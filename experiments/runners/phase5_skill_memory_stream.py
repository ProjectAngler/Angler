"""Meta-learned, skill-local procedural memory experiment.

The policy deliberately has no slow path from a public task to an answer.
Public item attributes create the 120 candidate permutation embeddings and
opaque procedure symbols create structural routing queries, but an empty
competence state always produces exactly zero logits.  Scalar outcome
feedback is the only online write signal.

Evaluator-owned targets are used only by the outer meta-loss.  The final
composition evaluator is imported only after every slow parameter is frozen.
Online evaluation advances exclusively from transactionally admitted memory
states; it never retrieves or replays an earlier experience.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.execution import SharedPrimitiveSequenceDecoder
from angler.procedures.learning import CompositeOperatorLearner
from angler.procedures.skill_memory import (
    ProceduralSkillRead,
    ProceduralSkillState,
    RoutedProceduralMemory,
    permute_procedural_skill_slots,
    procedural_skill_state_digest,
    restore_procedural_skill_state,
    snapshot_procedural_skill_state,
    zero_procedural_skill_content,
)
from angler.procedures.trunk import FrozenHashTextEncoder, NeuralOperatorCore
from angler.reasoning.recurrent_core import reasoning_state_digest


_ITEM_COUNT = 5
_ITEM_FEATURE_WIDTH = 14
_EXPECTED_PRIMITIVE_ROOTS = 10
_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(
    itertools.permutations(range(_ITEM_COUNT))
)
_PERMUTATION_TO_INDEX = {
    permutation: index for index, permutation in enumerate(_PERMUTATIONS)
}
_REPORT_VERSION = "angler.phase5-skill-memory-stream.v23"
_COMPATIBLE_PHASE5_RUNNERS = (
    "angler.phase5-skill-memory-stream.v2",
    "angler.phase5-skill-memory-stream.v3",
    "angler.phase5-skill-memory-stream.v4",
    "angler.phase5-skill-memory-stream.v5",
    "angler.phase5-skill-memory-stream.v6",
    "angler.phase5-skill-memory-stream.v7",
    "angler.phase5-skill-memory-stream.v8",
    "angler.phase5-skill-memory-stream.v10",
    "angler.phase5-skill-memory-stream.v11",
    "angler.phase5-skill-memory-stream.v12",
    "angler.phase5-skill-memory-stream.v13",
    "angler.phase5-skill-memory-stream.v14",
    "angler.phase5-skill-memory-stream.v15",
    "angler.phase5-skill-memory-stream.v16",
    "angler.phase5-skill-memory-stream.v17",
    "angler.phase5-skill-memory-stream.v18",
    "angler.phase5-skill-memory-stream.v19",
    "angler.phase5-skill-memory-stream.v20",
    "angler.phase5-skill-memory-stream.v21",
    "angler.phase5-skill-memory-stream.v22",
    _REPORT_VERSION,
)
_RUNNERS_WITHOUT_DIRECTION_MIXER = frozenset(
    runner
    for runner in _COMPATIBLE_PHASE5_RUNNERS
    if runner
    not in {
        "angler.phase5-skill-memory-stream.v15",
        "angler.phase5-skill-memory-stream.v16",
        "angler.phase5-skill-memory-stream.v17",
        "angler.phase5-skill-memory-stream.v18",
        "angler.phase5-skill-memory-stream.v19",
        "angler.phase5-skill-memory-stream.v20",
        "angler.phase5-skill-memory-stream.v21",
        "angler.phase5-skill-memory-stream.v22",
        _REPORT_VERSION,
    }
)
_RUNNERS_WITHOUT_FAST_ADAPTER = frozenset(
    runner
    for runner in _COMPATIBLE_PHASE5_RUNNERS
    if runner
    not in {
        "angler.phase5-skill-memory-stream.v17",
        "angler.phase5-skill-memory-stream.v18",
        "angler.phase5-skill-memory-stream.v19",
        "angler.phase5-skill-memory-stream.v20",
        "angler.phase5-skill-memory-stream.v21",
        "angler.phase5-skill-memory-stream.v22",
        _REPORT_VERSION,
    }
)
_RUNNERS_WITHOUT_GOAL_PROJECTION = frozenset(
    {
        "angler.phase5-skill-memory-stream.v2",
        "angler.phase5-skill-memory-stream.v3",
        "angler.phase5-skill-memory-stream.v4",
        "angler.phase5-skill-memory-stream.v5",
        "angler.phase5-skill-memory-stream.v6",
        "angler.phase5-skill-memory-stream.v7",
        "angler.phase5-skill-memory-stream.v8",
        "angler.phase5-skill-memory-stream.v10",
        "angler.phase5-skill-memory-stream.v11",
        "angler.phase5-skill-memory-stream.v12",
        "angler.phase5-skill-memory-stream.v13",
        "angler.phase5-skill-memory-stream.v14",
        "angler.phase5-skill-memory-stream.v15",
        "angler.phase5-skill-memory-stream.v16",
        "angler.phase5-skill-memory-stream.v17",
        "angler.phase5-skill-memory-stream.v18",
    }
)
_RUNNERS_WITHOUT_REVERSIBLE_TRANSITION = frozenset(
    runner
    for runner in _COMPATIBLE_PHASE5_RUNNERS
    if runner
    not in {
        "angler.phase5-skill-memory-stream.v22",
        _REPORT_VERSION,
    }
)
_TRAINING_STAGES = (
    "leaf_core",
    "integrated",
    "relational_acquisition",
    "harmonization",
    "procedural_adapter",
    "reverse_construction",
    "reverse_harmonization",
    "procedural_coadaptation",
    "reversible_transition_acquisition",
)
_PHASE4_CHECKPOINT = Path("outputs/phase4-continual-v5-clean.pt")
_PHASE4_CHECKPOINT_SHA256 = (
    "cbb470051bc56d84a128a163ad646f8fe83ab3d15335a8a9e5a53522036a2407"
)
_PHASE4_RESULT_DIGEST = (
    "sha256:b902a6c0e0dbded9213d433f90b646ff934d7c5a2e9e43fdb5077b5106789aeb"
)
_HARMONIZATION_SOURCE_CHECKPOINT_SHA256 = (
    "4ebe66c0949e0202a1af69f12f93f9dc2029563f07f451229fa8ff6d3b92b3de"
)
_HARMONIZATION_OUTER_STEPS = 16
_PROCEDURAL_ADAPTER_SOURCE_CHECKPOINT_SHA256 = (
    _HARMONIZATION_SOURCE_CHECKPOINT_SHA256
)
_PROCEDURAL_ADAPTER_OUTER_STEPS = 256
_REVERSE_CONSTRUCTION_OUTER_STEPS = 512
_REVERSE_CONSTRUCTION_ATTEMPTS = 4
_REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256 = (
    "fe22f6c6d1c4ce8157a702af3264fdf6cbacda4293e29a666b6107396192f88f"
)
_REVERSE_HARMONIZATION_OUTER_STEPS = 256
_PROCEDURAL_COADAPTATION_SOURCE_CHECKPOINT_SHA256 = (
    "411dc36a2a02c8b1cfc8327fd61446f1c8c9f3ce048ee3011c8158c250cde4ee"
)
_PROCEDURAL_COADAPTATION_OUTER_STEPS = 256
_PROCEDURAL_ADAPTER_UNARY_ACTUAL_WEIGHT = 0.5
_PROCEDURAL_ADAPTER_FORWARD_WEIGHT = 1.0
_PROCEDURAL_ADAPTER_SPECIFICITY_WEIGHT = 1.0
_PROCEDURAL_ADAPTER_BINARY_ACTUAL_WEIGHT = 1.0
_OPERATOR_AUDIT_REPORT_VERSION = "angler.phase5-operator-localization-audit.v1"
_OPERATOR_AUDIT_COHORTS = (
    "unary_depth2",
    "unary_depth3",
    "unary_direct_binary_child",
)
_PROCEDURAL_ADAPTER_TRAINING_COHORTS = (
    *_OPERATOR_AUDIT_COHORTS,
    "binary_root",
)
_OPERATOR_AUDIT_BRIDGES = {
    "operator": "compiler_operator_bridge.",
    "source": "compiler_source_bridge.",
    "successor": "compiler_successor_bridge.",
}
_OPERATOR_AUDIT_SEED_COUNT = 8
_OPERATOR_AUDIT_INSTANCES_PER_PROGRAM = 16
_OPERATOR_AUDIT_QUERY_INSTANCES = 4
_PERMITTED_CHECKPOINT_MIGRATION_PREFIXES = (
    "composition_item_encoder.",
    "composition_candidate_encoder.",
    "composition_state_encoder.",
    "composition_goal_encoder.",
    "composition_memory.",
    "phase4_direction_mixer.",
    "phase4_reliability_gate.",
    "procedural_fast_adapter.",
    "procedural_goal_projection.",
    "reversible_procedure_transition.",
    "reversible_transition_mode",
    "compiler_source_bridge.",
    "compiler_operator_bridge.",
    "compiler_successor_bridge.",
)
_CONDITION_AXIS_KEY = "relational_branch_router.condition_axis"
_CONDITION_AXIS_MIGRATION_RUNNERS = frozenset(
    {
        "angler.phase5-skill-memory-stream.v2",
        "angler.phase5-skill-memory-stream.v3",
        "angler.phase5-skill-memory-stream.v4",
        "angler.phase5-skill-memory-stream.v5",
        "angler.phase5-skill-memory-stream.v6",
        "angler.phase5-skill-memory-stream.v7",
        "angler.phase5-skill-memory-stream.v8",
        "angler.phase5-skill-memory-stream.v10",
        "angler.phase5-skill-memory-stream.v11",
    }
)
_CONDITION_AXIS_RESET_RUNNERS = frozenset(
    {
        # v12 learned this axis against a generic nonlinear direction basis.
        # v13's canonical outcome coordinates have different semantics, so
        # retaining those numbers would silently reinterpret old weights.
        "angler.phase5-skill-memory-stream.v12",
    }
)
# v13 through v16 use the same canonical left-versus-right outcome
# coordinate. Recursive child-policy execution and candidate-local evidence
# calibration do not reinterpret that sign, so acquired axes remain valid and
# are retained exactly.
_RELATIONAL_ACQUISITION_PREFIXES = (
    "relational_branch_router.",
)
_HARMONIZATION_TRAINABLE_PREFIXES = (
    "phase4_direction_mixer.",
)
_PROCEDURAL_ADAPTER_TRAINABLE_PREFIXES = (
    "procedural_fast_adapter.source_down.",
    "procedural_fast_adapter.code_gate.",
    "procedural_fast_adapter.forward_up.",
)
_REVERSE_CONSTRUCTION_TRAINABLE_PREFIXES = (
    "procedural_fast_adapter.source_down.",
    "procedural_fast_adapter.code_gate.",
    "procedural_fast_adapter.forward_up.",
    "procedural_fast_adapter.reverse_up.",
    "procedural_goal_projection.candidate_down.",
)
_REVERSE_CONSTRUCTION_TRAINABLE_NAMES = frozenset(
    {
        "memory.feedback_direction_encoder.3.weight",
        "composition_memory.feedback_direction_encoder.3.weight",
    }
)
_REVERSE_HARMONIZATION_TRAINABLE_PREFIXES = (
    "phase4_direction_mixer.",
    "phase4_reliability_gate.",
)
_REVERSIBLE_TRANSITION_TRAINABLE_PREFIXES = (
    "reversible_procedure_transition.",
)
_REVERSIBLE_TRANSITION_TRAINABLE_NAMES = frozenset(
    {
        "memory.feedback_direction_encoder.3.weight",
        "composition_memory.feedback_direction_encoder.3.weight",
    }
)
_PERMITTED_CHECKPOINT_DROP_PREFIXES = (
    "tree_own.",
    "tree_children.",
    "compiler_harmonizer.",
    "composition_fusion.",
    "recursive_branch_router.",
)
_V2_FOLDED_SEAM_PREFIXES = (
    "composition_item_encoder.",
    "composition_candidate_encoder.",
    "composition_state_encoder.",
    "composition_goal_encoder.",
    "composition_memory.",
    "phase4_direction_mixer.",
    "phase4_reliability_gate.",
    "procedural_fast_adapter.",
    "recursive_branch_router.",
    "compiler_operator_bridge.",
)
_COMPOSITION_TRAINABLE_PREFIXES = (
    "composition_item_encoder.",
    "composition_candidate_encoder.",
    "composition_state_encoder.",
    "composition_goal_encoder.",
    "composition_memory.",
    "phase4_direction_mixer.",
    "phase4_reliability_gate.",
    "relational_branch_router.",
    "procedural_fast_adapter.",
    "compiler_source_bridge.",
    "compiler_operator_bridge.",
    "compiler_successor_bridge.",
)


class _NonFinitePolicyScoresError(RuntimeError):
    """A policy rescore that is unsafe to commit but otherwise well formed."""


class _IncompleteMatchedDescendantError(RuntimeError):
    """A matched objective whose acquired tree is not yet executable."""


class RelationalBranchRouter(nn.Module):
    """Choose between children only through learned procedural polarity."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 1
            or width % 2
        ):
            raise ValueError("width must be a positive even integer")
        self.context_width = width
        self.outcome_start = width // 2
        self.condition_axis = nn.Parameter(
            torch.zeros(width - self.outcome_start)
        )

    def forward(
        self,
        procedure_context: torch.Tensor,
        public_flag: bool,
    ) -> torch.Tensor:
        if procedure_context.ndim != 2:
            raise ValueError("procedure context must be rank two")
        if procedure_context.shape[-1] != self.context_width:
            raise ValueError("procedure context width does not match router width")
        if type(public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        signed_flag = procedure_context.new_tensor(
            1.0 if public_flag else -1.0
        )
        # Slot summaries reserve the first half for reward-independent content
        # and the second half for centered outcome covariance.  Branch
        # polarity may use only the latter: opaque symbol/content geometry can
        # address a skill but cannot decide which child is correct.
        outcome_context = procedure_context[..., self.outcome_start :]
        polarity = (
            (outcome_context * self.condition_axis).sum(dim=-1, keepdim=True)
            / math.sqrt(self.condition_axis.numel())
        )
        contrast = signed_flag * polarity
        return torch.cat((-0.5 * contrast, 0.5 * contrast), dim=-1)


class ConditionalReversibleTransition(nn.Module):
    """A small code-gated additive coupling with an algebraic inverse."""

    def __init__(self, width: int, rank: int = 8) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 2:
            raise ValueError("reversible transition width must be at least two")
        if width % 2:
            raise ValueError("reversible transition width must be even")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("reversible transition rank must be positive")
        self.width = width
        self.half_width = width // 2
        self.rank = rank
        self.condition_gate = nn.Linear(2 * width, rank, bias=False)
        self.first_normalize = nn.LayerNorm(
            self.half_width,
            elementwise_affine=False,
        )
        self.first_down = nn.Linear(self.half_width, rank)
        self.first_up = nn.Linear(rank, self.half_width, bias=False)
        self.second_normalize = nn.LayerNorm(
            self.half_width,
            elementwise_affine=False,
        )
        self.second_down = nn.Linear(self.half_width, rank)
        self.second_up = nn.Linear(rank, self.half_width, bias=False)
        nn.init.zeros_(self.first_up.weight)
        nn.init.zeros_(self.second_up.weight)

    def _conditioned_delta(
        self,
        normalize: nn.Module,
        down: nn.Module,
        up: nn.Module,
        source_half: torch.Tensor,
        condition_gate: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.silu(down(normalize(source_half))) * condition_gate
        return up(hidden) / math.sqrt(self.rank)

    def forward(
        self,
        source: torch.Tensor,
        condition: torch.Tensor,
        *,
        reverse: bool = False,
        post_tanh_gate_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.shape[-1:] != (self.width,):
            raise ValueError("reversible source has the wrong width")
        if condition.shape[-1:] != (2 * self.width,):
            raise ValueError("reversible condition has the wrong width")
        if not isinstance(reverse, bool):
            raise TypeError("reverse must be boolean")
        try:
            expanded_condition = condition.expand(
                *source.shape[:-1],
                2 * self.width,
            )
        except RuntimeError as error:
            raise ValueError(
                "reversible condition cannot broadcast across source rows"
            ) from error
        condition_gate = torch.tanh(self.condition_gate(expanded_condition))
        if post_tanh_gate_residual is not None:
            if (
                not isinstance(post_tanh_gate_residual, torch.Tensor)
                or post_tanh_gate_residual.shape[-1:] != (self.rank,)
                or post_tanh_gate_residual.device != condition_gate.device
                or post_tanh_gate_residual.dtype != condition_gate.dtype
                or not bool(torch.isfinite(post_tanh_gate_residual).all().item())
            ):
                raise ValueError(
                    "post-tanh gate residual must be a finite rank-width tensor"
                )
            try:
                expanded_gate_residual = post_tanh_gate_residual.expand_as(
                    condition_gate
                )
            except RuntimeError as error:
                raise ValueError(
                    "post-tanh gate residual cannot broadcast across source rows"
                ) from error
            condition_gate = condition_gate + expanded_gate_residual
        first, second = source.split(self.half_width, dim=-1)
        if reverse:
            recovered_second = second - self._conditioned_delta(
                self.second_normalize,
                self.second_down,
                self.second_up,
                first,
                condition_gate,
            )
            recovered_first = first - self._conditioned_delta(
                self.first_normalize,
                self.first_down,
                self.first_up,
                recovered_second,
                condition_gate,
            )
            return torch.cat((recovered_first, recovered_second), dim=-1)
        transitioned_first = first + self._conditioned_delta(
            self.first_normalize,
            self.first_down,
            self.first_up,
            second,
            condition_gate,
        )
        transitioned_second = second + self._conditioned_delta(
            self.second_normalize,
            self.second_down,
            self.second_up,
            transitioned_first,
            condition_gate,
        )
        return torch.cat((transitioned_first, transitioned_second), dim=-1)


class CodeConditionedLowRankTransition(nn.Module):
    """Turn one acquired procedure code into a shared rank-bounded fast weight.

    The adapter predicts a latent transition, never candidate logits.  Its
    bias-free code gate makes the all-zero code an exact identity, while zero
    output matrices make legacy checkpoints bit-compatible before learning.
    """

    def __init__(self, width: int, *, rank: int = 8) -> None:
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 1
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 < rank < width
        ):
            raise ValueError("adapter width and rank must satisfy 0 < rank < width")
        self.width = width
        self.rank = rank
        self.source_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.source_down = nn.Linear(width, rank, bias=True)
        self.code_gate = nn.Linear(width, rank, bias=False)
        self.forward_up = nn.Linear(rank, width, bias=False)
        self.reverse_up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.forward_up.weight)
        nn.init.zeros_(self.reverse_up.weight)

    def forward(
        self,
        source: torch.Tensor,
        procedure_code: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        if source.ndim != 2 or source.shape[-1] != self.width:
            raise ValueError("adapter source must have shape [batch, width]")
        if (
            procedure_code.ndim != 2
            or procedure_code.shape[-1] != self.width
            or procedure_code.shape[0] not in (1, source.shape[0])
        ):
            raise ValueError(
                "adapter code must have shape [1, width] or [batch, width]"
            )
        if type(reverse) is not bool:
            raise TypeError("adapter reverse flag must be bool")
        hidden = self.latent_factors(source, procedure_code)
        output = self.reverse_up(hidden) if reverse else self.forward_up(hidden)
        return output / math.sqrt(self.rank)

    def latent_factors(
        self,
        source: torch.Tensor,
        procedure_code: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared code-conditioned rank factors without an output head."""

        if source.ndim != 2 or source.shape[-1] != self.width:
            raise ValueError("adapter source must have shape [batch, width]")
        if (
            procedure_code.ndim != 2
            or procedure_code.shape[-1] != self.width
            or procedure_code.shape[0] not in (1, source.shape[0])
        ):
            raise ValueError(
                "adapter code must have shape [1, width] or [batch, width]"
            )
        source_factors = self.source_down(self.source_norm(source))
        code_factors = torch.tanh(self.code_gate(procedure_code))
        if code_factors.shape[0] == 1 and source_factors.shape[0] != 1:
            code_factors = code_factors.expand(source_factors.shape[0], -1)
        return source_factors * code_factors


class CandidateEquivariantGoalProjection(nn.Module):
    """Foreshadow one latent destination without candidate identities or targets."""

    def __init__(self, width: int, *, rank: int = 8) -> None:
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 1
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 < rank < width
        ):
            raise ValueError("goal width and rank must satisfy 0 < rank < width")
        self.width = width
        self.rank = rank
        self.relative_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.candidate_down = nn.Linear(width, rank, bias=False)
        nn.init.zeros_(self.candidate_down.weight)

    def forward(
        self,
        source: torch.Tensor,
        candidates: torch.Tensor,
        query_factors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.shape != (1, self.width):
            raise ValueError("goal source must have shape [1, width]")
        if candidates.ndim != 2 or candidates.shape[-1] != self.width:
            raise ValueError("goal candidates must have shape [count, width]")
        if query_factors.shape != (1, self.rank):
            raise ValueError("goal query must have shape [1, rank]")
        relative = candidates - source
        keys = self.candidate_down(self.relative_norm(relative))
        energies = torch.matmul(keys, query_factors[0]) / math.sqrt(self.rank)
        probabilities = torch.softmax(energies, dim=0)
        centered_probabilities = probabilities - probabilities.new_full(
            probabilities.shape,
            1.0 / probabilities.numel(),
        )
        destination_residual = torch.einsum(
            "c,cw->w",
            centered_probabilities,
            candidates,
        )
        # Zero-initialized keys or a zero code yield uniform probability and
        # therefore an exact zero residual. Candidate reordering changes no value.
        return torch.tanh(destination_residual).unsqueeze(0), energies


def _execute_binary_branch(
    branch_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft routing, straight-through execution, and the exact tie bit.

    The same scale-aware predicate that chooses a soft tie is retained for
    evaluator reporting.  This prevents a float32 near-tie from being reported
    as a hard choice while execution actually blended both children.
    """

    if branch_logits.ndim != 2 or branch_logits.shape[-1] != 2:
        raise ValueError("binary branch logits must have shape [batch, 2]")
    if not bool(torch.isfinite(branch_logits).all().item()):
        raise ValueError("binary branch logits must be finite")
    branch_weights = torch.softmax(branch_logits, dim=-1)
    hard_branch_weights = F.one_hot(
        branch_weights.argmax(dim=-1),
        num_classes=2,
    ).to(dtype=branch_weights.dtype)
    straight_through_branch = hard_branch_weights + (
        branch_weights - branch_weights.detach()
    )
    contrast = (branch_logits[:, 1] - branch_logits[:, 0]).abs()
    contrast_tolerance = (
        32.0
        * torch.finfo(branch_logits.dtype).eps
        * branch_logits.abs().amax(dim=-1).clamp_min(1.0)
    )
    execution_tied = contrast <= contrast_tolerance
    executed_branch_weights = torch.where(
        execution_tied.unsqueeze(-1),
        branch_weights,
        straight_through_branch,
    )
    return branch_weights, executed_branch_weights, execution_tied


def _soft_reanchor_intermediate(
    source: torch.Tensor,
    successor: torch.Tensor,
    candidate_states: torch.Tensor,
) -> torch.Tensor:
    """Return a generated step to the learned public state manifold.

    The soft projection has no candidate identity, target, or hard selection.
    Its geometry remains differentiable because the experiment showed that a
    frozen, previously unaligned manifold prevented the new recursive seam
    from learning.
    """

    if source.ndim != 2 or successor.shape != source.shape:
        raise ValueError("source and successor must have the same rank-two shape")
    if (
        candidate_states.ndim != 2
        or candidate_states.shape[1] != source.shape[1]
    ):
        raise ValueError("candidate states must have shape [candidate, width]")
    anchors = candidate_states
    distances = (
        (anchors - successor[0].unsqueeze(0))
        .square()
        .mean(dim=-1)
    )
    centered = distances - distances.mean()
    scale = distances.std(unbiased=False).clamp_min(
        torch.finfo(distances.dtype).eps
    )
    weights = torch.softmax(-centered / scale, dim=-1)
    anchored = torch.matmul(weights.unsqueeze(0), anchors)
    effect = (successor - source).square().mean(dim=-1, keepdim=True)
    effect_gate = effect / (effect + torch.finfo(effect.dtype).eps)
    return successor + effect_gate * (anchored - successor)


def _seed_reproducible_stage(
    seed: int,
    domain: str,
    device: torch.device,
) -> int:
    """Reset stochastic evaluation state independently of prior execution."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(domain, str) or not domain:
        raise ValueError("seed domain must be a non-empty string")
    material = f"project-angler.phase5-stage-seed.v1\x00{seed}\x00{domain}".encode(
        "utf-8"
    )
    stage_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.seed(stage_seed)
    torch.manual_seed(stage_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(stage_seed)
    return stage_seed


@dataclass(frozen=True, slots=True)
class RunProfile:
    name: str
    width: int
    hidden_width: int
    hash_width: int
    slots: int
    heads: int
    read_top_k: int
    meta_steps: int
    meta_instances_per_program: int
    learning_rate: float
    gradient_clip: float
    encounters_per_primitive: int
    cases_per_component_probe: int
    cases_per_composition: int


_PROFILES: Mapping[str, RunProfile] = {
    "smoke": RunProfile(
        name="smoke",
        width=64,
        hidden_width=32,
        hash_width=32,
        slots=8,
        heads=4,
        read_top_k=1,
        meta_steps=1,
        meta_instances_per_program=8,
        learning_rate=2.0e-3,
        gradient_clip=5.0,
        encounters_per_primitive=2,
        cases_per_component_probe=2,
        cases_per_composition=2,
    ),
    "diagnostic": RunProfile(
        name="diagnostic",
        width=64,
        hidden_width=128,
        hash_width=128,
        slots=16,
        heads=8,
        read_top_k=1,
        meta_steps=32,
        meta_instances_per_program=8,
        learning_rate=8.0e-4,
        gradient_clip=5.0,
        encounters_per_primitive=8,
        cases_per_component_probe=8,
        cases_per_composition=4,
    ),
    "leaf": RunProfile(
        name="leaf",
        width=64,
        hidden_width=128,
        hash_width=128,
        slots=16,
        heads=8,
        read_top_k=1,
        meta_steps=128,
        meta_instances_per_program=8,
        learning_rate=8.0e-4,
        gradient_clip=5.0,
        encounters_per_primitive=8,
        cases_per_component_probe=8,
        cases_per_composition=4,
    ),
    "composition": RunProfile(
        name="composition",
        width=64,
        hidden_width=128,
        hash_width=128,
        slots=16,
        heads=8,
        read_top_k=1,
        meta_steps=64,
        meta_instances_per_program=8,
        learning_rate=8.0e-4,
        gradient_clip=5.0,
        encounters_per_primitive=8,
        cases_per_component_probe=8,
        cases_per_composition=8,
    ),
    "standard": RunProfile(
        name="standard",
        width=64,
        hidden_width=128,
        hash_width=128,
        slots=16,
        heads=8,
        read_top_k=1,
        meta_steps=128,
        meta_instances_per_program=32,
        learning_rate=8.0e-4,
        gradient_clip=5.0,
        encounters_per_primitive=64,
        cases_per_component_probe=40,
        cases_per_composition=40,
    ),
}


def _make_reference_compiler(
    *,
    width: int,
    hidden_width: int,
    hash_width: int,
) -> CompositeOperatorLearner:
    """Construct the Phase-4 compiler architecture without invoking a world."""

    core = NeuralOperatorCore(
        width=width,
        hidden_width=hidden_width,
        schema_hash_width=hash_width,
    )
    decoder = SharedPrimitiveSequenceDecoder(
        width=width,
        hidden_width=hidden_width,
        hash_width=hash_width,
        maximum_steps=4,
    )
    compiler = CompositeOperatorLearner(
        core,
        decoder,
        binding_hash_width=hash_width,
        hidden_width=hidden_width,
    )
    compiler.eval()
    compiler.requires_grad_(False)
    return compiler


def _load_phase4_compiler(
    checkpoint_path: str | Path,
) -> tuple[CompositeOperatorLearner, dict[str, Any]]:
    """Load the exact accepted Phase-4 checkpoint as a frozen dependency."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase-4 checkpoint is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _PHASE4_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Phase-4 checkpoint hash differs from the pinned skill-memory substrate"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("runner") != (
        "angler.phase4.causal-operator-compiler.v5"
    ):
        raise RuntimeError("Phase-4 checkpoint runner identity is invalid")
    if payload.get("result_digest") != _PHASE4_RESULT_DIGEST:
        raise RuntimeError("Phase-4 checkpoint result identity is invalid")
    profile = payload.get("profile")
    model_state = payload.get("model")
    if not isinstance(profile, dict) or not isinstance(model_state, dict):
        raise RuntimeError("Phase-4 checkpoint payload is incomplete")
    try:
        compiler = _make_reference_compiler(
            width=int(profile["width"]),
            hidden_width=int(profile["hidden_width"]),
            hash_width=int(profile["hash_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Phase-4 checkpoint profile is invalid") from error
    compiler.load_state_dict(model_state, strict=True)
    compiler.eval()
    compiler.requires_grad_(False)
    return compiler, {
        "path": str(path),
        "sha256": digest,
        "runner": payload["runner"],
        "result_digest": payload.get("result_digest"),
        "usage_scope": (
            "frozen Phase-4 effect geometry through learned "
            "source/operator/successor bridges"
        ),
        "executor_calls": 0,
    }


def _load_initial_policy_checkpoint(
    policy: "SkillMemoryPolicy",
    checkpoint_path: str | Path,
    profile: RunProfile,
) -> dict[str, Any]:
    """Restore compatible slow weights without importing online skill state."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"initial policy checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("runner") not in _COMPATIBLE_PHASE5_RUNNERS
    ):
        raise RuntimeError("initial policy checkpoint runner identity is invalid")
    source_runner = payload["runner"]
    source_profile = payload.get("profile")
    model_state = payload.get("model")
    result_digest = payload.get("result_digest")
    source_initialization = payload.get("initialization")
    source_training = payload.get("training")
    if (
        not isinstance(source_profile, dict)
        or not isinstance(model_state, dict)
        or not isinstance(result_digest, str)
        or not result_digest.startswith("sha256:")
        or payload.get("compiler_checkpoint_sha256")
        != _PHASE4_CHECKPOINT_SHA256
    ):
        raise RuntimeError("initial policy checkpoint payload is incomplete")
    architecture_fields = (
        "width",
        "hidden_width",
        "hash_width",
        "slots",
        "heads",
        "read_top_k",
    )
    expected = asdict(profile)
    if any(source_profile.get(field) != expected[field] for field in architecture_fields):
        raise RuntimeError("initial policy checkpoint architecture is incompatible")
    if (
        source_runner in _CONDITION_AXIS_MIGRATION_RUNNERS
        and _CONDITION_AXIS_KEY in model_state
    ):
        raise RuntimeError(
            "pre-v11 checkpoint contains an undeclared condition axis"
        )
    if (
        source_runner in _CONDITION_AXIS_RESET_RUNNERS
        and _CONDITION_AXIS_KEY not in model_state
    ):
        raise RuntimeError("v12 checkpoint is missing its versioned condition axis")
    if source_runner in _RUNNERS_WITHOUT_FAST_ADAPTER and any(
        key.startswith("procedural_fast_adapter.") for key in model_state
    ):
        raise RuntimeError(
            "legacy checkpoint contains an undeclared procedural fast adapter"
        )
    if source_runner in _RUNNERS_WITHOUT_GOAL_PROJECTION and any(
        key.startswith("procedural_goal_projection.") for key in model_state
    ):
        raise RuntimeError(
            "legacy checkpoint contains an undeclared procedural goal projection"
        )
    if source_runner in _RUNNERS_WITHOUT_REVERSIBLE_TRANSITION and any(
        key == "reversible_transition_mode"
        or key.startswith("reversible_procedure_transition.")
        for key in model_state
    ):
        raise RuntimeError(
            "legacy checkpoint contains an undeclared reversible transition"
        )
    if source_runner == "angler.phase5-skill-memory-stream.v2":
        folded_overlap = tuple(
            sorted(
                key
                for key in model_state
                if key.startswith(_V2_FOLDED_SEAM_PREFIXES)
            )
        )
        if folded_overlap:
            raise RuntimeError(
                "v2 checkpoint contains folded-seam state that cannot migrate "
                f"into recursive semantics: {list(folded_overlap)}"
            )
    compiler_before = reasoning_state_digest(policy.stable_compiler)
    dropped = tuple(
        sorted(
            key
            for key in model_state
            if key.startswith(_PERMITTED_CHECKPOINT_DROP_PREFIXES)
        )
    )
    reset = tuple(
        key
        for key in (_CONDITION_AXIS_KEY,)
        if source_runner in _CONDITION_AXIS_RESET_RUNNERS and key in model_state
    )
    loadable_state = {
        key: value
        for key, value in model_state.items()
        if key not in dropped and key not in reset
    }
    try:
        incompatible = policy.load_state_dict(loadable_state, strict=False)
    except RuntimeError as error:
        raise RuntimeError("initial policy checkpoint state is incompatible") from error
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    current_keys = set(policy.state_dict())
    permitted_missing = (
        {
            key
            for key in current_keys
            if key.startswith(_PERMITTED_CHECKPOINT_MIGRATION_PREFIXES)
        }
        if source_runner != _REPORT_VERSION
        else set()
    )
    if source_runner not in _RUNNERS_WITHOUT_DIRECTION_MIXER:
        permitted_missing = {
            key
            for key in permitted_missing
            if not key.startswith("phase4_direction_mixer.")
        }
    if source_runner not in _RUNNERS_WITHOUT_FAST_ADAPTER:
        permitted_missing = {
            key
            for key in permitted_missing
            if not key.startswith("procedural_fast_adapter.")
        }
    if source_runner not in _RUNNERS_WITHOUT_GOAL_PROJECTION:
        permitted_missing = {
            key
            for key in permitted_missing
            if not key.startswith("procedural_goal_projection.")
        }
    if (
        source_runner in _CONDITION_AXIS_MIGRATION_RUNNERS
        or source_runner in _CONDITION_AXIS_RESET_RUNNERS
    ):
        permitted_missing.add(_CONDITION_AXIS_KEY)
    missing_set = set(missing)
    atomic_migration = not (missing_set - permitted_missing)
    for prefix in _PERMITTED_CHECKPOINT_MIGRATION_PREFIXES:
        group = {key for key in current_keys if key.startswith(prefix)}
        group_missing = missing_set & group
        if group_missing and group_missing != group:
            atomic_migration = False
    if unexpected or not atomic_migration:
        raise RuntimeError(
            "initial policy checkpoint has an undeclared state migration: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    cloned_migrations = (
        (
            "composition_item_encoder.",
            policy.composition_item_encoder,
            policy.item_encoder,
        ),
        (
            "composition_candidate_encoder.",
            policy.composition_candidate_encoder,
            policy.candidate_encoder,
        ),
        (
            "composition_state_encoder.",
            policy.composition_state_encoder,
            policy.state_encoder,
        ),
        (
            "composition_goal_encoder.",
            policy.composition_goal_encoder,
            policy.goal_encoder,
        ),
        ("composition_memory.", policy.composition_memory, policy.memory),
    )
    cloned_prefixes: list[str] = []
    for prefix, destination, source in cloned_migrations:
        group = {key for key in current_keys if key.startswith(prefix)}
        if group and group.issubset(missing_set):
            destination.load_state_dict(source.state_dict(), strict=True)
            cloned_prefixes.append(prefix)
    if (
        _CONDITION_AXIS_KEY in missing_set
        and bool(policy.relational_branch_router.condition_axis.detach().count_nonzero())
    ):
        raise RuntimeError("migrated condition axis did not remain exactly zero")
    if reasoning_state_digest(policy.stable_compiler) != compiler_before:
        raise RuntimeError("initial policy checkpoint changed the frozen compiler")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "result_digest": result_digest,
        "source_stage": payload.get("stage", "unspecified"),
        "source_profile": source_profile.get("name", "unspecified"),
        "source_runner": source_runner,
        "source_initialization": source_initialization,
        "source_training": source_training,
        "dropped_parameter_keys": list(dropped),
        "reset_parameter_keys": list(reset),
        "online_state_restored": False,
        "slow_model_state_restored": True,
        "fresh_parameter_keys": list(missing),
        "fresh_cloned_prefixes": cloned_prefixes,
    }


@dataclass(frozen=True, slots=True)
class NodePolicyRead:
    """Learner-visible tensors for one expression node."""

    path: tuple[int, ...]
    child_count: int
    state_embedding: torch.Tensor
    goal_embedding: torch.Tensor
    candidate_embeddings: torch.Tensor
    memory_read: ProceduralSkillRead
    recursive_predecessor: torch.Tensor
    feedback_context: torch.Tensor
    feedback_available: torch.Tensor
    candidate_branch_advantages: torch.Tensor
    child_candidate_scores: torch.Tensor
    conditioned_child_candidate_scores: torch.Tensor
    branch_weights: torch.Tensor
    executed_branch_weights: torch.Tensor
    execution_tied: torch.Tensor
    subtree_context: torch.Tensor
    memory_tier: str


@dataclass(frozen=True, slots=True)
class PolicyScores:
    logits: torch.Tensor
    candidate_embeddings: torch.Tensor
    nodes: tuple[NodePolicyRead, ...]
    root_context: torch.Tensor
    composition_logits: torch.Tensor
    phase4_bridge_logits: torch.Tensor
    phase4_forward_evidence: torch.Tensor
    phase4_reverse_evidence: torch.Tensor
    phase4_direction_gains: torch.Tensor
    phase4_reliability: torch.Tensor
    memory_bias: torch.Tensor
    binary_policy_logits: torch.Tensor
    root_available: torch.Tensor
    public_feedback_evidence: torch.Tensor

    @property
    def root(self) -> NodePolicyRead:
        if not self.nodes or self.nodes[0].path:
            raise RuntimeError("policy score tree has no root node")
        return self.nodes[0]


@dataclass(frozen=True, slots=True)
class TaskProposal:
    """A frozen public permutation proposal with no evaluator state."""

    answer: tuple[str, ...]
    candidate_index: int
    scores: PolicyScores
    behavior_probabilities: torch.Tensor
    competence_digest: str
    public_task_digest: str


@dataclass(frozen=True, slots=True)
class DifferentiableFeedback:
    candidate_state: ProceduralSkillState
    write_slot: int
    delta_norm: float
    route_probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class _MatchedDifferentiableFeedback:
    """One row-local feedback transaction over three matched control arms."""

    candidate_state: ProceduralSkillState
    write_slots: torch.Tensor
    delta_norms: torch.Tensor


@dataclass(frozen=True, slots=True)
class TransactionalFeedback:
    state: ProceduralSkillState
    accepted: bool
    write_slot: int
    delta_norm: float
    before_loss: float
    after_loss: float
    core_accepted: bool


class SkillMemoryPolicy(nn.Module):
    """A memory-only policy over all five-item output permutations.

    Entity symbols are never encoded.  They are used only after candidate
    selection to render the public answer.  The tree composer is bias-free,
    and every output path is multiplied by acquired plastic context, so the
    empty competence state has exactly uniform probabilities.
    """

    def __init__(
        self,
        profile: RunProfile,
        stable_compiler: CompositeOperatorLearner | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        if stable_compiler is None:
            stable_compiler = _make_reference_compiler(
                width=profile.width,
                hidden_width=profile.hidden_width,
                hash_width=profile.hash_width,
            )
        if not isinstance(stable_compiler, CompositeOperatorLearner):
            raise TypeError("stable_compiler must be a CompositeOperatorLearner")
        stable_compiler.eval()
        stable_compiler.requires_grad_(False)
        self.stable_compiler = stable_compiler
        width = profile.width
        self.item_encoder = nn.Sequential(
            nn.Linear(_ITEM_FEATURE_WIDTH, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(_ITEM_COUNT * width),
            nn.Linear(_ITEM_COUNT * width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(width + 1),
            nn.Linear(width + 1, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.symbol_features = FrozenHashTextEncoder(profile.hash_width)
        self.goal_encoder = nn.Sequential(
            nn.LayerNorm(profile.hash_width + 7),
            nn.Linear(profile.hash_width + 7, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.composition_item_encoder = copy.deepcopy(self.item_encoder)
        self.composition_candidate_encoder = copy.deepcopy(self.candidate_encoder)
        self.composition_state_encoder = copy.deepcopy(self.state_encoder)
        self.composition_goal_encoder = copy.deepcopy(self.goal_encoder)
        self.memory = RoutedProceduralMemory(
            width,
            slots=profile.slots,
            heads=profile.heads,
            read_top_k=profile.read_top_k,
            hidden_width=profile.hidden_width,
        )
        # Non-leaf procedures acquire through an independent slow pathway.
        # It starts as an exact copy of the generic leaf learner but may adapt
        # during composition training without changing leaf acquisition.
        self.composition_memory = RoutedProceduralMemory(
            width,
            slots=profile.slots,
            heads=profile.heads,
            read_top_k=profile.read_top_k,
            hidden_width=profile.hidden_width,
        )
        self.composition_memory.load_state_dict(
            self.memory.state_dict(),
            strict=True,
        )

        # A binary branch preference may arise only from the interaction of a
        # public condition and acquired local procedure evidence.  There is no
        # generic condition-only MLP capable of learning a fixed-child shortcut.
        self.relational_branch_router = RelationalBranchRouter(width)
        # One shared candidate-local mixer learns how much to trust forward and
        # reverse procedural evidence. It has no candidate identity, operator
        # label, hidden target, or direct action output. Bounded gains start at
        # exactly one, preserving the retained v41 policy before calibration.
        self.phase4_direction_mixer = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )
        self.phase4_reliability_gate = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, 1),
        )
        self.compiler_source_bridge = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.compiler_operator_bridge = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.compiler_successor_bridge = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.procedural_fast_adapter = CodeConditionedLowRankTransition(
            width,
            rank=8,
        )
        self.procedural_goal_projection = CandidateEquivariantGoalProjection(
            width,
            rank=8,
        )
        self.reversible_procedure_transition = ConditionalReversibleTransition(
            width,
        )
        # The relational axis starts at zero, so both binary alternatives are
        # exactly equal before transferable procedural evidence is learned.
        nn.init.zeros_(self.phase4_direction_mixer[-1].weight)
        nn.init.zeros_(self.phase4_direction_mixer[-1].bias)
        nn.init.zeros_(self.phase4_reliability_gate[-1].weight)
        nn.init.zeros_(self.phase4_reliability_gate[-1].bias)
        self.register_buffer(
            "reversible_transition_mode",
            torch.zeros((), dtype=torch.bool),
        )
        self.register_buffer(
            "candidate_permutations",
            torch.tensor(_PERMUTATIONS, dtype=torch.long),
        )

    def _phase4_directional_evidence(
        self,
        base_bias: torch.Tensor,
        forward_delta: torch.Tensor,
        reverse_delta: torch.Tensor,
        score_limit: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calibrate two learned effect directions without decoding an answer."""

        if base_bias.ndim != 2 or base_bias.shape[0] != 1:
            raise ValueError("base bias must have shape [1, candidate count]")
        expected = (base_bias.shape[1],)
        if forward_delta.shape != expected or reverse_delta.shape != expected:
            raise ValueError("direction deltas must match the candidate count")
        if not math.isfinite(score_limit) or score_limit <= 0.0:
            raise ValueError("score limit must be finite and positive")
        centered_forward = forward_delta - forward_delta.mean()
        centered_reverse = reverse_delta - reverse_delta.mean()
        raw_forward = score_limit * torch.tanh(centered_forward).unsqueeze(0)
        raw_reverse = score_limit * torch.tanh(centered_reverse).unsqueeze(0)
        normalized_base = F.softsign(base_bias / score_limit)
        normalized_forward = F.softsign(raw_forward / score_limit)
        normalized_reverse = F.softsign(raw_reverse / score_limit)
        direction_features = torch.stack(
            (
                normalized_base,
                normalized_forward,
                normalized_reverse,
                normalized_base * normalized_forward,
                normalized_base * normalized_reverse,
                normalized_forward * normalized_reverse,
                (normalized_base - normalized_forward).abs(),
                (normalized_base - normalized_reverse).abs(),
                (normalized_forward - normalized_reverse).abs(),
            ),
            dim=-1,
        )
        direction_gains = 2.0 * torch.sigmoid(
            self.phase4_direction_mixer(direction_features)
        )
        # Express calibration as a residual around the retained v41 equation.
        # With neutral gains the correction is exactly zero and the nonlinear
        # combination remains limit*tanh(center(forward_delta+reverse_delta)).
        correction = (
            (direction_gains[..., 0] - 1.0) * centered_forward.unsqueeze(0)
            + (direction_gains[..., 1] - 1.0) * centered_reverse.unsqueeze(0)
        )
        combined_delta = forward_delta + reverse_delta
        combined_delta = combined_delta + correction.squeeze(0)
        combined_delta = combined_delta - combined_delta.mean()
        bridge_logits = score_limit * torch.tanh(combined_delta).unsqueeze(0)
        # Directional tensors remain raw, bounded diagnostics. Only their
        # effective pre-tanh combination is calibrated, preventing auxiliary
        # per-direction targets from teaching the mixer to preserve both.
        forward_evidence = raw_forward
        reverse_evidence = raw_reverse
        return (
            bridge_logits,
            forward_evidence,
            reverse_evidence,
            direction_gains,
        )

    def initial_state(self, batch_size: int = 1) -> ProceduralSkillState:
        reference = next(self.parameters())
        return self.memory.initial_state(
            batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    def memory_for_tier(self, tier: str) -> RoutedProceduralMemory:
        if tier == "leaf":
            return self.memory
        if tier == "composition":
            return self.composition_memory
        raise ValueError("memory tier must be leaf or composition")

    def score_task(
        self,
        public_task: Any,
        state: ProceduralSkillState,
        *,
        include_compiler: bool = True,
        include_phase4_bridge: bool = True,
        include_reverse_bridge: bool = True,
        include_descendants: bool = True,
        probe_leaf_bridge: bool = False,
        include_frozen_transition: bool = True,
        include_fast_adapter: bool = True,
        include_reversible_transition: bool = True,
        transition_only_composition: bool = False,
        matched_acquisition_batch: bool = False,
    ) -> PolicyScores:
        """Score one public task without changing its competence state.

        ``transition_only_composition`` removes direct utility-decoder bias
        from composed unary procedures while leaving standalone leaf scoring
        unchanged. Binary procedures still select between their recursively
        constructed child policies.
        """

        if not isinstance(include_compiler, bool):
            raise TypeError("include_compiler must be boolean")
        if not isinstance(include_phase4_bridge, bool):
            raise TypeError("include_phase4_bridge must be boolean")
        if not isinstance(include_reverse_bridge, bool):
            raise TypeError("include_reverse_bridge must be boolean")
        if not isinstance(include_descendants, bool):
            raise TypeError("include_descendants must be boolean")
        if not isinstance(probe_leaf_bridge, bool):
            raise TypeError("probe_leaf_bridge must be boolean")
        if not isinstance(include_frozen_transition, bool):
            raise TypeError("include_frozen_transition must be boolean")
        if not isinstance(include_fast_adapter, bool):
            raise TypeError("include_fast_adapter must be boolean")
        if not isinstance(include_reversible_transition, bool):
            raise TypeError("include_reversible_transition must be boolean")
        if not isinstance(transition_only_composition, bool):
            raise TypeError("transition_only_composition must be boolean")
        if not isinstance(matched_acquisition_batch, bool):
            raise TypeError("matched_acquisition_batch must be boolean")
        if not isinstance(state, ProceduralSkillState):
            raise TypeError("state must be a ProceduralSkillState")

        items = tuple(public_task.items)
        if len(items) != _ITEM_COUNT:
            raise ValueError(f"public task must have exactly {_ITEM_COUNT} items")
        if type(public_task.public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        batch_size = state.batch_size
        is_composition = bool(public_task.request.children)
        reversible_mode = bool(self.reversible_transition_mode.item())
        if batch_size != 1:
            if batch_size != 3:
                raise ValueError(
                    "batched task scoring requires exactly three matched state rows"
                )
            if not reversible_mode:
                raise ValueError(
                    "batched task scoring requires reversible transition mode"
                )
            if not transition_only_composition or (
                not is_composition and not matched_acquisition_batch
            ):
                raise ValueError(
                    "batched task scoring requires transition-only composition"
                )
            if not include_reversible_transition:
                raise ValueError(
                    "batched task scoring supports only transition-enabled rows"
                )
            if (
                not include_compiler
                or not include_phase4_bridge
                or not include_reverse_bridge
                or not include_descendants
                or probe_leaf_bridge
                or not include_frozen_transition
                or not include_fast_adapter
            ):
                raise ValueError(
                    "batched task scoring requires the complete default compiler path"
                )
            if bool(getattr(public_task, "demonstrations", ())):
                raise ValueError(
                    "batched task scoring requires a demonstration-free query"
                )
        elif matched_acquisition_batch:
            raise ValueError(
                "matched acquisition batching requires exactly three state rows"
            )
        reference = next(self.parameters())
        item_features = _public_item_features(
            items,
            public_task.public_flag,
            device=reference.device,
            dtype=reference.dtype,
        )
        # Experimental front ends may attach one shared, family-neutral
        # public-fact adapter after loading a frozen checkpoint.  The base
        # policy has no such module, so all retained checkpoints and ordinary
        # runs remain byte/behavior compatible.  The adapter receives public
        # facts only and must preserve the fixed [item, feature] contract.
        public_fact_adapter = getattr(self, "public_fact_adapter", None)
        adapter_applies = getattr(public_fact_adapter, "applies_to", None)
        adapter_active = public_fact_adapter is not None and (
            adapter_applies is None or bool(adapter_applies(public_task))
        )
        public_state_gate = getattr(
            public_fact_adapter,
            "reads_public_evidence_state",
            None,
        )
        public_evidence_state_active = bool(
            adapter_active
            and public_state_gate is not None
            and public_state_gate(public_task)
            and getattr(
                self.composition_memory,
                "public_evidence_reader",
                None,
            )
            is not None
        )
        if adapter_active:
            encode_public_task = getattr(
                public_fact_adapter,
                "encode_public_task",
                None,
            )
            item_features = (
                encode_public_task(public_task, item_features)
                if encode_public_task is not None
                else public_fact_adapter(item_features)
            )
            if item_features.shape != (_ITEM_COUNT, _ITEM_FEATURE_WIDTH):
                raise ValueError(
                    "public fact adapter must preserve the item-feature shape"
                )
            if not bool(torch.isfinite(item_features).all().item()):
                raise ValueError("public fact adapter produced non-finite features")
        flag_feature = torch.tensor(
            (float(public_task.public_flag),),
            device=reference.device,
            dtype=reference.dtype,
        )

        def encode_view(
            item_encoder: nn.Module,
            state_encoder: nn.Module,
            candidate_encoder: nn.Module,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            item_embeddings = item_encoder(item_features)
            state_embedding = state_encoder(
                torch.cat((item_embeddings.mean(dim=0), flag_feature)).unsqueeze(0)
            )
            ordered_items = item_embeddings[self.candidate_permutations]
            candidate_embeddings = candidate_encoder(
                ordered_items.reshape(len(_PERMUTATIONS), -1)
            ).unsqueeze(0)
            if batch_size != 1:
                state_embedding = state_embedding.expand(batch_size, -1)
                candidate_embeddings = candidate_embeddings.expand(
                    batch_size,
                    -1,
                    -1,
                )
            return state_embedding, candidate_embeddings

        leaf_view = encode_view(
            self.item_encoder,
            self.state_encoder,
            self.candidate_encoder,
        )
        composition_view = encode_view(
            self.composition_item_encoder,
            self.composition_state_encoder,
            self.composition_candidate_encoder,
        )

        nodes: list[NodePolicyRead] = []
        node_operator_codes: dict[tuple[int, ...], torch.Tensor] = {}
        node_availability: dict[tuple[int, ...], torch.Tensor] = {}
        node_incoming_sources: dict[tuple[int, ...], torch.Tensor] = {}
        node_successors: dict[tuple[int, ...], torch.Tensor] = {}
        node_execution_enabled: dict[tuple[int, ...], bool] = {}
        node_branch_weights: dict[tuple[int, ...], torch.Tensor] = {}
        node_executed_branch_weights: dict[tuple[int, ...], torch.Tensor] = {}
        node_execution_ties: dict[tuple[int, ...], torch.Tensor] = {}
        bridge_requested = is_composition or probe_leaf_bridge
        # Structural execution and learned child-policy routing are distinct
        # from the Phase-4 action-space evidence seam.  The Phase-4 ablation
        # must remove only that evidence, never the newly acquired selector.
        execution_enabled = include_compiler and bridge_requested
        compiler_source = (
            self.compiler_source_bridge(composition_view[0])
            if execution_enabled
            else torch.zeros_like(composition_view[0])
        )
        compiler_candidate_states = (
            self.compiler_successor_bridge(composition_view[1][0])
            if execution_enabled
            else None
        )
        def apply_local_transition(
            source: torch.Tensor,
            procedure_code: torch.Tensor,
            *,
            reverse: bool = False,
            public_transition_gate: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Apply one acquired node in either frozen Phase-4 direction."""

            if reversible_mode:
                if not include_reversible_transition:
                    return source
                operator = self.compiler_operator_bridge(procedure_code)
                null_operator = self.compiler_operator_bridge(
                    torch.zeros_like(procedure_code)
                )
                reversible_condition = torch.cat(
                    (procedure_code, operator - null_operator),
                    dim=-1,
                )
                if source.ndim == 3:
                    reversible_condition = reversible_condition.unsqueeze(1)
                    if public_transition_gate is not None:
                        public_transition_gate = public_transition_gate.unsqueeze(1)
                return self.reversible_procedure_transition(
                    source,
                    reversible_condition,
                    reverse=reverse,
                    post_tanh_gate_residual=public_transition_gate,
                )
            stable_delta = torch.zeros_like(source)
            if include_frozen_transition:
                operator = self.compiler_operator_bridge(procedure_code)
                null_operator = self.compiler_operator_bridge(
                    torch.zeros_like(procedure_code)
                )
                predicted = self.stable_compiler.core.predict_effects(
                    source,
                    operator,
                    reverse=reverse,
                )[:, 0, :]
                null_predicted = self.stable_compiler.core.predict_effects(
                    source,
                    null_operator,
                    reverse=reverse,
                )[:, 0, :]
                stable_delta = predicted - null_predicted
            fast_delta = torch.zeros_like(source)
            goal_delta = torch.zeros_like(source)
            if include_fast_adapter:
                fast_delta = self.procedural_fast_adapter(
                    source,
                    procedure_code,
                    reverse=reverse,
                )
                if not reverse:
                    if compiler_candidate_states is None:
                        raise RuntimeError(
                            "goal projection requires compiler candidate states"
                        )
                    factors = self.procedural_fast_adapter.latent_factors(
                        source,
                        procedure_code,
                    )
                    goal_delta, _ = self.procedural_goal_projection(
                        source,
                        compiler_candidate_states,
                        factors,
                    )
            # Phase 4 is residual.  Subtracting its same-source null transition
            # makes an all-zero acquired code an exact identity operation.
            return source + stable_delta + fast_delta + goal_delta

        def reanchor_intermediate(
            source: torch.Tensor,
            successor: torch.Tensor,
        ) -> torch.Tensor:
            """Softly return a model-generated step to public state geometry."""

            if reversible_mode:
                return successor
            if compiler_candidate_states is None:
                raise RuntimeError("intermediate re-anchoring requires compiler states")
            return _soft_reanchor_intermediate(
                source,
                successor,
                compiler_candidate_states,
            )

        def visit(
            expression: Any,
            path: tuple[int, ...],
            depth: int,
            incoming_source: torch.Tensor,
            *,
            execute: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            children = tuple(expression.children)
            if len(children) > 2:
                raise ValueError("skill expressions may have at most two children")
            memory_tier = "composition" if children else "leaf"
            state_embedding, candidate_embeddings = (
                composition_view if memory_tier == "composition" else leaf_view
            )
            goal_embedding = self._goal_embedding(
                expression.symbol,
                len(children),
                # A learned primitive keeps the same associative key when it
                # later appears deeper in a novel composition.  Tree position
                # is represented by the ordered recursive composer itself.
                0,
                reference,
                memory_tier,
            )
            if batch_size != 1:
                goal_embedding = goal_embedding.expand(batch_size, -1)
            memory_module = self.memory_for_tier(memory_tier)
            read = memory_module.read(
                state_embedding,
                goal_embedding,
                candidate_embeddings,
                state=state,
                include_public_evidence=(
                    public_evidence_state_active
                    and getattr(memory_module, "public_evidence_reader", None)
                    is not None
                ),
            )
            node_incoming_sources[path] = incoming_source
            node_execution_enabled[path] = execute
            use_children = include_descendants or bool(path)
            child_results = tuple(
                visit(
                    child,
                    path + (index,),
                    depth + 1,
                    incoming_source,
                    execute=execute and use_children,
                )
                for index, child in enumerate(children)
            )
            child_successors = tuple(item[0] for item in child_results)
            child_availability = tuple(item[1] for item in child_results)

            # Leaf and unary nodes are state transitions.  Unary children feed
            # their successor into the parent.  Binary nodes are conditional
            # combinators: both alternatives start from the same source and a
            # shared, candidate-blind router selects/mixes their successors.
            predecessor = incoming_source
            if execute and use_children and len(children) == 1:
                predecessor = reanchor_intermediate(
                    incoming_source,
                    child_successors[0],
                )
            elif execute and use_children and len(children) == 2:
                branch_logits = self.relational_branch_router(
                    read.plastic_context,
                    public_task.public_flag,
                )
                (
                    branch_weights,
                    executed_branch_weights,
                    execution_tied,
                ) = _execute_binary_branch(
                    branch_logits,
                )
                if reversible_mode:
                    hard_branch_weights = F.one_hot(
                        branch_weights.argmax(dim=-1),
                        num_classes=2,
                    ).to(dtype=branch_weights.dtype)
                    executed_branch_weights = hard_branch_weights + (
                        branch_weights - branch_weights.detach()
                    )
                node_branch_weights[path] = branch_weights
                node_executed_branch_weights[path] = executed_branch_weights
                node_execution_ties[path] = execution_tied
                predecessor = child_successors[0] + (
                    executed_branch_weights[:, 1:2]
                    * (child_successors[1] - child_successors[0])
                )
            operator_code = read.plastic_context
            node_operator_codes[path] = operator_code

            # The memory read already applies smooth count confidence to its
            # plastic context.  This is only an exact presence gate: a node or
            # required child with no evidence makes that subtree an identity.
            available = (read.evidence_count > 0).to(
                dtype=read.plastic_context.dtype
            )
            if execute and use_children:
                for child_available in child_availability:
                    available = available * child_available
            node_availability[path] = available
            if execute:
                # A binary node is a learned branch/combinator: its acquired
                # code already controls the candidate-blind router.  Applying
                # another effect after routing would turn selection into an
                # unrelated decoder.  Leaf and unary nodes remain ordinary
                # state transitions.
                transitioned = (
                    predecessor
                    if len(children) == 2
                    else apply_local_transition(
                        predecessor,
                        operator_code,
                        public_transition_gate=read.public_transition_gate,
                    )
                )
                successor = torch.where(
                    available.unsqueeze(-1).to(dtype=torch.bool),
                    transitioned,
                    incoming_source,
                )
            else:
                successor = incoming_source
            node_successors[path] = successor
            recursive_predecessor = (
                predecessor
                if children and execute
                else torch.zeros_like(read.plastic_context)
            )
            feedback_context = recursive_predecessor
            feedback_available = torch.ones(
                read.evidence_count.shape,
                device=read.evidence_count.device,
                dtype=torch.bool,
            )
            branch_weights = torch.zeros(
                (read.plastic_context.shape[0], 2),
                device=read.plastic_context.device,
                dtype=read.plastic_context.dtype,
            )
            executed_branch_weights = torch.zeros_like(branch_weights)
            execution_tied = torch.zeros(
                read.evidence_count.shape,
                device=read.evidence_count.device,
                dtype=torch.bool,
            )
            candidate_branch_advantages = torch.zeros(
                (read.plastic_context.shape[0], len(_PERMUTATIONS)),
                device=read.plastic_context.device,
                dtype=read.plastic_context.dtype,
            )
            child_candidate_scores = torch.zeros(
                (read.plastic_context.shape[0], 2, len(_PERMUTATIONS)),
                device=read.plastic_context.device,
                dtype=read.plastic_context.dtype,
            )
            conditioned_child_candidate_scores = torch.zeros(
                (read.plastic_context.shape[0], 2, len(_PERMUTATIONS)),
                device=read.plastic_context.device,
                dtype=read.plastic_context.dtype,
            )
            if len(children) == 2 and execute and use_children:
                signed_flag = incoming_source.new_tensor(
                    1.0 if public_task.public_flag else -1.0
                )
                feedback_context = signed_flag * (
                    child_successors[1] - child_successors[0]
                )
                branch_weights = node_branch_weights[path]
                executed_branch_weights = node_executed_branch_weights[path]
                execution_tied = node_execution_ties[path]
                for child_available in child_availability:
                    feedback_available = feedback_available & (
                        child_available > 0
                    )
                successor_scale = torch.maximum(
                    child_successors[0].abs().amax(dim=-1),
                    child_successors[1].abs().amax(dim=-1),
                ).clamp_min(1.0)
                distinct_tolerance = (
                    32.0
                    * torch.finfo(feedback_context.dtype).eps
                    * successor_scale
                )
                if not reversible_mode:
                    feedback_available = feedback_available & (
                        feedback_context.abs().amax(dim=-1)
                        > distinct_tolerance
                    )
            elif len(children) == 2:
                feedback_available = torch.zeros_like(feedback_available)
            subtree_context = successor - incoming_source
            nodes.append(
                NodePolicyRead(
                    path=path,
                    child_count=len(children),
                    state_embedding=state_embedding,
                    goal_embedding=goal_embedding,
                    candidate_embeddings=candidate_embeddings,
                    memory_read=read,
                    recursive_predecessor=recursive_predecessor,
                    feedback_context=feedback_context,
                    feedback_available=feedback_available,
                    candidate_branch_advantages=candidate_branch_advantages,
                    child_candidate_scores=child_candidate_scores,
                    conditioned_child_candidate_scores=(
                        conditioned_child_candidate_scores
                    ),
                    branch_weights=branch_weights,
                    executed_branch_weights=executed_branch_weights,
                    execution_tied=execution_tied,
                    subtree_context=subtree_context,
                    memory_tier=memory_tier,
                )
            )
            return successor, available

        root_successor, root_available = visit(
            public_task.request,
            (),
            0,
            compiler_source,
            execute=execution_enabled,
        )

        def reverse_visit(
            expression: Any,
            path: tuple[int, ...],
            target: torch.Tensor,
            *,
            execute: bool,
        ) -> torch.Tensor:
            """Reconstruct a source from every candidate destination.

            Traversal reverses the actual forward structure.  Candidate rows
            all use the same acquired codes and the binary routing weights
            cached from the public-input rollout, so reverse evidence cannot
            become a candidate-index answer decoder.
            """

            if not execute:
                return target
            children = tuple(expression.children)
            use_children = include_descendants or bool(path)
            procedure_code = node_operator_codes[path]
            if len(children) == 2:
                if use_children:
                    branch_weights = node_executed_branch_weights[path]
                    child_sources = tuple(
                        reverse_visit(
                            child,
                            path + (index,),
                            target,
                            execute=True,
                        )
                        for index, child in enumerate(children)
                    )
                    branch_contribution = branch_weights[:, 1:2]
                    if batch_size != 1:
                        branch_contribution = branch_contribution.unsqueeze(-1)
                    reconstructed = child_sources[0] + (
                        branch_contribution
                        * (child_sources[1] - child_sources[0])
                    )
                else:
                    reconstructed = target
            else:
                reconstructed = apply_local_transition(
                    target,
                    procedure_code,
                    reverse=True,
                    public_transition_gate=(
                        nodes_by_path[path].memory_read.public_transition_gate
                    ),
                )
                if len(children) == 1 and use_children:
                    reconstructed = reverse_visit(
                        children[0],
                        path + (0,),
                        reconstructed,
                        execute=True,
                    )
            available = node_availability[path]
            availability_mask = available.unsqueeze(-1)
            if batch_size != 1:
                availability_mask = availability_mask.unsqueeze(-1)
            return torch.where(
                availability_mask.to(dtype=torch.bool),
                reconstructed,
                target,
            )

        root_context = root_successor - compiler_source
        nodes_by_path = {node.path: node for node in nodes}
        phase4_components_by_path: dict[
            tuple[int, ...],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        action_policy_by_path: dict[tuple[int, ...], torch.Tensor] = {}

        def local_phase4_components(
            expression: Any,
            path: tuple[int, ...],
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            """Score one acquired subtree in the shared Phase-4 action space."""

            cached = phase4_components_by_path.get(path)
            if cached is not None:
                return cached
            node = nodes_by_path[path]
            # A binary node's generic utility decoder is never an action path.
            # Its action policy is assembled recursively from its children.
            base_bias = (
                torch.zeros_like(node.memory_read.score_bias)
                if node.child_count == 2
                else node.memory_read.score_bias
            )
            bridge_logits = torch.zeros_like(base_bias)
            forward_evidence = torch.zeros_like(base_bias)
            reverse_evidence = torch.zeros_like(base_bias)
            direction_gains = torch.ones(
                (*base_bias.shape, 2),
                device=base_bias.device,
                dtype=base_bias.dtype,
            )
            if include_compiler and bridge_requested and node_execution_enabled[path]:
                if self.stable_compiler.core.width != self.profile.width:
                    raise RuntimeError(
                        "Phase-4 compiler and skill memory widths differ"
                    )
                if include_phase4_bridge:
                    if compiler_candidate_states is None:
                        raise RuntimeError(
                            "compiler candidate states were not constructed"
                        )
                    if batch_size == 1:
                        successor_candidates = compiler_candidate_states
                        predicted = node_successors[path][0]
                        null_prediction = node_incoming_sources[path][0]
                        predicted_scores = -(
                            (successor_candidates - predicted.unsqueeze(0))
                            .square()
                            .mean(dim=-1)
                        )
                        null_scores = -(
                            (successor_candidates - null_prediction.unsqueeze(0))
                            .square()
                            .mean(dim=-1)
                        )
                    else:
                        successor_candidates = compiler_candidate_states.unsqueeze(
                            0
                        ).expand(batch_size, -1, -1)
                        predicted = node_successors[path].unsqueeze(1)
                        null_prediction = node_incoming_sources[path].unsqueeze(1)
                        predicted_scores = -(
                            (successor_candidates - predicted).square().mean(dim=-1)
                        )
                        null_scores = -(
                            (successor_candidates - null_prediction)
                            .square()
                            .mean(dim=-1)
                        )
                    forward_delta = predicted_scores - null_scores
                    if include_reverse_bridge:
                        reverse_sources = reverse_visit(
                            expression,
                            path,
                            successor_candidates,
                            execute=True,
                        )
                        source_targets = node_incoming_sources[path]
                        if batch_size != 1:
                            source_targets = source_targets.unsqueeze(1)
                        source_targets = source_targets.expand_as(reverse_sources)
                        reverse_scores = -(
                            (reverse_sources - source_targets).square().mean(dim=-1)
                        )
                        reverse_delta = reverse_scores - null_scores
                    else:
                        reverse_delta = torch.zeros_like(forward_delta)
                    memory_module = self.memory_for_tier(node.memory_tier)
                    availability = node_availability[path].unsqueeze(-1)
                    if reversible_mode:
                        if batch_size == 1:
                            centered_forward = forward_delta - forward_delta.mean()
                            centered_reverse = reverse_delta - reverse_delta.mean()
                            forward_evidence = memory_module.score_limit * torch.tanh(
                                centered_forward
                            ).unsqueeze(0)
                            reverse_evidence = memory_module.score_limit * torch.tanh(
                                centered_reverse
                            ).unsqueeze(0)
                            symmetric_delta = 0.5 * (
                                centered_forward + centered_reverse
                            )
                            symmetric_delta = (
                                symmetric_delta - symmetric_delta.mean()
                            )
                            bridge_logits = memory_module.score_limit * torch.tanh(
                                symmetric_delta
                            ).unsqueeze(0)
                        else:
                            centered_forward = forward_delta - forward_delta.mean(
                                dim=-1,
                                keepdim=True,
                            )
                            centered_reverse = reverse_delta - reverse_delta.mean(
                                dim=-1,
                                keepdim=True,
                            )
                            forward_evidence = memory_module.score_limit * torch.tanh(
                                centered_forward
                            )
                            reverse_evidence = memory_module.score_limit * torch.tanh(
                                centered_reverse
                            )
                            symmetric_delta = 0.5 * (
                                centered_forward + centered_reverse
                            )
                            symmetric_delta = symmetric_delta - symmetric_delta.mean(
                                dim=-1,
                                keepdim=True,
                            )
                            bridge_logits = memory_module.score_limit * torch.tanh(
                                symmetric_delta
                            )
                    else:
                        (
                            bridge_logits,
                            forward_evidence,
                            reverse_evidence,
                            direction_gains,
                        ) = self._phase4_directional_evidence(
                            base_bias,
                            forward_delta,
                            reverse_delta,
                            memory_module.score_limit,
                        )
                    bridge_logits = bridge_logits * availability
                    forward_evidence = forward_evidence * availability
                    reverse_evidence = reverse_evidence * availability
                if reversible_mode:
                    reliability = torch.ones_like(base_bias)
                else:
                    score_limit = self.memory_for_tier(node.memory_tier).score_limit
                    normalized_base = F.softsign(base_bias / score_limit)
                    normalized_phase4 = F.softsign(bridge_logits / score_limit)
                    reliability_features = torch.stack(
                        (
                            normalized_base,
                            normalized_phase4,
                            normalized_base * normalized_phase4,
                            (normalized_base - normalized_phase4).abs(),
                        ),
                        dim=-1,
                    )
                    # This gate can scale only candidate-local Phase-4 evidence.
                    reliability = 2.0 * torch.sigmoid(
                        self.phase4_reliability_gate(reliability_features)
                    ).squeeze(-1)
            else:
                reliability = torch.ones_like(base_bias)
            result = (
                bridge_logits,
                forward_evidence,
                reverse_evidence,
                direction_gains,
                reliability,
            )
            phase4_components_by_path[path] = result
            return result

        def subtree_action_policy(
            expression: Any,
            path: tuple[int, ...],
        ) -> torch.Tensor:
            """Return the action policy actually executed by this subtree.

            Leaf policies come directly from acquired memory.  Unary policies
            combine their own memory with their complete learned Phase-4
            transition.  Binary policies recursively execute one of those
            policies; their generic root decoder is never reopened.
            """

            cached = action_policy_by_path.get(path)
            if cached is not None:
                return cached
            node = nodes_by_path[path]
            children = tuple(expression.children)
            use_children = include_descendants or bool(path)
            if not children:
                policy_logits = node.memory_read.score_bias
            elif len(children) == 1:
                bridge_logits, _, _, _, reliability = local_phase4_components(
                    expression,
                    path,
                )
                direct_bias = (
                    torch.zeros_like(node.memory_read.score_bias)
                    if transition_only_composition and is_composition
                    else node.memory_read.score_bias
                )
                policy_logits = direct_bias + (
                    reliability * bridge_logits
                ) * node_availability[path].unsqueeze(-1)
            elif node_execution_enabled[path] and use_children:
                child_policies = torch.stack(
                    tuple(
                        subtree_action_policy(child, path + (index,))
                        for index, child in enumerate(children)
                    ),
                    dim=1,
                )
                signed_flag = child_policies.new_tensor(
                    1.0 if public_task.public_flag else -1.0
                )
                conditioned_children = signed_flag * child_policies.detach()
                advantages = conditioned_children[:, 1] - conditioned_children[:, 0]
                centered_advantages = advantages - advantages.mean(
                    dim=-1,
                    keepdim=True,
                )
                centered_children = conditioned_children - conditioned_children.mean(
                    dim=-1,
                    keepdim=True,
                )
                advantage_scale = child_policies.abs().amax(dim=-1).amax(
                    dim=1
                ).clamp_min(1.0)
                advantage_tolerance = (
                    32.0
                    * torch.finfo(centered_advantages.dtype).eps
                    * advantage_scale
                )
                feedback_available = node.feedback_available & (
                    centered_advantages.abs().amax(dim=-1) > advantage_tolerance
                )
                feedback_available = feedback_available & (
                    centered_children.abs().amax(dim=-1).min(dim=1).values
                    > advantage_tolerance
                )
                node = replace(
                    node,
                    feedback_available=feedback_available,
                    candidate_branch_advantages=advantages,
                    child_candidate_scores=child_policies,
                    conditioned_child_candidate_scores=conditioned_children,
                )
                nodes_by_path[path] = node
                policy_logits = (
                    node.executed_branch_weights.unsqueeze(-1) * child_policies
                ).sum(dim=1) * node_availability[path].unsqueeze(-1)
            else:
                # A binary root with descendant execution disabled has no
                # generic fallback decoder and therefore emits no policy.
                policy_logits = torch.zeros_like(node.memory_read.score_bias)
            action_policy_by_path[path] = policy_logits
            return policy_logits

        def refresh_binary_nodes(expression: Any, path: tuple[int, ...]) -> None:
            children = tuple(expression.children)
            for index, child in enumerate(children):
                refresh_binary_nodes(child, path + (index,))
            if len(children) == 2:
                subtree_action_policy(expression, path)

        # Refresh all binary credit records, including a binary child nested
        # beneath a unary parent, then compute the actual root action policy.
        refresh_binary_nodes(public_task.request, ())
        root_action_policy = subtree_action_policy(public_task.request, ())
        nodes = sorted(
            nodes_by_path.values(),
            key=lambda node: (len(node.path), node.path),
        )
        root = nodes[0]
        public_feedback_evidence = torch.zeros_like(root.feedback_context)
        if adapter_active:
            feedback_evidence = getattr(
                public_fact_adapter,
                "feedback_evidence",
                None,
            )
            if feedback_evidence is not None:
                public_feedback_evidence = feedback_evidence(
                    public_task,
                    root.feedback_context,
                )
                if public_feedback_evidence.shape != root.feedback_context.shape:
                    raise ValueError(
                        "public feedback evidence must match root context shape"
                    )
                if not bool(
                    torch.isfinite(public_feedback_evidence).all().item()
                ):
                    raise ValueError("public feedback evidence must be finite")
        # A binary procedure is a learned relational selector, not another
        # direct answer decoder.  Its plastic memory may influence output only
        # through the candidate-blind branch router and executed child
        # procedures.
        memory_bias = (
            torch.zeros_like(root.memory_read.score_bias)
            if (
                root.child_count == 2
                or (reversible_mode and is_composition)
                or (transition_only_composition and is_composition)
            )
            else root.memory_read.score_bias
        )
        binary_policy_logits = (
            root_action_policy
            if root.child_count == 2
            else torch.zeros_like(memory_bias)
        )
        if include_compiler and bridge_requested:
            (
                phase4_bridge_logits,
                phase4_forward_evidence,
                phase4_reverse_evidence,
                phase4_direction_gains,
                phase4_reliability,
            ) = local_phase4_components(public_task.request, ())
            composition_logits = (
                binary_policy_logits
                if root.child_count == 2
                else (
                    phase4_reliability * phase4_bridge_logits
                ) * root_available.unsqueeze(-1)
            )
            # Atomic probes align their acquired code to the shared operator
            # geometry without changing the already-validated leaf policy.
            # Only an actual composition consumes Phase-4 evidence as output.
            logits = memory_bias + composition_logits if is_composition else memory_bias
        else:
            composition_logits = torch.zeros_like(memory_bias)
            phase4_bridge_logits = torch.zeros_like(memory_bias)
            phase4_forward_evidence = torch.zeros_like(memory_bias)
            phase4_reverse_evidence = torch.zeros_like(memory_bias)
            phase4_direction_gains = torch.ones(
                (*memory_bias.shape, 2),
                device=memory_bias.device,
                dtype=memory_bias.dtype,
            )
            phase4_reliability = torch.ones_like(memory_bias)
            logits = memory_bias
        if logits.shape != (batch_size, len(_PERMUTATIONS)):
            raise RuntimeError("policy produced an invalid candidate score shape")
        if not bool(torch.isfinite(logits).all().item()):
            raise _NonFinitePolicyScoresError(
                "policy produced non-finite candidate scores"
            )
        return PolicyScores(
            logits=logits,
            candidate_embeddings=root.candidate_embeddings,
            nodes=tuple(nodes),
            root_context=root_context,
            composition_logits=composition_logits,
            phase4_bridge_logits=phase4_bridge_logits,
            phase4_forward_evidence=phase4_forward_evidence,
            phase4_reverse_evidence=phase4_reverse_evidence,
            phase4_direction_gains=phase4_direction_gains,
            phase4_reliability=phase4_reliability,
            memory_bias=memory_bias,
            binary_policy_logits=binary_policy_logits,
            root_available=root_available,
            public_feedback_evidence=public_feedback_evidence,
        )

    def _goal_embedding(
        self,
        symbol: str,
        arity: int,
        depth: int,
        reference: torch.Tensor,
        memory_tier: str,
    ) -> torch.Tensor:
        if not isinstance(symbol, str):
            raise TypeError("skill symbol must be text")
        if arity not in (0, 1, 2):
            raise ValueError("skill arity must be zero, one, or two")
        if not 0 <= depth <= 3:
            raise ValueError("skill expression depth exceeds the supported curriculum")
        symbol_row = self.symbol_features.encode_texts(
            (symbol,),
            device=reference.device,
            dtype=reference.dtype,
        )
        structure = torch.zeros((1, 7), device=reference.device, dtype=reference.dtype)
        structure[0, arity] = 1.0
        structure[0, 3 + depth] = 1.0
        encoder = (
            self.goal_encoder
            if memory_tier == "leaf"
            else self.composition_goal_encoder
        )
        return encoder(torch.cat((symbol_row, structure), dim=-1))


def propose_task(
    policy: SkillMemoryPolicy,
    public_task: Any,
    state: ProceduralSkillState,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
) -> TaskProposal:
    """Choose a public permutation without evaluator or routing identities."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scores = policy.score_task(public_task, state)
    sampling_probabilities = torch.softmax(scores.logits[0] / temperature, dim=-1)
    if greedy:
        candidate_index = int(scores.logits.argmax(dim=-1).item())
        behavior_probabilities = F.one_hot(
            torch.tensor(candidate_index, device=scores.logits.device),
            num_classes=len(_PERMUTATIONS),
        ).to(dtype=scores.logits.dtype)
    else:
        candidate_index = int(torch.multinomial(sampling_probabilities, 1).item())
        behavior_probabilities = sampling_probabilities.detach()
    permutation = _PERMUTATIONS[candidate_index]
    answer = tuple(public_task.items[index].symbol for index in permutation)
    return TaskProposal(
        answer,
        candidate_index,
        scores,
        behavior_probabilities,
        procedural_skill_state_digest(state),
        _public_task_digest(public_task),
    )


def _proposal_for_candidate(
    policy: SkillMemoryPolicy,
    public_task: Any,
    state: ProceduralSkillState,
    candidate_index: int,
    *,
    include_compiler: bool = True,
) -> TaskProposal:
    """Bind one externally fixed public candidate to a competence state.

    This is used only for matched offline counterfactual evidence sets.  It
    exposes neither an evaluator target nor a routing identity: the candidate
    was already selected by the ordinary learner trajectory.
    """

    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or not 0 <= candidate_index < len(_PERMUTATIONS)
    ):
        raise ValueError("candidate_index is outside the public candidate set")
    scores = policy.score_task(
        public_task,
        state,
        include_compiler=include_compiler,
    )
    permutation = _PERMUTATIONS[candidate_index]
    answer = tuple(public_task.items[index].symbol for index in permutation)
    behavior_probabilities = F.one_hot(
        torch.tensor(candidate_index, device=scores.logits.device),
        num_classes=len(_PERMUTATIONS),
    ).to(dtype=scores.logits.dtype)
    return TaskProposal(
        answer,
        candidate_index,
        scores,
        behavior_probabilities,
        procedural_skill_state_digest(state),
        _public_task_digest(public_task),
    )


def _canonical_binary_outcome_basis(
    proposal: TaskProposal,
    memory_module: RoutedProceduralMemory,
) -> torch.Tensor | None:
    """Return attempted-action relational credit in one canonical basis.

    Both acquired child policies are read before reward.  Centering and
    scaling each under the actual behavior policy removes confidence-scale and
    fixed-side bias; only the attempted action is gathered.  A deterministic
    or forced action has no identifiable relational variance and therefore
    supplies no binary write.
    """

    root = proposal.scores.root
    if root.child_count != 2:
        return None
    child_scores = root.conditioned_child_candidate_scores
    probabilities = proposal.behavior_probabilities
    if (
        child_scores.ndim != 3
        or child_scores.shape[:2] != (1, 2)
        or probabilities.shape != (child_scores.shape[2],)
        or probabilities.device != child_scores.device
        or probabilities.dtype != child_scores.dtype
        or not bool(torch.isfinite(child_scores).all().item())
        or not bool(torch.isfinite(probabilities).all().item())
        or bool((probabilities < 0.0).any().item())
        or not torch.allclose(
            probabilities.sum(),
            probabilities.new_tensor(1.0),
            atol=1e-6,
            rtol=0.0,
        )
    ):
        raise RuntimeError("binary proposal has invalid behavior-policy evidence")
    probability_rows = probabilities.view(1, 1, -1)
    child_means = (probability_rows * child_scores).sum(
        dim=-1,
        keepdim=True,
    )
    centered_children = child_scores - child_means
    child_variances = (probability_rows * centered_children.square()).sum(
        dim=-1,
        keepdim=True,
    )
    child_scale_reference = child_scores.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    child_tolerance = (
        32.0 * torch.finfo(child_scores.dtype).eps * child_scale_reference
    )
    if not bool((child_variances.sqrt() > child_tolerance).all().item()):
        return None
    normalized_children = centered_children / child_variances.sqrt()
    advantages = normalized_children[:, 1] - normalized_children[:, 0]
    probability_rows = probabilities.unsqueeze(0)
    mean = (probability_rows * advantages).sum(dim=-1, keepdim=True)
    centered = advantages - mean
    variance = (probability_rows * centered.square()).sum(
        dim=-1,
        keepdim=True,
    )
    scale_reference = advantages.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    tolerance = (
        32.0 * torch.finfo(advantages.dtype).eps * scale_reference
    )
    if not bool((variance.sqrt() > tolerance).all().item()):
        return None
    standardized = centered / variance.sqrt()
    bounded = torch.tanh(standardized)
    # A nonlinear bound can destroy the zero-mean property even when its
    # input is centered.  Re-center after bounding, then normalize by the
    # largest action magnitude so constant or action-independent feedback
    # cannot accumulate a fixed-side polarity under a skewed policy.
    bounded = bounded - (probability_rows * bounded).sum(
        dim=-1,
        keepdim=True,
    )
    bounded_scale = bounded.abs().amax(dim=-1, keepdim=True)
    if not bool((bounded_scale > torch.finfo(bounded.dtype).eps).all().item()):
        return None
    canonical_actions = bounded / bounded_scale
    selected = canonical_actions[:, proposal.candidate_index]
    canonical = selected.unsqueeze(-1)
    return canonical.expand(-1, memory_module.evidence_outcome_width).detach()


def _canonical_binary_outcome_basis_rows(
    scores: PolicyScores,
    behavior_probabilities: torch.Tensor,
    candidate_indices: torch.Tensor,
    memory_module: RoutedProceduralMemory,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-local binary credit and validity for a matched batch."""

    root = scores.root
    child_scores = root.conditioned_child_candidate_scores
    batch = scores.logits.shape[0]
    candidates = scores.logits.shape[1]
    if root.child_count != 2:
        raise ValueError("matched binary evidence requires a binary root")
    if (
        child_scores.shape != (batch, 2, candidates)
        or behavior_probabilities.shape != (batch, candidates)
        or behavior_probabilities.device != child_scores.device
        or behavior_probabilities.dtype != child_scores.dtype
        or candidate_indices.shape != (batch,)
        or candidate_indices.dtype != torch.long
        or candidate_indices.device != child_scores.device
        or not bool(torch.isfinite(child_scores).all().item())
        or not bool(torch.isfinite(behavior_probabilities).all().item())
        or bool((behavior_probabilities < 0.0).any().item())
        or not torch.allclose(
            behavior_probabilities.sum(dim=-1),
            behavior_probabilities.new_ones((batch,)),
            atol=1.0e-6,
            rtol=0.0,
        )
        or bool(
            ((candidate_indices < 0) | (candidate_indices >= candidates))
            .any()
            .item()
        )
    ):
        raise RuntimeError("matched binary proposal evidence is invalid")

    probability_rows = behavior_probabilities.unsqueeze(1)
    child_means = (probability_rows * child_scores).sum(
        dim=-1,
        keepdim=True,
    )
    centered_children = child_scores - child_means
    child_variances = (probability_rows * centered_children.square()).sum(
        dim=-1,
        keepdim=True,
    )
    child_scale_reference = child_scores.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    child_tolerance = (
        32.0 * torch.finfo(child_scores.dtype).eps * child_scale_reference
    )
    child_scales = child_variances.sqrt()
    valid = (child_scales > child_tolerance).all(dim=1).squeeze(-1)
    normalized_children = centered_children / child_scales.clamp_min(
        torch.finfo(child_scores.dtype).eps
    )
    advantages = normalized_children[:, 1] - normalized_children[:, 0]
    mean = (behavior_probabilities * advantages).sum(dim=-1, keepdim=True)
    centered = advantages - mean
    variance = (behavior_probabilities * centered.square()).sum(
        dim=-1,
        keepdim=True,
    )
    scale_reference = advantages.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    tolerance = 32.0 * torch.finfo(advantages.dtype).eps * scale_reference
    scale = variance.sqrt()
    valid = valid & (scale > tolerance).squeeze(-1)
    standardized = centered / scale.clamp_min(torch.finfo(centered.dtype).eps)
    bounded = torch.tanh(standardized)
    bounded = bounded - (behavior_probabilities * bounded).sum(
        dim=-1,
        keepdim=True,
    )
    bounded_scale = bounded.abs().amax(dim=-1, keepdim=True)
    valid = valid & (
        bounded_scale > torch.finfo(bounded.dtype).eps
    ).squeeze(-1)
    canonical_actions = bounded / bounded_scale.clamp_min(
        torch.finfo(bounded.dtype).eps
    )
    selected = canonical_actions.gather(1, candidate_indices.unsqueeze(-1))
    canonical = selected.expand(-1, memory_module.evidence_outcome_width)
    canonical = torch.where(
        valid.unsqueeze(-1),
        canonical,
        torch.zeros_like(canonical),
    ).detach()
    return canonical, valid


def propose_matched_differentiable_feedback(
    policy: SkillMemoryPolicy,
    scores: PolicyScores,
    candidate_index: int,
    behavior_probabilities: torch.Tensor,
    reward: float,
    state: ProceduralSkillState,
) -> _MatchedDifferentiableFeedback:
    """Stage three matched row-local writes without changing scalar APIs."""

    if not isinstance(scores, PolicyScores):
        raise TypeError("scores must be PolicyScores")
    if not isinstance(state, ProceduralSkillState) or state.batch_size != 3:
        raise ValueError("matched feedback requires exactly three state rows")
    if scores.logits.shape != (3, len(_PERMUTATIONS)):
        raise ValueError("matched feedback scores have the wrong shape")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or not 0 <= candidate_index < len(_PERMUTATIONS)
    ):
        raise ValueError("candidate_index is outside the public candidate set")
    if (
        not isinstance(behavior_probabilities, torch.Tensor)
        or behavior_probabilities.shape != (len(_PERMUTATIONS),)
        or behavior_probabilities.device != scores.logits.device
        or behavior_probabilities.dtype != scores.logits.dtype
        or not bool(torch.isfinite(behavior_probabilities).all().item())
        or bool((behavior_probabilities < 0.0).any().item())
        or not torch.allclose(
            behavior_probabilities.sum(),
            behavior_probabilities.new_tensor(1.0),
            atol=1.0e-6,
            rtol=0.0,
        )
    ):
        raise ValueError("matched behavior probabilities are invalid")

    numeric_reward = _validate_scalar_reward(reward)
    root = scores.root
    memory_module = policy.memory_for_tier(root.memory_tier)
    attempted = torch.full(
        (3,),
        candidate_index,
        device=scores.logits.device,
        dtype=torch.long,
    )
    probabilities = behavior_probabilities.unsqueeze(0).expand(3, -1)
    eligible = torch.ones((3,), device=scores.logits.device, dtype=torch.bool)
    outcome_direction_basis = None
    if root.child_count == 2:
        outcome_direction_basis, basis_valid = _canonical_binary_outcome_basis_rows(
            scores,
            probabilities,
            attempted,
            memory_module,
        )
        feedback_available = root.feedback_available.reshape(3).to(
            device=scores.logits.device,
            dtype=torch.bool,
        )
        eligible = feedback_available & basis_valid
        if not bool(eligible.any().item()):
            return _MatchedDifferentiableFeedback(
                candidate_state=state,
                write_slots=root.memory_read.write_slots.detach(),
                delta_norms=scores.logits.new_zeros((3,)),
            )

    rewards = scores.logits.new_full((3,), numeric_reward)
    base_logits = scores.logits - root.memory_read.score_bias
    structural_context = (
        torch.zeros_like(root.feedback_context)
        if root.child_count == 2
        else root.feedback_context
    )
    staged = memory_module.propose_feedback(
        root.state_embedding,
        root.goal_embedding,
        root.candidate_embeddings,
        attempted,
        rewards,
        base_logits,
        state=state,
        structural_context=structural_context,
        outcome_direction_basis=outcome_direction_basis,
        public_evidence=scores.public_feedback_evidence,
        include_public_evidence=root.memory_read.public_evidence_enabled,
    )
    if root.child_count == 2:
        staged = replace(
            staged,
            delta_norm=torch.where(
                eligible,
                staged.delta_norm,
                torch.zeros_like(staged.delta_norm),
            ),
        )
    committed = memory_module.commit_bounded_feedback(staged)
    return _MatchedDifferentiableFeedback(
        candidate_state=committed.state,
        write_slots=staged.write_slots.detach(),
        delta_norms=committed.delta_norm.detach(),
    )


def propose_differentiable_feedback(
    policy: SkillMemoryPolicy,
    proposal: TaskProposal,
    reward: float,
    state: ProceduralSkillState,
    *,
    validated_state_digest: str | None = None,
) -> DifferentiableFeedback:
    """Stage one root-skill write for an outer meta-gradient.

    ``validated_state_digest`` is an opt-in hot-path value.  A caller may pass
    it only after computing it from this exact current state and while treating
    the state as immutable.  Omitting it preserves the legacy independent
    digest computation and stale-proposal check.
    """

    numeric_reward = _validate_scalar_reward(reward)
    if validated_state_digest is None:
        current_state_digest = procedural_skill_state_digest(state)
    else:
        if not isinstance(validated_state_digest, str):
            raise TypeError("validated_state_digest must be text")
        current_state_digest = validated_state_digest
    if proposal.competence_digest != current_state_digest:
        raise ValueError("proposal is not bound to the supplied competence state")
    root = proposal.scores.root
    memory_module = policy.memory_for_tier(root.memory_tier)
    outcome_direction_basis = None
    if root.child_count == 2:
        outcome_direction_basis = _canonical_binary_outcome_basis(
            proposal,
            memory_module,
        )
    if root.child_count == 2 and (
        not bool(root.feedback_available.all().item())
        or outcome_direction_basis is None
    ):
        # A binary selector is meaningful only after both alternatives are
        # executable and distinct.  Refuse to turn root state/candidate/reward
        # correlations into a fixed-child shortcut while either child is
        # missing or both frozen child procedures predict the same successor.
        return DifferentiableFeedback(
            candidate_state=state,
            write_slot=int(root.memory_read.write_slots.item()),
            delta_norm=0.0,
            route_probabilities=root.memory_read.route_probabilities,
        )
    reward_tensor = proposal.scores.logits.new_tensor((numeric_reward,))
    attempted = torch.tensor(
        (proposal.candidate_index,),
        device=proposal.scores.logits.device,
        dtype=torch.long,
    )
    base_logits = proposal.scores.logits - root.memory_read.score_bias
    structural_context = (
        torch.zeros_like(root.feedback_context)
        if root.child_count == 2
        else root.feedback_context
    )
    staged = memory_module.propose_feedback(
        root.state_embedding,
        root.goal_embedding,
        root.candidate_embeddings,
        attempted,
        reward_tensor,
        base_logits,
        state=state,
        structural_context=structural_context,
        outcome_direction_basis=outcome_direction_basis,
        public_evidence=proposal.scores.public_feedback_evidence,
        include_public_evidence=root.memory_read.public_evidence_enabled,
    )
    # Meta-training must advance through the same bounded core transaction as
    # online use.  The accepted branch retains its differentiable state;
    # rejected/no-effect writes return the exact incoming bytes.
    committed = memory_module.commit_bounded_feedback(staged)
    return DifferentiableFeedback(
        candidate_state=committed.state,
        write_slot=int(staged.write_slots.item()),
        delta_norm=float(committed.delta_norm.detach().item()),
        route_probabilities=staged.read.route_probabilities,
    )


def apply_transactional_feedback(
    policy: SkillMemoryPolicy,
    public_task: Any,
    proposal: TaskProposal,
    reward: float,
    state: ProceduralSkillState,
) -> TransactionalFeedback:
    """Admit one scalar-feedback write or return the exact incoming state.

    The memory module commits the meta-learned update only when it produces a
    finite observable local effect.  The wrapper then recomputes the complete
    tree policy and rolls back on a non-finite result.  Immediate same-example
    improvement is recorded but is not required: useful procedural updates
    are trained for later varied instances rather than a one-step patch.
    """

    numeric_reward = _validate_scalar_reward(reward)
    if proposal.competence_digest != procedural_skill_state_digest(state):
        raise ValueError("proposal is not bound to the supplied competence state")
    if proposal.public_task_digest != _public_task_digest(public_task):
        raise ValueError("proposal is not bound to the supplied public task")
    root = proposal.scores.root
    memory_module = policy.memory_for_tier(root.memory_tier)
    outcome_direction_basis = None
    if root.child_count == 2:
        outcome_direction_basis = _canonical_binary_outcome_basis(
            proposal,
            memory_module,
        )
    if root.child_count == 2 and (
        not bool(root.feedback_available.all().item())
        or outcome_direction_basis is None
    ):
        before_loss = _scalar_feedback_loss(
            proposal.scores.logits,
            proposal.candidate_index,
            numeric_reward,
        )
        return TransactionalFeedback(
            state=state,
            accepted=False,
            write_slot=int(root.memory_read.write_slots.item()),
            delta_norm=0.0,
            before_loss=before_loss,
            after_loss=before_loss,
            core_accepted=False,
        )
    reward_tensor = proposal.scores.logits.new_tensor((numeric_reward,))
    attempted = torch.tensor(
        (proposal.candidate_index,),
        device=proposal.scores.logits.device,
        dtype=torch.long,
    )
    base_logits = proposal.scores.logits - root.memory_read.score_bias
    structural_context = (
        torch.zeros_like(root.feedback_context)
        if root.child_count == 2
        else root.feedback_context
    )
    staged = memory_module.propose_feedback(
        root.state_embedding,
        root.goal_embedding,
        root.candidate_embeddings,
        attempted,
        reward_tensor,
        base_logits,
        state=state,
        structural_context=structural_context,
        outcome_direction_basis=outcome_direction_basis,
        public_evidence=proposal.scores.public_feedback_evidence,
        include_public_evidence=root.memory_read.public_evidence_enabled,
    )
    write = memory_module.commit_bounded_feedback(staged)
    core_accepted = bool(write.accepted.item())
    before_loss = _scalar_feedback_loss(
        proposal.scores.logits,
        proposal.candidate_index,
        numeric_reward,
    )
    if core_accepted:
        try:
            rescored = policy.score_task(public_task, write.state)
        except _NonFinitePolicyScoresError:
            after_loss = before_loss
            accepted = False
        else:
            after_loss = _scalar_feedback_loss(
                rescored.logits,
                proposal.candidate_index,
                numeric_reward,
            )
            accepted = math.isfinite(after_loss)
    else:
        after_loss = before_loss
        accepted = False
    committed = write.state if accepted else state
    if not accepted and procedural_skill_state_digest(committed) != (
        procedural_skill_state_digest(state)
    ):
        raise RuntimeError("rejected transaction did not restore exact incoming state")
    return TransactionalFeedback(
        state=committed,
        accepted=accepted,
        write_slot=int(write.write_slots.item()),
        delta_norm=float(write.delta_norm.item()) if accepted else 0.0,
        before_loss=before_loss,
        after_loss=after_loss,
        core_accepted=core_accepted,
    )


def _public_item_features(
    items: Sequence[Any],
    public_flag: bool,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode attributes only; entity symbols and identities are excluded."""

    features = torch.zeros(
        (_ITEM_COUNT, _ITEM_FEATURE_WIDTH), device=device, dtype=dtype
    )
    for index, item in enumerate(items):
        if not 0 <= int(item.rank_a) < _ITEM_COUNT:
            raise ValueError("rank_a is outside the public item range")
        if not 0 <= int(item.rank_b) < _ITEM_COUNT:
            raise ValueError("rank_b is outside the public item range")
        if int(item.group) not in (0, 1):
            raise ValueError("group must be zero or one")
        features[index, int(item.rank_a)] = 1.0
        features[index, 5 + int(item.rank_b)] = 1.0
        features[index, 10 + int(item.group)] = 1.0
        features[index, 12] = float(bool(item.marked))
        features[index, 13] = float(public_flag)
    return features


def _public_task_digest(public_task: Any) -> str:
    """Bind a transaction to public bytes without exposing the digest as input."""

    canonicalizer = getattr(public_task, "to_canonical", None)
    if not callable(canonicalizer):
        raise TypeError("public task must expose canonical public content")
    encoded = json.dumps(
        canonicalizer(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"project-angler.public-skill-task.v1\x00" + encoded
    ).hexdigest()


def _validate_scalar_reward(reward: float) -> float:
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("reward must be a scalar number")
    numeric = float(reward)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("reward must be finite and between zero and one")
    return numeric


def _scalar_feedback_loss(
    logits: torch.Tensor,
    selected: int,
    reward: float,
) -> float:
    return float(_scalar_feedback_tensor(logits, selected, reward).item())


def _scalar_feedback_tensor(
    logits: torch.Tensor,
    selected: int,
    reward: float,
) -> torch.Tensor:
    selected_logit = logits[0, selected]
    target = logits.new_tensor(float(reward))
    return F.binary_cross_entropy_with_logits(selected_logit, target)


def _detached_state(state: ProceduralSkillState) -> ProceduralSkillState:
    return restore_procedural_skill_state(snapshot_procedural_skill_state(state))


def _parameter_identity_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256(b"project-angler.parameter-identities.v1\x00")
    for name, parameter in module.named_parameters():
        encoded = (
            f"{name}\x00{id(parameter)}\x00{parameter.data_ptr()}\x00"
            f"{tuple(parameter.shape)}\x00{parameter.dtype}\x00{parameter.device}"
        ).encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _named_state_fingerprint(
    module: nn.Module,
    *,
    include: Callable[[str], bool],
    domain: bytes,
) -> str:
    """Hash a declared state partition for exact consolidation checks."""

    digest = hashlib.sha256(domain + b"\x00")
    selected = 0
    for name, value in sorted(module.state_dict().items()):
        if not include(name):
            continue
        selected += 1
        tensor = value.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">I", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack(">I", tensor.ndim))
        digest.update(struct.pack(f">{tensor.ndim}Q", *tensor.shape))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    if not selected:
        raise RuntimeError("state fingerprint selected no tensors")
    return "sha256:" + digest.hexdigest()


def _is_composition_state(name: str) -> bool:
    return name.startswith(_COMPOSITION_TRAINABLE_PREFIXES)


def _is_harmonization_state(name: str) -> bool:
    return name.startswith(_HARMONIZATION_TRAINABLE_PREFIXES)


def _is_procedural_adapter_state(name: str) -> bool:
    return name.startswith("procedural_fast_adapter.")


def _is_procedural_adapter_trainable(name: str) -> bool:
    return name.startswith(_PROCEDURAL_ADAPTER_TRAINABLE_PREFIXES)


def _is_reverse_construction_state(name: str) -> bool:
    return name in _REVERSE_CONSTRUCTION_TRAINABLE_NAMES or name.startswith(
        _REVERSE_CONSTRUCTION_TRAINABLE_PREFIXES
    )


def _is_reverse_harmonization_state(name: str) -> bool:
    return name.startswith(_REVERSE_HARMONIZATION_TRAINABLE_PREFIXES)


def _is_procedural_coadaptation_state(name: str) -> bool:
    return _is_reverse_construction_state(name) or _is_reverse_harmonization_state(
        name
    )


def _is_reversible_transition_acquisition_state(name: str) -> bool:
    return name in _REVERSIBLE_TRANSITION_TRAINABLE_NAMES or name.startswith(
        _REVERSIBLE_TRANSITION_TRAINABLE_PREFIXES
    )


def _configure_stage_trainability(
    policy: SkillMemoryPolicy,
    stage: str,
) -> tuple[str, ...]:
    """Consolidate the leaf substrate while learning composition adapters."""

    if stage not in _TRAINING_STAGES:
        raise ValueError(f"training stage must be one of {_TRAINING_STAGES}")
    trainable: list[str] = []
    for name, parameter in policy.named_parameters():
        always_frozen = name.startswith(
            (
                "stable_compiler.",
                "memory.memory.",
                "composition_memory.memory.",
            )
        )
        if stage == "leaf_core":
            enabled = not always_frozen and not name.startswith(
                "composition_memory."
            )
        elif stage == "relational_acquisition":
            enabled = name.startswith(_RELATIONAL_ACQUISITION_PREFIXES)
        elif stage == "harmonization":
            # Preserve acquired memory, child policies, and relational branch
            # polarity. Only shared bounded evidence calibration may adapt;
            # the learned compiler bridges themselves remain consolidated.
            enabled = _is_harmonization_state(name)
        elif stage == "procedural_adapter":
            enabled = _is_procedural_adapter_trainable(name)
        elif stage == "reverse_construction":
            enabled = _is_reverse_construction_state(name)
        elif stage == "reverse_harmonization":
            enabled = _is_reverse_harmonization_state(name)
        elif stage == "procedural_coadaptation":
            enabled = _is_procedural_coadaptation_state(name)
        elif stage == "reversible_transition_acquisition":
            enabled = _is_reversible_transition_acquisition_state(name)
        else:
            enabled = not always_frozen and _is_composition_state(name)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("training stage selected no slow parameters")
    return tuple(trainable)


def _optimizer_identity_fingerprint(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
) -> str:
    by_id = {
        id(parameter): name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_parameters) != len(by_id) or {
        id(parameter) for parameter in optimizer_parameters
    } != set(by_id):
        raise RuntimeError("optimizer parameters are not exactly the policy slow weights")
    digest = hashlib.sha256(b"project-angler.optimizer-identities.v1\x00")
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(struct.pack(">I", group_index))
        for parameter in group["params"]:
            digest.update(by_id[id(parameter)].encode("utf-8") + b"\x00")
    return "sha256:" + digest.hexdigest()


def _load_training_partition(seed: int, instances_per_program: int) -> tuple[Any, Any]:
    from experiments.evaluators.skill_memory_suite import (
        make_skill_memory_meta_partition,
        score_skill_memory_answer,
    )

    return (
        make_skill_memory_meta_partition(
            seed, instances_per_program=instances_per_program
        ),
        score_skill_memory_answer,
    )


def _load_matched_descendant_queries(seed: int) -> tuple[Any, ...]:
    from experiments.evaluators.skill_memory_suite import (
        make_skill_memory_meta_matched_queries,
    )

    return make_skill_memory_meta_matched_queries(seed)


def _load_final_curriculum(
    seed: int,
    encounters_per_primitive: int,
    cases_per_component_probe: int,
    cases_per_composition: int,
) -> tuple[Any, Any]:
    # Calling this factory is the first operation that imports the sealed final
    # composition programs.  _evaluate verifies frozen slow weights first.
    from experiments.evaluators.skill_memory_suite import (
        make_skill_memory_composition_curriculum,
        score_skill_memory_answer,
    )

    return (
        make_skill_memory_composition_curriculum(
            seed,
            encounters_per_primitive=encounters_per_primitive,
            cases_per_component_probe=cases_per_component_probe,
            cases_per_composition=cases_per_composition,
        ),
        score_skill_memory_answer,
    )


def _load_final_binary_branch_grid(seed: int, cases_per_cell: int) -> Any:
    from experiments.evaluators.skill_memory_suite import (
        make_skill_memory_matched_binary_branch_grid,
    )

    return make_skill_memory_matched_binary_branch_grid(
        seed,
        cases_per_cell=cases_per_cell,
    )


def _group_evaluator_pairs(pairs: Sequence[Any]) -> list[list[Any]]:
    """Group repeated public request structures without evaluator identities."""

    grouped: dict[str, list[Any]] = {}
    for pair in pairs:
        key = json.dumps(
            pair.learner.request.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
        )
        grouped.setdefault(key, []).append(pair)
    return [grouped[key] for key in sorted(grouped)]


def _group_evaluator_pairs_by_root(pairs: Sequence[Any]) -> list[list[Any]]:
    """Group varied child contexts under their shared opaque root mechanism."""

    grouped: dict[str, list[Any]] = {}
    for pair in pairs:
        grouped.setdefault(pair.learner.request.symbol, []).append(pair)
    return [grouped[key] for key in sorted(grouped)]


def _judge_frozen_answer(pair: Any, answer: tuple[str, ...], judge: Callable[..., float]) -> float:
    return float(judge(pair.learner, pair.hidden, answer))


def _outer_target_candidate_index(pair: Any) -> int:
    """Evaluator-only target extraction, called only by the outer loss."""

    display_position = {
        item.symbol: index for index, item in enumerate(pair.learner.items)
    }
    target = tuple(
        display_position[symbol]
        for symbol in pair.hidden.generated.hidden.target_order
    )
    return _PERMUTATION_TO_INDEX[target]


def _outer_target_candidate_utilities(
    pair: Any,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return evaluator-only pairwise utility for every public candidate."""

    target = _PERMUTATIONS[_outer_target_candidate_index(pair)]
    target_pairs = tuple(
        (target[left], target[right])
        for left in range(_ITEM_COUNT)
        for right in range(left + 1, _ITEM_COUNT)
    )
    denominator = float(len(target_pairs))
    values: list[float] = []
    for candidate in _PERMUTATIONS:
        positions = {item: index for index, item in enumerate(candidate)}
        correct = sum(
            positions[first] < positions[second]
            for first, second in target_pairs
        )
        values.append(correct / denominator)
    return reference.new_tensor((values,))


def _outer_query_loss(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    evaluator_pair: Any,
    *,
    include_compiler: bool = True,
) -> torch.Tensor:
    scores = policy.score_task(
        evaluator_pair.learner,
        state,
        include_compiler=include_compiler,
    )
    return _outer_logits_loss(scores.logits, evaluator_pair)


def _outer_counterfactual_query_loss(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    evaluator_pair: Any,
    *,
    include_compiler: bool = True,
) -> torch.Tensor:
    """Fit the coherent procedure implied by complemented scalar outcomes."""

    scores = policy.score_task(
        evaluator_pair.learner,
        state,
        include_compiler=include_compiler,
    )
    utilities = _outer_target_candidate_utilities(evaluator_pair, scores.logits)
    target = _PERMUTATIONS[_outer_target_candidate_index(evaluator_pair)]
    reversed_target = _PERMUTATION_TO_INDEX[tuple(reversed(target))]
    return _outer_utility_loss(scores.logits, 1.0 - utilities, reversed_target)


def _outer_logits_loss(
    logits: torch.Tensor,
    evaluator_pair: Any,
) -> torch.Tensor:
    """Evaluator-only utility/ranking loss for already-produced logits."""

    utilities = _outer_target_candidate_utilities(
        evaluator_pair,
        logits,
    )
    return _outer_utility_loss(
        logits,
        utilities,
        _outer_target_candidate_index(evaluator_pair),
    )


def _outer_top_target_loss(
    logits: torch.Tensor,
    evaluator_pair: Any,
) -> torch.Tensor:
    """Rank one evaluator-owned target from a centered causal residual."""

    if logits.ndim != 2 or logits.shape != (1, len(_PERMUTATIONS)):
        raise ValueError("residual logits must have shape [1, candidate count]")
    target = torch.tensor(
        (_outer_target_candidate_index(evaluator_pair),),
        device=logits.device,
        dtype=torch.long,
    )
    return F.cross_entropy(logits, target)


def _outer_matched_descendant_loss(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    pair: Any,
    *,
    margin: float = 0.10,
    include_evidence_delta: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit two same-instance tasks whose child trees imply different answers."""

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("matched descendant margin must be finite and positive")
    if not isinstance(include_evidence_delta, bool):
        raise TypeError("include_evidence_delta must be a bool")
    left = pair.left
    right = pair.right
    left_scores = policy.score_task(left.learner, state)
    right_scores = policy.score_task(right.learner, state)
    if (
        float(left_scores.root_available.item()) != 1.0
        or float(right_scores.root_available.item()) != 1.0
    ):
        raise _IncompleteMatchedDescendantError(
            "matched descendant objective requires complete trees"
        )
    left_target = _outer_target_candidate_index(left)
    right_target = _outer_target_candidate_index(right)
    if left_target == right_target:
        raise RuntimeError("matched descendant targets must differ")
    absolute = 0.5 * (
        _outer_logits_loss(left_scores.logits, left)
        + _outer_logits_loss(right_scores.logits, right)
    )
    cross = 0.5 * (
        F.relu(
            margin
            + left_scores.logits[0, right_target]
            - left_scores.logits[0, left_target]
        )
        + F.relu(
            margin
            + right_scores.logits[0, left_target]
            - right_scores.logits[0, right_target]
        )
    )
    left_utilities = _outer_target_candidate_utilities(left, left_scores.logits)
    right_utilities = _outer_target_candidate_utilities(right, right_scores.logits)
    utility_delta = left_utilities - right_utilities
    score_limit = policy.composition_memory.score_limit

    def normalized_delta_loss(
        left_evidence: torch.Tensor,
        right_evidence: torch.Tensor,
    ) -> torch.Tensor:
        evidence_delta = (left_evidence - right_evidence) / (2.0 * score_limit)
        return F.smooth_l1_loss(evidence_delta, utility_delta)

    # Same-root memory and downstream reliability are deliberately absent.
    # Only descendant-conditioned frozen-compiler evidence can explain why
    # the two final candidate-utility vectors differ.  Separate directional
    # terms prevent forward/reverse cancellation from satisfying the loss.
    paired_delta = torch.stack(
        (
            normalized_delta_loss(
                left_scores.phase4_bridge_logits,
                right_scores.phase4_bridge_logits,
            ),
            normalized_delta_loss(
                left_scores.phase4_forward_evidence,
                right_scores.phase4_forward_evidence,
            ),
            normalized_delta_loss(
                left_scores.phase4_reverse_evidence,
                right_scores.phase4_reverse_evidence,
            ),
        )
    ).mean()
    evidence_term = paired_delta if include_evidence_delta else paired_delta.detach() * 0.0
    return absolute + cross + evidence_term, cross, paired_delta


def _outer_utility_loss(
    logits: torch.Tensor,
    utilities: torch.Tensor,
    target_index: int,
) -> torch.Tensor:
    """Proper utility and ranking loss for an evaluator-selected target."""

    if logits.shape != utilities.shape:
        raise ValueError("utility targets must match candidate logits")
    if not 0 <= target_index < logits.shape[1]:
        raise ValueError("target index is outside the candidate set")
    calibration = F.binary_cross_entropy_with_logits(logits, utilities)
    target_logit = logits[:, target_index : target_index + 1]
    competitors = torch.ones_like(logits, dtype=torch.bool)
    competitors[:, target_index] = False
    ranking = F.relu(0.10 + logits - target_logit)
    ranking = ranking[competitors].mean()
    return calibration + 0.25 * ranking


def _oracle_latent_state(
    policy: SkillMemoryPolicy,
    public_task: Any,
    raw_code: torch.Tensor,
    *,
    keyed: bool,
    evidence_count: int = 4,
) -> ProceduralSkillState:
    """Build an evaluator-only state that bypasses evidence acquisition.

    The bounded code is the sole optimized value.  Dense mode removes routing
    as a variable; keyed mode uses the ordinary single-slot read interface.
    Neither mode is available to the online learner or persisted afterward.
    """

    if not isinstance(keyed, bool):
        raise TypeError("keyed must be boolean")
    if (
        isinstance(evidence_count, bool)
        or not isinstance(evidence_count, int)
        or evidence_count <= 0
    ):
        raise ValueError("evidence_count must be a positive integer")
    if (
        not isinstance(raw_code, torch.Tensor)
        or raw_code.shape != (policy.profile.width,)
    ):
        raise ValueError("raw_code must have shape [policy width]")
    empty = policy.initial_state(1)
    if (
        raw_code.device != empty.slot_latents.device
        or raw_code.dtype != empty.slot_latents.dtype
    ):
        raise ValueError("raw_code must match the policy device and dtype")
    code = torch.tanh(raw_code)
    empty_scores = policy.score_task(public_task, empty, include_compiler=False)
    route_key = empty_scores.root.memory_read.route_key
    anchors = F.normalize(policy.memory.slot_anchors, dim=-1, eps=1e-8)
    if keyed:
        selector = F.one_hot(
            torch.zeros(1, device=code.device, dtype=torch.long),
            policy.profile.slots,
        ).to(dtype=code.dtype)
        slot_latents = selector.unsqueeze(-1) * code.view(1, 1, -1)
        desired_keys = route_key.unsqueeze(1).expand(-1, policy.profile.slots, -1)
        key_offsets = selector.unsqueeze(-1) * (
            desired_keys - anchors.unsqueeze(0)
        )
        occupied = selector.to(dtype=torch.bool)
        write_counts = occupied.to(dtype=torch.long) * evidence_count
    else:
        slot_latents = code.view(1, 1, -1).expand(
            1, policy.profile.slots, -1
        )
        key_offsets = (
            route_key.unsqueeze(1).expand(-1, policy.profile.slots, -1)
            - anchors.unsqueeze(0)
        )
        occupied = torch.ones_like(empty.occupied)
        write_counts = torch.full_like(empty.write_counts, evidence_count)
    return ProceduralSkillState(
        fast_weights=empty.fast_weights,
        slot_latents=slot_latents,
        key_offsets=key_offsets,
        occupied=occupied,
        write_counts=write_counts,
    )


def _oracle_readout_metrics(
    policy: SkillMemoryPolicy,
    grouped: Mapping[str, Sequence[Any]],
    operators: Sequence[str],
    raw_codes: torch.Tensor,
    *,
    keyed: bool,
    include_compiler: bool,
    code_shift: int = 0,
) -> dict[str, float]:
    utilities: list[float] = []
    exact: list[float] = []
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for operator_index, operator in enumerate(operators):
            code = raw_codes[(operator_index + code_shift) % len(operators)]
            for pair in grouped[operator]:
                state = _oracle_latent_state(
                    policy, pair.learner, code, keyed=keyed
                )
                logits = policy.score_task(
                    pair.learner,
                    state,
                    include_compiler=include_compiler,
                ).logits
                target = _outer_target_candidate_index(pair)
                chosen = int(logits.argmax(dim=-1).item())
                utility_vector = _outer_target_candidate_utilities(pair, logits)
                utilities.append(float(utility_vector[0, chosen].item()))
                exact.append(float(chosen == target))
                order = torch.argsort(logits[0], descending=True)
                rank = int((order == target).nonzero(as_tuple=False)[0, 0].item()) + 1
                reciprocal_ranks.append(1.0 / rank)
                other = logits[0].clone()
                other[target] = -torch.inf
                margins.append(float((logits[0, target] - other.max()).item()))
                losses.append(float(_outer_logits_loss(logits, pair).item()))
    return {
        "mean_utility": sum(utilities) / len(utilities),
        "exact_top1": sum(exact) / len(exact),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "mean_target_margin": sum(margins) / len(margins),
        "mean_outer_loss": sum(losses) / len(losses),
    }


def oracle_leaf_readout_gate(
    policy: SkillMemoryPolicy,
    *,
    seed: int,
    steps: int = 64,
    instances_per_operator: int = 8,
) -> dict[str, Any]:
    """Test whether the frozen decoder can use an ideal primitive code.

    Hidden operator names select four temporary diagnostic codes only inside
    this evaluator function.  Learner calls receive public tasks and numeric
    states only.  Codes are optimized on one half of fresh meta instances and
    evaluated on the disjoint half, then discarded.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("oracle readout gate requires a frozen policy")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(instances_per_operator, bool)
        or not isinstance(instances_per_operator, int)
        or instances_per_operator < 8
        or instances_per_operator % 2
    ):
        raise ValueError("instances_per_operator must be even and at least eight")
    partition, _ = _load_training_partition(seed, instances_per_operator)
    leaf_pairs = tuple(
        pair for pair in partition.tasks if pair.hidden.program.depth == 0
    )
    grouped_all: dict[str, list[Any]] = {}
    for pair in leaf_pairs:
        grouped_all.setdefault(pair.hidden.program.operator, []).append(pair)
    operators = tuple(sorted(grouped_all))
    if len(operators) != 4 or any(
        len(grouped_all[operator]) != instances_per_operator
        for operator in operators
    ):
        raise RuntimeError("oracle gate requires four balanced leaf operators")
    midpoint = instances_per_operator // 2
    fit = {operator: tuple(grouped_all[operator][:midpoint]) for operator in operators}
    heldout = {
        operator: tuple(grouped_all[operator][midpoint:]) for operator in operators
    }
    reference = next(policy.parameters())
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed + 4049)
    raw_codes = nn.Parameter(
        0.05
        * torch.randn(
            len(operators),
            policy.profile.width,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
    )
    optimizer = torch.optim.Adam((raw_codes,), lr=0.05)
    slow_before = reasoning_state_digest(policy)
    identity_before = _parameter_identity_fingerprint(policy)
    first_loss = 0.0
    last_loss = 0.0
    for step in range(steps):
        losses: list[torch.Tensor] = []
        for operator_index, operator in enumerate(operators):
            for pair in fit[operator]:
                state = _oracle_latent_state(
                    policy,
                    pair.learner,
                    raw_codes[operator_index],
                    keyed=True,
                )
                logits = policy.score_task(
                    pair.learner, state, include_compiler=False
                ).logits
                losses.append(_outer_logits_loss(logits, pair))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        numeric = float(loss.detach().item())
        if step == 0:
            first_loss = numeric
        last_loss = numeric
    if reasoning_state_digest(policy) != slow_before or (
        _parameter_identity_fingerprint(policy) != identity_before
    ):
        raise RuntimeError("oracle diagnostic changed frozen policy state")

    keyed = _oracle_readout_metrics(
        policy, heldout, operators, raw_codes, keyed=True, include_compiler=False
    )
    dense = _oracle_readout_metrics(
        policy, heldout, operators, raw_codes, keyed=False, include_compiler=False
    )
    compiler = _oracle_readout_metrics(
        policy, heldout, operators, raw_codes, keyed=True, include_compiler=True
    )
    permuted = _oracle_readout_metrics(
        policy,
        heldout,
        operators,
        raw_codes,
        keyed=True,
        include_compiler=False,
        code_shift=1,
    )
    zero_codes = torch.zeros_like(raw_codes)
    zero = _oracle_readout_metrics(
        policy, heldout, operators, zero_codes, keyed=True, include_compiler=False
    )
    readout_capable = (
        keyed["mean_utility"] >= 0.90
        and keyed["mean_utility"] - zero["mean_utility"] >= 0.20
        and keyed["exact_top1"] >= 0.75
        and keyed["mean_target_margin"] > 0.0
        and keyed["mean_utility"] - permuted["mean_utility"] >= 0.10
    )
    return {
        "operators": list(operators),
        "fit_instances_per_operator": midpoint,
        "heldout_instances_per_operator": midpoint,
        "optimization_steps": steps,
        "first_fit_loss": first_loss,
        "last_fit_loss": last_loss,
        "keyed_compiler_off": keyed,
        "dense_compiler_off": dense,
        "keyed_compiler_on": compiler,
        "permuted_code_control": permuted,
        "zero_code_control": zero,
        "dense_keyed_utility_gap": dense["mean_utility"] - keyed["mean_utility"],
        "readout_capable": readout_capable,
        "policy_fingerprint_unchanged": True,
        "codes_persisted": False,
    }


def _meta_variants_by_root(
    groups: Sequence[Sequence[Any]],
) -> dict[str, tuple[Sequence[Any], ...]]:
    by_root: dict[str, list[Sequence[Any]]] = {}
    for group in groups:
        if len(group) < 8 or len(group) % 2:
            raise RuntimeError(
                "meta episode groups require an even count of at least eight"
            )
        by_root.setdefault(group[0].learner.request.symbol, []).append(group)
    if len(by_root) != _EXPECTED_PRIMITIVE_ROOTS:
        raise RuntimeError(
            "meta episode does not expose every primitive root"
        )
    return {key: tuple(values) for key, values in by_root.items()}


def _ordered_meta_roots(
    by_root: Mapping[str, Sequence[Sequence[Any]]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            by_root,
            key=lambda symbol: (
                len(by_root[symbol][0][0].learner.request.children),
                symbol,
            ),
        )
    )


def _meta_local_sequences(
    groups: Sequence[Sequence[Any]],
    episode_index: int,
) -> tuple[tuple[tuple[Any, Any], ...], ...]:
    """One long varied skill stream, rotating roots and public contexts."""

    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer")
    by_root = _meta_variants_by_root(groups)
    roots = _ordered_meta_roots(by_root)
    sequences: list[tuple[tuple[Any, Any], ...]] = []
    leaf_roots = tuple(
        root
        for root in roots
        if not by_root[root][0][0].learner.request.children
    )
    if len(leaf_roots) != 4:
        raise RuntimeError("meta local episodes require four leaf procedures")
    for root_offset, root in enumerate(leaf_roots):
        variants = by_root[root]
        selected_index = (episode_index + root_offset) % len(variants)
        selected = variants[selected_index]
        heldout = variants[(selected_index + 1) % len(variants)]
        pair_count = min(len(selected), len(heldout)) // 2
        support_parity = episode_index % 2
        query_parity = 1 - support_parity
        sequences.append(
            tuple(
                (
                    selected[2 * index + support_parity],
                    heldout[2 * index + query_parity],
                )
                for index in range(pair_count)
            )
        )
    return tuple(sequences)


def _meta_episode_sequence(
    groups: Sequence[Sequence[Any]],
    episode_index: int,
) -> tuple[tuple[Any, Any], ...]:
    """Four varied observations per mechanism in one continual stream."""

    by_root = _meta_variants_by_root(groups)
    roots = _ordered_meta_roots(by_root)

    def cross_context_pair(root_index: int, occurrence: int) -> tuple[Any, Any]:
        variants = by_root[roots[root_index]]
        selected_index = (episode_index + occurrence) % len(variants)
        selected = variants[selected_index]
        heldout = variants[(selected_index + 1) % len(variants)]
        # Each freshly reset competence state must experience both public
        # conditions for every mechanism.  Alternating only between outer
        # episodes teaches two separate one-flag states and does not prepare
        # the learner to aggregate the mixed evidence seen online.
        support_parity = (episode_index + occurrence) % 2
        query_parity = 1 - support_parity
        support_index = 4 + 2 * occurrence + support_parity
        query_index = 4 + 2 * occurrence + query_parity
        return (
            selected[support_index % len(selected)],
            heldout[query_index % len(heldout)],
        )

    sequence = [
        cross_context_pair(root_index, occurrence)
        for occurrence in range(4)
        for root_index in range(len(roots))
    ]
    return tuple(sequence)


def _meta_composition_queries(
    groups: Sequence[Sequence[Any]],
    episode_index: int,
) -> tuple[Any, ...]:
    """Use private depth-two/three programs after primitive-root writes."""

    deep = tuple(
        group
        for group in groups
        if group and group[0].learner.request.depth >= 2
    )
    if not deep:
        raise RuntimeError("meta composition queries require non-empty groups")
    return tuple(
        group[episode_index % len(group)]
        for group in deep
    )


def _operator_audit_cohort(pair: Any) -> str | None:
    """Classify only roots whose acquired code enters the operator bridge."""

    request = pair.learner.request
    if len(request.children) != 1 or request.depth not in (2, 3):
        return None
    if len(request.children[0].children) == 2:
        return "unary_direct_binary_child"
    return f"unary_depth{request.depth}"


def _procedural_adapter_training_cohort(pair: Any) -> str | None:
    unary = _operator_audit_cohort(pair)
    if unary is not None:
        return unary
    request = pair.learner.request
    if len(request.children) == 2 and request.depth >= 2:
        return "binary_root"
    return None


def _operator_audit_arity_symbols(pairs: Sequence[Any]) -> dict[int, tuple[str, ...]]:
    """Recover evaluator-only opaque symbol sets without exposing them in reports."""

    arities: dict[str, int] = {}
    for pair in pairs:
        symbol = pair.learner.request.symbol
        arity = len(pair.learner.request.children)
        previous = arities.setdefault(symbol, arity)
        if previous != arity:
            raise RuntimeError("one opaque procedure symbol has conflicting arities")
    by_arity: dict[int, list[str]] = {}
    for symbol, arity in arities.items():
        by_arity.setdefault(arity, []).append(symbol)
    result = {
        arity: tuple(sorted(symbols))
        for arity, symbols in by_arity.items()
    }
    if len(result.get(1, ())) != 4:
        raise RuntimeError("operator audit requires exactly four unary procedures")
    return result


def _operator_audit_queries(
    groups: Sequence[Sequence[Any]],
) -> tuple[Any, ...]:
    """Return four fresh query instances for every deep program group."""

    deep = tuple(
        group
        for group in groups
        if group and group[0].learner.request.depth >= 2
    )
    if not deep or any(
        len(group) < _OPERATOR_AUDIT_QUERY_INSTANCES for group in deep
    ):
        raise RuntimeError("operator audit has insufficient deep query instances")
    return tuple(
        pair
        for group in deep
        for pair in group[:_OPERATOR_AUDIT_QUERY_INSTANCES]
    )


def _acquire_operator_audit_state(
    policy: SkillMemoryPolicy,
    groups: Sequence[Sequence[Any]],
    judge: Callable[..., float],
    *,
    episode_index: int = 0,
) -> tuple[ProceduralSkillState, tuple[str, ...]]:
    """Reproduce the shared meta episode without retaining an update graph."""

    state = policy.initial_state(1)
    support_identities: list[str] = []
    with torch.no_grad():
        for support_pair, _ in _meta_episode_sequence(groups, episode_index):
            proposal = propose_task(
                policy,
                support_pair.learner,
                state,
                greedy=False,
                temperature=1.25,
            )
            score = _judge_frozen_answer(support_pair, proposal.answer, judge)
            staged = propose_differentiable_feedback(
                policy,
                proposal,
                score,
                state,
            )
            state = _detached_state(staged.candidate_state)
            support_identities.append(support_pair.hidden.source_instance_identity)
    return state, tuple(support_identities)


def _acquire_reverse_construction_state(
    policy: SkillMemoryPolicy,
    groups: Sequence[Sequence[Any]],
    judge: Callable[..., float],
    *,
    episode_index: int,
) -> tuple[ProceduralSkillState, tuple[str, ...]]:
    """Retain the real scalar-feedback graph through one fresh mapping."""

    state = policy.initial_state(1)
    support_identities: list[str] = []
    for support_pair, _ in _meta_episode_sequence(groups, episode_index):
        proposal = propose_task(
            policy,
            support_pair.learner,
            state,
            greedy=False,
            temperature=1.25,
        )
        score = _judge_frozen_answer(support_pair, proposal.answer, judge)
        staged = propose_differentiable_feedback(
            policy,
            proposal,
            score,
            state,
        )
        state = staged.candidate_state
        support_identities.append(support_pair.hidden.source_instance_identity)
    return state, tuple(support_identities)


def _same_arity_root_codes(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    pair: Any,
    scores: PolicyScores,
    arity_symbols: Mapping[int, Sequence[str]],
) -> tuple[torch.Tensor, ...]:
    """Read every other same-arity code in the exact current task context."""

    root = scores.root
    arity = root.child_count
    current_symbol = pair.learner.request.symbol
    alternatives = tuple(
        symbol
        for symbol in arity_symbols.get(arity, ())
        if symbol != current_symbol
    )
    if arity != 1 or len(alternatives) != 3:
        raise RuntimeError("unary operator audit requires three alternate codes")
    memory = policy.memory_for_tier(root.memory_tier)
    codes: list[torch.Tensor] = []
    for symbol in alternatives:
        goal = policy._goal_embedding(
            symbol,
            arity,
            0,
            root.state_embedding,
            root.memory_tier,
        )
        read = memory.read(
            root.state_embedding,
            goal,
            root.candidate_embeddings,
            state=state,
        )
        codes.append(read.plastic_context)
    return tuple(codes)


def _root_operator_forward_evidence(
    policy: SkillMemoryPolicy,
    scores: PolicyScores,
    code: torch.Tensor,
    *,
    include_frozen_transition: bool = True,
    include_fast_adapter: bool = True,
) -> torch.Tensor:
    """Replay one unary root code while holding its executed descendants fixed."""

    root = scores.root
    if root.child_count != 1:
        raise ValueError("operator evidence replay requires a unary root")
    if code.shape != root.memory_read.plastic_context.shape:
        raise ValueError("counterfactual operator code has the wrong shape")
    incoming = policy.compiler_source_bridge(root.state_embedding)
    predecessor = root.recursive_predecessor
    candidates = policy.compiler_successor_bridge(root.candidate_embeddings[0])
    stable_delta = torch.zeros_like(predecessor)
    if include_frozen_transition:
        operator = policy.compiler_operator_bridge(code)
        null_operator = policy.compiler_operator_bridge(torch.zeros_like(code))
        stable_delta = (
            policy.stable_compiler.core.predict_effects(
                predecessor,
                operator,
                reverse=False,
            )[:, 0, :]
            - policy.stable_compiler.core.predict_effects(
                predecessor,
                null_operator,
                reverse=False,
            )[:, 0, :]
        )
    fast_delta = (
        policy.procedural_fast_adapter(predecessor, code, reverse=False)
        if include_fast_adapter
        else torch.zeros_like(predecessor)
    )
    goal_delta = torch.zeros_like(predecessor)
    if include_fast_adapter:
        factors = policy.procedural_fast_adapter.latent_factors(
            predecessor,
            code,
        )
        goal_delta, _ = policy.procedural_goal_projection(
            predecessor,
            candidates,
            factors,
        )
    predicted = predecessor + stable_delta + fast_delta + goal_delta
    predicted_scores = -(
        (candidates - predicted[0].unsqueeze(0)).square().mean(dim=-1)
    )
    null_scores = -(
        (candidates - incoming[0].unsqueeze(0)).square().mean(dim=-1)
    )
    residual = predicted_scores - null_scores
    residual = residual - residual.mean()
    limit = policy.memory_for_tier(root.memory_tier).score_limit
    return limit * torch.tanh(residual)


def _standardize_candidate_energy(energy: torch.Tensor) -> torch.Tensor:
    row = energy.reshape(-1)
    if row.numel() != len(_PERMUTATIONS) or not bool(
        torch.isfinite(row).all().item()
    ):
        raise ValueError("candidate energy must be one finite public action vector")
    centered = row - row.mean()
    # Smooth RMS normalization keeps an exactly neutral new head trainable
    # without the infinite sqrt(0) derivative or the arbitrary 1e6 first-step
    # gain produced by a post-sqrt clamp.
    scale = (centered.square().mean() + 1.0e-4).sqrt()
    return centered / scale


def _root_reverse_construction_energies(
    policy: SkillMemoryPolicy,
    scores: PolicyScores,
    code: torch.Tensor,
    *,
    include_frozen_transition: bool = True,
) -> dict[str, torch.Tensor]:
    """Foreshadow, execute, and reverse one unary procedure in shared geometry."""

    root = scores.root
    if root.child_count != 1:
        raise ValueError("reverse construction requires a unary root")
    if code.shape != root.memory_read.plastic_context.shape:
        raise ValueError("reverse construction code has the wrong shape")
    source = root.recursive_predecessor
    candidates = policy.compiler_successor_bridge(root.candidate_embeddings[0])
    stable_forward = torch.zeros_like(source)
    stable_reverse_candidates = torch.zeros_like(candidates)
    operator = policy.compiler_operator_bridge(code)
    null_operator = policy.compiler_operator_bridge(torch.zeros_like(code))
    if include_frozen_transition:
        stable_forward = (
            policy.stable_compiler.core.predict_effects(
                source,
                operator,
                reverse=False,
            )[:, 0, :]
            - policy.stable_compiler.core.predict_effects(
                source,
                null_operator,
                reverse=False,
            )[:, 0, :]
        )
        stable_reverse_candidates = (
            policy.stable_compiler.core.predict_effects(
                candidates,
                operator,
                reverse=True,
            )[:, 0, :]
            - policy.stable_compiler.core.predict_effects(
                candidates,
                null_operator,
                reverse=True,
            )[:, 0, :]
        )
    factors = policy.procedural_fast_adapter.latent_factors(source, code)
    goal_delta, goal_energy = policy.procedural_goal_projection(
        source,
        candidates,
        factors,
    )
    forward_successor = (
        source
        + stable_forward
        + policy.procedural_fast_adapter(source, code, reverse=False)
        + goal_delta
    )
    reverse_origins = (
        candidates
        + stable_reverse_candidates
        + policy.procedural_fast_adapter(candidates, code, reverse=True)
    )
    forward_energy = -(
        (candidates - forward_successor).square().mean(dim=-1)
    )
    reverse_energy = -(
        (reverse_origins - source).square().mean(dim=-1)
    )

    stable_cycle = torch.zeros_like(forward_successor)
    if include_frozen_transition:
        stable_cycle = (
            policy.stable_compiler.core.predict_effects(
                forward_successor,
                operator,
                reverse=True,
            )[:, 0, :]
            - policy.stable_compiler.core.predict_effects(
                forward_successor,
                null_operator,
                reverse=True,
            )[:, 0, :]
        )
    reconstructed_source = (
        forward_successor
        + stable_cycle
        + policy.procedural_fast_adapter(
            forward_successor,
            code,
            reverse=True,
        )
    )
    standardized_goal = _standardize_candidate_energy(goal_energy)
    standardized_forward = _standardize_candidate_energy(forward_energy)
    standardized_reverse = _standardize_candidate_energy(reverse_energy)
    return {
        "goal": standardized_goal,
        "forward": standardized_forward,
        "reverse": standardized_reverse,
        "combined": standardized_goal + standardized_forward + standardized_reverse,
        "cycle": F.smooth_l1_loss(reconstructed_source, source.detach()),
        "forward_successor": forward_successor,
        "reconstructed_source": reconstructed_source,
    }


def _operator_audit_alignment(
    evidence: torch.Tensor,
    utilities: torch.Tensor,
) -> dict[str, float] | None:
    """Return unweighted candidate covariance and scale-free alignment."""

    evidence_row = evidence.detach().reshape(-1).to(dtype=torch.float64)
    utility_row = utilities.detach().reshape(-1).to(dtype=torch.float64)
    if evidence_row.shape != utility_row.shape or evidence_row.numel() != len(
        _PERMUTATIONS
    ):
        raise ValueError("operator alignment requires one complete candidate vector")
    if not bool(torch.isfinite(evidence_row).all().item()) or not bool(
        torch.isfinite(utility_row).all().item()
    ):
        raise ValueError("operator alignment inputs must be finite")
    centered_evidence = evidence_row - evidence_row.mean()
    centered_utility = utility_row - utility_row.mean()
    evidence_variance = centered_evidence.square().mean()
    utility_variance = centered_utility.square().mean()
    tolerance = torch.finfo(torch.float64).eps
    if bool((evidence_variance <= tolerance).item()) or bool(
        (utility_variance <= tolerance).item()
    ):
        return None
    covariance = (centered_evidence * centered_utility).mean()
    correlation = covariance / torch.sqrt(evidence_variance * utility_variance)
    return {
        "covariance": float(covariance.item()),
        "correlation": float(correlation.item()),
    }


def _operator_audit_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("operator audit mean requires at least one value")
    return float(sum(values) / len(values))


def _summarize_operator_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate evaluator rows without exposing tasks, targets, or symbols."""

    if not rows:
        raise ValueError("operator audit summary requires at least one row")

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        condition_means = {
            condition: {
                metric: _operator_audit_mean(
                    [float(row[condition][metric]) for row in selected]
                )
                for metric in ("covariance", "correlation")
            }
            for condition in ("correct", "permuted", "zero")
        }
        return {
            "count": len(selected),
            "conditions": condition_means,
            "representation": {
                metric: _operator_audit_mean(
                    [float(row["representation"][metric]) for row in selected]
                )
                for metric in (
                    "correct_code_rms",
                    "correct_minus_permuted_code_rms",
                    "correct_minus_zero_operator_rms",
                    "correct_minus_permuted_operator_rms",
                    "correct_minus_zero_evidence_rms",
                    "correct_minus_permuted_evidence_rms",
                )
            },
            "deltas": {
                f"correct_minus_{control}_{metric}": (
                    condition_means["correct"][metric]
                    - condition_means[control][metric]
                )
                for control in ("permuted", "zero")
                for metric in ("covariance", "correlation")
            },
        }

    operators = sorted({str(row["hidden_operator"]) for row in rows})
    return {
        **summarize(rows),
        "by_hidden_operator": {
            operator: summarize(
                [row for row in rows if row["hidden_operator"] == operator]
            )
            for operator in operators
        },
    }


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().to(dtype=torch.float64, device="cpu")
    right = right.detach().to(dtype=torch.float64, device="cpu")
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return None
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


def _bridge_gradient_summary(
    vectors: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    """Summarize discarded no-step gradients across eight independent mappings."""

    seeds = tuple(sorted(vectors))
    if len(seeds) != _OPERATOR_AUDIT_SEED_COUNT:
        raise ValueError("bridge coherence requires exactly eight audit seeds")
    cpu_vectors = {
        seed: vectors[seed].detach().to(dtype=torch.float64, device="cpu")
        for seed in seeds
    }
    norms = {
        str(seed): float(torch.linalg.vector_norm(cpu_vectors[seed]).item())
        for seed in seeds
    }
    pairwise: list[dict[str, Any]] = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            pairwise.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "cosine": _cosine_similarity(
                        cpu_vectors[left_seed], cpu_vectors[right_seed]
                    ),
                }
            )
    numeric_pairwise = [
        float(row["cosine"])
        for row in pairwise
        if row["cosine"] is not None
    ]
    leave_one_out: dict[str, float | None] = {}
    for seed in seeds:
        others = torch.stack(
            [cpu_vectors[other] for other in seeds if other != seed]
        ).mean(dim=0)
        leave_one_out[str(seed)] = _cosine_similarity(cpu_vectors[seed], others)
    normalized = [
        cpu_vectors[seed] / torch.linalg.vector_norm(cpu_vectors[seed])
        for seed in seeds
        if norms[str(seed)] > 0.0
    ]
    consensus = (
        float(torch.linalg.vector_norm(torch.stack(normalized).mean(dim=0)).item())
        if len(normalized) == len(seeds)
        else 0.0
    )
    sorted_pairwise = sorted(numeric_pairwise)
    median_pairwise = (
        0.5
        * (
            sorted_pairwise[len(sorted_pairwise) // 2 - 1]
            + sorted_pairwise[len(sorted_pairwise) // 2]
        )
        if len(sorted_pairwise) % 2 == 0 and sorted_pairwise
        else (
            sorted_pairwise[len(sorted_pairwise) // 2]
            if sorted_pairwise
            else None
        )
    )
    positive_pairwise = sum(value > 0.0 for value in numeric_pairwise)
    numeric_leave_one_out = [
        float(value) for value in leave_one_out.values() if value is not None
    ]
    positive_leave_one_out = sum(value > 0.0 for value in numeric_leave_one_out)
    finite_nonzero = all(math.isfinite(value) and value > 0.0 for value in norms.values())
    mean_pairwise = (
        _operator_audit_mean(numeric_pairwise) if numeric_pairwise else None
    )
    passed = (
        finite_nonzero
        and len(numeric_pairwise) == 28
        and mean_pairwise is not None
        and mean_pairwise >= 0.10
        and positive_pairwise >= 21
        and len(numeric_leave_one_out) == 8
        and positive_leave_one_out >= 7
    )
    return {
        "gradient_norms": norms,
        "pairwise_cosines": pairwise,
        "mean_pairwise_cosine": mean_pairwise,
        "median_pairwise_cosine": median_pairwise,
        "positive_pairwise_count": positive_pairwise,
        "pairwise_count": len(numeric_pairwise),
        "leave_one_seed_out_cosines": leave_one_out,
        "positive_leave_one_seed_out_count": positive_leave_one_out,
        "normalized_consensus_magnitude": consensus,
        "finite_nonzero_every_seed": finite_nonzero,
        "passed": passed,
    }


def _operator_localization_seed(
    policy: SkillMemoryPolicy,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Run one read-only mapping-local audit and return private gradient vectors."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("operator localization requires a frozen policy")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("operator localization requires empty gradient fields")
    slow_before = reasoning_state_digest(policy)
    identity_before = _parameter_identity_fingerprint(policy)
    partition, judge = _load_training_partition(
        seed,
        _OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
    )
    groups = _group_evaluator_pairs(partition.tasks)
    arity_symbols = _operator_audit_arity_symbols(partition.tasks)
    state, support_identities = _acquire_operator_audit_state(
        policy,
        groups,
        judge,
    )
    queries = _operator_audit_queries(groups)
    query_identities = {
        pair.hidden.source_instance_identity for pair in queries
    }
    if set(support_identities) & query_identities:
        raise RuntimeError("operator audit support and query instances overlap")
    state_before = procedural_skill_state_digest(state)

    named_bridges = {
        label: tuple(
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if name.startswith(prefix)
        )
        for label, prefix in _OPERATOR_AUDIT_BRIDGES.items()
    }
    if any(not values for values in named_bridges.values()):
        raise RuntimeError("operator audit could not resolve every bridge module")
    ordered_parameters = tuple(
        parameter
        for label in _OPERATOR_AUDIT_BRIDGES
        for _, parameter in named_bridges[label]
    )
    for parameter in ordered_parameters:
        parameter.requires_grad_(True)

    rows: dict[str, list[dict[str, Any]]] = {
        cohort: [] for cohort in _OPERATOR_AUDIT_COHORTS
    }
    degenerate = {cohort: 0 for cohort in _OPERATOR_AUDIT_COHORTS}
    unavailable = {cohort: 0 for cohort in _OPERATOR_AUDIT_COHORTS}
    gradient_sums = {
        cohort: {
            label: torch.zeros(
                sum(parameter.numel() for _, parameter in named_bridges[label]),
                device=next(policy.parameters()).device,
                dtype=next(policy.parameters()).dtype,
            )
            for label in _OPERATOR_AUDIT_BRIDGES
        }
        for cohort in _OPERATOR_AUDIT_COHORTS
    }
    gradient_counts = {cohort: 0 for cohort in _OPERATOR_AUDIT_COHORTS}
    try:
        for pair in queries:
            cohort = _operator_audit_cohort(pair)
            if cohort is None:
                continue
            scores = policy.score_task(pair.learner, state)
            if float(scores.root_available.detach().item()) != 1.0:
                unavailable[cohort] += 1
                continue
            correct = _root_operator_forward_evidence(
                policy,
                scores,
                scores.root.memory_read.plastic_context,
            )
            if not torch.allclose(
                correct.unsqueeze(0),
                scores.phase4_forward_evidence,
                atol=1.0e-6,
                rtol=1.0e-5,
            ):
                raise RuntimeError(
                    "operator audit reconstruction differs from live forward evidence"
                )
            loss = _outer_top_target_loss(correct.unsqueeze(0), pair)
            gradients = torch.autograd.grad(
                loss,
                ordered_parameters,
                allow_unused=True,
            )
            gradient_index = 0
            for label in _OPERATOR_AUDIT_BRIDGES:
                flattened: list[torch.Tensor] = []
                for _, parameter in named_bridges[label]:
                    gradient = gradients[gradient_index]
                    gradient_index += 1
                    flattened.append(
                        torch.zeros_like(parameter).reshape(-1)
                        if gradient is None
                        else gradient.detach().reshape(-1)
                    )
                gradient_sums[cohort][label] += torch.cat(flattened)
            gradient_counts[cohort] += 1

            with torch.no_grad():
                alternatives = _same_arity_root_codes(
                    policy,
                    state,
                    pair,
                    scores,
                    arity_symbols,
                )
                alternate_evidence = tuple(
                    _root_operator_forward_evidence(policy, scores, code)
                    for code in alternatives
                )
                zero = _root_operator_forward_evidence(
                    policy,
                    scores,
                    torch.zeros_like(scores.root.memory_read.plastic_context),
                )
                correct_code = scores.root.memory_read.plastic_context
                correct_operator = policy.compiler_operator_bridge(correct_code)
                alternate_operators = tuple(
                    policy.compiler_operator_bridge(code) for code in alternatives
                )
                zero_operator = policy.compiler_operator_bridge(
                    torch.zeros_like(correct_code)
                )
                utilities = _outer_target_candidate_utilities(pair, correct)
                correct_alignment = _operator_audit_alignment(correct, utilities)
                alternate_alignments = tuple(
                    _operator_audit_alignment(evidence, utilities)
                    for evidence in alternate_evidence
                )
                zero_alignment = _operator_audit_alignment(zero, utilities)
                intervention_alignments: dict[str, dict[str, Any] | None] = {}
                for mode, include_frozen, include_fast in (
                    ("frozen_only", True, False),
                    ("fast_only", False, True),
                ):
                    mode_correct = _root_operator_forward_evidence(
                        policy,
                        scores,
                        correct_code,
                        include_frozen_transition=include_frozen,
                        include_fast_adapter=include_fast,
                    )
                    mode_alternates = tuple(
                        _root_operator_forward_evidence(
                            policy,
                            scores,
                            code,
                            include_frozen_transition=include_frozen,
                            include_fast_adapter=include_fast,
                        )
                        for code in alternatives
                    )
                    mode_zero = _root_operator_forward_evidence(
                        policy,
                        scores,
                        torch.zeros_like(correct_code),
                        include_frozen_transition=include_frozen,
                        include_fast_adapter=include_fast,
                    )
                    mode_correct_alignment = _operator_audit_alignment(
                        mode_correct,
                        utilities,
                    )
                    mode_alternate_alignments = tuple(
                        _operator_audit_alignment(evidence, utilities)
                        for evidence in mode_alternates
                    )
                    mode_zero_alignment = _operator_audit_alignment(
                        mode_zero,
                        utilities,
                    )
                    if (
                        mode_correct_alignment is None
                        or mode_zero_alignment is None
                        or any(
                            item is None for item in mode_alternate_alignments
                        )
                    ):
                        intervention_alignments[mode] = None
                        continue
                    numeric_mode_alternates = tuple(
                        item
                        for item in mode_alternate_alignments
                        if item is not None
                    )
                    intervention_alignments[mode] = {
                        "correct": mode_correct_alignment,
                        "permuted": {
                            metric: _operator_audit_mean(
                                [
                                    float(item[metric])
                                    for item in numeric_mode_alternates
                                ]
                            )
                            for metric in ("covariance", "correlation")
                        },
                        "zero": mode_zero_alignment,
                    }
            if (
                correct_alignment is None
                or zero_alignment is None
                or any(item is None for item in alternate_alignments)
            ):
                degenerate[cohort] += 1
                continue
            numeric_alternatives = tuple(
                item for item in alternate_alignments if item is not None
            )
            rows[cohort].append(
                {
                    "hidden_operator": pair.hidden.program.operator,
                    "correct": correct_alignment,
                    "permuted": {
                        metric: _operator_audit_mean(
                            [float(item[metric]) for item in numeric_alternatives]
                        )
                        for metric in ("covariance", "correlation")
                    },
                    "zero": zero_alignment,
                    "interventions": intervention_alignments,
                    "representation": {
                        "correct_code_rms": float(
                            correct_code.square().mean().sqrt().item()
                        ),
                        "correct_minus_permuted_code_rms": _operator_audit_mean(
                            [
                                float(
                                    (correct_code - code)
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .item()
                                )
                                for code in alternatives
                            ]
                        ),
                        "correct_minus_zero_operator_rms": float(
                            (correct_operator - zero_operator)
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "correct_minus_permuted_operator_rms": (
                            _operator_audit_mean(
                                [
                                    float(
                                        (correct_operator - operator)
                                        .square()
                                        .mean()
                                        .sqrt()
                                        .item()
                                    )
                                    for operator in alternate_operators
                                ]
                            )
                        ),
                        "correct_minus_zero_evidence_rms": float(
                            (correct - zero).square().mean().sqrt().item()
                        ),
                        "correct_minus_permuted_evidence_rms": (
                            _operator_audit_mean(
                                [
                                    float(
                                        (correct - evidence)
                                        .square()
                                        .mean()
                                        .sqrt()
                                        .item()
                                    )
                                    for evidence in alternate_evidence
                                ]
                            )
                        ),
                    },
                }
            )
    finally:
        for parameter in ordered_parameters:
            parameter.requires_grad_(False)

    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("no-step operator audit populated parameter gradients")
    if procedural_skill_state_digest(state) != state_before:
        raise RuntimeError("operator audit changed its acquired competence state")
    if reasoning_state_digest(policy) != slow_before or (
        _parameter_identity_fingerprint(policy) != identity_before
    ):
        raise RuntimeError("operator audit changed frozen slow state")
    if any(gradient_counts[cohort] == 0 for cohort in _OPERATOR_AUDIT_COHORTS):
        raise RuntimeError("operator audit did not cover every required cohort")
    gradient_vectors = {
        label: torch.stack(
            [
                gradient_sums[cohort][label] / gradient_counts[cohort]
                for cohort in _OPERATOR_AUDIT_COHORTS
            ]
        ).mean(dim=0).detach().cpu()
        for label in _OPERATOR_AUDIT_BRIDGES
    }
    public = {
        "seed": seed,
        "support_presentations": len(support_identities),
        "query_presentations": len(queries),
        "support_query_disjoint": True,
        "cohorts": {
            cohort: (
                _summarize_operator_audit_rows(rows[cohort])
                if rows[cohort]
                else {"count": 0}
            )
            for cohort in _OPERATOR_AUDIT_COHORTS
        },
        "degenerate_cases": degenerate,
        "unavailable_cases": unavailable,
        "state_digest_before": state_before,
        "state_digest_after": procedural_skill_state_digest(state),
        "slow_fingerprint_before": slow_before,
        "slow_fingerprint_after": reasoning_state_digest(policy),
        "parameter_identity_unchanged": True,
    }
    return public, rows, gradient_vectors


def _procedural_adapter_candidate_pair(
    outer_step: int,
    query_index: int,
) -> tuple[int, int]:
    """Choose two varied public interventions without evaluator information."""

    for name, value in (("outer_step", outer_step), ("query_index", query_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    first = (11 + 31 * outer_step + 43 * query_index) % len(_PERMUTATIONS)
    offset = 1 + (17 * outer_step + 29 * query_index) % (len(_PERMUTATIONS) - 1)
    second = (first + offset) % len(_PERMUTATIONS)
    if first == second:
        raise RuntimeError("procedural adapter interventions must be distinct")
    return first, second


def _reverse_construction_candidate_set(
    outer_step: int,
    query_index: int,
    *,
    count: int = _REVERSE_CONSTRUCTION_ATTEMPTS,
) -> tuple[int, ...]:
    """Predeclare varied public attempts without task, mapping, or reward access."""

    for name, value in (("outer_step", outer_step), ("query_index", query_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= len(
        _PERMUTATIONS
    ):
        raise ValueError("attempt count must be between two and the action count")
    schedule = random.Random(
        0xA69E_2026 ^ (outer_step + 1) * 1_000_003 ^ (query_index + 1) * 97_003
    )
    return tuple(schedule.sample(range(len(_PERMUTATIONS)), count))


def _on_policy_reward_candidate_set(
    logits: torch.Tensor,
    outer_step: int,
    query_index: int,
) -> tuple[int, ...]:
    """Judge the deployed choice plus three policy-sampled explorations."""

    for name, value in (("outer_step", outer_step), ("query_index", query_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    row = logits.detach().reshape(-1)
    if row.numel() != len(_PERMUTATIONS) or not bool(
        torch.isfinite(row).all().item()
    ):
        raise ValueError("on-policy attempt selection requires finite action logits")
    greedy = int(row.argmax().item())
    behavior = torch.softmax(row / 1.25, dim=0)
    behavior[greedy] = 0.0
    behavior = behavior / behavior.sum()
    generator = torch.Generator(device=row.device)
    generator.manual_seed(
        0xA69E_2026
        ^ (outer_step + 1) * 1_000_003
        ^ (query_index + 1) * 97_003
    )
    sampled = torch.multinomial(
        behavior,
        _REVERSE_CONSTRUCTION_ATTEMPTS - 1,
        replacement=False,
        generator=generator,
    )
    result = (greedy, *(int(index) for index in sampled.tolist()))
    if len(result) != _REVERSE_CONSTRUCTION_ATTEMPTS or len(set(result)) != len(
        result
    ):
        raise RuntimeError("on-policy preference attempts are not distinct")
    return result


def _candidate_answer(pair: Any, candidate_index: int) -> tuple[str, ...]:
    if not 0 <= candidate_index < len(_PERMUTATIONS):
        raise ValueError("candidate index is outside the public action set")
    return tuple(
        pair.learner.items[index].symbol
        for index in _PERMUTATIONS[candidate_index]
    )


def _scalar_pairwise_scores(
    pair: Any,
    candidate_pair: tuple[int, int],
    judge: Callable[..., float],
) -> tuple[float, float]:
    """Observe exactly two public attempts once, independent of model logits."""

    first, second = candidate_pair
    if first == second:
        raise ValueError("preference candidates must be distinct")
    if not all(0 <= index < len(_PERMUTATIONS) for index in candidate_pair):
        raise ValueError("preference candidate is outside the public action set")
    first_score = _judge_frozen_answer(
        pair,
        _candidate_answer(pair, first),
        judge,
    )
    second_score = _judge_frozen_answer(
        pair,
        _candidate_answer(pair, second),
        judge,
    )
    return first_score, second_score


def _scalar_attempt_scores(
    pair: Any,
    candidate_indices: Sequence[int],
    judge: Callable[..., float],
) -> tuple[float, ...]:
    indices = tuple(candidate_indices)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("scalar attempts must contain distinct public candidates")
    if any(not 0 <= index < len(_PERMUTATIONS) for index in indices):
        raise ValueError("scalar attempt is outside the public action set")
    return tuple(
        _judge_frozen_answer(
            pair,
            _candidate_answer(pair, index),
            judge,
        )
        for index in indices
    )


def _scalar_multi_preference_loss(
    logits: torch.Tensor,
    candidate_indices: Sequence[int],
    scalar_scores: Sequence[float],
    *,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, int]:
    """Fit every observed non-tied preference edge among attempted outputs."""

    row = logits.reshape(-1)
    if row.numel() != len(_PERMUTATIONS):
        raise ValueError("preference logits must cover every public candidate")
    indices = tuple(candidate_indices)
    scores = tuple(_validate_scalar_reward(value) for value in scalar_scores)
    if len(indices) != len(scores) or len(indices) < 2:
        raise ValueError("attempted candidates and scores must have equal length")
    if len(set(indices)) != len(indices) or any(
        not 0 <= index < len(_PERMUTATIONS) for index in indices
    ):
        raise ValueError("attempted preference candidates must be distinct and valid")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("preference temperature must be finite and positive")
    edges: list[torch.Tensor] = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            difference = scores[left] - scores[right]
            if difference == 0.0:
                continue
            direction = row.new_tensor(1.0 if difference > 0.0 else -1.0)
            margin = direction * (row[indices[left]] - row[indices[right]])
            edges.append(abs(difference) * F.softplus(-margin / temperature))
    if not edges:
        return row.sum() * 0.0, 0
    return torch.stack(edges).mean(), len(edges)


def _scalar_on_policy_reward_loss(
    logits: torch.Tensor,
    candidate_indices: Sequence[int],
    scalar_scores: Sequence[float],
) -> torch.Tensor:
    """Maximize observed reward under the complete deployed action policy."""

    row = logits.reshape(-1)
    if row.numel() != len(_PERMUTATIONS) or not bool(
        torch.isfinite(row).all().item()
    ):
        raise ValueError("on-policy reward logits must be one finite action vector")
    indices = tuple(candidate_indices)
    rewards = tuple(_validate_scalar_reward(value) for value in scalar_scores)
    if len(indices) != len(rewards) or len(indices) < 2:
        raise ValueError("attempted candidates and rewards must have equal length")
    if len(set(indices)) != len(indices) or any(
        not 0 <= index < len(_PERMUTATIONS) for index in indices
    ):
        raise ValueError("on-policy reward candidates must be distinct and valid")
    reward_tensor = row.new_tensor(rewards)
    advantages = reward_tensor - reward_tensor.mean()
    if not bool(advantages.count_nonzero()):
        return row.sum() * 0.0
    log_probabilities = F.log_softmax(row, dim=0)
    return -(
        advantages.detach()
        * log_probabilities[row.new_tensor(indices, dtype=torch.long)]
    ).mean()


def _scalar_pairwise_preference_loss(
    logits: torch.Tensor,
    candidate_pair: tuple[int, int],
    scalar_scores: tuple[float, float],
    *,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, float]:
    """Fit logits to one already-observed two-attempt scalar preference."""

    row = logits.reshape(-1)
    if row.numel() != len(_PERMUTATIONS):
        raise ValueError("preference logits must cover every public candidate")
    first, second = candidate_pair
    if first == second:
        raise ValueError("preference candidates must be distinct")
    if not all(0 <= index < len(_PERMUTATIONS) for index in candidate_pair):
        raise ValueError("preference candidate is outside the public action set")
    if len(scalar_scores) != 2:
        raise ValueError("preference observation must contain two scalar scores")
    first_score = _validate_scalar_reward(scalar_scores[0])
    second_score = _validate_scalar_reward(scalar_scores[1])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("preference temperature must be finite and positive")
    difference = first_score - second_score
    if difference == 0.0:
        return row.sum() * 0.0, 0.0
    direction = row.new_tensor(1.0 if difference > 0.0 else -1.0)
    margin = direction * (row[first] - row[second])
    return abs(difference) * F.softplus(-margin / temperature), abs(difference)


def _train_procedural_adapter(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Meta-learn a fast-weight executor from varied scalar comparisons only."""

    trainable_names = _configure_stage_trainability(policy, "procedural_adapter")
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=profile.learning_rate,
        weight_decay=0.0,
    )
    optimizer_identity = _optimizer_identity_fingerprint(optimizer, policy)
    parameter_identity = _parameter_identity_fingerprint(policy)
    compiler_before = reasoning_state_digest(policy.stable_compiler)
    adapter_before = _named_state_fingerprint(
        policy,
        include=_is_procedural_adapter_state,
        domain=b"project-angler.phase5-procedural-adapter.v1",
    )
    outside_before = _named_state_fingerprint(
        policy,
        include=lambda name: not _is_procedural_adapter_state(name),
        domain=b"project-angler.phase5-outside-procedural-adapter.v1",
    )
    reverse_before = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith(
            "procedural_fast_adapter.reverse_up."
        ),
        domain=b"project-angler.phase5-procedural-adapter-reverse.v1",
    )
    losses: list[float] = []
    actual_losses: list[float] = []
    forward_losses: list[float] = []
    specificity_losses: list[float] = []
    reward_separations: list[int] = []
    query_counts_per_mapping: list[int] = []
    gradient_norms: list[float] = []
    cohort_case_counts = {
        cohort: 0 for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
    }
    policy.train()
    policy.stable_compiler.eval()
    for step in range(profile.meta_steps):
        episode_seed = seed + 100_003 * (step + 1)
        _seed_reproducible_stage(
            episode_seed,
            "procedural-adapter-training",
            next(policy.parameters()).device,
        )
        partition, judge = _load_training_partition(
            episode_seed,
            _OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
        groups = _group_evaluator_pairs(partition.tasks)
        arity_symbols = _operator_audit_arity_symbols(partition.tasks)
        state, support_identities = _acquire_operator_audit_state(
            policy,
            groups,
            judge,
        )
        queries = _operator_audit_queries(groups)
        if len(support_identities) != 40 or len(queries) != 32:
            raise RuntimeError("adapter mapping has an unexpected presentation count")
        if set(support_identities) & {
            pair.hidden.source_instance_identity for pair in queries
        }:
            raise RuntimeError("adapter support and query instances overlap")
        by_cohort: dict[str, list[torch.Tensor]] = {
            cohort: [] for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
        }
        step_actual: list[torch.Tensor] = []
        step_forward: list[torch.Tensor] = []
        step_specificity: list[torch.Tensor] = []
        separated = 0
        for query_index, pair in enumerate(queries):
            cohort = _procedural_adapter_training_cohort(pair)
            if cohort is None:
                continue
            scores = policy.score_task(pair.learner, state)
            if float(scores.root_available.detach().item()) != 1.0:
                continue
            candidate_pair = _procedural_adapter_candidate_pair(step, query_index)
            scalar_scores = _scalar_pairwise_scores(
                pair,
                candidate_pair,
                judge,
            )
            actual_loss, actual_separation = _scalar_pairwise_preference_loss(
                scores.logits,
                candidate_pair,
                scalar_scores,
            )
            step_actual.append(actual_loss)
            if cohort == "binary_root":
                by_cohort[cohort].append(
                    _PROCEDURAL_ADAPTER_BINARY_ACTUAL_WEIGHT * actual_loss
                )
                separated += int(actual_separation > 0.0)
                cohort_case_counts[cohort] += 1
                continue
            correct = _root_operator_forward_evidence(
                policy,
                scores,
                scores.root.memory_read.plastic_context,
            )
            forward_loss, forward_separation = _scalar_pairwise_preference_loss(
                correct,
                candidate_pair,
                scalar_scores,
            )
            alternatives = _same_arity_root_codes(
                policy,
                state,
                pair,
                scores,
                arity_symbols,
            )
            control_evidence = tuple(
                _root_operator_forward_evidence(policy, scores, code)
                for code in alternatives
            ) + (
                _root_operator_forward_evidence(
                    policy,
                    scores,
                    torch.zeros_like(scores.root.memory_read.plastic_context),
                ),
            )
            control_losses = tuple(
                _scalar_pairwise_preference_loss(
                    evidence,
                    candidate_pair,
                    scalar_scores,
                )[0]
                for evidence in control_evidence
            )
            specificity = torch.stack(
                tuple(
                    F.relu(0.05 + forward_loss - control_loss)
                    for control_loss in control_losses
                )
            ).mean()
            case_loss = (
                _PROCEDURAL_ADAPTER_UNARY_ACTUAL_WEIGHT * actual_loss
                + _PROCEDURAL_ADAPTER_FORWARD_WEIGHT * forward_loss
                + _PROCEDURAL_ADAPTER_SPECIFICITY_WEIGHT * specificity
            )
            by_cohort[cohort].append(case_loss)
            step_forward.append(forward_loss)
            step_specificity.append(specificity)
            separated += int(actual_separation > 0.0 and forward_separation > 0.0)
            cohort_case_counts[cohort] += 1
        if any(
            not by_cohort[cohort]
            for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
        ):
            raise RuntimeError("adapter training did not cover every required cohort")
        covered_queries = sum(len(values) for values in by_cohort.values())
        if covered_queries != 32:
            raise RuntimeError("adapter training skipped a qualifying query presentation")
        query_counts_per_mapping.append(covered_queries)
        loss = torch.stack(
            tuple(
                torch.stack(by_cohort[cohort]).mean()
                for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
            )
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            profile.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        actual_losses.append(float(torch.stack(step_actual).mean().detach().item()))
        forward_losses.append(float(torch.stack(step_forward).mean().detach().item()))
        specificity_losses.append(
            float(torch.stack(step_specificity).mean().detach().item())
        )
        reward_separations.append(separated)
        gradient_norms.append(float(gradient_norm.detach().item()))
    policy.eval()
    policy.requires_grad_(False)
    adapter_after = _named_state_fingerprint(
        policy,
        include=_is_procedural_adapter_state,
        domain=b"project-angler.phase5-procedural-adapter.v1",
    )
    outside_after = _named_state_fingerprint(
        policy,
        include=lambda name: not _is_procedural_adapter_state(name),
        domain=b"project-angler.phase5-outside-procedural-adapter.v1",
    )
    reverse_after = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith(
            "procedural_fast_adapter.reverse_up."
        ),
        domain=b"project-angler.phase5-procedural-adapter-reverse.v1",
    )
    if reasoning_state_digest(policy.stable_compiler) != compiler_before:
        raise RuntimeError("procedural adapter training changed the frozen compiler")
    if outside_after != outside_before:
        raise RuntimeError("procedural adapter training changed pre-existing slow state")
    if reverse_after != reverse_before:
        raise RuntimeError("forward adapter training changed the reverse fast weight")
    if adapter_after == adapter_before:
        raise RuntimeError("procedural adapter training produced no learned update")
    if _parameter_identity_fingerprint(policy) != parameter_identity:
        raise RuntimeError("procedural adapter training replaced parameter identities")
    if query_counts_per_mapping != [32] * profile.meta_steps:
        raise RuntimeError("adapter query accounting differs across mappings")
    return {
        "training_stage": "procedural_adapter",
        "outer_steps": profile.meta_steps,
        "fresh_opaque_mappings": profile.meta_steps,
        "support_presentations_per_mapping": 40,
        "query_presentations_per_mapping": 32,
        "total_support_presentations": 40 * profile.meta_steps,
        "total_query_presentations": sum(cohort_case_counts.values()),
        "attempted_outputs_per_query": 2,
        "total_scored_query_attempts": 2 * sum(cohort_case_counts.values()),
        "scalar_comparisons_with_distinct_outcomes": sum(reward_separations),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "first_actual_preference_loss": actual_losses[0],
        "last_actual_preference_loss": actual_losses[-1],
        "first_forward_preference_loss": forward_losses[0],
        "last_forward_preference_loss": forward_losses[-1],
        "first_specificity_loss": specificity_losses[0],
        "last_specificity_loss": specificity_losses[-1],
        "mean_gradient_norm": _operator_audit_mean(gradient_norms),
        "cohort_case_counts": cohort_case_counts,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "optimizer_identity": optimizer_identity,
        "adapter_fingerprint_before": adapter_before,
        "adapter_fingerprint_after": adapter_after,
        "outside_adapter_fingerprint_before": outside_before,
        "outside_adapter_fingerprint_after": outside_after,
        "reverse_adapter_fingerprint_before": reverse_before,
        "reverse_adapter_fingerprint_after": reverse_after,
        "target_permutations_used_for_training": False,
        "candidate_utility_vectors_used_for_training": False,
        "feedback_signal": "two attempted public outputs plus scalar scores",
        "correct_code_control": "three same-arity codes plus zero code",
        "objective_weights": {
            "unary_actual": _PROCEDURAL_ADAPTER_UNARY_ACTUAL_WEIGHT,
            "unary_forward": _PROCEDURAL_ADAPTER_FORWARD_WEIGHT,
            "unary_specificity": _PROCEDURAL_ADAPTER_SPECIFICITY_WEIGHT,
            "binary_actual": _PROCEDURAL_ADAPTER_BINARY_ACTUAL_WEIGHT,
        },
        "optimizer_steps": profile.meta_steps,
    }


def _train_reverse_construction(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Jointly learn feedback-derived codes and bidirectional procedure use."""

    trainable_names = _configure_stage_trainability(policy, "reverse_construction")
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=profile.learning_rate,
        weight_decay=0.0,
    )
    optimizer_identity = _optimizer_identity_fingerprint(optimizer, policy)
    parameter_identity = _parameter_identity_fingerprint(policy)
    compiler_before = reasoning_state_digest(policy.stable_compiler)

    def fingerprint(include: Callable[[str], bool], domain: bytes) -> str:
        return _named_state_fingerprint(policy, include=include, domain=domain)

    learned_before = fingerprint(
        _is_reverse_construction_state,
        b"project-angler.phase5-reverse-construction.v1",
    )
    outside_before = fingerprint(
        lambda name: not _is_reverse_construction_state(name),
        b"project-angler.phase5-outside-reverse-construction.v1",
    )
    memory_before = fingerprint(
        lambda name: name in _REVERSE_CONSTRUCTION_TRAINABLE_NAMES,
        b"project-angler.phase5-reverse-code-acquisition.v1",
    )
    adapter_before = fingerprint(
        lambda name: name.startswith("procedural_fast_adapter."),
        b"project-angler.phase5-reverse-fast-adapter.v1",
    )
    goal_before = fingerprint(
        lambda name: name.startswith("procedural_goal_projection."),
        b"project-angler.phase5-reverse-goal-projection.v1",
    )
    losses: list[float] = []
    actual_losses: list[float] = []
    combined_losses: list[float] = []
    component_losses: list[float] = []
    specificity_losses: list[float] = []
    cycle_losses: list[float] = []
    gradient_norms: list[float] = []
    preference_edges: list[int] = []
    cohort_case_counts = {
        cohort: 0 for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
    }
    policy.train()
    policy.stable_compiler.eval()
    for step in range(profile.meta_steps):
        episode_seed = seed + 100_003 * (step + 1)
        _seed_reproducible_stage(
            episode_seed,
            "reverse-construction-training",
            next(policy.parameters()).device,
        )
        partition, judge = _load_training_partition(
            episode_seed,
            _OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
        groups = _group_evaluator_pairs(partition.tasks)
        arity_symbols = _operator_audit_arity_symbols(partition.tasks)
        state, support_identities = _acquire_reverse_construction_state(
            policy,
            groups,
            judge,
            episode_index=step,
        )
        queries = _operator_audit_queries(groups)
        if len(support_identities) != 40 or len(queries) != 32:
            raise RuntimeError(
                "reverse construction mapping has an unexpected presentation count"
            )
        if set(support_identities) & {
            pair.hidden.source_instance_identity for pair in queries
        }:
            raise RuntimeError(
                "reverse construction support and query instances overlap"
            )
        by_cohort: dict[str, list[torch.Tensor]] = {
            cohort: [] for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
        }
        step_actual: list[torch.Tensor] = []
        step_combined: list[torch.Tensor] = []
        step_components: list[torch.Tensor] = []
        step_specificity: list[torch.Tensor] = []
        step_cycles: list[torch.Tensor] = []
        step_edges = 0
        for query_index, pair in enumerate(queries):
            cohort = _procedural_adapter_training_cohort(pair)
            if cohort is None:
                continue
            scores = policy.score_task(pair.learner, state)
            if float(scores.root_available.detach().item()) != 1.0:
                continue
            candidate_indices = _reverse_construction_candidate_set(
                step,
                query_index,
            )
            scalar_scores = _scalar_attempt_scores(
                pair,
                candidate_indices,
                judge,
            )
            actual_loss, observed_edges = _scalar_multi_preference_loss(
                scores.logits,
                candidate_indices,
                scalar_scores,
            )
            step_edges += observed_edges
            step_actual.append(actual_loss)
            if cohort == "binary_root":
                by_cohort[cohort].append(actual_loss)
                cohort_case_counts[cohort] += 1
                continue

            correct = _root_reverse_construction_energies(
                policy,
                scores,
                scores.root.memory_read.plastic_context,
            )
            combined_loss, _ = _scalar_multi_preference_loss(
                correct["combined"],
                candidate_indices,
                scalar_scores,
            )
            per_component = torch.stack(
                tuple(
                    _scalar_multi_preference_loss(
                        correct[name],
                        candidate_indices,
                        scalar_scores,
                    )[0]
                    for name in ("goal", "forward", "reverse")
                )
            ).mean()
            alternatives = _same_arity_root_codes(
                policy,
                state,
                pair,
                scores,
                arity_symbols,
            )
            control_codes = alternatives + (
                torch.zeros_like(scores.root.memory_read.plastic_context),
            )
            control_losses = tuple(
                _scalar_multi_preference_loss(
                    _root_reverse_construction_energies(
                        policy,
                        scores,
                        code,
                    )["combined"],
                    candidate_indices,
                    scalar_scores,
                )[0]
                for code in control_codes
            )
            specificity = torch.stack(
                tuple(
                    F.relu(0.05 + combined_loss - control_loss)
                    for control_loss in control_losses
                )
            ).mean()
            case_loss = (
                0.5 * actual_loss
                + combined_loss
                + 0.25 * per_component
                + 0.5 * specificity
                + 0.05 * correct["cycle"]
            )
            by_cohort[cohort].append(case_loss)
            step_combined.append(combined_loss)
            step_components.append(per_component)
            step_specificity.append(specificity)
            step_cycles.append(correct["cycle"])
            cohort_case_counts[cohort] += 1
        if any(
            len(by_cohort[cohort]) != expected
            for cohort, expected in {
                "unary_depth2": 16,
                "unary_depth3": 4,
                "unary_direct_binary_child": 4,
                "binary_root": 8,
            }.items()
        ):
            raise RuntimeError(
                "reverse construction skipped a declared query cohort"
            )
        loss = torch.stack(
            tuple(
                torch.stack(by_cohort[cohort]).mean()
                for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
            )
        ).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("reverse construction produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            profile.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        actual_losses.append(float(torch.stack(step_actual).mean().detach().item()))
        combined_losses.append(
            float(torch.stack(step_combined).mean().detach().item())
        )
        component_losses.append(
            float(torch.stack(step_components).mean().detach().item())
        )
        specificity_losses.append(
            float(torch.stack(step_specificity).mean().detach().item())
        )
        cycle_losses.append(float(torch.stack(step_cycles).mean().detach().item()))
        gradient_norms.append(float(gradient_norm.detach().item()))
        preference_edges.append(step_edges)
    policy.eval()
    policy.requires_grad_(False)
    learned_after = fingerprint(
        _is_reverse_construction_state,
        b"project-angler.phase5-reverse-construction.v1",
    )
    outside_after = fingerprint(
        lambda name: not _is_reverse_construction_state(name),
        b"project-angler.phase5-outside-reverse-construction.v1",
    )
    memory_after = fingerprint(
        lambda name: name in _REVERSE_CONSTRUCTION_TRAINABLE_NAMES,
        b"project-angler.phase5-reverse-code-acquisition.v1",
    )
    adapter_after = fingerprint(
        lambda name: name.startswith("procedural_fast_adapter."),
        b"project-angler.phase5-reverse-fast-adapter.v1",
    )
    goal_after = fingerprint(
        lambda name: name.startswith("procedural_goal_projection."),
        b"project-angler.phase5-reverse-goal-projection.v1",
    )
    if reasoning_state_digest(policy.stable_compiler) != compiler_before:
        raise RuntimeError("reverse construction changed the frozen compiler")
    if outside_after != outside_before:
        raise RuntimeError("reverse construction changed undeclared slow state")
    if any(
        before == after
        for before, after in (
            (learned_before, learned_after),
            (memory_before, memory_after),
            (adapter_before, adapter_after),
            (goal_before, goal_after),
        )
    ):
        raise RuntimeError("reverse construction left a required learned group unchanged")
    if _parameter_identity_fingerprint(policy) != parameter_identity:
        raise RuntimeError("reverse construction replaced parameter identities")
    return {
        "training_stage": "reverse_construction",
        "outer_steps": profile.meta_steps,
        "fresh_opaque_mappings": profile.meta_steps,
        "support_presentations_per_mapping": 40,
        "query_presentations_per_mapping": 32,
        "attempted_outputs_per_query": _REVERSE_CONSTRUCTION_ATTEMPTS,
        "total_support_presentations": 40 * profile.meta_steps,
        "total_query_presentations": 32 * profile.meta_steps,
        "total_scored_query_attempts": (
            _REVERSE_CONSTRUCTION_ATTEMPTS * 32 * profile.meta_steps
        ),
        "total_observed_preference_edges": sum(preference_edges),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "first_actual_preference_loss": actual_losses[0],
        "last_actual_preference_loss": actual_losses[-1],
        "first_reverse_construction_loss": combined_losses[0],
        "last_reverse_construction_loss": combined_losses[-1],
        "first_component_loss": component_losses[0],
        "last_component_loss": component_losses[-1],
        "first_specificity_loss": specificity_losses[0],
        "last_specificity_loss": specificity_losses[-1],
        "first_cycle_loss": cycle_losses[0],
        "last_cycle_loss": cycle_losses[-1],
        "mean_gradient_norm": _operator_audit_mean(gradient_norms),
        "cohort_case_counts": cohort_case_counts,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "optimizer_identity": optimizer_identity,
        "learned_state_fingerprint_before": learned_before,
        "learned_state_fingerprint_after": learned_after,
        "outside_learned_fingerprint_before": outside_before,
        "outside_learned_fingerprint_after": outside_after,
        "code_acquisition_fingerprint_before": memory_before,
        "code_acquisition_fingerprint_after": memory_after,
        "fast_adapter_fingerprint_before": adapter_before,
        "fast_adapter_fingerprint_after": adapter_after,
        "goal_projection_fingerprint_before": goal_before,
        "goal_projection_fingerprint_after": goal_after,
        "support_graph_detached": False,
        "competence_state_crosses_mapping_boundary": False,
        "target_permutations_used_for_training": False,
        "candidate_utility_vectors_used_for_training": False,
        "hidden_operator_labels_used_for_training": False,
        "feedback_signal": (
            "four attempted public outputs plus their scalar scores"
        ),
        "optimizer_steps": profile.meta_steps,
    }


def _train_deployed_preference_adaptation(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
    *,
    stage: str,
    state_selector: Callable[[str], bool],
    support_graph_detached: bool,
    objective: str,
    fingerprint_domain: bytes,
    outside_fingerprint_domain: bytes,
) -> dict[str, Any]:
    """Adapt a declared procedural seam through deployed scalar preferences."""

    if objective not in {"pairwise_preference", "on_policy_reward"}:
        raise ValueError("deployed adaptation objective is invalid")
    trainable_names = _configure_stage_trainability(
        policy,
        stage,
    )
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    group_selectors: tuple[tuple[str, Callable[[str], bool]], ...] = (
        (
            "leaf_code_acquisition",
            lambda name: name == "memory.feedback_direction_encoder.3.weight",
        ),
        (
            "composition_code_acquisition",
            lambda name: (
                name == "composition_memory.feedback_direction_encoder.3.weight"
            ),
        ),
        (
            "fast_adapter",
            lambda name: name.startswith("procedural_fast_adapter."),
        ),
        (
            "goal_projection",
            lambda name: name.startswith("procedural_goal_projection."),
        ),
        (
            "direction_mixer",
            lambda name: name.startswith("phase4_direction_mixer."),
        ),
        (
            "reliability_gate",
            lambda name: name.startswith("phase4_reliability_gate."),
        ),
        (
            "reversible_transition",
            lambda name: name.startswith("reversible_procedure_transition."),
        ),
    )
    active_group_selectors = tuple(
        (label, selector)
        for label, selector in group_selectors
        if any(selector(name) for name in trainable_names)
    )
    grouped_trainable_names = {
        name
        for name in trainable_names
        if any(selector(name) for _, selector in active_group_selectors)
    }
    if grouped_trainable_names != set(trainable_names):
        raise RuntimeError(f"{stage} has an unclassified trainable parameter")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=profile.learning_rate,
        weight_decay=0.0,
    )
    optimizer_identity = _optimizer_identity_fingerprint(optimizer, policy)
    parameter_identity = _parameter_identity_fingerprint(policy)
    compiler_before = reasoning_state_digest(policy.stable_compiler)
    harmonizer_before = _named_state_fingerprint(
        policy,
        include=state_selector,
        domain=fingerprint_domain,
    )
    direction_mixer_before = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("phase4_direction_mixer."),
        domain=b"project-angler.phase5-reverse-direction-mixer.v1",
    )
    reliability_gate_before = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("phase4_reliability_gate."),
        domain=b"project-angler.phase5-reverse-reliability-gate.v1",
    )
    outside_before = _named_state_fingerprint(
        policy,
        include=lambda name: not state_selector(name),
        domain=outside_fingerprint_domain,
    )
    group_fingerprints_before = {
        label: _named_state_fingerprint(
            policy,
            include=selector,
            domain=(
                b"project-angler.phase5-deployed-preference-group."
                + label.encode("ascii")
            ),
        )
        for label, selector in active_group_selectors
    }
    losses: list[float] = []
    gradient_norms: list[float] = []
    preference_edges: list[int] = []
    cohort_loss_history: dict[str, list[float]] = {
        cohort: [] for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
    }
    cohort_case_counts = {
        cohort: 0 for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
    }
    gradient_reached = {label: False for label, _ in active_group_selectors}
    policy.train()
    policy.stable_compiler.eval()
    for step in range(profile.meta_steps):
        episode_seed = seed + 100_003 * (step + 1)
        _seed_reproducible_stage(
            episode_seed,
            f"{stage}-training",
            next(policy.parameters()).device,
        )
        partition, judge = _load_training_partition(
            episode_seed,
            _OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
        groups = _group_evaluator_pairs(partition.tasks)
        if support_graph_detached:
            with torch.no_grad():
                state, support_identities = _acquire_reverse_construction_state(
                    policy,
                    groups,
                    judge,
                    episode_index=step,
                )
        else:
            state, support_identities = _acquire_reverse_construction_state(
                policy,
                groups,
                judge,
                episode_index=step,
            )
        queries = _operator_audit_queries(groups)
        if len(support_identities) != 40 or len(queries) != 32:
            raise RuntimeError(
                "reverse harmonization mapping has an unexpected presentation count"
            )
        if set(support_identities) & {
            pair.hidden.source_instance_identity for pair in queries
        }:
            raise RuntimeError(
                "reverse harmonization support and query instances overlap"
            )
        by_cohort: dict[str, list[torch.Tensor]] = {
            cohort: [] for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
        }
        step_edges = 0
        for query_index, pair in enumerate(queries):
            cohort = _procedural_adapter_training_cohort(pair)
            if cohort is None:
                continue
            scores = policy.score_task(pair.learner, state)
            if float(scores.root_available.detach().item()) != 1.0:
                raise RuntimeError(
                    "reverse harmonization encountered unavailable root evidence"
                )
            candidate_indices = (
                _on_policy_reward_candidate_set(scores.logits, step, query_index)
                if objective == "on_policy_reward"
                else _reverse_construction_candidate_set(step, query_index)
            )
            scalar_scores = _scalar_attempt_scores(
                pair,
                candidate_indices,
                judge,
            )
            if objective == "on_policy_reward":
                actual_loss = _scalar_on_policy_reward_loss(
                    scores.logits,
                    candidate_indices,
                    scalar_scores,
                )
                observed_edges = sum(
                    scalar_scores[left] != scalar_scores[right]
                    for left in range(len(scalar_scores))
                    for right in range(left + 1, len(scalar_scores))
                )
            else:
                actual_loss, observed_edges = _scalar_multi_preference_loss(
                    scores.logits,
                    candidate_indices,
                    scalar_scores,
                )
            by_cohort[cohort].append(actual_loss)
            cohort_case_counts[cohort] += 1
            step_edges += observed_edges
        if any(
            len(by_cohort[cohort]) != expected
            for cohort, expected in {
                "unary_depth2": 16,
                "unary_depth3": 4,
                "unary_direct_binary_child": 4,
                "binary_root": 8,
            }.items()
        ):
            raise RuntimeError(
                "reverse harmonization skipped a declared query cohort"
            )
        cohort_losses = {
            cohort: torch.stack(by_cohort[cohort]).mean()
            for cohort in _PROCEDURAL_ADAPTER_TRAINING_COHORTS
        }
        loss = torch.stack(tuple(cohort_losses.values())).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("reverse harmonization produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for label, selector in active_group_selectors:
            gradient_reached[label] = gradient_reached[label] or any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                and bool(parameter.grad.detach().count_nonzero())
                for name, parameter in policy.named_parameters()
                if selector(name)
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            profile.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(gradient_norm.detach().item()))
        preference_edges.append(step_edges)
        for cohort, cohort_loss in cohort_losses.items():
            cohort_loss_history[cohort].append(
                float(cohort_loss.detach().item())
            )
    policy.eval()
    policy.requires_grad_(False)
    harmonizer_after = _named_state_fingerprint(
        policy,
        include=state_selector,
        domain=fingerprint_domain,
    )
    direction_mixer_after = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("phase4_direction_mixer."),
        domain=b"project-angler.phase5-reverse-direction-mixer.v1",
    )
    reliability_gate_after = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("phase4_reliability_gate."),
        domain=b"project-angler.phase5-reverse-reliability-gate.v1",
    )
    outside_after = _named_state_fingerprint(
        policy,
        include=lambda name: not state_selector(name),
        domain=outside_fingerprint_domain,
    )
    group_fingerprints_after = {
        label: _named_state_fingerprint(
            policy,
            include=selector,
            domain=(
                b"project-angler.phase5-deployed-preference-group."
                + label.encode("ascii")
            ),
        )
        for label, selector in active_group_selectors
    }
    if reasoning_state_digest(policy.stable_compiler) != compiler_before:
        raise RuntimeError(f"{stage} changed the frozen compiler")
    if outside_after != outside_before:
        raise RuntimeError(f"{stage} changed state outside its declared seam")
    if harmonizer_after == harmonizer_before:
        raise RuntimeError(f"{stage} left its complete trainable seam unchanged")
    active_group_labels = {label for label, _ in active_group_selectors}
    if (
        "direction_mixer" in active_group_labels
        and direction_mixer_after == direction_mixer_before
    ):
        raise RuntimeError("reverse harmonization left the direction mixer unchanged")
    if (
        "reliability_gate" in active_group_labels
        and reliability_gate_after == reliability_gate_before
    ):
        raise RuntimeError("reverse harmonization left the reliability gate unchanged")
    if not all(gradient_reached.values()):
        raise RuntimeError(
            f"deployed preference loss did not reach every {stage} group"
        )
    if _parameter_identity_fingerprint(policy) != parameter_identity:
        raise RuntimeError(f"{stage} replaced parameter identities")
    return {
        "training_stage": stage,
        "outer_steps": profile.meta_steps,
        "fresh_opaque_mappings": profile.meta_steps,
        "support_presentations_per_mapping": 40,
        "query_presentations_per_mapping": 32,
        "attempted_outputs_per_query": _REVERSE_CONSTRUCTION_ATTEMPTS,
        "total_support_presentations": 40 * profile.meta_steps,
        "total_query_presentations": 32 * profile.meta_steps,
        "total_scored_query_attempts": (
            _REVERSE_CONSTRUCTION_ATTEMPTS * 32 * profile.meta_steps
        ),
        "total_observed_preference_edges": sum(preference_edges),
        "first_deployed_preference_loss": losses[0],
        "last_deployed_preference_loss": losses[-1],
        "minimum_deployed_preference_loss": min(losses),
        "first_training_objective_loss": losses[0],
        "last_training_objective_loss": losses[-1],
        "minimum_training_objective_loss": min(losses),
        "first_cohort_preference_losses": {
            cohort: values[0] for cohort, values in cohort_loss_history.items()
        },
        "last_cohort_preference_losses": {
            cohort: values[-1] for cohort, values in cohort_loss_history.items()
        },
        "mean_gradient_norm": _operator_audit_mean(gradient_norms),
        "cohort_case_counts": cohort_case_counts,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "optimizer_identity": optimizer_identity,
        "harmonizer_fingerprint_before": harmonizer_before,
        "harmonizer_fingerprint_after": harmonizer_after,
        "direction_mixer_fingerprint_before": direction_mixer_before,
        "direction_mixer_fingerprint_after": direction_mixer_after,
        "reliability_gate_fingerprint_before": reliability_gate_before,
        "reliability_gate_fingerprint_after": reliability_gate_after,
        "outside_harmonizer_fingerprint_before": outside_before,
        "outside_harmonizer_fingerprint_after": outside_after,
        "trainable_group_fingerprints_before": group_fingerprints_before,
        "trainable_group_fingerprints_after": group_fingerprints_after,
        "deployed_preference_gradient_reached_groups": gradient_reached,
        "deployed_preference_gradient_reached_direction_mixer": gradient_reached.get(
            "direction_mixer", False
        ),
        "deployed_preference_gradient_reached_reliability_gate": gradient_reached.get(
            "reliability_gate", False
        ),
        "support_graph_detached": support_graph_detached,
        "competence_state_crosses_mapping_boundary": False,
        "target_permutations_used_for_training": False,
        "candidate_utility_vectors_used_for_training": False,
        "hidden_operator_labels_used_for_training": False,
        "auxiliary_ranking_objectives_used_for_training": False,
        "training_objective": objective,
        "reward_credit_baseline": (
            "mean of the observed attempts in the same public query"
            if objective == "on_policy_reward"
            else "not applicable"
        ),
        "current_deployed_greedy_attempted_per_query": (
            objective == "on_policy_reward"
        ),
        "complete_action_softmax_used_for_training": (
            objective == "on_policy_reward"
        ),
        "feedback_signal": (
            "four attempted public outputs plus their scalar scores"
        ),
        "optimizer_steps": profile.meta_steps,
    }


def _train_reverse_harmonization(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Learn arbitration among frozen procedural channels on deployed logits."""

    return _train_deployed_preference_adaptation(
        policy,
        profile,
        seed,
        stage="reverse_harmonization",
        state_selector=_is_reverse_harmonization_state,
        support_graph_detached=True,
        objective="pairwise_preference",
        fingerprint_domain=b"project-angler.phase5-reverse-harmonization.v1",
        outside_fingerprint_domain=(
            b"project-angler.phase5-outside-reverse-harmonization.v1"
        ),
    )


def _train_procedural_coadaptation(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Jointly adapt feedback acquisition, execution, and arbitration."""

    return _train_deployed_preference_adaptation(
        policy,
        profile,
        seed,
        stage="procedural_coadaptation",
        state_selector=_is_procedural_coadaptation_state,
        support_graph_detached=False,
        objective="on_policy_reward",
        fingerprint_domain=b"project-angler.phase5-procedural-coadaptation.v1",
        outside_fingerprint_domain=(
            b"project-angler.phase5-outside-procedural-coadaptation.v1"
        ),
    )


def _train_reversible_transition_acquisition(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Acquire one invertible procedure map directly from deployed outcomes."""

    if bool(policy.reversible_transition_mode.item()):
        raise RuntimeError("reversible transition mode was active before acquisition")
    policy.reversible_transition_mode.fill_(True)
    return _train_deployed_preference_adaptation(
        policy,
        profile,
        seed,
        stage="reversible_transition_acquisition",
        state_selector=_is_reversible_transition_acquisition_state,
        support_graph_detached=False,
        objective="on_policy_reward",
        fingerprint_domain=(
            b"project-angler.phase5-reversible-transition-acquisition.v1"
        ),
        outside_fingerprint_domain=(
            b"project-angler.phase5-outside-reversible-transition-acquisition.v1"
        ),
    )


def _stratified_leaf_candidate_index(
    episode_index: int,
    root_offset: int,
    support_index: int,
) -> int:
    """Choose a varied public intervention without task or reward access."""

    for name, value in (
        ("episode_index", episode_index),
        ("root_offset", root_offset),
        ("support_index", support_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return (11 + 31 * episode_index + 43 * support_index + 19 * root_offset) % len(
        _PERMUTATIONS
    )


def _train(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
    *,
    stage: str = "integrated",
) -> dict[str, Any]:
    if stage not in _TRAINING_STAGES:
        raise ValueError(f"training stage must be one of {_TRAINING_STAGES}")
    composition_training = stage != "leaf_core"
    harmonization_training = stage == "harmonization"
    feedback_causal_weight = 0.0 if harmonization_training else 5.0
    phase4_residual_weight = 0.5
    phase4_direction_weight = 0.0 if harmonization_training else 0.25
    local_phase4_alignment_weight = 0.0 if harmonization_training else 0.5
    support_consistency_weight = 0.0 if harmonization_training else 0.005
    route_balance_weight = 0.0 if harmonization_training else 0.01
    trainable_names = _configure_stage_trainability(policy, stage)
    leaf_substrate_before = _named_state_fingerprint(
        policy,
        include=lambda name: not name.startswith("stable_compiler.")
        and not _is_composition_state(name),
        domain=b"project-angler.phase5-leaf-substrate.v1",
    )
    preexisting_state_before = _named_state_fingerprint(
        policy,
        include=lambda name: not name.startswith(
            _RELATIONAL_ACQUISITION_PREFIXES
        ),
        domain=b"project-angler.phase5-preexisting-state.v1",
    )
    condition_axis_before = _named_state_fingerprint(
        policy,
        include=lambda name: name == _CONDITION_AXIS_KEY,
        domain=b"project-angler.phase5-condition-axis.v1",
    )
    relational_acquisition_before = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith(_RELATIONAL_ACQUISITION_PREFIXES),
        domain=b"project-angler.phase5-relational-acquisition.v1",
    )
    harmonization_before = _named_state_fingerprint(
        policy,
        include=_is_harmonization_state,
        domain=b"project-angler.phase5-harmonization.v1",
    )
    outside_harmonization_before = _named_state_fingerprint(
        policy,
        include=lambda name: not _is_harmonization_state(name),
        domain=b"project-angler.phase5-outside-harmonization.v1",
    )
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("meta-training has no trainable slow parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=profile.learning_rate, weight_decay=1.0e-4
    )
    parameter_identity_before = _parameter_identity_fingerprint(policy)
    optimizer_identity_before = _optimizer_identity_fingerprint(optimizer, policy)
    compiler_before = reasoning_state_digest(policy.stable_compiler)
    losses: list[float] = []
    task_losses: list[float] = []
    support_losses: list[float] = []
    causal_losses: list[float] = []
    counterfactual_losses: list[float] = []
    descendant_residual_losses: list[float] = []
    phase4_residual_losses: list[float] = []
    phase4_direction_losses: list[float] = []
    local_phase4_alignment_losses: list[float] = []
    matched_descendant_losses: list[float] = []
    matched_cross_losses: list[float] = []
    matched_delta_losses: list[float] = []
    matched_complete_counts: list[int] = []
    balance_losses: list[float] = []
    gradient_norms: list[float] = []
    phase4_residual_root_arities: set[int] = set()
    matched_pairs_per_step = 0
    policy.train()
    policy.stable_compiler.eval()
    for step in range(profile.meta_steps):
        # A fresh seed creates a fresh opaque mapping and fresh public symbols
        # for every outer episode; the slow model cannot memorize their names.
        episode_seed = seed + 100_003 * (step + 1)
        partition, judge = _load_training_partition(
            episode_seed, profile.meta_instances_per_program
        )
        groups = _group_evaluator_pairs(partition.tasks)
        query_losses: list[torch.Tensor] = []
        support_consistency_losses: list[torch.Tensor] = []
        feedback_causal_losses: list[torch.Tensor] = []
        step_counterfactual_losses: list[torch.Tensor] = []
        step_descendant_residual_losses: list[torch.Tensor] = []
        step_phase4_residual_losses: list[torch.Tensor] = []
        step_phase4_direction_losses: list[torch.Tensor] = []
        step_local_phase4_alignment_losses: list[torch.Tensor] = []
        step_matched_descendant_losses: list[torch.Tensor] = []
        step_matched_cross_losses: list[torch.Tensor] = []
        step_matched_delta_losses: list[torch.Tensor] = []
        interference_route_probabilities: list[torch.Tensor] = []
        torch.manual_seed(seed + 17_003 * (step + 1))

        def consume_pair(
            incoming: ProceduralSkillState,
            support_pair: Any,
            query_pair: Any,
            *,
            balance_route: bool,
            include_compiler_query: bool,
            candidate_index: int | None = None,
        ) -> tuple[ProceduralSkillState, TaskProposal, float]:
            proposal = (
                propose_task(
                    policy,
                    support_pair.learner,
                    incoming,
                    greedy=False,
                    temperature=1.25,
                )
                if candidate_index is None
                else _proposal_for_candidate(
                    policy,
                    support_pair.learner,
                    incoming,
                    candidate_index,
                    include_compiler=include_compiler_query,
                )
            )
            scalar_score = _judge_frozen_answer(support_pair, proposal.answer, judge)
            staged = propose_differentiable_feedback(
                policy, proposal, scalar_score, incoming
            )
            # This candidate state is intentionally unadmitted: it is the
            # differentiable inner update optimized by later-query outer loss.
            if balance_route:
                interference_route_probabilities.append(
                    staged.route_probabilities
                )
            support_logits = policy.score_task(
                support_pair.learner,
                staged.candidate_state,
                include_compiler=include_compiler_query,
            ).logits
            support_consistency_losses.append(
                _scalar_feedback_tensor(
                    support_logits,
                    proposal.candidate_index,
                    scalar_score,
                )
            )
            adapted_query_loss = _outer_query_loss(
                policy,
                staged.candidate_state,
                query_pair,
                include_compiler=include_compiler_query,
            )
            query_losses.append(adapted_query_loss)
            return staged.candidate_state, proposal, scalar_score

        # Short per-skill episodes teach repeated adaptation without a long
        # gradient chain.  A separate shared-state episode then teaches
        # interference resistance, returns, and cross-skill composition.
        for root_offset, local_sequence in enumerate(
            _meta_local_sequences(groups, step)
        ):
            local_state = policy.initial_state(1)
            inverted_state = policy.initial_state(1)
            local_queries: list[Any] = []
            for support_index, (support_pair, query_pair) in enumerate(local_sequence):
                local_state, ordinary_proposal, scalar_score = consume_pair(
                    local_state,
                    support_pair,
                    query_pair,
                    balance_route=False,
                    include_compiler_query=False,
                    candidate_index=_stratified_leaf_candidate_index(
                        step,
                        root_offset,
                        support_index,
                    ),
                )
                if composition_training and not harmonization_training:
                    if query_pair.learner.request.children:
                        raise RuntimeError(
                            "local Phase-4 alignment received a non-atomic query"
                        )
                    local_bridge = policy.score_task(
                        query_pair.learner,
                        local_state,
                        probe_leaf_bridge=True,
                    )
                    step_local_phase4_alignment_losses.append(
                        (
                            _outer_top_target_loss(
                                local_bridge.phase4_bridge_logits,
                                query_pair,
                            )
                            + _outer_top_target_loss(
                                local_bridge.phase4_forward_evidence,
                                query_pair,
                            )
                            + _outer_top_target_loss(
                                local_bridge.phase4_reverse_evidence,
                                query_pair,
                            )
                        )
                        / 3.0
                    )
                # One scalar observation is usually compatible with many
                # procedures.  Reward causality is therefore tested only after
                # a varied evidence set has accumulated.  The counterfactual
                # receives the exact same public tasks and attempted candidates
                # as the ordinary path, with only the scalar outcomes inverted.
                inverted_proposal = _proposal_for_candidate(
                    policy,
                    support_pair.learner,
                    inverted_state,
                    ordinary_proposal.candidate_index,
                )
                inverted_state = propose_differentiable_feedback(
                    policy,
                    inverted_proposal,
                    1.0 - scalar_score,
                    inverted_state,
                ).candidate_state
                local_queries.append(query_pair)
            ordinary_set_loss = torch.stack(
                tuple(
                    _outer_query_loss(
                        policy,
                        local_state,
                        pair,
                        include_compiler=False,
                    )
                    for pair in local_queries
                )
            ).mean()
            inverted_on_ordinary_loss = torch.stack(
                tuple(
                    _outer_query_loss(
                        policy,
                        inverted_state,
                        pair,
                        include_compiler=False,
                    )
                    for pair in local_queries
                )
            ).mean()
            inverted_counterfactual_loss = torch.stack(
                tuple(
                    _outer_counterfactual_query_loss(
                        policy,
                        inverted_state,
                        pair,
                        include_compiler=False,
                    )
                    for pair in local_queries
                )
            ).mean()
            # The final accumulated states need proper absolute objectives.
            # The old hinge alone could succeed by degrading the inverted
            # state while leaving the ordinary state mediocre.  Complemented
            # pairwise outcomes describe the exact reverse relation, so the
            # counterfactual target is coherent without exposing a procedure
            # label, rule, or solution to the online learner.
            query_losses.extend((ordinary_set_loss, inverted_counterfactual_loss))
            step_counterfactual_losses.append(inverted_counterfactual_loss)
            feedback_causal_losses.append(
                F.relu(0.02 + ordinary_set_loss - inverted_on_ordinary_loss)
            )
        if composition_training:
            state = policy.initial_state(1)
            for support_pair, query_pair in _meta_episode_sequence(groups, step):
                state, _, _ = consume_pair(
                    state,
                    support_pair,
                    query_pair,
                    balance_route=True,
                    include_compiler_query=True,
                )
            # These queries are the first point at which the complete acquired
            # primitive set is composed.  Their gradients traverse the frozen
            # Phase-4 compiler into the learned evidence memory and composer.
            for pair in _meta_composition_queries(groups, step):
                scores = policy.score_task(pair.learner, state)
                query_losses.append(_outer_logits_loss(scores.logits, pair))
                # Keep the root-only contrast as a diagnostic.  It is not a
                # training objective: optimizing a difference can satisfy the
                # contrast by degrading the ablated policy instead of making
                # the complete policy correct.  Absolute matched-descendant
                # objectives below hold child structure accountable instead.
                with torch.no_grad():
                    root_only = policy.score_task(
                        pair.learner,
                        state,
                        include_descendants=False,
                    )
                    step_descendant_residual_losses.append(
                        _outer_top_target_loss(
                            scores.logits.detach() - root_only.logits,
                            pair,
                        )
                    )
                if not harmonization_training or len(pair.learner.request.children) == 1:
                    phase4_residual_root_arities.add(
                        len(pair.learner.request.children)
                    )
                    step_phase4_residual_losses.append(
                        _outer_top_target_loss(scores.phase4_bridge_logits, pair)
                    )
                if not harmonization_training:
                    step_phase4_direction_losses.append(
                        0.5
                        * (
                            _outer_top_target_loss(
                                scores.phase4_forward_evidence,
                                pair,
                            )
                            + _outer_top_target_loss(
                                scores.phase4_reverse_evidence,
                                pair,
                            )
                        )
                    )
            matched_pairs = _load_matched_descendant_queries(episode_seed)
            partition_mapping = {
                pair.hidden.mapping.digest for pair in partition.tasks
            }
            matched_mapping = {
                item.left.hidden.mapping.digest for item in matched_pairs
            } | {
                item.right.hidden.mapping.digest for item in matched_pairs
            }
            if not matched_pairs or matched_mapping != partition_mapping:
                raise RuntimeError(
                    "matched descendant queries do not share the episode mapping"
                )
            if matched_pairs_per_step not in (0, len(matched_pairs)):
                raise RuntimeError("matched descendant query count changed by step")
            matched_pairs_per_step = len(matched_pairs)
            state_before_matched = procedural_skill_state_digest(state)
            for matched_pair in matched_pairs:
                try:
                    matched_loss, cross_loss, delta_loss = (
                        _outer_matched_descendant_loss(
                            policy,
                            state,
                            matched_pair,
                            include_evidence_delta=not harmonization_training,
                        )
                    )
                except _IncompleteMatchedDescendantError:
                    continue
                step_matched_descendant_losses.append(matched_loss)
                step_matched_cross_losses.append(cross_loss)
                step_matched_delta_losses.append(delta_loss)
            if procedural_skill_state_digest(state) != state_before_matched:
                raise RuntimeError("matched descendant query changed competence state")
        task_loss = torch.stack(query_losses).mean()
        support_consistency_loss = torch.stack(support_consistency_losses).mean()
        feedback_causal_loss = (
            torch.stack(feedback_causal_losses).mean()
            if feedback_causal_losses
            else task_loss.new_zeros(())
        )
        descendant_residual_loss = (
            torch.stack(step_descendant_residual_losses).mean()
            if step_descendant_residual_losses
            else task_loss.new_zeros(())
        )
        phase4_residual_loss = (
            torch.stack(step_phase4_residual_losses).mean()
            if step_phase4_residual_losses
            else task_loss.new_zeros(())
        )
        phase4_direction_loss = (
            torch.stack(step_phase4_direction_losses).mean()
            if step_phase4_direction_losses
            else task_loss.new_zeros(())
        )
        local_phase4_alignment_loss = (
            torch.stack(step_local_phase4_alignment_losses).mean()
            if step_local_phase4_alignment_losses
            else task_loss.new_zeros(())
        )
        matched_descendant_loss = (
            torch.stack(step_matched_descendant_losses).mean()
            if step_matched_descendant_losses
            else task_loss.new_zeros(())
        )
        matched_cross_loss = (
            torch.stack(step_matched_cross_losses).mean()
            if step_matched_cross_losses
            else task_loss.new_zeros(())
        )
        matched_delta_loss = (
            torch.stack(step_matched_delta_losses).mean()
            if step_matched_delta_losses
            else task_loss.new_zeros(())
        )
        matched_complete_counts.append(len(step_matched_descendant_losses))
        if interference_route_probabilities:
            mean_route = torch.cat(interference_route_probabilities, dim=0).mean(dim=0)
            load_balance_loss = profile.slots * mean_route.square().sum() - 1.0
        else:
            load_balance_loss = task_loss.new_zeros(())
        loss = (
            task_loss
            + feedback_causal_weight * feedback_causal_loss
            + phase4_residual_weight * phase4_residual_loss
            + phase4_direction_weight * phase4_direction_loss
            + local_phase4_alignment_weight * local_phase4_alignment_loss
            + matched_descendant_loss
            + support_consistency_weight * support_consistency_loss
            + route_balance_weight * load_balance_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("meta-training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, profile.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("meta-training produced a non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        task_losses.append(float(task_loss.detach().item()))
        support_losses.append(float(support_consistency_loss.detach().item()))
        causal_losses.append(float(feedback_causal_loss.detach().item()))
        counterfactual_losses.append(
            float(
                torch.stack(step_counterfactual_losses).mean().detach().item()
            )
        )
        descendant_residual_losses.append(
            float(descendant_residual_loss.detach().item())
        )
        phase4_residual_losses.append(
            float(phase4_residual_loss.detach().item())
        )
        phase4_direction_losses.append(
            float(phase4_direction_loss.detach().item())
        )
        local_phase4_alignment_losses.append(
            float(local_phase4_alignment_loss.detach().item())
        )
        matched_descendant_losses.append(
            float(matched_descendant_loss.detach().item())
        )
        matched_cross_losses.append(float(matched_cross_loss.detach().item()))
        matched_delta_losses.append(float(matched_delta_loss.detach().item()))
        balance_losses.append(float(load_balance_loss.detach().item()))
        gradient_norms.append(float(gradient_norm.detach().item()))

    parameter_identity_after = _parameter_identity_fingerprint(policy)
    optimizer_identity_after = _optimizer_identity_fingerprint(optimizer, policy)
    compiler_after = reasoning_state_digest(policy.stable_compiler)
    leaf_substrate_after = _named_state_fingerprint(
        policy,
        include=lambda name: not name.startswith("stable_compiler.")
        and not _is_composition_state(name),
        domain=b"project-angler.phase5-leaf-substrate.v1",
    )
    preexisting_state_after = _named_state_fingerprint(
        policy,
        include=lambda name: not name.startswith(
            _RELATIONAL_ACQUISITION_PREFIXES
        ),
        domain=b"project-angler.phase5-preexisting-state.v1",
    )
    condition_axis_after = _named_state_fingerprint(
        policy,
        include=lambda name: name == _CONDITION_AXIS_KEY,
        domain=b"project-angler.phase5-condition-axis.v1",
    )
    relational_acquisition_after = _named_state_fingerprint(
        policy,
        include=lambda name: name.startswith(_RELATIONAL_ACQUISITION_PREFIXES),
        domain=b"project-angler.phase5-relational-acquisition.v1",
    )
    harmonization_after = _named_state_fingerprint(
        policy,
        include=_is_harmonization_state,
        domain=b"project-angler.phase5-harmonization.v1",
    )
    outside_harmonization_after = _named_state_fingerprint(
        policy,
        include=lambda name: not _is_harmonization_state(name),
        domain=b"project-angler.phase5-outside-harmonization.v1",
    )
    if parameter_identity_after != parameter_identity_before:
        raise RuntimeError("meta-training replaced a slow parameter object")
    if optimizer_identity_after != optimizer_identity_before:
        raise RuntimeError("optimizer-to-slow-weight identity changed during training")
    if compiler_after != compiler_before:
        raise RuntimeError("meta-training changed the frozen Phase-4 compiler")
    if composition_training and leaf_substrate_after != leaf_substrate_before:
        raise RuntimeError("composition training changed the consolidated leaf substrate")
    if (
        stage == "relational_acquisition"
        and preexisting_state_after != preexisting_state_before
    ):
        raise RuntimeError("relational acquisition changed pre-existing slow state")
    if (
        stage == "relational_acquisition"
        and relational_acquisition_after == relational_acquisition_before
    ):
        raise RuntimeError("relational acquisition produced no learned update")
    if (
        stage == "harmonization"
        and outside_harmonization_after != outside_harmonization_before
    ):
        raise RuntimeError("harmonization changed state outside its neural seam")
    if stage == "harmonization" and harmonization_after == harmonization_before:
        raise RuntimeError("harmonization produced no learned update")
    return {
        "outer_steps": profile.meta_steps,
        "training_stage": stage,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "first_task_loss": task_losses[0],
        "last_task_loss": task_losses[-1],
        "first_support_consistency_loss": support_losses[0],
        "last_support_consistency_loss": support_losses[-1],
        "first_feedback_causal_loss": causal_losses[0],
        "last_feedback_causal_loss": causal_losses[-1],
        "first_counterfactual_set_loss": counterfactual_losses[0],
        "last_counterfactual_set_loss": counterfactual_losses[-1],
        "first_descendant_residual_loss": descendant_residual_losses[0],
        "last_descendant_residual_loss": descendant_residual_losses[-1],
        "descendant_residual_weight": 0.0,
        "first_phase4_residual_loss": phase4_residual_losses[0],
        "last_phase4_residual_loss": phase4_residual_losses[-1],
        "phase4_residual_weight": phase4_residual_weight,
        "phase4_residual_root_arities": sorted(phase4_residual_root_arities),
        "first_phase4_direction_loss": phase4_direction_losses[0],
        "last_phase4_direction_loss": phase4_direction_losses[-1],
        "phase4_direction_weight": phase4_direction_weight,
        "first_local_phase4_alignment_loss": local_phase4_alignment_losses[0],
        "last_local_phase4_alignment_loss": local_phase4_alignment_losses[-1],
        "local_phase4_alignment_weight": local_phase4_alignment_weight,
        "first_matched_descendant_loss": matched_descendant_losses[0],
        "last_matched_descendant_loss": matched_descendant_losses[-1],
        "first_matched_cross_loss": matched_cross_losses[0],
        "last_matched_cross_loss": matched_cross_losses[-1],
        "first_matched_delta_loss": matched_delta_losses[0],
        "last_matched_delta_loss": matched_delta_losses[-1],
        "matched_descendant_weight": 1.0,
        "matched_evidence_delta_is_objective": not harmonization_training,
        "matched_descendant_margin": 0.10,
        "matched_descendant_pairs_per_step": matched_pairs_per_step,
        "matched_complete_pairs_min": min(matched_complete_counts),
        "matched_complete_pairs_max": max(matched_complete_counts),
        "matched_complete_pairs_last": matched_complete_counts[-1],
        "feedback_causal_weight": feedback_causal_weight,
        "feedback_causal_scope": "matched_varied_evidence_set",
        "local_component_query_compiler": False,
        "local_support_policy": (
            "seeded_low_discrepancy_public_candidate_interventions"
        ),
        "route_balance_weight": route_balance_weight,
        "support_consistency_weight": support_consistency_weight,
        "last_route_balance_loss": balance_losses[-1],
        "associative_reuse_similarity_threshold": (
            policy.memory.reuse_similarity_threshold
        ),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "slow_parameter_identity": parameter_identity_after,
        "optimizer_identity": optimizer_identity_after,
        "compiler_fingerprint_before": compiler_before,
        "compiler_fingerprint_after": compiler_after,
        "leaf_substrate_fingerprint_before": leaf_substrate_before,
        "leaf_substrate_fingerprint_after": leaf_substrate_after,
        "leaf_substrate_consolidated": composition_training,
        "preexisting_state_fingerprint_before": preexisting_state_before,
        "preexisting_state_fingerprint_after": preexisting_state_after,
        "preexisting_state_consolidated": stage == "relational_acquisition",
        "condition_axis_fingerprint_before": condition_axis_before,
        "condition_axis_fingerprint_after": condition_axis_after,
        "relational_acquisition_fingerprint_before": (
            relational_acquisition_before
        ),
        "relational_acquisition_fingerprint_after": (
            relational_acquisition_after
        ),
        "harmonization_fingerprint_before": harmonization_before,
        "harmonization_fingerprint_after": harmonization_after,
        "outside_harmonization_fingerprint_before": (
            outside_harmonization_before
        ),
        "outside_harmonization_fingerprint_after": outside_harmonization_after,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if parameter.requires_grad
        ),
        "fresh_mapping_per_outer_episode": True,
        "outer_target_access_scope": "query_loss_only",
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("score summary requires at least one value")
    window = max(1, len(values) // 4)
    early = sum(values[:window]) / window
    late = sum(values[-window:]) / window
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "early_mean": early,
        "late_mean": late,
        "gain": late - early,
    }


def _run_support_segment(
    policy: SkillMemoryPolicy,
    incoming_state: ProceduralSkillState,
    pairs: Sequence[Any],
    judge: Callable[..., float],
    *,
    mode: str,
    probe_pairs: Sequence[Any] | None = None,
    recovery_target: float | None = None,
) -> tuple[ProceduralSkillState, dict[str, Any]]:
    if mode not in ("ordinary", "inverted", "no_write"):
        raise ValueError("unknown online support mode")
    state = _detached_state(incoming_state)
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    accepted = 0
    core_accepted = 0
    probe_trace: list[float] = []
    recovery_step: int | None = None
    if recovery_target is not None and probe_pairs is None:
        raise ValueError("recovery_target requires disjoint probe pairs")
    if recovery_target is not None and (
        not math.isfinite(recovery_target) or not 0.0 <= recovery_target <= 1.0
    ):
        raise ValueError("recovery_target must be between zero and one")
    if probe_pairs is not None and recovery_target is not None:
        initial_probe = _summary(
            _score_no_feedback(policy, state, probe_pairs, judge)
        )["mean"]
        if float(initial_probe) >= recovery_target:
            recovery_step = 0
    for pair in pairs:
        # Acquisition deliberately explores candidate procedures.  Repeated
        # varied scalar feedback, rather than one self-reinforcing greedy
        # action, is the information from which the fast state must adapt.
        proposal = propose_task(
            policy,
            pair.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        observed = _judge_frozen_answer(pair, proposal.answer, judge)
        scores.append(observed)
        row: dict[str, Any] = {
            "public_flag": pair.learner.public_flag,
            "score_before_write": observed,
            "candidate_index": proposal.candidate_index,
        }
        if mode != "no_write":
            presented = observed if mode == "ordinary" else 1.0 - observed
            transaction = apply_transactional_feedback(
                policy,
                pair.learner,
                proposal,
                presented,
                state,
            )
            # Online state advances only through the admitted write object.
            state = _detached_state(transaction.state)
            accepted += int(transaction.accepted)
            core_accepted += int(transaction.core_accepted)
            row.update(
                {
                    "accepted": transaction.accepted,
                    "core_accepted": transaction.core_accepted,
                    "write_slot": transaction.write_slot,
                    "delta_norm": transaction.delta_norm,
                }
            )
        rows.append(row)
        if probe_pairs is not None:
            probe_mean = float(
                _summary(_score_no_feedback(policy, state, probe_pairs, judge))["mean"]
            )
            probe_trace.append(probe_mean)
            if (
                recovery_target is not None
                and recovery_step is None
                and probe_mean >= recovery_target
            ):
                recovery_step = len(rows)
    attempted = 0 if mode == "no_write" else len(pairs)
    return state, {
        **_summary(scores),
        "mode": mode,
        "transactions": attempted,
        "accepted": accepted,
        "rejected": attempted - accepted,
        "core_accepted": core_accepted,
        "rejection_rate": 0.0 if not attempted else (attempted - accepted) / attempted,
        "probe_trace": probe_trace,
        "recovery_target": recovery_target,
        "recovery_step": recovery_step,
        "rows": rows,
    }


def _return_split(count: int) -> int:
    if count < 2:
        raise ValueError("returned skills require at least two encounters")
    if count < 8:
        return count - 1
    # At standard scale, the return exposure is no more than 25% of initial
    # acquisition exposure (12 return versus 52 initial for count 64).
    return count - max(2, count // 5)


def _slot_collision_diagnostics(
    rows_by_mechanism: Mapping[str, Sequence[Mapping[str, Any]]],
    slots: int,
) -> dict[str, Any]:
    writers: dict[int, set[str]] = {slot: set() for slot in range(slots)}
    attempts = 0
    for label, rows in rows_by_mechanism.items():
        for row in rows:
            if "write_slot" not in row:
                continue
            attempts += 1
            writers[int(row["write_slot"])].add(label)
    occupied = {slot: labels for slot, labels in writers.items() if labels}
    collisions = {
        str(slot): sorted(labels)
        for slot, labels in occupied.items()
        if len(labels) > 1
    }
    collision_assignments = sum(len(labels) - 1 for labels in occupied.values())
    distinct_assignments = sum(len(labels) for labels in occupied.values())
    return {
        "write_attempts": attempts,
        "distinct_mechanism_slot_assignments": distinct_assignments,
        "distinct_routed_slots": len(occupied),
        "cross_mechanism_collision_slots": len(collisions),
        "cross_mechanism_collision_assignments": collision_assignments,
        "cross_mechanism_collision_rate": (
            0.0
            if not distinct_assignments
            else collision_assignments / distinct_assignments
        ),
        "collision_map": collisions,
    }


def _score_no_feedback(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    pairs: Sequence[Any],
    judge: Callable[..., float],
    *,
    include_compiler: bool = True,
    include_phase4_bridge: bool = True,
    include_reverse_bridge: bool = True,
    include_descendants: bool = True,
    include_frozen_transition: bool = True,
    include_fast_adapter: bool = True,
    include_reversible_transition: bool = True,
) -> list[float]:
    before = procedural_skill_state_digest(state)
    values = [
        _judge_frozen_answer(
            pair,
            _propose_with_compiler_mode(
                policy,
                pair.learner,
                state,
                include_compiler=include_compiler,
                include_phase4_bridge=include_phase4_bridge,
                include_reverse_bridge=include_reverse_bridge,
                include_descendants=include_descendants,
                include_frozen_transition=include_frozen_transition,
                include_fast_adapter=include_fast_adapter,
                include_reversible_transition=include_reversible_transition,
            ).answer,
            judge,
        )
        for pair in pairs
    ]
    if procedural_skill_state_digest(state) != before:
        raise RuntimeError("no-feedback query changed competence state")
    return values


def _summarize_binary_branch_choices(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize an exact complementary-selector factorial.

    Rows are evaluator observations, not learning examples.  Ties and
    unavailable child pairs count as incorrect so a fixed-side shortcut or a
    missing procedure cannot receive partial credit.
    """

    if not rows:
        raise ValueError("binary branch summary requires at least one row")
    expected_cells = {
        "IF_FLAG:0",
        "IF_FLAG:1",
        "IF_NOT_FLAG:0",
        "IF_NOT_FLAG:1",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        operator = row.get("operator")
        public_flag = row.get("public_flag")
        expected_branch = row.get("expected_branch")
        weights = row.get("branch_weights")
        executed_weights = row.get("executed_branch_weights")
        execution_tied = row.get("execution_tied")
        eligible = row.get("eligible")
        if operator not in {"IF_FLAG", "IF_NOT_FLAG"}:
            raise ValueError("branch row has an unsupported operator")
        if type(public_flag) is not bool:
            raise TypeError("branch row public_flag must be bool")
        if expected_branch not in {0, 1}:
            raise ValueError("expected_branch must be zero or one")
        if type(eligible) is not bool:
            raise TypeError("branch row eligible must be bool")
        if type(execution_tied) is not bool:
            raise TypeError("branch row execution_tied must be bool")
        if (
            not isinstance(weights, Sequence)
            or isinstance(weights, (str, bytes))
            or len(weights) != 2
        ):
            raise ValueError("branch weights must contain exactly two values")
        if (
            not isinstance(executed_weights, Sequence)
            or isinstance(executed_weights, (str, bytes))
            or len(executed_weights) != 2
        ):
            raise ValueError(
                "executed branch weights must contain exactly two values"
            )
        numeric_weights = tuple(float(value) for value in weights)
        numeric_executed = tuple(float(value) for value in executed_weights)
        if (
            any(not math.isfinite(value) or value < 0.0 for value in numeric_weights)
            or not math.isclose(sum(numeric_weights), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("branch weights must be a finite probability pair")
        if (
            any(not math.isfinite(value) or value < 0.0 for value in numeric_executed)
            or not math.isclose(sum(numeric_executed), 1.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "executed branch weights must be a finite probability pair"
            )
        if execution_tied:
            if any(
                not math.isclose(left, right, abs_tol=1e-7)
                for left, right in zip(
                    numeric_executed,
                    numeric_weights,
                    strict=True,
                )
            ):
                raise ValueError("a tied execution must preserve the soft mixture")
        elif not (
            math.isclose(max(numeric_executed), 1.0, abs_tol=1e-7)
            and math.isclose(min(numeric_executed), 0.0, abs_tol=1e-7)
        ):
            raise ValueError("a non-tied execution must be one-hot")
        cell = f"{operator}:{int(public_flag)}"
        if cell not in expected_cells:
            raise RuntimeError("branch row produced an invalid factorial cell")
        grouped.setdefault(cell, []).append(
            {
                **row,
                "branch_weights": numeric_weights,
                "executed_branch_weights": numeric_executed,
            }
        )
    if set(grouped) != expected_cells:
        raise ValueError("binary branch summary does not cover the exact 2x2 grid")
    counts = {cell: len(values) for cell, values in grouped.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("binary branch grid is not balanced")

    def summarize_cell(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        eligible_count = sum(bool(row["eligible"]) for row in values)
        ties = 0
        correct = 0
        target_weights: list[float] = []
        margins: list[float] = []
        left_weights: list[float] = []
        right_weights: list[float] = []
        for row in values:
            weights = tuple(float(value) for value in row["branch_weights"])
            executed_weights = tuple(
                float(value) for value in row["executed_branch_weights"]
            )
            expected_branch = int(row["expected_branch"])
            target = weights[expected_branch]
            other = weights[1 - expected_branch]
            tied = bool(row["execution_tied"])
            ties += int(tied)
            correct += int(
                bool(row["eligible"])
                and not tied
                and executed_weights[expected_branch]
                > executed_weights[1 - expected_branch]
            )
            target_weights.append(target)
            margins.append(target - other)
            left_weights.append(weights[0])
            right_weights.append(weights[1])
        count = len(values)
        return {
            "count": count,
            "eligible_count": eligible_count,
            "mean_branch_weights": [
                sum(left_weights) / count,
                sum(right_weights) / count,
            ],
            "mean_target_weight": sum(target_weights) / count,
            "mean_signed_target_margin": sum(margins) / count,
            "hard_correct": correct,
            "hard_accuracy": correct / count,
            "eligible_hard_accuracy": (
                0.0 if not eligible_count else correct / eligible_count
            ),
            "tie_count": ties,
            "tie_rate": ties / count,
        }

    cells = {
        cell: summarize_cell(grouped[cell])
        for cell in sorted(expected_cells)
    }
    total = len(rows)
    total_eligible = sum(int(cell["eligible_count"]) for cell in cells.values())
    total_correct = sum(int(cell["hard_correct"]) for cell in cells.values())
    total_ties = sum(int(cell["tie_count"]) for cell in cells.values())
    return {
        "cells": cells,
        "count": total,
        "eligible_count": total_eligible,
        "hard_correct": total_correct,
        "hard_accuracy": total_correct / total,
        "eligible_hard_accuracy": (
            0.0 if not total_eligible else total_correct / total_eligible
        ),
        "tie_count": total_ties,
        "tie_rate": total_ties / total,
        "all_cells_balanced": True,
        "all_cases_eligible": total_eligible == total,
    }


def _binary_branch_choice_probe(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    cases: Sequence[Any],
) -> dict[str, Any]:
    """Read the branch actually executed on evaluator-owned matched cases."""

    state_before = procedural_skill_state_digest(state)
    slow_before = reasoning_state_digest(policy)
    rows: list[dict[str, Any]] = []
    for case in cases:
        pair = case.task
        expected_branch = int(case.expected_branch)
        proposal = propose_task(policy, pair.learner, state, greedy=True)
        root = proposal.scores.root
        weights = root.branch_weights
        executed_weights = root.executed_branch_weights
        if weights.shape != (1, 2) or not bool(torch.isfinite(weights).all().item()):
            raise RuntimeError("binary probe received invalid cached branch weights")
        if executed_weights.shape != (1, 2) or not bool(
            torch.isfinite(executed_weights).all().item()
        ):
            raise RuntimeError("binary probe received invalid executed branch weights")
        if bool((weights < 0.0).any().item()) or not torch.allclose(
            weights.sum(dim=-1),
            torch.ones(1, device=weights.device, dtype=weights.dtype),
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError("binary probe branch weights are not probabilities")
        rows.append(
            {
                "operator": pair.hidden.program.operator,
                "public_flag": pair.learner.public_flag,
                "expected_branch": expected_branch,
                "eligible": bool(proposal.scores.root_available.item())
                and bool(root.feedback_available.item()),
                "branch_weights": [
                    float(weights[0, 0].item()),
                    float(weights[0, 1].item()),
                ],
                "executed_branch_weights": [
                    float(executed_weights[0, 0].item()),
                    float(executed_weights[0, 1].item()),
                ],
                "execution_tied": bool(root.execution_tied.item()),
            }
        )
    if procedural_skill_state_digest(state) != state_before:
        raise RuntimeError("binary branch probe changed competence state")
    if reasoning_state_digest(policy) != slow_before:
        raise RuntimeError("binary branch probe changed frozen slow state")
    return {
        **_summarize_binary_branch_choices(rows),
        "feedback_writes": 0,
        "state_digest_before": state_before,
        "state_digest_after": procedural_skill_state_digest(state),
        "slow_fingerprint_before": slow_before,
        "slow_fingerprint_after": reasoning_state_digest(policy),
    }


def _candidate_indices_no_feedback(
    policy: SkillMemoryPolicy,
    state: ProceduralSkillState,
    pairs: Sequence[Any],
    *,
    include_compiler: bool = True,
    include_phase4_bridge: bool = True,
    include_reverse_bridge: bool = True,
    include_descendants: bool = True,
    include_frozen_transition: bool = True,
    include_fast_adapter: bool = True,
    include_reversible_transition: bool = True,
) -> tuple[list[int], list[float]]:
    """Return greedy decisions and root availability without writing state."""

    before = procedural_skill_state_digest(state)
    proposals = [
        _propose_with_compiler_mode(
            policy,
            pair.learner,
            state,
            include_compiler=include_compiler,
            include_phase4_bridge=include_phase4_bridge,
            include_reverse_bridge=include_reverse_bridge,
            include_descendants=include_descendants,
            include_frozen_transition=include_frozen_transition,
            include_fast_adapter=include_fast_adapter,
            include_reversible_transition=include_reversible_transition,
        )
        for pair in pairs
    ]
    if procedural_skill_state_digest(state) != before:
        raise RuntimeError("decision-only query changed competence state")
    return (
        [proposal.candidate_index for proposal in proposals],
        [float(proposal.scores.root_available.item()) for proposal in proposals],
    )


def _propose_with_compiler_mode(
    policy: SkillMemoryPolicy,
    public_task: Any,
    state: ProceduralSkillState,
    *,
    include_compiler: bool,
    include_phase4_bridge: bool = True,
    include_reverse_bridge: bool = True,
    include_descendants: bool = True,
    include_frozen_transition: bool = True,
    include_fast_adapter: bool = True,
    include_reversible_transition: bool = True,
) -> TaskProposal:
    """Greedy proposal helper used only for a declared compiler ablation."""

    scores = policy.score_task(
        public_task,
        state,
        include_compiler=include_compiler,
        include_phase4_bridge=include_phase4_bridge,
        include_reverse_bridge=include_reverse_bridge,
        include_descendants=include_descendants,
        include_frozen_transition=include_frozen_transition,
        include_fast_adapter=include_fast_adapter,
        include_reversible_transition=include_reversible_transition,
    )
    candidate_index = int(scores.logits.argmax(dim=-1).item())
    permutation = _PERMUTATIONS[candidate_index]
    behavior_probabilities = F.one_hot(
        torch.tensor(candidate_index, device=scores.logits.device),
        num_classes=len(_PERMUTATIONS),
    ).to(dtype=scores.logits.dtype)
    return TaskProposal(
        tuple(public_task.items[index].symbol for index in permutation),
        candidate_index,
        scores,
        behavior_probabilities,
        procedural_skill_state_digest(state),
        _public_task_digest(public_task),
    )


def _evaluate_leaf_core(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    """Evaluate isolated acquisition of all four atomic procedures."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("leaf-core evaluation requires frozen slow weights")
    slow_before = reasoning_state_digest(policy)
    identity_before = _parameter_identity_fingerprint(policy)
    partition, judge = _load_training_partition(seed + 8_000_003, 16)
    groups = tuple(
        group
        for group in _group_evaluator_pairs(partition.tasks)
        if group[0].hidden.program.depth == 0
    )
    if len(groups) != 4 or any(len(group) != 16 for group in groups):
        raise RuntimeError("leaf-core evaluation requires four balanced leaf roots")
    reports: dict[str, dict[str, Any]] = {}
    gains: list[float] = []
    causal_advantages: list[float] = []
    state_drops: list[float] = []
    for root_index, group in enumerate(groups):
        label = chr(ord("A") + root_index)
        supports = group[:8]
        probes = group[8:]
        reset_state = policy.initial_state(1)
        baseline_values = _score_no_feedback(
            policy,
            reset_state,
            probes,
            judge,
            include_compiler=False,
        )
        ordinary = policy.initial_state(1)
        inverted = policy.initial_state(1)
        curve: dict[str, float] = {"0": float(_summary(baseline_values)["mean"])}
        attempted_indices: list[int] = []
        for support_index, support in enumerate(supports):
            candidate_index = (17 + 37 * support_index + 13 * root_index) % len(
                _PERMUTATIONS
            )
            attempted_indices.append(candidate_index)
            proposal = _proposal_for_candidate(
                policy,
                support.learner,
                ordinary,
                candidate_index,
                include_compiler=False,
            )
            reward = _judge_frozen_answer(support, proposal.answer, judge)
            ordinary = _detached_state(
                propose_differentiable_feedback(
                    policy,
                    proposal,
                    reward,
                    ordinary,
                ).candidate_state
            )
            inverted_proposal = _proposal_for_candidate(
                policy,
                support.learner,
                inverted,
                candidate_index,
                include_compiler=False,
            )
            inverted = _detached_state(
                propose_differentiable_feedback(
                    policy,
                    inverted_proposal,
                    1.0 - reward,
                    inverted,
                ).candidate_state
            )
            count = support_index + 1
            if count in {1, 2, 4, 8}:
                curve[str(count)] = float(
                    _summary(
                        _score_no_feedback(
                            policy,
                            ordinary,
                            probes,
                            judge,
                            include_compiler=False,
                        )
                    )["mean"]
                )
        ordinary_summary = _summary(
            _score_no_feedback(
                policy, ordinary, probes, judge, include_compiler=False
            )
        )
        inverted_summary = _summary(
            _score_no_feedback(
                policy, inverted, probes, judge, include_compiler=False
            )
        )
        baseline_summary = _summary(baseline_values)
        gain = float(ordinary_summary["mean"]) - float(baseline_summary["mean"])
        causal_advantage = float(ordinary_summary["mean"]) - float(
            inverted_summary["mean"]
        )
        zero_summary = _summary(
            _score_no_feedback(
                policy,
                zero_procedural_skill_content(ordinary),
                probes,
                judge,
                include_compiler=False,
            )
        )
        state_drop = float(ordinary_summary["mean"]) - float(zero_summary["mean"])
        gains.append(gain)
        causal_advantages.append(causal_advantage)
        state_drops.append(state_drop)
        reports[label] = {
            "baseline": baseline_summary,
            "ordinary": ordinary_summary,
            "inverted": inverted_summary,
            "zero_content": zero_summary,
            "gain": gain,
            "ordinary_minus_inverted": causal_advantage,
            "ordinary_minus_zero": state_drop,
            "learning_curve": curve,
            "attempted_candidate_indices": attempted_indices,
            "ordinary_write_count": int(ordinary.write_counts.sum().item()),
            "inverted_write_count": int(inverted.write_counts.sum().item()),
            "ordinary_inverted_state_distance": float(
                (ordinary.slot_latents - inverted.slot_latents).norm().item()
            ),
        }
    mean_gain = sum(gains) / len(gains)
    mean_causal = sum(causal_advantages) / len(causal_advantages)
    mean_state_drop = sum(state_drops) / len(state_drops)
    criteria = {
        "three_of_four_leaf_gains_positive": sum(value > 0.0 for value in gains) >= 3,
        "mean_leaf_gain_at_least_0_05": mean_gain >= 0.05,
        "mean_correct_over_inverted_at_least_0_02": mean_causal >= 0.02,
        "mean_zero_content_drop_at_least_0_05": mean_state_drop >= 0.05,
    }
    if reasoning_state_digest(policy) != slow_before or (
        _parameter_identity_fingerprint(policy) != identity_before
    ):
        raise RuntimeError("leaf-core evaluation changed frozen slow state")
    return {
        "stage": "leaf_core",
        "roots": reports,
        "mean_gain": mean_gain,
        "mean_correct_over_inverted": mean_causal,
        "mean_zero_content_drop": mean_state_drop,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "final_partition_loaded_after_freeze": False,
        "online_replay_reads": 0,
        "history_retrievals": 0,
        "slow_fingerprint_before": slow_before,
        "slow_fingerprint_after": reasoning_state_digest(policy),
    }


def _evaluate(
    policy: SkillMemoryPolicy,
    profile: RunProfile,
    seed: int,
) -> dict[str, Any]:
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("final curriculum cannot load before slow weights freeze")
    slow_before = reasoning_state_digest(policy)
    parameter_identity_before = _parameter_identity_fingerprint(policy)
    final_seed = seed + 9_000_001
    curriculum, judge = _load_final_curriculum(
        final_seed,
        profile.encounters_per_primitive,
        profile.cases_per_component_probe,
        profile.cases_per_composition,
    )
    binary_branch_grid = _load_final_binary_branch_grid(
        final_seed,
        profile.cases_per_composition,
    )
    support_groups = _group_evaluator_pairs_by_root(curriculum.component_supports)
    probe_groups = _group_evaluator_pairs_by_root(curriculum.component_probes)
    if (
        len(support_groups) != _EXPECTED_PRIMITIVE_ROOTS
        or len(probe_groups) != _EXPECTED_PRIMITIVE_ROOTS
    ):
        raise RuntimeError("component curriculum must expose every primitive root")
    support_by_symbol = {
        group[0].learner.request.symbol: group for group in support_groups
    }
    probes_by_symbol = {
        group[0].learner.request.symbol: group for group in probe_groups
    }
    if set(support_by_symbol) != set(probes_by_symbol):
        raise RuntimeError("support and probe roots do not share one public mapping")
    ordered_symbols = tuple(
        sorted(
            support_by_symbol,
            key=lambda symbol: (
                len(support_by_symbol[symbol][0].learner.request.children),
                symbol,
            ),
        )
    )
    labels = tuple(chr(ord("A") + index) for index in range(len(ordered_symbols)))
    grouped = dict(
        zip(labels, (support_by_symbol[symbol] for symbol in ordered_symbols), strict=True)
    )
    probes = dict(
        zip(labels, (probes_by_symbol[symbol] for symbol in ordered_symbols), strict=True)
    )
    if any(len(grouped[label]) != profile.encounters_per_primitive for label in labels):
        raise RuntimeError("component support count differs from the run profile")
    if any(len(probes[label]) != profile.cases_per_component_probe for label in labels):
        raise RuntimeError("component probe count differs from the run profile")
    split_a = _return_split(len(grouped["A"]))
    split_b = _return_split(len(grouped["B"]))
    stages: list[tuple[str, str, Sequence[Any]]] = [
        ("A", "acquire", grouped["A"][:split_a]),
        ("B", "acquire", grouped["B"][:split_b]),
        ("C", "acquire", grouped["C"]),
        ("A", "return", grouped["A"][split_a:]),
        ("D", "acquire", grouped["D"]),
        ("B", "return", grouped["B"][split_b:]),
    ]
    stages.extend((label, "supplemental", grouped[label]) for label in labels[4:])

    state = policy.initial_state(1)
    initial_capacity = state.numel()
    baseline_probes = {
        label: _summary(_score_no_feedback(policy, state, probes[label], judge))
        for label in labels
    }
    route_order = tuple(range(1, profile.slots)) + (0,)
    reports: list[dict[str, Any]] = []
    ordinary_rows: dict[str, list[Mapping[str, Any]]] = {label: [] for label in labels}
    scores_by_mechanism: dict[str, list[float]] = {label: [] for label in labels}
    post_acquisition: dict[str, dict[str, float | int]] = {}
    return_metrics: dict[str, dict[str, Any]] = {}
    for label, phase, pairs in stages:
        incoming = _detached_state(state)
        probe_before = _summary(
            _score_no_feedback(policy, incoming, probes[label], judge)
        )
        recovery_target: float | None = None
        if phase == "return":
            recovery_target = max(
                0.0,
                float(post_acquisition[label]["mean"]) - 0.05,
            )
        state, ordinary = _run_support_segment(
            policy,
            incoming,
            pairs,
            judge,
            mode="ordinary",
            probe_pairs=probes[label] if phase == "return" else None,
            recovery_target=recovery_target,
        )
        probe_after = _summary(
            _score_no_feedback(policy, state, probes[label], judge)
        )
        ordinary_rows[label].extend(ordinary["rows"])
        scores_by_mechanism[label].extend(
            float(row["score_before_write"]) for row in ordinary["rows"]
        )
        if phase in {"acquire", "supplemental"} and label not in post_acquisition:
            post_acquisition[label] = probe_after
        if phase == "return":
            first_exposure = split_a if label == "A" else split_b
            budget = math.floor(first_exposure * 0.25)
            return_metrics[label] = {
                "probe_before": probe_before,
                "probe_after": probe_after,
                "target": recovery_target,
                "steps_used": ordinary["recovery_step"],
                "step_budget": budget,
                "return_presentations_available": len(pairs),
                "recovered_within_budget": ordinary["recovery_step"] is not None
                and int(ordinary["recovery_step"]) <= budget,
            }
        reports.append(
            {
                "mechanism": label,
                "phase": phase,
                "ordinary": ordinary,
                "probe_before": probe_before,
                "probe_after": probe_after,
            }
        )

    state_before_queries = procedural_skill_state_digest(state)
    reset_state = policy.initial_state(1)
    zero_state = zero_procedural_skill_content(state)
    permuted_state = permute_procedural_skill_slots(state, route_order)
    final_probes = {
        label: _summary(_score_no_feedback(policy, state, probes[label], judge))
        for label in labels
    }
    reset_probes = {
        label: _summary(_score_no_feedback(policy, reset_state, probes[label], judge))
        for label in labels
    }
    zero_probes = {
        label: _summary(_score_no_feedback(policy, zero_state, probes[label], judge))
        for label in labels
    }
    permuted_probes = {
        label: _summary(_score_no_feedback(policy, permuted_state, probes[label], judge))
        for label in labels
    }
    compiler_removed_probes = {
        label: _summary(
            _score_no_feedback(
                policy,
                state,
                probes[label],
                judge,
                include_compiler=False,
            )
        )
        for label in labels
    }
    composition_scores = _score_no_feedback(
        policy, state, curriculum.composition_queries, judge
    )
    reset_scores = _score_no_feedback(
        policy, reset_state, curriculum.composition_queries, judge
    )
    zero_scores = _score_no_feedback(
        policy, zero_state, curriculum.composition_queries, judge
    )
    permuted_scores = _score_no_feedback(
        policy, permuted_state, curriculum.composition_queries, judge
    )
    composition_removed_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_compiler=False,
    )
    phase4_removed_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_compiler=True,
        include_phase4_bridge=False,
    )
    fast_adapter_removed_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_fast_adapter=False,
    )
    fast_adapter_only_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_frozen_transition=False,
        include_fast_adapter=True,
    )
    reversible_transition_removed_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_reversible_transition=False,
    )
    reverse_removed_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_compiler=True,
        include_phase4_bridge=True,
        include_reverse_bridge=False,
    )
    root_only_scores = _score_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        judge,
        include_compiler=True,
        include_descendants=False,
    )
    composition_candidates, composition_availability = _candidate_indices_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
    )
    root_only_candidates, root_only_availability = _candidate_indices_no_feedback(
        policy,
        state,
        curriculum.composition_queries,
        include_descendants=False,
    )
    binary_branch_choice = _binary_branch_choice_probe(
        policy,
        state,
        binary_branch_grid.cells,
    )
    descendant_decision_change_rate = sum(
        left != right
        for left, right in zip(
            composition_candidates,
            root_only_candidates,
            strict=True,
        )
    ) / len(composition_candidates)
    equal_complete_availability = all(
        full == 1.0 and root_only == 1.0
        for full, root_only in zip(
            composition_availability,
            root_only_availability,
            strict=True,
        )
    )
    if procedural_skill_state_digest(state) != state_before_queries:
        raise RuntimeError("composition stage wrote feedback")

    slow_after = reasoning_state_digest(policy)
    parameter_identity_after = _parameter_identity_fingerprint(policy)
    if slow_after != slow_before or parameter_identity_after != parameter_identity_before:
        raise RuntimeError("online evaluation changed slow weights or their identities")
    if state.numel() != initial_capacity:
        raise RuntimeError("online stream changed fixed competence-state capacity")
    mechanisms: dict[str, dict[str, Any]] = {}
    for label in labels:
        baseline = baseline_probes[label]
        acquired = post_acquisition[label]
        final = final_probes[label]
        mechanisms[label] = {
            "online_support": _summary(scores_by_mechanism[label]),
            "baseline_probe": baseline,
            "post_acquisition_probe": acquired,
            "final_probe": final,
            "fresh_probe_gain": float(acquired["mean"]) - float(baseline["mean"]),
            "final_retention_loss": float(acquired["mean"]) - float(final["mean"]),
            "reset_probe": reset_probes[label],
            "zero_content_probe": zero_probes[label],
            "route_permuted_probe": permuted_probes[label],
            "compiler_removed_probe": compiler_removed_probes[label],
        }
        if label in return_metrics:
            mechanisms[label]["return"] = return_metrics[label]

    main_labels = labels[:4]
    deficient = tuple(
        label
        for label in main_labels
        if float(mechanisms[label]["baseline_probe"]["mean"]) < 0.70
    )
    improved = tuple(
        label
        for label in deficient
        if float(mechanisms[label]["fresh_probe_gain"]) >= 0.15
    )
    component_final_mean = sum(
        float(final_probes[label]["mean"]) for label in main_labels
    ) / len(main_labels)
    component_control_means = {
        "reset": sum(float(reset_probes[label]["mean"]) for label in main_labels)
        / len(main_labels),
        "zero_content": sum(
            float(zero_probes[label]["mean"]) for label in main_labels
        )
        / len(main_labels),
        "route_permuted": sum(
            float(permuted_probes[label]["mean"]) for label in main_labels
        )
        / len(main_labels),
    }
    component_control_drops = {
        name: component_final_mean - value
        for name, value in component_control_means.items()
    }
    composition_summary = _summary(composition_scores)
    composition_removed = _summary(composition_removed_scores)
    phase4_removed = _summary(phase4_removed_scores)
    fast_adapter_removed = _summary(fast_adapter_removed_scores)
    fast_adapter_only = _summary(fast_adapter_only_scores)
    reversible_transition_removed = _summary(
        reversible_transition_removed_scores
    )
    reverse_removed = _summary(reverse_removed_scores)
    root_only = _summary(root_only_scores)

    def composition_slices(
        keys: Sequence[int | str],
        eligible: Sequence[int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        pool = tuple(range(len(keys))) if eligible is None else tuple(eligible)
        if not pool:
            raise ValueError("composition slice requires at least one case")
        result: dict[str, dict[str, Any]] = {}
        for key in sorted({keys[index] for index in pool}):
            selected = tuple(index for index in pool if keys[index] == key)
            full = _summary([composition_scores[index] for index in selected])
            memory = _summary([phase4_removed_scores[index] for index in selected])
            no_fast = _summary(
                [fast_adapter_removed_scores[index] for index in selected]
            )
            fast_only = _summary(
                [fast_adapter_only_scores[index] for index in selected]
            )
            forward_only = _summary(
                [reverse_removed_scores[index] for index in selected]
            )
            no_children = _summary([root_only_scores[index] for index in selected])
            result[str(key)] = {
                **full,
                "phase4_removed_mean": memory["mean"],
                "fast_adapter_removed_mean": no_fast["mean"],
                "fast_adapter_only_mean": fast_only["mean"],
                "reverse_removed_mean": forward_only["mean"],
                "root_only_mean": no_children["mean"],
                "phase4_gain": float(full["mean"]) - float(memory["mean"]),
                "fast_adapter_gain": float(full["mean"])
                - float(no_fast["mean"]),
                "fast_only_gain_over_frozen_only": float(fast_only["mean"])
                - float(no_fast["mean"]),
                "descendant_gain": float(full["mean"])
                - float(no_children["mean"]),
            }
        return result

    composition_depths = tuple(
        int(pair.hidden.program.depth) for pair in curriculum.composition_queries
    )
    composition_root_arities = tuple(
        len(pair.learner.request.children)
        for pair in curriculum.composition_queries
    )
    composition_public_flags = tuple(
        int(pair.learner.public_flag) for pair in curriculum.composition_queries
    )
    composition_root_operators = tuple(
        pair.hidden.program.operator for pair in curriculum.composition_queries
    )
    composition_binary_cells = tuple(
        f"{operator}:{int(pair.learner.public_flag)}"
        for operator, pair in zip(
            composition_root_operators,
            curriculum.composition_queries,
            strict=True,
        )
    )
    binary_indices = tuple(
        index
        for index, arity in enumerate(composition_root_arities)
        if arity == 2
    )
    adapter_by_arity = composition_slices(composition_root_arities)
    unary_adapter = adapter_by_arity.get("1")
    binary_adapter = adapter_by_arity.get("2")
    adapter_harmonization_criteria = {
        "full_adds_at_least_0_02_over_adapter_off": float(
            composition_summary["mean"]
        )
        - float(fast_adapter_removed["mean"])
        >= 0.02,
        "fast_only_adds_at_least_0_02_over_frozen_only": float(
            fast_adapter_only["mean"]
        )
        - float(fast_adapter_removed["mean"])
        >= 0.02,
        "unary_root_gain_at_least_0_02": (
            unary_adapter is not None
            and float(unary_adapter["fast_adapter_gain"]) >= 0.02
        ),
        "binary_root_regression_at_most_0_02": (
            binary_adapter is not None
            and float(binary_adapter["fast_adapter_gain"]) >= -0.02
        ),
    }
    criteria = {
        "four_main_mechanisms_initially_deficient": len(deficient) == 4,
        "gain_at_least_0_15_on_three_of_four": len(improved) >= 3,
        "old_skill_retention_within_0_05": all(
            float(mechanisms[label]["final_retention_loss"]) <= 0.05
            for label in ("A", "B")
        ),
        "returned_skill_recovery_within_quarter_budget": all(
            bool(return_metrics[label]["recovered_within_budget"])
            for label in ("A", "B")
        ),
        "novel_composition_mean_at_least_0_70": float(composition_summary["mean"])
        >= 0.70,
        "reset_removes_at_least_0_15": component_control_drops["reset"] >= 0.15,
        "zero_content_removes_at_least_0_15": component_control_drops["zero_content"]
        >= 0.15,
        "route_permutation_removes_at_least_0_15": component_control_drops[
            "route_permuted"
        ]
        >= 0.15,
        "frozen_compiler_adds_at_least_0_05_on_compositions": float(
            composition_summary["mean"]
        )
        - float(phase4_removed["mean"])
        >= 0.05,
        "descendants_add_at_least_0_10_on_compositions": (
            equal_complete_availability
            and float(composition_summary["mean"])
            - float(root_only["mean"])
            >= 0.10
        ),
        "descendant_ablation_changes_at_least_quarter_of_decisions": (
            descendant_decision_change_rate >= 0.25
        ),
        "matched_binary_branch_accuracy_at_least_0_75": (
            binary_branch_choice["hard_accuracy"] >= 0.75
        ),
        "matched_binary_every_cell_at_least_half_correct": all(
            cell["hard_accuracy"] >= 0.50
            for cell in binary_branch_choice["cells"].values()
        ),
        "matched_binary_all_cases_eligible": bool(
            binary_branch_choice["all_cases_eligible"]
        ),
    }
    transactions = sum(
        int(stage["ordinary"]["transactions"]) for stage in reports
    )
    accepted = sum(int(stage["ordinary"]["accepted"]) for stage in reports)
    return {
        "phase_order": [stage["mechanism"] for stage in reports[:6]],
        "stage_reports": reports,
        "mechanisms": mechanisms,
        "composition": {
            **composition_summary,
            "feedback_writes": 0,
            "reset_mean": _summary(reset_scores)["mean"],
            "zero_content_mean": _summary(zero_scores)["mean"],
            "route_permuted_mean": _summary(permuted_scores)["mean"],
            "composition_removed_mean": composition_removed["mean"],
            "phase4_removed_mean": phase4_removed["mean"],
            "fast_adapter_removed_mean": fast_adapter_removed["mean"],
            "fast_adapter_only_mean": fast_adapter_only["mean"],
            "fast_adapter_gain": float(composition_summary["mean"])
            - float(fast_adapter_removed["mean"]),
            "fast_only_gain_over_frozen_only": float(fast_adapter_only["mean"])
            - float(fast_adapter_removed["mean"]),
            "legacy_adapter_controls_applicable": not bool(
                policy.reversible_transition_mode.item()
            ),
            "reversible_transition_removed_mean": (
                reversible_transition_removed["mean"]
            ),
            "reversible_transition_gain": float(composition_summary["mean"])
            - float(reversible_transition_removed["mean"]),
            "reverse_removed_mean": reverse_removed["mean"],
            "root_only_mean": root_only["mean"],
            "descendant_ablation_decision_change_rate": (
                descendant_decision_change_rate
            ),
            "full_root_availability": _summary(composition_availability),
            "root_only_availability": _summary(root_only_availability),
            "equal_complete_availability": equal_complete_availability,
            "by_depth": composition_slices(composition_depths),
            "by_root_arity": adapter_by_arity,
            "binary_by_public_flag": composition_slices(
                composition_public_flags,
                binary_indices,
            ),
            "binary_by_hidden_operator": composition_slices(
                composition_root_operators,
                binary_indices,
            ),
            "binary_by_operator_and_flag": composition_slices(
                composition_binary_cells,
                binary_indices,
            ),
            "binary_branch_choice": binary_branch_choice,
            "procedural_adapter_harmonization": {
                "comparison": "same acquired system under causal transition ablations",
                "criteria": adapter_harmonization_criteria,
                "passed": all(adapter_harmonization_criteria.values()),
            },
            "state_digest_before": state_before_queries,
            "state_digest_after": procedural_skill_state_digest(state),
        },
        "component_controls": {
            "ordinary_mean": component_final_mean,
            "control_means": component_control_means,
            "drops": component_control_drops,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "transactions": transactions,
        "accepted": accepted,
        "rejected": transactions - accepted,
        "rejection_rate": 0.0 if not transactions else (transactions - accepted) / transactions,
        "slot_collisions": _slot_collision_diagnostics(
            ordinary_rows, profile.slots
        ),
        "state_numel_initial": initial_capacity,
        "state_numel_final": state.numel(),
        "state_digest_final": procedural_skill_state_digest(state),
        "slow_fingerprint_before": slow_before,
        "slow_fingerprint_after": slow_after,
        "slow_parameter_identity_before": parameter_identity_before,
        "slow_parameter_identity_after": parameter_identity_after,
        "optimizer_reachable_online": False,
        "online_replay_reads": 0,
        "history_retrievals": 0,
        "ordinary_unique_presentations": len(curriculum.component_supports),
        "disjoint_component_probe_count": len(curriculum.component_probes),
        "composition_query_count": len(curriculum.composition_queries),
        "matched_binary_branch_probe_count": len(binary_branch_grid.cells),
        "counterfactual_feedback_presentations": 0,
        "final_partition_loaded_after_freeze": True,
    }


def _operator_intervention_rows(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        interventions = row.get("interventions")
        observation = (
            interventions.get(mode)
            if isinstance(interventions, Mapping)
            else None
        )
        if not isinstance(observation, Mapping):
            continue
        selected.append(
            {
                "hidden_operator": row["hidden_operator"],
                "correct": observation["correct"],
                "permuted": observation["permuted"],
                "zero": observation["zero"],
                "representation": row["representation"],
            }
        )
    return tuple(selected)


def _summarize_operator_interventions(
    seed_rows: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("frozen_only", "fast_only"):
        cohorts: dict[str, Any] = {}
        for cohort in _OPERATOR_AUDIT_COHORTS:
            all_rows = tuple(
                row
                for seed in sorted(seed_rows)
                for row in seed_rows[seed][cohort]
            )
            selected = _operator_intervention_rows(all_rows, mode)
            per_seed: dict[str, Any] = {}
            positive_both = 0
            for seed in sorted(seed_rows):
                original = seed_rows[seed][cohort]
                observed = _operator_intervention_rows(original, mode)
                if len(observed) != len(original) or not observed:
                    per_seed[str(seed)] = {
                        "count": len(observed),
                        "positive_against_both": False,
                    }
                    continue
                seed_summary = _summarize_operator_audit_rows(observed)
                seed_deltas = seed_summary["deltas"]
                positive = (
                    float(seed_deltas["correct_minus_permuted_correlation"])
                    > 0.0
                    and float(seed_deltas["correct_minus_zero_correlation"])
                    > 0.0
                )
                positive_both += int(positive)
                per_seed[str(seed)] = {
                    "count": len(observed),
                    "deltas": seed_deltas,
                    "positive_against_both": positive,
                }
            if not selected:
                cohorts[cohort] = {
                    "count": 0,
                    "expected_count": len(all_rows),
                    "all_cases_available": False,
                    "positive_against_both_seed_count": 0,
                    "per_seed": per_seed,
                    "passed": False,
                }
                continue
            aggregate = _summarize_operator_audit_rows(selected)
            deltas = aggregate["deltas"]
            all_available = len(selected) == len(all_rows)
            criteria = {
                "all_cases_available": all_available,
                "correct_minus_permuted_correlation_at_least_0_05": float(
                    deltas["correct_minus_permuted_correlation"]
                )
                >= 0.05,
                "correct_minus_zero_correlation_at_least_0_05": float(
                    deltas["correct_minus_zero_correlation"]
                )
                >= 0.05,
                "correct_minus_permuted_covariance_positive": float(
                    deltas["correct_minus_permuted_covariance"]
                )
                > 0.0,
                "correct_minus_zero_covariance_positive": float(
                    deltas["correct_minus_zero_covariance"]
                )
                > 0.0,
                "at_least_seven_of_eight_seeds_positive_against_both": (
                    positive_both >= 7
                ),
            }
            cohorts[cohort] = {
                **aggregate,
                "expected_count": len(all_rows),
                "all_cases_available": all_available,
                "positive_against_both_seed_count": positive_both,
                "per_seed": per_seed,
                "criteria": criteria,
                "passed": all(criteria.values()),
            }
        modes[mode] = cohorts
    return modes


def _summarize_operator_localization(
    seed_reports: Sequence[Mapping[str, Any]],
    seed_rows: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    seed_vectors: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    trained_fast_adapter: bool = False,
) -> dict[str, Any]:
    if len(seed_reports) != _OPERATOR_AUDIT_SEED_COUNT:
        raise ValueError("operator localization requires exactly eight seed reports")
    localization: dict[str, Any] = {}
    for cohort in _OPERATOR_AUDIT_COHORTS:
        pooled = tuple(
            row
            for seed in sorted(seed_rows)
            for row in seed_rows[seed][cohort]
        )
        if not pooled:
            raise RuntimeError(f"operator localization cohort is empty: {cohort}")
        aggregate = _summarize_operator_audit_rows(pooled)
        per_seed: dict[str, Any] = {}
        positive_both = 0
        for seed in sorted(seed_rows):
            rows = seed_rows[seed][cohort]
            if not rows:
                per_seed[str(seed)] = {"count": 0, "positive_against_both": False}
                continue
            summary = _summarize_operator_audit_rows(rows)
            deltas = summary["deltas"]
            positive = (
                float(deltas["correct_minus_permuted_correlation"]) > 0.0
                and float(deltas["correct_minus_zero_correlation"]) > 0.0
            )
            positive_both += int(positive)
            per_seed[str(seed)] = {
                "count": summary["count"],
                "deltas": deltas,
                "positive_against_both": positive,
            }
        deltas = aggregate["deltas"]
        no_degenerate = all(
            int(report["degenerate_cases"][cohort]) == 0
            for report in seed_reports
        )
        no_unavailable = all(
            int(report["unavailable_cases"][cohort]) == 0
            for report in seed_reports
        )
        criteria = {
            "correct_minus_permuted_correlation_at_least_0_05": float(
                deltas["correct_minus_permuted_correlation"]
            )
            >= 0.05,
            "correct_minus_zero_correlation_at_least_0_05": float(
                deltas["correct_minus_zero_correlation"]
            )
            >= 0.05,
            "correct_minus_permuted_covariance_positive": float(
                deltas["correct_minus_permuted_covariance"]
            )
            > 0.0,
            "correct_minus_zero_covariance_positive": float(
                deltas["correct_minus_zero_covariance"]
            )
            > 0.0,
            "at_least_seven_of_eight_seeds_positive_against_both": (
                positive_both >= 7
            ),
            "no_degenerate_cases": no_degenerate,
            "no_unavailable_cases": no_unavailable,
        }
        localization[cohort] = {
            **aggregate,
            "per_seed": per_seed,
            "positive_against_both_seed_count": positive_both,
            "criteria": criteria,
            "passed": all(criteria.values()),
        }
    gradients = {
        label: _bridge_gradient_summary(seed_vectors[label])
        for label in _OPERATOR_AUDIT_BRIDGES
    }
    localization_passed = all(
        report["passed"] for report in localization.values()
    )
    permitted_bridges = [
        label
        for label, report in gradients.items()
        if localization_passed and bool(report["passed"])
    ]
    operator_trial_authorized = (
        localization_passed and bool(gradients["operator"]["passed"])
    )
    interventions = _summarize_operator_interventions(seed_rows)
    causal_criteria: dict[str, Any] = {}
    for cohort in _OPERATOR_AUDIT_COHORTS:
        full_deltas = localization[cohort]["deltas"]
        frozen_report = interventions["frozen_only"][cohort]
        fast_report = interventions["fast_only"][cohort]
        frozen_deltas = frozen_report.get("deltas", {})
        causal_criteria[cohort] = {
            "full_localization_passed": bool(localization[cohort]["passed"]),
            "fast_only_localization_passed": bool(fast_report["passed"]),
            "adapter_adds_0_05_over_frozen_permuted_control": (
                "correct_minus_permuted_correlation" in frozen_deltas
                and float(full_deltas["correct_minus_permuted_correlation"])
                - float(frozen_deltas["correct_minus_permuted_correlation"])
                >= 0.05
            ),
            "adapter_adds_0_05_over_frozen_zero_control": (
                "correct_minus_zero_correlation" in frozen_deltas
                and float(full_deltas["correct_minus_zero_correlation"])
                - float(frozen_deltas["correct_minus_zero_correlation"])
                >= 0.05
            ),
        }
        causal_criteria[cohort]["passed"] = all(
            causal_criteria[cohort][name]
            for name in (
                "full_localization_passed",
                "fast_only_localization_passed",
                "adapter_adds_0_05_over_frozen_permuted_control",
                "adapter_adds_0_05_over_frozen_zero_control",
            )
        )
    fast_adapter_causal_passed = all(
        report["passed"] for report in causal_criteria.values()
    )
    passed = (
        fast_adapter_causal_passed
        if trained_fast_adapter
        else operator_trial_authorized
    )
    return {
        "localization": localization,
        "localization_passed": localization_passed,
        "interventions": interventions,
        "fast_adapter_causal_gate": {
            "evaluated": trained_fast_adapter,
            "by_cohort": causal_criteria,
            "passed": fast_adapter_causal_passed if trained_fast_adapter else False,
        },
        "bridge_gradient_coherence": gradients,
        "individually_coherent_bridges": permitted_bridges,
        "operator_bridge_trial_go": (
            operator_trial_authorized if not trained_fast_adapter else False
        ),
        "recommendation": (
            (
                "retain the learned fast procedural adapter"
                if fast_adapter_causal_passed
                else "reject the adapter trial; preserve retained v41"
            )
            if trained_fast_adapter
            else (
                "one bounded fresh-seed operator-bridge trial"
                if operator_trial_authorized
                else "no bridge training; revise the representation before another update"
            )
        ),
        "passed": passed,
    }


def _validate_retained_v41_neutral_migration(
    policy: SkillMemoryPolicy,
    initialization: Mapping[str, Any],
) -> None:
    """Require the exact retained policy plus only behavior-neutral new state."""

    if (
        initialization.get("sha256")
        != _PROCEDURAL_ADAPTER_SOURCE_CHECKPOINT_SHA256
        or initialization.get("source_runner")
        != "angler.phase5-skill-memory-stream.v13"
        or initialization.get("source_stage") != "relational_acquisition"
    ):
        raise RuntimeError(
            "procedural adapter work requires the exact retained v41 checkpoint"
        )
    expected_fresh = {
        name
        for name in policy.state_dict()
        if name.startswith(
            (
                "phase4_direction_mixer.",
                "procedural_fast_adapter.",
                "procedural_goal_projection.",
                "reversible_procedure_transition.",
                "reversible_transition_mode",
            )
        )
    }
    actual_fresh = set(initialization.get("fresh_parameter_keys", ()))
    if actual_fresh != expected_fresh:
        raise RuntimeError(
            "retained v41 migration contains state outside the neutral groups"
        )
    if bool(policy.phase4_direction_mixer[-1].weight.detach().count_nonzero()) or bool(
        policy.phase4_direction_mixer[-1].bias.detach().count_nonzero()
    ):
        raise RuntimeError("retained v41 direction mixer is not exactly neutral")
    if bool(policy.procedural_fast_adapter.forward_up.weight.detach().count_nonzero()) or bool(
        policy.procedural_fast_adapter.reverse_up.weight.detach().count_nonzero()
    ):
        raise RuntimeError("retained v41 fast adapter is not exactly neutral")
    if bool(
        policy.procedural_goal_projection.candidate_down.weight.detach().count_nonzero()
    ):
        raise RuntimeError("retained v41 goal projection is not exactly neutral")
    if bool(policy.reversible_transition_mode.item()):
        raise RuntimeError("retained v41 reversible transition is unexpectedly active")
    if bool(
        policy.reversible_procedure_transition.first_up.weight.detach().count_nonzero()
    ) or bool(
        policy.reversible_procedure_transition.second_up.weight.detach().count_nonzero()
    ):
        raise RuntimeError("retained v41 reversible transition is not exactly neutral")


def _validate_operator_audit_checkpoint_lineage(
    policy: SkillMemoryPolicy,
    initialization: Mapping[str, Any],
) -> str:
    """Validate retained or learned procedural state before a causal audit."""

    if initialization.get("sha256") == _PROCEDURAL_ADAPTER_SOURCE_CHECKPOINT_SHA256:
        _validate_retained_v41_neutral_migration(policy, initialization)
        return "retained_v41"
    source_stage = initialization.get("source_stage")
    source_runner = initialization.get("source_runner")
    compatible_learned_stage = (
        source_stage == "reverse_construction"
        and source_runner
        in {
            "angler.phase5-skill-memory-stream.v19",
            "angler.phase5-skill-memory-stream.v22",
            _REPORT_VERSION,
        }
    ) or (
        source_stage == "reverse_harmonization"
        and source_runner
        in {
            "angler.phase5-skill-memory-stream.v20",
            "angler.phase5-skill-memory-stream.v22",
            _REPORT_VERSION,
        }
    ) or (
        source_stage in {"procedural_adapter", "procedural_coadaptation"}
        and source_runner
        in {
            "angler.phase5-skill-memory-stream.v21",
            "angler.phase5-skill-memory-stream.v22",
            _REPORT_VERSION,
        }
    )
    if (
        not compatible_learned_stage
        or initialization.get("fresh_parameter_keys")
    ):
        raise RuntimeError(
            "operator audit requires retained v41 or a complete current learned checkpoint"
        )
    source_initialization = initialization.get("source_initialization")
    source_training = initialization.get("source_training")
    if source_stage == "procedural_coadaptation":
        group_selectors: tuple[tuple[str, Callable[[str], bool]], ...] = (
            (
                "leaf_code_acquisition",
                lambda name: name == "memory.feedback_direction_encoder.3.weight",
            ),
            (
                "composition_code_acquisition",
                lambda name: (
                    name
                    == "composition_memory.feedback_direction_encoder.3.weight"
                ),
            ),
            (
                "fast_adapter",
                lambda name: name.startswith("procedural_fast_adapter."),
            ),
            (
                "goal_projection",
                lambda name: name.startswith("procedural_goal_projection."),
            ),
            (
                "direction_mixer",
                lambda name: name.startswith("phase4_direction_mixer."),
            ),
            (
                "reliability_gate",
                lambda name: name.startswith("phase4_reliability_gate."),
            ),
        )
        current_groups = {
            label: _named_state_fingerprint(
                policy,
                include=selector,
                domain=(
                    b"project-angler.phase5-deployed-preference-group."
                    + label.encode("ascii")
                ),
            )
            for label, selector in group_selectors
        }
        current_joint = _named_state_fingerprint(
            policy,
            include=_is_procedural_coadaptation_state,
            domain=b"project-angler.phase5-procedural-coadaptation.v1",
        )
        current_outside = _named_state_fingerprint(
            policy,
            include=lambda name: not _is_procedural_coadaptation_state(name),
            domain=b"project-angler.phase5-outside-procedural-coadaptation.v1",
        )
        expected_trainable = [
            name
            for name, _ in policy.named_parameters()
            if _is_procedural_coadaptation_state(name)
        ]
        source_v47 = (
            source_initialization.get("source_initialization")
            if isinstance(source_initialization, Mapping)
            else None
        )
        source_v48_training = (
            source_initialization.get("source_training")
            if isinstance(source_initialization, Mapping)
            else None
        )
        expected_group_reach = {label: True for label, _ in group_selectors}
        if (
            not isinstance(source_initialization, Mapping)
            or source_initialization.get("sha256")
            != _PROCEDURAL_COADAPTATION_SOURCE_CHECKPOINT_SHA256
            or source_initialization.get("source_runner")
            != "angler.phase5-skill-memory-stream.v20"
            or source_initialization.get("source_stage")
            != "reverse_harmonization"
            or source_initialization.get("fresh_parameter_keys")
            or not isinstance(source_v47, Mapping)
            or source_v47.get("sha256")
            != _REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256
            or source_v47.get("source_runner")
            != "angler.phase5-skill-memory-stream.v19"
            or source_v47.get("source_stage") != "reverse_construction"
            or source_v47.get("fresh_parameter_keys")
            or not isinstance(source_v48_training, Mapping)
            or source_v48_training.get("training_stage")
            != "reverse_harmonization"
            or source_v48_training.get("outer_steps")
            != _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_v48_training.get("support_graph_detached") is not True
            or source_v48_training.get(
                "auxiliary_ranking_objectives_used_for_training"
            )
            is not False
            or not isinstance(source_training, Mapping)
            or source_training.get("training_stage")
            != "procedural_coadaptation"
            or source_training.get("harmonizer_fingerprint_after")
            != current_joint
            or source_training.get("outside_harmonizer_fingerprint_before")
            != source_training.get("outside_harmonizer_fingerprint_after")
            or source_training.get("outside_harmonizer_fingerprint_after")
            != current_outside
            or source_training.get("trainable_group_fingerprints_after")
            != current_groups
            or source_training.get(
                "deployed_preference_gradient_reached_groups"
            )
            != expected_group_reach
            or source_training.get("trainable_parameter_names")
            != expected_trainable
            or source_training.get("trainable_parameter_count") != 11_749
            or source_training.get("outer_steps")
            != _PROCEDURAL_COADAPTATION_OUTER_STEPS
            or source_training.get("fresh_opaque_mappings")
            != _PROCEDURAL_COADAPTATION_OUTER_STEPS
            or source_training.get("optimizer_steps")
            != _PROCEDURAL_COADAPTATION_OUTER_STEPS
            or source_training.get("support_presentations_per_mapping") != 40
            or source_training.get("query_presentations_per_mapping") != 32
            or source_training.get("attempted_outputs_per_query")
            != _REVERSE_CONSTRUCTION_ATTEMPTS
            or source_training.get("total_support_presentations")
            != 40 * _PROCEDURAL_COADAPTATION_OUTER_STEPS
            or source_training.get("total_query_presentations")
            != 32 * _PROCEDURAL_COADAPTATION_OUTER_STEPS
            or source_training.get("total_scored_query_attempts")
            != (
                _REVERSE_CONSTRUCTION_ATTEMPTS
                * 32
                * _PROCEDURAL_COADAPTATION_OUTER_STEPS
            )
            or source_training.get("cohort_case_counts")
            != {
                "unary_depth2": 16 * _PROCEDURAL_COADAPTATION_OUTER_STEPS,
                "unary_depth3": 4 * _PROCEDURAL_COADAPTATION_OUTER_STEPS,
                "unary_direct_binary_child": (
                    4 * _PROCEDURAL_COADAPTATION_OUTER_STEPS
                ),
                "binary_root": 8 * _PROCEDURAL_COADAPTATION_OUTER_STEPS,
            }
            or not isinstance(
                source_training.get("total_observed_preference_edges"), int
            )
            or source_training.get("total_observed_preference_edges", 0) <= 0
            or source_training.get("support_graph_detached") is not False
            or source_training.get("training_objective") != "on_policy_reward"
            or source_training.get("current_deployed_greedy_attempted_per_query")
            is not True
            or source_training.get("complete_action_softmax_used_for_training")
            is not True
            or source_training.get("target_permutations_used_for_training")
            is not False
            or source_training.get("candidate_utility_vectors_used_for_training")
            is not False
            or source_training.get("hidden_operator_labels_used_for_training")
            is not False
            or source_training.get("auxiliary_ranking_objectives_used_for_training")
            is not False
        ):
            raise RuntimeError(
                "procedural-coadaptation checkpoint does not prove exact v48 lineage"
            )
        return "procedural_coadaptation"
    if source_stage == "reverse_harmonization":
        current_harmonizer = _named_state_fingerprint(
            policy,
            include=_is_reverse_harmonization_state,
            domain=b"project-angler.phase5-reverse-harmonization.v1",
        )
        current_direction_mixer = _named_state_fingerprint(
            policy,
            include=lambda name: name.startswith("phase4_direction_mixer."),
            domain=b"project-angler.phase5-reverse-direction-mixer.v1",
        )
        current_reliability_gate = _named_state_fingerprint(
            policy,
            include=lambda name: name.startswith("phase4_reliability_gate."),
            domain=b"project-angler.phase5-reverse-reliability-gate.v1",
        )
        current_outside = _named_state_fingerprint(
            policy,
            include=lambda name: not _is_reverse_harmonization_state(name),
            domain=b"project-angler.phase5-outside-reverse-harmonization.v1",
        )
        expected_trainable = [
            name
            for name, _ in policy.named_parameters()
            if _is_reverse_harmonization_state(name)
        ]
        if (
            not isinstance(source_initialization, Mapping)
            or source_initialization.get("sha256")
            != _REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256
            or source_initialization.get("source_runner")
            != "angler.phase5-skill-memory-stream.v19"
            or source_initialization.get("source_stage")
            != "reverse_construction"
            or source_initialization.get("fresh_parameter_keys")
            or not isinstance(source_training, Mapping)
            or source_training.get("training_stage")
            != "reverse_harmonization"
            or source_training.get("outside_harmonizer_fingerprint_before")
            != source_training.get("outside_harmonizer_fingerprint_after")
            or source_training.get("harmonizer_fingerprint_before")
            == source_training.get("harmonizer_fingerprint_after")
            or source_training.get("harmonizer_fingerprint_after")
            != current_harmonizer
            or source_training.get("direction_mixer_fingerprint_before")
            == source_training.get("direction_mixer_fingerprint_after")
            or source_training.get("direction_mixer_fingerprint_after")
            != current_direction_mixer
            or source_training.get("reliability_gate_fingerprint_before")
            == source_training.get("reliability_gate_fingerprint_after")
            or source_training.get("reliability_gate_fingerprint_after")
            != current_reliability_gate
            or source_training.get("outside_harmonizer_fingerprint_after")
            != current_outside
            or source_training.get("trainable_parameter_names")
            != expected_trainable
            or source_training.get("outer_steps")
            != _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_training.get("fresh_opaque_mappings")
            != _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_training.get("optimizer_steps")
            != _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_training.get("support_presentations_per_mapping") != 40
            or source_training.get("query_presentations_per_mapping") != 32
            or source_training.get("attempted_outputs_per_query")
            != _REVERSE_CONSTRUCTION_ATTEMPTS
            or source_training.get("total_support_presentations")
            != 40 * _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_training.get("total_query_presentations")
            != 32 * _REVERSE_HARMONIZATION_OUTER_STEPS
            or source_training.get("total_scored_query_attempts")
            != (
                _REVERSE_CONSTRUCTION_ATTEMPTS
                * 32
                * _REVERSE_HARMONIZATION_OUTER_STEPS
            )
            or source_training.get("cohort_case_counts")
            != {
                "unary_depth2": 16 * _REVERSE_HARMONIZATION_OUTER_STEPS,
                "unary_depth3": 4 * _REVERSE_HARMONIZATION_OUTER_STEPS,
                "unary_direct_binary_child": (
                    4 * _REVERSE_HARMONIZATION_OUTER_STEPS
                ),
                "binary_root": 8 * _REVERSE_HARMONIZATION_OUTER_STEPS,
            }
            or not isinstance(
                source_training.get("total_observed_preference_edges"), int
            )
            or source_training.get("total_observed_preference_edges", 0) <= 0
            or source_training.get("support_graph_detached") is not True
            or source_training.get("target_permutations_used_for_training")
            is not False
            or source_training.get("candidate_utility_vectors_used_for_training")
            is not False
            or source_training.get("hidden_operator_labels_used_for_training")
            is not False
            or source_training.get("auxiliary_ranking_objectives_used_for_training")
            is not False
            or source_training.get(
                "deployed_preference_gradient_reached_direction_mixer"
            )
            is not True
            or source_training.get(
                "deployed_preference_gradient_reached_reliability_gate"
            )
            is not True
        ):
            raise RuntimeError(
                "reverse-harmonization checkpoint does not prove exact v47 lineage"
            )
        return "reverse_harmonization"
    expected_fresh = {
        name
        for name in policy.state_dict()
        if name.startswith(
            (
                "phase4_direction_mixer.",
                "procedural_fast_adapter.",
                "procedural_goal_projection.",
                "reversible_procedure_transition.",
                "reversible_transition_mode",
            )
        )
    }
    valid_source = (
        isinstance(source_initialization, Mapping)
        and source_initialization.get("sha256")
        == _PROCEDURAL_ADAPTER_SOURCE_CHECKPOINT_SHA256
        and source_initialization.get("source_runner")
        == "angler.phase5-skill-memory-stream.v13"
        and source_initialization.get("source_stage") == "relational_acquisition"
        and set(source_initialization.get("fresh_parameter_keys", ()))
        == expected_fresh
        and isinstance(source_training, Mapping)
    )
    if source_stage == "procedural_adapter":
        current_adapter = _named_state_fingerprint(
            policy,
            include=_is_procedural_adapter_state,
            domain=b"project-angler.phase5-procedural-adapter.v1",
        )
        current_outside = _named_state_fingerprint(
            policy,
            include=lambda name: not _is_procedural_adapter_state(name),
            domain=b"project-angler.phase5-outside-procedural-adapter.v1",
        )
        current_reverse = _named_state_fingerprint(
            policy,
            include=lambda name: name.startswith(
                "procedural_fast_adapter.reverse_up."
            ),
            domain=b"project-angler.phase5-procedural-adapter-reverse.v1",
        )
        if (
            not valid_source
            or source_training.get("training_stage") != "procedural_adapter"
            or source_training.get("outside_adapter_fingerprint_before")
            != source_training.get("outside_adapter_fingerprint_after")
            or source_training.get("reverse_adapter_fingerprint_before")
            != source_training.get("reverse_adapter_fingerprint_after")
            or source_training.get("adapter_fingerprint_before")
            == source_training.get("adapter_fingerprint_after")
            or source_training.get("adapter_fingerprint_after") != current_adapter
            or source_training.get("outside_adapter_fingerprint_after")
            != current_outside
            or source_training.get("reverse_adapter_fingerprint_after")
            != current_reverse
        ):
            raise RuntimeError(
                "trained adapter checkpoint does not prove exact v41 lineage"
            )
        return "procedural_adapter"

    fingerprints = {
        "learned_state": _named_state_fingerprint(
            policy,
            include=_is_reverse_construction_state,
            domain=b"project-angler.phase5-reverse-construction.v1",
        ),
        "outside_learned": _named_state_fingerprint(
            policy,
            include=lambda name: not _is_reverse_construction_state(name),
            domain=b"project-angler.phase5-outside-reverse-construction.v1",
        ),
        "code_acquisition": _named_state_fingerprint(
            policy,
            include=lambda name: name in _REVERSE_CONSTRUCTION_TRAINABLE_NAMES,
            domain=b"project-angler.phase5-reverse-code-acquisition.v1",
        ),
        "fast_adapter": _named_state_fingerprint(
            policy,
            include=lambda name: name.startswith("procedural_fast_adapter."),
            domain=b"project-angler.phase5-reverse-fast-adapter.v1",
        ),
        "goal_projection": _named_state_fingerprint(
            policy,
            include=lambda name: name.startswith("procedural_goal_projection."),
            domain=b"project-angler.phase5-reverse-goal-projection.v1",
        ),
    }
    changed_groups = (
        "learned_state",
        "code_acquisition",
        "fast_adapter",
        "goal_projection",
    )
    expected_trainable = [
        name for name, _ in policy.named_parameters() if _is_reverse_construction_state(name)
    ]
    if (
        not valid_source
        or source_training.get("training_stage") != "reverse_construction"
        or source_training.get("outside_learned_fingerprint_before")
        != source_training.get("outside_learned_fingerprint_after")
        or any(
            source_training.get(f"{group}_fingerprint_before")
            == source_training.get(f"{group}_fingerprint_after")
            for group in changed_groups
        )
        or any(
            source_training.get(f"{group}_fingerprint_after") != current
            for group, current in fingerprints.items()
        )
        or source_training.get("trainable_parameter_names") != expected_trainable
    ):
        raise RuntimeError(
            "reverse-construction checkpoint does not prove exact v41 lineage"
        )
    return "reverse_construction"


def run_operator_localization_audit(
    profile: str | RunProfile = "composition",
    *,
    seed: int = 85_031,
    device: str | torch.device = "cpu",
    compiler_checkpoint: str | Path = _PHASE4_CHECKPOINT,
    initial_checkpoint: str | Path,
) -> dict[str, Any]:
    """Test bridge-local causal signal without an optimizer or state mutation."""

    if isinstance(profile, str):
        try:
            settings = _PROFILES[profile]
        except KeyError as error:
            raise ValueError(f"unknown profile: {profile}") from error
    elif isinstance(profile, RunProfile):
        settings = profile
    else:
        raise TypeError("profile must be a name or RunProfile")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    target_device = torch.device(device)
    python_rng = random.getstate()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    result: dict[str, Any] | None = None
    try:
        stable_compiler, compiler_record = _load_phase4_compiler(
            compiler_checkpoint
        )
        if stable_compiler.core.width != settings.width:
            raise RuntimeError(
                "operator audit profile differs from the frozen Phase-4 compiler"
            )
        policy = SkillMemoryPolicy(settings, stable_compiler).to(
            device=target_device,
            dtype=torch.float32,
        )
        initialization = _load_initial_policy_checkpoint(
            policy,
            initial_checkpoint,
            settings,
        )
        checkpoint_kind = _validate_operator_audit_checkpoint_lineage(
            policy,
            initialization,
        )
        trained_adapter = checkpoint_kind != "retained_v41"
        policy.eval()
        policy.requires_grad_(False)
        slow_before = reasoning_state_digest(policy)
        identity_before = _parameter_identity_fingerprint(policy)
        reports: list[dict[str, Any]] = []
        rows_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}
        vectors_by_bridge: dict[str, dict[int, torch.Tensor]] = {
            label: {} for label in _OPERATOR_AUDIT_BRIDGES
        }
        seeds = tuple(seed + offset for offset in range(_OPERATOR_AUDIT_SEED_COUNT))
        for audit_seed in seeds:
            _seed_reproducible_stage(
                audit_seed,
                "operator-localization-audit",
                target_device,
            )
            report, private_rows, private_vectors = _operator_localization_seed(
                policy,
                seed=audit_seed,
            )
            reports.append(report)
            rows_by_seed[audit_seed] = private_rows
            for label, vector in private_vectors.items():
                vectors_by_bridge[label][audit_seed] = vector
        summary = _summarize_operator_localization(
            reports,
            rows_by_seed,
            vectors_by_bridge,
            trained_fast_adapter=trained_adapter,
        )
        if reasoning_state_digest(policy) != slow_before or (
            _parameter_identity_fingerprint(policy) != identity_before
        ):
            raise RuntimeError("operator localization changed the retained policy")
        if any(parameter.requires_grad for parameter in policy.parameters()) or any(
            parameter.grad is not None for parameter in policy.parameters()
        ):
            raise RuntimeError("operator localization left trainable or gradient state")
        result = {
            "report_version": _OPERATOR_AUDIT_REPORT_VERSION,
            "profile": settings.name,
            "device": str(target_device),
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "instances_per_program": _OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
            "query_instances_per_deep_program": _OPERATOR_AUDIT_QUERY_INSTANCES,
            "compiler_checkpoint": compiler_record,
            "initialization": initialization,
            "seed_reports": reports,
            "summary": summary,
            "slow_fingerprint_before": slow_before,
            "slow_fingerprint_after": reasoning_state_digest(policy),
            "parameter_identity_before": identity_before,
            "parameter_identity_after": _parameter_identity_fingerprint(policy),
            "claims": {
                "optimizer_created": False,
                "backward_called": False,
                "optimizer_steps": 0,
                "slow_weights_updated": False,
                "competence_state_persisted": False,
                "checkpoint_written": False,
                "hidden_data_scope": "discarded evaluator statistics and gradients only",
                "learner_api_changed": False,
            },
        }
    finally:
        random.setstate(python_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if result is None:
        raise RuntimeError("operator localization produced no result")
    rng_restored = random.getstate() == python_rng and torch.equal(
        torch.random.get_rng_state(), cpu_rng
    )
    if cuda_rng is not None:
        rng_restored = rng_restored and all(
            torch.equal(left, right)
            for left, right in zip(
                torch.cuda.get_rng_state_all(),
                cuda_rng,
                strict=True,
            )
        )
    if not rng_restored:
        raise RuntimeError("operator localization did not restore RNG state")
    result["rng_state_restored"] = True
    result["result_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def run(
    profile: str | RunProfile = "smoke",
    *,
    seed: int = 85_001,
    device: str | torch.device = "cpu",
    stage: str = "integrated",
    compiler_checkpoint: str | Path = _PHASE4_CHECKPOINT,
    initial_checkpoint: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(profile, str):
        try:
            settings = _PROFILES[profile]
        except KeyError as error:
            raise ValueError(f"unknown profile: {profile}") from error
    elif isinstance(profile, RunProfile):
        settings = profile
    else:
        raise TypeError("profile must be a name or RunProfile")
    if stage not in _TRAINING_STAGES:
        raise ValueError(f"stage must be one of {_TRAINING_STAGES}")
    if stage in {
        "relational_acquisition",
        "harmonization",
        "procedural_adapter",
        "reverse_construction",
        "reverse_harmonization",
        "procedural_coadaptation",
        "reversible_transition_acquisition",
    } and (
        initial_checkpoint is None
    ):
        raise ValueError(f"{stage} requires an initial checkpoint")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if stage == "harmonization" and settings.meta_steps > _HARMONIZATION_OUTER_STEPS:
        settings = replace(settings, meta_steps=_HARMONIZATION_OUTER_STEPS)
    if stage == "procedural_adapter":
        settings = replace(
            settings,
            meta_steps=_PROCEDURAL_ADAPTER_OUTER_STEPS,
            meta_instances_per_program=_OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
    if stage == "reverse_construction":
        settings = replace(
            settings,
            meta_steps=_REVERSE_CONSTRUCTION_OUTER_STEPS,
            meta_instances_per_program=_OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
    if stage == "reverse_harmonization":
        settings = replace(
            settings,
            meta_steps=_REVERSE_HARMONIZATION_OUTER_STEPS,
            meta_instances_per_program=_OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
    if stage == "procedural_coadaptation":
        settings = replace(
            settings,
            meta_steps=_PROCEDURAL_COADAPTATION_OUTER_STEPS,
            meta_instances_per_program=_OPERATOR_AUDIT_INSTANCES_PER_PROGRAM,
        )
    target_device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    stable_compiler, compiler_checkpoint_record = _load_phase4_compiler(
        compiler_checkpoint
    )
    if stable_compiler.core.width != settings.width:
        raise RuntimeError("run profile width differs from the frozen Phase-4 compiler")
    policy = SkillMemoryPolicy(settings, stable_compiler).to(
        device=target_device,
        dtype=torch.float32,
    )
    initialization = (
        {
            "kind": "fresh_slow_weights",
            "online_state_restored": False,
            "slow_model_state_restored": False,
        }
        if initial_checkpoint is None
        else {
            "kind": "compatible_phase5_checkpoint",
            **_load_initial_policy_checkpoint(
                policy,
                initial_checkpoint,
                settings,
            ),
        }
    )
    if stage == "relational_acquisition" and initialization.get(
        "source_stage"
    ) not in {"integrated", "relational_acquisition"}:
        raise RuntimeError(
            "relational_acquisition requires an integrated or "
            "relational-acquisition checkpoint"
        )
    if stage == "harmonization" and (
        initialization.get("source_stage") != "relational_acquisition"
        or initialization.get("source_runner")
        != "angler.phase5-skill-memory-stream.v13"
        or initialization.get("sha256")
        != _HARMONIZATION_SOURCE_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "harmonization requires the exact retained v41 relational checkpoint"
        )
    if stage == "procedural_adapter":
        _validate_retained_v41_neutral_migration(policy, initialization)
        training = _train_procedural_adapter(policy, settings, seed)
    elif stage == "reverse_construction":
        _validate_retained_v41_neutral_migration(policy, initialization)
        training = _train_reverse_construction(policy, settings, seed)
    elif stage == "reverse_harmonization":
        if (
            initialization.get("sha256")
            != _REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256
            or initialization.get("source_runner")
            != "angler.phase5-skill-memory-stream.v19"
            or initialization.get("source_stage") != "reverse_construction"
            or initialization.get("fresh_parameter_keys")
        ):
            raise RuntimeError(
                "reverse harmonization requires the exact v47 learned channels"
            )
        training = _train_reverse_harmonization(policy, settings, seed)
    elif stage == "procedural_coadaptation":
        if (
            initialization.get("sha256")
            != _PROCEDURAL_COADAPTATION_SOURCE_CHECKPOINT_SHA256
            or initialization.get("source_runner")
            != "angler.phase5-skill-memory-stream.v20"
            or initialization.get("source_stage") != "reverse_harmonization"
            or initialization.get("fresh_parameter_keys")
            or _validate_operator_audit_checkpoint_lineage(
                policy,
                initialization,
            )
            != "reverse_harmonization"
        ):
            raise RuntimeError(
                "procedural coadaptation requires the exact validated v48 policy"
            )
        training = _train_procedural_coadaptation(policy, settings, seed)
    elif stage == "reversible_transition_acquisition":
        _validate_retained_v41_neutral_migration(policy, initialization)
        training = _train_reversible_transition_acquisition(policy, settings, seed)
    else:
        training = _train(policy, settings, seed, stage=stage)
    trained_fingerprint = reasoning_state_digest(policy)
    policy.eval()
    policy.requires_grad_(False)
    stage_seeds: dict[str, int] = {}
    if stage == "procedural_adapter":
        oracle_readout = {
            "status": "not_run_unchanged_memory_substrate",
            "codes_persisted": False,
        }
    elif stage == "reverse_construction":
        oracle_readout = {
            "status": "not_run_joint_code_alignment_trained",
            "codes_persisted": False,
            "memory_substrate_updated": True,
        }
    elif stage == "reverse_harmonization":
        oracle_readout = {
            "status": "not_run_frozen_codes_and_channels_harmonized",
            "codes_persisted": False,
            "memory_substrate_updated": False,
        }
    elif stage == "procedural_coadaptation":
        oracle_readout = {
            "status": "not_run_end_to_end_procedural_credit_trained",
            "codes_persisted": False,
            "memory_substrate_updated": True,
        }
    elif stage == "reversible_transition_acquisition":
        oracle_readout = {
            "status": "not_run_reversible_procedure_map_trained",
            "codes_persisted": False,
            "memory_substrate_updated": True,
        }
    else:
        stage_seeds["oracle_readout"] = _seed_reproducible_stage(
            seed,
            "oracle-readout",
            target_device,
        )
        oracle_readout = (
            oracle_leaf_readout_gate(
                policy,
                seed=seed + 7_000_003,
                steps=64,
                instances_per_operator=8,
            )
            if settings.meta_steps >= 16
            else {
                "status": "not_run_below_16_outer_steps",
                "codes_persisted": False,
            }
        )
    if stage == "leaf_core":
        stage_seeds["leaf_core"] = _seed_reproducible_stage(
            seed,
            "leaf-core-evaluation",
            target_device,
        )
        online = _evaluate_leaf_core(policy, settings, seed)
        leaf_retention = online
    else:
        stage_seeds["leaf_retention"] = _seed_reproducible_stage(
            seed,
            "leaf-retention-evaluation",
            target_device,
        )
        leaf_retention = _evaluate_leaf_core(policy, settings, seed)
        stage_seeds["online"] = _seed_reproducible_stage(
            seed,
            "online-evaluation",
            target_device,
        )
        online = _evaluate(policy, settings, seed)
    if online["slow_fingerprint_before"] != trained_fingerprint:
        raise RuntimeError("frozen online policy differs from trained slow state")
    result: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "profile": settings.name,
        "stage": stage,
        "seed": seed,
        "device": str(target_device),
        "candidate_count": len(_PERMUTATIONS),
        "compiler_checkpoint": compiler_checkpoint_record,
        "initialization": initialization,
        "stage_seeds": stage_seeds,
        "training": training,
        "oracle_readout_diagnostic": oracle_readout,
        "leaf_retention_after_training": leaf_retention,
        "online": online,
        "claims": {
            "model_or_foundation_weights_updated": False,
            "online_slow_weights_updated": False,
            "online_task_or_domain_ids": False,
            "online_full_history_replay": False,
            "component_feedback": (
                "four attempted public outputs plus scalar scores"
                if stage
                in {
                    "reverse_construction",
                    "reverse_harmonization",
                    "procedural_coadaptation",
                    "reversible_transition_acquisition",
                }
                else "attempted public permutation plus scalar score"
            ),
            "composition_feedback": (
                "four attempted public outputs plus scalar scores"
                if stage
                in {
                    "reverse_construction",
                    "reverse_harmonization",
                    "procedural_coadaptation",
                    "reversible_transition_acquisition",
                }
                else "none"
            ),
            "phase4_compiler_scope": compiler_checkpoint_record["usage_scope"],
            "proposal_time_executor_calls": 0,
        },
    }
    result["result_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "compiler_checkpoint_sha256": compiler_checkpoint_record["sha256"],
                "model": policy.state_dict(),
                "profile": asdict(settings),
                "result_digest": result["result_digest"],
                "runner": _REPORT_VERSION,
                "seed": seed,
                "stage": stage,
                "initialization": initialization,
                "training": training,
            },
            checkpoint_path,
        )
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(_PROFILES), default="smoke")
    parser.add_argument(
        "--stage",
        choices=_TRAINING_STAGES,
        default="integrated",
    )
    parser.add_argument("--seed", type=int, default=85_001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--result-json")
    parser.add_argument("--compiler-checkpoint", default=str(_PHASE4_CHECKPOINT))
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--operator-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operator_audit:
        if args.initial_checkpoint is None:
            raise ValueError("--operator-audit requires --initial-checkpoint")
        if args.checkpoint is not None:
            raise ValueError("--operator-audit cannot write a checkpoint")
        result = run_operator_localization_audit(
            args.profile,
            seed=args.seed,
            device=args.device,
            compiler_checkpoint=args.compiler_checkpoint,
            initial_checkpoint=args.initial_checkpoint,
        )
    else:
        result = run(
            args.profile,
            seed=args.seed,
            device=args.device,
            stage=args.stage,
            compiler_checkpoint=args.compiler_checkpoint,
            initial_checkpoint=args.initial_checkpoint,
            checkpoint=args.checkpoint,
        )
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.result_json:
        destination = Path(args.result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
