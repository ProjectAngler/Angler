"""Learned, fixed-capacity procedural memory with skill-local evidence sets.

The module routes structural state/goal/candidate features into one numeric
competence state.  It has no task, domain, episode, adapter, or solution-ID
input.  Slow parameters learn how to encode evidence and infer candidate
utility; online feedback changes only one explicitly carried evidence slot.

Each slot is a running mean of independently encoded public observations, so
its content is invariant to observation order and cannot be overwritten by a
long recurrent chain.  An untouched state contributes exactly zero.  A
feedback write is first proposed and can then be admitted transactionally;
rejected transactions return the exact incoming state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.reasoning.self_referential_memory import (
    SelfReferentialMemory,
    SelfReferentialState,
    restore_self_referential_state,
)


_FAST_NAMES = ("delta_y", "delta_q", "delta_k", "delta_beta")
_SNAPSHOT_KEYS = tuple(f"fast.{name}" for name in _FAST_NAMES) + (
    "slot_latents",
    "key_offsets",
    "occupied",
    "write_counts",
)
_STATE_DIGEST_DOMAIN = b"project-angler.procedural-skill-state.v1\x00"
_PUBLIC_CODE_BETA_CHANNEL = 0
_PUBLIC_COUNT_BETA_CHANNEL = 1
_PUBLIC_COUNT_FLAT_INDEX = 0


@dataclass(frozen=True, slots=True)
class ProceduralSkillState:
    """One fixed-size competence state containing unnamed routed slots."""

    fast_weights: SelfReferentialState
    slot_latents: torch.Tensor
    key_offsets: torch.Tensor
    occupied: torch.Tensor
    write_counts: torch.Tensor

    def __post_init__(self) -> None:
        _validate_state_structure(self)

    @property
    def batch_size(self) -> int:
        return int(self.key_offsets.shape[0])

    @property
    def slot_count(self) -> int:
        return int(self.key_offsets.shape[1])

    @property
    def width(self) -> int:
        return int(self.key_offsets.shape[2])

    def numel(self) -> int:
        """Return the constant scalar capacity, including routing metadata."""

        return (
            self.fast_weights.numel()
            + self.slot_latents.numel()
            + self.key_offsets.numel()
            + self.occupied.numel()
            + self.write_counts.numel()
        )


@dataclass(frozen=True, slots=True)
class ProceduralSkillRead:
    """A read-only candidate-score residual and its learner-generated route."""

    state_embeddings: torch.Tensor
    goal_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    route_key: torch.Tensor
    route_query: torch.Tensor
    read_weights: torch.Tensor
    route_probabilities: torch.Tensor
    write_weights: torch.Tensor
    write_slots: torch.Tensor
    plastic_context: torch.Tensor
    evidence_count: torch.Tensor
    score_bias: torch.Tensor
    public_evidence_enabled: bool
    public_transition_gate: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class ProceduralSkillProposal:
    """A reversible one-slot feedback proposal awaiting local admission."""

    incoming_state: ProceduralSkillState
    candidate_state: ProceduralSkillState
    read: ProceduralSkillRead
    attempted_indices: torch.Tensor
    rewards: torch.Tensor
    base_logits: torch.Tensor
    feedback_event: torch.Tensor
    write_slots: torch.Tensor
    delta_norm: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProceduralSkillWrite:
    """Outcome of immediate feedback-consistency admission."""

    state: ProceduralSkillState
    accepted: torch.Tensor
    write_slots: torch.Tensor
    delta_norm: torch.Tensor
    before_loss: torch.Tensor
    after_loss: torch.Tensor


class PublicEvidenceResidualWriter(nn.Module):
    """Bounded learned bridge from typed public evidence into memory events.

    This module is optional and starts as an exact no-op.  It cannot route a
    skill, alter current logits, or decode an answer.  When attached to a
    memory tier, it may only add bounded residuals to the content and signed
    outcome directions already produced by that tier's frozen generic writer.
    """

    def __init__(
        self,
        width: int,
        *,
        hidden_width: int = 16,
        residual_limit: float = 0.25,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("public-evidence writer width must be positive and even")
        if hidden_width <= 0:
            raise ValueError("public-evidence writer hidden width must be positive")
        if not math.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("public-evidence writer residual limit must be positive")
        self.width = width
        self.content_width = width // 2
        self.outcome_width = width - self.content_width
        self.residual_limit = float(residual_limit)
        input_width = width + self.content_width + self.outcome_width
        self.input_norm = nn.LayerNorm(input_width, elementwise_affine=False)
        self.hidden = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
        )
        self.content_head = nn.Linear(
            hidden_width,
            self.content_width,
            bias=False,
        )
        self.direction_head = nn.Linear(
            hidden_width,
            self.outcome_width,
            bias=False,
        )
        nn.init.zeros_(self.content_head.weight)
        nn.init.zeros_(self.direction_head.weight)

    def forward(
        self,
        public_evidence: torch.Tensor,
        base_content: torch.Tensor,
        base_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(public_evidence, torch.Tensor)
            or public_evidence.ndim != 2
            or public_evidence.shape[-1] != self.width
        ):
            raise ValueError("public evidence must have shape [batch, width]")
        batch = public_evidence.shape[0]
        if base_content.shape != (batch, self.content_width):
            raise ValueError("base content has the wrong shape")
        if base_direction.shape != (batch, self.outcome_width):
            raise ValueError("base direction has the wrong shape")
        for value in (public_evidence, base_content, base_direction):
            if (
                not value.is_floating_point()
                or value.device != public_evidence.device
                or value.dtype != public_evidence.dtype
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("public-evidence writer inputs must match and be finite")
        joined = torch.cat(
            (public_evidence, base_content, base_direction),
            dim=-1,
        )
        hidden = self.hidden(self.input_norm(joined))
        gate = public_evidence.detach().abs().sum(dim=-1, keepdim=True) > 0.0
        content = self.residual_limit * torch.tanh(self.content_head(hidden))
        direction = self.residual_limit * torch.tanh(self.direction_head(hidden))
        return (
            torch.where(gate, content, torch.zeros_like(content)),
            torch.where(gate, direction, torch.zeros_like(direction)),
        )


class PublicEvidenceLatentReader(nn.Module):
    """Translate routed public evidence into the shared procedure-code space.

    The reader is optional, candidate-blind, and bias-free.  Its final map
    starts at zero, so attaching it preserves every pre-existing read exactly.
    It subtracts the same network evaluated with zero public evidence under the
    identical base context.  Consequently a learned base-context-only output
    cancels exactly; matched content objectives must still teach which public
    evidence is useful.  Public evidence is stored in a fixed-capacity state
    channel and can affect behavior only through the same procedure context
    consumed by the retained decoder and reversible transition.
    """

    def __init__(
        self,
        width: int,
        *,
        hidden_width: int = 64,
        residual_limit: float = 2.0,
        transition_rank: int = 8,
        transition_gate_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("public-evidence reader width must be positive")
        if hidden_width <= 0:
            raise ValueError("public-evidence reader hidden width must be positive")
        if not math.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("public-evidence reader residual limit must be positive")
        if (
            isinstance(transition_rank, bool)
            or not isinstance(transition_rank, int)
            or transition_rank <= 0
            or not math.isfinite(transition_gate_limit)
            or transition_gate_limit <= 0.0
        ):
            raise ValueError("public transition gate configuration is invalid")
        self.width = width
        self.residual_limit = float(residual_limit)
        self.transition_rank = transition_rank
        self.transition_gate_limit = float(transition_gate_limit)
        self.input_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.hidden = nn.Sequential(
            nn.Linear(2 * width, hidden_width, bias=False),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_width, width, bias=False)
        self.transition_output = nn.Linear(hidden_width, transition_rank, bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.transition_output.weight)

    def forward(
        self,
        public_context: torch.Tensor,
        base_context: torch.Tensor,
        *,
        public_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, _ = self.read_effects(
            public_context,
            base_context,
            public_confidence=public_confidence,
        )
        return residual

    def read_effects(
        self,
        public_context: torch.Tensor,
        base_context: torch.Tensor,
        *,
        public_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent and post-saturation gate residuals from public content."""

        if (
            not isinstance(public_context, torch.Tensor)
            or public_context.ndim != 2
            or public_context.shape[-1] != self.width
            or not public_context.is_floating_point()
            or not bool(torch.isfinite(public_context).all().item())
            or not isinstance(base_context, torch.Tensor)
            or base_context.shape != public_context.shape
            or base_context.device != public_context.device
            or base_context.dtype != public_context.dtype
            or not bool(torch.isfinite(base_context).all().item())
        ):
            raise ValueError(
                "public and base contexts must be finite, matching [batch, width] tensors"
            )
        if public_confidence is None:
            public_confidence = torch.ones(
                (public_context.shape[0], 1),
                device=public_context.device,
                dtype=public_context.dtype,
            )
        if (
            not isinstance(public_confidence, torch.Tensor)
            or public_confidence.shape != (public_context.shape[0], 1)
            or public_confidence.device != public_context.device
            or public_confidence.dtype != public_context.dtype
            or not bool(torch.isfinite(public_confidence).all().item())
            or bool((public_confidence < 0.0).any().item())
            or bool((public_confidence > 1.0).any().item())
        ):
            raise ValueError("public confidence must be finite [batch, 1] in [0, 1]")
        normalized_public = self.input_norm(public_context) * public_confidence
        normalized_base = self.input_norm(base_context)
        joined = torch.cat((normalized_public, normalized_base), dim=-1)
        counterfactual = torch.cat(
            (torch.zeros_like(normalized_public), normalized_base),
            dim=-1,
        )
        live_hidden = self.hidden(joined)
        counterfactual_hidden = self.hidden(counterfactual)
        residual = self.output(live_hidden) - self.output(counterfactual_hidden)
        transition_residual = self.transition_output(
            live_hidden
        ) - self.transition_output(counterfactual_hidden)
        bounded = self.residual_limit * torch.tanh(residual)
        bounded_transition = self.transition_gate_limit * torch.tanh(
            transition_residual
        )
        gate = public_context.detach().abs().sum(dim=-1, keepdim=True) > 0.0
        return (
            torch.where(gate, bounded, torch.zeros_like(bounded)),
            torch.where(
                gate,
                bounded_transition,
                torch.zeros_like(bounded_transition),
            ),
        )


def snapshot_procedural_skill_state(
    state: ProceduralSkillState,
) -> dict[str, torch.Tensor]:
    """Copy all competence tensors without retaining storage aliases."""

    _validate_state_structure(state)
    snapshot = {
        f"fast.{name}": getattr(state.fast_weights, name).detach().clone()
        for name in _FAST_NAMES
    }
    snapshot.update(
        {
            "slot_latents": state.slot_latents.detach().clone(),
            "key_offsets": state.key_offsets.detach().clone(),
            "occupied": state.occupied.detach().clone(),
            "write_counts": state.write_counts.detach().clone(),
        }
    )
    return snapshot


def restore_procedural_skill_state(
    snapshot: Mapping[str, torch.Tensor],
) -> ProceduralSkillState:
    """Restore an independent competence state from an exact tensor snapshot."""

    if set(snapshot) != set(_SNAPSHOT_KEYS):
        missing = sorted(set(_SNAPSHOT_KEYS) - set(snapshot))
        extra = sorted(set(snapshot) - set(_SNAPSHOT_KEYS))
        raise ValueError(
            f"skill-state snapshot keys differ; missing={missing}, extra={extra}"
        )
    if any(not isinstance(snapshot[name], torch.Tensor) for name in _SNAPSHOT_KEYS):
        raise TypeError("every skill-state snapshot value must be a tensor")
    fast = restore_self_referential_state(
        {
            name: snapshot[f"fast.{name}"]
            for name in _FAST_NAMES
        }
    )
    return ProceduralSkillState(
        fast_weights=fast,
        slot_latents=snapshot["slot_latents"].detach().clone(),
        key_offsets=snapshot["key_offsets"].detach().clone(),
        occupied=snapshot["occupied"].detach().clone(),
        write_counts=snapshot["write_counts"].detach().clone(),
    )


def procedural_skill_state_digest(state: ProceduralSkillState) -> str:
    """Return a deterministic identity for every competence-state tensor."""

    _validate_state_structure(state)
    digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
    for name in _SNAPSHOT_KEYS:
        tensor = (
            getattr(state.fast_weights, name.removeprefix("fast."))
            if name.startswith("fast.")
            else getattr(state, name)
        ).detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", tensor.ndim))
        digest.update(struct.pack(f">{tensor.ndim}Q", *tensor.shape))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def permute_procedural_skill_slots(
    state: ProceduralSkillState,
    permutation: Sequence[int],
) -> ProceduralSkillState:
    """Permute unnamed slots for causal/equivariance tests.

    The corresponding module anchors must be permuted in the same order.  This
    function is a state transformation, not an inference-time routing input.
    """

    _validate_state_structure(state)
    order = tuple(permutation)
    if (
        len(order) != state.slot_count
        or any(isinstance(value, bool) or not isinstance(value, int) for value in order)
        or sorted(order) != list(range(state.slot_count))
    ):
        raise ValueError("slot permutation must contain every slot exactly once")
    index = torch.tensor(order, device=state.key_offsets.device, dtype=torch.long)
    batch = state.batch_size
    slots = state.slot_count
    fast_values: dict[str, torch.Tensor] = {}
    for name in _FAST_NAMES:
        value = getattr(state.fast_weights, name)
        shaped = value.reshape(batch, slots, *value.shape[1:])
        fast_values[name] = shaped.index_select(1, index).reshape_as(value)
    return ProceduralSkillState(
        fast_weights=SelfReferentialState(**fast_values),
        slot_latents=state.slot_latents.index_select(1, index),
        key_offsets=state.key_offsets.index_select(1, index),
        occupied=state.occupied.index_select(1, index),
        write_counts=state.write_counts.index_select(1, index),
    )


def zero_procedural_skill_content(
    state: ProceduralSkillState,
) -> ProceduralSkillState:
    """Remove learned fast-weight content while retaining routing metadata.

    This is a causal intervention rather than an online operation.  Keeping
    occupancy, learned key offsets, and write counts distinguishes erased
    procedural content from a completely fresh/reset competence state.
    """

    _validate_state_structure(state)
    fast = SelfReferentialState(
        **{
            name: torch.zeros_like(getattr(state.fast_weights, name))
            for name in _FAST_NAMES
        }
    )
    return ProceduralSkillState(
        fast_weights=fast,
        slot_latents=torch.zeros_like(state.slot_latents),
        key_offsets=state.key_offsets.detach().clone(),
        occupied=state.occupied.detach().clone(),
        write_counts=state.write_counts.detach().clone(),
    )


def public_evidence_skill_content(
    state: ProceduralSkillState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed public codes and their fixed per-slot observation counts.

    The layout reuses an otherwise dormant compatibility tensor without
    changing state capacity: beta channel zero carries one width-sized code per
    slot, and one scalar in beta channel one carries its exchangeable-mean
    count.  The embedded recurrent memory must remain bypassed while this
    versioned layout is active.
    """

    _validate_state_structure(state)
    beta = state.fast_weights.delta_beta.reshape(
        state.batch_size,
        state.slot_count,
        *state.fast_weights.delta_beta.shape[1:],
    )
    codes = beta[..., _PUBLIC_CODE_BETA_CHANNEL].reshape(
        state.batch_size,
        state.slot_count,
        state.width,
    )
    count_cells = beta[..., _PUBLIC_COUNT_BETA_CHANNEL].reshape(
        state.batch_size,
        state.slot_count,
        state.width,
    )
    counts = count_cells[..., _PUBLIC_COUNT_FLAT_INDEX]
    return codes, counts


def zero_public_evidence_skill_content(
    state: ProceduralSkillState,
) -> ProceduralSkillState:
    """Erase only public procedure codes while preserving their count metadata."""

    return _zero_public_evidence_skill_content(state, preserve_graph=False)


def differentiable_zero_public_evidence_skill_content(
    state: ProceduralSkillState,
) -> ProceduralSkillState:
    """Erase public codes while retaining the non-code computation graph.

    A paired training loss can use this counterfactual to subtract every
    shared base-state and query pathway.  The removed public-code channel has
    no gradient; all unaffected floating-point tensors retain theirs.
    """

    return _zero_public_evidence_skill_content(state, preserve_graph=True)


def _zero_public_evidence_skill_content(
    state: ProceduralSkillState,
    *,
    preserve_graph: bool,
) -> ProceduralSkillState:
    _validate_state_structure(state)
    codes, _ = public_evidence_skill_content(state)
    beta = state.fast_weights.delta_beta.reshape(
        state.batch_size,
        state.slot_count,
        *state.fast_weights.delta_beta.shape[1:],
    )
    channels = list(torch.unbind(beta, dim=-1))
    channels[_PUBLIC_CODE_BETA_CHANNEL] = torch.zeros_like(codes).reshape_as(
        channels[_PUBLIC_CODE_BETA_CHANNEL]
    )
    next_fast = {
        name: (
            getattr(state.fast_weights, name).clone()
            if preserve_graph
            else getattr(state.fast_weights, name).detach().clone()
        )
        for name in _FAST_NAMES
    }
    next_fast["delta_beta"] = torch.stack(channels, dim=-1).reshape_as(
        state.fast_weights.delta_beta
    )
    return ProceduralSkillState(
        fast_weights=SelfReferentialState(**next_fast),
        slot_latents=(
            state.slot_latents.clone()
            if preserve_graph
            else state.slot_latents.detach().clone()
        ),
        key_offsets=(
            state.key_offsets.clone()
            if preserve_graph
            else state.key_offsets.detach().clone()
        ),
        occupied=state.occupied.detach().clone(),
        write_counts=state.write_counts.detach().clone(),
    )


class RoutedProceduralMemory(nn.Module):
    """Learned routing over isolated, exchangeable procedural evidence sets."""

    def __init__(
        self,
        width: int,
        *,
        slots: int = 16,
        heads: int = 8,
        read_top_k: int = 2,
        hidden_width: int | None = None,
        score_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if width < 2 or slots <= 0 or heads <= 0:
            raise ValueError("width must be at least two; slots and heads must be positive")
        if width % heads:
            raise ValueError("heads must divide width")
        if not 1 <= read_top_k <= slots:
            raise ValueError("read_top_k must be between one and slots")
        if not math.isfinite(score_limit) or score_limit <= 0.0:
            raise ValueError("score_limit must be finite and positive")
        hidden = width * 2 if hidden_width is None else hidden_width
        if hidden <= 0:
            raise ValueError("hidden_width must be positive")

        self.width = width
        self.slots = slots
        self.read_top_k = read_top_k
        self.score_limit = float(score_limit)
        # Preserve the Phase-5 state/snapshot shape while the recurrent SRWM
        # channel is ablated.  Its slow parameters remain frozen and it is
        # never executed.  An optional public-evidence codec may version and
        # reuse a documented beta subchannel; otherwise every fast tensor
        # remains exact zero and competence lives in the slot summaries below.
        self.memory = SelfReferentialMemory(width, heads=heads)
        self.memory.requires_grad_(False)
        self.goal_route = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.goal_route.weight)
        self.slot_anchors = nn.Parameter(torch.empty(slots, width))
        self.reuse_similarity_threshold = 0.999
        # The opaque goal token is an address, never procedural content.  Each
        # event is encoded independently from public state and the attempted
        # public candidate.  No prior memory or innovation enters the event,
        # which makes the accumulated set exchangeable.  Unweighted context
        # and centered outcome covariance occupy disjoint channels: a mean
        # reward cannot masquerade as a procedure, complemented feedback is
        # exactly antisymmetric, and neutral feedback still retains the full
        # public observation.
        self.evidence_content_width = width // 2
        self.evidence_outcome_width = width - self.evidence_content_width
        feedback_width = width * 3
        self.feedback_encoder = nn.Sequential(
            nn.LayerNorm(feedback_width),
            nn.Linear(feedback_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.evidence_content_width),
        )
        self.feedback_direction_encoder = nn.Sequential(
            nn.LayerNorm(feedback_width),
            nn.Linear(feedback_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.evidence_outcome_width, bias=False),
        )
        # A shared nonlinear conditional decoder is strictly more expressive
        # than the old bilinear readout.  Subtracting its zero-context response
        # makes empty-memory output bit-exact zero without a learned bypass.
        decoder_width = width * 5
        self.utility_decoder = nn.Sequential(
            nn.LayerNorm(decoder_width),
            nn.Linear(decoder_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        nn.init.normal_(self.slot_anchors, mean=0.0, std=1.0 / math.sqrt(width))

    def state_numel(self, batch_size: int = 1) -> int:
        """Return fixed capacity, independent of the number of feedback events."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        fast = self.memory.state_numel(batch_size * self.slots)
        routing = batch_size * self.slots * self.width * 2
        metadata = batch_size * self.slots * 2
        return fast + routing + metadata

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ProceduralSkillState:
        """Create an empty state whose candidate-score residual is exactly zero."""

        reference = self.memory.base_y
        target_device = reference.device if device is None else torch.device(device)
        target_dtype = reference.dtype if dtype is None else dtype
        if target_device != reference.device or target_dtype != reference.dtype:
            raise ValueError(
                "initial skill state must match the memory parameter device and dtype"
            )
        fast = self.memory.initial_state(
            batch_size * self.slots,
            device=target_device,
            dtype=target_dtype,
        )
        target_device = fast.delta_y.device
        target_dtype = fast.delta_y.dtype
        return ProceduralSkillState(
            fast_weights=fast,
            slot_latents=torch.zeros(
                batch_size,
                self.slots,
                self.width,
                device=target_device,
                dtype=target_dtype,
            ),
            key_offsets=torch.zeros(
                batch_size,
                self.slots,
                self.width,
                device=target_device,
                dtype=target_dtype,
            ),
            occupied=torch.zeros(
                batch_size,
                self.slots,
                device=target_device,
                dtype=torch.bool,
            ),
            write_counts=torch.zeros(
                batch_size,
                self.slots,
                device=target_device,
                dtype=torch.long,
            ),
        )

    def forward(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        *,
        state: ProceduralSkillState,
        candidate_mask: torch.Tensor | None = None,
        include_public_evidence: bool = False,
    ) -> ProceduralSkillRead:
        return self.read(
            state_embeddings,
            goal_embeddings,
            candidate_embeddings,
            state=state,
            candidate_mask=candidate_mask,
            include_public_evidence=include_public_evidence,
        )

    def read(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        *,
        state: ProceduralSkillState,
        candidate_mask: torch.Tensor | None = None,
        include_public_evidence: bool = False,
    ) -> ProceduralSkillRead:
        """Read a bounded score residual without changing competence state."""

        if type(include_public_evidence) is not bool:
            raise TypeError("include_public_evidence must be bool")
        candidates, mask = self._validate_read_inputs(
            state_embeddings,
            goal_embeddings,
            candidate_embeddings,
            candidate_mask,
            state,
        )
        safe_candidates = torch.where(
            mask.unsqueeze(-1),
            candidates,
            torch.zeros_like(candidates),
        )
        route_key = F.normalize(
            self.goal_route(goal_embeddings),
            dim=-1,
            eps=1e-8,
        )
        route_query = route_key
        anchors = F.normalize(self.slot_anchors, dim=-1, eps=1e-8)
        effective_keys = F.normalize(
            anchors.unsqueeze(0) + state.key_offsets,
            dim=-1,
            eps=1e-8,
        )
        route_logits = torch.einsum("bw,bsw->bs", route_key, effective_keys)
        route_logits = route_logits * math.sqrt(self.width)

        occupied_scores = route_logits.masked_fill(~state.occupied, -torch.inf)
        best_occupied = occupied_scores.max(dim=-1).values
        has_occupied = state.occupied.any(dim=-1)
        has_free = (~state.occupied).any(dim=-1)
        reuse = has_occupied & (
            best_occupied
            >= self.reuse_similarity_threshold * math.sqrt(self.width)
        )
        allowed = torch.where(
            reuse.unsqueeze(-1),
            state.occupied,
            torch.where(
                has_free.unsqueeze(-1),
                ~state.occupied,
                torch.ones_like(state.occupied),
            ),
        )
        write_route_logits = route_logits.masked_fill(~allowed, -torch.inf)
        soft_write = torch.softmax(write_route_logits, dim=-1)
        write_slots = write_route_logits.argmax(dim=-1)
        hard_write = F.one_hot(write_slots, self.slots).to(dtype=soft_write.dtype)
        # Forward values are hard one-hot; the soft residual carries a
        # meta-gradient into the learned router.
        write_weights = hard_write + (soft_write - soft_write.detach())
        read_weights = self._top_k_read_weights(route_logits, state.occupied)

        mean_context = torch.einsum(
            "bs,bsw->bw",
            read_weights,
            state.slot_latents,
        )
        evidence_count = torch.einsum(
            "bs,bs->b",
            read_weights,
            state.write_counts.to(dtype=state_embeddings.dtype),
        )
        confidence = 1.0 - torch.exp(-evidence_count)
        plastic_context = mean_context * confidence.unsqueeze(-1)
        public_transition_gate: torch.Tensor | None = None
        public_reader = getattr(self, "public_evidence_reader", None)
        if include_public_evidence:
            if public_reader is None:
                raise RuntimeError(
                    "public evidence was requested without an attached reader"
                )
            if not isinstance(public_reader, PublicEvidenceLatentReader):
                raise TypeError("public_evidence_reader has the wrong type")
            public_codes, public_counts = public_evidence_skill_content(state)
            routed_public = torch.einsum(
                "bs,bsw->bw",
                read_weights,
                public_codes,
            )
            routed_public_count = torch.einsum(
                "bs,bs->b",
                read_weights,
                public_counts,
            )
            public_confidence = 1.0 - torch.exp(-routed_public_count)
            public_context = routed_public
            public_residual, public_transition_gate = public_reader.read_effects(
                public_context,
                plastic_context,
                public_confidence=public_confidence.unsqueeze(-1),
            )
            plastic_context = torch.where(
                (routed_public_count > 0.0).unsqueeze(-1),
                plastic_context + public_residual,
                plastic_context,
            )
        score_bias = self.decode_context_bias(
            state_embeddings,
            safe_candidates,
            plastic_context,
            candidate_mask=mask,
        )
        score_bias = torch.where(mask, score_bias, torch.zeros_like(score_bias))
        if not (
            bool(torch.isfinite(route_query).all().item())
            and bool(torch.isfinite(plastic_context).all().item())
            and bool(torch.isfinite(score_bias).all().item())
        ):
            raise RuntimeError("procedural skill read produced a non-finite value")
        return ProceduralSkillRead(
            state_embeddings=state_embeddings,
            goal_embeddings=goal_embeddings,
            candidate_embeddings=candidates,
            candidate_mask=mask,
            route_key=route_key,
            route_query=route_query,
            read_weights=read_weights,
            route_probabilities=soft_write,
            write_weights=write_weights,
            write_slots=write_slots,
            plastic_context=plastic_context,
            evidence_count=evidence_count,
            score_bias=score_bias,
            public_evidence_enabled=include_public_evidence,
            public_transition_gate=public_transition_gate,
        )

    def decode_context_bias(
        self,
        state_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        plastic_context: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode candidate utility from a supplied procedural summary.

        This seam permits oracle-latent diagnostics without exposing any task
        identity to the online learner.  The same shared decoder is used by
        ordinary reads, and a zero summary always returns exact zero.
        """

        if (
            not isinstance(state_embeddings, torch.Tensor)
            or state_embeddings.ndim != 2
            or state_embeddings.shape[-1] != self.width
            or not isinstance(plastic_context, torch.Tensor)
            or plastic_context.shape != state_embeddings.shape
        ):
            raise ValueError("state and plastic context must share shape [batch, width]")
        if not isinstance(candidate_embeddings, torch.Tensor):
            raise TypeError("candidate_embeddings must be a tensor")
        if candidate_embeddings.ndim == 2:
            candidate_embeddings = candidate_embeddings.unsqueeze(0).expand(
                state_embeddings.shape[0], -1, -1
            )
        if (
            candidate_embeddings.ndim != 3
            or candidate_embeddings.shape[0] != state_embeddings.shape[0]
            or candidate_embeddings.shape[1] <= 0
            or candidate_embeddings.shape[2] != self.width
        ):
            raise ValueError("candidate embeddings must have shape [batch, count, width]")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                candidate_embeddings.shape[:2],
                device=candidate_embeddings.device,
                dtype=torch.bool,
            )
        if (
            not isinstance(candidate_mask, torch.Tensor)
            or candidate_mask.shape != candidate_embeddings.shape[:2]
            or candidate_mask.dtype != torch.bool
            or candidate_mask.device != candidate_embeddings.device
        ):
            raise ValueError("candidate_mask must be boolean and match candidates")
        for value in (state_embeddings, candidate_embeddings, plastic_context):
            if (
                not value.is_floating_point()
                or value.device != self.memory.base_y.device
                or value.dtype != self.memory.base_y.dtype
            ):
                raise ValueError("decoder inputs must match memory device and dtype")
        if not (
            bool(torch.isfinite(state_embeddings).all().item())
            and bool(torch.isfinite(plastic_context).all().item())
            and bool(torch.isfinite(candidate_embeddings[candidate_mask]).all().item())
        ):
            raise ValueError("active decoder inputs must be finite")

        candidates = torch.where(
            candidate_mask.unsqueeze(-1),
            candidate_embeddings,
            torch.zeros_like(candidate_embeddings),
        )
        context = plastic_context.unsqueeze(1).expand_as(candidates)
        states = state_embeddings.unsqueeze(1).expand_as(candidates)
        live_features = torch.cat(
            (context, states, candidates, context * candidates, candidates - states),
            dim=-1,
        )
        zero_context = torch.zeros_like(context)
        zero_features = torch.cat(
            (
                zero_context,
                states,
                candidates,
                zero_context,
                candidates - states,
            ),
            dim=-1,
        )
        residual = self.utility_decoder(live_features).squeeze(-1) - (
            self.utility_decoder(zero_features).squeeze(-1)
        )
        score_bias = self.score_limit * torch.tanh(residual)
        return torch.where(candidate_mask, score_bias, torch.zeros_like(score_bias))

    def propose_feedback(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        attempted_indices: torch.Tensor,
        reward: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: ProceduralSkillState,
        candidate_mask: torch.Tensor | None = None,
        structural_context: torch.Tensor | None = None,
        outcome_direction_basis: torch.Tensor | None = None,
        public_evidence: torch.Tensor | None = None,
        include_public_evidence: bool = False,
    ) -> ProceduralSkillProposal:
        """Propose one internally routed write without committing it."""

        read = self.read(
            state_embeddings,
            goal_embeddings,
            candidate_embeddings,
            state=state,
            candidate_mask=candidate_mask,
            include_public_evidence=include_public_evidence,
        )
        batch = state.batch_size
        candidates = read.candidate_embeddings
        if (
            not isinstance(attempted_indices, torch.Tensor)
            or attempted_indices.shape != (batch,)
            or attempted_indices.dtype != torch.long
            or attempted_indices.device != candidates.device
        ):
            raise ValueError("attempted_indices must be torch.long with shape [batch]")
        if bool(
            ((attempted_indices < 0) | (attempted_indices >= candidates.shape[1]))
            .any()
            .item()
        ):
            raise ValueError("attempted candidate index is outside the candidate set")
        batch_indices = torch.arange(batch, device=candidates.device)
        if not bool(read.candidate_mask[batch_indices, attempted_indices].all().item()):
            raise ValueError("attempted candidate must be active")
        rewards = _validate_rewards(reward, batch, candidates)
        if (
            not isinstance(base_logits, torch.Tensor)
            or base_logits.shape != read.score_bias.shape
            or base_logits.device != read.score_bias.device
            or base_logits.dtype != read.score_bias.dtype
        ):
            raise ValueError("base_logits must match the candidate score shape")
        if not bool(torch.isfinite(base_logits[read.candidate_mask]).all().item()):
            raise ValueError("active base logits must be finite")
        if structural_context is None:
            structural_context = torch.zeros_like(read.state_embeddings)
        if (
            not isinstance(structural_context, torch.Tensor)
            or structural_context.shape != read.state_embeddings.shape
            or structural_context.device != read.state_embeddings.device
            or structural_context.dtype != read.state_embeddings.dtype
            or not bool(torch.isfinite(structural_context).all().item())
        ):
            raise ValueError(
                "structural_context must be finite and match state embeddings"
            )
        attempted = candidates[batch_indices, attempted_indices]
        feedback_inputs = torch.cat(
            (
                read.state_embeddings,
                attempted,
                structural_context,
            ),
            dim=-1,
        )
        # Preserve the observation in one channel and put only its learned
        # reward covariance in the other.  This generic contextual-bandit
        # statistic contains no hidden procedure or domain-specific solver.
        base_content = torch.tanh(self.feedback_encoder(feedback_inputs))
        if outcome_direction_basis is None:
            base_direction = torch.tanh(
                self.feedback_direction_encoder(feedback_inputs)
            )
        else:
            if (
                not isinstance(outcome_direction_basis, torch.Tensor)
                or outcome_direction_basis.shape
                != (batch, self.evidence_outcome_width)
                or outcome_direction_basis.device != candidates.device
                or outcome_direction_basis.dtype != candidates.dtype
                or not bool(torch.isfinite(outcome_direction_basis).all().item())
                or bool((outcome_direction_basis.abs() > 1.0).any().item())
            ):
                raise ValueError(
                    "outcome_direction_basis must be finite, bounded, and "
                    "have shape [batch, evidence_outcome_width]"
                )
            # A caller may supply a learner-derived canonical action basis.
            # Reward is still applied only below; no target or counterfactual
            # outcome enters this interface.
            # This is pre-reward evidence supplied by the caller, never a
            # trainable path into whatever produced the child policies.
            base_direction = outcome_direction_basis.detach()
        content_residual = torch.zeros_like(base_content)
        direction_residual = torch.zeros_like(base_direction)
        public_reader = getattr(self, "public_evidence_reader", None)
        if public_reader is not None and not isinstance(
            public_reader,
            PublicEvidenceLatentReader,
        ):
            raise TypeError("public_evidence_reader has the wrong type")
        if public_evidence is not None:
            if (
                not isinstance(public_evidence, torch.Tensor)
                or public_evidence.shape != (batch, self.width)
                or public_evidence.device != candidates.device
                or public_evidence.dtype != candidates.dtype
                or not bool(torch.isfinite(public_evidence).all().item())
                or bool((public_evidence.abs() > 1.0 + 1.0e-6).any().item())
            ):
                raise ValueError(
                    "public_evidence must be finite, bounded, and have shape "
                    "[batch, width]"
                )
            evidence_writer = getattr(self, "public_evidence_writer", None)
            if evidence_writer is not None and public_reader is not None:
                raise RuntimeError(
                    "residual writer and public latent reader cannot be active together"
                )
            if evidence_writer is None and public_reader is None:
                if bool(public_evidence.count_nonzero().item()):
                    raise ValueError(
                        "nonzero public evidence requires an attached learned interface"
                    )
            elif not isinstance(evidence_writer, PublicEvidenceResidualWriter):
                if evidence_writer is not None:
                    raise TypeError("public_evidence_writer has the wrong type")
            if evidence_writer is not None:
                content_residual, direction_residual = evidence_writer(
                    public_evidence,
                    base_content,
                    base_direction,
                )
        # A zero residual must preserve the retained writer bit-for-bit.  Each
        # additive residual is independently bounded by the optional bridge.
        content = base_content + content_residual
        direction_basis = base_direction + direction_residual
        signed_outcome = 2.0 * rewards - 1.0
        feedback_event = torch.cat(
            (
                content,
                signed_outcome.unsqueeze(-1) * direction_basis,
            ),
            dim=-1,
        )
        # The recurrent SRWM channel is intentionally never called.  A learned
        # public reader may reserve beta channel zero for one width-sized code
        # per slot and one scalar in channel one for its own running-mean count.
        # Every other compatibility value remains untouched.
        next_fast_values = {
            name: getattr(state.fast_weights, name)
            for name in _FAST_NAMES
        }
        hard_slots = F.one_hot(read.write_slots, self.slots).to(dtype=torch.bool)
        if public_reader is not None and public_evidence is not None:
            public_gate = public_evidence.detach().abs().sum(dim=-1) > 0.0
            if bool(public_gate.any().item()):
                public_codes, public_counts = public_evidence_skill_content(state)
                if bool((public_counts < 0.0).any().item()) or not torch.equal(
                    public_counts,
                    public_counts.round(),
                ):
                    raise RuntimeError("public evidence count storage is invalid")
                selected_public = hard_slots & public_gate.unsqueeze(-1)
                if bool(
                    (
                        selected_public
                        & (public_counts >= float(2**24 - 1))
                    ).any().item()
                ):
                    raise RuntimeError("public evidence count exhausted exact capacity")
                proposed_public_means = public_codes + (
                    public_evidence.unsqueeze(1) - public_codes
                ) / (public_counts + 1.0).unsqueeze(-1)
                next_public_codes = torch.where(
                    selected_public.unsqueeze(-1),
                    proposed_public_means,
                    public_codes,
                )
                next_public_counts = public_counts + selected_public.to(
                    dtype=public_counts.dtype
                )

                beta = state.fast_weights.delta_beta.reshape(
                    batch,
                    self.slots,
                    *state.fast_weights.delta_beta.shape[1:],
                )
                channels = list(torch.unbind(beta, dim=-1))
                channels[_PUBLIC_CODE_BETA_CHANNEL] = next_public_codes.reshape_as(
                    channels[_PUBLIC_CODE_BETA_CHANNEL]
                )
                count_cells = channels[_PUBLIC_COUNT_BETA_CHANNEL].reshape(
                    batch,
                    self.slots,
                    self.width,
                )
                count_basis = F.one_hot(
                    torch.tensor(
                        _PUBLIC_COUNT_FLAT_INDEX,
                        device=count_cells.device,
                        dtype=torch.long,
                    ),
                    self.width,
                ).to(dtype=count_cells.dtype)
                next_count_cells = count_cells + (
                    next_public_counts - public_counts
                ).unsqueeze(-1) * count_basis.reshape(1, 1, -1)
                channels[_PUBLIC_COUNT_BETA_CHANNEL] = next_count_cells.reshape_as(
                    channels[_PUBLIC_COUNT_BETA_CHANNEL]
                )
                next_fast_values["delta_beta"] = torch.stack(
                    channels,
                    dim=-1,
                ).reshape_as(state.fast_weights.delta_beta)

        anchors = F.normalize(self.slot_anchors, dim=-1, eps=1e-8)
        current_keys = anchors.unsqueeze(0) + state.key_offsets
        proposed_keys = read.route_key.unsqueeze(1).expand_as(current_keys)
        proposed_offsets = proposed_keys - anchors.unsqueeze(0)
        routing_weights = read.write_weights
        next_key_offsets = state.key_offsets + routing_weights.unsqueeze(-1) * (
            proposed_offsets - state.key_offsets
        )

        old_counts = state.write_counts.to(dtype=state.slot_latents.dtype)
        proposed_means = state.slot_latents + (
            feedback_event.unsqueeze(1) - state.slot_latents
        ) / (old_counts + 1.0).unsqueeze(-1)
        next_slot_latents = state.slot_latents + read.write_weights.unsqueeze(-1) * (
            proposed_means - state.slot_latents
        )

        occupied = state.occupied | hard_slots
        write_counts = state.write_counts + hard_slots.to(dtype=torch.long)
        candidate_state = ProceduralSkillState(
            fast_weights=SelfReferentialState(**next_fast_values),
            slot_latents=next_slot_latents,
            key_offsets=next_key_offsets,
            occupied=occupied,
            write_counts=write_counts,
        )
        delta_norm = self._delta_norm(state, candidate_state)
        return ProceduralSkillProposal(
            incoming_state=state,
            candidate_state=candidate_state,
            read=read,
            attempted_indices=attempted_indices,
            rewards=rewards,
            # Bind admission to the exact pre-write evidence.  A clone keeps
            # autograd connectivity while preventing later caller mutation of
            # the original tensor from changing the transaction.
            base_logits=base_logits.clone(),
            feedback_event=feedback_event,
            write_slots=read.write_slots,
            delta_norm=delta_norm,
        )

    def admit_feedback(
        self,
        proposal: ProceduralSkillProposal,
        *,
        minimum_improvement: float = 0.0,
    ) -> ProceduralSkillWrite:
        """Commit only rows that immediately better fit their scalar outcome."""

        if not isinstance(proposal, ProceduralSkillProposal):
            raise TypeError("proposal must be a ProceduralSkillProposal")
        if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be finite and nonnegative")
        read = proposal.read
        base_logits = proposal.base_logits
        after_read = self.read(
            read.state_embeddings,
            read.goal_embeddings,
            read.candidate_embeddings,
            state=proposal.candidate_state,
            candidate_mask=read.candidate_mask,
            include_public_evidence=read.public_evidence_enabled,
        )
        before_loss, viable = _feedback_consistency_loss(
            base_logits + read.score_bias,
            read.candidate_mask,
            proposal.attempted_indices,
            proposal.rewards,
        )
        after_loss, after_viable = _feedback_consistency_loss(
            base_logits + after_read.score_bias,
            read.candidate_mask,
            proposal.attempted_indices,
            proposal.rewards,
        )
        viable = viable & after_viable
        accepted = (
            viable
            & torch.isfinite(before_loss)
            & torch.isfinite(after_loss)
            & (proposal.delta_norm > 0.0)
            & (after_loss + minimum_improvement < before_loss)
        )
        if bool(accepted.all().item()):
            committed = proposal.candidate_state
        elif not bool(accepted.any().item()):
            committed = proposal.incoming_state
        else:
            committed = _select_state_rows(
                proposal.incoming_state,
                proposal.candidate_state,
                accepted,
            )
        return ProceduralSkillWrite(
            state=committed,
            accepted=accepted,
            write_slots=proposal.write_slots,
            delta_norm=torch.where(
                accepted,
                proposal.delta_norm,
                torch.zeros_like(proposal.delta_norm),
            ),
            before_loss=before_loss,
            after_loss=after_loss,
        )

    def commit_bounded_feedback(
        self,
        proposal: ProceduralSkillProposal,
        *,
        minimum_effect: float = 0.0,
    ) -> ProceduralSkillWrite:
        """Commit a finite, bounded evidence-state update.

        Unlike ``admit_feedback``, this transaction does not demand that one
        update improve or even alter the same example immediately.  A valid
        observation may only become informative together with later evidence.
        The candidate remains locally reversible and is rejected when its
        bounded competence-state transition is non-finite or empty.
        """

        if not isinstance(proposal, ProceduralSkillProposal):
            raise TypeError("proposal must be a ProceduralSkillProposal")
        if not math.isfinite(minimum_effect) or minimum_effect < 0.0:
            raise ValueError("minimum_effect must be finite and nonnegative")
        read = proposal.read
        after_read = self.read(
            read.state_embeddings,
            read.goal_embeddings,
            read.candidate_embeddings,
            state=proposal.candidate_state,
            candidate_mask=read.candidate_mask,
            include_public_evidence=read.public_evidence_enabled,
        )
        before_loss, viable = _feedback_consistency_loss(
            proposal.base_logits + read.score_bias,
            read.candidate_mask,
            proposal.attempted_indices,
            proposal.rewards,
        )
        after_loss, after_viable = _feedback_consistency_loss(
            proposal.base_logits + after_read.score_bias,
            read.candidate_mask,
            proposal.attempted_indices,
            proposal.rewards,
        )
        accepted = (
            viable
            & after_viable
            & torch.isfinite(before_loss)
            & torch.isfinite(after_loss)
            & torch.isfinite(proposal.delta_norm)
            & (proposal.delta_norm > minimum_effect)
        )
        if bool(accepted.all().item()):
            committed = proposal.candidate_state
        elif not bool(accepted.any().item()):
            committed = proposal.incoming_state
        else:
            committed = _select_state_rows(
                proposal.incoming_state,
                proposal.candidate_state,
                accepted,
            )
        return ProceduralSkillWrite(
            state=committed,
            accepted=accepted,
            write_slots=proposal.write_slots,
            delta_norm=torch.where(
                accepted,
                proposal.delta_norm,
                torch.zeros_like(proposal.delta_norm),
            ),
            before_loss=before_loss,
            after_loss=after_loss,
        )

    def incorporate_feedback(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        attempted_indices: torch.Tensor,
        reward: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: ProceduralSkillState,
        candidate_mask: torch.Tensor | None = None,
        minimum_effect: float = 0.0,
        structural_context: torch.Tensor | None = None,
        outcome_direction_basis: torch.Tensor | None = None,
        public_evidence: torch.Tensor | None = None,
        include_public_evidence: bool = False,
    ) -> ProceduralSkillWrite:
        """Propose and transactionally retain one bounded evidence update."""

        proposal = self.propose_feedback(
            state_embeddings,
            goal_embeddings,
            candidate_embeddings,
            attempted_indices,
            reward,
            base_logits,
            state=state,
            candidate_mask=candidate_mask,
            structural_context=structural_context,
            outcome_direction_basis=outcome_direction_basis,
            public_evidence=public_evidence,
            include_public_evidence=include_public_evidence,
        )
        return self.commit_bounded_feedback(
            proposal,
            minimum_effect=minimum_effect,
        )

    def _validate_read_inputs(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None,
        state: ProceduralSkillState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state, ProceduralSkillState):
            raise TypeError("state must be a ProceduralSkillState")
        self._validate_state_topology(state)
        if (
            not isinstance(states, torch.Tensor)
            or states.ndim != 2
            or states.shape != (state.batch_size, self.width)
            or not isinstance(goals, torch.Tensor)
            or goals.shape != states.shape
        ):
            raise ValueError("states and goals must share shape [batch, width]")
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("candidates must be a tensor")
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(state.batch_size, -1, -1)
        if (
            candidates.ndim != 3
            or candidates.shape[0] != state.batch_size
            or candidates.shape[1] <= 0
            or candidates.shape[2] != self.width
        ):
            raise ValueError(
                "candidates must have shape [count, width] or [batch, count, width]"
            )
        reference = self.memory.base_y
        for value in (states, goals, candidates):
            if not value.is_floating_point():
                raise ValueError("structural inputs must be floating point")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("structural inputs must match the memory device and dtype")
        if mask is None:
            mask = torch.ones(
                candidates.shape[:2],
                device=candidates.device,
                dtype=torch.bool,
            )
        elif mask.shape == (candidates.shape[1],):
            mask = mask.unsqueeze(0).expand(candidates.shape[0], -1)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != candidates.shape[:2]
            or mask.dtype != torch.bool
            or mask.device != candidates.device
        ):
            raise ValueError("candidate_mask must be boolean and match candidates")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every batch row requires one active candidate")
        if not (
            bool(torch.isfinite(states).all().item())
            and bool(torch.isfinite(goals).all().item())
            and bool(torch.isfinite(candidates[mask]).all().item())
        ):
            raise ValueError("active structural inputs must be finite")
        return candidates, mask

    def _validate_state_topology(self, state: ProceduralSkillState) -> None:
        if (
            state.slot_count != self.slots
            or state.width != self.width
            or state.key_offsets.device != self.memory.base_y.device
            or state.key_offsets.dtype != self.memory.base_y.dtype
        ):
            raise ValueError("skill state topology does not match this memory")
        expected_batch = state.batch_size * self.slots
        if state.fast_weights.batch_size != expected_batch:
            raise ValueError("flattened fast-weight slots have the wrong batch size")

    def _top_k_read_weights(
        self,
        route_logits: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        matching = occupied & (
            route_logits
            >= self.reuse_similarity_threshold * math.sqrt(self.width)
        )
        active_rows = matching.any(dim=-1)
        masked = route_logits.masked_fill(~matching, -torch.inf)
        safe = torch.where(active_rows.unsqueeze(-1), masked, torch.zeros_like(masked))
        probabilities = torch.softmax(safe, dim=-1)
        _, indices = torch.topk(safe, k=self.read_top_k, dim=-1)
        selected = torch.zeros_like(occupied)
        selected.scatter_(1, indices, True)
        selected = selected & matching
        weights = probabilities * selected.to(dtype=probabilities.dtype)
        total = weights.sum(dim=-1, keepdim=True)
        return torch.where(
            total > 0.0,
            weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )

    @staticmethod
    def _delta_norm(
        before: ProceduralSkillState,
        after: ProceduralSkillState,
    ) -> torch.Tensor:
        batch = before.batch_size
        squared = (after.key_offsets - before.key_offsets).square().flatten(1).sum(1)
        squared = squared + (
            after.slot_latents - before.slot_latents
        ).square().flatten(1).sum(1)
        squared = squared + (
            after.write_counts - before.write_counts
        ).to(dtype=before.slot_latents.dtype).square().flatten(1).sum(1)
        for name in _FAST_NAMES:
            difference = getattr(after.fast_weights, name) - getattr(
                before.fast_weights,
                name,
            )
            squared = squared + difference.reshape(batch, -1).square().sum(1)
        return torch.sqrt(squared)


def _validate_state_structure(state: ProceduralSkillState) -> None:
    if not isinstance(state.fast_weights, SelfReferentialState):
        raise TypeError("fast_weights must be a SelfReferentialState")
    if not isinstance(state.key_offsets, torch.Tensor) or state.key_offsets.ndim != 3:
        raise ValueError("key_offsets must have shape [batch, slots, width]")
    batch, slots, width = state.key_offsets.shape
    if batch <= 0 or slots <= 0 or width <= 0:
        raise ValueError("skill-state dimensions must be positive")
    if not state.key_offsets.is_floating_point():
        raise ValueError("key_offsets must be floating point")
    if (
        not isinstance(state.slot_latents, torch.Tensor)
        or state.slot_latents.shape != (batch, slots, width)
        or not state.slot_latents.is_floating_point()
    ):
        raise ValueError(
            "slot_latents must be floating point with shape [batch, slots, width]"
        )
    if (
        not isinstance(state.occupied, torch.Tensor)
        or state.occupied.shape != (batch, slots)
        or state.occupied.dtype != torch.bool
    ):
        raise ValueError("occupied must be boolean with shape [batch, slots]")
    if (
        not isinstance(state.write_counts, torch.Tensor)
        or state.write_counts.shape != (batch, slots)
        or state.write_counts.dtype != torch.long
    ):
        raise ValueError("write_counts must be torch.long with shape [batch, slots]")
    matrices = (
        state.fast_weights.delta_y,
        state.fast_weights.delta_q,
        state.fast_weights.delta_k,
    )
    tensors = (*matrices, state.fast_weights.delta_beta)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 4 for value in tensors):
        raise ValueError("every fast-weight tensor must be rank four")
    if any(value.shape[0] != batch * slots for value in tensors):
        raise ValueError("fast-weight batch must flatten batch and slots")
    if any(value.shape != matrices[0].shape for value in matrices[1:]):
        raise ValueError("y, q, and k fast-weight shapes must match")
    flat_batch, heads, input_width, output_width = matrices[0].shape
    if (
        flat_batch != batch * slots
        or heads <= 0
        or input_width <= 0
        or input_width != output_width
    ):
        raise ValueError("fast-weight matrices must be nonempty and square")
    if state.fast_weights.delta_beta.shape != (
        flat_batch,
        heads,
        input_width,
        4,
    ):
        raise ValueError("beta fast weights must have shape [batch*slots, heads, width, 4]")
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("fast weights must be floating point")
    if any(
        value.device != state.key_offsets.device
        or value.dtype != state.key_offsets.dtype
        for value in tensors
    ):
        raise ValueError("fast weights and key offsets must share device and dtype")
    if (
        state.slot_latents.device != state.key_offsets.device
        or state.slot_latents.dtype != state.key_offsets.dtype
    ):
        raise ValueError("slot latents and key offsets must share device and dtype")
    if (
        state.occupied.device != state.key_offsets.device
        or state.write_counts.device != state.key_offsets.device
    ):
        raise ValueError("all skill-state tensors must share one device")
    if bool((state.write_counts < 0).any().item()):
        raise ValueError("write_counts must be nonnegative")
    if not torch.equal(state.occupied, state.write_counts > 0):
        raise ValueError("occupied slots must exactly match positive write counts")
    if not (
        bool(torch.isfinite(state.key_offsets).all().item())
        and bool(torch.isfinite(state.slot_latents).all().item())
        and all(bool(torch.isfinite(value).all().item()) for value in tensors)
    ):
        raise ValueError("skill-state floating tensors must be finite")


def _validate_rewards(
    reward: torch.Tensor,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(reward, torch.Tensor) or reward.shape != (batch_size,):
        raise ValueError("reward must have shape [batch]")
    numeric = reward.detach().to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(numeric).all().item()):
        raise ValueError("reward must be finite")
    if bool(((numeric < 0.0) | (numeric > 1.0)).any().item()):
        raise ValueError("reward must be between zero and one")
    return numeric


def _feedback_consistency_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    selected: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = logits.shape[0]
    rows = torch.arange(batch, device=logits.device)
    selected_logits = logits[rows, selected]
    # The observed score is this candidate's scalar utility.  A categorical
    # selected-vs-rest log-odds would make the prediction depend on the number
    # of unrelated alternatives (uniform over 120 predicts 1/120, not 1/2).
    viable = mask[rows, selected]
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, rewards, reduction="none"
    )
    return loss, viable


def _select_state_rows(
    before: ProceduralSkillState,
    after: ProceduralSkillState,
    select_after: torch.Tensor,
) -> ProceduralSkillState:
    batch = before.batch_size
    slots = before.slot_count
    fast_values: dict[str, torch.Tensor] = {}
    for name in _FAST_NAMES:
        old = getattr(before.fast_weights, name).reshape(
            batch,
            slots,
            *getattr(before.fast_weights, name).shape[1:],
        )
        new = getattr(after.fast_weights, name).reshape_as(old)
        condition = select_after.reshape(batch, *([1] * (old.ndim - 1)))
        fast_values[name] = torch.where(condition, new, old).reshape_as(
            getattr(before.fast_weights, name)
        )
    floating_condition = select_after[:, None, None]
    discrete_condition = select_after[:, None]
    return ProceduralSkillState(
        fast_weights=SelfReferentialState(**fast_values),
        slot_latents=torch.where(
            floating_condition,
            after.slot_latents,
            before.slot_latents,
        ),
        key_offsets=torch.where(
            floating_condition,
            after.key_offsets,
            before.key_offsets,
        ),
        occupied=torch.where(
            discrete_condition,
            after.occupied,
            before.occupied,
        ),
        write_counts=torch.where(
            discrete_condition,
            after.write_counts,
            before.write_counts,
        ),
    )


__all__ = [
    "PublicEvidenceLatentReader",
    "PublicEvidenceResidualWriter",
    "ProceduralSkillProposal",
    "ProceduralSkillRead",
    "ProceduralSkillState",
    "ProceduralSkillWrite",
    "RoutedProceduralMemory",
    "differentiable_zero_public_evidence_skill_content",
    "permute_procedural_skill_slots",
    "public_evidence_skill_content",
    "procedural_skill_state_digest",
    "restore_procedural_skill_state",
    "snapshot_procedural_skill_state",
    "zero_public_evidence_skill_content",
    "zero_procedural_skill_content",
]
