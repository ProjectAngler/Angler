"""Resource-scalable latent procedural core for Project Angler.

The core is not a language model and does not predict answer tokens.  It reads
detached foundation representations, retrieved evidence, Moving Origin
coordinates, and a bounded plastic state.  It produces evidence attribution,
latent procedure slots, and prefix embeddings that a frozen language model can
consume.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ProceduralCoreConfig:
    content_width: int
    temporal_width: int
    model_width: int
    depth: int
    heads: int
    procedure_tokens: int
    plastic_slots: int
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    tier: str = "custom"

    def __post_init__(self) -> None:
        for name in (
            "content_width",
            "temporal_width",
            "model_width",
            "depth",
            "heads",
            "procedure_tokens",
            "plastic_slots",
            "feedforward_multiplier",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_width % self.heads:
            raise ValueError("model_width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if type(self.tier) is not str or not self.tier:
            raise ValueError("tier must be non-empty text")


@dataclass(frozen=True, slots=True)
class PlasticProcedureState:
    keys: torch.Tensor
    values: torch.Tensor
    strengths: torch.Tensor
    step: int

    @property
    def bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in (self.keys, self.values, self.strengths))

    def detached_clone(self) -> "PlasticProcedureState":
        return PlasticProcedureState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            strengths=self.strengths.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class ProceduralCoreOutput:
    candidate_scores: torch.Tensor
    candidate_weights: torch.Tensor
    procedure_slots: torch.Tensor
    qwen_prefix: torch.Tensor
    query_state: torch.Tensor
    procedure_summary: torch.Tensor
    plastic_attention: torch.Tensor


class _ProceduralBlock(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.evidence_norm = nn.LayerNorm(width)
        self.evidence_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.memory_norm = nn.LayerNorm(width)
        self.memory_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, width),
        )

    def forward(
        self,
        slots: torch.Tensor,
        evidence: torch.Tensor,
        evidence_padding: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_padding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.slot_norm(slots)
        delta, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + delta
        normalized = self.evidence_norm(slots)
        delta, _ = self.evidence_attention(
            normalized,
            evidence,
            evidence,
            key_padding_mask=evidence_padding,
            need_weights=False,
        )
        slots = slots + delta
        normalized = self.memory_norm(slots)
        delta, _ = self.memory_attention(
            normalized,
            memory_keys,
            memory_values,
            key_padding_mask=memory_padding,
            need_weights=False,
        )
        slots = slots + delta
        return slots + self.feedforward(self.feedforward_norm(slots))


class ScalableProceduralCore(nn.Module):
    """Multi-round evidence reasoner with fixed-capacity learned plastic memory."""

    def __init__(self, config: ProceduralCoreConfig) -> None:
        super().__init__()
        self.config = config
        width = config.model_width
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.content_width),
            nn.Linear(config.content_width, width),
            nn.GELU(),
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(config.content_width),
            nn.Linear(config.content_width, width),
            nn.GELU(),
        )
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(config.temporal_width),
            nn.Linear(config.temporal_width, width),
            nn.GELU(),
        )
        self.initial_slots = nn.Parameter(torch.empty(config.procedure_tokens, width))
        self.null_memory = nn.Parameter(torch.zeros(1, width))
        nn.init.normal_(self.initial_slots, std=1.0 / math.sqrt(width))
        self.blocks = nn.ModuleList(
            _ProceduralBlock(
                width,
                config.heads,
                width * config.feedforward_multiplier,
                config.dropout,
            )
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width)
        self.candidate_score = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.prefix_projection = nn.Linear(width, config.content_width)
        self.write_key = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.Tanh())
        self.write_value = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.Tanh())
        self.write_gate = nn.Sequential(nn.Linear(width * 2 + 1, 1), nn.Sigmoid())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_plastic_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PlasticProcedureState:
        reference = next(self.parameters())
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        return PlasticProcedureState(
            keys=torch.zeros(self.config.plastic_slots, self.config.model_width, device=device, dtype=dtype),
            values=torch.zeros(self.config.plastic_slots, self.config.model_width, device=device, dtype=dtype),
            strengths=torch.zeros(self.config.plastic_slots, device=device, dtype=dtype),
            step=0,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        plastic_state: PlasticProcedureState | None = None,
    ) -> ProceduralCoreOutput:
        self._validate_inputs(query_features, candidate_features, temporal_features, candidate_mask)
        state = plastic_state or self.initial_plastic_state()
        self._validate_state(state)
        query_features = query_features.detach()
        candidate_features = candidate_features.detach()
        temporal_features = temporal_features.detach()
        query = self.query_projection(query_features)
        evidence = self.evidence_projection(candidate_features) + self.temporal_projection(temporal_features)
        batch = query.shape[0]
        slots = self.initial_slots.unsqueeze(0).expand(batch, -1, -1) + query.unsqueeze(1)
        memory_keys = torch.cat((self.null_memory, state.keys), dim=0).unsqueeze(0).expand(batch, -1, -1)
        memory_values = torch.cat((self.null_memory, state.values), dim=0).unsqueeze(0).expand(batch, -1, -1)
        active = state.strengths > 1.0e-6
        memory_padding = torch.cat(
            (torch.zeros(1, device=active.device, dtype=torch.bool), ~active), dim=0
        ).unsqueeze(0).expand(batch, -1)
        evidence_padding = ~candidate_mask
        for block in self.blocks:
            slots = block(
                slots,
                evidence,
                evidence_padding,
                memory_keys,
                memory_values,
                memory_padding,
            )
        slots = self.final_norm(slots)
        summary = slots.mean(dim=1)
        expanded = summary.unsqueeze(1).expand_as(evidence)
        candidate_scores = self.candidate_score(
            torch.cat((evidence, expanded, evidence * expanded), dim=-1)
        ).squeeze(-1)
        candidate_scores = candidate_scores.masked_fill(~candidate_mask, -torch.inf)
        candidate_weights = torch.softmax(candidate_scores, dim=-1)
        if bool(active.any().item()):
            similarities = torch.einsum("bw,sw->bs", summary, state.keys) / math.sqrt(self.config.model_width)
            similarities = similarities.masked_fill(~active.unsqueeze(0), -torch.inf)
            plastic_attention = torch.softmax(similarities, dim=-1)
        else:
            plastic_attention = torch.zeros(batch, self.config.plastic_slots, device=summary.device, dtype=summary.dtype)
        return ProceduralCoreOutput(
            candidate_scores=candidate_scores,
            candidate_weights=candidate_weights,
            procedure_slots=slots,
            qwen_prefix=self.prefix_projection(slots),
            query_state=query,
            procedure_summary=summary,
            plastic_attention=plastic_attention,
        )

    def apply_feedback(
        self,
        state: PlasticProcedureState,
        output: ProceduralCoreOutput,
        outcome: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> PlasticProcedureState:
        self._validate_state(state)
        if output.query_state.shape[0] != 1 or output.procedure_summary.shape[0] != 1:
            raise ValueError("persistent feedback currently requires one experience at a time")
        if outcome.shape != (1,) or outcome.dtype != output.query_state.dtype:
            raise ValueError("outcome must be one floating-point value matching the core dtype")
        if outcome.device != output.query_state.device or not bool(((outcome == 1) | (outcome == -1)).all().item()):
            raise ValueError("outcome must be +1 success or -1 failure on the core device")
        write_input = torch.cat((output.query_state, output.procedure_summary, outcome.unsqueeze(1)), dim=-1)
        key = self.write_key(write_input).squeeze(0)
        value = self.write_value(write_input).squeeze(0)
        gate = self.write_gate(write_input).squeeze()
        similarity = torch.mv(state.keys, key) / math.sqrt(self.config.model_width)
        write_weights = torch.softmax(similarity - 4.0 * state.strengths, dim=0)
        rates = (gate * write_weights).clamp(0.0, 1.0)
        keys = (1.0 - rates.unsqueeze(1)) * state.keys + rates.unsqueeze(1) * key
        values = (1.0 - rates.unsqueeze(1)) * state.values + rates.unsqueeze(1) * value
        strengths = (state.strengths + rates * (1.0 - state.strengths)).clamp(0.0, 1.0)
        updated = PlasticProcedureState(keys=keys, values=values, strengths=strengths, step=state.step + 1)
        return updated.detached_clone() if detach_state else updated

    def _validate_inputs(self, query, candidates, temporal, mask) -> None:
        reference = next(self.parameters())
        if query.ndim != 2 or query.shape[-1] != self.config.content_width:
            raise ValueError("query_features must be [batch, content_width]")
        if candidates.ndim != 3 or candidates.shape != (query.shape[0], candidates.shape[1], self.config.content_width):
            raise ValueError("candidate_features must be [batch, candidates, content_width]")
        if temporal.shape != (query.shape[0], candidates.shape[1], self.config.temporal_width):
            raise ValueError("temporal_features must be [batch, candidates, temporal_width]")
        if mask.dtype is not torch.bool or mask.shape != candidates.shape[:2] or not bool(mask.any(dim=1).all().item()):
            raise ValueError("candidate_mask must expose at least one candidate per row")
        for name, value in (("query", query), ("candidates", candidates), ("temporal", temporal)):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match the core device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if mask.device != reference.device:
            raise ValueError("candidate_mask must match the core device")

    def _validate_state(self, state: PlasticProcedureState) -> None:
        reference = next(self.parameters())
        expected = (self.config.plastic_slots, self.config.model_width)
        if state.keys.shape != expected or state.values.shape != expected:
            raise ValueError("plastic key/value state has the wrong shape")
        if state.strengths.shape != (self.config.plastic_slots,):
            raise ValueError("plastic strength state has the wrong shape")
        if type(state.step) is not int or state.step < 0:
            raise ValueError("plastic state step must be a non-negative integer")
        for value in (state.keys, state.values, state.strengths):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("plastic state must match the core device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("plastic state must be finite")


def procedural_core_config(
    tier: str,
    *,
    content_width: int,
    temporal_width: int,
) -> ProceduralCoreConfig:
    settings = {
        "compact": (256, 4, 8, 8, 16),
        "workstation": (512, 8, 8, 16, 64),
        "dedicated": (768, 12, 12, 32, 128),
    }
    try:
        width, depth, heads, tokens, slots = settings[tier]
    except KeyError as exc:
        raise ValueError(f"unknown procedural-core tier: {tier}") from exc
    return ProceduralCoreConfig(
        content_width=content_width,
        temporal_width=temporal_width,
        model_width=width,
        depth=depth,
        heads=heads,
        procedure_tokens=tokens,
        plastic_slots=slots,
        tier=tier,
    )


def select_procedural_core_tier(
    available_bytes: int,
    *,
    content_width: int,
    temporal_width: int,
    training: bool,
    reserve_fraction: float = 0.25,
) -> ProceduralCoreConfig:
    if type(available_bytes) is not int or available_bytes <= 0:
        raise ValueError("available_bytes must be positive")
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")
    usable = int(available_bytes * (1.0 - reserve_fraction))
    multiplier = 16 if training else 4
    choices = []
    for tier in ("compact", "workstation", "dedicated"):
        config = procedural_core_config(tier, content_width=content_width, temporal_width=temporal_width)
        model = ScalableProceduralCore(config)
        estimate = model.parameter_count * multiplier + config.plastic_slots * config.model_width * 12
        choices.append((config, estimate))
    viable = [config for config, estimate in choices if estimate <= usable]
    if not viable:
        raise RuntimeError("no procedural-core tier fits the declared available bytes")
    return viable[-1]


def plastic_state_digest(state: PlasticProcedureState) -> str:
    digest = hashlib.sha256(b"project-angler.scalable-procedural-state.v1\x00")
    digest.update(state.step.to_bytes(8, "big"))
    for name, value in (("keys", state.keys), ("values", state.values), ("strengths", state.strengths)):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(str(tensor.dtype).encode("ascii") + b"\x00")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


__all__ = [
    "PlasticProcedureState",
    "ProceduralCoreConfig",
    "ProceduralCoreOutput",
    "ScalableProceduralCore",
    "plastic_state_digest",
    "procedural_core_config",
    "select_procedural_core_tier",
]
