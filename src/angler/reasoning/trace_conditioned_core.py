"""Plastic procedural memory that records the procedure which earned feedback."""

from __future__ import annotations

import math

import torch
from torch import nn

from .scalable_procedural_core import (
    PlasticProcedureState,
    ProceduralCoreConfig,
    ProceduralCoreOutput,
    ScalableProceduralCore,
)


class TraceConditionedProceduralCore(ScalableProceduralCore):
    """Extend the scalable core with ordered, action-conditioned feedback.

    The inherited outcome-only writer is preserved for checkpoint compatibility.
    This successor writer additionally observes the ordered action embeddings
    that produced the externally supplied outcome.  It learns what to retain;
    it does not contain action labels, a solver, or a task-specific lookup.
    """

    def __init__(
        self,
        config: ProceduralCoreConfig,
        *,
        maximum_trace_steps: int,
    ) -> None:
        super().__init__(config)
        if type(maximum_trace_steps) is not int or maximum_trace_steps <= 0:
            raise ValueError("maximum_trace_steps must be a positive integer")
        self.maximum_trace_steps = maximum_trace_steps
        width = config.model_width
        self.trace_projection = nn.Sequential(
            nn.LayerNorm(config.content_width),
            nn.Linear(config.content_width, width),
            nn.GELU(),
        )
        self.trace_positions = nn.Parameter(torch.empty(maximum_trace_steps, width))
        nn.init.normal_(self.trace_positions, std=1.0 / math.sqrt(width))
        self.trace_recurrent = nn.GRU(width, width, batch_first=True)
        write_width = width * 3 + 1
        self.trace_write_key = nn.Sequential(nn.Linear(write_width, width), nn.Tanh())
        self.trace_write_value = nn.Sequential(nn.Linear(write_width, width), nn.Tanh())
        self.trace_write_gate = nn.Sequential(nn.Linear(write_width, 1), nn.Sigmoid())

    def encode_procedure_trace(
        self,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one or more ordered, prefix-masked executed procedures."""

        reference = next(self.parameters())
        if (
            action_features.ndim != 3
            or action_features.shape[1:] != (
                self.maximum_trace_steps,
                self.config.content_width,
            )
        ):
            raise ValueError(
                "action_features must be [batch, maximum_trace_steps, content_width]"
            )
        if action_mask.dtype is not torch.bool or action_mask.shape != action_features.shape[:2]:
            raise ValueError("action_mask must be boolean [batch, maximum_trace_steps]")
        if not bool(action_mask.any(dim=1).all().item()):
            raise ValueError("every procedure trace must contain at least one action")
        if bool((action_mask & ((~action_mask).cumsum(dim=1) > 0)).any().item()):
            raise ValueError("action_mask must expose one contiguous action prefix")
        if action_features.device != reference.device or action_features.dtype != reference.dtype:
            raise ValueError("action_features must match the core device and dtype")
        if action_mask.device != reference.device:
            raise ValueError("action_mask must match the core device")
        if not bool(torch.isfinite(action_features).all().item()):
            raise ValueError("action_features must be finite")

        projected = self.trace_projection(action_features.detach())
        projected = projected + self.trace_positions.unsqueeze(0)
        projected = projected * action_mask.unsqueeze(-1)
        lengths = action_mask.sum(dim=1).to(device="cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.trace_recurrent(packed)
        return hidden[-1]

    def apply_procedural_feedback(
        self,
        state: PlasticProcedureState,
        output: ProceduralCoreOutput,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        outcome: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> PlasticProcedureState:
        """Write the observed ordered procedure and its external outcome."""

        self._validate_state(state)
        if output.query_state.shape[0] != 1 or output.procedure_summary.shape[0] != 1:
            raise ValueError("persistent procedural feedback requires one experience at a time")
        if action_features.shape[0] != 1:
            raise ValueError("persistent procedural feedback requires one action trace")
        if outcome.shape != (1,) or outcome.dtype != output.query_state.dtype:
            raise ValueError("outcome must be one floating-point value matching the core dtype")
        if outcome.device != output.query_state.device or not bool(
            ((outcome == 1) | (outcome == -1)).all().item()
        ):
            raise ValueError("outcome must be +1 success or -1 failure on the core device")

        trace = self.encode_procedure_trace(action_features, action_mask)
        write_input = torch.cat(
            (output.query_state, output.procedure_summary, trace, outcome.unsqueeze(1)),
            dim=-1,
        )
        key = self.trace_write_key(write_input).squeeze(0)
        value = self.trace_write_value(write_input).squeeze(0)
        gate = self.trace_write_gate(write_input).squeeze()
        similarity = torch.mv(state.keys, key) / math.sqrt(self.config.model_width)
        write_weights = torch.softmax(similarity - 4.0 * state.strengths, dim=0)
        rates = (gate * write_weights).clamp(0.0, 1.0)
        keys = (1.0 - rates.unsqueeze(1)) * state.keys + rates.unsqueeze(1) * key
        values = (1.0 - rates.unsqueeze(1)) * state.values + rates.unsqueeze(1) * value
        strengths = (state.strengths + rates * (1.0 - state.strengths)).clamp(0.0, 1.0)
        updated = PlasticProcedureState(
            keys=keys,
            values=values,
            strengths=strengths,
            step=state.step + 1,
        )
        return updated.detached_clone() if detach_state else updated


__all__ = ["TraceConditionedProceduralCore"]
