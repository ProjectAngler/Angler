"""Action-level procedural memory and learned cross-context correspondence."""

from __future__ import annotations

import math

import torch
from torch import nn

from .candidate_procedure_decoder import CandidateProcedureDecode, CandidateProcedureDecoder
from .scalable_procedural_core import (
    PlasticProcedureState,
    ProceduralCoreConfig,
    ProceduralCoreOutput,
    ScalableProceduralCore,
)


class ActionTraceProceduralCore(ScalableProceduralCore):
    """Write each observed procedure action as an ordered plastic-memory item."""

    def __init__(self, config: ProceduralCoreConfig, *, maximum_trace_steps: int) -> None:
        super().__init__(config)
        if type(maximum_trace_steps) is not int or maximum_trace_steps <= 0:
            raise ValueError("maximum_trace_steps must be a positive integer")
        self.maximum_trace_steps = maximum_trace_steps
        width = config.model_width
        self.action_trace_projection = nn.Sequential(
            nn.LayerNorm(config.content_width),
            nn.Linear(config.content_width, width),
            nn.GELU(),
        )
        self.action_trace_positions = nn.Parameter(torch.empty(maximum_trace_steps, width))
        nn.init.normal_(self.action_trace_positions, std=1.0 / math.sqrt(width))
        write_width = width * 3 + 1
        self.action_write_key = nn.Sequential(nn.Linear(write_width, width), nn.Tanh())
        self.action_write_value = nn.Sequential(nn.Linear(write_width, width), nn.Tanh())
        self.action_write_gate = nn.Sequential(nn.Linear(write_width, 1), nn.Sigmoid())

    def apply_action_trace_feedback(
        self,
        state: PlasticProcedureState,
        output: ProceduralCoreOutput,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        outcome: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> PlasticProcedureState:
        self._validate_state(state)
        reference = next(self.parameters())
        expected = (1, self.maximum_trace_steps, self.config.content_width)
        if action_features.shape != expected:
            raise ValueError("action_features must be [1, maximum_trace_steps, content_width]")
        if action_mask.dtype is not torch.bool or action_mask.shape != expected[:2]:
            raise ValueError("action_mask must be boolean [1, maximum_trace_steps]")
        if not bool(action_mask.any().item()):
            raise ValueError("the procedure trace must contain at least one action")
        if bool((action_mask & ((~action_mask).cumsum(dim=1) > 0)).any().item()):
            raise ValueError("action_mask must expose one contiguous action prefix")
        if action_features.device != reference.device or action_features.dtype != reference.dtype:
            raise ValueError("action_features must match the core device and dtype")
        if action_mask.device != reference.device:
            raise ValueError("action_mask must match the core device")
        if output.query_state.shape[0] != 1 or output.procedure_summary.shape[0] != 1:
            raise ValueError("action-trace feedback requires one experience at a time")
        if outcome.shape != (1,) or outcome.device != reference.device or outcome.dtype != reference.dtype:
            raise ValueError("outcome must match the core device and dtype")
        if not bool(((outcome == 1) | (outcome == -1)).all().item()):
            raise ValueError("outcome must be +1 success or -1 failure")

        actions = self.action_trace_projection(action_features.detach())
        actions = actions + self.action_trace_positions.unsqueeze(0)
        current = state
        for step in range(int(action_mask.sum().item())):
            write_input = torch.cat(
                (
                    output.query_state,
                    output.procedure_summary,
                    actions[:, step],
                    outcome.unsqueeze(1),
                ),
                dim=-1,
            )
            key = self.action_write_key(write_input).squeeze(0)
            value = self.action_write_value(write_input).squeeze(0)
            gate = self.action_write_gate(write_input).squeeze()
            similarity = torch.mv(current.keys, key) / math.sqrt(self.config.model_width)
            weights = torch.softmax(similarity - 4.0 * current.strengths, dim=0)
            rates = (gate * weights).clamp(0.0, 1.0)
            current = PlasticProcedureState(
                keys=(1.0 - rates.unsqueeze(1)) * current.keys + rates.unsqueeze(1) * key,
                values=(1.0 - rates.unsqueeze(1)) * current.values + rates.unsqueeze(1) * value,
                strengths=(current.strengths + rates * (1.0 - current.strengths)).clamp(0.0, 1.0),
                step=state.step,
            )
        updated = PlasticProcedureState(
            keys=current.keys,
            values=current.values,
            strengths=current.strengths,
            step=state.step + 1,
        )
        return updated.detached_clone() if detach_state else updated


class ActionTraceMatcherDecoder(CandidateProcedureDecoder):
    """Match current candidates against active action-level procedure memory."""

    def __init__(self, **config: int) -> None:
        super().__init__(**config)
        width = self.hidden_width
        self.matcher_memory_norm = nn.LayerNorm(self.procedure_width)
        self.matcher_memory_projection = nn.Linear(self.procedure_width, width)
        self.matcher_update = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.matcher_update[-1].weight)
        nn.init.zeros_(self.matcher_update[-1].bias)

    def forward(
        self,
        procedure_slots: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        memory_values: torch.Tensor,
        memory_mask: torch.Tensor,
        teacher_actions: torch.Tensor | None = None,
        correspondence: bool = True,
    ) -> CandidateProcedureDecode:
        self._validate(procedure_slots, action_features, action_mask, teacher_actions)
        reference = next(self.parameters())
        if memory_values.ndim != 2 or memory_values.shape[1] != self.procedure_width:
            raise ValueError("memory_values must be [memory_slots, procedure_width]")
        if memory_mask.dtype is not torch.bool or memory_mask.shape != (memory_values.shape[0],):
            raise ValueError("memory_mask must be boolean [memory_slots]")
        if memory_values.device != reference.device or memory_values.dtype != reference.dtype:
            raise ValueError("memory_values must match the decoder device and dtype")
        if memory_mask.device != reference.device:
            raise ValueError("memory_mask must match the decoder device")
        if torch.is_inference(procedure_slots):
            procedure_slots = procedure_slots.clone()
        slots = self.procedure_projection(procedure_slots)
        actions = self.action_projection(action_features.detach())
        if correspondence and bool(memory_mask.any().item()):
            memory = self.matcher_memory_projection(self.matcher_memory_norm(memory_values))
            scores = torch.einsum("baw,sw->bas", actions, memory) / math.sqrt(self.hidden_width)
            scores = scores.masked_fill(~memory_mask.view(1, 1, -1), -torch.inf)
            aligned = torch.softmax(scores, dim=-1) @ memory.unsqueeze(0)
            mismatch = actions - aligned
            actions = actions + self.matcher_update(torch.cat((actions, mismatch), dim=-1))

        batch = slots.shape[0]
        hidden = torch.tanh(self.initial_hidden(slots.mean(dim=1)))
        previous = self.start_token.unsqueeze(0).expand(batch, -1)
        logits_rows, selected_rows = [], []
        for step in range(self.maximum_steps):
            query = (hidden + self.position_tokens[step]).unsqueeze(1)
            context, _ = self.slot_attention(query, slots, slots, need_weights=False)
            hidden = self.recurrent(torch.cat((previous, context.squeeze(1)), dim=-1), hidden)
            pointer = self.output_query(self.output_norm(hidden))
            action_logits = torch.einsum("bw,baw->ba", pointer, actions) / math.sqrt(self.hidden_width)
            action_logits = action_logits.masked_fill(~action_mask, -torch.inf)
            logits = torch.cat((action_logits, self.stop_score(pointer)), dim=-1)
            logits_rows.append(logits)
            selected = logits.argmax(dim=-1)
            selected_rows.append(selected)
            previous_indices = selected if teacher_actions is None else teacher_actions[:, step]
            previous = self._selected_embedding(previous_indices, actions)
        return CandidateProcedureDecode(
            logits=torch.stack(logits_rows, dim=1),
            selected_indices=torch.stack(selected_rows, dim=1),
        )


__all__ = ["ActionTraceMatcherDecoder", "ActionTraceProceduralCore"]
