"""Learned structured decoding from latent procedures to candidate actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CandidateProcedureDecode:
    logits: torch.Tensor
    selected_indices: torch.Tensor


class CandidateProcedureDecoder(nn.Module):
    """Autoregressively point into task-local candidates or emit STOP.

    Candidate meanings arrive as detached foundation embeddings.  The decoder
    has no global action table, task identity, or solution rule; it learns how
    latent Angler procedure slots address whatever candidates the caller
    presents for this task.
    """

    def __init__(
        self,
        *,
        content_width: int,
        procedure_width: int,
        hidden_width: int,
        heads: int,
        maximum_actions: int,
        maximum_steps: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("content_width", content_width),
            ("procedure_width", procedure_width),
            ("hidden_width", hidden_width),
            ("heads", heads),
            ("maximum_actions", maximum_actions),
            ("maximum_steps", maximum_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_width % heads:
            raise ValueError("hidden_width must be divisible by heads")
        self.content_width = content_width
        self.procedure_width = procedure_width
        self.hidden_width = hidden_width
        self.maximum_actions = maximum_actions
        self.maximum_steps = maximum_steps
        self.stop_index = maximum_actions
        self.procedure_projection = nn.Linear(procedure_width, hidden_width)
        self.action_projection = nn.Sequential(
            nn.LayerNorm(content_width),
            nn.Linear(content_width, hidden_width),
            nn.GELU(),
        )
        self.initial_hidden = nn.Linear(hidden_width, hidden_width)
        self.start_token = nn.Parameter(torch.empty(hidden_width))
        self.stop_token = nn.Parameter(torch.empty(hidden_width))
        self.position_tokens = nn.Parameter(torch.empty(maximum_steps, hidden_width))
        nn.init.normal_(self.start_token, std=1.0 / math.sqrt(hidden_width))
        nn.init.normal_(self.stop_token, std=1.0 / math.sqrt(hidden_width))
        nn.init.normal_(self.position_tokens, std=1.0 / math.sqrt(hidden_width))
        self.slot_attention = nn.MultiheadAttention(
            hidden_width,
            heads,
            batch_first=True,
        )
        self.recurrent = nn.GRUCell(hidden_width * 2, hidden_width)
        self.output_norm = nn.LayerNorm(hidden_width)
        self.output_query = nn.Linear(hidden_width, hidden_width)
        self.stop_score = nn.Linear(hidden_width, 1)

    def forward(
        self,
        procedure_slots: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        teacher_actions: torch.Tensor | None = None,
    ) -> CandidateProcedureDecode:
        self._validate(procedure_slots, action_features, action_mask, teacher_actions)
        # A frozen upstream reasoner may legitimately publish its slots from
        # inference_mode.  Decoder-weight gradients still need an ordinary
        # detached input tensor saved for backward; cloning changes neither
        # values nor scientific inputs.
        if torch.is_inference(procedure_slots):
            procedure_slots = procedure_slots.clone()
        slots = self.procedure_projection(procedure_slots)
        actions = self.action_projection(action_features.detach())
        batch = slots.shape[0]
        hidden = torch.tanh(self.initial_hidden(slots.mean(dim=1)))
        previous = self.start_token.unsqueeze(0).expand(batch, -1)
        logits_rows = []
        selected_rows = []
        for step in range(self.maximum_steps):
            query = (hidden + self.position_tokens[step]).unsqueeze(1)
            context, _ = self.slot_attention(query, slots, slots, need_weights=False)
            hidden = self.recurrent(
                torch.cat((previous, context.squeeze(1)), dim=-1),
                hidden,
            )
            pointer = self.output_query(self.output_norm(hidden))
            action_logits = torch.einsum("bw,baw->ba", pointer, actions) / math.sqrt(
                self.hidden_width
            )
            action_logits = action_logits.masked_fill(~action_mask, -torch.inf)
            logits = torch.cat((action_logits, self.stop_score(pointer)), dim=-1)
            logits_rows.append(logits)
            selected = logits.argmax(dim=-1)
            selected_rows.append(selected)
            if teacher_actions is None:
                previous_indices = selected
            else:
                previous_indices = teacher_actions[:, step]
            previous = self._selected_embedding(
                previous_indices,
                actions,
            )
        return CandidateProcedureDecode(
            logits=torch.stack(logits_rows, dim=1),
            selected_indices=torch.stack(selected_rows, dim=1),
        )

    def _selected_embedding(
        self,
        indices: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        safe = indices.clamp(min=0, max=self.stop_index)
        stop = self.stop_token.view(1, 1, -1).expand(actions.shape[0], -1, -1)
        choices = torch.cat((actions, stop), dim=1)
        selected = choices[torch.arange(actions.shape[0], device=actions.device), safe]
        ignored = indices < 0
        return torch.where(ignored.unsqueeze(1), self.stop_token.unsqueeze(0), selected)

    def _validate(self, slots, actions, mask, teacher) -> None:
        reference = next(self.parameters())
        if slots.ndim != 3 or slots.shape[-1] != self.procedure_width:
            raise ValueError("procedure_slots must be [batch, slots, procedure_width]")
        if actions.shape != (slots.shape[0], self.maximum_actions, self.content_width):
            raise ValueError("action_features must expose the configured padded action shape")
        if mask.dtype is not torch.bool or mask.shape != actions.shape[:2]:
            raise ValueError("action_mask must be boolean [batch, maximum_actions]")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every task needs at least one action candidate")
        for name, value in (("procedure_slots", slots), ("action_features", actions)):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match decoder device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if mask.device != reference.device:
            raise ValueError("action_mask must match decoder device")
        if teacher is not None:
            if teacher.dtype != torch.long or teacher.shape != (slots.shape[0], self.maximum_steps):
                raise ValueError("teacher_actions must be long [batch, maximum_steps]")
            if teacher.device != reference.device:
                raise ValueError("teacher_actions must match decoder device")
            valid = (teacher == -100) | ((teacher >= 0) & (teacher <= self.stop_index))
            if not bool(valid.all().item()):
                raise ValueError("teacher action is outside candidate/STOP space")


def candidate_procedure_loss(
    decoded: CandidateProcedureDecode,
    targets: torch.Tensor,
) -> torch.Tensor:
    if targets.dtype != torch.long or targets.shape != decoded.logits.shape[:2]:
        raise ValueError("targets must be long [batch, steps]")
    return F.cross_entropy(
        decoded.logits.reshape(-1, decoded.logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


__all__ = [
    "CandidateProcedureDecode",
    "CandidateProcedureDecoder",
    "candidate_procedure_loss",
]
