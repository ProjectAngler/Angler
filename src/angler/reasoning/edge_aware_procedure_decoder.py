"""Learned edge-aware set-to-procedure decoding."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_procedure_decoder import CandidateProcedureDecode
from .structured_relational_encoder import StructuredActionTraceMatcherDecoder


class EdgeAwareActionTraceDecoder(StructuredActionTraceMatcherDecoder):
    """Pointer decoder with learned public-edge messages and coverage.

    Deterministic plumbing supplies relation indicators only. Learned modules
    decide their meaning, candidate priority, ordering, and STOP behavior.
    Grounded candidates are selected without replacement within one bounded
    procedure, matching the public one-component/one-candidate contract.
    """

    def __init__(self, *, relation_count: int, message_rounds: int, **config: int) -> None:
        super().__init__(**config)
        if type(relation_count) is not int or relation_count <= 0:
            raise ValueError("relation_count must be positive")
        if type(message_rounds) is not int or message_rounds <= 0:
            raise ValueError("message_rounds must be positive")
        self.relation_count = relation_count
        self.message_rounds = message_rounds
        width = self.hidden_width
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(relation_count),
            nn.Linear(relation_count, width),
            nn.SiLU(),
        )
        self.edge_message = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.SiLU(),
        )
        self.edge_gate = nn.Linear(3 * width, 1)
        self.edge_update = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.edge_update[-1].weight)
        nn.init.zeros_(self.edge_update[-1].bias)
        self.selected_edge_projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.donor_prior_raw_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)))
        )

    def _edge_refine(
        self,
        actions: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_codes = self.edge_encoder(edge_features)
        pair_mask = action_mask.unsqueeze(2) & action_mask.unsqueeze(1)
        current = actions
        for _ in range(self.message_rounds):
            left = current.unsqueeze(2).expand(-1, -1, current.shape[1], -1)
            right = current.unsqueeze(1).expand(-1, current.shape[1], -1, -1)
            joined = torch.cat((left, right, edge_codes), dim=-1)
            messages = self.edge_message(joined)
            gates = torch.sigmoid(self.edge_gate(joined)).squeeze(-1)
            gates = gates * pair_mask.to(gates.dtype)
            outgoing = (gates.unsqueeze(-1) * messages).sum(dim=2) / gates.sum(
                dim=2, keepdim=True
            ).clamp_min(1.0)
            incoming = (gates.unsqueeze(-1) * messages).sum(dim=1) / gates.sum(
                dim=1
            ).unsqueeze(-1).clamp_min(1.0)
            current = current + self.edge_update(
                torch.cat((current, outgoing, incoming), dim=-1)
            )
        return current, edge_codes

    def forward(
        self,
        procedure_slots: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        edge_features: torch.Tensor,
        donor_scores: torch.Tensor,
        memory_values: torch.Tensor,
        memory_mask: torch.Tensor,
        teacher_actions: torch.Tensor | None = None,
        correspondence: bool = True,
        donor_prior: bool = True,
        edge_reasoning: bool = True,
        coverage: bool = True,
    ) -> CandidateProcedureDecode:
        self._validate(procedure_slots, action_features, action_mask, teacher_actions)
        reference = next(self.parameters())
        batch, actions_count = action_mask.shape
        if (
            edge_features.shape
            != (batch, actions_count, actions_count, self.relation_count)
            or donor_scores.shape != (batch, actions_count)
            or edge_features.device != reference.device
            or donor_scores.device != reference.device
            or edge_features.dtype != reference.dtype
            or donor_scores.dtype != reference.dtype
            or not bool(torch.isfinite(edge_features).all().item())
            or not bool(torch.isfinite(donor_scores).all().item())
        ):
            raise ValueError("edge features and donor scores are invalid")
        if memory_values.ndim != 2 or memory_values.shape[1] != self.procedure_width:
            raise ValueError("memory_values must be [memory_slots, procedure_width]")
        if memory_mask.dtype is not torch.bool or memory_mask.shape != (memory_values.shape[0],):
            raise ValueError("memory_mask must be boolean [memory_slots]")
        if torch.is_inference(procedure_slots):
            procedure_slots = procedure_slots.clone()
        slots = self.procedure_projection(procedure_slots)
        actions = self.action_projection(action_features)
        if correspondence and bool(memory_mask.any().item()):
            memory = self.matcher_memory_projection(self.matcher_memory_norm(memory_values))
            match = torch.einsum("baw,sw->bas", actions, memory) / math.sqrt(self.hidden_width)
            match = match.masked_fill(~memory_mask.view(1, 1, -1), -torch.inf)
            aligned = torch.softmax(match, dim=-1) @ memory.unsqueeze(0)
            actions = actions + self.matcher_update(
                torch.cat((actions, actions - aligned), dim=-1)
            )
        if edge_reasoning:
            actions, edge_codes = self._edge_refine(actions, edge_features, action_mask)
        else:
            edge_codes = torch.zeros(
                (*edge_features.shape[:3], self.hidden_width),
                device=reference.device,
                dtype=reference.dtype,
            )

        hidden = torch.tanh(self.initial_hidden(slots.mean(dim=1)))
        previous = self.start_token.unsqueeze(0).expand(batch, -1)
        used = torch.zeros_like(action_mask)
        previous_indices = torch.full(
            (batch,), self.stop_index, device=reference.device, dtype=torch.long
        )
        logits_rows, selected_rows = [], []
        for step in range(self.maximum_steps):
            query = (hidden + self.position_tokens[step]).unsqueeze(1)
            context, _ = self.slot_attention(query, slots, slots, need_weights=False)
            hidden = self.recurrent(torch.cat((previous, context.squeeze(1)), dim=-1), hidden)
            pointer = self.output_query(self.output_norm(hidden))
            step_actions = actions
            if step > 0 and edge_reasoning:
                safe_previous = previous_indices.clamp(0, actions_count - 1)
                row = edge_codes[
                    torch.arange(batch, device=reference.device), safe_previous
                ]
                active = (previous_indices < actions_count).view(batch, 1, 1)
                step_actions = actions + torch.where(
                    active,
                    self.selected_edge_projection(row),
                    torch.zeros_like(row),
                )
            action_logits = torch.einsum(
                "bw,baw->ba", pointer, step_actions
            ) / math.sqrt(self.hidden_width)
            if donor_prior:
                action_logits = action_logits + F.softplus(
                    self.donor_prior_raw_scale
                ) * donor_scores
            available = action_mask & (~used if coverage else torch.ones_like(used))
            action_logits = action_logits.masked_fill(~available, -torch.inf)
            logits = torch.cat((action_logits, self.stop_score(pointer)), dim=-1)
            logits_rows.append(logits)
            selected = logits.argmax(dim=-1)
            selected_rows.append(selected)
            previous_indices = selected if teacher_actions is None else teacher_actions[:, step]
            valid = (previous_indices >= 0) & (previous_indices < actions_count)
            if coverage:
                used = used | (
                    F.one_hot(
                        previous_indices.clamp(0, actions_count - 1),
                        num_classes=actions_count,
                    ).to(torch.bool)
                    & valid.unsqueeze(1)
                )
            previous = self._selected_embedding(previous_indices, actions)
        return CandidateProcedureDecode(
            logits=torch.stack(logits_rows, dim=1),
            selected_indices=torch.stack(selected_rows, dim=1),
        )


__all__ = ["EdgeAwareActionTraceDecoder"]
