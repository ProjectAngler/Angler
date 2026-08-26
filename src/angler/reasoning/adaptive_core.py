"""Feedback-plastic reasoning core built around one shared numeric state.

The frozen language-model connector remains outside this module.  This bridge
receives detached public-observation features, lets the recurrent core propose
one action, and encodes only the observation summary, attempted action, and
bounded scalar outcome into the self-referential state.  It has no task-type
channel and contains no task-solving procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .recurrent_core import (
    ReasoningCoreConfig,
    ReasoningTrajectory,
    RecurrentReasoningCore,
)
from .self_referential_memory import (
    SelfReferentialMemory,
    SelfReferentialState,
)


@dataclass(frozen=True, slots=True)
class AdaptiveFeedbackContext:
    """Learner-visible trace needed to assign an outcome to its own action."""

    observation_summary: torch.Tensor
    action_summary: torch.Tensor


@dataclass(frozen=True, slots=True)
class AdaptiveReasoningTrajectory:
    """One action plus the public trace required for later feedback."""

    action: ReasoningTrajectory
    feedback_context: AdaptiveFeedbackContext
    plastic_context: torch.Tensor


@dataclass(frozen=True, slots=True)
class AdaptiveFeedbackWrite:
    """Result of one explicit outcome-bearing state update."""

    state: SelfReferentialState
    event: torch.Tensor
    delta_norm: torch.Tensor


class AdaptiveReasoningCore(nn.Module):
    """Compose recurrent reasoning with one persistent SRWM competence state."""

    def __init__(
        self,
        config: ReasoningCoreConfig,
        *,
        memory_heads: int | None = None,
    ) -> None:
        super().__init__()
        width = config.core_width
        heads = config.attention_heads if memory_heads is None else memory_heads
        if heads <= 0 or width % heads:
            raise ValueError("memory_heads must divide core_width")

        self.config = config
        self.core = RecurrentReasoningCore(config)
        self.memory = SelfReferentialMemory(width, heads=heads)
        self.observation_encoder = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.context_modulation = nn.Linear(width, width * 2)
        self.adaptive_entity_norm = nn.LayerNorm(width)
        self.action_cell = nn.GRUCell(width, width)
        self.feedback_encoder = nn.Sequential(
            nn.LayerNorm(width * 2 + 2),
            nn.Linear(width * 2 + 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def initial_state(self, batch_size: int) -> SelfReferentialState:
        """Create the single-state lineage root for one or more test streams."""

        return self.memory.initial_state(batch_size)

    def act(
        self,
        fact_features: torch.Tensor,
        fact_mask: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        mention_features: torch.Tensor,
        mention_mask: torch.Tensor,
        mention_fact_indices: torch.Tensor,
        mention_entity_indices: torch.Tensor,
        *,
        state: SelfReferentialState,
        greedy: bool = False,
        reasoning_steps: int | None = None,
        temperature: float = 1.0,
    ) -> AdaptiveReasoningTrajectory:
        """Read the state and act without mutating the state."""

        adapted_entities, observation_summary, plastic_context = (
            self._condition_entities(
                fact_features,
                fact_mask,
                entity_features,
                entity_mask,
                mention_features,
                mention_mask,
                mention_fact_indices,
                mention_entity_indices,
                state=state,
                reasoning_steps=reasoning_steps,
            )
        )
        action = self.core.act_encoded(
            adapted_entities,
            entity_mask,
            samples_per_task=1,
            greedy=greedy,
            temperature=temperature,
        )
        action_summary = self._summarize_action(
            adapted_entities,
            action.order_indices[:, 0],
        )
        return AdaptiveReasoningTrajectory(
            action=action,
            feedback_context=AdaptiveFeedbackContext(
                observation_summary=observation_summary,
                action_summary=action_summary,
            ),
            plastic_context=plastic_context,
        )

    def score_training_order(
        self,
        fact_features: torch.Tensor,
        fact_mask: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        mention_features: torch.Tensor,
        mention_mask: torch.Tensor,
        mention_fact_indices: torch.Tensor,
        mention_entity_indices: torch.Tensor,
        prescribed_order: torch.Tensor,
        *,
        state: SelfReferentialState,
        reasoning_steps: int | None = None,
        temperature: float = 1.0,
    ) -> ReasoningTrajectory:
        """Score a visible meta-training target without writing state."""

        adapted_entities, _, _ = self._condition_entities(
            fact_features,
            fact_mask,
            entity_features,
            entity_mask,
            mention_features,
            mention_mask,
            mention_fact_indices,
            mention_entity_indices,
            state=state,
            reasoning_steps=reasoning_steps,
        )
        return self.core.score_encoded_order(
            adapted_entities,
            entity_mask,
            prescribed_order,
            temperature=temperature,
        )

    def _condition_entities(
        self,
        fact_features: torch.Tensor,
        fact_mask: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        mention_features: torch.Tensor,
        mention_mask: torch.Tensor,
        mention_fact_indices: torch.Tensor,
        mention_entity_indices: torch.Tensor,
        *,
        state: SelfReferentialState,
        reasoning_steps: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entities, slots = self.core.encode(
            fact_features,
            fact_mask,
            entity_features,
            entity_mask,
            mention_features,
            mention_mask,
            mention_fact_indices,
            mention_entity_indices,
            reasoning_steps=reasoning_steps,
        )
        entity_summary = _masked_mean(entities, entity_mask)
        slot_summary = slots.mean(dim=1)
        observation_summary = self.observation_encoder(
            torch.cat((entity_summary, slot_summary), dim=-1)
        )
        plastic_context = self.memory.read(
            observation_summary.unsqueeze(1),
            state,
        ).squeeze(1)

        scale, shift = self.context_modulation(plastic_context).chunk(2, dim=-1)
        adapted_entities = self.adaptive_entity_norm(
            entities * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
            + 0.1 * shift.unsqueeze(1)
        )
        adapted_entities = adapted_entities.masked_fill(
            ~entity_mask.unsqueeze(-1),
            0.0,
        )
        return (
            adapted_entities,
            observation_summary,
            plastic_context,
        )

    def incorporate_feedback(
        self,
        context: AdaptiveFeedbackContext,
        *,
        reward: torch.Tensor,
        state: SelfReferentialState,
    ) -> AdaptiveFeedbackWrite:
        """Write one attempted-action/outcome event into the shared state."""

        batch_size = context.observation_summary.shape[0]
        width = self.config.core_width
        expected = (batch_size, width)
        if (
            context.observation_summary.shape != expected
            or context.action_summary.shape != expected
        ):
            raise ValueError("feedback summaries do not match the core topology")
        if (
            context.observation_summary.device != self.memory.base_y.device
            or context.action_summary.device != self.memory.base_y.device
            or context.observation_summary.dtype != self.memory.base_y.dtype
            or context.action_summary.dtype != self.memory.base_y.dtype
        ):
            raise ValueError("feedback summaries must match the adaptive core")

        scalar_reward = _validate_scalar_feedback(
            "reward",
            reward,
            batch_size,
        )
        scalars = torch.stack(
            (
                scalar_reward,
                torch.ones_like(scalar_reward),
            ),
            dim=-1,
        ).to(
            device=context.observation_summary.device,
            dtype=context.observation_summary.dtype,
        )
        event = self.feedback_encoder(
            torch.cat(
                (
                    context.observation_summary,
                    context.action_summary,
                    scalars,
                ),
                dim=-1,
            )
        )
        _, next_state = self.memory.write(event.unsqueeze(1), state)
        squared_delta = torch.zeros(
            (),
            device=event.device,
            dtype=event.dtype,
        )
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            difference = getattr(next_state, name) - getattr(state, name)
            squared_delta = squared_delta + difference.square().sum()
        return AdaptiveFeedbackWrite(
            state=next_state,
            event=event,
            delta_norm=torch.sqrt(squared_delta),
        )

    def _summarize_action(
        self,
        entities: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        if order.ndim != 2 or order.shape != entities.shape[:2]:
            raise ValueError("attempted order must match the entity topology")
        active = order >= 0
        safe_order = order.masked_fill(~active, 0)
        width = entities.shape[-1]
        ordered_entities = entities.gather(
            1,
            safe_order.unsqueeze(-1).expand(-1, -1, width),
        )
        positions = self.core.position_embeddings[: order.shape[1]].unsqueeze(0)
        action_state = torch.zeros(
            entities.shape[0],
            width,
            device=entities.device,
            dtype=entities.dtype,
        )
        ordered_events = ordered_entities + positions
        for position in range(order.shape[1]):
            next_state = self.action_cell(
                ordered_events[:, position],
                action_state,
            )
            action_state = torch.where(
                active[:, position].unsqueeze(-1),
                next_state,
                action_state,
            )
        return action_state


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / count


def _validate_scalar_feedback(
    name: str,
    value: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [batch]")
    numeric = value.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(numeric).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool(((numeric < 0.0) | (numeric > 1.0)).any().item()):
        raise ValueError(f"{name} must be between zero and one")
    return numeric


__all__ = [
    "AdaptiveFeedbackContext",
    "AdaptiveFeedbackWrite",
    "AdaptiveReasoningCore",
    "AdaptiveReasoningTrajectory",
]
