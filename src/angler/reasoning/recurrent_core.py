"""A separate recurrent neural reasoning core for frozen knowledge features.

The module never loads, updates, or selects a language model.  It receives
detached public-observation features, integrates them through a shared
recurrent workspace, and emits a structured action policy.  Environment truth
and outcome scoring remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

_STATE_DIGEST_DOMAIN = b"project-angler.reasoning-core.v1\x00"


@dataclass(frozen=True, slots=True)
class ReasoningCoreConfig:
    """Resource-scalable topology for one uniformly active reasoning core."""

    knowledge_width: int = 2560
    core_width: int = 512
    workspace_slots: int = 16
    attention_heads: int = 8
    feedforward_width: int = 2048
    reasoning_steps: int = 8
    maximum_reasoning_steps: int = 12
    maximum_entities: int = 16

    def __post_init__(self) -> None:
        positive = {
            "knowledge_width": self.knowledge_width,
            "core_width": self.core_width,
            "workspace_slots": self.workspace_slots,
            "attention_heads": self.attention_heads,
            "feedforward_width": self.feedforward_width,
            "reasoning_steps": self.reasoning_steps,
            "maximum_reasoning_steps": self.maximum_reasoning_steps,
            "maximum_entities": self.maximum_entities,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("reasoning-core dimensions must be positive: " + ", ".join(invalid))
        if self.core_width % self.attention_heads:
            raise ValueError("core_width must be divisible by attention_heads")
        if self.reasoning_steps > self.maximum_reasoning_steps:
            raise ValueError("reasoning_steps exceeds maximum_reasoning_steps")


@dataclass(frozen=True, slots=True)
class ReasoningTrajectory:
    """Sampled or greedy public-entity actions and their policy statistics."""

    order_indices: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class _SharedReasoningStep(nn.Module):
    def __init__(self, config: ReasoningCoreConfig) -> None:
        super().__init__()
        width = config.core_width
        heads = config.attention_heads

        self.edge_context_norm = nn.LayerNorm(width * 4)
        self.edge_context = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.edge_message_norm = nn.LayerNorm(width)
        self.edge_to_entity = nn.Linear(width, width)
        self.edge_to_fact = nn.Linear(width, width)
        self.entity_node_norm = nn.LayerNorm(width)
        self.fact_node_norm = nn.LayerNorm(width)
        self.node_feedforward = nn.Sequential(
            nn.Linear(width, config.feedforward_width),
            nn.SiLU(),
            nn.Linear(config.feedforward_width, width),
        )

        self.slot_memory_norm = nn.LayerNorm(width)
        self.slot_memory_attention = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
            dropout=0.0,
        )
        self.slot_self_norm = nn.LayerNorm(width)
        self.slot_self_attention = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
            dropout=0.0,
        )
        self.slot_ff_norm = nn.LayerNorm(width)
        self.slot_feedforward = nn.Sequential(
            nn.Linear(width, config.feedforward_width),
            nn.SiLU(),
            nn.Linear(config.feedforward_width, width),
        )

        self.mention_gate = nn.Parameter(torch.tensor(-1.0))
        self.entity_message_gate = nn.Parameter(torch.tensor(-1.0))
        self.fact_message_gate = nn.Parameter(torch.tensor(-1.0))
        self.entity_node_gate = nn.Parameter(torch.tensor(-1.0))
        self.fact_node_gate = nn.Parameter(torch.tensor(-1.0))
        self.slot_memory_gate = nn.Parameter(torch.tensor(-1.0))
        self.slot_self_gate = nn.Parameter(torch.tensor(-1.0))
        self.slot_ff_gate = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _gated_residual(base: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return base + torch.sigmoid(gate) * delta

    def forward(
        self,
        slots: torch.Tensor,
        facts: torch.Tensor,
        entities: torch.Tensor,
        mentions: torch.Tensor,
        *,
        fact_mask: torch.Tensor,
        entity_mask: torch.Tensor,
        mention_mask: torch.Tensor,
        mention_fact_indices: torch.Tensor,
        mention_entity_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_fact_indices = mention_fact_indices.masked_fill(~mention_mask, 0)
        safe_entity_indices = mention_entity_indices.masked_fill(~mention_mask, 0)
        width = mentions.shape[-1]
        fact_at_mention = facts.gather(
            1,
            safe_fact_indices.unsqueeze(-1).expand(-1, -1, width),
        )
        entity_at_mention = entities.gather(
            1,
            safe_entity_indices.unsqueeze(-1).expand(-1, -1, width),
        )
        slot_context = slots.mean(dim=1, keepdim=True).expand(
            -1,
            mentions.shape[1],
            -1,
        )
        edge_input = torch.cat(
            (mentions, fact_at_mention, entity_at_mention, slot_context),
            dim=-1,
        )
        edge_delta = self.edge_context(self.edge_context_norm(edge_input))
        edge_delta = edge_delta.masked_fill(~mention_mask.unsqueeze(-1), 0.0)
        mentions = self._gated_residual(
            mentions,
            edge_delta,
            self.mention_gate,
        )
        mentions = mentions.masked_fill(~mention_mask.unsqueeze(-1), 0.0)

        normalized_mentions = self.edge_message_norm(mentions)
        entity_messages = _scatter_incident_messages(
            self.edge_to_entity(normalized_mentions),
            safe_entity_indices,
            mention_mask,
            target_count=entities.shape[1],
        )
        fact_messages = _scatter_incident_messages(
            self.edge_to_fact(normalized_mentions),
            safe_fact_indices,
            mention_mask,
            target_count=facts.shape[1],
        )
        entities = self._gated_residual(
            entities,
            entity_messages,
            self.entity_message_gate,
        )
        facts = self._gated_residual(
            facts,
            fact_messages,
            self.fact_message_gate,
        )
        entities = self._gated_residual(
            entities,
            self.node_feedforward(self.entity_node_norm(entities)),
            self.entity_node_gate,
        )
        facts = self._gated_residual(
            facts,
            self.node_feedforward(self.fact_node_norm(facts)),
            self.fact_node_gate,
        )
        entities = entities.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        facts = facts.masked_fill(~fact_mask.unsqueeze(-1), 0.0)

        memory = torch.cat((facts, entities, mentions), dim=1)
        memory_mask = torch.cat((fact_mask, entity_mask, mention_mask), dim=1)
        normalized_slots = self.slot_memory_norm(slots)
        memory_delta, _ = self.slot_memory_attention(
            normalized_slots,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        slots = self._gated_residual(
            slots,
            memory_delta,
            self.slot_memory_gate,
        )

        normalized_slots = self.slot_self_norm(slots)
        self_delta, _ = self.slot_self_attention(
            normalized_slots,
            normalized_slots,
            normalized_slots,
            need_weights=False,
        )
        slots = self._gated_residual(slots, self_delta, self.slot_self_gate)
        slots = self._gated_residual(
            slots,
            self.slot_feedforward(self.slot_ff_norm(slots)),
            self.slot_ff_gate,
        )
        return slots, facts, entities, mentions


class RecurrentReasoningCore(nn.Module):
    """Integrate detached knowledge features and choose public entity actions."""

    def __init__(self, config: ReasoningCoreConfig) -> None:
        super().__init__()
        self.config = config
        width = config.core_width
        self.knowledge_projection = nn.Linear(config.knowledge_width, width)
        self.fact_type = nn.Parameter(torch.zeros(width))
        self.entity_type = nn.Parameter(torch.zeros(width))
        self.mention_type = nn.Parameter(torch.zeros(width))
        self.seed_slots = nn.Parameter(
            torch.empty(config.workspace_slots, width)
        )
        self.reasoning_step = _SharedReasoningStep(config)
        self.output_norm = nn.LayerNorm(width)

        self.pointer_keys = nn.Linear(width, width, bias=False)
        self.pointer_query = nn.Linear(width, width, bias=False)
        self.decoder_initial = nn.Linear(width, width)
        self.decoder_cell = nn.GRUCell(width, width)
        self.position_embeddings = nn.Parameter(
            torch.empty(config.maximum_entities, width)
        )
        value_width = max(1, width // 4)
        self.value_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, value_width),
            nn.SiLU(),
            nn.Linear(value_width, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.seed_slots, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.fact_type, mean=0.0, std=0.02)
        nn.init.normal_(self.entity_type, mean=0.0, std=0.02)
        nn.init.normal_(self.mention_type, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(
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
        reasoning_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            fact_features,
            fact_mask,
            entity_features,
            entity_mask,
            mention_features,
            mention_mask,
            mention_fact_indices,
            mention_entity_indices,
        )
        steps = self.config.reasoning_steps if reasoning_steps is None else reasoning_steps
        if not 0 <= steps <= self.config.maximum_reasoning_steps:
            raise ValueError("reasoning_steps is outside the configured range")

        projection_dtype = self.knowledge_projection.weight.dtype
        facts = self.knowledge_projection(fact_features.to(dtype=projection_dtype))
        entities = self.knowledge_projection(entity_features.to(dtype=projection_dtype))
        mentions = self.knowledge_projection(mention_features.to(dtype=projection_dtype))
        facts = facts + self.fact_type
        entities = entities + self.entity_type
        mentions = mentions + self.mention_type
        facts = facts.masked_fill(~fact_mask.unsqueeze(-1), 0.0)
        entities = entities.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        mentions = mentions.masked_fill(~mention_mask.unsqueeze(-1), 0.0)
        slots = self.seed_slots.unsqueeze(0).expand(facts.shape[0], -1, -1)

        for _ in range(steps):
            slots, facts, entities, mentions = self.reasoning_step(
                slots,
                facts,
                entities,
                mentions,
                fact_mask=fact_mask,
                entity_mask=entity_mask,
                mention_mask=mention_mask,
                mention_fact_indices=mention_fact_indices,
                mention_entity_indices=mention_entity_indices,
            )
        return self.output_norm(entities), self.output_norm(slots)

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
        samples_per_task: int = 1,
        greedy: bool = False,
        reasoning_steps: int | None = None,
        temperature: float = 1.0,
    ) -> ReasoningTrajectory:
        if samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        entities, _ = self.encode(
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
        return self.act_encoded(
            entities,
            entity_mask,
            samples_per_task=samples_per_task,
            greedy=greedy,
            temperature=temperature,
        )

    def act_encoded(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        samples_per_task: int = 1,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> ReasoningTrajectory:
        """Decode an action from already-reasoned public entity states."""

        if samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if entities.ndim != 3 or entities.shape[-1] != self.config.core_width:
            raise ValueError(
                "entities must have shape [batch, entities, core_width]"
            )
        if (
            entity_mask.shape != entities.shape[:2]
            or entity_mask.dtype != torch.bool
        ):
            raise ValueError("entity_mask must be bool with the entity shape")
        if entity_mask.device != entities.device:
            raise ValueError("entities and entity_mask must share a device")
        if entities.device != self.pointer_keys.weight.device:
            raise ValueError("encoded entities must share the core device")
        if entities.dtype != self.pointer_keys.weight.dtype:
            raise ValueError("encoded entities must share the core dtype")
        if entities.shape[1] > self.config.maximum_entities:
            raise ValueError("entity count exceeds maximum_entities")
        if bool((entity_mask.sum(dim=1) == 0).any().item()):
            raise ValueError("every task requires at least one entity")

        batch_size, maximum_entities, width = entities.shape
        entity_summary = _masked_mean(entities, entity_mask)
        value = self.value_head(entity_summary).squeeze(-1)
        if samples_per_task > 1:
            entities = entities.repeat_interleave(samples_per_task, dim=0)
            entity_mask = entity_mask.repeat_interleave(samples_per_task, dim=0)
            entity_summary = entity_summary.repeat_interleave(
                samples_per_task,
                dim=0,
            )

        expanded_batch = entities.shape[0]
        keys = self.pointer_keys(entities)
        decoder = self.decoder_initial(entity_summary)
        selected = torch.zeros_like(entity_mask)
        entity_counts = entity_mask.sum(dim=1)
        orders: list[torch.Tensor] = []
        log_probability = torch.zeros(expanded_batch, device=entities.device)
        entropy = torch.zeros_like(log_probability)

        for position in range(maximum_entities):
            active = entity_counts > position
            query = self.pointer_query(
                decoder + self.position_embeddings[position].unsqueeze(0)
            )
            logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
            logits = logits / (math.sqrt(width) * temperature)
            logits = logits.masked_fill(~entity_mask | selected, -torch.inf)
            if bool((~active).any().item()):
                logits = logits.clone()
                logits[~active] = -torch.inf
                logits[~active, 0] = 0.0
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = torch.multinomial(probabilities, 1).squeeze(-1)
            step_log_probability = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            # Masked actions have log-probability -inf.  Replacing those values
            # before multiplication avoids the undefined 0 * -inf backward path
            # while leaving the categorical entropy of valid actions unchanged.
            finite_log_probs = torch.where(
                torch.isfinite(log_probs),
                log_probs,
                torch.zeros_like(log_probs),
            )
            step_entropy = -(probabilities * finite_log_probs).sum(dim=-1)
            log_probability = log_probability + torch.where(
                active,
                step_log_probability,
                torch.zeros_like(step_log_probability),
            )
            entropy = entropy + torch.where(
                active,
                step_entropy,
                torch.zeros_like(step_entropy),
            )
            orders.append(torch.where(active, action, torch.full_like(action, -1)))

            selected_update = torch.zeros_like(selected)
            selected_update.scatter_(1, action.unsqueeze(-1), True)
            selected = selected | (selected_update & active.unsqueeze(-1))
            selected_entity = entities.gather(
                1,
                action.view(-1, 1, 1).expand(-1, 1, width),
            ).squeeze(1)
            next_decoder = self.decoder_cell(
                selected_entity + self.position_embeddings[position],
                decoder,
            )
            decoder = torch.where(active.unsqueeze(-1), next_decoder, decoder)

        order_tensor = torch.stack(orders, dim=-1)
        order_tensor = order_tensor.view(batch_size, samples_per_task, maximum_entities)
        log_probability = log_probability.view(batch_size, samples_per_task)
        entropy = entropy.view(batch_size, samples_per_task)
        return ReasoningTrajectory(order_tensor, log_probability, entropy, value)

    def _validate_inputs(
        self,
        fact_features: torch.Tensor,
        fact_mask: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        mention_features: torch.Tensor,
        mention_mask: torch.Tensor,
        mention_fact_indices: torch.Tensor,
        mention_entity_indices: torch.Tensor,
    ) -> None:
        tensors = (fact_features, entity_features, mention_features)
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("knowledge feature tensors must have shape [batch, items, width]")
        if any(tensor.shape[-1] != self.config.knowledge_width for tensor in tensors):
            raise ValueError("knowledge feature width does not match the core configuration")
        batch_size = fact_features.shape[0]
        if entity_features.shape[0] != batch_size or mention_features.shape[0] != batch_size:
            raise ValueError("all knowledge tensors must share a batch dimension")
        if any(tensor.device != fact_features.device for tensor in tensors):
            raise ValueError("features, masks, and indices must share a device")
        expected_masks = (
            (fact_mask, fact_features.shape[:2], "fact_mask"),
            (entity_mask, entity_features.shape[:2], "entity_mask"),
            (mention_mask, mention_features.shape[:2], "mention_mask"),
        )
        for mask, shape, name in expected_masks:
            if mask.shape != shape or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be bool with shape {shape}")
            if mask.device != fact_features.device:
                raise ValueError("features, masks, and indices must share a device")
        index_specs = (
            (mention_fact_indices, "mention_fact_indices"),
            (mention_entity_indices, "mention_entity_indices"),
        )
        for indices, name in index_specs:
            if indices.shape != mention_mask.shape:
                raise ValueError(f"{name} must have the mention mask shape")
            if indices.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
            if indices.device != fact_features.device:
                raise ValueError("features, masks, and indices must share a device")
        if entity_features.shape[1] > self.config.maximum_entities:
            raise ValueError("entity count exceeds maximum_entities")
        if bool((fact_mask.sum(dim=1) == 0).any().item()):
            raise ValueError("every task requires at least one fact")
        if bool((entity_mask.sum(dim=1) == 0).any().item()):
            raise ValueError("every task requires at least one entity")
        self._validate_incidence_indices(
            mention_fact_indices,
            mention_mask,
            fact_mask,
            item_name="fact",
        )
        self._validate_incidence_indices(
            mention_entity_indices,
            mention_mask,
            entity_mask,
            item_name="entity",
        )

    @staticmethod
    def _validate_incidence_indices(
        indices: torch.Tensor,
        mention_mask: torch.Tensor,
        item_mask: torch.Tensor,
        *,
        item_name: str,
    ) -> None:
        safe_indices = indices.masked_fill(~mention_mask, 0)
        valid_indices = indices.masked_select(mention_mask)
        if valid_indices.numel() and (
            bool((valid_indices < 0).any().item())
            or bool((valid_indices >= item_mask.shape[1]).any().item())
        ):
            raise ValueError(
                f"a mention references a {item_name} outside the public list"
            )
        if valid_indices.numel():
            referenced_items = item_mask.gather(1, safe_indices)
            if not bool(referenced_items.masked_select(mention_mask).all().item()):
                raise ValueError(f"a mention references a padded {item_name}")


def reasoning_state_digest(module: nn.Module) -> str:
    """Hash the complete persistent neural state independently of Qwen."""

    digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def snapshot_reasoning_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Capture exact CPU tensors for candidate rollback and causal swapping."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def restore_reasoning_state(
    module: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> None:
    """Restore an exact snapshot after validating names, shapes, and dtypes."""

    current = module.state_dict()
    if set(current) != set(snapshot):
        raise ValueError("reasoning-state snapshot names do not match the module")
    restored: dict[str, torch.Tensor] = {}
    for name, current_value in current.items():
        saved = snapshot[name]
        if saved.shape != current_value.shape or saved.dtype != current_value.dtype:
            raise ValueError(f"reasoning-state tensor {name!r} is incompatible")
        restored[name] = saved.to(device=current_value.device)
    module.load_state_dict(restored, strict=True)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _scatter_incident_messages(
    messages: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_count: int,
) -> torch.Tensor:
    """Route learned edge messages through public incidence only.

    Square-root count normalization keeps message scale bounded without
    discarding multiplicity.  The routing contains no relation direction,
    target rank, answer, or task-specific transformation.
    """

    width = messages.shape[-1]
    result = messages.new_zeros((messages.shape[0], target_count, width))
    counts = messages.new_zeros((messages.shape[0], target_count, 1))
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, width)
    masked_messages = messages * mask.unsqueeze(-1).to(dtype=messages.dtype)
    result.scatter_add_(1, expanded_indices, masked_messages)
    counts.scatter_add_(
        1,
        indices.unsqueeze(-1),
        mask.unsqueeze(-1).to(dtype=messages.dtype),
    )
    return result / counts.clamp_min(1.0).sqrt()
