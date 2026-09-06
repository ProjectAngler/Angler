"""Sparse memory-dependent causal routing for procedural apprenticeship V8."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .content_addressed_causal_memory import (
    ContentAddressedCausalMemoryCore,
    ContentAddressedCausalMemoryOutput,
    ContentAddressedCausalMemoryState,
)


class _MemoryDependentScore(nn.Module):
    """A bias-free residual that is exactly zero when memory reads are zero."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.first = nn.Linear(rank * 2, rank, bias=False)
        self.activation = nn.SiLU()
        self.second = nn.Linear(rank, 1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        rank = values.shape[-1] // 3
        read = values[..., rank : rank * 2]
        interaction = values[..., rank * 2 :]
        return self.second(self.activation(self.first(torch.cat((read, interaction), dim=-1))))


class SparseMemoryCausalRoutingCore(ContentAddressedCausalMemoryCore):
    """V7 bounded memory with generic learned top-2 routing and no score bypass."""

    routing_k = 2

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.score_network = _MemoryDependentScore(self.rank)

    @staticmethod
    def _top_k_preserve_mass(weights: torch.Tensor, k: int) -> torch.Tensor:
        if weights.shape[-1] < k:
            raise ValueError("routing width is smaller than top-k")
        mass = weights.sum(dim=-1, keepdim=True)
        values, indices = torch.topk(weights, k=k, dim=-1)
        sparse = torch.zeros_like(weights).scatter(-1, indices, values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1.0e-8) * mass

    def _content_weights(self, *args, **kwargs) -> torch.Tensor:
        dense = super()._content_weights(*args, **kwargs)
        return self._top_k_preserve_mass(dense, self.routing_k)

    def _allocation_weights(self, usage: torch.Tensor) -> torch.Tensor:
        dense = super()._allocation_weights(usage)
        return self._top_k_preserve_mass(dense, self.routing_k)

    def forward(self, *args, **kwargs) -> ContentAddressedCausalMemoryOutput:
        output = super().forward(*args, **kwargs)
        # Content and allocation may nominate different pairs. The final write
        # is routed once more so every actual update touches at most two slots.
        write_weights = self._top_k_preserve_mass(output.write_weights, self.routing_k)
        return replace(output, write_weights=write_weights)

    def apply_mixed_outcome_update(
        self,
        state: ContentAddressedCausalMemoryState,
        output: ContentAddressedCausalMemoryOutput,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> ContentAddressedCausalMemoryState:
        self._validate_state(state)
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        if candidate_mask.shape[0] != 1:
            raise ValueError("one persistent memory accepts one chronological event at a time")
        self._validate_output(output, candidate_mask)
        keys, values, usage = state.keys, state.values, state.usage
        for candidate in range(candidate_mask.shape[1]):
            slot_weights = output.write_weights[0, candidate]
            slot = slot_weights.unsqueeze(-1)
            write_key = output.write_keys[0, candidate].unsqueeze(0)
            write_value = output.write_values[0, candidate].unsqueeze(0)
            erase = output.erase_gates[0, candidate].unsqueeze(0)
            keys = (1.0 - slot) * keys + slot * write_key
            erased_values = values * (1.0 - slot * erase)
            values = (1.0 - slot) * erased_values + slot * write_value
            usage = usage + slot_weights * (1.0 - usage)
        updated = ContentAddressedCausalMemoryState(
            keys=keys,
            values=values,
            usage=usage,
            step=state.step + 1,
        )
        self._validate_state(updated)
        return updated.detached_clone() if detach_state else updated


__all__ = ["SparseMemoryCausalRoutingCore"]

