"""Sufficient-horizon DNC allocation for factorized causal memory."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from .content_addressed_causal_memory import (
    ContentAddressedCausalMemoryOutput,
    ContentAddressedCausalMemoryState,
)
from .factorized_key_value_causal_memory import (
    FactorizedKeyValueCausalMemoryCore,
)


class DncAllocatedFactorizedCausalMemoryCore(
    FactorizedKeyValueCausalMemoryCore
):
    """64 shared slots with generic least-used top-1 episodic writes."""

    memory_slot_identity = 64

    def __init__(self, **kwargs) -> None:
        requested_slots = kwargs.pop("memory_slots", self.memory_slot_identity)
        if requested_slots != self.memory_slot_identity:
            raise ValueError("V10 memory identity requires exactly 64 slots")
        # Build the inherited shared networks under their frozen V9 identity,
        # then replace only the slot-shaped state. No slot-specific learned
        # tensor survives into V10.
        super().__init__(memory_slots=16, **kwargs)
        self.memory_slots = self.memory_slot_identity
        del self.slot_address_anchors
        del self.allocation_logits
        del self.allocation_usage_pressure

    def initial_memory_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContentAddressedCausalMemoryState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        return ContentAddressedCausalMemoryState(
            keys=torch.zeros(
                self.memory_slots,
                self.rank,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            values=torch.zeros(
                self.memory_slots,
                self.rank,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            usage=torch.zeros(
                self.memory_slots,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            step=0,
        )

    initial_plastic_state = initial_memory_state

    def _allocation_weights(self, usage: torch.Tensor) -> torch.Tensor:
        """Return the top DNC usage-sorted allocation as an exact one-hot."""

        if usage.shape != (self.memory_slots,):
            raise ValueError("allocation usage has the wrong shape")
        sorted_usage, order = torch.sort(usage, stable=True)
        exclusive_product = torch.cumprod(
            torch.cat((torch.ones_like(sorted_usage[:1]), sorted_usage[:-1])),
            dim=0,
        )
        sorted_allocation = (1.0 - sorted_usage) * exclusive_product
        allocation = torch.zeros_like(usage).scatter(0, order, sorted_allocation)
        # Before exhaustion this is the first unused slot. After exhaustion it
        # is the least-used slot. The discrete choice is generic memory
        # bookkeeping and carries no task, content, outcome, or answer signal.
        winner = int(allocation.argmax().item())
        return F.one_hot(
            torch.tensor(winner, device=usage.device),
            num_classes=self.memory_slots,
        ).to(dtype=usage.dtype)

    def forward(self, *args, **kwargs) -> ContentAddressedCausalMemoryOutput:
        state = kwargs.get("memory_state")
        if state is None:
            state = self.initial_memory_state()
            kwargs["memory_state"] = state
        output = super().forward(*args, **kwargs)
        allocation = self._allocation_weights(state.usage).view(1, 1, -1)
        write_weights = output.write_strengths.unsqueeze(-1) * allocation
        return replace(
            output,
            allocation_mix=torch.zeros_like(output.allocation_mix),
            write_weights=write_weights,
        )


__all__ = ["DncAllocatedFactorizedCausalMemoryCore"]
