"""Outcome-only credit assignment for sampled Angler actions."""

from __future__ import annotations

import torch


def leave_one_out_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Center each sampled outcome on the other outcomes for the same task.

    The final dimension is the set of independently sampled actions for one
    task encounter.  Excluding an action from its own baseline keeps its
    policy-gradient credit unbiased while requiring neither a learned critic
    nor another presentation of the task.
    """

    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("leave-one-out credit requires at least two samples")
    if not rewards.is_floating_point():
        raise ValueError("rewards must use a floating-point dtype")
    if rewards.requires_grad:
        raise ValueError("externally scored rewards must not carry gradients")
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("rewards must be finite")

    sample_count = rewards.shape[-1]
    other_mean = (
        rewards.sum(dim=-1, keepdim=True) - rewards
    ) / (sample_count - 1)
    return rewards - other_mean
