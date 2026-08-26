"""Read-only knowledge representations from a frozen causal language model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def freeze_knowledge_model(model: nn.Module) -> None:
    """Make a foundation model a representation source, never an optimizer target."""

    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("knowledge model still exposes trainable parameters")


def encode_detached_segments(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    hidden_state_index: int = -1,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Encode independent text segments without joining them or retaining gradients.

    Each returned row is the final non-padding token representation of exactly
    one supplied segment.  Callers, not this function, own observation
    segmentation; this function cannot assemble a multi-fact problem.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        raise ValueError("at least one text segment is required")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("knowledge segments must be non-empty strings")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("knowledge-model parameters must be frozen")

    model_device = next(model.parameters()).device
    encoded_batches: list[torch.Tensor] = []
    prior_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            text_batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            encoded = encoded.to(model_device)
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("knowledge model did not return hidden states")
            try:
                selected_hidden = hidden_states[hidden_state_index]
            except IndexError as error:
                raise ValueError("hidden_state_index is outside the model output") from error
            attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
            positions = torch.arange(
                attention_mask.shape[1],
                device=attention_mask.device,
            ).unsqueeze(0)
            final_positions = positions.masked_fill(~attention_mask, -1).max(dim=1).values
            if bool((final_positions < 0).any().item()):
                raise RuntimeError("tokenizer produced an empty encoded segment")
            batch_indices = torch.arange(selected_hidden.shape[0], device=model_device)
            pooled = selected_hidden[batch_indices, final_positions]
            encoded_batches.append(
                pooled.detach().to(device="cpu", dtype=storage_dtype)
            )
    model.train(prior_training)
    result = torch.cat(encoded_batches, dim=0)
    if result.requires_grad or result.grad_fn is not None:
        raise RuntimeError("knowledge representations unexpectedly retain a gradient graph")
    return result
