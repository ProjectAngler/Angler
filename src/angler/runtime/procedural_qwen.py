"""Inject Angler latent procedure tokens into a frozen causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ProceduralQwenGeneration:
    response: str
    generated_token_ids: tuple[int, ...]
    prompt_tokens: int
    procedure_tokens: int


@dataclass(frozen=True, slots=True)
class ProceduralQwenTrainingLoss:
    loss: torch.Tensor
    prompt_tokens: int
    target_tokens: int
    procedure_tokens: int


def qwen_inputs_with_procedural_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    procedure_prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build frozen-Qwen inputs from explicit text plus Angler soft tokens."""

    if type(prompt) is not str or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    if procedure_prefix.ndim != 3 or procedure_prefix.shape[0] != 1:
        raise ValueError("procedure_prefix must be [1, tokens, Qwen width]")
    if procedure_prefix.shape[1] < 1:
        raise ValueError("procedure_prefix must contain at least one token")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Qwen must be frozen before procedural-prefix use")
    device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    if procedure_prefix.shape[-1] != embedding.embedding_dim:
        raise ValueError("procedure-prefix width does not match Qwen embeddings")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = encoded["input_ids"]
    text_embeddings = embedding(input_ids)
    prefix = procedure_prefix.to(device=device, dtype=text_embeddings.dtype)
    inputs = torch.cat((prefix, text_embeddings), dim=1)
    prefix_mask = torch.ones(
        (1, prefix.shape[1]),
        device=device,
        dtype=encoded["attention_mask"].dtype,
    )
    attention_mask = torch.cat((prefix_mask, encoded["attention_mask"]), dim=1)
    return inputs, attention_mask, int(input_ids.shape[1])


def generate_with_procedural_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    procedure_prefix: torch.Tensor,
    *,
    max_new_tokens: int = 256,
) -> ProceduralQwenGeneration:
    """Greedily generate from frozen Qwen conditioned by Angler latent state."""

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    inputs, attention_mask, prompt_tokens = qwen_inputs_with_procedural_prefix(
        model,
        tokenizer,
        prompt,
        procedure_prefix,
    )
    with torch.inference_mode():
        generated = model.generate(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decoder-only GenerationMixin returns only the generated identity when it
    # starts from soft inputs.  A leading BOS is harmless and removed by the
    # tokenizer's special-token handling.
    token_ids = tuple(int(value) for value in generated[0].detach().cpu().tolist())
    response = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return ProceduralQwenGeneration(
        response=response,
        generated_token_ids=token_ids,
        prompt_tokens=prompt_tokens,
        procedure_tokens=int(procedure_prefix.shape[1]),
    )


def procedural_prefix_language_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    target: str,
    procedure_prefix: torch.Tensor,
) -> ProceduralQwenTrainingLoss:
    """Backpropagate a frozen LM consequence only into Angler's soft prefix."""

    if type(target) is not str or not target.strip():
        raise ValueError("target must be non-empty text")
    prompt_inputs, prompt_mask, prompt_tokens = qwen_inputs_with_procedural_prefix(
        model,
        tokenizer,
        prompt,
        procedure_prefix,
    )
    encoded_target = tokenizer(
        target,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(prompt_inputs.device)
    target_ids = encoded_target["input_ids"]
    if target_ids.ndim != 2 or target_ids.shape[0] != 1 or target_ids.shape[1] < 1:
        raise ValueError("target tokenizer output must contain one non-empty row")
    target_embeddings = model.get_input_embeddings()(target_ids)
    inputs = torch.cat((prompt_inputs, target_embeddings), dim=1)
    attention_mask = torch.cat(
        (prompt_mask, encoded_target["attention_mask"]),
        dim=1,
    )
    outputs = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[:2] != inputs.shape[:2]:
        raise RuntimeError("Qwen returned logits with an incompatible shape")
    first_prediction = prompt_inputs.shape[1] - 1
    prediction_logits = logits[
        :, first_prediction : first_prediction + target_ids.shape[1], :
    ]
    loss = F.cross_entropy(
        prediction_logits.reshape(-1, prediction_logits.shape[-1]).float(),
        target_ids.reshape(-1),
    )
    return ProceduralQwenTrainingLoss(
        loss=loss,
        prompt_tokens=prompt_tokens,
        target_tokens=int(target_ids.shape[1]),
        procedure_tokens=int(procedure_prefix.shape[1]),
    )


__all__ = [
    "ProceduralQwenGeneration",
    "ProceduralQwenTrainingLoss",
    "generate_with_procedural_prefix",
    "procedural_prefix_language_loss",
    "qwen_inputs_with_procedural_prefix",
]
