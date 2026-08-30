from __future__ import annotations

import unittest

import torch
from torch import nn

from angler.runtime import (
    generate_with_procedural_prefix,
    procedural_prefix_language_loss,
    qwen_inputs_with_procedural_prefix,
)


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, prompt, **kwargs):
        return _Encoding(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )

    def decode(self, values, **kwargs):
        return "procedure response"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.output = nn.Linear(8, 10, bias=False)
        self.last_inputs = None
        self.last_mask = None

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, *, inputs_embeds, attention_mask, **kwargs):
        self.last_inputs = inputs_embeds
        self.last_mask = attention_mask
        return torch.tensor([[4, 5, 9]], device=inputs_embeds.device)

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        del attention_mask, kwargs
        contextual = inputs_embeds.cumsum(dim=1)
        return type("Output", (), {"logits": self.output(contextual)})()


class ProceduralQwenTests(unittest.TestCase):
    def test_prefix_is_prepended_without_unfreezing_qwen(self) -> None:
        model = _Model()
        model.requires_grad_(False)
        tokenizer = _Tokenizer()
        prefix = torch.randn(1, 4, 8, requires_grad=True)

        inputs, mask, prompt_tokens = qwen_inputs_with_procedural_prefix(
            model, tokenizer, "problem", prefix
        )
        self.assertEqual(inputs.shape, (1, 7, 8))
        self.assertEqual(mask.shape, (1, 7))
        self.assertEqual(prompt_tokens, 3)

        generated = generate_with_procedural_prefix(
            model, tokenizer, "problem", prefix, max_new_tokens=3
        )
        self.assertEqual(generated.response, "procedure response")
        self.assertEqual(generated.procedure_tokens, 4)
        self.assertEqual(generated.generated_token_ids, (4, 5, 9))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_prefix_width_must_match_qwen(self) -> None:
        model = _Model()
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            qwen_inputs_with_procedural_prefix(
                model,
                _Tokenizer(),
                "problem",
                torch.randn(1, 4, 7),
            )

    def test_teacher_forced_loss_reaches_prefix_but_not_qwen(self) -> None:
        model = _Model()
        model.requires_grad_(False)
        prefix = torch.randn(1, 4, 8, requires_grad=True)

        measured = procedural_prefix_language_loss(
            model,
            _Tokenizer(),
            "problem",
            "A B STOP",
            prefix,
        )
        measured.loss.backward()

        self.assertTrue(torch.isfinite(measured.loss))
        self.assertGreater(float(prefix.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(measured.prompt_tokens, 3)
        self.assertEqual(measured.target_tokens, 3)
        self.assertEqual(measured.procedure_tokens, 4)


if __name__ == "__main__":
    unittest.main()
