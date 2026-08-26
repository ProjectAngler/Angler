from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from angler.runtime import encode_detached_segments, freeze_knowledge_model


class DetachedKnowledgeEncodingTests(unittest.TestCase):
    def test_segments_remain_independent_and_gradient_free(self) -> None:
        model = _ToyKnowledgeModel(width=6)
        tokenizer = _ToyTokenizer()
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            encode_detached_segments(model, tokenizer, ("alpha", "beta"))

        freeze_knowledge_model(model)
        together = encode_detached_segments(
            model,
            tokenizer,
            ("alpha", "beta"),
            batch_size=2,
        )
        separately = torch.cat(
            (
                encode_detached_segments(model, tokenizer, ("alpha",)),
                encode_detached_segments(model, tokenizer, ("beta",)),
            ),
            dim=0,
        )

        self.assertTrue(torch.equal(together, separately))
        self.assertEqual(together.shape, (2, 6))
        self.assertEqual(together.dtype, torch.bfloat16)
        self.assertFalse(together.requires_grad)
        self.assertIsNone(together.grad_fn)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


class _ToyBatch(dict):
    def to(self, device: torch.device) -> "_ToyBatch":
        return _ToyBatch(
            {name: tensor.to(device) for name, tensor in self.items()}
        )


class _ToyTokenizer:
    def __call__(
        self,
        texts,
        *,
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool,
    ) -> _ToyBatch:
        if return_tensors != "pt" or not padding or not add_special_tokens:
            raise AssertionError("unexpected tokenizer options")
        lengths = [len(text.encode("utf-8")) + 1 for text in texts]
        maximum = max(lengths)
        input_ids = torch.zeros((len(texts), maximum), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            values = [1, *text.encode("utf-8")]
            input_ids[row, : len(values)] = torch.tensor(values)
            attention_mask[row, : len(values)] = 1
        return _ToyBatch(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )


class _ToyKnowledgeModel(nn.Module):
    def __init__(self, *, width: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(width))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        if not output_hidden_states or use_cache or not return_dict:
            raise AssertionError("unexpected model options")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        values = input_ids.float().unsqueeze(-1) + positions.view(1, -1, 1)
        hidden = values * self.anchor.view(1, 1, -1)
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))


if __name__ == "__main__":
    unittest.main()
