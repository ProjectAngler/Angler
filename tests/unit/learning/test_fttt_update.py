"""CPU-only tests for the bounded feedback-time LoRA update primitive."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn
from transformers import PretrainedConfig

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.learning.fttt_update import (  # noqa: E402
    BoundedUpdateError,
    LoraUpdateBudget,
    TeacherForcedReflectionBatch,
    propose_bounded_lora_update,
    teacher_forced_causal_loss,
)
from angler.runtime import (  # noqa: E402
    PlasticStateError,
    adapter_tensor_digest,
    attach_single_adapter,
    build_causal_lm_lora_config,
    enumerate_parameter_scopes,
)


class TinyCausalLM(nn.Module):
    """PEFT-compatible tiny model with no checkpoint or network dependency."""

    def __init__(self, *, fail_on_forward: int | None = None) -> None:
        super().__init__()
        self.embed = nn.Embedding(13, 8)
        self.proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 13, bias=False)
        self.config = PretrainedConfig()
        self.config.model_type = "angler_tiny"
        self.fail_on_forward = fail_on_forward
        self.forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask
        if input_ids is None:
            raise ValueError("input_ids is required")
        self.forward_calls += 1
        logits = self.lm_head(torch.tanh(self.proj(self.embed(input_ids))))
        if self.fail_on_forward == self.forward_calls:
            logits = logits * torch.tensor(float("nan"), device=logits.device)
        return SimpleNamespace(logits=logits)

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> dict:
        del args
        return kwargs


def make_model(*, fail_on_forward: int | None = None) -> nn.Module:
    foundation = TinyCausalLM(fail_on_forward=fail_on_forward)
    config = build_causal_lm_lora_config(
        target_modules=("proj",),
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    return attach_single_adapter(foundation, config)


def make_batch(*, target_count: int = 2) -> TeacherForcedReflectionBatch:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, -target_count:] = input_ids[0, -target_count:]
    return TeacherForcedReflectionBatch(
        input_ids=input_ids,
        reflection_labels=labels,
        attention_mask=torch.ones_like(input_ids),
    )


def make_budget(**overrides: object) -> LoraUpdateBudget:
    values: dict[str, object] = {
        "max_steps": 2,
        "max_input_tokens": 10,
        "max_supervised_tokens": 4,
        "learning_rate": 0.05,
        "max_gradient_norm": 0.05,
        "max_adapter_delta_norm": 1.0,
    }
    values.update(overrides)
    return LoraUpdateBudget(**values)  # type: ignore[arg-type]


def foundation_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    inventory = enumerate_parameter_scopes(model)
    named = dict(model.named_parameters())
    return {
        record.name: named[record.name].detach().clone()
        for record in inventory.foundation
    }


class TeacherForcedLossTests(unittest.TestCase):
    def test_loss_uses_only_causally_shifted_revision_targets(self) -> None:
        logits = torch.tensor(
            [
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                    [3.0, 0.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[-100, -100, 2, 0]], dtype=torch.long)
        actual = teacher_forced_causal_loss(logits, labels)
        expected = torch.nn.functional.cross_entropy(
            logits[:, 1:3, :].reshape(-1, 3),
            torch.tensor([2, 0]),
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_loss_rejects_absent_or_invalid_revision_targets(self) -> None:
        logits = torch.zeros((1, 3, 5))
        with self.assertRaises(ValueError):
            teacher_forced_causal_loss(
                logits,
                torch.full((1, 3), -100, dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            teacher_forced_causal_loss(
                logits,
                torch.tensor([[-100, -100, 5]], dtype=torch.long),
            )


class BoundedLoraUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = make_model()
        self.parent_digest = adapter_tensor_digest(self.model)
        self.foundation_before = foundation_snapshot(self.model)

    def assert_foundation_exact(self, model: nn.Module) -> None:
        named = dict(model.named_parameters())
        for name, expected in self.foundation_before.items():
            self.assertTrue(torch.equal(named[name], expected), name)
            self.assertFalse(named[name].requires_grad, name)
            self.assertIsNone(named[name].grad, name)

    def test_candidate_changes_only_lora_and_can_be_retained(self) -> None:
        candidate = propose_bounded_lora_update(
            self.model,
            (make_batch(),),
            make_budget(),
        )
        receipt = candidate.receipt
        self.assertEqual(receipt.parent_adapter_digest, self.parent_digest)
        self.assertNotEqual(receipt.candidate_adapter_digest, self.parent_digest)
        self.assertEqual(receipt.total_input_tokens, 5)
        self.assertEqual(receipt.total_supervised_tokens, 2)
        self.assertEqual(len(receipt.steps), 1)
        self.assertGreater(receipt.steps[0].gradient_norm_before_clip, 0.0)
        self.assertGreater(receipt.candidate_adapter_delta_norm, 0.0)
        self.assertLessEqual(
            receipt.candidate_adapter_delta_norm,
            receipt.max_adapter_delta_norm,
        )
        self.assert_foundation_exact(self.model)

        finalized = candidate.retain()
        self.assertEqual(finalized.disposition, "candidate_retained")
        self.assertEqual(
            finalized.final_adapter_digest,
            receipt.candidate_adapter_digest,
        )
        with self.assertRaises(BoundedUpdateError):
            candidate.reject()

    def test_rejection_restores_exact_parent_adapter_and_behavior(self) -> None:
        inputs = make_batch().input_ids
        with torch.no_grad():
            parent_logits = self.model(input_ids=inputs).logits.detach().clone()

        candidate = propose_bounded_lora_update(
            self.model,
            (make_batch(),),
            make_budget(),
        )
        self.assertNotEqual(adapter_tensor_digest(self.model), self.parent_digest)
        finalized = candidate.reject()

        self.assertEqual(finalized.disposition, "rejected_rolled_back")
        self.assertEqual(finalized.final_adapter_digest, self.parent_digest)
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)
        with torch.no_grad():
            restored_logits = self.model(input_ids=inputs).logits
        self.assertTrue(torch.equal(restored_logits, parent_logits))
        self.assert_foundation_exact(self.model)

    def test_unresolved_context_candidate_rolls_back_on_exception(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "external evaluation failed"):
            with propose_bounded_lora_update(
                self.model,
                (make_batch(),),
                make_budget(),
            ):
                raise RuntimeError("external evaluation failed")
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)
        self.assert_foundation_exact(self.model)

    def test_runtime_failure_after_a_step_rolls_back_exactly(self) -> None:
        torch.manual_seed(101)
        model = make_model(fail_on_forward=2)
        parent_digest = adapter_tensor_digest(model)
        foundation_before = foundation_snapshot(model)

        with self.assertRaises(BoundedUpdateError):
            propose_bounded_lora_update(
                model,
                (make_batch(), make_batch()),
                make_budget(),
            )

        self.assertEqual(adapter_tensor_digest(model), parent_digest)
        named = dict(model.named_parameters())
        for name, expected in foundation_before.items():
            self.assertTrue(torch.equal(named[name], expected), name)

    def test_scope_and_budgets_fail_before_mutation(self) -> None:
        with self.assertRaises(BoundedUpdateError):
            propose_bounded_lora_update(
                self.model,
                (make_batch(), make_batch()),
                make_budget(max_steps=1),
            )
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)

        with self.assertRaisesRegex(BoundedUpdateError, "adapter delta norm"):
            propose_bounded_lora_update(
                self.model,
                (make_batch(),),
                make_budget(max_adapter_delta_norm=1e-12),
            )
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)

        with self.assertRaises(BoundedUpdateError):
            propose_bounded_lora_update(
                self.model,
                (make_batch(),),
                make_budget(max_supervised_tokens=1),
            )
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)

        inventory = enumerate_parameter_scopes(self.model)
        foundation_parameter = dict(self.model.named_parameters())[
            inventory.foundation[0].name
        ]
        foundation_parameter.requires_grad_(True)
        with self.assertRaises(PlasticStateError):
            propose_bounded_lora_update(
                self.model,
                (make_batch(),),
                make_budget(),
            )
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)

    def test_supervision_must_copy_tokens_from_model_generated_text(self) -> None:
        batch = make_batch()
        injected_labels = batch.reflection_labels.clone()
        injected_labels[0, -1] = (injected_labels[0, -1] + 1) % 13
        with self.assertRaisesRegex(ValueError, "supplied model output"):
            propose_bounded_lora_update(
                self.model,
                (
                    TeacherForcedReflectionBatch(
                        input_ids=batch.input_ids,
                        reflection_labels=injected_labels,
                        attention_mask=batch.attention_mask,
                    ),
                ),
                make_budget(),
            )
        self.assertEqual(adapter_tensor_digest(self.model), self.parent_digest)


class BudgetValidationTests(unittest.TestCase):
    def test_invalid_update_budgets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_budget(max_steps=0)
        with self.assertRaises(ValueError):
            make_budget(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            make_budget(max_gradient_norm=0.0)
        with self.assertRaises(ValueError):
            make_budget(max_adapter_delta_norm=float("inf"))


if __name__ == "__main__":
    unittest.main()
