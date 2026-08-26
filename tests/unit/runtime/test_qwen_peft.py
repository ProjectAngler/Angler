"""CPU-only tests for the Phase 1 PEFT plastic-state substrate."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from transformers import PretrainedConfig

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.runtime import (  # noqa: E402 - explicit src path for local tests
    PlasticStateError,
    adapter_tensor_digest,
    attach_single_adapter,
    build_causal_lm_lora_config,
    enumerate_parameter_scopes,
    foundation_tensor_digest,
    freeze_foundation_parameters,
    reload_adapter_local,
    save_adapter_local,
    validate_foundation_frozen,
)


class TinyCausalLM(nn.Module):
    """Small PEFT-compatible causal-LM-shaped module; it loads no checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.config = PretrainedConfig()
        self.config.model_type = "angler_tiny"

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids is required")
        return self.proj(input_ids)

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> dict:
        del args
        return kwargs


def make_config():
    return build_causal_lm_lora_config(
        target_modules=("proj",),
        rank=2,
        alpha=4,
        dropout=0.0,
    )


class LoraConfigTests(unittest.TestCase):
    def test_builder_is_causal_lm_and_resource_configurable(self) -> None:
        config = build_causal_lm_lora_config(
            target_modules=("proj",),
            rank=3,
            alpha=9,
            dropout=0.25,
            use_rslora=True,
        )
        self.assertEqual(str(config.task_type), "TaskType.CAUSAL_LM")
        self.assertEqual(config.r, 3)
        self.assertEqual(config.lora_alpha, 9)
        self.assertEqual(config.lora_dropout, 0.25)
        self.assertEqual(set(config.target_modules), {"proj"})
        self.assertEqual(config.bias, "none")
        self.assertIsNone(config.modules_to_save)
        self.assertTrue(config.use_rslora)

    def test_builder_rejects_invalid_capacity(self) -> None:
        with self.assertRaises(ValueError):
            build_causal_lm_lora_config(target_modules=("proj",), rank=0)
        with self.assertRaises(ValueError):
            build_causal_lm_lora_config(target_modules=(), rank=2)
        with self.assertRaises(ValueError):
            build_causal_lm_lora_config(
                target_modules=("proj",), rank=2, dropout=1.1
            )


class ParameterScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.foundation = TinyCausalLM()
        self.foundation_weight = self.foundation.proj.weight.detach().clone()
        self.foundation_digest = foundation_tensor_digest(self.foundation)
        self.model = attach_single_adapter(self.foundation, make_config())

    def test_attach_has_one_active_adapter_and_only_lora_is_trainable(self) -> None:
        inventory = validate_foundation_frozen(self.model)
        self.assertEqual(tuple(self.model.peft_config), ("default",))
        self.assertEqual(tuple(self.model.active_adapters), ("default",))
        self.assertGreater(inventory.foundation_numel, 0)
        self.assertGreater(inventory.lora_numel, 0)
        self.assertEqual(inventory.trainable_foundation_numel, 0)
        self.assertEqual(inventory.trainable_lora_numel, inventory.lora_numel)
        self.assertEqual(foundation_tensor_digest(self.model), self.foundation_digest)
        self.assertTrue(
            torch.equal(
                self.model.base_model.model.proj.base_layer.weight,
                self.foundation_weight,
            )
        )

    def test_existing_adapter_cannot_be_wrapped_or_selected_as_a_second(self) -> None:
        with self.assertRaises(PlasticStateError):
            attach_single_adapter(self.model, make_config())
        with self.assertRaises(PlasticStateError):
            enumerate_parameter_scopes(self.model, adapter_name="another")

    def test_foundation_validation_detects_and_refreezes_base_parameter(self) -> None:
        foundation_parameter = next(
            parameter
            for name, parameter in self.model.named_parameters()
            if ".lora_" not in name
        )
        foundation_parameter.requires_grad_(True)
        with self.assertRaises(PlasticStateError):
            validate_foundation_frozen(self.model)

        repaired = freeze_foundation_parameters(self.model)
        self.assertEqual(repaired.trainable_foundation_numel, 0)
        self.assertEqual(repaired.trainable_lora_numel, repaired.lora_numel)


class AdapterIdentityAndRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        foundation = TinyCausalLM()
        self.foundation_state = copy.deepcopy(foundation.state_dict())
        self.foundation_digest = foundation_tensor_digest(foundation)
        self.model = attach_single_adapter(foundation, make_config())
        with torch.no_grad():
            for index, parameter in enumerate(
                parameter
                for name, parameter in self.model.named_parameters()
                if ".lora_" in name
            ):
                parameter.fill_(0.125 * (index + 1))

    def test_digest_is_repeatable_and_changes_with_tensor_content(self) -> None:
        first = adapter_tensor_digest(self.model)
        self.assertEqual(first, adapter_tensor_digest(self.model))
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")

        lora_parameter = next(
            parameter
            for name, parameter in self.model.named_parameters()
            if ".lora_" in name
        )
        with torch.no_grad():
            lora_parameter.add_(1.0)
        self.assertNotEqual(first, adapter_tensor_digest(self.model))

    def test_safe_local_save_reload_preserves_digest_and_behavior(self) -> None:
        expected_digest = adapter_tensor_digest(self.model)
        inputs = torch.tensor(
            [[0.25, -0.5, 0.75, 1.0], [-1.0, 0.5, 0.25, -0.25]]
        )
        with torch.no_grad():
            expected_output = self.model(input_ids=inputs)

        with tempfile.TemporaryDirectory() as temporary:
            saved = save_adapter_local(self.model, Path(temporary) / "adapter")
            self.assertTrue((saved / "adapter_config.json").is_file())
            self.assertTrue((saved / "adapter_model.safetensors").is_file())

            restored_foundation = TinyCausalLM()
            restored_foundation.load_state_dict(self.foundation_state)
            restored = reload_adapter_local(restored_foundation, saved)
            self.assertEqual(adapter_tensor_digest(restored), expected_digest)
            self.assertEqual(
                foundation_tensor_digest(restored), self.foundation_digest
            )
            validate_foundation_frozen(restored)
            with torch.no_grad():
                actual_output = restored(input_ids=inputs)
            self.assertTrue(torch.equal(actual_output, expected_output))

    def test_save_and_reload_require_explicit_absolute_paths(self) -> None:
        with self.assertRaises(ValueError):
            save_adapter_local(self.model, "relative-adapter")
        with self.assertRaises(ValueError):
            reload_adapter_local(TinyCausalLM(), "relative-adapter")


if __name__ == "__main__":
    unittest.main()
