"""Runtime primitives for Project Angler's single plastic LoRA state."""

from .qwen_peft import (
    DEFAULT_ADAPTER_NAME,
    ParameterInventory,
    ParameterRecord,
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

__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "ParameterInventory",
    "ParameterRecord",
    "PlasticStateError",
    "adapter_tensor_digest",
    "attach_single_adapter",
    "build_causal_lm_lora_config",
    "enumerate_parameter_scopes",
    "foundation_tensor_digest",
    "freeze_foundation_parameters",
    "reload_adapter_local",
    "save_adapter_local",
    "validate_foundation_frozen",
]
