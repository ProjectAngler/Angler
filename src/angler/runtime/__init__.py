"""Frozen-model connectors and reversible plastic-state runtime primitives."""

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
from .qwen_knowledge import encode_detached_segments, freeze_knowledge_model
from .situated_qwen import (
    LocalQwenIO,
    SituatedQwenCoordinator,
    SituatedEvidenceSelection,
    SituatedQwenTurn,
    build_qwen_prompt,
    package_outcome_feedback,
    select_situated_evidence,
)
from .procedural_qwen import (
    ProceduralQwenGeneration,
    generate_with_procedural_prefix,
    qwen_inputs_with_procedural_prefix,
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
    "encode_detached_segments",
    "freeze_knowledge_model",
    "LocalQwenIO",
    "SituatedQwenCoordinator",
    "SituatedEvidenceSelection",
    "SituatedQwenTurn",
    "build_qwen_prompt",
    "package_outcome_feedback",
    "select_situated_evidence",
    "ProceduralQwenGeneration",
    "generate_with_procedural_prefix",
    "qwen_inputs_with_procedural_prefix",
]
