"""Bounded PEFT LoRA state for a frozen causal-language-model substrate.

The helpers in this module deliberately own only adapter attachment, parameter
scope, tensor identity, and local serialization.  They do not choose adapters
from requests, compute updates, or implement inference policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
from typing import Any, Sequence

import torch
from torch import nn

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)

DEFAULT_ADAPTER_NAME = "default"

_DIGEST_DOMAIN = b"project-angler.adapter-tensors.v1\x00"
_FOUNDATION_DIGEST_DOMAIN = b"project-angler.foundation-parameters.v1\x00"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_LORA_PARAMETER_CONTAINERS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


class PlasticStateError(ValueError):
    """Raised when a model violates the single-plastic-state contract."""


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """Stable metadata for one named model parameter."""

    name: str
    numel: int
    dtype: str
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    """Foundation and active-LoRA parameter scopes for one PEFT model."""

    adapter_name: str
    foundation: tuple[ParameterRecord, ...]
    lora: tuple[ParameterRecord, ...]

    @property
    def foundation_numel(self) -> int:
        return sum(parameter.numel for parameter in self.foundation)

    @property
    def lora_numel(self) -> int:
        return sum(parameter.numel for parameter in self.lora)

    @property
    def trainable_foundation_numel(self) -> int:
        return sum(
            parameter.numel
            for parameter in self.foundation
            if parameter.requires_grad
        )

    @property
    def trainable_lora_numel(self) -> int:
        return sum(
            parameter.numel for parameter in self.lora if parameter.requires_grad
        )


def build_causal_lm_lora_config(
    *,
    target_modules: Sequence[str] | str,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    use_rslora: bool = False,
) -> LoraConfig:
    """Build a trainable, LoRA-only PEFT config for a causal language model.

    Target modules and capacity are caller supplied so an execution plan can
    scale the same interface from a tiny test model to a local Qwen checkpoint.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be between 0 and 1")

    if isinstance(target_modules, str):
        normalized_targets: Sequence[str] | str = target_modules.strip()
        if not normalized_targets:
            raise ValueError("target_modules must not be empty")
    else:
        normalized_targets = tuple(target_modules)
        if not normalized_targets or any(
            not isinstance(name, str) or not name.strip()
            for name in normalized_targets
        ):
            raise ValueError("target_modules must contain non-empty names")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=normalized_targets,
        bias="none",
        modules_to_save=None,
        use_rslora=use_rslora,
    )


def attach_single_adapter(
    foundation_model: nn.Module,
    config: LoraConfig,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    autocast_adapter_dtype: bool = True,
) -> PeftModel:
    """Attach and activate exactly one trainable LoRA adapter."""

    if hasattr(foundation_model, "peft_config"):
        raise PlasticStateError("the model already has a PEFT adapter")
    if not adapter_name:
        raise ValueError("adapter_name must not be empty")
    if config.task_type not in (TaskType.CAUSAL_LM, TaskType.CAUSAL_LM.value):
        raise PlasticStateError("the adapter config must target a causal LM")
    if config.bias != "none" or config.modules_to_save:
        raise PlasticStateError(
            "only LoRA tensors may be trainable; bias/modules_to_save are unsupported"
        )

    model = get_peft_model(
        foundation_model,
        config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
    )
    model.set_adapter(adapter_name)
    freeze_foundation_parameters(model, adapter_name=adapter_name)
    return model


def enumerate_parameter_scopes(
    model: nn.Module,
    *,
    adapter_name: str | None = None,
) -> ParameterInventory:
    """Enumerate foundation parameters separately from active LoRA tensors."""

    active_name = _require_one_active_adapter(model, adapter_name)
    lora_parameter_ids = _active_lora_parameter_ids(model, active_name)

    foundation: list[ParameterRecord] = []
    lora: list[ParameterRecord] = []
    for name, parameter in model.named_parameters():
        record = ParameterRecord(
            name=name,
            numel=parameter.numel(),
            dtype=str(parameter.dtype),
            requires_grad=parameter.requires_grad,
        )
        if id(parameter) in lora_parameter_ids:
            lora.append(record)
        else:
            foundation.append(record)

    if not lora:
        raise PlasticStateError(
            f"adapter {active_name!r} exposes no LoRA parameters"
        )
    return ParameterInventory(
        adapter_name=active_name,
        foundation=tuple(foundation),
        lora=tuple(lora),
    )


def freeze_foundation_parameters(
    model: nn.Module,
    *,
    adapter_name: str | None = None,
    trainable_adapter: bool = True,
) -> ParameterInventory:
    """Freeze every foundation parameter and set active LoRA trainability."""

    active_name = _require_one_active_adapter(model, adapter_name)
    lora_parameter_ids = _active_lora_parameter_ids(model, active_name)
    if not lora_parameter_ids:
        raise PlasticStateError(
            f"adapter {active_name!r} exposes no LoRA parameters"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(
            trainable_adapter and id(parameter) in lora_parameter_ids
        )

    return validate_foundation_frozen(
        model,
        adapter_name=active_name,
        require_trainable_lora=trainable_adapter,
    )


def validate_foundation_frozen(
    model: nn.Module,
    *,
    adapter_name: str | None = None,
    require_trainable_lora: bool = True,
) -> ParameterInventory:
    """Validate foundation immutability and, optionally, LoRA trainability."""

    inventory = enumerate_parameter_scopes(model, adapter_name=adapter_name)
    mutable_foundation = [
        parameter.name
        for parameter in inventory.foundation
        if parameter.requires_grad
    ]
    if mutable_foundation:
        raise PlasticStateError(
            "foundation parameters are trainable: " + ", ".join(mutable_foundation)
        )

    if require_trainable_lora:
        frozen_lora = [
            parameter.name for parameter in inventory.lora if not parameter.requires_grad
        ]
        if frozen_lora:
            raise PlasticStateError(
                "active LoRA parameters are frozen: " + ", ".join(frozen_lora)
            )
    return inventory


def adapter_tensor_digest(
    model: nn.Module,
    *,
    adapter_name: str | None = None,
) -> str:
    """Return a deterministic SHA-256 digest of the active adapter tensors."""

    active_name = _require_one_active_adapter(model, adapter_name)
    state = get_peft_model_state_dict(model, adapter_name=active_name)
    if not state:
        raise PlasticStateError(f"adapter {active_name!r} has no tensor state")

    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise PlasticStateError(f"adapter state {name!r} is not a tensor")
        if tensor.layout is not torch.strided:
            raise PlasticStateError(f"adapter state {name!r} is not strided")

        value = tensor.detach().to(device="cpu").contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _digest_field(digest, name.encode("utf-8"))
        _digest_field(digest, str(value.dtype).encode("ascii"))
        digest.update(struct.pack(">Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack(">Q", dimension))
        _digest_field(digest, raw)
    return "sha256:" + digest.hexdigest()


def foundation_tensor_digest(model: nn.Module) -> str:
    """Fingerprint foundation parameters across plain and PEFT-wrapped forms.

    Parameter order, shape, dtype, and exact bytes are committed.  Names are
    intentionally excluded because PEFT wraps selected modules under
    ``base_layer`` while retaining the original foundation parameter order.
    """

    lora_parameter_ids: set[int] = set()
    if hasattr(model, "peft_config"):
        active_name = _require_one_active_adapter(model, None)
        lora_parameter_ids = _active_lora_parameter_ids(model, active_name)

    parameters = tuple(
        parameter
        for parameter in model.parameters()
        if id(parameter) not in lora_parameter_ids
    )
    if not parameters:
        raise PlasticStateError("the model exposes no foundation parameters")

    digest = hashlib.sha256()
    digest.update(_FOUNDATION_DIGEST_DOMAIN)
    digest.update(struct.pack(">Q", len(parameters)))
    for parameter in parameters:
        _digest_tensor(digest, parameter)
    return "sha256:" + digest.hexdigest()


def save_adapter_local(
    model: PeftModel,
    directory: str | os.PathLike[str],
    *,
    adapter_name: str | None = None,
) -> Path:
    """Save the sole adapter to an explicit absolute local directory."""

    active_name = _require_one_active_adapter(model, adapter_name)
    validate_foundation_frozen(
        model,
        adapter_name=active_name,
        require_trainable_lora=False,
    )
    target = _absolute_local_directory(directory, must_exist=False)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(target),
        safe_serialization=True,
        selected_adapters=[active_name],
    )

    candidates = (target, target / active_name)
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate.resolve()
    raise PlasticStateError("PEFT did not emit an adapter_config.json")


def reload_adapter_local(
    foundation_model: nn.Module,
    directory: str | os.PathLike[str],
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    trainable_adapter: bool = True,
    autocast_adapter_dtype: bool = True,
) -> PeftModel:
    """Reload one adapter from an explicit local path without hub resolution."""

    if hasattr(foundation_model, "peft_config"):
        raise PlasticStateError("reload requires an unwrapped foundation model")
    if not adapter_name:
        raise ValueError("adapter_name must not be empty")

    source = _absolute_local_directory(directory, must_exist=True)
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(source / "adapter_config.json")

    model = PeftModel.from_pretrained(
        foundation_model,
        str(source),
        adapter_name=adapter_name,
        is_trainable=trainable_adapter,
        autocast_adapter_dtype=autocast_adapter_dtype,
        local_files_only=True,
    )
    model.set_adapter(adapter_name)
    freeze_foundation_parameters(
        model,
        adapter_name=adapter_name,
        trainable_adapter=trainable_adapter,
    )
    return model


def _require_one_active_adapter(
    model: nn.Module,
    requested_name: str | None,
) -> str:
    configs = getattr(model, "peft_config", None)
    if configs is None:
        raise PlasticStateError("the model has no PEFT adapter")
    adapter_names = tuple(configs)
    if len(adapter_names) != 1:
        raise PlasticStateError(
            f"expected exactly one adapter, found {len(adapter_names)}"
        )

    adapter_name = adapter_names[0]
    if requested_name is not None and requested_name != adapter_name:
        raise PlasticStateError(
            f"requested adapter {requested_name!r} is not the sole adapter "
            f"{adapter_name!r}"
        )

    active = getattr(model, "active_adapters", ())
    active_names = (active,) if isinstance(active, str) else tuple(active)
    if active_names != (adapter_name,):
        raise PlasticStateError(
            f"sole adapter {adapter_name!r} is not uniformly active"
        )
    return adapter_name


def _active_lora_parameter_ids(model: nn.Module, adapter_name: str) -> set[int]:
    parameter_ids: set[int] = set()
    for module in model.modules():
        for container_name in _LORA_PARAMETER_CONTAINERS:
            container = getattr(module, container_name, None)
            if container is None:
                continue
            try:
                present = adapter_name in container
            except TypeError:
                continue
            if not present:
                continue
            value = container[adapter_name]
            if isinstance(value, nn.Parameter):
                parameter_ids.add(id(value))
            elif isinstance(value, nn.Module):
                parameter_ids.update(id(parameter) for parameter in value.parameters())
    return parameter_ids


def _absolute_local_directory(
    directory: str | os.PathLike[str],
    *,
    must_exist: bool,
) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("adapter directory must be an absolute local path")
    if must_exist and not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


def _digest_field(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _digest_tensor(digest: Any, tensor: torch.Tensor) -> None:
    if tensor.layout is not torch.strided:
        raise PlasticStateError("foundation parameter is not strided")
    value = tensor.detach().to(device="cpu").contiguous()
    _digest_field(digest, str(value.dtype).encode("ascii"))
    digest.update(struct.pack(">Q", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack(">Q", dimension))

    raw = value.reshape(-1).view(torch.uint8)
    digest.update(struct.pack(">Q", raw.numel()))
    for start in range(0, raw.numel(), _HASH_CHUNK_BYTES):
        chunk = raw[start : start + _HASH_CHUNK_BYTES]
        digest.update(chunk.numpy().tobytes(order="C"))
