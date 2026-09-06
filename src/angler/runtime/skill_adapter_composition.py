"""Offline exact composition of uniformly active LoRA skill adapters.

The pinned SGLang runtime applies one adapter to each request.  Compatible
skill deltas are therefore compiled into one rank-concatenated adapter without
loading or changing foundation weights.  This module owns artifact integrity
and algebra only; it never trains, selects, or activates an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_WEIGHT_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)
_MAX_COMPONENTS = 8
_MAX_COMBINED_RANK = 64
_MAX_CONFIG_BYTES = 1_048_576
_MAX_JSON_BYTES = 4_194_304
_MAX_ADAPTER_BYTES = 2 * 1024 * 1024 * 1024


class SkillAdapterCompositionError(ValueError):
    """Raised when source artifacts cannot form one exact composition."""


@dataclass(frozen=True, slots=True)
class SkillAdapterComponent:
    """One caller-bound LoRA input."""

    name: str
    path: str | Path
    adapter_model_sha256: str
    adapter_config_sha256: str
    weight: str = "1"


@dataclass(frozen=True, slots=True)
class SkillAdapterComposition:
    """Content identities emitted by :func:`compose_skill_adapters`."""

    output_root: Path
    adapter_path: Path
    manifest_path: Path
    training_result_path: Path
    adapter_model_sha256: str
    adapter_config_sha256: str
    manifest_ref: str
    manifest_sha256: str
    training_result_sha256: str


@dataclass(frozen=True, slots=True)
class _LoadedComponent:
    specification: SkillAdapterComponent
    path: Path
    config: dict[str, object]
    tensors: dict[str, torch.Tensor]
    rank: int
    alpha: int
    normalized_weight: str
    applied_weight: float


def compose_skill_adapters(
    components: Sequence[SkillAdapterComponent],
    *,
    training_result_source: str | Path,
    expected_training_result_sha256: str,
    output_root: str | Path,
) -> SkillAdapterComposition:
    """Compile compatible LoRAs and package an existing-binding-compatible run.

    Every source must use ``alpha/r = 2``.  For source ``i`` the compiler
    writes ``A*=cat(weight_i*A_i, axis=0)`` and
    ``B*=cat(B_i, axis=1)``.  The output also uses ``alpha*/rank*=2``, so
    ``2*B*@A*`` is algebraically the exact sum of the weighted source deltas.

    ``training_result_source`` must be the PASS result that produced exactly
    one component (normally the new residual skill).  The output preserves
    that evidence and adds explicit composition lineage while updating the
    served adapter hashes and rank.  This keeps the run consumable by the
    existing ``SGLangLoRABinding.from_training_run`` validator.
    """

    if isinstance(components, (str, bytes)) or not 2 <= len(components) <= _MAX_COMPONENTS:
        raise SkillAdapterCompositionError(
            f"composition requires 2 through {_MAX_COMPONENTS} components"
        )
    _require_sha256(expected_training_result_sha256, "training result SHA-256")
    loaded = tuple(_load_component(component) for component in components)
    _validate_component_collection(loaded)
    source_result_path = _canonical_regular_file(
        training_result_source,
        label="training result",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if _sha256_file(source_result_path) != expected_training_result_sha256:
        raise SkillAdapterCompositionError("training result SHA-256 differs")
    source_result = _read_json_object(source_result_path, maximum_bytes=_MAX_JSON_BYTES)
    trained_component_name = _validate_training_result_source(source_result, loaded)

    output = _new_output_root(output_root)
    adapter_path = output / "adapter"
    manifest_path = output / "composition-manifest.json"
    result_path = output / "training-result.json"
    try:
        adapter_path.mkdir(mode=0o700)
        composite_tensors, shape_audit = _compose_tensors(loaded)
        total_rank = sum(component.rank for component in loaded)
        composite_config = dict(loaded[0].config)
        composite_config.update(
            {
                "alpha_pattern": {},
                "inference_mode": True,
                "lora_alpha": total_rank * 2,
                "r": total_rank,
                "rank_pattern": {},
                "target_modules": sorted(_target_modules(loaded[0].config)),
            }
        )

        config_path = adapter_path / "adapter_config.json"
        model_path = adapter_path / "adapter_model.safetensors"
        _write_exclusive(config_path, _pretty_json_bytes(composite_config))
        save_file(composite_tensors, str(model_path), metadata={"format": "pt"})
        os.chmod(model_path, 0o600)
        _fsync_file(model_path)
        config_sha256 = _sha256_file(config_path)
        model_sha256 = _sha256_file(model_path)
        _verify_composite_readback(
            model_path,
            expected_tensors=composite_tensors,
            expected_rank=total_rank,
        )

        unsigned_manifest = _manifest_payload(
            loaded=loaded,
            composite_config=composite_config,
            composite_tensors=composite_tensors,
            config_sha256=config_sha256,
            model_sha256=model_sha256,
            shape_audit=shape_audit,
            source_training_result_sha256=expected_training_result_sha256,
            trained_component_name=trained_component_name,
        )
        manifest_ref = "sha256:" + hashlib.sha256(
            _canonical_json_bytes(unsigned_manifest)
        ).hexdigest()
        manifest = {**unsigned_manifest, "manifest_ref": manifest_ref}
        _write_exclusive(manifest_path, _pretty_json_bytes(manifest))
        _verify_manifest(manifest_path, expected_ref=manifest_ref)
        manifest_sha256 = _sha256_file(manifest_path)

        packaged_result = dict(source_result)
        source_rank = packaged_result["rank"]
        packaged_result.update(
            {
                "adapter_artifact_kind": "exact-rank-concatenated-skill-composite",
                "adapter_config_sha256": config_sha256,
                "adapter_model_sha256": model_sha256,
                "composition_manifest_ref": manifest_ref,
                "composition_manifest_sha256": manifest_sha256,
                "composition_source_training_result_sha256": expected_training_result_sha256,
                "composition_trained_component": trained_component_name,
                "latest_checkpoint": str(adapter_path),
                "latest_checkpoint_sha256": model_sha256,
                "rank": total_rank,
                "trained_residual_rank": source_rank,
            }
        )
        _write_exclusive(result_path, _pretty_json_bytes(packaged_result))
        _verify_packaged_result(
            result_path,
            model_sha256=model_sha256,
            config_sha256=config_sha256,
            rank=total_rank,
            manifest_ref=manifest_ref,
            manifest_sha256=manifest_sha256,
        )
        result_sha256 = _sha256_file(result_path)

        for path in (config_path, model_path, manifest_path, result_path):
            os.chmod(path, 0o444)
        _fsync_directory(adapter_path)
        _fsync_directory(output)
        return SkillAdapterComposition(
            output_root=output,
            adapter_path=adapter_path,
            manifest_path=manifest_path,
            training_result_path=result_path,
            adapter_model_sha256=model_sha256,
            adapter_config_sha256=config_sha256,
            manifest_ref=manifest_ref,
            manifest_sha256=manifest_sha256,
            training_result_sha256=result_sha256,
        )
    except BaseException:
        _remove_owned_output(output)
        raise


def _load_component(specification: SkillAdapterComponent) -> _LoadedComponent:
    if not isinstance(specification, SkillAdapterComponent):
        raise TypeError("every component must be SkillAdapterComponent")
    if _NAME_RE.fullmatch(specification.name) is None:
        raise SkillAdapterCompositionError("component name is malformed")
    _require_sha256(specification.adapter_model_sha256, "adapter model SHA-256")
    _require_sha256(specification.adapter_config_sha256, "adapter config SHA-256")
    path = _canonical_existing_directory(specification.path, label="component")
    config_path = _canonical_regular_file(
        path / "adapter_config.json",
        label="adapter config",
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    model_path = _canonical_regular_file(
        path / "adapter_model.safetensors",
        label="adapter model",
        maximum_bytes=_MAX_ADAPTER_BYTES,
    )
    if _sha256_file(config_path) != specification.adapter_config_sha256:
        raise SkillAdapterCompositionError(
            f"component {specification.name!r} config SHA-256 differs"
        )
    if _sha256_file(model_path) != specification.adapter_model_sha256:
        raise SkillAdapterCompositionError(
            f"component {specification.name!r} model SHA-256 differs"
        )
    config = _read_json_object(config_path, maximum_bytes=_MAX_CONFIG_BYTES)
    rank, alpha = _validate_standard_config(config, name=specification.name)
    normalized_weight, applied_weight = _parse_weight(specification.weight)
    with safe_open(str(model_path), framework="pt", device="cpu") as source:
        if source.metadata() != {"format": "pt"}:
            raise SkillAdapterCompositionError(
                f"component {specification.name!r} safetensors metadata differs"
            )
    tensors = dict(load_file(str(model_path), device="cpu"))
    _validate_tensor_map(tensors, rank=rank, name=specification.name)
    return _LoadedComponent(
        specification=specification,
        path=path,
        config=config,
        tensors=tensors,
        rank=rank,
        alpha=alpha,
        normalized_weight=normalized_weight,
        applied_weight=applied_weight,
    )


def _validate_standard_config(
    config: Mapping[str, object], *, name: str
) -> tuple[int, int]:
    exact = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "lora_bias": False,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise SkillAdapterCompositionError(
                f"component {name!r} has unsupported {field}"
            )
    for field in ("alpha_pattern", "rank_pattern", "loftq_config"):
        if config.get(field) not in (None, {}):
            raise SkillAdapterCompositionError(
                f"component {name!r} has unsupported {field}"
            )
    for field in (
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "eva_config",
        "exclude_modules",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "lora_ga_config",
        "megatron_config",
        "modules_to_save",
        "target_parameters",
        "trainable_token_indices",
        "velora_config",
    ):
        if config.get(field) is not None:
            raise SkillAdapterCompositionError(
                f"component {name!r} has unsupported {field}"
            )
    for field in ("use_bdlora", "use_qalora"):
        if config.get(field) not in (None, False):
            raise SkillAdapterCompositionError(
                f"component {name!r} has unsupported {field}"
            )
    dropout = config.get("lora_dropout")
    if (
        type(dropout) not in (int, float)
        or isinstance(dropout, bool)
        or float(dropout) != 0.0
    ):
        raise SkillAdapterCompositionError(
            f"component {name!r} must have zero LoRA dropout"
        )
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if type(rank) is not int or isinstance(rank, bool) or not 1 <= rank <= 64:
        raise SkillAdapterCompositionError(f"component {name!r} rank is invalid")
    if type(alpha) is not int or isinstance(alpha, bool) or alpha != rank * 2:
        raise SkillAdapterCompositionError(
            f"component {name!r} must use alpha/r = 2"
        )
    base = config.get("base_model_name_or_path")
    if type(base) is not str or not base:
        raise SkillAdapterCompositionError(
            f"component {name!r} base model identity is missing"
        )
    _target_modules(config)
    return rank, alpha


def _target_modules(config: Mapping[str, object]) -> frozenset[str]:
    value = config.get("target_modules")
    if type(value) is not list or not value or any(
        type(item) is not str or not item for item in value
    ):
        raise SkillAdapterCompositionError(
            "target_modules must be a non-empty string list"
        )
    if len(set(value)) != len(value):
        raise SkillAdapterCompositionError("target_modules contains duplicates")
    return frozenset(value)


def _validate_tensor_map(
    tensors: Mapping[str, torch.Tensor], *, rank: int, name: str
) -> None:
    if not tensors:
        raise SkillAdapterCompositionError(f"component {name!r} has no tensors")
    keys = set(tensors)
    a_keys = {key for key in keys if key.endswith(_A_SUFFIX)}
    b_keys = {key for key in keys if key.endswith(_B_SUFFIX)}
    if keys != a_keys | b_keys or not a_keys or len(a_keys) != len(b_keys):
        raise SkillAdapterCompositionError(
            f"component {name!r} contains unsupported or unpaired tensors"
        )
    expected_b = {key[: -len(_A_SUFFIX)] + _B_SUFFIX for key in a_keys}
    if b_keys != expected_b:
        raise SkillAdapterCompositionError(
            f"component {name!r} A/B tensor keys differ"
        )
    for a_key in sorted(a_keys):
        b_key = a_key[: -len(_A_SUFFIX)] + _B_SUFFIX
        a = tensors[a_key]
        b = tensors[b_key]
        if (
            a.layout is not torch.strided
            or b.layout is not torch.strided
            or a.device.type != "cpu"
            or b.device.type != "cpu"
            or a.dtype is not torch.float32
            or b.dtype is not torch.float32
            or a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != rank
            or b.shape[1] != rank
        ):
            raise SkillAdapterCompositionError(
                f"component {name!r} tensor shape or dtype differs at {a_key}"
            )
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise SkillAdapterCompositionError(
                f"component {name!r} contains non-finite data at {a_key}"
            )


def _validate_component_collection(components: Sequence[_LoadedComponent]) -> None:
    names = [item.specification.name for item in components]
    paths = [item.path for item in components]
    hashes = [item.specification.adapter_model_sha256 for item in components]
    if len(set(names)) != len(names):
        raise SkillAdapterCompositionError("component names must be unique")
    if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise SkillAdapterCompositionError("component artifacts must be distinct")
    total_rank = sum(item.rank for item in components)
    if total_rank > _MAX_COMBINED_RANK:
        raise SkillAdapterCompositionError(
            f"combined rank {total_rank} exceeds {_MAX_COMBINED_RANK}"
        )
    first = components[0]
    first_keys = set(first.tensors)
    first_targets = _target_modules(first.config)
    for item in components[1:]:
        for field in (
            "base_model_name_or_path",
            "revision",
            "peft_type",
            "peft_version",
            "task_type",
        ):
            if item.config.get(field) != first.config.get(field):
                raise SkillAdapterCompositionError(
                    f"component {item.specification.name!r} {field} differs"
                )
        if _target_modules(item.config) != first_targets:
            raise SkillAdapterCompositionError(
                f"component {item.specification.name!r} target modules differ"
            )
        if set(item.tensors) != first_keys:
            raise SkillAdapterCompositionError(
                f"component {item.specification.name!r} tensor key set differs"
            )


def _validate_training_result_source(
    result: Mapping[str, object], components: Sequence[_LoadedComponent]
) -> str:
    if result.get("schema") != "jenny2.qlora-training.v1" or result.get("status") != "PASS":
        raise SkillAdapterCompositionError("source training result is not a passed QLoRA run")
    if result.get("base_revision") is None or result.get("text_only_training") is not True:
        raise SkillAdapterCompositionError("source training result base lineage differs")
    if result.get("lora_scaling") != "standard-alpha-over-r":
        raise SkillAdapterCompositionError("source training result LoRA scaling differs")
    rank = result.get("rank")
    if type(rank) is not int or isinstance(rank, bool):
        raise SkillAdapterCompositionError("source training result rank is invalid")
    result_targets = result.get("target_modules")
    if type(result_targets) is not list or set(result_targets) != _target_modules(
        components[0].config
    ):
        raise SkillAdapterCompositionError("source training result target modules differ")
    model_sha = result.get("adapter_model_sha256")
    config_sha = result.get("adapter_config_sha256")
    matches = [
        item
        for item in components
        if item.specification.adapter_model_sha256 == model_sha
        and item.specification.adapter_config_sha256 == config_sha
        and item.rank == rank
    ]
    if len(matches) != 1:
        raise SkillAdapterCompositionError(
            "source training result must bind exactly one component"
        )
    if type(result.get("held_out_teacher_forced_evaluation")) is not dict:
        raise SkillAdapterCompositionError("source training result lacks held-out evaluation")
    steps = result.get("steps_completed")
    if type(steps) is not int or steps < 1 or steps != result.get("max_steps"):
        raise SkillAdapterCompositionError("source training result is incomplete")
    for field in (
        "training_sha256",
        "evaluation_sha256",
        "curriculum_manifest_sha256",
        "base_config_sha256",
        "base_index_sha256",
    ):
        _require_sha256(result.get(field), f"source training result {field}")
    return matches[0].specification.name


def _compose_tensors(
    components: Sequence[_LoadedComponent],
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    first = components[0]
    a_keys = sorted(key for key in first.tensors if key.endswith(_A_SUFFIX))
    composite: dict[str, torch.Tensor] = {}
    audit: list[dict[str, object]] = []
    for a_key in a_keys:
        prefix = a_key[: -len(_A_SUFFIX)]
        b_key = prefix + _B_SUFFIX
        input_width = first.tensors[a_key].shape[1]
        output_width = first.tensors[b_key].shape[0]
        a_blocks: list[torch.Tensor] = []
        b_blocks: list[torch.Tensor] = []
        slices: list[dict[str, object]] = []
        rank_offset = 0
        for item in components:
            a = item.tensors[a_key]
            b = item.tensors[b_key]
            if a.shape[1] != input_width or b.shape[0] != output_width:
                raise SkillAdapterCompositionError(
                    f"component {item.specification.name!r} dimensions differ at {prefix}"
                )
            weighted_a = (
                a.clone() if item.applied_weight == 1.0 else a * item.applied_weight
            )
            a_blocks.append(weighted_a)
            b_blocks.append(b.clone())
            slices.append(
                {
                    "component": item.specification.name,
                    "rank_start": rank_offset,
                    "rank_stop": rank_offset + item.rank,
                    "requested_weight": item.normalized_weight,
                    "applied_float32": repr(item.applied_weight),
                    "applied_float32_hex": item.applied_weight.hex(),
                }
            )
            rank_offset += item.rank
        combined_a = torch.cat(a_blocks, dim=0).contiguous()
        combined_b = torch.cat(b_blocks, dim=1).contiguous()
        if not bool(torch.isfinite(combined_a).all()) or not bool(
            torch.isfinite(combined_b).all()
        ):
            raise SkillAdapterCompositionError(
                f"composition produced non-finite data at {prefix}"
            )
        offset = 0
        for item, a_block in zip(components, a_blocks):
            stop = offset + item.rank
            if not torch.equal(combined_a[offset:stop], a_block) or not torch.equal(
                combined_b[:, offset:stop], item.tensors[b_key]
            ):
                raise SkillAdapterCompositionError(
                    f"exact rank-slice audit failed at {prefix}"
                )
            offset = stop
        composite[a_key] = combined_a
        composite[b_key] = combined_b
        audit.append(
            {
                "module": prefix,
                "dtype": "torch.float32",
                "input_width": input_width,
                "output_width": output_width,
                "composite_a_shape": list(combined_a.shape),
                "composite_b_shape": list(combined_b.shape),
                "component_slices": slices,
                "exact_rank_slice_audit": True,
            }
        )
    return {key: composite[key] for key in sorted(composite)}, audit


def _manifest_payload(
    *,
    loaded: Sequence[_LoadedComponent],
    composite_config: Mapping[str, object],
    composite_tensors: Mapping[str, torch.Tensor],
    config_sha256: str,
    model_sha256: str,
    shape_audit: Sequence[Mapping[str, object]],
    source_training_result_sha256: str,
    trained_component_name: str,
) -> dict[str, object]:
    return {
        "schema": "jenny2.skill-adapter-composition.v1",
        "status": "PASS",
        "algorithm": "exact-lora-rank-concatenation-v1",
        "foundation_weights_changed": False,
        "query_conditioned_adapter_selection": False,
        "formula": {
            "source_delta": "weight_i * (alpha_i / rank_i) * B_i @ A_i",
            "composite_a": "concat(weight_i * A_i, axis=0)",
            "composite_b": "concat(B_i, axis=1)",
            "composite_delta": "2 * composite_B @ composite_A",
            "required_source_alpha_over_rank": 2,
            "composite_alpha_over_rank": 2,
        },
        "base": {
            "base_model_name_or_path": loaded[0].config["base_model_name_or_path"],
            "revision": loaded[0].config.get("revision"),
            "peft_type": loaded[0].config["peft_type"],
            "peft_version": loaded[0].config.get("peft_version"),
            "task_type": loaded[0].config["task_type"],
            "target_modules": sorted(_target_modules(loaded[0].config)),
        },
        "source_training_result_sha256": source_training_result_sha256,
        "trained_component": trained_component_name,
        "components": [
            {
                "ordinal": index,
                "name": item.specification.name,
                "path": str(item.path),
                "adapter_model_sha256": item.specification.adapter_model_sha256,
                "adapter_config_sha256": item.specification.adapter_config_sha256,
                "rank": item.rank,
                "alpha": item.alpha,
                "alpha_over_rank": 2,
                "requested_weight": item.normalized_weight,
                "applied_float32": repr(item.applied_weight),
                "applied_float32_hex": item.applied_weight.hex(),
                "tensor_count": len(item.tensors),
            }
            for index, item in enumerate(loaded)
        ],
        "composite": {
            "adapter_model_sha256": model_sha256,
            "adapter_config_sha256": config_sha256,
            "rank": composite_config["r"],
            "alpha": composite_config["lora_alpha"],
            "alpha_over_rank": 2,
            "tensor_count": len(composite_tensors),
            "parameter_count": sum(tensor.numel() for tensor in composite_tensors.values()),
            "raw_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in composite_tensors.values()
            ),
        },
        "shape_audit": list(shape_audit),
        "limitations": [
            "Artifact compatibility and exact LoRA algebra do not establish behavioral quality.",
            "Rank and runtime LoRA memory grow with the sum of component ranks.",
            "Activation requires strict-load, retention, context, throughput, and whole-system checks.",
        ],
    }


def _verify_composite_readback(
    model_path: Path,
    *,
    expected_tensors: Mapping[str, torch.Tensor],
    expected_rank: int,
) -> None:
    with safe_open(str(model_path), framework="pt", device="cpu") as source:
        if source.metadata() != {"format": "pt"}:
            raise SkillAdapterCompositionError("composite safetensors metadata differs")
        if set(source.keys()) != set(expected_tensors):
            raise SkillAdapterCompositionError("composite tensor key set changed on write")
        for key in source.keys():
            observed = source.get_tensor(key)
            if not torch.equal(observed, expected_tensors[key]):
                raise SkillAdapterCompositionError(
                    f"composite tensor changed on write at {key}"
                )
            if key.endswith(_A_SUFFIX) and observed.shape[0] != expected_rank:
                raise SkillAdapterCompositionError("composite A rank differs")
            if key.endswith(_B_SUFFIX) and observed.shape[1] != expected_rank:
                raise SkillAdapterCompositionError("composite B rank differs")


def _verify_manifest(path: Path, *, expected_ref: str) -> None:
    payload = _read_json_object(path, maximum_bytes=_MAX_JSON_BYTES)
    observed_ref = payload.pop("manifest_ref", None)
    recomputed = "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if observed_ref != expected_ref or recomputed != expected_ref:
        raise SkillAdapterCompositionError("composition manifest identity differs")


def _verify_packaged_result(
    path: Path,
    *,
    model_sha256: str,
    config_sha256: str,
    rank: int,
    manifest_ref: str,
    manifest_sha256: str,
) -> None:
    payload = _read_json_object(path, maximum_bytes=_MAX_JSON_BYTES)
    expected = {
        "schema": "jenny2.qlora-training.v1",
        "status": "PASS",
        "adapter_artifact_kind": "exact-rank-concatenated-skill-composite",
        "adapter_model_sha256": model_sha256,
        "adapter_config_sha256": config_sha256,
        "rank": rank,
        "composition_manifest_ref": manifest_ref,
        "composition_manifest_sha256": manifest_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise SkillAdapterCompositionError(
                f"packaged training result {field} differs"
            )


def _parse_weight(value: object) -> tuple[str, float]:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 64
        or _WEIGHT_RE.fullmatch(value) is None
    ):
        raise SkillAdapterCompositionError(
            "component weight must be a bounded decimal string"
        )
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise SkillAdapterCompositionError("component weight is malformed") from exc
    if not decimal.is_finite() or decimal == 0 or abs(decimal) > 16:
        raise SkillAdapterCompositionError(
            "component weight must be finite, nonzero, and at most 16"
        )
    applied = float(torch.tensor(float(decimal), dtype=torch.float32).item())
    if not math.isfinite(applied) or applied == 0.0:
        raise SkillAdapterCompositionError(
            "component weight is not representable as float32"
        )
    normalized = format(decimal, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized, applied


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SkillAdapterCompositionError(f"{label} is malformed")


def _canonical_existing_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SkillAdapterCompositionError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SkillAdapterCompositionError(f"{label} path does not exist") from exc
    if str(path) != str(resolved) or path.is_symlink() or not path.is_dir():
        raise SkillAdapterCompositionError(
            f"{label} path must be a canonical real directory"
        )
    return resolved


def _canonical_regular_file(
    value: str | Path, *, label: str, maximum_bytes: int
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SkillAdapterCompositionError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        stat = path.lstat()
    except OSError as exc:
        raise SkillAdapterCompositionError(f"{label} is unavailable") from exc
    if (
        str(path) != str(resolved)
        or path.is_symlink()
        or not path.is_file()
        or stat.st_nlink != 1
    ):
        raise SkillAdapterCompositionError(
            f"{label} must be a canonical private regular file"
        )
    if stat.st_size > maximum_bytes:
        raise SkillAdapterCompositionError(f"{label} exceeds its byte boundary")
    return resolved


def _new_output_root(value: str | Path) -> Path:
    output = Path(value)
    if not output.is_absolute() or str(output) != str(output.resolve()):
        raise SkillAdapterCompositionError(
            "output root must be canonical and absolute"
        )
    parent = _canonical_existing_directory(output.parent, label="output parent")
    output = parent / output.name
    if not output.name or os.path.lexists(output):
        raise FileExistsError(output)
    output.mkdir(mode=0o700, exist_ok=False)
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    _canonical_regular_file(path, label="JSON artifact", maximum_bytes=maximum_bytes)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillAdapterCompositionError(f"JSON artifact is malformed: {path}") from exc
    if type(payload) is not dict:
        raise SkillAdapterCompositionError(f"JSON artifact must be an object: {path}")
    return payload


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("artifact write did not advance")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_output(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        return
    shutil.rmtree(path)


__all__ = [
    "SkillAdapterComponent",
    "SkillAdapterComposition",
    "SkillAdapterCompositionError",
    "compose_skill_adapters",
]
