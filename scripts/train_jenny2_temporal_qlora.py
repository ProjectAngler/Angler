#!/usr/bin/env python3
"""Bounded QLoRA training for one uniformly active Jenny 2.0 adapter."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random
import re
import resource
import time

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


BASE_REVISION = "e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_ADAPTER_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_adapter_name(value: str, *, field: str) -> str:
    if _ADAPTER_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{field} must match [a-z][a-z0-9_]{{0,63}} so PEFT can bind it exactly"
        )
    return value


def _validate_exact_local_adapter(
    path: Path,
    *,
    expected_model_sha256: str | None,
    expected_config_sha256: str | None,
) -> tuple[str, str]:
    expected = (expected_model_sha256, expected_config_sha256)
    if any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in expected
    ):
        raise ValueError("continuing an adapter requires both exact SHA-256 values")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("initial adapter must be a real local directory")
    model_path = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    observed_model = _sha(model_path)
    observed_config = _sha(config_path)
    if observed_model != expected_model_sha256:
        raise ValueError("initial adapter tensor SHA-256 differs")
    if observed_config != expected_config_sha256:
        raise ValueError("initial adapter config SHA-256 differs")
    return observed_model, observed_config


def _adapter_parameter_items(model, adapter_name: str):
    """Return exact named PEFT tensors for one adapter component."""

    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if adapter_name in name.split(".")
    )


def _adapter_parameter_fingerprint(model, adapter_name: str) -> str:
    items = _adapter_parameter_items(model, adapter_name)
    if not items:
        raise RuntimeError(f"adapter {adapter_name!r} has no named parameters")
    digest = hashlib.sha256()
    digest.update(b"jenny2.peft-adapter-parameters.v1\0")
    for name, parameter in sorted(items):
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _activate_adapter_components(
    model, adapter_names: tuple[str, ...], *, inference_mode: bool
) -> None:
    """Activate one or more additive PEFT components at the tuner boundary."""

    if not adapter_names:
        raise ValueError("at least one adapter component is required")
    if len(set(adapter_names)) != len(adapter_names):
        raise ValueError("adapter components must be unique")
    setter = getattr(getattr(model, "base_model", None), "set_adapter", None)
    if setter is None:
        raise RuntimeError("PEFT base_model does not expose multi-adapter activation")
    selection: str | list[str]
    selection = adapter_names[0] if len(adapter_names) == 1 else list(adapter_names)
    setter(selection, inference_mode=inference_mode)


def _enforce_residual_skill_trainability(
    model, *, parent_adapter_name: str, skill_adapter_name: str
) -> dict[str, object]:
    """Freeze every tensor except the named residual skill component."""

    if parent_adapter_name == skill_adapter_name:
        raise ValueError("parent and skill adapter names must differ")
    parent_items = _adapter_parameter_items(model, parent_adapter_name)
    skill_items = _adapter_parameter_items(model, skill_adapter_name)
    if not parent_items:
        raise RuntimeError("the frozen parent component has no parameters")
    if not skill_items:
        raise RuntimeError("the trainable skill component has no parameters")
    skill_ids = {id(parameter) for _name, parameter in skill_items}
    for _name, parameter in model.named_parameters():
        parameter.requires_grad_(id(parameter) in skill_ids)
    trainable_items = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    unexpected = [
        name
        for name, _parameter in trainable_items
        if skill_adapter_name not in name.split(".")
    ]
    if unexpected:
        raise RuntimeError(
            "non-skill tensors remained trainable: " + ", ".join(unexpected[:3])
        )
    if {id(parameter) for _name, parameter in trainable_items} != skill_ids:
        raise RuntimeError("not every skill tensor is trainable")
    names = sorted(name for name, _parameter in trainable_items)
    return {
        "parent_tensor_count": len(parent_items),
        "parent_trainable_tensor_count": sum(
            int(parameter.requires_grad) for _name, parameter in parent_items
        ),
        "skill_tensor_count": len(skill_items),
        "skill_trainable_tensor_count": len(trainable_items),
        "trainable_parameters": sum(
            parameter.numel() for _name, parameter in trainable_items
        ),
        "trainable_name_set_sha256": hashlib.sha256(
            ("\n".join(names) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def _skill_lora_config(rank: int) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        target_modules=list(TARGET_MODULES),
        bias="none",
        # The pinned live server applies alpha/r but not PEFT's rsLoRA marker.
        use_rslora=False,
    )


def _encode(tokenizer, row: dict[str, object], maximum_tokens: int):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )["input_ids"]
    if full[: len(prompt)] != prompt:
        raise RuntimeError("assistant answer is not a suffix of the live chat template")
    if len(full) > maximum_tokens:
        raise RuntimeError(f"lesson has {len(full)} tokens, above {maximum_tokens}")
    labels = [-100] * len(prompt) + full[len(prompt) :]
    if not any(label != -100 for label in labels):
        raise RuntimeError("lesson contains no assistant target tokens")
    return (
        torch.tensor([full], dtype=torch.long),
        torch.ones((1, len(full)), dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


def _supervised_window(labels: torch.Tensor) -> tuple[int, torch.Tensor]:
    supervised_tokens = int((labels != -100).sum().item())
    logits_to_keep = supervised_tokens + 1
    shift_labels = functional.pad(labels, (0, 1), value=-100)[:, 1:][
        :, -logits_to_keep:
    ].contiguous()
    return logits_to_keep, shift_labels


def _evaluate_expected_suffixes(
    *, model, tokenizer, rows: list[dict[str, object]], maximum_tokens: int, input_device
) -> dict[str, object]:
    aggregates: dict[str, dict[str, float | int]] = {}
    model.eval()
    with torch.inference_mode():
        for row in rows:
            input_ids, attention_mask, labels = _encode(tokenizer, row, maximum_tokens)
            logits_to_keep, shift_labels = _supervised_window(labels)
            batch = {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
                "labels": labels.to(input_device),
                "logits_to_keep": logits_to_keep,
                "shift_labels": shift_labels.to(input_device),
                "use_cache": False,
            }
            with model.disable_adapter():
                base = model(**batch)
            adapted = model(**batch)
            target = batch["shift_labels"].to(adapted.logits.device)
            mask = target != -100
            token_count = int(mask.sum().item())
            if token_count < 1:
                raise RuntimeError("evaluation row contains no supervised tokens")
            skill = str(row.get("skill", "unspecified"))
            group = aggregates.setdefault(
                skill,
                {
                    "examples": 0,
                    "tokens": 0,
                    "base_loss_sum": 0.0,
                    "adapted_loss_sum": 0.0,
                    "base_correct": 0,
                    "adapted_correct": 0,
                },
            )
            group["examples"] += 1
            group["tokens"] += token_count
            group["base_loss_sum"] += float(base.loss) * token_count
            group["adapted_loss_sum"] += float(adapted.loss) * token_count
            group["base_correct"] += int(
                ((base.logits.argmax(dim=-1) == target) & mask).sum().item()
            )
            group["adapted_correct"] += int(
                ((adapted.logits.argmax(dim=-1) == target) & mask).sum().item()
            )
    summary: dict[str, object] = {}
    totals = {
        "examples": 0,
        "tokens": 0,
        "base_loss_sum": 0.0,
        "adapted_loss_sum": 0.0,
        "base_correct": 0,
        "adapted_correct": 0,
    }
    for skill, group in sorted(aggregates.items()):
        tokens = int(group["tokens"])
        summary[skill] = {
            "examples": int(group["examples"]),
            "tokens": tokens,
            "base_mean_loss": float(group["base_loss_sum"]) / tokens,
            "adapted_mean_loss": float(group["adapted_loss_sum"]) / tokens,
            "base_token_accuracy": int(group["base_correct"]) / tokens,
            "adapted_token_accuracy": int(group["adapted_correct"]) / tokens,
        }
        for key in totals:
            totals[key] += group[key]
    tokens = int(totals["tokens"])
    return {
        "examples": int(totals["examples"]),
        "tokens": tokens,
        "base_mean_loss": float(totals["base_loss_sum"]) / tokens,
        "adapted_mean_loss": float(totals["adapted_loss_sum"]) / tokens,
        "base_token_accuracy": int(totals["base_correct"]) / tokens,
        "adapted_token_accuracy": int(totals["adapted_correct"]) / tokens,
        "by_skill": summary,
    }


def _summarize_variant_groups(
    aggregates: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    totals: dict[str, float | int] = {
        "examples": 0,
        "tokens": 0,
        "loss_sum": 0.0,
        "correct": 0,
    }
    for skill, group in sorted(aggregates.items()):
        tokens = int(group["tokens"])
        summary[skill] = {
            "examples": int(group["examples"]),
            "tokens": tokens,
            "mean_loss": float(group["loss_sum"]) / tokens,
            "token_accuracy": int(group["correct"]) / tokens,
        }
        for key in totals:
            totals[key] += group[key]
    tokens = int(totals["tokens"])
    if tokens < 1:
        raise RuntimeError("variant evaluation produced no supervised tokens")
    return {
        "examples": int(totals["examples"]),
        "tokens": tokens,
        "mean_loss": float(totals["loss_sum"]) / tokens,
        "token_accuracy": int(totals["correct"]) / tokens,
        "by_skill": summary,
    }


def _evaluate_adapter_variants(
    *,
    model,
    tokenizer,
    rows: list[dict[str, object]],
    maximum_tokens: int,
    input_device,
    variants: dict[str, tuple[str, ...]],
    restore_adapters: tuple[str, ...],
    trainable_adapter: str | None,
    parent_adapter: str | None = None,
) -> dict[str, object]:
    """Teacher-force the same sealed rows under explicit adapter removals."""

    if not variants:
        raise ValueError("at least one evaluation variant is required")
    aggregates: dict[str, dict[str, dict[str, float | int]]] = {
        variant: {} for variant in variants
    }
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for row in rows:
                input_ids, attention_mask, labels = _encode(
                    tokenizer, row, maximum_tokens
                )
                logits_to_keep, shift_labels = _supervised_window(labels)
                batch = {
                    "input_ids": input_ids.to(input_device),
                    "attention_mask": attention_mask.to(input_device),
                    "labels": labels.to(input_device),
                    "logits_to_keep": logits_to_keep,
                    "shift_labels": shift_labels.to(input_device),
                    "use_cache": False,
                }
                target = batch["shift_labels"]
                mask = target != -100
                token_count = int(mask.sum().item())
                if token_count < 1:
                    raise RuntimeError("evaluation row contains no supervised tokens")
                skill = str(row.get("skill", "unspecified"))
                for variant, adapter_names in variants.items():
                    if adapter_names:
                        _activate_adapter_components(
                            model, adapter_names, inference_mode=True
                        )
                        context = nullcontext()
                    else:
                        context = model.disable_adapter()
                    with context:
                        outputs = model(**batch)
                    observed_target = target.to(outputs.logits.device)
                    observed_mask = observed_target != -100
                    group = aggregates[variant].setdefault(
                        skill,
                        {
                            "examples": 0,
                            "tokens": 0,
                            "loss_sum": 0.0,
                            "correct": 0,
                        },
                    )
                    group["examples"] += 1
                    group["tokens"] += token_count
                    group["loss_sum"] += float(outputs.loss) * token_count
                    group["correct"] += int(
                        (
                            (outputs.logits.argmax(dim=-1) == observed_target)
                            & observed_mask
                        )
                        .sum()
                        .item()
                    )
    finally:
        _activate_adapter_components(
            model, restore_adapters, inference_mode=trainable_adapter is None
        )
        if trainable_adapter is not None:
            if parent_adapter is None:
                raise RuntimeError("trainable restoration requires the parent adapter")
            _enforce_residual_skill_trainability(
                model,
                parent_adapter_name=parent_adapter,
                skill_adapter_name=trainable_adapter,
            )
        if was_training:
            model.train()
    return {
        "schema": "jenny2.adapter-removal-teacher-forced-evaluation.v1",
        "variant_components": {
            name: list(adapter_names) for name, adapter_names in variants.items()
        },
        "variants": {
            name: _summarize_variant_groups(groups)
            for name, groups in aggregates.items()
        },
    }


def _memory() -> dict[str, object]:
    return {
        "ram_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda": [
            {
                "index": index,
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(index),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _prepare_kbit_without_bulk_upcast(model):
    """Freeze the foundation without PEFT's blanket BF16-to-FP32 conversion.

    The generic helper upcasts every non-Params4bit tensor, including this
    architecture's multi-gigabyte language head.  Only small normalization
    parameters need the conventional FP32 stability cast.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "norm" in name.lower() and parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.to(torch.float32)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        embedding = model.get_input_embeddings()
        embedding.register_forward_hook(
            lambda _module, _input, output: output.requires_grad_(True)
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def _dual_gpu_device_map(layer_count: int) -> dict[str, int]:
    """Keep the complete quantized text model on two GPUs without TP training."""

    if layer_count < 2:
        raise ValueError("the text model must expose at least two layers")
    split = 20 if layer_count == 64 else layer_count // 2
    mapping: dict[str, int] = {"model.embed_tokens": 0}
    mapping.update({f"model.layers.{index}": 0 for index in range(split)})
    mapping.update(
        {f"model.layers.{index}": 1 for index in range(split, layer_count)}
    )
    mapping.update({"model.norm": 1, "model.rotary_emb": 1, "lm_head": 0})
    return mapping


def _save_selected_adapter(model, output_root: Path, adapter_name: str | None) -> Path:
    if adapter_name is None:
        model.save_pretrained(output_root, safe_serialization=True)
        adapter_path = output_root
    else:
        model.save_pretrained(
            output_root,
            safe_serialization=True,
            selected_adapters=[adapter_name],
        )
        nested_path = output_root / adapter_name
        if not nested_path.is_dir():
            raise RuntimeError("PEFT did not emit the selected named adapter")
        for source in nested_path.iterdir():
            target = output_root / source.name
            if target.exists():
                raise RuntimeError(f"selected adapter output collides with {target.name}")
            source.replace(target)
        nested_path.rmdir()
        adapter_path = output_root
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise RuntimeError("saved adapter tensor artifact is absent")
    if not (adapter_path / "adapter_config.json").is_file():
        raise RuntimeError("saved adapter configuration is absent")
    return adapter_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/opt/angler/models/Qwen3.8-27B-BF16")
    parser.add_argument(
        "--initial-adapter",
        help="existing exact LoRA adapter to continue without discarding learned skills",
    )
    parser.add_argument("--expected-initial-adapter-sha256")
    parser.add_argument("--expected-initial-adapter-config-sha256")
    parser.add_argument(
        "--residual-skill-name",
        help=(
            "add and train this separate rank-8 skill component while freezing "
            "--initial-adapter"
        ),
    )
    parser.add_argument(
        "--parent-adapter-name",
        default="prior_competence",
        help="stable in-memory component name for the frozen initial adapter",
    )
    parser.add_argument("--train", default="/opt/angler/training/jenny2-temporal-v1/train.jsonl")
    parser.add_argument("--eval")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument(
        "--retention-eval",
        help=(
            "optional sealed prior-skill evaluation comparing parent-only with "
            "parent-plus-skill"
        ),
    )
    parser.add_argument("--retention-eval-limit", type=int, default=0)
    parser.add_argument("--curriculum-manifest")
    parser.add_argument("--profile", default="temporal-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--gpu-weight-gib", type=int, default=7)
    parser.add_argument("--cpu-memory-gib", type=int, default=40)
    parser.add_argument(
        "--placement",
        choices=("balanced-offload", "dual-gpu-native"),
        default="balanced-offload",
    )
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 10_000:
        raise ValueError("max-steps must be 1 through 10000")
    if not 0 <= args.eval_limit <= 10_000:
        raise ValueError("eval-limit must be zero through 10000")
    if not 0 <= args.retention_eval_limit <= 10_000:
        raise ValueError("retention-eval-limit must be zero through 10000")
    if not 256 <= args.maximum_tokens <= 4096:
        raise ValueError("maximum-tokens must be 256 through 4096")
    if not 1 <= args.rank <= 64:
        raise ValueError("rank must be 1 through 64")
    residual_skill_name = args.residual_skill_name
    if residual_skill_name is not None:
        residual_skill_name = _validate_adapter_name(
            residual_skill_name, field="residual-skill-name"
        )
        _validate_adapter_name(
            args.parent_adapter_name, field="parent-adapter-name"
        )
        if residual_skill_name == args.parent_adapter_name:
            raise ValueError("parent-adapter-name and residual-skill-name must differ")
        if args.rank != 8:
            raise ValueError("residual skill adapters use an exact rank of 8")
        if args.initial_adapter is None:
            raise ValueError("a residual skill requires --initial-adapter as its parent")
    elif args.retention_eval is not None:
        raise ValueError("retention-eval is available only for a residual skill run")
    if not 1 <= args.gradient_accumulation <= 128:
        raise ValueError("gradient-accumulation must be 1 through 128")
    if not 0 <= args.checkpoint_every <= args.max_steps:
        raise ValueError("checkpoint-every must be zero through max-steps")
    if not 4 <= args.gpu_weight_gib <= 12:
        raise ValueError("gpu-weight-gib must be 4 through 12")
    if not 16 <= args.cpu_memory_gib <= 52:
        raise ValueError("cpu-memory-gib must be 16 through 52")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("this training shape requires exactly two visible GPUs")

    base = Path(args.base).resolve()
    initial_adapter = (
        None if args.initial_adapter is None else Path(args.initial_adapter).resolve()
    )
    expected_initial_hashes = (
        args.expected_initial_adapter_sha256,
        args.expected_initial_adapter_config_sha256,
    )
    if initial_adapter is None:
        if any(value is not None for value in expected_initial_hashes):
            raise ValueError("initial adapter hashes require --initial-adapter")
    else:
        _validate_exact_local_adapter(
            initial_adapter,
            expected_model_sha256=expected_initial_hashes[0],
            expected_config_sha256=expected_initial_hashes[1],
        )
    training_path = Path(args.train).resolve()
    evaluation_path = None if args.eval is None else Path(args.eval).resolve()
    retention_evaluation_path = (
        None
        if args.retention_eval is None
        else Path(args.retention_eval).resolve()
    )
    manifest_path = (
        None
        if args.curriculum_manifest is None
        else Path(args.curriculum_manifest).resolve()
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "training-result.json"
    result: dict[str, object] = {
        "schema": "jenny2.qlora-training.v1",
        "training_profile": args.profile,
        "status": "STARTED",
        "base_revision": BASE_REVISION,
        "base_config_sha256": _sha(base / "config.json"),
        "base_index_sha256": _sha(base / "model.safetensors.index.json"),
        "training_sha256": _sha(training_path),
        "evaluation_sha256": (
            None if evaluation_path is None else _sha(evaluation_path)
        ),
        "evaluation_limit": args.eval_limit,
        "retention_evaluation_sha256": (
            None
            if retention_evaluation_path is None
            else _sha(retention_evaluation_path)
        ),
        "retention_evaluation_limit": args.retention_eval_limit,
        "curriculum_manifest_sha256": (
            None if manifest_path is None else _sha(manifest_path)
        ),
        "target_modules": list(TARGET_MODULES),
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "maximum_tokens": args.maximum_tokens,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "checkpoint_every": args.checkpoint_every,
        "loss_logits_scope": "supervised-assistant-suffix-only",
        "hub_kernels_required": True,
        "lora_scaling": "standard-alpha-over-r",
        "runtime_versions": {
            package: metadata.version(package)
            for package in (
                "accelerate",
                "bitsandbytes",
                "kernels",
                "peft",
                "torch",
                "transformers",
            )
        },
        "gpu_weight_gib": args.gpu_weight_gib,
        "cpu_memory_gib": args.cpu_memory_gib,
        "placement": args.placement,
        "text_only_training": True,
        "initial_adapter_path": (
            None if initial_adapter is None else str(initial_adapter)
        ),
        "initial_adapter_model_sha256": expected_initial_hashes[0],
        "initial_adapter_config_sha256": expected_initial_hashes[1],
        "adapter_training_mode": (
            "residual-skill"
            if residual_skill_name is not None
            else "new" if initial_adapter is None else "continue"
        ),
        "parent_adapter_name": (
            args.parent_adapter_name if residual_skill_name is not None else None
        ),
        "residual_skill_name": residual_skill_name,
        "active_adapter_components": (
            [args.parent_adapter_name, residual_skill_name]
            if residual_skill_name is not None
            else ["default"]
        ),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        random.seed(2026090203)
        torch.manual_seed(2026090203)
        rows = [json.loads(line) for line in training_path.read_text(encoding="utf-8").splitlines()]
        random.shuffle(rows)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        full_config = AutoConfig.from_pretrained(base, local_files_only=True)
        text_config = full_config.text_config
        text_config.architectures = ["Qwen3_5ForCausalLM"]
        device_map: str | dict[str, int]
        maximum_memory: dict[int | str, str] | None
        if args.placement == "dual-gpu-native":
            device_map = _dual_gpu_device_map(text_config.num_hidden_layers)
            maximum_memory = None
        else:
            device_map = "balanced"
            maximum_memory = {
                0: f"{args.gpu_weight_gib}GiB",
                1: f"{args.gpu_weight_gib}GiB",
                "cpu": f"{args.cpu_memory_gib}GiB",
            }
        model = AutoModelForCausalLM.from_pretrained(
            base,
            config=text_config,
            key_mapping={r"^model\.language_model\.": "model."},
            local_files_only=True,
            quantization_config=quantization,
            device_map=device_map,
            max_memory=maximum_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_kernels=True,
        )
        model = _prepare_kbit_without_bulk_upcast(model)
        model.config.use_cache = False
        result["memory_after_foundation_load"] = _memory()
        residual_trainability_audit = None
        parent_parameter_fingerprint_before = None
        skill_parameter_fingerprint_before = None
        if initial_adapter is None:
            model = get_peft_model(
                model,
                _skill_lora_config(args.rank),
            )
        elif residual_skill_name is None:
            model = PeftModel.from_pretrained(
                model,
                initial_adapter,
                is_trainable=True,
                local_files_only=True,
            )
            adapter_config = model.peft_config.get("default")
            if adapter_config is None:
                raise RuntimeError("continued adapter has no default configuration")
            continued_targets = set(adapter_config.target_modules or ())
            if (
                adapter_config.r != args.rank
                or adapter_config.lora_alpha != args.rank * 2
                or bool(getattr(adapter_config, "use_rslora", False))
                or continued_targets != set(TARGET_MODULES)
            ):
                raise RuntimeError("continued adapter configuration differs")
        else:
            model = PeftModel.from_pretrained(
                model,
                initial_adapter,
                adapter_name=args.parent_adapter_name,
                is_trainable=False,
                local_files_only=True,
            )
            parent_config = model.peft_config.get(args.parent_adapter_name)
            if parent_config is None:
                raise RuntimeError("frozen parent adapter has no named configuration")
            parent_targets = set(parent_config.target_modules or ())
            if (
                parent_config.r != args.rank
                or parent_config.lora_alpha != args.rank * 2
                or bool(getattr(parent_config, "use_rslora", False))
                or parent_targets != set(TARGET_MODULES)
            ):
                raise RuntimeError("frozen parent adapter configuration differs")
            model.add_adapter(
                residual_skill_name,
                _skill_lora_config(args.rank),
            )
            _activate_adapter_components(
                model,
                (args.parent_adapter_name, residual_skill_name),
                inference_mode=False,
            )
            residual_trainability_audit = _enforce_residual_skill_trainability(
                model,
                parent_adapter_name=args.parent_adapter_name,
                skill_adapter_name=residual_skill_name,
            )
            parent_parameter_fingerprint_before = _adapter_parameter_fingerprint(
                model, args.parent_adapter_name
            )
            skill_parameter_fingerprint_before = _adapter_parameter_fingerprint(
                model, residual_skill_name
            )
            result.update(
                {
                    "residual_trainability_audit": residual_trainability_audit,
                    "parent_parameter_fingerprint_before": (
                        parent_parameter_fingerprint_before
                    ),
                    "skill_parameter_fingerprint_before": (
                        skill_parameter_fingerprint_before
                    ),
                    "composition_operator": "peft-additive-active-adapter-sum",
                }
            )
        model.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        foundation = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if trainable <= 0 or foundation <= trainable:
            raise RuntimeError("PEFT parameter separation failed")
        result["memory_after_adapter_attach"] = _memory()
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.learning_rate,
        )
        input_device = model.get_input_embeddings().weight.device
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        step_seconds: list[float] = []
        if (
            evaluation_path is not None
            and initial_adapter is not None
            and residual_skill_name is None
        ):
            parent_rows = [
                json.loads(line)
                for line in evaluation_path.read_text(encoding="utf-8").splitlines()
            ]
            if args.eval_limit:
                parent_rows = parent_rows[: args.eval_limit]
            if not parent_rows:
                raise RuntimeError("parent adapter evaluation is empty")
            result["parent_teacher_forced_evaluation"] = _evaluate_expected_suffixes(
                model=model,
                tokenizer=tokenizer,
                rows=parent_rows,
                maximum_tokens=args.maximum_tokens,
                input_device=input_device,
            )
            model.train()
        for step in range(args.max_steps):
            step_started = time.monotonic()
            accumulated = 0.0
            for offset in range(args.gradient_accumulation):
                row = rows[(step * args.gradient_accumulation + offset) % len(rows)]
                input_ids, attention_mask, labels = _encode(
                    tokenizer, row, args.maximum_tokens
                )
                # The live-format context is long, but only the assistant suffix
                # is supervised.  Qwen's full causal-LM loss would otherwise
                # materialize FP32 logits for every prompt token.  Keep the
                # prompt-final predictor plus the supervised suffix and provide
                # the exactly aligned shifted targets explicitly.
                supervised_tokens = int((labels != -100).sum().item())
                logits_to_keep = supervised_tokens + 1
                shift_labels = functional.pad(labels, (0, 1), value=-100)[
                    :, 1:
                ][:, -logits_to_keep:].contiguous()
                outputs = model(
                    input_ids=input_ids.to(input_device),
                    attention_mask=attention_mask.to(input_device),
                    labels=labels.to(input_device),
                    logits_to_keep=logits_to_keep,
                    shift_labels=shift_labels.to(input_device),
                    use_cache=False,
                )
                loss = outputs.loss / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("training loss is not finite")
                loss.backward()
                accumulated += float(loss.detach())
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(accumulated)
            step_seconds.append(time.monotonic() - step_started)
            print(
                f"STEP={step + 1} LOSS={accumulated:.8f} "
                f"SECONDS={step_seconds[-1]:.6f}",
                flush=True,
            )
            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                checkpoint_root = output / f"checkpoint-step-{step + 1:05d}"
                checkpoint_path = _save_selected_adapter(
                    model, checkpoint_root, residual_skill_name
                )
                result.update(
                    {
                        "status": "RUNNING",
                        "elapsed_seconds": time.monotonic() - started,
                        "steps_completed": step + 1,
                        "losses": losses,
                        "step_seconds": step_seconds,
                        "latest_checkpoint": str(checkpoint_path),
                        "latest_checkpoint_sha256": _sha(
                            checkpoint_path / "adapter_model.safetensors"
                        ),
                        "memory": _memory(),
                    }
                )
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        adapter_path = _save_selected_adapter(
            model, output / "adapter", residual_skill_name
        )
        tokenizer.save_pretrained(adapter_path)
        if residual_skill_name is not None:
            parent_parameter_fingerprint_after = _adapter_parameter_fingerprint(
                model, args.parent_adapter_name
            )
            if (
                parent_parameter_fingerprint_after
                != parent_parameter_fingerprint_before
            ):
                raise RuntimeError("frozen parent adapter changed during skill training")
            result.update(
                {
                    "parent_parameter_fingerprint_after": (
                        parent_parameter_fingerprint_after
                    ),
                    "parent_parameter_fingerprint_unchanged": True,
                    "skill_parameter_fingerprint_after": (
                        _adapter_parameter_fingerprint(model, residual_skill_name)
                    ),
                    "standalone_skill_adapter_path": str(adapter_path),
                }
            )
        held_out_evaluation = None
        if evaluation_path is not None:
            evaluation_rows = [
                json.loads(line)
                for line in evaluation_path.read_text(encoding="utf-8").splitlines()
            ]
            if args.eval_limit:
                evaluation_rows = evaluation_rows[: args.eval_limit]
            if not evaluation_rows:
                raise RuntimeError("held-out evaluation is empty")
            result.update(
                {
                    "status": "EVALUATING",
                    "elapsed_seconds": time.monotonic() - started,
                    "steps_completed": len(losses),
                    "losses": losses,
                    "step_seconds": step_seconds,
                }
            )
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if residual_skill_name is None:
                held_out_evaluation = _evaluate_expected_suffixes(
                    model=model,
                    tokenizer=tokenizer,
                    rows=evaluation_rows,
                    maximum_tokens=args.maximum_tokens,
                    input_device=input_device,
                )
            else:
                held_out_evaluation = _evaluate_adapter_variants(
                    model=model,
                    tokenizer=tokenizer,
                    rows=evaluation_rows,
                    maximum_tokens=args.maximum_tokens,
                    input_device=input_device,
                    variants={
                        "base": (),
                        "parent_only": (args.parent_adapter_name,),
                        "skill_only": (residual_skill_name,),
                        "parent_plus_skill": (
                            args.parent_adapter_name,
                            residual_skill_name,
                        ),
                    },
                    restore_adapters=(
                        args.parent_adapter_name,
                        residual_skill_name,
                    ),
                    trainable_adapter=None,
                )
        retention_evaluation = None
        if retention_evaluation_path is not None:
            if residual_skill_name is None:
                raise RuntimeError("retention evaluation requires a residual skill")
            retention_rows = [
                json.loads(line)
                for line in retention_evaluation_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            if args.retention_eval_limit:
                retention_rows = retention_rows[: args.retention_eval_limit]
            if not retention_rows:
                raise RuntimeError("retention evaluation is empty")
            retention_evaluation = _evaluate_adapter_variants(
                model=model,
                tokenizer=tokenizer,
                rows=retention_rows,
                maximum_tokens=args.maximum_tokens,
                input_device=input_device,
                variants={
                    "parent_only": (args.parent_adapter_name,),
                    "parent_plus_skill": (
                        args.parent_adapter_name,
                        residual_skill_name,
                    ),
                },
                restore_adapters=(args.parent_adapter_name, residual_skill_name),
                trainable_adapter=None,
            )
        adapter_config_sha256 = _sha(adapter_path / "adapter_config.json")
        adapter_model_sha256 = _sha(adapter_path / "adapter_model.safetensors")
        skill_lineage_sha256 = None
        skill_lineage_path = None
        if residual_skill_name is not None:
            skill_lineage_path = adapter_path / "jenny2-skill-lineage.json"
            lineage = {
                "schema": "jenny2.skill-adapter-lineage.v1",
                "base": {
                    "revision": BASE_REVISION,
                    "config_sha256": result["base_config_sha256"],
                    "index_sha256": result["base_index_sha256"],
                },
                "frozen_parent": {
                    "component_name": args.parent_adapter_name,
                    "adapter_model_sha256": expected_initial_hashes[0],
                    "adapter_config_sha256": expected_initial_hashes[1],
                    "parameter_fingerprint": parent_parameter_fingerprint_before,
                },
                "skill": {
                    "component_name": residual_skill_name,
                    "adapter_model_sha256": adapter_model_sha256,
                    "adapter_config_sha256": adapter_config_sha256,
                    "rank": args.rank,
                    "alpha": args.rank * 2,
                    "target_modules": list(TARGET_MODULES),
                },
                "composition": {
                    "operator": "peft-additive-active-adapter-sum",
                    "ordered_components": [
                        args.parent_adapter_name,
                        residual_skill_name,
                    ],
                },
                "curriculum": {
                    "train_sha256": result["training_sha256"],
                    "evaluation_sha256": result["evaluation_sha256"],
                    "retention_evaluation_sha256": result[
                        "retention_evaluation_sha256"
                    ],
                    "manifest_sha256": result["curriculum_manifest_sha256"],
                },
                "training": {
                    "seed": 2026090203,
                    "steps": len(losses),
                    "learning_rate": args.learning_rate,
                    "gradient_accumulation": args.gradient_accumulation,
                    "trainer_sha256": _sha(Path(__file__).resolve()),
                    "peft_version": result["runtime_versions"]["peft"],
                },
            }
            skill_lineage_path.write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            skill_lineage_sha256 = _sha(skill_lineage_path)
        result.update(
            {
                "status": "PASS",
                "elapsed_seconds": time.monotonic() - started,
                "steps_completed": len(losses),
                "losses": losses,
                "step_seconds": step_seconds,
                "trainable_parameters": trainable,
                "frozen_parameters": foundation,
                "memory": _memory(),
                "adapter_config_sha256": adapter_config_sha256,
                "adapter_model_sha256": adapter_model_sha256,
                "adapter_path": str(adapter_path),
                "skill_lineage_path": (
                    None if skill_lineage_path is None else str(skill_lineage_path)
                ),
                "skill_lineage_sha256": skill_lineage_sha256,
                "held_out_teacher_forced_evaluation": held_out_evaluation,
                "retention_teacher_forced_evaluation": retention_evaluation,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("TRAINING_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        result.update(
            {
                "status": "FAIL",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "memory": _memory(),
            }
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
