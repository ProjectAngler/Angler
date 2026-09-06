#!/usr/bin/env python3
"""Run the frozen Jenny reading-residual calibration and one-shot final gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from angler.runtime.skill_adapter_calibration import (
    CompiledCandidate,
    EvaluationRow,
    EvaluationTarget,
    PRODUCTION_PROTOCOL,
    RowSufficientStatistic,
    compiler_contract_payload,
    evaluator_contract_payload,
    run_skill_adapter_calibration,
    validate_calibration_runtime_inputs,
)


EVALUATION_MICROBATCH_SIZE = 2


@dataclass(frozen=True, slots=True)
class _EncodedEvaluationRow:
    original_index: int
    row: EvaluationRow
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def _length_bucketed_batches(
    rows: Sequence[_EncodedEvaluationRow],
    *,
    batch_size: int = EVALUATION_MICROBATCH_SIZE,
) -> tuple[tuple[_EncodedEvaluationRow, ...], ...]:
    if type(batch_size) is not int or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("evaluation microbatch size must be positive")
    if any(len(row.input_ids) != len(row.labels) or not row.input_ids for row in rows):
        raise ValueError("encoded evaluation row lengths differ")
    ordered = sorted(rows, key=lambda row: (len(row.input_ids), row.original_index))
    return tuple(
        tuple(ordered[start : start + batch_size])
        for start in range(0, len(ordered), batch_size)
    )


def _right_padded_batch_payload(
    rows: Sequence[_EncodedEvaluationRow], *, pad_token_id: int
) -> dict[str, object]:
    if not rows:
        raise ValueError("evaluation microbatch is empty")
    if type(pad_token_id) is not int or isinstance(pad_token_id, bool):
        raise ValueError("tokenizer pad token id is absent")
    maximum_length = max(len(row.input_ids) for row in rows)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    supervised_tokens: list[int] = []
    for row in rows:
        if len(row.input_ids) != len(row.labels) or not row.input_ids:
            raise ValueError("encoded evaluation row lengths differ")
        padding = maximum_length - len(row.input_ids)
        padded_labels = list(row.labels) + [-100] * padding
        tokens = sum(value != -100 for value in padded_labels)
        if tokens < 1:
            raise ValueError("evaluation row has no assistant target tokens")
        input_ids.append(list(row.input_ids) + [pad_token_id] * padding)
        attention_mask.append([1] * len(row.input_ids) + [0] * padding)
        labels.append(padded_labels)
        supervised_tokens.append(tokens)
    full_shift_labels = [row_labels[1:] + [-100] for row_labels in labels]
    supervised_indices = [
        index
        for index in range(maximum_length)
        if any(row_labels[index] != -100 for row_labels in full_shift_labels)
    ]
    if not supervised_indices:
        raise ValueError("evaluation microbatch has no supervised prediction")
    first_supervised_index = supervised_indices[0]
    logits_to_keep = maximum_length - first_supervised_index
    shift_labels = [
        row_labels[first_supervised_index:] for row_labels in full_shift_labels
    ]
    if any(
        sum(value != -100 for value in row_labels) != expected
        for row_labels, expected in zip(shift_labels, supervised_tokens)
    ):
        raise ValueError("shifted batch labels lost supervised tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "shift_labels": shift_labels,
        "supervised_tokens": supervised_tokens,
        "logits_to_keep": logits_to_keep,
    }


def _row_statistics_from_logits(
    *,
    rows: Sequence[_EncodedEvaluationRow],
    logits,
    shift_labels,
    functional,
    torch_module,
) -> tuple[tuple[int, RowSufficientStatistic], ...]:
    if logits.ndim != 3 or shift_labels.ndim != 2:
        raise RuntimeError("batched evaluation tensor rank differs")
    if tuple(logits.shape[:2]) != tuple(shift_labels.shape):
        raise RuntimeError("batched evaluation logits/labels shape differs")
    if logits.shape[0] != len(rows):
        raise RuntimeError("batched evaluation row count differs")
    targets = shift_labels.to(logits.device)
    mask = targets != -100
    losses = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)
    predictions = logits.argmax(dim=-1)
    result: list[tuple[int, RowSufficientStatistic]] = []
    for batch_index, row in enumerate(rows):
        row_mask = mask[batch_index]
        supervised_tokens = int(row_mask.sum().item())
        correct_tokens = int(
            ((predictions[batch_index] == targets[batch_index]) & row_mask)
            .sum()
            .item()
        )
        nll_sum = float(
            losses[batch_index][row_mask]
            .sum(dtype=torch_module.float64)
            .item()
        )
        if supervised_tokens < 1 or not math.isfinite(nll_sum) or nll_sum < 0:
            raise RuntimeError("model emitted invalid per-row evaluation statistics")
        result.append(
            (
                row.original_index,
                RowSufficientStatistic(
                    identity=row.row.identity,
                    skill=row.row.skill,
                    boundary=row.row.boundary,
                    supervised_tokens=supervised_tokens,
                    correct_tokens=correct_tokens,
                    nll_sum=nll_sum,
                ),
            )
        )
    return tuple(result)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _evaluator_identity() -> dict[str, object]:
    # This is collected without importing torch or touching CUDA and is sealed
    # before the first metric. The entrypoint hash binds the concrete backend
    # implementation as well as its load/evaluation settings.
    import angler.runtime.skill_adapter_calibration as calibration_module

    entrypoint = Path(__file__).resolve()
    orchestrator = Path(calibration_module.__file__).resolve()
    composition_spec = find_spec("angler.runtime.skill_adapter_composition")
    if composition_spec is None or composition_spec.origin is None:
        raise RuntimeError("exact composition module identity is unavailable")
    composition_module = Path(composition_spec.origin).resolve()
    return {
        "schema": "jenny2.skill-adapter-evaluator-identity.v1",
        "entrypoint": {"path": str(entrypoint), "sha256": _sha(entrypoint)},
        "orchestrator": {
            "path": str(orchestrator),
            "sha256": _sha(orchestrator),
        },
        "python_runtime": sys.version,
        "dependencies": {
            name: version(name)
            for name in (
                "accelerate",
                "bitsandbytes",
                "numpy",
                "peft",
                "safetensors",
                "torch",
                "transformers",
            )
        },
        "contract": evaluator_contract_payload(PRODUCTION_PROTOCOL),
        "compiler": {
            "schema": "jenny2.skill-adapter-compiler-identity.v1",
            "implementation_files": [
                {"path": str(entrypoint), "sha256": _sha(entrypoint)},
                {
                    "path": str(composition_module),
                    "sha256": _sha(composition_module),
                },
            ],
            "dependencies": {
                name: version(name) for name in ("numpy", "safetensors", "torch")
            },
            "contract": compiler_contract_payload(PRODUCTION_PROTOCOL),
        },
    }


def _compiler(weight: str, output: Path) -> CompiledCandidate:
    # Delayed so importing this CLI cannot compile artifacts or initialize torch.
    from angler.runtime.skill_adapter_composition import (
        SkillAdapterComponent,
        compose_skill_adapters,
    )

    protocol = PRODUCTION_PROTOCOL
    composition = compose_skill_adapters(
        (
            SkillAdapterComponent(
                name=protocol.parent.name,
                path=protocol.parent.path,
                adapter_model_sha256=protocol.parent.adapter_model_sha256,
                adapter_config_sha256=protocol.parent.adapter_config_sha256,
                weight="1",
            ),
            SkillAdapterComponent(
                name=protocol.residual.name,
                path=protocol.residual.path,
                adapter_model_sha256=protocol.residual.adapter_model_sha256,
                adapter_config_sha256=protocol.residual.adapter_config_sha256,
                weight=weight,
            ),
        ),
        training_result_source=protocol.source_training_result.path,
        expected_training_result_sha256=protocol.source_training_result.sha256,
        output_root=output,
    )
    config = json.loads(
        (composition.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    rank = config.get("r")
    if type(rank) is not int:
        raise RuntimeError("compiled adapter rank is absent")
    return CompiledCandidate(
        weight=weight,
        output_root=composition.output_root,
        adapter_path=composition.adapter_path,
        manifest_path=composition.manifest_path,
        training_result_path=composition.training_result_path,
        adapter_model_sha256=composition.adapter_model_sha256,
        adapter_config_sha256=composition.adapter_config_sha256,
        manifest_ref=composition.manifest_ref,
        manifest_sha256=composition.manifest_sha256,
        training_result_sha256=composition.training_result_sha256,
        rank=rank,
    )


class _LocalNF4DualGPUBackend:
    """Exact-adapter teacher-forced evaluator; construction loads the model."""

    def __init__(self) -> None:
        # Re-hash every indexed BF16 shard, tokenizer/config input, and both
        # source adapters in the same process immediately before model load.
        validate_calibration_runtime_inputs(PRODUCTION_PROTOCOL)
        # Heavy imports and all CUDA interaction remain behind main's factory.
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._torch = torch
        self._functional = functional
        self._parent_name = "parent_control"
        self._candidate_name: str | None = None
        self._candidate_key: tuple[str, str] | None = None
        protocol = PRODUCTION_PROTOCOL
        if torch.cuda.device_count() != 2:
            raise RuntimeError("calibration requires exactly two visible GPUs")
        tokenizer = AutoTokenizer.from_pretrained(
            protocol.base_root, local_files_only=True
        )
        full_config = AutoConfig.from_pretrained(
            protocol.base_root, local_files_only=True
        )
        text_config = full_config.text_config
        if text_config.num_hidden_layers != 64:
            raise RuntimeError("frozen dual-GPU placement requires 64 text layers")
        text_config.architectures = ["Qwen3_5ForCausalLM"]
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            protocol.base_root,
            config=text_config,
            key_mapping={r"^model\.language_model\.": "model."},
            local_files_only=True,
            quantization_config=quantization,
            device_map=_dual_gpu_device_map(text_config.num_hidden_layers),
            max_memory=None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_kernels=False,
        )
        model.config.use_cache = False
        parent = EvaluationTarget(
            name="parent_control",
            weight=None,
            adapter_path=protocol.parent.path,
            adapter_model_sha256=protocol.parent.adapter_model_sha256,
            adapter_config_sha256=protocol.parent.adapter_config_sha256,
            rank=protocol.parent.rank,
            artifact_kind="source-parent-adapter",
        )
        self._verify_target(parent)
        self._model = PeftModel.from_pretrained(
            model,
            parent.adapter_path,
            adapter_name=self._parent_name,
            is_trainable=False,
            local_files_only=True,
        )
        self._model.eval()
        self._tokenizer = tokenizer
        self._input_device = self._model.get_input_embeddings().weight.device

    def evaluate(
        self,
        target: EvaluationTarget,
        rows: Sequence[EvaluationRow],
        *,
        phase: str,
        suite: str,
        maximum_tokens: int,
    ) -> Sequence[RowSufficientStatistic]:
        del phase, suite
        if maximum_tokens != 3072:
            raise RuntimeError("maximum token boundary differs")
        self._verify_target(target)
        self._activate(target)
        torch = self._torch
        encoded = tuple(
            self._encode(row, maximum_tokens, original_index=index)
            for index, row in enumerate(rows)
        )
        pad_token_id = self._tokenizer.pad_token_id
        result: dict[int, RowSufficientStatistic] = {}
        self._model.eval()
        with torch.inference_mode():
            for microbatch in _length_bucketed_batches(encoded):
                payload = _right_padded_batch_payload(
                    microbatch, pad_token_id=pad_token_id
                )
                input_ids = torch.tensor(payload["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(
                    payload["attention_mask"], dtype=torch.long
                )
                shift_labels = torch.tensor(
                    payload["shift_labels"], dtype=torch.long
                )
                batch = {
                    "input_ids": input_ids.to(self._input_device),
                    "attention_mask": attention_mask.to(self._input_device),
                    "logits_to_keep": payload["logits_to_keep"],
                    "use_cache": False,
                }
                output = self._model(**batch)
                for original_index, statistic in _row_statistics_from_logits(
                    rows=microbatch,
                    logits=output.logits,
                    shift_labels=shift_labels,
                    functional=self._functional,
                    torch_module=torch,
                ):
                    if original_index in result:
                        raise RuntimeError("evaluation row was scored more than once")
                    result[original_index] = statistic
        if set(result) != set(range(len(rows))):
            raise RuntimeError("evaluation microbatches did not cover every row")
        return tuple(result[index] for index in range(len(rows)))

    def close(self) -> None:
        torch = self._torch
        self._model = None
        torch.cuda.empty_cache()

    def _encode(
        self,
        row: EvaluationRow,
        maximum_tokens: int,
        *,
        original_index: int,
    ) -> _EncodedEvaluationRow:
        messages = row.payload["messages"]
        prompt = self._tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"]
        full = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )["input_ids"]
        if full[: len(prompt)] != prompt:
            raise RuntimeError("assistant answer is not a suffix of the chat template")
        if len(full) > maximum_tokens:
            raise RuntimeError("evaluation row exceeds 3072 tokens")
        labels = [-100] * len(prompt) + full[len(prompt) :]
        if not any(value != -100 for value in labels):
            raise RuntimeError("evaluation row has no assistant target tokens")
        if (
            type(original_index) is not int
            or isinstance(original_index, bool)
            or original_index < 0
            or any(type(value) is not int for value in full)
        ):
            raise RuntimeError("encoded evaluation row differs")
        return _EncodedEvaluationRow(
            original_index=original_index,
            row=row,
            input_ids=tuple(full),
            labels=tuple(labels),
        )

    def _activate(self, target: EvaluationTarget) -> None:
        if target.name == "parent_control":
            self._discard_candidate()
            self._model.set_adapter(self._parent_name, inference_mode=True)
            return
        if (
            target.name != "weighted_composite"
            or target.artifact_kind
            != "exact-rank-concatenated-skill-composite"
            or target.weight is None
        ):
            raise RuntimeError("only an exact compiled weighted candidate is permitted")
        key = (target.adapter_model_sha256, target.adapter_config_sha256)
        if key != self._candidate_key:
            self._discard_candidate()
            name = "weighted_" + target.weight.replace(".", "p")
            # Hash verification occurs immediately before this exact-artifact load.
            self._verify_target(target)
            self._model.load_adapter(
                target.adapter_path,
                adapter_name=name,
                is_trainable=False,
                local_files_only=True,
            )
            config = self._model.peft_config.get(name)
            if (
                config is None
                or config.r != target.rank
                or config.lora_alpha != target.rank * 2
            ):
                raise RuntimeError("loaded candidate adapter configuration differs")
            self._candidate_name = name
            self._candidate_key = key
        assert self._candidate_name is not None
        self._model.set_adapter(self._candidate_name, inference_mode=True)

    def _discard_candidate(self) -> None:
        if self._candidate_name is not None:
            self._model.set_adapter(self._parent_name, inference_mode=True)
            self._model.delete_adapter(self._candidate_name)
            self._candidate_name = None
            self._candidate_key = None

    @staticmethod
    def _verify_target(target: EvaluationTarget) -> None:
        root = target.adapter_path
        if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
            raise RuntimeError("adapter path is not a canonical real directory")
        model = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        if (
            not root.is_dir()
            or model.is_symlink()
            or config.is_symlink()
            or not model.is_file()
            or not config.is_file()
            or _sha(model) != target.adapter_model_sha256
            or _sha(config) != target.adapter_config_sha256
        ):
            raise RuntimeError("adapter artifact changed before load")


def _dual_gpu_device_map(layer_count: int) -> dict[str, int]:
    if layer_count != 64:
        raise ValueError("frozen placement requires exactly 64 layers")
    mapping: dict[str, int] = {"model.embed_tokens": 0}
    mapping.update({f"model.layers.{index}": 0 for index in range(20)})
    mapping.update({f"model.layers.{index}": 1 for index in range(20, 64)})
    mapping.update({"model.norm": 1, "model.rotary_emb": 1, "lm_head": 0})
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and seal six exact weighted adapters, calibrate on the frozen "
            "rows, then test only the selected candidate. No service is changed."
        )
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = run_skill_adapter_calibration(
        output_root=Path(args.output),
        compiler=_compiler,
        evaluator_factory=_LocalNF4DualGPUBackend,
        evaluator_identity=_evaluator_identity(),
    )
    print(
        "JENNY2_SKILL_ADAPTER_CALIBRATION="
        + json.dumps(
            {
                "live_state_changed": False,
                "output_root": str(run.output_root),
                "pre_evaluation_manifest_ref": run.pre_evaluation_manifest_ref,
                "qualification_ref": run.qualification_ref,
                "result_path": str(run.result_path),
                "result_sha256": run.result_sha256,
                "runtime_changed": False,
                "selection_manifest_ref": run.selection_manifest_ref,
                "status": run.status,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if run.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
