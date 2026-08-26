"""Run a local-only Qwen3-4B LoRA attach/save/reload smoke check."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from angler.runtime import (  # noqa: E402
    adapter_tensor_digest,
    attach_single_adapter,
    build_causal_lm_lora_config,
    foundation_tensor_digest,
    reload_adapter_local,
    save_adapter_local,
    validate_foundation_frozen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument(
        "--adapter-dir",
        default="/opt/angler/project/work/qwen3-4b-r8-qv-smoke",
    )
    return parser.parse_args()


def load_foundation(model_path: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-checkpoint smoke")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    foundation = load_foundation(args.model)
    first_load_finished = time.perf_counter()
    initial_foundation_digest = foundation_tensor_digest(foundation)

    config = build_causal_lm_lora_config(
        target_modules=("q_proj", "v_proj"),
        rank=8,
        alpha=16,
        dropout=0.0,
    )
    model = attach_single_adapter(foundation, config)
    inventory = validate_foundation_frozen(model)
    adapter_digest_before = adapter_tensor_digest(model)
    saved_path = save_adapter_local(model, Path(args.adapter_dir))
    attached_foundation_digest = foundation_tensor_digest(model)
    first_stage_finished = time.perf_counter()

    if initial_foundation_digest != attached_foundation_digest:
        raise RuntimeError("foundation fingerprint changed after LoRA attachment/save")
    if inventory.trainable_foundation_numel != 0:
        raise RuntimeError("foundation parameters are trainable")
    if inventory.trainable_lora_numel != inventory.lora_numel:
        raise RuntimeError("not every LoRA parameter is trainable")

    del model, foundation
    release_cuda()
    after_first_release_allocated = torch.cuda.memory_allocated()

    clean_foundation = load_foundation(args.model)
    second_load_finished = time.perf_counter()
    restored = reload_adapter_local(clean_foundation, saved_path)
    restored_inventory = validate_foundation_frozen(restored)
    adapter_digest_after = adapter_tensor_digest(restored)
    restored_foundation_digest = foundation_tensor_digest(restored)
    finished = time.perf_counter()

    if adapter_digest_before != adapter_digest_after:
        raise RuntimeError("adapter digest changed across save/reload")
    if initial_foundation_digest != restored_foundation_digest:
        raise RuntimeError("foundation fingerprint changed in the clean runtime")
    if restored_inventory.trainable_foundation_numel != 0:
        raise RuntimeError("reloaded foundation parameters are trainable")
    if restored_inventory.trainable_lora_numel != restored_inventory.lora_numel:
        raise RuntimeError("reloaded LoRA parameter scope is incomplete")

    torch.cuda.synchronize()
    result = {
        "model_path": str(Path(args.model).resolve()),
        "adapter_path": str(saved_path),
        "adapter_topology": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        "adapter_digest": adapter_digest_after,
        "foundation_fingerprint": restored_foundation_digest,
        "foundation_unchanged": True,
        "adapter_round_trip_exact": True,
        "parameter_counts": {
            "foundation": restored_inventory.foundation_numel,
            "lora": restored_inventory.lora_numel,
            "trainable_foundation": restored_inventory.trainable_foundation_numel,
            "trainable_lora": restored_inventory.trainable_lora_numel,
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "device": torch.cuda.get_device_name(0),
        "dtype": "torch.bfloat16",
        "timing_seconds": {
            "first_load": round(first_load_finished - started, 3),
            "attach_hash_save": round(first_stage_finished - first_load_finished, 3),
            "release_and_second_load": round(
                second_load_finished - first_stage_finished, 3
            ),
            "reload_and_verify": round(finished - second_load_finished, 3),
            "total": round(finished - started, 3),
        },
        "cuda_memory_bytes": {
            "allocated_after_first_release": after_first_release_allocated,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        },
        "local_files_only": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
