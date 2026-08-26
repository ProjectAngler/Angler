"""Run deterministic, local-only Qwen inference for the Phase 1 Angler substrate."""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument(
        "--cache-implementation", choices=("dynamic", "static"), default="dynamic"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local inference check")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    loaded = time.perf_counter()

    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")

    if args.warmup_tokens:
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=args.warmup_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                cache_implementation=args.cache_implementation,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    generated_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            cache_implementation=args.cache_implementation,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()

    new_ids = output_ids[0, inputs.input_ids.shape[1] :]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    result = {
        "model_path": args.model,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(next(model.parameters()).dtype),
        "prompt": args.prompt,
        "answer": answer,
        "input_tokens": int(inputs.input_ids.shape[1]),
        "output_tokens": int(new_ids.shape[0]),
        "warmup_tokens": args.warmup_tokens,
        "cache_implementation": args.cache_implementation,
        "load_seconds": round(loaded - started, 3),
        "generation_seconds": round(finished - generated_started, 3),
        "tokens_per_second": round(new_ids.shape[0] / (finished - generated_started), 3),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "local_files_only": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
