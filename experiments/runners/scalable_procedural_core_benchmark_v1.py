"""RTX/Qwen interface benchmark for the scalable Angler procedural core."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

from angler.reasoning import (
    ScalableProceduralCore,
    encode_situated_features,
    plastic_state_digest,
    procedural_core_config,
    select_procedural_core_tier,
)
from angler.runtime import (
    encode_detached_segments,
    freeze_knowledge_model,
    generate_with_procedural_prefix,
)
from experiments.runners.situated_memory_cognee_evaluate_v2 import _recall
from experiments.runners.situated_memory_value_v2 import SPEC_V2


IDENTITY = "angler.scalable-procedural-core-benchmark.v1"
SEED = 20260843


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--recall", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scalable core benchmark requires CUDA")
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("scalable core benchmark output already exists")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    source = json.loads(Path(args.recall).read_text(encoding="utf-8"))
    rows = source["rows"]
    recalls = tuple(_recall(row) for row in rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    texts = tuple(
        text
        for row, recall in zip(rows, recalls, strict=True)
        for text in (row["query_text"], *(item.text for item in recall.items))
    )
    unique = tuple(dict.fromkeys(texts))
    encoded = encode_detached_segments(
        model,
        tokenizer,
        unique,
        batch_size=64,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: encoded[index].float() for index, text in enumerate(unique)}
    query = torch.stack([embeddings[row["query_text"]] for row in rows]).cuda()
    candidates = torch.stack(
        [torch.stack([embeddings[item.text] for item in recall.items]) for recall in recalls]
    ).cuda()
    temporal, mask = encode_situated_features(
        recalls,
        now=tuple(int(row["origin_now"]) for row in rows),
        spec=SPEC_V2,
        device="cuda",
        dtype=torch.float32,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    selected_config = select_procedural_core_tier(
        int(free_bytes),
        content_width=query.shape[-1],
        temporal_width=SPEC_V2.width,
        training=True,
    )
    gc.collect()
    tiers = []
    selected_prefix = None
    selected_tier = selected_config.tier

    for tier_index, tier in enumerate(("compact", "workstation", "dedicated")):
        torch.manual_seed(SEED + tier_index)
        torch.cuda.manual_seed_all(SEED + tier_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = procedural_core_config(
            tier,
            content_width=query.shape[-1],
            temporal_width=SPEC_V2.width,
        )
        core = ScalableProceduralCore(config).cuda()
        initial = core.initial_plastic_state()
        initial_digest = plastic_state_digest(initial)
        core.train()
        output = core(query, candidates, temporal, mask, plastic_state=initial)
        loss = output.qwen_prefix.square().mean() + output.candidate_weights.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        gradients_finite = bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)
        core.zero_grad(set_to_none=True)
        core.eval()
        with torch.inference_mode():
            baseline = core(query[:1], candidates[:1], temporal[:1], mask[:1], plastic_state=initial)
            updated = core.apply_feedback(initial, baseline, torch.tensor([1.0], device="cuda"))
            live = core(query[:1], candidates[:1], temporal[:1], mask[:1], plastic_state=updated)
            state_effect = float((live.procedure_slots - baseline.procedure_slots).square().mean().sqrt().item())
            prefix_effect = float((live.qwen_prefix - baseline.qwen_prefix).square().mean().sqrt().item())
            timings = []
            for _ in range(3):
                torch.cuda.synchronize()
                tick = time.monotonic()
                measured = core(query, candidates, temporal, mask, plastic_state=updated)
                torch.cuda.synchronize()
                timings.append(time.monotonic() - tick)
            entropy = float(
                (-(measured.candidate_weights * measured.candidate_weights.clamp_min(1e-8).log()).sum(dim=1)).mean().item()
            )
            if tier == selected_tier:
                selected_prefix = live.qwen_prefix.detach().cpu()
        tiers.append(
            {
                "tier": tier,
                "parameters": core.parameter_count,
                "parameter_bytes_fp32": core.parameter_count * 4,
                "plastic_state_bytes": initial.bytes,
                "procedure_tokens": config.procedure_tokens,
                "plastic_slots": config.plastic_slots,
                "model_width": config.model_width,
                "depth": config.depth,
                "heads": config.heads,
                "gradient_path_finite": gradients_finite,
                "initial_state_digest": initial_digest,
                "updated_state_digest": plastic_state_digest(updated),
                "state_effect_rms": state_effect,
                "prefix_effect_rms": prefix_effect,
                "candidate_entropy": entropy,
                "mean_forward_seconds_batch8": sum(timings) / len(timings),
                "peak_allocated_bytes_with_qwen": torch.cuda.max_memory_allocated(),
            }
        )
        del core, output, baseline, live, measured, updated, gradients
        gc.collect()
        torch.cuda.empty_cache()

    if selected_prefix is None:
        raise RuntimeError("resource-selected tier was not benchmarked")
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return exactly the word READY."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    generation = generate_with_procedural_prefix(
        model,
        tokenizer,
        chat,
        selected_prefix[:1].cuda(),
        max_new_tokens=16,
    )
    counts = [row["parameters"] for row in tiers]
    ready = (
        counts == sorted(counts)
        and all(row["gradient_path_finite"] for row in tiers)
        and all(row["state_effect_rms"] > 0.0 and row["prefix_effect_rms"] > 0.0 for row in tiers)
        and bool(generation.generated_token_ids)
        and selected_tier in {"workstation", "dedicated"}
    )
    torch.cuda.synchronize()
    report = {
        "identity": IDENTITY,
        "classification": "SCALED_CORE_INTERFACE_READY" if ready else "NOT_READY",
        "resource_selection": {
            "gpu": torch.cuda.get_device_name(0),
            "total_bytes": int(total_bytes),
            "free_after_qwen_and_embeddings": int(free_bytes),
            "selected_training_tier": selected_tier,
            "reserve_fraction": 0.25,
        },
        "tiers": tiers,
        "qwen_prefix_smoke": {
            "tier": selected_tier,
            "response": generation.response,
            "generated_token_ids": generation.generated_token_ids,
            "prompt_tokens": generation.prompt_tokens,
            "procedure_tokens": generation.procedure_tokens,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "wall_seconds": time.monotonic() - started,
            "seed": SEED,
        },
        "limits": [
            "This is capacity/interface evidence, not a trained reasoning result.",
            "Random initial procedure prefixes have no expected semantic benefit.",
            "Complex-task training and component-removal evaluation remain next.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
