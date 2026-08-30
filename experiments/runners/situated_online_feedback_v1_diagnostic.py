"""Reconstruct V1 fast-state credit from its preserved Qwen outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from angler.reasoning import LearnedSituatedMemoryReader, SituatedFeedbackPolicy, situated_outcome_loss
from angler.runtime import encode_detached_segments, freeze_knowledge_model
from experiments.runners.situated_online_feedback_v1 import (
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAXIMUM_RESIDUAL,
    RANK,
    SEED,
    SPEC,
    _encode_samples,
    _selection,
    build_stream,
)


SOURCE_SHA256 = "6513ab17b21bc9ccbed4c77ecf3992be1af436906ea8e7ae2d69f461e69d53f9"
SOURCE_TERMINAL_DIGEST = "sha256:e26bae6ee69b0e9db6d6564b788abb5c586217f499f8ddebc2b1ebc28fb65868"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visible_margin(scores: torch.Tensor) -> float:
    values = torch.topk(scores[0], k=2).values
    return float((values[0] - values[1]).detach().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path = Path(args.source_result)
    if _sha256(source_path) != SOURCE_SHA256:
        raise RuntimeError("source V1 result does not match the preserved identity")
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("V1 diagnostic output already exists")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("classification") != "NOT_SUPPORTED" or len(source.get("records", ())) != 96:
        raise RuntimeError("source V1 result structure is invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires the preserved CUDA representation path")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    stream, probes = build_stream()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True)
    reader = LearnedSituatedMemoryReader(
        content_width=int(checkpoint["content_width"]),
        temporal_width=SPEC.width,
        hidden_width=256,
        action_count=4,
    ).cuda()
    reader.load_state_dict(checkpoint["state_dict"], strict=True)
    reader.requires_grad_(False)
    reader.eval()
    texts = tuple(
        text
        for sample in stream + probes
        for text in (sample.query, *(item.text for item in sample.recall.items))
    )
    unique = tuple(dict.fromkeys(texts))
    features = encode_detached_segments(
        model,
        tokenizer,
        unique,
        batch_size=64,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: features[index].float() for index, text in enumerate(unique)}
    rows = _encode_samples(stream, embeddings, reader)
    probe_rows = _encode_samples(probes, embeddings, reader)

    torch.manual_seed(SEED + 1)
    policy = SituatedFeedbackPolicy(
        content_width=reader.content_width,
        temporal_width=SPEC.width,
        rank=RANK,
        maximum_residual=MAXIMUM_RESIDUAL,
    ).cuda()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    steps = []
    for index, (row, record) in enumerate(zip(rows, source["records"], strict=True)):
        output, selected_index, selected_action, _ = _selection(policy, row)
        if selected_action != record["live"]["selected_action"]:
            raise RuntimeError(f"reconstruction selection diverged at step {index}")
        reward = 1.0 if record["live"]["qwen_correct"] else -1.0
        optimizer.zero_grad(set_to_none=True)
        loss = situated_outcome_loss(
            output,
            torch.tensor([selected_index], device="cuda"),
            torch.tensor([reward], device="cuda"),
        )
        loss.backward()
        gradient_norm = float(
            torch.sqrt(
                sum(
                    parameter.grad.detach().square().sum()
                    for parameter in policy.parameters()
                    if parameter.grad is not None
                )
            ).item()
        )
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRADIENT_CLIP)
        optimizer.step()
        steps.append(
            {
                "step": index,
                "reward": reward,
                "selected_probability": float(output.weights[0, selected_index].detach().item()),
                "base_margin": _visible_margin(output.normalized_base_scores),
                "adaptive_margin": _visible_margin(output.scores),
                "maximum_abs_residual": float(output.residuals.detach().abs().max().item()),
                "gradient_norm": gradient_norm,
            }
        )
    terminal_digest = policy.plastic_state_digest()
    if terminal_digest != SOURCE_TERMINAL_DIGEST:
        raise RuntimeError("reconstructed fast state does not match the V1 terminal digest")

    def probe_detail():
        details = []
        with torch.inference_mode():
            for row in probe_rows:
                output, _, action, _ = _selection(policy, row)
                details.append(
                    {
                        "family": row.sample.family_index,
                        "correct": action == ("stage", "seal", "invert", "trace")[row.sample.family_index],
                        "base_margin": _visible_margin(output.normalized_base_scores),
                        "adaptive_margin": _visible_margin(output.scores),
                        "maximum_abs_residual": float(output.residuals.abs().max().item()),
                    }
                )
        return details

    probe = probe_detail()
    report = {
        "identity": "angler.situated-online-feedback.v1-diagnostic",
        "source_result_sha256": SOURCE_SHA256,
        "terminal_digest": terminal_digest,
        "summary": {
            "mean_base_margin": sum(row["base_margin"] for row in steps) / len(steps),
            "mean_adaptive_margin": sum(row["adaptive_margin"] for row in steps) / len(steps),
            "mean_maximum_abs_residual": sum(row["maximum_abs_residual"] for row in steps) / len(steps),
            "terminal_maximum_abs_residual": max(row["maximum_abs_residual"] for row in probe),
            "mean_gradient_norm": sum(row["gradient_norm"] for row in steps) / len(steps),
            "clipped_gradient_fraction": sum(row["gradient_norm"] > GRADIENT_CLIP for row in steps) / len(steps),
            "probe_accuracy": sum(row["correct"] for row in probe) / len(probe),
        },
        "steps": steps,
        "probe": probe,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "limits": [
            "Reconstructs the consumed V1 state from preserved outcomes; no new Qwen generation or outcome is accepted.",
            "Diagnostic values cannot reclassify V1.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
