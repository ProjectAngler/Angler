"""Evaluation-only publication-boundary successor to scaled software V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from angler.reasoning import PlasticProcedureState, ProceduralCoreConfig, ScalableProceduralCore
from angler.runtime import encode_detached_segments, foundation_tensor_digest, freeze_knowledge_model
import experiments.runners.scaled_procedural_software_v1 as v1


IDENTITY = "angler.scaled-procedural-software-reconstruction.v2-publication"
SEED = v1.SEED
V1_RESULT_SHA256 = "ae0523155e5b8c1428887767fd0ebbe8acb2b0acbdd3a3456a8469a531ef48e7"
V1_CHECKPOINT_SHA256 = "08cb21c1485cee41565a79b28afdabf1b14a3a379f0d8044cb06870a3d7a1df2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--v1-result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_prompt_v2(query_text: str, evidence=()) -> str:
    return v1.build_prompt(query_text, evidence) + "\nanswer="


def _needed_texts(evidence, final) -> tuple[str, ...]:
    values = []
    for stream in final:
        for example in (*stream.supports, *stream.queries):
            values.append(example.query_text)
            for ref in (*example.candidate_refs, *example.unrelated_refs):
                values.append(evidence[ref])
    return tuple(dict.fromkeys(values))


def _evaluate_query(core, model, tokenizer, example, evidence, embeddings, live_state):
    with torch.inference_mode():
        full = core(*v1._inputs(example, embeddings), plastic_state=live_state)
        reset = core(
            *v1._inputs(example, embeddings),
            plastic_state=core.initial_plastic_state(),
        )
        no_coordinates = core(
            *v1._inputs(example, embeddings, coordinates=False),
            plastic_state=live_state,
        )
        unrelated = core(
            *v1._inputs(
                example,
                embeddings,
                refs=example.unrelated_refs,
                temporal=example.unrelated_temporal,
            ),
            plastic_state=live_state,
        )
    selected = v1._selected_refs(example, full.candidate_weights[0])
    reset_selected = v1._selected_refs(example, reset.candidate_weights[0])
    coordinate_selected = v1._selected_refs(example, no_coordinates.candidate_weights[0])
    unrelated_selected = v1._selected_refs(example, unrelated.candidate_weights[0])
    prompts = {
        "qwen_alone": build_prompt_v2(example.query_text),
        "fair_retrieval": build_prompt_v2(
            example.query_text,
            [evidence[ref] for ref in example.candidate_refs],
        ),
        "prefix_removed": build_prompt_v2(
            example.query_text,
            [evidence[ref] for ref in selected],
        ),
        "full": build_prompt_v2(example.query_text, [evidence[ref] for ref in selected]),
        "reset_state": build_prompt_v2(
            example.query_text,
            [evidence[ref] for ref in reset_selected],
        ),
        "coordinates_removed": build_prompt_v2(
            example.query_text,
            [evidence[ref] for ref in coordinate_selected],
        ),
        "unrelated_evidence": build_prompt_v2(
            example.query_text,
            [evidence[ref] for ref in unrelated_selected],
        ),
    }
    responses = {
        "qwen_alone": v1._generate_plain(model, tokenizer, prompts["qwen_alone"]),
        "fair_retrieval": v1._generate_plain(model, tokenizer, prompts["fair_retrieval"]),
        "prefix_removed": v1._generate_plain(model, tokenizer, prompts["prefix_removed"]),
        "full": v1.generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["full"],
            full.qwen_prefix,
            max_new_tokens=v1.MAX_NEW_TOKENS,
        ).response,
        "reset_state": v1.generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["reset_state"],
            reset.qwen_prefix,
            max_new_tokens=v1.MAX_NEW_TOKENS,
        ).response,
        "coordinates_removed": v1.generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["coordinates_removed"],
            no_coordinates.qwen_prefix,
            max_new_tokens=v1.MAX_NEW_TOKENS,
        ).response,
        "unrelated_evidence": v1.generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["unrelated_evidence"],
            unrelated.qwen_prefix,
            max_new_tokens=v1.MAX_NEW_TOKENS,
        ).response,
    }
    scores = {name: v1._judge(example.pair, response) for name, response in responses.items()}
    relevance = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
    return {
        "scores": scores,
        "responses": responses,
        "selected_refs": list(selected),
        "target_evidence_mass": float(full.candidate_weights[0, relevance].sum()),
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.v1_result)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("scaled procedural V2 output already exists")
    if _sha256(result_path) != V1_RESULT_SHA256 or _sha256(checkpoint_path) != V1_CHECKPOINT_SHA256:
        raise RuntimeError("V1 result/checkpoint identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("scaled procedural V2 requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    evidence, _, _, final = v1.prepare_corpus()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    sealed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    foundation = foundation_tensor_digest(model)
    if foundation != sealed["foundation_digest"]:
        raise RuntimeError("checkpoint and frozen Qwen foundation do not match")
    texts = _needed_texts(evidence, final)
    encoded = encode_detached_segments(
        model,
        tokenizer,
        texts,
        batch_size=args.embedding_batch_size,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: encoded[index] for index, text in enumerate(texts)}
    for stream in final:
        for example in (*stream.supports, *stream.queries):
            for ref in (*example.candidate_refs, *example.unrelated_refs):
                embeddings[ref] = embeddings[evidence[ref]]

    config = ProceduralCoreConfig(**sealed["config"])
    core = ScalableProceduralCore(config).cuda()
    core.load_state_dict(sealed["state_dict"], strict=True)
    core.eval()
    stored = sealed["plastic_state"]
    persistent = PlasticProcedureState(
        stored["keys"].cuda(),
        stored["values"].cuda(),
        stored["strengths"].cuda(),
        int(stored["step"]),
    )
    v1_result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = []
    for stream_index, stream in enumerate(final):
        live = persistent.detached_clone()
        with torch.inference_mode():
            for support in stream.supports:
                output = core(*v1._inputs(support, embeddings), plastic_state=live)
                live = core.apply_feedback(
                    live,
                    output,
                    torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                )
        for query_index, example in enumerate(stream.queries):
            row = _evaluate_query(core, model, tokenizer, example, evidence, embeddings, live)
            row["stream_index"] = stream_index
            row["query_index"] = query_index
            rows.append(row)

    arms = tuple(rows[0]["scores"])
    metrics = {
        arm: v1._mean([float(row["scores"][arm]) for row in rows])
        for arm in arms
    }
    attribution = v1._mean([float(row["target_evidence_mass"]) for row in rows])
    loss_reduction = float(v1_result["loss_reduction"])
    supported = (
        metrics["full"] >= 0.60
        and metrics["full"] - metrics["qwen_alone"] >= 0.15
        and metrics["full"] - metrics["fair_retrieval"] >= 0.15
        and metrics["full"] - metrics["prefix_removed"] >= 0.10
        and metrics["full"] - metrics["reset_state"] >= 0.10
        and metrics["full"] - metrics["coordinates_removed"] >= 0.10
        and attribution >= 0.75
        and loss_reduction >= 0.20
    )
    report = {
        "identity": IDENTITY,
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "v1_result_sha256": V1_RESULT_SHA256,
        "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
        "change": "append literal answer= publication boundary only",
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "inherited_v1_loss_reduction": loss_reduction,
        "evaluation": rows,
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "runner_sha256": _sha256(Path(__file__)),
    }
    if foundation_tensor_digest(model) != foundation or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen Qwen changed during V2 evaluation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("identity", "classification", "metrics", "target_evidence_attribution", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
