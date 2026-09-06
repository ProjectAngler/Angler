"""Train a learned structured decoder over the preserved scaled Angler core."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import torch

from angler.reasoning import (
    CandidateProcedureDecoder,
    PlasticProcedureState,
    ProceduralCoreConfig,
    ScalableProceduralCore,
    candidate_procedure_loss,
)
from angler.runtime import encode_detached_segments, foundation_tensor_digest, freeze_knowledge_model
import experiments.runners.scaled_procedural_software_v1 as v1


IDENTITY = "angler.structured-procedure-decoder.v3"
SEED = 20260845
PARENT_CHECKPOINT_SHA256 = "08cb21c1485cee41565a79b28afdabf1b14a3a379f0d8044cb06870a3d7a1df2"
PARENT_V2_RESULT_SHA256 = "902a7eb87275eb7b125484f7393bff1fb891ac6d56ee20f1dfb7011b8a3c6f4d"
MAXIMUM_ACTIONS = 6
MAXIMUM_STEPS = 4
DECODER_WIDTH = 512
DECODER_HEADS = 8
EPOCHS = 4
LEARNING_RATE = 3.0e-4
GRADIENT_LIMIT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--v2-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _needed_texts(evidence, partitions) -> tuple[str, ...]:
    values = list(evidence.values())
    for partition in partitions:
        for stream in partition:
            for example in (*stream.supports, *stream.queries):
                values.append(example.query_text)
                values.extend(example.action_texts)
    return tuple(dict.fromkeys(values))


def _action_inputs(example, embeddings):
    if len(example.action_texts) > MAXIMUM_ACTIONS:
        raise RuntimeError("task exceeds the frozen action ceiling")
    width = next(iter(embeddings.values())).shape[0]
    features = torch.zeros(1, MAXIMUM_ACTIONS, width, device="cuda", dtype=torch.float32)
    mask = torch.zeros(1, MAXIMUM_ACTIONS, device="cuda", dtype=torch.bool)
    for index, text in enumerate(example.action_texts):
        features[0, index] = embeddings[text].float().cuda()
        mask[0, index] = True
    return features, mask


def _targets(example) -> torch.Tensor:
    if example.target_text is None:
        raise ValueError("visible support target is absent")
    values = []
    for token in example.target_text.split():
        if token == "STOP":
            values.append(MAXIMUM_ACTIONS)
            break
        if len(token) != 1 or not "A" <= token <= "F":
            raise ValueError("public target contains an invalid local label")
        values.append(ord(token) - ord("A"))
    values = values[:MAXIMUM_STEPS]
    values.extend([-100] * (MAXIMUM_STEPS - len(values)))
    return torch.tensor([values], device="cuda", dtype=torch.long)


def _decoder_pipeline(task, selected: torch.Tensor):
    actions = []
    stopped = False
    for value in selected.detach().cpu().tolist():
        if value == MAXIMUM_ACTIONS:
            stopped = True
            break
        if value < 0 or value >= len(task.grounded_candidates):
            actions = []
            stopped = True
            break
        actions.append(task.grounded_candidates[value])
    if len(actions) < task.max_steps:
        stopped = True
    return v1.commit_software_pipeline(task, actions, stopped=stopped)


def _decode(
    core,
    decoder,
    example,
    embeddings,
    state,
    *,
    coordinates=True,
    unrelated=False,
    remove_slots=False,
):
    if unrelated:
        core_inputs = v1._inputs(
            example,
            embeddings,
            refs=example.unrelated_refs,
            temporal=example.unrelated_temporal,
        )
    else:
        core_inputs = v1._inputs(example, embeddings, coordinates=coordinates)
    with torch.inference_mode():
        output = core(*core_inputs, plastic_state=state)
        slots = torch.zeros_like(output.procedure_slots) if remove_slots else output.procedure_slots
        action_features, action_mask = _action_inputs(example, embeddings)
        decoded = decoder(slots, action_features, action_mask)
    return output, decoded


def _exact_visible(decoded, targets) -> float:
    predicted = decoded.selected_indices[0]
    wanted = targets[0]
    active = wanted != -100
    return float(torch.equal(predicted[active], wanted[active]))


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    parent_path = Path(args.parent_checkpoint)
    v2_path = Path(args.v2_result)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("structured decoder V3 identity already exists")
    if _sha256(parent_path) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent core checkpoint identity mismatch")
    if _sha256(v2_path) != PARENT_V2_RESULT_SHA256:
        raise RuntimeError("parent V2 result identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("structured decoder V3 requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    evidence, train, development, final = v1.prepare_corpus()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    sealed = torch.load(parent_path, map_location="cpu", weights_only=True)
    foundation = foundation_tensor_digest(model)
    if foundation != sealed["foundation_digest"]:
        raise RuntimeError("parent checkpoint binds a different frozen Qwen")
    texts = _needed_texts(evidence, (train, development, final))
    encoded = encode_detached_segments(
        model,
        tokenizer,
        texts,
        batch_size=args.embedding_batch_size,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: encoded[index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]
    del encoded, model
    gc.collect()
    torch.cuda.empty_cache()

    config = ProceduralCoreConfig(**sealed["config"])
    core = ScalableProceduralCore(config).cuda()
    core.load_state_dict(sealed["state_dict"], strict=True)
    core.requires_grad_(False)
    core.eval()
    decoder = CandidateProcedureDecoder(
        content_width=config.content_width,
        procedure_width=config.model_width,
        hidden_width=DECODER_WIDTH,
        heads=DECODER_HEADS,
        maximum_actions=MAXIMUM_ACTIONS,
        maximum_steps=MAXIMUM_STEPS,
    ).cuda()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    epoch_rows = []
    for epoch in range(EPOCHS):
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        exact = []
        decoder.train()
        for stream_index in order:
            state = core.initial_plastic_state()
            for example in train[stream_index].supports:
                with torch.inference_mode():
                    output = core(*v1._inputs(example, embeddings), plastic_state=state)
                action_features, action_mask = _action_inputs(example, embeddings)
                targets = _targets(example)
                decoded = decoder(
                    output.procedure_slots,
                    action_features,
                    action_mask,
                    teacher_actions=targets,
                )
                loss = candidate_procedure_loss(decoded, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRADIENT_LIMIT)
                if not bool(torch.isfinite(norm).item()):
                    raise RuntimeError("decoder gradient became non-finite")
                optimizer.step()
                losses.append(float(loss.detach()))
                exact.append(_exact_visible(decoded, targets))
                with torch.inference_mode():
                    state = core.apply_feedback(
                        state,
                        output,
                        torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                    )
        epoch_rows.append(
            {"epoch": epoch, "loss": _mean(losses), "teacher_exact": _mean(exact)}
        )

    decoder.eval()
    development_exact = []
    with torch.inference_mode():
        for stream in development:
            state = core.initial_plastic_state()
            for example in stream.supports:
                output = core(*v1._inputs(example, embeddings), plastic_state=state)
                action_features, action_mask = _action_inputs(example, embeddings)
                decoded = decoder(output.procedure_slots, action_features, action_mask)
                development_exact.append(_exact_visible(decoded, _targets(example)))
                state = core.apply_feedback(
                    state,
                    output,
                    torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                )

    stored = sealed["plastic_state"]
    persistent = PlasticProcedureState(
        stored["keys"].cuda(),
        stored["values"].cuda(),
        stored["strengths"].cuda(),
        int(stored["step"]),
    )
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
            arms = {
                "full": _decode(core, decoder, example, embeddings, live),
                "reset_state": _decode(
                    core,
                    decoder,
                    example,
                    embeddings,
                    core.initial_plastic_state(),
                ),
                "coordinates_removed": _decode(
                    core,
                    decoder,
                    example,
                    embeddings,
                    live,
                    coordinates=False,
                ),
                "unrelated_evidence": _decode(
                    core,
                    decoder,
                    example,
                    embeddings,
                    live,
                    unrelated=True,
                ),
                "procedure_slots_removed": _decode(
                    core,
                    decoder,
                    example,
                    embeddings,
                    live,
                    remove_slots=True,
                ),
            }
            scores = {}
            sequences = {}
            for name, (_, decoded) in arms.items():
                pipeline = _decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
                scores[name] = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
                sequences[name] = decoded.selected_indices[0].detach().cpu().tolist()
            full_output = arms["full"][0]
            relevance = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
            rows.append(
                {
                    "stream_index": stream_index,
                    "query_index": query_index,
                    "scores": scores,
                    "sequences": sequences,
                    "target_evidence_mass": float(
                        full_output.candidate_weights[0, relevance].sum()
                    ),
                }
            )

    arm_names = tuple(rows[0]["scores"])
    metrics = {
        name: _mean([float(row["scores"][name]) for row in rows])
        for name in arm_names
    }
    attribution = _mean([row["target_evidence_mass"] for row in rows])
    dev_exact = _mean(development_exact)
    supported = (
        metrics["full"] >= 0.60
        and all(
            metrics["full"] - metrics[name] >= 0.10
            for name in (
                "reset_state",
                "coordinates_removed",
                "unrelated_evidence",
                "procedure_slots_removed",
            )
        )
        and dev_exact >= 0.75
        and attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_v2_result_sha256": PARENT_V2_RESULT_SHA256,
        "decoder": {
            "parameters": sum(value.numel() for value in decoder.parameters()),
            "epochs": EPOCHS,
            "updates": EPOCHS * v1.SLOW_UPDATES,
            "hidden_width": DECODER_WIDTH,
            "heads": DECODER_HEADS,
            "maximum_actions": MAXIMUM_ACTIONS,
            "maximum_steps": MAXIMUM_STEPS,
        },
        "training": epoch_rows,
        "development_exact": dev_exact,
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "evaluation": rows,
        "preserved_qwen_controls": {"qwen_alone": 0.0, "fair_retrieval": 0.0},
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
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "decoder_config": {
                "content_width": config.content_width,
                "procedure_width": config.model_width,
                "hidden_width": DECODER_WIDTH,
                "heads": DECODER_HEADS,
                "maximum_actions": MAXIMUM_ACTIONS,
                "maximum_steps": MAXIMUM_STEPS,
            },
            "state_dict": {name: value.detach().cpu() for name, value in decoder.state_dict().items()},
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        },
        checkpoint_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("identity", "classification", "development_exact", "metrics", "target_evidence_attribution", "training", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
