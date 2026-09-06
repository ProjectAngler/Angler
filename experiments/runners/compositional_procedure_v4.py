"""Jointly train Angler slots and its structured decoder on public composition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
import experiments.runners.structured_procedure_decoder_v3 as v3


IDENTITY = "angler.compositional-procedure.v4"
SEED = 20260846
PARENT_CHECKPOINT_SHA256 = "08cb21c1485cee41565a79b28afdabf1b14a3a379f0d8044cb06870a3d7a1df2"
V3_RESULT_SHA256 = "ef2c56461cc8cc72200bf2e03b94e07d6c12ea41ba778a9c8d79e5a836174698"
EPOCHS = 8
CORE_LEARNING_RATE = 3.0e-5
DECODER_LEARNING_RATE = 3.0e-4
RETRIEVAL_WEIGHT = 0.1
GRADIENT_LIMIT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--v3-result", required=True)
    parser.add_argument("--embedding-cache", required=True)
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


def supervised_composed_target(example: v1.PreparedExample) -> str:
    """Materialize a label only for a generated training-partition example."""

    pair = example.pair
    if pair is None or pair.hidden.mechanism_partition != "train":
        raise ValueError("composed supervision is restricted to the train partition")
    candidates = pair.learner.grounded_candidates
    labels = []
    for motif in pair.hidden.required_motifs:
        chosen = sorted(
            (
                item
                for item in pair.hidden.implementations
                if item.motif == motif and not item.distractor
            ),
            key=lambda item: item.stage,
        )
        if len(chosen) != 2 or tuple(item.stage for item in chosen) != (0, 1):
            raise RuntimeError("training composition does not contain two ordered actions")
        for implementation in chosen:
            matches = tuple(
                index
                for index, action in enumerate(candidates)
                if action.digest == implementation.action_digest
            )
            if len(matches) != 1:
                raise RuntimeError("training action does not map to one public candidate")
            labels.append(chr(ord("A") + matches[0]))
    if len(labels) != v3.MAXIMUM_STEPS:
        raise RuntimeError("training composition does not fill the four-step horizon")
    return " ".join(labels)


def _public_training_streams(streams):
    return tuple(
        replace(
            stream,
            queries=tuple(
                replace(example, target_text=supervised_composed_target(example))
                for example in stream.queries
            ),
        )
        for stream in streams
    )


def _relevance_loss(output, example) -> tuple[torch.Tensor, float]:
    relevant = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
    mass = output.candidate_weights[0, relevant].sum().clamp_min(1.0e-8)
    return -torch.log(mass), float(mass.detach())


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_core(sealed) -> ScalableProceduralCore:
    config = ProceduralCoreConfig(**sealed["config"])
    core = ScalableProceduralCore(config).cuda()
    core.load_state_dict(sealed["state_dict"], strict=True)
    return core


def _new_decoder(config: ProceduralCoreConfig) -> CandidateProcedureDecoder:
    return CandidateProcedureDecoder(
        content_width=config.content_width,
        procedure_width=config.model_width,
        hidden_width=v3.DECODER_WIDTH,
        heads=v3.DECODER_HEADS,
        maximum_actions=v3.MAXIMUM_ACTIONS,
        maximum_steps=v3.MAXIMUM_STEPS,
    ).cuda()


def _advance_supports(core, stream, embeddings, state):
    with torch.inference_mode():
        for example in stream.supports:
            output = core(*v1._inputs(example, embeddings), plastic_state=state)
            state = core.apply_feedback(
                state,
                output,
                torch.ones(1, device="cuda", dtype=output.query_state.dtype),
            )
    return state


def _evaluate_streams(core, decoder, streams, embeddings, persistent=None):
    scores = []
    attribution = []
    rows = []
    for stream_index, stream in enumerate(streams):
        state = (
            core.initial_plastic_state()
            if persistent is None
            else persistent.detached_clone()
        )
        state = _advance_supports(core, stream, embeddings, state)
        for query_index, example in enumerate(stream.queries):
            output, decoded = v3._decode(core, decoder, example, embeddings, state)
            pipeline = v3._decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
            score = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
            relevant = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
            mass = float(output.candidate_weights[0, relevant].sum())
            scores.append(score)
            attribution.append(mass)
            rows.append(
                {
                    "stream_index": stream_index,
                    "query_index": query_index,
                    "score": score,
                    "sequence": decoded.selected_indices[0].detach().cpu().tolist(),
                    "target_evidence_mass": mass,
                }
            )
    return _mean(scores), _mean(attribution), rows


def main() -> None:
    args = parse_args()
    parent_path = Path(args.parent_checkpoint)
    v3_path = Path(args.v3_result)
    cache_path = Path(args.embedding_cache)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if any(path.exists() for path in (cache_path, output_path, checkpoint_path)):
        raise RuntimeError("compositional V4 output identity already exists")
    if _sha256(parent_path) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent core checkpoint identity mismatch")
    if _sha256(v3_path) != V3_RESULT_SHA256:
        raise RuntimeError("V3 result identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("compositional V4 requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    evidence, raw_train, development, final = v1.prepare_corpus()
    train = _public_training_streams(raw_train)
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
    texts = v3._needed_texts(evidence, (train, development, final))
    encoded = encode_detached_segments(
        model,
        tokenizer,
        texts,
        batch_size=args.embedding_batch_size,
        storage_dtype=torch.bfloat16,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "foundation_digest": foundation,
            "texts": texts,
            "encoded": encoded,
        },
        cache_path,
    )
    embeddings = {text: encoded[index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    core = _load_core(sealed)
    decoder = _new_decoder(core.config)
    optimizer = torch.optim.AdamW(
        (
            {"params": core.parameters(), "lr": CORE_LEARNING_RATE},
            {"params": decoder.parameters(), "lr": DECODER_LEARNING_RATE},
        ),
        weight_decay=1.0e-4,
    )
    epoch_rows = []
    for epoch in range(EPOCHS):
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        support_exact = []
        composition_exact = []
        relevance_rows = []
        core.train()
        decoder.train()
        for stream_index in order:
            stream = train[stream_index]
            state = core.initial_plastic_state()
            episode_losses = []
            for example in (*stream.supports, *stream.queries):
                output = core(*v1._inputs(example, embeddings), plastic_state=state)
                action_features, action_mask = v3._action_inputs(example, embeddings)
                targets = v3._targets(example)
                decoded = decoder(
                    output.procedure_slots,
                    action_features,
                    action_mask,
                    teacher_actions=targets,
                )
                sequence_loss = candidate_procedure_loss(decoded, targets)
                retrieval_loss, mass = _relevance_loss(output, example)
                episode_losses.append(sequence_loss + RETRIEVAL_WEIGHT * retrieval_loss)
                relevance_rows.append(mass)
                exact = v3._exact_visible(decoded, targets)
                if example.pair is None:
                    support_exact.append(exact)
                    state = core.apply_feedback(
                        state,
                        output,
                        torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                        detach_state=False,
                    )
                else:
                    composition_exact.append(exact)
            loss = torch.stack(episode_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                tuple(core.parameters()) + tuple(decoder.parameters()),
                GRADIENT_LIMIT,
            )
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("joint procedural gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": _mean(losses),
                "support_teacher_exact": _mean(support_exact),
                "composition_teacher_exact": _mean(composition_exact),
                "relevance_mass": _mean(relevance_rows),
            }
        )

    core.eval()
    decoder.eval()
    persistent = core.initial_plastic_state()
    with torch.inference_mode():
        for stream in train:
            for example in stream.supports:
                output = core(*v1._inputs(example, embeddings), plastic_state=persistent)
                persistent = core.apply_feedback(
                    persistent,
                    output,
                    torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                )

    development_execution, development_attribution, development_rows = _evaluate_streams(
        core,
        decoder,
        development,
        embeddings,
    )
    final_rows = []
    for stream_index, stream in enumerate(final):
        live = _advance_supports(core, stream, embeddings, persistent.detached_clone())
        for query_index, example in enumerate(stream.queries):
            arms = {
                "full": v3._decode(core, decoder, example, embeddings, live),
                "reset_state": v3._decode(
                    core, decoder, example, embeddings, core.initial_plastic_state()
                ),
                "coordinates_removed": v3._decode(
                    core, decoder, example, embeddings, live, coordinates=False
                ),
                "unrelated_evidence": v3._decode(
                    core, decoder, example, embeddings, live, unrelated=True
                ),
                "procedure_slots_removed": v3._decode(
                    core, decoder, example, embeddings, live, remove_slots=True
                ),
            }
            scores = {}
            sequences = {}
            for name, (_, decoded) in arms.items():
                pipeline = v3._decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
                scores[name] = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
                sequences[name] = decoded.selected_indices[0].detach().cpu().tolist()
            full_output = arms["full"][0]
            relevant = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
            final_rows.append(
                {
                    "stream_index": stream_index,
                    "query_index": query_index,
                    "scores": scores,
                    "sequences": sequences,
                    "target_evidence_mass": float(
                        full_output.candidate_weights[0, relevant].sum()
                    ),
                }
            )

    arm_names = tuple(final_rows[0]["scores"])
    metrics = {
        name: _mean([float(row["scores"][name]) for row in final_rows])
        for name in arm_names
    }
    attribution = _mean([row["target_evidence_mass"] for row in final_rows])
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
        and development_execution >= 0.60
        and attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "v3_result_sha256": V3_RESULT_SHA256,
        "curriculum": {
            "train_mechanisms": len(train),
            "two_action_presentations": sum(len(stream.supports) for stream in train),
            "four_action_presentations": sum(len(stream.queries) for stream in train),
            "epochs": EPOCHS,
            "optimizer_updates": EPOCHS * len(train),
        },
        "training": epoch_rows,
        "development_execution": development_execution,
        "development_target_evidence_attribution": development_attribution,
        "development": development_rows,
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "evaluation": final_rows,
        "embedding_cache_sha256": _sha256(cache_path),
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "core": {"config": asdict(core.config), "parameters": core.parameter_count},
        "decoder_parameters": sum(value.numel() for value in decoder.parameters()),
        "runner_sha256": _sha256(Path(__file__)),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "foundation_digest": foundation,
            "core_config": asdict(core.config),
            "core_state_dict": {
                name: value.detach().cpu() for name, value in core.state_dict().items()
            },
            "decoder_config": {
                "content_width": core.config.content_width,
                "procedure_width": core.config.model_width,
                "hidden_width": v3.DECODER_WIDTH,
                "heads": v3.DECODER_HEADS,
                "maximum_actions": v3.MAXIMUM_ACTIONS,
                "maximum_steps": v3.MAXIMUM_STEPS,
            },
            "decoder_state_dict": {
                name: value.detach().cpu() for name, value in decoder.state_dict().items()
            },
            "plastic_state": {
                "keys": persistent.keys.detach().cpu(),
                "values": persistent.values.detach().cpu(),
                "strengths": persistent.strengths.detach().cpu(),
                "step": persistent.step,
            },
        },
        checkpoint_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "identity",
                    "classification",
                    "training",
                    "development_execution",
                    "metrics",
                    "target_evidence_attribution",
                    "runtime",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
