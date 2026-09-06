"""Action-level trace memory with learned candidate correspondence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import torch

from angler.reasoning import (
    ActionTraceMatcherDecoder,
    ActionTraceProceduralCore,
    PlasticProcedureState,
    ProceduralCoreConfig,
    candidate_procedure_loss,
)
import experiments.runners.compositional_procedure_v4 as v4
import experiments.runners.compositional_procedure_v4_r1 as v4r1
import experiments.runners.scaled_procedural_software_v1 as v1
import experiments.runners.structured_procedure_decoder_v3 as v3
import experiments.runners.trace_conditioned_procedure_v5 as v5


IDENTITY = "angler.action-trace-matching.v6"
SEED = 20260849
PARENT_CHECKPOINT_SHA256 = "f9f21bd778dd553468e70a000fe9d8033e02ab18298422237940d8126319332b"
EMBEDDING_CACHE_SHA256 = v5.EMBEDDING_CACHE_SHA256
EPOCHS = 8
LEARNING_RATE = 1.0e-4
RETRIEVAL_WEIGHT = v4.RETRIEVAL_WEIGHT
GRADIENT_LIMIT = v4.GRADIENT_LIMIT
CAUSAL_MARGIN_ARMS = v4r1.CAUSAL_MARGIN_ARMS
DIAGNOSTIC_ARMS = (*v4r1.DIAGNOSTIC_ARMS, "correspondence_removed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--angler-device", default="cuda:1")
    return parser.parse_args()


def _load_core(sealed, device):
    config = ProceduralCoreConfig(**sealed["core_config"])
    core = ActionTraceProceduralCore(config, maximum_trace_steps=v3.MAXIMUM_STEPS).to(device)
    current = core.state_dict()
    parent = sealed["core_state_dict"]
    inherited = {name: value for name, value in parent.items() if name in current}
    action_names = {name for name in current if name.startswith("action_")}
    if set(current) - set(inherited) != action_names:
        raise RuntimeError("V6 core topology is not a base-plus-action-trace successor")
    incompatible = core.load_state_dict(inherited, strict=False)
    if set(incompatible.missing_keys) != action_names or incompatible.unexpected_keys:
        raise RuntimeError("V5 base core did not load exactly into V6")
    core.requires_grad_(False)
    for name, parameter in core.named_parameters():
        if name.startswith("action_"):
            parameter.requires_grad_(True)
    return core


def _load_decoder(sealed, device):
    decoder = ActionTraceMatcherDecoder(**sealed["decoder_config"]).to(device)
    current = decoder.state_dict()
    parent = sealed["decoder_state_dict"]
    matcher_names = {name for name in current if name.startswith("matcher_")}
    incompatible = decoder.load_state_dict(parent, strict=False)
    if set(incompatible.missing_keys) != matcher_names or incompatible.unexpected_keys:
        raise RuntimeError("V5 decoder did not load with only matcher parameters fresh")
    decoder.requires_grad_(False)
    for name, parameter in decoder.named_parameters():
        if name.startswith("matcher_"):
            parameter.requires_grad_(True)
    return decoder


def _feedback(core, state, output, example, embeddings, *, detach_state=True):
    trace, mask = v5._procedure_trace(example, embeddings)
    return core.apply_action_trace_feedback(
        state,
        output,
        trace,
        mask,
        torch.ones(1, device=output.query_state.device, dtype=output.query_state.dtype),
        detach_state=detach_state,
    )


def _advance_supports(core, stream, embeddings, state):
    with torch.inference_mode():
        for example in stream.supports:
            output = core(*v1._inputs(example, embeddings), plastic_state=state)
            state = _feedback(core, state, output, example, embeddings)
    return state


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
    correspondence=True,
):
    if unrelated:
        inputs = v1._inputs(
            example,
            embeddings,
            refs=example.unrelated_refs,
            temporal=example.unrelated_temporal,
        )
    else:
        inputs = v1._inputs(example, embeddings, coordinates=coordinates)
    with torch.inference_mode():
        output = core(*inputs, plastic_state=state)
        slots = torch.zeros_like(output.procedure_slots) if remove_slots else output.procedure_slots
        action_features, action_mask = v3._action_inputs(example, embeddings)
        decoded = decoder(
            slots,
            action_features,
            action_mask,
            memory_values=state.values,
            memory_mask=state.strengths > 1.0e-6,
            correspondence=correspondence,
        )
    return output, decoded


def _evaluate_streams(core, decoder, streams, embeddings, persistent=None):
    scores, attribution, rows = [], [], []
    for stream_index, stream in enumerate(streams):
        state = v4r1._anchored_initial_state(core) if persistent is None else persistent.detached_clone()
        state = _advance_supports(core, stream, embeddings, state)
        for query_index, example in enumerate(stream.queries):
            output, decoded = _decode(core, decoder, example, embeddings, state)
            pipeline = v3._decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
            score = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
            relevant = torch.tensor(example.relevant, device=output.query_state.device, dtype=torch.bool)
            mass = float(output.candidate_weights[0, relevant].sum())
            scores.append(score)
            attribution.append(mass)
            rows.append({
                "stream_index": stream_index,
                "query_index": query_index,
                "score": score,
                "sequence": decoded.selected_indices[0].detach().cpu().tolist(),
                "target_evidence_mass": mass,
            })
    return v4._mean(scores), v4._mean(attribution), rows


def main() -> None:
    args = parse_args()
    parent_path, cache_path = Path(args.parent_checkpoint), Path(args.embedding_cache)
    output_path, checkpoint_path = Path(args.output), Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("V6 output identity already exists")
    if v5._sha256(parent_path) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V5 parent checkpoint identity mismatch")
    if v5._sha256(cache_path) != EMBEDDING_CACHE_SHA256:
        raise RuntimeError("embedding cache identity mismatch")
    device = torch.device(args.angler_device)
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
        raise ValueError("Angler device must be an available explicit CUDA ordinal")

    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device.index)
    evidence, raw_train, development, final = v1.prepare_corpus()
    train = v4r1._public_training_streams(raw_train)
    sealed = torch.load(parent_path, map_location="cpu", weights_only=True)
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    texts = v3._needed_texts(evidence, (train, development, final))
    if cached["foundation_digest"] != sealed["foundation_digest"] or tuple(cached["texts"]) != texts:
        raise RuntimeError("frozen cache does not match the V6 corpus/foundation")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    core = _load_core(sealed, device)
    decoder = _load_decoder(sealed, device)
    trainable = [parameter for parameter in (*core.parameters(), *decoder.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1.0e-4)
    epoch_rows = []
    for epoch in range(EPOCHS):
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses, support_exact, composition_exact, relevance_rows = [], [], [], []
        core.train(); decoder.train()
        for stream_index in order:
            stream = train[stream_index]
            state = v4r1._anchored_initial_state(core)
            episode_losses = []
            for is_support, examples in ((True, stream.supports), (False, stream.queries)):
                for example in examples:
                    output = core(*v1._inputs(example, embeddings), plastic_state=state)
                    action_features, action_mask = v3._action_inputs(example, embeddings)
                    targets = v3._targets(example)
                    decoded = decoder(
                        output.procedure_slots,
                        action_features,
                        action_mask,
                        memory_values=state.values,
                        memory_mask=state.strengths > 1.0e-6,
                        teacher_actions=targets,
                    )
                    sequence_loss = candidate_procedure_loss(decoded, targets)
                    retrieval_loss, mass = v4._relevance_loss(output, example)
                    episode_losses.append(sequence_loss + RETRIEVAL_WEIGHT * retrieval_loss)
                    relevance_rows.append(mass)
                    exact = v3._exact_visible(decoded, targets)
                    if is_support:
                        support_exact.append(exact)
                        state = _feedback(core, state, output, example, embeddings, detach_state=False)
                    else:
                        composition_exact.append(exact)
            loss = torch.stack(episode_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_LIMIT)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("V6 gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append({
            "epoch": epoch,
            "loss": v4._mean(losses),
            "support_teacher_exact": v4._mean(support_exact),
            "composition_teacher_exact": v4._mean(composition_exact),
            "relevance_mass": v4._mean(relevance_rows),
        })

    core.eval(); decoder.eval()
    persistent = v4r1._anchored_initial_state(core)
    with torch.inference_mode():
        for stream in train:
            persistent = _advance_supports(core, stream, embeddings, persistent)
    development_execution, development_attribution, development_rows = _evaluate_streams(core, decoder, development, embeddings)
    final_rows = []
    for stream_index, stream in enumerate(final):
        reset = v4r1._anchored_initial_state(core)
        persistent_only = persistent.detached_clone()
        local_only = _advance_supports(core, stream, embeddings, reset.detached_clone())
        live = _advance_supports(core, stream, embeddings, persistent.detached_clone())
        for query_index, example in enumerate(stream.queries):
            arms = {
                "full": _decode(core, decoder, example, embeddings, live),
                "reset_state": _decode(core, decoder, example, embeddings, reset),
                "coordinates_removed": _decode(core, decoder, example, embeddings, live, coordinates=False),
                "unrelated_evidence": _decode(core, decoder, example, embeddings, live, unrelated=True),
                "procedure_slots_removed": _decode(core, decoder, example, embeddings, live, remove_slots=True),
                "local_only": _decode(core, decoder, example, embeddings, local_only),
                "persistent_only": _decode(core, decoder, example, embeddings, persistent_only),
                "correspondence_removed": _decode(core, decoder, example, embeddings, live, correspondence=False),
            }
            scores, sequences = {}, {}
            for name, (_, decoded) in arms.items():
                pipeline = v3._decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
                scores[name] = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
                sequences[name] = decoded.selected_indices[0].detach().cpu().tolist()
            full_output = arms["full"][0]
            relevant = torch.tensor(example.relevant, device=device, dtype=torch.bool)
            final_rows.append({
                "stream_index": stream_index,
                "query_index": query_index,
                "scores": scores,
                "sequences": sequences,
                "target_evidence_mass": float(full_output.candidate_weights[0, relevant].sum()),
            })

    names = tuple(final_rows[0]["scores"])
    metrics = {name: v4._mean([row["scores"][name] for row in final_rows]) for name in names}
    attribution = v4._mean([row["target_evidence_mass"] for row in final_rows])
    supported = (
        metrics["full"] >= 0.60
        and all(metrics["full"] - metrics[name] >= 0.10 for name in CAUSAL_MARGIN_ARMS)
        and development_execution >= 0.60
        and attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "embedding_cache_sha256": EMBEDDING_CACHE_SHA256,
        "training": epoch_rows,
        "development_execution": development_execution,
        "development_target_evidence_attribution": development_attribution,
        "development": development_rows,
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "evaluation": final_rows,
        "runtime": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "core_config": asdict(core.config),
        "runner_sha256": v5._sha256(Path(__file__)),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "identity": IDENTITY,
        "foundation_digest": sealed["foundation_digest"],
        "core_type": "action-trace-matching-v1",
        "core_config": asdict(core.config),
        "maximum_trace_steps": core.maximum_trace_steps,
        "core_state_dict": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "decoder_config": sealed["decoder_config"],
        "decoder_state_dict": {name: value.detach().cpu() for name, value in decoder.state_dict().items()},
        "slot_anchor": sealed["slot_anchor"],
        "plastic_state": {
            "keys": persistent.keys.detach().cpu(),
            "values": persistent.values.detach().cpu(),
            "strengths": persistent.strengths.detach().cpu(),
            "step": persistent.step,
        },
    }, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "identity", "classification", "training", "development_execution",
        "metrics", "target_evidence_attribution", "runtime", "trainable_parameters",
    )}, indent=2))


if __name__ == "__main__":
    main()

