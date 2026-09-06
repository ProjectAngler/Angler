"""Learn a typed relational correction for V6's collapsed text features."""

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
    SemanticRelationalFusion,
    StructuredActionTraceMatcherDecoder,
    StructuredActionTraceProceduralCore,
    StructuredRelationalEncoder,
    build_relational_incidence,
    candidate_procedure_loss,
    parse_public_relations,
)
import experiments.runners.action_trace_matching_v6 as v6
import experiments.runners.compositional_procedure_v4 as v4
import experiments.runners.compositional_procedure_v4_r1 as v4r1
import experiments.runners.scaled_procedural_software_v1 as v1
import experiments.runners.structured_procedure_decoder_v3 as v3
import experiments.runners.trace_conditioned_procedure_v5 as v5


IDENTITY = "angler.structured-relational-transfer.v7"
SEED = 20260850
PARENT_CHECKPOINT_SHA256 = "08a966424ca678e9742a8d80b985ba48fc0c0ff8e3df4bc86c0262b9c53fad7a"
EMBEDDING_CACHE_SHA256 = v5.EMBEDDING_CACHE_SHA256
EPOCHS = 8
LEARNING_RATE = 1.0e-4
RELATIONAL_WIDTH = 128
RELATIONAL_STEPS = 6
RELATIONAL_SLOTS = 8
FUSION_HIDDEN_WIDTH = 512
CAUSAL_MARGIN_ARMS = (*v4r1.CAUSAL_MARGIN_ARMS, "structured_removed")
DIAGNOSTIC_ARMS = (
    *v4r1.DIAGNOSTIC_ARMS,
    "correspondence_removed",
    "structured_removed",
    "topology_removed",
    "action_semantics_removed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--angler-device", default="cuda:1")
    return parser.parse_args()


def _load_parent(sealed, device):
    from angler.reasoning import ProceduralCoreConfig

    config = ProceduralCoreConfig(**sealed["core_config"])
    core = StructuredActionTraceProceduralCore(
        config,
        maximum_trace_steps=sealed["maximum_trace_steps"],
    ).to(device)
    core.load_state_dict(sealed["core_state_dict"], strict=True)
    decoder = StructuredActionTraceMatcherDecoder(**sealed["decoder_config"]).to(device)
    decoder.load_state_dict(sealed["decoder_state_dict"], strict=True)
    core.requires_grad_(False)
    decoder.requires_grad_(False)
    return core, decoder


def _relational_modules(content_width: int, device: torch.device):
    encoder = StructuredRelationalEncoder(
        width=RELATIONAL_WIDTH,
        reasoning_steps=RELATIONAL_STEPS,
        workspace_slots=RELATIONAL_SLOTS,
    ).to(device)
    fusion = SemanticRelationalFusion(
        semantic_width=content_width,
        relational_width=RELATIONAL_WIDTH,
        hidden_width=FUSION_HIDDEN_WIDTH,
    ).to(device)
    return encoder, fusion


def _action_features(
    example,
    embeddings,
    encoder,
    fusion,
    *,
    structured: bool = True,
    topology: bool = True,
    semantics: bool = True,
):
    reference = next(encoder.parameters())
    width = next(iter(embeddings.values())).shape[0]
    semantic = torch.zeros(
        1,
        v3.MAXIMUM_ACTIONS,
        width,
        device=reference.device,
        dtype=reference.dtype,
    )
    mask = torch.zeros(
        1,
        v3.MAXIMUM_ACTIONS,
        device=reference.device,
        dtype=torch.bool,
    )
    for index, text in enumerate(example.action_texts):
        semantic[0, index] = embeddings[text].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        mask[0, index] = True
    graph = parse_public_relations(example.query_text, example.action_texts)
    tensors = build_relational_incidence(
        graph,
        device=semantic.device,
        dtype=semantic.dtype,
        include_topology_edges=topology,
    )
    relational, _ = encoder(tensors)
    padded = relational.new_zeros((1, v3.MAXIMUM_ACTIONS, relational.shape[-1]))
    padded[:, : relational.shape[1]] = relational
    return (
        fusion(
            semantic,
            padded,
            include_semantic=semantics,
            include_relational=structured,
        ),
        mask,
    )


def _trace_from_actions(action_features: torch.Tensor, target: torch.Tensor):
    trace = action_features.new_zeros((1, v3.MAXIMUM_STEPS, action_features.shape[-1]))
    mask = torch.zeros((1, v3.MAXIMUM_STEPS), device=action_features.device, dtype=torch.bool)
    position = 0
    for value in target[0].tolist():
        if value in (-100, v3.MAXIMUM_ACTIONS):
            break
        if value < 0 or value >= v3.MAXIMUM_ACTIONS:
            raise ValueError("trace target is outside the public candidates")
        trace[0, position] = action_features[0, value]
        mask[0, position] = True
        position += 1
    if position == 0:
        raise ValueError("successful procedure trace contains no action")
    return trace, mask


def _feedback(
    core,
    state,
    output,
    example,
    embeddings,
    encoder,
    fusion,
    *,
    detach_state=True,
    structured=True,
    topology=True,
    semantics=True,
):
    action_features, _ = _action_features(
        example,
        embeddings,
        encoder,
        fusion,
        structured=structured,
        topology=topology,
        semantics=semantics,
    )
    trace, mask = _trace_from_actions(action_features, v3._targets(example))
    return core.apply_structured_action_trace_feedback(
        state,
        output,
        trace,
        mask,
        torch.ones(1, device=output.query_state.device, dtype=output.query_state.dtype),
        detach_state=detach_state,
    )


def _advance_supports(
    core,
    stream,
    embeddings,
    state,
    encoder,
    fusion,
    *,
    structured=True,
    topology=True,
    semantics=True,
):
    with torch.inference_mode():
        for example in stream.supports:
            output = core(*v1._inputs(example, embeddings), plastic_state=state)
            state = _feedback(
                core,
                state,
                output,
                example,
                embeddings,
                encoder,
                fusion,
                structured=structured,
                topology=topology,
                semantics=semantics,
            )
    return state


def _decode(
    core,
    decoder,
    example,
    embeddings,
    state,
    encoder,
    fusion,
    *,
    coordinates=True,
    unrelated=False,
    remove_slots=False,
    correspondence=True,
    structured=True,
    topology=True,
    semantics=True,
    teacher_actions=None,
):
    inputs = (
        v1._inputs(
            example,
            embeddings,
            refs=example.unrelated_refs,
            temporal=example.unrelated_temporal,
        )
        if unrelated
        else v1._inputs(example, embeddings, coordinates=coordinates)
    )
    output = core(*inputs, plastic_state=state)
    slots = torch.zeros_like(output.procedure_slots) if remove_slots else output.procedure_slots
    action_features, action_mask = _action_features(
        example,
        embeddings,
        encoder,
        fusion,
        structured=structured,
        topology=topology,
        semantics=semantics,
    )
    decoded = decoder(
        slots,
        action_features,
        action_mask,
        memory_values=state.values,
        memory_mask=state.strengths > 1.0e-6,
        correspondence=correspondence,
        teacher_actions=teacher_actions,
    )
    return output, decoded


def _evaluate_streams(core, decoder, streams, embeddings, encoder, fusion):
    scores, attribution, rows = [], [], []
    for stream_index, stream in enumerate(streams):
        state = _advance_supports(
            core,
            stream,
            embeddings,
            v4r1._anchored_initial_state(core),
            encoder,
            fusion,
        )
        for query_index, example in enumerate(stream.queries):
            with torch.inference_mode():
                output, decoded = _decode(
                    core, decoder, example, embeddings, state, encoder, fusion
                )
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


def _persistent_states(core, train, embeddings, encoder, fusion):
    modes = {
        "full": {},
        "structured_removed": {"structured": False},
        "topology_removed": {"topology": False},
        "action_semantics_removed": {"semantics": False},
    }
    states = {name: v4r1._anchored_initial_state(core) for name in modes}
    for stream in train:
        for name, options in modes.items():
            states[name] = _advance_supports(
                core,
                stream,
                embeddings,
                states[name],
                encoder,
                fusion,
                **options,
            )
    return states


def main() -> None:
    args = parse_args()
    parent_path, cache_path = Path(args.parent_checkpoint), Path(args.embedding_cache)
    output_path, checkpoint_path = Path(args.output), Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("V7 output identity already exists")
    if v5._sha256(parent_path) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("V6 parent checkpoint identity mismatch")
    if v5._sha256(cache_path) != EMBEDDING_CACHE_SHA256:
        raise RuntimeError("embedding cache identity mismatch")
    device = torch.device(args.angler_device)
    if (
        not torch.cuda.is_available()
        or device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
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
        raise RuntimeError("frozen cache does not match the V7 corpus/foundation")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    core, decoder = _load_parent(sealed, device)
    encoder, fusion = _relational_modules(core.config.content_width, device)
    trainable = [*encoder.parameters(), *fusion.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1.0e-4)
    epoch_rows = []
    core.eval(); decoder.eval()
    for epoch in range(EPOCHS):
        encoder.train(); fusion.train()
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses, support_exact, composition_exact, relevance_rows = [], [], [], []
        for stream_index in order:
            stream = train[stream_index]
            state = v4r1._anchored_initial_state(core)
            episode_losses = []
            for is_support, examples in ((True, stream.supports), (False, stream.queries)):
                for example in examples:
                    targets = v3._targets(example)
                    output, decoded = _decode(
                        core,
                        decoder,
                        example,
                        embeddings,
                        state,
                        encoder,
                        fusion,
                        teacher_actions=targets,
                    )
                    sequence_loss = candidate_procedure_loss(decoded, targets)
                    retrieval_loss, mass = v4._relevance_loss(output, example)
                    episode_losses.append(sequence_loss + v4.RETRIEVAL_WEIGHT * retrieval_loss)
                    relevance_rows.append(mass)
                    exact = v3._exact_visible(decoded, targets)
                    if is_support:
                        support_exact.append(exact)
                        state = _feedback(
                            core,
                            state,
                            output,
                            example,
                            embeddings,
                            encoder,
                            fusion,
                            detach_state=False,
                        )
                    else:
                        composition_exact.append(exact)
            loss = torch.stack(episode_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, v4.GRADIENT_LIMIT)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("V7 gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append({
            "epoch": epoch,
            "loss": v4._mean(losses),
            "support_teacher_exact": v4._mean(support_exact),
            "composition_teacher_exact": v4._mean(composition_exact),
            "relevance_mass": v4._mean(relevance_rows),
        })

    encoder.eval(); fusion.eval()
    persistent = _persistent_states(core, train, embeddings, encoder, fusion)
    development_execution, development_attribution, development_rows = _evaluate_streams(
        core, decoder, development, embeddings, encoder, fusion
    )
    final_rows = []
    for stream_index, stream in enumerate(final):
        reset = v4r1._anchored_initial_state(core)
        full_live = _advance_supports(
            core, stream, embeddings, persistent["full"].detached_clone(), encoder, fusion
        )
        local_only = _advance_supports(
            core, stream, embeddings, reset.detached_clone(), encoder, fusion
        )
        component_states = {}
        for name, options in (
            ("structured_removed", {"structured": False}),
            ("topology_removed", {"topology": False}),
            ("action_semantics_removed", {"semantics": False}),
        ):
            component_states[name] = _advance_supports(
                core,
                stream,
                embeddings,
                persistent[name].detached_clone(),
                encoder,
                fusion,
                **options,
            )
        for query_index, example in enumerate(stream.queries):
            with torch.inference_mode():
                arms = {
                    "full": _decode(core, decoder, example, embeddings, full_live, encoder, fusion),
                    "reset_state": _decode(core, decoder, example, embeddings, reset, encoder, fusion),
                    "coordinates_removed": _decode(core, decoder, example, embeddings, full_live, encoder, fusion, coordinates=False),
                    "unrelated_evidence": _decode(core, decoder, example, embeddings, full_live, encoder, fusion, unrelated=True),
                    "procedure_slots_removed": _decode(core, decoder, example, embeddings, full_live, encoder, fusion, remove_slots=True),
                    "local_only": _decode(core, decoder, example, embeddings, local_only, encoder, fusion),
                    "persistent_only": _decode(core, decoder, example, embeddings, persistent["full"], encoder, fusion),
                    "correspondence_removed": _decode(core, decoder, example, embeddings, full_live, encoder, fusion, correspondence=False),
                    "structured_removed": _decode(core, decoder, example, embeddings, component_states["structured_removed"], encoder, fusion, structured=False),
                    "topology_removed": _decode(core, decoder, example, embeddings, component_states["topology_removed"], encoder, fusion, topology=False),
                    "action_semantics_removed": _decode(core, decoder, example, embeddings, component_states["action_semantics_removed"], encoder, fusion, semantics=False),
                }
            scores, sequences = {}, {}
            for name, (_, decoded) in arms.items():
                pipeline = v3._decoder_pipeline(example.pair.learner, decoded.selected_indices[0])
                scores[name] = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
                sequences[name] = decoded.selected_indices[0].detach().cpu().tolist()
            output = arms["full"][0]
            relevant = torch.tensor(example.relevant, device=device, dtype=torch.bool)
            final_rows.append({
                "stream_index": stream_index,
                "query_index": query_index,
                "scores": scores,
                "sequences": sequences,
                "target_evidence_mass": float(output.candidate_weights[0, relevant].sum()),
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
            "torch_threads": torch.get_num_threads(),
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "relational_config": {
            "width": RELATIONAL_WIDTH,
            "steps": RELATIONAL_STEPS,
            "workspace_slots": RELATIONAL_SLOTS,
            "feature_boundary": "typed-incidence-v1",
        },
        "runner_sha256": v5._sha256(Path(__file__)),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "identity": IDENTITY,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "foundation_digest": sealed["foundation_digest"],
        "encoder_state_dict": {name: value.detach().cpu() for name, value in encoder.state_dict().items()},
        "fusion_state_dict": {name: value.detach().cpu() for name, value in fusion.state_dict().items()},
        "plastic_state": {
            "keys": persistent["full"].keys.detach().cpu(),
            "values": persistent["full"].values.detach().cpu(),
            "strengths": persistent["full"].strengths.detach().cpu(),
            "step": persistent["full"].step,
        },
        "relational_config": report["relational_config"],
    }, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "identity", "classification", "training", "development_execution",
        "metrics", "target_evidence_attribution", "runtime", "trainable_parameters",
    )}, indent=2))


if __name__ == "__main__":
    main()
