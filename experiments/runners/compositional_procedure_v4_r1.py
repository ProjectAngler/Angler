"""Recover V4 with diverse plastic writes and sealed evaluation labels."""

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

from angler.reasoning import PlasticProcedureState, candidate_procedure_loss
from angler.runtime import encode_detached_segments, foundation_tensor_digest, freeze_knowledge_model
import experiments.runners.compositional_procedure_v4 as v4
import experiments.runners.scaled_procedural_software_v1 as v1
import experiments.runners.structured_procedure_decoder_v3 as v3


IDENTITY = "angler.compositional-procedure.v4-r1"
SEED = 20260847
PARENT_V4_RUNNER_SHA256 = "511d845bbbb0d6d8c9521e4c65bb101c3e37c6164fc5c64459e60d4ebebf2dff"
SLOT_ANCHOR_SEED = 2026084701
SLOT_ANCHOR_SCALE = 1.0
EPOCHS = v4.EPOCHS
CORE_LEARNING_RATE = v4.CORE_LEARNING_RATE
DECODER_LEARNING_RATE = v4.DECODER_LEARNING_RATE
RETRIEVAL_WEIGHT = v4.RETRIEVAL_WEIGHT
GRADIENT_LIMIT = v4.GRADIENT_LIMIT
CAUSAL_MARGIN_ARMS = (
    "reset_state",
    "coordinates_removed",
    "unrelated_evidence",
    "procedure_slots_removed",
)
DIAGNOSTIC_ARMS = ("local_only", "persistent_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--v3-result", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--angler-device", default="cuda:1")
    return parser.parse_args()


def _resolve_device_split(
    qwen_device: str,
    angler_device: str,
    *,
    device_count: int | None = None,
) -> tuple[torch.device, torch.device]:
    qwen = torch.device(qwen_device)
    angler = torch.device(angler_device)
    count = torch.cuda.device_count() if device_count is None else device_count
    for name, device in (("Qwen", qwen), ("Angler", angler)):
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"{name} device must be an explicit CUDA ordinal")
        if device.index < 0 or device.index >= count:
            raise ValueError(f"{name} CUDA ordinal is unavailable")
    if qwen == angler:
        raise ValueError("Qwen and Angler must use different CUDA devices")
    return qwen, angler


def _slot_anchor_keys(config, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Create content-neutral fixed addresses without activating reset memory."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SLOT_ANCHOR_SEED)
    bits = torch.randint(
        0,
        2,
        (config.plastic_slots, config.model_width),
        generator=generator,
        device="cpu",
        dtype=torch.int8,
    )
    anchors = bits.to(torch.float32).mul_(2.0).sub_(1.0).mul_(SLOT_ANCHOR_SCALE)
    return anchors.to(device=device, dtype=dtype)


def _slot_anchor_digest(config) -> str:
    anchors = _slot_anchor_keys(config).contiguous()
    return hashlib.sha256(anchors.numpy().tobytes()).hexdigest()


def _slot_anchor_record(config) -> dict[str, object]:
    return {
        "algorithm": "seeded-rademacher-v1",
        "seed": SLOT_ANCHOR_SEED,
        "scale": SLOT_ANCHOR_SCALE,
        "shape": [config.plastic_slots, config.model_width],
        "sha256": _slot_anchor_digest(config),
        "initial_values": "zero",
        "initial_strengths": "zero",
    }


def _anchored_initial_state(core) -> PlasticProcedureState:
    reference = next(core.parameters())
    keys = _slot_anchor_keys(core.config, device=reference.device, dtype=reference.dtype)
    return PlasticProcedureState(
        keys=keys,
        values=torch.zeros_like(keys),
        strengths=torch.zeros(
            core.config.plastic_slots,
            device=reference.device,
            dtype=reference.dtype,
        ),
        step=0,
    )


def supervised_composed_target(example: v1.PreparedExample) -> str:
    return v4.supervised_composed_target(example)


def _public_training_streams(streams):
    """Materialize train labels once, then discard every hidden pair object."""

    return tuple(
        replace(
            stream,
            queries=tuple(
                replace(
                    example,
                    target_text=supervised_composed_target(example),
                    pair=None,
                )
                for example in stream.queries
            ),
        )
        for stream in streams
    )


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
            _anchored_initial_state(core)
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
    return v4._mean(scores), v4._mean(attribution), rows


def main() -> None:
    args = parse_args()
    parent_path = Path(args.parent_checkpoint)
    v3_path = Path(args.v3_result)
    cache_path = Path(args.embedding_cache)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if any(path.exists() for path in (cache_path, output_path, checkpoint_path)):
        raise RuntimeError("compositional V4-R1 output identity already exists")
    if v4._sha256(Path(v4.__file__)) != PARENT_V4_RUNNER_SHA256:
        raise RuntimeError("preserved V4 runner identity mismatch")
    if v4._sha256(parent_path) != v4.PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent core checkpoint identity mismatch")
    if v4._sha256(v3_path) != v4.V3_RESULT_SHA256:
        raise RuntimeError("V3 result identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("compositional V4-R1 requires CUDA")
    qwen_device, angler_device = _resolve_device_split(
        args.qwen_device,
        args.angler_device,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Initialize and meter each CUDA context explicitly.  PyTorch's memory
    # API does not accept an uninitialized ``torch.device`` on every build.
    torch.cuda.set_device(qwen_device)
    torch.cuda.reset_peak_memory_stats(qwen_device.index)
    torch.cuda.set_device(angler_device)
    torch.cuda.reset_peak_memory_stats(angler_device.index)
    evidence, raw_train, development, final = v1.prepare_corpus()
    train = _public_training_streams(raw_train)
    torch.cuda.set_device(qwen_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": qwen_device.index},
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
    torch.cuda.synchronize(qwen_device)
    qwen_peak_cuda_bytes = int(torch.cuda.max_memory_allocated(qwen_device))
    del model, tokenizer
    gc.collect()
    torch.cuda.set_device(qwen_device)
    torch.cuda.empty_cache()

    # The preserved V1/V3/V4 helpers use bare ``cuda`` allocations. Setting
    # the current device here keeps all trainable Angler tensors on the
    # declared second GPU without modifying any consumed parent source.
    torch.cuda.set_device(angler_device)
    core = v4._load_core(sealed)
    decoder = v4._new_decoder(core.config)
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
            state = _anchored_initial_state(core)
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
                        teacher_actions=targets,
                    )
                    sequence_loss = candidate_procedure_loss(decoded, targets)
                    retrieval_loss, mass = v4._relevance_loss(output, example)
                    episode_losses.append(sequence_loss + RETRIEVAL_WEIGHT * retrieval_loss)
                    relevance_rows.append(mass)
                    exact = v3._exact_visible(decoded, targets)
                    if is_support:
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
                "loss": v4._mean(losses),
                "support_teacher_exact": v4._mean(support_exact),
                "composition_teacher_exact": v4._mean(composition_exact),
                "relevance_mass": v4._mean(relevance_rows),
            }
        )

    core.eval()
    decoder.eval()
    persistent = _anchored_initial_state(core)
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
        reset = _anchored_initial_state(core)
        persistent_only = persistent.detached_clone()
        local_only = _advance_supports(core, stream, embeddings, reset.detached_clone())
        live = _advance_supports(core, stream, embeddings, persistent.detached_clone())
        for query_index, example in enumerate(stream.queries):
            arms = {
                "full": v3._decode(core, decoder, example, embeddings, live),
                "reset_state": v3._decode(core, decoder, example, embeddings, reset),
                "coordinates_removed": v3._decode(
                    core, decoder, example, embeddings, live, coordinates=False
                ),
                "unrelated_evidence": v3._decode(
                    core, decoder, example, embeddings, live, unrelated=True
                ),
                "procedure_slots_removed": v3._decode(
                    core, decoder, example, embeddings, live, remove_slots=True
                ),
                "local_only": v3._decode(core, decoder, example, embeddings, local_only),
                "persistent_only": v3._decode(
                    core, decoder, example, embeddings, persistent_only
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
        name: v4._mean([float(row["scores"][name]) for row in final_rows])
        for name in arm_names
    }
    attribution = v4._mean([row["target_evidence_mass"] for row in final_rows])
    supported = (
        metrics["full"] >= 0.60
        and all(metrics["full"] - metrics[name] >= 0.10 for name in CAUSAL_MARGIN_ARMS)
        and development_execution >= 0.60
        and attribution >= 0.75
    )
    anchor_record = _slot_anchor_record(core.config)
    report = {
        "identity": IDENTITY,
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "parent_v4_runner_sha256": PARENT_V4_RUNNER_SHA256,
        "parent_checkpoint_sha256": v4.PARENT_CHECKPOINT_SHA256,
        "v3_result_sha256": v4.V3_RESULT_SHA256,
        "slot_anchor": anchor_record,
        "causal_margin_arms": list(CAUSAL_MARGIN_ARMS),
        "diagnostic_arms": list(DIAGNOSTIC_ARMS),
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
        "embedding_cache_sha256": v4._sha256(cache_path),
        "runtime": {
            "device": torch.cuda.get_device_name(angler_device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(angler_device)),
            "placement": {
                "qwen": {
                    "device": str(qwen_device),
                    "name": torch.cuda.get_device_name(qwen_device),
                    "peak_cuda_bytes": qwen_peak_cuda_bytes,
                },
                "angler": {
                    "device": str(angler_device),
                    "name": torch.cuda.get_device_name(angler_device),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(angler_device)),
                },
            },
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "core": {"config": asdict(core.config), "parameters": core.parameter_count},
        "decoder_parameters": sum(value.numel() for value in decoder.parameters()),
        "runner_sha256": v4._sha256(Path(__file__)),
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
            "slot_anchor": {
                **anchor_record,
                "keys": _slot_anchor_keys(core.config),
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
                    "slot_anchor",
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
