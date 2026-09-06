"""V8: V20 correspondence + Angler procedures + situated episode context.

Cognee-compatible candidate references determine which canonical public
episodes are available. Moving Origin coordinates remain inputs to Angler's
procedural core. Frozen V20/V19 produces public per-candidate correspondence
features, while a learned fusion and Angler's existing action-trace decoder
must compose the procedure. No deterministic procedure ordering is present.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Mapping, Sequence

import torch

from angler.reasoning import DonorCandidateFusion, candidate_procedure_loss
from experiments.runners import phase6_software_pipeline_reconstruction as legacy
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_oml_relation_representation as v20
import experiments.runners.compositional_procedure_v4 as v4
import experiments.runners.compositional_procedure_v4_r1 as v4r1
import experiments.runners.scaled_procedural_software_v1 as v1
import experiments.runners.structured_procedure_decoder_v3 as v3
import experiments.runners.structured_relational_transfer_v7 as v7
import experiments.runners.trace_conditioned_procedure_v5 as v5


IDENTITY = "angler.trifecta-correspondence-composition.v8"
SEED = 20260852
PARENT_CHECKPOINT_SHA256 = v7.PARENT_CHECKPOINT_SHA256
EMBEDDING_CACHE_SHA256 = v5.EMBEDDING_CACHE_SHA256
V19_CHECKPOINT_SHA256 = v20.SOURCE_CHECKPOINT_SHA256.lower()
V20_CHECKPOINT_SHA256 = (
    "d49e4caab64a264a11c675b295a8c453ac4475f078311eb7283a4f9a8817ef48"
)
EPOCHS = 8
FUSION_LEARNING_RATE = 1.0e-4
DECODER_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
DONOR_WIDTH = 97
FUSION_HIDDEN_WIDTH = 512

GATING_ARMS = (
    "donor_removed",
    "oml_removed",
    "moving_origin_removed",
    "cognee_unrelated",
    "procedure_slots_removed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--v19-checkpoint", required=True)
    parser.add_argument("--v20-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--angler-device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_partition(name: str, count: int, offset: int):
    commitments = legacy.software_pipeline_mechanism_partition(name)
    return tuple(
        legacy.make_software_pipeline_stream(
            v1.SEED + offset + index * 101,
            surface_seed=v1.SEED + 10_000_000 + offset + index * 103,
            supports_per_motif=2,
            queries=v1.QUERIES_PER_MECHANISM,
            maximum_steps=4,
            mechanism_commitment=commitment,
            mechanism_partition=name,
        )
        for index, commitment in enumerate(commitments[:count])
    )


def _raw_corpus():
    return (
        _raw_partition("train", v1.TRAIN_MECHANISMS, 1_000_000),
        _raw_partition("development", v1.DEVELOPMENT_MECHANISMS, 2_000_000),
        _raw_partition("final", v1.FINAL_MECHANISMS, 3_000_000),
    )


def _episode_maps(raw_groups):
    support_by_key = {}
    task_by_query_text = {}
    pair_by_ref = {}
    for streams in raw_groups:
        for stream in streams:
            for pair in stream.supports:
                query = v1.serialize_public_task(
                    legacy.replace(pair.learner, observations=())
                )
                target = v1.visible_procedure(pair.learner)
                key = (query, target)
                if key in support_by_key:
                    raise RuntimeError("V8 public support key repeated")
                support_by_key[key] = pair
                text = v1.support_evidence(pair.learner)
                ref = v1._event_ref(text)
                if ref in pair_by_ref:
                    raise RuntimeError("V8 episode projection reference repeated")
                pair_by_ref[ref] = pair
            for pair in stream.queries:
                text = v1.serialize_public_task(pair.learner)
                existing = task_by_query_text.get(text)
                if existing is not None and existing.public_digest != pair.learner.public_digest:
                    raise RuntimeError("V8 query serialization collision")
                task_by_query_text[text] = pair.learner
    return support_by_key, task_by_query_text, pair_by_ref


def _task_for_example(example, support_by_key, task_by_query_text):
    if example.pair is not None:
        return example.pair.learner
    try:
        return support_by_key[(example.query_text, example.target_text)].learner
    except KeyError as error:
        try:
            return task_by_query_text[example.query_text]
        except KeyError:
            raise RuntimeError("V8 example lost its canonical public task") from error


def _acquire_refs(controller, refs: Sequence[str], pair_by_ref):
    state = controller.initial_state()
    for ref in refs:
        try:
            pair = pair_by_ref[ref]
        except KeyError as error:
            raise RuntimeError("V8 retrieved an unknown episode reference") from error
        state = v19.acquire_v19_public_pipeline_traces(
            controller, pair.learner, state
        ).state
    return state


def public_donor_features(controller, task, state) -> torch.Tensor:
    """Return learned public candidate features; evaluator state is absent."""

    encoded = controller.encode_task(legacy.replace(task, observations=()))
    scores = controller._paired_graph_evidence_scores(encoded, state).unsqueeze(-1)
    features = torch.cat(
        (
            encoded.relation_component_embeddings,
            encoded.relation_context_embeddings,
            encoded.operator_embeddings,
            scores,
        ),
        dim=-1,
    )
    if features.shape != (len(task.grounded_candidates), DONOR_WIDTH):
        raise RuntimeError("V8 donor feature boundary changed")
    return features


def build_donor_bank(
    controller,
    prepared_groups,
    support_by_key,
    task_by_query_text,
    pair_by_ref,
    *,
    unrelated: bool = False,
    lesion: str | None = None,
) -> dict[int, torch.Tensor]:
    controller.requires_grad_(False)
    controller.eval()
    bank: dict[int, torch.Tensor] = {}
    context = controller.paired_graph_lesion(lesion) if lesion else nullcontext()
    with context, torch.inference_mode():
        for streams in prepared_groups:
            for stream in streams:
                for example in (*stream.supports, *stream.queries):
                    refs = example.unrelated_refs if unrelated else example.candidate_refs
                    state = _acquire_refs(controller, refs, pair_by_ref)
                    task = _task_for_example(example, support_by_key, task_by_query_text)
                    value = public_donor_features(controller, task, state).detach().cpu()
                    if id(example) in bank:
                        raise RuntimeError("V8 example identity repeated in donor bank")
                    bank[id(example)] = value
    return bank


def _action_features(example, embeddings, fusion, bank, *, include_donor=True):
    reference = next(fusion.parameters())
    width = next(iter(embeddings.values())).shape[0]
    semantic = torch.zeros(
        1, v3.MAXIMUM_ACTIONS, width, device=reference.device, dtype=reference.dtype
    )
    donor = torch.zeros(
        1, v3.MAXIMUM_ACTIONS, DONOR_WIDTH, device=reference.device, dtype=reference.dtype
    )
    mask = torch.zeros(
        1, v3.MAXIMUM_ACTIONS, device=reference.device, dtype=torch.bool
    )
    try:
        donor_rows = bank[id(example)]
    except KeyError as error:
        raise RuntimeError("V8 donor bank does not cover an example") from error
    for index, text in enumerate(example.action_texts):
        semantic[0, index] = embeddings[text].to(
            device=reference.device, dtype=reference.dtype
        )
        donor[0, index] = donor_rows[index].to(
            device=reference.device, dtype=reference.dtype
        )
        mask[0, index] = True
    return fusion(semantic, donor, include_donor=include_donor), mask


def _feedback(core, state, output, example, embeddings, fusion, bank, *, detach_state=True, include_donor=True):
    action_features, _ = _action_features(
        example, embeddings, fusion, bank, include_donor=include_donor
    )
    trace, trace_mask = v7._trace_from_actions(action_features, v3._targets(example))
    return core.apply_structured_action_trace_feedback(
        state,
        output,
        trace,
        trace_mask,
        torch.ones(1, device=output.query_state.device, dtype=output.query_state.dtype),
        detach_state=detach_state,
    )


def _advance_supports(
    core,
    stream,
    embeddings,
    state,
    fusion,
    bank,
    *,
    include_donor=True,
    core_unrelated=False,
):
    with torch.inference_mode():
        for example in stream.supports:
            inputs = (
                v1._inputs(
                    example,
                    embeddings,
                    refs=example.unrelated_refs,
                    temporal=example.unrelated_temporal,
                )
                if core_unrelated
                else v1._inputs(example, embeddings)
            )
            output = core(*inputs, plastic_state=state)
            state = _feedback(
                core,
                state,
                output,
                example,
                embeddings,
                fusion,
                bank,
                include_donor=include_donor,
            )
    return state


def _decode(
    core,
    decoder,
    example,
    embeddings,
    state,
    fusion,
    bank,
    *,
    coordinates=True,
    core_unrelated=False,
    include_donor=True,
    remove_slots=False,
    correspondence=True,
    teacher_actions=None,
):
    inputs = (
        v1._inputs(
            example,
            embeddings,
            refs=example.unrelated_refs,
            temporal=example.unrelated_temporal,
        )
        if core_unrelated
        else v1._inputs(example, embeddings, coordinates=coordinates)
    )
    output = core(*inputs, plastic_state=state)
    slots = torch.zeros_like(output.procedure_slots) if remove_slots else output.procedure_slots
    action_features, action_mask = _action_features(
        example, embeddings, fusion, bank, include_donor=include_donor
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


def _persistent_states(core, train, embeddings, fusion, banks):
    modes = {
        "full": (banks["full"], True, False),
        "donor_removed": (banks["full"], False, False),
        "oml_removed": (banks["oml_removed"], True, False),
        "paired_graph_removed": (banks["paired_graph_removed"], True, False),
        "cognee_unrelated": (banks["cognee_unrelated"], True, True),
    }
    states = {name: v4r1._anchored_initial_state(core) for name in modes}
    for stream in train:
        for name, (bank, include_donor, unrelated) in modes.items():
            states[name] = _advance_supports(
                core,
                stream,
                embeddings,
                states[name],
                fusion,
                bank,
                include_donor=include_donor,
                core_unrelated=unrelated,
            )
    return states


def _evaluate(core, decoder, streams, embeddings, fusion, banks, persistent):
    rows = []
    for stream_index, stream in enumerate(streams):
        local = v4r1._anchored_initial_state(core)
        states = {
            "full": _advance_supports(
                core, stream, embeddings, persistent["full"].detached_clone(), fusion, banks["full"]
            ),
            "donor_removed": _advance_supports(
                core, stream, embeddings, persistent["donor_removed"].detached_clone(), fusion,
                banks["full"], include_donor=False
            ),
            "oml_removed": _advance_supports(
                core, stream, embeddings, persistent["oml_removed"].detached_clone(), fusion,
                banks["oml_removed"]
            ),
            "paired_graph_removed": _advance_supports(
                core, stream, embeddings, persistent["paired_graph_removed"].detached_clone(), fusion,
                banks["paired_graph_removed"]
            ),
            "cognee_unrelated": _advance_supports(
                core, stream, embeddings, persistent["cognee_unrelated"].detached_clone(), fusion,
                banks["cognee_unrelated"], core_unrelated=True
            ),
            "local_only": _advance_supports(
                core, stream, embeddings, local.detached_clone(), fusion, banks["full"]
            ),
        }
        for query_index, example in enumerate(stream.queries):
            with torch.inference_mode():
                arms = {
                    "full": _decode(core, decoder, example, embeddings, states["full"], fusion, banks["full"]),
                    "donor_removed": _decode(core, decoder, example, embeddings, states["donor_removed"], fusion, banks["full"], include_donor=False),
                    "oml_removed": _decode(core, decoder, example, embeddings, states["oml_removed"], fusion, banks["oml_removed"]),
                    "paired_graph_removed": _decode(core, decoder, example, embeddings, states["paired_graph_removed"], fusion, banks["paired_graph_removed"]),
                    "moving_origin_removed": _decode(core, decoder, example, embeddings, states["full"], fusion, banks["full"], coordinates=False),
                    "cognee_unrelated": _decode(core, decoder, example, embeddings, states["cognee_unrelated"], fusion, banks["cognee_unrelated"], core_unrelated=True),
                    "procedure_slots_removed": _decode(core, decoder, example, embeddings, states["full"], fusion, banks["full"], remove_slots=True),
                    "correspondence_removed": _decode(core, decoder, example, embeddings, states["full"], fusion, banks["full"], correspondence=False),
                    "reset_state": _decode(core, decoder, example, embeddings, local, fusion, banks["full"]),
                    "local_only": _decode(core, decoder, example, embeddings, states["local_only"], fusion, banks["full"]),
                    "persistent_only": _decode(core, decoder, example, embeddings, persistent["full"], fusion, banks["full"]),
                }
            scores = {}
            sequences = {}
            for name, (_, decoded) in arms.items():
                pipeline = v3._decoder_pipeline(
                    example.pair.learner, decoded.selected_indices[0]
                )
                scores[name] = float(
                    v1.judge_software_pipeline_attempt(example.pair, pipeline)
                )
                sequences[name] = decoded.selected_indices[0].detach().cpu().tolist()
            full_output = arms["full"][0]
            relevant = torch.tensor(
                example.relevant,
                device=full_output.query_state.device,
                dtype=torch.bool,
            )
            rows.append(
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
    names = tuple(rows[0]["scores"])
    metrics = {
        name: v4._mean([row["scores"][name] for row in rows]) for name in names
    }
    attribution = v4._mean([row["target_evidence_mass"] for row in rows])
    return metrics, attribution, rows


def main() -> None:
    args = parse_args()
    parent = Path(args.parent_checkpoint)
    cache = Path(args.embedding_cache)
    output = Path(args.output)
    checkpoint = Path(args.checkpoint)
    if output.exists() or checkpoint.exists():
        raise RuntimeError("V8 output identity already exists")
    for path, expected, label in (
        (parent, PARENT_CHECKPOINT_SHA256, "V6 parent"),
        (cache, EMBEDDING_CACHE_SHA256, "embedding cache"),
        (Path(args.v19_checkpoint), V19_CHECKPOINT_SHA256, "V19 donor"),
        (Path(args.v20_checkpoint), V20_CHECKPOINT_SHA256, "V20 donor"),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"V8 {label} identity mismatch")
    device = torch.device(args.angler_device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise ValueError("V8 requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()

    evidence, prepared_train, development, final = v1.prepare_corpus()
    train = v4r1._public_training_streams(prepared_train)
    raw_groups = _raw_corpus()
    support_by_key, task_by_query_text, pair_by_ref = _episode_maps(raw_groups)
    sealed = torch.load(parent, map_location="cpu", weights_only=True)
    cached = torch.load(cache, map_location="cpu", weights_only=True)
    texts = v3._needed_texts(evidence, (train, development, final))
    if cached["foundation_digest"] != sealed["foundation_digest"] or tuple(cached["texts"]) != texts:
        raise RuntimeError("V8 frozen Qwen cache does not cover the corpus")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    core, decoder = v7._load_parent(sealed, device)
    decoder.requires_grad_(True)
    fusion = DonorCandidateFusion(
        semantic_width=core.config.content_width,
        donor_width=DONOR_WIDTH,
        hidden_width=FUSION_HIDDEN_WIDTH,
    ).to(device)
    donor_system = v20.load_oml_checkpoint(
        args.v20_checkpoint, args.v19_checkpoint, device=device
    )
    prepared_groups = (train, development, final)
    banks = {
        "full": build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
        ),
        "oml_removed": build_donor_bank(
            donor_system.source_v19.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
        ),
        "paired_graph_removed": build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
            lesion="zero_residual",
        ),
        "cognee_unrelated": build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
            unrelated=True,
        ),
    }
    donor_system = None
    trainable = [*fusion.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(
        (
            {"params": fusion.parameters(), "lr": FUSION_LEARNING_RATE},
            {"params": decoder.parameters(), "lr": DECODER_LEARNING_RATE},
        ),
        weight_decay=WEIGHT_DECAY,
    )
    core.eval()
    epoch_rows = []
    for epoch in range(EPOCHS):
        decoder.train()
        fusion.train()
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        support_exact = []
        composition_exact = []
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
                        fusion,
                        banks["full"],
                        teacher_actions=targets,
                    )
                    sequence = candidate_procedure_loss(decoded, targets)
                    retrieval, _ = v4._relevance_loss(output, example)
                    episode_losses.append(sequence + v4.RETRIEVAL_WEIGHT * retrieval)
                    exact = v3._exact_visible(decoded, targets)
                    if is_support:
                        support_exact.append(exact)
                        state = _feedback(
                            core,
                            state,
                            output,
                            example,
                            embeddings,
                            fusion,
                            banks["full"],
                            detach_state=False,
                        )
                    else:
                        composition_exact.append(exact)
            loss = torch.stack(episode_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, v4.GRADIENT_LIMIT)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("V8 gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": v4._mean(losses),
                "support_teacher_exact": v4._mean(support_exact),
                "composition_teacher_exact": v4._mean(composition_exact),
            }
        )

    decoder.eval()
    fusion.eval()
    persistent = _persistent_states(core, train, embeddings, fusion, banks)
    development_metrics, development_attribution, development_rows = _evaluate(
        core, decoder, development, embeddings, fusion, banks, persistent
    )
    final_metrics, final_attribution, final_rows = _evaluate(
        core, decoder, final, embeddings, fusion, banks, persistent
    )
    full = final_metrics["full"]
    supported = (
        development_metrics["full"] >= 0.60
        and full >= 0.60
        and full - final_metrics["donor_removed"] >= 0.20
        and full - final_metrics["oml_removed"] >= 0.10
        and full - final_metrics["moving_origin_removed"] >= 0.10
        and full - final_metrics["cognee_unrelated"] >= 0.10
        and full - final_metrics["procedure_slots_removed"] >= 0.10
        and final_attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "TRIFECTA_COMPOSITION_SUPPORTED" if supported else "NOT_SUPPORTED",
        "first_result_accepted_without_tuning": True,
        "training": epoch_rows,
        "development_metrics": development_metrics,
        "development_target_evidence_attribution": development_attribution,
        "development": development_rows,
        "metrics": final_metrics,
        "target_evidence_attribution": final_attribution,
        "evaluation": final_rows,
        "source": {
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "embedding_cache_sha256": EMBEDDING_CACHE_SHA256,
            "v19_checkpoint_sha256": V19_CHECKPOINT_SHA256,
            "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
            "donor_arm": v20.ARM_SECOND_ORDER,
            "cognee_boundary": "validated episode refs plus canonical public sidecars",
            "moving_origin_boundary": "situated temporal coordinates in Angler core",
        },
        "trainable_parameters": sum(value.numel() for value in trainable),
        "runtime": {
            "device": str(device),
            "name": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "deterministic_solver_used": False,
        "hidden_fields_used_for_training": False,
        "runner_sha256": _sha256(Path(__file__)),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "source": report["source"],
            "foundation_digest": sealed["foundation_digest"],
            "fusion_config": {
                "semantic_width": core.config.content_width,
                "donor_width": DONOR_WIDTH,
                "hidden_width": FUSION_HIDDEN_WIDTH,
            },
            "fusion_state_dict": {
                name: value.detach().cpu() for name, value in fusion.state_dict().items()
            },
            "decoder_config": sealed["decoder_config"],
            "decoder_state_dict": {
                name: value.detach().cpu() for name, value in decoder.state_dict().items()
            },
            "plastic_state": {
                "keys": persistent["full"].keys.detach().cpu(),
                "values": persistent["full"].values.detach().cpu(),
                "strengths": persistent["full"].strengths.detach().cpu(),
                "step": persistent["full"].step,
            },
        },
        checkpoint,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "identity",
                    "classification",
                    "training",
                    "development_metrics",
                    "metrics",
                    "target_evidence_attribution",
                    "runtime",
                    "trainable_parameters",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

