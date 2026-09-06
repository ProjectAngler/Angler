"""V10: fresh neural graph planner over the frozen trifecta boundary."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import random
import time

import torch

from angler.reasoning import (
    DonorCandidateFusion,
    PUBLIC_ROLE_WIDTH,
    RoleAwareGraphProcedureDecoder,
    build_public_candidate_roles,
    candidate_procedure_loss,
    parse_public_relations,
)
from experiments.runners import compositional_procedure_v4 as v4
from experiments.runners import compositional_procedure_v4_r1 as v4r1
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import scaled_procedural_software_v1 as v1
from experiments.runners import structured_procedure_decoder_v3 as v3
from experiments.runners import structured_relational_transfer_v7 as v7
from experiments.runners import trifecta_correspondence_composition_v8 as v8
from experiments.runners import trifecta_edge_procedure_v9 as v9


IDENTITY = "angler.trifecta-role-graph.v10"
SEED = 20260854
EPOCHS = 8
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
V8_CHECKPOINT_SHA256 = v9.V8_CHECKPOINT_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--v19-checkpoint", required=True)
    parser.add_argument("--v20-checkpoint", required=True)
    parser.add_argument("--v8-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--angler-device", default="cuda:0")
    return parser.parse_args()


def public_role_features(example, reference: torch.Tensor) -> torch.Tensor:
    graph = parse_public_relations(example.query_text, example.action_texts)
    return build_public_candidate_roles(
        graph,
        maximum_actions=v3.MAXIMUM_ACTIONS,
        device=reference.device,
        dtype=reference.dtype,
    )


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
    donor_prior=True,
    edge_reasoning=True,
    coverage=True,
    include_roles=True,
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
    with torch.no_grad():
        core_output = core(*inputs, plastic_state=state)
        slots = (
            torch.zeros_like(core_output.procedure_slots)
            if remove_slots
            else core_output.procedure_slots
        )
        actions, action_mask, edges, donor_scores = v9._decoder_inputs(
            example, embeddings, fusion, bank, include_donor=include_donor
        )
        roles = public_role_features(example, actions)
    decoded = decoder(
        slots.detach(),
        actions,
        action_mask,
        role_features=roles,
        include_roles=include_roles,
        edge_features=edges,
        donor_scores=donor_scores,
        memory_values=state.values,
        memory_mask=state.strengths > 1.0e-6,
        correspondence=correspondence,
        donor_prior=donor_prior,
        edge_reasoning=edge_reasoning,
        coverage=coverage,
        teacher_actions=teacher_actions,
    )
    return core_output, decoded


@contextmanager
def _v9_evaluator_decode():
    original = v9._decode
    v9._decode = _decode
    try:
        yield
    finally:
        v9._decode = original


def _role_removed(core, decoder, streams, embeddings, fusion, bank, persistent):
    scores = []
    for stream in streams:
        state = v8._advance_supports(
            core,
            stream,
            embeddings,
            persistent.detached_clone(),
            fusion,
            bank,
        )
        for example in stream.queries:
            with torch.inference_mode():
                _, decoded = _decode(
                    core,
                    decoder,
                    example,
                    embeddings,
                    state,
                    fusion,
                    bank,
                    include_roles=False,
                )
            pipeline = v3._decoder_pipeline(
                example.pair.learner, decoded.selected_indices[0]
            )
            scores.append(float(v1.judge_software_pipeline_attempt(example.pair, pipeline)))
    return v4._mean(scores)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("V10 output identity already exists")
    for path, expected, label in (
        (args.parent_checkpoint, v8.PARENT_CHECKPOINT_SHA256, "V6 parent"),
        (args.embedding_cache, v8.EMBEDDING_CACHE_SHA256, "embedding cache"),
        (args.v19_checkpoint, v8.V19_CHECKPOINT_SHA256, "V19 donor"),
        (args.v20_checkpoint, v8.V20_CHECKPOINT_SHA256, "V20 donor"),
        (args.v8_checkpoint, V8_CHECKPOINT_SHA256, "V8 parent"),
    ):
        if v9._sha256(path) != expected:
            raise RuntimeError(f"V10 {label} identity mismatch")
    device = torch.device(args.angler_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("V10 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()

    evidence, prepared_train, development, final = v1.prepare_corpus()
    train = v4r1._public_training_streams(prepared_train)
    support, tasks, pairs = v8._episode_maps(v8._raw_corpus())
    sealed = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    cached = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
    parent_v8 = torch.load(args.v8_checkpoint, map_location="cpu", weights_only=True)
    texts = v3._needed_texts(evidence, (train, development, final))
    if cached["foundation_digest"] != sealed["foundation_digest"] or tuple(cached["texts"]) != texts:
        raise RuntimeError("V10 Qwen cache/corpus binding changed")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    core, inherited = v7._load_parent(sealed, device)
    del inherited
    fusion = DonorCandidateFusion(**parent_v8["fusion_config"]).to(device)
    fusion.load_state_dict(parent_v8["fusion_state_dict"], strict=True)
    fusion.requires_grad_(False).eval()
    decoder = RoleAwareGraphProcedureDecoder(
        **parent_v8["decoder_config"],
        relation_count=v9.RELATION_COUNT,
        message_rounds=v9.MESSAGE_ROUNDS,
        role_width=PUBLIC_ROLE_WIDTH,
    ).to(device)
    trainable = tuple(decoder.parameters())
    core.eval()

    donor_system = v20.load_oml_checkpoint(
        args.v20_checkpoint, args.v19_checkpoint, device=device
    )
    groups = (train, development, final)
    banks = {
        "full": v8.build_donor_bank(donor_system.second_order_oml.controller, groups, support, tasks, pairs),
        "oml_removed": v8.build_donor_bank(donor_system.source_v19.controller, groups, support, tasks, pairs),
        "paired_graph_removed": v8.build_donor_bank(
            donor_system.second_order_oml.controller, groups, support, tasks, pairs,
            lesion="zero_residual"
        ),
        "cognee_unrelated": v8.build_donor_bank(
            donor_system.second_order_oml.controller, groups, support, tasks, pairs,
            unrelated=True
        ),
    }
    donor_system = None
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = []
    for epoch in range(EPOCHS):
        decoder.train()
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        losses, support_exact, query_exact = [], [], []
        for stream_index in order:
            stream = train[stream_index]
            state = v4r1._anchored_initial_state(core)
            terms = []
            for is_support, examples in ((True, stream.supports), (False, stream.queries)):
                for example in examples:
                    targets = v3._targets(example)
                    core_output, decoded = _decode(
                        core, decoder, example, embeddings, state, fusion,
                        banks["full"], teacher_actions=targets
                    )
                    terms.append(candidate_procedure_loss(decoded, targets))
                    (support_exact if is_support else query_exact).append(
                        v3._exact_visible(decoded, targets)
                    )
                    if is_support:
                        state = v8._feedback(
                            core, state, core_output, example, embeddings, fusion,
                            banks["full"], detach_state=True
                        )
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, v4.GRADIENT_LIMIT)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("V10 gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach()))
        epochs.append({
            "epoch": epoch,
            "loss": v4._mean(losses),
            "support_teacher_exact": v4._mean(support_exact),
            "composition_teacher_exact": v4._mean(query_exact),
        })

    decoder.eval()
    persistent = {
        "full": v9._restore_plastic_state(parent_v8["plastic_state"], device),
        **v9._control_persistent_states(core, train, embeddings, fusion, banks),
    }
    with _v9_evaluator_decode():
        dev_metrics, dev_attr, dev_rows = v9._evaluate(
            core, decoder, development, embeddings, fusion, banks, persistent
        )
        metrics, attribution, rows = v9._evaluate(
            core, decoder, final, embeddings, fusion, banks, persistent
        )
    dev_metrics["public_roles_removed"] = _role_removed(
        core, decoder, development, embeddings, fusion, banks["full"], persistent["full"]
    )
    metrics["public_roles_removed"] = _role_removed(
        core, decoder, final, embeddings, fusion, banks["full"], persistent["full"]
    )
    full = metrics["full"]
    supported = (
        dev_metrics["full"] >= 0.60 and full >= 0.60
        and all(
            full - metrics[name] >= 0.10
            for name in (
                "donor_removed", "public_roles_removed", "edge_reasoning_removed",
                "moving_origin_removed", "cognee_unrelated",
                "procedure_slots_removed", "reset_state",
            )
        )
        and attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "TRIFECTA_ROLE_GRAPH_SUPPORTED" if supported else "NOT_SUPPORTED",
        "first_result_accepted_without_tuning": True,
        "training": epochs,
        "development_metrics": dev_metrics,
        "development_target_evidence_attribution": dev_attr,
        "development": dev_rows,
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "evaluation": rows,
        "source": {
            "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "parent_checkpoint_sha256": v8.PARENT_CHECKPOINT_SHA256,
            "embedding_cache_sha256": v8.EMBEDDING_CACHE_SHA256,
            "v19_checkpoint_sha256": v8.V19_CHECKPOINT_SHA256,
            "v20_checkpoint_sha256": v8.V20_CHECKPOINT_SHA256,
        },
        "trainable_parameters": sum(value.numel() for value in trainable),
        "runtime": {
            "device": str(device), "name": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "deterministic_solver_used": False,
        "hidden_fields_used_for_training": False,
        "runner_sha256": v9._sha256(Path(__file__)),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "identity": IDENTITY,
        "foundation_digest": sealed["foundation_digest"],
        "core_config": sealed["core_config"],
        "core_state_dict": sealed["core_state_dict"],
        "fusion_config": parent_v8["fusion_config"],
        "fusion_state_dict": parent_v8["fusion_state_dict"],
        "decoder_config": parent_v8["decoder_config"],
        "graph_decoder_config": {
            "relation_count": v9.RELATION_COUNT,
            "message_rounds": v9.MESSAGE_ROUNDS,
            "role_width": PUBLIC_ROLE_WIDTH,
        },
        "decoder_state_dict": {name: value.detach().cpu() for name, value in decoder.state_dict().items()},
        "plastic_state": parent_v8["plastic_state"],
        "source": report["source"],
    }, checkpoint_path)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({
        "identity": IDENTITY, "classification": report["classification"],
        "training": epochs, "development_metrics": dev_metrics,
        "metrics": metrics, "target_evidence_attribution": attribution,
        "runtime": report["runtime"], "trainable_parameters": report["trainable_parameters"],
    }, indent=2))


if __name__ == "__main__":
    main()
