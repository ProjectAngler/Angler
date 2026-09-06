"""V9: learned donor-prior and public-edge set-to-procedure decoding.

The exact V8 fusion and plastic state remain frozen.  Deterministic plumbing
only exposes declared pairwise relations and the frozen donor score; a learned
pointer decoder must infer candidate priority and order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Mapping, Sequence

import torch

from angler.reasoning import (
    DonorCandidateFusion,
    EdgeAwareActionTraceDecoder,
    PlasticProcedureState,
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


IDENTITY = "angler.trifecta-edge-procedure.v9"
SEED = 20260853
V8_CHECKPOINT_SHA256 = (
    "210b439ea1295c7545e5283e30e8243d1c47cec8e2f73607681ff3100c8a562c"
)
EPOCHS = 8
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
RELATION_COUNT = 8
MESSAGE_ROUNDS = 2
NEW_DECODER_PREFIXES = (
    "edge_encoder.",
    "edge_message.",
    "edge_gate.",
    "edge_update.",
    "selected_edge_projection.",
    "donor_prior_raw_scale",
)
GATING_ARMS = (
    "donor_removed",
    "donor_prior_removed",
    "edge_reasoning_removed",
    "coverage_removed",
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
    parser.add_argument("--v8-checkpoint", required=True)
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


def public_edge_features(
    query_text: str,
    action_texts: Sequence[str],
    *,
    maximum_actions: int = v3.MAXIMUM_ACTIONS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode public candidate relations without interpreting their utility."""

    graph = parse_public_relations(query_text, action_texts)
    if len(graph.components) > maximum_actions:
        raise ValueError("public candidate count exceeds decoder capacity")
    edges = torch.zeros(
        (1, maximum_actions, maximum_actions, RELATION_COUNT),
        device=device,
        dtype=dtype,
    )
    for left_index, left in enumerate(graph.components):
        left_reads = frozenset(left.reads)
        left_writes = frozenset(left.writes)
        left_state = left_reads | left_writes
        for right_index, right in enumerate(graph.components):
            right_reads = frozenset(right.reads)
            right_writes = frozenset(right.writes)
            right_state = right_reads | right_writes
            edges[0, left_index, right_index] = torch.tensor(
                (
                    left_index == right_index,
                    left.output_type == right.input_type,
                    left.input_type == right.output_type,
                    bool(left_writes & right_reads),
                    bool(left_reads & right_writes),
                    left.input_type == right.input_type,
                    left.output_type == right.output_type,
                    bool(left_state & right_state),
                ),
                device=device,
                dtype=dtype,
            )
    return edges


def migrate_v8_decoder_state(
    decoder: EdgeAwareActionTraceDecoder,
    v8_state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load inherited V8 decoder bytes and permit only declared fresh keys."""

    missing, unexpected = decoder.load_state_dict(v8_state, strict=False)
    if unexpected or any(
        not any(name.startswith(prefix) for prefix in NEW_DECODER_PREFIXES)
        for name in missing
    ):
        raise RuntimeError("V9 decoder migration changed an inherited tensor")
    expected = {
        name
        for name in decoder.state_dict()
        if any(name.startswith(prefix) for prefix in NEW_DECODER_PREFIXES)
    }
    if set(missing) != expected:
        raise RuntimeError("V9 decoder migration omitted a declared fresh tensor")
    return tuple(sorted(missing))


def _restore_plastic_state(record, device: torch.device) -> PlasticProcedureState:
    if not isinstance(record, dict) or set(record) != {
        "keys",
        "values",
        "strengths",
        "step",
    }:
        raise RuntimeError("V9 parent plastic-state fields changed")
    return PlasticProcedureState(
        keys=record["keys"].to(device),
        values=record["values"].to(device),
        strengths=record["strengths"].to(device),
        step=int(record["step"]),
    )


def _decoder_inputs(example, embeddings, fusion, bank, *, include_donor=True):
    action_features, action_mask = v8._action_features(
        example,
        embeddings,
        fusion,
        bank,
        include_donor=include_donor,
    )
    try:
        donor_rows = bank[id(example)]
    except KeyError as error:
        raise RuntimeError("V9 donor bank does not cover an example") from error
    donor_scores = torch.zeros(
        (1, v3.MAXIMUM_ACTIONS),
        device=action_features.device,
        dtype=action_features.dtype,
    )
    if include_donor:
        donor_scores[0, : donor_rows.shape[0]] = donor_rows[:, -1].to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
    edges = public_edge_features(
        example.query_text,
        example.action_texts,
        device=action_features.device,
        dtype=action_features.dtype,
    )
    return action_features.detach(), action_mask, edges, donor_scores


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
        action_features, action_mask, edges, donor_scores = _decoder_inputs(
            example,
            embeddings,
            fusion,
            bank,
            include_donor=include_donor,
        )
    decoded = decoder(
        slots.detach(),
        action_features,
        action_mask,
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


def _control_persistent_states(core, train, embeddings, fusion, banks):
    modes = {
        "donor_removed": (banks["full"], False, False),
        "oml_removed": (banks["oml_removed"], True, False),
        "paired_graph_removed": (banks["paired_graph_removed"], True, False),
        "cognee_unrelated": (banks["cognee_unrelated"], True, True),
    }
    states = {name: v4r1._anchored_initial_state(core) for name in modes}
    for stream in train:
        for name, (bank, include_donor, unrelated) in modes.items():
            states[name] = v8._advance_supports(
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
            "full": v8._advance_supports(
                core, stream, embeddings, persistent["full"].detached_clone(),
                fusion, banks["full"]
            ),
            "donor_removed": v8._advance_supports(
                core, stream, embeddings, persistent["donor_removed"].detached_clone(),
                fusion, banks["full"], include_donor=False
            ),
            "oml_removed": v8._advance_supports(
                core, stream, embeddings, persistent["oml_removed"].detached_clone(),
                fusion, banks["oml_removed"]
            ),
            "paired_graph_removed": v8._advance_supports(
                core, stream, embeddings,
                persistent["paired_graph_removed"].detached_clone(), fusion,
                banks["paired_graph_removed"]
            ),
            "cognee_unrelated": v8._advance_supports(
                core, stream, embeddings,
                persistent["cognee_unrelated"].detached_clone(), fusion,
                banks["cognee_unrelated"], core_unrelated=True
            ),
            "local_only": v8._advance_supports(
                core, stream, embeddings, local.detached_clone(), fusion,
                banks["full"]
            ),
        }
        for query_index, example in enumerate(stream.queries):
            with torch.inference_mode():
                common = (core, decoder, example, embeddings, states["full"], fusion, banks["full"])
                arms = {
                    "full": _decode(*common),
                    "donor_removed": _decode(
                        core, decoder, example, embeddings, states["donor_removed"],
                        fusion, banks["full"], include_donor=False
                    ),
                    "donor_prior_removed": _decode(*common, donor_prior=False),
                    "edge_reasoning_removed": _decode(*common, edge_reasoning=False),
                    "coverage_removed": _decode(*common, coverage=False),
                    "oml_removed": _decode(
                        core, decoder, example, embeddings, states["oml_removed"],
                        fusion, banks["oml_removed"]
                    ),
                    "paired_graph_removed": _decode(
                        core, decoder, example, embeddings,
                        states["paired_graph_removed"], fusion,
                        banks["paired_graph_removed"]
                    ),
                    "moving_origin_removed": _decode(*common, coordinates=False),
                    "cognee_unrelated": _decode(
                        core, decoder, example, embeddings,
                        states["cognee_unrelated"], fusion,
                        banks["cognee_unrelated"], core_unrelated=True
                    ),
                    "procedure_slots_removed": _decode(*common, remove_slots=True),
                    "correspondence_removed": _decode(*common, correspondence=False),
                    "reset_state": _decode(
                        core, decoder, example, embeddings, local, fusion, banks["full"]
                    ),
                    "local_only": _decode(
                        core, decoder, example, embeddings, states["local_only"],
                        fusion, banks["full"]
                    ),
                    "persistent_only": _decode(
                        core, decoder, example, embeddings, persistent["full"],
                        fusion, banks["full"]
                    ),
                }
            scores = {}
            sequences = {}
            for name, (_, decoded) in arms.items():
                pipeline = v3._decoder_pipeline(
                    example.pair.learner, decoded.selected_indices[0]
                )
                scores[name] = float(v1.judge_software_pipeline_attempt(example.pair, pipeline))
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
    metrics = {name: v4._mean([row["scores"][name] for row in rows]) for name in names}
    attribution = v4._mean([row["target_evidence_mass"] for row in rows])
    return metrics, attribution, rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("V9 output identity already exists")
    identities = (
        (args.parent_checkpoint, v8.PARENT_CHECKPOINT_SHA256, "V6 parent"),
        (args.embedding_cache, v8.EMBEDDING_CACHE_SHA256, "embedding cache"),
        (args.v19_checkpoint, v8.V19_CHECKPOINT_SHA256, "V19 donor"),
        (args.v20_checkpoint, v8.V20_CHECKPOINT_SHA256, "V20 donor"),
        (args.v8_checkpoint, V8_CHECKPOINT_SHA256, "V8 parent"),
    )
    for path, expected, label in identities:
        if _sha256(path) != expected:
            raise RuntimeError(f"V9 {label} identity mismatch")
    device = torch.device(args.angler_device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise ValueError("V9 requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()

    evidence, prepared_train, development, final = v1.prepare_corpus()
    train = v4r1._public_training_streams(prepared_train)
    raw_groups = v8._raw_corpus()
    support_by_key, task_by_query_text, pair_by_ref = v8._episode_maps(raw_groups)
    sealed = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    cached = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
    texts = v3._needed_texts(evidence, (train, development, final))
    if cached["foundation_digest"] != sealed["foundation_digest"] or tuple(cached["texts"]) != texts:
        raise RuntimeError("V9 frozen Qwen cache does not cover the corpus")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    parent_v8 = torch.load(args.v8_checkpoint, map_location="cpu", weights_only=True)
    if parent_v8.get("identity") != v8.IDENTITY:
        raise RuntimeError("V9 parent payload identity changed")
    core, inherited_decoder = v7._load_parent(sealed, device)
    del inherited_decoder
    fusion = DonorCandidateFusion(**parent_v8["fusion_config"]).to(device)
    fusion.load_state_dict(parent_v8["fusion_state_dict"], strict=True)
    fusion.requires_grad_(False)
    decoder = EdgeAwareActionTraceDecoder(
        **parent_v8["decoder_config"],
        relation_count=RELATION_COUNT,
        message_rounds=MESSAGE_ROUNDS,
    ).to(device)
    fresh_decoder_keys = migrate_v8_decoder_state(
        decoder, parent_v8["decoder_state_dict"]
    )
    decoder.requires_grad_(False)
    for name, parameter in decoder.named_parameters():
        if any(name.startswith(prefix) for prefix in NEW_DECODER_PREFIXES):
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("V9 declared no trainable decoder tensors")
    core.eval()
    fusion.eval()

    donor_system = v20.load_oml_checkpoint(
        args.v20_checkpoint, args.v19_checkpoint, device=device
    )
    prepared_groups = (train, development, final)
    banks = {
        "full": v8.build_donor_bank(
            donor_system.second_order_oml.controller, prepared_groups,
            support_by_key, task_by_query_text, pair_by_ref
        ),
        "oml_removed": v8.build_donor_bank(
            donor_system.source_v19.controller, prepared_groups,
            support_by_key, task_by_query_text, pair_by_ref
        ),
        "paired_graph_removed": v8.build_donor_bank(
            donor_system.second_order_oml.controller, prepared_groups,
            support_by_key, task_by_query_text, pair_by_ref, lesion="zero_residual"
        ),
        "cognee_unrelated": v8.build_donor_bank(
            donor_system.second_order_oml.controller, prepared_groups,
            support_by_key, task_by_query_text, pair_by_ref, unrelated=True
        ),
    }
    donor_system = None
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    epoch_rows = []
    for epoch in range(EPOCHS):
        decoder.train()
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
                    core_output, decoded = _decode(
                        core, decoder, example, embeddings, state, fusion,
                        banks["full"], teacher_actions=targets
                    )
                    episode_losses.append(candidate_procedure_loss(decoded, targets))
                    exact = v3._exact_visible(decoded, targets)
                    if is_support:
                        support_exact.append(exact)
                        state = v8._feedback(
                            core, state, core_output, example, embeddings, fusion,
                            banks["full"], detach_state=True
                        )
                    else:
                        composition_exact.append(exact)
            loss = torch.stack(episode_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, v4.GRADIENT_LIMIT)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("V9 gradient became non-finite")
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
    persistent = {
        "full": _restore_plastic_state(parent_v8["plastic_state"], device),
        **_control_persistent_states(core, train, embeddings, fusion, banks),
    }
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
        and full - final_metrics["donor_prior_removed"] >= 0.10
        and full - final_metrics["edge_reasoning_removed"] >= 0.10
        and full - final_metrics["coverage_removed"] >= 0.10
        and full - final_metrics["oml_removed"] >= 0.10
        and full - final_metrics["moving_origin_removed"] >= 0.10
        and full - final_metrics["cognee_unrelated"] >= 0.10
        and full - final_metrics["procedure_slots_removed"] >= 0.10
        and final_attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "classification": "TRIFECTA_EDGE_PROCEDURE_SUPPORTED" if supported else "NOT_SUPPORTED",
        "first_result_accepted_without_tuning": True,
        "training": epoch_rows,
        "development_metrics": development_metrics,
        "development_target_evidence_attribution": development_attribution,
        "development": development_rows,
        "metrics": final_metrics,
        "target_evidence_attribution": final_attribution,
        "evaluation": final_rows,
        "source": {
            "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "parent_checkpoint_sha256": v8.PARENT_CHECKPOINT_SHA256,
            "embedding_cache_sha256": v8.EMBEDDING_CACHE_SHA256,
            "v19_checkpoint_sha256": v8.V19_CHECKPOINT_SHA256,
            "v20_checkpoint_sha256": v8.V20_CHECKPOINT_SHA256,
            "cognee_boundary": "episode refs select canonical public support sidecars",
            "moving_origin_boundary": "situated temporal coordinates enter frozen Angler core",
        },
        "fresh_decoder_keys": list(fresh_decoder_keys),
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
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "source": report["source"],
            "foundation_digest": sealed["foundation_digest"],
            "fusion_config": parent_v8["fusion_config"],
            "fusion_state_dict": parent_v8["fusion_state_dict"],
            "decoder_config": parent_v8["decoder_config"],
            "edge_decoder_config": {
                "relation_count": RELATION_COUNT,
                "message_rounds": MESSAGE_ROUNDS,
            },
            "decoder_state_dict": {
                name: value.detach().cpu() for name, value in decoder.state_dict().items()
            },
            "plastic_state": parent_v8["plastic_state"],
        },
        checkpoint_path,
    )
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
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
