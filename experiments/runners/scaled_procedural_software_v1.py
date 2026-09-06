"""Train the scaled Angler core from public software-procedure experience."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from angler.memory import MovingOriginIndex, RecallBatch, SituatedRecall
from angler.reasoning import (
    PlasticProcedureState,
    ScalableProceduralCore,
    SituatedFeatureSpec,
    encode_situated_features,
    plastic_state_digest,
    procedural_core_config,
)
from angler.runtime import (
    encode_detached_segments,
    foundation_tensor_digest,
    freeze_knowledge_model,
    generate_with_procedural_prefix,
    procedural_prefix_language_loss,
)
from experiments.evaluators.software_pipeline_reconstruction_suite import (
    PublicComponentContract,
    PublicSoftwarePipelineTask,
    commit_software_pipeline,
    judge_software_pipeline_attempt,
    make_software_pipeline_stream,
    software_pipeline_mechanism_partition,
)


IDENTITY = "angler.scaled-procedural-software-reconstruction.v1"
SEED = 20260844
TRAIN_MECHANISMS = 64
DEVELOPMENT_MECHANISMS = 16
FINAL_MECHANISMS = 16
SUPPORTS_PER_MECHANISM = 4
QUERIES_PER_MECHANISM = 2
SLOW_UPDATES = TRAIN_MECHANISMS * SUPPORTS_PER_MECHANISM
META_EPISODES = 32
QWEN_LOSS_INTERVAL = 4
LEARNING_RATE = 2.0e-4
LANGUAGE_LOSS_WEIGHT = 0.05
GRADIENT_LIMIT = 1.0
SELECTED_EVIDENCE = 4
MAX_NEW_TOKENS = 16
SPEC = SituatedFeatureSpec()
_LABELS = tuple(chr(ord("A") + index) for index in range(26))
_TOKEN = re.compile(r"\b(?:STOP|[A-Z])\b")


@dataclass(frozen=True, slots=True)
class PreparedExample:
    query_text: str
    target_text: str | None
    action_texts: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    relevant: tuple[bool, ...]
    temporal: torch.Tensor
    unrelated_refs: tuple[str, ...]
    unrelated_temporal: torch.Tensor
    pair: Any | None = None


@dataclass(frozen=True, slots=True)
class PreparedStream:
    supports: tuple[PreparedExample, ...]
    queries: tuple[PreparedExample, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    return parser.parse_args()


def _local_label(index: int) -> str:
    if index < 0 or index >= len(_LABELS):
        raise ValueError("task exposes too many local action labels")
    return _LABELS[index]


def _component_order(task: PublicSoftwarePipelineTask) -> tuple[PublicComponentContract, ...]:
    ordered = []
    for action in task.grounded_candidates:
        matches = tuple(component for component in task.components if component.schema == action.schema)
        if len(matches) != 1:
            raise ValueError("every candidate must bind exactly one public component")
        ordered.append(matches[0])
    return tuple(ordered)


def _atom_map(task: PublicSoftwarePipelineTask) -> dict[str, str]:
    atoms = {
        argument
        for record in (*task.origin.records, *task.required_output.required, *task.required_output.forbidden)
        for argument in record.arguments
    }
    for component in task.components:
        atoms.update(component.state_reads)
        atoms.update(component.state_writes)
    return {value: f"S{index}" for index, value in enumerate(sorted(atoms))}


def _type_map(task: PublicSoftwarePipelineTask) -> dict[str, str]:
    values = {
        value
        for component in task.components
        for value in (component.input_type, component.output_type, component.error_type)
    }
    return {value: f"T{index}" for index, value in enumerate(sorted(values))}


def _graph_text(component: PublicComponentContract) -> str:
    edges = tuple(
        (record.arguments[-2], record.arguments[-1])
        for record in component.incidence
        if record.predicate.endswith(".relates") and len(record.arguments) >= 3
    )
    nodes = sorted({value for edge in edges for value in edge})
    names = {value: str(index) for index, value in enumerate(nodes)}
    return ",".join(sorted(f"{names[left]}>{names[right]}" for left, right in edges))


def serialize_public_task(task: PublicSoftwarePipelineTask) -> str:
    """Expose structure with package-local labels, never opaque identities."""

    atoms = _atom_map(task)
    types = _type_map(task)
    origin = ",".join(sorted(atoms[item.arguments[0]] for item in task.origin.records))
    required = ",".join(sorted(atoms[item.arguments[0]] for item in task.required_output.required))
    forbidden = ",".join(sorted(atoms[item.arguments[0]] for item in task.required_output.forbidden)) or "none"
    rows = [
        f"origin={origin}; goal={required}; forbidden={forbidden}; budget={task.max_steps}",
        "components:",
    ]
    rows.extend(serialize_action_candidates(task))
    return "\n".join(rows)


def serialize_action_candidates(task: PublicSoftwarePipelineTask) -> tuple[str, ...]:
    atoms = _atom_map(task)
    types = _type_map(task)
    rows = []
    for index, component in enumerate(_component_order(task)):
        reads = ",".join(atoms[value] for value in component.state_reads)
        writes = ",".join(atoms[value] for value in component.state_writes)
        rows.append(
            f"{_local_label(index)} in={types[component.input_type]} "
            f"out={types[component.output_type]} reads={reads} writes={writes} "
            f"graph={_graph_text(component)}"
        )
    return tuple(rows)


def visible_procedure(task: PublicSoftwarePipelineTask) -> str:
    if len(task.observations) != 1 or not task.observations[0].transitions:
        raise ValueError("support task must expose exactly one non-empty public observation")
    actions = task.grounded_candidates
    labels = []
    for transition in task.observations[0].transitions:
        labels.append(_local_label(actions.index(transition.action)))
    labels.append("STOP")
    return " ".join(labels)


def support_evidence(task: PublicSoftwarePipelineTask) -> str:
    masked = replace(task, observations=())
    return f"{serialize_public_task(masked)}\nsuccessful procedure={visible_procedure(task)}"


def build_prompt(query_text: str, evidence: Sequence[str] = ()) -> str:
    rows = [
        "Reconstruct the software pipeline from public structure and experience.",
        "Answer only a space-separated sequence of component labels, optionally ending in STOP.",
        query_text,
    ]
    if evidence:
        rows.append("public experience:")
        rows.extend(f"experience {index + 1}:\n{text}" for index, text in enumerate(evidence))
    return "\n".join(rows)


def parse_response(task: PublicSoftwarePipelineTask, response: str):
    allowed = {_local_label(index): action for index, action in enumerate(task.grounded_candidates)}
    actions = []
    stopped = False
    for token in _TOKEN.findall(response.upper()):
        if token == "STOP":
            stopped = True
            break
        if token not in allowed:
            actions = []
            stopped = True
            break
        actions.append(allowed[token])
        if len(actions) >= task.max_steps:
            break
    if len(actions) < task.max_steps:
        stopped = True
    return commit_software_pipeline(task, actions, stopped=stopped)


def _event_ref(text: str) -> str:
    return "support-" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _temporal_rows(
    origin: MovingOriginIndex,
    refs: Sequence[str],
    *,
    now: int,
) -> torch.Tensor:
    items = []
    for ref in refs:
        anchor = origin.anchor(ref)
        items.append(
            SituatedRecall(
                artifact_ref=ref,
                text="synthetic software experience",
                source_ref=ref,
                context=(),
                visibility="LEARNER_VISIBLE",
                acquired_ordinal=anchor.ordinal,
                age=now - anchor.ordinal,
                landmark_relations=(),
                world_valid_from=None,
                world_valid_until=None,
                world_valid_at_query=None,
                backend_score=None,
                backend_ref=None,
            )
        )
    values, mask = encode_situated_features(
        (RecallBatch(tuple(items)),),
        now=(now,),
        spec=SPEC,
    )
    if not bool(mask.all().item()):
        raise RuntimeError("prepared evidence unexpectedly contains padding")
    return values[0].cpu()


def prepare_corpus() -> tuple[
    dict[str, str],
    tuple[PreparedStream, ...],
    tuple[PreparedStream, ...],
    tuple[PreparedStream, ...],
]:
    origin = MovingOriginIndex()
    texts: dict[str, str] = {}
    history: list[str] = []
    rng = random.Random(SEED)

    def partition(name: str, count: int, offset: int) -> tuple[PreparedStream, ...]:
        commitments = software_pipeline_mechanism_partition(name)
        if len(commitments) < count:
            raise RuntimeError("declared partition is smaller than the frozen profile")
        prepared = []
        for index, commitment in enumerate(commitments[:count]):
            stream = make_software_pipeline_stream(
                SEED + offset + index * 101,
                surface_seed=SEED + 10_000_000 + offset + index * 103,
                supports_per_motif=2,
                queries=QUERIES_PER_MECHANISM,
                maximum_steps=4,
                mechanism_commitment=commitment,
                mechanism_partition=name,
            )
            stale = tuple(history[-SUPPORTS_PER_MECHANISM:])
            unrelated = tuple(history[-(2 * SUPPORTS_PER_MECHANISM) : -SUPPORTS_PER_MECHANISM])
            current = []
            support_texts = []
            for pair in stream.supports:
                text = support_evidence(pair.learner)
                ref = _event_ref(text)
                if ref in texts:
                    raise RuntimeError("support projection identity repeated")
                texts[ref] = text
                origin.append(ref, "projection:" + ref)
                history.append(ref)
                current.append(ref)
                support_texts.append(text)
            now = origin.now
            support_examples = []
            for support_index, pair in enumerate(stream.supports):
                refs = tuple(
                    ref for local_index, ref in enumerate(current) if local_index != support_index
                ) + stale
                order = list(range(len(refs)))
                rng.shuffle(order)
                refs = tuple(refs[item] for item in order)
                relevant = tuple(ref in current for ref in refs)
                unrelated_refs = unrelated or stale or tuple(current)
                support_examples.append(
                    PreparedExample(
                        query_text=serialize_public_task(replace(pair.learner, observations=())),
                        target_text=visible_procedure(pair.learner),
                        action_texts=serialize_action_candidates(pair.learner),
                        candidate_refs=refs,
                        relevant=relevant,
                        temporal=_temporal_rows(origin, refs, now=now),
                        unrelated_refs=unrelated_refs,
                        unrelated_temporal=_temporal_rows(origin, unrelated_refs, now=now),
                    )
                )
            query_examples = []
            for pair in stream.queries:
                refs = tuple(current) + stale
                order = list(range(len(refs)))
                rng.shuffle(order)
                refs = tuple(refs[item] for item in order)
                relevant = tuple(ref in current for ref in refs)
                unrelated_refs = unrelated or stale or tuple(current)
                query_examples.append(
                    PreparedExample(
                        query_text=serialize_public_task(pair.learner),
                        target_text=None,
                        action_texts=serialize_action_candidates(pair.learner),
                        candidate_refs=refs,
                        relevant=relevant,
                        temporal=_temporal_rows(origin, refs, now=now),
                        unrelated_refs=unrelated_refs,
                        unrelated_temporal=_temporal_rows(origin, unrelated_refs, now=now),
                        pair=pair,
                    )
                )
            prepared.append(PreparedStream(tuple(support_examples), tuple(query_examples)))
        return tuple(prepared)

    train = partition("train", TRAIN_MECHANISMS, 1_000_000)
    development = partition("development", DEVELOPMENT_MECHANISMS, 2_000_000)
    final = partition("final", FINAL_MECHANISMS, 3_000_000)
    return texts, train, development, final


def _all_texts(
    evidence: dict[str, str],
    groups: Sequence[Sequence[PreparedStream]],
) -> tuple[str, ...]:
    values = list(evidence.values())
    for partition in groups:
        for stream in partition:
            for example in (*stream.supports, *stream.queries):
                values.append(example.query_text)
                if example.target_text is not None:
                    values.append(example.target_text)
    return tuple(dict.fromkeys(values))


def _inputs(
    example: PreparedExample,
    embeddings: dict[str, torch.Tensor],
    *,
    refs: Sequence[str] | None = None,
    temporal: torch.Tensor | None = None,
    coordinates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = example.candidate_refs if refs is None else tuple(refs)
    temporal_value = example.temporal if temporal is None else temporal
    query = embeddings[example.query_text].float().unsqueeze(0).cuda()
    candidates = torch.stack([embeddings[ref].float() for ref in chosen]).unsqueeze(0).cuda()
    coordinates_value = temporal_value.float().unsqueeze(0).cuda()
    if not coordinates:
        coordinates_value = torch.zeros_like(coordinates_value)
    mask = torch.ones((1, len(chosen)), device="cuda", dtype=torch.bool)
    return query, candidates, coordinates_value, mask


def _loss_for_example(
    core: ScalableProceduralCore,
    model,
    tokenizer,
    example: PreparedExample,
    evidence: dict[str, str],
    embeddings: dict[str, torch.Tensor],
    state: PlasticProcedureState,
    *,
    include_language: bool,
) -> tuple[torch.Tensor, dict[str, float], Any]:
    if example.target_text is None:
        raise ValueError("training example is missing its visible target")
    output = core(*_inputs(example, embeddings), plastic_state=state)
    target_embedding = embeddings[example.target_text].float().unsqueeze(0).cuda()
    latent = 1.0 - F.cosine_similarity(output.qwen_prefix.mean(dim=1), target_embedding).mean()
    relevant = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
    relevance_mass = output.candidate_weights[0, relevant].sum().clamp_min(1.0e-8)
    retrieval = -torch.log(relevance_mass)
    language = latent.new_zeros(())
    if include_language:
        # Evidence already enters the procedural core as detached Qwen
        # representations.  Repeating it in the language prompt would both
        # bypass the prefix bottleneck and retain redundant 4B activations.
        prompt = build_prompt(example.query_text)
        language = procedural_prefix_language_loss(
            model,
            tokenizer,
            prompt,
            example.target_text,
            output.qwen_prefix,
        ).loss
    total = latent + retrieval + LANGUAGE_LOSS_WEIGHT * language
    return total, {
        "total": float(total.detach()),
        "latent": float(latent.detach()),
        "retrieval": float(retrieval.detach()),
        "language": float(language.detach()),
        "relevance_mass": float(relevance_mass.detach()),
    }, output


def _step(
    core: ScalableProceduralCore,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_LIMIT)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("scaled-core gradient became non-finite")
    optimizer.step()
    return float(norm.detach())


def _generate_plain(model, tokenizer, prompt: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    suffix = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(suffix, skip_special_tokens=True).strip()


def _judge(pair, response: str) -> float:
    pipeline = parse_response(pair.learner, response)
    return float(judge_software_pipeline_attempt(pair, pipeline))


def _selected_refs(example: PreparedExample, weights: torch.Tensor) -> tuple[str, ...]:
    count = min(SELECTED_EVIDENCE, len(example.candidate_refs))
    indices = torch.topk(weights.detach().cpu(), count).indices.tolist()
    return tuple(example.candidate_refs[index] for index in indices)


def _evaluate_query(
    core: ScalableProceduralCore,
    model,
    tokenizer,
    example: PreparedExample,
    evidence: dict[str, str],
    embeddings: dict[str, torch.Tensor],
    live_state: PlasticProcedureState,
) -> dict[str, Any]:
    if example.pair is None:
        raise ValueError("query example is missing its evaluator pairing")
    with torch.inference_mode():
        full = core(*_inputs(example, embeddings), plastic_state=live_state)
        reset = core(
            *_inputs(example, embeddings),
            plastic_state=core.initial_plastic_state(),
        )
        no_coordinates = core(
            *_inputs(example, embeddings, coordinates=False),
            plastic_state=live_state,
        )
        unrelated = core(
            *_inputs(
                example,
                embeddings,
                refs=example.unrelated_refs,
                temporal=example.unrelated_temporal,
            ),
            plastic_state=live_state,
        )
    selected = _selected_refs(example, full.candidate_weights[0])
    reset_selected = _selected_refs(example, reset.candidate_weights[0])
    coordinate_selected = _selected_refs(example, no_coordinates.candidate_weights[0])
    unrelated_selected = _selected_refs(example, unrelated.candidate_weights[0])
    prompts = {
        "qwen_alone": build_prompt(example.query_text),
        "fair_retrieval": build_prompt(
            example.query_text,
            [evidence[ref] for ref in example.candidate_refs],
        ),
        "prefix_removed": build_prompt(example.query_text, [evidence[ref] for ref in selected]),
        "full": build_prompt(example.query_text, [evidence[ref] for ref in selected]),
        "reset_state": build_prompt(example.query_text, [evidence[ref] for ref in reset_selected]),
        "coordinates_removed": build_prompt(
            example.query_text,
            [evidence[ref] for ref in coordinate_selected],
        ),
        "unrelated_evidence": build_prompt(
            example.query_text,
            [evidence[ref] for ref in unrelated_selected],
        ),
    }
    responses = {
        "qwen_alone": _generate_plain(model, tokenizer, prompts["qwen_alone"]),
        "fair_retrieval": _generate_plain(model, tokenizer, prompts["fair_retrieval"]),
        "prefix_removed": _generate_plain(model, tokenizer, prompts["prefix_removed"]),
        "full": generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["full"],
            full.qwen_prefix,
            max_new_tokens=MAX_NEW_TOKENS,
        ).response,
        "reset_state": generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["reset_state"],
            reset.qwen_prefix,
            max_new_tokens=MAX_NEW_TOKENS,
        ).response,
        "coordinates_removed": generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["coordinates_removed"],
            no_coordinates.qwen_prefix,
            max_new_tokens=MAX_NEW_TOKENS,
        ).response,
        "unrelated_evidence": generate_with_procedural_prefix(
            model,
            tokenizer,
            prompts["unrelated_evidence"],
            unrelated.qwen_prefix,
            max_new_tokens=MAX_NEW_TOKENS,
        ).response,
    }
    scores = {name: _judge(example.pair, response) for name, response in responses.items()}
    relevance = torch.tensor(example.relevant, device=full.candidate_weights.device, dtype=torch.bool)
    return {
        "scores": scores,
        "responses": responses,
        "selected_refs": list(selected),
        "target_evidence_mass": float(full.candidate_weights[0, relevance].sum()),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if output_path.exists() or checkpoint_path.exists():
        raise RuntimeError("scaled procedural V1 output identity already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("scaled procedural V1 requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    evidence, train, development, final = prepare_corpus()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    foundation_before = foundation_tensor_digest(model)
    texts = _all_texts(evidence, (train, development, final))
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

    config = procedural_core_config(
        "workstation",
        content_width=encoded.shape[1],
        temporal_width=SPEC.width,
    )
    core = ScalableProceduralCore(config).cuda()
    optimizer = torch.optim.AdamW(core.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    slow_rows = []
    core.train()
    update_index = 0
    for stream in train:
        for example in stream.supports:
            loss, row, _ = _loss_for_example(
                core,
                model,
                tokenizer,
                example,
                evidence,
                embeddings,
                core.initial_plastic_state(),
                include_language=update_index % QWEN_LOSS_INTERVAL == 0,
            )
            row["gradient_norm"] = _step(core, optimizer, loss)
            row["update"] = update_index
            slow_rows.append(row)
            update_index += 1

    meta_rows = []
    for episode_index, stream in enumerate(train[:META_EPISODES]):
        state = core.initial_plastic_state()
        for support in stream.supports[:3]:
            output = core(*_inputs(support, embeddings), plastic_state=state)
            state = core.apply_feedback(
                state,
                output,
                torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                detach_state=False,
            )
        held = stream.supports[3]
        loss, row, _ = _loss_for_example(
            core,
            model,
            tokenizer,
            held,
            evidence,
            embeddings,
            state,
            include_language=episode_index % QWEN_LOSS_INTERVAL == 0,
        )
        row["gradient_norm"] = _step(core, optimizer, loss)
        row["episode"] = episode_index
        meta_rows.append(row)

    core.eval()
    development_rows = []
    with torch.inference_mode():
        for stream in development:
            for example in stream.supports:
                output = core(
                    *_inputs(example, embeddings),
                    plastic_state=core.initial_plastic_state(),
                )
                target = embeddings[example.target_text].float().unsqueeze(0).cuda()
                latent = 1.0 - F.cosine_similarity(output.qwen_prefix.mean(1), target).mean()
                relevance = torch.tensor(example.relevant, device="cuda", dtype=torch.bool)
                mass = output.candidate_weights[0, relevance].sum()
                development_rows.append(
                    {"latent": float(latent), "relevance_mass": float(mass)}
                )

    persistent = core.initial_plastic_state()
    with torch.inference_mode():
        for stream in train:
            for example in stream.supports:
                output = core(*_inputs(example, embeddings), plastic_state=persistent)
                persistent = core.apply_feedback(
                    persistent,
                    output,
                    torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                )

    evaluation_rows = []
    for stream_index, stream in enumerate(final):
        live = persistent.detached_clone()
        with torch.inference_mode():
            for support in stream.supports:
                output = core(*_inputs(support, embeddings), plastic_state=live)
                live = core.apply_feedback(
                    live,
                    output,
                    torch.ones(1, device="cuda", dtype=output.query_state.dtype),
                )
        for query_index, example in enumerate(stream.queries):
            row = _evaluate_query(
                core,
                model,
                tokenizer,
                example,
                evidence,
                embeddings,
                live,
            )
            row["stream_index"] = stream_index
            row["query_index"] = query_index
            evaluation_rows.append(row)

    foundation_after = foundation_tensor_digest(model)
    if foundation_before != foundation_after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen Qwen changed during scaled-core training")
    arms = tuple(evaluation_rows[0]["scores"])
    metrics = {
        arm: _mean([float(row["scores"][arm]) for row in evaluation_rows])
        for arm in arms
    }
    attribution = _mean([float(row["target_evidence_mass"]) for row in evaluation_rows])
    initial_loss = _mean([row["total"] for row in slow_rows[:32]])
    terminal_loss = _mean([row["total"] for row in slow_rows[-32:]])
    loss_reduction = 0.0 if initial_loss == 0.0 else (initial_loss - terminal_loss) / initial_loss
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
        "seed": SEED,
        "profile": {
            "train_mechanisms": TRAIN_MECHANISMS,
            "development_mechanisms": DEVELOPMENT_MECHANISMS,
            "final_mechanisms": FINAL_MECHANISMS,
            "slow_updates": SLOW_UPDATES,
            "meta_episodes": META_EPISODES,
            "qwen_loss_interval": QWEN_LOSS_INTERVAL,
            "tier": config.tier,
            "parameters": core.parameter_count,
        },
        "metrics": metrics,
        "target_evidence_attribution": attribution,
        "initial_loss": initial_loss,
        "terminal_loss": terminal_loss,
        "loss_reduction": loss_reduction,
        "development_latent_loss": _mean([row["latent"] for row in development_rows]),
        "development_relevance_mass": _mean([row["relevance_mass"] for row in development_rows]),
        "plastic_state_digest": plastic_state_digest(persistent),
        "foundation_digest": foundation_before,
        "slow_training": slow_rows,
        "meta_training": meta_rows,
        "evaluation": evaluation_rows,
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "source_sha256": {
            "runner": _sha256(Path(__file__)),
            "procedural_core": _sha256(
                Path(__file__).resolve().parents[2]
                / "src/angler/reasoning/scalable_procedural_core.py"
            ),
            "qwen_bridge": _sha256(
                Path(__file__).resolve().parents[2]
                / "src/angler/runtime/procedural_qwen.py"
            ),
        },
        "nonclaims": [
            "The task world is synthetic software reconstruction.",
            "A pass is not unrestricted software engineering, conversation readiness, AGI, or deployment approval.",
        ],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": IDENTITY,
            "config": asdict(config),
            "state_dict": {name: value.detach().cpu() for name, value in core.state_dict().items()},
            "plastic_state": {
                "keys": persistent.keys.detach().cpu(),
                "values": persistent.values.detach().cpu(),
                "strengths": persistent.strengths.detach().cpu(),
                "step": persistent.step,
            },
            "foundation_digest": foundation_before,
        },
        checkpoint_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("identity", "classification", "metrics", "loss_reduction", "target_evidence_attribution", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
