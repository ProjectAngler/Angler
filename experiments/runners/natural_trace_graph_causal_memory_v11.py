"""Frozen natural-trace graph causal-memory evaluation V11."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from angler.reasoning import (
    NaturalTraceGraphCausalMemoryCore,
    NaturalTraceGraphMemoryState,
    parse_step_trace,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import content_addressed_causal_memory_v7 as v7
from experiments.runners import dnc_allocated_factorized_causal_memory_v10 as v10
from experiments.runners import factorized_key_value_causal_memory_v9 as v9


IDENTITY = "angler.natural-trace-graph-causal-memory.v11-first-result"
SEED = 2026083111
RANK = v10.RANK
MEMORY_SLOTS = v10.MEMORY_SLOTS
TEMPORAL_WIDTH = v10.TEMPORAL_WIDTH
MAXIMUM_TRACE_STEPS = 8
GRAPH_SUCCESS_ADVANTAGE = 2
GRAPH_TARGET_ADVANTAGE = 0.10
DIRECTION_SUCCESS_ADVANTAGE = 1
DIRECTION_TARGET_ADVANTAGE = 0.05


@dataclass(frozen=True, slots=True)
class EncodedGraphMechanism:
    mechanism_ref: str
    generator_family: str
    episode_queries: torch.Tensor
    episode_candidates: torch.Tensor
    episode_temporal: torch.Tensor
    episode_outcomes: torch.Tensor
    challenge_query: torch.Tensor
    challenge_candidates: torch.Tensor
    challenge_temporal: torch.Tensor
    target_index: int
    failed_indices: tuple[int, ...]
    episode_step_features: torch.Tensor
    episode_step_mask: torch.Tensor
    challenge_step_features: torch.Tensor
    challenge_step_mask: torch.Tensor

    def to(self, device: torch.device) -> "EncodedGraphMechanism":
        floating = {
            name: getattr(self, name).to(device=device, dtype=torch.float32)
            for name in (
                "episode_queries",
                "episode_candidates",
                "episode_temporal",
                "episode_outcomes",
                "challenge_query",
                "challenge_candidates",
                "challenge_temporal",
                "episode_step_features",
                "challenge_step_features",
            )
        }
        return replace(
            self,
            **floating,
            episode_step_mask=self.episode_step_mask.to(device=device),
            challenge_step_mask=self.challenge_step_mask.to(device=device),
        )


def _encode_partition(qwen: Any, mechanisms: Sequence[Any]) -> tuple[EncodedGraphMechanism, ...]:
    """Keep V6 pooled observations exact and separately embed raw step spans."""

    base_rows = v7.v6._encode_partition(qwen, mechanisms)
    sentences: list[str] = []
    mechanism_spans: list[tuple[tuple[int, int], ...]] = []
    for mechanism in mechanisms:
        spans = []
        traces = [episode.action_trace_text for episode in mechanism.public.episodes]
        traces.extend(mechanism.public.challenge.candidate_action_trace_texts)
        for trace in traces:
            steps = parse_step_trace(trace)
            if len(steps) > MAXIMUM_TRACE_STEPS:
                raise ValueError("public trace exceeds the frozen eight-step ceiling")
            start = len(sentences)
            sentences.extend(steps)
            spans.append((start, len(steps)))
        mechanism_spans.append(tuple(spans))
    encoded = qwen.embed(sentences).to(dtype=torch.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(sentences):
        raise RuntimeError("Qwen step embeddings do not align with parsed public sentences")
    step_width = int(encoded.shape[-1])
    rows = []
    for base, spans in zip(base_rows, mechanism_spans):
        episode_features = encoded.new_zeros((6, MAXIMUM_TRACE_STEPS, step_width))
        episode_mask = torch.zeros(
            6, MAXIMUM_TRACE_STEPS, dtype=torch.bool, device=encoded.device
        )
        challenge_features = encoded.new_zeros((4, MAXIMUM_TRACE_STEPS, step_width))
        challenge_mask = torch.zeros(
            4, MAXIMUM_TRACE_STEPS, dtype=torch.bool, device=encoded.device
        )
        for index, (start, length) in enumerate(spans):
            target_features = episode_features if index < 6 else challenge_features
            target_mask = episode_mask if index < 6 else challenge_mask
            target_index = index if index < 6 else index - 6
            target_features[target_index, :length] = encoded[start : start + length]
            target_mask[target_index, :length] = True
        rows.append(
            EncodedGraphMechanism(
                **{name: getattr(base, name) for name in (
                    "mechanism_ref", "generator_family", "episode_queries",
                    "episode_candidates", "episode_temporal", "episode_outcomes",
                    "challenge_query", "challenge_candidates", "challenge_temporal",
                    "target_index", "failed_indices",
                )},
                episode_step_features=episode_features,
                episode_step_mask=episode_mask,
                challenge_step_features=challenge_features,
                challenge_step_mask=challenge_mask,
            )
        )
    return tuple(rows)


_INCLUDE_DIRECTION = True
_INCLUDE_GRAPH_ADDRESS = True


@contextmanager
def _graph_lesion_scope(*, include_direction: bool, include_graph_address: bool):
    global _INCLUDE_DIRECTION, _INCLUDE_GRAPH_ADDRESS
    previous = (_INCLUDE_DIRECTION, _INCLUDE_GRAPH_ADDRESS)
    _INCLUDE_DIRECTION = include_direction
    _INCLUDE_GRAPH_ADDRESS = include_graph_address
    try:
        yield
    finally:
        _INCLUDE_DIRECTION, _INCLUDE_GRAPH_ADDRESS = previous


def _apply_stream(
    core: NaturalTraceGraphCausalMemoryCore,
    row: EncodedGraphMechanism,
    state: NaturalTraceGraphMemoryState,
    *,
    outcome_values: Sequence[int | float] | None = None,
    use_moving_origin: bool = True,
    detach_state: bool,
    collect_outputs: bool = False,
):
    outcomes = row.episode_outcomes if outcome_values is None else torch.tensor(
        tuple(outcome_values), device=row.episode_outcomes.device, dtype=torch.float32
    )
    if outcomes.shape != (6,):
        raise ValueError("V11 stream must contain exactly six outcomes")
    outputs = []
    for index in range(6):
        query = row.episode_queries[index].unsqueeze(0)
        candidates = row.episode_candidates[index].reshape(1, 1, -1)
        temporal = row.episode_temporal[index].reshape(1, 1, -1)
        if not use_moving_origin:
            temporal = torch.zeros_like(temporal)
        event_outcome = outcomes[index].reshape(1, 1)
        mask = torch.ones((1, 1), device=query.device, dtype=torch.bool)
        output = core(
            query,
            candidates,
            temporal,
            event_outcome,
            mask,
            v7._base_scores(query, candidates),
            candidate_step_features=row.episode_step_features[index].reshape(
                1, 1, MAXIMUM_TRACE_STEPS, -1
            ),
            candidate_step_mask=row.episode_step_mask[index].reshape(
                1, 1, MAXIMUM_TRACE_STEPS
            ),
            memory_state=state,
            include_direction=_INCLUDE_DIRECTION,
            include_graph_address=_INCLUDE_GRAPH_ADDRESS,
        )
        state = core.apply_mixed_outcome_update(
            state, output, event_outcome, mask, detach_state=detach_state
        )
        if collect_outputs:
            outputs.append(output)
    return state, tuple(outputs)


def _challenge(
    core: NaturalTraceGraphCausalMemoryCore,
    row: EncodedGraphMechanism,
    state: NaturalTraceGraphMemoryState,
    *,
    use_moving_origin: bool = True,
):
    query = row.challenge_query.unsqueeze(0)
    candidates = row.challenge_candidates.unsqueeze(0)
    temporal = row.challenge_temporal.unsqueeze(0)
    if not use_moving_origin:
        temporal = torch.zeros_like(temporal)
    mask = torch.ones(candidates.shape[:2], device=query.device, dtype=torch.bool)
    outcomes = torch.zeros_like(mask, dtype=torch.float32)
    return core(
        query,
        candidates,
        temporal,
        outcomes,
        mask,
        v7._base_scores(query, candidates),
        candidate_step_features=row.challenge_step_features.unsqueeze(0),
        candidate_step_mask=row.challenge_step_mask.unsqueeze(0),
        memory_state=state,
        include_direction=_INCLUDE_DIRECTION,
        include_graph_address=_INCLUDE_GRAPH_ADDRESS,
    )


_ORIGINAL_APPLY_STREAM = v7._apply_stream
_ORIGINAL_CHALLENGE = v7._challenge


@contextmanager
def _v11_runner_patch():
    """Patch the two inherited state boundaries and restore on every exit."""

    if v7._apply_stream is not _ORIGINAL_APPLY_STREAM or v7._challenge is not _ORIGINAL_CHALLENGE:
        raise RuntimeError("inherited V7 runner boundary was already patched")
    v7._apply_stream = _apply_stream
    v7._challenge = _challenge
    try:
        yield
    finally:
        v7._apply_stream = _ORIGINAL_APPLY_STREAM
        v7._challenge = _ORIGINAL_CHALLENGE


def _assert_graph_gradients(core: NaturalTraceGraphCausalMemoryCore) -> dict[str, float]:
    required = {
        "step_projection": core.trace_graph_encoder.step_projection.weight,
        "message_update": core.trace_graph_encoder.message_update[1].weight,
        "pooling": core.trace_graph_encoder.pooling[1].weight,
        "graph_comparator": core.graph_comparator[-1].weight,
        "address": core.read_key_network[1].weight,
        "value": core.writer[-1].weight,
        "scorer": core.score_network.second.weight,
    }
    values = {}
    for name, parameter in required.items():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError(f"V11 graph gradient is missing or non-finite: {name}")
        magnitude = float(parameter.grad.abs().sum().item())
        if magnitude <= 0.0:
            raise RuntimeError(f"V11 graph gradient did not reach {name}")
        values[f"{name}_gradient_l1"] = magnitude
    return values


def _train_core(core, rows):
    with _v11_runner_patch():
        diagnostics = v10._train_core(core, rows)
    gradients = _assert_graph_gradients(core)
    diagnostics[-1].update(gradients)
    return diagnostics


def _per_family_report(arms: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for name, arm in arms.items():
        # State-swap summaries without per-mechanism records are intentionally
        # excluded; every ordinary behavioral arm is reported family-by-family.
        for row in arm.get("per_mechanism", ()):
            family = families.setdefault(row["generator_family"], {})
            bucket = family.setdefault(name, {"successes": 0, "count": 0, "target": 0.0})
            bucket["successes"] += int(row["success"])
            bucket["count"] += 1
            bucket["target"] += float(row["target_probability_after"])
    for family in families.values():
        for values in family.values():
            values["mean_target_probability"] = values.pop("target") / values["count"]
    return families


def _state_swap_family_rows(core, rows) -> dict[str, list[dict[str, Any]]]:
    """Recover per-mechanism swap observations omitted by the inherited summary."""

    states = v7._local_states(core, rows)
    records = {"equivalent_state_swap": [], "unrelated_state_swap": []}
    with torch.no_grad():
        for index, row in enumerate(rows):
            equivalent_index = next(
                (index + offset) % len(rows)
                for offset in range(1, len(rows) + 1)
                if rows[(index + offset) % len(rows)].generator_family
                == row.generator_family
            )
            unrelated_index = next(
                other_index
                for other_index, other in enumerate(rows)
                if other.generator_family != row.generator_family
            )
            for name, state_index in (
                ("equivalent_state_swap", equivalent_index),
                ("unrelated_state_swap", unrelated_index),
            ):
                output = _challenge(core, row, states[state_index])
                records[name].append(
                    {
                        "mechanism_ref": row.mechanism_ref,
                        "generator_family": row.generator_family,
                        "success": int(output.weights.argmax(dim=-1).item())
                        == row.target_index,
                        "target_probability_after": float(
                            output.weights[0, row.target_index].item()
                        ),
                    }
                )
    return records


def _evaluate_all(core, rows):
    with _v11_runner_patch():
        metrics = v10._evaluate_all(core, rows)
        with v10._v10_slot_scope(), _graph_lesion_scope(
            include_direction=True, include_graph_address=False
        ):
            graph_removed = v7._evaluate_sequence(core, rows, mode="full")
        with v10._v10_slot_scope(), _graph_lesion_scope(
            include_direction=False, include_graph_address=True
        ):
            direction_removed = v7._evaluate_sequence(core, rows, mode="full")
        swap_rows = _state_swap_family_rows(core, rows)
    graph_removed["mode"] = "graph_address_removed"
    direction_removed["mode"] = "direction_removed"
    metrics["arms"]["graph_address_removed"] = graph_removed
    metrics["arms"]["direction_removed"] = direction_removed
    metrics["successes"]["graph_address_removed"] = int(graph_removed["successes"])
    metrics["successes"]["direction_removed"] = int(direction_removed["successes"])
    metrics["accuracies"]["graph_address_removed"] = float(graph_removed["accuracy"])
    metrics["accuracies"]["direction_removed"] = float(direction_removed["accuracy"])
    for name, records in swap_rows.items():
        metrics["arms"][name]["per_mechanism"] = records
    full = metrics["arms"]["full"]
    metrics["graph_component"] = {
        "full_minus_graph_removed_successes": int(full["successes"] - graph_removed["successes"]),
        "full_minus_graph_removed_target_probability": float(
            full["mean_target_probability_after"]
            - graph_removed["mean_target_probability_after"]
        ),
        "full_minus_direction_removed_successes": int(
            full["successes"] - direction_removed["successes"]
        ),
        "full_minus_direction_removed_target_probability": float(
            full["mean_target_probability_after"]
            - direction_removed["mean_target_probability_after"]
        ),
        "per_family": _per_family_report(metrics["arms"]),
    }
    return metrics


def _factorization_metrics(core, rows):
    with _v11_runner_patch():
        return v9._factorization_metrics(core, rows)


def _allocation_metrics(core, rows):
    persistent = core.initial_memory_state().detached_clone()
    persistent_winners: list[int] = []
    persistent_weights: list[torch.Tensor] = []
    local_unique_counts = []
    maximum_nonzero_write_slots = 0
    attribution = []
    with torch.no_grad():
        for row in rows:
            local = core.initial_memory_state().detached_clone()
            local, local_outputs = _apply_stream(
                core, row, local, detach_state=True, collect_outputs=True
            )
            local_winners = [int(output.write_weights[0, 0].argmax().item()) for output in local_outputs]
            local_unique_counts.append(len(set(local_winners)))
            maximum_nonzero_write_slots = max(
                maximum_nonzero_write_slots,
                max(int((output.write_weights[0, 0] > 0).sum().item()) for output in local_outputs),
            )
            challenge = _challenge(core, row, local)
            success_slots = [
                slot for slot, outcome in zip(local_winners, row.episode_outcomes.tolist())
                if int(outcome) == 1
            ]
            failure_slots = [
                slot for slot, outcome in zip(local_winners, row.episode_outcomes.tolist())
                if int(outcome) == -1
            ]
            target_read = challenge.read_weights[0, row.target_index]
            failed_read = challenge.read_weights[0, list(row.failed_indices)]
            attribution.append({
                "mechanism_ref": row.mechanism_ref,
                "generator_family": row.generator_family,
                "target_success_slot_mass": float(target_read[success_slots].sum().item()),
                "target_failure_slot_mass": float(target_read[failure_slots].sum().item()),
                "failed_candidates_success_slot_mass": float(
                    failed_read[:, success_slots].sum(dim=-1).mean().item()
                ),
                "failed_candidates_failure_slot_mass": float(
                    failed_read[:, failure_slots].sum(dim=-1).mean().item()
                ),
            })
            persistent, outputs = _apply_stream(
                core, row, persistent, detach_state=True, collect_outputs=True
            )
            persistent_winners.extend(int(output.write_weights[0, 0].argmax().item()) for output in outputs)
            persistent_weights.extend(output.write_weights[0, 0] for output in outputs)
    with v10._v10_slot_scope():
        distribution = v7._distribution_metrics(torch.stack(persistent_weights))
    occupied = persistent.usage > 0
    sidecar_occupied = persistent.graph_step_mask.any(dim=-1)
    permutation = torch.arange(MEMORY_SLOTS - 1, -1, -1, device=persistent.keys.device)
    permuted = type(persistent)(
        keys=persistent.keys[permutation].clone(),
        values=persistent.values[permutation].clone(),
        usage=persistent.usage[permutation].clone(),
        step=persistent.step,
        graph_step_features=persistent.graph_step_features[permutation].clone(),
        graph_step_mask=persistent.graph_step_mask[permutation].clone(),
    )
    original_read = _challenge(core, rows[0], persistent)
    permuted_read = _challenge(core, rows[0], permuted)
    stress = persistent.detached_clone()
    for row in rows[:4]:
        stress, _ = _apply_stream(core, row, stress, detach_state=True)
    stress_state = v7._state_metrics(stress)
    return {
        "event_count": len(persistent_winners),
        "persistent_unique_winning_slots": len(set(persistent_winners)),
        "minimum_local_unique_winning_slots": min(local_unique_counts),
        "no_reuse_before_capacity": len(set(persistent_winners)) == len(persistent_winners),
        "unused_slots_after_persistent": MEMORY_SLOTS - int(occupied.sum().item()),
        "maximum_nonzero_write_slots": maximum_nonzero_write_slots,
        "aggregate_write": distribution,
        "sidecar_occupied_slots": int(sidecar_occupied.sum().item()),
        "sidecar_usage_alignment": bool(torch.equal(occupied, sidecar_occupied)),
        "slot_permutation_read_value_difference": float(
            (original_read.read_values - permuted_read.read_values).abs().max().item()
        ),
        "slot_permutation_residual_difference": float(
            (original_read.residuals - permuted_read.residuals).abs().max().item()
        ),
        "slot_permutation_score_difference": float(
            (original_read.scores - permuted_read.scores).abs().max().item()
        ),
        "stress_step": stress.step,
        "stress_bounds_valid": stress_state["bounds_valid"],
        "stress_sidecar_usage_alignment": bool(
            torch.equal(stress.usage > 0, stress.graph_step_mask.any(dim=-1))
        ),
        "stress_state": stress_state,
        "evaluator_only_read_attribution": attribution,
    }


def _graph_invariance_metrics(core, rows):
    maxima = {
        "maximum_graph_logit_difference_under_shuffle": 0.0,
        "maximum_graph_sidecar_difference_under_shuffle": 0.0,
        "maximum_graph_sidecar_mask_difference_under_shuffle": 0.0,
        "maximum_absolute_graph_logit": 0.0,
    }
    with torch.no_grad():
        for row in rows:
            initial = core.initial_memory_state().detached_clone()
            true_state, true_outputs = _apply_stream(
                core, row, initial, detach_state=True, collect_outputs=True
            )
            shuffled_values = v7.v6._counterfactual_outcomes(row, "outcome_shuffled")
            shuffled_state, shuffled_outputs = _apply_stream(
                core,
                row,
                initial,
                outcome_values=shuffled_values,
                detach_state=True,
                collect_outputs=True,
            )
            for true_output, shuffled_output in zip(true_outputs, shuffled_outputs):
                maxima["maximum_graph_logit_difference_under_shuffle"] = max(
                    maxima["maximum_graph_logit_difference_under_shuffle"],
                    float((true_output.graph_address_logits - shuffled_output.graph_address_logits).abs().max().item()),
                )
            maxima["maximum_graph_sidecar_difference_under_shuffle"] = max(
                maxima["maximum_graph_sidecar_difference_under_shuffle"],
                float((true_state.graph_step_features - shuffled_state.graph_step_features).abs().max().item()),
            )
            maxima["maximum_graph_sidecar_mask_difference_under_shuffle"] = max(
                maxima["maximum_graph_sidecar_mask_difference_under_shuffle"],
                float((true_state.graph_step_mask != shuffled_state.graph_step_mask).any().item()),
            )
            true_challenge = _challenge(core, row, true_state)
            shuffled_challenge = _challenge(core, row, shuffled_state)
            maxima["maximum_graph_logit_difference_under_shuffle"] = max(
                maxima["maximum_graph_logit_difference_under_shuffle"],
                float((true_challenge.graph_address_logits - shuffled_challenge.graph_address_logits).abs().max().item()),
            )
            maxima["maximum_absolute_graph_logit"] = max(
                maxima["maximum_absolute_graph_logit"],
                float(true_challenge.graph_address_logits.abs().max().item()),
            )
    return {"mechanism_count": len(rows), **maxima}


def _development_gate(metrics: dict[str, Any], identities_exact: bool) -> bool:
    graph = metrics.get("graph_component", {})
    invariant = metrics.get("graph_invariance", {})
    allocation = metrics.get("allocation", {})
    try:
        return bool(
            v10._development_gate(metrics, identities_exact)
            and int(metrics["full_failed_to_pass_repairs"]) >= 2
            and int(graph["full_minus_graph_removed_successes"]) >= GRAPH_SUCCESS_ADVANTAGE
            and float(graph["full_minus_graph_removed_target_probability"]) >= GRAPH_TARGET_ADVANTAGE
            and (
                int(graph["full_minus_direction_removed_successes"]) >= DIRECTION_SUCCESS_ADVANTAGE
                or float(graph["full_minus_direction_removed_target_probability"]) >= DIRECTION_TARGET_ADVANTAGE
            )
            and invariant.get("mechanism_count") == 8
            and float(invariant["maximum_absolute_graph_logit"]) > 0.0
            and float(invariant["maximum_graph_logit_difference_under_shuffle"]) <= 1.0e-6
            and float(invariant["maximum_graph_sidecar_difference_under_shuffle"]) <= 1.0e-6
            and float(invariant["maximum_graph_sidecar_mask_difference_under_shuffle"]) <= 1.0e-6
            and allocation.get("sidecar_occupied_slots") == 48
            and allocation.get("sidecar_usage_alignment") is True
            and allocation.get("stress_sidecar_usage_alignment") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inherited = v10._source_hashes()
    return {
        "runner": v7.v6.v5._sha256(Path(__file__).resolve()),
        "core": v7.v6.v5._sha256(root / "src/angler/reasoning/natural_trace_graph_causal_memory.py"),
        **{f"inherited_{name}": value for name, value in inherited.items()},
    }


def _new_core(content_width: int, device: torch.device):
    return NaturalTraceGraphCausalMemoryCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        step_width=content_width,
        rank=RANK,
        memory_slots=MEMORY_SLOTS,
        maximum_trace_steps=MAXIMUM_TRACE_STEPS,
    ).to(device=device, dtype=torch.float32)


def _checkpoint_payload(core, content_width, qwen_digest, metrics):
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "content_width": content_width,
        "temporal_width": TEMPORAL_WIDTH,
        "rank": RANK,
        "memory_slots": MEMORY_SLOTS,
        "maximum_trace_steps": MAXIMUM_TRACE_STEPS,
        "qwen_digest": qwen_digest,
        "core_digest": v7.v6.v5._module_digest(core),
        "source_hashes": _source_hashes(),
        "development_metrics": metrics,
        "core_state_dict": {name: tensor.detach().cpu() for name, tensor in core.state_dict().items()},
    }


def _validate_development_authorization(development: dict[str, Any], checkpoint_path: Path) -> None:
    identity_checks = development.get("identity_checks", {})
    metrics = development.get("development_metrics")
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "DEVELOPMENT_GATE_PASSED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == v7.v6.v5._sha256(checkpoint_path)
        and isinstance(metrics, dict)
        and identity_checks.get("exact") is True
        and identity_checks.get("foundation_before") == identity_checks.get("foundation_after")
        and identity_checks.get("core_before_evaluation") == identity_checks.get("core_after_evaluation")
        and _development_gate(metrics, True)
    )
    v7.v6.v5._validate_final_authorization(valid)


def _complete_metrics(core, rows):
    metrics = _evaluate_all(core, rows)
    metrics["factorization"] = _factorization_metrics(core, rows)
    metrics["allocation"] = _allocation_metrics(core, rows)
    metrics["graph_invariance"] = _graph_invariance_metrics(core, rows)
    return metrics


def _run_train_development(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V11 train-development identity is already consumed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v7.v6.v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_causal_neuromodulated_apprenticeship_v6()
    train_cpu = _encode_partition(qwen, corpus.train)
    development_cpu = _encode_partition(qwen, corpus.development)
    content_width = int(train_cpu[0].episode_queries.shape[-1])
    train_rows = tuple(row.to(device) for row in train_cpu)
    development_rows = tuple(row.to(device) for row in development_cpu)
    core = _new_core(content_width, device)
    donor = v10._load_semantic_donor(core, Path(args.v9_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v7.v6.v5._module_digest(core)
    metrics = _complete_metrics(core, development_rows)
    core_after = v7.v6.v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    exact = core_before == core_after and foundation_before == foundation_after
    authorized = _development_gate(metrics, exact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(core, content_width, foundation_before, metrics), checkpoint_path)
    result = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": "DEVELOPMENT_GATE_PASSED" if authorized else "DEVELOPMENT_NOT_SUPPORTED",
        "development_authorized": authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "training_mechanisms": 64, "development_mechanisms": 8,
            "final_mechanisms_opened": 0, "meta_pairs": 256,
            "optimizer_steps": 64, "rank": RANK, "memory_slots": MEMORY_SLOTS,
            "maximum_trace_steps": MAXIMUM_TRACE_STEPS,
        },
        "training_diagnostics": training,
        "development_metrics": metrics,
        "identity_checks": {
            "foundation_before": foundation_before, "foundation_after": foundation_after,
            "core_before_evaluation": core_before, "core_after_evaluation": core_after,
            "exact": exact,
        },
        "v9_semantic_donor": donor,
        "source_hashes": _source_hashes(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Development tests natural public-trace graph addressing in bounded synthetic "
            "factorized memory only; it is not arbitrary reasoning, AGI, or deployment authority."
        ),
    }
    v7.v6.v5._write_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"], "development_result": str(result_path),
        "checkpoint": str(checkpoint_path), "full_successes": metrics["successes"]["full"],
        "graph_target_advantage": metrics["graph_component"]["full_minus_graph_removed_target_probability"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _run_final(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V11 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_development_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != _source_hashes()
        or checkpoint.get("development_metrics")
        != development.get("development_metrics")
    ):
        raise RuntimeError("V11 checkpoint identity or source mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v7.v6.v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]:
        raise RuntimeError("frozen Qwen changed")
    corpus = build_causal_neuromodulated_apprenticeship_v6(include_final=True)
    rows = tuple(row.to(device) for row in _encode_partition(qwen, corpus.final))
    core = _new_core(checkpoint["content_width"], device)
    core.load_state_dict(checkpoint["core_state_dict"], strict=True)
    if v7.v6.v5._module_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V11 core digest mismatch")
    core_before = v7.v6.v5._module_digest(core)
    metrics = _complete_metrics(core, rows)
    core_after = v7.v6.v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    exact = core_before == core_after and foundation_before == foundation_after
    supported = _development_gate(metrics, exact)
    result = {
        "identity": IDENTITY, "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported, "corpus_id": CORPUS_ID, "seed": SEED,
        "development_result_sha256": v7.v6.v5._sha256(development_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "final_metrics": metrics,
        "identity_checks": {
            "foundation_before": foundation_before, "foundation_after": foundation_after,
            "core_before_evaluation": core_before, "core_after_evaluation": core_after,
            "exact": exact,
        },
        "source_hashes": _source_hashes(),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    v7.v6.v5._write_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"], "final_result": str(result_path),
        "full_successes": metrics["successes"]["full"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v9-checkpoint", default="/opt/angler/results/factorized-key-value-causal-memory-v9.pt")
    parser.add_argument("--checkpoint", default="/opt/angler/results/natural-trace-graph-causal-memory-v11.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/natural-trace-graph-causal-memory-v11-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/natural-trace-graph-causal-memory-v11-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
