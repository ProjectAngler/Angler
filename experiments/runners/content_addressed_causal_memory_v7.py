"""Frozen bounded content-addressed causal apprenticeship evaluation V7."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from angler.reasoning import (
    ContentAddressedCausalMemoryCore,
    ContentAddressedCausalMemoryState,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import causal_neuromodulated_apprenticeship_v6 as v6


IDENTITY = "angler.content-addressed-causal-memory.v7-first-result"
SEED = 2026083107
RANK = 32
MEMORY_SLOTS = 16
TEMPORAL_WIDTH = 8
LEARNING_RATE = 3.0e-4
META_BATCH_PAIRS = 4
GRADIENT_CLIP = 5.0
OCCUPANCY_WEIGHT = 0.02
MINIMUM_EFFECTIVE_WRITE_SLOTS = 4.0
MAXIMUM_AGGREGATE_WRITE_SHARE = 0.50
EPSILON = 1.0e-8


def _state_metrics(state: ContentAddressedCausalMemoryState) -> dict[str, Any]:
    usage_mass = state.usage.sum().clamp_min(EPSILON)
    distribution = state.usage / usage_mass
    entropy = -(distribution * torch.log(distribution.clamp_min(EPSILON))).sum()
    return {
        "step": state.step,
        "key_norm": float(state.keys.norm().item()),
        "value_norm": float(state.values.norm().item()),
        "usage_norm": float(state.usage.norm().item()),
        "key_min": float(state.keys.min().item()),
        "key_max": float(state.keys.max().item()),
        "value_min": float(state.values.min().item()),
        "value_max": float(state.values.max().item()),
        "usage_min": float(state.usage.min().item()),
        "usage_max": float(state.usage.max().item()),
        "occupied_slots": int((state.usage > 1.0e-3).sum().item()),
        "usage_entropy": float(entropy.item()),
        "effective_usage_slots": float(torch.exp(entropy).item()),
        "maximum_usage_share": float(distribution.max().item()),
        "bounds_valid": bool(
            (state.keys.abs() <= 1.0 + 1.0e-6).all().item()
            and (state.values.abs() <= 1.0 + 1.0e-6).all().item()
            and (state.usage >= -1.0e-6).all().item()
            and (state.usage <= 1.0 + 1.0e-6).all().item()
        ),
    }


def _distribution_metrics(weights: torch.Tensor) -> dict[str, float]:
    values = weights.detach().reshape(-1, MEMORY_SLOTS).sum(dim=0)
    distribution = values / values.sum().clamp_min(EPSILON)
    entropy = -(distribution * torch.log(distribution.clamp_min(EPSILON))).sum()
    return {
        "entropy": float(entropy.item()),
        "effective_slots": float(torch.exp(entropy).item()),
        "maximum_share": float(distribution.max().item()),
        "mass": float(values.sum().item()),
    }


def _base_scores(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return v6.v5._base_scores(query, candidates)


def _apply_stream(
    core: ContentAddressedCausalMemoryCore,
    row: v6.v5.EncodedMechanism,
    state: ContentAddressedCausalMemoryState,
    *,
    outcome_values: Sequence[int | float] | None = None,
    use_moving_origin: bool = True,
    detach_state: bool,
    collect_outputs: bool = False,
) -> tuple[ContentAddressedCausalMemoryState, tuple[Any, ...]]:
    outcomes = row.episode_outcomes if outcome_values is None else torch.tensor(
        tuple(outcome_values), device=row.episode_outcomes.device, dtype=torch.float32
    )
    if outcomes.shape != (6,):
        raise ValueError("V7 stream must contain exactly six outcomes")
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
            _base_scores(query, candidates),
            memory_state=state,
        )
        state = core.apply_mixed_outcome_update(
            state,
            output,
            event_outcome,
            mask,
            detach_state=detach_state,
        )
        if collect_outputs:
            outputs.append(output)
    return state, tuple(outputs)


def _challenge(
    core: ContentAddressedCausalMemoryCore,
    row: v6.v5.EncodedMechanism,
    state: ContentAddressedCausalMemoryState,
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
        _base_scores(query, candidates),
        memory_state=state,
    )


def _adapt_pair(
    core: ContentAddressedCausalMemoryCore,
    anchor: v6.v5.EncodedMechanism,
    current: v6.v5.EncodedMechanism,
    *,
    outcome_mode: str | None,
    detach_state: bool,
    collect_outputs: bool,
) -> dict[str, Any]:
    state = core.initial_memory_state()
    anchor_outcomes = None if outcome_mode is None else v6._counterfactual_outcomes(anchor, outcome_mode)
    current_outcomes = None if outcome_mode is None else v6._counterfactual_outcomes(current, outcome_mode)
    state, anchor_writes = _apply_stream(
        core,
        anchor,
        state,
        outcome_values=anchor_outcomes,
        detach_state=detach_state,
        collect_outputs=collect_outputs,
    )
    anchor_immediate = _challenge(core, anchor, state)
    current_before = _challenge(core, current, state)
    state, current_writes = _apply_stream(
        core,
        current,
        state,
        outcome_values=current_outcomes,
        detach_state=detach_state,
        collect_outputs=collect_outputs,
    )
    return {
        "state": state,
        "anchor_immediate": anchor_immediate,
        "current_before": current_before,
        "current_after": _challenge(core, current, state),
        "anchor_retained": _challenge(core, anchor, state),
        "writes": (*anchor_writes, *current_writes),
    }


def _challenge_values(output: Any, row: v6.v5.EncodedMechanism):
    return v6.v5._challenge_values(output, row)


def _margin_loss(output: Any, row: v6.v5.EncodedMechanism) -> torch.Tensor:
    return v6.v5._margin_loss(output, row)


def _causal_reference_loss(
    true_pair: dict[str, Any],
    reference_pair: dict[str, Any],
    anchor: v6.v5.EncodedMechanism,
    current: v6.v5.EncodedMechanism,
) -> torch.Tensor:
    terms = []
    for key, row in (("current_after", current), ("anchor_retained", anchor)):
        true_target, true_failed = _challenge_values(true_pair[key], row)
        reference_target, reference_failed = _challenge_values(reference_pair[key], row)
        terms.append(v6._smooth_margin(0.10 + reference_target - true_target))
        terms.append(v6._smooth_margin(0.10 + true_failed - reference_failed))
    return 0.25 * torch.stack(terms).sum()


def _occupancy_loss(outputs: Sequence[Any]) -> tuple[torch.Tensor, dict[str, float]]:
    write_masses = torch.stack(tuple(output.write_weights[0, 0] for output in outputs)).sum(dim=0)
    distribution = (write_masses + EPSILON) / (
        write_masses.sum() + MEMORY_SLOTS * EPSILON
    )
    entropy = -(distribution * torch.log(distribution)).sum()
    shortfall = F.relu(entropy.new_tensor(math.log(MINIMUM_EFFECTIVE_WRITE_SLOTS)) - entropy)
    loss = OCCUPANCY_WEIGHT * (shortfall / math.log(MINIMUM_EFFECTIVE_WRITE_SLOTS)).square()
    return loss, {
        "aggregate_write_entropy": float(entropy.detach().item()),
        "aggregate_effective_write_slots": float(torch.exp(entropy.detach()).item()),
        "aggregate_maximum_write_share": float(distribution.detach().max().item()),
        "aggregate_write_mass": float(write_masses.detach().sum().item()),
    }


def _train_core(
    core: ContentAddressedCausalMemoryCore,
    rows: Sequence[v6.v5.EncodedMechanism],
) -> list[dict[str, float]]:
    if len(rows) != 64:
        raise ValueError("V7 training requires exactly 64 mechanisms")
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0
    )
    schedule = v6._pair_schedule()
    diagnostics = []
    core.train()
    for start in range(0, len(schedule), META_BATCH_PAIRS):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        causal_values = []
        occupancy_values = []
        slot_metrics = []
        for anchor_index, current_index in schedule[start : start + META_BATCH_PAIRS]:
            anchor, current = rows[anchor_index], rows[current_index]
            with torch.no_grad():
                deranged = _adapt_pair(
                    core, anchor, current, outcome_mode="outcome_shuffled",
                    detach_state=True, collect_outputs=False,
                )
                zeroed = _adapt_pair(
                    core, anchor, current, outcome_mode="outcome_zeroed",
                    detach_state=True, collect_outputs=False,
                )
            true_pair = _adapt_pair(
                core, anchor, current, outcome_mode=None,
                detach_state=False, collect_outputs=True,
            )
            current_target, current_failed = _challenge_values(true_pair["current_after"], current)
            anchor_target, _ = _challenge_values(true_pair["anchor_retained"], anchor)
            anchor_immediate, _ = _challenge_values(true_pair["anchor_immediate"], anchor)
            _, current_failed_before = _challenge_values(true_pair["current_before"], current)
            useful = -0.5 * torch.log(current_target.clamp_min(EPSILON))
            useful = useful - 0.5 * torch.log(anchor_target.clamp_min(EPSILON))
            useful = useful + 0.25 * _margin_loss(true_pair["current_after"], current)
            useful = useful + 0.25 * _margin_loss(true_pair["anchor_retained"], anchor)
            anti_repeat = 0.5 * F.relu(current_failed - current_failed_before + 0.05)
            retention = F.relu(anchor_immediate.detach() - anchor_target - 0.05)
            causal = 0.5 * _causal_reference_loss(true_pair, deranged, anchor, current)
            causal = causal + 0.5 * _causal_reference_loss(true_pair, zeroed, anchor, current)
            occupancy, occupancy_metrics = _occupancy_loss(true_pair["writes"])
            loss = useful + anti_repeat + retention + causal + occupancy
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite V7 training loss")
            step_losses.append(loss)
            causal_values.append(float(causal.detach().item()))
            occupancy_values.append(float(occupancy.detach().item()))
            slot_metrics.append(occupancy_metrics)
        meta_loss = torch.stack(step_losses).mean()
        meta_loss.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
            raise RuntimeError("missing or non-finite V7 gradients")
        required = (
            core.outcome_embedding.weight.grad,
            core.writer[-1].weight.grad,
            core.neuromodulator[-1].weight.grad,
            core.read_key_network[1].weight.grad,
        )
        if any(value is None or float(value.abs().sum().item()) <= 0.0 for value in required):
            raise RuntimeError("V7 causal path did not reach outcome/read/write/modulator parameters")
        gradient_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_CLIP)
        optimizer.step()
        diagnostics.append(
            {
                "optimizer_step": float(start // META_BATCH_PAIRS + 1),
                "loss": float(meta_loss.detach().item()),
                "causal_loss": sum(causal_values) / len(causal_values),
                "occupancy_loss": sum(occupancy_values) / len(occupancy_values),
                "gradient_norm": float(gradient_norm.detach().item()),
                "effective_write_slots": sum(x["aggregate_effective_write_slots"] for x in slot_metrics) / len(slot_metrics),
                "maximum_write_share": max(x["aggregate_maximum_write_share"] for x in slot_metrics),
            }
        )
    return diagnostics


def _evaluate_sequence(
    core: ContentAddressedCausalMemoryCore,
    rows: Sequence[v6.v5.EncodedMechanism],
    *,
    mode: str,
) -> dict[str, Any]:
    core.eval()
    persistent = core.initial_memory_state().detached_clone()
    immediate, before_probs, before_failed, after_failed = [], [], [], []
    predictions, before_predictions, per_mechanism, local_states = [], [], [], []
    write_weights = []
    with torch.no_grad():
        for row in rows:
            state = core.initial_memory_state().detached_clone() if mode == "state_reset" else persistent
            use_mor = mode != "moving_origin_removed"
            before = _challenge(core, row, state, use_moving_origin=use_mor)
            before_target, failed_before = _challenge_values(before, row)
            before_prediction = int(before.weights.argmax(dim=-1).item())
            outputs = ()
            if mode not in ("memory_disabled", "fair_semantic_retrieval", "state_zero"):
                outcomes = None
                if mode == "outcome_shuffled":
                    outcomes = v6._counterfactual_outcomes(row, mode)
                elif mode == "outcome_zeroed":
                    outcomes = (0,) * 6
                state, outputs = _apply_stream(
                    core, row, state, outcome_values=outcomes,
                    use_moving_origin=use_mor, detach_state=True, collect_outputs=True,
                )
            after = _challenge(core, row, state, use_moving_origin=use_mor)
            if mode == "fair_semantic_retrieval":
                weights = torch.softmax(after.normalized_base_scores, dim=-1)
                prediction = int(weights.argmax(dim=-1).item())
                target = weights[0, row.target_index]
                failed_after = weights[0, list(row.failed_indices)].sum()
            else:
                prediction = int(after.weights.argmax(dim=-1).item())
                target, failed_after = _challenge_values(after, row)
            if mode != "state_reset":
                persistent = state
            local_states.append(state)
            predictions.append(prediction)
            before_predictions.append(before_prediction)
            immediate.append(float(target.item()))
            before_probs.append(float(before_target.item()))
            before_failed.append(float(failed_before.item()))
            after_failed.append(float(failed_after.item()))
            write_weights.extend(output.write_weights[0, 0] for output in outputs)
            per_mechanism.append(
                {
                    "mechanism_ref": row.mechanism_ref,
                    "generator_family": row.generator_family,
                    "target_index": row.target_index,
                    "prediction": prediction,
                    "success": prediction == row.target_index,
                    "target_probability_before": float(before_target.item()),
                    "target_probability_after": float(target.item()),
                    "failed_probability_before": float(failed_before.item()),
                    "failed_probability_after": float(failed_after.item()),
                    "memory": _state_metrics(state),
                    "challenge_read": _distribution_metrics(after.read_weights),
                }
            )
        retained, retained_successes = [], 0
        for index, row in enumerate(rows):
            state = local_states[index] if mode == "state_reset" else persistent
            output = _challenge(core, row, state, use_moving_origin=mode != "moving_origin_removed")
            target, _ = _challenge_values(output, row)
            retained.append(float(target.item()))
            retained_successes += int(output.weights.argmax(dim=-1).item() == row.target_index)
    successes = sum(int(p == row.target_index) for p, row in zip(predictions, rows))
    aggregate_write = (
        _distribution_metrics(torch.stack(write_weights))
        if write_weights
        else {"entropy": 0.0, "effective_slots": 0.0, "maximum_share": 1.0, "mass": 0.0}
    )
    return {
        "mode": mode,
        "successes": successes,
        "accuracy": successes / len(rows),
        "mean_target_probability_before": sum(before_probs) / len(rows),
        "mean_target_probability_after": sum(immediate) / len(rows),
        "mean_failed_probability_before": sum(before_failed) / len(rows),
        "mean_failed_probability_after": sum(after_failed) / len(rows),
        "failed_to_pass_repairs": sum(
            int(b != row.target_index and p == row.target_index)
            for b, p, row in zip(before_predictions, predictions, rows)
        ),
        "sequential_retained_successes": retained_successes,
        "sequential_target_probability_retention": min(1.0, sum(retained) / max(sum(immediate), EPSILON)),
        "final_memory": _state_metrics(persistent),
        "aggregate_write": aggregate_write,
        "per_mechanism": per_mechanism,
    }


def _local_states(core, rows):
    values = []
    with torch.no_grad():
        for row in rows:
            state = core.initial_memory_state().detached_clone()
            state, _ = _apply_stream(core, row, state, detach_state=True)
            values.append(state)
    return tuple(values)


def _evaluate_state_swaps(core, rows):
    states = _local_states(core, rows)
    equivalent_predictions, unrelated_predictions = [], []
    equivalent_probabilities, own_probabilities = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            equivalent_index = next(
                (index + offset) % len(rows)
                for offset in range(1, len(rows) + 1)
                if rows[(index + offset) % len(rows)].generator_family == row.generator_family
            )
            unrelated_index = next(i for i, other in enumerate(rows) if other.generator_family != row.generator_family)
            equivalent = _challenge(core, row, states[equivalent_index])
            unrelated = _challenge(core, row, states[unrelated_index])
            own = _challenge(core, row, states[index])
            equivalent_predictions.append(int(equivalent.weights.argmax(dim=-1).item()))
            unrelated_predictions.append(int(unrelated.weights.argmax(dim=-1).item()))
            equivalent_probabilities.append(float(equivalent.weights[0, row.target_index].item()))
            own_probabilities.append(float(own.weights[0, row.target_index].item()))
    equivalent_successes = sum(int(p == row.target_index) for p, row in zip(equivalent_predictions, rows))
    unrelated_successes = sum(int(p == row.target_index) for p, row in zip(unrelated_predictions, rows))
    return (
        {
            "mode": "equivalent_state_swap",
            "successes": equivalent_successes,
            "accuracy": equivalent_successes / len(rows),
            "target_probability_retention": min(1.0, sum(equivalent_probabilities) / max(sum(own_probabilities), EPSILON)),
        },
        {"mode": "unrelated_state_swap", "successes": unrelated_successes, "accuracy": unrelated_successes / len(rows)},
    )


def _evaluate_all(core, rows):
    modes = (
        "full", "state_reset", "outcome_shuffled", "outcome_zeroed",
        "memory_disabled", "fair_semantic_retrieval", "moving_origin_removed", "state_zero",
    )
    arms = {mode: _evaluate_sequence(core, rows, mode=mode) for mode in modes}
    equivalent, unrelated = _evaluate_state_swaps(core, rows)
    arms["equivalent_state_swap"], arms["unrelated_state_swap"] = equivalent, unrelated
    successes = {name: int(value["successes"]) for name, value in arms.items()}
    metrics = {
        "mechanism_count": len(rows),
        "successes": successes,
        "accuracies": {name: float(value["accuracy"]) for name, value in arms.items()},
        "equivalent_state_target_probability_retention": equivalent["target_probability_retention"],
        "full_failed_to_pass_repairs": arms["full"]["failed_to_pass_repairs"],
        "full_negative_feedback_reduced_failed_probability": (
            arms["full"]["mean_failed_probability_after"] < arms["full"]["mean_failed_probability_before"]
        ),
        "full_target_probability_non_decreasing": (
            arms["full"]["mean_target_probability_after"] >= arms["full"]["mean_target_probability_before"]
        ),
        "sequential_target_probability_retention": arms["full"]["sequential_target_probability_retention"],
        "memory_structure": {
            "all_bounds_valid": all(row["memory"]["bounds_valid"] for row in arms["full"]["per_mechanism"]),
            "aggregate_effective_write_slots": arms["full"]["aggregate_write"]["effective_slots"],
            "aggregate_maximum_write_share": arms["full"]["aggregate_write"]["maximum_share"],
            "slot_collapse_observed": (
                arms["full"]["aggregate_write"]["effective_slots"] < MINIMUM_EFFECTIVE_WRITE_SLOTS
                or arms["full"]["aggregate_write"]["maximum_share"] > MAXIMUM_AGGREGATE_WRITE_SHARE
            ),
        },
        "arms": arms,
    }
    return v6._augment_causal_metrics(metrics)


def _development_gate(metrics: dict[str, Any], identities_exact: bool) -> bool:
    structure = metrics.get("memory_structure", {})
    return bool(
        v6._development_gate(metrics, identities_exact)
        and structure.get("all_bounds_valid") is True
        and structure.get("aggregate_effective_write_slots", 0.0) >= MINIMUM_EFFECTIVE_WRITE_SLOTS
        and structure.get("aggregate_maximum_write_share", 1.0) <= MAXIMUM_AGGREGATE_WRITE_SHARE
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "runner": v6.v5._sha256(Path(__file__).resolve()),
        "core": v6.v5._sha256(root / "src/angler/reasoning/content_addressed_causal_memory.py"),
        "corpus": v6.v5._sha256(root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py"),
    }


def _load_semantic_donor(core: ContentAddressedCausalMemoryCore, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != v6.IDENTITY:
        raise RuntimeError("V6 semantic donor identity mismatch")
    donor = checkpoint["core_state_dict"]
    mapping = {
        "query_projection.weight": "query_projection.weight",
        "candidate_projection.weight": "candidate_projection.weight",
        "temporal_projection.weight": "temporal_projection.weight",
        "feature_network.0.weight": "semantic_network.0.weight",
        "feature_network.0.bias": "semantic_network.0.bias",
        "feature_network.1.weight": "semantic_network.1.weight",
        "feature_network.1.bias": "semantic_network.1.bias",
    }
    state = core.state_dict()
    with torch.no_grad():
        for source, destination in mapping.items():
            if donor[source].shape != state[destination].shape:
                raise RuntimeError("V6 semantic donor tensor shape mismatch")
            target = dict(core.named_parameters())[destination]
            target.copy_(donor[source].to(device=target.device, dtype=target.dtype))
    loaded = core.state_dict()
    if any(not torch.equal(loaded[destination].cpu(), donor[source]) for source, destination in mapping.items()):
        raise RuntimeError("V6 semantic donor was not loaded byte-exactly")
    return {"path": str(checkpoint_path), "sha256": v6.v5._sha256(checkpoint_path), "mapping": mapping}


def _checkpoint_payload(core, content_width, qwen_digest, metrics):
    return {
        "identity": IDENTITY, "corpus_id": CORPUS_ID, "seed": SEED,
        "content_width": content_width, "temporal_width": TEMPORAL_WIDTH,
        "rank": RANK, "memory_slots": MEMORY_SLOTS, "qwen_digest": qwen_digest,
        "core_digest": v6.v5._module_digest(core), "source_hashes": _source_hashes(),
        "development_metrics": metrics,
        "core_state_dict": {name: tensor.detach().cpu() for name, tensor in core.state_dict().items()},
    }


def _run_train_development(args):
    checkpoint_path, result_path = Path(args.checkpoint), Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V7 train-development identity is already consumed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v6.v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_causal_neuromodulated_apprenticeship_v6()
    train_cpu = v6._encode_partition(qwen, corpus.train)
    development_cpu = v6._encode_partition(qwen, corpus.development)
    content_width = int(train_cpu[0].episode_queries.shape[-1])
    train_rows = tuple(row.to(device) for row in train_cpu)
    development_rows = tuple(row.to(device) for row in development_cpu)
    core = ContentAddressedCausalMemoryCore(
        content_width=content_width, temporal_width=TEMPORAL_WIDTH,
        rank=RANK, memory_slots=MEMORY_SLOTS,
    ).to(device=device, dtype=torch.float32)
    donor = _load_semantic_donor(core, Path(args.v6_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v6.v5._module_digest(core)
    metrics = _evaluate_all(core, development_rows)
    core_after = v6.v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    exact = core_before == core_after and foundation_before == foundation_after
    authorized = _development_gate(metrics, exact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(core, content_width, foundation_before, metrics), checkpoint_path)
    result = {
        "identity": IDENTITY, "phase": "train-development",
        "classification": "DEVELOPMENT_GATE_PASSED" if authorized else "DEVELOPMENT_NOT_SUPPORTED",
        "development_authorized": authorized, "corpus_id": CORPUS_ID, "seed": SEED,
        "frozen_compute": {"training_mechanisms":64,"development_mechanisms":8,"final_mechanisms_opened":0,"meta_pairs":256,"optimizer_steps":64,"rank":RANK,"memory_slots":MEMORY_SLOTS},
        "training_diagnostics": training, "development_metrics": metrics,
        "identity_checks": {"foundation_before":foundation_before,"foundation_after":foundation_after,"core_before_evaluation":core_before,"core_after_evaluation":core_after,"exact":exact},
        "v6_semantic_donor": donor, "source_hashes": _source_hashes(),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": v6.v5._sha256(checkpoint_path),
        "environment": v6.v5._environment_record(device), "elapsed_seconds": time.perf_counter()-started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": "Development tests bounded content-addressed outcome-causal adaptation in synthetic software mechanisms; it is not final, arbitrary reasoning, AGI, or deployment authority.",
    }
    v6.v5._write_json(result_path, result)
    print(json.dumps({"classification":result["classification"],"development_result":str(result_path),"checkpoint":str(checkpoint_path),"full_successes":metrics["successes"]["full"],"effective_write_slots":metrics["memory_structure"]["aggregate_effective_write_slots"],"elapsed_seconds":result["elapsed_seconds"]},sort_keys=True),flush=True)


def _run_final(args):
    checkpoint_path, development_path, result_path = Path(args.checkpoint), Path(args.development_result), Path(args.final_result)
    if result_path.exists(): raise FileExistsError("V7 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    v6.v5._validate_final_authorization(development.get("development_authorized") is True)
    if development.get("checkpoint_sha256") != v6.v5._sha256(checkpoint_path): raise RuntimeError("V7 checkpoint changed")
    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if checkpoint.get("identity") != IDENTITY or checkpoint.get("source_hashes") != _source_hashes(): raise RuntimeError("V7 identity/source mismatch")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.use_deterministic_algorithms(True,warn_only=False)
    device=torch.device(args.device); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device); started=time.perf_counter()
    qwen=v6.v5._load_qwen(args.model_path); foundation_before=foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]: raise RuntimeError("frozen Qwen changed")
    corpus=build_causal_neuromodulated_apprenticeship_v6(include_final=True)
    rows=tuple(row.to(device) for row in v6._encode_partition(qwen,corpus.final))
    core=ContentAddressedCausalMemoryCore(content_width=checkpoint["content_width"],temporal_width=checkpoint["temporal_width"],rank=checkpoint["rank"],memory_slots=checkpoint["memory_slots"]).to(device=device,dtype=torch.float32)
    core.load_state_dict(checkpoint["core_state_dict"],strict=True)
    if v6.v5._module_digest(core)!=checkpoint["core_digest"]: raise RuntimeError("V7 core digest mismatch")
    core_before=v6.v5._module_digest(core); metrics=_evaluate_all(core,rows); core_after=v6.v5._module_digest(core); foundation_after=foundation_tensor_digest(qwen.model)
    exact=core_before==core_after and foundation_before==foundation_after; supported=_development_gate(metrics,exact)
    result={"identity":IDENTITY,"phase":"final","classification":"SUPPORTED" if supported else "NOT_SUPPORTED","supported":supported,"corpus_id":CORPUS_ID,"seed":SEED,"development_result_sha256":v6.v5._sha256(development_path),"checkpoint_sha256":v6.v5._sha256(checkpoint_path),"final_metrics":metrics,"identity_checks":{"foundation_before":foundation_before,"foundation_after":foundation_after,"core_before_evaluation":core_before,"core_after_evaluation":core_after,"exact":exact},"source_hashes":_source_hashes(),"environment":v6.v5._environment_record(device),"elapsed_seconds":time.perf_counter()-started,"peak_angler_gpu_bytes":torch.cuda.max_memory_allocated(device),"interpretation":"Supported would establish bounded synthetic content-addressed outcome-causal apprenticeship only, not arbitrary intelligence, AGI, or deployment readiness."}
    v6.v5._write_json(result_path,result); print(json.dumps({"classification":result["classification"],"final_result":str(result_path),"full_successes":metrics["successes"]["full"],"elapsed_seconds":result["elapsed_seconds"]},sort_keys=True),flush=True)


def _parser():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase",choices=("train-dev","final"),required=True); parser.add_argument("--model-path",default="/opt/angler/models/Qwen3-4B"); parser.add_argument("--device",default="cuda:1"); parser.add_argument("--v6-checkpoint",default="/opt/angler/results/causal-neuromodulated-apprenticeship-v6.pt"); parser.add_argument("--checkpoint",default="/opt/angler/results/content-addressed-causal-memory-v7.pt"); parser.add_argument("--development-result",default="/opt/angler/results/content-addressed-causal-memory-v7-development.json"); parser.add_argument("--final-result",default="/opt/angler/results/content-addressed-causal-memory-v7-final.json"); return parser


def main(argv: Sequence[str] | None=None):
    args=_parser().parse_args(argv); _run_train_development(args) if args.phase=="train-dev" else _run_final(args)


if __name__ == "__main__": main()
