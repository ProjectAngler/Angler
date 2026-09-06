"""Frozen causal-outcome OML/ANML apprenticeship evaluation V6."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from angler.reasoning import CausalNeuromodulatedApprenticeshipCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import outcome_aware_apprenticeship_v5 as v5


IDENTITY = "angler.causal-neuromodulated-apprenticeship.v6-first-result"
SEED = 2026083106
RANK = 32
TEMPORAL_WIDTH = 8
LEARNING_RATE = 3.0e-4
META_BATCH_PAIRS = 4
GRADIENT_CLIP = 5.0
CAUSAL_TRAIN_MARGIN = 0.10
CAUSAL_GATE_MARGIN = 0.10
RETENTION_TOLERANCE = 0.05
PROBABILITY_EPSILON = 1.0e-8
REQUIRED_CONTROLS = (
    "state_reset",
    "outcome_shuffled",
    "outcome_zeroed",
    "memory_disabled",
    "fair_semantic_retrieval",
    "moving_origin_removed",
)


def _pair_schedule() -> tuple[tuple[int, int], ...]:
    return tuple(
        (anchor, (anchor + offset) % 64)
        for offset in (5, 17, 29, 43)
        for anchor in range(64)
    )


def _episode_query_text(mechanism: Any, index: int) -> str:
    """Expose task and request only; outcome enters solely through its token."""

    episode = mechanism.public.episodes[index]
    return f"Task: {episode.task_text}\nRequest: {episode.request_text}"


def _challenge_query_text(mechanism: Any) -> str:
    challenge = mechanism.public.challenge
    return f"Task: {challenge.task_text}\nRequest: {challenge.request_text}"


def _encode_partition(qwen: Any, mechanisms: Sequence[Any]) -> tuple[v5.EncodedMechanism, ...]:
    texts: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    for mechanism in mechanisms:
        query_start = len(texts)
        texts.extend(_episode_query_text(mechanism, index) for index in range(6))
        candidate_start = len(texts)
        texts.extend(episode.action_trace_text for episode in mechanism.public.episodes)
        challenge_query_index = len(texts)
        texts.append(_challenge_query_text(mechanism))
        challenge_candidate_start = len(texts)
        texts.extend(mechanism.public.challenge.candidate_action_trace_texts)
        spans.append((query_start, candidate_start, challenge_query_index, challenge_candidate_start))
    encoded = qwen.embed(texts).to(dtype=torch.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(texts):
        raise RuntimeError("Qwen embeddings do not align with the frozen V6 corpus")
    rows: list[v5.EncodedMechanism] = []
    for mechanism, span in zip(mechanisms, spans):
        query_start, candidate_start, challenge_query_index, challenge_candidate_start = span
        outcomes = tuple(int(episode.outcome_value) for episode in mechanism.public.episodes)
        target_indices = mechanism.supervision.target_successful_candidate_indices
        if len(target_indices) != 1:
            raise RuntimeError("V6 requires exactly one successful challenge candidate")
        candidate_count = len(mechanism.public.challenge.candidate_action_trace_texts)
        rows.append(
            v5.EncodedMechanism(
                mechanism_ref=mechanism.metadata.mechanism_ref,
                generator_family=mechanism.metadata.generator_family,
                episode_queries=encoded[query_start : query_start + 6].clone(),
                episode_candidates=encoded[candidate_start : candidate_start + 6].clone(),
                episode_temporal=torch.tensor(
                    [v5._episode_temporal(index) for index in range(6)],
                    dtype=torch.float32,
                ),
                episode_outcomes=torch.tensor(outcomes, dtype=torch.float32),
                challenge_query=encoded[challenge_query_index].clone(),
                challenge_candidates=encoded[
                    challenge_candidate_start : challenge_candidate_start + candidate_count
                ].clone(),
                challenge_temporal=v5._challenge_temporal(candidate_count),
                target_index=target_indices[0],
                failed_indices=mechanism.supervision.relevant_failed_candidate_indices,
            )
        )
    return tuple(rows)


def _counterfactual_outcomes(row: v5.EncodedMechanism, mode: str) -> tuple[int, ...]:
    observed = tuple(int(value) for value in row.episode_outcomes.tolist())
    if mode == "outcome_shuffled":
        return v5._deranged_outcomes(observed)
    if mode == "outcome_zeroed":
        return (0,) * 6
    raise ValueError("unknown V6 counterfactual outcome mode")


def _adapt_pair(
    core: CausalNeuromodulatedApprenticeshipCore,
    anchor: v5.EncodedMechanism,
    current: v5.EncodedMechanism,
    *,
    outcome_mode: str | None,
):
    state = core.initial_plastic_state(detach=False)
    anchor_outcomes = None if outcome_mode is None else _counterfactual_outcomes(anchor, outcome_mode)
    current_outcomes = None if outcome_mode is None else _counterfactual_outcomes(current, outcome_mode)
    state = v5._apply_stream(
        core,
        anchor,
        state,
        outcome_values=anchor_outcomes,
        detach_state=False,
    )
    anchor_immediate = v5._challenge(core, anchor, state)
    current_before = v5._challenge(core, current, state)
    state = v5._apply_stream(
        core,
        current,
        state,
        outcome_values=current_outcomes,
        detach_state=False,
    )
    return {
        "state": state,
        "anchor_immediate": anchor_immediate,
        "current_before": current_before,
        "current_after": v5._challenge(core, current, state),
        "anchor_retained": v5._challenge(core, anchor, state),
    }


def _smooth_margin(value: torch.Tensor) -> torch.Tensor:
    scale = value.new_tensor(0.05)
    return scale * F.softplus(value / scale)


def _causal_probability_terms(
    true_output: Any,
    counterfactual_output: Any,
    row: v5.EncodedMechanism,
) -> tuple[torch.Tensor, torch.Tensor]:
    true_target, true_failed = v5._challenge_values(true_output, row)
    counter_target, counter_failed = v5._challenge_values(counterfactual_output, row)
    target_contrast = _smooth_margin(
        true_target.new_tensor(CAUSAL_TRAIN_MARGIN) - true_target + counter_target
    )
    failure_contrast = _smooth_margin(
        true_failed.new_tensor(CAUSAL_TRAIN_MARGIN) + true_failed - counter_failed
    )
    return target_contrast, failure_contrast


def _causal_branch_loss(
    true_pair: dict[str, Any],
    counterfactual_pair: dict[str, Any],
    anchor: v5.EncodedMechanism,
    current: v5.EncodedMechanism,
) -> torch.Tensor:
    current_target, current_failure = _causal_probability_terms(
        true_pair["current_after"], counterfactual_pair["current_after"], current
    )
    anchor_target, anchor_failure = _causal_probability_terms(
        true_pair["anchor_retained"], counterfactual_pair["anchor_retained"], anchor
    )
    return 0.5 * torch.stack((current_target, anchor_target)).mean() + 0.5 * torch.stack(
        (current_failure, anchor_failure)
    ).mean()


def _train_core(
    core: CausalNeuromodulatedApprenticeshipCore,
    rows: Sequence[v5.EncodedMechanism],
) -> list[dict[str, float]]:
    if len(rows) != 64:
        raise ValueError("V6 training requires exactly 64 mechanisms")
    optimizer = torch.optim.AdamW(
        core.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    schedule = _pair_schedule()
    diagnostics: list[dict[str, float]] = []
    core.train()
    for start in range(0, len(schedule), META_BATCH_PAIRS):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        causal_values: list[float] = []
        useful_values: list[float] = []
        for anchor_index, current_index in schedule[start : start + META_BATCH_PAIRS]:
            anchor = rows[anchor_index]
            current = rows[current_index]
            true_pair = _adapt_pair(core, anchor, current, outcome_mode=None)
            deranged_pair = _adapt_pair(
                core, anchor, current, outcome_mode="outcome_shuffled"
            )
            zeroed_pair = _adapt_pair(
                core, anchor, current, outcome_mode="outcome_zeroed"
            )

            current_target, current_failed = v5._challenge_values(
                true_pair["current_after"], current
            )
            anchor_target, _ = v5._challenge_values(
                true_pair["anchor_retained"], anchor
            )
            anchor_immediate_target, _ = v5._challenge_values(
                true_pair["anchor_immediate"], anchor
            )
            _, current_failed_before = v5._challenge_values(
                true_pair["current_before"], current
            )
            useful = -0.5 * torch.log(current_target.clamp_min(PROBABILITY_EPSILON))
            useful = useful - 0.5 * torch.log(anchor_target.clamp_min(PROBABILITY_EPSILON))
            useful = useful + 0.25 * v5._margin_loss(true_pair["current_after"], current)
            useful = useful + 0.25 * v5._margin_loss(true_pair["anchor_retained"], anchor)
            anti_repeat = 0.5 * F.relu(current_failed - current_failed_before + 0.05)
            retention = F.relu(
                anchor_immediate_target.detach() - anchor_target - RETENTION_TOLERANCE
            )
            deranged_causal = _causal_branch_loss(
                true_pair, deranged_pair, anchor, current
            )
            zeroed_causal = _causal_branch_loss(true_pair, zeroed_pair, anchor, current)
            causal = 0.5 * deranged_causal + 0.5 * zeroed_causal
            state = true_pair["state"]
            state_penalty = 1.0e-4 * (
                state.fast_weight.square().mean() + state.fast_bias.square()
            )
            loss = useful + anti_repeat + retention + causal + state_penalty
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite V6 meta-training loss")
            losses.append(loss)
            causal_values.append(float(causal.detach().item()))
            useful_values.append(float(useful.detach().item()))
        meta_loss = torch.stack(losses).mean()
        meta_loss.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
            raise RuntimeError("missing or non-finite V6 gradients")
        outcome_gradient = core.outcome_embedding.weight.grad
        modulator_gradient = core.neuromodulator[-1].weight.grad
        if (
            outcome_gradient is None
            or modulator_gradient is None
            or float(outcome_gradient.abs().sum().item()) <= 0.0
            or float(modulator_gradient.abs().sum().item()) <= 0.0
        ):
            raise RuntimeError("V6 causal loss did not reach outcome and neuromodulator parameters")
        gradient_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all().item()) for parameter in core.parameters()):
            raise RuntimeError("non-finite V6 parameter after optimizer step")
        diagnostics.append(
            {
                "optimizer_step": float(start // META_BATCH_PAIRS + 1),
                "loss": float(meta_loss.detach().item()),
                "useful_loss": sum(useful_values) / len(useful_values),
                "causal_loss": sum(causal_values) / len(causal_values),
                "gradient_norm": float(gradient_norm.detach().item()),
                "outcome_gradient_l1": float(outcome_gradient.abs().sum().item()),
                "neuromodulator_gradient_l1": float(modulator_gradient.abs().sum().item()),
            }
        )
    return diagnostics


def _evaluate_sequence(
    core: CausalNeuromodulatedApprenticeshipCore,
    rows: Sequence[v5.EncodedMechanism],
    *,
    mode: str,
) -> dict[str, Any]:
    core.eval()
    persistent_state = core.initial_plastic_state(detach=True)
    immediate_probabilities: list[float] = []
    before_probabilities: list[float] = []
    before_failed_probabilities: list[float] = []
    after_failed_probabilities: list[float] = []
    predictions: list[int] = []
    before_predictions: list[int] = []
    per_mechanism: list[dict[str, Any]] = []
    local_states: list[Any] = []
    with torch.no_grad():
        for row in rows:
            state = (
                core.initial_plastic_state(detach=True)
                if mode == "state_reset"
                else persistent_state
            )
            use_moving_origin = mode != "moving_origin_removed"
            before = v5._challenge(core, row, state, use_moving_origin=use_moving_origin)
            before_target, before_failed = v5._challenge_values(before, row)
            before_prediction = int(before.weights.argmax(dim=-1).item())
            if mode not in ("memory_disabled", "fair_semantic_retrieval", "state_zero"):
                outcomes = None
                if mode == "outcome_shuffled":
                    outcomes = _counterfactual_outcomes(row, mode)
                elif mode == "outcome_zeroed":
                    outcomes = (0,) * 6
                state = v5._apply_stream(
                    core,
                    row,
                    state,
                    outcome_values=outcomes,
                    use_moving_origin=use_moving_origin,
                    detach_state=True,
                )
            after = v5._challenge(core, row, state, use_moving_origin=use_moving_origin)
            if mode == "fair_semantic_retrieval":
                weights = torch.softmax(after.normalized_base_scores, dim=-1)
                prediction = int(weights.argmax(dim=-1).item())
                target_probability = weights[0, row.target_index]
                failed_after = weights[0, list(row.failed_indices)].sum()
            else:
                prediction = int(after.weights.argmax(dim=-1).item())
                target_probability, failed_after = v5._challenge_values(after, row)
            if mode != "state_reset":
                persistent_state = state
            local_states.append(state)
            predictions.append(prediction)
            before_predictions.append(before_prediction)
            immediate_probabilities.append(float(target_probability.item()))
            before_probabilities.append(float(before_target.item()))
            before_failed_probabilities.append(float(before_failed.item()))
            after_failed_probabilities.append(float(failed_after.item()))
            per_mechanism.append(
                {
                    "mechanism_ref": row.mechanism_ref,
                    "generator_family": row.generator_family,
                    "target_index": row.target_index,
                    "prediction": prediction,
                    "success": prediction == row.target_index,
                    "target_probability_before": float(before_target.item()),
                    "target_probability_after": float(target_probability.item()),
                    "failed_probability_before": float(before_failed.item()),
                    "failed_probability_after": float(failed_after.item()),
                    "state_norm": float(state.fast_weight.norm().item()),
                }
            )
        retained_probabilities: list[float] = []
        retained_successes = 0
        for index, row in enumerate(rows):
            retained_state = local_states[index] if mode == "state_reset" else persistent_state
            output = v5._challenge(
                core,
                row,
                retained_state,
                use_moving_origin=mode != "moving_origin_removed",
            )
            target, _ = v5._challenge_values(output, row)
            retained_probabilities.append(float(target.item()))
            retained_successes += int(output.weights.argmax(dim=-1).item() == row.target_index)
    successes = sum(
        int(prediction == row.target_index) for prediction, row in zip(predictions, rows)
    )
    return {
        "mode": mode,
        "successes": successes,
        "accuracy": successes / len(rows),
        "mean_target_probability_before": sum(before_probabilities) / len(rows),
        "mean_target_probability_after": sum(immediate_probabilities) / len(rows),
        "mean_failed_probability_before": sum(before_failed_probabilities) / len(rows),
        "mean_failed_probability_after": sum(after_failed_probabilities) / len(rows),
        "failed_to_pass_repairs": sum(
            int(before_prediction != row.target_index and prediction == row.target_index)
            for before_prediction, prediction, row in zip(before_predictions, predictions, rows)
        ),
        "sequential_retained_successes": retained_successes,
        "sequential_target_probability_retention": min(
            1.0,
            sum(retained_probabilities)
            / max(sum(immediate_probabilities), PROBABILITY_EPSILON),
        ),
        "per_mechanism": per_mechanism,
    }


def _evaluate_all(
    core: CausalNeuromodulatedApprenticeshipCore,
    rows: Sequence[v5.EncodedMechanism],
) -> dict[str, Any]:
    modes = (
        "full",
        "state_reset",
        "outcome_shuffled",
        "outcome_zeroed",
        "memory_disabled",
        "fair_semantic_retrieval",
        "moving_origin_removed",
        "state_zero",
    )
    arms = {mode: _evaluate_sequence(core, rows, mode=mode) for mode in modes}
    equivalent, unrelated = v5._evaluate_state_swaps(core, rows)
    arms["equivalent_state_swap"] = equivalent
    arms["unrelated_state_swap"] = unrelated
    successes = {name: int(value["successes"]) for name, value in arms.items()}
    return {
        "mechanism_count": len(rows),
        "successes": successes,
        "accuracies": {name: float(value["accuracy"]) for name, value in arms.items()},
        "equivalent_state_target_probability_retention": float(
            equivalent["target_probability_retention"]
        ),
        "full_failed_to_pass_repairs": int(arms["full"]["failed_to_pass_repairs"]),
        "full_negative_feedback_reduced_failed_probability": (
            arms["full"]["mean_failed_probability_after"]
            < arms["full"]["mean_failed_probability_before"]
        ),
        "full_target_probability_non_decreasing": (
            arms["full"]["mean_target_probability_after"]
            >= arms["full"]["mean_target_probability_before"]
        ),
        "sequential_target_probability_retention": float(
            arms["full"]["sequential_target_probability_retention"]
        ),
        "arms": arms,
    }


def _augment_causal_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    arms = metrics["arms"]
    full = arms["full"]
    target_deltas = {
        mode: float(
            full["mean_target_probability_after"]
            - arms[mode]["mean_target_probability_after"]
        )
        for mode in ("outcome_shuffled", "outcome_zeroed")
    }
    failure_deltas = {
        mode: float(
            arms[mode]["mean_failed_probability_after"]
            - full["mean_failed_probability_after"]
        )
        for mode in ("outcome_shuffled", "outcome_zeroed")
    }
    paired = []
    for full_row, deranged_row, zeroed_row in zip(
        full["per_mechanism"],
        arms["outcome_shuffled"]["per_mechanism"],
        arms["outcome_zeroed"]["per_mechanism"],
    ):
        if not (
            full_row["mechanism_ref"]
            == deranged_row["mechanism_ref"]
            == zeroed_row["mechanism_ref"]
        ):
            raise RuntimeError("V6 causal arms are not mechanism-aligned")
        paired.append(
            {
                "mechanism_ref": full_row["mechanism_ref"],
                "full_minus_deranged_target_probability": (
                    full_row["target_probability_after"]
                    - deranged_row["target_probability_after"]
                ),
                "full_minus_zeroed_target_probability": (
                    full_row["target_probability_after"]
                    - zeroed_row["target_probability_after"]
                ),
                "deranged_minus_full_failed_probability": (
                    deranged_row["failed_probability_after"]
                    - full_row["failed_probability_after"]
                ),
                "zeroed_minus_full_failed_probability": (
                    zeroed_row["failed_probability_after"]
                    - full_row["failed_probability_after"]
                ),
            }
        )
    metrics["outcome_causality"] = {
        "target_probability_advantage": target_deltas,
        "failed_probability_reduction": failure_deltas,
        "minimum_target_probability_advantage": min(target_deltas.values()),
        "minimum_failed_probability_reduction": min(failure_deltas.values()),
        "paired_mechanism_deltas": paired,
    }
    return metrics


def _development_gate(metrics: dict[str, Any], identities_exact: bool = True) -> bool:
    try:
        successes = metrics["successes"]
        full = successes["full"]
        causality = metrics["outcome_causality"]
        return bool(
            identities_exact
            and type(metrics["mechanism_count"]) is int
            and metrics["mechanism_count"] == 8
            and type(full) is int
            and full >= 6
            and all(type(successes[name]) is int and full - successes[name] >= 2 for name in REQUIRED_CONTROLS)
            and metrics["full_failed_to_pass_repairs"] >= 1
            and metrics["full_negative_feedback_reduced_failed_probability"]
            and metrics["full_target_probability_non_decreasing"]
            and successes["equivalent_state_swap"] >= full
            and metrics["equivalent_state_target_probability_retention"] >= 0.90
            and metrics["sequential_target_probability_retention"] >= 0.90
            and causality["minimum_target_probability_advantage"] >= CAUSAL_GATE_MARGIN
            and causality["minimum_failed_probability_reduction"] >= CAUSAL_GATE_MARGIN
        )
    except (KeyError, TypeError, ValueError):
        return False


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "runner": Path(__file__).resolve(),
        "core": root / "src/angler/reasoning/causal_neuromodulated_apprenticeship.py",
        "corpus": root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py",
    }
    return {name: v5._sha256(path) for name, path in paths.items()}


def _checkpoint_payload(
    core: CausalNeuromodulatedApprenticeshipCore,
    *,
    content_width: int,
    qwen_digest: str,
    development_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "content_width": content_width,
        "temporal_width": TEMPORAL_WIDTH,
        "rank": RANK,
        "qwen_digest": qwen_digest,
        "core_digest": v5._module_digest(core),
        "source_hashes": _source_hashes(),
        "development_metrics": development_metrics,
        "core_state_dict": {
            name: tensor.detach().cpu() for name, tensor in core.state_dict().items()
        },
    }


def _load_v5_initialization(
    core: CausalNeuromodulatedApprenticeshipCore,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != v5.IDENTITY or checkpoint.get("corpus_id") != v5.CORPUS_ID:
        raise RuntimeError("V5 donor checkpoint identity mismatch")
    donor = checkpoint["core_state_dict"]
    incompatible = core.load_state_dict(donor, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("V5 donor contains unexpected tensors")
    if not incompatible.missing_keys or any(
        not name.startswith("neuromodulator.") for name in incompatible.missing_keys
    ):
        raise RuntimeError("only fresh V6 neuromodulator tensors may be missing from V5")
    loaded = core.state_dict()
    if any(not torch.equal(loaded[name].cpu(), value) for name, value in donor.items()):
        raise RuntimeError("an inherited V5 tensor was not loaded byte-exactly")
    return {
        "path": str(checkpoint_path),
        "sha256": v5._sha256(checkpoint_path),
        "core_digest": checkpoint["core_digest"],
        "missing_fresh_keys": list(incompatible.missing_keys),
    }


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V6 train-development identity is already consumed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_causal_neuromodulated_apprenticeship_v6()
    train_cpu = _encode_partition(qwen, corpus.train)
    development_cpu = _encode_partition(qwen, corpus.development)
    content_width = int(train_cpu[0].episode_queries.shape[-1])
    train_rows = tuple(row.to(device) for row in train_cpu)
    development_rows = tuple(row.to(device) for row in development_cpu)
    del train_cpu, development_cpu
    core = CausalNeuromodulatedApprenticeshipCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        rank=RANK,
    ).to(device=device, dtype=torch.float32)
    donor_initialization = _load_v5_initialization(core, Path(args.v5_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v5._module_digest(core)
    development_metrics = _augment_causal_metrics(_evaluate_all(core, development_rows))
    core_after = v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    identities_exact = core_before == core_after and foundation_before == foundation_after
    authorized = _development_gate(development_metrics, identities_exact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            core,
            content_width=content_width,
            qwen_digest=foundation_before,
            development_metrics=development_metrics,
        ),
        checkpoint_path,
    )
    result = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": "DEVELOPMENT_GATE_PASSED" if authorized else "DEVELOPMENT_NOT_SUPPORTED",
        "development_authorized": authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "training_mechanisms": 64,
            "development_mechanisms": 8,
            "final_mechanisms_opened": 0,
            "meta_pairs": 256,
            "optimizer_steps": 64,
            "counterfactual_streams_per_pair": 2,
            "learning_rate": LEARNING_RATE,
            "rank": RANK,
        },
        "training_diagnostics": training,
        "development_metrics": development_metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_before,
            "core_after_evaluation": core_after,
            "exact": identities_exact,
        },
        "source_hashes": _source_hashes(),
        "v5_donor_initialization": donor_initialization,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": v5._sha256(checkpoint_path),
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Development tests outcome-causal adaptation across generator-family-disjoint synthetic software "
            "mechanisms. It is not final evidence, arbitrary reasoning, AGI, or deployment authorization."
        ),
    }
    v5._write_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "development_result": str(result_path),
        "checkpoint": str(checkpoint_path),
        "full_successes": development_metrics["successes"]["full"],
        "causal_target_margin": development_metrics["outcome_causality"]["minimum_target_probability_advantage"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V6 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    v5._validate_final_authorization(development.get("development_authorized") is True)
    if development.get("checkpoint_sha256") != v5._sha256(checkpoint_path):
        raise RuntimeError("V6 checkpoint changed after development authorization")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != IDENTITY or checkpoint.get("source_hashes") != _source_hashes():
        raise RuntimeError("V6 checkpoint identity or frozen source changed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]:
        raise RuntimeError("frozen Qwen identity changed")
    corpus = build_causal_neuromodulated_apprenticeship_v6(include_final=True)
    final_cpu = _encode_partition(qwen, corpus.final)
    final_rows = tuple(row.to(device) for row in final_cpu)
    core = CausalNeuromodulatedApprenticeshipCore(
        content_width=int(checkpoint["content_width"]),
        temporal_width=int(checkpoint["temporal_width"]),
        rank=int(checkpoint["rank"]),
    ).to(device=device, dtype=torch.float32)
    core.load_state_dict(checkpoint["core_state_dict"], strict=True)
    if v5._module_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V6 trained core digest mismatch")
    core_before = v5._module_digest(core)
    final_metrics = _augment_causal_metrics(_evaluate_all(core, final_rows))
    core_after = v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    identities_exact = core_before == core_after and foundation_before == foundation_after
    supported = _development_gate(final_metrics, identities_exact)
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "development_result_sha256": v5._sha256(development_path),
        "checkpoint_sha256": v5._sha256(checkpoint_path),
        "final_metrics": final_metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_before,
            "core_after_evaluation": core_after,
            "exact": identities_exact,
        },
        "source_hashes": _source_hashes(),
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Supported would establish bounded outcome-causal synthetic procedural apprenticeship only, not "
            "arbitrary-domain intelligence, consciousness, AGI, deployment safety, or production readiness."
        ),
    }
    v5._write_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "final_result": str(result_path),
        "full_successes": final_metrics["successes"]["full"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--v5-checkpoint",
        default="/opt/angler/results/outcome-aware-apprenticeship-v5.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/causal-neuromodulated-apprenticeship-v6.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/causal-neuromodulated-apprenticeship-v6-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/causal-neuromodulated-apprenticeship-v6-final.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
