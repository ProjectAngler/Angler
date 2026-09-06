"""Frozen OML evaluation for mixed-outcome procedural apprenticeship V5.

The runner has two explicit phases. ``train-dev`` may train on the 32 declared
training mechanisms and evaluate the eight development mechanisms. ``final``
refuses to open the final partition unless the preserved development result
passes the frozen gate. Qwen is a detached encoder only; all learned adaptation
lives in :class:`OutcomeAwareApprenticeshipCore`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.reasoning import (
    OutcomeAwareApprenticeshipCore,
    OutcomeAwarePlasticState,
)
from angler.runtime import LocalQwenIO, foundation_tensor_digest, freeze_knowledge_model
from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    CORPUS_ID,
    CorpusMechanism,
    build_outcome_aware_apprenticeship_v5,
)


IDENTITY = "angler.outcome-aware-apprenticeship.v5-first-result"
SEED = 20260831
TEMPORAL_WIDTH = 8
RANK = 32
LEARNING_RATE = 3.0e-4
META_BATCH_PAIRS = 4
GRADIENT_CLIP = 5.0
RETENTION_TOLERANCE = 0.05
PROBABILITY_EPSILON = 1.0e-8
REQUIRED_ADVANTAGE_SUCCESSES = 2
REQUIRED_CONTROLS = (
    "state_reset",
    "outcome_shuffled",
    "memory_disabled",
    "fair_semantic_retrieval",
    "moving_origin_removed",
    "state_zero",
)


@dataclass(frozen=True, slots=True)
class EncodedMechanism:
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

    def to(self, device: torch.device) -> "EncodedMechanism":
        return replace(
            self,
            episode_queries=self.episode_queries.to(device=device, dtype=torch.float32),
            episode_candidates=self.episode_candidates.to(device=device, dtype=torch.float32),
            episode_temporal=self.episode_temporal.to(device=device, dtype=torch.float32),
            episode_outcomes=self.episode_outcomes.to(device=device, dtype=torch.float32),
            challenge_query=self.challenge_query.to(device=device, dtype=torch.float32),
            challenge_candidates=self.challenge_candidates.to(device=device, dtype=torch.float32),
            challenge_temporal=self.challenge_temporal.to(device=device, dtype=torch.float32),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256(b"project-angler.module-state.v1\x00")
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(str(tensor.dtype).encode("ascii") + b"\x00")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\x00")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _pair_schedule() -> tuple[tuple[int, int], ...]:
    """Return the frozen balanced 128-pair OML meta-training schedule."""

    return tuple(
        (anchor, (anchor + offset) % 32)
        for offset in (1, 7, 13, 19)
        for anchor in range(32)
    )


def _deranged_outcomes(outcomes: Sequence[int | float]) -> tuple[int, ...]:
    """Return the frozen opposite-label permutation for a balanced stream."""

    values = tuple(int(value) for value in outcomes)
    if (
        len(values) != 6
        or values.count(1) != 3
        or values.count(-1) != 3
        or len(set(values[:3])) == 1
    ):
        raise ValueError("outcome derangement requires a balanced interleaved six-label stream")
    return tuple(-value for value in values)


def _development_gate(metrics: dict[str, object]) -> bool:
    """Apply the frozen eight-mechanism development boundary defensively."""

    try:
        count = metrics["mechanism_count"]
        successes = metrics["successes"]
        retention = metrics["equivalent_state_target_probability_retention"]
        if type(count) is not int or count != 8 or not isinstance(successes, dict):
            return False
        if not isinstance(retention, (int, float)) or isinstance(retention, bool):
            return False
        if not math.isfinite(float(retention)) or float(retention) < 0.90:
            return False
        required = (*REQUIRED_CONTROLS, "full", "equivalent_state_swap")
        if any(type(successes.get(name)) is not int for name in required):
            return False
        full = successes["full"]
        equivalent = successes["equivalent_state_swap"]
        if full < 6 or equivalent < full:
            return False
        if any(full - successes[name] < REQUIRED_ADVANTAGE_SUCCESSES for name in REQUIRED_CONTROLS):
            return False
        if equivalent - successes["state_reset"] < REQUIRED_ADVANTAGE_SUCCESSES:
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _validate_final_authorization(development_authorized: bool) -> None:
    if development_authorized is not True:
        raise RuntimeError("final partition remains sealed until the frozen development gate passes")


def _load_qwen(model_path: str) -> LocalQwenIO:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    freeze_knowledge_model(model)
    return LocalQwenIO(
        model,
        tokenizer,
        embedding_batch_size=16,
        max_new_tokens=32,
        enable_thinking=False,
    )


def _episode_query_text(mechanism: CorpusMechanism, index: int) -> str:
    episode = mechanism.public.episodes[index]
    return (
        f"Task: {episode.task_text}\nRequest: {episode.request_text}\n"
        f"Observed result: {episode.objective_diagnostic_text}"
    )


def _challenge_query_text(mechanism: CorpusMechanism) -> str:
    challenge = mechanism.public.challenge
    return f"Task: {challenge.task_text}\nRequest: {challenge.request_text}"


def _episode_temporal(index: int) -> tuple[float, ...]:
    age = 5 - index
    return (
        age / 5.0,
        index / 5.0,
        1.0 if index == 0 else 0.0,
        0.0 if index == 0 else 1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def _challenge_temporal(candidate_count: int) -> torch.Tensor:
    row = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    return torch.tensor([row] * candidate_count, dtype=torch.float32)


def _encode_partition(
    qwen: LocalQwenIO,
    mechanisms: Sequence[CorpusMechanism],
) -> tuple[EncodedMechanism, ...]:
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
        spans.append(
            (query_start, candidate_start, challenge_query_index, challenge_candidate_start)
        )
    encoded = qwen.embed(texts).to(dtype=torch.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(texts):
        raise RuntimeError("Qwen embedding output does not align with the frozen corpus")
    rows: list[EncodedMechanism] = []
    for mechanism, span in zip(mechanisms, spans):
        query_start, candidate_start, challenge_query_index, challenge_candidate_start = span
        outcomes = tuple(int(episode.outcome_value) for episode in mechanism.public.episodes)
        target_indices = mechanism.supervision.target_successful_candidate_indices
        if len(target_indices) != 1:
            raise RuntimeError("V5 requires one successful challenge candidate")
        candidate_count = len(mechanism.public.challenge.candidate_action_trace_texts)
        rows.append(
            EncodedMechanism(
                mechanism_ref=mechanism.metadata.mechanism_ref,
                generator_family=mechanism.metadata.generator_family,
                episode_queries=encoded[query_start : query_start + 6].clone(),
                episode_candidates=encoded[candidate_start : candidate_start + 6].clone(),
                episode_temporal=torch.tensor(
                    [_episode_temporal(index) for index in range(6)],
                    dtype=torch.float32,
                ),
                episode_outcomes=torch.tensor(outcomes, dtype=torch.float32),
                challenge_query=encoded[challenge_query_index].clone(),
                challenge_candidates=encoded[
                    challenge_candidate_start : challenge_candidate_start + candidate_count
                ].clone(),
                challenge_temporal=_challenge_temporal(candidate_count),
                target_index=target_indices[0],
                failed_indices=mechanism.supervision.relevant_failed_candidate_indices,
            )
        )
    return tuple(rows)


def _base_scores(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    query_normalized = F.normalize(query, dim=-1)
    candidate_normalized = F.normalize(candidates, dim=-1)
    return torch.einsum("bc,bnc->bn", query_normalized, candidate_normalized)


def _apply_stream(
    core: OutcomeAwareApprenticeshipCore,
    row: EncodedMechanism,
    state: OutcomeAwarePlasticState,
    *,
    outcome_values: Sequence[int | float] | None = None,
    use_moving_origin: bool = True,
    detach_state: bool,
) -> OutcomeAwarePlasticState:
    values = row.episode_outcomes if outcome_values is None else torch.tensor(
        tuple(outcome_values), device=row.episode_outcomes.device, dtype=torch.float32
    )
    if values.shape != (6,):
        raise ValueError("one V5 stream must expose exactly six outcomes")
    for index in range(6):
        query = row.episode_queries[index].unsqueeze(0)
        candidates = row.episode_candidates[index].reshape(1, 1, -1)
        temporal = row.episode_temporal[index].reshape(1, 1, -1)
        if not use_moving_origin:
            temporal = torch.zeros_like(temporal)
        outcomes = values[index].reshape(1, 1)
        mask = torch.ones((1, 1), device=query.device, dtype=torch.bool)
        output = core(
            query,
            candidates,
            temporal,
            outcomes,
            mask,
            _base_scores(query, candidates),
            plastic_state=state,
        )
        state = core.apply_mixed_outcome_update(
            state,
            output,
            outcomes,
            mask,
            detach_state=detach_state,
        )
    return state


def _challenge(
    core: OutcomeAwareApprenticeshipCore,
    row: EncodedMechanism,
    state: OutcomeAwarePlasticState,
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
        plastic_state=state,
    )


def _challenge_values(output: Any, row: EncodedMechanism) -> tuple[torch.Tensor, torch.Tensor]:
    target = output.weights[0, row.target_index]
    failed = output.weights[0, list(row.failed_indices)].sum()
    return target, failed


def _margin_loss(output: Any, row: EncodedMechanism, margin: float = 0.35) -> torch.Tensor:
    target = output.scores[0, row.target_index]
    failed = output.scores[0, list(row.failed_indices)].mean()
    return F.relu(target.new_tensor(margin) - target + failed)


def _train_core(
    core: OutcomeAwareApprenticeshipCore,
    rows: Sequence[EncodedMechanism],
) -> list[dict[str, float]]:
    if len(rows) != 32:
        raise ValueError("V5 training requires exactly 32 encoded mechanisms")
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
    for step in range(0, len(schedule), META_BATCH_PAIRS):
        optimizer.zero_grad(set_to_none=True)
        pair_losses: list[torch.Tensor] = []
        useful_values: list[float] = []
        anti_values: list[float] = []
        retention_values: list[float] = []
        for anchor_index, current_index in schedule[step : step + META_BATCH_PAIRS]:
            anchor = rows[anchor_index]
            current = rows[current_index]
            state = core.initial_plastic_state(detach=False)
            state = _apply_stream(core, anchor, state, detach_state=False)
            anchor_immediate = _challenge(core, anchor, state)
            anchor_immediate_target, _ = _challenge_values(anchor_immediate, anchor)
            current_before = _challenge(core, current, state)
            _, failed_before = _challenge_values(current_before, current)
            state = _apply_stream(core, current, state, detach_state=False)
            current_after = _challenge(core, current, state)
            anchor_retained = _challenge(core, anchor, state)
            current_target, failed_after = _challenge_values(current_after, current)
            anchor_target, _ = _challenge_values(anchor_retained, anchor)

            useful = -0.5 * torch.log(current_target.clamp_min(PROBABILITY_EPSILON))
            useful = useful - 0.5 * torch.log(anchor_target.clamp_min(PROBABILITY_EPSILON))
            margins = 0.25 * _margin_loss(current_after, current)
            margins = margins + 0.25 * _margin_loss(anchor_retained, anchor)
            anti_repeat = 0.5 * F.relu(failed_after - failed_before + 0.05)
            retention = F.relu(
                anchor_immediate_target.detach() - anchor_target - RETENTION_TOLERANCE
            )
            state_penalty = 1.0e-4 * (
                state.fast_weight.square().mean() + state.fast_bias.square()
            )
            loss = useful + margins + anti_repeat + retention + state_penalty
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite V5 meta-training loss")
            pair_losses.append(loss)
            useful_values.append(float(useful.detach().item()))
            anti_values.append(float(anti_repeat.detach().item()))
            retention_values.append(float(retention.detach().item()))
        meta_loss = torch.stack(pair_losses).mean()
        meta_loss.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
            raise RuntimeError("missing or non-finite V5 gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all().item()) for parameter in core.parameters()):
            raise RuntimeError("non-finite V5 parameter after optimizer step")
        diagnostics.append(
            {
                "optimizer_step": float(step // META_BATCH_PAIRS + 1),
                "loss": float(meta_loss.detach().item()),
                "useful_loss": sum(useful_values) / len(useful_values),
                "anti_repeat_loss": sum(anti_values) / len(anti_values),
                "retention_loss": sum(retention_values) / len(retention_values),
                "gradient_norm": float(gradient_norm.detach().item()),
            }
        )
    return diagnostics


def _evaluate_sequence(
    core: OutcomeAwareApprenticeshipCore,
    rows: Sequence[EncodedMechanism],
    *,
    mode: str,
) -> dict[str, Any]:
    core.eval()
    state = core.initial_plastic_state(detach=True)
    immediate_probabilities: list[float] = []
    before_target_probabilities: list[float] = []
    before_failed_probabilities: list[float] = []
    after_failed_probabilities: list[float] = []
    before_predictions: list[int] = []
    predictions: list[int] = []
    per_mechanism: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            before = _challenge(core, row, state, use_moving_origin=mode != "moving_origin_removed")
            before_target, before_failed = _challenge_values(before, row)
            before_predictions.append(int(before.weights.argmax(dim=-1).item()))
            if mode not in ("state_reset", "memory_disabled", "fair_semantic_retrieval", "state_zero"):
                if mode == "outcome_shuffled":
                    outcomes = _deranged_outcomes(tuple(int(value) for value in row.episode_outcomes.tolist()))
                elif mode == "outcome_zeroed":
                    outcomes = (0,) * 6
                else:
                    outcomes = None
                state = _apply_stream(
                    core,
                    row,
                    state,
                    outcome_values=outcomes,
                    use_moving_origin=mode != "moving_origin_removed",
                    detach_state=True,
                )
            if mode in ("state_reset", "state_zero"):
                challenge_state = core.initial_plastic_state(detach=True)
            else:
                challenge_state = state
            after = _challenge(
                core,
                row,
                challenge_state,
                use_moving_origin=mode != "moving_origin_removed",
            )
            if mode == "fair_semantic_retrieval":
                weights = torch.softmax(after.normalized_base_scores, dim=-1)
                prediction = int(weights.argmax(dim=-1).item())
                target_probability = weights[0, row.target_index]
                failed_after = weights[0, list(row.failed_indices)].sum()
            else:
                prediction = int(after.weights.argmax(dim=-1).item())
                target_probability, failed_after = _challenge_values(after, row)
            predictions.append(prediction)
            immediate_probabilities.append(float(target_probability.item()))
            before_target_probabilities.append(float(before_target.item()))
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
                }
            )
        retained_probabilities: list[float] = []
        retained_successes = 0
        for row in rows:
            output = _challenge(
                core,
                row,
                state,
                use_moving_origin=mode != "moving_origin_removed",
            )
            target, _ = _challenge_values(output, row)
            retained_probabilities.append(float(target.item()))
            retained_successes += int(output.weights.argmax(dim=-1).item() == row.target_index)
    probability_denominator = max(sum(immediate_probabilities), PROBABILITY_EPSILON)
    return {
        "mode": mode,
        "successes": sum(
            int(prediction == row.target_index) for prediction, row in zip(predictions, rows)
        ),
        "accuracy": sum(
            int(prediction == row.target_index) for prediction, row in zip(predictions, rows)
        ) / len(rows),
        "mean_target_probability_before": sum(before_target_probabilities) / len(rows),
        "mean_target_probability_after": sum(immediate_probabilities) / len(rows),
        "mean_failed_probability_before": sum(before_failed_probabilities) / len(rows),
        "mean_failed_probability_after": sum(after_failed_probabilities) / len(rows),
        "failed_to_pass_repairs": sum(
            int(before_prediction != row.target_index and after_prediction == row.target_index)
            for before_prediction, after_prediction, row in zip(before_predictions, predictions, rows)
        ),
        "sequential_retained_successes": retained_successes,
        "sequential_target_probability_retention": min(
            1.0,
            sum(retained_probabilities) / probability_denominator,
        ),
        "per_mechanism": per_mechanism,
    }


def _individual_states(
    core: OutcomeAwareApprenticeshipCore,
    rows: Sequence[EncodedMechanism],
) -> tuple[OutcomeAwarePlasticState, ...]:
    values = []
    with torch.no_grad():
        for row in rows:
            state = core.initial_plastic_state(detach=True)
            values.append(_apply_stream(core, row, state, detach_state=True))
    return tuple(values)


def _evaluate_state_swaps(
    core: OutcomeAwareApprenticeshipCore,
    rows: Sequence[EncodedMechanism],
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = _individual_states(core, rows)
    equivalent_predictions: list[int] = []
    unrelated_predictions: list[int] = []
    equivalent_probabilities: list[float] = []
    own_probabilities: list[float] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            equivalent_index = next(
                candidate
                for candidate in range(1, len(rows) + 1)
                if rows[(index + candidate) % len(rows)].generator_family == row.generator_family
            )
            equivalent_index = (index + equivalent_index) % len(rows)
            unrelated_index = next(
                candidate
                for candidate in range(len(rows))
                if rows[candidate].generator_family != row.generator_family
            )
            equivalent = _challenge(core, row, states[equivalent_index])
            unrelated = _challenge(core, row, states[unrelated_index])
            own = _challenge(core, row, states[index])
            equivalent_predictions.append(int(equivalent.weights.argmax(dim=-1).item()))
            unrelated_predictions.append(int(unrelated.weights.argmax(dim=-1).item()))
            equivalent_probabilities.append(float(equivalent.weights[0, row.target_index].item()))
            own_probabilities.append(float(own.weights[0, row.target_index].item()))
    denominator = max(sum(own_probabilities), PROBABILITY_EPSILON)
    equivalent_result = {
        "mode": "equivalent_state_swap",
        "successes": sum(
            int(prediction == row.target_index)
            for prediction, row in zip(equivalent_predictions, rows)
        ),
        "accuracy": sum(
            int(prediction == row.target_index)
            for prediction, row in zip(equivalent_predictions, rows)
        ) / len(rows),
        "target_probability_retention": min(1.0, sum(equivalent_probabilities) / denominator),
    }
    unrelated_result = {
        "mode": "unrelated_state_swap",
        "successes": sum(
            int(prediction == row.target_index)
            for prediction, row in zip(unrelated_predictions, rows)
        ),
        "accuracy": sum(
            int(prediction == row.target_index)
            for prediction, row in zip(unrelated_predictions, rows)
        ) / len(rows),
    }
    return equivalent_result, unrelated_result


def _evaluate_all(
    core: OutcomeAwareApprenticeshipCore,
    rows: Sequence[EncodedMechanism],
) -> dict[str, Any]:
    modes = (
        "full",
        "state_reset",
        "outcome_shuffled",
        "memory_disabled",
        "fair_semantic_retrieval",
        "moving_origin_removed",
        "state_zero",
        "outcome_zeroed",
    )
    arms = {mode: _evaluate_sequence(core, rows, mode=mode) for mode in modes}
    equivalent, unrelated = _evaluate_state_swaps(core, rows)
    arms["equivalent_state_swap"] = equivalent
    arms["unrelated_state_swap"] = unrelated
    successes = {name: int(value["successes"]) for name, value in arms.items()}
    return {
        "mechanism_count": len(rows),
        "successes": successes,
        "accuracies": {
            name: float(value["accuracy"]) for name, value in arms.items()
        },
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


def _final_supported(metrics: dict[str, Any], identities_exact: bool) -> bool:
    successes = metrics["successes"]
    full = successes["full"]
    return bool(
        identities_exact
        and metrics["mechanism_count"] == 8
        and full >= 6
        and all(full - successes[name] >= 2 for name in REQUIRED_CONTROLS)
        and metrics["full_failed_to_pass_repairs"] >= 1
        and metrics["full_negative_feedback_reduced_failed_probability"]
        and metrics["full_target_probability_non_decreasing"]
        and successes["equivalent_state_swap"] >= full
        and metrics["equivalent_state_target_probability_retention"] >= 0.90
        and metrics["sequential_target_probability_retention"] >= 0.90
    )


def _environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_payload(
    core: OutcomeAwareApprenticeshipCore,
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
        "core_digest": _module_digest(core),
        "development_metrics": development_metrics,
        "core_state_dict": {
            name: tensor.detach().cpu() for name, tensor in core.state_dict().items()
        },
    }


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V5 train-development identity is already consumed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = _load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_outcome_aware_apprenticeship_v5()
    # Only train and development tensors are materialized in this phase.
    train_rows_cpu = _encode_partition(qwen, corpus.train)
    development_rows_cpu = _encode_partition(qwen, corpus.development)
    content_width = int(train_rows_cpu[0].episode_queries.shape[-1])
    train_rows = tuple(row.to(device) for row in train_rows_cpu)
    development_rows = tuple(row.to(device) for row in development_rows_cpu)
    del train_rows_cpu, development_rows_cpu
    core = OutcomeAwareApprenticeshipCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        rank=RANK,
    ).to(device=device, dtype=torch.float32)
    training = _train_core(core, train_rows)
    core_digest_before_evaluation = _module_digest(core)
    development_metrics = _evaluate_all(core, development_rows)
    core_digest_after_evaluation = _module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    identities_exact = (
        foundation_before == foundation_after
        and core_digest_before_evaluation == core_digest_after_evaluation
    )
    development_authorized = _development_gate(development_metrics) and identities_exact
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
        "classification": "DEVELOPMENT_GATE_PASSED" if development_authorized else "DEVELOPMENT_NOT_SUPPORTED",
        "development_authorized": development_authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "training_mechanisms": 32,
            "development_mechanisms": 8,
            "final_mechanisms_opened": 0,
            "chronological_updates_per_pair": 12,
            "meta_pairs": 128,
            "optimizer_steps": 32,
            "learning_rate": LEARNING_RATE,
            "rank": RANK,
        },
        "training_diagnostics": training,
        "development_metrics": development_metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_digest_before_evaluation,
            "core_after_evaluation": core_digest_after_evaluation,
            "exact": identities_exact,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "environment": _environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angeler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "This gate tests transfer to generator-disjoint software mechanisms after mixed-outcome OML updates. "
            "It is not a final result, arbitrary-domain reasoning claim, AGI claim, or deployment authorization."
        ),
    }
    _write_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "development_result": str(result_path),
        "checkpoint": str(checkpoint_path),
        "full_successes": development_metrics["successes"]["full"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_result_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V5 final identity is already consumed")
    development_result = json.loads(development_result_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development_result.get("development_authorized") is True)
    if development_result.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise RuntimeError("V5 checkpoint no longer matches the development authorization")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != IDENTITY or checkpoint.get("corpus_id") != CORPUS_ID:
        raise RuntimeError("V5 checkpoint identity mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = _load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]:
        raise RuntimeError("frozen Qwen identity changed between development and final")
    corpus = build_outcome_aware_apprenticeship_v5()
    # This is the first operation in either phase that materializes final embeddings.
    final_rows_cpu = _encode_partition(qwen, corpus.final)
    final_rows = tuple(row.to(device) for row in final_rows_cpu)
    content_width = int(checkpoint["content_width"])
    core = OutcomeAwareApprenticeshipCore(
        content_width=content_width,
        temporal_width=int(checkpoint["temporal_width"]),
        rank=int(checkpoint["rank"]),
    ).to(device=device, dtype=torch.float32)
    core.load_state_dict(checkpoint["core_state_dict"], strict=True)
    if _module_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("loaded V5 core digest does not match the authorized checkpoint")
    core_before = _module_digest(core)
    final_metrics = _evaluate_all(core, final_rows)
    core_after = _module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    identities_exact = core_before == core_after and foundation_before == foundation_after
    supported = _final_supported(final_metrics, identities_exact)
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "development_result": str(development_result_path),
        "development_result_sha256": _sha256(development_result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": final_metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_before,
            "core_after_evaluation": core_after,
            "exact": identities_exact,
        },
        "environment": _environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angeler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "A supported result demonstrates bounded outcome-aware procedural adaptation across fresh synthetic "
            "software mechanisms and frozen removal controls. It does not establish arbitrary-domain intelligence, "
            "consciousness, AGI, autonomous deployment, or general project competence."
        ),
    }
    _write_json(result_path, result)
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
        "--checkpoint",
        default="/opt/angler/results/outcome-aware-apprenticeship-v5.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/outcome-aware-apprenticeship-v5-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/outcome-aware-apprenticeship-v5-final.json",
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
