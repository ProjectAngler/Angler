"""Meta-train and test Angler's persistent update dynamics.

This isolating experiment uses a structured public-observation connector rather
than a language model so it measures the reasoning/plasticity path directly.
Meta-training targets teach only the slow update dynamics.  At final evaluation
all module parameters are frozen; unique support encounters expose only the
attempted ordering and one scalar outcome, and only SRWM offsets change.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from angler.reasoning import (  # noqa: E402
    AdaptiveReasoningCore,
    ReasoningCoreConfig,
    SelfReferentialState,
    detach_self_referential_state,
    reasoning_state_digest,
    self_referential_state_digest,
)
from angler.worlds.latent_order_programs import (  # noqa: E402
    ITEM_COUNT as LATENT_ORDER_ITEM_COUNT,
    GeneratedLatentOrderingTask,
    LatentOrderingTask,
    OrderingProgram,
    TRAIN_PROGRAMS,
    generate_latent_ordering_task,
    score_latent_ordering_answer,
)


_KNOWLEDGE_WIDTH = 24
_EXPERIMENT_ID = "ANGLER-PREQUENTIAL-SRWM-V1"


@dataclass(frozen=True, slots=True)
class RunSettings:
    outer_steps: int
    batch_size: int
    minimum_supports: int
    maximum_supports: int
    reentry_supports: int
    reacquisition_supports: int
    queries_per_stage: int
    core_width: int
    workspace_slots: int
    attention_heads: int
    feedforward_width: int
    reasoning_steps: int
    learning_rate: float
    gradient_clip: float
    evaluation_supports: int


_PROFILES = {
    "smoke": RunSettings(
        outer_steps=8,
        batch_size=2,
        minimum_supports=3,
        maximum_supports=4,
        reentry_supports=1,
        reacquisition_supports=4,
        queries_per_stage=2,
        core_width=32,
        workspace_slots=4,
        attention_heads=4,
        feedforward_width=96,
        reasoning_steps=2,
        learning_rate=8e-4,
        gradient_clip=1.0,
        evaluation_supports=8,
    ),
    "full": RunSettings(
        outer_steps=256,
        batch_size=4,
        minimum_supports=5,
        maximum_supports=8,
        reentry_supports=2,
        reacquisition_supports=16,
        queries_per_stage=4,
        core_width=64,
        workspace_slots=8,
        attention_heads=4,
        feedforward_width=256,
        reasoning_steps=3,
        learning_rate=5e-4,
        gradient_clip=1.0,
        evaluation_supports=64,
    ),
}


class _UniqueSeedStream:
    def __init__(self, root_seed: int) -> None:
        self._root_seed = root_seed
        self._counter = 0
        self.instance_ids: set[str] = set()

    def take(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("seed count must be positive")
        result = tuple(
            self._root_seed * 1_000_000 + self._counter + offset
            for offset in range(count)
        )
        self._counter += count
        return result

    def record(self, tasks: Iterable[GeneratedLatentOrderingTask]) -> None:
        for generated in tasks:
            instance_id = generated.learner.instance_id
            if instance_id in self.instance_ids:
                raise RuntimeError("a public instance was presented more than once")
            self.instance_ids.add(instance_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(_PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--result-json",
        default="/opt/angler/project/work/prequential-srwm-result.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/project/work/prequential-srwm-slow.safetensors",
    )
    return parser.parse_args()


def _make_model(settings: RunSettings, device: torch.device) -> AdaptiveReasoningCore:
    config = ReasoningCoreConfig(
        knowledge_width=_KNOWLEDGE_WIDTH,
        core_width=settings.core_width,
        workspace_slots=settings.workspace_slots,
        attention_heads=settings.attention_heads,
        feedforward_width=settings.feedforward_width,
        reasoning_steps=settings.reasoning_steps,
        maximum_reasoning_steps=max(settings.reasoning_steps, 4),
        maximum_entities=LATENT_ORDER_ITEM_COUNT,
    )
    return AdaptiveReasoningCore(config).to(device=device, dtype=torch.float32)


def _generate_batch(
    programs: Sequence[OrderingProgram],
    seeds: Sequence[int],
    stream: _UniqueSeedStream,
    *,
    public_flags: Sequence[bool | None] | None = None,
) -> tuple[GeneratedLatentOrderingTask, ...]:
    if len(programs) != len(seeds):
        raise ValueError("program and seed batches must match")
    if public_flags is None:
        public_flags = (None,) * len(programs)
    if len(public_flags) != len(programs):
        raise ValueError("public flag and program batches must match")
    generated = tuple(
        generate_latent_ordering_task(
            program,
            seed,
            public_flag=public_flag,
        )
        for program, seed, public_flag in zip(
            programs,
            seeds,
            public_flags,
            strict=True,
        )
    )
    stream.record(generated)
    return generated


def _balanced_flags(count: int) -> tuple[bool, ...]:
    if count <= 0 or count % 2:
        raise ValueError("balanced evaluation counts must be positive and even")
    return tuple(bool(index % 2) for index in range(count))


def _structured_inputs(
    generated: Sequence[GeneratedLatentOrderingTask],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(generated)
    features = torch.zeros(
        batch_size,
        LATENT_ORDER_ITEM_COUNT,
        _KNOWLEDGE_WIDTH,
        device=device,
        dtype=torch.float32,
    )
    for batch_index, task_pair in enumerate(generated):
        task = task_pair.learner
        for display_index, item in enumerate(task.items):
            vector = features[batch_index, display_index]
            vector[item.rank_a] = 1.0
            vector[5 + item.rank_b] = 1.0
            vector[10 + item.group] = 1.0
            vector[12 + int(item.marked)] = 1.0
            vector[14 + int(task.public_flag)] = 1.0
            vector[16 + display_index] = 1.0
            vector[21] = 1.0

    fact_features = features.clone()
    fact_features[:, :, 22] = 1.0
    mention_features = features.clone()
    mention_features[:, :, 23] = 1.0
    mask = torch.ones(
        batch_size,
        LATENT_ORDER_ITEM_COUNT,
        device=device,
        dtype=torch.bool,
    )
    incidence = torch.arange(
        LATENT_ORDER_ITEM_COUNT,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    return {
        "fact_features": fact_features,
        "fact_mask": mask,
        "entity_features": features,
        "entity_mask": mask,
        "mention_features": mention_features,
        "mention_mask": mask,
        "mention_fact_indices": incidence,
        "mention_entity_indices": incidence,
    }


def _target_indices(
    generated: Sequence[GeneratedLatentOrderingTask],
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[int]] = []
    for task_pair in generated:
        display_position = {
            symbol: index
            for index, symbol in enumerate(task_pair.learner.symbols)
        }
        rows.append(
            [display_position[symbol] for symbol in task_pair.hidden.target_order]
        )
    return torch.tensor(rows, device=device, dtype=torch.long)


def _answers_from_indices(
    tasks: Sequence[LatentOrderingTask],
    order_indices: torch.Tensor,
) -> tuple[tuple[str, ...], ...]:
    cpu_orders = order_indices.detach().cpu().tolist()
    return tuple(
        tuple(task.symbols[index] for index in order)
        for task, order in zip(tasks, cpu_orders, strict=True)
    )


def _support_step(
    model: AdaptiveReasoningCore,
    state: SelfReferentialState,
    generated: Sequence[GeneratedLatentOrderingTask],
    device: torch.device,
    *,
    reward_transform: str = "ordinary",
) -> tuple[SelfReferentialState, tuple[float, ...]]:
    inputs = _structured_inputs(generated, device)
    trajectory = model.act(
        **inputs,
        state=state,
        greedy=False,
        temperature=1.25,
    )
    answers = _answers_from_indices(
        tuple(pair.learner for pair in generated),
        trajectory.action.order_indices[:, 0],
    )
    rewards = tuple(
        score_latent_ordering_answer(pair.learner, pair.hidden, answer).pairwise_accuracy
        for pair, answer in zip(generated, answers, strict=True)
    )
    if reward_transform == "inverted":
        presented_rewards = tuple(1.0 - reward for reward in rewards)
    elif reward_transform == "ordinary":
        presented_rewards = rewards
    else:
        raise ValueError("unknown reward transform")
    write = model.incorporate_feedback(
        trajectory.feedback_context,
        reward=torch.tensor(
            presented_rewards,
            device=device,
            dtype=torch.float32,
        ),
        state=state,
    )
    return write.state, rewards


def _training_query_loss(
    model: AdaptiveReasoningCore,
    state: SelfReferentialState,
    generated: Sequence[GeneratedLatentOrderingTask],
    device: torch.device,
) -> torch.Tensor:
    inputs = _structured_inputs(generated, device)
    scored = model.score_training_order(
        **inputs,
        prescribed_order=_target_indices(generated, device),
        state=state,
    )
    return -scored.log_probability.mean() / LATENT_ORDER_ITEM_COUNT


def _training_stage_loss(
    model: AdaptiveReasoningCore,
    state: SelfReferentialState,
    programs: Sequence[OrderingProgram],
    settings: RunSettings,
    device: torch.device,
    stream: _UniqueSeedStream,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for _ in range(settings.queries_per_stage):
        generated = _generate_batch(
            programs,
            stream.take(settings.batch_size),
            stream,
        )
        losses.append(_training_query_loss(model, state, generated, device))
    return torch.stack(losses).mean()


def _train(
    model: AdaptiveReasoningCore,
    settings: RunSettings,
    seed: int,
    device: torch.device,
    stream: _UniqueSeedStream,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=1e-4,
    )
    rng = random.Random(seed)
    losses: list[float] = []
    gradient_norms: list[float] = []
    model.train()

    for _ in range(settings.outer_steps):
        program_a: list[OrderingProgram] = []
        program_b: list[OrderingProgram] = []
        for _batch in range(settings.batch_size):
            first, second = rng.sample(TRAIN_PROGRAMS, 2)
            program_a.append(first)
            program_b.append(second)
        support_count_a = rng.randint(
            settings.minimum_supports,
            settings.maximum_supports,
        )
        support_count_b = rng.randint(
            settings.minimum_supports,
            settings.maximum_supports,
        )
        state = model.initial_state(settings.batch_size)

        for _support in range(support_count_a):
            tasks = _generate_batch(
                program_a,
                stream.take(settings.batch_size),
                stream,
            )
            state, _ = _support_step(model, state, tasks, device)
        loss_a = _training_stage_loss(
            model,
            state,
            program_a,
            settings,
            device,
            stream,
        )

        for _support in range(support_count_b):
            tasks = _generate_batch(
                program_b,
                stream.take(settings.batch_size),
                stream,
            )
            state, _ = _support_step(model, state, tasks, device)
        loss_b = _training_stage_loss(
            model,
            state,
            program_b,
            settings,
            device,
            stream,
        )

        for _support in range(settings.reentry_supports):
            tasks = _generate_batch(
                program_a,
                stream.take(settings.batch_size),
                stream,
            )
            state, _ = _support_step(model, state, tasks, device)
        loss_reentry = _training_stage_loss(
            model,
            state,
            program_a,
            settings,
            device,
            stream,
        )

        loss = (loss_a + loss_b + loss_reentry) / 3.0
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("meta-training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            settings.gradient_clip,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("meta-training produced a non-finite gradient")
        optimizer.step()
        losses.append(float(loss.item()))
        gradient_norms.append(float(gradient_norm.item()))

    return {
        "outer_steps": settings.outer_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "loss_tail": losses[-min(16, len(losses)):],
    }


def _online_summary(
    scores: Sequence[float],
    flags: Sequence[bool],
) -> dict[str, object]:
    if not scores or len(scores) != len(flags):
        raise ValueError("online scores and flags must be nonempty and aligned")

    def summarize(values: Sequence[float]) -> dict[str, float | int]:
        window = max(1, len(values) // 4)
        early = sum(values[:window]) / window
        late = sum(values[-window:]) / window
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "early_mean": early,
            "late_mean": late,
            "gain": late - early,
        }

    by_flag: dict[str, dict[str, float | int]] = {}
    for flag in (False, True):
        flag_scores = [
            score
            for score, observed_flag in zip(scores, flags, strict=True)
            if observed_flag is flag
        ]
        if not flag_scores:
            raise RuntimeError("online sequence did not cover both public flags")
        by_flag[str(flag).lower()] = summarize(flag_scores)
    overall = summarize(scores)
    return {
        **overall,
        "by_public_flag": by_flag,
        "worst_flag_gain": min(
            float(record["gain"]) for record in by_flag.values()
        ),
        "trajectory": [
            {
                "step": index + 1,
                "public_flag": flag,
                "score_before_write": score,
            }
            for index, (score, flag) in enumerate(
                zip(scores, flags, strict=True)
            )
        ],
    }


def _flag_metric(
    summary: dict[str, object],
    flag: str,
    metric: str,
) -> float:
    by_flag = summary["by_public_flag"]
    return float(by_flag[flag][metric])


def _run_online_sequence(
    model: AdaptiveReasoningCore,
    state: SelfReferentialState,
    generated: Sequence[GeneratedLatentOrderingTask],
    device: torch.device,
    *,
    mode: str,
    sampling_seed: int,
) -> tuple[SelfReferentialState, dict[str, object]]:
    if not generated:
        raise ValueError("online sequence must contain at least one task")
    if mode not in ("ordinary", "no_write", "inverted"):
        raise ValueError("unknown online support mode")

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device() if device.index is None else device.index
        )
    scores: list[float] = []
    flags: list[bool] = []
    state = detach_self_referential_state(state)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(sampling_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sampling_seed)
        for pair in generated:
            candidate, observed = _support_step(
                model,
                state,
                (pair,),
                device,
                reward_transform=(
                    "inverted" if mode == "inverted" else "ordinary"
                ),
            )
            scores.append(observed[0])
            flags.append(pair.learner.public_flag)
            if mode in ("ordinary", "inverted"):
                state = candidate
    return state, _online_summary(scores, flags)


def _evaluate(
    model: AdaptiveReasoningCore,
    settings: RunSettings,
    seed: int,
    device: torch.device,
    stream: _UniqueSeedStream,
) -> dict[str, object]:
    # The exact evaluator mechanisms remain outside the learner-facing world
    # package and are loaded only after training has ended and the caller has
    # fingerprinted the frozen candidate.
    from experiments.evaluators.latent_order_suite import evaluator_programs

    for count in (
        settings.evaluation_supports,
        settings.reacquisition_supports,
    ):
        _balanced_flags(count)
    rng = random.Random(seed + 900_000)
    program_order = list(evaluator_programs())
    rng.shuffle(program_order)
    state = model.initial_state(1)
    sequential: list[dict[str, object]] = []
    counterfactual_presentations = 0

    model.eval()
    slow_before = reasoning_state_digest(model)
    with torch.no_grad():
        for position, program in enumerate(program_order):
            support_count = settings.evaluation_supports
            support_seeds = stream.take(support_count)
            support_flags = _balanced_flags(support_count)
            support_tasks = _generate_batch(
                (program,) * support_count,
                support_seeds,
                stream,
                public_flags=support_flags,
            )
            incoming_state = detach_self_referential_state(state)
            sampling_seed = seed + 10_000 + position
            state, ordinary = _run_online_sequence(
                model,
                incoming_state,
                support_tasks,
                device,
                mode="ordinary",
                sampling_seed=sampling_seed,
            )
            _, no_write = _run_online_sequence(
                model,
                incoming_state,
                support_tasks,
                device,
                mode="no_write",
                sampling_seed=sampling_seed,
            )
            _, inverted = _run_online_sequence(
                model,
                incoming_state,
                support_tasks,
                device,
                mode="inverted",
                sampling_seed=sampling_seed,
            )
            counterfactual_presentations += support_count * 2
            causal_by_flag = {
                flag: (
                    _flag_metric(ordinary, flag, "late_mean")
                    - max(
                        _flag_metric(no_write, flag, "late_mean"),
                        _flag_metric(inverted, flag, "late_mean"),
                    )
                )
                for flag in ("false", "true")
            }
            sequential.append(
                {
                    "mechanism": f"mechanism_{position + 1}",
                    "support_count": support_count,
                    "ordinary": ordinary,
                    "no_write": no_write,
                    "inverted_feedback": inverted,
                    "causal_advantage_by_public_flag": causal_by_flag,
                    "worst_flag_causal_advantage": min(
                        causal_by_flag.values()
                    ),
                }
            )

        final_state = detach_self_referential_state(state)
        reacquisition: list[dict[str, object]] = []
        for index, program in enumerate(program_order):
            support_seeds = stream.take(settings.reacquisition_supports)
            support_flags = _balanced_flags(settings.reacquisition_supports)
            reentry_tasks = _generate_batch(
                (program,) * settings.reacquisition_supports,
                support_seeds,
                stream,
                public_flags=support_flags,
            )
            sampling_seed = seed + 20_000 + index
            _, history = _run_online_sequence(
                model,
                detach_self_referential_state(final_state),
                reentry_tasks,
                device,
                mode="ordinary",
                sampling_seed=sampling_seed,
            )
            _, cold = _run_online_sequence(
                model,
                model.initial_state(1),
                reentry_tasks,
                device,
                mode="ordinary",
                sampling_seed=sampling_seed,
            )
            counterfactual_presentations += settings.reacquisition_supports
            savings_by_flag = {
                flag: (
                    _flag_metric(history, flag, "mean")
                    - _flag_metric(cold, flag, "mean")
                )
                for flag in ("false", "true")
            }
            early_savings_by_flag = {
                flag: (
                    _flag_metric(history, flag, "early_mean")
                    - _flag_metric(cold, flag, "early_mean")
                )
                for flag in ("false", "true")
            }
            reacquisition.append(
                {
                    "mechanism": f"mechanism_{index + 1}",
                    "history": history,
                    "cold": cold,
                    "savings_by_public_flag": savings_by_flag,
                    "worst_flag_savings": min(savings_by_flag.values()),
                    "early_savings_by_public_flag": early_savings_by_flag,
                    "worst_flag_early_savings": min(
                        early_savings_by_flag.values()
                    ),
                }
            )

    slow_after = reasoning_state_digest(model)
    if slow_after != slow_before:
        raise RuntimeError("frozen slow parameters changed during evaluation")
    worst_flag_gains = [
        float(record["ordinary"]["worst_flag_gain"])
        for record in sequential
    ]
    causal_advantages = [
        float(record["worst_flag_causal_advantage"])
        for record in sequential
    ]
    reacquisition_savings = [
        float(record["worst_flag_savings"])
        for record in reacquisition
    ]
    early_savings = [
        float(record["worst_flag_early_savings"])
        for record in reacquisition
    ]
    return {
        "sequential": sequential,
        "reacquisition": reacquisition,
        "macro_worst_flag_gain": sum(worst_flag_gains) / len(worst_flag_gains),
        "positive_gain_mechanisms": sum(
            gain > 0.0 for gain in worst_flag_gains
        ),
        "macro_worst_flag_causal_advantage": (
            sum(causal_advantages) / len(causal_advantages)
        ),
        "causal_control_mechanisms": sum(
            advantage > 0.0 for advantage in causal_advantages
        ),
        "macro_worst_flag_reacquisition_savings": (
            sum(reacquisition_savings) / len(reacquisition_savings)
        ),
        "positive_reacquisition_mechanisms": sum(
            saving > 0.0 for saving in reacquisition_savings
        ),
        "macro_worst_flag_early_savings": (
            sum(early_savings) / len(early_savings)
        ),
        "counterfactual_branch_presentations": counterfactual_presentations,
        "slow_state_unchanged": slow_before == slow_after,
        "final_fast_state_digest": self_referential_state_digest(final_state),
    }


def _classify_effect(
    evaluation: dict[str, object],
) -> tuple[str, dict[str, bool]]:
    online_effect = (
        float(evaluation["macro_worst_flag_gain"]) > 0.0
        and int(evaluation["positive_gain_mechanisms"]) >= 3
    )
    causal_effect = (
        float(evaluation["macro_worst_flag_causal_advantage"]) > 0.0
        and int(evaluation["causal_control_mechanisms"]) >= 3
    )
    persistence_effect = (
        float(evaluation["macro_worst_flag_reacquisition_savings"]) > 0.0
        and int(evaluation["positive_reacquisition_mechanisms"]) >= 3
    )
    if online_effect and causal_effect and persistence_effect:
        status = "ADAPTIVE_PERSISTENT_EFFECT_OBSERVED"
    elif online_effect and causal_effect:
        status = "ONLINE_EFFECT_WITHOUT_PERSISTENCE"
    else:
        status = "NO_CAUSAL_ADAPTIVE_EFFECT_OBSERVED"
    return status, {
        "online_gain": online_effect,
        "matched_causal_controls": causal_effect,
        "reacquisition_savings": persistence_effect,
    }


def main() -> None:
    args = parse_args()
    settings = _PROFILES[args.profile]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = _make_model(settings, device)
    training_stream = _UniqueSeedStream(args.seed)
    evaluation_stream = _UniqueSeedStream(args.seed + 10_000_000)
    start = time.perf_counter()
    training = _train(
        model,
        settings,
        args.seed,
        device,
        training_stream,
    )
    slow_digest = reasoning_state_digest(model)
    model.requires_grad_(False)
    evaluation = _evaluate(
        model,
        settings,
        args.seed,
        device,
        evaluation_stream,
    )
    wall_seconds = time.perf_counter() - start

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        str(checkpoint),
        metadata={
            "experiment": _EXPERIMENT_ID,
            "slow_digest": slow_digest,
            "profile": args.profile,
        },
    )
    status, effect_checks = _classify_effect(evaluation)
    result = {
        "experiment": _EXPERIMENT_ID,
        "profile": args.profile,
        "seed": args.seed,
        "settings": asdict(settings),
        "device": str(device),
        "slow_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "fast_state_scalars": model.memory.state_numel(),
        "training_unique_public_instances": len(training_stream.instance_ids),
        "evaluation_unique_public_instances": len(
            evaluation_stream.instance_ids
        ),
        "stream_protocol": {
            "within_branch_feedback_replay_count": 0,
            "counterfactual_branch_presentations": evaluation[
                "counterfactual_branch_presentations"
            ],
            "counterfactual_note": (
                "Matched experimental branches reuse the same unique public "
                "instances from the same incoming state and Torch random "
                "stream; no single learning branch sees an instance twice."
            ),
            "training_and_evaluation_seed_domains_separate": True,
        },
        "training": training,
        "evaluation": evaluation,
        "slow_state_digest": slow_digest,
        "checkpoint": str(checkpoint),
        "wall_seconds": wall_seconds,
        "status": status,
        "effect_checks": effect_checks,
        "claim_limit": (
            "One directional run is not a universal-learning or statistical "
            "claim. The primary evidence is the score trajectory on unique "
            "attempts before each feedback write, plus matched causal branches "
            "and reacquisition. Structurally new compositions only rule out "
            "exact-program memorization; conditional branches include familiar "
            "subprograms and are reported separately by public flag."
        ),
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
