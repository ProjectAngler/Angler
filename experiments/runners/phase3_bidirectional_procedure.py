"""Train and evaluate Angler's first bidirectional procedure constructor.

Training data consists only of transitions produced by random interaction with
the public reversible world.  No shortest-path solver or benchmark solution
trace is used.  Held-out success is counted only after the independent world
executes one committed primitive sequence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.reasoning.bidirectional_procedure_core import (  # noqa: E402
    BidirectionalProcedureConfig,
    BidirectionalProcedureCore,
    ProcedureLearningBatch,
    permutations_to_tensor,
    procedure_core_digest,
)
from angler.worlds.reversible_transition_world import (  # noqa: E402
    ACTION_COUNT,
    TOKEN_COUNT,
    ReversibleTransitionTask,
    commit_procedure,
    execute_committed_procedure,
    generate_reversible_transition_task,
    observe_primitive_transition,
)
from experiments.evaluators.bidirectional_procedure_suite import (  # noqa: E402
    ProcedureChallenge,
    evaluate_procedure,
    make_heldout_procedure_suite,
    summarize_results,
)


@dataclass(frozen=True, slots=True)
class RunProfile:
    hidden_width: int
    action_width: int
    unique_transitions: int
    trajectory_count: int
    trajectory_minimum: int
    trajectory_maximum: int
    optimizer_steps: int
    batch_size: int
    learning_rate: float
    cases_per_distance: int
    evaluation_distances: tuple[int, ...]
    maximum_steps: int
    maximum_expansions: int
    actions_per_state: int
    qwen_cases: int


PROFILES = {
    "smoke": RunProfile(
        hidden_width=96,
        action_width=24,
        unique_transitions=900,
        trajectory_count=700,
        trajectory_minimum=2,
        trajectory_maximum=10,
        optimizer_steps=500,
        batch_size=256,
        learning_rate=3e-3,
        cases_per_distance=2,
        evaluation_distances=(2, 4, 6, 8),
        maximum_steps=10,
        maximum_expansions=900,
        actions_per_state=2,
        qwen_cases=4,
    ),
    "full": RunProfile(
        hidden_width=192,
        action_width=48,
        unique_transitions=2600,
        trajectory_count=2600,
        trajectory_minimum=3,
        trajectory_maximum=12,
        optimizer_steps=1800,
        batch_size=512,
        learning_rate=2e-3,
        cases_per_distance=4,
        evaluation_distances=(2, 4, 6, 8, 10),
        maximum_steps=12,
        maximum_expansions=1400,
        actions_per_state=2,
        qwen_cases=8,
    ),
}


@dataclass(frozen=True, slots=True)
class _Trajectory:
    states: tuple[tuple[int, ...], ...]
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Corpus:
    transition_states: torch.Tensor
    transition_actions: torch.Tensor
    transition_next_states: torch.Tensor
    origins: torch.Tensor
    goals: torch.Tensor
    horizons: torch.Tensor
    first_actions: torch.Tensor
    last_actions: torch.Tensor
    midpoints: torch.Tensor
    transition_keys: frozenset[tuple[tuple[int, ...], int]]
    auxiliary_transition_keys: frozenset[tuple[tuple[int, ...], int]]
    future_goal_pairs: frozenset[tuple[tuple[int, ...], tuple[int, ...]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--goal-source", choices=("structured", "qwen"), default="structured")
    parser.add_argument("--qwen-model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def _domain_rng(seed: int, domain: str) -> random.Random:
    material = f"angler.phase3.bidirectional.v1\x00{seed}\x00{domain}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _random_origin(rng: random.Random) -> tuple[int, ...]:
    values = list(range(TOKEN_COUNT))
    rng.shuffle(values)
    return tuple(values)


def _random_trajectory(
    rng: random.Random,
    minimum: int,
    maximum: int,
) -> _Trajectory:
    length = rng.randint(minimum, maximum)
    states = [_random_origin(rng)]
    actions: list[int] = []
    previous: int | None = None
    for _ in range(length):
        choices = list(range(ACTION_COUNT))
        # Avoid an immediate undo while retaining stochastic, non-solver data.
        if previous is not None and len(choices) > 1:
            choices.remove(previous)
        action = rng.choice(choices)
        actions.append(action)
        states.append(observe_primitive_transition(states[-1], action).after)
        previous = action
    return _Trajectory(tuple(states), tuple(actions))


def build_experience_corpus(
    profile: RunProfile,
    seed: int,
    *,
    excluded_goal_pairs: Sequence[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = (),
) -> _Corpus:
    """Create unique transitions and HER-style future-state relabels."""

    rng = _domain_rng(seed, "experience")
    trajectories: list[_Trajectory] = []
    transitions: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[int, ...], int, tuple[int, ...]],
    ] = {}
    attempts = 0
    while (
        len(trajectories) < profile.trajectory_count
        or len(transitions) < profile.unique_transitions
    ):
        attempts += 1
        if attempts > profile.trajectory_count * 20:
            raise RuntimeError("experience generator could not meet its unique-transition target")
        trajectory = _random_trajectory(
            rng,
            profile.trajectory_minimum,
            profile.trajectory_maximum,
        )
        trajectories.append(trajectory)
        for index, action in enumerate(trajectory.actions):
            key = (trajectory.states[index], action)
            transitions.setdefault(
                key,
                (trajectory.states[index], action, trajectory.states[index + 1]),
            )
        if (
            len(trajectories) >= profile.trajectory_count
            and len(transitions) >= profile.unique_transitions
        ):
            break

    selected_transition_keys = tuple(transitions)[: profile.unique_transitions]
    selected_transition_key_set = frozenset(selected_transition_keys)
    selected_transitions = tuple(transitions[key] for key in selected_transition_keys)
    excluded_pairs = frozenset(excluded_goal_pairs)

    relabels: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            int,
            int,
            int,
            tuple[int, ...],
        ]
    ] = []
    auxiliary_transition_keys: set[tuple[tuple[int, ...], int]] = set()
    for trajectory in trajectories[: profile.trajectory_count]:
        final_index = len(trajectory.states) - 1
        for start in range(final_index):
            maximum_end = min(
                final_index,
                start + profile.maximum_steps,
            )
            if maximum_end <= start:
                continue
            # Each observed prefix contributes two future goals when possible;
            # no goal or action was chosen to solve an evaluation case.
            end_candidates = {maximum_end, rng.randint(start + 1, maximum_end)}
            for end in sorted(end_candidates):
                segment_keys = tuple(
                    (trajectory.states[index], trajectory.actions[index])
                    for index in range(start, end)
                )
                # The transition split applies to every supervision head, not
                # merely the direct F/B rows.  Otherwise a horizon-one HER
                # relabel can silently disclose a supposedly held-out edge.
                if any(key not in selected_transition_key_set for key in segment_keys):
                    continue
                goal_pair = (trajectory.states[start], trajectory.states[end])
                if goal_pair in excluded_pairs:
                    continue
                horizon = end - start
                midpoint = start + horizon // 2
                relabels.append(
                    (
                        trajectory.states[start],
                        trajectory.states[end],
                        horizon,
                        trajectory.actions[start],
                        trajectory.actions[end - 1],
                        trajectory.states[midpoint],
                    )
                )
                auxiliary_transition_keys.update(segment_keys)
    if not relabels:
        raise RuntimeError("experience generator produced no future-state relabels")

    return _Corpus(
        transition_states=permutations_to_tensor(
            [item[0] for item in selected_transitions]
        ),
        transition_actions=torch.tensor(
            [item[1] for item in selected_transitions], dtype=torch.long
        ),
        transition_next_states=permutations_to_tensor(
            [item[2] for item in selected_transitions]
        ),
        origins=permutations_to_tensor([item[0] for item in relabels]),
        goals=permutations_to_tensor([item[1] for item in relabels]),
        horizons=torch.tensor([item[2] for item in relabels], dtype=torch.long),
        first_actions=torch.tensor([item[3] for item in relabels], dtype=torch.long),
        last_actions=torch.tensor([item[4] for item in relabels], dtype=torch.long),
        midpoints=permutations_to_tensor([item[5] for item in relabels]),
        transition_keys=selected_transition_key_set,
        auxiliary_transition_keys=frozenset(auxiliary_transition_keys),
        future_goal_pairs=frozenset((item[0], item[1]) for item in relabels),
    )


def _sample_learning_batch(
    corpus: _Corpus,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> ProcedureLearningBatch:
    transition_indices = torch.randint(
        corpus.transition_states.shape[0],
        (batch_size,),
        generator=generator,
    )
    relabel_indices = torch.randint(
        corpus.origins.shape[0],
        (batch_size,),
        generator=generator,
    )
    return ProcedureLearningBatch(
        states=corpus.transition_states[transition_indices].to(device),
        actions=corpus.transition_actions[transition_indices].to(device),
        next_states=corpus.transition_next_states[transition_indices].to(device),
        origins=corpus.origins[relabel_indices].to(device),
        goals=corpus.goals[relabel_indices].to(device),
        horizons=corpus.horizons[relabel_indices].to(device),
        first_actions=corpus.first_actions[relabel_indices].to(device),
        last_actions=corpus.last_actions[relabel_indices].to(device),
        midpoints=corpus.midpoints[relabel_indices].to(device),
    )


def train_core(
    model: BidirectionalProcedureCore,
    corpus: _Corpus,
    profile: RunProfile,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=1e-5,
    )
    first_losses: dict[str, float] | None = None
    final_losses: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    for step in range(profile.optimizer_steps):
        batch = _sample_learning_batch(
            corpus,
            profile.batch_size,
            generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = model.learning_losses(batch)
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("procedure-core gradient became non-finite")
        optimizer.step()
        numeric = {name: float(value.detach().item()) for name, value in losses.items()}
        if first_losses is None:
            first_losses = numeric
        final_losses = numeric
        if not all(math_is_finite(value) for value in numeric.values()):
            raise RuntimeError(f"non-finite loss at optimizer step {step}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "optimizer_steps": profile.optimizer_steps,
        "batch_size": profile.batch_size,
        "unique_transition_experiences": len(corpus.transition_keys),
        "future_goal_examples": int(corpus.origins.shape[0]),
        "first_losses": first_losses,
        "final_losses": final_losses,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _all_transition_holdout(corpus: _Corpus) -> tuple[torch.Tensor, ...]:
    states: list[tuple[int, ...]] = []
    actions: list[int] = []
    next_states: list[tuple[int, ...]] = []
    for state in itertools.permutations(range(TOKEN_COUNT)):
        for action in range(ACTION_COUNT):
            if (state, action) in corpus.transition_keys:
                continue
            states.append(state)
            actions.append(action)
            next_states.append(observe_primitive_transition(state, action).after)
    if not states:
        raise RuntimeError("training consumed the entire finite transition graph")
    return (
        permutations_to_tensor(states),
        torch.tensor(actions, dtype=torch.long),
        permutations_to_tensor(next_states),
    )


@torch.no_grad()
def transition_metrics(
    model: BidirectionalProcedureCore,
    corpus: _Corpus,
    device: torch.device,
) -> dict[str, Any]:
    states, actions, next_states = _all_transition_holdout(corpus)
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)
    forward = model.forward_state_logits(states, actions).argmax(-1)
    backward = model.backward_state_logits(next_states, actions).argmax(-1)
    inverse = model.inverse_action_logits(states, next_states).argmax(-1)
    expected_next = next_states.argmax(-1)
    expected_state = states.argmax(-1)
    return {
        "heldout_transitions": int(states.shape[0]),
        "forward_exact": float((forward == expected_next).all(dim=1).float().mean().item()),
        "backward_exact": float((backward == expected_state).all(dim=1).float().mean().item()),
        "inverse_action_exact": float((inverse == actions).float().mean().item()),
        "forward_token_accuracy": float((forward == expected_next).float().mean().item()),
        "backward_token_accuracy": float((backward == expected_state).float().mean().item()),
    }


def _evaluate_branch(
    model: BidirectionalProcedureCore,
    challenges: Sequence[ProcedureChallenge],
    profile: RunProfile,
    *,
    use_backward: bool,
    policy_action_permutation: Sequence[int] | None = None,
    backward_action_permutation: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = []
    details: list[dict[str, Any]] = []
    for challenge in challenges:
        plan = model.construct_procedure(
            challenge.origin,
            challenge.goal,
            maximum_steps=profile.maximum_steps,
            maximum_expansions=profile.maximum_expansions,
            use_backward=use_backward,
            actions_per_state=profile.actions_per_state,
            policy_action_permutation=policy_action_permutation,
            backward_action_permutation=backward_action_permutation,
        )
        # A no-plan branch still commits exactly one empty candidate so every
        # branch has the same terminal judging semantics and no retries.
        actions = plan.actions if plan.found else ()
        outcome = evaluate_procedure(
            challenge,
            actions,
            expansions=plan.total_expansions,
        )
        results.append(outcome)
        details.append(
            {
                "case_id": challenge.case_id,
                "origin": list(challenge.origin),
                "goal": list(challenge.goal),
                "plan_found": plan.found,
                "actions": list(actions),
                "meeting_state": None if plan.meeting_state is None else list(plan.meeting_state),
                "exact_frontier_join": plan.exact_frontier_join,
                "reason": plan.reason,
                "forward_expansions": plan.forward_expansions,
                "backward_expansions": plan.backward_expansions,
                "environment_exact": outcome.exact,
                "reached_state": list(outcome.reached_state),
            }
        )
    summary = summarize_results(results)
    return asdict(summary), details


def _goal_swap_check(
    model: BidirectionalProcedureCore,
    challenges: Sequence[ProcedureChallenge],
    profile: RunProfile,
) -> dict[str, Any]:
    for source in challenges:
        for donor in challenges:
            if source.case_id == donor.case_id or donor.goal == source.goal:
                continue
            distance = _inversion_distance(source.origin, donor.goal)
            if distance > profile.maximum_steps:
                continue
            plan = model.construct_procedure(
                source.origin,
                donor.goal,
                maximum_steps=profile.maximum_steps,
                maximum_expansions=profile.maximum_expansions,
                actions_per_state=profile.actions_per_state,
            )
            actions = plan.actions if plan.found else ()
            committed = commit_procedure(source.task, actions)
            execution = execute_committed_procedure(
                source.task,
                committed,
                donor.goal,
            )
            return {
                "source_case_id": source.case_id,
                "donor_case_id": donor.case_id,
                "swapped_goal_distance": distance,
                "plan_found": plan.found,
                "followed_swapped_goal": execution.exact,
                "reached_original_goal": execution.reached_state == source.goal,
                "actions": list(actions),
            }
    return {"plan_found": False, "reason": "no_bounded_goal_swap_pair"}


def _inversion_distance(origin: Sequence[int], goal: Sequence[int]) -> int:
    positions = {token: index for index, token in enumerate(origin)}
    relative = [positions[token] for token in goal]
    return sum(
        relative[left] > relative[right]
        for left in range(len(relative))
        for right in range(left + 1, len(relative))
    )


def _parse_goal_json(text: str) -> tuple[int, ...] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"goal"}:
        return None
    goal = payload["goal"]
    if not isinstance(goal, list) or any(type(value) is not int for value in goal):
        return None
    candidate = tuple(goal)
    return candidate if sorted(candidate) == list(range(TOKEN_COUNT)) else None


def _make_qwen_semantic_tasks(
    profile: RunProfile,
    seed: int,
) -> tuple[
    tuple[ReversibleTransitionTask, tuple[int, ...], str],
    ...,
]:
    """Define connector tasks before training so their goal pairs can be excluded."""

    maximum_steps = TOKEN_COUNT * (TOKEN_COUNT - 1) // 2
    tasks: list[tuple[ReversibleTransitionTask, tuple[int, ...], str]] = []
    for index in range(profile.qwen_cases):
        task = generate_reversible_transition_task(
            seed + 700_001 + index,
            max_steps=maximum_steps,
        )
        ascending = index % 2 == 0
        direction = "ascending" if ascending else "descending"
        target = tuple(range(TOKEN_COUNT))
        if not ascending:
            target = tuple(reversed(target))
        tasks.append((task, target, direction))
    return tuple(tasks)


def _run_qwen_goal_suite(
    core: BidirectionalProcedureCore,
    profile: RunProfile,
    *,
    model_path: Path,
    device: torch.device,
    semantic_tasks: Sequence[
        tuple[ReversibleTransitionTask, tuple[int, ...], str]
    ],
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("the local Qwen goal suite requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device.index or 0},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen was not frozen")

    # Unlike the stratified evaluator, a semantic goal proposer can choose any
    # state in the six-token space.  Give this connector the full graph
    # diameter and enough expansions to avoid misclassifying a correct goal as
    # a planner failure merely because it happened to be farther than the
    # structured suite's comparison budget.
    qwen_maximum_steps = TOKEN_COUNT * (TOKEN_COUNT - 1) // 2
    qwen_maximum_expansions = math.factorial(TOKEN_COUNT) * ACTION_COUNT
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    tasks = []
    targets: list[tuple[int, ...]] = []
    for task, target, direction in semantic_tasks:
        prompt = (
            "Act only as a destination-state proposer. The origin is "
            f"{list(task.origin)}. The public objective is to arrange every "
            f"integer in strictly {direction} numerical order. Return only "
            'strict JSON of the form {"goal":[...]}; use every input integer '
            "exactly once and include no explanation."
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(rendered)
        tasks.append(task)
        targets.append(target)

    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for index, (task, intended) in enumerate(zip(tasks, targets, strict=True)):
        new_tokens = generated[index, encoded.input_ids.shape[1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        proposed = _parse_goal_json(raw)
        plan = None
        reached = None
        endpoint_follows_candidate = False
        intended_exact = False
        if proposed is not None:
            plan = core.construct_procedure(
                task.origin,
                proposed,
                maximum_steps=qwen_maximum_steps,
                maximum_expansions=qwen_maximum_expansions,
                actions_per_state=profile.actions_per_state,
            )
            committed = commit_procedure(task, plan.actions if plan.found else ())
            execution = execute_committed_procedure(task, committed, intended)
            reached = execution.reached_state
            endpoint_follows_candidate = reached == proposed
            intended_exact = execution.exact
        records.append(
            {
                "origin": list(task.origin),
                "intended_goal": list(intended),
                "raw_qwen_output": raw,
                "proposed_goal": None if proposed is None else list(proposed),
                "goal_proposal_valid": proposed is not None,
                "goal_proposal_correct": proposed == intended,
                "plan_found": bool(plan is not None and plan.found),
                "endpoint_follows_candidate": endpoint_follows_candidate,
                "end_to_end_exact": intended_exact,
                "reached_state": None if reached is None else list(reached),
            }
        )
    count = len(records)
    return {
        "model_path": str(model_path),
        "model_class": type(model).__name__,
        "frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "cases": count,
        "maximum_steps": qwen_maximum_steps,
        "maximum_expansions": qwen_maximum_expansions,
        "valid_goal_rate": sum(item["goal_proposal_valid"] for item in records) / count,
        "correct_goal_rate": sum(item["goal_proposal_correct"] for item in records) / count,
        "procedure_reaches_candidate_rate": sum(
            item["endpoint_follows_candidate"] for item in records
        ) / count,
        "end_to_end_exact_rate": sum(item["end_to_end_exact"] for item in records) / count,
        "records": records,
    }


def _save_checkpoint(model: BidirectionalProcedureCore, path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        str(path),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILES[args.profile]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    challenges = make_heldout_procedure_suite(
        args.seed + 400_009,
        inversion_distances=profile.evaluation_distances,
        cases_per_distance=profile.cases_per_distance,
        max_steps=profile.maximum_steps,
    )
    evaluation_goal_pairs = frozenset(
        (challenge.origin, challenge.goal) for challenge in challenges
    )
    qwen_semantic_tasks = _make_qwen_semantic_tasks(profile, args.seed)
    qwen_intended_goal_pairs = frozenset(
        (task.origin, target) for task, target, _ in qwen_semantic_tasks
    )
    excluded_goal_pairs = evaluation_goal_pairs | qwen_intended_goal_pairs
    corpus = build_experience_corpus(
        profile,
        args.seed,
        excluded_goal_pairs=tuple(excluded_goal_pairs),
    )
    if corpus.future_goal_pairs & evaluation_goal_pairs:
        raise RuntimeError("an evaluation origin/goal pair entered training relabels")
    if not corpus.auxiliary_transition_keys <= corpus.transition_keys:
        raise RuntimeError("an excluded transition entered auxiliary supervision")
    if corpus.future_goal_pairs & qwen_intended_goal_pairs:
        raise RuntimeError("a Qwen connector origin/goal pair entered training relabels")
    config = BidirectionalProcedureConfig(
        item_count=TOKEN_COUNT,
        hidden_width=profile.hidden_width,
        action_width=profile.action_width,
        maximum_horizon=max(15, profile.maximum_steps),
    )
    model = BidirectionalProcedureCore(config).to(device)
    initial_digest = procedure_core_digest(model)
    training = train_core(
        model,
        corpus,
        profile,
        seed=args.seed,
        device=device,
    )
    trained_digest = procedure_core_digest(model)
    if trained_digest == initial_digest:
        raise RuntimeError("training did not change the procedure core")
    model.eval().requires_grad_(False)
    frozen_digest = procedure_core_digest(model)
    transition_result = transition_metrics(model, corpus, device)

    bidirectional, bidirectional_details = _evaluate_branch(
        model,
        challenges,
        profile,
        use_backward=True,
    )
    forward_only, forward_details = _evaluate_branch(
        model,
        challenges,
        profile,
        use_backward=False,
    )
    corrupted, corrupted_details = _evaluate_branch(
        model,
        challenges,
        profile,
        use_backward=True,
        backward_action_permutation=tuple(range(1, ACTION_COUNT)) + (0,),
    )
    policy_permuted, policy_permuted_details = _evaluate_branch(
        model,
        challenges,
        profile,
        use_backward=True,
        policy_action_permutation=tuple(range(1, ACTION_COUNT)) + (0,),
    )

    torch.manual_seed(args.seed)
    untrained = BidirectionalProcedureCore(config).to(device)
    untrained.eval().requires_grad_(False)
    untrained_summary, untrained_details = _evaluate_branch(
        untrained,
        challenges,
        profile,
        use_backward=True,
    )
    goal_swap = _goal_swap_check(model, challenges, profile)
    qwen = None
    if args.goal_source == "qwen":
        qwen = _run_qwen_goal_suite(
            model,
            profile,
            model_path=Path(args.qwen_model),
            device=device,
            semantic_tasks=qwen_semantic_tasks,
        )
    if procedure_core_digest(model) != frozen_digest:
        raise RuntimeError("evaluation mutated the frozen procedure core")

    checks = {
        "heldout_forward_dynamics": transition_result["forward_exact"] >= 0.98,
        "heldout_backward_dynamics": transition_result["backward_exact"] >= 0.98,
        "verified_procedure_success": bidirectional["exact_success_rate"] >= 0.80,
        "causal_over_untrained": bidirectional["exact_success_rate"] > untrained_summary["exact_success_rate"],
        "backward_integrity_matters": (
            bidirectional["exact_success_rate"] > corrupted["exact_success_rate"]
            or bidirectional["mean_expansions"] < corrupted["mean_expansions"]
        ),
        "bidirectional_search_advantage": (
            bidirectional["exact_success_rate"] > forward_only["exact_success_rate"]
            or (
                bidirectional["exact_success_rate"]
                == forward_only["exact_success_rate"]
                and bidirectional["mean_expansions"]
                < forward_only["mean_expansions"]
            )
        ),
        "learned_action_guidance_matters": (
            bidirectional["exact_success_rate"]
            > policy_permuted["exact_success_rate"]
            or (
                bidirectional["exact_success_rate"]
                == policy_permuted["exact_success_rate"]
                and bidirectional["mean_expansions"]
                < policy_permuted["mean_expansions"]
            )
        ),
        "goal_choke_point": bool(goal_swap.get("followed_swapped_goal")),
        "frozen_during_evaluation": procedure_core_digest(model) == frozen_digest,
    }
    status = (
        "PROCEDURAL_EFFECT_OBSERVED"
        if all(checks.values())
        else "PROCEDURAL_EFFECT_NOT_YET_ESTABLISHED"
    )
    result = {
        "experiment": "ANGLER-BIDIRECTIONAL-PROCEDURE-V1",
        "status": status,
        "claim_level": (
            "learned reversible dynamics and learned action guidance composed "
            "by bounded generic bidirectional search into independently verified procedures"
        ),
        "profile": args.profile,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config),
        "profile_settings": asdict(profile),
        "donor_concepts": {
            "backward_learning": "reversed transition learning and backward rollout; MIT; clean-room",
            "her": "future-state relabeling from observed trajectories",
            "sgt_pg": (
                "shared midpoint auxiliary target; MIT; clean-room; trained but "
                "not yet used causally by planning"
            ),
        },
        "data_boundary": {
            "random_executed_trajectories_only": True,
            "shortest_path_solver_used_for_training": False,
            "evaluation_solution_trace_visible": False,
            "task_or_program_id_entered_model": False,
            "unique_transition_experiences": len(corpus.transition_keys),
            "heldout_transition_keys_in_auxiliary_labels": len(
                corpus.auxiliary_transition_keys - corpus.transition_keys
            ),
            "evaluation_goal_pair_collisions": len(
                corpus.future_goal_pairs & evaluation_goal_pairs
            ),
            "qwen_intended_goal_pair_collisions": len(
                corpus.future_goal_pairs & qwen_intended_goal_pairs
            ),
        },
        "state": {
            "initial_digest": initial_digest,
            "trained_digest": trained_digest,
            "frozen_evaluation_digest": frozen_digest,
            "parameters": model.parameter_count(),
        },
        "training": training,
        "heldout_transition_metrics": transition_result,
        "verified_branches": {
            "bidirectional": bidirectional,
            "forward_only_equal_budget": forward_only,
            "corrupted_backward_equal_compute": corrupted,
            "permuted_policy_equal_compute": policy_permuted,
            "untrained": untrained_summary,
        },
        "goal_swap": goal_swap,
        "checks": checks,
        "qwen_goal_suite": qwen,
        "case_details": {
            "bidirectional": bidirectional_details,
            "forward_only": forward_details,
            "corrupted_backward": corrupted_details,
            "permuted_policy": policy_permuted_details,
            "untrained": untrained_details,
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "local_only": True,
    }
    if args.checkpoint is not None:
        _save_checkpoint(model, args.checkpoint)
        result["checkpoint"] = str(args.checkpoint)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
