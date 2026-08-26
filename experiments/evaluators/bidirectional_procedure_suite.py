"""Held-out permutation challenges for bidirectional procedure learning.

The suite exposes only an origin, a candidate destination, the world's
primitive-action vocabulary, and execution ceilings.  Exact-distance target
selection and result aggregation remain evaluator concerns.  In particular,
this module never constructs, stores, or returns a solution path.

Submitted procedures are committed and executed exclusively through
``angler.worlds.reversible_transition_world``.  The evaluator does not replay
adjacent swaps itself and therefore cannot quietly become a second execution
engine or a task-specific solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from typing import Iterable, Sequence

from angler.worlds.reversible_transition_world import (
    DEFAULT_MAX_STEPS,
    TOKEN_COUNT,
    ProcedureExecution,
    ReversibleTransitionTask,
    commit_procedure,
    execute_committed_procedure,
    generate_reversible_transition_task,
)


DEFAULT_INVERSION_DISTANCES = (2, 4, 6, 8, 10)
DEFAULT_CASES_PER_DISTANCE = 4
MAX_INVERSION_DISTANCE = TOKEN_COUNT * (TOKEN_COUNT - 1) // 2


@dataclass(frozen=True, slots=True)
class ProcedureChallenge:
    """Complete learner-visible projection of one held-out challenge.

    ``task`` supplies the origin, primitive actions, and execution ceiling;
    ``goal`` is the candidate destination.  No distance, predecessor, midpoint,
    next-action hint, or solution trace is included.
    """

    case_id: str
    task: ReversibleTransitionTask
    goal: tuple[int, ...]

    @property
    def origin(self) -> tuple[int, ...]:
        return self.task.origin

    @property
    def available_actions(self) -> tuple[int, ...]:
        return self.task.available_actions

    @property
    def max_steps(self) -> int:
        return self.task.max_steps


@dataclass(frozen=True, slots=True)
class ProcedureCaseResult:
    """Terminal world outcome plus planner work for one unique challenge."""

    case_id: str
    inversion_distance: int
    exact: bool
    expansions: int
    steps_executed: int
    reached_state: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DistanceSummary:
    """Exact-success and search-work totals for one inversion distance."""

    inversion_distance: int
    attempts: int
    exact_successes: int
    exact_success_rate: float
    total_expansions: int
    mean_expansions: float


@dataclass(frozen=True, slots=True)
class ProcedureSuiteSummary:
    """Deterministic aggregate over one result per held-out challenge."""

    attempts: int
    exact_successes: int
    exact_success_rate: float
    total_expansions: int
    mean_expansions: float
    by_distance: tuple[DistanceSummary, ...]


def make_heldout_procedure_suite(
    seed: int,
    *,
    inversion_distances: Sequence[int] = DEFAULT_INVERSION_DISTANCES,
    cases_per_distance: int = DEFAULT_CASES_PER_DISTANCE,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[ProcedureChallenge, ...]:
    """Create replayable, unique origin/goal challenges at exact distances.

    Targets are sampled from all permutations at the requested inversion
    distance.  Exhaustive target filtering establishes difficulty without
    constructing a path.  The domain-separated seed keeps this held-out suite
    independent from learner experience streams.
    """

    _validate_seed(seed)
    distances = _validate_suite_parameters(
        inversion_distances,
        cases_per_distance,
        max_steps,
    )
    rng = _domain_rng(seed, "suite")
    challenges: list[ProcedureChallenge] = []
    seen_case_ids: set[str] = set()
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for distance in distances:
        created = 0
        attempts = 0
        while created < cases_per_distance:
            attempts += 1
            if attempts > 20_000:
                raise RuntimeError(
                    "could not generate enough unique held-out challenges"
                )
            world_seed = rng.randrange(0, 2**63)
            task = generate_reversible_transition_task(
                world_seed,
                max_steps=max_steps,
            )
            candidates = _goals_at_distance(task.origin, distance)
            goal = candidates[rng.randrange(len(candidates))]
            pair = (task.origin, goal)
            if pair in seen_pairs:
                continue

            case_id = _case_id(task, goal)
            if case_id in seen_case_ids:
                continue
            seen_pairs.add(pair)
            seen_case_ids.add(case_id)
            challenges.append(
                ProcedureChallenge(
                    case_id=case_id,
                    task=task,
                    goal=goal,
                )
            )
            created += 1

    return tuple(challenges)


def evaluate_procedure(
    challenge: ProcedureChallenge,
    actions: Sequence[int],
    *,
    expansions: int,
) -> ProcedureCaseResult:
    """Commit and execute one procedure through the real world exactly once."""

    _validate_challenge(challenge)
    _validate_expansions(expansions)
    committed = commit_procedure(challenge.task, actions)
    execution = execute_committed_procedure(
        challenge.task,
        committed,
        challenge.goal,
    )
    _validate_execution(challenge, execution)
    return ProcedureCaseResult(
        case_id=challenge.case_id,
        inversion_distance=_inversion_distance(
            challenge.origin,
            challenge.goal,
        ),
        exact=execution.exact,
        expansions=expansions,
        steps_executed=execution.steps_executed,
        reached_state=execution.reached_state,
    )


def summarize_results(
    results: Iterable[ProcedureCaseResult],
) -> ProcedureSuiteSummary:
    """Aggregate exact success and expansions without re-executing a case."""

    ordered = tuple(results)
    if not ordered:
        raise ValueError("results must not be empty")
    case_ids: set[str] = set()
    for result in ordered:
        if not isinstance(result, ProcedureCaseResult):
            raise TypeError("every result must be a ProcedureCaseResult")
        if result.case_id in case_ids:
            raise ValueError("results contain a duplicate case_id")
        case_ids.add(result.case_id)
        _validate_result(result)

    by_distance = tuple(
        _summarize_distance(distance, ordered)
        for distance in sorted(
            {result.inversion_distance for result in ordered}
        )
    )
    attempts = len(ordered)
    exact_successes = sum(result.exact for result in ordered)
    total_expansions = sum(result.expansions for result in ordered)
    return ProcedureSuiteSummary(
        attempts=attempts,
        exact_successes=exact_successes,
        exact_success_rate=exact_successes / attempts,
        total_expansions=total_expansions,
        mean_expansions=total_expansions / attempts,
        by_distance=by_distance,
    )


def _goals_at_distance(
    origin: tuple[int, ...],
    distance: int,
) -> tuple[tuple[int, ...], ...]:
    candidates = tuple(
        candidate
        for candidate in itertools.permutations(origin)
        if _inversion_distance(origin, candidate) == distance
    )
    if not candidates:
        raise ValueError(
            f"no goal exists at inversion distance {distance}"
        )
    return candidates


def _inversion_distance(
    origin: tuple[int, ...],
    goal: tuple[int, ...],
) -> int:
    if len(origin) != TOKEN_COUNT or len(goal) != TOKEN_COUNT:
        raise ValueError(f"states must contain exactly {TOKEN_COUNT} tokens")
    if len(set(origin)) != TOKEN_COUNT or set(origin) != set(goal):
        raise ValueError("origin and goal must be permutations of the same tokens")
    origin_index = {token: index for index, token in enumerate(origin)}
    relative = tuple(origin_index[token] for token in goal)
    return sum(
        relative[left] > relative[right]
        for left in range(TOKEN_COUNT)
        for right in range(left + 1, TOKEN_COUNT)
    )


def _summarize_distance(
    distance: int,
    results: tuple[ProcedureCaseResult, ...],
) -> DistanceSummary:
    selected = tuple(
        result
        for result in results
        if result.inversion_distance == distance
    )
    attempts = len(selected)
    exact_successes = sum(result.exact for result in selected)
    total_expansions = sum(result.expansions for result in selected)
    return DistanceSummary(
        inversion_distance=distance,
        attempts=attempts,
        exact_successes=exact_successes,
        exact_success_rate=exact_successes / attempts,
        total_expansions=total_expansions,
        mean_expansions=total_expansions / attempts,
    )


def _validate_suite_parameters(
    inversion_distances: Sequence[int],
    cases_per_distance: int,
    max_steps: int,
) -> tuple[int, ...]:
    if isinstance(inversion_distances, (str, bytes)):
        raise TypeError("inversion_distances must be a sequence of integers")
    distances = tuple(inversion_distances)
    if not distances:
        raise ValueError("inversion_distances must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in distances):
        raise TypeError("every inversion distance must be an integer")
    if len(set(distances)) != len(distances):
        raise ValueError("inversion_distances must be unique")
    if any(not 1 <= value <= MAX_INVERSION_DISTANCE for value in distances):
        raise ValueError(
            "every inversion distance must be between 1 and "
            f"{MAX_INVERSION_DISTANCE}"
        )
    if isinstance(cases_per_distance, bool) or not isinstance(
        cases_per_distance,
        int,
    ):
        raise TypeError("cases_per_distance must be an integer")
    if cases_per_distance <= 0:
        raise ValueError("cases_per_distance must be positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer")
    if max_steps < max(distances):
        raise ValueError(
            "max_steps must be at least the largest inversion distance"
        )
    return distances


def _validate_challenge(challenge: ProcedureChallenge) -> None:
    if not isinstance(challenge, ProcedureChallenge):
        raise TypeError("challenge must be a ProcedureChallenge")
    if challenge.case_id != _case_id(challenge.task, challenge.goal):
        raise ValueError("challenge case_id does not match its task and goal")
    _inversion_distance(challenge.origin, challenge.goal)


def _validate_execution(
    challenge: ProcedureChallenge,
    execution: ProcedureExecution,
) -> None:
    if not isinstance(execution, ProcedureExecution):
        raise TypeError("world returned an invalid execution record")
    if execution.task_id != challenge.task.instance_id:
        raise RuntimeError("world execution task identity changed")
    expected_exact = execution.reached_state == challenge.goal
    if execution.exact is not expected_exact:
        raise RuntimeError("world execution exact flag is internally inconsistent")


def _validate_result(result: ProcedureCaseResult) -> None:
    if not isinstance(result.exact, bool):
        raise TypeError("result exact must be a bool")
    _validate_expansions(result.expansions)
    if (
        isinstance(result.inversion_distance, bool)
        or not isinstance(result.inversion_distance, int)
        or not 1 <= result.inversion_distance <= MAX_INVERSION_DISTANCE
    ):
        raise ValueError("result inversion_distance is outside suite bounds")
    if (
        isinstance(result.steps_executed, bool)
        or not isinstance(result.steps_executed, int)
        or result.steps_executed < 0
    ):
        raise ValueError("result steps_executed must be a non-negative integer")


def _validate_expansions(expansions: int) -> None:
    if (
        isinstance(expansions, bool)
        or not isinstance(expansions, int)
        or expansions < 0
    ):
        raise ValueError("expansions must be a non-negative integer")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def _case_id(
    task: ReversibleTransitionTask,
    goal: tuple[int, ...],
) -> str:
    material = {
        "world_task_id": task.instance_id,
        "goal": list(goal),
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _domain_rng(seed: int, domain: str) -> random.Random:
    material = (
        f"angler.bidirectional-procedure-suite.v1\x00{seed}\x00{domain}"
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


__all__ = [
    "DEFAULT_CASES_PER_DISTANCE",
    "DEFAULT_INVERSION_DISTANCES",
    "DistanceSummary",
    "MAX_INVERSION_DISTANCE",
    "ProcedureCaseResult",
    "ProcedureChallenge",
    "ProcedureSuiteSummary",
    "evaluate_procedure",
    "make_heldout_procedure_suite",
    "summarize_results",
]
