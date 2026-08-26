"""Deterministic reversible transitions for procedure-learning experiments.

This module defines environment physics and terminal judging only.  It exposes
adjacent swaps so a learner can observe forward/inverse transition pairs, but
it deliberately provides no path, distance, next-action, or solving API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import random


FAMILY_ID = "angler.reversible-transition"
FAMILY_VERSION = "1.0.0"
TOKEN_COUNT = 6
ACTION_COUNT = TOKEN_COUNT - 1
DEFAULT_MAX_STEPS = 24
MAX_PROCEDURE_STEPS = 64

PermutationState = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReversibleTransitionTask:
    """Immutable learner-visible origin and primitive-action vocabulary."""

    instance_id: str
    origin: PermutationState
    available_actions: tuple[int, ...]
    max_steps: int
    generation_commitment: str


@dataclass(frozen=True, slots=True)
class PrimitiveTransition:
    """One observed reversible environment transition."""

    before: PermutationState
    action: int
    after: PermutationState
    inverse_action: int


@dataclass(frozen=True, slots=True)
class CommittedProcedure:
    """A complete immutable procedure bound to one task identity."""

    task_id: str
    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.actions, tuple):
            raise TypeError("actions must be an immutable tuple")
        for action in self.actions:
            _validate_action(action)


@dataclass(frozen=True, slots=True)
class ProcedureExecution:
    """Terminal execution report with no intermediate trace or hint."""

    task_id: str
    reached_state: PermutationState
    exact: bool
    steps_executed: int


def generate_reversible_transition_task(
    seed: int,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ReversibleTransitionTask:
    """Generate a replayable public origin without generating a solution."""

    if type(seed) is not int:
        raise TypeError("seed must be an int")
    _validate_max_steps(max_steps)

    rng = random.Random(_domain_seed(seed, "origin"))
    origin = list(range(TOKEN_COUNT))
    rng.shuffle(origin)
    origin_state = tuple(origin)
    available_actions = tuple(range(ACTION_COUNT))
    generation_commitment = "sha256:" + hashlib.sha256(
        f"{FAMILY_ID}@{FAMILY_VERSION}\x00generation\x00{seed}".encode("utf-8")
    ).hexdigest()
    instance_id = _task_identity(
        origin_state,
        available_actions,
        max_steps,
        generation_commitment,
    )
    return ReversibleTransitionTask(
        instance_id=instance_id,
        origin=origin_state,
        available_actions=available_actions,
        max_steps=max_steps,
        generation_commitment=generation_commitment,
    )


def _task_identity(
    origin: PermutationState,
    available_actions: tuple[int, ...],
    max_steps: int,
    generation_commitment: str,
) -> str:
    identity_material = {
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "origin": origin,
        "available_actions": available_actions,
        "max_steps": max_steps,
        "generation_commitment": generation_commitment,
    }
    encoded = json.dumps(
        identity_material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def observe_primitive_transition(
    state: Sequence[int],
    action: int,
) -> PrimitiveTransition:
    """Apply one adjacent swap and return its self-inverse observation."""

    before = _validate_state(state)
    _validate_action(action)
    swapped = list(before)
    swapped[action], swapped[action + 1] = (
        swapped[action + 1],
        swapped[action],
    )
    return PrimitiveTransition(
        before=before,
        action=action,
        after=tuple(swapped),
        inverse_action=action,
    )


def commit_procedure(
    task: ReversibleTransitionTask,
    actions: Sequence[int],
) -> CommittedProcedure:
    """Validate and freeze a complete action sequence before execution."""

    _validate_task(task)
    if not isinstance(actions, Sequence) or isinstance(
        actions,
        (str, bytes, bytearray),
    ):
        raise TypeError("actions must be a finite sequence")
    committed_actions = tuple(actions)
    if len(committed_actions) > task.max_steps:
        raise ValueError("procedure exceeds the task step budget")
    for action in committed_actions:
        _validate_action(action)
        if action not in task.available_actions:
            raise ValueError("action is unavailable for this task")
    return CommittedProcedure(task.instance_id, committed_actions)


def execute_committed_procedure(
    task: ReversibleTransitionTask,
    procedure: CommittedProcedure,
    verification_target: Sequence[int],
) -> ProcedureExecution:
    """Atomically execute a committed procedure, then compare its endpoint."""

    _validate_task(task)
    if not isinstance(procedure, CommittedProcedure):
        raise TypeError("procedure must be a CommittedProcedure")
    if procedure.task_id != task.instance_id:
        raise ValueError("procedure and task identities do not match")
    if len(procedure.actions) > task.max_steps:
        raise ValueError("procedure exceeds the task step budget")

    # Validate every input before the local execution state is created.  An
    # invalid procedure therefore cannot yield a partial execution result.
    target = _validate_state(verification_target)
    for action in procedure.actions:
        _validate_action(action)
        if action not in task.available_actions:
            raise ValueError("action is unavailable for this task")

    reached = task.origin
    for action in procedure.actions:
        reached = observe_primitive_transition(reached, action).after

    return ProcedureExecution(
        task_id=task.instance_id,
        reached_state=reached,
        exact=reached == target,
        steps_executed=len(procedure.actions),
    )


def _validate_task(task: ReversibleTransitionTask) -> None:
    if not isinstance(task, ReversibleTransitionTask):
        raise TypeError("task must be a ReversibleTransitionTask")
    if not isinstance(task.instance_id, str) or not task.instance_id.startswith(
        "sha256:"
    ):
        raise ValueError("task instance_id must be a sha256 identity")
    if len(task.instance_id) != len("sha256:") + 64:
        raise ValueError("task instance_id must contain a full sha256 digest")
    try:
        int(task.instance_id.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError("task instance_id must contain hexadecimal digits") from error
    _validate_state(task.origin)
    if task.available_actions != tuple(range(ACTION_COUNT)):
        raise ValueError("task action vocabulary is invalid")
    _validate_max_steps(task.max_steps)
    _validate_digest(task.generation_commitment, "generation_commitment")
    expected_identity = _task_identity(
        task.origin,
        task.available_actions,
        task.max_steps,
        task.generation_commitment,
    )
    if task.instance_id != expected_identity:
        raise ValueError("task instance_id does not match its immutable content")


def _validate_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be a sha256 identity")
    if len(value) != len("sha256:") + 64:
        raise ValueError(f"{name} must contain a full sha256 digest")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"{name} must contain hexadecimal digits") from error


def _validate_state(state: Sequence[int]) -> PermutationState:
    if not isinstance(state, Sequence) or isinstance(
        state,
        (str, bytes, bytearray),
    ):
        raise TypeError("state must be a finite integer sequence")
    normalized = tuple(state)
    if any(type(token) is not int for token in normalized):
        raise TypeError("state tokens must be ints")
    if len(normalized) != TOKEN_COUNT or set(normalized) != set(range(TOKEN_COUNT)):
        raise ValueError(
            f"state must be a permutation of integers 0 through {TOKEN_COUNT - 1}"
        )
    return normalized


def _validate_action(action: int) -> None:
    if type(action) is not int:
        raise TypeError("action must be an int")
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(
            f"action must be an adjacent-swap index from 0 through {ACTION_COUNT - 1}"
        )


def _validate_max_steps(max_steps: int) -> None:
    if type(max_steps) is not int:
        raise TypeError("max_steps must be an int")
    if not 0 <= max_steps <= MAX_PROCEDURE_STEPS:
        raise ValueError(
            f"max_steps must be between 0 and {MAX_PROCEDURE_STEPS}"
        )


def _domain_seed(seed: int, domain: str) -> int:
    material = (
        f"{FAMILY_ID}@{FAMILY_VERSION}\x00{seed}\x00{domain}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


__all__ = [
    "ACTION_COUNT",
    "DEFAULT_MAX_STEPS",
    "FAMILY_ID",
    "FAMILY_VERSION",
    "MAX_PROCEDURE_STEPS",
    "TOKEN_COUNT",
    "CommittedProcedure",
    "PermutationState",
    "PrimitiveTransition",
    "ProcedureExecution",
    "ReversibleTransitionTask",
    "commit_procedure",
    "execute_committed_procedure",
    "generate_reversible_transition_task",
    "observe_primitive_transition",
]
