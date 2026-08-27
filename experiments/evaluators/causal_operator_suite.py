"""Held-out execution suite for cross-domain causal operator learning.

Challenges expose an origin, a declarative goal, typed primitive schemas, and
an execution ceiling.  They never contain a route, grounded action sequence,
or solution trace.  Learner submissions are frozen before this evaluator
executes each action through the owning domain and judges only the observed
terminal state with that domain's verifier.

Every domain contributes an unseen-binding two-step relocation case and a
four-step case that requires composing two such retained procedural chunks.
The suite constructs state pairs directly; it never constructs or retains the
actions that connect them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    State,
    Trace,
)
from angler.worlds import relational_boxes as boxes
from angler.worlds import relational_files as files
from angler.worlds import relational_tokens as tokens


DomainName = Literal["tokens", "files", "boxes"]
SUPPORTED_DOMAINS: tuple[DomainName, ...] = ("tokens", "files", "boxes")
DEFAULT_CASES_PER_DOMAIN = 2
SINGLE_OPERATOR_STEPS = 2
COMPOSED_OPERATOR_STEPS = 4
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUITE_VERSION = "angler.causal-operator-suite.v1"


@dataclass(frozen=True, slots=True)
class OperatorChallenge:
    """Complete learner-visible projection of one held-out challenge."""

    case_id: str
    domain: DomainName
    origin: State
    goal: Goal
    allowed_action_schemas: tuple[ActionSchema, ...]
    maximum_steps: int

    def __post_init__(self) -> None:
        _require_digest(self.case_id, "challenge case_id")
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError("challenge domain is unsupported")
        if not isinstance(self.origin, State) or not isinstance(self.goal, Goal):
            raise TypeError("challenge origin and goal must be relational records")
        if self.origin.namespace != self.goal.namespace:
            raise ValueError("challenge origin and goal must share a namespace")
        if type(self.allowed_action_schemas) is not tuple:
            raise TypeError("allowed_action_schemas must be an immutable tuple")
        if not self.allowed_action_schemas or any(
            not isinstance(item, ActionSchema)
            for item in self.allowed_action_schemas
        ):
            raise TypeError("allowed_action_schemas must contain action schemas")
        if len(set(self.allowed_action_schemas)) != len(
            self.allowed_action_schemas
        ):
            raise ValueError("allowed_action_schemas must be unique")
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or self.maximum_steps <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer")


@dataclass(frozen=True, slots=True)
class CommittedActionSequence:
    """One immutable learner submission bound to exactly one challenge."""

    challenge_id: str
    actions: tuple[GroundAction, ...]

    def __post_init__(self) -> None:
        _require_digest(self.challenge_id, "commitment challenge_id")
        if type(self.actions) is not tuple:
            raise TypeError("committed actions must be an immutable tuple")
        if any(not isinstance(item, GroundAction) for item in self.actions):
            raise TypeError("committed actions must contain GroundAction values")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "actions": [item.digest for item in self.actions],
                "challenge_id": self.challenge_id,
            }
        )


@dataclass(frozen=True, slots=True)
class OperatorCaseResult:
    """Objective execution feedback for one frozen submission."""

    case_id: str
    domain: DomainName
    success: bool
    tool_calls: int
    applied_actions: int
    commitment_digest: str
    trace: Trace

    def __post_init__(self) -> None:
        _require_digest(self.case_id, "result case_id")
        _require_digest(self.commitment_digest, "result commitment_digest")
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError("result domain is unsupported")
        if type(self.success) is not bool:
            raise TypeError("result success must be bool")
        for value, label in (
            (self.tool_calls, "tool_calls"),
            (self.applied_actions, "applied_actions"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.applied_actions > self.tool_calls:
            raise ValueError("applied_actions cannot exceed tool_calls")
        if not isinstance(self.trace, Trace):
            raise TypeError("result trace must be an executed Trace")
        if len(self.trace.transitions) != self.tool_calls:
            raise ValueError("tool_calls must equal the executed trace length")
        if sum(item.applied for item in self.trace.transitions) != self.applied_actions:
            raise ValueError("applied_actions must match the executed trace")

    @property
    def final_state(self) -> State:
        return self.trace.final_state

    @property
    def trace_digest(self) -> str:
        return self.trace.digest

    @property
    def digest(self) -> str:
        return _digest(
            {
                "applied_actions": self.applied_actions,
                "case_id": self.case_id,
                "commitment_digest": self.commitment_digest,
                "domain": self.domain,
                "success": self.success,
                "tool_calls": self.tool_calls,
                "trace_digest": self.trace.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class DomainOperatorSummary:
    """Success and real executor-call totals for one domain."""

    domain: DomainName
    attempts: int
    successes: int
    success_rate: float
    total_tool_calls: int
    mean_tool_calls: float


@dataclass(frozen=True, slots=True)
class OperatorSuiteSummary:
    """Aggregate held-out performance without re-executing submissions."""

    attempts: int
    successes: int
    success_rate: float
    total_tool_calls: int
    mean_tool_calls: float
    by_domain: tuple[DomainOperatorSummary, ...]


def make_heldout_operator_suite(
    seed: int,
    *,
    cases_per_domain: int = DEFAULT_CASES_PER_DOMAIN,
) -> tuple[OperatorChallenge, ...]:
    """Create deterministic unseen-binding cases without constructing routes.

    At least two cases per domain are required so every suite contains both a
    two-step held-out application and a four-step composition challenge.
    Additional cases alternate those two declarative shapes with fresh entity
    bindings.
    """

    _validate_seed(seed)
    if (
        isinstance(cases_per_domain, bool)
        or not isinstance(cases_per_domain, int)
    ):
        raise TypeError("cases_per_domain must be an integer")
    if cases_per_domain < 2:
        raise ValueError(
            "cases_per_domain must be at least two to retain composition coverage"
        )

    challenges: list[OperatorChallenge] = []
    for domain in SUPPORTED_DOMAINS:
        for case_index in range(cases_per_domain):
            step_count = (
                SINGLE_OPERATOR_STEPS
                if case_index % 2 == 0
                else COMPOSED_OPERATOR_STEPS
            )
            entities = tuple(
                _heldout_entity(seed, domain, case_index, entity_index)
                for entity_index in range(step_count)
            )
            origin, goal, schema = _make_state_pair(domain, entities)
            case_id = _case_id(
                domain,
                origin,
                goal,
                (schema,),
                step_count,
            )
            challenges.append(
                OperatorChallenge(
                    case_id=case_id,
                    domain=domain,
                    origin=origin,
                    goal=goal,
                    allowed_action_schemas=(schema,),
                    maximum_steps=step_count,
                )
            )
    return tuple(challenges)


def commit_action_sequence(
    challenge: OperatorChallenge,
    actions: Sequence[GroundAction],
) -> CommittedActionSequence:
    """Freeze one complete proposal before any domain action is executed."""

    _validate_challenge(challenge)
    if isinstance(actions, (str, bytes, bytearray)) or not isinstance(
        actions,
        Sequence,
    ):
        raise TypeError("actions must be a finite sequence")
    frozen = tuple(actions)
    if len(frozen) > challenge.maximum_steps:
        raise ValueError("action sequence exceeds the challenge execution ceiling")
    allowed = set(challenge.allowed_action_schemas)
    if any(not isinstance(item, GroundAction) for item in frozen):
        raise TypeError("actions must contain GroundAction values")
    if any(item.schema not in allowed for item in frozen):
        raise ValueError("action sequence contains a schema not allowed by challenge")
    return CommittedActionSequence(challenge.case_id, frozen)


def evaluate_committed_sequence(
    challenge: OperatorChallenge,
    commitment: CommittedActionSequence,
) -> OperatorCaseResult:
    """Execute a frozen submission exactly once, then verify terminal facts."""

    _validate_challenge(challenge)
    if not isinstance(commitment, CommittedActionSequence):
        raise TypeError("commitment must be a CommittedActionSequence")
    if commitment.challenge_id != challenge.case_id:
        raise ValueError("commitment is bound to a different challenge")
    if len(commitment.actions) > challenge.maximum_steps:
        raise ValueError("commitment exceeds the challenge execution ceiling")
    allowed = set(challenge.allowed_action_schemas)
    if any(action.schema not in allowed for action in commitment.actions):
        raise ValueError("commitment contains a schema not allowed by challenge")

    executor, verifier = _execution_boundary(challenge.domain)
    state = challenge.origin
    transitions = []
    for action in commitment.actions:
        transition = executor(state, action)
        transitions.append(transition)
        state = transition.after
    trace = Trace(challenge.origin, tuple(transitions), challenge.goal)
    success = verifier(trace.final_state, challenge.goal)
    return OperatorCaseResult(
        case_id=challenge.case_id,
        domain=challenge.domain,
        success=success,
        tool_calls=len(commitment.actions),
        applied_actions=sum(item.applied for item in transitions),
        commitment_digest=commitment.digest,
        trace=trace,
    )


def summarize_operator_results(
    results: Iterable[OperatorCaseResult],
) -> OperatorSuiteSummary:
    """Aggregate success and tool calls without replaying any action."""

    ordered = tuple(results)
    if not ordered:
        raise ValueError("results must not be empty")
    if any(not isinstance(item, OperatorCaseResult) for item in ordered):
        raise TypeError("results must contain OperatorCaseResult values")
    case_ids = tuple(item.case_id for item in ordered)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("results contain a duplicate case_id")

    by_domain = tuple(
        _summarize_domain(domain, ordered)
        for domain in SUPPORTED_DOMAINS
        if any(item.domain == domain for item in ordered)
    )
    attempts = len(ordered)
    successes = sum(item.success for item in ordered)
    total_tool_calls = sum(item.tool_calls for item in ordered)
    return OperatorSuiteSummary(
        attempts=attempts,
        successes=successes,
        success_rate=successes / attempts,
        total_tool_calls=total_tool_calls,
        mean_tool_calls=total_tool_calls / attempts,
        by_domain=by_domain,
    )


def _make_state_pair(
    domain: DomainName,
    entities: tuple[str, ...],
) -> tuple[State, Goal, ActionSchema]:
    position_count = len(entities) + 1
    if domain == "tokens":
        origin = tokens.make_token_state(entities + (None,))
        goal = tokens.make_token_goal((None,) + entities)
        return origin, goal, tokens.MOVE_TOKEN

    placements = tuple(
        (entity, _position(index))
        for index, entity in enumerate(entities)
    )
    target = tuple(
        (entity, _position(index + 1))
        for index, entity in enumerate(entities)
    )
    if domain == "files":
        links = tuple(
            pair
            for index in range(position_count - 1)
            for pair in (
                (_position(index), _position(index + 1)),
                (_position(index + 1), _position(index)),
            )
        )
        origin = files.make_file_state(placements, links)
        target_state = files.make_file_state(target, links)
        goal = Goal(
            namespace=files.NAMESPACE,
            required=target_state.records,
            exact=True,
        )
        return origin, goal, files.RELOCATE_FILE

    if domain == "boxes":
        origin_contents = {
            _position(index): ((entities[index],) if index < len(entities) else ())
            for index in range(position_count)
        }
        target_contents = {
            _position(index): (() if index == 0 else (entities[index - 1],))
            for index in range(position_count)
        }
        capacities = {
            _position(index): 1
            for index in range(position_count)
        }
        origin = boxes.make_box_state(origin_contents, capacities)
        goal = boxes.make_box_goal(target_contents, capacities)
        return origin, goal, boxes.TRANSFER_ITEM

    raise ValueError("unsupported causal-operator domain")


def _execution_boundary(domain: DomainName):
    if domain == "tokens":
        return tokens.execute_token_action, tokens.verify_token_goal
    if domain == "files":
        return files.execute_file_action, files.verify_file_goal
    if domain == "boxes":
        return boxes.execute_box_action, boxes.verify_box_goal
    raise ValueError("unsupported causal-operator domain")


def _expected_boundary(domain: DomainName) -> tuple[str, ActionSchema]:
    if domain == "tokens":
        return tokens.NAMESPACE, tokens.MOVE_TOKEN
    if domain == "files":
        return files.NAMESPACE, files.RELOCATE_FILE
    if domain == "boxes":
        return boxes.NAMESPACE, boxes.TRANSFER_ITEM
    raise ValueError("unsupported causal-operator domain")


def _validate_challenge(challenge: OperatorChallenge) -> None:
    if not isinstance(challenge, OperatorChallenge):
        raise TypeError("challenge must be an OperatorChallenge")
    namespace, schema = _expected_boundary(challenge.domain)
    if challenge.origin.namespace != namespace or challenge.goal.namespace != namespace:
        raise ValueError("challenge records do not match its declared domain")
    if challenge.allowed_action_schemas != (schema,):
        raise ValueError("challenge action vocabulary does not match its domain")
    expected_id = _case_id(
        challenge.domain,
        challenge.origin,
        challenge.goal,
        challenge.allowed_action_schemas,
        challenge.maximum_steps,
    )
    if challenge.case_id != expected_id:
        raise ValueError("challenge case_id does not match its public content")
    _, verifier = _execution_boundary(challenge.domain)
    if verifier(challenge.origin, challenge.goal):
        raise ValueError("challenge origin must not already satisfy its goal")


def _summarize_domain(
    domain: DomainName,
    results: tuple[OperatorCaseResult, ...],
) -> DomainOperatorSummary:
    selected = tuple(item for item in results if item.domain == domain)
    attempts = len(selected)
    successes = sum(item.success for item in selected)
    total_tool_calls = sum(item.tool_calls for item in selected)
    return DomainOperatorSummary(
        domain=domain,
        attempts=attempts,
        successes=successes,
        success_rate=successes / attempts,
        total_tool_calls=total_tool_calls,
        mean_tool_calls=total_tool_calls / attempts,
    )


def _case_id(
    domain: DomainName,
    origin: State,
    goal: Goal,
    schemas: tuple[ActionSchema, ...],
    maximum_steps: int,
) -> str:
    return _digest(
        {
            "domain": domain,
            "goal": goal.digest,
            "maximum_steps": maximum_steps,
            "origin": origin.digest,
            "schemas": [item.digest for item in schemas],
        }
    )


def _heldout_entity(
    seed: int,
    domain: DomainName,
    case_index: int,
    entity_index: int,
) -> str:
    material = (
        f"{_SUITE_VERSION}\x00{seed}\x00{domain}\x00{case_index}\x00{entity_index}"
    ).encode("utf-8")
    suffix = hashlib.sha256(material).hexdigest()[:16]
    return f"heldout_{domain}_{suffix}"


def _position(index: int) -> str:
    return f"position_{index}"


def _digest(payload: dict[str, object]) -> str:
    material = {
        "suite": _SUITE_VERSION,
        "payload": payload,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


__all__ = [
    "COMPOSED_OPERATOR_STEPS",
    "CommittedActionSequence",
    "DEFAULT_CASES_PER_DOMAIN",
    "DomainOperatorSummary",
    "OperatorCaseResult",
    "OperatorChallenge",
    "OperatorSuiteSummary",
    "SINGLE_OPERATOR_STEPS",
    "SUPPORTED_DOMAINS",
    "commit_action_sequence",
    "evaluate_committed_sequence",
    "make_heldout_operator_suite",
    "summarize_operator_results",
]
