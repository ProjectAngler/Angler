"""Bounded early-teacher search over learned symbolic operator mirrors.

The search proposes grounded learned bodies, but only a caller-supplied trial
boundary can produce state changes or success evidence.  This module imports
no world or evaluator, performs no domain transition itself, and never treats
predicted effects as observed facts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Literal, Protocol

from angler.procedures.grounding import (
    DEFAULT_MAX_BINDINGS,
    DEFAULT_MAX_MATCH_ATTEMPTS,
    GroundedOperatorPrediction,
    GroundingError,
    StateOperatorBinding,
    enumerate_operator_bindings,
    instantiate_operator,
    score_goal_effect_overlap,
)
from angler.procedures.operators import LearnedOperator
from angler.procedures.records import Goal, GroundAction, State


DEFAULT_MAX_OPERATOR_DEPTH = 2
DEFAULT_MAX_EXPANSIONS = 64
HARD_MAX_OPERATOR_DEPTH = 8
HARD_MAX_EXPANSIONS = 4_096
HARD_MAX_OPERATORS = 256
_EXPERT_VERSION = "angler.expert-iteration.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
Termination = Literal["verified", "expansion_limit", "exhausted"]


class ExpertIterationError(ValueError):
    """Raised when the teacher boundary or search inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class TrialRequest:
    """One immutable learned-body proposal submitted to an external boundary."""

    trial_index: int
    operator_depth: int
    parent_trial_digest: str | None
    origin: State
    goal: Goal
    binding: StateOperatorBinding
    actions: tuple[GroundAction, ...]
    goal_effect_overlap: int | None

    def __post_init__(self) -> None:
        for value, label in (
            (self.trial_index, "trial_index"),
            (self.operator_depth, "operator_depth"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExpertIterationError(f"{label} must be a positive integer")
        if self.parent_trial_digest is not None:
            _require_digest(self.parent_trial_digest, "parent_trial_digest")
        if not isinstance(self.origin, State) or not isinstance(self.goal, Goal):
            raise TypeError("trial origin and goal must be relational records")
        if self.origin.namespace != self.goal.namespace:
            raise ExpertIterationError("trial origin and goal must share a namespace")
        if not isinstance(self.binding, StateOperatorBinding):
            raise TypeError("trial binding must be a StateOperatorBinding")
        if self.binding.namespace != self.origin.namespace:
            raise ExpertIterationError("trial binding must match the state namespace")
        if type(self.actions) is not tuple or not self.actions:
            raise ExpertIterationError("trial actions must be a non-empty tuple")
        if any(not isinstance(item, GroundAction) for item in self.actions):
            raise TypeError("trial actions must contain GroundAction values")
        if any(item.namespace != self.origin.namespace for item in self.actions):
            raise ExpertIterationError("trial actions must match the state namespace")
        if self.goal_effect_overlap is not None and (
            isinstance(self.goal_effect_overlap, bool)
            or not isinstance(self.goal_effect_overlap, int)
        ):
            raise TypeError("goal_effect_overlap must be an integer or None")

    @property
    def digest(self) -> str:
        return _digest(
            "trial_request",
            {
                "actions": [item.digest for item in self.actions],
                "binding": self.binding.digest,
                "goal": self.goal.digest,
                "goal_effect_overlap": self.goal_effect_overlap,
                "operator_depth": self.operator_depth,
                "origin": self.origin.digest,
                "parent_trial_digest": self.parent_trial_digest,
                "trial_index": self.trial_index,
            },
        )


@dataclass(frozen=True, slots=True)
class TrialEvidence:
    """Immutable observation returned by the caller-owned trial boundary."""

    request_digest: str
    observed_state: State
    success: bool
    applied_actions: int
    cost: int

    def __post_init__(self) -> None:
        _require_digest(self.request_digest, "evidence request_digest")
        if not isinstance(self.observed_state, State):
            raise TypeError("evidence observed_state must be a State")
        if type(self.success) is not bool:
            raise TypeError("evidence success must be bool")
        for value, label in (
            (self.applied_actions, "applied_actions"),
            (self.cost, "cost"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExpertIterationError(
                    f"evidence {label} must be a non-negative integer"
                )

    @property
    def digest(self) -> str:
        return _digest(
            "trial_evidence",
            {
                "applied_actions": self.applied_actions,
                "cost": self.cost,
                "observed_state": self.observed_state.digest,
                "request_digest": self.request_digest,
                "success": self.success,
            },
        )


class ExternalTrialCallback(Protocol):
    """Caller-owned immutable proposal/evidence boundary."""

    def __call__(self, request: TrialRequest) -> TrialEvidence:
        """Submit one proposal once and return its observed evidence."""


@dataclass(frozen=True, slots=True)
class TeacherTrial:
    """One exact proposal paired with the evidence returned for it."""

    request: TrialRequest
    evidence: TrialEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.request, TrialRequest):
            raise TypeError("teacher trial request must be a TrialRequest")
        if not isinstance(self.evidence, TrialEvidence):
            raise TypeError("teacher trial evidence must be TrialEvidence")
        _validate_evidence(self.request, self.evidence)

    @property
    def digest(self) -> str:
        return _digest(
            "teacher_trial",
            {
                "evidence": self.evidence.digest,
                "request": self.request.digest,
            },
        )


@dataclass(frozen=True, slots=True)
class TeacherPlan:
    """Externally verified operator/binding/action chain plus all search costs."""

    initial: State
    goal: Goal
    chain: tuple[TeacherTrial, ...]
    accounting: tuple[TeacherTrial, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initial, State) or not isinstance(self.goal, Goal):
            raise TypeError("teacher plan endpoints must be relational records")
        if self.initial.namespace != self.goal.namespace:
            raise ExpertIterationError("teacher plan endpoints must share a namespace")
        if type(self.chain) is not tuple or not self.chain:
            raise ExpertIterationError("verified teacher plan requires a non-empty chain")
        if type(self.accounting) is not tuple or not self.accounting:
            raise ExpertIterationError("teacher plan requires complete trial accounting")
        if any(not isinstance(item, TeacherTrial) for item in self.chain + self.accounting):
            raise TypeError("teacher plan entries must be TeacherTrial values")
        accounting_digests = tuple(item.digest for item in self.accounting)
        if len(set(accounting_digests)) != len(accounting_digests):
            raise ExpertIterationError("teacher accounting cannot repeat a trial")
        if any(item.digest not in set(accounting_digests) for item in self.chain):
            raise ExpertIterationError("teacher chain must be contained in accounting")
        current = self.initial
        for index, trial in enumerate(self.chain):
            if trial.request.origin != current:
                raise ExpertIterationError("teacher chain states must be contiguous")
            if trial.request.goal != self.goal:
                raise ExpertIterationError("teacher chain trials must share its goal")
            if trial.request.operator_depth != index + 1:
                raise ExpertIterationError("teacher chain depth must be contiguous")
            if index and trial.request.parent_trial_digest != self.chain[index - 1].digest:
                raise ExpertIterationError("teacher chain parent evidence is inconsistent")
            if not index and trial.request.parent_trial_digest is not None:
                raise ExpertIterationError("first teacher step cannot have a parent")
            if index < len(self.chain) - 1 and trial.evidence.success:
                raise ExpertIterationError("teacher search must stop at first success")
            current = trial.evidence.observed_state
        terminal = self.chain[-1]
        if not terminal.evidence.success:
            raise ExpertIterationError("teacher plan requires external success evidence")
        if not _state_satisfies_goal(terminal.evidence.observed_state, self.goal):
            raise ExpertIterationError(
                "external success evidence does not satisfy the declarative goal"
            )

    @property
    def expansions(self) -> int:
        return len(self.accounting)

    @property
    def total_cost(self) -> int:
        return sum(item.evidence.cost for item in self.accounting)

    @property
    def total_applied_actions(self) -> int:
        return sum(item.evidence.applied_actions for item in self.accounting)

    @property
    def digest(self) -> str:
        return _digest(
            "teacher_plan",
            {
                "accounting": [item.digest for item in self.accounting],
                "chain": [item.digest for item in self.chain],
                "goal": self.goal.digest,
                "initial": self.initial.digest,
            },
        )


@dataclass(frozen=True, slots=True)
class TeacherSearchResult:
    """Verified plan or bounded failure, always retaining every real trial."""

    plan: TeacherPlan | None
    trials: tuple[TeacherTrial, ...]
    termination: Termination

    def __post_init__(self) -> None:
        if self.plan is not None and not isinstance(self.plan, TeacherPlan):
            raise TypeError("search plan must be a TeacherPlan or None")
        if type(self.trials) is not tuple or any(
            not isinstance(item, TeacherTrial) for item in self.trials
        ):
            raise TypeError("search trials must be an immutable TeacherTrial tuple")
        if len({item.digest for item in self.trials}) != len(self.trials):
            raise ExpertIterationError("search trials cannot contain duplicates")
        if self.termination not in ("verified", "expansion_limit", "exhausted"):
            raise ExpertIterationError("search termination is unsupported")
        if self.termination == "verified":
            if self.plan is None or self.plan.accounting != self.trials:
                raise ExpertIterationError(
                    "verified result requires a plan with exact accounting"
                )
        elif self.plan is not None:
            raise ExpertIterationError("unverified result cannot contain a plan")

    @property
    def expansions(self) -> int:
        return len(self.trials)

    @property
    def total_cost(self) -> int:
        return sum(item.evidence.cost for item in self.trials)

    @property
    def total_applied_actions(self) -> int:
        return sum(item.evidence.applied_actions for item in self.trials)


@dataclass(frozen=True, slots=True)
class _Candidate:
    operator_index: int
    binding: StateOperatorBinding
    prediction: GroundedOperatorPrediction
    overlap: int | None


@dataclass(frozen=True, slots=True)
class _SearchNode:
    state: State
    chain: tuple[TeacherTrial, ...]

    @property
    def depth(self) -> int:
        return len(self.chain)


def search_teacher_plan(
    initial: State,
    goal: Goal,
    operators: Sequence[LearnedOperator],
    trial_callback: ExternalTrialCallback | Callable[[TrialRequest], TrialEvidence],
    *,
    maximum_operator_depth: int = DEFAULT_MAX_OPERATOR_DEPTH,
    maximum_expansions: int = DEFAULT_MAX_EXPANSIONS,
    maximum_bindings_per_operator: int = DEFAULT_MAX_BINDINGS,
    maximum_match_attempts: int = DEFAULT_MAX_MATCH_ATTEMPTS,
    order_by_goal_effect_overlap: bool = True,
) -> TeacherSearchResult:
    """Search learned operator applications under external observed feedback."""

    ordered_operators = _validate_search_inputs(
        initial,
        goal,
        operators,
        trial_callback,
        maximum_operator_depth=maximum_operator_depth,
        maximum_expansions=maximum_expansions,
        order_by_goal_effect_overlap=order_by_goal_effect_overlap,
    )
    queue = deque((_SearchNode(initial, ()),))
    seen_states = {initial.digest}
    attempted_actions: set[tuple[str, tuple[str, ...]]] = set()
    trials: list[TeacherTrial] = []

    while queue:
        node = queue.popleft()
        if node.depth >= maximum_operator_depth:
            continue
        candidates = _ground_candidates(
            node.state,
            goal,
            ordered_operators,
            maximum_bindings_per_operator=maximum_bindings_per_operator,
            maximum_match_attempts=maximum_match_attempts,
            order_by_goal_effect_overlap=order_by_goal_effect_overlap,
        )
        for candidate in candidates:
            action_key = (
                node.state.digest,
                tuple(item.digest for item in candidate.prediction.actions),
            )
            if action_key in attempted_actions:
                continue
            attempted_actions.add(action_key)
            if len(trials) >= maximum_expansions:
                return TeacherSearchResult(None, tuple(trials), "expansion_limit")

            request = TrialRequest(
                trial_index=len(trials) + 1,
                operator_depth=node.depth + 1,
                parent_trial_digest=(
                    None if not node.chain else node.chain[-1].digest
                ),
                origin=node.state,
                goal=goal,
                binding=candidate.binding,
                actions=candidate.prediction.actions,
                goal_effect_overlap=candidate.overlap,
            )
            evidence = trial_callback(request)
            if not isinstance(evidence, TrialEvidence):
                raise TypeError("trial callback must return TrialEvidence")
            trial = TeacherTrial(request, evidence)
            trials.append(trial)

            if evidence.success:
                if not _state_satisfies_goal(evidence.observed_state, goal):
                    raise ExpertIterationError(
                        "trial callback claimed success outside the goal"
                    )
                plan = TeacherPlan(
                    initial=initial,
                    goal=goal,
                    chain=node.chain + (trial,),
                    accounting=tuple(trials),
                )
                return TeacherSearchResult(plan, tuple(trials), "verified")

            if (
                node.depth + 1 < maximum_operator_depth
                and evidence.observed_state.digest not in seen_states
            ):
                seen_states.add(evidence.observed_state.digest)
                queue.append(_SearchNode(evidence.observed_state, node.chain + (trial,)))

    return TeacherSearchResult(None, tuple(trials), "exhausted")


def _ground_candidates(
    state: State,
    goal: Goal,
    operators: tuple[LearnedOperator, ...],
    *,
    maximum_bindings_per_operator: int,
    maximum_match_attempts: int,
    order_by_goal_effect_overlap: bool,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for operator_index, operator in enumerate(operators):
        bindings = enumerate_operator_bindings(
            operator,
            state,
            maximum_bindings=maximum_bindings_per_operator,
            maximum_match_attempts=maximum_match_attempts,
        )
        for binding in bindings:
            # Relational matching may legally enumerate a co-reference that
            # collapses a learned add/delete pair.  That is not an executable
            # operator instance, so omit it as an invalid proposal instead of
            # aborting the bounded search over the remaining bindings.
            try:
                prediction = instantiate_operator(operator, binding)
            except GroundingError:
                continue
            overlap = (
                score_goal_effect_overlap(prediction, goal)
                if order_by_goal_effect_overlap
                else None
            )
            candidates.append(
                _Candidate(operator_index, binding, prediction, overlap)
            )
    if order_by_goal_effect_overlap:
        candidates.sort(
            key=lambda item: (
                -int(item.overlap),
                item.operator_index,
                item.binding.digest,
            )
        )
    return tuple(candidates)


def _validate_search_inputs(
    initial: State,
    goal: Goal,
    operators: Sequence[LearnedOperator],
    trial_callback: ExternalTrialCallback | Callable[[TrialRequest], TrialEvidence],
    *,
    maximum_operator_depth: int,
    maximum_expansions: int,
    order_by_goal_effect_overlap: bool,
) -> tuple[LearnedOperator, ...]:
    if not isinstance(initial, State) or not isinstance(goal, Goal):
        raise TypeError("search initial and goal must be relational records")
    if initial.namespace != goal.namespace:
        raise ExpertIterationError("search initial and goal must share a namespace")
    if isinstance(operators, (str, bytes, bytearray)) or not isinstance(
        operators,
        Sequence,
    ):
        raise TypeError("operators must be a finite sequence")
    frozen = tuple(operators)
    if any(not isinstance(item, LearnedOperator) for item in frozen):
        raise TypeError("operators must contain LearnedOperator values")
    if len(frozen) > HARD_MAX_OPERATORS:
        raise ExpertIterationError("operators exceed the hard ceiling")
    if len({item.digest for item in frozen}) != len(frozen):
        raise ExpertIterationError("operators must be unique revisions")
    if any(item.namespace != initial.namespace for item in frozen):
        raise ExpertIterationError("operators must match the search namespace")
    if not callable(trial_callback):
        raise TypeError("trial_callback must be callable")
    _validate_budget(
        maximum_operator_depth,
        "maximum_operator_depth",
        HARD_MAX_OPERATOR_DEPTH,
    )
    _validate_budget(
        maximum_expansions,
        "maximum_expansions",
        HARD_MAX_EXPANSIONS,
    )
    if not isinstance(order_by_goal_effect_overlap, bool):
        raise TypeError("order_by_goal_effect_overlap must be bool")
    return frozen


def _validate_evidence(request: TrialRequest, evidence: TrialEvidence) -> None:
    if evidence.request_digest != request.digest:
        raise ExpertIterationError("trial evidence is bound to another request")
    if evidence.observed_state.namespace != request.origin.namespace:
        raise ExpertIterationError("trial evidence changed the state namespace")
    if evidence.applied_actions > len(request.actions):
        raise ExpertIterationError(
            "trial evidence applied_actions exceeds the submitted sequence"
        )


def _state_satisfies_goal(state: State, goal: Goal) -> bool:
    if state.namespace != goal.namespace:
        return False
    if goal.exact:
        return state.records == goal.required
    records = set(state.records)
    return set(goal.required) <= records and not (set(goal.forbidden) & records)


def _validate_budget(value: int, label: str, hard_maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0 or value > hard_maximum:
        raise ExpertIterationError(
            f"{label} must be between 1 and {hard_maximum}"
        )


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ExpertIterationError(f"{label} must be a canonical sha256 digest")


def _digest(kind: str, payload: dict[str, Any]) -> str:
    material = {
        "expert_iteration": _EXPERT_VERSION,
        "kind": kind,
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


__all__ = [
    "DEFAULT_MAX_EXPANSIONS",
    "DEFAULT_MAX_OPERATOR_DEPTH",
    "ExternalTrialCallback",
    "ExpertIterationError",
    "HARD_MAX_EXPANSIONS",
    "HARD_MAX_OPERATOR_DEPTH",
    "HARD_MAX_OPERATORS",
    "TeacherPlan",
    "TeacherSearchResult",
    "TeacherTrial",
    "TrialEvidence",
    "TrialRequest",
    "search_teacher_plan",
]
