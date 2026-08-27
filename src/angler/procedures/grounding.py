"""Domain-neutral grounding for learned symbolic operator mirrors.

This module matches learned precondition patterns against immutable relational
states.  It can instantiate the resulting typed bindings into proposed
primitive actions and predicted record deltas, but it has no world adapter and
cannot execute or validate those predictions.  Goal overlap is only a
declarative ordering signal for a teacher or fallback policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any

from angler.procedures.operators import (
    Constant,
    LearnedOperator,
    RecordPattern,
    SymbolicTerm,
    TypedVariable,
)
from angler.procedures.records import Goal, GroundAction, Record, State


DEFAULT_MAX_BINDINGS = 128
DEFAULT_MAX_MATCH_ATTEMPTS = 10_000
HARD_MAX_BINDINGS = 1_024
HARD_MAX_MATCH_ATTEMPTS = 100_000
HARD_MAX_PRECONDITIONS = 64
HARD_MAX_STATE_RECORDS = 4_096
_GROUNDING_VERSION = "angler.operator-grounding.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_VALUE_LENGTH = 512


class GroundingError(ValueError):
    """Raised when symbolic grounding inputs are inconsistent."""


class GroundingLimitError(RuntimeError):
    """Raised instead of returning a silently incomplete grounding set."""


@dataclass(frozen=True, slots=True, order=True)
class StateBindingAssignment:
    """One typed learned role bound to one concrete state atom."""

    variable: TypedVariable
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.variable, TypedVariable):
            raise TypeError("assignment variable must be a TypedVariable")
        _require_atom(self.value, "assignment value")

    def to_canonical(self) -> dict[str, str]:
        return {
            "name": self.variable.name,
            "type_name": self.variable.type_name,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class StateOperatorBinding:
    """A state-derived typed binding with lossless execution-contract export."""

    operator_digest: str
    namespace: str
    assignments: tuple[StateBindingAssignment, ...]

    def __post_init__(self) -> None:
        _require_digest(self.operator_digest, "binding operator_digest")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise GroundingError("binding namespace must be non-empty")
        if type(self.assignments) is not tuple:
            raise TypeError("binding assignments must be an immutable tuple")
        if any(
            not isinstance(item, StateBindingAssignment)
            for item in self.assignments
        ):
            raise TypeError(
                "binding assignments must be StateBindingAssignment values"
            )
        expected = tuple(
            sorted(
                self.assignments,
                key=lambda item: (item.variable.name, item.variable.type_name),
            )
        )
        if self.assignments != expected:
            raise GroundingError("binding assignments must be canonical")
        names = tuple(item.variable.name for item in self.assignments)
        if len(set(names)) != len(names):
            raise GroundingError("a binding cannot assign one variable twice")

    def value_for(self, variable: TypedVariable | str) -> str:
        """Return one concrete value while preserving type checks when supplied."""

        if isinstance(variable, TypedVariable):
            for assignment in self.assignments:
                if assignment.variable.name == variable.name:
                    if assignment.variable != variable:
                        raise GroundingError(
                            "binding variable name refers to a different type"
                        )
                    return assignment.value
            raise KeyError(variable.name)
        if not isinstance(variable, str) or not variable:
            raise TypeError("variable must be a TypedVariable or non-empty name")
        for assignment in self.assignments:
            if assignment.variable.name == variable:
                return assignment.value
        raise KeyError(variable)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "assignments": [item.to_canonical() for item in self.assignments],
            "namespace": self.namespace,
            "operator_digest": self.operator_digest,
        }

    @property
    def digest(self) -> str:
        return _digest("operator_binding", self.to_canonical())

    def to_execution_binding(self, operator: LearnedOperator):
        """Convert losslessly to the canonical neural/execution binding type.

        The import is intentionally local: symbolic state grounding remains
        usable in lightweight environments where the optional neural runtime
        is absent.  When present, the returned value is exactly
        ``execution.OperatorBinding``, not a look-alike protocol.
        """

        _validate_operator_binding(operator, self)
        from angler.procedures.execution import (  # optional neural boundary
            BindingAssignment,
            OperatorBinding,
            TypedEntityCandidate,
        )

        return OperatorBinding(
            operator,
            tuple(
                BindingAssignment(
                    item.variable,
                    TypedEntityCandidate(
                        item.value,
                        item.variable.type_name,
                    ),
                )
                for item in self.assignments
            ),
        )


@dataclass(frozen=True, slots=True)
class GroundedOperatorPrediction:
    """Ground actions and record deltas predicted by a learned mirror."""

    binding: StateOperatorBinding
    actions: tuple[GroundAction, ...]
    predicted_additions: tuple[Record, ...]
    predicted_deletions: tuple[Record, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, StateOperatorBinding):
            raise TypeError("grounded prediction requires a StateOperatorBinding")
        if type(self.actions) is not tuple or not self.actions:
            raise GroundingError("grounded actions must be a non-empty tuple")
        if any(not isinstance(item, GroundAction) for item in self.actions):
            raise TypeError("grounded actions must contain GroundAction values")
        if any(item.namespace != self.binding.namespace for item in self.actions):
            raise GroundingError("grounded actions must remain in one namespace")
        for records, label in (
            (self.predicted_additions, "predicted additions"),
            (self.predicted_deletions, "predicted deletions"),
        ):
            _require_canonical_records(records, label)
            if any(item.namespace != self.binding.namespace for item in records):
                raise GroundingError(
                    "predicted records must remain in the binding namespace"
                )
        if set(self.predicted_additions) & set(self.predicted_deletions):
            raise GroundingError(
                "one grounded record cannot be both added and deleted"
            )

    @property
    def digest(self) -> str:
        return _digest(
            "grounded_operator_prediction",
            {
                "actions": [item.digest for item in self.actions],
                "binding": self.binding.digest,
                "predicted_additions": [
                    item.digest for item in self.predicted_additions
                ],
                "predicted_deletions": [
                    item.digest for item in self.predicted_deletions
                ],
            },
        )


def enumerate_operator_bindings(
    operator: LearnedOperator,
    state: State,
    *,
    maximum_bindings: int = DEFAULT_MAX_BINDINGS,
    maximum_match_attempts: int = DEFAULT_MAX_MATCH_ATTEMPTS,
) -> tuple[StateOperatorBinding, ...]:
    """Enumerate every complete typed binding within explicit hard ceilings.

    A limit breach raises ``GroundingLimitError``; it never returns a prefix
    that could be mistaken for an exhaustive result.
    """

    _validate_grounding_inputs(
        operator,
        state,
        maximum_bindings=maximum_bindings,
        maximum_match_attempts=maximum_match_attempts,
    )
    records_by_shape: dict[tuple[str, int], tuple[Record, ...]] = {}
    for pattern in operator.preconditions:
        shape = (pattern.predicate, len(pattern.arguments))
        if shape not in records_by_shape:
            records_by_shape[shape] = tuple(
                record
                for record in state.records
                if record.predicate == pattern.predicate
                and len(record.arguments) == len(pattern.arguments)
            )

    work = tuple(
        sorted(
            [
                (
                    pattern,
                    records_by_shape[
                        (pattern.predicate, len(pattern.arguments))
                    ],
                )
                for pattern in operator.preconditions
            ],
            key=lambda item: (len(item[1]), _pattern_key(item[0])),
        )
    )
    if any(not candidates for _, candidates in work):
        return ()

    variable_by_name = {item.name: item for item in operator.variables}
    results: dict[str, StateOperatorBinding] = {}
    match_attempts = 0

    def visit(index: int, values: dict[str, str]) -> None:
        nonlocal match_attempts
        if index == len(work):
            if set(values) != set(variable_by_name):
                return
            assignments = tuple(
                StateBindingAssignment(variable, values[variable.name])
                for variable in operator.variables
            )
            binding = StateOperatorBinding(
                operator_digest=operator.digest,
                namespace=operator.namespace,
                assignments=assignments,
            )
            if binding.digest not in results:
                if len(results) >= maximum_bindings:
                    raise GroundingLimitError(
                        "operator grounding exceeded maximum_bindings"
                    )
                results[binding.digest] = binding
            return

        pattern, candidates = work[index]
        for record in candidates:
            match_attempts += 1
            if match_attempts > maximum_match_attempts:
                raise GroundingLimitError(
                    "operator grounding exceeded maximum_match_attempts"
                )
            extended = _unify_pattern(pattern, record, values)
            if extended is not None:
                visit(index + 1, extended)

    visit(0, {})
    return tuple(results[key] for key in sorted(results))


def instantiate_operator(
    operator: LearnedOperator,
    binding: StateOperatorBinding,
) -> GroundedOperatorPrediction:
    """Instantiate a mirror without asserting that its predictions are true."""

    _validate_operator_binding(operator, binding)
    values = {
        item.variable.name: item.value
        for item in binding.assignments
    }
    actions = tuple(
        pattern.schema.ground(
            *(_ground_term(term, values) for term in pattern.arguments)
        )
        for pattern in operator.body
    )
    additions: set[Record] = set()
    deletions: set[Record] = set()
    for effect in operator.effects:
        record = Record(
            effect.record.predicate,
            tuple(
                _ground_term(term, values)
                for term in effect.record.arguments
            ),
        )
        if effect.kind == "add":
            additions.add(record)
        else:
            deletions.add(record)
    if additions & deletions:
        raise GroundingError(
            "binding collapses learned effects into a contradiction"
        )
    return GroundedOperatorPrediction(
        binding=binding,
        actions=actions,
        predicted_additions=tuple(sorted(additions)),
        predicted_deletions=tuple(sorted(deletions)),
    )


def score_goal_effect_overlap(
    prediction: GroundedOperatorPrediction,
    goal: Goal,
) -> int:
    """Return helpful minus conflicting declarative effects.

    This score is suitable only for ordering teacher/fallback proposals.  It is
    not an applicability proof, world transition, success judgment, distance,
    or promotion signal.
    """

    if not isinstance(prediction, GroundedOperatorPrediction):
        raise TypeError("prediction must be a GroundedOperatorPrediction")
    if not isinstance(goal, Goal):
        raise TypeError("goal must be a Goal")
    if prediction.binding.namespace != goal.namespace:
        raise GroundingError("prediction and goal namespaces must match")
    additions = set(prediction.predicted_additions)
    deletions = set(prediction.predicted_deletions)
    required = set(goal.required)
    forbidden = set(goal.forbidden)
    if goal.exact:
        helpful = len(additions & required) + len(deletions - required)
        conflicting = len(additions - required) + len(deletions & required)
    else:
        helpful = len(additions & required) + len(deletions & forbidden)
        conflicting = len(additions & forbidden) + len(deletions & required)
    return helpful - conflicting


def _unify_pattern(
    pattern: RecordPattern,
    record: Record,
    values: dict[str, str],
) -> dict[str, str] | None:
    extended = dict(values)
    for term, concrete in zip(pattern.arguments, record.arguments, strict=True):
        if isinstance(term, Constant):
            if term.value != concrete:
                return None
            continue
        if not isinstance(term, TypedVariable):
            raise GroundingError("record pattern contains an unsupported term")
        existing = extended.get(term.name)
        if existing is not None and existing != concrete:
            return None
        extended[term.name] = concrete
    return extended


def _ground_term(term: SymbolicTerm, values: dict[str, str]) -> str:
    if isinstance(term, Constant):
        return term.value
    if isinstance(term, TypedVariable):
        try:
            return values[term.name]
        except KeyError as error:
            raise GroundingError(
                f"binding does not assign variable {term.name!r}"
            ) from error
    raise GroundingError("operator contains an unsupported symbolic term")


def _validate_grounding_inputs(
    operator: LearnedOperator,
    state: State,
    *,
    maximum_bindings: int,
    maximum_match_attempts: int,
) -> None:
    if not isinstance(operator, LearnedOperator):
        raise TypeError("operator must be a LearnedOperator")
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if operator.namespace != state.namespace:
        raise GroundingError("operator and state namespaces must match")
    _validate_ceiling(
        maximum_bindings,
        "maximum_bindings",
        HARD_MAX_BINDINGS,
    )
    _validate_ceiling(
        maximum_match_attempts,
        "maximum_match_attempts",
        HARD_MAX_MATCH_ATTEMPTS,
    )
    if len(operator.preconditions) > HARD_MAX_PRECONDITIONS:
        raise GroundingLimitError("operator exceeds the precondition hard ceiling")
    if len(state.records) > HARD_MAX_STATE_RECORDS:
        raise GroundingLimitError("state exceeds the record hard ceiling")


def _validate_operator_binding(
    operator: LearnedOperator,
    binding: StateOperatorBinding,
) -> None:
    if not isinstance(operator, LearnedOperator):
        raise TypeError("operator must be a LearnedOperator")
    if not isinstance(binding, StateOperatorBinding):
        raise TypeError("binding must be a StateOperatorBinding")
    if binding.operator_digest != operator.digest:
        raise GroundingError("binding belongs to another operator revision")
    if binding.namespace != operator.namespace:
        raise GroundingError("binding belongs to another namespace")
    expected = operator.variables
    observed = tuple(item.variable for item in binding.assignments)
    if observed != expected:
        raise GroundingError("binding does not assign every operator variable exactly")


def _validate_ceiling(value: int, label: str, hard_maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0 or value > hard_maximum:
        raise ValueError(f"{label} must be between 1 and {hard_maximum}")


def _pattern_key(pattern: RecordPattern) -> bytes:
    return json.dumps(
        pattern.to_canonical(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_canonical_records(records: Any, label: str) -> None:
    if type(records) is not tuple:
        raise TypeError(f"{label} must be an immutable tuple")
    if any(not isinstance(item, Record) for item in records):
        raise TypeError(f"{label} must contain Record values")
    if records != tuple(sorted(records)) or len(set(records)) != len(records):
        raise GroundingError(f"{label} must be canonical and unique")


def _require_atom(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_VALUE_LENGTH
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(unicodedata.category(item).startswith("C") for item in value)
    ):
        raise GroundingError(f"{label} must be a canonical bounded atom")


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise GroundingError(f"{label} must be a canonical sha256 digest")


def _digest(kind: str, payload: dict[str, Any]) -> str:
    material = {
        "grounding": _GROUNDING_VERSION,
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
    "DEFAULT_MAX_BINDINGS",
    "DEFAULT_MAX_MATCH_ATTEMPTS",
    "GroundedOperatorPrediction",
    "GroundingError",
    "GroundingLimitError",
    "HARD_MAX_BINDINGS",
    "HARD_MAX_MATCH_ATTEMPTS",
    "HARD_MAX_PRECONDITIONS",
    "HARD_MAX_STATE_RECORDS",
    "StateBindingAssignment",
    "StateOperatorBinding",
    "enumerate_operator_bindings",
    "instantiate_operator",
    "score_goal_effect_overlap",
]
