"""Canonical immutable contracts for relational procedure learning.

The contracts describe observations and executed transitions.  They do not
contain a planner, a solver, or domain transition semantics.  Domain modules
own execution and verification; learned components may depend on these stable
records without importing any particular world.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any


_CONTRACT_VERSION = "angler.relational-records.v1"
_LOCAL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_QUALIFIED_NAME = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"
)
_MAX_ATOM_LENGTH = 512


def _validate_local_name(value: str, label: str) -> str:
    if type(value) is not str or not _LOCAL_NAME.fullmatch(value):
        raise ValueError(f"{label} must be a canonical lower-case local name")
    return value


def _validate_qualified_name(value: str, label: str) -> str:
    if (
        type(value) is not str
        or "." not in value
        or not _QUALIFIED_NAME.fullmatch(value)
    ):
        raise ValueError(f"{label} must be a canonical qualified name")
    return value


def _validate_atom(value: str, label: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_ATOM_LENGTH:
        raise ValueError(f"{label} must be a non-empty bounded string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{label} must be stripped NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


def _namespace_of(name: str) -> str:
    return name.rsplit(".", maxsplit=1)[0]


def _record_key(record: "Record") -> tuple[str, tuple[str, ...]]:
    return record.predicate, record.arguments


def _canonical_records(
    records: Iterable["Record"],
    *,
    label: str,
) -> tuple["Record", ...]:
    result = tuple(records)
    if any(not isinstance(record, Record) for record in result):
        raise TypeError(f"{label} must contain only Record values")
    ordered = tuple(sorted(result, key=_record_key))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{label} cannot contain duplicate records")
    return ordered


def _require_canonical_records(records: Any, *, label: str) -> None:
    if type(records) is not tuple:
        raise TypeError(f"{label} must be an immutable tuple")
    canonical = _canonical_records(records, label=label)
    if records != canonical:
        raise ValueError(f"{label} must be in canonical sorted order")


def _digest(kind: str, payload: dict[str, Any]) -> str:
    material = {
        "contract": _CONTRACT_VERSION,
        "kind": kind,
        "payload": payload,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class Parameter:
    """One typed positional parameter in an action schema."""

    name: str
    type_name: str

    def __post_init__(self) -> None:
        _validate_local_name(self.name, "parameter name")
        _validate_qualified_name(self.type_name, "parameter type")

    @property
    def digest(self) -> str:
        return _digest(
            "parameter",
            {"name": self.name, "type_name": self.type_name},
        )


@dataclass(frozen=True, slots=True, order=True)
class Record:
    """One canonical namespaced relational fact."""

    predicate: str
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_qualified_name(self.predicate, "record predicate")
        if type(self.arguments) is not tuple:
            raise TypeError("record arguments must be an immutable tuple")
        for index, argument in enumerate(self.arguments):
            _validate_atom(argument, f"record argument {index}")

    @property
    def namespace(self) -> str:
        return _namespace_of(self.predicate)

    @property
    def digest(self) -> str:
        return _digest(
            "record",
            {"predicate": self.predicate, "arguments": self.arguments},
        )


@dataclass(frozen=True, slots=True)
class State:
    """A canonical set of facts owned by exactly one domain namespace."""

    namespace: str
    records: tuple[Record, ...]

    def __post_init__(self) -> None:
        _validate_qualified_name(self.namespace, "state namespace")
        _require_canonical_records(self.records, label="state records")
        if any(record.namespace != self.namespace for record in self.records):
            raise ValueError("every state record must belong to the state namespace")

    @classmethod
    def from_records(cls, namespace: str, records: Iterable[Record]) -> "State":
        """Build a State while canonicalizing input iteration order."""

        return cls(
            namespace=namespace,
            records=_canonical_records(records, label="state records"),
        )

    @property
    def digest(self) -> str:
        return _digest(
            "state",
            {
                "namespace": self.namespace,
                "records": [
                    {
                        "predicate": record.predicate,
                        "arguments": record.arguments,
                    }
                    for record in self.records
                ],
            },
        )


@dataclass(frozen=True, slots=True)
class ActionSchema:
    """A fully qualified operator name with typed positional parameters."""

    name: str
    parameters: tuple[Parameter, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_qualified_name(self.name, "action schema name")
        if type(self.parameters) is not tuple:
            raise TypeError("action parameters must be an immutable tuple")
        if any(not isinstance(parameter, Parameter) for parameter in self.parameters):
            raise TypeError("action parameters must contain only Parameter values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("action parameter names must be unique")
        if any(
            _namespace_of(parameter.type_name) != self.namespace
            for parameter in self.parameters
        ):
            raise ValueError("action parameter types must belong to its namespace")
        if self.description is not None:
            _validate_atom(self.description, "action schema description")

    @property
    def namespace(self) -> str:
        return _namespace_of(self.name)

    def ground(self, *arguments: str) -> "GroundAction":
        return GroundAction(schema=self, arguments=tuple(arguments))

    @property
    def digest(self) -> str:
        return _digest(
            "action_schema",
            {
                "name": self.name,
                "parameters": [
                    {"name": item.name, "type_name": item.type_name}
                    for item in self.parameters
                ],
                "description": self.description,
            },
        )


@dataclass(frozen=True, slots=True)
class GroundAction:
    """One completely bound instance of an ActionSchema."""

    schema: ActionSchema
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ActionSchema):
            raise TypeError("ground action schema must be an ActionSchema")
        if type(self.arguments) is not tuple:
            raise TypeError("ground action arguments must be an immutable tuple")
        if len(self.arguments) != len(self.schema.parameters):
            raise ValueError("ground action argument count does not match its schema")
        for parameter, argument in zip(
            self.schema.parameters,
            self.arguments,
            strict=True,
        ):
            _validate_atom(argument, f"action argument {parameter.name}")

    @property
    def namespace(self) -> str:
        return self.schema.namespace

    @property
    def digest(self) -> str:
        return _digest(
            "ground_action",
            {
                "schema_digest": self.schema.digest,
                "arguments": self.arguments,
            },
        )


@dataclass(frozen=True, slots=True)
class Goal:
    """A declarative conjunction of required and forbidden facts.

    ``exact`` means the complete state fact set must equal ``required``.  It is
    intentionally a condition, not a procedure for reaching that condition.
    """

    namespace: str
    required: tuple[Record, ...]
    forbidden: tuple[Record, ...] = ()
    exact: bool = False

    def __post_init__(self) -> None:
        _validate_qualified_name(self.namespace, "goal namespace")
        _require_canonical_records(self.required, label="required goal records")
        _require_canonical_records(self.forbidden, label="forbidden goal records")
        if any(
            record.namespace != self.namespace
            for record in self.required + self.forbidden
        ):
            raise ValueError("every goal record must belong to the goal namespace")
        if set(self.required) & set(self.forbidden):
            raise ValueError("a goal record cannot be both required and forbidden")
        if type(self.exact) is not bool:
            raise TypeError("goal exact must be bool")
        if self.exact and self.forbidden:
            raise ValueError("an exact goal cannot also contain forbidden records")

    @classmethod
    def from_records(
        cls,
        namespace: str,
        required: Iterable[Record],
        *,
        forbidden: Iterable[Record] = (),
        exact: bool = False,
    ) -> "Goal":
        return cls(
            namespace=namespace,
            required=_canonical_records(required, label="required goal records"),
            forbidden=_canonical_records(forbidden, label="forbidden goal records"),
            exact=exact,
        )

    @property
    def digest(self) -> str:
        return _digest(
            "goal",
            {
                "namespace": self.namespace,
                "required": [record.digest for record in self.required],
                "forbidden": [record.digest for record in self.forbidden],
                "exact": self.exact,
            },
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """One domain-executed action and its observed successor state."""

    before: State
    action: GroundAction
    after: State
    applied: bool
    outcome: str

    def __post_init__(self) -> None:
        if not isinstance(self.before, State) or not isinstance(self.after, State):
            raise TypeError("transition endpoints must be State values")
        if not isinstance(self.action, GroundAction):
            raise TypeError("transition action must be a GroundAction")
        if self.before.namespace != self.after.namespace:
            raise ValueError("transition endpoints must share a namespace")
        if self.action.namespace != self.before.namespace:
            raise ValueError("transition action must belong to the state namespace")
        if type(self.applied) is not bool:
            raise TypeError("transition applied must be bool")
        _validate_qualified_name(self.outcome, "transition outcome")
        if _namespace_of(self.outcome) != self.before.namespace:
            raise ValueError("transition outcome must belong to the state namespace")
        if self.applied and self.before == self.after:
            raise ValueError("an applied transition must change state")
        if not self.applied and self.before != self.after:
            raise ValueError("a rejected transition cannot change state")

    @property
    def digest(self) -> str:
        return _digest(
            "transition",
            {
                "before": self.before.digest,
                "action": self.action.digest,
                "after": self.after.digest,
                "applied": self.applied,
                "outcome": self.outcome,
            },
        )


@dataclass(frozen=True, slots=True)
class Trace:
    """A contiguous immutable sequence of independently executed transitions."""

    initial: State
    transitions: tuple[Transition, ...]
    goal: Goal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.initial, State):
            raise TypeError("trace initial must be a State")
        if type(self.transitions) is not tuple:
            raise TypeError("trace transitions must be an immutable tuple")
        current = self.initial
        for transition in self.transitions:
            if not isinstance(transition, Transition):
                raise TypeError("trace transitions must contain Transition values")
            if transition.before != current:
                raise ValueError("trace transitions must form one contiguous chain")
            current = transition.after
        if self.goal is not None:
            if not isinstance(self.goal, Goal):
                raise TypeError("trace goal must be a Goal or None")
            if self.goal.namespace != self.initial.namespace:
                raise ValueError("trace goal must belong to the state namespace")

    @property
    def final_state(self) -> State:
        if not self.transitions:
            return self.initial
        return self.transitions[-1].after

    @property
    def digest(self) -> str:
        return _digest(
            "trace",
            {
                "initial": self.initial.digest,
                "transitions": [item.digest for item in self.transitions],
                "goal": None if self.goal is None else self.goal.digest,
            },
        )


__all__ = [
    "ActionSchema",
    "Goal",
    "GroundAction",
    "Parameter",
    "Record",
    "State",
    "Trace",
    "Transition",
]
