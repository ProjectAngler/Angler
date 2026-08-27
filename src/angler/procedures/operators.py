"""Canonical symbolic mirrors for procedures learned from executed traces.

The objects in this module are descriptions, not executors.  In particular,
``LearnedOperator`` cannot call a tool, mutate a project, or certify that its
effects will occur.  It records a falsifiable abstraction over observed trace
subsegments while retaining exact provenance back to those observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Literal, TypeAlias

from angler.procedures.records import ActionSchema, GroundAction, Record


_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class OperatorValidationError(ValueError):
    """Raised when a learned symbolic mirror is internally inconsistent."""


def _require_symbol(value: str, field: str) -> None:
    if not isinstance(value, str) or _SYMBOL.fullmatch(value) is None:
        raise OperatorValidationError(f"{field} must be a non-empty symbolic name")


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise OperatorValidationError(f"{field} must be a canonical sha256 digest")


def _under_namespace(symbol: str, namespace: str) -> bool:
    return symbol == namespace or symbol.startswith(namespace + ".")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class TypedVariable:
    """One typed role shared across lifted records and action arguments."""

    name: str
    type_name: str

    def __post_init__(self) -> None:
        _require_symbol(self.name, "variable name")
        _require_symbol(self.type_name, "variable type_name")

    def to_canonical(self) -> dict[str, str]:
        return {"kind": "variable", "name": self.name, "type_name": self.type_name}


@dataclass(frozen=True, slots=True, order=True)
class Constant:
    """A typed entity that remained identical across all supporting examples."""

    value: str
    type_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise OperatorValidationError("constant value must be a non-empty string")
        _require_symbol(self.type_name, "constant type_name")

    def to_canonical(self) -> dict[str, str]:
        return {"kind": "constant", "type_name": self.type_name, "value": self.value}


SymbolicTerm: TypeAlias = TypedVariable | Constant


def _term_key(term: SymbolicTerm) -> bytes:
    if not isinstance(term, (TypedVariable, Constant)):
        raise OperatorValidationError("symbolic arguments must be typed terms")
    return _canonical_bytes(term.to_canonical())


@dataclass(frozen=True, slots=True)
class RecordPattern:
    """A lifted project-state record used as a precondition or effect target."""

    predicate: str
    arguments: tuple[SymbolicTerm, ...]

    def __post_init__(self) -> None:
        _require_symbol(self.predicate, "record-pattern predicate")
        if not isinstance(self.arguments, tuple):
            raise OperatorValidationError("record-pattern arguments must be a tuple")
        for argument in self.arguments:
            _term_key(argument)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "arguments": [argument.to_canonical() for argument in self.arguments],
            "predicate": self.predicate,
        }


@dataclass(frozen=True, slots=True)
class ActionPattern:
    """A typed, lifted reference to an observed primitive action schema."""

    schema: ActionSchema
    arguments: tuple[SymbolicTerm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ActionSchema):
            raise OperatorValidationError("action-pattern schema must be an ActionSchema")
        if not isinstance(self.arguments, tuple):
            raise OperatorValidationError("action-pattern arguments must be a tuple")
        if len(self.arguments) != len(self.schema.parameters):
            raise OperatorValidationError(
                "action-pattern arity must match its action schema"
            )
        for parameter, argument in zip(
            self.schema.parameters,
            self.arguments,
            strict=True,
        ):
            _term_key(argument)
            if argument.type_name != parameter.type_name:
                raise OperatorValidationError(
                    "action-pattern argument types must match schema parameters"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "arguments": [argument.to_canonical() for argument in self.arguments],
            "schema": {
                "digest": self.schema.digest,
                "description": self.schema.description,
                "name": self.schema.name,
                "parameters": [
                    {"name": item.name, "type_name": item.type_name}
                    for item in self.schema.parameters
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class Effect:
    """An observed add/delete delta generalized into a record pattern."""

    kind: Literal["add", "delete"]
    record: RecordPattern

    def __post_init__(self) -> None:
        if self.kind not in ("add", "delete"):
            raise OperatorValidationError("effect kind must be 'add' or 'delete'")
        if not isinstance(self.record, RecordPattern):
            raise OperatorValidationError("effect record must be a RecordPattern")

    def to_canonical(self) -> dict[str, Any]:
        return {"kind": self.kind, "record": self.record.to_canonical()}


@dataclass(frozen=True, slots=True)
class ReconstructionExemplar:
    """Bounded canonical instance data sufficient to refit/replay a mirror.

    The payload contains observations and actions only.  It has no executor,
    verifier result, or authority to prescribe the recorded sequence.
    """

    namespace: str
    start_records: tuple[Record, ...]
    variable_bindings: tuple[tuple[str, str], ...]
    constant_values: tuple[str, ...]
    actions: tuple[GroundAction, ...]
    end_records: tuple[Record, ...]

    def __post_init__(self) -> None:
        _require_symbol(self.namespace, "reconstruction namespace")
        for values, label in (
            (self.start_records, "start_records"),
            (self.end_records, "end_records"),
        ):
            if type(values) is not tuple:
                raise OperatorValidationError(f"{label} must be a tuple")
            if len(values) > 32:
                raise OperatorValidationError(f"{label} cannot exceed 32 records")
            if any(not isinstance(item, Record) for item in values):
                raise OperatorValidationError(f"{label} must contain Record values")
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise OperatorValidationError(f"{label} must be canonical and unique")
            if any(item.namespace != self.namespace for item in values):
                raise OperatorValidationError(
                    "reconstruction records must remain in one namespace"
                )
        if type(self.variable_bindings) is not tuple:
            raise OperatorValidationError("variable_bindings must be a tuple")
        bindings = tuple(sorted(self.variable_bindings))
        if len({item[0] for item in bindings}) != len(bindings):
            raise OperatorValidationError("variable binding names must be unique")
        for name, value in bindings:
            _require_symbol(name, "variable binding name")
            if not isinstance(value, str) or not value:
                raise OperatorValidationError("variable binding values must be non-empty")
        if type(self.constant_values) is not tuple:
            raise OperatorValidationError("constant_values must be a tuple")
        constants = tuple(sorted(self.constant_values))
        if len(set(constants)) != len(constants) or any(
            not isinstance(value, str) or not value for value in constants
        ):
            raise OperatorValidationError("constant_values must be unique strings")
        if type(self.actions) is not tuple or not self.actions:
            raise OperatorValidationError("reconstruction actions must be non-empty")
        if any(not isinstance(item, GroundAction) for item in self.actions):
            raise OperatorValidationError("reconstruction actions must be GroundAction")
        if any(item.namespace != self.namespace for item in self.actions):
            raise OperatorValidationError(
                "reconstruction actions must remain in one namespace"
            )
        bound_values = {value for _, value in bindings} | set(constants)
        for record in self.start_records + self.end_records:
            if any(argument not in bound_values for argument in record.arguments):
                raise OperatorValidationError(
                    "reconstruction records may mention only bound entities"
                )
        for action in self.actions:
            if any(argument not in bound_values for argument in action.arguments):
                raise OperatorValidationError(
                    "reconstruction actions may mention only bound entities"
                )
        object.__setattr__(self, "variable_bindings", bindings)
        object.__setattr__(self, "constant_values", constants)

    def to_canonical(self) -> dict[str, Any]:
        def record_payload(record: Record) -> dict[str, Any]:
            return {
                "arguments": list(record.arguments),
                "predicate": record.predicate,
            }

        return {
            "actions": [
                {
                    "arguments": list(action.arguments),
                    "schema_digest": action.schema.digest,
                    "schema_name": action.schema.name,
                }
                for action in self.actions
            ],
            "constant_values": list(self.constant_values),
            "end_records": [record_payload(item) for item in self.end_records],
            "namespace": self.namespace,
            "start_records": [record_payload(item) for item in self.start_records],
            "variable_bindings": [list(item) for item in self.variable_bindings],
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class OperatorExemplar:
    """Immutable provenance for one trace subsegment supporting an operator."""

    trace_digest: str
    start_index: int
    stop_index: int
    before_state_digest: str
    after_state_digest: str
    action_digests: tuple[str, ...]
    reconstruction: ReconstructionExemplar

    def __post_init__(self) -> None:
        _require_digest(self.trace_digest, "trace_digest")
        _require_digest(self.before_state_digest, "before_state_digest")
        _require_digest(self.after_state_digest, "after_state_digest")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or isinstance(self.stop_index, bool)
            or not isinstance(self.stop_index, int)
            or self.start_index < 0
            or self.stop_index <= self.start_index
        ):
            raise OperatorValidationError("exemplar indices must form a non-empty range")
        if not isinstance(self.action_digests, tuple):
            raise OperatorValidationError("action_digests must be a tuple")
        if len(self.action_digests) != self.stop_index - self.start_index:
            raise OperatorValidationError(
                "one action digest is required for each exemplar transition"
            )
        for value in self.action_digests:
            _require_digest(value, "action digest")
        if not isinstance(self.reconstruction, ReconstructionExemplar):
            raise OperatorValidationError(
                "operator exemplar requires reconstructable canonical instance data"
            )
        reconstructed_actions = tuple(
            item.digest for item in self.reconstruction.actions
        )
        if reconstructed_actions != self.action_digests:
            raise OperatorValidationError(
                "reconstruction actions must match exemplar action digests"
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "action_digests": list(self.action_digests),
            "after_state_digest": self.after_state_digest,
            "before_state_digest": self.before_state_digest,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "trace_digest": self.trace_digest,
            "reconstruction": self.reconstruction.to_canonical(),
        }


def _pattern_key(pattern: RecordPattern) -> bytes:
    return _canonical_bytes(pattern.to_canonical())


def _effect_key(effect: Effect) -> bytes:
    return _canonical_bytes(effect.to_canonical())


def _exemplar_key(exemplar: OperatorExemplar) -> bytes:
    return _canonical_bytes(exemplar.to_canonical())


@dataclass(frozen=True, slots=True)
class LearnedOperator:
    """Canonical symbolic hypothesis induced from one domain's observations.

    ``body`` is an abstract action-schema sequence, never a grounded solution
    path.  ``effects`` are predictions derived from observed deltas and are not
    treated as authoritative environment physics.
    """

    name: str
    namespace: str
    variables: tuple[TypedVariable, ...]
    preconditions: tuple[RecordPattern, ...]
    effects: tuple[Effect, ...]
    body: tuple[ActionPattern, ...]
    exemplars: tuple[OperatorExemplar, ...]
    revision: int = 1
    parent_digest: str | None = None
    _digest_cache: str = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        _require_symbol(self.name, "operator name")
        _require_symbol(self.namespace, "operator namespace")
        if not _under_namespace(self.name, self.namespace):
            raise OperatorValidationError("operator name must be under its namespace")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise OperatorValidationError("operator revision must be an integer")
        if self.revision <= 0:
            raise OperatorValidationError("operator revision must be positive")
        if self.parent_digest is not None:
            _require_digest(self.parent_digest, "parent_digest")

        tuple_fields = (
            (self.variables, "variables"),
            (self.preconditions, "preconditions"),
            (self.effects, "effects"),
            (self.body, "body"),
            (self.exemplars, "exemplars"),
        )
        for values, field in tuple_fields:
            if not isinstance(values, tuple):
                raise OperatorValidationError(f"{field} must be a tuple")
        if not self.body:
            raise OperatorValidationError("an operator requires an abstract action body")
        if not self.effects:
            raise OperatorValidationError("an operator requires at least one observed effect")
        if not self.exemplars:
            raise OperatorValidationError("an operator requires exemplar provenance")

        variables = tuple(sorted(self.variables))
        if len(set(variables)) != len(variables):
            raise OperatorValidationError("operator variables must be unique")
        if len({item.name for item in variables}) != len(variables):
            raise OperatorValidationError("variable names must identify one type")
        preconditions = tuple(sorted(self.preconditions, key=_pattern_key))
        effects = tuple(sorted(self.effects, key=_effect_key))
        exemplars = tuple(sorted(self.exemplars, key=_exemplar_key))
        if len(set(_pattern_key(item) for item in preconditions)) != len(preconditions):
            raise OperatorValidationError("operator preconditions must be unique")
        if len(set(_effect_key(item) for item in effects)) != len(effects):
            raise OperatorValidationError("operator effects must be unique")
        if len(set(_exemplar_key(item) for item in exemplars)) != len(exemplars):
            raise OperatorValidationError("operator exemplars must be unique")

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "exemplars", exemplars)

        declared = {item.name: item for item in variables}
        used: dict[str, TypedVariable] = {}
        for pattern in preconditions:
            self._validate_record_pattern(pattern, declared, used)
        for effect in effects:
            self._validate_record_pattern(effect.record, declared, used)
        for action in self.body:
            if not isinstance(action, ActionPattern):
                raise OperatorValidationError("operator body entries must be ActionPattern")
            if not _under_namespace(action.schema.name, self.namespace):
                raise OperatorValidationError(
                    "operator action schemas must remain within one namespace"
                )
            for term in action.arguments:
                self._validate_term(term, declared, used)
        if set(used) != set(declared):
            raise OperatorValidationError("every declared variable must be used")
        object.__setattr__(self, "_digest_cache", _digest(self.to_canonical()))

    def _validate_record_pattern(
        self,
        pattern: RecordPattern,
        declared: dict[str, TypedVariable],
        used: dict[str, TypedVariable],
    ) -> None:
        if not isinstance(pattern, RecordPattern):
            raise OperatorValidationError("operator records must be RecordPattern")
        if not _under_namespace(pattern.predicate, self.namespace):
            raise OperatorValidationError(
                "operator record predicates must remain within one namespace"
            )
        for term in pattern.arguments:
            self._validate_term(term, declared, used)

    @staticmethod
    def _validate_term(
        term: SymbolicTerm,
        declared: dict[str, TypedVariable],
        used: dict[str, TypedVariable],
    ) -> None:
        _term_key(term)
        if isinstance(term, TypedVariable):
            if declared.get(term.name) != term:
                raise OperatorValidationError(
                    "every variable term must match a declared typed variable"
                )
            used[term.name] = term

    def to_canonical(self) -> dict[str, Any]:
        """Return the complete deterministic identity material."""

        return {
            "body": [item.to_canonical() for item in self.body],
            "effects": [item.to_canonical() for item in self.effects],
            "exemplars": [item.to_canonical() for item in self.exemplars],
            "name": self.name,
            "namespace": self.namespace,
            "parent_digest": self.parent_digest,
            "preconditions": [item.to_canonical() for item in self.preconditions],
            "revision": self.revision,
            "variables": [item.to_canonical() for item in self.variables],
        }

    @property
    def digest(self) -> str:
        return self._digest_cache


__all__ = [
    "ActionPattern",
    "Constant",
    "Effect",
    "LearnedOperator",
    "OperatorExemplar",
    "OperatorValidationError",
    "ReconstructionExemplar",
    "RecordPattern",
    "SymbolicTerm",
    "TypedVariable",
]
