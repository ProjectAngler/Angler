"""Certificate-gated cross-domain transfer adapters.

This module only projects immutable symbolic contracts.  It imports no world,
executor, evaluator, or planner and never treats an alias as evidence unless
that alias was admitted with a passing external certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import TYPE_CHECKING, Literal

from angler.procedures.alignment import AliasTable, SymbolAlias
from angler.procedures.grounding import StateOperatorBinding
from angler.procedures.operators import LearnedOperator, TypedVariable
from angler.procedures.records import ActionSchema, Goal, GroundAction, Record, State


if TYPE_CHECKING:  # pragma: no cover - the neural runtime is optional here
    from angler.procedures.execution import OperatorBinding


AliasKind = Literal["operator", "predicate", "action", "type"]
_QUALIFIED_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_SYMBOL_DOMAIN = b"project-angler.certified-transfer-symbol.v1\x00"


class CertifiedTransferError(ValueError):
    """Raised when certified evidence cannot support an exact projection."""


class TransferRuntimeUnavailableError(RuntimeError):
    """Raised when the optional execution binding runtime is unavailable."""


def _validate_alias_table(alias_table: AliasTable) -> None:
    if not isinstance(alias_table, AliasTable):
        raise TypeError("alias_table must be an AliasTable")
    for entry in alias_table.entries:
        if (
            entry.certificate.result != "pass"
            or entry.certificate.candidate_digest != entry.candidate_digest
        ):
            raise CertifiedTransferError(
                "every transfer alias requires a passing candidate-bound certificate"
            )


def _namespace_of(symbol: str) -> str:
    return symbol.rsplit(".", maxsplit=1)[0]


def _require_namespace(value: str, label: str) -> None:
    if not isinstance(value, str) or not _QUALIFIED_NAME.fullmatch(value):
        raise CertifiedTransferError(f"{label} must be a qualified name")


def _aliases(alias_table: AliasTable, kind: AliasKind) -> tuple[SymbolAlias, ...]:
    return tuple(
        alias
        for entry in alias_table.entries
        for alias in entry.aliases
        if alias.kind == kind
    )


def _reachable(start, edges: tuple[tuple[object, object], ...]) -> set:
    reached = {start}
    changed = True
    while changed:
        changed = False
        for left, right in edges:
            if left in reached or right in reached:
                before = len(reached)
                reached.update((left, right))
                changed = changed or len(reached) != before
    return reached


def _symbols_connected(
    alias_table: AliasTable,
    kind: AliasKind,
    source: str,
    target: str,
) -> bool:
    if source == target:
        return True
    edges = tuple((item.source, item.target) for item in _aliases(alias_table, kind))
    return target in _reachable(source, edges)


def _operators_connected(
    alias_table: AliasTable,
    source_digest: str,
    target_digest: str,
) -> bool:
    if source_digest == target_digest:
        return True
    edges = tuple(
        (entry.source_operator_digest, entry.target_operator_digest)
        for entry in alias_table.entries
    )
    return target_digest in _reachable(source_digest, edges)


def _body_variables(operator: LearnedOperator) -> tuple[TypedVariable, ...]:
    names = {
        term.name
        for action in operator.body
        for term in action.arguments
        if isinstance(term, TypedVariable)
    }
    return tuple(item for item in operator.variables if item.name in names)


def _validate_state_binding(
    target: LearnedOperator,
    binding: StateOperatorBinding,
) -> dict[str, str]:
    if not isinstance(binding, StateOperatorBinding):
        raise TypeError("target_binding must be a StateOperatorBinding")
    if binding.operator_digest != target.digest or binding.namespace != target.namespace:
        raise CertifiedTransferError(
            "target binding does not belong to the supplied target operator"
        )
    declared = {item.name: item for item in target.variables}
    values: dict[str, str] = {}
    for assignment in binding.assignments:
        expected = declared.get(assignment.variable.name)
        if expected is None or expected != assignment.variable:
            raise CertifiedTransferError(
                "target binding contains a foreign or retyped variable"
            )
        values[assignment.variable.name] = assignment.value
    return values


def _execution_binding(
    operator: LearnedOperator,
    values: dict[str, str],
) -> "OperatorBinding":
    try:
        from angler.procedures.execution import (
            BindingAssignment,
            OperatorBinding,
            TypedEntityCandidate,
        )
    except ModuleNotFoundError as error:  # optional neural dependency boundary
        raise TransferRuntimeUnavailableError(
            "the execution binding runtime is unavailable"
        ) from error

    assignments = tuple(
        BindingAssignment(
            variable,
            TypedEntityCandidate(values[variable.name], variable.type_name),
        )
        for variable in operator.variables
    )
    return OperatorBinding(operator, assignments)


def certified_transfer_binding(
    alias_table: AliasTable,
    canonical_operator: LearnedOperator,
    target_operator: LearnedOperator,
    target_binding: StateOperatorBinding,
) -> "OperatorBinding":
    """Map a target state binding into the chosen canonical execution mirror.

    Cross-domain variable correspondences are taken only from certified alias
    entries.  Target variables absent from the primitive body are residual
    constraint witnesses and do not participate in transfer.
    """

    _validate_alias_table(alias_table)
    if not isinstance(canonical_operator, LearnedOperator) or not isinstance(
        target_operator, LearnedOperator
    ):
        raise TypeError("binding transfer endpoints must be LearnedOperator values")
    target_values = _validate_state_binding(target_operator, target_binding)

    if canonical_operator.digest == target_operator.digest:
        missing = {
            item.name for item in canonical_operator.variables
        } - set(target_values)
        if missing:
            raise CertifiedTransferError(
                f"identity binding is incomplete; missing={sorted(missing)}"
            )
        return _execution_binding(canonical_operator, target_values)

    if not alias_table.entries or not _operators_connected(
        alias_table,
        canonical_operator.digest,
        target_operator.digest,
    ):
        raise CertifiedTransferError(
            "cross-domain binding transfer lacks a passing certified chain"
        )

    canonical_core = _body_variables(canonical_operator)
    target_core = _body_variables(target_operator)
    canonical_core_names = {item.name for item in canonical_core}
    target_core_names = {item.name for item in target_core}
    canonical_residual = {
        item.name for item in canonical_operator.variables
    } - canonical_core_names
    if canonical_residual:
        raise CertifiedTransferError(
            "the chosen canonical operator has residual-only variables and cannot "
            "form a complete execution.OperatorBinding"
        )

    variable_edges = tuple(
        (
            (entry.source_operator_digest, source_name),
            (entry.target_operator_digest, target_name),
        )
        for entry in alias_table.entries
        for source_name, target_name in entry.variable_map
    )
    target_variables = {item.name: item for item in target_core}
    mapped_targets: dict[str, str] = {}
    canonical_values: dict[str, str] = {}
    for canonical_variable in canonical_core:
        component = _reachable(
            (canonical_operator.digest, canonical_variable.name),
            variable_edges,
        )
        matches = sorted(
            name
            for digest, name in component
            if digest == target_operator.digest and name in target_core_names
        )
        if not matches:
            raise CertifiedTransferError(
                f"canonical executable variable {canonical_variable.name!r} is unmapped"
            )
        if len(matches) != 1:
            raise CertifiedTransferError(
                f"canonical executable variable {canonical_variable.name!r} is ambiguous"
            )
        target_name = matches[0]
        prior = mapped_targets.get(target_name)
        if prior is not None and prior != canonical_variable.name:
            raise CertifiedTransferError(
                "certified variable chain is not one-to-one at the target"
            )
        mapped_targets[target_name] = canonical_variable.name
        if target_name not in target_values:
            raise CertifiedTransferError(
                f"target executable variable {target_name!r} has no assignment"
            )
        target_variable = target_variables[target_name]
        if not _symbols_connected(
            alias_table,
            "type",
            canonical_variable.type_name,
            target_variable.type_name,
        ):
            raise CertifiedTransferError(
                "mapped executable variables lack a certified type correspondence"
            )
        canonical_values[canonical_variable.name] = target_values[target_name]

    return _execution_binding(canonical_operator, canonical_values)


def _projected_symbol(
    canonical_namespace: str,
    category: Literal["shared", "residual"],
    identity: str,
) -> str:
    digest = hashlib.sha256(
        _SYMBOL_DOMAIN
        + category.encode("ascii")
        + b"\x00"
        + identity.encode("utf-8")
    ).hexdigest()
    return f"{canonical_namespace}.{category}_{digest}"


@dataclass(frozen=True, slots=True)
class CertifiedPredicateProjector:
    """Project certified predicate aliases and isolated residuals into one domain."""

    alias_table: AliasTable
    canonical_namespace: str
    canonical_source_namespace: str

    def __post_init__(self) -> None:
        _validate_alias_table(self.alias_table)
        _require_namespace(self.canonical_namespace, "canonical_namespace")
        _require_namespace(
            self.canonical_source_namespace,
            "canonical_source_namespace",
        )

    def _require_source(self, source_namespace: str) -> None:
        _require_namespace(source_namespace, "source namespace")
        if source_namespace == self.canonical_source_namespace:
            return
        namespace_edges = tuple(
            (_namespace_of(alias.source), _namespace_of(alias.target))
            for entry in self.alias_table.entries
            for alias in entry.aliases
        )
        if self.canonical_source_namespace not in _reachable(
            source_namespace,
            namespace_edges,
        ):
            raise CertifiedTransferError(
                "noncanonical projection lacks a passing certified chain"
            )

    def project_predicate(self, predicate: str, source_namespace: str) -> str:
        self._require_source(source_namespace)
        if not isinstance(predicate, str) or _namespace_of(predicate) != source_namespace:
            raise CertifiedTransferError(
                "predicate does not belong to its declared source namespace"
            )
        aliases = _aliases(self.alias_table, "predicate")
        if source_namespace == self.canonical_source_namespace:
            return _projected_symbol(
                self.canonical_namespace,
                "shared",
                predicate,
            )
        predicate_edges = tuple((item.source, item.target) for item in aliases)
        component = _reachable(predicate, predicate_edges)
        canonical_members = sorted(
            symbol
            for symbol in component
            if _namespace_of(symbol) == self.canonical_source_namespace
        )
        if len(canonical_members) > 1:
            raise CertifiedTransferError(
                "target predicate has an ambiguous canonical-source identity"
            )
        if canonical_members:
            return _projected_symbol(
                self.canonical_namespace,
                "shared",
                canonical_members[0],
            )
        return _projected_symbol(
            self.canonical_namespace,
            "residual",
            source_namespace + "\x00" + predicate,
        )

    def project_record(self, record: Record) -> Record:
        if not isinstance(record, Record):
            raise TypeError("record must be a Record")
        return Record(
            self.project_predicate(record.predicate, record.namespace),
            record.arguments,
        )

    def project_state(self, state: State) -> State:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        self._require_source(state.namespace)
        try:
            return State.from_records(
                self.canonical_namespace,
                (self.project_record(item) for item in state.records),
            )
        except ValueError as error:
            raise CertifiedTransferError(
                "state projection would collide or violate canonical records"
            ) from error

    def project_goal(self, goal: Goal) -> Goal:
        if not isinstance(goal, Goal):
            raise TypeError("goal must be a Goal")
        self._require_source(goal.namespace)
        try:
            return Goal.from_records(
                self.canonical_namespace,
                (self.project_record(item) for item in goal.required),
                forbidden=(self.project_record(item) for item in goal.forbidden),
                exact=goal.exact,
            )
        except ValueError as error:
            raise CertifiedTransferError(
                "goal projection would collide or violate canonical records"
            ) from error


@dataclass(frozen=True, slots=True)
class CertifiedActionAdapter:
    """Exact positional action projection backed by certified symbol aliases."""

    alias_table: AliasTable
    local_schema: ActionSchema
    canonical_schema: ActionSchema

    def __post_init__(self) -> None:
        _validate_alias_table(self.alias_table)
        if not isinstance(self.local_schema, ActionSchema) or not isinstance(
            self.canonical_schema,
            ActionSchema,
        ):
            raise TypeError("action adapter schemas must be ActionSchema values")
        if self.local_schema == self.canonical_schema:
            return
        if not self.alias_table.entries or not _symbols_connected(
            self.alias_table,
            "action",
            self.local_schema.name,
            self.canonical_schema.name,
        ):
            raise CertifiedTransferError(
                "action projection lacks a passing certified alias chain"
            )
        if len(self.local_schema.parameters) != len(self.canonical_schema.parameters):
            raise CertifiedTransferError(
                "action projection requires exact positional arity"
            )
        for local, canonical in zip(
            self.local_schema.parameters,
            self.canonical_schema.parameters,
            strict=True,
        ):
            if not _symbols_connected(
                self.alias_table,
                "type",
                local.type_name,
                canonical.type_name,
            ):
                raise CertifiedTransferError(
                    "action parameter order lacks certified type correspondence"
                )

    def project_schema(self, schema: ActionSchema) -> ActionSchema:
        if schema != self.local_schema:
            raise CertifiedTransferError("schema is not this adapter's local schema")
        return self.canonical_schema

    def reverse_schema(self, schema: ActionSchema) -> ActionSchema:
        if schema != self.canonical_schema:
            raise CertifiedTransferError(
                "schema is not this adapter's canonical schema"
            )
        return self.local_schema

    def project_action(self, action: GroundAction) -> GroundAction:
        if not isinstance(action, GroundAction):
            raise TypeError("action must be a GroundAction")
        if action.schema != self.local_schema:
            raise CertifiedTransferError("action is not grounded in the local schema")
        return self.canonical_schema.ground(*action.arguments)

    def reverse_action(self, action: GroundAction) -> GroundAction:
        if not isinstance(action, GroundAction):
            raise TypeError("action must be a GroundAction")
        if action.schema != self.canonical_schema:
            raise CertifiedTransferError(
                "action is not grounded in the canonical schema"
            )
        return self.local_schema.ground(*action.arguments)


__all__ = [
    "CertifiedActionAdapter",
    "CertifiedPredicateProjector",
    "CertifiedTransferError",
    "TransferRuntimeUnavailableError",
    "certified_transfer_binding",
]
