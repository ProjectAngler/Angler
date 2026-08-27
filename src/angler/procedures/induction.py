"""Domain-agnostic trace induction for learned symbolic operator mirrors.

Induction is deliberately observational: it extracts deltas from already
executed transitions and proposes hypotheses.  Nothing in this module invokes
an action, consults a domain solver, or treats a learned effect as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Protocol, Sequence

from angler.procedures.operators import (
    ActionPattern,
    Constant,
    Effect,
    LearnedOperator,
    OperatorExemplar,
    ReconstructionExemplar,
    RecordPattern,
    SymbolicTerm,
    TypedVariable,
)
from angler.procedures.records import GroundAction, Record, State, Trace, Transition


class InductionError(ValueError):
    """Raised when observations cannot support the requested abstraction."""


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


def _record_payload(record: Record) -> dict[str, Any]:
    return {"predicate": record.predicate, "arguments": list(record.arguments)}


@dataclass(frozen=True, slots=True)
class TransitionDelta:
    """Exact set delta between two observed states in the same namespace."""

    namespace: str
    added: tuple[Record, ...]
    deleted: tuple[Record, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise InductionError("delta namespace must be non-empty")
        for values, label in ((self.added, "added"), (self.deleted, "deleted")):
            if type(values) is not tuple:
                raise InductionError(f"delta {label} records must be a tuple")
            if any(not isinstance(item, Record) for item in values):
                raise InductionError(f"delta {label} values must be records")
            expected = tuple(sorted(values))
            if values != expected or len(set(values)) != len(values):
                raise InductionError(f"delta {label} records must be canonical")
            if any(item.namespace != self.namespace for item in values):
                raise InductionError("delta records must remain within one namespace")
        if set(self.added) & set(self.deleted):
            raise InductionError("a record cannot be both added and deleted")

    @classmethod
    def between(cls, before: State, after: State) -> "TransitionDelta":
        if not isinstance(before, State) or not isinstance(after, State):
            raise TypeError("delta endpoints must be State values")
        if before.namespace != after.namespace:
            raise InductionError("delta endpoints must share a namespace")
        before_records = set(before.records)
        after_records = set(after.records)
        return cls(
            namespace=before.namespace,
            added=tuple(sorted(after_records - before_records)),
            deleted=tuple(sorted(before_records - after_records)),
        )

    @property
    def changed(self) -> bool:
        return bool(self.added or self.deleted)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "added": [_record_payload(item) for item in self.added],
            "deleted": [_record_payload(item) for item in self.deleted],
            "namespace": self.namespace,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class TraceSubsegment:
    """A non-empty contiguous window over one immutable executed trace."""

    trace: Trace
    start_index: int
    stop_index: int
    delta: TransitionDelta

    def __post_init__(self) -> None:
        if not isinstance(self.trace, Trace):
            raise TypeError("subsegment trace must be a Trace")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or isinstance(self.stop_index, bool)
            or not isinstance(self.stop_index, int)
            or self.start_index < 0
            or self.stop_index <= self.start_index
            or self.stop_index > len(self.trace.transitions)
        ):
            raise InductionError("subsegment indices must select a non-empty trace window")
        if not isinstance(self.delta, TransitionDelta):
            raise TypeError("subsegment delta must be a TransitionDelta")
        expected = TransitionDelta.between(self.before, self.after)
        if self.delta != expected:
            raise InductionError("subsegment delta does not match its trace endpoints")

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return self.trace.transitions[self.start_index : self.stop_index]

    @property
    def before(self) -> State:
        return self.trace.initial if self.start_index == 0 else self.trace.transitions[
            self.start_index - 1
        ].after

    @property
    def after(self) -> State:
        return self.trace.transitions[self.stop_index - 1].after

    @property
    def actions(self) -> tuple[GroundAction, ...]:
        return tuple(item.action for item in self.transitions)

    @property
    def namespace(self) -> str:
        return self.before.namespace

    @property
    def relevant_preconditions(self) -> tuple[Record, ...]:
        """Minimal deterministic start facts needed to bind the action roles."""

        return _minimal_connected_precondition_basis(self)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "delta": self.delta.to_canonical(),
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "trace_digest": self.trace.digest,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())

    def reconstruction_exemplar(
        self,
        variable_bindings: tuple[tuple[str, str], ...],
        constant_values: tuple[str, ...],
    ) -> ReconstructionExemplar:
        bound_values = {value for _, value in variable_bindings} | set(
            constant_values
        )
        start_records = tuple(
            item
            for item in self.before.records
            if all(argument in bound_values for argument in item.arguments)
        )
        end_records = tuple(
            item
            for item in self.after.records
            if all(argument in bound_values for argument in item.arguments)
        )
        if len(start_records) > 32 or len(end_records) > 32:
            raise InductionError(
                "a reconstruction exemplar cannot exceed 32 bound records per state"
            )
        return ReconstructionExemplar(
            namespace=self.namespace,
            start_records=start_records,
            variable_bindings=variable_bindings,
            constant_values=constant_values,
            actions=self.actions,
            end_records=end_records,
        )

    def operator_exemplar(
        self,
        reconstruction: ReconstructionExemplar,
    ) -> OperatorExemplar:
        return OperatorExemplar(
            trace_digest=self.trace.digest,
            start_index=self.start_index,
            stop_index=self.stop_index,
            before_state_digest=self.before.digest,
            after_state_digest=self.after.digest,
            action_digests=tuple(item.digest for item in self.actions),
            reconstruction=reconstruction,
        )


def extract_subsegment_delta(
    trace: Trace,
    start_index: int,
    stop_index: int,
) -> TraceSubsegment:
    """Extract one exact observed subsegment without consulting its goal."""

    if not isinstance(trace, Trace):
        raise TypeError("trace must be a Trace")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or isinstance(stop_index, bool)
        or not isinstance(stop_index, int)
        or start_index < 0
        or stop_index <= start_index
        or stop_index > len(trace.transitions)
    ):
        raise InductionError("indices must select a non-empty trace window")
    before = trace.initial if start_index == 0 else trace.transitions[start_index - 1].after
    after = trace.transitions[stop_index - 1].after
    return TraceSubsegment(
        trace=trace,
        start_index=start_index,
        stop_index=stop_index,
        delta=TransitionDelta.between(before, after),
    )


def _minimal_connected_precondition_basis(
    segment: TraceSubsegment,
) -> tuple[Record, ...]:
    """Select an irredundant relational basis without consulting domain rules.

    Deleted effects are causal start facts and therefore fixed seeds.  Other
    invariant facts survive only when they introduce at least one primitive
    action entity not yet bound by those seeds.  A deterministic MDL-style
    greedy cover prefers one fact that binds more missing roles, then a fact
    connected to already-bound roles, then lower literal/arity cost.  A final
    reverse pass removes any residual fact made redundant by later choices.
    """

    action_values = tuple(
        argument
        for action in segment.actions
        for argument in action.arguments
    )
    required = set(action_values)
    before = set(segment.before.records)
    seeds = tuple(
        sorted(record for record in segment.delta.deleted if record in before)
    )
    selected = list(seeds)
    covered = {
        argument
        for record in selected
        for argument in record.arguments
        if argument in required
    }
    candidates = [
        record
        for record in segment.before.records
        if record not in set(seeds)
        and any(argument in required for argument in record.arguments)
    ]

    while required - covered:
        missing = required - covered
        useful = tuple(
            record
            for record in candidates
            if any(argument in missing for argument in record.arguments)
        )
        if not useful:
            break

        def candidate_key(record: Record) -> tuple[Any, ...]:
            arguments = set(record.arguments)
            introduced = len(arguments & missing)
            connected = len(arguments & covered)
            literal_cost = sum(argument not in required for argument in record.arguments)
            return (
                -introduced,
                -connected,
                literal_cost,
                len(record.arguments),
                _basis_record_key(record, action_values),
                record,
            )

        chosen = min(useful, key=candidate_key)
        selected.append(chosen)
        candidates.remove(chosen)
        covered.update(set(chosen.arguments) & required)

    seed_set = set(seeds)
    for record in tuple(reversed(selected)):
        if record in seed_set:
            continue
        without = [item for item in selected if item != record]
        remaining_coverage = {
            argument
            for item in without
            for argument in item.arguments
            if argument in required
        }
        if required <= remaining_coverage:
            selected.remove(record)
    return tuple(sorted(selected))


def _basis_record_key(
    record: Record,
    action_values: tuple[str, ...],
) -> bytes:
    """Structural key stable under renaming of action-bound entities."""

    literal_ids: dict[str, int] = {}
    argument_roles: list[dict[str, Any]] = []
    for argument in record.arguments:
        roles = tuple(
            index
            for index, value in enumerate(action_values)
            if value == argument
        )
        if roles:
            argument_roles.append({"roles": roles})
        else:
            if argument not in literal_ids:
                literal_ids[argument] = len(literal_ids)
            argument_roles.append({"literal": literal_ids[argument]})
    return _canonical_bytes(
        {
            "arguments": argument_roles,
            "predicate": record.predicate,
        }
    )


def extract_trace_subsegments(
    trace: Trace,
    *,
    minimum_length: int = 1,
    maximum_length: int | None = None,
) -> tuple[TraceSubsegment, ...]:
    """Enumerate every bounded contiguous window, independent of outcome/goal."""

    if isinstance(minimum_length, bool) or not isinstance(minimum_length, int):
        raise TypeError("minimum_length must be an integer")
    if minimum_length <= 0:
        raise InductionError("minimum_length must be positive")
    if maximum_length is None:
        maximum_length = len(trace.transitions)
    if isinstance(maximum_length, bool) or not isinstance(maximum_length, int):
        raise TypeError("maximum_length must be an integer or None")
    if maximum_length < minimum_length:
        raise InductionError("maximum_length cannot be smaller than minimum_length")
    result: list[TraceSubsegment] = []
    for start in range(len(trace.transitions)):
        stop_ceiling = min(len(trace.transitions), start + maximum_length)
        for stop in range(start + minimum_length, stop_ceiling + 1):
            result.append(extract_subsegment_delta(trace, start, stop))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EntityAntiUnification:
    """Least-specific typed terms and per-example variable substitutions."""

    terms: tuple[SymbolicTerm, ...]
    variables: tuple[TypedVariable, ...]
    substitutions: tuple[tuple[tuple[str, str], ...], ...]

    def __post_init__(self) -> None:
        if type(self.terms) is not tuple or type(self.variables) is not tuple:
            raise InductionError("anti-unification terms and variables must be tuples")
        if type(self.substitutions) is not tuple:
            raise InductionError("anti-unification substitutions must be a tuple")


def anti_unify_entities(
    rows: Sequence[Sequence[str]],
    *,
    type_rows: Sequence[Sequence[str | None]] | None = None,
    fallback_type: str,
) -> EntityAntiUnification:
    """Anti-unify corresponding entity slots while preserving co-reference.

    Columns whose values are identical in every example remain constants.
    Other columns become variables.  Two columns share a variable exactly when
    their concrete values co-refer in every example.
    """

    return _anti_unify_entities(
        rows,
        type_rows=type_rows,
        fallback_type=fallback_type,
        forced_variable_columns=None,
    )


def _anti_unify_entities(
    rows: Sequence[Sequence[str]],
    *,
    type_rows: Sequence[Sequence[str | None]] | None,
    fallback_type: str,
    forced_variable_columns: Sequence[bool] | None,
) -> EntityAntiUnification:
    """Internal anti-unification with induction-only entity-role forcing."""

    if not isinstance(fallback_type, str) or not fallback_type:
        raise InductionError("fallback_type must be non-empty")
    normalized = tuple(tuple(row) for row in rows)
    if not normalized:
        raise InductionError("anti-unification requires at least one example")
    width = len(normalized[0])
    if any(len(row) != width for row in normalized):
        raise InductionError("anti-unification rows must have equal width")
    if any(not isinstance(value, str) or not value for row in normalized for value in row):
        raise InductionError("anti-unification entities must be non-empty strings")
    if forced_variable_columns is None:
        forced_columns = tuple(False for _ in range(width))
    else:
        forced_columns = tuple(forced_variable_columns)
        if len(forced_columns) != width or any(
            type(value) is not bool for value in forced_columns
        ):
            raise InductionError(
                "forced variable columns must be one bool per entity slot"
            )
    if type_rows is None:
        normalized_types = tuple(
            tuple(fallback_type for _ in range(width)) for _ in normalized
        )
        known_types = tuple(tuple(False for _ in range(width)) for _ in normalized)
    else:
        supplied_types = tuple(tuple(row) for row in type_rows)
        if len(supplied_types) != len(normalized) or any(
            len(row) != width for row in supplied_types
        ):
            raise InductionError("type rows must match entity row shape")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for row in supplied_types
            for value in row
        ):
            raise InductionError("type evidence must be a non-empty string or None")
        normalized_types = tuple(
            tuple(fallback_type if value is None else value for value in row)
            for row in supplied_types
        )
        known_types = tuple(
            tuple(value is not None for value in row) for row in supplied_types
        )

    forced_vectors = {
        tuple(row[column] for row in normalized)
        for column, forced in enumerate(forced_columns)
        if forced
    }
    vector_types: dict[tuple[str, ...], str] = {}
    for column in range(width):
        values = tuple(row[column] for row in normalized)
        if (
            all(value == values[0] for value in values)
            and values not in forced_vectors
        ):
            continue
        concrete_types = {
            normalized_types[row_index][column]
            for row_index in range(len(normalized_types))
            if known_types[row_index][column]
        }
        if len(concrete_types) > 1:
            raise InductionError("one entity role cannot have incompatible types")
        inferred = next(iter(concrete_types), fallback_type)
        prior = vector_types.get(values)
        if prior is None or prior == fallback_type:
            vector_types[values] = inferred
        elif inferred != fallback_type and inferred != prior:
            raise InductionError("co-referent entity slots disagree on type")

    vector_terms: dict[tuple[str, ...], TypedVariable] = {}
    terms: list[SymbolicTerm] = []
    for column in range(width):
        values = tuple(row[column] for row in normalized)
        types = {
            normalized_types[row_index][column]
            for row_index in range(len(normalized_types))
            if known_types[row_index][column]
        }
        if len(types) > 1:
            raise InductionError("one entity role cannot have incompatible types")
        type_name = next(iter(types), fallback_type)
        if (
            all(value == values[0] for value in values)
            and values not in forced_vectors
        ):
            terms.append(Constant(values[0], type_name))
            continue
        type_name = vector_types[values]
        variable = vector_terms.get(values)
        if variable is None:
            variable = TypedVariable(f"v{len(vector_terms)}", type_name)
            vector_terms[values] = variable
        elif variable.type_name != type_name:
            raise InductionError("co-referent entity slots disagree on type")
        terms.append(variable)

    variables = tuple(sorted(vector_terms.values()))
    substitutions = tuple(
        tuple(
            sorted(
                (variable.name, values[row_index])
                for values, variable in vector_terms.items()
            )
        )
        for row_index in range(len(normalized))
    )
    return EntityAntiUnification(tuple(terms), variables, substitutions)


def _record_shape(records: Sequence[Record]) -> tuple[tuple[str, int], ...]:
    return tuple((item.predicate, len(item.arguments)) for item in records)


def _action_shape(action: GroundAction) -> tuple[Any, ...]:
    return (
        action.schema.digest,
        action.schema.name,
        tuple(parameter.type_name for parameter in action.schema.parameters),
    )


def _segment_signature(segment: TraceSubsegment) -> tuple[Any, ...]:
    values, types, forced = _flatten_segment(segment)
    first_occurrence: dict[str, int] = {}
    equality_pattern: list[int] = []
    for value in values:
        if value not in first_occurrence:
            first_occurrence[value] = len(first_occurrence)
        equality_pattern.append(first_occurrence[value])
    return (
        segment.namespace,
        tuple(_action_shape(item) for item in segment.actions),
        tuple((item.applied, item.outcome) for item in segment.transitions),
        _record_shape(segment.relevant_preconditions),
        _record_shape(segment.delta.deleted),
        _record_shape(segment.delta.added),
        tuple(equality_pattern),
        tuple(types),
        tuple(forced),
    )


@dataclass(frozen=True, slots=True)
class SubsegmentCluster:
    """Canonical within-domain group sharing one structural trace signature."""

    namespace: str
    signature: tuple[Any, ...]
    segments: tuple[TraceSubsegment, ...]

    def __post_init__(self) -> None:
        if type(self.segments) is not tuple or not self.segments:
            raise InductionError("a cluster requires a non-empty segment tuple")
        ordered = tuple(sorted(self.segments, key=lambda item: item.digest))
        if len({item.digest for item in ordered}) != len(ordered):
            raise InductionError("a cluster cannot repeat a trace subsegment")
        if any(item.namespace != self.namespace for item in ordered):
            raise InductionError("a cluster cannot cross domain namespaces")
        if any(_segment_signature(item) != self.signature for item in ordered):
            raise InductionError("all clustered segments must share a signature")
        object.__setattr__(self, "segments", ordered)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "namespace": self.namespace,
                "segments": [item.digest for item in self.segments],
                "signature": self.signature,
            }
        )


def cluster_subsegments(
    segments: Iterable[TraceSubsegment],
) -> tuple[SubsegmentCluster, ...]:
    """Group only structurally compatible subsegments from the same namespace."""

    groups: dict[bytes, list[TraceSubsegment]] = {}
    signatures: dict[bytes, tuple[Any, ...]] = {}
    for segment in segments:
        if not isinstance(segment, TraceSubsegment):
            raise TypeError("segments must contain only TraceSubsegment values")
        signature = _segment_signature(segment)
        key = _canonical_bytes(signature)
        signatures[key] = signature
        groups.setdefault(key, []).append(segment)
    return tuple(
        SubsegmentCluster(
            namespace=items[0].namespace,
            signature=signatures[key],
            segments=tuple(items),
        )
        for key, items in sorted(groups.items(), key=lambda item: item[0])
    )


def _entity_types(segment: TraceSubsegment) -> dict[str, str]:
    result: dict[str, str] = {}
    for action in segment.actions:
        for parameter, value in zip(
            action.schema.parameters,
            action.arguments,
            strict=True,
        ):
            existing = result.get(value)
            if existing is not None and existing != parameter.type_name:
                raise InductionError("one observed entity has incompatible action types")
            result[value] = parameter.type_name
    return result


def _flatten_segment(
    segment: TraceSubsegment,
) -> tuple[tuple[str, ...], tuple[str | None, ...], tuple[bool, ...]]:
    values: list[str] = []
    types: list[str | None] = []
    forced: list[bool] = []
    type_map = _entity_types(segment)
    record_groups = (
        segment.relevant_preconditions,
        segment.delta.deleted,
        segment.delta.added,
    )
    for records in record_groups:
        for record in records:
            values.extend(record.arguments)
            types.extend(type_map.get(argument) for argument in record.arguments)
            forced.extend(argument in type_map for argument in record.arguments)
    for action in segment.actions:
        values.extend(action.arguments)
        types.extend(parameter.type_name for parameter in action.schema.parameters)
        forced.extend(True for _ in action.arguments)
    return tuple(values), tuple(types), tuple(forced)


def _lift_records(
    records: Sequence[Record],
    terms: Sequence[SymbolicTerm],
    cursor: int,
) -> tuple[tuple[RecordPattern, ...], int]:
    result: list[RecordPattern] = []
    for record in records:
        stop = cursor + len(record.arguments)
        result.append(RecordPattern(record.predicate, tuple(terms[cursor:stop])))
        cursor = stop
    return tuple(result), cursor


def _induce_operator(
    cluster: SubsegmentCluster,
    *,
    revision: int = 1,
    parent_digest: str | None = None,
) -> LearnedOperator:
    if not any(item.delta.changed for item in cluster.segments):
        raise InductionError("effect-free segments cannot allocate an operator")
    rows_types_and_forcing = tuple(
        _flatten_segment(item) for item in cluster.segments
    )
    forced_columns = tuple(
        any(item[2][column] for item in rows_types_and_forcing)
        for column in range(len(rows_types_and_forcing[0][0]))
    )
    unification = _anti_unify_entities(
        tuple(item[0] for item in rows_types_and_forcing),
        type_rows=tuple(item[1] for item in rows_types_and_forcing),
        fallback_type=cluster.namespace + ".untyped",
        forced_variable_columns=forced_columns,
    )
    reference = cluster.segments[0]
    cursor = 0
    preconditions, cursor = _lift_records(
        reference.relevant_preconditions,
        unification.terms,
        cursor,
    )
    deleted, cursor = _lift_records(reference.delta.deleted, unification.terms, cursor)
    added, cursor = _lift_records(reference.delta.added, unification.terms, cursor)
    body: list[ActionPattern] = []
    for action in reference.actions:
        stop = cursor + len(action.arguments)
        body.append(ActionPattern(action.schema, tuple(unification.terms[cursor:stop])))
        cursor = stop
    if cursor != len(unification.terms):
        raise InductionError("internal anti-unification layout mismatch")
    effects = tuple(
        [Effect("delete", item) for item in deleted]
        + [Effect("add", item) for item in added]
    )
    name_seed = _digest({"namespace": cluster.namespace, "signature": cluster.signature})
    name = f"{cluster.namespace}.learned_{name_seed.removeprefix('sha256:')[:16]}"
    constant_values = tuple(
        sorted(
            {
                term.value
                for term in unification.terms
                if isinstance(term, Constant)
            }
        )
    )
    exemplars: list[OperatorExemplar] = []
    for segment, bindings in zip(
        cluster.segments,
        unification.substitutions,
        strict=True,
    ):
        reconstruction = segment.reconstruction_exemplar(bindings, constant_values)
        exemplars.append(segment.operator_exemplar(reconstruction))
    return LearnedOperator(
        name=name,
        namespace=cluster.namespace,
        variables=unification.variables,
        preconditions=preconditions,
        effects=effects,
        body=tuple(body),
        exemplars=tuple(exemplars),
        revision=revision,
        parent_digest=parent_digest,
    )


def _symbol_cost(name: str) -> int:
    return max(1, math.ceil(len(name.encode("utf-8")) / 8))


def _record_cost(record: Record) -> int:
    return _symbol_cost(record.predicate) + len(record.arguments)


@dataclass(frozen=True, slots=True)
class MDLScore:
    """Integer description-cost comparison; lower candidate cost is better."""

    model_cost: int
    data_cost: int
    exception_cost: int
    raw_cost: int

    def __post_init__(self) -> None:
        for value in (
            self.model_cost,
            self.data_cost,
            self.exception_cost,
            self.raw_cost,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InductionError("MDL costs must be non-negative integers")

    @property
    def candidate_cost(self) -> int:
        return self.model_cost + self.data_cost + self.exception_cost

    @property
    def savings(self) -> int:
        return self.raw_cost - self.candidate_cost


def score_mdl(operator: LearnedOperator, cluster: SubsegmentCluster) -> MDLScore:
    """Score abstraction versus literal storage using deterministic token costs."""

    if not isinstance(operator, LearnedOperator):
        raise TypeError("operator must be a LearnedOperator")
    if not isinstance(cluster, SubsegmentCluster):
        raise TypeError("cluster must be a SubsegmentCluster")
    if operator.namespace != cluster.namespace:
        raise InductionError("operator and cluster namespaces must match")

    model_cost = (
        1
        + len(operator.variables)
        + sum(1 + len(item.arguments) for item in operator.preconditions)
        + sum(1 + len(item.record.arguments) for item in operator.effects)
        + sum(1 + len(item.arguments) for item in operator.body)
    )
    data_cost = len(cluster.segments) * len(operator.variables)
    exception_cost = sum(
        1
        for segment in cluster.segments
        for transition in segment.transitions
        if not transition.applied
    )
    raw_cost = 0
    for segment in cluster.segments:
        raw_cost += sum(_record_cost(item) for item in segment.relevant_preconditions)
        raw_cost += sum(_record_cost(item) for item in segment.delta.added)
        raw_cost += sum(_record_cost(item) for item in segment.delta.deleted)
        raw_cost += sum(
            _symbol_cost(action.schema.name) + len(action.arguments)
            for action in segment.actions
        )
    return MDLScore(model_cost, data_cost, exception_cost, raw_cost)


@dataclass(frozen=True, slots=True)
class OperatorCandidate:
    """An unpromoted operator hypothesis plus the observations needed to revise it."""

    operator: LearnedOperator
    score: MDLScore
    cluster_digest: str
    supporting_segments: tuple[TraceSubsegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operator, LearnedOperator):
            raise TypeError("candidate operator must be a LearnedOperator")
        if not isinstance(self.score, MDLScore):
            raise TypeError("candidate score must be an MDLScore")
        if not isinstance(self.cluster_digest, str) or not self.cluster_digest.startswith(
            "sha256:"
        ):
            raise InductionError("candidate cluster_digest must be sha256")
        if type(self.supporting_segments) is not tuple or not self.supporting_segments:
            raise InductionError("candidate requires supporting trace subsegments")


class CandidateAllocator(Protocol):
    def allocate(self, cluster: SubsegmentCluster) -> OperatorCandidate | None:
        """Allocate an unpromoted hypothesis, or reject insufficient evidence."""


class CandidateRefiner(Protocol):
    def refine(
        self,
        candidate: OperatorCandidate,
        additional: Iterable[TraceSubsegment],
    ) -> OperatorCandidate | None:
        """Return a new candidate revision without mutating the prior candidate."""


@dataclass(frozen=True, slots=True)
class MDLOperatorInducer:
    """Allocate/refine candidates under explicit support and MDL thresholds."""

    minimum_support: int = 2
    minimum_savings: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_support, bool)
            or not isinstance(self.minimum_support, int)
            or self.minimum_support <= 0
        ):
            raise InductionError("minimum_support must be a positive integer")
        if (
            isinstance(self.minimum_savings, bool)
            or not isinstance(self.minimum_savings, int)
        ):
            raise InductionError("minimum_savings must be an integer")

    def allocate(self, cluster: SubsegmentCluster) -> OperatorCandidate | None:
        if not isinstance(cluster, SubsegmentCluster):
            raise TypeError("cluster must be a SubsegmentCluster")
        if len(cluster.segments) < self.minimum_support:
            return None
        try:
            operator = _induce_operator(cluster)
        except InductionError:
            return None
        score = score_mdl(operator, cluster)
        if score.savings < self.minimum_savings:
            return None
        return OperatorCandidate(operator, score, cluster.digest, cluster.segments)

    def refine(
        self,
        candidate: OperatorCandidate,
        additional: Iterable[TraceSubsegment],
    ) -> OperatorCandidate | None:
        if not isinstance(candidate, OperatorCandidate):
            raise TypeError("candidate must be an OperatorCandidate")
        combined = {
            item.digest: item
            for item in candidate.supporting_segments
        }
        for item in additional:
            if not isinstance(item, TraceSubsegment):
                raise TypeError("additional values must be TraceSubsegment")
            combined[item.digest] = item
        clusters = cluster_subsegments(combined.values())
        if len(clusters) != 1:
            return None
        cluster = clusters[0]
        try:
            operator = _induce_operator(
                cluster,
                revision=candidate.operator.revision + 1,
                parent_digest=candidate.operator.digest,
            )
        except InductionError:
            return None
        score = score_mdl(operator, cluster)
        if len(cluster.segments) < self.minimum_support or score.savings < self.minimum_savings:
            return None
        return OperatorCandidate(operator, score, cluster.digest, cluster.segments)


__all__ = [
    "CandidateAllocator",
    "CandidateRefiner",
    "EntityAntiUnification",
    "InductionError",
    "MDLOperatorInducer",
    "MDLScore",
    "OperatorCandidate",
    "SubsegmentCluster",
    "TraceSubsegment",
    "TransitionDelta",
    "anti_unify_entities",
    "cluster_subsegments",
    "extract_subsegment_delta",
    "extract_trace_subsegments",
    "score_mdl",
]
