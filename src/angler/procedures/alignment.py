"""Cross-domain structural alignment for learned symbolic operators.

This module proposes isomorphisms; it never executes either domain.  Aliases
and merge authorization require an opaque counterfactual-execution certificate
supplied by an external evaluator.  No function here can manufacture or pass
such a certificate from an operator prediction alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import re
from typing import Any, Iterable, Literal

from angler.procedures.operators import (
    ActionPattern,
    Constant,
    Effect,
    LearnedOperator,
    RecordPattern,
    SymbolicTerm,
    TypedVariable,
)
from angler.procedures.records import Goal, Record, State


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
AliasKind = Literal["operator", "predicate", "action", "type"]
CANONICAL_PROJECTION_NAMESPACE = "angler.certified_alias"


class AlignmentError(ValueError):
    """Raised when an alignment or evidence binding is invalid."""


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


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AlignmentError(f"{label} must be a canonical sha256 digest")


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AlignmentError(f"{label} must be non-empty stripped text")


@dataclass(frozen=True, slots=True, order=True)
class SymbolAlias:
    """One directional symbolic correspondence proposed by an isomorphism."""

    kind: AliasKind
    source: str
    target: str

    def __post_init__(self) -> None:
        if self.kind not in ("operator", "predicate", "action", "type"):
            raise AlignmentError("unsupported alias kind")
        _require_text(self.source, "alias source")
        _require_text(self.target, "alias target")

    def to_canonical(self) -> dict[str, str]:
        return {"kind": self.kind, "source": self.source, "target": self.target}


def _term_payload(term: SymbolicTerm) -> dict[str, str]:
    return term.to_canonical()


def _pattern_payload(pattern: RecordPattern) -> dict[str, Any]:
    return {
        "arguments": [_term_payload(item) for item in pattern.arguments],
        "predicate": pattern.predicate,
    }


def _effect_payload(effect: Effect) -> dict[str, Any]:
    return {"kind": effect.kind, "record": _pattern_payload(effect.record)}


@dataclass(frozen=True, slots=True)
class AlignmentCoverage:
    """Exact common-template coverage; counts never imply certification."""

    matched_preconditions: int
    source_preconditions: int
    target_preconditions: int
    matched_effects: int
    source_effects: int
    target_effects: int

    def __post_init__(self) -> None:
        values = (
            self.matched_preconditions,
            self.source_preconditions,
            self.target_preconditions,
            self.matched_effects,
            self.source_effects,
            self.target_effects,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise AlignmentError("alignment coverage counts must be non-negative integers")
        if self.matched_preconditions > min(
            self.source_preconditions, self.target_preconditions
        ):
            raise AlignmentError("matched preconditions exceed an endpoint")
        if self.matched_effects > min(self.source_effects, self.target_effects):
            raise AlignmentError("matched effects exceed an endpoint")

    def to_canonical(self) -> dict[str, int]:
        return {
            "matched_effects": self.matched_effects,
            "matched_preconditions": self.matched_preconditions,
            "source_effects": self.source_effects,
            "source_preconditions": self.source_preconditions,
            "target_effects": self.target_effects,
            "target_preconditions": self.target_preconditions,
        }


@dataclass(frozen=True, slots=True)
class DomainResiduals:
    """Constraints/effects deliberately kept local to each aligned domain."""

    source_preconditions: tuple[RecordPattern, ...]
    target_preconditions: tuple[RecordPattern, ...]
    source_effects: tuple[Effect, ...]
    target_effects: tuple[Effect, ...]

    def __post_init__(self) -> None:
        fields = (
            (self.source_preconditions, RecordPattern, "source_preconditions"),
            (self.target_preconditions, RecordPattern, "target_preconditions"),
            (self.source_effects, Effect, "source_effects"),
            (self.target_effects, Effect, "target_effects"),
        )
        for values, expected, label in fields:
            if type(values) is not tuple or any(
                not isinstance(item, expected) for item in values
            ):
                raise AlignmentError(f"{label} has invalid residual values")
        object.__setattr__(
            self,
            "source_preconditions",
            tuple(sorted(self.source_preconditions, key=lambda item: _canonical_bytes(_pattern_payload(item)))),
        )
        object.__setattr__(
            self,
            "target_preconditions",
            tuple(sorted(self.target_preconditions, key=lambda item: _canonical_bytes(_pattern_payload(item)))),
        )
        object.__setattr__(
            self,
            "source_effects",
            tuple(sorted(self.source_effects, key=lambda item: _canonical_bytes(_effect_payload(item)))),
        )
        object.__setattr__(
            self,
            "target_effects",
            tuple(sorted(self.target_effects, key=lambda item: _canonical_bytes(_effect_payload(item)))),
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "source_effects": [_effect_payload(item) for item in self.source_effects],
            "source_preconditions": [
                _pattern_payload(item) for item in self.source_preconditions
            ],
            "target_effects": [_effect_payload(item) for item in self.target_effects],
            "target_preconditions": [
                _pattern_payload(item) for item in self.target_preconditions
            ],
        }


@dataclass(frozen=True, slots=True)
class StructuralIsomorphismCandidate:
    """Unverified structural correspondence between two operator mirrors."""

    source_operator_digest: str
    target_operator_digest: str
    aliases: tuple[SymbolAlias, ...]
    variable_map: tuple[tuple[str, str], ...]
    matched_precondition_pairs: tuple[tuple[int, int], ...]
    matched_effect_pairs: tuple[tuple[int, int], ...]
    coverage: AlignmentCoverage
    residuals: DomainResiduals

    def __post_init__(self) -> None:
        _require_digest(self.source_operator_digest, "source_operator_digest")
        _require_digest(self.target_operator_digest, "target_operator_digest")
        if self.source_operator_digest == self.target_operator_digest:
            raise AlignmentError("alignment requires distinct operator identities")
        if type(self.aliases) is not tuple or not self.aliases:
            raise AlignmentError("an isomorphism candidate requires aliases")
        aliases = tuple(sorted(self.aliases))
        if len(set(aliases)) != len(aliases):
            raise AlignmentError("candidate aliases must be unique")
        if sum(item.kind == "operator" for item in aliases) != 1:
            raise AlignmentError("candidate requires exactly one operator alias")
        _validate_injective_aliases(aliases)
        if type(self.variable_map) is not tuple:
            raise AlignmentError("variable_map must be a tuple")
        variable_map = tuple(sorted(self.variable_map))
        if any(
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
            for source, target in variable_map
        ):
            raise AlignmentError("variable correspondences must contain names")
        if len({item[0] for item in variable_map}) != len(variable_map) or len(
            {item[1] for item in variable_map}
        ) != len(variable_map):
            raise AlignmentError("variable correspondence must be bijective")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "variable_map", variable_map)
        for pairs, label in (
            (self.matched_precondition_pairs, "matched_precondition_pairs"),
            (self.matched_effect_pairs, "matched_effect_pairs"),
        ):
            if type(pairs) is not tuple or any(
                type(item) is not tuple
                or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in item)
                for item in pairs
            ):
                raise AlignmentError(f"{label} must contain index pairs")
            if len({item[0] for item in pairs}) != len(pairs) or len(
                {item[1] for item in pairs}
            ) != len(pairs):
                raise AlignmentError(f"{label} must be one-to-one")
            object.__setattr__(self, label, tuple(sorted(pairs)))
        if not isinstance(self.coverage, AlignmentCoverage):
            raise TypeError("candidate coverage has the wrong type")
        if not isinstance(self.residuals, DomainResiduals):
            raise TypeError("candidate residuals have the wrong type")
        if self.coverage.matched_preconditions != len(self.matched_precondition_pairs):
            raise AlignmentError("precondition coverage does not match pair evidence")
        if self.coverage.matched_effects != len(self.matched_effect_pairs):
            raise AlignmentError("effect coverage does not match pair evidence")
        if any(
            source >= self.coverage.source_preconditions
            or target >= self.coverage.target_preconditions
            for source, target in self.matched_precondition_pairs
        ):
            raise AlignmentError("matched precondition index exceeds coverage bounds")
        if any(
            source >= self.coverage.source_effects
            or target >= self.coverage.target_effects
            for source, target in self.matched_effect_pairs
        ):
            raise AlignmentError("matched effect index exceeds coverage bounds")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "aliases": [item.to_canonical() for item in self.aliases],
            "source_operator_digest": self.source_operator_digest,
            "target_operator_digest": self.target_operator_digest,
            "variable_map": [list(item) for item in self.variable_map],
            "matched_precondition_pairs": [
                list(item) for item in self.matched_precondition_pairs
            ],
            "matched_effect_pairs": [list(item) for item in self.matched_effect_pairs],
            "coverage": self.coverage.to_canonical(),
            "residuals": self.residuals.to_canonical(),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


def _validate_injective_aliases(aliases: Iterable[SymbolAlias]) -> None:
    forward: dict[tuple[str, str], str] = {}
    reverse: dict[tuple[str, str], str] = {}
    for item in aliases:
        key = (item.kind, item.source)
        reverse_key = (item.kind, item.target)
        if key in forward and forward[key] != item.target:
            raise AlignmentError("one source symbol cannot alias multiple targets")
        if reverse_key in reverse and reverse[reverse_key] != item.source:
            raise AlignmentError("symbol aliases must be injective")
        forward[key] = item.target
        reverse[reverse_key] = item.source


@dataclass(slots=True)
class _Mapping:
    symbols: dict[str, dict[str, str]]
    symbol_reverse: dict[str, dict[str, str]]
    constants: dict[tuple[str, str], tuple[str, str]]
    constant_reverse: dict[tuple[str, str], tuple[str, str]]

    @classmethod
    def empty(cls) -> "_Mapping":
        kinds = ("operator", "predicate", "action", "type")
        return cls(
            symbols={kind: {} for kind in kinds},
            symbol_reverse={kind: {} for kind in kinds},
            constants={},
            constant_reverse={},
        )

    def copy(self) -> "_Mapping":
        return _Mapping(
            symbols={kind: dict(values) for kind, values in self.symbols.items()},
            symbol_reverse={
                kind: dict(values) for kind, values in self.symbol_reverse.items()
            },
            constants=dict(self.constants),
            constant_reverse=dict(self.constant_reverse),
        )

    def bind(self, kind: str, source: str, target: str) -> bool:
        prior = self.symbols[kind].get(source)
        reverse = self.symbol_reverse[kind].get(target)
        if (prior is not None and prior != target) or (
            reverse is not None and reverse != source
        ):
            return False
        self.symbols[kind][source] = target
        self.symbol_reverse[kind][target] = source
        return True

    def bind_constant(self, source: Constant, target: Constant) -> bool:
        source_key = (source.type_name, source.value)
        target_key = (target.type_name, target.value)
        prior = self.constants.get(source_key)
        reverse = self.constant_reverse.get(target_key)
        if (prior is not None and prior != target_key) or (
            reverse is not None and reverse != source_key
        ):
            return False
        self.constants[source_key] = target_key
        self.constant_reverse[target_key] = source_key
        return self.bind("type", source.type_name, target.type_name)


def _pair_term(
    source: SymbolicTerm,
    target: SymbolicTerm,
    variable_map: dict[str, str],
    mapping: _Mapping,
) -> bool:
    if isinstance(source, TypedVariable) and isinstance(target, TypedVariable):
        return variable_map.get(source.name) == target.name and mapping.bind(
            "type", source.type_name, target.type_name
        )
    if isinstance(source, Constant) and isinstance(target, Constant):
        return mapping.bind_constant(source, target)
    return False


def _pair_record(
    source: RecordPattern,
    target: RecordPattern,
    variable_map: dict[str, str],
    mapping: _Mapping,
) -> bool:
    if len(source.arguments) != len(target.arguments):
        return False
    if not mapping.bind("predicate", source.predicate, target.predicate):
        return False
    return all(
        _pair_term(left, right, variable_map, mapping)
        for left, right in zip(source.arguments, target.arguments, strict=True)
    )


def _pair_action(
    source: ActionPattern,
    target: ActionPattern,
    variable_map: dict[str, str],
    mapping: _Mapping,
) -> bool:
    if len(source.arguments) != len(target.arguments):
        return False
    if len(source.schema.parameters) != len(target.schema.parameters):
        return False
    if not mapping.bind("action", source.schema.name, target.schema.name):
        return False
    for left, right in zip(
        source.schema.parameters,
        target.schema.parameters,
        strict=True,
    ):
        if not mapping.bind("type", left.type_name, right.type_name):
            return False
    return all(
        _pair_term(left, right, variable_map, mapping)
        for left, right in zip(source.arguments, target.arguments, strict=True)
    )


@dataclass(slots=True)
class _PartialMatch:
    mapping: _Mapping
    pairs: tuple[tuple[int, int], ...]


def _maximal_record_matches(
    source: tuple[RecordPattern, ...],
    target: tuple[RecordPattern, ...],
    variable_map: dict[str, str],
    mapping: _Mapping,
) -> tuple[_PartialMatch, ...]:
    def visit(
        index: int,
        remaining: tuple[int, ...],
        state: _Mapping,
        pairs: tuple[tuple[int, int], ...],
    ) -> list[_PartialMatch]:
        if index == len(source):
            return [_PartialMatch(state, pairs)]
        results = visit(index + 1, remaining, state.copy(), pairs)
        current = source[index]
        for target_index in remaining:
            candidate = target[target_index]
            if len(current.arguments) != len(candidate.arguments):
                continue
            next_state = state.copy()
            if not _pair_record(current, candidate, variable_map, next_state):
                continue
            results.extend(
                visit(
                    index + 1,
                    tuple(item for item in remaining if item != target_index),
                    next_state,
                    pairs + ((index, target_index),),
                )
            )
        return results

    results = visit(0, tuple(range(len(target))), mapping, ())
    maximum = max((len(item.pairs) for item in results), default=0)
    return tuple(item for item in results if len(item.pairs) == maximum)


def _maximal_effect_matches(
    source: tuple[Effect, ...],
    target: tuple[Effect, ...],
    variable_map: dict[str, str],
    mapping: _Mapping,
) -> tuple[_PartialMatch, ...]:
    def visit(
        index: int,
        remaining: tuple[int, ...],
        state: _Mapping,
        pairs: tuple[tuple[int, int], ...],
    ) -> list[_PartialMatch]:
        if index == len(source):
            return [_PartialMatch(state, pairs)]
        results = visit(index + 1, remaining, state.copy(), pairs)
        current = source[index]
        for target_index in remaining:
            candidate = target[target_index]
            if current.kind != candidate.kind:
                continue
            next_state = state.copy()
            if not _pair_record(current.record, candidate.record, variable_map, next_state):
                continue
            results.extend(
                visit(
                    index + 1,
                    tuple(item for item in remaining if item != target_index),
                    next_state,
                    pairs + ((index, target_index),),
                )
            )
        return results

    results = visit(0, tuple(range(len(target))), mapping, ())
    maximum = max((len(item.pairs) for item in results), default=0)
    return tuple(item for item in results if len(item.pairs) == maximum)


def _term_identity(term: SymbolicTerm) -> tuple[str, ...]:
    if isinstance(term, TypedVariable):
        return ("variable", term.name)
    return ("constant", term.type_name, term.value)


def _is_relocation_pair(delete: Effect, add: Effect) -> bool:
    if (
        delete.kind != "delete"
        or add.kind != "add"
        or delete.record.predicate != add.record.predicate
        or len(delete.record.arguments) != len(add.record.arguments)
        or len(delete.record.arguments) < 2
    ):
        return False
    before = tuple(_term_identity(item) for item in delete.record.arguments)
    after = tuple(_term_identity(item) for item in add.record.arguments)
    return any(left == right for left, right in zip(before, after, strict=True)) and any(
        left != right for left, right in zip(before, after, strict=True)
    )


def _contains_paired_relocation(
    source_effects: tuple[Effect, ...],
    target_effects: tuple[Effect, ...],
    pairs: tuple[tuple[int, int], ...],
) -> bool:
    for source_delete, target_delete in pairs:
        for source_add, target_add in pairs:
            if _is_relocation_pair(
                source_effects[source_delete], source_effects[source_add]
            ) and _is_relocation_pair(
                target_effects[target_delete], target_effects[target_add]
            ):
                return True
    return False


def _body_variables(operator: LearnedOperator) -> tuple[TypedVariable, ...]:
    """Return variables participating in the executable causal core.

    Variables mentioned only by preconditions are domain-local constraint
    witnesses.  Requiring them to participate in a cross-domain bijection
    would make residual-aware alignment equivalent to full-precondition
    equality, contrary to the purpose of retaining unmatched constraints.
    """

    names = {
        term.name
        for action in operator.body
        for term in action.arguments
        if isinstance(term, TypedVariable)
    }
    return tuple(item for item in operator.variables if item.name in names)


def find_structural_isomorphisms(
    source: LearnedOperator,
    target: LearnedOperator,
    *,
    maximum_candidates: int = 64,
    minimum_common_effects: int = 2,
) -> tuple[StructuralIsomorphismCandidate, ...]:
    """Find name/description-independent structural bijections.

    Action order, add/delete polarity, arity, typed-role equality, constants,
    and variable co-reference are preserved.  Predicate, action, type, operator,
    namespace, and description text are allowed to differ.
    """

    if not isinstance(source, LearnedOperator) or not isinstance(target, LearnedOperator):
        raise TypeError("alignment endpoints must be LearnedOperator values")
    if source.digest == target.digest:
        return ()
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates <= 0
    ):
        raise AlignmentError("maximum_candidates must be a positive integer")
    if (
        isinstance(minimum_common_effects, bool)
        or not isinstance(minimum_common_effects, int)
        or minimum_common_effects < 2
    ):
        raise AlignmentError("minimum_common_effects must be at least two")
    if len(source.body) != len(target.body):
        return ()

    results: dict[str, StructuralIsomorphismCandidate] = {}
    source_variables = _body_variables(source)
    target_variables = _body_variables(target)
    if len(source_variables) != len(target_variables):
        return ()
    for permutation in itertools.permutations(target_variables):
        variable_map = {
            left.name: right.name
            for left, right in zip(source_variables, permutation, strict=True)
        }
        mapping = _Mapping.empty()
        if not mapping.bind("operator", source.name, target.name):
            continue
        compatible = True
        for left, right in zip(source_variables, permutation, strict=True):
            if not mapping.bind("type", left.type_name, right.type_name):
                compatible = False
                break
        if not compatible:
            continue
        for left, right in zip(source.body, target.body, strict=True):
            if not _pair_action(left, right, variable_map, mapping):
                compatible = False
                break
        if not compatible:
            continue
        effect_matches = _maximal_effect_matches(
            source.effects,
            target.effects,
            variable_map,
            mapping,
        )
        for effect_match in effect_matches:
            if len(effect_match.pairs) < minimum_common_effects or not _contains_paired_relocation(
                source.effects,
                target.effects,
                effect_match.pairs,
            ):
                continue
            precondition_matches = _maximal_record_matches(
                source.preconditions,
                target.preconditions,
                variable_map,
                effect_match.mapping,
            )
            for precondition_match in precondition_matches:
                final_mapping = precondition_match.mapping
                aliases = tuple(
                    SymbolAlias(kind, left, right)
                    for kind in ("operator", "predicate", "action", "type")
                    for left, right in final_mapping.symbols[kind].items()
                )
                candidate = StructuralIsomorphismCandidate(
                    source_operator_digest=source.digest,
                    target_operator_digest=target.digest,
                    aliases=aliases,
                    variable_map=tuple(variable_map.items()),
                    matched_precondition_pairs=precondition_match.pairs,
                    matched_effect_pairs=effect_match.pairs,
                    coverage=AlignmentCoverage(
                        matched_preconditions=len(precondition_match.pairs),
                        source_preconditions=len(source.preconditions),
                        target_preconditions=len(target.preconditions),
                        matched_effects=len(effect_match.pairs),
                        source_effects=len(source.effects),
                        target_effects=len(target.effects),
                    ),
                    residuals=DomainResiduals(
                        source_preconditions=tuple(
                            item
                            for index, item in enumerate(source.preconditions)
                            if index
                            not in {pair[0] for pair in precondition_match.pairs}
                        ),
                        target_preconditions=tuple(
                            item
                            for index, item in enumerate(target.preconditions)
                            if index
                            not in {pair[1] for pair in precondition_match.pairs}
                        ),
                        source_effects=tuple(
                            item
                            for index, item in enumerate(source.effects)
                            if index not in {pair[0] for pair in effect_match.pairs}
                        ),
                        target_effects=tuple(
                            item
                            for index, item in enumerate(target.effects)
                            if index not in {pair[1] for pair in effect_match.pairs}
                        ),
                    ),
                )
                results[candidate.digest] = candidate
                if len(results) >= maximum_candidates:
                    return tuple(results[key] for key in sorted(results))
    return tuple(results[key] for key in sorted(results))


@dataclass(frozen=True, slots=True)
class CounterfactualExecutionCertificate:
    """Opaque evidence issued by an evaluator outside this module."""

    candidate_digest: str
    execution_digest: str
    result_digest: str
    result: Literal["pass", "fail"]
    issued_by: str

    def __post_init__(self) -> None:
        _require_digest(self.candidate_digest, "certificate candidate_digest")
        _require_digest(self.execution_digest, "certificate execution_digest")
        _require_digest(self.result_digest, "certificate result_digest")
        if self.result not in ("pass", "fail"):
            raise AlignmentError("certificate result must be 'pass' or 'fail'")
        _require_text(self.issued_by, "certificate issued_by")

    def to_canonical(self) -> dict[str, str]:
        return {
            "candidate_digest": self.candidate_digest,
            "execution_digest": self.execution_digest,
            "issued_by": self.issued_by,
            "result": self.result,
            "result_digest": self.result_digest,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class VerifiedAliasEntry:
    """Aliases admitted only with a passing, candidate-bound certificate."""

    candidate_digest: str
    source_operator_digest: str
    target_operator_digest: str
    aliases: tuple[SymbolAlias, ...]
    variable_map: tuple[tuple[str, str], ...]
    certificate: CounterfactualExecutionCertificate

    def __post_init__(self) -> None:
        _require_digest(self.candidate_digest, "alias-entry candidate_digest")
        _require_digest(self.source_operator_digest, "alias-entry source_operator_digest")
        _require_digest(self.target_operator_digest, "alias-entry target_operator_digest")
        if type(self.aliases) is not tuple or not self.aliases:
            raise AlignmentError("verified alias entry requires aliases")
        aliases = tuple(sorted(self.aliases))
        _validate_injective_aliases(aliases)
        if type(self.variable_map) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in self.variable_map
        ) or len({item[0] for item in self.variable_map}) != len(
            self.variable_map
        ) or len({item[1] for item in self.variable_map}) != len(self.variable_map):
            raise AlignmentError("alias-entry variable_map must be bijective")
        object.__setattr__(self, "variable_map", tuple(sorted(self.variable_map)))
        if not isinstance(self.certificate, CounterfactualExecutionCertificate):
            raise TypeError("alias-entry certificate has the wrong type")
        if self.certificate.candidate_digest != self.candidate_digest:
            raise AlignmentError("certificate does not bind the alias candidate")
        if self.certificate.result != "pass":
            raise AlignmentError("failed counterfactual evidence cannot admit aliases")
        object.__setattr__(self, "aliases", aliases)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "aliases": [item.to_canonical() for item in self.aliases],
            "candidate_digest": self.candidate_digest,
            "certificate": self.certificate.to_canonical(),
            "source_operator_digest": self.source_operator_digest,
            "target_operator_digest": self.target_operator_digest,
            "variable_map": [list(item) for item in self.variable_map],
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class CanonicalStateProjection:
    """Immutable record view for certified cross-domain state encoding."""

    source_state_digest: str
    records: tuple[Record, ...]
    namespace: str = CANONICAL_PROJECTION_NAMESPACE

    def __post_init__(self) -> None:
        _require_digest(self.source_state_digest, "source_state_digest")
        if self.namespace != CANONICAL_PROJECTION_NAMESPACE:
            raise AlignmentError("canonical state projection namespace is fixed")
        if type(self.records) is not tuple or any(
            not isinstance(item, Record) for item in self.records
        ):
            raise AlignmentError("canonical state records must be a tuple")
        canonical = tuple(sorted(set(self.records)))
        object.__setattr__(self, "records", canonical)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "namespace": self.namespace,
                "records": [
                    {"predicate": item.predicate, "arguments": list(item.arguments)}
                    for item in self.records
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class CanonicalGoalProjection:
    """Immutable required/forbidden view for certified goal encoding."""

    source_goal_digest: str
    required: tuple[Record, ...]
    forbidden: tuple[Record, ...]
    exact: bool
    namespace: str = CANONICAL_PROJECTION_NAMESPACE

    def __post_init__(self) -> None:
        _require_digest(self.source_goal_digest, "source_goal_digest")
        if self.namespace != CANONICAL_PROJECTION_NAMESPACE:
            raise AlignmentError("canonical goal projection namespace is fixed")
        for values, label in (
            (self.required, "required"),
            (self.forbidden, "forbidden"),
        ):
            if type(values) is not tuple or any(
                not isinstance(item, Record) for item in values
            ):
                raise AlignmentError(f"canonical goal {label} must be a tuple")
            object.__setattr__(self, label, tuple(sorted(set(values))))
        if type(self.exact) is not bool:
            raise AlignmentError("canonical goal exact must be bool")
        if set(self.required) & set(self.forbidden):
            raise AlignmentError("canonical goal facts cannot conflict")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "exact": self.exact,
                "namespace": self.namespace,
                "forbidden": [
                    {"predicate": item.predicate, "arguments": list(item.arguments)}
                    for item in self.forbidden
                ],
                "required": [
                    {"predicate": item.predicate, "arguments": list(item.arguments)}
                    for item in self.required
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class AliasTable:
    """An immutable, append-only set of externally certified aliases."""

    entries: tuple[VerifiedAliasEntry, ...] = ()

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise AlignmentError("alias-table entries must be a tuple")
        if any(not isinstance(item, VerifiedAliasEntry) for item in self.entries):
            raise TypeError("alias table contains an invalid entry")
        entries = tuple(sorted(self.entries, key=lambda item: item.digest))
        if len({item.candidate_digest for item in entries}) != len(entries):
            raise AlignmentError("an alias candidate can be admitted only once")
        _validate_injective_aliases(
            alias for entry in entries for alias in entry.aliases
        )
        object.__setattr__(self, "entries", entries)

    def with_certificate(
        self,
        candidate: StructuralIsomorphismCandidate,
        certificate: CounterfactualExecutionCertificate,
    ) -> "AliasTable":
        if not isinstance(candidate, StructuralIsomorphismCandidate):
            raise TypeError("candidate has the wrong type")
        if not isinstance(certificate, CounterfactualExecutionCertificate):
            raise TypeError("certificate has the wrong type")
        entry = VerifiedAliasEntry(
            candidate.digest,
            candidate.source_operator_digest,
            candidate.target_operator_digest,
            candidate.aliases,
            candidate.variable_map,
            certificate,
        )
        return AliasTable(self.entries + (entry,))

    def entry_for(self, candidate_digest: str) -> VerifiedAliasEntry | None:
        return next(
            (item for item in self.entries if item.candidate_digest == candidate_digest),
            None,
        )

    def canonical_symbol(self, kind: AliasKind, symbol: str) -> str:
        """Return the stable source representative of a certified alias chain."""

        if kind not in ("operator", "predicate", "action", "type"):
            raise AlignmentError("unsupported alias kind")
        _require_text(symbol, "symbol")
        edges = tuple(
            (alias.source, alias.target)
            for entry in self.entries
            for alias in entry.aliases
            if alias.kind == kind
        )
        component = {symbol}
        changed = True
        while changed:
            changed = False
            for source, target in edges:
                if source in component or target in component:
                    before = len(component)
                    component.update((source, target))
                    changed = changed or len(component) != before
        targets = {target for source, target in edges if source in component}
        roots = sorted(component - targets)
        return roots[0] if roots else min(component)

    def canonicalize_record_pattern(self, pattern: RecordPattern) -> dict[str, Any]:
        if not isinstance(pattern, RecordPattern):
            raise TypeError("pattern must be a RecordPattern")
        return {
            "arguments": [self._canonicalize_term(item) for item in pattern.arguments],
            "predicate": self.canonical_symbol("predicate", pattern.predicate),
        }

    def canonicalize_record(self, record: Record) -> Record:
        """Return a new fact using only externally admitted predicate aliases."""

        if not isinstance(record, Record):
            raise TypeError("record must be a Record")
        return Record(
            self.canonical_symbol("predicate", record.predicate),
            record.arguments,
        )

    def canonicalize_state(self, state: State) -> CanonicalStateProjection:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        return CanonicalStateProjection(
            source_state_digest=state.digest,
            records=tuple(self.canonicalize_record(item) for item in state.records),
        )

    def canonicalize_goal(self, goal: Goal) -> CanonicalGoalProjection:
        if not isinstance(goal, Goal):
            raise TypeError("goal must be a Goal")
        return CanonicalGoalProjection(
            source_goal_digest=goal.digest,
            required=tuple(self.canonicalize_record(item) for item in goal.required),
            forbidden=tuple(self.canonicalize_record(item) for item in goal.forbidden),
            exact=goal.exact,
        )

    def _canonicalize_term(self, term: SymbolicTerm) -> dict[str, str]:
        payload = term.to_canonical()
        payload["type_name"] = self.canonical_symbol("type", term.type_name)
        return payload

    def _canonical_variable(self, operator_digest: str, variable_name: str) -> str:
        node = (operator_digest, variable_name)
        edges = tuple(
            (
                (entry.source_operator_digest, source),
                (entry.target_operator_digest, target),
            )
            for entry in self.entries
            for source, target in entry.variable_map
        )
        component = {node}
        changed = True
        while changed:
            changed = False
            for source, target in edges:
                if source in component or target in component:
                    before = len(component)
                    component.update((source, target))
                    changed = changed or len(component) != before
        targets = {target for source, target in edges if source in component}
        roots = sorted(component - targets)
        return (roots[0] if roots else min(component))[1]

    def canonicalize_operator(self, operator: LearnedOperator) -> dict[str, Any]:
        """Project a mirror through certified aliases for learning features.

        Evidence, lineage, and schema descriptions are intentionally excluded;
        unmatched domain-local constraints remain named and visible.
        """

        if not isinstance(operator, LearnedOperator):
            raise TypeError("operator must be a LearnedOperator")
        payload = {
            "body": [
                {
                    "arguments": [
                        self._canonicalize_operator_term(operator, term)
                        for term in action.arguments
                    ],
                    "schema": {
                        "name": self.canonical_symbol("action", action.schema.name),
                        "parameters": [
                            {
                                "name": f"p{index}",
                                "type_name": self.canonical_symbol(
                                    "type", parameter.type_name
                                ),
                            }
                            for index, parameter in enumerate(
                                action.schema.parameters
                            )
                        ],
                    },
                }
                for action in operator.body
            ],
            "effects": [
                {
                    "kind": effect.kind,
                    "record": self._canonicalize_operator_record(
                        operator, effect.record
                    ),
                }
                for effect in operator.effects
            ],
            "name": self.canonical_symbol("operator", operator.name),
            "preconditions": [
                self._canonicalize_operator_record(operator, item)
                for item in operator.preconditions
            ],
            "variables": [
                {
                    "kind": "variable",
                    "name": self._canonical_variable(operator.digest, item.name),
                    "type_name": self.canonical_symbol("type", item.type_name),
                }
                for item in operator.variables
            ],
        }
        for field in ("effects", "preconditions", "variables"):
            payload[field] = sorted(payload[field], key=_canonical_bytes)
        return payload

    def _canonicalize_operator_term(
        self,
        operator: LearnedOperator,
        term: SymbolicTerm,
    ) -> dict[str, str]:
        payload = self._canonicalize_term(term)
        if isinstance(term, TypedVariable):
            payload["name"] = self._canonical_variable(operator.digest, term.name)
        return payload

    def _canonicalize_operator_record(
        self,
        operator: LearnedOperator,
        pattern: RecordPattern,
    ) -> dict[str, Any]:
        return {
            "arguments": [
                self._canonicalize_operator_term(operator, item)
                for item in pattern.arguments
            ],
            "predicate": self.canonical_symbol("predicate", pattern.predicate),
        }

    def to_canonical(self) -> dict[str, Any]:
        return {"entries": [item.to_canonical() for item in self.entries]}

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class MergeProposal:
    """A non-executing request to retain one of two aligned mirrors."""

    candidate: StructuralIsomorphismCandidate
    survivor_operator_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, StructuralIsomorphismCandidate):
            raise TypeError("merge proposal candidate has the wrong type")
        _require_digest(self.survivor_operator_digest, "survivor_operator_digest")
        if self.survivor_operator_digest not in (
            self.candidate.source_operator_digest,
            self.candidate.target_operator_digest,
        ):
            raise AlignmentError("merge survivor must be one aligned operator")

    @property
    def retired_operator_digest(self) -> str:
        if self.survivor_operator_digest == self.candidate.source_operator_digest:
            return self.candidate.target_operator_digest
        return self.candidate.source_operator_digest

    def to_canonical(self) -> dict[str, str]:
        return {
            "candidate_digest": self.candidate.digest,
            "retired_operator_digest": self.retired_operator_digest,
            "survivor_operator_digest": self.survivor_operator_digest,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_canonical())


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Authorization result only; actual state mutation belongs elsewhere."""

    proposal_digest: str
    candidate_digest: str
    status: Literal["authorized", "rejected"]
    alias_table_digest: str
    certificate: CounterfactualExecutionCertificate | None
    reason: str

    def __post_init__(self) -> None:
        _require_digest(self.proposal_digest, "merge-result proposal_digest")
        _require_digest(self.candidate_digest, "merge-result candidate_digest")
        _require_digest(self.alias_table_digest, "merge-result alias_table_digest")
        if self.status not in ("authorized", "rejected"):
            raise AlignmentError("merge-result status is invalid")
        _require_text(self.reason, "merge-result reason")
        if self.status == "authorized":
            if not isinstance(self.certificate, CounterfactualExecutionCertificate):
                raise AlignmentError("authorized merge requires an external certificate")
            if (
                self.certificate.result != "pass"
                or self.certificate.candidate_digest != self.candidate_digest
            ):
                raise AlignmentError(
                    "authorized merge requires passing candidate-bound evidence"
                )


def authorize_merge(
    proposal: MergeProposal,
    alias_table: AliasTable,
    certificate: CounterfactualExecutionCertificate,
) -> MergeResult:
    """Authorize a merge contract only after externally certified admission."""

    if not isinstance(proposal, MergeProposal):
        raise TypeError("proposal has the wrong type")
    if not isinstance(alias_table, AliasTable):
        raise TypeError("alias_table has the wrong type")
    if not isinstance(certificate, CounterfactualExecutionCertificate):
        raise TypeError("certificate has the wrong type")
    entry = alias_table.entry_for(proposal.candidate.digest)
    if (
        entry is None
        or entry.certificate != certificate
        or certificate.result != "pass"
        or certificate.candidate_digest != proposal.candidate.digest
    ):
        return MergeResult(
            proposal_digest=proposal.digest,
            candidate_digest=proposal.candidate.digest,
            status="rejected",
            alias_table_digest=alias_table.digest,
            certificate=certificate,
            reason="missing_passing_counterfactual_certificate",
        )
    return MergeResult(
        proposal_digest=proposal.digest,
        candidate_digest=proposal.candidate.digest,
        status="authorized",
        alias_table_digest=alias_table.digest,
        certificate=certificate,
        reason="external_counterfactual_certificate_passed",
    )


__all__ = [
    "AliasTable",
    "AlignmentCoverage",
    "AlignmentError",
    "CanonicalGoalProjection",
    "CanonicalStateProjection",
    "CANONICAL_PROJECTION_NAMESPACE",
    "CounterfactualExecutionCertificate",
    "DomainResiduals",
    "MergeProposal",
    "MergeResult",
    "StructuralIsomorphismCandidate",
    "SymbolAlias",
    "VerifiedAliasEntry",
    "authorize_merge",
    "find_structural_isomorphisms",
]
