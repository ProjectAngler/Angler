"""Evaluator-owned streams for persistent procedure induction from examples.

One sealed position permutation governs many fresh-symbol tasks. Online
supports expose ordinary public input/output demonstrations and return only a
scalar score for one attempted query ordering. Held-out queries omit every
demonstration, so success requires the acquired procedure to survive in
Angler's bounded competence state.

The learner receives raw public symbol pairs. Entity matching and tensorization
belong to a learned typed sensory port; this evaluator never serializes the
examples into a normalized permutation or target ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
import re
from typing import Literal, Sequence

from angler.worlds.latent_order_programs import PublicOrderingItem
from angler.worlds.symbolic_rule_induction import (
    GeneratedSymbolicRuleTask,
    HiddenSymbolicRuleSolution,
    SymbolicRuleLearnerTask,
    generate_symbolic_rule_task,
    verify_symbolic_rule_answer,
)
from experiments.evaluators.skill_memory_suite import PublicSkillExpression


_ITEM_COUNT = 5
_DEMONSTRATION_COUNT = 2
_SKILL_SYMBOL = re.compile(r"^skill_[0-9a-f]{20}$")
PermutationPartition = Literal["train", "development", "final"]
_PARTITION_SIZES = {"train": 80, "development": 19, "final": 20}


@dataclass(frozen=True, slots=True)
class PublicSymbolicDemonstration:
    """One raw, learner-visible example; no normalized rule is stored."""

    input_symbols: tuple[str, ...]
    output_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, symbols in (
            ("input_symbols", self.input_symbols),
            ("output_symbols", self.output_symbols),
        ):
            if type(symbols) is not tuple or len(symbols) != _ITEM_COUNT:
                raise ValueError(f"{name} must contain exactly {_ITEM_COUNT} symbols")
            if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
                raise ValueError(f"{name} must contain non-empty text symbols")
            if len(set(symbols)) != _ITEM_COUNT:
                raise ValueError(f"{name} symbols must be unique")
        if set(self.input_symbols) != set(self.output_symbols):
            raise ValueError("demonstration input/output entities must match")

    def to_canonical(self) -> dict[str, object]:
        return {
            "input_symbols": list(self.input_symbols),
            "output_symbols": list(self.output_symbols),
        }


@dataclass(frozen=True, slots=True)
class PublicDemonstrationProcedureTask:
    """Learner view with optional raw examples and no private identity."""

    items: tuple[PublicOrderingItem, ...]
    public_flag: bool
    request: PublicSkillExpression
    demonstrations: tuple[PublicSymbolicDemonstration, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or len(self.items) != _ITEM_COUNT:
            raise ValueError(f"public task must contain exactly {_ITEM_COUNT} items")
        if any(not isinstance(item, PublicOrderingItem) for item in self.items):
            raise TypeError("items must be PublicOrderingItem values")
        if len({item.symbol for item in self.items}) != _ITEM_COUNT:
            raise ValueError("public query symbols must be unique")
        if type(self.public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        if not isinstance(self.request, PublicSkillExpression):
            raise TypeError("request must be a PublicSkillExpression")
        if type(self.demonstrations) is not tuple or len(self.demonstrations) not in (
            0,
            _DEMONSTRATION_COUNT,
        ):
            raise ValueError("task must contain zero or two raw demonstrations")
        if any(
            not isinstance(demonstration, PublicSymbolicDemonstration)
            for demonstration in self.demonstrations
        ):
            raise TypeError("demonstrations must use the raw public schema")

    @property
    def demonstrations_visible(self) -> bool:
        return bool(self.demonstrations)

    def to_canonical(self) -> dict[str, object]:
        return {
            "demonstrations": [
                demonstration.to_canonical()
                for demonstration in self.demonstrations
            ],
            "items": [
                {
                    "group": item.group,
                    "marked": item.marked,
                    "rank_a": item.rank_a,
                    "rank_b": item.rank_b,
                    "symbol": item.symbol,
                }
                for item in self.items
            ],
            "public_flag": self.public_flag,
            "request": self.request.to_canonical(),
        }


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenDemonstrationProcedureSolution:
    public_digest: str
    source_task: SymbolicRuleLearnerTask
    source_solution: HiddenSymbolicRuleSolution
    source_instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.public_digest, str) or not self.public_digest.startswith(
            "sha256:"
        ):
            raise ValueError("public_digest must be a canonical digest")
        if not isinstance(self.source_task, SymbolicRuleLearnerTask):
            raise TypeError("source_task must be a SymbolicRuleLearnerTask")
        if not isinstance(self.source_solution, HiddenSymbolicRuleSolution):
            raise TypeError("source_solution must remain evaluator-owned")
        if self.source_instance_id != self.source_task.instance_id or (
            self.source_instance_id != self.source_solution.instance_id
        ):
            raise ValueError("source identities are inconsistent")


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedDemonstrationProcedureTask:
    learner: PublicDemonstrationProcedureTask
    hidden: _HiddenDemonstrationProcedureSolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class DemonstrationProcedureTransferStream:
    supports: tuple[GeneratedDemonstrationProcedureTask, ...]
    queries: tuple[GeneratedDemonstrationProcedureTask, ...]
    mechanism_commitment: str
    mechanism_partition: str

    def __post_init__(self) -> None:
        if not self.supports or not self.queries:
            raise ValueError("demonstration transfer stream requires supports and queries")
        if self.mechanism_partition not in (*_PARTITION_SIZES, "unpartitioned"):
            raise ValueError("mechanism partition is invalid")
        support_ids = {pair.hidden.source_instance_id for pair in self.supports}
        query_ids = {pair.hidden.source_instance_id for pair in self.queries}
        if len(support_ids) != len(self.supports):
            raise ValueError("support source identities must be unique")
        if len(query_ids) != len(self.queries):
            raise ValueError("query source identities must be unique")
        if support_ids & query_ids:
            raise ValueError("support and query source identities overlap")
        if any(pair.learner.demonstrations_visible for pair in self.queries):
            raise ValueError("held-out queries must not expose demonstrations")

        identity = tuple(range(_ITEM_COUNT))
        transform_pairs = tuple(
            pair for pair in self.supports if pair.learner.request.children
        ) + self.queries
        transforms = {
            pair.hidden.source_solution.position_permutation
            for pair in transform_pairs
        }
        if len(transforms) != 1 or identity in transforms:
            raise ValueError("stream must hold one non-identity transform")
        transform = next(iter(transforms))
        identity_pairs = tuple(
            pair for pair in self.supports if not pair.learner.request.children
        )
        if any(
            pair.hidden.source_solution.position_permutation != identity
            for pair in identity_pairs
        ):
            raise ValueError("leaf supports must use the identity procedure")
        if self.mechanism_commitment != _permutation_commitment(transform):
            raise ValueError("mechanism commitment does not bind the stream")
        _assert_global_symbol_freshness((*self.supports, *self.queries))


def demonstration_permutation_partition(
    partition: PermutationPartition,
) -> tuple[tuple[int, ...], ...]:
    """Return a fixed evaluator-only train/development/final partition of S5."""

    if partition not in _PARTITION_SIZES:
        raise ValueError("partition must be train, development, or final")
    identity = tuple(range(_ITEM_COUNT))
    candidates = [
        permutation
        for permutation in itertools.permutations(range(_ITEM_COUNT))
        if permutation != identity
    ]
    ordered = tuple(
        sorted(
            candidates,
            key=lambda permutation: hashlib.sha256(
                b"project-angler.demonstration-mechanism-partition.v1\x00"
                + bytes(permutation)
            ).digest(),
        )
    )
    train_end = _PARTITION_SIZES["train"]
    development_end = train_end + _PARTITION_SIZES["development"]
    slices = {
        "train": ordered[:train_end],
        "development": ordered[train_end:development_end],
        "final": ordered[development_end:],
    }
    selected = slices[partition]
    if len(selected) != _PARTITION_SIZES[partition]:
        raise RuntimeError("mechanism partition size is inconsistent")
    return selected


def make_demonstration_procedure_transfer_stream(
    seed: int,
    *,
    supports_per_procedure: int = 64,
    queries_per_procedure: int = 40,
    position_permutation: Sequence[int] | None = None,
    mechanism_partition: str = "unpartitioned",
    expose_transform_demonstrations: bool = True,
    demonstration_permutation: Sequence[int] | None = None,
    rotate_demonstration_outputs: int = 0,
) -> DemonstrationProcedureTransferStream:
    """Create identity-child and demonstrated-transform acquisition streams."""

    _validate_seed(seed)
    for name, value in (
        ("supports_per_procedure", supports_per_procedure),
        ("queries_per_procedure", queries_per_procedure),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if mechanism_partition not in (*_PARTITION_SIZES, "unpartitioned"):
        raise ValueError("mechanism_partition is invalid")
    if type(expose_transform_demonstrations) is not bool:
        raise TypeError("expose_transform_demonstrations must be bool")
    if (
        isinstance(rotate_demonstration_outputs, bool)
        or not isinstance(rotate_demonstration_outputs, int)
        or not 0 <= rotate_demonstration_outputs < _ITEM_COUNT
    ):
        raise ValueError("rotate_demonstration_outputs must be between zero and four")
    if demonstration_permutation is not None and rotate_demonstration_outputs:
        raise ValueError(
            "demonstration permutation and public output rotation are exclusive"
        )

    if position_permutation is None:
        sampled = generate_symbolic_rule_task(
            _domain_seed(seed, "sealed-procedure", 0, 0),
            item_count=_ITEM_COUNT,
            demonstration_count=_DEMONSTRATION_COUNT,
        )
        transform = sampled.hidden.position_permutation
    else:
        transform = _validate_non_identity_permutation(position_permutation)
    if mechanism_partition in _PARTITION_SIZES and transform not in (
        demonstration_permutation_partition(mechanism_partition)  # type: ignore[arg-type]
    ):
        raise ValueError("position_permutation is outside the declared partition")
    shown_transform = (
        transform
        if demonstration_permutation is None
        else _validate_non_identity_permutation(demonstration_permutation)
    )

    identity = tuple(range(_ITEM_COUNT))
    identity_request = PublicSkillExpression(_opaque_symbol(seed, 0))
    transform_request = PublicSkillExpression(
        _opaque_symbol(seed, 1),
        (identity_request,),
    )

    supports: list[GeneratedDemonstrationProcedureTask] = []
    for index in range(supports_per_procedure):
        for permutation, request, expose_demonstrations, variant in (
            (identity, identity_request, False, 0),
            (
                transform,
                transform_request,
                expose_transform_demonstrations,
                1,
            ),
        ):
            task_seed = _domain_seed(seed, "support", index, variant)
            generated = generate_symbolic_rule_task(
                task_seed,
                item_count=_ITEM_COUNT,
                demonstration_count=_DEMONSTRATION_COUNT,
                position_permutation=permutation,
                public_symbols=_fresh_public_symbols(
                    seed,
                    "support",
                    index,
                    variant,
                    (_DEMONSTRATION_COUNT + 1) * _ITEM_COUNT,
                ),
            )
            supports.append(
                _pair_task(
                    generated,
                    request=request,
                    expose_demonstrations=expose_demonstrations,
                    demonstration_permutation=(
                        shown_transform if variant == 1 else None
                    ),
                    rotate_demonstration_outputs=(
                        rotate_demonstration_outputs if variant == 1 else 0
                    ),
                    public_flag=bool(
                        _domain_seed(seed, "support-flag", index, variant) & 1
                    ),
                )
            )

    queries: list[GeneratedDemonstrationProcedureTask] = []
    for index in range(queries_per_procedure):
        task_seed = _domain_seed(seed, "query", index, 1)
        generated = generate_symbolic_rule_task(
            task_seed,
            item_count=_ITEM_COUNT,
            demonstration_count=_DEMONSTRATION_COUNT,
            position_permutation=transform,
            public_symbols=_fresh_public_symbols(
                seed,
                "query",
                index,
                1,
                (_DEMONSTRATION_COUNT + 1) * _ITEM_COUNT,
            ),
        )
        queries.append(
            _pair_task(
                generated,
                request=transform_request,
                expose_demonstrations=False,
                public_flag=bool(_domain_seed(seed, "query-flag", index, 0) & 1),
            )
        )
    random.Random(_domain_seed(seed, "query-order", 0, 0)).shuffle(queries)
    return DemonstrationProcedureTransferStream(
        tuple(supports),
        tuple(queries),
        _permutation_commitment(transform),
        mechanism_partition,
    )


def score_demonstration_procedure_answer(
    task: PublicDemonstrationProcedureTask,
    solution: _HiddenDemonstrationProcedureSolution,
    answer: str | Sequence[str],
) -> float:
    """Return only scalar pairwise agreement for one frozen public answer."""

    _validate_pairing(task, solution)
    return float(
        verify_symbolic_rule_answer(
            solution.source_task,
            solution.source_solution,
            answer,
        ).pairwise_order_agreement
    )


def _pair_task(
    generated: GeneratedSymbolicRuleTask,
    *,
    request: PublicSkillExpression,
    expose_demonstrations: bool,
    demonstration_permutation: tuple[int, ...] | None = None,
    rotate_demonstration_outputs: int = 0,
    public_flag: bool,
) -> GeneratedDemonstrationProcedureTask:
    demonstrations = (
        tuple(
            PublicSymbolicDemonstration(
                demonstration.input_symbols,
                output[rotate_demonstration_outputs:]
                + output[:rotate_demonstration_outputs],
            )
            for demonstration in generated.learner.demonstrations
            for output in (
                (
                    demonstration.output_symbols
                    if demonstration_permutation is None
                    else tuple(
                        demonstration.input_symbols[position]
                        for position in demonstration_permutation
                    )
                ),
            )
        )
        if expose_demonstrations
        else ()
    )
    items = tuple(
        PublicOrderingItem(
            symbol=symbol,
            rank_a=index,
            rank_b=index,
            group=index % 2,
            marked=index == 0,
        )
        for index, symbol in enumerate(generated.learner.query_symbols)
    )
    learner = PublicDemonstrationProcedureTask(
        items=items,
        public_flag=public_flag,
        request=request,
        demonstrations=demonstrations,
    )
    hidden = _HiddenDemonstrationProcedureSolution(
        public_digest=_public_digest(learner),
        source_task=generated.learner,
        source_solution=generated.hidden,
        source_instance_id=generated.learner.instance_id,
    )
    return GeneratedDemonstrationProcedureTask(learner, hidden)


def _fresh_public_symbols(
    seed: int,
    scope: str,
    index: int,
    variant: int,
    count: int,
) -> tuple[str, ...]:
    return tuple(
        "entity_"
        + hashlib.sha256(
            (
                "project-angler.demonstration-public-entity.v1\x00"
                f"{seed}\x00{scope}\x00{index}\x00{variant}\x00{position}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        for position in range(count)
    )


def _assert_global_symbol_freshness(
    pairs: Sequence[GeneratedDemonstrationProcedureTask],
) -> None:
    seen: set[str] = set()
    for pair in pairs:
        source = pair.hidden.source_task
        namespace = {
            symbol
            for demonstration in source.demonstrations
            for symbol in demonstration.input_symbols
        } | set(source.query_symbols)
        expected = (_DEMONSTRATION_COUNT + 1) * _ITEM_COUNT
        if len(namespace) != expected:
            raise ValueError("one source task reused a public entity")
        if seen & namespace:
            raise ValueError("public entity namespaces overlap across encounters")
        seen.update(namespace)


def _validate_non_identity_permutation(
    supplied: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("position_permutation must be an integer sequence")
    permutation = tuple(supplied)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in permutation):
        raise TypeError("position_permutation must contain only integers")
    if sorted(permutation) != list(range(_ITEM_COUNT)):
        raise ValueError("position_permutation must cover every item position once")
    if permutation == tuple(range(_ITEM_COUNT)):
        raise ValueError("transfer transform must not be identity")
    return permutation


def _permutation_commitment(permutation: tuple[int, ...]) -> str:
    return "sha256:" + hashlib.sha256(
        b"project-angler.demonstration-mechanism.v1\x00" + bytes(permutation)
    ).hexdigest()


def _validate_pairing(
    task: PublicDemonstrationProcedureTask,
    solution: _HiddenDemonstrationProcedureSolution,
) -> None:
    if not isinstance(task, PublicDemonstrationProcedureTask):
        raise TypeError("task must be a PublicDemonstrationProcedureTask")
    if not isinstance(solution, _HiddenDemonstrationProcedureSolution):
        raise TypeError("solution must remain evaluator-owned")
    if _public_digest(task) != solution.public_digest:
        raise ValueError("public task and evaluator binding do not match")
    if tuple(item.symbol for item in task.items) != solution.source_task.query_symbols:
        raise ValueError("public query symbols differ from the bound source task")


def _public_digest(task: PublicDemonstrationProcedureTask) -> str:
    encoded = json.dumps(
        task.to_canonical(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"project-angler.demonstration-procedure-public.v2\x00" + encoded
    ).hexdigest()


def _opaque_symbol(seed: int, index: int) -> str:
    material = f"project-angler.demonstration-symbol.v1\x00{seed}\x00{index}".encode(
        "utf-8"
    )
    value = "skill_" + hashlib.sha256(material).hexdigest()[:20]
    if _SKILL_SYMBOL.fullmatch(value) is None:
        raise RuntimeError("opaque symbol generation failed")
    return value


def _domain_seed(seed: int, scope: str, index: int, variant: int) -> int:
    material = (
        f"project-angler.demonstration-procedure-seed.v2\x00{seed}\x00{scope}"
        f"\x00{index}\x00{variant}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


__all__ = [
    "DemonstrationProcedureTransferStream",
    "GeneratedDemonstrationProcedureTask",
    "PublicDemonstrationProcedureTask",
    "PublicSymbolicDemonstration",
    "demonstration_permutation_partition",
    "make_demonstration_procedure_transfer_stream",
    "score_demonstration_procedure_answer",
]
