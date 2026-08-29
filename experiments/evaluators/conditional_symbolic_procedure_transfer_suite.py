"""Evaluator-owned transfer streams for a learned conditional procedure.

One opaque conditional root binds ``public_flag=False`` and
``public_flag=True`` to two different, evaluator-private S5 position
permutations.  Its public expression is an executable tree rather than a leaf
shortcut::

    A
    P0(A), P1(A)
    C(P0(A), P1(A))

``A`` is a scalar-taught identity anchor.  The two unary components receive
ordinary raw demonstrations under counterbalanced flags, so component
identity cannot disclose the conditional binding.  Binding supports use the
full tree and show examples for the flag-selected rule.  Queries use that same
tree, expose the flag and fresh input entities, and contain no examples.

Pair identity is partitioned as an *unordered* mechanism before a separately
domain-separated hash bit fixes the False/True orientation.  Consequently a
reversed pair cannot appear in another partition.  Learner projections never
contain either normalized permutation, a target ordering, or a private task
identity; evaluation returns only scalar pairwise-order agreement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
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
from experiments.evaluators.symbolic_procedure_transfer_suite import (
    PublicDemonstrationProcedureTask,
    PublicSymbolicDemonstration,
)


_ITEM_COUNT = 5
_DEMONSTRATION_COUNT = 2
_SKILL_SYMBOL = re.compile(r"^skill_[0-9a-f]{20}$")
ConditionalMechanismPartition = Literal["train", "development", "final"]
ConditionalPermutationPair = tuple[tuple[int, ...], tuple[int, ...]]
_PARTITION_SIZES = {"train": 512, "development": 64, "final": 20}
_OPENED_PAIR_COUNT = sum(_PARTITION_SIZES.values())
_RETAINED_TRAIN_COUNT = 80
_RETAINED_DEVELOPMENT_COUNT = 19
_RETAINED_FINAL_COUNT = 20
_RETAINED_FINAL_START = _RETAINED_TRAIN_COUNT + _RETAINED_DEVELOPMENT_COUNT
_EXPANSION_START = _RETAINED_FINAL_START + _RETAINED_FINAL_COUNT
_PAIR_PARTITION_DOMAIN = b"project-angler.conditional-mechanism-pair-partition.v1\x00"
_PAIR_ORIENTATION_DOMAIN = b"project-angler.conditional-mechanism-pair-orientation.v1\x00"


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenConditionalProcedureSolution:
    """Evaluator-only binding for one public encounter."""

    public_digest: str
    public_flag: bool
    mechanism_commitment: str
    source_task: SymbolicRuleLearnerTask
    source_solution: HiddenSymbolicRuleSolution
    source_instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.public_digest, str) or not self.public_digest.startswith(
            "sha256:"
        ):
            raise ValueError("public_digest must be a canonical digest")
        if type(self.public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        if not isinstance(
            self.mechanism_commitment, str
        ) or not self.mechanism_commitment.startswith("sha256:"):
            raise ValueError("mechanism_commitment must be a canonical digest")
        if not isinstance(self.source_task, SymbolicRuleLearnerTask):
            raise TypeError("source_task must be a SymbolicRuleLearnerTask")
        if not isinstance(self.source_solution, HiddenSymbolicRuleSolution):
            raise TypeError("source_solution must remain evaluator-owned")
        if self.source_instance_id != self.source_task.instance_id or (
            self.source_instance_id != self.source_solution.instance_id
        ):
            raise ValueError("source identities are inconsistent")


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedConditionalProcedureTask:
    """Evaluator pairing; learner code should receive only ``learner``."""

    learner: PublicDemonstrationProcedureTask
    hidden: _HiddenConditionalProcedureSolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class ConditionalProcedureTransferStream:
    """Explicit staged stream for one sealed conditional mechanism."""

    anchor_supports: tuple[GeneratedConditionalProcedureTask, ...]
    component_supports: tuple[GeneratedConditionalProcedureTask, ...]
    binding_supports: tuple[GeneratedConditionalProcedureTask, ...]
    queries: tuple[GeneratedConditionalProcedureTask, ...]
    mechanism_commitment: str
    mechanism_partition: ConditionalMechanismPartition

    def __post_init__(self) -> None:
        _validate_partition(self.mechanism_partition)
        for values, label in (
            (self.anchor_supports, "anchor supports"),
            (self.component_supports, "component supports"),
            (self.binding_supports, "binding supports"),
            (self.queries, "queries"),
        ):
            if type(values) is not tuple or not values:
                raise ValueError(f"conditional stream requires non-empty {label}")
            if any(
                not isinstance(value, GeneratedConditionalProcedureTask)
                for value in values
            ):
                raise TypeError(f"{label} contain a non-conditional task")

        all_supports = self.supports
        all_pairs = (*all_supports, *self.queries)
        identities = tuple(pair.hidden.source_instance_id for pair in all_pairs)
        if len(set(identities)) != len(identities):
            raise ValueError("conditional encounter source identities must be unique")
        if any(
            pair.learner.demonstrations_visible for pair in self.anchor_supports
        ):
            raise ValueError("identity-anchor supports must not expose demonstrations")
        if any(
            not pair.learner.demonstrations_visible
            for pair in (*self.component_supports, *self.binding_supports)
        ):
            raise ValueError(
                "component and binding supports must expose demonstrations"
            )
        if any(pair.learner.demonstrations_visible for pair in self.queries):
            raise ValueError("conditional queries must not expose demonstrations")

        for values, label in (
            (self.anchor_supports, "anchor supports"),
            (self.binding_supports, "binding supports"),
            (self.queries, "queries"),
        ):
            counts = Counter(pair.learner.public_flag for pair in values)
            if counts != {False: len(values) // 2, True: len(values) // 2}:
                raise ValueError(f"{label} must be balanced across both public flags")

        anchor_requests = {pair.learner.request for pair in self.anchor_supports}
        component_requests = {
            pair.learner.request for pair in self.component_supports
        }
        binding_requests = {
            pair.learner.request for pair in (*self.binding_supports, *self.queries)
        }
        if (
            len(anchor_requests) != 1
            or len(component_requests) != 2
            or len(binding_requests) != 1
        ):
            raise ValueError("conditional stream request identities are inconsistent")
        anchor_request = next(iter(anchor_requests))
        binding_request = next(iter(binding_requests))
        if anchor_request.children:
            raise ValueError("identity anchor must be a leaf")
        if (
            len(binding_request.children) != 2
            or binding_request.depth != 2
            or set(binding_request.children) != component_requests
        ):
            raise ValueError("binding request must contain both unary components")
        first_component, second_component = binding_request.children
        for component in (first_component, second_component):
            if component.children != (anchor_request,) or component.depth != 1:
                raise ValueError("each component must contain the shared anchor")
        symbols = {
            anchor_request.symbol,
            first_component.symbol,
            second_component.symbol,
            binding_request.symbol,
        }
        if len(symbols) != 4:
            raise ValueError("anchor, components, and binding require opaque symbols")

        for component in (first_component, second_component):
            members = tuple(
                pair
                for pair in self.component_supports
                if pair.learner.request == component
            )
            counts = Counter(pair.learner.public_flag for pair in members)
            if not members or counts != {
                False: len(members) // 2,
                True: len(members) // 2,
            }:
                raise ValueError(
                    "each unary component must be flag-counterbalanced"
                )

        identity = tuple(range(_ITEM_COUNT))
        if {
            pair.hidden.source_solution.position_permutation
            for pair in self.anchor_supports
        } != {identity}:
            raise ValueError("anchor supports must use only the identity procedure")
        first_permutations = {
            pair.hidden.source_solution.position_permutation
            for pair in self.component_supports
            if pair.learner.request == first_component
        }
        second_permutations = {
            pair.hidden.source_solution.position_permutation
            for pair in self.component_supports
            if pair.learner.request == second_component
        }
        if len(first_permutations) != 1 or len(second_permutations) != 1:
            raise ValueError("each unary component must bind one procedure")
        mechanism_pair = (
            next(iter(first_permutations)),
            next(iter(second_permutations)),
        )
        _validate_conditional_pair(mechanism_pair)
        for pair in (*self.binding_supports, *self.queries):
            if pair.hidden.source_solution.position_permutation != mechanism_pair[
                int(pair.learner.public_flag)
            ]:
                raise ValueError("public flag does not select its bound procedure")
        if mechanism_pair not in conditional_mechanism_partition(
            self.mechanism_partition
        ):
            raise ValueError("conditional mechanism is outside its declared partition")
        if self.mechanism_commitment != _mechanism_commitment(mechanism_pair):
            raise ValueError("mechanism commitment does not bind the stream")
        if any(
            pair.hidden.mechanism_commitment != self.mechanism_commitment
            for pair in all_pairs
        ):
            raise ValueError("encounter commitment differs from its stream")
        _assert_global_symbol_freshness(all_pairs)

    @property
    def supports(self) -> tuple[GeneratedConditionalProcedureTask, ...]:
        """Return acquisition encounters in their required stage order."""

        return (
            self.anchor_supports
            + self.component_supports
            + self.binding_supports
        )

    @property
    def acquisition_stages(
        self,
    ) -> tuple[tuple[GeneratedConditionalProcedureTask, ...], ...]:
        """Return the three explicit acquisition stages."""

        return (
            self.anchor_supports,
            self.component_supports,
            self.binding_supports,
        )


def conditional_mechanism_partition(
    partition: ConditionalMechanismPartition,
) -> tuple[ConditionalPermutationPair, ...]:
    """Return one fixed opened partition of oriented conditional mechanisms.

    All 7,021 unordered pairs of the 119 nonidentity S5 permutations are
    hash-sorted.  The original 20 final mechanisms remain sealed at their
    existing identities while training expands from 80 to 512 and development
    from 19 to 64 using the next unopened pairs.  In total 596 mechanisms are
    opened and the remaining 6,425 are not returned by this API.  A separate
    hash domain assigns each opened pair's False/True orientation.
    """

    _validate_partition(partition)
    ordered = _ordered_conditional_mechanism_pairs()
    retained_train = ordered[:_RETAINED_TRAIN_COUNT]
    retained_development = ordered[
        _RETAINED_TRAIN_COUNT:_RETAINED_FINAL_START
    ]
    retained_final = ordered[
        _RETAINED_FINAL_START:_EXPANSION_START
    ]
    expansion = ordered[_EXPANSION_START:_OPENED_PAIR_COUNT]
    added_train_count = _PARTITION_SIZES["train"] - len(retained_train)
    added_development_count = (
        _PARTITION_SIZES["development"] - len(retained_development)
    )
    slices = {
        "train": retained_train + expansion[:added_train_count],
        "development": retained_development
        + expansion[
            added_train_count:added_train_count + added_development_count
        ],
        "final": retained_final,
    }
    selected = slices[partition]
    if len(selected) != _PARTITION_SIZES[partition]:
        raise RuntimeError("conditional mechanism partition size is inconsistent")
    return selected


def make_conditional_procedure_transfer_stream(
    seed: int,
    *,
    supports_per_flag: int = 64,
    queries_per_flag: int = 40,
    mechanism_pair: Sequence[Sequence[int]] | None = None,
    mechanism_partition: ConditionalMechanismPartition = "train",
) -> ConditionalProcedureTransferStream:
    """Create balanced demonstrated supports and demonstration-free queries."""

    _validate_seed(seed)
    _validate_positive_count(supports_per_flag, "supports_per_flag")
    _validate_positive_count(queries_per_flag, "queries_per_flag")
    _validate_partition(mechanism_partition)
    available = conditional_mechanism_partition(mechanism_partition)
    if mechanism_pair is None:
        mechanism = available[
            _domain_seed(seed, "mechanism-selection", 0, 0) % len(available)
        ]
    else:
        mechanism = _validate_conditional_pair(mechanism_pair)
        if mechanism not in available:
            raise ValueError(
                "mechanism_pair is outside the declared partition or orientation"
            )

    commitment = _mechanism_commitment(mechanism)
    anchor_request = PublicSkillExpression(_opaque_symbol(seed, commitment, 0))
    first_component_request = PublicSkillExpression(
        _opaque_symbol(seed, commitment, 1),
        (anchor_request,),
    )
    second_component_request = PublicSkillExpression(
        _opaque_symbol(seed, commitment, 2),
        (anchor_request,),
    )
    binding_request = PublicSkillExpression(
        _opaque_symbol(seed, commitment, 3),
        (first_component_request, second_component_request),
    )
    identity = tuple(range(_ITEM_COUNT))

    anchor_supports = [
        _make_encounter(
            seed,
            scope="anchor-support",
            index=index,
            public_flag=public_flag,
            permutation=identity,
            request=anchor_request,
            commitment=commitment,
            expose_demonstrations=False,
        )
        for index in range(supports_per_flag)
        for public_flag in (False, True)
    ]
    component_supports = [
        _make_encounter(
            seed,
            scope=f"component-{component_index}-support",
            index=index,
            public_flag=public_flag,
            permutation=mechanism[component_index],
            request=component_request,
            commitment=commitment,
            expose_demonstrations=True,
        )
        for component_index, component_request in enumerate(
            (first_component_request, second_component_request)
        )
        for index in range(supports_per_flag)
        for public_flag in (False, True)
    ]
    binding_supports = [
        _make_encounter(
            seed,
            scope="binding-support",
            index=index,
            public_flag=public_flag,
            permutation=mechanism[int(public_flag)],
            request=binding_request,
            commitment=commitment,
            expose_demonstrations=True,
        )
        for index in range(supports_per_flag)
        for public_flag in (False, True)
    ]
    queries = [
        _make_encounter(
            seed,
            scope="query",
            index=index,
            public_flag=public_flag,
            permutation=mechanism[int(public_flag)],
            request=binding_request,
            commitment=commitment,
            expose_demonstrations=False,
        )
        for index in range(queries_per_flag)
        for public_flag in (False, True)
    ]
    random.Random(_domain_seed(seed, "anchor-order", 0, 0)).shuffle(
        anchor_supports
    )
    random.Random(_domain_seed(seed, "component-order", 0, 0)).shuffle(
        component_supports
    )
    random.Random(_domain_seed(seed, "binding-order", 0, 0)).shuffle(
        binding_supports
    )
    random.Random(_domain_seed(seed, "query-order", 0, 0)).shuffle(queries)
    return ConditionalProcedureTransferStream(
        anchor_supports=tuple(anchor_supports),
        component_supports=tuple(component_supports),
        binding_supports=tuple(binding_supports),
        queries=tuple(queries),
        mechanism_commitment=commitment,
        mechanism_partition=mechanism_partition,
    )


def score_conditional_procedure_answer(
    task: PublicDemonstrationProcedureTask,
    solution: _HiddenConditionalProcedureSolution,
    answer: str | Sequence[str],
) -> float:
    """Return scalar pairwise agreement for one frozen public answer."""

    _validate_pairing(task, solution)
    return float(
        verify_symbolic_rule_answer(
            solution.source_task,
            solution.source_solution,
            answer,
        ).pairwise_order_agreement
    )


def _make_encounter(
    seed: int,
    *,
    scope: str,
    index: int,
    public_flag: bool,
    permutation: tuple[int, ...],
    request: PublicSkillExpression,
    commitment: str,
    expose_demonstrations: bool,
) -> GeneratedConditionalProcedureTask:
    variant = int(public_flag)
    generated = generate_symbolic_rule_task(
        _domain_seed(seed, scope, index, variant),
        item_count=_ITEM_COUNT,
        demonstration_count=_DEMONSTRATION_COUNT,
        position_permutation=permutation,
        public_symbols=_fresh_public_symbols(
            seed,
            scope,
            index,
            variant,
            (_DEMONSTRATION_COUNT + 1) * _ITEM_COUNT,
        ),
    )
    demonstrations = (
        tuple(
            PublicSymbolicDemonstration(
                demonstration.input_symbols,
                demonstration.output_symbols,
            )
            for demonstration in generated.learner.demonstrations
        )
        if expose_demonstrations
        else ()
    )
    learner = PublicDemonstrationProcedureTask(
        items=tuple(
            PublicOrderingItem(
                symbol=symbol,
                rank_a=position,
                rank_b=position,
                group=position % 2,
                marked=position == 0,
            )
            for position, symbol in enumerate(generated.learner.query_symbols)
        ),
        public_flag=public_flag,
        request=request,
        demonstrations=demonstrations,
    )
    hidden = _HiddenConditionalProcedureSolution(
        public_digest=_public_digest(learner),
        public_flag=public_flag,
        mechanism_commitment=commitment,
        source_task=generated.learner,
        source_solution=generated.hidden,
        source_instance_id=generated.learner.instance_id,
    )
    return GeneratedConditionalProcedureTask(learner, hidden)


@lru_cache(maxsize=1)
def _ordered_conditional_mechanism_pairs() -> tuple[ConditionalPermutationPair, ...]:
    identity = tuple(range(_ITEM_COUNT))
    permutations = tuple(
        permutation
        for permutation in itertools.permutations(range(_ITEM_COUNT))
        if permutation != identity
    )
    unordered = tuple(itertools.combinations(permutations, 2))
    expected = len(permutations) * (len(permutations) - 1) // 2
    if len(permutations) != 119 or len(unordered) != expected or expected != 7_021:
        raise RuntimeError("conditional S5 mechanism universe is inconsistent")
    ordered = sorted(unordered, key=lambda pair: _pair_hash(_PAIR_PARTITION_DOMAIN, pair))
    return tuple(_orient_pair(pair) for pair in ordered)


def _orient_pair(pair: ConditionalPermutationPair) -> ConditionalPermutationPair:
    return (
        (pair[1], pair[0])
        if _pair_hash(_PAIR_ORIENTATION_DOMAIN, pair)[0] & 1
        else pair
    )


def _pair_hash(domain: bytes, pair: ConditionalPermutationPair) -> bytes:
    return hashlib.sha256(domain + bytes(pair[0]) + b"\x00" + bytes(pair[1])).digest()


def _mechanism_commitment(pair: ConditionalPermutationPair) -> str:
    return "sha256:" + hashlib.sha256(
        b"project-angler.conditional-mechanism.v1\x00"
        + b"false\x00"
        + bytes(pair[0])
        + b"\x00true\x00"
        + bytes(pair[1])
    ).hexdigest()


def _fresh_public_symbols(
    seed: int,
    scope: str,
    index: int,
    variant: int,
    count: int,
) -> tuple[str, ...]:
    return tuple(
        "conditional_entity_"
        + hashlib.sha256(
            (
                "project-angler.conditional-public-entity.v1\x00"
                f"{seed}\x00{scope}\x00{index}\x00{variant}\x00{position}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        for position in range(count)
    )


def _assert_global_symbol_freshness(
    pairs: Sequence[GeneratedConditionalProcedureTask],
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
            raise ValueError("one conditional encounter reused a public entity")
        if seen & namespace:
            raise ValueError("public entity namespaces overlap across encounters")
        seen.update(namespace)


def _validate_conditional_pair(
    supplied: Sequence[Sequence[int]],
) -> ConditionalPermutationPair:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("mechanism_pair must contain two permutations")
    values = tuple(supplied)
    if len(values) != 2:
        raise ValueError("mechanism_pair must contain False and True permutations")
    pair = (
        _validate_nonidentity_permutation(values[0]),
        _validate_nonidentity_permutation(values[1]),
    )
    if pair[0] == pair[1]:
        raise ValueError("conditional procedures must be distinct")
    return pair


def _validate_nonidentity_permutation(supplied: Sequence[int]) -> tuple[int, ...]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("each conditional procedure must be an integer sequence")
    permutation = tuple(supplied)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in permutation):
        raise TypeError("conditional procedures must contain only integers")
    if sorted(permutation) != list(range(_ITEM_COUNT)):
        raise ValueError("conditional procedure must cover every item position once")
    if permutation == tuple(range(_ITEM_COUNT)):
        raise ValueError("conditional procedure must not be identity")
    return permutation


def _validate_pairing(
    task: PublicDemonstrationProcedureTask,
    solution: _HiddenConditionalProcedureSolution,
) -> None:
    if not isinstance(task, PublicDemonstrationProcedureTask):
        raise TypeError("task must be a PublicDemonstrationProcedureTask")
    if not isinstance(solution, _HiddenConditionalProcedureSolution):
        raise TypeError("solution must remain evaluator-owned")
    if _public_digest(task) != solution.public_digest:
        raise ValueError("public task and evaluator binding do not match")
    if task.public_flag is not solution.public_flag:
        raise ValueError("public flag and evaluator binding do not match")
    if tuple(item.symbol for item in task.items) != solution.source_task.query_symbols:
        raise ValueError("public query symbols differ from the bound source task")
    if task.demonstrations:
        expected = tuple(
            PublicSymbolicDemonstration(
                demonstration.input_symbols,
                demonstration.output_symbols,
            )
            for demonstration in solution.source_task.demonstrations
        )
        if task.demonstrations != expected:
            raise ValueError("public demonstrations differ from the bound source task")


def _public_digest(task: PublicDemonstrationProcedureTask) -> str:
    encoded = json.dumps(
        task.to_canonical(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"project-angler.conditional-procedure-public.v1\x00" + encoded
    ).hexdigest()


def _opaque_symbol(seed: int, commitment: str, role: int) -> str:
    material = (
        "project-angler.conditional-skill-symbol.v1\x00"
        f"{seed}\x00{commitment}\x00{role}"
    ).encode("utf-8")
    value = "skill_" + hashlib.sha256(material).hexdigest()[:20]
    if _SKILL_SYMBOL.fullmatch(value) is None:
        raise RuntimeError("opaque skill generation failed")
    return value


def _domain_seed(seed: int, scope: str, index: int, variant: int) -> int:
    material = (
        "project-angler.conditional-procedure-seed.v1\x00"
        f"{seed}\x00{scope}\x00{index}\x00{variant}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validate_partition(partition: str) -> None:
    if partition not in _PARTITION_SIZES:
        raise ValueError("partition must be train, development, or final")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _validate_positive_count(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


__all__ = [
    "ConditionalMechanismPartition",
    "ConditionalPermutationPair",
    "ConditionalProcedureTransferStream",
    "GeneratedConditionalProcedureTask",
    "conditional_mechanism_partition",
    "make_conditional_procedure_transfer_stream",
    "score_conditional_procedure_answer",
]
