"""Changing latent ordering programs with scalar outcome-only feedback.

The program and target live only in generator/evaluator objects.  Learner code
receives public item attributes and must emit one ordering.  This module may
generate and judge tasks; it never provides a solving procedure to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Sequence


ITEM_COUNT = 5
PAIR_COUNT = ITEM_COUNT * (ITEM_COUNT - 1) // 2

_SYMBOLS = (
    "amber", "birch", "coral", "delta", "ember", "fjord", "gale",
    "harbor", "iris", "jade", "kestrel", "lumen", "maple", "nova",
    "onyx", "pearl", "quartz", "raven", "solar", "tulip", "umber",
    "violet", "willow", "xenon", "yarrow", "zephyr", "acorn", "brook",
    "cinder", "dune", "elm", "flint", "grove", "heath", "indigo",
    "juniper", "kelp", "lotus", "moss", "nectar", "opal", "pine",
    "reed", "sage", "thistle", "upland", "vale", "wren", "yucca",
    "zinnia",
)

_LEAF_OPERATORS = frozenset(("A_ASC", "A_DESC", "B_ASC", "B_DESC"))
_UNARY_OPERATORS = frozenset(("GROUP_01", "GROUP_10", "ZIGZAG", "ROTATE"))
_BINARY_OPERATORS = frozenset(("IF_FLAG", "IF_NOT_FLAG"))


@dataclass(frozen=True, slots=True, repr=False)
class OrderingProgram:
    """Evaluator-only operator tree."""

    operator: str
    children: tuple["OrderingProgram", ...] = ()

    def __post_init__(self) -> None:
        if self.operator in _LEAF_OPERATORS:
            expected = 0
        elif self.operator in _UNARY_OPERATORS:
            expected = 1
        elif self.operator in _BINARY_OPERATORS:
            expected = 2
        else:
            raise ValueError(f"unknown ordering operator: {self.operator}")
        if len(self.children) != expected:
            raise ValueError(
                f"{self.operator} requires {expected} child program(s)"
            )

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(
            child.depth for child in self.children
        )

    @property
    def canonical(self) -> str:
        if not self.children:
            return self.operator
        nested = ",".join(child.canonical for child in self.children)
        return f"{self.operator}({nested})"

    @property
    def structural_skeleton(self) -> str:
        """Canonical topology with leaf attribute/direction erased."""

        if not self.children:
            return "LEAF"
        nested = ",".join(
            child.structural_skeleton for child in self.children
        )
        return f"{self.operator}({nested})"


@dataclass(frozen=True, slots=True)
class PublicOrderingItem:
    """One learner-visible item with two ranks and two public attributes."""

    symbol: str
    rank_a: int
    rank_b: int
    group: int
    marked: bool


@dataclass(frozen=True, slots=True)
class LatentOrderingTask:
    """Complete learner projection; it contains no program or target."""

    instance_id: str
    items: tuple[PublicOrderingItem, ...]
    public_flag: bool

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.items)


@dataclass(frozen=True, slots=True, repr=False)
class HiddenLatentOrderSolution:
    """Sealed program identity and target for the outcome judge."""

    instance_id: str
    program: OrderingProgram
    target_order: tuple[str, ...]
    generator_seed: int


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedLatentOrderingTask:
    learner: LatentOrderingTask
    hidden: HiddenLatentOrderSolution


@dataclass(frozen=True, slots=True)
class LatentOrderFeedback:
    """Scalar result; no target, violated pair, or program is disclosed."""

    valid: bool
    exact: bool
    pairwise_accuracy: float


def _leaf(name: str) -> OrderingProgram:
    return OrderingProgram(name)


def _unary(name: str, child: OrderingProgram) -> OrderingProgram:
    return OrderingProgram(name, (child,))


def _conditional(
    when_false: OrderingProgram,
    when_true: OrderingProgram,
) -> OrderingProgram:
    return OrderingProgram("IF_FLAG", (when_false, when_true))


def _inverse_conditional(
    when_false: OrderingProgram,
    when_true: OrderingProgram,
) -> OrderingProgram:
    return OrderingProgram("IF_NOT_FLAG", (when_false, when_true))


A_ASC = _leaf("A_ASC")
A_DESC = _leaf("A_DESC")
B_ASC = _leaf("B_ASC")
B_DESC = _leaf("B_DESC")

TRAIN_PROGRAMS = (
    A_ASC,
    A_DESC,
    B_ASC,
    B_DESC,
    _unary("GROUP_01", A_ASC),
    _unary("GROUP_10", B_DESC),
    _unary("ZIGZAG", A_ASC),
    _unary("ZIGZAG", B_DESC),
    _unary("ROTATE", A_ASC),
    _unary("ROTATE", B_DESC),
    _conditional(A_ASC, B_ASC),
    _conditional(A_DESC, B_DESC),
    _inverse_conditional(A_ASC, B_DESC),
    _inverse_conditional(A_DESC, B_ASC),
)

VALIDATION_PROGRAMS = (
    _unary("GROUP_01", _unary("ZIGZAG", B_ASC)),
    _unary("ROTATE", _unary("GROUP_10", A_ASC)),
    _conditional(_unary("ZIGZAG", A_DESC), B_ASC),
    _inverse_conditional(_unary("GROUP_01", B_DESC), A_ASC),
    _unary("ZIGZAG", _unary("ROTATE", B_DESC)),
)

def generate_latent_ordering_task(
    program: OrderingProgram,
    seed: int,
    *,
    public_flag: bool | None = None,
) -> GeneratedLatentOrderingTask:
    """Generate one unique public instance for a sealed ordering program."""

    if not isinstance(program, OrderingProgram):
        raise TypeError("program must be an OrderingProgram")
    # Public input sampling is deliberately independent of the sealed program,
    # allowing paired counterfactual tasks and preventing marginal shortcuts.
    rng = _domain_rng(seed, "public")
    symbols = rng.sample(_SYMBOLS, ITEM_COUNT)
    ranks_a = list(range(ITEM_COUNT))
    ranks_b = list(range(ITEM_COUNT))
    rng.shuffle(ranks_a)
    rng.shuffle(ranks_b)
    if ranks_b == ranks_a:
        ranks_b = ranks_b[1:] + ranks_b[:1]
    groups = [0, 0, 1, 1, rng.randrange(2)]
    rng.shuffle(groups)
    marked_index = rng.randrange(ITEM_COUNT)
    sampled_flag = bool(rng.randrange(2))
    if public_flag is None:
        public_flag = sampled_flag
    elif not isinstance(public_flag, bool):
        raise TypeError("public_flag must be a bool or None")

    items = [
        PublicOrderingItem(
            symbol=symbols[index],
            rank_a=ranks_a[index],
            rank_b=ranks_b[index],
            group=groups[index],
            marked=index == marked_index,
        )
        for index in range(ITEM_COUNT)
    ]
    rng.shuffle(items)
    public_items = tuple(items)
    target = tuple(
        item.symbol
        for item in _execute_program(program, public_items, public_flag)
    )
    instance_id = _instance_id(
        public_items,
        public_flag,
    )
    return GeneratedLatentOrderingTask(
        learner=LatentOrderingTask(
            instance_id=instance_id,
            items=public_items,
            public_flag=public_flag,
        ),
        hidden=HiddenLatentOrderSolution(
            instance_id=instance_id,
            program=program,
            target_order=target,
            generator_seed=seed,
        ),
    )


def make_renamed_latent_variant(
    source: GeneratedLatentOrderingTask,
    *,
    seed: int,
) -> GeneratedLatentOrderingTask:
    """Change only opaque symbols and display order, preserving semantics."""

    _validate_pairing(source.learner, source.hidden)
    rng = _domain_rng(seed, "rename")
    old_symbols = source.learner.symbols
    available = tuple(symbol for symbol in _SYMBOLS if symbol not in old_symbols)
    replacements = rng.sample(available, ITEM_COUNT)
    rename = dict(zip(old_symbols, replacements, strict=True))
    items = [
        PublicOrderingItem(
            symbol=rename[item.symbol],
            rank_a=item.rank_a,
            rank_b=item.rank_b,
            group=item.group,
            marked=item.marked,
        )
        for item in source.learner.items
    ]
    rng.shuffle(items)
    public_items = tuple(items)
    program = source.hidden.program
    target = tuple(
        item.symbol
        for item in _execute_program(
            program,
            public_items,
            source.learner.public_flag,
        )
    )
    instance_id = _instance_id(
        public_items,
        source.learner.public_flag,
    )
    return GeneratedLatentOrderingTask(
        learner=LatentOrderingTask(
            instance_id=instance_id,
            items=public_items,
            public_flag=source.learner.public_flag,
        ),
        hidden=HiddenLatentOrderSolution(
            instance_id=instance_id,
            program=program,
            target_order=target,
            generator_seed=seed,
        ),
    )


def score_latent_ordering_answer(
    task: LatentOrderingTask,
    solution: HiddenLatentOrderSolution,
    answer: str | Sequence[str],
) -> LatentOrderFeedback:
    """Judge only a final permutation and return one bounded scalar score."""

    _validate_pairing(task, solution)
    submitted = _parse_answer(answer)
    if (
        submitted is None
        or len(submitted) != ITEM_COUNT
        or len(set(submitted)) != ITEM_COUNT
        or set(submitted) != set(task.symbols)
    ):
        return LatentOrderFeedback(False, False, 0.0)

    submitted_position = {
        symbol: index for index, symbol in enumerate(submitted)
    }
    target = solution.target_order
    agreements = sum(
        submitted_position[target[left]] < submitted_position[target[right]]
        for left in range(ITEM_COUNT)
        for right in range(left + 1, ITEM_COUNT)
    )
    return LatentOrderFeedback(
        valid=True,
        exact=submitted == target,
        pairwise_accuracy=agreements / PAIR_COUNT,
    )


def _execute_program(
    program: OrderingProgram,
    items: tuple[PublicOrderingItem, ...],
    public_flag: bool,
) -> tuple[PublicOrderingItem, ...]:
    operator = program.operator
    if operator == "A_ASC":
        return tuple(sorted(items, key=lambda item: item.rank_a))
    if operator == "A_DESC":
        return tuple(sorted(items, key=lambda item: item.rank_a, reverse=True))
    if operator == "B_ASC":
        return tuple(sorted(items, key=lambda item: item.rank_b))
    if operator == "B_DESC":
        return tuple(sorted(items, key=lambda item: item.rank_b, reverse=True))
    if operator == "IF_FLAG":
        selected = program.children[int(public_flag)]
        return _execute_program(selected, items, public_flag)
    if operator == "IF_NOT_FLAG":
        selected = program.children[1 - int(public_flag)]
        return _execute_program(selected, items, public_flag)

    base = _execute_program(program.children[0], items, public_flag)
    if operator == "GROUP_01":
        return tuple(item for group in (0, 1) for item in base if item.group == group)
    if operator == "GROUP_10":
        return tuple(item for group in (1, 0) for item in base if item.group == group)
    if operator == "ZIGZAG":
        result: list[PublicOrderingItem] = []
        left, right = 0, len(base) - 1
        while left <= right:
            result.append(base[left])
            left += 1
            if left <= right:
                result.append(base[right])
                right -= 1
        return tuple(result)
    if operator == "ROTATE":
        marked = next(index for index, item in enumerate(base) if item.marked)
        return base[marked:] + base[:marked]
    raise AssertionError(f"unhandled validated operator: {operator}")


def _validate_pairing(
    task: LatentOrderingTask,
    solution: HiddenLatentOrderSolution,
) -> None:
    if task.instance_id != solution.instance_id:
        raise ValueError("task and hidden solution identities do not match")
    if (
        len(solution.target_order) != ITEM_COUNT
        or set(solution.target_order) != set(task.symbols)
    ):
        raise ValueError("hidden target is not a permutation of public symbols")
    expected = tuple(
        item.symbol
        for item in _execute_program(
            solution.program,
            task.items,
            task.public_flag,
        )
    )
    if expected != solution.target_order:
        raise ValueError("hidden target does not match the sealed program")


def _parse_answer(answer: str | Sequence[str]) -> tuple[str, ...] | None:
    if isinstance(answer, str):
        submitted = tuple(part.strip() for part in answer.split(","))
    elif isinstance(answer, Sequence) and all(
        isinstance(part, str) for part in answer
    ):
        submitted = tuple(part.strip() for part in answer)
    else:
        return None
    return None if any(not part for part in submitted) else submitted


def _domain_rng(seed: int, domain: str) -> random.Random:
    material = f"angler.latent-order.v1\x00{seed}\x00{domain}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _instance_id(
    items: tuple[PublicOrderingItem, ...],
    public_flag: bool,
) -> str:
    public_material = {
        "items": [
            {
                "symbol": item.symbol,
                "a": item.rank_a,
                "b": item.rank_b,
                "g": item.group,
                "marked": item.marked,
            }
            for item in items
        ],
        "flag": public_flag,
    }
    encoded = json.dumps(
        public_material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ITEM_COUNT",
    "PAIR_COUNT",
    "LatentOrderingTask",
    "PublicOrderingItem",
]
