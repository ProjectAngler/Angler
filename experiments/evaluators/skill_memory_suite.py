"""Evaluator-owned streams for skill-local procedural memory.

The learner sees public item attributes, a public flag, and a tree made from
opaque skill symbols.  The meaning of each symbol, the corresponding
``OrderingProgram``, the target ordering, partition identity, and generator
seed remain in the evaluator-owned half of a paired object.

Every answer is frozen before this module returns one scalar pairwise score.
No target order, violated pair, program label, trace, or task identifier is
returned to the learner.  Public instances are sampled independently of the
hidden program by ``latent_order_programs`` and are unique across the
deterministic train/development/final partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Literal, Sequence

from angler.worlds.latent_order_programs import (
    ITEM_COUNT,
    GeneratedLatentOrderingTask,
    OrderingProgram,
    PublicOrderingItem,
    TRAIN_PROGRAMS,
    VALIDATION_PROGRAMS,
    generate_latent_ordering_task,
    make_renamed_latent_variant,
    score_latent_ordering_answer,
)


PartitionName = Literal["train", "development", "final"]
_PARTITIONS: tuple[PartitionName, ...] = (
    "train",
    "development",
    "final",
)
_SKILL_SYMBOL = re.compile(r"^skill_[0-9a-f]{20}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUITE_VERSION = "angler.skill-memory-suite.v2"
_PRIMITIVE_ARITY = {
    "A_ASC": 0,
    "A_DESC": 0,
    "B_ASC": 0,
    "B_DESC": 0,
    "GROUP_01": 1,
    "GROUP_10": 1,
    "ZIGZAG": 1,
    "ROTATE": 1,
    "IF_FLAG": 2,
    "IF_NOT_FLAG": 2,
}
_MATCHED_BINARY_SCOPE = "evaluator-matched-binary-grid"
_MATCHED_BINARY_CELL_ORDER = (
    ("IF_FLAG", False, 0),
    ("IF_FLAG", True, 1),
    ("IF_NOT_FLAG", False, 1),
    ("IF_NOT_FLAG", True, 0),
)


@dataclass(frozen=True, slots=True)
class PublicSkillExpression:
    """A public procedure tree whose node meanings are evaluator-private."""

    symbol: str
    children: tuple["PublicSkillExpression", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or _SKILL_SYMBOL.fullmatch(
            self.symbol
        ) is None:
            raise ValueError("skill symbol must be an opaque canonical token")
        if type(self.children) is not tuple or any(
            not isinstance(child, PublicSkillExpression)
            for child in self.children
        ):
            raise TypeError("skill-expression children must be an immutable tuple")

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(
            child.depth for child in self.children
        )

    def to_canonical(self) -> dict[str, object]:
        return {
            "children": [child.to_canonical() for child in self.children],
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class PublicSkillMemoryTask:
    """Complete learner view; deliberately contains no task identity."""

    items: tuple[PublicOrderingItem, ...]
    public_flag: bool
    request: PublicSkillExpression

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or len(self.items) != ITEM_COUNT:
            raise ValueError(f"public task must contain exactly {ITEM_COUNT} items")
        if any(not isinstance(item, PublicOrderingItem) for item in self.items):
            raise TypeError("public task items must be PublicOrderingItem values")
        if len({item.symbol for item in self.items}) != ITEM_COUNT:
            raise ValueError("public item symbols must be unique")
        if type(self.public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        if not isinstance(self.request, PublicSkillExpression):
            raise TypeError("request must be a PublicSkillExpression")

    def to_canonical(self) -> dict[str, object]:
        return {
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
class _HiddenSkillMapping:
    """Evaluator-only bijection from public symbols to primitive meanings."""

    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or len(self.entries) != len(
            _PRIMITIVE_ARITY
        ):
            raise ValueError("hidden mapping must cover every primitive exactly once")
        symbols = tuple(symbol for symbol, _ in self.entries)
        operators = tuple(operator for _, operator in self.entries)
        if any(_SKILL_SYMBOL.fullmatch(symbol) is None for symbol in symbols):
            raise ValueError("hidden mapping contains a non-canonical symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("hidden mapping symbols must be unique")
        if set(operators) != set(_PRIMITIVE_ARITY):
            raise ValueError("hidden mapping operator coverage is invalid")

    @property
    def digest(self) -> str:
        return _digest("mapping", {"entries": self.entries})

    def express(self, program: OrderingProgram) -> PublicSkillExpression:
        if not isinstance(program, OrderingProgram):
            raise TypeError("program must be an OrderingProgram")
        by_operator = {operator: symbol for symbol, operator in self.entries}
        try:
            symbol = by_operator[program.operator]
        except KeyError as error:
            raise ValueError("program uses an unsupported primitive") from error
        return PublicSkillExpression(
            symbol,
            tuple(self.express(child) for child in program.children),
        )

    def resolve(self, expression: PublicSkillExpression) -> OrderingProgram:
        if not isinstance(expression, PublicSkillExpression):
            raise TypeError("expression must be a PublicSkillExpression")
        by_symbol = dict(self.entries)
        try:
            operator = by_symbol[expression.symbol]
        except KeyError as error:
            raise ValueError("expression contains an unknown skill symbol") from error
        expected = _PRIMITIVE_ARITY[operator]
        if len(expression.children) != expected:
            raise ValueError("expression topology is invalid for its hidden mapping")
        return OrderingProgram(
            operator,
            tuple(self.resolve(child) for child in expression.children),
        )


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenSkillMemorySolution:
    """Sealed evaluator state paired with exactly one public task."""

    instance_identity: str
    source_instance_identity: str
    mechanism_identity: str
    partition: PartitionName
    mapping: _HiddenSkillMapping
    generated: GeneratedLatentOrderingTask

    def __post_init__(self) -> None:
        for name in (
            "instance_identity",
            "source_instance_identity",
            "mechanism_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{name} must be a canonical sha256 digest")
        _validate_partition(self.partition)
        if not isinstance(self.mapping, _HiddenSkillMapping):
            raise TypeError("mapping must remain an evaluator-owned mapping")
        if not isinstance(self.generated, GeneratedLatentOrderingTask):
            raise TypeError("generated must be a latent-order task pair")

    @property
    def program(self) -> OrderingProgram:
        return self.generated.hidden.program


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedSkillMemoryTask:
    """Evaluator-owned pairing; pass only ``learner`` into learner code."""

    learner: PublicSkillMemoryTask
    hidden: _HiddenSkillMemorySolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class MatchedDescendantQuery:
    """Evaluator-owned pair differing only in the requested child tree."""

    left: GeneratedSkillMemoryTask
    right: GeneratedSkillMemoryTask

    def __post_init__(self) -> None:
        if not isinstance(self.left, GeneratedSkillMemoryTask) or not isinstance(
            self.right, GeneratedSkillMemoryTask
        ):
            raise TypeError("matched descendants must contain generated tasks")
        left = self.left
        right = self.right
        if (
            left.learner.items != right.learner.items
            or left.learner.public_flag is not right.learner.public_flag
            or left.hidden.source_instance_identity
            != right.hidden.source_instance_identity
            or left.hidden.mapping.digest != right.hidden.mapping.digest
            or left.learner.request.symbol != right.learner.request.symbol
            or left.hidden.program.operator != right.hidden.program.operator
            or len(left.learner.request.children)
            != len(right.learner.request.children)
        ):
            raise ValueError("matched descendants differ outside the child tree")
        if (
            left.learner.request.children == right.learner.request.children
            or left.hidden.generated.hidden.target_order
            == right.hidden.generated.hidden.target_order
        ):
            raise ValueError("matched descendants require distinct trees and targets")


@dataclass(frozen=True, slots=True, repr=False)
class MatchedBinaryBranchCell:
    """One evaluator-owned cell in a matched conditional 2x2 grid."""

    task: GeneratedSkillMemoryTask
    expected_branch: int

    def __post_init__(self) -> None:
        if not isinstance(self.task, GeneratedSkillMemoryTask):
            raise TypeError("matched binary cell must contain a generated task")
        operator = self.task.hidden.program.operator
        if operator not in {"IF_FLAG", "IF_NOT_FLAG"}:
            raise ValueError("matched binary cell root must be conditional")
        if self.expected_branch not in (0, 1):
            raise ValueError("expected branch index must be zero or one")
        expected = (
            int(self.task.learner.public_flag)
            if operator == "IF_FLAG"
            else 1 - int(self.task.learner.public_flag)
        )
        if self.expected_branch != expected:
            raise ValueError("expected branch index contradicts operator semantics")

    @property
    def hidden_operator(self) -> str:
        return self.task.hidden.program.operator

    @property
    def public_flag(self) -> bool:
        return self.task.learner.public_flag


@dataclass(frozen=True, slots=True, repr=False)
class MatchedBinaryBranchCase:
    """Four cells that differ only by conditional operator and public flag."""

    cells: tuple[MatchedBinaryBranchCell, ...]

    def __post_init__(self) -> None:
        if type(self.cells) is not tuple or len(self.cells) != 4:
            raise ValueError("matched binary case must contain exactly four cells")
        if any(not isinstance(cell, MatchedBinaryBranchCell) for cell in self.cells):
            raise TypeError("matched binary case contains an invalid cell")
        observed = tuple(
            (
                cell.hidden_operator,
                cell.public_flag,
                cell.expected_branch,
            )
            for cell in self.cells
        )
        if observed != _MATCHED_BINARY_CELL_ORDER:
            raise ValueError("matched binary cells are not in canonical grid order")

        generated = tuple(cell.task for cell in self.cells)
        exemplar = generated[0]
        if any(task.learner.items != exemplar.learner.items for task in generated[1:]):
            raise ValueError("matched binary cells must share exact public items")
        if any(
            task.hidden.mapping.digest != exemplar.hidden.mapping.digest
            for task in generated[1:]
        ):
            raise ValueError("matched binary cells must share one opaque mapping")
        child_programs = exemplar.hidden.program.children
        child_requests = exemplar.learner.request.children
        if any(
            task.hidden.program.children != child_programs
            or task.learner.request.children != child_requests
            for task in generated[1:]
        ):
            raise ValueError("matched binary cells must share an ordered child pair")

        targets_by_branch: dict[int, tuple[str, ...]] = {}
        for cell in self.cells:
            target = cell.task.hidden.generated.hidden.target_order
            prior = targets_by_branch.setdefault(cell.expected_branch, target)
            if prior != target:
                raise ValueError("matched cells disagree about a child target")
        if (
            set(targets_by_branch) != {0, 1}
            or targets_by_branch[0] == targets_by_branch[1]
        ):
            raise ValueError("matched binary children must produce distinct targets")


@dataclass(frozen=True, slots=True, repr=False)
class MatchedBinaryBranchGrid:
    """Evaluator-only cases for the conditional operator-by-flag matrix."""

    cases: tuple[MatchedBinaryBranchCase, ...]

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("matched binary grid must contain at least one case")
        if any(not isinstance(case, MatchedBinaryBranchCase) for case in self.cases):
            raise TypeError("matched binary grid contains an invalid case")
        cells = self.cells
        identities = tuple(cell.task.hidden.instance_identity for cell in cells)
        if len(set(identities)) != len(identities):
            raise ValueError("matched binary grid contains duplicate task identities")
        mappings = {
            cell.task.hidden.mapping.digest
            for cell in cells
        }
        if len(mappings) != 1:
            raise ValueError("matched binary grid must share one opaque mapping")

    @property
    def cases_per_cell(self) -> int:
        return len(self.cases)

    @property
    def cells(self) -> tuple[MatchedBinaryBranchCell, ...]:
        return tuple(cell for case in self.cases for cell in case.cells)


@dataclass(frozen=True, slots=True, repr=False)
class SkillMemoryPartition:
    """One deterministic evaluator stream."""

    name: PartitionName
    tasks: tuple[GeneratedSkillMemoryTask, ...]

    def __post_init__(self) -> None:
        _validate_partition(self.name)
        if type(self.tasks) is not tuple or not self.tasks:
            raise ValueError("partition tasks must be a non-empty tuple")
        if any(not isinstance(task, GeneratedSkillMemoryTask) for task in self.tasks):
            raise TypeError("partition contains a non-skill-memory task")
        if any(task.hidden.partition != self.name for task in self.tasks):
            raise ValueError("task partition identity does not match its container")
        identities = tuple(task.hidden.instance_identity for task in self.tasks)
        sources = tuple(task.hidden.source_instance_identity for task in self.tasks)
        if len(set(identities)) != len(identities):
            raise ValueError("partition contains a duplicate task identity")
        if len(set(sources)) != len(sources):
            raise ValueError("partition contains a duplicate public instance")

    @property
    def learner_tasks(self) -> tuple[PublicSkillMemoryTask, ...]:
        return tuple(task.learner for task in self.tasks)


@dataclass(frozen=True, slots=True, repr=False)
class SkillMemoryPartitions:
    train: SkillMemoryPartition
    development: SkillMemoryPartition
    final: SkillMemoryPartition

    def __post_init__(self) -> None:
        if (
            self.train.name,
            self.development.name,
            self.final.name,
        ) != _PARTITIONS:
            raise ValueError("partition bundle order or identities are invalid")
        partitions = (self.train, self.development, self.final)
        identity_sets = [
            {task.hidden.instance_identity for task in partition.tasks}
            for partition in partitions
        ]
        source_sets = [
            {task.hidden.source_instance_identity for task in partition.tasks}
            for partition in partitions
        ]
        for left in range(len(partitions)):
            for right in range(left + 1, len(partitions)):
                if identity_sets[left] & identity_sets[right]:
                    raise ValueError("partition task identities overlap")
                if source_sets[left] & source_sets[right]:
                    raise ValueError("partition public instances overlap")


@dataclass(frozen=True, slots=True, repr=False)
class SkillMemoryCompositionCurriculum:
    """One opaque mapping shared by support, probe, and composition stages.

    ``component_supports`` covers every hidden primitive as the root of its
    smallest valid program context and supplies scalar feedback.
    ``component_probes`` is a source- and instance-disjoint no-feedback stage
    over those same primitive roots.  ``composition_queries`` contains only
    structurally deeper programs and is also query-only.  This object belongs
    to the evaluator; learner code should receive only the three public-task
    properties below.
    """

    component_supports: tuple[GeneratedSkillMemoryTask, ...]
    component_probes: tuple[GeneratedSkillMemoryTask, ...]
    composition_queries: tuple[GeneratedSkillMemoryTask, ...]

    def __post_init__(self) -> None:
        for values, label in (
            (self.component_supports, "component supports"),
            (self.component_probes, "component probes"),
            (self.composition_queries, "composition queries"),
        ):
            if type(values) is not tuple or not values:
                raise ValueError(f"{label} must be a non-empty tuple")
            if any(not isinstance(item, GeneratedSkillMemoryTask) for item in values):
                raise TypeError(f"{label} contains an invalid task")

        all_tasks = (
            self.component_supports
            + self.component_probes
            + self.composition_queries
        )
        mappings = {task.hidden.mapping.digest for task in all_tasks}
        if len(mappings) != 1:
            raise ValueError("curriculum stages must share exactly one opaque mapping")
        for values, label in (
            (self.component_supports, "component supports"),
            (self.component_probes, "component probes"),
        ):
            roots = {task.hidden.program.operator for task in values}
            if roots != set(_PRIMITIVE_ARITY):
                raise ValueError(f"{label} do not cover all primitives")
            if any(task.hidden.program.depth > 1 for task in values):
                raise ValueError(f"{label} use a non-minimal program context")
            counts: dict[str, int] = {}
            for task in values:
                root = task.hidden.program.operator
                counts[root] = counts.get(root, 0) + 1
            if len(set(counts.values())) != 1:
                raise ValueError(f"{label} are not balanced by primitive root")
        composition_depths = {
            task.hidden.program.depth for task in self.composition_queries
        }
        if not {2, 3}.issubset(composition_depths):
            raise ValueError("composition queries must cover depths two and three")

        stages = (
            ("component supports", self.component_supports),
            ("component probes", self.component_probes),
            ("composition queries", self.composition_queries),
        )
        identity_sets: list[set[str]] = []
        source_sets: list[set[str]] = []
        for label, values in stages:
            identities = {task.hidden.instance_identity for task in values}
            sources = {task.hidden.source_instance_identity for task in values}
            if len(identities) != len(values) or len(sources) != len(values):
                raise ValueError(f"{label} are not unique")
            identity_sets.append(identities)
            source_sets.append(sources)
        for left in range(len(stages)):
            for right in range(left + 1, len(stages)):
                if identity_sets[left] & identity_sets[right]:
                    raise ValueError("curriculum stage task identities overlap")
                if source_sets[left] & source_sets[right]:
                    raise ValueError("curriculum stage public instances overlap")

    @property
    def learner_component_supports(self) -> tuple[PublicSkillMemoryTask, ...]:
        return tuple(task.learner for task in self.component_supports)

    @property
    def learner_component_probes(self) -> tuple[PublicSkillMemoryTask, ...]:
        return tuple(task.learner for task in self.component_probes)

    @property
    def learner_composition_queries(self) -> tuple[PublicSkillMemoryTask, ...]:
        return tuple(task.learner for task in self.composition_queries)


def make_skill_memory_partition(
    partition: PartitionName,
    seed: int,
    *,
    instances_per_program: int = 8,
) -> SkillMemoryPartition:
    """Build a deterministic interleaved stream with unique public instances."""

    _validate_partition(partition)
    _validate_seed(seed)
    if (
        isinstance(instances_per_program, bool)
        or not isinstance(instances_per_program, int)
        or instances_per_program <= 0
    ):
        raise ValueError("instances_per_program must be a positive integer")

    programs = _partition_programs(partition)
    mapping = _make_mapping(seed, partition)
    rng = _domain_rng(seed, partition, "stream-order")
    tasks: list[GeneratedSkillMemoryTask] = []
    for presentation_index in range(instances_per_program):
        order = list(range(len(programs)))
        rng.shuffle(order)
        for program_index in order:
            program = programs[program_index]
            request = mapping.express(program)
            generated = generate_latent_ordering_task(
                program,
                _instance_seed(
                    seed,
                    partition,
                    program_index,
                    presentation_index,
                ),
                public_flag=bool(presentation_index % 2),
            )
            tasks.append(
                _pair_task(
                    partition,
                    mapping,
                    request,
                    generated,
                )
            )
    return SkillMemoryPartition(partition, tuple(tasks))


def make_skill_memory_meta_partition(
    seed: int,
    *,
    instances_per_program: int = 8,
) -> SkillMemoryPartition:
    """Build the private context-diverse partition used only for meta-training.

    Every unary primitive is presented over all four leaf contexts, the
    conditional varies both ordered branches, and private depth-two/three
    programs teach composition without importing the sealed final evaluator.
    All public symbols remain freshly permuted and opaque for each call.
    """

    _validate_seed(seed)
    if (
        isinstance(instances_per_program, bool)
        or not isinstance(instances_per_program, int)
        or instances_per_program < 8
        or instances_per_program % 2
    ):
        raise ValueError(
            "instances_per_program must be an even integer of at least eight"
        )
    programs = _meta_training_programs()
    mapping = _make_scoped_mapping(seed, "meta-train")
    rng = _domain_rng(seed, "meta-train", "stream-order")
    tasks: list[GeneratedSkillMemoryTask] = []
    for presentation_index in range(instances_per_program):
        order = list(range(len(programs)))
        rng.shuffle(order)
        for program_index in order:
            program = programs[program_index]
            generated = generate_latent_ordering_task(
                program,
                _curriculum_instance_seed(
                    seed,
                    "meta-train",
                    program_index,
                    presentation_index,
                ),
                public_flag=bool(presentation_index % 2),
            )
            tasks.append(
                _pair_task(
                    "train",
                    mapping,
                    mapping.express(program),
                    generated,
                )
            )
    return SkillMemoryPartition("train", tuple(tasks))


def make_skill_memory_meta_matched_queries(
    seed: int,
) -> tuple[MatchedDescendantQuery, ...]:
    """Create varied same-instance counterfactual trees for every non-leaf root.

    Public instance sampling is independent of the hidden program, so using
    one seed for both sides holds every public input fixed.  The pair remains
    evaluator-owned; learner code receives each ordinary public task alone.
    Three structural pairs per root are sampled from the broad meta-training
    program set.  A fresh outer seed changes both the public instance and the
    selected tree pairs, preventing the composer from memorizing five fixed
    counterfactual templates.
    """

    _validate_seed(seed)
    by_root: dict[str, list[OrderingProgram]] = {}
    for program in _meta_training_programs():
        if program.children:
            by_root.setdefault(program.operator, []).append(program)
    expected_roots = {
        operator for operator, arity in _PRIMITIVE_ARITY.items() if arity > 0
    }
    if set(by_root) != expected_roots:
        raise RuntimeError("matched descendants do not cover every non-leaf root")
    template_rng = _domain_rng(seed, "meta-matched-descendants", "tree-pairs")
    templates: list[tuple[OrderingProgram, OrderingProgram]] = []
    for operator in sorted(by_root):
        variants = sorted(by_root[operator], key=lambda program: program.canonical)
        candidates = [
            (variants[left], variants[right])
            for left in range(len(variants))
            for right in range(left + 1, len(variants))
        ]
        if len(candidates) < 3:
            raise RuntimeError("non-leaf root has insufficient tree variation")
        template_rng.shuffle(candidates)
        templates.extend(candidates[:3])
    mapping = _make_scoped_mapping(seed, "meta-train")
    pairs: list[MatchedDescendantQuery] = []
    for pair_index, (left_program, right_program) in enumerate(templates):
        for attempt in range(64):
            instance_seed = _curriculum_instance_seed(
                seed,
                "meta-matched-descendants",
                pair_index,
                attempt,
            )
            public_flag = bool((seed + pair_index + attempt) % 2)
            left_generated = generate_latent_ordering_task(
                left_program,
                instance_seed,
                public_flag=public_flag,
            )
            right_generated = generate_latent_ordering_task(
                right_program,
                instance_seed,
                public_flag=public_flag,
            )
            if (
                left_generated.hidden.target_order
                == right_generated.hidden.target_order
            ):
                continue
            pairs.append(
                MatchedDescendantQuery(
                    _pair_task(
                        "train",
                        mapping,
                        mapping.express(left_program),
                        left_generated,
                    ),
                    _pair_task(
                        "train",
                        mapping,
                        mapping.express(right_program),
                        right_generated,
                    ),
                )
            )
            break
        else:
            raise RuntimeError("could not generate distinct matched targets")
    return tuple(pairs)


def make_skill_memory_matched_binary_branch_grid(
    seed: int,
    *,
    cases_per_cell: int = 8,
) -> MatchedBinaryBranchGrid:
    """Build a sealed same-instance conditional operator-by-flag grid.

    Each case holds public items and an exact ordered child pair fixed while
    varying only ``IF_FLAG`` versus ``IF_NOT_FLAG`` and false versus true.
    The grid shares the final curriculum's opaque mapping so acquired symbols
    remain meaningful, while using a fresh evaluator-only instance-seed
    domain.  Returned tasks are final-evaluation tasks and are never inserted
    into any train, meta-train, support, or curriculum partition.
    """

    _validate_seed(seed)
    if (
        isinstance(cases_per_cell, bool)
        or not isinstance(cases_per_cell, int)
        or cases_per_cell <= 0
    ):
        raise ValueError("cases_per_cell must be a positive integer")

    # Reuse the final curriculum's opaque vocabulary so a learned policy can
    # resolve these symbols, but never reuse its public-instance seed domain.
    mapping = _make_scoped_mapping(seed, "component-composition")
    leaves = tuple(
        OrderingProgram(operator)
        for operator in ("A_ASC", "A_DESC", "B_ASC", "B_DESC")
    )
    child_pairs = [
        (left, right)
        for left in leaves
        for right in leaves
        if left != right
    ]
    pair_rng = _domain_rng(seed, _MATCHED_BINARY_SCOPE, "child-pairs")
    pair_rng.shuffle(child_pairs)

    cases: list[MatchedBinaryBranchCase] = []
    for case_index in range(cases_per_cell):
        for attempt in range(128):
            children = child_pairs[(case_index + attempt) % len(child_pairs)]
            instance_seed = _curriculum_instance_seed(
                seed,
                _MATCHED_BINARY_SCOPE,
                case_index,
                attempt,
            )
            cells: list[MatchedBinaryBranchCell] = []
            for operator, public_flag, expected_branch in _MATCHED_BINARY_CELL_ORDER:
                program = OrderingProgram(operator, children)
                generated = generate_latent_ordering_task(
                    program,
                    instance_seed,
                    public_flag=public_flag,
                )
                cells.append(
                    MatchedBinaryBranchCell(
                        _pair_task(
                            "final",
                            mapping,
                            mapping.express(program),
                            generated,
                        ),
                        expected_branch,
                    )
                )
            left_target = cells[0].task.hidden.generated.hidden.target_order
            right_target = cells[1].task.hidden.generated.hidden.target_order
            if left_target == right_target:
                continue
            cases.append(MatchedBinaryBranchCase(tuple(cells)))
            break
        else:
            raise RuntimeError(
                "could not generate distinct matched binary child targets"
            )
    return MatchedBinaryBranchGrid(tuple(cases))


def make_skill_memory_partitions(
    seed: int,
    *,
    instances_per_program: int = 8,
) -> SkillMemoryPartitions:
    """Build mutually disjoint train, development, and final streams."""

    _validate_seed(seed)
    return SkillMemoryPartitions(
        train=make_skill_memory_partition(
            "train",
            seed,
            instances_per_program=instances_per_program,
        ),
        development=make_skill_memory_partition(
            "development",
            seed,
            instances_per_program=instances_per_program,
        ),
        final=make_skill_memory_partition(
            "final",
            seed,
            instances_per_program=instances_per_program,
        ),
    )


def make_skill_memory_composition_curriculum(
    seed: int,
    *,
    encounters_per_primitive: int = 8,
    cases_per_component_probe: int = 8,
    cases_per_composition: int = 8,
) -> SkillMemoryCompositionCurriculum:
    """Create feedback supports followed by two disjoint no-feedback stages.

    Counts must be even so every hidden mechanism is presented equally under
    both public-flag values.  Public instances, including renamed entity
    surfaces and display order, are unique.  All three stages use one randomly
    permuted opaque-symbol mapping, while their source-seed domains remain
    mutually disjoint.
    """

    _validate_seed(seed)
    _validate_balanced_count(
        encounters_per_primitive,
        "encounters_per_primitive",
    )
    _validate_balanced_count(
        cases_per_component_probe,
        "cases_per_component_probe",
    )
    _validate_balanced_count(
        cases_per_composition,
        "cases_per_composition",
    )
    mapping = _make_scoped_mapping(seed, "component-composition")

    supports: list[GeneratedSkillMemoryTask] = []
    support_programs = _component_support_program_variants()
    support_rng = _domain_rng(seed, "component-composition", "support-order")
    for encounter_index in range(encounters_per_primitive):
        order = list(range(len(support_programs)))
        support_rng.shuffle(order)
        for program_index in order:
            variants = support_programs[program_index]
            program = variants[(encounter_index // 2) % len(variants)]
            generated = generate_latent_ordering_task(
                program,
                _curriculum_instance_seed(
                    seed,
                    "component-support",
                    program_index,
                    encounter_index,
                ),
                public_flag=bool(encounter_index % 2),
            )
            supports.append(
                _pair_task(
                    "train",
                    mapping,
                    mapping.express(program),
                    generated,
                )
            )

    probes: list[GeneratedSkillMemoryTask] = []
    probe_programs = _component_support_program_variants()
    probe_rng = _domain_rng(seed, "component-composition", "probe-order")
    for case_index in range(cases_per_component_probe):
        order = list(range(len(probe_programs)))
        probe_rng.shuffle(order)
        for program_index in order:
            variants = probe_programs[program_index]
            program = variants[(1 + case_index // 2) % len(variants)]
            generated = generate_latent_ordering_task(
                program,
                _curriculum_instance_seed(
                    seed,
                    "component-probe",
                    program_index,
                    case_index,
                ),
                public_flag=bool(case_index % 2),
            )
            probes.append(
                _pair_task(
                    "development",
                    mapping,
                    mapping.express(program),
                    generated,
                )
            )

    query_programs = _composition_query_programs()
    queries: list[GeneratedSkillMemoryTask] = []
    query_rng = _domain_rng(seed, "component-composition", "query-order")
    for case_index in range(cases_per_composition):
        order = list(range(len(query_programs)))
        query_rng.shuffle(order)
        for program_index in order:
            program = query_programs[program_index]
            generated = generate_latent_ordering_task(
                program,
                _curriculum_instance_seed(
                    seed,
                    "composition-query",
                    program_index,
                    case_index,
                ),
                public_flag=bool(case_index % 2),
            )
            queries.append(
                _pair_task(
                    "final",
                    mapping,
                    mapping.express(program),
                    generated,
                )
            )
    return SkillMemoryCompositionCurriculum(
        tuple(supports),
        tuple(probes),
        tuple(queries),
    )


def make_renamed_skill_variant(
    source: GeneratedSkillMemoryTask,
    *,
    seed: int,
) -> GeneratedSkillMemoryTask:
    """Rename only entity surfaces while retaining the opaque procedure."""

    if not isinstance(source, GeneratedSkillMemoryTask):
        raise TypeError("source must be a GeneratedSkillMemoryTask")
    _validate_seed(seed)
    generated = make_renamed_latent_variant(source.hidden.generated, seed=seed)
    renamed = _pair_task(
        source.hidden.partition,
        source.hidden.mapping,
        source.learner.request,
        generated,
    )
    if renamed.hidden.source_instance_identity == (
        source.hidden.source_instance_identity
    ):
        raise RuntimeError("renamed variant did not produce a fresh public instance")
    return renamed


def score_skill_memory_answer(
    task: PublicSkillMemoryTask,
    solution: _HiddenSkillMemorySolution,
    answer: str | Sequence[str],
) -> float:
    """Return only scalar pairwise accuracy for one frozen answer."""

    _validate_pairing(task, solution)
    feedback = score_latent_ordering_answer(
        solution.generated.learner,
        solution.generated.hidden,
        answer,
    )
    return float(feedback.pairwise_accuracy)


def _pair_task(
    partition: PartitionName,
    mapping: _HiddenSkillMapping,
    request: PublicSkillExpression,
    generated: GeneratedLatentOrderingTask,
) -> GeneratedSkillMemoryTask:
    learner = PublicSkillMemoryTask(
        items=generated.learner.items,
        public_flag=generated.learner.public_flag,
        request=request,
    )
    mechanism_identity = _digest(
        "mechanism",
        {
            "mapping": mapping.digest,
            "program": generated.hidden.program.canonical,
        },
    )
    identity = _task_identity(
        partition,
        learner,
        generated.learner.instance_id,
        mapping,
    )
    hidden = _HiddenSkillMemorySolution(
        instance_identity=identity,
        source_instance_identity=generated.learner.instance_id,
        mechanism_identity=mechanism_identity,
        partition=partition,
        mapping=mapping,
        generated=generated,
    )
    return GeneratedSkillMemoryTask(learner, hidden)


def _validate_pairing(
    task: PublicSkillMemoryTask,
    solution: _HiddenSkillMemorySolution,
) -> None:
    if not isinstance(task, PublicSkillMemoryTask):
        raise TypeError("task must be a PublicSkillMemoryTask")
    if not isinstance(solution, _HiddenSkillMemorySolution):
        raise TypeError("solution must be evaluator-owned hidden state")
    source = solution.generated.learner
    if (
        task.items != source.items
        or task.public_flag is not source.public_flag
        or solution.source_instance_identity != source.instance_id
    ):
        raise ValueError("public task and hidden solution do not match")
    resolved = solution.mapping.resolve(task.request)
    if resolved.canonical != solution.program.canonical:
        raise ValueError("opaque request and hidden program do not match")
    expected_mechanism = _digest(
        "mechanism",
        {
            "mapping": solution.mapping.digest,
            "program": solution.program.canonical,
        },
    )
    if solution.mechanism_identity != expected_mechanism:
        raise ValueError("hidden mechanism identity is inconsistent")
    expected_identity = _task_identity(
        solution.partition,
        task,
        source.instance_id,
        solution.mapping,
    )
    if solution.instance_identity != expected_identity:
        raise ValueError("hidden task identity is inconsistent")


def _partition_programs(partition: PartitionName) -> tuple[OrderingProgram, ...]:
    if partition == "train":
        return tuple(TRAIN_PROGRAMS)
    if partition == "development":
        return tuple(VALIDATION_PROGRAMS)
    if partition == "final":
        # Keep the final exact compositions in the evaluator package.  This
        # import is deliberately delayed until a final partition is requested.
        from experiments.evaluators.latent_order_suite import evaluator_programs

        return evaluator_programs()
    raise AssertionError("validated partition was not handled")


def _component_support_program_variants(
) -> tuple[tuple[OrderingProgram, ...], ...]:
    """Return repeated root mechanisms across varied shallow child contexts."""

    leaves = tuple(
        OrderingProgram(operator)
        for operator in ("A_ASC", "A_DESC", "B_ASC", "B_DESC")
    )
    conditional_children = (
        (leaves[0], leaves[2]),
        (leaves[1], leaves[3]),
        (leaves[0], leaves[3]),
        (leaves[2], leaves[1]),
    )
    programs: list[tuple[OrderingProgram, ...]] = []
    for operator, arity in _PRIMITIVE_ARITY.items():
        if arity == 0:
            programs.append((OrderingProgram(operator),))
        elif arity == 1:
            programs.append(
                tuple(OrderingProgram(operator, (child,)) for child in leaves)
            )
        elif arity == 2:
            programs.append(
                tuple(
                    OrderingProgram(operator, children)
                    for children in conditional_children
                )
            )
        else:
            raise AssertionError("unsupported primitive arity")
    return tuple(programs)


def _meta_training_programs() -> tuple[OrderingProgram, ...]:
    """Return broad train-only contexts without loading final programs."""

    leaves = tuple(
        OrderingProgram(operator)
        for operator in ("A_ASC", "A_DESC", "B_ASC", "B_DESC")
    )
    programs: list[OrderingProgram] = list(leaves)
    for operator in ("GROUP_01", "GROUP_10", "ZIGZAG", "ROTATE"):
        programs.extend(OrderingProgram(operator, (child,)) for child in leaves)

    conditional_pairs = (
        (leaves[0], leaves[1]),
        (leaves[1], leaves[0]),
        (leaves[2], leaves[3]),
        (leaves[3], leaves[2]),
        (leaves[0], leaves[2]),
        (leaves[2], leaves[0]),
        (leaves[1], leaves[3]),
        (leaves[3], leaves[1]),
    )
    for operator in ("IF_FLAG", "IF_NOT_FLAG"):
        programs.extend(
            OrderingProgram(operator, children)
            for children in conditional_pairs
        )
    programs.extend(
        (
            OrderingProgram(
                "GROUP_01",
                (OrderingProgram("ROTATE", (leaves[1],)),),
            ),
            OrderingProgram(
                "GROUP_10",
                (OrderingProgram("ZIGZAG", (leaves[2],)),),
            ),
            OrderingProgram(
                "ROTATE",
                (OrderingProgram("GROUP_01", (leaves[3],)),),
            ),
            OrderingProgram(
                "ZIGZAG",
                (OrderingProgram("GROUP_10", (leaves[1],)),),
            ),
            OrderingProgram(
                "IF_FLAG",
                (
                    OrderingProgram("GROUP_01", (leaves[0],)),
                    OrderingProgram("ROTATE", (leaves[2],)),
                ),
            ),
            OrderingProgram(
                "IF_NOT_FLAG",
                (
                    OrderingProgram("ROTATE", (leaves[1],)),
                    OrderingProgram("GROUP_10", (leaves[2],)),
                ),
            ),
            OrderingProgram(
                "GROUP_01",
                (
                    OrderingProgram(
                        "IF_NOT_FLAG",
                        (leaves[2], leaves[1]),
                    ),
                ),
            ),
            OrderingProgram(
                "ROTATE",
                (
                    OrderingProgram(
                        "ZIGZAG",
                        (OrderingProgram("GROUP_01", (leaves[2],)),),
                    ),
                ),
            ),
        )
    )
    result = tuple(programs)
    if len({program.canonical for program in result}) != len(result):
        raise RuntimeError("meta-training programs must be unique")
    depths = {program.depth for program in result}
    if not {0, 1, 2, 3}.issubset(depths):
        raise RuntimeError("meta-training programs must cover depths zero to three")
    return result


def _composition_query_programs() -> tuple[OrderingProgram, ...]:
    from experiments.evaluators.latent_order_suite import evaluator_programs

    programs = tuple(VALIDATION_PROGRAMS) + evaluator_programs()
    if len({program.canonical for program in programs}) != len(programs):
        raise RuntimeError("composition query programs must be unique")
    if any(program.depth < 2 for program in programs):
        raise RuntimeError("composition query contains a shallow program")
    return programs


def _make_mapping(seed: int, partition: PartitionName) -> _HiddenSkillMapping:
    return _make_scoped_mapping(seed, partition)


def _make_scoped_mapping(seed: int, scope: str) -> _HiddenSkillMapping:
    operators = list(_PRIMITIVE_ARITY)
    rng = _domain_rng(seed, scope, "mapping")
    rng.shuffle(operators)
    symbols = tuple(
        "skill_"
        + hashlib.sha256(
            (
                f"{_SUITE_VERSION}\x00{seed}\x00{scope}\x00symbol\x00{index}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        for index in range(len(operators))
    )
    return _HiddenSkillMapping(tuple(zip(symbols, operators, strict=True)))


def _task_identity(
    partition: PartitionName,
    task: PublicSkillMemoryTask,
    source_instance_identity: str,
    mapping: _HiddenSkillMapping,
) -> str:
    return _digest(
        "task",
        {
            "mapping": mapping.digest,
            "partition": partition,
            "public": task.to_canonical(),
            "source_instance_identity": source_instance_identity,
        },
    )


def _instance_seed(
    seed: int,
    partition: PartitionName,
    program_index: int,
    presentation_index: int,
) -> int:
    material = (
        f"{_SUITE_VERSION}\x00{seed}\x00{partition}\x00instance\x00"
        f"{program_index}\x00{presentation_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _curriculum_instance_seed(
    seed: int,
    stage: str,
    program_index: int,
    presentation_index: int,
) -> int:
    material = (
        f"{_SUITE_VERSION}\x00{seed}\x00{stage}\x00instance\x00"
        f"{program_index}\x00{presentation_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _domain_rng(seed: int, partition: str, domain: str) -> random.Random:
    material = (
        f"{_SUITE_VERSION}\x00{seed}\x00{partition}\x00{domain}"
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _digest(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {
            "kind": kind,
            "payload": payload,
            "suite": _SUITE_VERSION,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_partition(partition: str) -> None:
    if partition not in _PARTITIONS:
        raise ValueError("partition must be train, development, or final")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def _validate_balanced_count(value: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value % 2
    ):
        raise ValueError(f"{label} must be a positive even integer")


__all__ = [
    "MatchedBinaryBranchCase",
    "MatchedBinaryBranchCell",
    "MatchedBinaryBranchGrid",
    "MatchedDescendantQuery",
    "GeneratedSkillMemoryTask",
    "PartitionName",
    "PublicSkillExpression",
    "PublicSkillMemoryTask",
    "SkillMemoryCompositionCurriculum",
    "SkillMemoryPartition",
    "SkillMemoryPartitions",
    "make_renamed_skill_variant",
    "make_skill_memory_composition_curriculum",
    "make_skill_memory_matched_binary_branch_grid",
    "make_skill_memory_meta_partition",
    "make_skill_memory_meta_matched_queries",
    "make_skill_memory_partition",
    "make_skill_memory_partitions",
    "score_skill_memory_answer",
]
