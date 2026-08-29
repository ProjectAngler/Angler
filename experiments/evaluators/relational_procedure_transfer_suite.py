"""Evaluator-owned cross-family streams for procedural transfer.

The learner receives a lossless packing of one public precedence graph into
the existing five-item permutation interface plus an opaque procedure tree.
It never receives an ordered target, generator seed, family identifier, or
orientation label.  The evaluator returns only one scalar constraint score.

This module deliberately does not topologically sort, rank, or otherwise
solve a graph on the learner's behalf.  ``rank_a`` names a displayed node and
``rank_b`` names its immediate successor; ``group`` is an outgoing-edge bit
and ``marked`` is the public start-node bit.  Those fields are a serialization
of the four visible edges, not the answer order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Sequence

from angler.worlds.latent_order_programs import PublicOrderingItem
from angler.worlds.procedural_constraints import (
    LearnerTask,
    generate_relational_task,
    score_constraint_satisfaction,
)
from experiments.evaluators.skill_memory_suite import PublicSkillExpression


_ITEM_COUNT = 5
_SKILL_SYMBOL = re.compile(r"^skill_[0-9a-f]{20}$")


@dataclass(frozen=True, slots=True)
class PublicPrecedenceEdge:
    """One learner-visible directed edge in displayed-node coordinates."""

    earlier_index: int
    later_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("earlier_index", self.earlier_index),
            ("later_index", self.later_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < _ITEM_COUNT:
                raise ValueError(f"{name} is outside the displayed item range")
        if self.earlier_index == self.later_index:
            raise ValueError("a precedence edge cannot be a self-edge")

    def to_canonical(self) -> dict[str, int]:
        return {
            "earlier_index": self.earlier_index,
            "later_index": self.later_index,
        }


@dataclass(frozen=True, slots=True)
class PublicRelationalProcedureTask:
    """Complete learner view with no task, family, or solution identity."""

    items: tuple[PublicOrderingItem, ...]
    public_flag: bool
    request: PublicSkillExpression
    precedence_edges: tuple[PublicPrecedenceEdge, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or len(self.items) != _ITEM_COUNT:
            raise ValueError(f"public task must contain exactly {_ITEM_COUNT} items")
        if any(not isinstance(item, PublicOrderingItem) for item in self.items):
            raise TypeError("items must be PublicOrderingItem values")
        if len({item.symbol for item in self.items}) != _ITEM_COUNT:
            raise ValueError("public item symbols must be unique")
        if type(self.public_flag) is not bool:
            raise TypeError("public_flag must be bool")
        if not isinstance(self.request, PublicSkillExpression):
            raise TypeError("request must be a PublicSkillExpression")
        if type(self.precedence_edges) is not tuple or len(
            self.precedence_edges
        ) != (_ITEM_COUNT - 1):
            raise ValueError("public task must contain four precedence edges")
        if any(
            not isinstance(edge, PublicPrecedenceEdge)
            for edge in self.precedence_edges
        ):
            raise TypeError("precedence_edges must be PublicPrecedenceEdge values")

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
            "precedence_edges": [
                edge.to_canonical() for edge in self.precedence_edges
            ],
            "public_flag": self.public_flag,
            "request": self.request.to_canonical(),
        }


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenRelationalProcedureSolution:
    """Evaluator-only judge binding; contains no stored target ordering."""

    public_digest: str
    source_task: LearnerTask
    reverse: bool
    source_instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.public_digest, str) or not self.public_digest.startswith(
            "sha256:"
        ):
            raise ValueError("public_digest must be a canonical digest")
        if not isinstance(self.source_task, LearnerTask):
            raise TypeError("source_task must be a relational LearnerTask")
        if type(self.reverse) is not bool:
            raise TypeError("reverse must be bool")
        if self.source_instance_id != self.source_task.instance_id:
            raise ValueError("source task identity is inconsistent")


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedRelationalProcedureTask:
    """Evaluator-owned public/private pair; pass only ``learner`` to policy."""

    learner: PublicRelationalProcedureTask
    hidden: _HiddenRelationalProcedureSolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class RelationalProcedureTransferStream:
    """One opaque mapping with disjoint online supports and no-write queries."""

    supports: tuple[GeneratedRelationalProcedureTask, ...]
    queries: tuple[GeneratedRelationalProcedureTask, ...]

    def __post_init__(self) -> None:
        if not self.supports or not self.queries:
            raise ValueError("transfer stream requires supports and queries")
        support_ids = {pair.hidden.source_instance_id for pair in self.supports}
        query_ids = {pair.hidden.source_instance_id for pair in self.queries}
        if len(support_ids) != len(self.supports):
            raise ValueError("support source identities must be unique")
        if len(query_ids) != len(self.queries):
            raise ValueError("query source identities must be unique")
        if support_ids & query_ids:
            raise ValueError("support and query source identities overlap")


def make_relational_procedure_transfer_stream(
    seed: int,
    *,
    supports_per_procedure: int = 64,
    queries_per_procedure: int = 40,
) -> RelationalProcedureTransferStream:
    """Create forward-path and reverse-path procedures over fresh graphs."""

    _validate_seed(seed)
    for name, value in (
        ("supports_per_procedure", supports_per_procedure),
        ("queries_per_procedure", queries_per_procedure),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    path_symbol = _opaque_symbol(seed, 0)
    reverse_symbol = _opaque_symbol(seed, 1)
    path_request = PublicSkillExpression(path_symbol)
    reverse_request = PublicSkillExpression(
        reverse_symbol,
        (path_request,),
    )

    def build(scope: str, count: int) -> tuple[GeneratedRelationalProcedureTask, ...]:
        rows: list[GeneratedRelationalProcedureTask] = []
        for index in range(count):
            # The first forward observation makes the shared child available;
            # subsequent forward/reverse observations are interleaved.
            for reverse, request in (
                (False, path_request),
                (True, reverse_request),
            ):
                instance_seed = _domain_seed(seed, scope, index, int(reverse))
                generated = generate_relational_task(
                    instance_seed,
                    item_count=_ITEM_COUNT,
                )
                learner = _pack_public_graph(
                    generated.learner,
                    request=request,
                    public_flag=bool(_domain_seed(seed, scope, index, 2) & 1),
                )
                hidden = _HiddenRelationalProcedureSolution(
                    public_digest=_public_digest(learner),
                    source_task=generated.learner,
                    reverse=reverse,
                    source_instance_id=generated.learner.instance_id,
                )
                rows.append(GeneratedRelationalProcedureTask(learner, hidden))
        return tuple(rows)

    supports = build("support", supports_per_procedure)
    queries = list(build("query", queries_per_procedure))
    random.Random(_domain_seed(seed, "query-order", 0, 0)).shuffle(queries)
    return RelationalProcedureTransferStream(supports, tuple(queries))


def score_relational_procedure_answer(
    task: PublicRelationalProcedureTask,
    solution: _HiddenRelationalProcedureSolution,
    answer: str | Sequence[str],
) -> float:
    """Return only scalar visible-constraint satisfaction for one frozen answer."""

    _validate_pairing(task, solution)
    submitted = (
        tuple(part.strip() for part in answer.split(","))
        if isinstance(answer, str)
        else tuple(str(part).strip() for part in answer)
    )
    evaluated = tuple(reversed(submitted)) if solution.reverse else submitted
    return float(
        score_constraint_satisfaction(
            solution.source_task,
            evaluated,
        ).constraint_satisfaction
    )


def _pack_public_graph(
    task: LearnerTask,
    *,
    request: PublicSkillExpression,
    public_flag: bool,
) -> PublicRelationalProcedureTask:
    """Losslessly serialize public edges without deriving a solution rank."""

    if len(task.symbols) != _ITEM_COUNT:
        raise ValueError("relational transfer requires exactly five symbols")
    positions = {symbol: index for index, symbol in enumerate(task.symbols)}
    outgoing = {
        positions[constraint.earlier]: positions[constraint.later]
        for constraint in task.constraints
    }
    incoming = {later for later in outgoing.values()}
    edges = tuple(
        PublicPrecedenceEdge(
            positions[constraint.earlier],
            positions[constraint.later],
        )
        for constraint in task.constraints
    )
    items = tuple(
        PublicOrderingItem(
            symbol=symbol,
            rank_a=index,
            rank_b=outgoing.get(index, index),
            group=int(index in outgoing),
            marked=index not in incoming,
        )
        for index, symbol in enumerate(task.symbols)
    )
    return PublicRelationalProcedureTask(
        items=items,
        public_flag=public_flag,
        request=request,
        precedence_edges=edges,
    )


def _validate_pairing(
    task: PublicRelationalProcedureTask,
    solution: _HiddenRelationalProcedureSolution,
) -> None:
    if not isinstance(task, PublicRelationalProcedureTask):
        raise TypeError("task must be a PublicRelationalProcedureTask")
    if not isinstance(solution, _HiddenRelationalProcedureSolution):
        raise TypeError("solution must remain evaluator-owned")
    if _public_digest(task) != solution.public_digest:
        raise ValueError("public task and evaluator binding do not match")
    source = solution.source_task
    if tuple(item.symbol for item in task.items) != source.symbols:
        raise ValueError("public symbols differ from the bound source task")


def _public_digest(task: PublicRelationalProcedureTask) -> str:
    encoded = json.dumps(
        task.to_canonical(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"project-angler.relational-procedure-public.v1\x00" + encoded
    ).hexdigest()


def _opaque_symbol(seed: int, index: int) -> str:
    material = f"project-angler.cross-family-symbol.v1\x00{seed}\x00{index}".encode(
        "utf-8"
    )
    value = "skill_" + hashlib.sha256(material).hexdigest()[:20]
    if _SKILL_SYMBOL.fullmatch(value) is None:
        raise RuntimeError("opaque symbol generation failed")
    return value


def _domain_seed(seed: int, scope: str, index: int, variant: int) -> int:
    material = (
        f"project-angler.relational-procedure-seed.v1\x00{seed}\x00{scope}"
        f"\x00{index}\x00{variant}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


__all__ = [
    "GeneratedRelationalProcedureTask",
    "PublicPrecedenceEdge",
    "PublicRelationalProcedureTask",
    "RelationalProcedureTransferStream",
    "make_relational_procedure_transfer_stream",
    "score_relational_procedure_answer",
]
