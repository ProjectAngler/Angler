"""Seeded relational-order tasks with outcome-only verification.

The module stores the learner projection and final hidden solution in distinct
types.  It generates task facts and verifies submitted outcomes; it never
produces a reasoning trace, strategy, plan, critique, or reflection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Sequence

FAMILY_ID = "angler.relational-order"
FAMILY_VERSION = "1.4.0"

_LABELS = (
    "amber",
    "birch",
    "coral",
    "delta",
    "ember",
    "fjord",
    "gale",
    "harbor",
    "iris",
    "jade",
    "kestrel",
    "lumen",
    "maple",
    "nova",
    "onyx",
    "pearl",
    "quartz",
    "raven",
    "solar",
    "tulip",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "acorn",
    "brook",
    "cinder",
    "dune",
    "elm",
    "flint",
)

DEFAULT_RELATION_SURFACE_FORMS = (
    "{earlier} is before {later}.",
    "{later} is after {earlier}.",
    "{earlier} precedes {later}.",
)


@dataclass(frozen=True, slots=True)
class PrecedenceConstraint:
    """One learner-visible relation with normalized endpoint semantics."""

    earlier: str
    later: str


@dataclass(frozen=True, slots=True)
class LearnerTask:
    """Complete learner-visible projection; it contains no hidden answer."""

    instance_id: str
    family_id: str
    family_version: str
    symbols: tuple[str, ...]
    constraints: tuple[PrecedenceConstraint, ...]
    fact_statements: tuple[str, ...]
    problem_statement: str
    prompt: str


@dataclass(frozen=True, slots=True, repr=False)
class HiddenOrderSolution:
    """Sealed generator material kept outside the learner projection."""

    instance_id: str
    ordered_symbols: tuple[str, ...]
    generator_seed: int


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedRelationalTask:
    """Generator-side pairing of a learner projection and sealed solution."""

    learner: LearnerTask
    hidden: HiddenOrderSolution


@dataclass(frozen=True, slots=True)
class OutcomeFeedback:
    """Bounded final-answer outcome with no answer or method disclosure."""

    task_id: str
    disposition: str
    correct: bool
    score: int
    code: str | None
    violated_visible_constraints: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScalarOutcomeFeedback:
    """Final scalar reward that discloses neither an answer nor a failed fact."""

    task_id: str
    valid: bool
    exact: bool
    constraint_satisfaction: float


def generate_relational_task(
    seed: int,
    *,
    item_count: int = 5,
    surface_forms: Sequence[str] | None = None,
) -> GeneratedRelationalTask:
    """Generate one replayable, uniquely ordered relational task."""

    _validate_item_count(item_count)
    selected_surface_forms = _validate_surface_forms(surface_forms)
    rng = random.Random(seed)
    ordered_symbols = tuple(rng.sample(_LABELS, item_count))

    relation_pattern = _relation_pattern(item_count, rng)
    display_symbols = _display_order(ordered_symbols, rng)
    constraints = tuple(
        PrecedenceConstraint(ordered_symbols[left], ordered_symbols[right])
        for left, right in relation_pattern
    )
    fact_statements = _render_fact_statements(
        constraints,
        rng,
        selected_surface_forms,
    )
    problem_statement = _render_problem_statement(display_symbols, fact_statements)
    prompt = _render_prompt(problem_statement)
    instance_id = _instance_id(prompt)
    return GeneratedRelationalTask(
        learner=LearnerTask(
            instance_id=instance_id,
            family_id=FAMILY_ID,
            family_version=FAMILY_VERSION,
            symbols=display_symbols,
            constraints=constraints,
            fact_statements=fact_statements,
            problem_statement=problem_statement,
            prompt=prompt,
        ),
        hidden=HiddenOrderSolution(
            instance_id=instance_id,
            ordered_symbols=ordered_symbols,
            generator_seed=seed,
        ),
    )


def make_held_out_variant(
    source: GeneratedRelationalTask,
    *,
    seed: int,
    surface_forms: Sequence[str] | None = None,
) -> GeneratedRelationalTask:
    """Rename symbols and reorder statements while preserving relations."""

    source_order = source.hidden.ordered_symbols
    if source.hidden.instance_id != source.learner.instance_id:
        raise ValueError("source learner and hidden identities do not match")

    item_count = len(source_order)
    selected_surface_forms = _validate_surface_forms(surface_forms)
    rng = random.Random(seed)
    available = tuple(label for label in _LABELS if label not in source_order)
    renamed_order = tuple(rng.sample(available, item_count))

    source_rank = {symbol: rank for rank, symbol in enumerate(source_order)}
    source_pattern = tuple(
        (source_rank[constraint.earlier], source_rank[constraint.later])
        for constraint in source.learner.constraints
    )
    variant_pattern = list(source_pattern)
    rng.shuffle(variant_pattern)
    if tuple(variant_pattern) == source_pattern:
        variant_pattern = variant_pattern[1:] + variant_pattern[:1]

    constraints = tuple(
        PrecedenceConstraint(renamed_order[left], renamed_order[right])
        for left, right in variant_pattern
    )
    display_symbols = _display_order(renamed_order, rng)
    fact_statements = _render_fact_statements(
        constraints,
        rng,
        selected_surface_forms,
    )
    problem_statement = _render_problem_statement(display_symbols, fact_statements)
    prompt = _render_prompt(problem_statement)
    instance_id = _instance_id(prompt)
    return GeneratedRelationalTask(
        learner=LearnerTask(
            instance_id=instance_id,
            family_id=FAMILY_ID,
            family_version=FAMILY_VERSION,
            symbols=display_symbols,
            constraints=constraints,
            fact_statements=fact_statements,
            problem_statement=problem_statement,
            prompt=prompt,
        ),
        hidden=HiddenOrderSolution(
            instance_id=instance_id,
            ordered_symbols=renamed_order,
            generator_seed=seed,
        ),
    )


def verify_final_answer(
    task: LearnerTask,
    answer: str | Sequence[str],
) -> OutcomeFeedback:
    """Check only the final ordering and return outcome-only feedback."""

    submitted = _parse_answer(answer)
    if (
        len(submitted) != len(task.symbols)
        or len(set(submitted)) != len(submitted)
        or set(submitted) != set(task.symbols)
    ):
        return OutcomeFeedback(
            task_id=task.instance_id,
            disposition="INVALID_ATTEMPT",
            correct=False,
            score=0,
            code="INVALID_FINAL_ANSWER",
            violated_visible_constraints=(),
        )

    positions = {symbol: index for index, symbol in enumerate(submitted)}
    violated = tuple(
        index
        for index, constraint in enumerate(task.constraints, start=1)
        if positions[constraint.earlier] >= positions[constraint.later]
    )
    correct = not violated
    return OutcomeFeedback(
        task_id=task.instance_id,
        disposition="VALID_RESULT",
        correct=correct,
        score=int(correct),
        code=None if correct else "ORDER_INCORRECT",
        violated_visible_constraints=violated,
    )


def score_constraint_satisfaction(
    task: LearnerTask,
    answer: str | Sequence[str],
) -> ScalarOutcomeFeedback:
    """Score a complete outcome while withholding solution and relation details."""

    outcome = verify_final_answer(task, answer)
    valid = outcome.disposition == "VALID_RESULT"
    satisfied = (
        len(task.constraints) - len(outcome.violated_visible_constraints)
        if valid
        else 0
    )
    return ScalarOutcomeFeedback(
        task_id=task.instance_id,
        valid=valid,
        exact=outcome.correct,
        constraint_satisfaction=(
            satisfied / len(task.constraints) if task.constraints else 0.0
        ),
    )


def _relation_pattern(item_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    # Adjacent links make the complete order unique without leaking rank through
    # the in/out-degree counts created by redundant forward edges.  The learner
    # must combine the shuffled links; no one-fact or local-degree shortcut can
    # recover the interior order.
    pattern = [(index, index + 1) for index in range(item_count - 1)]
    rng.shuffle(pattern)
    if pattern == [(index, index + 1) for index in range(item_count - 1)]:
        pattern = pattern[1:] + pattern[:1]
    return tuple(pattern)


def _display_order(
    ordered_symbols: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    display = list(ordered_symbols)
    rng.shuffle(display)
    if tuple(display) == ordered_symbols:
        display = display[1:] + display[:1]
    return tuple(display)


def _render_fact_statements(
    constraints: tuple[PrecedenceConstraint, ...],
    rng: random.Random,
    surface_forms: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        rng.choice(surface_forms).format(
            earlier=constraint.earlier,
            later=constraint.later,
        )
        for constraint in constraints
    )


def _render_problem_statement(
    display_symbols: tuple[str, ...],
    fact_statements: tuple[str, ...],
) -> str:
    numbered = "\n".join(
        f"{index}. {statement}"
        for index, statement in enumerate(fact_statements, 1)
    )
    return (
        "Arrange every symbol in one sequence from earliest to latest.\n"
        f"Symbols (display order only): {', '.join(display_symbols)}\n"
        "Constraints:\n"
        f"{numbered}"
    )


def _render_prompt(problem_statement: str) -> str:
    return (
        f"{problem_statement}\n"
        "Return only the final comma-separated symbol sequence."
    )


def _parse_answer(answer: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(answer, str):
        return tuple(part.strip() for part in answer.split(","))
    return tuple(str(part).strip() for part in answer)


def _instance_id(prompt: str) -> str:
    material = f"{FAMILY_ID}@{FAMILY_VERSION}\x00{prompt}".encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _validate_item_count(item_count: int) -> None:
    if not 4 <= item_count <= 8:
        raise ValueError("item_count must be between 4 and 8")


def _validate_surface_forms(
    surface_forms: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = (
        DEFAULT_RELATION_SURFACE_FORMS
        if surface_forms is None
        else tuple(surface_forms)
    )
    if not selected:
        raise ValueError("surface_forms must not be empty")
    for template in selected:
        valid_template = (
            isinstance(template, str)
            and template.count("{earlier}") == 1
            and template.count("{later}") == 1
        )
        if not valid_template:
            raise ValueError(
                "every surface form must contain {earlier} and {later} exactly once"
            )
    return selected
