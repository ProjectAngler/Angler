"""Seeded symbolic permutation tasks with outcome-only verification.

Each task contains several demonstrations of one shared, hidden position
permutation.  Demonstration and query symbols are all public and mutually
disjoint, while the normalized permutation and query target remain in a
separate generator-side object.  This module generates and scores outcomes;
it does not prescribe or implement a method for inducing the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from string import Formatter
from typing import Sequence

FAMILY_ID = "angler.symbolic-rule-induction"
FAMILY_VERSION = "1.0.0"

_SYMBOLS = (
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
    "grove",
    "heath",
    "indigo",
    "juniper",
    "kelp",
    "lotus",
    "moss",
    "nectar",
    "opal",
    "pine",
    "reed",
    "sage",
    "thistle",
    "upland",
    "vale",
    "wren",
    "yucca",
    "zinnia",
    "atlas",
    "beacon",
    "cobalt",
    "drift",
    "echo",
    "fable",
    "glint",
    "haven",
    "islet",
    "jewel",
    "knoll",
    "lyric",
    "meteor",
    "north",
    "orbit",
    "prism",
    "ripple",
    "spruce",
    "terra",
    "velvet",
    "wander",
    "zenith",
)

DEFAULT_GOAL_SURFACE_FORMS = (
    (
        "Infer the one position rearrangement shared by all {demo_count} "
        "examples, then apply it to the new list of {item_count} symbols."
    ),
    (
        "The {demo_count} demonstrations follow one common positional rule. "
        "Use that rule on the {item_count}-symbol query."
    ),
)

DEFAULT_DEMONSTRATION_SURFACE_FORMS = (
    "Example {demo_number}: input [{inputs}] becomes output [{outputs}].",
    "Demonstration {demo_number}: [{inputs}] maps to [{outputs}].",
    "Case {demo_number}: given [{inputs}], the result is [{outputs}].",
)

DEFAULT_QUERY_SURFACE_FORMS = (
    "New input (display order): [{inputs}]",
    "Apply the shared rule to this input: [{inputs}]",
    "Query list, in its given order: [{inputs}]",
)


@dataclass(frozen=True, slots=True)
class SymbolicRuleDemonstration:
    """One learner-visible input/output example."""

    input_symbols: tuple[str, ...]
    output_symbols: tuple[str, ...]
    statement: str


@dataclass(frozen=True, slots=True)
class SymbolicRuleLearnerTask:
    """Complete learner projection, excluding the normalized rule and target."""

    instance_id: str
    family_id: str
    family_version: str
    goal_text: str
    demonstrations: tuple[SymbolicRuleDemonstration, ...]
    query_symbols: tuple[str, ...]
    query_statement: str
    prompt: str


@dataclass(frozen=True, slots=True, repr=False)
class HiddenSymbolicRuleSolution:
    """Sealed generator state that must not enter the learner observation."""

    instance_id: str
    position_permutation: tuple[int, ...]
    target_order: tuple[str, ...]
    generator_seed: int
    surface_seed: int


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedSymbolicRuleTask:
    """Generator-side pairing of public task material and sealed solution."""

    learner: SymbolicRuleLearnerTask
    hidden: HiddenSymbolicRuleSolution


@dataclass(frozen=True, slots=True)
class SymbolicRuleFeedback:
    """Bounded scalar outcome with no target or mistake identity."""

    valid: bool
    exact: bool
    pairwise_order_agreement: float


def generate_symbolic_rule_task(
    seed: int,
    *,
    item_count: int = 5,
    demonstration_count: int = 3,
    position_permutation: Sequence[int] | None = None,
    public_symbols: Sequence[str] | None = None,
    surface_seed: int | None = None,
    goal_surface_forms: Sequence[str] | None = None,
    demonstration_surface_forms: Sequence[str] | None = None,
    query_surface_forms: Sequence[str] | None = None,
) -> GeneratedSymbolicRuleTask:
    """Generate one replayable shared-permutation induction task.

    Structural sampling and surface rendering use domain-separated random
    streams.  Callers can therefore vary wording through ``surface_seed`` or
    injected templates without changing symbols, demonstrations, or solution.
    An evaluator may supply a sealed position permutation to generate a stream
    of fresh-symbol instances governed by one persistent procedure.  The
    supplied value remains generator-side and is never added to the learner
    projection.
    """

    _validate_item_count(item_count)
    _validate_demonstration_count(demonstration_count)
    goal_forms = _validate_surface_forms(
        goal_surface_forms,
        DEFAULT_GOAL_SURFACE_FORMS,
        required_fields=("demo_count", "item_count"),
        label="goal_surface_forms",
    )
    demo_forms = _validate_surface_forms(
        demonstration_surface_forms,
        DEFAULT_DEMONSTRATION_SURFACE_FORMS,
        required_fields=("demo_number", "inputs", "outputs"),
        label="demonstration_surface_forms",
    )
    query_forms = _validate_surface_forms(
        query_surface_forms,
        DEFAULT_QUERY_SURFACE_FORMS,
        required_fields=("inputs",),
        label="query_surface_forms",
    )

    public_symbol_count = (demonstration_count + 1) * item_count
    structure_rng = random.Random(seed)
    sampled_symbols = (
        structure_rng.sample(_SYMBOLS, public_symbol_count)
        if public_symbols is None
        else _validate_public_symbols(public_symbols, public_symbol_count)
    )
    selected_permutation = (
        _sample_non_identity_permutation(item_count, structure_rng)
        if position_permutation is None
        else _validate_position_permutation(position_permutation, item_count)
    )

    effective_surface_seed = seed if surface_seed is None else surface_seed
    goal_rng = _domain_rng(effective_surface_seed, "goal")
    demo_rng = _domain_rng(effective_surface_seed, "demonstrations")
    query_rng = _domain_rng(effective_surface_seed, "query")

    demonstrations: list[SymbolicRuleDemonstration] = []
    for index in range(demonstration_count):
        start = index * item_count
        input_symbols = tuple(sampled_symbols[start : start + item_count])
        output_symbols = _apply_position_permutation(
            input_symbols,
            selected_permutation,
        )
        statement = demo_rng.choice(demo_forms).format(
            demo_number=index + 1,
            inputs=", ".join(input_symbols),
            outputs=", ".join(output_symbols),
        )
        demonstrations.append(
            SymbolicRuleDemonstration(
                input_symbols=input_symbols,
                output_symbols=output_symbols,
                statement=statement,
            )
        )

    query_start = demonstration_count * item_count
    query_symbols = tuple(sampled_symbols[query_start:])
    target_order = _apply_position_permutation(
        query_symbols,
        selected_permutation,
    )
    goal_text = goal_rng.choice(goal_forms).format(
        demo_count=demonstration_count,
        item_count=item_count,
    )
    query_statement = query_rng.choice(query_forms).format(
        inputs=", ".join(query_symbols),
    )
    prompt = _render_prompt(
        goal_text,
        tuple(demonstrations),
        query_statement,
    )
    instance_id = _instance_id(
        prompt,
        seed=seed,
        surface_seed=effective_surface_seed,
    )

    return GeneratedSymbolicRuleTask(
        learner=SymbolicRuleLearnerTask(
            instance_id=instance_id,
            family_id=FAMILY_ID,
            family_version=FAMILY_VERSION,
            goal_text=goal_text,
            demonstrations=tuple(demonstrations),
            query_symbols=query_symbols,
            query_statement=query_statement,
            prompt=prompt,
        ),
        hidden=HiddenSymbolicRuleSolution(
            instance_id=instance_id,
            position_permutation=selected_permutation,
            target_order=target_order,
            generator_seed=seed,
            surface_seed=effective_surface_seed,
        ),
    )


def verify_symbolic_rule_answer(
    task: SymbolicRuleLearnerTask,
    solution: HiddenSymbolicRuleSolution,
    answer: str | Sequence[str],
) -> SymbolicRuleFeedback:
    """Score a proposed full ordering without disclosing answer details."""

    _validate_pairing(task, solution)
    submitted = _parse_answer(answer)
    if (
        submitted is None
        or len(submitted) != len(task.query_symbols)
        or len(set(submitted)) != len(submitted)
        or set(submitted) != set(task.query_symbols)
    ):
        return SymbolicRuleFeedback(
            valid=False,
            exact=False,
            pairwise_order_agreement=0.0,
        )

    target = solution.target_order
    submitted_position = {
        symbol: index for index, symbol in enumerate(submitted)
    }
    pair_count = len(target) * (len(target) - 1) // 2
    agreeing_pairs = sum(
        submitted_position[target[left]] < submitted_position[target[right]]
        for left in range(len(target))
        for right in range(left + 1, len(target))
    )
    agreement = agreeing_pairs / pair_count
    exact = submitted == target
    return SymbolicRuleFeedback(
        valid=True,
        exact=exact,
        pairwise_order_agreement=agreement,
    )


def _apply_position_permutation(
    symbols: tuple[str, ...],
    permutation: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(symbols[position] for position in permutation)


def _sample_non_identity_permutation(
    item_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    identity = tuple(range(item_count))
    permutation = list(identity)
    rng.shuffle(permutation)
    if tuple(permutation) == identity:
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return tuple(permutation)


def _validate_position_permutation(
    supplied: Sequence[int],
    item_count: int,
) -> tuple[int, ...]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("position_permutation must be an integer sequence")
    permutation = tuple(supplied)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in permutation):
        raise TypeError("position_permutation must contain only integers")
    if sorted(permutation) != list(range(item_count)):
        raise ValueError("position_permutation must cover every item position once")
    return permutation


def _validate_public_symbols(
    supplied: Sequence[str],
    required_count: int,
) -> tuple[str, ...]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("public_symbols must be a sequence of text tokens")
    symbols = tuple(supplied)
    if len(symbols) != required_count:
        raise ValueError(
            f"public_symbols must contain exactly {required_count} tokens"
        )
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
        raise ValueError("public_symbols must contain non-empty text tokens")
    if any("," in symbol or "\n" in symbol or "\r" in symbol for symbol in symbols):
        raise ValueError("public_symbols cannot contain commas or line breaks")
    if len(set(symbols)) != len(symbols):
        raise ValueError("public_symbols must be unique")
    return symbols


def _domain_rng(seed: int, domain: str) -> random.Random:
    material = f"{FAMILY_ID}@{FAMILY_VERSION}\x00{seed}\x00{domain}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(derived_seed)


def _render_prompt(
    goal_text: str,
    demonstrations: tuple[SymbolicRuleDemonstration, ...],
    query_statement: str,
) -> str:
    rendered_demonstrations = "\n".join(
        demonstration.statement for demonstration in demonstrations
    )
    return (
        f"{goal_text}\n"
        "Demonstrations:\n"
        f"{rendered_demonstrations}\n"
        f"{query_statement}\n"
        "Return only the complete comma-separated output ordering."
    )


def _parse_answer(answer: str | Sequence[str]) -> tuple[str, ...] | None:
    if isinstance(answer, str):
        submitted = tuple(part.strip() for part in answer.split(","))
    else:
        if not isinstance(answer, Sequence) or any(
            not isinstance(part, str) for part in answer
        ):
            return None
        submitted = tuple(part.strip() for part in answer)
    if any(not part for part in submitted):
        return None
    return submitted


def _instance_id(prompt: str, *, seed: int, surface_seed: int) -> str:
    material = (
        f"{FAMILY_ID}@{FAMILY_VERSION}\x00{seed}\x00{surface_seed}\x00{prompt}"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _validate_pairing(
    task: SymbolicRuleLearnerTask,
    solution: HiddenSymbolicRuleSolution,
) -> None:
    if task.instance_id != solution.instance_id:
        raise ValueError("task and hidden solution identities do not match")
    if (
        len(solution.target_order) != len(task.query_symbols)
        or set(solution.target_order) != set(task.query_symbols)
    ):
        raise ValueError("hidden target is not a permutation of the query symbols")
    item_count = len(task.query_symbols)
    if sorted(solution.position_permutation) != list(range(item_count)):
        raise ValueError("hidden position permutation is malformed")
    expected_target = _apply_position_permutation(
        task.query_symbols,
        solution.position_permutation,
    )
    if expected_target != solution.target_order:
        raise ValueError("hidden target and position permutation disagree")


def _validate_item_count(item_count: int) -> None:
    if not 4 <= item_count <= 7:
        raise ValueError("item_count must be between 4 and 7")


def _validate_demonstration_count(demonstration_count: int) -> None:
    if not 2 <= demonstration_count <= 4:
        raise ValueError("demonstration_count must be between 2 and 4")


def _validate_surface_forms(
    supplied: Sequence[str] | None,
    defaults: tuple[str, ...],
    *,
    required_fields: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    selected = defaults if supplied is None else tuple(supplied)
    if not selected:
        raise ValueError(f"{label} must not be empty")

    for template in selected:
        if not isinstance(template, str):
            raise ValueError(f"every {label} entry must be text")
        try:
            parsed_fields = tuple(
                field_name
                for _, field_name, _, _ in Formatter().parse(template)
                if field_name is not None
            )
        except ValueError as error:
            raise ValueError(f"{label} contains an invalid format string") from error
        if sorted(parsed_fields) != sorted(required_fields):
            required = ", ".join(f"{{{field}}}" for field in required_fields)
            raise ValueError(
                f"every {label} entry must contain exactly once: {required}"
            )
    return selected


__all__ = [
    "DEFAULT_DEMONSTRATION_SURFACE_FORMS",
    "DEFAULT_GOAL_SURFACE_FORMS",
    "DEFAULT_QUERY_SURFACE_FORMS",
    "FAMILY_ID",
    "FAMILY_VERSION",
    "GeneratedSymbolicRuleTask",
    "HiddenSymbolicRuleSolution",
    "SymbolicRuleDemonstration",
    "SymbolicRuleFeedback",
    "SymbolicRuleLearnerTask",
    "generate_symbolic_rule_task",
    "verify_symbolic_rule_answer",
]
