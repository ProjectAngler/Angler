"""An abstract relational file-routing world with independent semantics.

Files move across declared directed directory links only when the destination
does not already contain the same file name.  The world never touches the host
filesystem and exposes no route, path, or solution operation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    Parameter,
    Record,
    State,
    Transition,
)


NAMESPACE = "angler.relational.files"
FILE_AT = f"{NAMESPACE}.file_at"
DIRECTORY_LINK = f"{NAMESPACE}.directory_link"
RELOCATE_FILE = ActionSchema(
    name=f"{NAMESPACE}.relocate_file",
    parameters=(
        Parameter("file", f"{NAMESPACE}.file"),
        Parameter("source", f"{NAMESPACE}.directory"),
        Parameter("destination", f"{NAMESPACE}.directory"),
    ),
    description="Move one file across a link when its name is free at the destination.",
)

_APPLIED = f"{NAMESPACE}.relocated"
_BLOCKED = f"{NAMESPACE}.blocked"


def make_file_state(
    locations: Mapping[str, str] | Iterable[tuple[str, str]],
    links: Iterable[tuple[str, str]],
) -> State:
    """Create an abstract state; repeated names may exist in different directories."""

    placements = _normalize_placements(locations, label="file locations")
    link_items = tuple(links)
    for link in link_items:
        if type(link) is not tuple or len(link) != 2:
            raise TypeError("each directory link must be a two-item tuple")
        if any(type(item) is not str for item in link):
            raise TypeError("directory link endpoints must be strings")
        if link[0] == link[1]:
            raise ValueError("directory links cannot be self-links")
    if len(set(link_items)) != len(link_items):
        raise ValueError("directory links must be unique")

    records = [Record(FILE_AT, item) for item in placements]
    records.extend(Record(DIRECTORY_LINK, link) for link in link_items)
    state = State.from_records(NAMESPACE, records)
    _decode_file_state(state)
    return state


def make_file_goal(
    locations: Mapping[str, str] | Iterable[tuple[str, str]],
) -> Goal:
    """Require file-name placements while leaving directory links unchanged."""

    placements = _normalize_placements(locations, label="file goal locations")
    return Goal.from_records(
        NAMESPACE,
        (Record(FILE_AT, item) for item in placements),
    )


def execute_file_action(state: State, action: GroundAction) -> Transition:
    """Execute one abstract file relocation or record a blocked attempt."""

    placements, links = _decode_file_state(state)
    if not isinstance(action, GroundAction) or action.schema != RELOCATE_FILE:
        raise ValueError("the file world accepts only RELOCATE_FILE actions")
    file, source, destination = action.arguments
    applicable = (
        source != destination
        and (file, source) in placements
        and (source, destination) in links
        and (file, destination) not in placements
    )
    if not applicable:
        return Transition(state, action, state, False, _BLOCKED)

    records = set(state.records)
    records.remove(Record(FILE_AT, (file, source)))
    records.add(Record(FILE_AT, (file, destination)))
    successor = State.from_records(NAMESPACE, records)
    _decode_file_state(successor)
    return Transition(state, action, successor, True, _APPLIED)


def verify_file_goal(state: State, goal: Goal) -> bool:
    """Judge file placements without deriving a route."""

    _decode_file_state(state)
    if not isinstance(goal, Goal) or goal.namespace != NAMESPACE:
        raise ValueError("file goal belongs to another domain")
    allowed = {FILE_AT, DIRECTORY_LINK} if goal.exact else {FILE_AT}
    if any(
        record.predicate not in allowed
        for record in goal.required + goal.forbidden
    ):
        raise ValueError("file goal contains an unsupported predicate")
    records = set(state.records)
    if goal.exact:
        return state.records == goal.required
    return set(goal.required) <= records and not (set(goal.forbidden) & records)


def _normalize_placements(
    locations: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    if isinstance(locations, Mapping):
        items = tuple(locations.items())
    else:
        items = tuple(locations)
    if not items:
        raise ValueError(f"{label} must be non-empty")
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{label} entries must be two-item tuples")
        if any(type(value) is not str for value in item):
            raise TypeError(f"{label} names must be strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} cannot contain duplicate placements")
    return items


def _decode_file_state(
    state: State,
) -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    if not isinstance(state, State) or state.namespace != NAMESPACE:
        raise ValueError("file state belongs to another domain")
    placements: set[tuple[str, str]] = set()
    links: set[tuple[str, str]] = set()
    for record in state.records:
        if len(record.arguments) != 2:
            raise ValueError("file state records must have two arguments")
        left, right = record.arguments
        if record.predicate == FILE_AT:
            placements.add((left, right))
        elif record.predicate == DIRECTORY_LINK:
            if left == right:
                raise ValueError("directory links cannot be self-links")
            links.add((left, right))
        else:
            raise ValueError("file state contains an unknown predicate")
    if not placements:
        raise ValueError("file state must contain at least one file")
    return frozenset(placements), frozenset(links)


__all__ = [
    "DIRECTORY_LINK",
    "FILE_AT",
    "NAMESPACE",
    "RELOCATE_FILE",
    "execute_file_action",
    "make_file_goal",
    "make_file_state",
    "verify_file_goal",
]
