"""A relational item/container world with independent transition semantics.

An item can move between containers only while the destination has declared
spare capacity.  This module executes and verifies states but exposes no
planning, path, route, or solution operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    Parameter,
    Record,
    State,
    Transition,
)


NAMESPACE = "angler.relational.boxes"
ITEM_IN = f"{NAMESPACE}.item_in"
CONTAINER_CAPACITY = f"{NAMESPACE}.container_capacity"
TRANSFER_ITEM = ActionSchema(
    name=f"{NAMESPACE}.transfer_item",
    parameters=(
        Parameter("item", f"{NAMESPACE}.item"),
        Parameter("source", f"{NAMESPACE}.container"),
        Parameter("destination", f"{NAMESPACE}.container"),
    ),
    description="Move one item between containers when destination capacity permits.",
)

_APPLIED = f"{NAMESPACE}.transferred"
_BLOCKED = f"{NAMESPACE}.blocked"
_CAPACITY_PREFIX = "limit_"


def make_box_state(
    contents: Mapping[str, Sequence[str]],
    capacities: Mapping[str, int] | None = None,
) -> State:
    """Create a state from container contents and positive capacities."""

    if not isinstance(contents, Mapping) or len(contents) < 2:
        raise ValueError("the box world requires at least two containers")
    normalized: dict[str, tuple[str, ...]] = {}
    all_items: set[str] = set()
    for container, items in contents.items():
        if type(container) is not str:
            raise TypeError("container names must be strings")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise TypeError("container contents must be finite item sequences")
        sequence = tuple(items)
        if any(type(item) is not str for item in sequence):
            raise TypeError("item names must be strings")
        if len(set(sequence)) != len(sequence) or all_items.intersection(sequence):
            raise ValueError("each item must occur exactly once")
        all_items.update(sequence)
        normalized[container] = sequence
    if not all_items:
        raise ValueError("the box world requires at least one item")

    if capacities is None:
        default = max(1, len(all_items))
        normalized_capacities = {container: default for container in normalized}
    else:
        if not isinstance(capacities, Mapping) or set(capacities) != set(normalized):
            raise ValueError("capacities must name every container exactly once")
        normalized_capacities = dict(capacities)
    for container, capacity in normalized_capacities.items():
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("container capacities must be positive integers")
        if len(normalized[container]) > capacity:
            raise ValueError("container contents exceed declared capacity")

    records: list[Record] = []
    for container, capacity in normalized_capacities.items():
        records.append(
            Record(CONTAINER_CAPACITY, (container, _capacity_name(capacity)))
        )
    for container, items in normalized.items():
        records.extend(Record(ITEM_IN, (item, container)) for item in items)
    return State.from_records(NAMESPACE, records)


def make_box_goal(
    contents: Mapping[str, Sequence[str]],
    capacities: Mapping[str, int] | None = None,
) -> Goal:
    """Describe an exact target allocation without exposing transfers."""

    target = make_box_state(contents, capacities)
    return Goal(namespace=NAMESPACE, required=target.records, exact=True)


def execute_box_action(state: State, action: GroundAction) -> Transition:
    """Execute one item transfer or record an unchanged blocked attempt."""

    placements, capacities = _decode_box_state(state)
    if not isinstance(action, GroundAction) or action.schema != TRANSFER_ITEM:
        raise ValueError("the box world accepts only TRANSFER_ITEM actions")
    item, source, destination = action.arguments
    destination_count = sum(place == destination for place in placements.values())
    applicable = (
        source != destination
        and placements.get(item) == source
        and destination in capacities
        and destination_count < capacities[destination]
    )
    if not applicable:
        return Transition(state, action, state, False, _BLOCKED)

    records = set(state.records)
    records.remove(Record(ITEM_IN, (item, source)))
    records.add(Record(ITEM_IN, (item, destination)))
    successor = State.from_records(NAMESPACE, records)
    _decode_box_state(successor)
    return Transition(state, action, successor, True, _APPLIED)


def verify_box_goal(state: State, goal: Goal) -> bool:
    """Judge item allocation and capacity facts without deriving moves."""

    _decode_box_state(state)
    if not isinstance(goal, Goal) or goal.namespace != NAMESPACE:
        raise ValueError("box goal belongs to another domain")
    allowed = {ITEM_IN, CONTAINER_CAPACITY}
    if any(record.predicate not in allowed for record in goal.required + goal.forbidden):
        raise ValueError("box goal contains an unknown predicate")
    records = set(state.records)
    if goal.exact:
        return state.records == goal.required
    return set(goal.required) <= records and not (set(goal.forbidden) & records)


def _capacity_name(value: int) -> str:
    return f"{_CAPACITY_PREFIX}{value}"


def _parse_capacity(value: str) -> int:
    if not value.startswith(_CAPACITY_PREFIX):
        raise ValueError("container capacity has an invalid name")
    suffix = value.removeprefix(_CAPACITY_PREFIX)
    if not suffix.isascii() or not suffix.isdigit():
        raise ValueError("container capacity has an invalid value")
    capacity = int(suffix)
    if capacity <= 0 or suffix != str(capacity):
        raise ValueError("container capacity must be a canonical positive integer")
    return capacity


def _decode_box_state(state: State) -> tuple[dict[str, str], dict[str, int]]:
    if not isinstance(state, State) or state.namespace != NAMESPACE:
        raise ValueError("box state belongs to another domain")
    placements: dict[str, str] = {}
    capacities: dict[str, int] = {}
    for record in state.records:
        if record.predicate == ITEM_IN:
            if len(record.arguments) != 2:
                raise ValueError("item records require item and container")
            item, container = record.arguments
            if item in placements:
                raise ValueError("an item cannot occupy multiple containers")
            placements[item] = container
        elif record.predicate == CONTAINER_CAPACITY:
            if len(record.arguments) != 2:
                raise ValueError("capacity records require container and limit")
            container, encoded = record.arguments
            if container in capacities:
                raise ValueError("a container cannot have multiple capacities")
            capacities[container] = _parse_capacity(encoded)
        else:
            raise ValueError("box state contains an unknown predicate")
    if len(capacities) < 2 or not placements:
        raise ValueError("box state requires two containers and one item")
    counts = {container: 0 for container in capacities}
    for container in placements.values():
        if container not in capacities:
            raise ValueError("item references an undeclared container")
        counts[container] += 1
    if any(counts[name] > capacity for name, capacity in capacities.items()):
        raise ValueError("box state exceeds a container capacity")
    return placements, capacities


__all__ = [
    "CONTAINER_CAPACITY",
    "ITEM_IN",
    "NAMESPACE",
    "TRANSFER_ITEM",
    "execute_box_action",
    "make_box_goal",
    "make_box_state",
    "verify_box_goal",
]
