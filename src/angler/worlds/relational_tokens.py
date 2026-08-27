"""A relational token-sliding world with independently owned semantics.

A token can move only from its current position into the sole unoccupied
adjacent position.  Emptiness is inferred from declared positions and token
placements rather than represented as a mutable fact, so relocation has the
same two-record add/delete boundary as the other domains.  This module exposes
no search, route, distance, path, or solution-construction operation.
"""

from __future__ import annotations

from collections.abc import Sequence

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    Parameter,
    Record,
    State,
    Transition,
)


NAMESPACE = "angler.relational.tokens"
TOKEN_IN = f"{NAMESPACE}.token_in"
POSITION_ADJACENT = f"{NAMESPACE}.position_adjacent"
MOVE_TOKEN = ActionSchema(
    name=f"{NAMESPACE}.slide_token",
    parameters=(
        Parameter("token", f"{NAMESPACE}.token"),
        Parameter("source", f"{NAMESPACE}.position"),
        Parameter("destination", f"{NAMESPACE}.position"),
    ),
    description="Move one token from its position into an adjacent empty position.",
)

_APPLIED = f"{NAMESPACE}.slid"
_BLOCKED = f"{NAMESPACE}.blocked"
_POSITION_PREFIX = "position_"


def make_token_state(slots: Sequence[str | None]) -> State:
    """Create a line of positions containing unique tokens and one empty slot."""

    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes, bytearray)):
        raise TypeError("token slots must be a finite sequence")
    contents = tuple(slots)
    if len(contents) < 3:
        raise ValueError("the token world requires at least three positions")
    if sum(item is None for item in contents) != 1:
        raise ValueError("the token world requires exactly one empty position")
    tokens = tuple(item for item in contents if item is not None)
    if any(type(token) is not str for token in tokens):
        raise TypeError("tokens must be strings or one None empty marker")
    if len(set(tokens)) != len(tokens):
        raise ValueError("tokens must be unique")

    records: list[Record] = []
    for index, token in enumerate(contents):
        position = _position(index)
        if token is not None:
            records.append(Record(TOKEN_IN, (token, position)))
    for index in range(len(contents) - 1):
        left = _position(index)
        right = _position(index + 1)
        records.append(Record(POSITION_ADJACENT, (left, right)))
        records.append(Record(POSITION_ADJACENT, (right, left)))
    return State.from_records(NAMESPACE, records)


def make_token_goal(slots: Sequence[str | None]) -> Goal:
    """Describe an exact target state without describing moves to reach it."""

    target = make_token_state(slots)
    return Goal(namespace=NAMESPACE, required=target.records, exact=True)


def execute_token_action(state: State, action: GroundAction) -> Transition:
    """Execute one declared token move or record an unchanged blocked attempt."""

    token_locations, empty, adjacency = _decode_token_state(state)
    if not isinstance(action, GroundAction) or action.schema != MOVE_TOKEN:
        raise ValueError("the token world accepts only MOVE_TOKEN actions")
    token, source, destination = action.arguments
    applicable = (
        source != destination
        and token_locations.get(token) == source
        and destination == empty
        and (source, destination) in adjacency
    )
    if not applicable:
        return Transition(state, action, state, False, _BLOCKED)

    records = set(state.records)
    records.remove(Record(TOKEN_IN, (token, source)))
    records.add(Record(TOKEN_IN, (token, destination)))
    successor = State.from_records(NAMESPACE, records)
    _decode_token_state(successor)
    return Transition(state, action, successor, True, _APPLIED)


def verify_token_goal(state: State, goal: Goal) -> bool:
    """Judge observable facts only; never derive a move sequence."""

    _decode_token_state(state)
    if not isinstance(goal, Goal) or goal.namespace != NAMESPACE:
        raise ValueError("token goal belongs to another domain")
    allowed = {TOKEN_IN, POSITION_ADJACENT}
    if any(record.predicate not in allowed for record in goal.required + goal.forbidden):
        raise ValueError("token goal contains an unknown predicate")
    records = set(state.records)
    if goal.exact:
        return state.records == goal.required
    return set(goal.required) <= records and not (set(goal.forbidden) & records)


def _position(index: int) -> str:
    return f"{_POSITION_PREFIX}{index}"


def _parse_position(value: str) -> int:
    if not value.startswith(_POSITION_PREFIX):
        raise ValueError("token position has an invalid name")
    suffix = value.removeprefix(_POSITION_PREFIX)
    if not suffix.isascii() or not suffix.isdigit():
        raise ValueError("token position has an invalid index")
    index = int(suffix)
    if suffix != str(index):
        raise ValueError("token position index is not canonical")
    return index


def _decode_token_state(
    state: State,
) -> tuple[dict[str, str], str, frozenset[tuple[str, str]]]:
    if not isinstance(state, State) or state.namespace != NAMESPACE:
        raise ValueError("token state belongs to another domain")
    token_locations: dict[str, str] = {}
    occupied: set[str] = set()
    adjacency: set[tuple[str, str]] = set()
    for record in state.records:
        if record.predicate == TOKEN_IN:
            if len(record.arguments) != 2:
                raise ValueError("token records require token and position")
            token, position = record.arguments
            _parse_position(position)
            if token in token_locations or position in occupied:
                raise ValueError("token placements must be one-to-one")
            token_locations[token] = position
            occupied.add(position)
        elif record.predicate == POSITION_ADJACENT:
            if len(record.arguments) != 2 or record.arguments[0] == record.arguments[1]:
                raise ValueError("adjacency records require two distinct positions")
            _parse_position(record.arguments[0])
            _parse_position(record.arguments[1])
            adjacency.add(record.arguments)
        else:
            raise ValueError("token state contains an unknown predicate")
    positions = {position for pair in adjacency for position in pair}
    if len(positions) < 3:
        raise ValueError("token state requires at least three positions")
    empty_positions = positions - occupied
    if len(empty_positions) != 1:
        raise ValueError("token state requires exactly one unoccupied position")
    indices = {_parse_position(position) for position in positions}
    if indices != set(range(len(positions))):
        raise ValueError("token positions must be contiguous from zero")
    expected_adjacency = {
        pair
        for index in range(len(positions) - 1)
        for pair in (
            (_position(index), _position(index + 1)),
            (_position(index + 1), _position(index)),
        )
    }
    if adjacency != expected_adjacency:
        raise ValueError("token state must declare the complete line adjacency")
    return token_locations, next(iter(empty_positions)), frozenset(adjacency)


__all__ = [
    "MOVE_TOKEN",
    "NAMESPACE",
    "POSITION_ADJACENT",
    "TOKEN_IN",
    "execute_token_action",
    "make_token_goal",
    "make_token_state",
    "verify_token_goal",
]
