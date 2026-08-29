"""Evaluator-owned streams for small opaque reversible state machines.

The learner receives typed states, zero-argument action schemas, raw public
transition traces, an origin, an exact goal, and a bounded procedure budget.
The evaluator alone retains the transition table.  It can commit and judge a
learner-proposed action sequence, but deliberately exposes no planning,
distance, target-procedure, next-action, or diagnostic API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import itertools
import json
import random
import re
from typing import Literal

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    Record,
    State,
    Trace,
    Transition,
)


GlyphMachinePartition = Literal["train", "development", "final"]
GlyphMachineControlArm = Literal["correct", "no_trace", "wrong_trace"]
GlyphMachineMechanism = tuple[int, tuple[tuple[int, ...], ...]]

_NAMESPACE = "angler.glyph_machine"
_STATE_PREDICATE = f"{_NAMESPACE}.at"
_APPLIED_OUTCOME = f"{_NAMESPACE}.applied"
_NO_CHANGE_OUTCOME = f"{_NAMESPACE}.no_change"
_PARTITION_SIZES = {"train": 64, "development": 16, "final": 16}
_SEALED_COUNT = 20
_EXPECTED_SEMANTIC_CLASS_COUNT = 116
_MECHANISM_ORDER_DOMAIN = b"project-angler.glyph-machine.semantic-order.v1\x00"
_MECHANISM_COMMITMENT_DOMAIN = (
    b"project-angler.glyph-machine.mechanism-commitment.v1\x00"
)
_PUBLIC_DIGEST_DOMAIN = b"project-angler.glyph-machine.public-task.v1\x00"
_OPAQUE_SURFACE_DOMAIN = b"project-angler.glyph-machine.opaque-surface.v1\x00"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PublicGlyphMachineTask:
    """Complete learner view with no instance or mechanism identity."""

    states: tuple[State, ...]
    actions: tuple[ActionSchema, ...]
    observations: tuple[Trace, ...]
    origin: State
    goal: Goal
    max_steps: int

    def __post_init__(self) -> None:
        _validate_public_task(self)

    def to_canonical(self) -> dict[str, object]:
        return {
            "actions": [_action_payload(action) for action in self.actions],
            "goal": _goal_payload(self.goal),
            "max_steps": self.max_steps,
            "observations": [
                _trace_payload(observation) for observation in self.observations
            ],
            "origin": _state_payload(self.origin),
            "states": [_state_payload(state) for state in self.states],
        }


@dataclass(frozen=True, slots=True)
class CommittedGlyphProcedure:
    """Immutable typed procedure bound to one public task projection."""

    public_digest: str
    actions: tuple[GroundAction, ...]
    stopped: bool

    def __post_init__(self) -> None:
        _require_digest(self.public_digest, "procedure public_digest")
        if type(self.actions) is not tuple or any(
            not isinstance(action, GroundAction) for action in self.actions
        ):
            raise TypeError("procedure actions must be an immutable GroundAction tuple")
        if type(self.stopped) is not bool:
            raise TypeError("procedure stopped flag must be bool")


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenGlyphMachineSolution:
    """Evaluator-only transition physics aligned to opaque public values."""

    public_digest: str
    state_digests: tuple[str, ...]
    action_digests: tuple[str, ...]
    transition_rows: tuple[tuple[int, ...], ...]
    mechanism_commitment: str
    mechanism_partition: GlyphMachinePartition

    def __post_init__(self) -> None:
        _validate_hidden_solution(self)


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedGlyphMachineTask:
    """Evaluator pairing; learner code should receive only ``learner``."""

    learner: PublicGlyphMachineTask
    hidden: _HiddenGlyphMachineSolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class GlyphMachineTraceStream:
    """One acquired machine followed by observation-free query tasks."""

    supports: tuple[GeneratedGlyphMachineTask, ...]
    queries: tuple[GeneratedGlyphMachineTask, ...]
    mechanism_commitment: str
    mechanism_partition: GlyphMachinePartition
    control_arm: GlyphMachineControlArm = "correct"

    def __post_init__(self) -> None:
        _validate_partition(self.mechanism_partition)
        if self.control_arm not in ("correct", "no_trace", "wrong_trace"):
            raise ValueError("stream control arm is invalid")
        _require_digest(self.mechanism_commitment, "mechanism_commitment")
        if type(self.supports) is not tuple or not self.supports:
            raise ValueError("glyph-machine stream requires public supports")
        if type(self.queries) is not tuple or not self.queries:
            raise ValueError("glyph-machine stream requires public queries")
        pairs = (*self.supports, *self.queries)
        if any(not isinstance(pair, GeneratedGlyphMachineTask) for pair in pairs):
            raise TypeError("glyph-machine streams contain generated task pairs")
        if self.control_arm == "no_trace":
            if any(pair.learner.observations for pair in self.supports):
                raise ValueError("no-trace supports cannot expose observations")
        elif any(not pair.learner.observations for pair in self.supports):
            raise ValueError("trace-bearing supports must expose observations")
        if any(pair.learner.observations for pair in self.queries):
            raise ValueError("query tasks must not replay transition observations")
        if any(
            pair.hidden.mechanism_commitment != self.mechanism_commitment
            or pair.hidden.mechanism_partition != self.mechanism_partition
            for pair in pairs
        ):
            raise ValueError("stream tasks differ from the declared mechanism")
        if self.mechanism_commitment not in glyph_machine_mechanism_partition(
            self.mechanism_partition
        ):
            raise ValueError("stream mechanism is outside its declared partition")
        public_digests = tuple(_public_digest(pair.learner) for pair in pairs)
        if len(set(public_digests)) != len(public_digests):
            raise ValueError("glyph-machine public encounters must be unique")
        state_sets = {frozenset(state.digest for state in pair.learner.states) for pair in pairs}
        action_sets = {
            frozenset(action.digest for action in pair.learner.actions)
            for pair in pairs
        }
        if len(state_sets) != 1 or len(action_sets) != 1:
            raise ValueError("one stream must retain one opaque state/action vocabulary")


def glyph_machine_mechanism_partition(
    partition: GlyphMachinePartition,
) -> tuple[str, ...]:
    """Return only commitments for one fixed semantic partition.

    Raw transition tables remain evaluator-private.  The 116 mechanically
    enumerated equivalence classes are canonical under simultaneous state
    renaming and action reordering.  Ninety-six are opened as 64/16/16 and
    twenty remain outside this API.
    """

    _validate_partition(partition)
    return tuple(
        _mechanism_commitment(mechanism)
        for mechanism in _semantic_partition(partition)
    )


def make_glyph_machine_trace_stream(
    seed: int,
    *,
    surface_seed: int | None = None,
    supports: int = 2,
    queries: int = 2,
    observations_per_support: int = 2,
    maximum_steps: int = 4,
    mechanism_commitment: str | None = None,
    mechanism_partition: GlyphMachinePartition = "train",
) -> GlyphMachineTraceStream:
    """Create one replayable public acquisition/query stream."""

    _validate_seed(seed, "seed")
    if surface_seed is None:
        surface_seed = _domain_seed(seed, "default-surface", 0, 0)
    _validate_seed(surface_seed, "surface_seed")
    _validate_positive_count(supports, "supports")
    _validate_positive_count(queries, "queries")
    _validate_positive_count(observations_per_support, "observations_per_support")
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or not 1 <= maximum_steps <= 4
    ):
        raise ValueError("maximum_steps must be an integer from one through four")
    _validate_partition(mechanism_partition)

    mechanisms = _semantic_partition(mechanism_partition)
    by_commitment = {
        _mechanism_commitment(mechanism): mechanism for mechanism in mechanisms
    }
    if mechanism_commitment is None:
        mechanism = mechanisms[
            _domain_seed(seed, "mechanism-selection", 0, 0) % len(mechanisms)
        ]
        commitment = _mechanism_commitment(mechanism)
    else:
        _require_digest(mechanism_commitment, "mechanism_commitment")
        try:
            mechanism = by_commitment[mechanism_commitment]
        except KeyError as error:
            raise ValueError(
                "mechanism_commitment is outside the declared partition"
            ) from error
        commitment = mechanism_commitment

    (
        states_by_index,
        actions_by_index,
        presented_states,
        presented_actions,
    ) = _surface_values(surface_seed, commitment, mechanism)
    # Keep public origin/goal/budget encounters unique even after a no-trace
    # control removes support observations.  This prevents an exact support
    # task from being replayed as a nominally held-out query.
    used_public_cases: set[tuple[int, int, int]] = set()

    support_pairs = [
        _make_encounter(
            seed,
            scope="support",
            index=index,
            mechanism=mechanism,
            commitment=commitment,
            partition=mechanism_partition,
            states_by_index=states_by_index,
            actions_by_index=actions_by_index,
            presented_states=presented_states,
            presented_actions=presented_actions,
            observations=_make_observations(
                seed,
                support_index=index,
                count=observations_per_support,
                transition_rows=mechanism[1],
                states_by_index=states_by_index,
                actions_by_index=actions_by_index,
            ),
            maximum_steps=maximum_steps,
            used_cases=used_public_cases,
        )
        for index in range(supports)
    ]
    query_pairs = [
        _make_encounter(
            seed,
            scope="query",
            index=index,
            mechanism=mechanism,
            commitment=commitment,
            partition=mechanism_partition,
            states_by_index=states_by_index,
            actions_by_index=actions_by_index,
            presented_states=presented_states,
            presented_actions=presented_actions,
            observations=(),
            maximum_steps=maximum_steps,
            used_cases=used_public_cases,
        )
        for index in range(queries)
    ]
    random.Random(_domain_seed(seed, "support-order", 0, 0)).shuffle(
        support_pairs
    )
    random.Random(_domain_seed(seed, "query-order", 0, 0)).shuffle(query_pairs)
    return GlyphMachineTraceStream(
        supports=tuple(support_pairs),
        queries=tuple(query_pairs),
        mechanism_commitment=commitment,
        mechanism_partition=mechanism_partition,
        control_arm="correct",
    )


def commit_glyph_procedure(
    task: PublicGlyphMachineTask,
    actions: Sequence[GroundAction],
    *,
    stopped: bool,
) -> CommittedGlyphProcedure:
    """Validate and snapshot one typed procedure without executing it."""

    if not isinstance(task, PublicGlyphMachineTask):
        raise TypeError("task must be a PublicGlyphMachineTask")
    if not isinstance(actions, Sequence) or isinstance(
        actions,
        (str, bytes, bytearray),
    ):
        raise TypeError("actions must be a finite sequence")
    if type(stopped) is not bool:
        raise TypeError("stopped must be bool")
    procedure = CommittedGlyphProcedure(
        public_digest=_public_digest(task),
        actions=tuple(actions),
        stopped=stopped,
    )
    _validate_procedure_for_task(task, procedure)
    return procedure


def score_glyph_procedure(
    task: PublicGlyphMachineTask,
    solution: _HiddenGlyphMachineSolution,
    procedure: CommittedGlyphProcedure,
) -> float:
    """Return only terminal exactness for one frozen typed procedure."""

    _validate_pairing(task, solution)
    _validate_procedure_for_task(task, procedure)
    if procedure.public_digest != solution.public_digest:
        raise ValueError("procedure and evaluator solution are bound to different tasks")
    state_indices = {
        digest: index for index, digest in enumerate(solution.state_digests)
    }
    action_indices = {
        digest: index for index, digest in enumerate(solution.action_digests)
    }
    current = state_indices[task.origin.digest]
    for action in procedure.actions:
        current = solution.transition_rows[action_indices[action.schema.digest]][
            current
        ]
    target = next(
        index
        for index, digest in enumerate(solution.state_digests)
        if digest == _goal_state(task).digest
    )
    return float(current == target)


def judge_glyph_procedure_attempt(
    pair: GeneratedGlyphMachineTask,
    procedure: CommittedGlyphProcedure,
) -> float:
    """Judge one committed attempt and expose only its terminal scalar."""

    if not isinstance(pair, GeneratedGlyphMachineTask):
        raise TypeError("pair must be a GeneratedGlyphMachineTask")
    if not isinstance(procedure, CommittedGlyphProcedure):
        raise TypeError("procedure must be a CommittedGlyphProcedure")
    return score_glyph_procedure(pair.learner, pair.hidden, procedure)


def make_glyph_machine_control_stream(
    stream: GlyphMachineTraceStream,
    arm: GlyphMachineControlArm,
) -> GlyphMachineTraceStream:
    """Create a matched public-evidence control without exposing physics.

    The wrong-trace arm rotates each publicly observed successor through the
    declared public state order.  It never consults evaluator transition
    semantics.  Hidden task bindings are refreshed here, on the evaluator
    side, solely so the ordinary scalar judge can validate the altered public
    projection.
    """

    if not isinstance(stream, GlyphMachineTraceStream):
        raise TypeError("stream must be a GlyphMachineTraceStream")
    if arm not in ("correct", "no_trace", "wrong_trace"):
        raise ValueError("control arm must be correct, no_trace, or wrong_trace")
    if arm == "correct":
        return stream
    supports = tuple(
        _reproject_control_pair(pair, arm)
        for pair in stream.supports
    )
    return GlyphMachineTraceStream(
        supports=supports,
        queries=stream.queries,
        mechanism_commitment=stream.mechanism_commitment,
        mechanism_partition=stream.mechanism_partition,
        control_arm=arm,
    )


def _reproject_control_pair(
    pair: GeneratedGlyphMachineTask,
    arm: GlyphMachineControlArm,
) -> GeneratedGlyphMachineTask:
    task = pair.learner
    observations = (
        ()
        if arm == "no_trace"
        else _rotated_public_observations(task)
    )
    learner = replace(task, observations=observations)
    solution = replace(pair.hidden, public_digest=_public_digest(learner))
    return GeneratedGlyphMachineTask(learner, solution)


def _rotated_public_observations(
    task: PublicGlyphMachineTask,
) -> tuple[Trace, ...]:
    state_indices = {
        state.digest: index for index, state in enumerate(task.states)
    }
    traces: list[Trace] = []
    for trace in task.observations:
        current = trace.initial
        transitions: list[Transition] = []
        for transition in trace.transitions:
            observed_index = state_indices[transition.after.digest]
            rotated_after = task.states[(observed_index + 1) % len(task.states)]
            applied = current != rotated_after
            transitions.append(
                Transition(
                    before=current,
                    action=transition.action,
                    after=rotated_after,
                    applied=applied,
                    outcome=(
                        _APPLIED_OUTCOME if applied else _NO_CHANGE_OUTCOME
                    ),
                )
            )
            current = rotated_after
        traces.append(Trace(initial=trace.initial, transitions=tuple(transitions)))
    return tuple(traces)


def _make_encounter(
    seed: int,
    *,
    scope: str,
    index: int,
    mechanism: GlyphMachineMechanism,
    commitment: str,
    partition: GlyphMachinePartition,
    states_by_index: tuple[State, ...],
    actions_by_index: tuple[ActionSchema, ...],
    presented_states: tuple[State, ...],
    presented_actions: tuple[ActionSchema, ...],
    observations: tuple[Trace, ...],
    maximum_steps: int,
    used_cases: set[tuple[int, int, int]] | None,
) -> GeneratedGlyphMachineTask:
    origin_index, target_index, budget = _select_public_case(
        seed,
        scope,
        index,
        mechanism[1],
        maximum_steps,
        used_cases,
    )
    task = PublicGlyphMachineTask(
        states=presented_states,
        actions=presented_actions,
        observations=observations,
        origin=states_by_index[origin_index],
        goal=Goal.from_records(
            _NAMESPACE,
            states_by_index[target_index].records,
            exact=True,
        ),
        max_steps=budget,
    )
    hidden = _HiddenGlyphMachineSolution(
        public_digest=_public_digest(task),
        state_digests=tuple(state.digest for state in states_by_index),
        action_digests=tuple(action.digest for action in actions_by_index),
        transition_rows=mechanism[1],
        mechanism_commitment=commitment,
        mechanism_partition=partition,
    )
    return GeneratedGlyphMachineTask(task, hidden)


def _surface_values(
    surface_seed: int,
    commitment: str,
    mechanism: GlyphMachineMechanism,
) -> tuple[
    tuple[State, ...],
    tuple[ActionSchema, ...],
    tuple[State, ...],
    tuple[ActionSchema, ...],
]:
    state_count, transition_rows = mechanism
    states_by_index = tuple(
        State.from_records(
            _NAMESPACE,
            (
                Record(
                    _STATE_PREDICATE,
                    (_opaque_token(surface_seed, commitment, "state", index),),
                ),
            ),
        )
        for index in range(state_count)
    )
    actions_by_index = tuple(
        ActionSchema(
            f"{_NAMESPACE}.action_{_opaque_token(surface_seed, commitment, 'action', index)}",
            (),
        )
        for index in range(len(transition_rows))
    )
    presented_states = list(states_by_index)
    presented_actions = list(actions_by_index)
    random.Random(_domain_seed(surface_seed, "state-presentation", 0, 0)).shuffle(
        presented_states
    )
    random.Random(_domain_seed(surface_seed, "action-presentation", 0, 0)).shuffle(
        presented_actions
    )
    return (
        states_by_index,
        actions_by_index,
        tuple(presented_states),
        tuple(presented_actions),
    )


def _make_observations(
    seed: int,
    *,
    support_index: int,
    count: int,
    transition_rows: tuple[tuple[int, ...], ...],
    states_by_index: tuple[State, ...],
    actions_by_index: tuple[ActionSchema, ...],
) -> tuple[Trace, ...]:
    traces: list[Trace] = []
    for observation_index in range(count):
        rng = random.Random(
            _domain_seed(seed, "observation", support_index, observation_index)
        )
        current_index = rng.randrange(len(states_by_index))
        initial = states_by_index[current_index]
        step_count = 1 + rng.randrange(2)
        transitions: list[Transition] = []
        for step in range(step_count):
            if step == 0:
                action_index = (
                    support_index * count + observation_index
                ) % len(actions_by_index)
            else:
                action_index = rng.randrange(len(actions_by_index))
            successor_index = transition_rows[action_index][current_index]
            before = states_by_index[current_index]
            after = states_by_index[successor_index]
            applied = before != after
            transitions.append(
                Transition(
                    before=before,
                    action=actions_by_index[action_index].ground(),
                    after=after,
                    applied=applied,
                    outcome=(
                        _APPLIED_OUTCOME if applied else _NO_CHANGE_OUTCOME
                    ),
                )
            )
            current_index = successor_index
        traces.append(Trace(initial=initial, transitions=tuple(transitions)))
    random.Random(_domain_seed(seed, "observation-order", support_index, 0)).shuffle(
        traces
    )
    return tuple(traces)


def _select_public_case(
    seed: int,
    scope: str,
    index: int,
    transition_rows: tuple[tuple[int, ...], ...],
    maximum_steps: int,
    used_cases: set[tuple[int, int, int]] | None,
) -> tuple[int, int, int]:
    state_count = len(transition_rows[0])
    for attempt in range(4_096):
        rng = random.Random(_domain_seed(seed, scope, index, attempt))
        budget = 1 + rng.randrange(maximum_steps)
        origin = rng.randrange(state_count)
        current = origin
        step_count = 1 + rng.randrange(budget)
        for _ in range(step_count):
            action_index = rng.randrange(len(transition_rows))
            current = transition_rows[action_index][current]
        case = (origin, current, budget)
        if current == origin or (used_cases is not None and case in used_cases):
            continue
        if used_cases is not None:
            used_cases.add(case)
        return case
    raise RuntimeError("could not generate a fresh nontrivial glyph-machine case")


@lru_cache(maxsize=1)
def _semantic_machine_universe() -> tuple[GlyphMachineMechanism, ...]:
    classes: set[GlyphMachineMechanism] = set()
    for state_count in range(2, 5):
        identity = tuple(range(state_count))
        permutations = tuple(
            permutation
            for permutation in itertools.permutations(range(state_count))
            if permutation != identity
        )
        for action_count in range(1, min(3, len(permutations)) + 1):
            for actions in itertools.combinations(permutations, action_count):
                if not _is_connected(actions, state_count):
                    continue
                classes.add(
                    (
                        state_count,
                        _canonicalize_transition_rows(state_count, actions),
                    )
                )
    if len(classes) != _EXPECTED_SEMANTIC_CLASS_COUNT:
        raise RuntimeError(
            "glyph-machine semantic universe changed: "
            f"expected {_EXPECTED_SEMANTIC_CLASS_COUNT}, observed {len(classes)}"
        )
    return tuple(sorted(classes, key=_mechanism_order_key))


@lru_cache(maxsize=4)
def _semantic_partition(
    partition: str,
) -> tuple[GlyphMachineMechanism, ...]:
    if partition not in (*_PARTITION_SIZES, "sealed"):
        raise ValueError("partition must be train, development, final, or sealed")
    by_size = {
        state_count: tuple(
            mechanism
            for mechanism in _semantic_machine_universe()
            if mechanism[0] == state_count
        )
        for state_count in (2, 3, 4)
    }
    if tuple(len(by_size[value]) for value in (2, 3, 4)) != (1, 7, 108):
        raise RuntimeError("glyph-machine state-count strata changed")
    slices = {
        "train": by_size[2] + by_size[3][:3] + by_size[4][:60],
        "development": by_size[3][3:5] + by_size[4][60:74],
        "final": by_size[3][5:7] + by_size[4][74:88],
        "sealed": by_size[4][88:],
    }
    selected = slices[partition]
    expected = _SEALED_COUNT if partition == "sealed" else _PARTITION_SIZES[partition]
    if len(selected) != expected:
        raise RuntimeError("glyph-machine partition size is inconsistent")
    return selected


def _canonicalize_transition_rows(
    state_count: int,
    transition_rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(row) for row in transition_rows)
    if (
        isinstance(state_count, bool)
        or not isinstance(state_count, int)
        or not 2 <= state_count <= 4
    ):
        raise ValueError("state_count must be an integer from two through four")
    if not 1 <= len(rows) <= 3:
        raise ValueError("a mechanism must contain one through three actions")
    identity = tuple(range(state_count))
    if any(len(row) != state_count or set(row) != set(identity) for row in rows):
        raise ValueError("every transition row must be a state permutation")
    if any(row == identity for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("mechanism actions must be distinct and nonidentity")

    candidates: list[tuple[tuple[int, ...], ...]] = []
    for renaming in itertools.permutations(range(state_count)):
        inverse = [0] * state_count
        for source, target in enumerate(renaming):
            inverse[target] = source
        renamed = tuple(
            sorted(
                tuple(
                    renaming[row[inverse[public_state]]]
                    for public_state in range(state_count)
                )
                for row in rows
            )
        )
        candidates.append(renamed)
    return min(candidates)


def _is_connected(
    transition_rows: Sequence[Sequence[int]],
    state_count: int,
) -> bool:
    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for row in transition_rows:
            target = row[source]
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return len(reached) == state_count


def _mechanism_order_key(mechanism: GlyphMachineMechanism) -> bytes:
    payload = json.dumps(mechanism, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(_MECHANISM_ORDER_DOMAIN + payload).digest()


def _mechanism_commitment(mechanism: GlyphMachineMechanism) -> str:
    payload = json.dumps(mechanism, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        _MECHANISM_COMMITMENT_DOMAIN + payload
    ).hexdigest()


def _validate_public_task(task: PublicGlyphMachineTask) -> None:
    if type(task.states) is not tuple or not 2 <= len(task.states) <= 4:
        raise ValueError("public task must declare two through four states")
    if any(not isinstance(state, State) for state in task.states):
        raise TypeError("public states must contain only State values")
    if len(set(task.states)) != len(task.states):
        raise ValueError("public states must be unique")
    for state in task.states:
        if (
            state.namespace != _NAMESPACE
            or len(state.records) != 1
            or state.records[0].predicate != _STATE_PREDICATE
            or len(state.records[0].arguments) != 1
        ):
            raise ValueError("public state does not use the glyph-machine schema")
    if type(task.actions) is not tuple or not 1 <= len(task.actions) <= 3:
        raise ValueError("public task must declare one through three actions")
    if any(not isinstance(action, ActionSchema) for action in task.actions):
        raise TypeError("public actions must contain only ActionSchema values")
    if len(set(task.actions)) != len(task.actions):
        raise ValueError("public actions must be unique")
    if any(
        action.namespace != _NAMESPACE or action.parameters
        for action in task.actions
    ):
        raise ValueError("glyph-machine actions must be zero-argument schemas")
    if type(task.observations) is not tuple or any(
        not isinstance(observation, Trace) for observation in task.observations
    ):
        raise TypeError("observations must be an immutable Trace tuple")
    state_set = set(task.states)
    action_set = set(task.actions)
    for observation in task.observations:
        if not observation.transitions or observation.goal is not None:
            raise ValueError("public observations must be nonempty goal-free traces")
        if observation.initial not in state_set:
            raise ValueError("observation initial state is undeclared")
        for transition in observation.transitions:
            if (
                transition.before not in state_set
                or transition.after not in state_set
                or transition.action.schema not in action_set
                or transition.action.arguments
            ):
                raise ValueError("observation uses an undeclared state or action")
            expected_outcome = (
                _APPLIED_OUTCOME if transition.applied else _NO_CHANGE_OUTCOME
            )
            if transition.outcome != expected_outcome:
                raise ValueError("observation outcome is inconsistent")
    if task.origin not in state_set:
        raise ValueError("task origin must be one of the declared states")
    if not isinstance(task.goal, Goal):
        raise TypeError("task goal must be a Goal")
    if (
        task.goal.namespace != _NAMESPACE
        or not task.goal.exact
        or task.goal.forbidden
    ):
        raise ValueError("glyph-machine goals must be exact public states")
    _goal_state(task)
    if (
        isinstance(task.max_steps, bool)
        or not isinstance(task.max_steps, int)
        or not 1 <= task.max_steps <= 4
    ):
        raise ValueError("task max_steps must be an integer from one through four")


def _validate_hidden_solution(solution: _HiddenGlyphMachineSolution) -> None:
    _require_digest(solution.public_digest, "hidden public_digest")
    _require_digest(solution.mechanism_commitment, "mechanism_commitment")
    _validate_partition(solution.mechanism_partition)
    if type(solution.state_digests) is not tuple or not 2 <= len(
        solution.state_digests
    ) <= 4:
        raise ValueError("hidden state binding has the wrong size")
    if type(solution.action_digests) is not tuple or not 1 <= len(
        solution.action_digests
    ) <= 3:
        raise ValueError("hidden action binding has the wrong size")
    for value in (*solution.state_digests, *solution.action_digests):
        _require_digest(value, "hidden public value digest")
    if len(set(solution.state_digests)) != len(solution.state_digests) or len(
        set(solution.action_digests)
    ) != len(solution.action_digests):
        raise ValueError("hidden public bindings must be unique")
    canonical_rows = _canonicalize_transition_rows(
        len(solution.state_digests),
        solution.transition_rows,
    )
    mechanism = (len(solution.state_digests), canonical_rows)
    if solution.transition_rows != canonical_rows:
        raise ValueError("hidden transition rows must use canonical semantics")
    if solution.mechanism_commitment != _mechanism_commitment(mechanism):
        raise ValueError("hidden mechanism commitment is inconsistent")
    if solution.mechanism_commitment not in glyph_machine_mechanism_partition(
        solution.mechanism_partition
    ):
        raise ValueError("hidden mechanism is outside its declared partition")


def _validate_pairing(
    task: PublicGlyphMachineTask,
    solution: _HiddenGlyphMachineSolution,
) -> None:
    if not isinstance(task, PublicGlyphMachineTask):
        raise TypeError("learner task must be a PublicGlyphMachineTask")
    if not isinstance(solution, _HiddenGlyphMachineSolution):
        raise TypeError("solution must remain evaluator-owned")
    if solution.public_digest != _public_digest(task):
        raise ValueError("public task and hidden solution do not match")
    if set(solution.state_digests) != {state.digest for state in task.states}:
        raise ValueError("hidden state binding differs from public states")
    if set(solution.action_digests) != {action.digest for action in task.actions}:
        raise ValueError("hidden action binding differs from public actions")


def _validate_procedure_for_task(
    task: PublicGlyphMachineTask,
    procedure: CommittedGlyphProcedure,
) -> None:
    if not isinstance(procedure, CommittedGlyphProcedure):
        raise TypeError("procedure must be a CommittedGlyphProcedure")
    if procedure.public_digest != _public_digest(task):
        raise ValueError("procedure and public task do not match")
    if len(procedure.actions) > task.max_steps:
        raise ValueError("procedure exceeds the public step budget")
    if not procedure.stopped and len(procedure.actions) < task.max_steps:
        raise ValueError("a short procedure must contain an explicit STOP")
    declared = set(task.actions)
    if any(
        action.schema not in declared or action.arguments
        for action in procedure.actions
    ):
        raise ValueError("procedure contains an undeclared glyph-machine action")


def _goal_state(task: PublicGlyphMachineTask) -> State:
    matches = tuple(
        state for state in task.states if state.records == task.goal.required
    )
    if len(matches) != 1:
        raise ValueError("glyph-machine goal must identify exactly one public state")
    return matches[0]


def _public_digest(task: PublicGlyphMachineTask) -> str:
    payload = json.dumps(
        task.to_canonical(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_PUBLIC_DIGEST_DOMAIN + payload).hexdigest()


def _state_payload(state: State) -> dict[str, object]:
    return {
        "namespace": state.namespace,
        "records": [
            {
                "arguments": list(record.arguments),
                "predicate": record.predicate,
            }
            for record in state.records
        ],
    }


def _action_payload(action: ActionSchema) -> dict[str, object]:
    return {
        "description": action.description,
        "name": action.name,
        "parameters": [
            {"name": parameter.name, "type_name": parameter.type_name}
            for parameter in action.parameters
        ],
    }


def _goal_payload(goal: Goal) -> dict[str, object]:
    return {
        "exact": goal.exact,
        "forbidden": [
            {"arguments": list(record.arguments), "predicate": record.predicate}
            for record in goal.forbidden
        ],
        "namespace": goal.namespace,
        "required": [
            {"arguments": list(record.arguments), "predicate": record.predicate}
            for record in goal.required
        ],
    }


def _trace_payload(trace: Trace) -> dict[str, object]:
    return {
        "initial": _state_payload(trace.initial),
        "transitions": [
            {
                "action": {
                    "arguments": list(transition.action.arguments),
                    "schema": _action_payload(transition.action.schema),
                },
                "after": _state_payload(transition.after),
                "applied": transition.applied,
                "before": _state_payload(transition.before),
                "outcome": transition.outcome,
            }
            for transition in trace.transitions
        ],
    }


def _opaque_token(
    surface_seed: int,
    commitment: str,
    role: str,
    index: int,
) -> str:
    material = (
        _OPAQUE_SURFACE_DOMAIN
        + str(surface_seed).encode("ascii")
        + b"\x00"
        + commitment.encode("ascii")
        + b"\x00"
        + role.encode("ascii")
        + b"\x00"
        + str(index).encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()[:20]


def _domain_seed(seed: int, scope: str, index: int, variant: int) -> int:
    material = (
        f"project-angler.glyph-machine.stream.v1\x00{seed}\x00{scope}\x00"
        f"{index}\x00{variant}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _validate_partition(partition: str) -> None:
    if partition not in _PARTITION_SIZES:
        raise ValueError("partition must be train, development, or final")


def _validate_seed(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _validate_positive_count(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


__all__ = [
    "CommittedGlyphProcedure",
    "GeneratedGlyphMachineTask",
    "GlyphMachinePartition",
    "GlyphMachineControlArm",
    "GlyphMachineTraceStream",
    "PublicGlyphMachineTask",
    "commit_glyph_procedure",
    "glyph_machine_mechanism_partition",
    "judge_glyph_procedure_attempt",
    "make_glyph_machine_control_stream",
    "make_glyph_machine_trace_stream",
    "score_glyph_procedure",
]
