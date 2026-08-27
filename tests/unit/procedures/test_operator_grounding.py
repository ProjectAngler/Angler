"""Tests for domain-neutral learned-operator grounding."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures import grounding  # noqa: E402
from angler.procedures.grounding import (  # noqa: E402
    GroundingError,
    GroundingLimitError,
    enumerate_operator_bindings,
    instantiate_operator,
    score_goal_effect_overlap,
)
from angler.procedures.induction import (  # noqa: E402
    MDLOperatorInducer,
    cluster_subsegments,
    extract_subsegment_delta,
)
from angler.procedures.operators import Constant, TypedVariable  # noqa: E402
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    Parameter,
    Record,
    State,
    Trace,
    Transition,
)
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402


NAMESPACE = "grounding.demo"
ENTITY = f"{NAMESPACE}.entity"
LOCATION = f"{NAMESPACE}.location"
AT = f"{NAMESPACE}.at"
CLEAR = f"{NAMESPACE}.clear"
MOVE = ActionSchema(
    f"{NAMESPACE}.move",
    (
        Parameter("entity", ENTITY),
        Parameter("source", LOCATION),
        Parameter("destination", LOCATION),
    ),
    description="Move one entity from its location into a clear location.",
)


def _state(entity: str, occupied: str, clear: str) -> State:
    return State.from_records(
        NAMESPACE,
        (
            Record(AT, (entity, occupied)),
            Record(CLEAR, (clear,)),
        ),
    )


def _move_trace(entity: str, source: str, destination: str) -> Trace:
    before = _state(entity, source, destination)
    after = _state(entity, destination, source)
    transition = Transition(
        before,
        MOVE.ground(entity, source, destination),
        after,
        True,
        f"{NAMESPACE}.applied",
    )
    return Trace(before, (transition,))


def _induced_operator(*, constant_destination: bool = False):
    destinations = ("hub", "hub") if constant_destination else ("beta", "delta")
    traces = (
        _move_trace("robot_1", "alpha", destinations[0]),
        _move_trace("robot_2", "gamma", destinations[1]),
    )
    segments = tuple(
        extract_subsegment_delta(trace, 0, 1)
        for trace in traces
    )
    cluster = cluster_subsegments(segments)[0]
    candidate = MDLOperatorInducer(
        minimum_support=2,
        minimum_savings=-100,
    ).allocate(cluster)
    if candidate is None:
        raise AssertionError("test traces must induce an operator")
    return candidate.operator


class OperatorGroundingTests(unittest.TestCase):
    def test_unseen_state_entities_produce_complete_typed_binding(self) -> None:
        operator = _induced_operator()
        unseen = _state("novel_agent", "north", "south")

        bindings = enumerate_operator_bindings(operator, unseen)

        self.assertEqual(len(bindings), 1)
        binding = bindings[0]
        self.assertEqual(binding.operator_digest, operator.digest)
        self.assertEqual(
            tuple(item.variable for item in binding.assignments),
            operator.variables,
        )
        self.assertEqual(
            {item.variable.type_name for item in binding.assignments},
            {ENTITY, LOCATION},
        )
        self.assertEqual(
            {item.value for item in binding.assignments},
            {"novel_agent", "north", "south"},
        )
        self.assertRegex(binding.digest, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(FrozenInstanceError):
            binding.namespace = "changed.demo"  # type: ignore[misc]

    def test_instantiation_preserves_cross_pattern_coreference(self) -> None:
        operator = _induced_operator()
        state = _state("unit_9", "left", "right")
        binding = enumerate_operator_bindings(operator, state)[0]

        prediction = instantiate_operator(operator, binding)

        self.assertEqual(prediction.actions, (MOVE.ground("unit_9", "left", "right"),))
        self.assertEqual(
            set(prediction.predicted_additions),
            {Record(AT, ("unit_9", "right")), Record(CLEAR, ("left",))},
        )
        self.assertEqual(
            set(prediction.predicted_deletions),
            {Record(AT, ("unit_9", "left")), Record(CLEAR, ("right",))},
        )
        self.assertRegex(prediction.digest, r"^sha256:[0-9a-f]{64}$")

    def test_identical_typed_action_values_remain_groundable_variables(self) -> None:
        operator = _induced_operator(constant_destination=True)
        state = State.from_records(
            NAMESPACE,
            (
                Record(AT, ("new_unit", "outer")),
                Record(CLEAR, ("decoy",)),
                Record(CLEAR, ("hub",)),
            ),
        )

        bindings = enumerate_operator_bindings(operator, state)

        self.assertEqual(len(bindings), 2)
        predictions = tuple(
            instantiate_operator(operator, binding) for binding in bindings
        )
        self.assertEqual(
            {item.actions for item in predictions},
            {
                (MOVE.ground("new_unit", "outer", "decoy"),),
                (MOVE.ground("new_unit", "outer", "hub"),),
            },
        )
        self.assertTrue(
            all(
                isinstance(term, TypedVariable)
                for term in operator.body[0].arguments
            )
        )

    def test_unsatisfied_precondition_yields_no_binding(self) -> None:
        operator = _induced_operator()
        blocked = State.from_records(
            NAMESPACE,
            (Record(AT, ("unit", "left")),),
        )
        self.assertEqual(enumerate_operator_bindings(operator, blocked), ())

    def test_state_binding_exports_losslessly_to_execution_contract(self) -> None:
        operator = _induced_operator()
        binding = enumerate_operator_bindings(
            operator,
            _state("unit", "left", "right"),
        )[0]

        class Candidate:
            def __init__(self, value, type_name):
                self.value = value
                self.type_name = type_name

        class Assignment:
            def __init__(self, variable, entity):
                self.variable = variable
                self.entity = entity

        class CanonicalBinding:
            def __init__(self, bound_operator, assignments):
                self.operator = bound_operator
                self.assignments = assignments

        execution_contract = ModuleType("angler.procedures.execution")
        execution_contract.TypedEntityCandidate = Candidate
        execution_contract.BindingAssignment = Assignment
        execution_contract.OperatorBinding = CanonicalBinding
        with patch.dict(
            sys.modules,
            {"angler.procedures.execution": execution_contract},
        ):
            exported = binding.to_execution_binding(operator)

        self.assertIs(exported.operator, operator)
        self.assertEqual(
            tuple(item.variable for item in exported.assignments),
            operator.variables,
        )
        self.assertEqual(
            tuple(item.entity.value for item in exported.assignments),
            tuple(item.value for item in binding.assignments),
        )
        self.assertEqual(
            tuple(item.entity.type_name for item in exported.assignments),
            tuple(item.variable.type_name for item in binding.assignments),
        )

    def test_goal_effect_overlap_is_only_declarative_ordering_score(self) -> None:
        operator = _induced_operator()
        state = _state("unit", "left", "right")
        binding = enumerate_operator_bindings(operator, state)[0]
        prediction = instantiate_operator(operator, binding)
        helpful = Goal.from_records(
            NAMESPACE,
            (Record(AT, ("unit", "right")),),
            forbidden=(Record(AT, ("unit", "left")),),
        )
        conflicting = Goal.from_records(
            NAMESPACE,
            (Record(AT, ("unit", "left")),),
            forbidden=(Record(AT, ("unit", "right")),),
        )
        exact = Goal(
            namespace=NAMESPACE,
            required=_state("unit", "right", "left").records,
            exact=True,
        )

        self.assertEqual(score_goal_effect_overlap(prediction, helpful), 2)
        self.assertEqual(score_goal_effect_overlap(prediction, conflicting), -2)
        self.assertEqual(score_goal_effect_overlap(prediction, exact), 4)

    def test_enumeration_is_deduplicated_and_fails_closed_at_ceilings(self) -> None:
        operator = _induced_operator()
        state = State.from_records(
            NAMESPACE,
            (
                Record(AT, ("alpha", "left")),
                Record(AT, ("beta", "middle")),
                Record(CLEAR, ("north",)),
                Record(CLEAR, ("south",)),
            ),
        )
        bindings = enumerate_operator_bindings(operator, state)
        self.assertEqual(len(bindings), 4)
        self.assertEqual(len({item.digest for item in bindings}), 4)
        self.assertEqual(bindings, enumerate_operator_bindings(operator, state))

        with self.assertRaisesRegex(GroundingLimitError, "maximum_bindings"):
            enumerate_operator_bindings(operator, state, maximum_bindings=2)
        with self.assertRaisesRegex(GroundingLimitError, "maximum_match_attempts"):
            enumerate_operator_bindings(
                operator,
                state,
                maximum_match_attempts=1,
            )
        with self.assertRaises(ValueError):
            enumerate_operator_bindings(
                operator,
                state,
                maximum_bindings=grounding.HARD_MAX_BINDINGS + 1,
            )

    def test_module_has_no_world_or_executor_dependency(self) -> None:
        source_path = SRC / "angler" / "procedures" / "grounding.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any(name.startswith("angler.worlds") for name in imported))
        public_names = set(grounding.__all__)
        forbidden = ("execute", "solver", "route", "path", "shortest")
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in public_names
                for fragment in forbidden
            )
        )

    def test_two_step_body_grounds_and_executes_on_unseen_roles(self) -> None:
        capacities = {"source": 1, "middle": 2, "destination": 1}

        def training_trace(first: str, second: str) -> Trace:
            initial = boxes.make_box_state(
                {
                    "source": (first,),
                    "middle": (second,),
                    "destination": (),
                },
                capacities,
            )
            first_transition = boxes.execute_box_action(
                initial,
                boxes.TRANSFER_ITEM.ground(
                    second,
                    "middle",
                    "destination",
                ),
            )
            second_transition = boxes.execute_box_action(
                first_transition.after,
                boxes.TRANSFER_ITEM.ground(first, "source", "middle"),
            )
            return Trace(initial, (first_transition, second_transition))

        segments = tuple(
            extract_subsegment_delta(training_trace(*names), 0, 2)
            for names in (("amber", "blue"), ("cyan", "gold"))
        )
        candidate = MDLOperatorInducer(
            minimum_support=2,
            minimum_savings=-100,
        ).allocate(cluster_subsegments(segments)[0])
        self.assertIsNotNone(candidate)
        assert candidate is not None
        operator = candidate.operator

        self.assertTrue(
            all(
                isinstance(term, TypedVariable)
                for action in operator.body
                for term in action.arguments
            )
        )
        record_literals = {
            term.value
            for pattern in operator.preconditions
            for term in pattern.arguments
            if isinstance(term, Constant)
        }
        self.assertEqual(record_literals, {"limit_1"})

        unseen_capacities = {
            "new_source": 1,
            "new_middle": 2,
            "new_destination": 1,
        }
        unseen = boxes.make_box_state(
            {
                "new_source": ("heldout_amber",),
                "new_middle": ("heldout_blue",),
                "new_destination": (),
            },
            unseen_capacities,
        )
        expected_actions = (
            boxes.TRANSFER_ITEM.ground(
                "heldout_blue",
                "new_middle",
                "new_destination",
            ),
            boxes.TRANSFER_ITEM.ground(
                "heldout_amber",
                "new_source",
                "new_middle",
            ),
        )
        predictions = []
        for binding in enumerate_operator_bindings(operator, unseen):
            try:
                predictions.append(instantiate_operator(operator, binding))
            except GroundingError:
                continue
        prediction = next(
            item for item in predictions if item.actions == expected_actions
        )

        state = unseen
        for action in prediction.actions:
            transition = boxes.execute_box_action(state, action)
            self.assertTrue(transition.applied)
            state = transition.after
        goal = boxes.make_box_goal(
            {
                "new_source": (),
                "new_middle": ("heldout_amber",),
                "new_destination": ("heldout_blue",),
            },
            unseen_capacities,
        )
        self.assertTrue(boxes.verify_box_goal(state, goal))

    def test_large_topology_operator_grounds_in_smaller_unseen_topology(self) -> None:
        def training_trace(first: str, second: str, noise: str) -> Trace:
            initial = tokens.make_token_state(
                (noise + "_0", noise + "_1", first, second, None)
            )
            first_transition = tokens.execute_token_action(
                initial,
                tokens.MOVE_TOKEN.ground(
                    second,
                    "position_3",
                    "position_4",
                ),
            )
            second_transition = tokens.execute_token_action(
                first_transition.after,
                tokens.MOVE_TOKEN.ground(
                    first,
                    "position_2",
                    "position_3",
                ),
            )
            return Trace(initial, (first_transition, second_transition))

        segments = tuple(
            extract_subsegment_delta(training_trace(*values), 0, 2)
            for values in (
                ("amber", "blue", "noise_a"),
                ("cyan", "gold", "noise_b"),
            )
        )
        for segment in segments:
            self.assertEqual(len(segment.relevant_preconditions), 3)
            self.assertFalse(
                any(
                    argument in {"position_0", "position_1"}
                    for record in segment.relevant_preconditions
                    for argument in record.arguments
                )
            )
        candidate = MDLOperatorInducer(
            minimum_support=2,
            minimum_savings=-100,
        ).allocate(cluster_subsegments(segments)[0])
        self.assertIsNotNone(candidate)
        assert candidate is not None

        unseen = tokens.make_token_state(
            ("heldout_amber", "heldout_blue", None)
        )
        expected_actions = (
            tokens.MOVE_TOKEN.ground(
                "heldout_blue",
                "position_1",
                "position_2",
            ),
            tokens.MOVE_TOKEN.ground(
                "heldout_amber",
                "position_0",
                "position_1",
            ),
        )
        predictions = []
        for binding in enumerate_operator_bindings(candidate.operator, unseen):
            try:
                predictions.append(
                    instantiate_operator(candidate.operator, binding)
                )
            except GroundingError:
                continue
        prediction = next(
            item for item in predictions if item.actions == expected_actions
        )

        state = unseen
        for action in prediction.actions:
            transition = tokens.execute_token_action(state, action)
            self.assertTrue(transition.applied)
            state = transition.after
        self.assertTrue(
            tokens.verify_token_goal(
                state,
                tokens.make_token_goal(
                    (None, "heldout_amber", "heldout_blue")
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
