"""Contract and independence tests for relational procedure domains."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    GroundAction,
    Parameter,
    Record,
    State,
    Trace,
    Transition,
)
from angler.procedures.alignment import find_structural_isomorphisms  # noqa: E402
from angler.procedures.induction import (  # noqa: E402
    MDLOperatorInducer,
    cluster_subsegments,
    extract_subsegment_delta,
)
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402


class RelationalRecordContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = "angler.test"
        self.left = Record("angler.test.at", ("item", "left"))
        self.right = Record("angler.test.at", ("item", "right"))
        self.schema = ActionSchema(
            "angler.test.move",
            (
                Parameter("item", "angler.test.item"),
                Parameter("destination", "angler.test.place"),
            ),
            description="Move an item to one destination.",
        )

    def test_state_is_canonical_immutable_and_content_addressed(self) -> None:
        first = State.from_records(self.namespace, (self.right, self.left))
        second = State.from_records(self.namespace, (self.left, self.right))
        self.assertEqual(first.records, (self.left, self.right))
        self.assertEqual(first, second)
        self.assertEqual(first.digest, second.digest)
        self.assertRegex(first.digest, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(FrozenInstanceError):
            first.namespace = "angler.changed"  # type: ignore[misc]

    def test_direct_contracts_reject_noncanonical_or_cross_namespace_data(self) -> None:
        with self.assertRaises(ValueError):
            State(self.namespace, (self.right, self.left))
        with self.assertRaises(ValueError):
            State.from_records(self.namespace, (self.left, self.left))
        with self.assertRaises(ValueError):
            State.from_records(
                self.namespace,
                (Record("angler.other.at", ("item", "left")),),
            )
        with self.assertRaises(TypeError):
            Record("angler.test.at", ["item", "left"])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Record("not_namespaced", ("item",))

    def test_action_schema_is_typed_grounded_and_description_bound(self) -> None:
        action = self.schema.ground("item", "right")
        self.assertIsInstance(action, GroundAction)
        self.assertEqual(action.schema.parameters[0].type_name, "angler.test.item")
        self.assertEqual(action.arguments, ("item", "right"))
        with self.assertRaises(ValueError):
            self.schema.ground("item")
        with self.assertRaises(ValueError):
            ActionSchema(
                "angler.test.move",
                (Parameter("item", "angler.other.item"),),
            )
        changed_description = ActionSchema(
            self.schema.name,
            self.schema.parameters,
            description="Move the same typed item using changed semantics.",
        )
        self.assertNotEqual(self.schema.digest, changed_description.digest)

    def test_goal_transition_and_trace_validate_namespace_and_chain(self) -> None:
        start = State.from_records(self.namespace, (self.left,))
        end = State.from_records(self.namespace, (self.right,))
        action = self.schema.ground("item", "right")
        goal = Goal.from_records(self.namespace, (self.right,), exact=True)
        transition = Transition(
            start,
            action,
            end,
            True,
            "angler.test.applied",
        )
        trace = Trace(start, (transition,), goal)
        self.assertEqual(trace.final_state, end)
        self.assertRegex(goal.digest, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(transition.digest, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(trace.digest, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(ValueError):
            Transition(start, action, end, False, "angler.test.blocked")
        with self.assertRaises(ValueError):
            Trace(end, (transition,), goal)


class IndependentRelationalDomainTests(unittest.TestCase):
    def test_tokens_require_adjacent_moves_into_the_empty_position(self) -> None:
        state = tokens.make_token_state(("a", "b", None))
        goal = tokens.make_token_goal((None, "a", "b"))
        self.assertFalse(tokens.verify_token_goal(state, goal))
        blocked = tokens.execute_token_action(
            state,
            tokens.MOVE_TOKEN.ground("a", "position_0", "position_2"),
        )
        self.assertFalse(blocked.applied)
        transitions = []
        for action in (
            tokens.MOVE_TOKEN.ground("b", "position_1", "position_2"),
            tokens.MOVE_TOKEN.ground("a", "position_0", "position_1"),
        ):
            transition = tokens.execute_token_action(state, action)
            self.assertTrue(transition.applied)
            transitions.append(transition)
            state = transition.after
        self.assertTrue(tokens.verify_token_goal(state, goal))
        trace = Trace(transitions[0].before, tuple(transitions), goal)
        self.assertEqual(trace.final_state, state)

    def test_files_require_linked_multi_step_relocation(self) -> None:
        state = files.make_file_state(
            {"report": "inbox", "index": "archive"},
            (("inbox", "review"), ("review", "archive")),
        )
        goal = files.make_file_goal({"report": "archive"})
        blocked = files.execute_file_action(
            state,
            files.RELOCATE_FILE.ground("report", "inbox", "archive"),
        )
        self.assertFalse(blocked.applied)
        self.assertEqual(blocked.after, state)
        first = files.execute_file_action(
            state,
            files.RELOCATE_FILE.ground("report", "inbox", "review"),
        )
        second = files.execute_file_action(
            first.after,
            files.RELOCATE_FILE.ground("report", "review", "archive"),
        )
        self.assertTrue(files.verify_file_goal(second.after, goal))
        self.assertEqual(Trace(state, (first, second), goal).final_state, second.after)

    def test_files_block_a_same_name_collision_at_the_destination(self) -> None:
        state = files.make_file_state(
            (("report", "inbox"), ("report", "archive")),
            (("inbox", "archive"),),
        )
        transition = files.execute_file_action(
            state,
            files.RELOCATE_FILE.ground("report", "inbox", "archive"),
        )
        self.assertFalse(transition.applied)
        self.assertEqual(transition.after, state)

    def test_boxes_require_capacity_permitting_multi_step_transfers(self) -> None:
        state = boxes.make_box_state(
            {"source": ("amber", "blue"), "destination": ()},
            {"source": 2, "destination": 2},
        )
        goal = boxes.make_box_goal(
            {"source": (), "destination": ("amber", "blue")},
            {"source": 2, "destination": 2},
        )
        first = boxes.execute_box_action(
            state,
            boxes.TRANSFER_ITEM.ground("amber", "source", "destination"),
        )
        second = boxes.execute_box_action(
            first.after,
            boxes.TRANSFER_ITEM.ground("blue", "source", "destination"),
        )
        self.assertTrue(first.applied)
        self.assertTrue(second.applied)
        self.assertTrue(boxes.verify_box_goal(second.after, goal))
        self.assertEqual(Trace(state, (first, second), goal).final_state, second.after)

        constrained = boxes.make_box_state(
            {"source": ("amber", "blue"), "destination": ()},
            {"source": 2, "destination": 1},
        )
        moved = boxes.execute_box_action(
            constrained,
            boxes.TRANSFER_ITEM.ground("amber", "source", "destination"),
        )
        blocked = boxes.execute_box_action(
            moved.after,
            boxes.TRANSFER_ITEM.ground("blue", "source", "destination"),
        )
        self.assertFalse(blocked.applied)

    def test_domain_vocabularies_and_executors_are_not_interchangeable(self) -> None:
        self.assertEqual(
            len({tokens.NAMESPACE, files.NAMESPACE, boxes.NAMESPACE}),
            3,
        )
        token_state = tokens.make_token_state(("a", "b", None))
        with self.assertRaises(ValueError):
            tokens.execute_token_action(
                token_state,
                files.RELOCATE_FILE.ground("a", "left", "right"),
            )
        public_names = {
            name
            for module in (tokens, files, boxes)
            for name in module.__all__
        }
        self.assertFalse(any("solver" in name.lower() for name in public_names))
        self.assertFalse(any("path" in name.lower() for name in public_names))

    def test_executed_multi_step_traces_induce_isomorphic_relocation_cores(self) -> None:
        def token_trace(first: str, second: str) -> Trace:
            initial = tokens.make_token_state((first, second, None))
            goal = tokens.make_token_goal((None, first, second))
            actions = (
                tokens.MOVE_TOKEN.ground(
                    second,
                    "position_1",
                    "position_2",
                ),
                tokens.MOVE_TOKEN.ground(
                    first,
                    "position_0",
                    "position_1",
                ),
            )
            return self._execute_trace(
                initial,
                goal,
                actions,
                tokens.execute_token_action,
            )

        links = (
            ("position_0", "position_1"),
            ("position_1", "position_0"),
            ("position_1", "position_2"),
            ("position_2", "position_1"),
        )

        def file_trace(first: str, second: str) -> Trace:
            initial = files.make_file_state(
                ((first, "position_0"), (second, "position_1")),
                links,
            )
            goal = files.make_file_goal(
                ((first, "position_1"), (second, "position_2"))
            )
            actions = (
                files.RELOCATE_FILE.ground(
                    second,
                    "position_1",
                    "position_2",
                ),
                files.RELOCATE_FILE.ground(
                    first,
                    "position_0",
                    "position_1",
                ),
            )
            return self._execute_trace(
                initial,
                goal,
                actions,
                files.execute_file_action,
            )

        capacities = {
            "position_0": 1,
            "position_1": 2,
            "position_2": 1,
        }

        def box_trace(first: str, second: str) -> Trace:
            initial = boxes.make_box_state(
                {
                    "position_0": (first,),
                    "position_1": (second,),
                    "position_2": (),
                },
                capacities,
            )
            goal = boxes.make_box_goal(
                {
                    "position_0": (),
                    "position_1": (first,),
                    "position_2": (second,),
                },
                capacities,
            )
            actions = (
                boxes.TRANSFER_ITEM.ground(
                    second,
                    "position_1",
                    "position_2",
                ),
                boxes.TRANSFER_ITEM.ground(
                    first,
                    "position_0",
                    "position_1",
                ),
            )
            return self._execute_trace(
                initial,
                goal,
                actions,
                boxes.execute_box_action,
            )

        inducer = MDLOperatorInducer(minimum_support=2, minimum_savings=1)
        operators = []
        for trace_factory in (token_trace, file_trace, box_trace):
            segments = tuple(
                extract_subsegment_delta(trace_factory(*names), 0, 2)
                for names in (("amber", "blue"), ("cyan", "gold"))
            )
            clusters = cluster_subsegments(segments)
            self.assertEqual(len(clusters), 1)
            candidate = inducer.allocate(clusters[0])
            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual(len(candidate.operator.body), 2)
            self.assertEqual(len(candidate.operator.effects), 4)
            operators.append(candidate.operator)

        for source_index, target_index in ((0, 1), (0, 2), (1, 2)):
            matches = find_structural_isomorphisms(
                operators[source_index],
                operators[target_index],
            )
            self.assertTrue(matches)
            self.assertEqual(
                matches[0].coverage.matched_effects,
                4,
            )
            self.assertFalse(matches[0].residuals.source_effects)
            self.assertFalse(matches[0].residuals.target_effects)

    @staticmethod
    def _execute_trace(initial, goal, actions, executor) -> Trace:
        state = initial
        transitions = []
        for action in actions:
            transition = executor(state, action)
            if not transition.applied:
                raise AssertionError("test action sequence must be executable")
            transitions.append(transition)
            state = transition.after
        return Trace(initial, tuple(transitions), goal)


if __name__ == "__main__":
    unittest.main()
