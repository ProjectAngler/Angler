"""Tests for the solver-free reversible primitive-transition world."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.worlds import reversible_transition_world as world  # noqa: E402
from angler.worlds.reversible_transition_world import (  # noqa: E402
    ACTION_COUNT,
    CommittedProcedure,
    ProcedureExecution,
    ReversibleTransitionTask,
    commit_procedure,
    execute_committed_procedure,
    generate_reversible_transition_task,
    observe_primitive_transition,
)


def replay_actions(
    origin: tuple[int, ...],
    actions: tuple[int, ...],
) -> tuple[int, ...]:
    state = origin
    for action in actions:
        state = observe_primitive_transition(state, action).after
    return state


class ReversibleTaskGenerationTests(unittest.TestCase):
    def test_seeded_generation_replays_exactly_and_has_unique_identity(self) -> None:
        first = generate_reversible_transition_task(4101, max_steps=17)
        replay = generate_reversible_transition_task(4101, max_steps=17)
        changed_seed = generate_reversible_transition_task(4102, max_steps=17)
        changed_budget = generate_reversible_transition_task(4101, max_steps=18)

        self.assertEqual(first, replay)
        self.assertNotEqual(first.instance_id, changed_seed.instance_id)
        self.assertNotEqual(first.instance_id, changed_budget.instance_id)
        self.assertEqual(first.available_actions, tuple(range(ACTION_COUNT)))
        self.assertEqual(sorted(first.origin), list(range(world.TOKEN_COUNT)))
        self.assertEqual(len(first.instance_id), len("sha256:") + 64)
        int(first.instance_id.removeprefix("sha256:"), 16)
        self.assertEqual(len(first.generation_commitment), len("sha256:") + 64)
        int(first.generation_commitment.removeprefix("sha256:"), 16)

        identities = {
            generate_reversible_transition_task(seed).instance_id
            for seed in range(100)
        }
        self.assertEqual(len(identities), 100)

    def test_task_and_all_public_value_objects_are_immutable(self) -> None:
        task = generate_reversible_transition_task(22)
        procedure = commit_procedure(task, (0, 1))
        result = execute_committed_procedure(
            task,
            procedure,
            replay_actions(task.origin, procedure.actions),
        )

        for value, field_name, replacement in (
            (task, "origin", tuple(reversed(task.origin))),
            (procedure, "actions", (1, 0)),
            (result, "exact", False),
        ):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, field_name, replacement)

    def test_generation_rejects_invalid_seed_and_step_bounds(self) -> None:
        for invalid in (True, 1.5, "1"):
            with self.subTest(seed=invalid):
                with self.assertRaises(TypeError):
                    generate_reversible_transition_task(invalid)  # type: ignore[arg-type]

        for invalid in (-1, world.MAX_PROCEDURE_STEPS + 1):
            with self.subTest(max_steps=invalid):
                with self.assertRaises(ValueError):
                    generate_reversible_transition_task(1, max_steps=invalid)
        with self.assertRaises(TypeError):
            generate_reversible_transition_task(1, max_steps=True)


class PrimitiveTransitionTests(unittest.TestCase):
    def test_every_primitive_is_an_exact_self_inverse_round_trip(self) -> None:
        for seed in range(12):
            state = generate_reversible_transition_task(seed).origin
            for action in range(ACTION_COUNT):
                with self.subTest(seed=seed, action=action):
                    forward = observe_primitive_transition(state, action)
                    backward = observe_primitive_transition(
                        forward.after,
                        forward.inverse_action,
                    )
                    self.assertEqual(forward.before, state)
                    self.assertEqual(forward.action, action)
                    self.assertEqual(forward.inverse_action, action)
                    self.assertEqual(backward.after, state)

                    changed = {
                        index
                        for index, (before, after) in enumerate(
                            zip(forward.before, forward.after, strict=True)
                        )
                        if before != after
                    }
                    self.assertEqual(changed, {action, action + 1})

    def test_transition_normalizes_sequence_and_never_mutates_input(self) -> None:
        source = [0, 1, 2, 3, 4, 5]
        transition = observe_primitive_transition(source, 2)

        self.assertEqual(source, [0, 1, 2, 3, 4, 5])
        self.assertEqual(transition.before, tuple(source))
        self.assertEqual(transition.after, (0, 1, 3, 2, 4, 5))

    def test_invalid_states_and_actions_are_rejected(self) -> None:
        invalid_states: tuple[object, ...] = (
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4, 4),
            (0, 1, 2, 3, 4, 6),
            (0, 1, 2, 3, 4, True),
            "0,1,2,3,4,5",
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises((TypeError, ValueError)):
                    observe_primitive_transition(state, 0)  # type: ignore[arg-type]

        for action in (-1, ACTION_COUNT, True, 1.0, "1"):
            with self.subTest(action=action):
                with self.assertRaises((TypeError, ValueError)):
                    observe_primitive_transition(
                        (0, 1, 2, 3, 4, 5),
                        action,  # type: ignore[arg-type]
                    )


class CommittedExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = generate_reversible_transition_task(7331, max_steps=8)

    def test_commit_snapshots_then_terminal_execution_reports_only_endpoint(self) -> None:
        source_actions = [0, 2, 1, 4, 3]
        target = replay_actions(self.task.origin, tuple(source_actions))
        committed = commit_procedure(self.task, source_actions)
        source_actions[:] = [4]

        result = execute_committed_procedure(self.task, committed, target)

        self.assertEqual(committed.actions, (0, 2, 1, 4, 3))
        self.assertEqual(
            result,
            ProcedureExecution(
                task_id=self.task.instance_id,
                reached_state=target,
                exact=True,
                steps_executed=5,
            ),
        )
        self.assertEqual(
            {field.name for field in fields(ProcedureExecution)},
            {"task_id", "reached_state", "exact", "steps_executed"},
        )
        self.assertFalse(
            {"path", "trace", "distance", "reward", "next_action"}
            & {field.name for field in fields(ProcedureExecution)}
        )

    def test_terminal_target_swap_changes_only_exact_judgment(self) -> None:
        actions = (1, 3, 2)
        committed = commit_procedure(self.task, actions)
        reached = replay_actions(self.task.origin, actions)
        swapped_target = observe_primitive_transition(reached, 0).after

        exact = execute_committed_procedure(self.task, committed, reached)
        wrong = execute_committed_procedure(
            self.task,
            committed,
            swapped_target,
        )

        self.assertTrue(exact.exact)
        self.assertFalse(wrong.exact)
        self.assertEqual(exact.reached_state, wrong.reached_state)
        self.assertEqual(exact.steps_executed, wrong.steps_executed)

    def test_empty_committed_procedure_can_verify_the_origin(self) -> None:
        committed = commit_procedure(self.task, ())
        result = execute_committed_procedure(
            self.task,
            committed,
            self.task.origin,
        )

        self.assertTrue(result.exact)
        self.assertEqual(result.reached_state, self.task.origin)
        self.assertEqual(result.steps_executed, 0)

    def test_invalid_or_overbudget_procedures_are_rejected_atomically(self) -> None:
        original_origin = self.task.origin
        for actions in (
            (0, 1, ACTION_COUNT),
            (0, 1, -1),
            (0, 1, True),
            tuple(range(9)),
        ):
            with self.subTest(actions=actions):
                with self.assertRaises((TypeError, ValueError)):
                    commit_procedure(self.task, actions)
                self.assertEqual(self.task.origin, original_origin)

        with self.assertRaises(TypeError):
            commit_procedure(self.task, (action for action in (0, 1)))  # type: ignore[arg-type]

        overbudget = CommittedProcedure(
            task_id=self.task.instance_id,
            actions=(0,) * (self.task.max_steps + 1),
        )
        with self.assertRaises(ValueError):
            execute_committed_procedure(
                self.task,
                overbudget,
                self.task.origin,
            )
        self.assertEqual(self.task.origin, original_origin)

    def test_mismatched_identity_invalid_target_and_forged_task_are_rejected(self) -> None:
        committed = commit_procedure(self.task, (0, 1))
        other = generate_reversible_transition_task(7332, max_steps=8)
        with self.assertRaises(ValueError):
            execute_committed_procedure(other, committed, other.origin)

        for target in ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 4)):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    execute_committed_procedure(self.task, committed, target)

        forged = replace(
            self.task,
            available_actions=(0, 1),
        )
        with self.assertRaises(ValueError):
            commit_procedure(forged, (0,))

        forged_origin = replace(
            self.task,
            origin=tuple(reversed(self.task.origin)),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            commit_procedure(forged_origin, ())

        forged_commitment = replace(
            self.task,
            generation_commitment="sha256:" + "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            commit_procedure(forged_commitment, ())


class NoSolverSurfaceTests(unittest.TestCase):
    def test_public_api_contains_physics_and_judging_but_no_solver(self) -> None:
        forbidden_fragments = (
            "solve",
            "solver",
            "path",
            "shortest",
            "distance",
            "hint",
            "optimal",
            "next_action",
        )
        public_names = tuple(world.__all__)
        for name in public_names:
            lowered = name.lower()
            self.assertFalse(
                any(fragment in lowered for fragment in forbidden_fragments),
                name,
            )

        expected_functions = {
            "commit_procedure",
            "execute_committed_procedure",
            "generate_reversible_transition_task",
            "observe_primitive_transition",
        }
        actual_functions = {
            name
            for name in public_names
            if inspect.isfunction(getattr(world, name, None))
        }
        self.assertEqual(actual_functions, expected_functions)
        self.assertFalse(
            set(forbidden_fragments)
            & {field.name for field in fields(ReversibleTransitionTask)}
        )


if __name__ == "__main__":
    unittest.main()
