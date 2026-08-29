from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
import inspect
import itertools
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from angler.procedures.records import ActionSchema  # noqa: E402
from experiments.evaluators import glyph_machine_trace_suite as suite  # noqa: E402
from experiments.evaluators.glyph_machine_trace_suite import (  # noqa: E402
    CommittedGlyphProcedure,
    commit_glyph_procedure,
    glyph_machine_mechanism_partition,
    judge_glyph_procedure_attempt,
    make_glyph_machine_control_stream,
    make_glyph_machine_trace_stream,
    score_glyph_procedure,
)


def _goal_state_digest(task: suite.PublicGlyphMachineTask) -> str:
    return next(
        state.digest for state in task.states if state.records == task.goal.required
    )


def _exact_public_actions(
    pair: suite.GeneratedGlyphMachineTask,
) -> tuple:
    task = pair.learner
    hidden = pair.hidden
    origin = hidden.state_digests.index(task.origin.digest)
    target = hidden.state_digests.index(_goal_state_digest(task))
    actions_by_digest = {action.digest: action for action in task.actions}
    for length in range(task.max_steps + 1):
        for action_indices in itertools.product(
            range(len(hidden.transition_rows)),
            repeat=length,
        ):
            current = origin
            for action_index in action_indices:
                current = hidden.transition_rows[action_index][current]
            if current == target:
                return tuple(
                    actions_by_digest[hidden.action_digests[index]].ground()
                    for index in action_indices
                )
    raise AssertionError("generated public case has no bounded exact procedure")


class GlyphMachineMechanismTests(unittest.TestCase):
    def test_universe_count_strata_and_opened_partitions_are_exact(self) -> None:
        universe = suite._semantic_machine_universe()
        strata = Counter(
            (state_count, len(rows)) for state_count, rows in universe
        )

        self.assertEqual(len(universe), 116)
        self.assertEqual(len(set(universe)), 116)
        self.assertEqual(
            strata,
            {
                (2, 1): 1,
                (3, 1): 1,
                (3, 2): 3,
                (3, 3): 3,
                (4, 1): 1,
                (4, 2): 14,
                (4, 3): 93,
            },
        )
        train = glyph_machine_mechanism_partition("train")
        development = glyph_machine_mechanism_partition("development")
        final = glyph_machine_mechanism_partition("final")
        sealed = tuple(
            suite._mechanism_commitment(value)
            for value in suite._semantic_partition("sealed")
        )
        groups = (train, development, final, sealed)

        self.assertEqual(tuple(map(len, groups)), (64, 16, 16, 20))
        self.assertEqual(len(set().union(*(set(group) for group in groups))), 116)
        for index, left in enumerate(groups):
            self.assertTrue(
                all(set(left).isdisjoint(right) for right in groups[index + 1 :])
            )
        self.assertEqual(
            tuple(value[0] for value in suite._semantic_partition("train")).count(2),
            1,
        )
        self.assertEqual(
            tuple(value[0] for value in suite._semantic_partition("development")).count(3),
            2,
        )
        self.assertEqual(
            tuple(value[0] for value in suite._semantic_partition("final")).count(3),
            2,
        )

    def test_canonicalization_erases_state_and_action_renaming(self) -> None:
        mechanism = next(
            value
            for value in suite._semantic_machine_universe()
            if value[0] == 4 and len(value[1]) == 3
        )
        state_count, rows = mechanism
        renaming = (2, 0, 3, 1)
        inverse = [0] * state_count
        for source, target in enumerate(renaming):
            inverse[target] = source
        renamed = tuple(
            reversed(
                tuple(
                    tuple(
                        renaming[row[inverse[public_state]]]
                        for public_state in range(state_count)
                    )
                    for row in rows
                )
            )
        )

        self.assertEqual(
            suite._canonicalize_transition_rows(state_count, renamed),
            rows,
        )
        self.assertEqual(
            suite._canonicalize_transition_rows(state_count, tuple(reversed(rows))),
            rows,
        )


class GlyphMachineStreamTests(unittest.TestCase):
    def test_stream_replays_and_surface_seed_changes_only_opaque_projection(self) -> None:
        commitment = glyph_machine_mechanism_partition("development")[5]
        first = make_glyph_machine_trace_stream(
            98_101,
            surface_seed=7_001,
            supports=2,
            queries=2,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )
        replay = make_glyph_machine_trace_stream(
            98_101,
            surface_seed=7_001,
            supports=2,
            queries=2,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )
        renamed = make_glyph_machine_trace_stream(
            98_101,
            surface_seed=7_002,
            supports=2,
            queries=2,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )

        self.assertEqual(first, replay)
        self.assertEqual(first.mechanism_commitment, renamed.mechanism_commitment)
        self.assertNotEqual(
            first.queries[0].learner.to_canonical(),
            renamed.queries[0].learner.to_canonical(),
        )
        for left, right in zip(first.queries, renamed.queries, strict=True):
            self.assertEqual(left.hidden.transition_rows, right.hidden.transition_rows)
            self.assertEqual(
                left.hidden.state_digests.index(left.learner.origin.digest),
                right.hidden.state_digests.index(right.learner.origin.digest),
            )
            self.assertEqual(
                left.hidden.state_digests.index(_goal_state_digest(left.learner)),
                right.hidden.state_digests.index(_goal_state_digest(right.learner)),
            )
        first_states = {
            state.records[0].arguments[0] for state in first.queries[0].learner.states
        }
        renamed_states = {
            state.records[0].arguments[0]
            for state in renamed.queries[0].learner.states
        }
        self.assertTrue(first_states.isdisjoint(renamed_states))
        self.assertTrue(
            {action.name for action in first.queries[0].learner.actions}.isdisjoint(
                action.name for action in renamed.queries[0].learner.actions
            )
        )

    def test_public_projection_contains_only_typed_observations_and_goals(self) -> None:
        stream = make_glyph_machine_trace_stream(
            98_103,
            surface_seed=7_103,
            supports=3,
            queries=2,
            mechanism_commitment=glyph_machine_mechanism_partition("final")[3],
            mechanism_partition="final",
        )

        self.assertTrue(all(pair.learner.observations for pair in stream.supports))
        self.assertTrue(all(not pair.learner.observations for pair in stream.queries))
        self.assertEqual(
            {field.name for field in fields(suite.PublicGlyphMachineTask)},
            {"states", "actions", "observations", "origin", "goal", "max_steps"},
        )
        for pair in (*stream.supports, *stream.queries):
            task = pair.learner
            self.assertIn(len(task.states), (2, 3, 4))
            self.assertIn(len(task.actions), (1, 2, 3))
            self.assertIn(task.max_steps, (1, 2, 3, 4))
            self.assertIn(task.origin, task.states)
            self.assertTrue(task.goal.exact)
            self.assertIn(_goal_state_digest(task), {state.digest for state in task.states})
            for observation in task.observations:
                self.assertTrue(observation.transitions)
                current = observation.initial
                for transition in observation.transitions:
                    self.assertEqual(transition.before, current)
                    self.assertIn(transition.before, task.states)
                    self.assertIn(transition.after, task.states)
                    self.assertIn(transition.action.schema, task.actions)
                    current = transition.after

            public_json = json.dumps(task.to_canonical(), sort_keys=True)
            for forbidden in (
                "transition_rows",
                "mechanism_commitment",
                "public_digest",
                "state_digests",
                "action_digests",
                "target_procedure",
                "instance_id",
                "surface_seed",
                "generator_seed",
            ):
                self.assertNotIn(forbidden, public_json)

    def test_partition_binding_and_stream_visibility_are_enforced(self) -> None:
        train_commitment = glyph_machine_mechanism_partition("train")[0]
        with self.assertRaisesRegex(ValueError, "outside"):
            make_glyph_machine_trace_stream(
                98_105,
                mechanism_commitment=train_commitment,
                mechanism_partition="final",
            )
        stream = make_glyph_machine_trace_stream(
            98_107,
            mechanism_commitment=glyph_machine_mechanism_partition("train")[1],
            mechanism_partition="train",
        )
        with self.assertRaisesRegex(ValueError, "must not replay"):
            replace(
                stream,
                queries=(
                    stream.supports[0],
                    *stream.queries[1:],
                ),
            )


class GlyphMachineScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_glyph_machine_trace_stream(
            98_201,
            surface_seed=8_201,
            supports=2,
            queries=2,
            mechanism_commitment=glyph_machine_mechanism_partition("final")[7],
            mechanism_partition="final",
        )

    def test_correct_wrong_and_cross_task_scoring_are_scalar_only(self) -> None:
        pair = self.stream.queries[0]
        exact_actions = _exact_public_actions(pair)
        exact = commit_glyph_procedure(
            pair.learner,
            exact_actions,
            stopped=len(exact_actions) < pair.learner.max_steps,
        )
        wrong = commit_glyph_procedure(pair.learner, (), stopped=True)

        self.assertEqual(score_glyph_procedure(pair.learner, pair.hidden, exact), 1.0)
        self.assertEqual(score_glyph_procedure(pair.learner, pair.hidden, wrong), 0.0)
        self.assertIsInstance(
            score_glyph_procedure(pair.learner, pair.hidden, exact),
            float,
        )
        other = self.stream.queries[1]
        with self.assertRaisesRegex(ValueError, "do not match"):
            score_glyph_procedure(other.learner, other.hidden, exact)
        tampered = replace(pair.learner, origin=_goal_state_for_test(pair.learner))
        with self.assertRaisesRegex(ValueError, "do not match"):
            score_glyph_procedure(tampered, pair.hidden, exact)

    def test_budget_stop_and_declared_action_constraints_are_rechecked(self) -> None:
        pair = self.stream.queries[0]
        task = pair.learner
        repeated = (task.actions[0].ground(),) * (task.max_steps + 1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            commit_glyph_procedure(task, repeated, stopped=False)
        with self.assertRaisesRegex(ValueError, "explicit STOP"):
            commit_glyph_procedure(task, (), stopped=False)
        undeclared = ActionSchema(
            "angler.glyph_machine.action_ffffffffffffffffffff",
            (),
        ).ground()
        with self.assertRaisesRegex(ValueError, "undeclared"):
            commit_glyph_procedure(task, (undeclared,), stopped=True)

        forged = CommittedGlyphProcedure(
            public_digest=suite._public_digest(task),
            actions=repeated,
            stopped=False,
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            score_glyph_procedure(task, pair.hidden, forged)

    def test_evaluator_owned_judge_and_public_only_controls_remain_scalar(self) -> None:
        pair = self.stream.queries[0]
        exact_actions = _exact_public_actions(pair)
        exact = commit_glyph_procedure(
            pair.learner,
            exact_actions,
            stopped=len(exact_actions) < pair.learner.max_steps,
        )
        value = judge_glyph_procedure_attempt(pair, exact)
        self.assertIsInstance(value, float)
        self.assertEqual(value, 1.0)

        correct = make_glyph_machine_control_stream(self.stream, "correct")
        no_trace = make_glyph_machine_control_stream(self.stream, "no_trace")
        wrong_trace = make_glyph_machine_control_stream(self.stream, "wrong_trace")
        self.assertIs(correct, self.stream)
        self.assertTrue(
            all(not item.learner.observations for item in no_trace.supports)
        )
        self.assertTrue(
            all(item.learner.observations for item in wrong_trace.supports)
        )
        self.assertEqual(
            tuple(item.learner for item in no_trace.queries),
            tuple(item.learner for item in wrong_trace.queries),
        )
        for original, altered in zip(
            self.stream.supports,
            wrong_trace.supports,
            strict=True,
        ):
            self.assertEqual(original.learner.states, altered.learner.states)
            self.assertEqual(original.learner.actions, altered.learner.actions)
            self.assertEqual(original.learner.origin, altered.learner.origin)
            self.assertEqual(original.learner.goal, altered.learner.goal)
            self.assertEqual(original.learner.max_steps, altered.learner.max_steps)
            original_after = tuple(
                transition.after
                for trace in original.learner.observations
                for transition in trace.transitions
            )
            altered_after = tuple(
                transition.after
                for trace in altered.learner.observations
                for transition in trace.transitions
            )
            self.assertEqual(len(original_after), len(altered_after))
            self.assertTrue(
                all(left != right for left, right in zip(original_after, altered_after))
            )


def _goal_state_for_test(task: suite.PublicGlyphMachineTask):
    return next(state for state in task.states if state.records == task.goal.required)


class GlyphMachineNoSolverSurfaceTests(unittest.TestCase):
    def test_public_api_exposes_generation_commit_and_scalar_judging_only(self) -> None:
        forbidden = (
            "solve",
            "solver",
            "shortest",
            "distance",
            "next_action",
            "target_procedure",
            "transition_table",
        )
        for name in suite.__all__:
            self.assertFalse(any(fragment in name.lower() for fragment in forbidden))
        public_functions = {
            name
            for name in suite.__all__
            if inspect.isfunction(getattr(suite, name, None))
        }
        self.assertEqual(
            public_functions,
            {
                "commit_glyph_procedure",
                "glyph_machine_mechanism_partition",
                "judge_glyph_procedure_attempt",
                "make_glyph_machine_control_stream",
                "make_glyph_machine_trace_stream",
                "score_glyph_procedure",
            },
        )
        self.assertNotIn("_HiddenGlyphMachineSolution", suite.__all__)


if __name__ == "__main__":
    unittest.main()
