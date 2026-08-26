"""Evaluator invariants for the bidirectional procedure experiment."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.worlds.reversible_transition_world import (  # noqa: E402
    commit_procedure as world_commit_procedure,
    execute_committed_procedure as world_execute_committed_procedure,
)
from experiments.evaluators import bidirectional_procedure_suite as suite  # noqa: E402
from experiments.runners.phase3_bidirectional_procedure import (  # noqa: E402
    RunProfile,
    _parse_goal_json,
    build_experience_corpus,
)


def _test_oracle_actions(
    origin: tuple[int, ...],
    goal: tuple[int, ...],
) -> tuple[int, ...]:
    """Construct a shortest adjacent-swap route for test assertions only."""

    state = list(origin)
    actions: list[int] = []
    for goal_index, token in enumerate(goal):
        current_index = state.index(token, goal_index)
        while current_index > goal_index:
            action = current_index - 1
            state[action], state[action + 1] = state[action + 1], state[action]
            actions.append(action)
            current_index -= 1
    if tuple(state) != goal:
        raise AssertionError("test oracle failed to reach the supplied goal")
    return tuple(actions)


class HeldOutProcedureSuiteTests(unittest.TestCase):
    def test_seeded_generation_is_unique_replayable_and_exactly_stratified(self) -> None:
        first = suite.make_heldout_procedure_suite(
            8101,
            inversion_distances=(1, 3, 5),
            cases_per_distance=3,
            max_steps=8,
        )
        replay = suite.make_heldout_procedure_suite(
            8101,
            inversion_distances=(1, 3, 5),
            cases_per_distance=3,
            max_steps=8,
        )
        changed = suite.make_heldout_procedure_suite(
            8102,
            inversion_distances=(1, 3, 5),
            cases_per_distance=3,
            max_steps=8,
        )

        self.assertEqual(first, replay)
        self.assertEqual(len(first), 9)
        self.assertEqual(len({case.case_id for case in first}), len(first))
        self.assertEqual(
            len({(case.origin, case.goal) for case in first}),
            len(first),
        )
        self.assertTrue(
            {case.case_id for case in first}.isdisjoint(
                {case.case_id for case in changed}
            )
        )

        observed_distances = [
            len(_test_oracle_actions(case.origin, case.goal))
            for case in first
        ]
        self.assertEqual(observed_distances.count(1), 3)
        self.assertEqual(observed_distances.count(3), 3)
        self.assertEqual(observed_distances.count(5), 3)
        for case in first:
            self.assertNotEqual(case.origin, case.goal)
            self.assertEqual(set(case.origin), set(case.goal))
            self.assertEqual(case.max_steps, 8)

    def test_learner_projection_contains_no_route_or_distance_hint(self) -> None:
        challenge = suite.make_heldout_procedure_suite(
            8103,
            inversion_distances=(4,),
            cases_per_distance=1,
            max_steps=8,
        )[0]

        self.assertEqual(
            {field.name for field in fields(suite.ProcedureChallenge)},
            {"case_id", "task", "goal"},
        )
        public_names = set(suite.__all__)
        forbidden = (
            "path",
            "route",
            "solution",
            "predecessor",
            "next_action",
            "optimal_action",
        )
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in public_names
                for fragment in forbidden
            )
        )
        self.assertFalse(
            any(hasattr(challenge, name) for name in forbidden)
        )

    def test_evaluation_delegates_commit_and_terminal_execution_to_world(self) -> None:
        challenge = suite.make_heldout_procedure_suite(
            8104,
            inversion_distances=(6,),
            cases_per_distance=1,
            max_steps=10,
        )[0]
        actions = _test_oracle_actions(challenge.origin, challenge.goal)

        with (
            patch.object(
                suite,
                "commit_procedure",
                wraps=world_commit_procedure,
            ) as committed,
            patch.object(
                suite,
                "execute_committed_procedure",
                wraps=world_execute_committed_procedure,
            ) as executed,
        ):
            result = suite.evaluate_procedure(
                challenge,
                actions,
                expansions=37,
            )

        committed.assert_called_once_with(challenge.task, actions)
        self.assertEqual(executed.call_count, 1)
        self.assertIs(executed.call_args.args[0], challenge.task)
        self.assertEqual(executed.call_args.args[2], challenge.goal)
        self.assertTrue(result.exact)
        self.assertEqual(result.reached_state, challenge.goal)
        self.assertEqual(result.inversion_distance, 6)
        self.assertEqual(result.steps_executed, len(actions))
        self.assertEqual(result.expansions, 37)

    def test_summary_aggregates_exact_outcomes_and_expansions_by_distance(self) -> None:
        challenges = suite.make_heldout_procedure_suite(
            8105,
            inversion_distances=(1, 2),
            cases_per_distance=2,
            max_steps=6,
        )
        expansions = (10, 20, 30, 40)
        results = []
        for index, (challenge, expanded) in enumerate(
            zip(challenges, expansions, strict=True)
        ):
            actions = (
                ()
                if index == len(challenges) - 1
                else _test_oracle_actions(challenge.origin, challenge.goal)
            )
            results.append(
                suite.evaluate_procedure(
                    challenge,
                    actions,
                    expansions=expanded,
                )
            )

        summary = suite.summarize_results(results)

        self.assertEqual(summary.attempts, 4)
        self.assertEqual(summary.exact_successes, 3)
        self.assertEqual(summary.exact_success_rate, 0.75)
        self.assertEqual(summary.total_expansions, 100)
        self.assertEqual(summary.mean_expansions, 25.0)
        self.assertEqual(
            tuple(item.inversion_distance for item in summary.by_distance),
            (1, 2),
        )
        self.assertEqual(summary.by_distance[0].exact_successes, 2)
        self.assertEqual(summary.by_distance[0].total_expansions, 30)
        self.assertEqual(summary.by_distance[0].mean_expansions, 15.0)
        self.assertEqual(summary.by_distance[1].exact_successes, 1)
        self.assertEqual(summary.by_distance[1].total_expansions, 70)
        self.assertEqual(summary.by_distance[1].mean_expansions, 35.0)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            suite.summarize_results(())
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            suite.summarize_results((results[0], results[0]))

    def test_generation_and_expansion_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            suite.make_heldout_procedure_suite(
                8106,
                inversion_distances=(2, 2),
            )
        with self.assertRaisesRegex(ValueError, "largest inversion distance"):
            suite.make_heldout_procedure_suite(
                8106,
                inversion_distances=(7,),
                max_steps=6,
            )
        challenge = suite.make_heldout_procedure_suite(
            8106,
            inversion_distances=(2,),
            cases_per_distance=1,
        )[0]
        for invalid in (-1, True, 1.5):
            with self.subTest(expansions=invalid):
                with self.assertRaises(ValueError):
                    suite.evaluate_procedure(
                        challenge,
                        (),
                        expansions=invalid,  # type: ignore[arg-type]
                    )


class ProcedureRunnerBoundaryTests(unittest.TestCase):
    def test_experience_is_unique_observed_transitions_and_future_states(self) -> None:
        profile = RunProfile(
            hidden_width=24,
            action_width=8,
            unique_transitions=120,
            trajectory_count=80,
            trajectory_minimum=2,
            trajectory_maximum=6,
            optimizer_steps=1,
            batch_size=8,
            learning_rate=1e-3,
            cases_per_distance=1,
            evaluation_distances=(2,),
            maximum_steps=6,
            maximum_expansions=50,
            actions_per_state=2,
            qwen_cases=1,
        )
        corpus = build_experience_corpus(profile, 8201)

        self.assertEqual(len(corpus.transition_keys), 120)
        self.assertEqual(corpus.transition_states.shape[0], 120)
        self.assertGreater(corpus.origins.shape[0], 80)
        self.assertTrue(corpus.auxiliary_transition_keys <= corpus.transition_keys)
        self.assertTrue(bool((corpus.horizons >= 1).all().item()))
        self.assertTrue(bool((corpus.horizons <= profile.maximum_steps).all().item()))

        states = corpus.transition_states.argmax(-1).tolist()
        actions = corpus.transition_actions.tolist()
        next_states = corpus.transition_next_states.argmax(-1).tolist()
        for state, action, observed_next in zip(
            states,
            actions,
            next_states,
            strict=True,
        ):
            expected = list(state)
            expected[action], expected[action + 1] = (
                expected[action + 1],
                expected[action],
            )
            self.assertEqual(observed_next, expected)

        excluded = next(iter(corpus.future_goal_pairs))
        rebuilt = build_experience_corpus(
            profile,
            8201,
            excluded_goal_pairs=(excluded,),
        )
        self.assertNotIn(excluded, rebuilt.future_goal_pairs)
        self.assertEqual(rebuilt.transition_keys, corpus.transition_keys)

    def test_qwen_goal_parser_accepts_only_one_complete_permutation(self) -> None:
        self.assertEqual(
            _parse_goal_json('{"goal":[5,4,3,2,1,0]}'),
            (5, 4, 3, 2, 1, 0),
        )
        invalid = (
            "[5,4,3,2,1,0]",
            '{"goal":[0,1,2,3,4,4]}',
            '{"goal":[0,1,2,3,4,5],"actions":[0]}',
            '{"goal":[0,1,2,3,4,true]}',
            "not json",
        )
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(_parse_goal_json(value))


if __name__ == "__main__":
    unittest.main()
