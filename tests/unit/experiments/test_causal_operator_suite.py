"""Evaluator invariants for held-out cross-domain causal operators."""

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

from experiments.evaluators import causal_operator_suite as suite  # noqa: E402
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402


_PLACEMENT_PREDICATE = {
    "tokens": tokens.TOKEN_IN,
    "files": files.FILE_AT,
    "boxes": boxes.ITEM_IN,
}


def _test_oracle_actions(
    challenge: suite.OperatorChallenge,
):
    """Construct the generated relocation sequence for test assertions only."""

    predicate = _PLACEMENT_PREDICATE[challenge.domain]
    entity_by_position = {
        record.arguments[1]: record.arguments[0]
        for record in challenge.origin.records
        if record.predicate == predicate
    }
    schema = challenge.allowed_action_schemas[0]
    return tuple(
        schema.ground(
            entity_by_position[f"position_{index}"],
            f"position_{index}",
            f"position_{index + 1}",
        )
        for index in reversed(range(challenge.maximum_steps))
    )


class HeldOutCausalOperatorSuiteTests(unittest.TestCase):
    def test_generation_is_replayable_cross_domain_and_compositional(self) -> None:
        first = suite.make_heldout_operator_suite(9201)
        replay = suite.make_heldout_operator_suite(9201)
        changed = suite.make_heldout_operator_suite(9202)

        self.assertEqual(first, replay)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({item.case_id for item in first}), 6)
        self.assertTrue(
            {item.case_id for item in first}.isdisjoint(
                {item.case_id for item in changed}
            )
        )
        for domain in suite.SUPPORTED_DOMAINS:
            selected = tuple(item for item in first if item.domain == domain)
            self.assertEqual(
                tuple(item.maximum_steps for item in selected),
                (
                    suite.SINGLE_OPERATOR_STEPS,
                    suite.COMPOSED_OPERATOR_STEPS,
                ),
            )
            self.assertEqual(
                selected[0].allowed_action_schemas[0].namespace,
                selected[0].origin.namespace,
            )

        training_bindings = {"amber", "blue", "cyan", "gold"}
        heldout_bindings = {
            record.arguments[0]
            for challenge in first
            for record in challenge.origin.records
            if record.predicate == _PLACEMENT_PREDICATE[challenge.domain]
        }
        self.assertTrue(heldout_bindings.isdisjoint(training_bindings))
        self.assertTrue(
            all(value.startswith("heldout_") for value in heldout_bindings)
        )

    def test_public_challenge_contains_no_route_or_ground_actions(self) -> None:
        challenge = suite.make_heldout_operator_suite(9203)[0]
        self.assertEqual(
            {field.name for field in fields(suite.OperatorChallenge)},
            {
                "allowed_action_schemas",
                "case_id",
                "domain",
                "goal",
                "maximum_steps",
                "origin",
            },
        )
        forbidden = (
            "path",
            "route",
            "solution",
            "predecessor",
            "next_action",
            "optimal_action",
            "trace",
        )
        self.assertFalse(any(hasattr(challenge, name) for name in forbidden))
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in suite.__all__
                for fragment in forbidden[:6]
            )
        )

    def test_committed_sequences_execute_once_through_each_domain_boundary(self) -> None:
        challenges = suite.make_heldout_operator_suite(9204)
        executor_names = {
            "tokens": (tokens, "execute_token_action"),
            "files": (files, "execute_file_action"),
            "boxes": (boxes, "execute_box_action"),
        }
        for challenge in challenges:
            actions = _test_oracle_actions(challenge)
            commitment = suite.commit_action_sequence(challenge, actions)
            module, name = executor_names[challenge.domain]
            original = getattr(module, name)
            with patch.object(module, name, wraps=original) as executed:
                result = suite.evaluate_committed_sequence(
                    challenge,
                    commitment,
                )

            self.assertTrue(result.success)
            self.assertEqual(executed.call_count, challenge.maximum_steps)
            self.assertEqual(result.tool_calls, challenge.maximum_steps)
            self.assertEqual(result.applied_actions, challenge.maximum_steps)
            self.assertEqual(len(result.trace.transitions), result.tool_calls)
            self.assertRegex(result.trace_digest, r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(result.digest, r"^sha256:[0-9a-f]{64}$")

    def test_blocked_submission_is_feedback_not_a_hidden_retry(self) -> None:
        challenge = next(
            item
            for item in suite.make_heldout_operator_suite(9205)
            if item.domain == "tokens"
            and item.maximum_steps == suite.SINGLE_OPERATOR_STEPS
        )
        placement = next(
            record
            for record in challenge.origin.records
            if record.predicate == tokens.TOKEN_IN
        )
        token, position = placement.arguments
        blocked = tokens.MOVE_TOKEN.ground(token, position, position)
        commitment = suite.commit_action_sequence(challenge, (blocked,))
        result = suite.evaluate_committed_sequence(challenge, commitment)

        self.assertFalse(result.success)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.applied_actions, 0)
        self.assertEqual(result.final_state, challenge.origin)

    def test_commitment_validation_rejects_oracle_boundary_bypasses(self) -> None:
        challenge = suite.make_heldout_operator_suite(9206)[0]
        with self.assertRaisesRegex(ValueError, "execution ceiling"):
            suite.commit_action_sequence(
                challenge,
                _test_oracle_actions(challenge)
                + (_test_oracle_actions(challenge)[0],),
            )
        with self.assertRaisesRegex(ValueError, "schema not allowed"):
            suite.commit_action_sequence(
                challenge,
                (files.RELOCATE_FILE.ground("x", "y", "z"),),
            )

        other = suite.make_heldout_operator_suite(9207)[0]
        commitment = suite.commit_action_sequence(challenge, ())
        with self.assertRaisesRegex(ValueError, "different challenge"):
            suite.evaluate_committed_sequence(other, commitment)

    def test_summary_reports_success_and_tool_calls_by_domain(self) -> None:
        challenges = suite.make_heldout_operator_suite(9208)
        results = []
        for index, challenge in enumerate(challenges):
            actions = () if index == len(challenges) - 1 else _test_oracle_actions(
                challenge
            )
            commitment = suite.commit_action_sequence(challenge, actions)
            results.append(
                suite.evaluate_committed_sequence(challenge, commitment)
            )

        summary = suite.summarize_operator_results(results)
        self.assertEqual(summary.attempts, 6)
        self.assertEqual(summary.successes, 5)
        self.assertEqual(summary.total_tool_calls, 14)
        self.assertEqual(tuple(item.domain for item in summary.by_domain), ("tokens", "files", "boxes"))
        self.assertEqual(tuple(item.attempts for item in summary.by_domain), (2, 2, 2))
        self.assertEqual(tuple(item.successes for item in summary.by_domain), (2, 2, 1))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            suite.summarize_operator_results(())
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            suite.summarize_operator_results((results[0], results[0]))

    def test_generation_parameters_fail_closed(self) -> None:
        for invalid in (True, 1.5, "seed"):
            with self.subTest(seed=invalid):
                with self.assertRaises(TypeError):
                    suite.make_heldout_operator_suite(invalid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "at least two"):
            suite.make_heldout_operator_suite(9209, cases_per_domain=1)


if __name__ == "__main__":
    unittest.main()
