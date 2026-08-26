"""Tests for the original Project Angler relational procedural world."""

from __future__ import annotations

from dataclasses import fields
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.worlds import (  # noqa: E402 - explicit src path for local tests
    HiddenOrderSolution,
    LearnerTask,
    OutcomeFeedback,
    generate_relational_task,
    make_held_out_variant,
    verify_final_answer,
)


def answer_text(order: tuple[str, ...]) -> str:
    return ", ".join(order)


def structural_pattern(instance) -> tuple[tuple[int, int], ...]:
    rank = {
        symbol: index
        for index, symbol in enumerate(instance.hidden.ordered_symbols)
    }
    return tuple(
        (rank[constraint.earlier], rank[constraint.later])
        for constraint in instance.learner.constraints
    )


class GenerationTests(unittest.TestCase):
    def test_seed_replay_is_exact_and_visible_state_has_no_solution_field(self) -> None:
        first = generate_relational_task(1729, item_count=5)
        second = generate_relational_task(1729, item_count=5)
        self.assertEqual(first.learner, second.learner)
        self.assertEqual(first.hidden, second.hidden)
        self.assertEqual(first.learner.instance_id, first.hidden.instance_id)
        self.assertNotEqual(
            first.learner,
            generate_relational_task(1730, item_count=5).learner,
        )

        visible_fields = {field.name for field in fields(LearnerTask)}
        self.assertFalse(
            visible_fields
            & {"answer", "solution", "reasoning", "strategy", "plan", "seed"}
        )
        self.assertEqual(
            {field.name for field in fields(HiddenOrderSolution)},
            {"instance_id", "ordered_symbols", "generator_seed"},
        )
        self.assertNotIn(str(first.hidden.generator_seed), first.learner.prompt)
        self.assertNotIn(answer_text(first.hidden.ordered_symbols), first.learner.prompt)
        self.assertNotEqual(first.learner.symbols, first.hidden.ordered_symbols)

    def test_generated_constraints_have_one_accepted_permutation(self) -> None:
        instance = generate_relational_task(81, item_count=5)
        accepted = [
            candidate
            for candidate in itertools.permutations(instance.learner.symbols)
            if verify_final_answer(instance.learner, candidate).correct
        ]
        self.assertEqual(accepted, [instance.hidden.ordered_symbols])

    def test_item_count_is_bounded(self) -> None:
        for invalid in (3, 9):
            with self.subTest(item_count=invalid):
                with self.assertRaises(ValueError):
                    generate_relational_task(1, item_count=invalid)


class HeldOutVariantTests(unittest.TestCase):
    def test_variant_renames_and_reorders_but_preserves_relations(self) -> None:
        source = generate_relational_task(44, item_count=6)
        variant = make_held_out_variant(source, seed=9901)

        self.assertTrue(
            set(source.learner.symbols).isdisjoint(variant.learner.symbols)
        )
        self.assertNotEqual(
            structural_pattern(source),
            structural_pattern(variant),
        )
        self.assertEqual(
            sorted(structural_pattern(source)),
            sorted(structural_pattern(variant)),
        )
        self.assertTrue(
            verify_final_answer(
                variant.learner,
                answer_text(variant.hidden.ordered_symbols),
            ).correct
        )
        self.assertEqual(
            variant,
            make_held_out_variant(source, seed=9901),
        )


class OutcomeVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_relational_task(31337, item_count=5)

    def test_correct_wrong_and_malformed_answers_return_only_outcomes(self) -> None:
        correct = verify_final_answer(
            self.instance.learner,
            answer_text(self.instance.hidden.ordered_symbols),
        )
        wrong = verify_final_answer(
            self.instance.learner,
            tuple(reversed(self.instance.hidden.ordered_symbols)),
        )
        malformed = verify_final_answer(self.instance.learner, "not, enough")
        trailing_delimiter = verify_final_answer(
            self.instance.learner,
            answer_text(self.instance.hidden.ordered_symbols) + ",",
        )

        self.assertEqual(
            correct,
            OutcomeFeedback(
                task_id=self.instance.learner.instance_id,
                disposition="VALID_RESULT",
                correct=True,
                score=1,
                code=None,
                violated_visible_constraints=(),
            ),
        )
        self.assertEqual((wrong.correct, wrong.score, wrong.code), (False, 0, "ORDER_INCORRECT"))
        self.assertTrue(wrong.violated_visible_constraints)
        self.assertTrue(
            all(
                1 <= index <= len(self.instance.learner.constraints)
                for index in wrong.violated_visible_constraints
            )
        )
        self.assertEqual(
            (malformed.disposition, malformed.score, malformed.code),
            ("INVALID_ATTEMPT", 0, "INVALID_FINAL_ANSWER"),
        )
        self.assertEqual(trailing_delimiter.disposition, "INVALID_ATTEMPT")

        feedback_fields = {field.name for field in fields(OutcomeFeedback)}
        self.assertEqual(
            feedback_fields,
            {
                "task_id",
                "disposition",
                "correct",
                "score",
                "code",
                "violated_visible_constraints",
            },
        )
        for feedback in (correct, wrong, malformed, trailing_delimiter):
            rendered = repr(feedback)
            self.assertNotIn(answer_text(self.instance.hidden.ordered_symbols), rendered)
            self.assertNotIn("reason", rendered.lower())
            self.assertNotIn("strategy", rendered.lower())


if __name__ == "__main__":
    unittest.main()
