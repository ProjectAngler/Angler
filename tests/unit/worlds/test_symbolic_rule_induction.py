"""Tests for the symbolic shared-permutation induction world."""

from __future__ import annotations

from dataclasses import fields, replace
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.worlds import (  # noqa: E402 - explicit src path for local tests
    HiddenSymbolicRuleSolution,
    SymbolicRuleFeedback,
    SymbolicRuleLearnerTask,
    generate_symbolic_rule_task,
    verify_symbolic_rule_answer,
)


class SymbolicRuleGenerationTests(unittest.TestCase):
    def test_generation_is_seeded_replayable_and_unique(self) -> None:
        first = generate_symbolic_rule_task(
            4101,
            item_count=6,
            demonstration_count=4,
            surface_seed=91,
        )
        replay = generate_symbolic_rule_task(
            4101,
            item_count=6,
            demonstration_count=4,
            surface_seed=91,
        )
        different = generate_symbolic_rule_task(
            4102,
            item_count=6,
            demonstration_count=4,
            surface_seed=91,
        )

        self.assertEqual(first.learner, replay.learner)
        self.assertEqual(first.hidden, replay.hidden)
        self.assertNotEqual(first.learner.instance_id, different.learner.instance_id)
        self.assertNotEqual(first.hidden.target_order, different.hidden.target_order)
        self.assertEqual(first.learner.instance_id, first.hidden.instance_id)

    def test_every_demonstration_and_query_uses_fresh_symbols(self) -> None:
        instance = generate_symbolic_rule_task(
            8128,
            item_count=7,
            demonstration_count=4,
        )
        groups = [
            demonstration.input_symbols
            for demonstration in instance.learner.demonstrations
        ] + [instance.learner.query_symbols]

        flattened = tuple(symbol for group in groups for symbol in group)
        self.assertEqual(len(flattened), len(set(flattened)))
        for left, right in itertools.combinations(groups, 2):
            self.assertTrue(set(left).isdisjoint(right))
        for demonstration in instance.learner.demonstrations:
            self.assertEqual(
                set(demonstration.input_symbols),
                set(demonstration.output_symbols),
            )

    def test_target_and_normalized_permutation_are_withheld(self) -> None:
        instance = generate_symbolic_rule_task(991, item_count=6)
        visible_fields = {field.name for field in fields(SymbolicRuleLearnerTask)}
        hidden_fields = {field.name for field in fields(HiddenSymbolicRuleSolution)}

        self.assertFalse(
            visible_fields
            & {
                "target",
                "target_order",
                "permutation",
                "position_permutation",
                "seed",
                "generator_seed",
                "surface_seed",
            }
        )
        self.assertEqual(
            hidden_fields,
            {
                "instance_id",
                "position_permutation",
                "target_order",
                "generator_seed",
                "surface_seed",
            },
        )
        rendered_target = ", ".join(instance.hidden.target_order)
        self.assertNotIn(rendered_target, instance.learner.prompt)
        self.assertNotIn("position_permutation", repr(instance.learner))
        self.assertNotIn(rendered_target, repr(instance.hidden))

    def test_shared_permutation_is_arbitrary_and_never_identity(self) -> None:
        observed: set[tuple[int, ...]] = set()
        for seed in range(40):
            instance = generate_symbolic_rule_task(
                seed,
                item_count=6,
                demonstration_count=3,
            )
            permutation = instance.hidden.position_permutation
            observed.add(permutation)
            self.assertEqual(sorted(permutation), list(range(6)))
            self.assertNotEqual(permutation, tuple(range(6)))
            for demonstration in instance.learner.demonstrations:
                self.assertEqual(
                    demonstration.output_symbols,
                    tuple(
                        demonstration.input_symbols[position]
                        for position in permutation
                    ),
                )
            self.assertEqual(
                instance.hidden.target_order,
                tuple(instance.learner.query_symbols[position] for position in permutation),
            )

        self.assertGreater(len(observed), 30)
        rotations = {
            tuple(range(offset, 6)) + tuple(range(offset))
            for offset in range(1, 6)
        }
        self.assertTrue(observed - rotations - {tuple(reversed(range(6)))})

    def test_structural_bounds_are_enforced(self) -> None:
        for item_count in (3, 8):
            with self.subTest(item_count=item_count):
                with self.assertRaises(ValueError):
                    generate_symbolic_rule_task(1, item_count=item_count)
        for demonstration_count in (1, 5):
            with self.subTest(demonstration_count=demonstration_count):
                with self.assertRaises(ValueError):
                    generate_symbolic_rule_task(
                        1,
                        demonstration_count=demonstration_count,
                    )

    def test_evaluator_can_hold_one_sealed_permutation_across_fresh_symbols(self) -> None:
        permutation = (2, 4, 1, 0, 3)
        first = generate_symbolic_rule_task(
            101,
            position_permutation=permutation,
        )
        second = generate_symbolic_rule_task(
            202,
            position_permutation=permutation,
        )

        self.assertEqual(first.hidden.position_permutation, permutation)
        self.assertEqual(second.hidden.position_permutation, permutation)
        self.assertNotEqual(first.learner.query_symbols, second.learner.query_symbols)
        for instance in (first, second):
            for demonstration in instance.learner.demonstrations:
                self.assertEqual(
                    demonstration.output_symbols,
                    tuple(
                        demonstration.input_symbols[position]
                        for position in permutation
                    ),
                )

    def test_supplied_permutation_is_validated_without_entering_learner_view(self) -> None:
        identity = tuple(range(5))
        generated = generate_symbolic_rule_task(
            303,
            position_permutation=identity,
        )
        self.assertEqual(generated.hidden.position_permutation, identity)
        self.assertNotIn("position_permutation", repr(generated.learner))

        for malformed in ((0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 1, 2, 3, True)):
            with self.subTest(malformed=malformed):
                with self.assertRaises((TypeError, ValueError)):
                    generate_symbolic_rule_task(
                        303,
                        position_permutation=malformed,
                    )

    def test_evaluator_can_supply_fresh_public_symbol_namespace(self) -> None:
        symbols = tuple(f"fresh_entity_{index}" for index in range(20))
        generated = generate_symbolic_rule_task(
            404,
            demonstration_count=3,
            public_symbols=symbols,
        )
        visible = tuple(
            symbol
            for demonstration in generated.learner.demonstrations
            for symbol in demonstration.input_symbols
        ) + generated.learner.query_symbols

        self.assertEqual(visible, symbols)
        for malformed in (
            symbols[:-1],
            (*symbols[:-1], symbols[0]),
            (*symbols[:-1], "contains,comma"),
        ):
            with self.subTest(malformed=malformed[-1]):
                with self.assertRaises(ValueError):
                    generate_symbolic_rule_task(
                        404,
                        demonstration_count=3,
                        public_symbols=malformed,
                    )

    def test_surface_forms_vary_independently_of_structure(self) -> None:
        first = generate_symbolic_rule_task(
            500,
            item_count=5,
            demonstration_count=3,
            surface_seed=1,
            demonstration_surface_forms=(
                "Training {demo_number}: ({inputs}) => ({outputs}).",
            ),
        )
        second = generate_symbolic_rule_task(
            500,
            item_count=5,
            demonstration_count=3,
            surface_seed=1,
            demonstration_surface_forms=(
                "Worked case {demo_number}: ({inputs}) rearranges as ({outputs}).",
            ),
        )

        self.assertEqual(first.hidden.position_permutation, second.hidden.position_permutation)
        self.assertEqual(first.hidden.target_order, second.hidden.target_order)
        self.assertEqual(first.learner.query_symbols, second.learner.query_symbols)
        self.assertEqual(first.learner.goal_text, second.learner.goal_text)
        self.assertEqual(first.learner.query_statement, second.learner.query_statement)
        self.assertNotEqual(
            tuple(demo.statement for demo in first.learner.demonstrations),
            tuple(demo.statement for demo in second.learner.demonstrations),
        )
        self.assertNotEqual(first.learner.prompt, second.learner.prompt)

        varied_surface_seed = generate_symbolic_rule_task(
            500,
            item_count=5,
            demonstration_count=3,
            surface_seed=2,
        )
        default_surface_seed = generate_symbolic_rule_task(
            500,
            item_count=5,
            demonstration_count=3,
            surface_seed=1,
        )
        self.assertEqual(
            default_surface_seed.hidden.position_permutation,
            varied_surface_seed.hidden.position_permutation,
        )
        self.assertEqual(
            default_surface_seed.hidden.target_order,
            varied_surface_seed.hidden.target_order,
        )

        with self.assertRaises(ValueError):
            generate_symbolic_rule_task(
                500,
                demonstration_surface_forms=("missing placeholders",),
            )


class SymbolicRuleVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_symbolic_rule_task(
            7331,
            item_count=4,
            demonstration_count=2,
        )

    def test_only_the_hidden_target_is_exact(self) -> None:
        exact_answers = [
            candidate
            for candidate in itertools.permutations(self.instance.learner.query_symbols)
            if verify_symbolic_rule_answer(
                self.instance.learner,
                self.instance.hidden,
                candidate,
            ).exact
        ]
        self.assertEqual(exact_answers, [self.instance.hidden.target_order])

    def test_scalar_feedback_is_bounded_and_does_not_leak_identity(self) -> None:
        exact = verify_symbolic_rule_answer(
            self.instance.learner,
            self.instance.hidden,
            self.instance.hidden.target_order,
        )
        wrong = verify_symbolic_rule_answer(
            self.instance.learner,
            self.instance.hidden,
            tuple(reversed(self.instance.hidden.target_order)),
        )

        self.assertEqual(
            exact,
            SymbolicRuleFeedback(
                valid=True,
                exact=True,
                pairwise_order_agreement=1.0,
            ),
        )
        self.assertTrue(wrong.valid)
        self.assertFalse(wrong.exact)
        self.assertGreaterEqual(wrong.pairwise_order_agreement, 0.0)
        self.assertLess(wrong.pairwise_order_agreement, 1.0)
        self.assertEqual(
            {field.name for field in fields(SymbolicRuleFeedback)},
            {"valid", "exact", "pairwise_order_agreement"},
        )
        for feedback in (exact, wrong):
            rendered = repr(feedback)
            for symbol in self.instance.learner.query_symbols:
                self.assertNotIn(symbol, rendered)
            self.assertNotIn("target", rendered.lower())
            self.assertNotIn("mistake", rendered.lower())

    def test_invalid_full_orderings_receive_only_zero_scalar_feedback(self) -> None:
        query = self.instance.learner.query_symbols
        invalid_answers: tuple[object, ...] = (
            query[:-1],
            query[:-1] + (query[0],),
            query[:-1] + ("unknown",),
            ", ".join(query) + ",",
            (1, 2, 3, 4),
        )
        for answer in invalid_answers:
            with self.subTest(answer=answer):
                feedback = verify_symbolic_rule_answer(
                    self.instance.learner,
                    self.instance.hidden,
                    answer,  # type: ignore[arg-type]
                )
                self.assertEqual(
                    feedback,
                    SymbolicRuleFeedback(
                        valid=False,
                        exact=False,
                        pairwise_order_agreement=0.0,
                    ),
                )

    def test_verifier_rejects_mismatched_or_corrupt_hidden_state(self) -> None:
        mismatched = generate_symbolic_rule_task(7332, item_count=4).hidden
        with self.assertRaises(ValueError):
            verify_symbolic_rule_answer(
                self.instance.learner,
                mismatched,
                self.instance.hidden.target_order,
            )

        malformed = replace(
            self.instance.hidden,
            position_permutation=(0, 0, 2, 3),
        )
        with self.assertRaises(ValueError):
            verify_symbolic_rule_answer(
                self.instance.learner,
                malformed,
                self.instance.hidden.target_order,
            )


if __name__ == "__main__":
    unittest.main()
