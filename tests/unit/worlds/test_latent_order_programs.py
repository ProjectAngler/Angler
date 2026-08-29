from __future__ import annotations

import unittest

import angler.worlds as public_worlds
from angler.worlds.latent_order_programs import (
    HiddenLatentOrderSolution,
    ITEM_COUNT as LATENT_ORDER_ITEM_COUNT,
    LatentOrderingTask,
    OrderingProgram,
    PublicOrderingItem,
    TRAIN_PROGRAMS,
    VALIDATION_PROGRAMS,
    generate_latent_ordering_task,
    make_renamed_latent_variant,
    score_latent_ordering_answer,
)
from experiments.evaluators.latent_order_suite import evaluator_programs


EVALUATION_PROGRAMS = evaluator_programs()


class LatentOrderProgramTests(unittest.TestCase):
    def test_seeded_generation_is_deterministic_unique_and_well_formed(self) -> None:
        identities: set[str] = set()
        for program_index, program in enumerate(
            (*TRAIN_PROGRAMS, *VALIDATION_PROGRAMS, *EVALUATION_PROGRAMS)
        ):
            for example in range(3):
                seed = 1000 * program_index + example
                generated = generate_latent_ordering_task(program, seed)
                repeated = generate_latent_ordering_task(program, seed)
                self.assertEqual(generated, repeated)
                self.assertNotIn(generated.learner.instance_id, identities)
                identities.add(generated.learner.instance_id)

                task = generated.learner
                self.assertEqual(len(task.items), LATENT_ORDER_ITEM_COUNT)
                self.assertEqual(len(set(task.symbols)), LATENT_ORDER_ITEM_COUNT)
                self.assertEqual(
                    sorted(item.rank_a for item in task.items),
                    list(range(LATENT_ORDER_ITEM_COUNT)),
                )
                self.assertEqual(
                    sorted(item.rank_b for item in task.items),
                    list(range(LATENT_ORDER_ITEM_COUNT)),
                )
                self.assertEqual(sum(item.marked for item in task.items), 1)
                self.assertEqual({item.group for item in task.items}, {0, 1})

    def test_public_observation_distribution_is_program_independent(self) -> None:
        first = generate_latent_ordering_task(TRAIN_PROGRAMS[0], 7721)
        second = generate_latent_ordering_task(EVALUATION_PROGRAMS[0], 7721)

        self.assertEqual(first.learner.items, second.learner.items)
        self.assertEqual(
            first.learner.public_flag,
            second.learner.public_flag,
        )
        self.assertEqual(first.learner, second.learner)
        self.assertNotEqual(first.hidden.program, second.hidden.program)

        forced_false = generate_latent_ordering_task(
            TRAIN_PROGRAMS[0],
            7721,
            public_flag=False,
        )
        forced_true = generate_latent_ordering_task(
            TRAIN_PROGRAMS[0],
            7721,
            public_flag=True,
        )
        self.assertEqual(forced_false.learner.items, forced_true.learner.items)
        self.assertFalse(forced_false.learner.public_flag)
        self.assertTrue(forced_true.learner.public_flag)
        self.assertNotEqual(
            forced_false.learner.instance_id,
            forced_true.learner.instance_id,
        )

    def test_exact_evaluation_programs_are_not_in_public_world_api(self) -> None:
        evaluator_only = (
            "GeneratedLatentOrderingTask",
            "HiddenLatentOrderSolution",
            "LatentOrderFeedback",
            "OrderingProgram",
            "TRAIN_PROGRAMS",
            "VALIDATION_PROGRAMS",
            "generate_latent_ordering_task",
            "score_latent_ordering_answer",
        )
        for name in evaluator_only:
            self.assertFalse(hasattr(public_worlds, name), name)

    def test_program_and_target_are_absent_from_learner_projection(self) -> None:
        generated = generate_latent_ordering_task(EVALUATION_PROGRAMS[1], 81)
        public_text = repr(generated.learner)

        self.assertNotIn(generated.hidden.program.canonical, public_text)
        self.assertNotIn(str(generated.hidden.target_order), public_text)
        self.assertNotIn("program", tuple(generated.learner.__slots__))
        self.assertNotIn("target", tuple(generated.learner.__slots__))

    def test_complete_program_structures_are_disjoint_and_deeper(self) -> None:
        train = {program.canonical for program in TRAIN_PROGRAMS}
        validation = {program.canonical for program in VALIDATION_PROGRAMS}
        sealed = {program.canonical for program in EVALUATION_PROGRAMS}

        self.assertFalse(train & validation)
        self.assertFalse(train & sealed)
        self.assertFalse(validation & sealed)
        self.assertTrue(all(program.depth <= 1 for program in TRAIN_PROGRAMS))
        self.assertTrue(all(program.depth >= 2 for program in VALIDATION_PROGRAMS))
        self.assertTrue(all(program.depth >= 2 for program in EVALUATION_PROGRAMS))

        train_skeletons = {
            program.structural_skeleton for program in TRAIN_PROGRAMS
        }
        validation_skeletons = {
            program.structural_skeleton for program in VALIDATION_PROGRAMS
        }
        test_skeletons = {
            program.structural_skeleton for program in EVALUATION_PROGRAMS
        }
        self.assertFalse(train_skeletons & validation_skeletons)
        self.assertFalse(train_skeletons & test_skeletons)
        self.assertFalse(validation_skeletons & test_skeletons)

        train_operators = set().union(
            *(_operators(program) for program in TRAIN_PROGRAMS)
        )
        test_operators = set().union(
            *(_operators(program) for program in EVALUATION_PROGRAMS)
        )
        self.assertLessEqual(test_operators, train_operators)

        train_programs = {program.canonical for program in TRAIN_PROGRAMS}
        conditional_branches = {
            child.canonical
            for program in EVALUATION_PROGRAMS
            for child in _conditional_children(program)
        }
        self.assertTrue(conditional_branches)
        self.assertLessEqual(conditional_branches, train_programs)

        semantic_fingerprints: set[tuple[tuple[int, ...], ...]] = set()
        for program in (
            *TRAIN_PROGRAMS,
            *VALIDATION_PROGRAMS,
            *EVALUATION_PROGRAMS,
        ):
            fingerprint: list[tuple[int, ...]] = []
            for seed in range(7000, 7064):
                generated = generate_latent_ordering_task(program, seed)
                position = {
                    symbol: index
                    for index, symbol in enumerate(generated.learner.symbols)
                }
                fingerprint.append(
                    tuple(position[symbol] for symbol in generated.hidden.target_order)
                )
            semantic_fingerprints.add(tuple(fingerprint))
        self.assertEqual(
            len(semantic_fingerprints),
            len(TRAIN_PROGRAMS) + len(VALIDATION_PROGRAMS) + len(EVALUATION_PROGRAMS),
        )

    def test_targets_depend_on_each_fresh_query_not_a_cached_permutation(self) -> None:
        position_patterns: set[tuple[int, ...]] = set()
        program = EVALUATION_PROGRAMS[2]
        for seed in range(20, 36):
            generated = generate_latent_ordering_task(program, seed)
            display_position = {
                symbol: index
                for index, symbol in enumerate(generated.learner.symbols)
            }
            position_patterns.add(
                tuple(
                    display_position[symbol]
                    for symbol in generated.hidden.target_order
                )
            )
        self.assertGreater(len(position_patterns), 8)

    def test_scalar_judge_scores_exact_partial_invalid_and_no_details(self) -> None:
        generated = generate_latent_ordering_task(EVALUATION_PROGRAMS[3], 991)
        exact = score_latent_ordering_answer(
            generated.learner,
            generated.hidden,
            generated.hidden.target_order,
        )
        reversed_answer = tuple(reversed(generated.hidden.target_order))
        reverse = score_latent_ordering_answer(
            generated.learner,
            generated.hidden,
            reversed_answer,
        )
        invalid = score_latent_ordering_answer(
            generated.learner,
            generated.hidden,
            (generated.learner.symbols[0],) * LATENT_ORDER_ITEM_COUNT,
        )
        adjacent_answer = list(generated.hidden.target_order)
        adjacent_answer[1], adjacent_answer[2] = (
            adjacent_answer[2],
            adjacent_answer[1],
        )
        adjacent = score_latent_ordering_answer(
            generated.learner,
            generated.hidden,
            adjacent_answer,
        )

        self.assertTrue(exact.valid)
        self.assertTrue(exact.exact)
        self.assertEqual(exact.pairwise_accuracy, 1.0)
        self.assertTrue(reverse.valid)
        self.assertFalse(reverse.exact)
        self.assertEqual(reverse.pairwise_accuracy, 0.0)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.pairwise_accuracy, 0.0)
        self.assertTrue(adjacent.valid)
        self.assertFalse(adjacent.exact)
        self.assertAlmostEqual(adjacent.pairwise_accuracy, 0.9)
        self.assertEqual(
            tuple(exact.__slots__),
            ("valid", "exact", "pairwise_accuracy"),
        )

    def test_opaque_symbol_renaming_preserves_attribute_semantics(self) -> None:
        source = generate_latent_ordering_task(EVALUATION_PROGRAMS[0], 450)
        renamed = make_renamed_latent_variant(source, seed=451)

        source_by_symbol = {
            item.symbol: (item.rank_a, item.rank_b, item.group, item.marked)
            for item in source.learner.items
        }
        renamed_by_symbol = {
            item.symbol: (item.rank_a, item.rank_b, item.group, item.marked)
            for item in renamed.learner.items
        }
        source_target = tuple(
            source_by_symbol[symbol] for symbol in source.hidden.target_order
        )
        renamed_target = tuple(
            renamed_by_symbol[symbol] for symbol in renamed.hidden.target_order
        )

        self.assertEqual(source_target, renamed_target)
        self.assertTrue(
            score_latent_ordering_answer(
                renamed.learner,
                renamed.hidden,
                renamed.hidden.target_order,
            ).exact
        )

    def test_operator_semantics_against_independent_hand_fixtures(self) -> None:
        items = (
            PublicOrderingItem("alpha", 2, 4, 1, False),
            PublicOrderingItem("beta", 0, 2, 0, False),
            PublicOrderingItem("gamma", 4, 0, 1, True),
            PublicOrderingItem("delta", 1, 3, 0, False),
            PublicOrderingItem("epsilon", 3, 1, 1, False),
        )
        a_asc = OrderingProgram("A_ASC")
        a_desc = OrderingProgram("A_DESC")
        b_asc = OrderingProgram("B_ASC")
        b_desc = OrderingProgram("B_DESC")
        fixtures = (
            (a_asc, False, ("beta", "delta", "alpha", "epsilon", "gamma")),
            (a_desc, False, ("gamma", "epsilon", "alpha", "delta", "beta")),
            (b_asc, False, ("gamma", "epsilon", "beta", "delta", "alpha")),
            (b_desc, False, ("alpha", "delta", "beta", "epsilon", "gamma")),
            (
                OrderingProgram("GROUP_01", (a_asc,)),
                False,
                ("beta", "delta", "alpha", "epsilon", "gamma"),
            ),
            (
                OrderingProgram("GROUP_10", (a_asc,)),
                False,
                ("alpha", "epsilon", "gamma", "beta", "delta"),
            ),
            (
                OrderingProgram("ZIGZAG", (a_asc,)),
                False,
                ("beta", "gamma", "delta", "epsilon", "alpha"),
            ),
            (
                OrderingProgram("ROTATE", (a_asc,)),
                False,
                ("gamma", "beta", "delta", "alpha", "epsilon"),
            ),
            (
                OrderingProgram("IF_FLAG", (a_asc, b_asc)),
                False,
                ("beta", "delta", "alpha", "epsilon", "gamma"),
            ),
            (
                OrderingProgram("IF_FLAG", (a_asc, b_asc)),
                True,
                ("gamma", "epsilon", "beta", "delta", "alpha"),
            ),
            (
                OrderingProgram("IF_NOT_FLAG", (a_asc, b_asc)),
                False,
                ("gamma", "epsilon", "beta", "delta", "alpha"),
            ),
            (
                OrderingProgram("IF_NOT_FLAG", (a_asc, b_asc)),
                True,
                ("beta", "delta", "alpha", "epsilon", "gamma"),
            ),
        )
        for index, (program, flag, expected) in enumerate(fixtures):
            with self.subTest(operator=program.operator, flag=flag):
                task = LatentOrderingTask(f"manual-{index}", items, flag)
                solution = HiddenLatentOrderSolution(
                    f"manual-{index}",
                    program,
                    expected,
                    0,
                )
                self.assertTrue(
                    score_latent_ordering_answer(task, solution, expected).exact
                )


def _operators(program: object) -> set[str]:
    operator = getattr(program, "operator")
    children = getattr(program, "children")
    result = {operator}
    for child in children:
        result.update(_operators(child))
    return result


def _conditional_children(program: OrderingProgram) -> tuple[OrderingProgram, ...]:
    nested: list[OrderingProgram] = []
    if program.operator in {"IF_FLAG", "IF_NOT_FLAG"}:
        nested.extend(program.children)
    for child in program.children:
        nested.extend(_conditional_children(child))
    return tuple(nested)


if __name__ == "__main__":
    unittest.main()
