from __future__ import annotations

from dataclasses import replace
import json
import unittest

from experiments.evaluators.symbolic_procedure_transfer_suite import (
    demonstration_permutation_partition,
    make_demonstration_procedure_transfer_stream,
    score_demonstration_procedure_answer,
)


class SymbolicProcedureTransferSuiteTests(unittest.TestCase):
    def test_mechanism_partitions_are_complete_disjoint_and_nonidentity(self) -> None:
        train = demonstration_permutation_partition("train")
        development = demonstration_permutation_partition("development")
        final = demonstration_permutation_partition("final")

        self.assertEqual((len(train), len(development), len(final)), (80, 19, 20))
        self.assertFalse(set(train) & set(development))
        self.assertFalse(set(train) & set(final))
        self.assertFalse(set(development) & set(final))
        self.assertEqual(len(set((*train, *development, *final))), 119)
        self.assertNotIn(tuple(range(5)), (*train, *development, *final))

    def test_stream_holds_one_final_transform_across_fresh_encounters(self) -> None:
        transform = demonstration_permutation_partition("final")[3]
        stream = make_demonstration_procedure_transfer_stream(
            94_001,
            supports_per_procedure=4,
            queries_per_procedure=3,
            position_permutation=transform,
            mechanism_partition="final",
        )
        identity = tuple(range(5))
        identity_supports = stream.supports[0::2]
        transform_supports = stream.supports[1::2]

        self.assertEqual(len(stream.supports), 8)
        self.assertEqual(len(stream.queries), 3)
        self.assertEqual(
            {
                pair.hidden.source_solution.position_permutation
                for pair in identity_supports
            },
            {identity},
        )
        self.assertEqual(
            {
                pair.hidden.source_solution.position_permutation
                for pair in (*transform_supports, *stream.queries)
            },
            {transform},
        )
        self.assertTrue(
            all(pair.learner.demonstrations_visible for pair in transform_supports)
        )
        self.assertTrue(
            all(not pair.learner.demonstrations_visible for pair in stream.queries)
        )

        namespaces = []
        for pair in (*stream.supports, *stream.queries):
            source = pair.hidden.source_task
            namespace = {
                symbol
                for demonstration in source.demonstrations
                for symbol in demonstration.input_symbols
            } | set(source.query_symbols)
            self.assertEqual(len(namespace), 15)
            namespaces.append(namespace)
        for index, left in enumerate(namespaces):
            self.assertTrue(
                all(left.isdisjoint(right) for right in namespaces[index + 1 :])
            )

    def test_public_examples_remain_raw_symbol_pairs(self) -> None:
        pair = make_demonstration_procedure_transfer_stream(
            94_003,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[1]

        self.assertEqual(len(pair.learner.demonstrations), 2)
        for public, source in zip(
            pair.learner.demonstrations,
            pair.hidden.source_task.demonstrations,
            strict=True,
        ):
            self.assertEqual(public.input_symbols, source.input_symbols)
            self.assertEqual(public.output_symbols, source.output_symbols)
        encoded = json.dumps(pair.learner.to_canonical(), sort_keys=True)
        self.assertNotIn("demonstration_position_rows", encoded)
        for forbidden in (
            "family_id",
            "family_version",
            "instance_id",
            "position_permutation",
            "target_order",
            "generator_seed",
            "surface_seed",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_queries_withhold_examples_but_scalar_judge_accepts_target(self) -> None:
        pair = make_demonstration_procedure_transfer_stream(
            94_007,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).queries[0]

        self.assertEqual(pair.learner.demonstrations, ())
        self.assertEqual(
            score_demonstration_procedure_answer(
                pair.learner,
                pair.hidden,
                pair.hidden.source_solution.target_order,
            ),
            1.0,
        )

    def test_same_public_query_can_bind_different_private_mechanisms(self) -> None:
        pair = make_demonstration_procedure_transfer_stream(
            94_009,
            supports_per_procedure=1,
            queries_per_procedure=1,
            position_permutation=demonstration_permutation_partition("final")[0],
            mechanism_partition="final",
        ).queries[0]
        alternative = demonstration_permutation_partition("final")[1]
        symbols = tuple(item.symbol for item in pair.learner.items)
        alternative_solution = replace(
            pair.hidden.source_solution,
            position_permutation=alternative,
            target_order=tuple(symbols[position] for position in alternative),
        )
        alternative_hidden = replace(
            pair.hidden,
            source_solution=alternative_solution,
        )
        proposed = pair.hidden.source_solution.target_order

        original_score = score_demonstration_procedure_answer(
            pair.learner,
            pair.hidden,
            proposed,
        )
        alternative_score = score_demonstration_procedure_answer(
            pair.learner,
            alternative_hidden,
            proposed,
        )
        self.assertEqual(original_score, 1.0)
        self.assertNotEqual(alternative_score, original_score)

    def test_partition_membership_and_public_binding_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            make_demonstration_procedure_transfer_stream(
                94_011,
                supports_per_procedure=1,
                queries_per_procedure=1,
                position_permutation=demonstration_permutation_partition("train")[0],
                mechanism_partition="final",
            )

        pair = make_demonstration_procedure_transfer_stream(
            94_013,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).queries[0]
        tampered = replace(pair.learner, public_flag=not pair.learner.public_flag)
        with self.assertRaisesRegex(ValueError, "do not match"):
            score_demonstration_procedure_answer(
                tampered,
                pair.hidden,
                tuple(item.symbol for item in tampered.items),
            )

    def test_matched_demo_off_and_wrong_demo_controls_preserve_hidden_target(self) -> None:
        target = demonstration_permutation_partition("final")[4]
        wrong = demonstration_permutation_partition("final")[5]
        correct = make_demonstration_procedure_transfer_stream(
            94_017,
            supports_per_procedure=1,
            queries_per_procedure=1,
            position_permutation=target,
            mechanism_partition="final",
        )
        demo_off = make_demonstration_procedure_transfer_stream(
            94_017,
            supports_per_procedure=1,
            queries_per_procedure=1,
            position_permutation=target,
            mechanism_partition="final",
            expose_transform_demonstrations=False,
        )
        wrong_demo = make_demonstration_procedure_transfer_stream(
            94_017,
            supports_per_procedure=1,
            queries_per_procedure=1,
            position_permutation=target,
            mechanism_partition="final",
            demonstration_permutation=wrong,
        )

        self.assertTrue(correct.supports[1].learner.demonstrations_visible)
        self.assertFalse(demo_off.supports[1].learner.demonstrations_visible)
        self.assertNotEqual(
            correct.supports[1].learner.demonstrations,
            wrong_demo.supports[1].learner.demonstrations,
        )
        for control in (demo_off, wrong_demo):
            self.assertEqual(
                control.queries[0].learner.to_canonical(),
                correct.queries[0].learner.to_canonical(),
            )
            self.assertEqual(
                control.queries[0].hidden.source_solution,
                correct.queries[0].hidden.source_solution,
            )
            self.assertEqual(
                control.supports[1].hidden.source_solution.position_permutation,
                target,
            )


if __name__ == "__main__":
    unittest.main()
