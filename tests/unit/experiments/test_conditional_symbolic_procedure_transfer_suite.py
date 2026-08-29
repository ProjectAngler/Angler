from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import unittest

import experiments.evaluators.conditional_symbolic_procedure_transfer_suite as suite
from experiments.evaluators.conditional_symbolic_procedure_transfer_suite import (
    conditional_mechanism_partition,
    make_conditional_procedure_transfer_stream,
    score_conditional_procedure_answer,
)


class ConditionalSymbolicProcedureTransferSuiteTests(unittest.TestCase):
    def test_pair_universe_and_opened_partitions_are_disjoint_and_oriented(self) -> None:
        ordered = suite._ordered_conditional_mechanism_pairs()
        train = conditional_mechanism_partition("train")
        development = conditional_mechanism_partition("development")
        final = conditional_mechanism_partition("final")
        opened = (*train, *development, *final)

        self.assertEqual(len(ordered), 7_021)
        self.assertEqual((len(train), len(development), len(final)), (512, 64, 20))
        self.assertEqual(len(opened), 596)
        self.assertEqual(len(set(opened)), 596)
        self.assertFalse(set(train) & set(development))
        self.assertFalse(set(train) & set(final))
        self.assertFalse(set(development) & set(final))
        self.assertEqual(final, ordered[99:119])
        self.assertEqual(train, ordered[:80] + ordered[119:551])
        self.assertEqual(development, ordered[80:99] + ordered[551:596])

        identity = tuple(range(5))
        unordered = {frozenset(pair) for pair in ordered}
        self.assertEqual(len(unordered), 7_021)
        for false_rule, true_rule in ordered:
            self.assertNotEqual(false_rule, true_rule)
            self.assertNotIn(identity, (false_rule, true_rule))
        opened_set = set(opened)
        self.assertTrue(
            all(
                (true_rule, false_rule) not in opened_set
                for false_rule, true_rule in opened
            )
        )
        self.assertEqual(ordered, suite._ordered_conditional_mechanism_pairs())

    def test_stream_is_replayable_and_seed_changes_only_public_encounters(self) -> None:
        mechanism = conditional_mechanism_partition("development")[3]
        first = make_conditional_procedure_transfer_stream(
            96_001,
            supports_per_flag=2,
            queries_per_flag=2,
            mechanism_pair=mechanism,
            mechanism_partition="development",
        )
        replay = make_conditional_procedure_transfer_stream(
            96_001,
            supports_per_flag=2,
            queries_per_flag=2,
            mechanism_pair=mechanism,
            mechanism_partition="development",
        )
        fresh = make_conditional_procedure_transfer_stream(
            96_002,
            supports_per_flag=2,
            queries_per_flag=2,
            mechanism_pair=mechanism,
            mechanism_partition="development",
        )

        self.assertEqual(first, replay)
        self.assertEqual(first.mechanism_commitment, fresh.mechanism_commitment)
        self.assertNotEqual(
            [pair.learner.to_canonical() for pair in first.binding_supports],
            [pair.learner.to_canonical() for pair in fresh.binding_supports],
        )

    def test_stages_have_executable_tree_balance_visibility_and_freshness(self) -> None:
        mechanism = conditional_mechanism_partition("final")[4]
        stream = make_conditional_procedure_transfer_stream(
            96_003,
            supports_per_flag=3,
            queries_per_flag=2,
            mechanism_pair=mechanism,
            mechanism_partition="final",
        )

        self.assertEqual(
            Counter(
                pair.learner.public_flag for pair in stream.anchor_supports
            ),
            {False: 3, True: 3},
        )
        self.assertEqual(len(stream.component_supports), 12)
        self.assertEqual(
            Counter(
                pair.learner.public_flag for pair in stream.binding_supports
            ),
            {False: 3, True: 3},
        )
        self.assertEqual(
            Counter(pair.learner.public_flag for pair in stream.queries),
            {False: 2, True: 2},
        )
        self.assertTrue(
            all(
                not pair.learner.demonstrations_visible
                for pair in stream.anchor_supports
            )
        )
        self.assertTrue(
            all(
                pair.learner.demonstrations_visible
                for pair in stream.component_supports
            )
        )
        self.assertTrue(
            all(
                pair.learner.demonstrations_visible
                for pair in stream.binding_supports
            )
        )
        self.assertTrue(
            all(
                not pair.learner.demonstrations_visible
                for pair in stream.queries
            )
        )

        anchor_request = stream.anchor_supports[0].learner.request
        component_requests = {
            pair.learner.request for pair in stream.component_supports
        }
        binding_request = stream.binding_supports[0].learner.request
        self.assertEqual(anchor_request.children, ())
        self.assertEqual(len(component_requests), 2)
        self.assertTrue(
            all(request.children == (anchor_request,) for request in component_requests)
        )
        self.assertEqual(set(binding_request.children), component_requests)
        self.assertEqual(binding_request.depth, 2)
        self.assertTrue(
            all(pair.learner.request == binding_request for pair in stream.queries)
        )
        symbols = {
            anchor_request.symbol,
            binding_request.symbol,
            *(request.symbol for request in component_requests),
        }
        self.assertEqual(len(symbols), 4)

        identity = tuple(range(5))
        self.assertEqual(
            {
                pair.hidden.source_solution.position_permutation
                for pair in stream.anchor_supports
            },
            {identity},
        )
        for component_index, request in enumerate(binding_request.children):
            members = [
                pair
                for pair in stream.component_supports
                if pair.learner.request == request
            ]
            self.assertEqual(len(members), 6)
            self.assertEqual(
                Counter(pair.learner.public_flag for pair in members),
                {False: 3, True: 3},
            )
            self.assertEqual(
                {
                    pair.hidden.source_solution.position_permutation
                    for pair in members
                },
                {mechanism[component_index]},
            )

        namespaces: list[set[str]] = []
        for pair in (*stream.supports, *stream.queries):
            if pair in (*stream.binding_supports, *stream.queries):
                self.assertEqual(
                    pair.hidden.source_solution.position_permutation,
                    mechanism[int(pair.learner.public_flag)],
                )
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
                all(
                    left.isdisjoint(right)
                    for right in namespaces[index + 1 :]
                )
            )

        for pair in (*stream.component_supports, *stream.binding_supports):
            self.assertEqual(
                tuple(
                    (demo.input_symbols, demo.output_symbols)
                    for demo in pair.learner.demonstrations
                ),
                tuple(
                    (demo.input_symbols, demo.output_symbols)
                    for demo in pair.hidden.source_task.demonstrations
                ),
            )
        public = json.dumps(
            stream.binding_supports[0].learner.to_canonical(),
            sort_keys=True,
        )
        for forbidden in (
            "position_permutation",
            "target_order",
            "generator_seed",
            "surface_seed",
            "instance_id",
        ):
            self.assertNotIn(forbidden, public)

    def test_scalar_scoring_uses_the_flag_selected_hidden_solution(self) -> None:
        stream = make_conditional_procedure_transfer_stream(
            96_005,
            supports_per_flag=1,
            queries_per_flag=1,
            mechanism_pair=conditional_mechanism_partition("final")[6],
            mechanism_partition="final",
        )
        for pair in stream.queries:
            target = pair.hidden.source_solution.target_order
            self.assertEqual(
                score_conditional_procedure_answer(pair.learner, pair.hidden, target),
                1.0,
            )
            self.assertEqual(
                score_conditional_procedure_answer(
                    pair.learner,
                    pair.hidden,
                    tuple(reversed(target)),
                ),
                0.0,
            )
            self.assertIsInstance(
                score_conditional_procedure_answer(pair.learner, pair.hidden, target),
                float,
            )

    def test_partition_orientation_public_binding_and_balance_are_enforced(self) -> None:
        train_pair = conditional_mechanism_partition("train")[0]
        final_pair = conditional_mechanism_partition("final")[0]
        with self.assertRaisesRegex(ValueError, "outside"):
            make_conditional_procedure_transfer_stream(
                96_007,
                supports_per_flag=1,
                queries_per_flag=1,
                mechanism_pair=train_pair,
                mechanism_partition="final",
            )
        with self.assertRaisesRegex(ValueError, "orientation"):
            make_conditional_procedure_transfer_stream(
                96_007,
                supports_per_flag=1,
                queries_per_flag=1,
                mechanism_pair=(final_pair[1], final_pair[0]),
                mechanism_partition="final",
            )

        stream = make_conditional_procedure_transfer_stream(
            96_009,
            supports_per_flag=1,
            queries_per_flag=1,
            mechanism_pair=final_pair,
            mechanism_partition="final",
        )
        pair = stream.queries[0]
        tampered = replace(pair.learner, public_flag=not pair.learner.public_flag)
        with self.assertRaisesRegex(ValueError, "do not match"):
            score_conditional_procedure_answer(
                tampered,
                pair.hidden,
                tuple(item.symbol for item in tampered.items),
            )
        with self.assertRaisesRegex(ValueError, "balanced"):
            replace(stream, binding_supports=stream.binding_supports[:-1])
        one_component = stream.component_supports[0].learner.request
        with self.assertRaisesRegex(ValueError, "counterbalanced"):
            replace(
                stream,
                component_supports=tuple(
                    pair
                    for pair in stream.component_supports
                    if pair.learner.request != one_component
                    or pair.learner.public_flag
                ),
            )
        with self.assertRaisesRegex(ValueError, "commitment"):
            replace(stream, mechanism_commitment="sha256:" + "0" * 64)


if __name__ == "__main__":
    unittest.main()
