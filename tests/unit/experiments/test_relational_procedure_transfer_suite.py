from __future__ import annotations

from dataclasses import replace
import json
import unittest

from experiments.evaluators.relational_procedure_transfer_suite import (
    make_relational_procedure_transfer_stream,
    score_relational_procedure_answer,
)


def _public_path(pair: object) -> tuple[str, ...]:
    source = pair.hidden.source_task
    outgoing = {
        constraint.earlier: constraint.later
        for constraint in source.constraints
    }
    incoming = set(outgoing.values())
    start = next(symbol for symbol in source.symbols if symbol not in incoming)
    ordered = [start]
    while ordered[-1] in outgoing:
        ordered.append(outgoing[ordered[-1]])
    return tuple(ordered)


class RelationalProcedureTransferSuiteTests(unittest.TestCase):
    def test_stream_is_disjoint_balanced_and_deterministic(self) -> None:
        first = make_relational_procedure_transfer_stream(
            91_001,
            supports_per_procedure=4,
            queries_per_procedure=3,
        )
        second = make_relational_procedure_transfer_stream(
            91_001,
            supports_per_procedure=4,
            queries_per_procedure=3,
        )

        self.assertEqual(len(first.supports), 8)
        self.assertEqual(len(first.queries), 6)
        self.assertEqual(first, second)
        self.assertEqual(
            sum(not pair.hidden.reverse for pair in first.supports),
            4,
        )
        self.assertEqual(sum(pair.hidden.reverse for pair in first.supports), 4)
        self.assertFalse(
            {pair.hidden.source_instance_id for pair in first.supports}
            & {pair.hidden.source_instance_id for pair in first.queries}
        )

    def test_public_packing_is_lossless_edges_not_solution_ranks(self) -> None:
        pair = make_relational_procedure_transfer_stream(
            91_003,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[0]
        task = pair.learner

        packed_edges = {
            (item.rank_a, item.rank_b)
            for item in task.items
            if item.group == 1
        }
        public_edges = {
            (edge.earlier_index, edge.later_index)
            for edge in task.precedence_edges
        }
        self.assertEqual(packed_edges, public_edges)
        self.assertEqual(tuple(item.rank_a for item in task.items), tuple(range(5)))
        self.assertEqual(sum(item.marked for item in task.items), 1)

        correct_positions = {
            symbol: index for index, symbol in enumerate(_public_path(pair))
        }
        self.assertNotEqual(
            tuple(item.rank_a for item in task.items),
            tuple(correct_positions[item.symbol] for item in task.items),
        )

    def test_judge_returns_only_scalar_for_forward_and_reverse(self) -> None:
        stream = make_relational_procedure_transfer_stream(
            91_007,
            supports_per_procedure=1,
            queries_per_procedure=1,
        )
        forward, reverse = stream.supports
        ordered = _public_path(forward)

        self.assertEqual(
            score_relational_procedure_answer(
                forward.learner,
                forward.hidden,
                ordered,
            ),
            1.0,
        )
        self.assertEqual(
            score_relational_procedure_answer(
                reverse.learner,
                reverse.hidden,
                tuple(reversed(_public_path(reverse))),
            ),
            1.0,
        )
        self.assertEqual(
            score_relational_procedure_answer(
                forward.learner,
                forward.hidden,
                tuple(reversed(ordered)),
            ),
            0.0,
        )

    def test_public_projection_excludes_private_identifiers(self) -> None:
        pair = make_relational_procedure_transfer_stream(
            91_011,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[0]
        encoded = json.dumps(pair.learner.to_canonical(), sort_keys=True)
        for forbidden in (
            "family_id",
            "family_version",
            "instance_id",
            "orientation",
            "reverse",
            "target_order",
            "generator_seed",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_judge_rejects_public_task_tampering(self) -> None:
        pair = make_relational_procedure_transfer_stream(
            91_013,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[0]
        tampered = replace(pair.learner, public_flag=not pair.learner.public_flag)
        with self.assertRaisesRegex(ValueError, "do not match"):
            score_relational_procedure_answer(
                tampered,
                pair.hidden,
                tuple(item.symbol for item in tampered.items),
            )


if __name__ == "__main__":
    unittest.main()
