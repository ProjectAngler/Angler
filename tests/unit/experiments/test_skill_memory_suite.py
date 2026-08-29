"""Integrity tests for the evaluator-owned skill-memory stream."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from experiments.evaluators import skill_memory_suite as suite


class SkillMemorySuiteTests(unittest.TestCase):
    def test_public_view_has_only_observations_flag_and_opaque_request(self) -> None:
        generated = suite.make_skill_memory_partition(
            "final",
            83_001,
            instances_per_program=2,
        ).tasks[0]
        public = generated.learner

        self.assertEqual(
            {field.name for field in fields(public)},
            {"items", "public_flag", "request"},
        )
        self.assertNotIn("instance_id", public.to_canonical())
        self.assertNotIn("domain", public.to_canonical())
        rendered = repr(public)
        for hidden_literal in (
            "A_ASC",
            "A_DESC",
            "B_ASC",
            "B_DESC",
            "GROUP_01",
            "GROUP_10",
            "ZIGZAG",
            "ROTATE",
            "IF_FLAG",
            "IF_NOT_FLAG",
            "target_order",
            "generator_seed",
        ):
            self.assertNotIn(hidden_literal, rendered)

        def check_opaque(expression: suite.PublicSkillExpression) -> None:
            self.assertRegex(expression.symbol, r"^skill_[0-9a-f]{20}$")
            for child in expression.children:
                check_opaque(child)

        check_opaque(public.request)

    def test_partition_generation_is_deterministic_unique_and_disjoint(self) -> None:
        first = suite.make_skill_memory_partitions(
            83_002,
            instances_per_program=3,
        )
        replay = suite.make_skill_memory_partitions(
            83_002,
            instances_per_program=3,
        )
        changed = suite.make_skill_memory_partitions(
            83_003,
            instances_per_program=3,
        )

        self.assertEqual(
            tuple(task.learner for task in first.train.tasks),
            tuple(task.learner for task in replay.train.tasks),
        )
        self.assertEqual(
            tuple(task.hidden.instance_identity for task in first.final.tasks),
            tuple(task.hidden.instance_identity for task in replay.final.tasks),
        )
        self.assertNotEqual(
            tuple(task.learner.request for task in first.train.tasks),
            tuple(task.learner.request for task in changed.train.tasks),
        )
        self.assertEqual(len(first.train.tasks), 14 * 3)
        self.assertEqual(len(first.development.tasks), 5 * 3)
        self.assertEqual(len(first.final.tasks), 6 * 3)

        partitions = (first.train, first.development, first.final)
        for index, left in enumerate(partitions):
            left_ids = {task.hidden.instance_identity for task in left.tasks}
            left_sources = {
                task.hidden.source_instance_identity for task in left.tasks
            }
            self.assertEqual(len(left_ids), len(left.tasks))
            self.assertEqual(len(left_sources), len(left.tasks))
            for right in partitions[index + 1 :]:
                self.assertFalse(
                    left_ids
                    & {task.hidden.instance_identity for task in right.tasks}
                )
                self.assertFalse(
                    left_sources
                    & {
                        task.hidden.source_instance_identity
                        for task in right.tasks
                    }
                )

    def test_train_is_shallow_and_final_contains_novel_depth_two_and_three(
        self,
    ) -> None:
        partitions = suite.make_skill_memory_partitions(
            83_004,
            instances_per_program=1,
        )
        train_programs = {
            task.hidden.program.canonical for task in partitions.train.tasks
        }
        development_programs = {
            task.hidden.program.canonical for task in partitions.development.tasks
        }
        final_programs = {
            task.hidden.program.canonical for task in partitions.final.tasks
        }

        self.assertTrue(
            all(task.hidden.program.depth <= 1 for task in partitions.train.tasks)
        )
        self.assertEqual(
            {task.hidden.program.depth for task in partitions.development.tasks},
            {2},
        )
        self.assertEqual(
            {task.hidden.program.depth for task in partitions.final.tasks},
            {2, 3},
        )
        self.assertFalse(train_programs & development_programs)
        self.assertFalse(train_programs & final_programs)
        self.assertFalse(development_programs & final_programs)

    def test_stream_interleaves_mechanisms_and_balances_public_flags(self) -> None:
        partition = suite.make_skill_memory_partition(
            "final",
            83_005,
            instances_per_program=4,
        )
        identities = [task.hidden.mechanism_identity for task in partition.tasks]

        self.assertEqual(len(set(identities)), 6)
        for offset in range(0, len(identities), 6):
            self.assertEqual(len(set(identities[offset : offset + 6])), 6)
        self.assertEqual(
            sum(task.learner.public_flag for task in partition.tasks),
            len(partition.tasks) // 2,
        )

    def test_meta_partition_varies_contexts_and_covers_private_depths(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            83_016,
            instances_per_program=8,
        )
        programs = {
            task.hidden.program.canonical: task.hidden.program
            for task in partition.tasks
        }
        self.assertEqual({program.depth for program in programs.values()}, {0, 1, 2, 3})

        unary_children: dict[str, set[str]] = {}
        conditional_children: dict[str, set[tuple[str, str]]] = {}
        for program in programs.values():
            if program.operator in {"GROUP_01", "GROUP_10", "ZIGZAG", "ROTATE"}:
                if program.depth == 1:
                    unary_children.setdefault(program.operator, set()).add(
                        program.children[0].operator
                    )
            elif program.operator in {"IF_FLAG", "IF_NOT_FLAG"} and program.depth == 1:
                conditional_children.setdefault(program.operator, set()).add(
                    tuple(child.operator for child in program.children)
                )
        self.assertEqual(
            set(unary_children),
            {"GROUP_01", "GROUP_10", "ZIGZAG", "ROTATE"},
        )
        self.assertTrue(
            all(
                children == {"A_ASC", "A_DESC", "B_ASC", "B_DESC"}
                for children in unary_children.values()
            )
        )
        self.assertEqual(set(conditional_children), {"IF_FLAG", "IF_NOT_FLAG"})
        self.assertTrue(
            all(len(children) >= 8 for children in conditional_children.values())
        )
        self.assertEqual(
            len({task.hidden.mapping.digest for task in partition.tasks}),
            1,
        )

    def test_matched_descendant_queries_hold_public_instance_and_root_fixed(self) -> None:
        pairs = suite.make_skill_memory_meta_matched_queries(83_016)
        replay = suite.make_skill_memory_meta_matched_queries(83_016)

        self.assertEqual(len(pairs), 18)
        self.assertEqual(
            tuple(pair.left.learner for pair in pairs),
            tuple(pair.left.learner for pair in replay),
        )
        roots = {}
        for pair in pairs:
            left = pair.left
            right = pair.right
            root = left.hidden.program.operator
            roots[root] = roots.get(root, 0) + 1
            self.assertEqual(left.learner.items, right.learner.items)
            self.assertIs(left.learner.public_flag, right.learner.public_flag)
            self.assertEqual(
                left.hidden.source_instance_identity,
                right.hidden.source_instance_identity,
            )
            self.assertEqual(
                left.hidden.mapping.digest,
                right.hidden.mapping.digest,
            )
            self.assertEqual(
                left.learner.request.symbol,
                right.learner.request.symbol,
            )
            self.assertNotEqual(
                left.learner.request.children,
                right.learner.request.children,
            )
            self.assertNotEqual(
                left.hidden.generated.hidden.target_order,
                right.hidden.generated.hidden.target_order,
            )
        self.assertEqual(
            roots,
            {
                "GROUP_01": 3,
                "GROUP_10": 3,
                "IF_FLAG": 3,
                "IF_NOT_FLAG": 3,
                "ROTATE": 3,
                "ZIGZAG": 3,
            },
        )
        other = suite.make_skill_memory_meta_matched_queries(83_017)
        self.assertNotEqual(
            tuple(
                (
                    pair.left.hidden.program.canonical,
                    pair.right.hidden.program.canonical,
                )
                for pair in pairs
            ),
            tuple(
                (
                    pair.left.hidden.program.canonical,
                    pair.right.hidden.program.canonical,
                )
                for pair in other
            ),
        )

    def test_meta_compositions_do_not_repeat_final_compositions(self) -> None:
        meta = {
            program.canonical for program in suite._meta_training_programs()
        }
        final = {
            program.canonical for program in suite._composition_query_programs()
        }
        self.assertFalse(meta & final)

    def test_matched_binary_grid_holds_instance_and_children_across_2x2_cells(
        self,
    ) -> None:
        grid = suite.make_skill_memory_matched_binary_branch_grid(
            83_018,
            cases_per_cell=5,
        )

        self.assertEqual(grid.cases_per_cell, 5)
        self.assertEqual(len(grid.cells), 20)
        expected_cells = (
            ("IF_FLAG", False, 0),
            ("IF_FLAG", True, 1),
            ("IF_NOT_FLAG", False, 1),
            ("IF_NOT_FLAG", True, 0),
        )
        for case in grid.cases:
            self.assertEqual(
                tuple(
                    (
                        cell.hidden_operator,
                        cell.public_flag,
                        cell.expected_branch,
                    )
                    for cell in case.cells
                ),
                expected_cells,
            )
            generated = tuple(cell.task for cell in case.cells)
            self.assertTrue(
                all(task.learner.items == generated[0].learner.items for task in generated)
            )
            self.assertTrue(
                all(
                    task.learner.request.children
                    == generated[0].learner.request.children
                    for task in generated
                )
            )
            self.assertTrue(
                all(
                    task.hidden.program.children
                    == generated[0].hidden.program.children
                    for task in generated
                )
            )
            self.assertEqual(
                generated[0].learner.request.symbol,
                generated[1].learner.request.symbol,
            )
            self.assertEqual(
                generated[2].learner.request.symbol,
                generated[3].learner.request.symbol,
            )
            self.assertNotEqual(
                generated[0].learner.request.symbol,
                generated[2].learner.request.symbol,
            )

            targets = tuple(
                task.hidden.generated.hidden.target_order for task in generated
            )
            self.assertEqual(targets[0], targets[3])
            self.assertEqual(targets[1], targets[2])
            self.assertNotEqual(targets[0], targets[1])

    def test_matched_binary_grid_is_deterministic_and_evaluator_only(self) -> None:
        first = suite.make_skill_memory_matched_binary_branch_grid(
            83_019,
            cases_per_cell=3,
        )
        replay = suite.make_skill_memory_matched_binary_branch_grid(
            83_019,
            cases_per_cell=3,
        )
        changed = suite.make_skill_memory_matched_binary_branch_grid(
            83_020,
            cases_per_cell=3,
        )

        self.assertEqual(
            tuple(cell.task.learner for cell in first.cells),
            tuple(cell.task.learner for cell in replay.cells),
        )
        self.assertNotEqual(
            first.cells[0].task.hidden.mapping.digest,
            changed.cells[0].task.hidden.mapping.digest,
        )
        self.assertTrue(
            all(cell.task.hidden.partition == "final" for cell in first.cells)
        )

        meta_train = suite.make_skill_memory_meta_partition(
            83_019,
            instances_per_program=8,
        )
        standard_train = suite.make_skill_memory_partition(
            "train",
            83_019,
            instances_per_program=1,
        )
        grid_ids = {
            cell.task.hidden.instance_identity for cell in first.cells
        }
        grid_sources = {
            cell.task.hidden.source_instance_identity for cell in first.cells
        }
        training = meta_train.tasks + standard_train.tasks
        self.assertFalse(
            grid_ids & {task.hidden.instance_identity for task in training}
        )
        self.assertFalse(
            grid_sources
            & {task.hidden.source_instance_identity for task in training}
        )
        curriculum = suite.make_skill_memory_composition_curriculum(
            83_019,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        )
        self.assertEqual(
            first.cells[0].task.hidden.mapping.digest,
            curriculum.component_supports[0].hidden.mapping.digest,
        )

    def test_matched_binary_grid_requires_positive_cases_per_cell(self) -> None:
        for invalid in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                suite.make_skill_memory_matched_binary_branch_grid(
                    83_021,
                    cases_per_cell=invalid,  # type: ignore[arg-type]
                )

    def test_score_returns_only_scalar_and_rejects_mismatched_pair(self) -> None:
        partition = suite.make_skill_memory_partition(
            "final",
            83_006,
            instances_per_program=2,
        )
        generated = partition.tasks[0]
        correct = generated.hidden.generated.hidden.target_order

        score = suite.score_skill_memory_answer(
            generated.learner,
            generated.hidden,
            correct,
        )
        self.assertIs(type(score), float)
        self.assertEqual(score, 1.0)
        self.assertEqual(
            suite.score_skill_memory_answer(
                generated.learner,
                generated.hidden,
                ("not", "a", "valid", "answer", "set"),
            ),
            0.0,
        )
        other = partition.tasks[1]
        with self.assertRaisesRegex(ValueError, "do not match"):
            suite.score_skill_memory_answer(
                generated.learner,
                other.hidden,
                correct,
            )

    def test_renamed_variant_preserves_skill_and_changes_public_instance(self) -> None:
        source = suite.make_skill_memory_partition(
            "development",
            83_007,
            instances_per_program=1,
        ).tasks[0]
        renamed = suite.make_renamed_skill_variant(source, seed=91_007)

        self.assertEqual(source.learner.request, renamed.learner.request)
        self.assertEqual(source.learner.public_flag, renamed.learner.public_flag)
        self.assertEqual(
            source.hidden.mechanism_identity,
            renamed.hidden.mechanism_identity,
        )
        self.assertEqual(
            source.hidden.program.canonical,
            renamed.hidden.program.canonical,
        )
        self.assertNotEqual(source.learner.items, renamed.learner.items)
        self.assertNotEqual(
            source.hidden.source_instance_identity,
            renamed.hidden.source_instance_identity,
        )
        self.assertEqual(
            suite.score_skill_memory_answer(
                renamed.learner,
                renamed.hidden,
                renamed.hidden.generated.hidden.target_order,
            ),
            1.0,
        )

    def test_tampered_request_cannot_be_scored_against_hidden_solution(self) -> None:
        generated = suite.make_skill_memory_partition(
            "train",
            83_008,
            instances_per_program=1,
        ).tasks[0]
        other = suite.make_skill_memory_partition(
            "train",
            83_009,
            instances_per_program=1,
        ).tasks[0]
        tampered = replace(generated.learner, request=other.learner.request)

        with self.assertRaises(ValueError):
            suite.score_skill_memory_answer(
                tampered,
                generated.hidden,
                generated.hidden.generated.hidden.target_order,
            )

    def test_component_curriculum_covers_primitives_then_deep_compositions(
        self,
    ) -> None:
        curriculum = suite.make_skill_memory_composition_curriculum(
            83_010,
            encounters_per_primitive=4,
            cases_per_component_probe=4,
            cases_per_composition=4,
        )
        supports = curriculum.component_supports
        probes = curriculum.component_probes
        queries = curriculum.composition_queries

        self.assertEqual(len(supports), 10 * 4)
        self.assertEqual(len(probes), 10 * 4)
        self.assertEqual(len(queries), 11 * 4)
        primitives = {
            "A_ASC",
            "A_DESC",
            "B_ASC",
            "B_DESC",
            "GROUP_01",
            "GROUP_10",
            "ZIGZAG",
            "ROTATE",
            "IF_FLAG",
            "IF_NOT_FLAG",
        }
        for tasks in (supports, probes):
            self.assertEqual(
                {task.hidden.program.operator for task in tasks},
                primitives,
            )
            self.assertTrue(all(task.hidden.program.depth <= 1 for task in tasks))
            by_root_programs: dict[str, set[str]] = {}
            for task in tasks:
                by_root_programs.setdefault(
                    task.hidden.program.operator,
                    set(),
                ).add(task.hidden.program.canonical)
            for operator in {
                "GROUP_01",
                "GROUP_10",
                "ZIGZAG",
                "ROTATE",
                "IF_FLAG",
                "IF_NOT_FLAG",
            }:
                self.assertGreaterEqual(len(by_root_programs[operator]), 2)
        self.assertEqual({task.hidden.program.depth for task in queries}, {2, 3})
        self.assertEqual(
            len(
                {
                    task.hidden.mapping.digest
                    for task in supports + probes + queries
                }
            ),
            1,
        )

        for tasks in (supports, probes, queries):
            by_mechanism: dict[str, list[bool]] = {}
            for task in tasks:
                by_mechanism.setdefault(
                    task.hidden.mechanism_identity,
                    [],
                ).append(task.learner.public_flag)
            self.assertTrue(by_mechanism)
            for flags in by_mechanism.values():
                self.assertEqual(flags.count(False), flags.count(True))

        support_sources = {
            task.hidden.source_instance_identity for task in supports
        }
        probe_sources = {
            task.hidden.source_instance_identity for task in probes
        }
        query_sources = {
            task.hidden.source_instance_identity for task in queries
        }
        support_ids = {task.hidden.instance_identity for task in supports}
        probe_ids = {task.hidden.instance_identity for task in probes}
        query_ids = {task.hidden.instance_identity for task in queries}
        self.assertEqual(len(support_sources), len(supports))
        self.assertEqual(len(probe_sources), len(probes))
        self.assertEqual(len(query_sources), len(queries))
        self.assertEqual(len(support_ids), len(supports))
        self.assertEqual(len(probe_ids), len(probes))
        self.assertEqual(len(query_ids), len(queries))
        self.assertFalse(support_sources & probe_sources)
        self.assertFalse(support_sources & query_sources)
        self.assertFalse(probe_sources & query_sources)
        self.assertFalse(support_ids & probe_ids)
        self.assertFalse(support_ids & query_ids)
        self.assertFalse(probe_ids & query_ids)
        self.assertEqual(
            curriculum.learner_component_supports,
            tuple(task.learner for task in supports),
        )
        self.assertEqual(
            curriculum.learner_component_probes,
            tuple(task.learner for task in probes),
        )
        self.assertEqual(
            curriculum.learner_composition_queries,
            tuple(task.learner for task in queries),
        )

    def test_component_curriculum_is_deterministic_and_seed_randomized(self) -> None:
        first = suite.make_skill_memory_composition_curriculum(
            83_011,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        )
        replay = suite.make_skill_memory_composition_curriculum(
            83_011,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        )
        changed = suite.make_skill_memory_composition_curriculum(
            83_012,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        )

        self.assertEqual(
            tuple(task.learner for task in first.component_supports),
            tuple(task.learner for task in replay.component_supports),
        )
        self.assertEqual(
            tuple(task.learner for task in first.component_probes),
            tuple(task.learner for task in replay.component_probes),
        )
        self.assertEqual(
            tuple(task.learner for task in first.composition_queries),
            tuple(task.learner for task in replay.composition_queries),
        )
        self.assertNotEqual(
            first.component_supports[0].hidden.mapping.digest,
            changed.component_supports[0].hidden.mapping.digest,
        )

    def test_composition_learner_view_has_no_target_or_task_identity(self) -> None:
        query = suite.make_skill_memory_composition_curriculum(
            83_013,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        ).composition_queries[0]

        self.assertEqual(
            {field.name for field in fields(query.learner)},
            {"items", "public_flag", "request"},
        )
        canonical = query.learner.to_canonical()
        for forbidden in (
            "target",
            "target_order",
            "task_id",
            "case_id",
            "instance_id",
            "domain",
            "program",
            "generator_seed",
        ):
            self.assertNotIn(forbidden, canonical)
            self.assertNotIn(forbidden, repr(query.learner))
        self.assertEqual(query.learner.request.depth, query.hidden.program.depth)

    def test_composition_query_can_be_renamed_without_changing_request(self) -> None:
        query = suite.make_skill_memory_composition_curriculum(
            83_014,
            encounters_per_primitive=2,
            cases_per_component_probe=2,
            cases_per_composition=2,
        ).composition_queries[0]
        renamed = suite.make_renamed_skill_variant(query, seed=91_014)

        self.assertEqual(query.learner.request, renamed.learner.request)
        self.assertEqual(
            query.hidden.mechanism_identity,
            renamed.hidden.mechanism_identity,
        )
        self.assertNotEqual(query.learner.items, renamed.learner.items)
        self.assertEqual(
            suite.score_skill_memory_answer(
                renamed.learner,
                renamed.hidden,
                renamed.hidden.generated.hidden.target_order,
            ),
            1.0,
        )

    def test_invalid_partition_and_count_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "partition"):
            suite.make_skill_memory_partition("test", 83_015)  # type: ignore[arg-type]
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                suite.make_skill_memory_partition(
                    "train",
                    83_015,
                    instances_per_program=invalid,  # type: ignore[arg-type]
                )
        for invalid in (0, 1, -2, True, 2.5):
            with self.assertRaises(ValueError):
                suite.make_skill_memory_composition_curriculum(
                    83_015,
                    encounters_per_primitive=invalid,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                suite.make_skill_memory_composition_curriculum(
                    83_015,
                    cases_per_component_probe=invalid,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                suite.make_skill_memory_composition_curriculum(
                    83_015,
                    cases_per_composition=invalid,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
