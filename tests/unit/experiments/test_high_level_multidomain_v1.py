"""Invariants for the phase-gated high-level multi-domain evaluator."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import hmac
import itertools
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402
from experiments.evaluators import high_level_multidomain_v1 as suite  # noqa: E402
from experiments.evaluators.glyph_machine_trace_suite import (  # noqa: E402
    glyph_machine_mechanism_partition,
)
from experiments.evaluators.symbolic_procedure_transfer_suite import (  # noqa: E402
    demonstration_permutation_partition,
)


_SEEDS = (bytes(range(32)), bytes(range(32, 64)))
_COMMITMENTS = tuple(suite.commit_replicate_seed(seed) for seed in _SEEDS)
_ADMISSION = "sha256:" + "a" * 64
_PLACEMENT_PREDICATE = {
    "tokens": tokens.TOKEN_IN,
    "files": files.FILE_AT,
    "boxes": boxes.ITEM_IN,
}


def _make_evaluation(
    admission: str = _ADMISSION,
) -> suite.HighLevelMultidomainEvaluator:
    return suite.make_evaluation_evaluator(_SEEDS, _COMMITMENTS, admission)


def _attempt_ref(task_id: str, arm: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"test-attempt\x00{task_id}\x00{arm}".encode("ascii")
    ).hexdigest()


def _phase_arms(
    evaluator: suite.HighLevelMultidomainEvaluator,
    phase: str,
) -> tuple[str, ...]:
    if evaluator.purpose == "qualification":
        return (suite.QUALIFICATION_ARM,)
    if phase == "adaptation":
        return ("FULL", "RANDOM_FEEDBACK")
    return suite.EVALUATION_ARMS


def _judge_phase(
    evaluator: suite.HighLevelMultidomainEvaluator,
    tasks: tuple[suite.PublicEvaluationTask, ...],
) -> None:
    for task in tasks:
        for arm in _phase_arms(evaluator, task.phase):
            if (task.task_id, arm) in evaluator._judgments:
                continue
            evaluator.judge_response(
                task.task_id,
                arm,
                _attempt_ref(task.task_id, arm),
                "MALFORMED",
            )


def _release_and_complete(
    evaluator: suite.HighLevelMultidomainEvaluator,
    phase: suite.Phase,
) -> tuple[suite.PublicEvaluationTask, ...]:
    tasks = evaluator.release_phase(phase)
    _judge_phase(evaluator, tasks)
    evaluator.complete_phase(phase)
    return tasks


def _expected_round_robin(
    symbolic_count: int,
    glyph_count: int,
    causal_count: int,
) -> list[str]:
    groups = (
        ["symbolic-demonstration-transfer"] * symbolic_count,
        ["glyph-machine"] * glyph_count,
        ["causal-operator"] * causal_count,
    )
    return [
        group[index]
        for index in range(max(map(len, groups)))
        for group in groups
        if index < len(group)
    ]


def _binding(
    evaluator: suite.HighLevelMultidomainEvaluator,
    task: suite.PublicEvaluationTask,
):
    return evaluator._bindings[task.task_id]


def _exact_glyph_response(binding) -> str:
    pair = binding.source
    task = pair.learner
    hidden = pair.hidden
    goal = next(
        state for state in task.states if state.records == task.goal.required
    )
    origin_index = hidden.state_digests.index(task.origin.digest)
    goal_index = hidden.state_digests.index(goal.digest)
    alias_by_digest = {
        action.digest: alias for alias, action in binding.aliases
    }
    for length in range(task.max_steps + 1):
        for indices in itertools.product(
            range(len(hidden.transition_rows)), repeat=length
        ):
            current = origin_index
            for index in indices:
                current = hidden.transition_rows[index][current]
            if current == goal_index:
                actions = [
                    alias_by_digest[hidden.action_digests[index]]
                    for index in indices
                ]
                return ",".join((*actions, "STOP"))
    raise AssertionError("generated glyph case has no bounded exact procedure")


def _exact_causal_response(binding) -> str:
    challenge = binding.source
    aliases = {raw: alias for alias, raw in binding.aliases}
    predicate = _PLACEMENT_PREDICATE[challenge.domain]
    entity_by_position = {
        record.arguments[1]: record.arguments[0]
        for record in challenge.origin.records
        if record.predicate == predicate
    }
    actions = []
    for index in reversed(range(challenge.maximum_steps)):
        entity = aliases[entity_by_position[f"position_{index}"]]
        source = aliases[f"position_{index}"]
        destination = aliases[f"position_{index + 1}"]
        actions.append(f"ACT({entity},{source},{destination})")
    return ";".join(actions)


class SeedAndConstructionTests(unittest.TestCase):
    def test_family_seed_uses_exact_hmac_domain_and_is_role_separated(self) -> None:
        seed = bytes(range(32))
        family = "glyph-machine"
        role = "development"
        message = (
            f"{suite.SUITE_SCHEMA}\x00{family}\x00{role}"
        ).encode("utf-8")
        expected = int.from_bytes(
            hmac.new(seed, message, hashlib.sha256).digest()[:8], "big"
        )

        self.assertEqual(suite.derive_family_seed(seed, family, role), expected)
        applicable = {
            "symbolic-demonstration-transfer": (
                "adaptation",
                "development",
                "final",
            ),
            "glyph-machine": (
                "adaptation",
                "development",
                "final",
                "surface",
            ),
            "causal-operator": ("adaptation", "development", "final"),
        }
        values = {
            suite.derive_family_seed(seed, declared_family, declared_role)
            for declared_family, roles in applicable.items()
            for declared_role in roles
        }
        self.assertEqual(len(values), 10)
        random_feedback = suite.derive_family_seed(
            seed, "random-feedback", "random-feedback"
        )
        self.assertNotIn(random_feedback, values)
        with self.assertRaisesRegex(ValueError, "not applicable"):
            suite.derive_family_seed(seed, "causal-operator", "surface")

    def test_seed_commitments_are_checked_and_raw_seeds_never_leave_metadata(self) -> None:
        evaluator = _make_evaluation()
        canonical = json.dumps(
            evaluator.commitments.to_canonical(), sort_keys=True
        )

        self.assertEqual(
            evaluator.commitments.replicate_commitments,
            _COMMITMENTS,
        )
        self.assertIsNone(evaluator.commitments.qualification_seed)
        for raw_seed in _SEEDS:
            self.assertNotIn(raw_seed.hex(), canonical)
        for raw_seed in _SEEDS:
            derived = suite.derive_family_seed(
                raw_seed, "random-feedback", "random-feedback"
            )
            self.assertNotIn(str(derived), canonical)
        with self.assertRaisesRegex(ValueError, "do not match"):
            suite.make_evaluation_evaluator(
                _SEEDS,
                (_COMMITMENTS[1], _COMMITMENTS[0]),
                _ADMISSION,
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            duplicate = (_SEEDS[0], _SEEDS[0])
            suite.make_evaluation_evaluator(
                duplicate,
                tuple(suite.commit_replicate_seed(item) for item in duplicate),
                _ADMISSION,
            )
        for invalid in (b"short", bytearray(32), "not-bytes"):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    suite.commit_replicate_seed(invalid)  # type: ignore[arg-type]

    def test_replay_counts_commitments_and_identities_are_exact(self) -> None:
        first = _make_evaluation()
        replay = _make_evaluation()
        qualification = suite.make_qualification_evaluator()

        self.assertEqual(first.commitments, replay.commitments)
        self.assertEqual(first.commitments.digest, replay.commitments.digest)
        self.assertEqual(len(first.commitments.tasks), 84)
        self.assertEqual(len(qualification.commitments.tasks), 22)
        self.assertEqual(
            qualification.commitments.qualification_seed,
            suite.QUALIFICATION_SEED,
        )
        self.assertEqual(
            tuple(suite.expected_phase_count("evaluation", phase) for phase in suite.PHASES),
            (24, 20, 40),
        )
        self.assertEqual(
            tuple(
                suite.expected_phase_count("qualification", phase)
                for phase in suite.PHASES
            ),
            (12, 10, 0),
        )
        for commitments in (first.commitments, qualification.commitments):
            task_ids = [item.task_id for item in commitments.tasks]
            public = [item.public_commitment for item in commitments.tasks]
            private = [item.private_commitment for item in commitments.tasks]
            self.assertEqual(len(task_ids), len(set(task_ids)))
            self.assertEqual(len(public), len(set(public)))
            self.assertEqual(len(private), len(set(private)))
            self.assertTrue(
                all(value.startswith("sha256:") for value in (*public, *private))
            )
            self.assertEqual(
                len(commitments.random_feedback),
                len(commitments.replicate_commitments),
            )
            self.assertTrue(
                all(
                    item.schedule_commitment.startswith("sha256:")
                    for item in commitments.random_feedback
                )
            )

    def test_random_feedback_schedule_is_exact_balanced_and_label_independent(self) -> None:
        evaluator = _make_evaluation()
        with self.assertRaisesRegex(RuntimeError, "before adaptation"):
            evaluator.random_feedback_schedule(_COMMITMENTS[0])
        adaptation = evaluator.release_phase("adaptation")

        for raw_seed, replicate_commitment in zip(
            _SEEDS, _COMMITMENTS, strict=True
        ):
            task_ids = tuple(
                task.task_id
                for task in adaptation
                if task.replicate_commitment == replicate_commitment
            )
            seed = suite.derive_family_seed(
                raw_seed, "random-feedback", "random-feedback"
            )
            seed_bytes = seed.to_bytes(8, "big")
            ordered = sorted(
                task_ids,
                key=lambda task_id: (
                    hashlib.sha256(
                        seed_bytes + task_id.encode("ascii")
                    ).digest(),
                    task_id,
                ),
            )
            expected = {
                task_id: (0.0 if index < 6 else 1.0)
                for index, task_id in enumerate(ordered)
            }
            schedule = evaluator.random_feedback_schedule(
                replicate_commitment
            )
            self.assertEqual(tuple(task_id for task_id, _ in schedule), task_ids)
            self.assertEqual(dict(schedule), expected)
            self.assertEqual(Counter(dict(schedule).values()), {0.0: 6, 1.0: 6})
            assignment_bytes = json.dumps(
                {
                    "assignments": [
                        {"score": score, "task_id": task_id}
                        for task_id, score in schedule
                    ],
                    "replicate_commitment": replicate_commitment,
                    "schema": suite.SUITE_SCHEMA,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            expected_commitment = "sha256:" + hmac.new(
                raw_seed,
                b"angler.high-level-multidomain.v1\x00random-feedback-schedule\x00"
                + assignment_bytes,
                hashlib.sha256,
            ).hexdigest()
            actual_commitment = next(
                item.schedule_commitment
                for item in evaluator.commitments.random_feedback
                if item.replicate_commitment == replicate_commitment
            )
            self.assertEqual(actual_commitment, expected_commitment)
            for task_id, score in schedule:
                self.assertEqual(evaluator.random_feedback_for(task_id), score)

        development_id = next(
            item.task_id
            for item in evaluator.commitments.tasks
            if item.phase == "development"
        )
        with self.assertRaisesRegex(ValueError, "not an adaptation"):
            evaluator.random_feedback_for(development_id)
        replay = _make_evaluation()
        self.assertEqual(
            evaluator.commitments.random_feedback,
            replay.commitments.random_feedback,
        )


class PartitionOrderingAndVisibilityTests(unittest.TestCase):
    def test_phase_release_is_fail_closed_and_qualification_has_no_final(self) -> None:
        qualification = suite.make_qualification_evaluator()
        unreleased_task_id = next(
            item.task_id
            for item in qualification.commitments.tasks
            if item.phase == "development"
        )
        with self.assertRaisesRegex(RuntimeError, "unreleased"):
            qualification.judge_response(
                unreleased_task_id,
                suite.QUALIFICATION_ARM,
                _attempt_ref(unreleased_task_id, suite.QUALIFICATION_ARM),
                "anything",
            )
        with self.assertRaisesRegex(RuntimeError, "preceding"):
            qualification.release_phase("development")

        adaptation = qualification.release_phase("adaptation")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            qualification.judge_response(
                adaptation[0].task_id,
                "FULL",
                _attempt_ref(adaptation[0].task_id, "FULL"),
                "MALFORMED",
            )
        for task in adaptation[:-1]:
            qualification.judge_response(
                task.task_id,
                suite.QUALIFICATION_ARM,
                _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
                "MALFORMED",
            )
        with self.assertRaisesRegex(RuntimeError, "task/arm judgment pair"):
            qualification.complete_phase("adaptation")
        final_adaptation_task = adaptation[-1]
        qualification.judge_response(
            final_adaptation_task.task_id,
            suite.QUALIFICATION_ARM,
            _attempt_ref(
                final_adaptation_task.task_id, suite.QUALIFICATION_ARM
            ),
            "MALFORMED",
        )
        qualification.complete_phase("adaptation")
        development = _release_and_complete(qualification, "development")
        self.assertEqual(len(development), 10)
        with self.assertRaisesRegex(ValueError, "cannot materialize final"):
            qualification.release_phase("final")
        with self.assertRaisesRegex(ValueError, "no final admission"):
            qualification.admit_final("sha256:" + "0" * 64)

    def test_evaluation_requires_development_and_explicit_immutable_admission(self) -> None:
        evaluator = _make_evaluation()
        with self.assertRaisesRegex(RuntimeError, "development"):
            evaluator.admit_final("sha256:" + "0" * 64)
        _release_and_complete(evaluator, "adaptation")
        _release_and_complete(evaluator, "development")
        with self.assertRaisesRegex(RuntimeError, "explicit admission"):
            evaluator.release_phase("final")
        with self.assertRaisesRegex(ValueError, "construction binding"):
            evaluator.admit_final("sha256:" + "2" * 64)
        evaluator.admit_final(_ADMISSION)
        evaluator.admit_final(_ADMISSION)
        final = evaluator.release_phase("final")
        self.assertEqual(len(final), 40)

        alternate = _make_evaluation("sha256:" + "b" * 64)
        self.assertEqual(evaluator.commitments, alternate.commitments)
        self.assertEqual(
            tuple(evaluator._bindings),
            tuple(alternate._bindings),
        )
        with self.assertRaisesRegex(ValueError, "requires a final admission"):
            suite.HighLevelMultidomainEvaluator(
                "evaluation", _SEEDS, _COMMITMENTS, None
            )
        with self.assertRaisesRegex(ValueError, "cannot bind"):
            suite.HighLevelMultidomainEvaluator(
                "qualification",
                (suite.QUALIFICATION_SEED.to_bytes(32, "big"),),
                suite.make_qualification_evaluator().commitments.replicate_commitments,
                _ADMISSION,
            )

    def test_family_counts_and_round_robin_order_hold_per_replicate(self) -> None:
        evaluator = _make_evaluation()
        expected = {
            "adaptation": (4, 2, 6),
            "development": (2, 2, 6),
            "final": (4, 4, 12),
        }
        for phase in ("adaptation", "development"):
            tasks = _release_and_complete(evaluator, phase)
            for replicate in ("replicate-01", "replicate-02"):
                selected = [task for task in tasks if task.replicate == replicate]
                counts = expected[phase]
                self.assertEqual(
                    Counter(task.family for task in selected),
                    Counter(dict(zip(suite.FAMILIES, counts, strict=True))),
                )
                self.assertEqual(
                    [task.family for task in selected],
                    _expected_round_robin(*counts),
                )
        evaluator.admit_final(_ADMISSION)
        final = evaluator.release_phase("final")
        for replicate in ("replicate-01", "replicate-02"):
            selected = [task for task in final if task.replicate == replicate]
            self.assertEqual(
                [task.family for task in selected],
                _expected_round_robin(4, 4, 12),
            )

    def test_one_partitioned_mechanism_and_surface_mapping_span_each_replicate(self) -> None:
        for purpose, evaluator, partition in (
            ("qualification", suite.make_qualification_evaluator(), "development"),
            ("evaluation", _make_evaluation(), "final"),
        ):
            replicate_labels = (
                ("qualification-01",)
                if purpose == "qualification"
                else ("replicate-01", "replicate-02")
            )
            for replicate in replicate_labels:
                bindings = [
                    binding
                    for binding in evaluator._bindings.values()
                    if binding.public.replicate == replicate
                ]
                symbolic_bindings = [
                    item
                    for item in bindings
                    if item.public.family == "symbolic-demonstration-transfer"
                ]
                transforms = {
                    item.source.hidden.source_solution.position_permutation
                    for item in symbolic_bindings
                }
                self.assertEqual(len(transforms), 1)
                self.assertIn(
                    next(iter(transforms)),
                    demonstration_permutation_partition(partition),
                )

                glyph_bindings = [
                    item
                    for item in bindings
                    if item.public.family == "glyph-machine"
                ]
                mechanisms = {
                    item.source.hidden.mechanism_commitment
                    for item in glyph_bindings
                }
                self.assertEqual(len(mechanisms), 1)
                self.assertIn(
                    next(iter(mechanisms)),
                    glyph_machine_mechanism_partition(partition),
                )
                state_vocabularies = {
                    item.source.hidden.state_digests for item in glyph_bindings
                }
                action_vocabularies = {
                    item.source.hidden.action_digests for item in glyph_bindings
                }
                self.assertEqual(len(state_vocabularies), 1)
                self.assertEqual(len(action_vocabularies), 1)

    def test_evaluation_mechanisms_are_selected_without_replacement_exactly(self) -> None:
        evaluator = _make_evaluation()
        remaining_symbolic = list(demonstration_permutation_partition("final"))
        remaining_glyph = list(glyph_machine_mechanism_partition("final"))
        expected_symbolic = []
        expected_glyph = []
        for seed in _SEEDS:
            symbolic_index = suite.derive_family_seed(
                seed, "symbolic-demonstration-transfer", "adaptation"
            ) % len(remaining_symbolic)
            expected_symbolic.append(remaining_symbolic.pop(symbolic_index))
            glyph_index = suite.derive_family_seed(
                seed, "glyph-machine", "adaptation"
            ) % len(remaining_glyph)
            expected_glyph.append(remaining_glyph.pop(glyph_index))

        observed_symbolic = []
        observed_glyph = []
        for replicate in ("replicate-01", "replicate-02"):
            bindings = tuple(
                item
                for item in evaluator._bindings.values()
                if item.public.replicate == replicate
            )
            observed_symbolic.append(
                next(
                    item.source.hidden.source_solution.position_permutation
                    for item in bindings
                    if item.public.family == "symbolic-demonstration-transfer"
                )
            )
            observed_glyph.append(
                next(
                    item.source.hidden.mechanism_commitment
                    for item in bindings
                    if item.public.family == "glyph-machine"
                )
            )
        self.assertEqual(observed_symbolic, expected_symbolic)
        self.assertEqual(observed_glyph, expected_glyph)
        self.assertEqual(len(set(observed_symbolic)), 2)
        self.assertEqual(len(set(observed_glyph)), 2)

    def test_symbolic_and_glyph_roles_are_slices_of_one_replicate_stream(self) -> None:
        evaluator = _make_evaluation()
        for index, (replicate, raw_seed) in enumerate(
            zip(("replicate-01", "replicate-02"), _SEEDS, strict=True)
        ):
            actual = tuple(
                item
                for item in evaluator._bindings.values()
                if item.public.replicate == replicate
            )
            symbolic_mechanism = next(
                item.source.hidden.source_solution.position_permutation
                for item in actual
                if item.public.family == "symbolic-demonstration-transfer"
            )
            symbolic_stream = suite.symbolic.make_demonstration_procedure_transfer_stream(
                suite.derive_family_seed(
                    raw_seed, "symbolic-demonstration-transfer", "adaptation"
                ),
                supports_per_procedure=4,
                queries_per_procedure=6,
                position_permutation=symbolic_mechanism,
                mechanism_partition="final",
                expose_transform_demonstrations=True,
            )
            expected_symbolic = {
                "adaptation": tuple(
                    item
                    for item in symbolic_stream.supports
                    if item.learner.demonstrations_visible
                ),
                "development": symbolic_stream.queries[:2],
                "final": symbolic_stream.queries[2:],
            }
            glyph_mechanism = next(
                item.source.hidden.mechanism_commitment
                for item in actual
                if item.public.family == "glyph-machine"
            )
            glyph_stream = suite.glyph.make_glyph_machine_trace_stream(
                suite.derive_family_seed(raw_seed, "glyph-machine", "adaptation"),
                surface_seed=suite.derive_family_seed(
                    raw_seed, "glyph-machine", "surface"
                ),
                supports=2,
                queries=6,
                observations_per_support=2,
                maximum_steps=4,
                mechanism_commitment=glyph_mechanism,
                mechanism_partition="final",
            )
            expected_glyph = {
                "adaptation": glyph_stream.supports,
                "development": glyph_stream.queries[:2],
                "final": glyph_stream.queries[2:],
            }
            for phase in suite.PHASES:
                observed_symbolic = tuple(
                    item.source
                    for item in sorted(
                        (
                            item
                            for item in actual
                            if item.public.family
                            == "symbolic-demonstration-transfer"
                            and item.public.phase == phase
                        ),
                        key=lambda item: item.public.ordinal,
                    )
                )
                observed_glyph = tuple(
                    item.source
                    for item in sorted(
                        (
                            item
                            for item in actual
                            if item.public.family == "glyph-machine"
                            and item.public.phase == phase
                        ),
                        key=lambda item: item.public.ordinal,
                    )
                )
                self.assertEqual(observed_symbolic, expected_symbolic[phase])
                self.assertEqual(observed_glyph, expected_glyph[phase])
            self.assertEqual(index, int(replicate.removeprefix("replicate-")) - 1)

        qualification = suite.make_qualification_evaluator()
        qualification_key = suite.QUALIFICATION_SEED.to_bytes(32, "big")
        qualification_bindings = tuple(qualification._bindings.values())
        symbolic_mechanism = next(
            item.source.hidden.source_solution.position_permutation
            for item in qualification_bindings
            if item.public.family == "symbolic-demonstration-transfer"
        )
        symbolic_stream = suite.symbolic.make_demonstration_procedure_transfer_stream(
            suite.derive_family_seed(
                qualification_key,
                "symbolic-demonstration-transfer",
                "adaptation",
            ),
            supports_per_procedure=4,
            queries_per_procedure=2,
            position_permutation=symbolic_mechanism,
            mechanism_partition="development",
            expose_transform_demonstrations=True,
        )
        self.assertEqual(
            tuple(
                item.source
                for item in sorted(
                    (
                        item
                        for item in qualification_bindings
                        if item.public.family == "symbolic-demonstration-transfer"
                        and item.public.phase == "development"
                    ),
                    key=lambda item: item.public.ordinal,
                )
            ),
            symbolic_stream.queries,
        )
        glyph_mechanism = next(
            item.source.hidden.mechanism_commitment
            for item in qualification_bindings
            if item.public.family == "glyph-machine"
        )
        glyph_stream = suite.glyph.make_glyph_machine_trace_stream(
            suite.derive_family_seed(
                qualification_key, "glyph-machine", "adaptation"
            ),
            surface_seed=suite.derive_family_seed(
                qualification_key, "glyph-machine", "surface"
            ),
            supports=2,
            queries=2,
            observations_per_support=2,
            maximum_steps=4,
            mechanism_commitment=glyph_mechanism,
            mechanism_partition="development",
        )
        self.assertEqual(
            tuple(
                item.source
                for item in sorted(
                    (
                        item
                        for item in qualification_bindings
                        if item.public.family == "glyph-machine"
                        and item.public.phase == "development"
                    ),
                    key=lambda item: item.public.ordinal,
                )
            ),
            glyph_stream.queries,
        )

    def test_causal_domain_order_and_role_seeds_remain_distinct(self) -> None:
        evaluator = _make_evaluation()
        for replicate in ("replicate-01", "replicate-02"):
            for phase, per_domain in (
                ("adaptation", 2),
                ("development", 2),
                ("final", 4),
            ):
                selected = [
                    binding.source
                    for binding in evaluator._bindings.values()
                    if binding.public.replicate == replicate
                    and binding.public.phase == phase
                    and binding.public.family == "causal-operator"
                ]
                self.assertEqual(
                    [item.domain for item in selected],
                    [
                        domain
                        for domain in causal_domains()
                        for _ in range(per_domain)
                    ],
                )
            ids_by_phase = {
                phase: {
                    binding.source.case_id
                    for binding in evaluator._bindings.values()
                    if binding.public.replicate == replicate
                    and binding.public.phase == phase
                    and binding.public.family == "causal-operator"
                }
                for phase in suite.PHASES
            }
            self.assertTrue(ids_by_phase["adaptation"].isdisjoint(ids_by_phase["development"]))
            self.assertTrue(ids_by_phase["adaptation"].isdisjoint(ids_by_phase["final"]))
            self.assertTrue(ids_by_phase["development"].isdisjoint(ids_by_phase["final"]))

    def test_causal_surface_slots_and_all_public_payloads_are_collision_free(self) -> None:
        evaluator = _make_evaluation()
        payloads = tuple(
            binding.public.payload_json for binding in evaluator._bindings.values()
        )
        self.assertEqual(len(payloads), 84)
        self.assertEqual(len(set(payloads)), 84)

        causal_payloads = []
        for binding in evaluator._bindings.values():
            task = binding.public
            if task.family != "causal-operator":
                continue
            causal_payloads.append(task.payload_json)
            alias_by_raw = {raw: alias for alias, raw in binding.aliases}
            entities = tuple(
                sorted(raw for raw in alias_by_raw if not raw.startswith("position_"))
            )
            locations = tuple(
                sorted(
                    (raw for raw in alias_by_raw if raw.startswith("position_")),
                    key=lambda value: int(value.removeprefix("position_")),
                )
            )
            entity_aliases = tuple(f"E{index}" for index in range(1, len(entities) + 1))
            location_aliases = tuple(f"L{index}" for index in range(1, len(locations) + 1))
            cases_per_domain = 4 if task.phase == "final" else 2
            if task.phase == "adaptation":
                phase_slot = 0
            elif task.phase == "development":
                phase_slot = 1
            else:
                phase_slot = 2 + (task.ordinal % cases_per_domain) // 2
            replicate_index = int(task.replicate.removeprefix("replicate-")) - 1
            combined_slot = replicate_index * 4 + phase_slot
            entity_surface, location_surface = next(
                itertools.islice(
                    itertools.product(
                        itertools.permutations(entity_aliases),
                        itertools.permutations(location_aliases),
                    ),
                    combined_slot,
                    None,
                )
            )
            self.assertEqual(
                tuple(alias_by_raw[value] for value in entities), entity_surface
            )
            self.assertEqual(
                tuple(alias_by_raw[value] for value in locations), location_surface
            )
        self.assertEqual(len(causal_payloads), 48)
        self.assertEqual(len(set(causal_payloads)), 48)


def causal_domains() -> tuple[str, ...]:
    return ("tokens", "files", "boxes")


class ArmAdmissionAndMetricsTests(unittest.TestCase):
    def test_attempt_receipts_are_globally_one_use_without_burning_replay_targets(
        self,
    ) -> None:
        evaluator = _make_evaluation()
        adaptation = evaluator.release_phase("adaptation")
        first, second = adaptation[:2]
        shared_receipt = _attempt_ref(first.task_id, "FULL")

        malformed = evaluator.judge_response(
            first.task_id,
            "FULL",
            shared_receipt,
            "MALFORMED",
        )
        self.assertEqual(malformed.score, 0.0)

        replay_targets = (
            (first, "RANDOM_FEEDBACK"),
            (second, "FULL"),
        )
        for target, arm in replay_targets:
            with self.assertRaisesRegex(RuntimeError, "attempt receipt is already consumed"):
                evaluator.judge_response(
                    target.task_id,
                    arm,
                    shared_receipt,
                    "MALFORMED",
                )

            fresh_receipt = _attempt_ref(target.task_id, arm)
            accepted = evaluator.judge_response(
                target.task_id,
                arm,
                fresh_receipt,
                "MALFORMED",
            )
            self.assertEqual(accepted.attempt_receipt_ref, fresh_receipt)

        _judge_phase(evaluator, adaptation)
        evaluator.complete_phase("adaptation")
        self.assertEqual(evaluator.completed_phases, ("adaptation",))

    def test_arm_receipt_binding_one_shot_and_evaluation_completion_are_exact(self) -> None:
        evaluator = _make_evaluation()
        adaptation = evaluator.release_phase("adaptation")
        task = adaptation[0]
        with self.assertRaisesRegex(ValueError, "not admitted"):
            evaluator.judge_response(
                task.task_id,
                "QWEN_ONLY",
                _attempt_ref(task.task_id, "QWEN_ONLY"),
                "MALFORMED",
            )
        with self.assertRaisesRegex(ValueError, "not admitted"):
            evaluator.judge_response(
                task.task_id,
                suite.QUALIFICATION_ARM,
                _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
                "MALFORMED",
            )
        with self.assertRaisesRegex(ValueError, "attempt_receipt_ref"):
            evaluator.judge_response(
                task.task_id,
                "FULL",
                "not-a-digest",
                "MALFORMED",
            )

        full = evaluator.judge_response(
            task.task_id,
            "FULL",
            _attempt_ref(task.task_id, "FULL"),
            "MALFORMED",
        )
        random_feedback = evaluator.judge_response(
            task.task_id,
            "RANDOM_FEEDBACK",
            _attempt_ref(task.task_id, "RANDOM_FEEDBACK"),
            "MALFORMED",
        )
        self.assertNotEqual(full.response_commitment, random_feedback.response_commitment)
        self.assertEqual(full.arm, "FULL")
        self.assertEqual(random_feedback.arm, "RANDOM_FEEDBACK")
        with self.assertRaisesRegex(ValueError, "does not bind"):
            replace(full, response_commitment="sha256:" + "0" * 64)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            evaluator.judge_response(
                task.task_id,
                "FULL",
                "sha256:" + "9" * 64,
                "MALFORMED",
            )
        with self.assertRaisesRegex(RuntimeError, "task/arm judgment pair"):
            evaluator.complete_phase("adaptation")

        _judge_phase(evaluator, adaptation)
        evaluator.complete_phase("adaptation")
        development = _release_and_complete(evaluator, "development")
        self.assertEqual(
            {
                arm
                for task_item in development
                for task_id, arm in evaluator._judgments
                if task_id == task_item.task_id
            },
            set(suite.EVALUATION_ARMS),
        )
        with self.assertRaisesRegex(RuntimeError, "suite completion"):
            evaluator.final_metrics()
        evaluator.admit_final(_ADMISSION)
        final = evaluator.release_phase("final")
        _judge_phase(evaluator, final)
        with self.assertRaisesRegex(RuntimeError, "suite completion"):
            evaluator.final_metrics()
        evaluator.complete_phase("final")
        metrics = evaluator.final_metrics()
        self.assertEqual(len(metrics.aggregates), 48)
        self.assertEqual(metrics.digest, evaluator.final_metrics().digest)

    def test_pairwise_measurement_is_aggregate_only_after_qualification_closes(self) -> None:
        evaluator = suite.make_qualification_evaluator()
        for phase in ("adaptation", "development"):
            tasks = evaluator.release_phase(phase)
            for task in tasks:
                response = "MALFORMED"
                if task.family == "symbolic-demonstration-transfer":
                    response = ",".join(
                        _binding(evaluator, task)
                        .source.hidden.source_solution.target_order
                    )
                evaluator.judge_response(
                    task.task_id,
                    suite.QUALIFICATION_ARM,
                    _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
                    response,
                )
            evaluator.complete_phase(phase)
            if phase == "adaptation":
                with self.assertRaisesRegex(RuntimeError, "suite completion"):
                    evaluator.final_metrics()

        metrics = evaluator.final_metrics()
        symbolic = tuple(
            item
            for item in metrics.aggregates
            if item.family == "symbolic-demonstration-transfer"
        )
        self.assertEqual(len(symbolic), 2)
        for item in symbolic:
            self.assertEqual(item.binary_success_total, item.attempts)
            self.assertEqual(item.pairwise_agreement_total, float(item.attempts))
        for item in metrics.aggregates:
            if item.family != "symbolic-demonstration-transfer":
                self.assertIsNone(item.pairwise_agreement_total)
        for public in (repr(metrics), json.dumps(metrics.to_canonical())):
            self.assertNotIn("secondary", public.lower())
            self.assertNotIn("hidden", public.lower())
            self.assertNotIn("solution", public.lower())


class PublicProjectionAndParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = _make_evaluation()
        self.adaptation = _release_and_complete(self.evaluator, "adaptation")
        self.development = _release_and_complete(self.evaluator, "development")
        self.evaluator.admit_final(_ADMISSION)
        self.final = self.evaluator.release_phase("final")

    def test_public_renderers_are_canonical_and_contain_no_private_material(self) -> None:
        all_tasks = (*self.adaptation, *self.development, *self.final)
        rendered = []
        for task in all_tasks:
            projection = suite.render_public_task(task)
            self.assertEqual(
                projection,
                json.dumps(
                    json.loads(projection),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            self.assertEqual(json.loads(projection)["family"], task.family)
            rendered.append(projection)
            for public in (repr(task), json.dumps(task.to_canonical())):
                self.assertNotIn("secondary", public.lower())
                self.assertNotIn("hidden", public.lower())
                self.assertNotIn("solution", public.lower())
            with self.assertRaises(ValueError):
                replace(task, payload_json=projection + " ")

        public_text = "\n".join(rendered)
        for raw_seed in _SEEDS:
            self.assertNotIn(raw_seed.hex(), public_text)
        for forbidden in (
            "case_id",
            "generator_seed",
            "heldout_",
            "mechanism_commitment",
            "position_",
            "source_instance_id",
            "target_order",
            "transition_rows",
        ):
            self.assertNotIn(forbidden, public_text)

    def test_role_visibility_and_alias_surfaces_are_exact(self) -> None:
        for task in (*self.adaptation, *self.development, *self.final):
            payload = json.loads(task.payload_json)
            if task.family == "symbolic-demonstration-transfer":
                expected = 2 if task.phase == "adaptation" else 0
                self.assertEqual(len(payload["demonstrations"]), expected)
                self.assertEqual(len(payload["query"]), 5)
            elif task.family == "glyph-machine":
                self.assertEqual(bool(payload["observations"]), task.phase == "adaptation")
                self.assertTrue(
                    all(
                        re.fullmatch(r"S_[0-9a-f]{16}", alias)
                        for alias in payload["states"]
                    )
                )
                self.assertTrue(
                    all(
                        re.fullmatch(r"A_[0-9a-f]{16}", alias)
                        for alias in payload["actions"]
                    )
                )
                self.assertIn(payload["maximum_steps"], (1, 2, 3, 4))
            else:
                self.assertTrue(all(alias.startswith("E") for alias in payload["entities"]))
                self.assertTrue(all(alias.startswith("L") for alias in payload["locations"]))
                self.assertIn(payload["domain"], causal_domains())
                self.assertIn(payload["maximum_steps"], (2, 4))

    def test_symbolic_parser_is_exact_and_non_repairing(self) -> None:
        task = next(
            item
            for item in self.adaptation
            if item.family == "symbolic-demonstration-transfer"
        )
        symbols = json.loads(task.payload_json)["query"]
        valid = ",".join(symbols)
        self.assertTrue(suite.parse_public_response(task, valid).valid)
        invalid = (
            " " + valid,
            valid + "\n",
            ", ".join(symbols),
            ",".join((*symbols[:-1], symbols[0])),
            ",".join((*symbols[:-1], "UNKNOWN")),
            valid + ",EXTRA",
        )
        for response in invalid:
            with self.subTest(response=response):
                parsed = suite.parse_public_response(task, response)
                self.assertFalse(parsed.valid)
                self.assertEqual(parsed.tokens, ())
                self.assertEqual(parsed.error, "MALFORMED_RESPONSE")

    def test_glyph_parser_requires_one_terminal_stop_and_obeys_ceiling(self) -> None:
        task = next(
            item for item in self.adaptation if item.family == "glyph-machine"
        )
        action = json.loads(task.payload_json)["actions"][0]
        for valid in ("STOP", f"{action},STOP"):
            self.assertTrue(suite.parse_public_response(task, valid).valid)
        invalid = (
            action,
            f"STOP,{action}",
            "STOP,STOP",
            "UNKNOWN,STOP",
            ",".join((action,) * 5 + ("STOP",)),
            f"{action}, STOP",
            f"{action},STOP\n",
        )
        for response in invalid:
            with self.subTest(response=response):
                self.assertFalse(suite.parse_public_response(task, response).valid)

    def test_causal_parser_rejects_empty_unknown_overlength_and_trailing_text(self) -> None:
        task = next(
            item for item in self.adaptation if item.family == "causal-operator"
        )
        payload = json.loads(task.payload_json)
        action = f"ACT({payload['entities'][0]},{payload['locations'][0]},{payload['locations'][1]})"
        self.assertTrue(suite.parse_public_response(task, action).valid)
        invalid = (
            "",
            action + ";",
            action + "\n",
            action.replace("ACT(", "ACT( "),
            action.replace(payload["entities"][0], "E999"),
            ";".join((action,) * (payload["maximum_steps"] + 1)),
            action + " trailing",
        )
        for response in invalid:
            with self.subTest(response=response):
                self.assertFalse(suite.parse_public_response(task, response).valid)


class ObjectiveJudgmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = suite.make_qualification_evaluator()
        self.tasks = self.evaluator.release_phase("adaptation")

    def test_all_three_families_round_trip_through_frozen_objective_judges(self) -> None:
        symbolic_task = next(
            task
            for task in self.tasks
            if task.family == "symbolic-demonstration-transfer"
        )
        symbolic_binding = _binding(self.evaluator, symbolic_task)
        symbolic_response = ",".join(
            symbolic_binding.source.hidden.source_solution.target_order
        )
        glyph_task = next(
            task for task in self.tasks if task.family == "glyph-machine"
        )
        glyph_response = _exact_glyph_response(
            _binding(self.evaluator, glyph_task)
        )

        exact = (
            (symbolic_task, symbolic_response),
            (glyph_task, glyph_response),
        )
        for task, response in exact:
            judgment = self.evaluator.judge_response(
                task.task_id,
                suite.QUALIFICATION_ARM,
                _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
                response,
            )
            self.assertEqual(judgment.score, 1.0)
            self.assertEqual(judgment.disposition, "SUCCESS")
            self.assertEqual(judgment.raw_response, response)
            self.assertEqual(judgment.arm, suite.QUALIFICATION_ARM)
            self.assertEqual(
                judgment.attempt_receipt_ref,
                _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
            )
            for public in (repr(judgment), json.dumps(judgment.to_canonical())):
                self.assertNotIn("secondary", public.lower())
                self.assertNotIn("hidden", public.lower())
                self.assertNotIn("solution", public.lower())

        for domain in causal_domains():
            task = next(
                task
                for task in self.tasks
                if task.family == "causal-operator"
                and _binding(self.evaluator, task).source.domain == domain
            )
            response = _exact_causal_response(_binding(self.evaluator, task))
            judgment = self.evaluator.judge_response(
                task.task_id,
                suite.QUALIFICATION_ARM,
                _attempt_ref(task.task_id, suite.QUALIFICATION_ARM),
                response,
            )
            self.assertEqual(judgment.score, 1.0)

    def test_malformed_and_valid_wrong_attempts_return_bounded_scalar_only(self) -> None:
        symbolic_tasks = tuple(
            task
            for task in self.tasks
            if task.family == "symbolic-demonstration-transfer"
        )
        symbolic_task = symbolic_tasks[0]
        malformed = "not,a,valid,response"
        malformed_judgment = self.evaluator.judge_response(
            symbolic_task.task_id,
            suite.QUALIFICATION_ARM,
            _attempt_ref(symbolic_task.task_id, suite.QUALIFICATION_ARM),
            malformed,
        )
        self.assertEqual(malformed_judgment.score, 0.0)
        self.assertEqual(malformed_judgment.raw_response, malformed)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            self.evaluator.judge_response(
                symbolic_task.task_id,
                suite.QUALIFICATION_ARM,
                "sha256:" + "e" * 64,
                malformed,
            )

        symbolic_task = symbolic_tasks[1]
        source = _binding(self.evaluator, symbolic_task).source
        target = source.hidden.source_solution.target_order
        wrong = ",".join(reversed(target))
        wrong_judgment = self.evaluator.judge_response(
            symbolic_task.task_id,
            suite.QUALIFICATION_ARM,
            _attempt_ref(symbolic_task.task_id, suite.QUALIFICATION_ARM),
            wrong,
        )
        self.assertEqual(wrong_judgment.score, 0.0)
        self.assertEqual(wrong_judgment.disposition, "UNSUCCESSFUL")

        causal_task = next(
            task for task in self.tasks if task.family == "causal-operator"
        )
        payload = json.loads(causal_task.payload_json)
        blocked = (
            f"ACT({payload['entities'][0]},{payload['locations'][0]},"
            f"{payload['locations'][0]})"
        )
        blocked_judgment = self.evaluator.judge_response(
            causal_task.task_id,
            suite.QUALIFICATION_ARM,
            _attempt_ref(causal_task.task_id, suite.QUALIFICATION_ARM),
            blocked,
        )
        self.assertEqual(blocked_judgment.score, 0.0)

    def test_cross_evaluator_and_unreleased_task_ids_fail_closed(self) -> None:
        other = suite.make_qualification_evaluator()
        task = self.tasks[0]
        other_development = next(
            item.task_id
            for item in other.commitments.tasks
            if item.phase == "development"
        )
        with self.assertRaisesRegex(RuntimeError, "unreleased"):
            other.judge_response(
                other_development,
                suite.QUALIFICATION_ARM,
                _attempt_ref(other_development, suite.QUALIFICATION_ARM),
                "anything",
            )
        with self.assertRaisesRegex(ValueError, "not owned"):
            missing = "sha256:" + "f" * 64
            other.judge_response(
                missing,
                suite.QUALIFICATION_ARM,
                _attempt_ref(missing, suite.QUALIFICATION_ARM),
                "anything",
            )
        self.assertEqual(
            other.commitments.tasks[0].task_id,
            task.task_id,
            "qualification replay must retain identical task identities",
        )


if __name__ == "__main__":
    unittest.main()
