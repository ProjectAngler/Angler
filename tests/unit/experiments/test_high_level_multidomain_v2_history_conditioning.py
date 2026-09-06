"""CPU-only invariants for the V2 history-conditioning evaluator."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


# This focused construction test must never make a CUDA device visible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from experiments.evaluators import glyph_machine_trace_suite as glyph  # noqa: E402
from experiments.evaluators import (  # noqa: E402
    high_level_multidomain_v2_history_conditioning as suite,
)


_SEEDS = (
    bytes(range(0, 32)),
    bytes(range(32, 64)),
    bytes(range(64, 96)),
)
_COMMITMENTS = tuple(suite.commit_replicate_seed(seed) for seed in _SEEDS)
_ADMISSION = "sha256:" + "a" * 64
_PREFIX = "angler.high-level-multidomain.v2-history-conditioning-isolation"
_PROTOCOL = _PREFIX + ".v3"
_CONSUMED = frozenset(
    {
        "sha256:86383b64d268bc944abffa1481d81cb730b75766d7d9f0f06d153ca9837cf4f9",
        "sha256:4ec98ef2fed6da44b2593301876d59ee051112e28d9c9caabc243f76481d7c4d",
        "sha256:869675ad3b43bdd300f6ed58b8bf0af2b87520318b42e36d721565a7b7e9d9f5",
        "sha256:44ea52eb9a4085e0e803a965b53ad0fa6ac58cd642fc82ae130a583371b9316f",
        "sha256:a6b8f252b7b8482d0e6648204899dff9d35f4a7c20160d948854bc271d293c8f",
        "sha256:a7327638bef71c33166f1cce2b6285fefcf1c496d3d4b2708346ee534d2965ae",
        "sha256:07d4cbc1ed6fc94466622677bdc1092c96ebfc3024d6080d483a2dfd0c876670",
        "sha256:980d453d2e54799283429d38bdffc6efc0f11264b4ff500bccdcb7a3979e69ff",
        "sha256:646c8207543482e6cee8255fe2e81697a0c2fffa9eb94ecb56644a0c3ba5788b",
        "sha256:613957a33dee19a992d06e7854b9c8435226c559ce7cdf0b734b02654d90a4e0",
        "sha256:2382474330f7e2d63e7e02bbe4d02112eecd5ce6d8c5be810b8b3671b9e030af",
    }
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _make_evaluation(
    admission: str = _ADMISSION,
) -> suite.HighLevelMultidomainEvaluator:
    return suite.make_evaluation_evaluator(
        _SEEDS,
        _COMMITMENTS,
        admission,
    )


def _attempt_ref(task_id: str, arm: str) -> str:
    return _digest(f"v2-unit-attempt\x00{task_id}\x00{arm}")


def _finalize_phase_as_unadmitted(
    evaluator: suite.HighLevelMultidomainEvaluator,
    tasks: tuple[suite.PublicEvaluationTask, ...],
) -> None:
    for task in tasks:
        for arm in evaluator.expected_arms_for(task.task_id):
            evaluator.record_attempt(
                task.task_id,
                arm,
                _attempt_ref(task.task_id, arm),
                "",
                proposal_admitted=False,
            )
    evaluator.complete_phase(tasks[0].phase)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested
            for child in value.values()
            for nested in _walk_keys(child)
        }
    if isinstance(value, list):
        return {nested for child in value for nested in _walk_keys(child)}
    return set()


class IdentityAndSeedTests(unittest.TestCase):
    def test_identities_and_all_literal_domains_are_frozen(self) -> None:
        self.assertEqual(suite.PROTOCOL_IDENTITY, _PROTOCOL)
        self.assertEqual(suite.SUITE_SCHEMA, _PROTOCOL)
        self.assertEqual(
            suite.QUALIFICATION_IDENTITY,
            _PREFIX + ".qualification.v6",
        )
        self.assertEqual(
            suite.EVALUATION_IDENTITY,
            _PREFIX + ".evaluation.v3",
        )
        self.assertEqual(suite.QUALIFICATION_SEED, 2_026_090_220)
        self.assertEqual(
            suite._REPLICATE_SEED_DOMAIN,
            (_PREFIX + "\x00replicate-seed\x00").encode(),
        )
        self.assertEqual(
            suite._PRIVATE_TASK_DOMAIN,
            (_PREFIX + "\x00private-task\x00").encode(),
        )
        self.assertEqual(
            suite._PUBLIC_TASK_DOMAIN,
            (_PREFIX + "\x00public-task\x00").encode(),
        )
        self.assertEqual(
            suite._TASK_ID_DOMAIN,
            (_PREFIX + "\x00task-id\x00").encode(),
        )
        self.assertEqual(
            suite._EVALUATOR_RECORD_DOMAIN,
            (_PREFIX + "\x00evaluator-record\x00").encode(),
        )
        self.assertEqual(
            suite.CONTROL_INDEX_DOMAIN,
            (_PREFIX + "\x00control-index\x00").encode(),
        )
        self.assertEqual(
            suite._CAUSAL_SURFACE_DOMAIN,
            (_PROTOCOL + "\x00causal-surface\x00").encode(),
        )

    def test_seed_commitment_and_family_derivation_use_exact_domains(self) -> None:
        qualification_commitment = (
            "sha256:453878d4fab9e2304e8bb6d72f16535e"
            "32ffe407362cd1bbe631d2943c9f3697"
        )
        self.assertEqual(
            suite.commit_replicate_seed(
                suite.QUALIFICATION_SEED.to_bytes(32, "big")
            ),
            qualification_commitment,
        )
        self.assertEqual(
            suite.make_qualification_evaluator().commitments.replicate_commitments,
            (qualification_commitment,),
        )

        raw_seed = _SEEDS[0]
        expected_commitment = "sha256:" + hashlib.sha256(
            (_PREFIX + "\x00replicate-seed\x00").encode() + raw_seed
        ).hexdigest()
        self.assertEqual(
            suite.commit_replicate_seed(raw_seed),
            expected_commitment,
        )

        expected_roles = {
            "symbolic-demonstration-transfer": ("adaptation",),
            "glyph-machine": ("adaptation", "surface"),
            "causal-operator": ("adaptation", "development", "final"),
        }
        observed: set[int] = set()
        for family, roles in expected_roles.items():
            for role in roles:
                message = f"{_PROTOCOL}\x00{family}\x00{role}".encode()
                expected = int.from_bytes(
                    hmac.new(raw_seed, message, hashlib.sha256).digest()[:8],
                    "big",
                )
                actual = suite.derive_family_seed(raw_seed, family, role)
                self.assertEqual(actual, expected)
                observed.add(actual)
        self.assertEqual(len(observed), 6)

        for family, role in (
            ("symbolic-demonstration-transfer", "development"),
            ("symbolic-demonstration-transfer", "final"),
            ("glyph-machine", "development"),
            ("glyph-machine", "final"),
            ("causal-operator", "surface"),
        ):
            with self.subTest(family=family, role=role):
                with self.assertRaisesRegex(ValueError, "not applicable"):
                    suite.derive_family_seed(raw_seed, family, role)

    def test_seed_inputs_and_consumed_commitments_fail_closed(self) -> None:
        self.assertEqual(suite.CONSUMED_REPLICATE_COMMITMENTS, _CONSUMED)
        for invalid in (b"short", bytearray(32), "not-bytes"):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, "exactly 32 bytes"):
                    suite.commit_replicate_seed(invalid)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "do not match"):
            suite.make_evaluation_evaluator(
                _SEEDS,
                tuple(reversed(_COMMITMENTS)),
                _ADMISSION,
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            duplicate = (_SEEDS[0], _SEEDS[0], _SEEDS[2])
            suite.make_evaluation_evaluator(
                duplicate,
                tuple(suite.commit_replicate_seed(item) for item in duplicate),
                _ADMISSION,
            )
        with mock.patch.object(
            suite,
            "CONSUMED_REPLICATE_COMMITMENTS",
            frozenset({_COMMITMENTS[0]}),
        ):
            with self.assertRaisesRegex(ValueError, "already consumed"):
                _make_evaluation()

        consumed_r5_key = (2_026_090_214).to_bytes(32, "big")
        consumed_r5_keys = (consumed_r5_key, _SEEDS[1], _SEEDS[2])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            suite.make_evaluation_evaluator(
                consumed_r5_keys,
                tuple(
                    suite.commit_replicate_seed(item)
                    for item in consumed_r5_keys
                ),
                _ADMISSION,
            )

    def test_public_task_refs_and_ids_use_the_exact_v2_domains(self) -> None:
        evaluator = suite.make_qualification_evaluator()
        task = evaluator.release_phase("adaptation")[0]
        public_payload = {
            "family": task.family,
            "ordinal": task.ordinal,
            "payload": task.payload,
            "phase": task.phase,
            "protocol_identity": _PROTOCOL,
            "replicate": task.replicate,
            "replicate_commitment": task.replicate_commitment,
        }
        expected_public = "sha256:" + hashlib.sha256(
            (_PREFIX + "\x00public-task\x00").encode()
            + _canonical(public_payload)
        ).hexdigest()
        expected_task_id = "sha256:" + hashlib.sha256(
            (_PREFIX + "\x00task-id\x00").encode()
            + _canonical(
                {
                    "family": task.family,
                    "ordinal": task.ordinal,
                    "phase": task.phase,
                    "protocol_identity": _PROTOCOL,
                    "public_task_ref": expected_public,
                    "replicate": task.replicate,
                }
            )
        ).hexdigest()
        self.assertEqual(task.public_task_ref, expected_public)
        self.assertEqual(task.task_id, expected_task_id)


class ConstructionAndPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qualification = suite.make_qualification_evaluator()
        cls.evaluation = _make_evaluation()

    def test_replay_identities_cardinalities_and_family_splits_are_exact(self) -> None:
        qualification_replay = suite.make_qualification_evaluator()
        evaluation_replay = _make_evaluation()
        self.assertEqual(
            self.qualification.commitments,
            qualification_replay.commitments,
        )
        self.assertEqual(
            self.evaluation.commitments,
            evaluation_replay.commitments,
        )
        self.assertEqual(self.qualification.identity, suite.QUALIFICATION_IDENTITY)
        self.assertEqual(self.evaluation.identity, suite.EVALUATION_IDENTITY)
        self.assertEqual(self.qualification.replicate_labels, ("qualification-01",))
        self.assertEqual(
            self.evaluation.replicate_labels,
            ("replicate-01", "replicate-02", "replicate-03"),
        )
        self.assertEqual(len(self.qualification.commitments.tasks), 48)
        self.assertEqual(len(self.evaluation.commitments.tasks), 216)

        qualification_counts = Counter(
            (item.replicate, item.family, item.phase)
            for item in self.qualification.commitments.tasks
        )
        for family in suite.FAMILIES:
            self.assertEqual(
                qualification_counts[("qualification-01", family, "adaptation")],
                15,
            )
            self.assertEqual(
                qualification_counts[("qualification-01", family, "development")],
                1,
            )

        evaluation_counts = Counter(
            (item.replicate, item.family, item.phase)
            for item in self.evaluation.commitments.tasks
        )
        for replicate in self.evaluation.replicate_labels:
            for family in suite.FAMILIES:
                self.assertEqual(
                    tuple(
                        evaluation_counts[(replicate, family, phase)]
                        for phase in suite.PHASES
                    ),
                    (15, 3, 6),
                )
        self.assertEqual(
            tuple(
                suite.expected_phase_count("qualification", phase)
                for phase in suite.PHASES
            ),
            (45, 3, 0),
        )
        self.assertEqual(
            tuple(
                suite.expected_phase_count("evaluation", phase)
                for phase in suite.PHASES
            ),
            (135, 27, 54),
        )
        self.assertEqual(
            tuple(
                suite.expected_attempt_count("qualification", phase)
                for phase in suite.PHASES
            ),
            (48, 24, 0),
        )
        self.assertEqual(
            tuple(
                suite.expected_attempt_count("evaluation", phase)
                for phase in suite.PHASES
            ),
            (270, 216, 432),
        )

    def test_glyph_candidate_hashes_and_four_state_restriction_are_exact(self) -> None:
        expected_aggregate = {
            "qualification": (
                "sha256:f4261dcf9227bf7f08e7191264fd5f089688658ccf995f1b9f3ae2e724a83d1b"
            ),
            "evaluation": (
                "sha256:5cbdd4f602e332c6fd0581d29a6b990f36726d3b7bfd8464c4b44d8731e8d7ab"
            ),
        }
        for purpose, partition in (
            ("qualification", "development"),
            ("evaluation", "final"),
        ):
            with self.subTest(purpose=purpose):
                complete = glyph.glyph_machine_mechanism_partition(partition)
                candidates = suite.glyph_candidate_commitments(purpose)
                self.assertEqual(candidates, complete[2:16])
                self.assertEqual(len(candidates), 14)
                self.assertTrue(set(candidates).isdisjoint(complete[:2]))
                observed_aggregate = "sha256:" + hashlib.sha256(
                    _canonical(list(candidates))
                ).hexdigest()
                self.assertEqual(observed_aggregate, expected_aggregate[purpose])

                # The public commitments and semantic strata have identical
                # order, making this an exact proof that indices 2:16 are the
                # four-state stratum and the excluded prefix is three-state.
                mechanisms = glyph._semantic_partition(partition)
                state_count_by_commitment = {
                    commitment: mechanism[0]
                    for commitment, mechanism in zip(
                        complete,
                        mechanisms,
                        strict=True,
                    )
                }
                self.assertEqual(
                    {state_count_by_commitment[item] for item in candidates},
                    {4},
                )
                self.assertEqual(
                    {state_count_by_commitment[item] for item in complete[:2]},
                    {3},
                )

        qualification_key = suite.QUALIFICATION_SEED.to_bytes(32, "big")
        _symbolic, selected_glyph = suite._select_replicate_mechanisms(
            "qualification",
            (qualification_key,),
        )
        complete = glyph.glyph_machine_mechanism_partition("development")
        self.assertEqual(len(selected_glyph), 1)
        self.assertEqual(complete.index(selected_glyph[0]), 13)

    def test_causal_evaluation_is_15_3_6_per_replicate_and_aliases_do_not_collide(self) -> None:
        by_id = self.evaluation._bindings
        causal_bindings = tuple(
            by_id[item.task_id]
            for item in self.evaluation.commitments.tasks
            if item.family == "causal-operator"
        )
        self.assertEqual(len(causal_bindings), 72)
        counts = Counter(
            (item.public.replicate, item.public.phase)
            for item in causal_bindings
        )
        domain_counts = Counter(
            (
                item.public.replicate,
                item.public.phase,
                item.public.payload["domain"],
            )
            for item in causal_bindings
        )
        for replicate in self.evaluation.replicate_labels:
            self.assertEqual(
                tuple(counts[(replicate, phase)] for phase in suite.PHASES),
                (15, 3, 6),
            )
            for domain in ("tokens", "files", "boxes"):
                self.assertEqual(
                    tuple(
                        domain_counts[(replicate, phase, domain)]
                        for phase in suite.PHASES
                    ),
                    (5, 1, 2),
                )

        seen_entities: set[str] = set()
        seen_locations: set[str] = set()
        for binding in causal_bindings:
            payload = binding.public.payload
            entities = set(payload["entities"])
            locations = set(payload["locations"])
            self.assertEqual(len(entities), len(payload["entities"]))
            self.assertEqual(len(locations), len(payload["locations"]))
            self.assertTrue(seen_entities.isdisjoint(entities))
            self.assertTrue(seen_locations.isdisjoint(locations))
            seen_entities.update(entities)
            seen_locations.update(locations)


class PublicBoundaryAndParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = _make_evaluation()
        cls.tasks = cls.evaluator.release_phase("adaptation")

    def test_public_tasks_and_metadata_do_not_expose_private_bindings_or_raw_seeds(self) -> None:
        self.assertEqual(
            {item.name for item in fields(suite.PublicEvaluationTask)},
            {
                "task_id",
                "replicate",
                "replicate_commitment",
                "family",
                "phase",
                "ordinal",
                "payload_json",
                "public_commitment",
            },
        )
        forbidden_keys = {
            "action_digests",
            "allowed_action_schemas",
            "case_id",
            "goal_digest",
            "mechanism_commitment",
            "origin_digest",
            "private",
            "private_commitment",
            "source_instance_id",
            "state_digests",
            "surface_option_index",
            "task_slot",
            "transform",
            "transition_rows",
        }
        for task in self.tasks:
            self.assertTrue(forbidden_keys.isdisjoint(_walk_keys(task.payload)))
            self.assertEqual(suite.render_public_task(task), task.payload_json)
            self.assertEqual(task.public_prompt, task.payload_json)
            self.assertNotIn("raw_seed", task.to_canonical())
        metadata_json = json.dumps(
            self.evaluator.commitments.to_canonical(),
            ensure_ascii=False,
            sort_keys=True,
        )
        for raw_seed in _SEEDS:
            self.assertNotIn(raw_seed.hex(), metadata_json)
        self.assertIsNone(self.evaluator.commitments.qualification_seed)
        self.assertFalse(
            any(
                name in suite.__all__
                for name in (
                    "answer",
                    "solution",
                    "private_binding",
                    "repair_response",
                )
            )
        )

    def test_each_family_parser_accepts_only_its_exact_public_grammar(self) -> None:
        by_family = {
            family: next(task for task in self.tasks if task.family == family)
            for family in suite.FAMILIES
        }

        symbolic_task = by_family["symbolic-demonstration-transfer"]
        symbolic_response = ",".join(symbolic_task.payload["query"])
        parsed_symbolic = suite.parse_public_response(
            symbolic_task,
            symbolic_response,
        )
        self.assertTrue(parsed_symbolic.valid)
        self.assertEqual(parsed_symbolic.tokens, tuple(symbolic_task.payload["query"]))

        glyph_task = by_family["glyph-machine"]
        parsed_glyph = suite.parse_public_response(glyph_task, "STOP")
        self.assertTrue(parsed_glyph.valid)
        self.assertEqual(parsed_glyph.tokens, ())

        causal_task = by_family["causal-operator"]
        causal_payload = causal_task.payload
        causal_response = "ACT({},{},{})".format(
            causal_payload["entities"][0],
            causal_payload["locations"][0],
            causal_payload["locations"][1],
        )
        parsed_causal = suite.parse_public_response(causal_task, causal_response)
        self.assertTrue(parsed_causal.valid)
        self.assertEqual(parsed_causal.tokens, (causal_response,))

        malformed = (
            (symbolic_task, symbolic_response + "\n"),
            (symbolic_task, symbolic_response.split(",")[0] + "," * 4),
            (glyph_task, " STOP"),
            (glyph_task, "A_0000000000000000,STOP"),
            (causal_task, causal_response + " extra"),
            (causal_task, "ACT(E999999,L999999,L999998)"),
        )
        for task, response in malformed:
            with self.subTest(family=task.family, response=response):
                parsed = suite.parse_public_response(task, response)
                self.assertFalse(parsed.valid)
                self.assertEqual(parsed.tokens, ())
                self.assertEqual(parsed.error, "MALFORMED_RESPONSE")


class AttemptAndPhaseGateTests(unittest.TestCase):
    def test_qualification_has_45_io_attempts_plus_first_ordinal_per_family_in(self) -> None:
        evaluator = suite.make_qualification_evaluator()
        with self.assertRaisesRegex(RuntimeError, "preceding phase"):
            evaluator.release_phase("development")
        adaptation = evaluator.release_phase("adaptation")
        arm_counts: Counter[str] = Counter()
        in_tasks: list[suite.PublicEvaluationTask] = []
        for task in adaptation:
            arms = evaluator.expected_arms_for(task.task_id)
            arm_counts.update(arms)
            if "IN_FULL_NEUTRAL_12" in arms:
                in_tasks.append(task)
        self.assertEqual(
            arm_counts,
            {
                "IO_FULL_ORDINARY_12": 45,
                "IN_FULL_NEUTRAL_12": 3,
            },
        )
        self.assertEqual(
            {(task.family, task.ordinal) for task in in_tasks},
            {(family, 0) for family in suite.FAMILIES},
        )
        _finalize_phase_as_unadmitted(evaluator, adaptation)

        development = evaluator.release_phase("development")
        self.assertEqual(len(development), 3)
        self.assertTrue(
            all(
                evaluator.expected_arms_for(task.task_id)
                == suite.EVALUATION_ARMS
                for task in development
            )
        )
        _finalize_phase_as_unadmitted(evaluator, development)
        metrics = evaluator.final_metrics()
        self.assertEqual(sum(item.attempts for item in metrics.aggregates), 72)
        with self.assertRaisesRegex(ValueError, "cannot materialize final"):
            evaluator.release_phase("final")

    def test_attempt_receipts_and_task_arm_pairs_are_one_use(self) -> None:
        evaluator = suite.make_qualification_evaluator()
        tasks = evaluator.release_phase("adaptation")
        first = tasks[0]
        first_arm = evaluator.expected_arms_for(first.task_id)[0]
        receipt = _attempt_ref(first.task_id, first_arm)
        judgment = evaluator.record_attempt(
            first.task_id,
            first_arm,
            receipt,
            "",
            proposal_admitted=False,
        )
        self.assertFalse(judgment.proposal_admitted)
        self.assertFalse(judgment.response_conforms)
        self.assertFalse(judgment.success)
        self.assertEqual(judgment.score, 0.0)
        self.assertEqual(judgment.raw_response, "")
        self.assertEqual(judgment.disposition, "UNSUCCESSFUL")

        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            evaluator.record_attempt(
                first.task_id,
                first_arm,
                _digest("different-receipt"),
                "",
                proposal_admitted=False,
            )
        second = tasks[1]
        second_arm = evaluator.expected_arms_for(second.task_id)[0]
        with self.assertRaisesRegex(RuntimeError, "receipt is already consumed"):
            evaluator.record_attempt(
                second.task_id,
                second_arm,
                receipt,
                "",
                proposal_admitted=False,
            )

    def test_unadmitted_and_malformed_admitted_attempts_finalize_without_repair(self) -> None:
        evaluator = suite.make_qualification_evaluator()
        tasks = evaluator.release_phase("adaptation")
        unadmitted = tasks[0]
        unadmitted_arm = evaluator.expected_arms_for(unadmitted.task_id)[0]
        with self.assertRaisesRegex(ValueError, "empty raw_response"):
            evaluator.record_attempt(
                unadmitted.task_id,
                unadmitted_arm,
                _attempt_ref(unadmitted.task_id, unadmitted_arm),
                "a malformed proposal must not execute",
                proposal_admitted=False,
            )

        malformed = tasks[1]
        malformed_arm = evaluator.expected_arms_for(malformed.task_id)[0]
        judgment = evaluator.record_attempt(
            malformed.task_id,
            malformed_arm,
            _attempt_ref(malformed.task_id, malformed_arm),
            "MALFORMED",
            proposal_admitted=True,
        )
        self.assertTrue(judgment.proposal_admitted)
        self.assertFalse(judgment.response_conforms)
        self.assertFalse(judgment.success)
        self.assertEqual(judgment.score, 0.0)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            evaluator.record_attempt(
                malformed.task_id,
                malformed_arm,
                _digest("repair-attempt"),
                "MALFORMED",
                proposal_admitted=True,
            )

    def test_evaluation_final_requires_completed_development_and_exact_admission(self) -> None:
        evaluator = _make_evaluation()
        with self.assertRaisesRegex(RuntimeError, "development must complete"):
            evaluator.admit_final(_ADMISSION)
        with self.assertRaisesRegex(RuntimeError, "preceding phase"):
            evaluator.release_phase("development")

        adaptation = evaluator.release_phase("adaptation")
        self.assertTrue(
            all(
                evaluator.expected_arms_for(task.task_id)
                == suite.ADAPTATION_ARMS
                for task in adaptation
            )
        )
        _finalize_phase_as_unadmitted(evaluator, adaptation)
        development = evaluator.release_phase("development")
        self.assertTrue(
            all(
                evaluator.expected_arms_for(task.task_id)
                == suite.EVALUATION_ARMS
                for task in development
            )
        )
        _finalize_phase_as_unadmitted(evaluator, development)

        with self.assertRaisesRegex(RuntimeError, "explicit admission"):
            evaluator.release_phase("final")
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluator.admit_final("sha256:" + "b" * 64)
        evaluator.admit_final(_ADMISSION)
        evaluator.admit_final(_ADMISSION)
        final = evaluator.release_phase("final")
        self.assertEqual(len(final), 54)
        self.assertTrue(
            all(
                evaluator.expected_arms_for(task.task_id)
                == suite.EVALUATION_ARMS
                for task in final
            )
        )


if __name__ == "__main__":
    unittest.main()
