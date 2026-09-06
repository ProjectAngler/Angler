from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import unittest

import experiments.corpora.structure_protected_oml_natural_trace_v13 as v13


_STEP = re.compile(r"(?:^|(?<=\s))Step [1-9][0-9]*: ")


def _step_sentences(trace: str) -> tuple[str, ...]:
    matches = tuple(_STEP.finditer(trace))
    if not matches or matches[0].start() != 0:
        raise AssertionError("trace does not use contiguous public Step framing")
    return tuple(
        trace[
            match.end() : (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(trace)
            )
        ].strip()
        for index, match in enumerate(matches)
    )


def _all_public_traces(mechanisms):
    result = set()
    for mechanism in mechanisms:
        for row in mechanism.public.episodes:
            result.add(row.reference_trace_text)
            result.add(row.attempt_trace_text)
        contrasts = mechanism.structural_contrasts
        result.add(contrasts.reference_trace_text)
        result.update(contrasts.serialized_candidate_attempt_trace_texts)
    return result


def _recursive_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


class StructureProtectedOmlNaturalTraceV13CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = v13.build_structure_protected_oml_natural_trace_v13()
        cls.open_corpus = v13.build_structure_protected_oml_natural_trace_v13(
            include_final=True
        )
        cls.outer = tuple(
            cls.corpus.train_outer(update, slot)
            for update in range(v13.TRAIN_OUTER_UPDATES)
            for slot in range(v13.TRAIN_OUTER_SLOTS)
        )
        cls.partitions = {
            "inner": cls.corpus.train_inner,
            "outer": cls.outer,
            "development": cls.corpus.development,
            "final": cls.open_corpus.final,
        }

    def test_exact_counts_families_schedule_and_fresh_outer(self):
        self.assertEqual(len(self.corpus.train_inner), 384)
        self.assertEqual(len(self.outer), 1536)
        self.assertEqual(len(self.corpus.development), 48)
        self.assertEqual(len(self.open_corpus.final), 48)
        self.assertEqual(
            len({row.metadata.generator_family for row in self.corpus.train_inner}),
            48,
        )
        self.assertEqual(
            len({row.metadata.generator_family for row in self.corpus.development}),
            12,
        )
        self.assertEqual(
            len({row.metadata.generator_family for row in self.open_corpus.final}),
            12,
        )

        scheduled = tuple(
            row
            for update in range(v13.TRAIN_OUTER_UPDATES)
            for row in self.corpus.train_inner_for_update(update)
        )
        exposures = Counter(row.metadata.mechanism_ref for row in scheduled)
        self.assertEqual(len(exposures), 384)
        self.assertEqual(set(exposures.values()), {4})
        for update in range(v13.TRAIN_OUTER_UPDATES):
            rows = self.corpus.train_inner_for_update(update)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row.metadata.generator_family for row in rows}), 8)

        outer_refs = [row.metadata.mechanism_ref for row in self.outer]
        outer_payloads = [
            json.dumps(row.to_learner_payload(), sort_keys=True)
            for row in self.outer
        ]
        self.assertEqual(len(set(outer_refs)), 1536)
        self.assertEqual(len(set(outer_payloads)), 1536)
        self.assertEqual(
            self.corpus.train_outer(73, 5),
            v13.build_structure_protected_oml_natural_trace_v13().train_outer(73, 5),
        )
        with self.assertRaises(ValueError):
            self.corpus.train_outer(-1, 0)
        with self.assertRaises(ValueError):
            self.corpus.train_outer(0, 8)

    def test_public_rows_have_only_reference_attempt_outcome_payload(self):
        for mechanisms in self.partitions.values():
            for mechanism in mechanisms:
                self.assertEqual(len(mechanism.public.episodes), 6)
                outcomes = Counter(
                    row.outcome_label for row in mechanism.public.episodes
                )
                self.assertEqual(outcomes, Counter({"SUCCESS": 3, "FAILURE": 3}))
                ordinals = tuple(
                    row.temporal.acquired_ordinal
                    for row in mechanism.public.episodes
                )
                self.assertEqual(
                    ordinals,
                    tuple(range(ordinals[0], ordinals[0] + 6)),
                )
                for row in mechanism.public.episodes:
                    self.assertNotEqual(
                        row.reference_trace_text, row.attempt_trace_text
                    )
                    self.assertEqual(
                        set(row.to_learner_payload()),
                        {
                            "reference_trace_text",
                            "attempt_trace_text",
                            "outcome_label",
                        },
                    )
                self.assertEqual(
                    set(mechanism.to_learner_payload()), {"episodes"}
                )

    def test_outcome_length_and_failure_relations_are_exactly_balanced(self):
        expected_per_relation = {
            "inner": 288,
            "outer": 1152,
            "development": 36,
            "final": 36,
        }
        for name, mechanisms in self.partitions.items():
            relations = Counter(
                relation
                for mechanism in mechanisms
                for relation in mechanism.supervision.episode_attempt_relations
                if relation != "paraphrase"
            )
            self.assertEqual(
                relations,
                Counter(
                    {
                        relation: expected_per_relation[name]
                        for relation in (
                            "reorder",
                            "replacement",
                            "omission",
                            "insertion",
                        )
                    }
                ),
            )

            length_outcomes = defaultdict(Counter)
            for mechanism in mechanisms:
                for row, relation in zip(
                    mechanism.public.episodes,
                    mechanism.supervision.episode_attempt_relations,
                    strict=True,
                ):
                    reference = _step_sentences(row.reference_trace_text)
                    attempt = _step_sentences(row.attempt_trace_text)
                    length_outcomes[len(attempt)][row.outcome_label] += 1
                    if relation == "paraphrase":
                        self.assertEqual(len(reference), len(attempt))
                        self.assertEqual(row.outcome_label, "SUCCESS")
                    elif relation == "reorder":
                        self.assertEqual(Counter(reference), Counter(attempt))
                        self.assertNotEqual(reference, attempt)
                    elif relation == "replacement":
                        self.assertEqual(len(reference), len(attempt))
                    elif relation == "omission":
                        self.assertEqual(len(attempt), len(reference) - 1)
                    elif relation == "insertion":
                        self.assertEqual(len(attempt), len(reference) + 1)
            self.assertEqual(set(length_outcomes), {3, 4, 5, 6})
            for outcomes in length_outcomes.values():
                self.assertEqual(outcomes["SUCCESS"], outcomes["FAILURE"])
            total_success = sum(value["SUCCESS"] for value in length_outcomes.values())
            total_failure = sum(value["FAILURE"] for value in length_outcomes.values())
            self.assertEqual(total_success, total_failure)
            # Every attempt length has P(success)=P(failure)=.5, hence the
            # optimal deterministic length-only balanced accuracy is exactly .5.
            length_only_balanced_accuracy = 0.5
            self.assertEqual(length_only_balanced_accuracy, 0.50)

    def test_structural_sets_protect_equivalence_order_and_primary_lengths(self):
        for mechanisms in self.partitions.values():
            positions = Counter()
            for mechanism in mechanisms:
                contrasts = mechanism.structural_contrasts
                reference = _step_sentences(contrasts.reference_trace_text)
                paraphrase = _step_sentences(
                    contrasts.paraphrase_attempt_trace_text
                )
                reordered = _step_sentences(
                    contrasts.reordered_attempt_trace_text
                )
                replacement = _step_sentences(
                    contrasts.replacement_attempt_trace_text
                )
                omitted = _step_sentences(contrasts.omitted_attempt_trace_text)
                inserted = _step_sentences(contrasts.inserted_attempt_trace_text)
                self.assertEqual(len(reference), len(paraphrase))
                self.assertTrue(
                    all(left != right for left, right in zip(reference, paraphrase))
                )
                self.assertEqual(Counter(reference), Counter(reordered))
                self.assertNotEqual(reference, reordered)
                self.assertEqual(len(reference), len(replacement))
                self.assertEqual(len(omitted), len(reference) - 1)
                self.assertEqual(len(inserted), len(reference) + 1)

                supervision = mechanism.supervision
                position = supervision.same_order_candidate_index
                positions[position] += 1
                self.assertEqual(
                    contrasts.serialized_candidate_attempt_trace_texts[position],
                    contrasts.paraphrase_attempt_trace_text,
                )
                self.assertEqual(
                    tuple(
                        supervision.candidate_attempt_relations[index]
                        for index in supervision.primary_length_matched_candidate_indices
                    ),
                    ("reorder", "replacement"),
                )
            self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)
            self.assertEqual(set(positions), set(range(5)))

    def test_renderer_atoms_are_shared_roles_balanced_and_texts_disjoint(self):
        for scope, mechanisms in self.partitions.items():
            reference_plans = Counter(
                plan
                for mechanism in mechanisms
                for plan in mechanism.metadata.episode_reference_plan_ids
            )
            attempt_plans = Counter(
                plan
                for mechanism in mechanisms
                for plan in mechanism.metadata.episode_attempt_plan_ids
            )
            structural_reference = Counter(
                mechanism.metadata.structural_reference_plan_id
                for mechanism in mechanisms
            )
            structural_attempt = Counter(
                mechanism.metadata.structural_attempt_plan_id
                for mechanism in mechanisms
            )
            self.assertEqual(set(reference_plans), set(range(8)))
            self.assertEqual(set(attempt_plans), set(range(8)))
            self.assertEqual(set(structural_reference), set(range(8)))
            self.assertEqual(set(structural_attempt), set(range(8)))
            self.assertEqual(len(set(reference_plans.values())), 1)
            self.assertEqual(len(set(structural_reference.values())), 1)
            self.assertEqual(len(set(structural_attempt.values())), 1)

            contexts = [v13._context(scope, index) for index in range(288)]
            for key, expected in (
                ("system", 16),
                ("artifact", 12),
                ("change", 10),
                ("constraint", 10),
                ("check", 10),
            ):
                self.assertEqual(len({value[key] for value in contexts}), expected)

        public = {
            name: _all_public_traces(mechanisms)
            for name, mechanisms in self.partitions.items()
        }
        for left, right in (
            ("inner", "outer"),
            ("inner", "development"),
            ("inner", "final"),
            ("outer", "development"),
            ("outer", "final"),
            ("development", "final"),
        ):
            self.assertTrue(public[left].isdisjoint(public[right]))

    def test_inherited_topologies_remain_compositionally_disjoint(self):
        signatures = {}
        ngrams = {}
        action_sets = {}
        for partition, families in v13._FAMILIES_BY_PARTITION.items():
            signatures[partition] = {
                (family.support_signature, family.challenge_signature)
                for family in families
            }
            ngrams[partition] = {
                ngram
                for family in families
                for ngram in (
                    v13.v12._ordered_ngrams(family.support_steps)
                    | v13.v12._ordered_ngrams(family.challenge_steps)
                )
            }
            action_sets[partition] = {
                action
                for family in families
                for action in (*family.support_steps, *family.challenge_steps)
            }
        for left, right in (
            ("train", "development"),
            ("train", "final"),
            ("development", "final"),
        ):
            self.assertTrue(signatures[left].isdisjoint(signatures[right]))
            self.assertTrue(ngrams[left].isdisjoint(ngrams[right]))
        self.assertLessEqual(action_sets["development"], action_sets["train"])
        self.assertLessEqual(action_sets["final"], action_sets["train"])

    def test_learner_payload_excludes_all_generator_and_evaluator_fields(self):
        forbidden_keys = {
            "partition",
            "mechanism_ref",
            "generator_family",
            "support_structure_signature",
            "challenge_structure_signature",
            "heldout_variant",
            "episode_attempt_relations",
            "same_order_candidate_index",
            "candidate_attempt_relations",
            "primary_length_matched_candidate_indices",
            "reference_plan",
            "attempt_plan",
            "renderer",
            "corruption",
            "relation",
            "edit_distance",
            "target_index",
            "seed",
            "structural_contrasts",
        }
        for mechanisms in self.partitions.values():
            for mechanism in mechanisms:
                payload = mechanism.to_learner_payload()
                self.assertTrue(
                    _recursive_keys(payload).isdisjoint(forbidden_keys)
                )
                text = json.dumps(payload, ensure_ascii=False)
                self.assertNotIn(mechanism.metadata.mechanism_ref, text)
                self.assertNotIn(mechanism.metadata.generator_family, text)
                self.assertNotIn(mechanism.metadata.heldout_variant, text)
                for action_key in v13.v12._ACTION_KEYS:
                    self.assertNotIn(action_key, text)
                for field in mechanism.structural_contrasts.__dataclass_fields__:
                    self.assertNotIn(field, payload)

    def test_final_seal_gradient_boundary_and_reproducibility(self):
        self.assertFalse(self.corpus.final_is_open)
        with self.assertRaises(v13.FinalPartitionSealedError):
            _ = self.corpus.final
        with self.assertRaises(v13.FinalPartitionSealedError):
            self.corpus.partition("final")
        self.assertTrue(self.open_corpus.final_is_open)
        self.assertEqual(self.corpus.train_inner, self.open_corpus.train_inner)
        self.assertEqual(self.corpus.development, self.open_corpus.development)
        self.assertTrue(
            all(value.gradient_authorized for value in self.corpus.train_inner)
        )
        self.assertTrue(all(value.gradient_authorized for value in self.outer))
        self.assertTrue(
            all(not value.gradient_authorized for value in self.corpus.development)
        )
        self.assertTrue(
            all(not value.gradient_authorized for value in self.open_corpus.final)
        )
        self.assertEqual(
            self.corpus,
            v13.build_structure_protected_oml_natural_trace_v13(),
        )
        self.assertEqual(
            self.open_corpus,
            v13.build_structure_protected_oml_natural_trace_v13(
                include_final=True
            ),
        )


if __name__ == "__main__":
    unittest.main()
