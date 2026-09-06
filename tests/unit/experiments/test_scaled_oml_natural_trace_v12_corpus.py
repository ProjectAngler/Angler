from __future__ import annotations

from collections import Counter
import json
import re
import unittest

import experiments.corpora.scaled_oml_natural_trace_v12 as v12


_STEP = re.compile(r"(?:^|(?<=\s))Step [1-9][0-9]*: ")


def _step_sentences(trace: str) -> tuple[str, ...]:
    matches = tuple(_STEP.finditer(trace))
    if not matches or matches[0].start() != 0:
        raise AssertionError("trace does not use the public Step N framing")
    return tuple(
        trace[match.end() : (matches[index + 1].start() if index + 1 < len(matches) else len(trace))].strip()
        for index, match in enumerate(matches)
    )


def _public_values(mechanisms):
    traces = []
    tasks = []
    requests = []
    for mechanism in mechanisms:
        for episode in mechanism.public.episodes:
            traces.append(episode.action_trace_text)
            tasks.append(episode.task_text)
            requests.append(episode.request_text)
        challenge = mechanism.public.challenge
        traces.extend(challenge.candidate_action_trace_texts)
        tasks.append(challenge.task_text)
        requests.append(challenge.request_text)
    return set(traces), set(tasks), set(requests)


class ScaledOmlNaturalTraceV12CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = v12.build_scaled_oml_natural_trace_v12()
        cls.open_corpus = v12.build_scaled_oml_natural_trace_v12(include_final=True)
        cls.outer = tuple(
            cls.corpus.train_outer(update, slot)
            for update in range(v12.TRAIN_OUTER_UPDATES)
            for slot in range(v12.TRAIN_OUTER_SLOTS)
        )

    def test_exact_counts_families_and_frozen_schedule_exposures(self):
        self.assertEqual(len(self.corpus.train_inner), 384)
        self.assertEqual(len(self.outer), 1536)
        self.assertEqual(len(self.corpus.development), 48)
        self.assertEqual(len(self.open_corpus.final), 48)
        self.assertEqual(
            len({value.metadata.generator_family for value in self.corpus.train_inner}),
            48,
        )
        self.assertEqual(
            len({value.metadata.generator_family for value in self.corpus.development}),
            12,
        )
        self.assertEqual(
            len({value.metadata.generator_family for value in self.open_corpus.final}),
            12,
        )

        scheduled = tuple(
            value
            for update in range(v12.TRAIN_OUTER_UPDATES)
            for value in self.corpus.train_inner_for_update(update)
        )
        exposures = Counter(value.metadata.mechanism_ref for value in scheduled)
        self.assertEqual(len(exposures), 384)
        self.assertEqual(set(exposures.values()), {4})
        for update in range(v12.TRAIN_OUTER_UPDATES):
            values = self.corpus.train_inner_for_update(update)
            self.assertEqual(len(values), 8)
            self.assertEqual(len({value.metadata.generator_family for value in values}), 8)

    def test_outer_mechanisms_are_fresh_unique_and_reproducible(self):
        refs = [value.metadata.mechanism_ref for value in self.outer]
        payloads = [
            json.dumps(value.to_learner_payload(), sort_keys=True, ensure_ascii=False)
            for value in self.outer
        ]
        self.assertEqual(len(set(refs)), 1536)
        self.assertEqual(len(set(payloads)), 1536)
        self.assertEqual(
            self.corpus.train_outer(73, 5),
            v12.build_scaled_oml_natural_trace_v12().train_outer(73, 5),
        )
        with self.assertRaises(ValueError):
            self.corpus.train_outer(-1, 0)
        with self.assertRaises(ValueError):
            self.corpus.train_outer(0, 8)

    def test_family_lengths_transitions_signatures_ngrams_and_vocabulary(self):
        by_partition = {
            partition: tuple(
                value for value in v12._FAMILIES if value.partition == partition
            )
            for partition in ("train", "development", "final")
        }
        expected = {"train": 12, "development": 3, "final": 3}
        signatures = {}
        ngrams = {}
        actions = {}
        for partition, families in by_partition.items():
            self.assertEqual(
                Counter(len(value.support_steps) for value in families),
                Counter({3: expected[partition], 4: expected[partition], 5: expected[partition], 6: expected[partition]}),
            )
            self.assertEqual(
                Counter(value.transition for value in families),
                Counter({
                    "insertion": expected[partition],
                    "removal": expected[partition],
                    "reorder": expected[partition],
                    "replacement": expected[partition],
                }),
            )
            signatures[partition] = {
                (value.support_signature, value.challenge_signature)
                for value in families
            }
            self.assertEqual(len(signatures[partition]), len(families))
            ngrams[partition] = {
                ngram
                for value in families
                for ngram in (
                    v12._ordered_ngrams(value.support_steps)
                    | v12._ordered_ngrams(value.challenge_steps)
                )
            }
            actions[partition] = {
                action
                for value in families
                for action in (*value.support_steps, *value.challenge_steps)
            }
        for left, right in (
            ("train", "development"),
            ("train", "final"),
            ("development", "final"),
        ):
            self.assertTrue(signatures[left].isdisjoint(signatures[right]))
            self.assertTrue(ngrams[left].isdisjoint(ngrams[right]))
        self.assertLessEqual(actions["development"], actions["train"])
        self.assertLessEqual(actions["final"], actions["train"])

    def test_paraphrase_plans_and_latin_contexts_are_diverse(self):
        family = v12._TRAIN_FAMILY_VALUES[0]
        contexts = [v12._context("inner", family, index) for index in range(128)]
        context_tuples = {
            tuple(value[name] for name in ("system", "artifact", "change", "constraint", "check"))
            for value in contexts
        }
        self.assertEqual(len(context_tuples), 128)
        context = contexts[0]
        for action in v12._ACTION_KEYS:
            rendered = {
                v12._render_action(
                    action,
                    context,
                    surface_ordinal=0,
                    plan_override=plan,
                )
                for plan in range(8)
            }
            self.assertEqual(len(rendered), 8)

        for family_index, family in enumerate(v12._TRAIN_FAMILY_VALUES):
            task_ids = []
            request_ids = []
            for local_index in range(8):
                mechanism_index = family_index * 8 + local_index
                base = mechanism_index * 17
                for offset in (*range(6), 7):
                    task_ids.append(
                        v12._template_id(
                            "inner", family, base + offset, request=False
                        )
                    )
                    request_ids.append(
                        v12._template_id(
                            "inner", family, base + offset, request=True
                        )
                    )
            self.assertEqual(Counter(task_ids), Counter({0: 14, 1: 14, 2: 14, 3: 14}))
            self.assertEqual(Counter(request_ids), Counter({0: 14, 1: 14, 2: 14, 3: 14}))

            outer_task_ids = []
            outer_request_ids = []
            for local_index in range(32):
                mechanism_index = family_index + 48 * local_index
                base = 1_000_000 + mechanism_index * 17 + local_index
                for offset in (*range(6), 7):
                    outer_task_ids.append(
                        v12._template_id(
                            "outer", family, base + offset, request=False
                        )
                    )
                    outer_request_ids.append(
                        v12._template_id(
                            "outer", family, base + offset, request=True
                        )
                    )
            self.assertEqual(Counter(outer_task_ids), Counter({4: 112, 5: 112}))
            self.assertEqual(Counter(outer_request_ids), Counter({4: 112, 5: 112}))

    def test_inner_trace_uniqueness_and_all_split_boundaries(self):
        inner_episode_traces = [
            episode.action_trace_text
            for mechanism in self.corpus.train_inner
            for episode in mechanism.public.episodes
        ]
        self.assertEqual(len(inner_episode_traces), 2304)
        self.assertGreaterEqual(
            len(set(inner_episode_traces)) / len(inner_episode_traces),
            0.95,
        )
        partitions = {
            "inner": self.corpus.train_inner,
            "outer": self.outer,
            "development": self.corpus.development,
            "final": self.open_corpus.final,
        }
        public = {name: _public_values(values) for name, values in partitions.items()}
        for left, right in (
            ("inner", "outer"),
            ("inner", "development"),
            ("inner", "final"),
            ("outer", "development"),
            ("outer", "final"),
            ("development", "final"),
        ):
            with self.subTest(left=left, right=right):
                for field in range(3):
                    self.assertTrue(public[left][field].isdisjoint(public[right][field]))

    def test_balanced_outcomes_candidates_positions_lengths_and_transitions(self):
        partitions = {
            "inner": self.corpus.train_inner,
            "outer": self.outer,
            "development": self.corpus.development,
            "final": self.open_corpus.final,
        }
        for name, mechanisms in partitions.items():
            positions = Counter()
            transitions = Counter()
            shorter = 0
            longer = 0
            for mechanism in mechanisms:
                outcomes = [value.outcome_label for value in mechanism.public.episodes]
                self.assertEqual(outcomes.count("SUCCESS"), 3)
                self.assertEqual(outcomes.count("FAILURE"), 3)
                self.assertGreater(
                    sum(left != right for left, right in zip(outcomes, outcomes[1:])),
                    1,
                )
                candidates = mechanism.public.challenge.candidate_action_trace_texts
                target = mechanism.supervision.target_successful_candidate_indices[0]
                lengths = [len(_step_sentences(value)) for value in candidates]
                target_length = lengths[target]
                self.assertGreaterEqual(lengths.count(target_length), 3)
                shorter += int(any(value < target_length for value in lengths))
                longer += int(any(value > target_length for value in lengths))
                positions[target] += 1
                transitions[mechanism.metadata.heldout_variant] += 1
                serialized = " ".join(candidates).lower()
                self.assertNotIn("blind_change", serialized)
                self.assertNotIn("blind change", serialized)
            self.assertEqual(set(positions.values()), {len(mechanisms) // 4})
            self.assertEqual(set(transitions.values()), {len(mechanisms) // 4})
            self.assertEqual(shorter, len(mechanisms) // 2)
            self.assertEqual(longer, len(mechanisms) // 2)

    def test_evaluator_contrasts_preserve_multiset_and_stay_out_of_payload(self):
        mechanism = self.corpus.development[0]
        contrasts = v12.build_trace_representation_contrasts(mechanism)
        reference = _step_sentences(contrasts.reference_trace_text)
        reordered = _step_sentences(contrasts.reordered_trace_text)
        paraphrase = _step_sentences(contrasts.paraphrase_trace_text)
        self.assertEqual(Counter(reference), Counter(reordered))
        self.assertNotEqual(reference, reordered)
        self.assertEqual(len(reference), len(paraphrase))
        self.assertTrue(all(left != right for left, right in zip(reference, paraphrase)))
        self.assertEqual(len(_step_sentences(contrasts.omitted_trace_text)), len(reference) - 1)
        self.assertEqual(len(_step_sentences(contrasts.inserted_trace_text)), len(reference) + 1)
        self.assertEqual(len(_step_sentences(contrasts.replacement_trace_text)), len(reference))
        payload = json.dumps(mechanism.to_learner_payload(), sort_keys=True)
        for value in contrasts.__dataclass_fields__:
            self.assertNotIn(value, payload)

    def test_learner_payload_has_only_public_schema_and_no_hidden_tokens(self):
        forbidden_keys = {
            "partition",
            "mechanism_ref",
            "generator_family",
            "support_structure_signature",
            "challenge_structure_signature",
            "heldout_variant",
            "target_successful_candidate_indices",
            "relevant_failed_candidate_indices",
            "transition",
            "renderer",
            "plan_id",
            "seed",
            "action_key",
        }

        def keys(value):
            if isinstance(value, dict):
                result = set(value)
                for child in value.values():
                    result.update(keys(child))
                return result
            if isinstance(value, list):
                result = set()
                for child in value:
                    result.update(keys(child))
                return result
            return set()

        samples = (
            *self.corpus.train_inner,
            *self.outer,
            *self.corpus.development,
            *self.open_corpus.final,
        )
        for mechanism in samples:
            payload = mechanism.to_learner_payload()
            self.assertTrue(keys(payload).isdisjoint(forbidden_keys))
            text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(mechanism.metadata.mechanism_ref, text)
            self.assertNotIn(mechanism.metadata.generator_family, text)
            self.assertNotIn(mechanism.metadata.heldout_variant, text)
            for action in v12._ACTION_KEYS:
                self.assertNotIn(action, text)

    def test_final_seal_gradient_boundary_and_reproducibility(self):
        self.assertFalse(self.corpus.final_is_open)
        with self.assertRaises(v12.FinalPartitionSealedError):
            _ = self.corpus.final
        with self.assertRaises(v12.FinalPartitionSealedError):
            self.corpus.partition("final")
        self.assertTrue(self.open_corpus.final_is_open)
        self.assertEqual(self.corpus.train_inner, self.open_corpus.train_inner)
        self.assertEqual(self.corpus.development, self.open_corpus.development)
        self.assertTrue(all(value.gradient_authorized for value in self.corpus.train_inner))
        self.assertTrue(all(value.gradient_authorized for value in self.outer))
        self.assertTrue(all(not value.gradient_authorized for value in self.corpus.development))
        self.assertTrue(all(not value.gradient_authorized for value in self.open_corpus.final))
        self.assertEqual(self.corpus, v12.build_scaled_oml_natural_trace_v12())
        self.assertEqual(
            self.open_corpus,
            v12.build_scaled_oml_natural_trace_v12(include_final=True),
        )


if __name__ == "__main__":
    unittest.main()
