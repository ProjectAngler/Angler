import json
import unittest

from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CANDIDATES_PER_CHALLENGE,
    DEVELOPMENT_MECHANISMS,
    EPISODES_PER_MECHANISM,
    FINAL_MECHANISMS,
    FinalPartitionSealedError,
    TRAIN_GENERATOR_FAMILIES,
    TRAIN_MECHANISMS,
    build_causal_neuromodulated_apprenticeship_v6,
)


class CausalNeuromodulatedApprenticeshipV6CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_causal_neuromodulated_apprenticeship_v6()
        cls.open_corpus = build_causal_neuromodulated_apprenticeship_v6(
            include_final=True
        )

    def test_exact_counts_and_broad_disjoint_generators(self):
        self.assertEqual(len(self.corpus.train), TRAIN_MECHANISMS)
        self.assertEqual(len(self.corpus.development), DEVELOPMENT_MECHANISMS)
        self.assertEqual(len(self.open_corpus.final), FINAL_MECHANISMS)
        families = {
            partition: {
                value.metadata.generator_family
                for value in self.open_corpus.partition(partition)
            }
            for partition in ("train", "development", "final")
        }
        self.assertEqual(len(families["train"]), TRAIN_GENERATOR_FAMILIES)
        self.assertGreaterEqual(len(families["train"]), 12)
        self.assertEqual(len(families["development"]), 4)
        self.assertEqual(len(families["final"]), 4)
        self.assertTrue(families["train"].isdisjoint(families["development"]))
        self.assertTrue(families["train"].isdisjoint(families["final"]))
        self.assertTrue(families["development"].isdisjoint(families["final"]))

        topology = {
            partition: {
                (
                    value.metadata.support_structure_signature,
                    value.metadata.challenge_structure_signature,
                )
                for value in self.open_corpus.partition(partition)
            }
            for partition in ("train", "development", "final")
        }
        self.assertTrue(topology["train"].isdisjoint(topology["development"]))
        self.assertTrue(topology["train"].isdisjoint(topology["final"]))
        self.assertTrue(topology["development"].isdisjoint(topology["final"]))
        self.assertTrue(
            all(
                value.metadata.support_structure_signature
                != value.metadata.challenge_structure_signature
                for value in self.open_corpus.all_mechanisms
            )
        )

    def test_balanced_interleaved_chronology_and_moving_origin(self):
        expected_ordinal = 0
        for mechanism in self.open_corpus.all_mechanisms:
            episodes = mechanism.public.episodes
            self.assertEqual(len(episodes), EPISODES_PER_MECHANISM)
            self.assertEqual(
                [value.temporal.acquired_ordinal for value in episodes],
                list(range(expected_ordinal, expected_ordinal + EPISODES_PER_MECHANISM)),
            )
            self.assertEqual(
                [value.temporal.age for value in episodes],
                list(reversed(range(EPISODES_PER_MECHANISM))),
            )
            outcomes = [value.outcome_label for value in episodes]
            self.assertEqual(outcomes.count("SUCCESS"), 3)
            self.assertEqual(outcomes.count("FAILURE"), 3)
            self.assertGreater(
                sum(left != right for left, right in zip(outcomes, outcomes[1:])),
                1,
            )
            self.assertEqual(
                episodes[0].temporal.landmark_relations,
                (("stream_start", "AT"),),
            )
            self.assertTrue(
                all(
                    value.temporal.landmark_relations == (("stream_start", "AFTER"),)
                    for value in episodes[1:]
                )
            )
            expected_ordinal += EPISODES_PER_MECHANISM

    def test_learner_payload_excludes_generator_and_answer_metadata(self):
        forbidden_keys = {
            "partition",
            "mechanism_ref",
            "generator_family",
            "support_structure_signature",
            "challenge_structure_signature",
            "heldout_variant",
            "target_successful_candidate_indices",
            "relevant_failed_candidate_indices",
            "seed",
            "event_id",
        }

        def recursive_keys(value):
            if isinstance(value, dict):
                result = set(value)
                for child in value.values():
                    result.update(recursive_keys(child))
                return result
            if isinstance(value, list):
                result = set()
                for child in value:
                    result.update(recursive_keys(child))
                return result
            return set()

        for mechanism in self.open_corpus.all_mechanisms:
            payload = mechanism.to_learner_payload()
            self.assertTrue(recursive_keys(payload).isdisjoint(forbidden_keys))
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(mechanism.metadata.mechanism_ref, encoded)
            self.assertNotIn(mechanism.metadata.generator_family, encoded)
            for episode in mechanism.public.episodes:
                self.assertIn("workspace", episode.task_text)
                self.assertIn("Step 1:", episode.action_trace_text)
                self.assertIn("Objective verification", episode.objective_diagnostic_text)

    def test_four_candidate_targets_are_exactly_position_balanced(self):
        expected_per_position = {
            "train": TRAIN_MECHANISMS // CANDIDATES_PER_CHALLENGE,
            "development": DEVELOPMENT_MECHANISMS // CANDIDATES_PER_CHALLENGE,
            "final": FINAL_MECHANISMS // CANDIDATES_PER_CHALLENGE,
        }
        for partition in ("train", "development", "final"):
            counts = [0] * CANDIDATES_PER_CHALLENGE
            for mechanism in self.open_corpus.partition(partition):
                candidates = mechanism.public.challenge.candidate_action_trace_texts
                self.assertEqual(len(candidates), CANDIDATES_PER_CHALLENGE)
                targets = mechanism.supervision.target_successful_candidate_indices
                failed = mechanism.supervision.relevant_failed_candidate_indices
                self.assertEqual(len(targets), 1)
                self.assertEqual(len(failed), CANDIDATES_PER_CHALLENGE - 1)
                self.assertEqual(set((*targets, *failed)), set(range(CANDIDATES_PER_CHALLENGE)))
                counts[targets[0]] += 1
            self.assertEqual(counts, [expected_per_position[partition]] * CANDIDATES_PER_CHALLENGE)

    def test_only_training_partition_authorizes_gradients(self):
        self.assertTrue(all(value.gradient_authorized for value in self.corpus.train))
        self.assertTrue(all(not value.gradient_authorized for value in self.corpus.development))
        self.assertTrue(all(not value.gradient_authorized for value in self.open_corpus.final))

    def test_final_is_lazy_and_requires_explicit_authorization(self):
        self.assertFalse(self.corpus.final_is_open)
        self.assertTrue(self.open_corpus.final_is_open)
        with self.assertRaises(FinalPartitionSealedError):
            _ = self.corpus.final
        with self.assertRaises(FinalPartitionSealedError):
            self.corpus.partition("final")
        self.assertEqual(
            self.corpus.train,
            self.open_corpus.train,
        )
        self.assertEqual(
            self.corpus.development,
            self.open_corpus.development,
        )

    def test_reproducible_and_small_enough_for_dual_gpu_runner(self):
        rebuilt = build_causal_neuromodulated_apprenticeship_v6()
        self.assertEqual(self.corpus, rebuilt)
        opened_rebuilt = build_causal_neuromodulated_apprenticeship_v6(
            include_final=True
        )
        self.assertEqual(self.open_corpus, opened_rebuilt)
        self.assertLess(self.open_corpus.learner_payload_size_bytes(), 2_000_000)


if __name__ == "__main__":
    unittest.main()
