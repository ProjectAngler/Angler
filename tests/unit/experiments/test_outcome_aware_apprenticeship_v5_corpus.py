import json
import unittest

from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    CANDIDATES_PER_CHALLENGE,
    DEVELOPMENT_MECHANISMS,
    EPISODES_PER_MECHANISM,
    FINAL_MECHANISMS,
    TRAIN_MECHANISMS,
    build_outcome_aware_apprenticeship_v5,
)


class OutcomeAwareApprenticeshipV5CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_outcome_aware_apprenticeship_v5()

    def test_frozen_partition_sizes_and_generator_disjointness(self):
        self.assertEqual(len(self.corpus.train), TRAIN_MECHANISMS)
        self.assertEqual(len(self.corpus.development), DEVELOPMENT_MECHANISMS)
        self.assertEqual(len(self.corpus.final), FINAL_MECHANISMS)

        family_sets = []
        structure_sets = []
        for partition in ("train", "development", "final"):
            values = self.corpus.partition(partition)
            family_sets.append({value.metadata.generator_family for value in values})
            structure_sets.append(
                {
                    (
                        value.metadata.support_structure_signature,
                        value.metadata.challenge_structure_signature,
                    )
                    for value in values
                }
            )
            self.assertTrue(
                all(
                    value.metadata.support_structure_signature
                    != value.metadata.challenge_structure_signature
                    for value in values
                )
            )
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(family_sets[left].isdisjoint(family_sets[right]))
                self.assertTrue(structure_sets[left].isdisjoint(structure_sets[right]))

    def test_chronology_is_contiguous_balanced_and_mixed(self):
        expected_ordinal = 0
        for mechanism in self.corpus.all_mechanisms:
            episodes = mechanism.public.episodes
            self.assertEqual(len(episodes), EPISODES_PER_MECHANISM)
            self.assertEqual(
                [row.temporal.acquired_ordinal for row in episodes],
                list(range(expected_ordinal, expected_ordinal + EPISODES_PER_MECHANISM)),
            )
            self.assertEqual(
                [row.temporal.age for row in episodes],
                list(reversed(range(EPISODES_PER_MECHANISM))),
            )
            outcomes = [row.outcome_label for row in episodes]
            self.assertEqual(outcomes.count("SUCCESS"), 3)
            self.assertEqual(outcomes.count("FAILURE"), 3)
            self.assertGreater(sum(left != right for left, right in zip(outcomes, outcomes[1:])), 1)
            self.assertEqual(episodes[0].temporal.landmark_relations, (("stream_start", "AT"),))
            self.assertTrue(
                all(
                    row.temporal.landmark_relations == (("stream_start", "AFTER"),)
                    for row in episodes[1:]
                )
            )
            expected_ordinal += EPISODES_PER_MECHANISM

    def test_public_projection_contains_natural_experience_but_no_hidden_labels(self):
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

        for mechanism in self.corpus.all_mechanisms:
            payload = mechanism.to_learner_payload()
            self.assertTrue(keys(payload).isdisjoint(forbidden_keys))
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(mechanism.metadata.mechanism_ref, encoded)
            self.assertNotIn(mechanism.metadata.generator_family, encoded)
            for episode in mechanism.public.episodes:
                self.assertIn("workspace", episode.task_text)
                self.assertIn("Step 1:", episode.action_trace_text)
                self.assertIn("Objective verification", episode.objective_diagnostic_text)

    def test_candidate_supervision_is_valid_separate_and_position_balanced(self):
        target_positions = set()
        for mechanism in self.corpus.all_mechanisms:
            candidate_count = len(mechanism.public.challenge.candidate_action_trace_texts)
            self.assertEqual(candidate_count, CANDIDATES_PER_CHALLENGE)
            targets = mechanism.supervision.target_successful_candidate_indices
            failed = mechanism.supervision.relevant_failed_candidate_indices
            self.assertEqual(len(targets), 1)
            self.assertEqual(len(failed), CANDIDATES_PER_CHALLENGE - 1)
            self.assertTrue(set(targets).isdisjoint(failed))
            self.assertEqual(set((*targets, *failed)), set(range(candidate_count)))
            target_positions.update(targets)
            payload = mechanism.to_learner_payload()
            self.assertNotIn("target_successful_candidate_indices", payload)
            self.assertNotIn("relevant_failed_candidate_indices", payload)
        self.assertEqual(target_positions, set(range(CANDIDATES_PER_CHALLENGE)))

    def test_only_train_partition_is_gradient_authorized(self):
        self.assertTrue(all(value.gradient_authorized for value in self.corpus.train))
        self.assertTrue(all(not value.gradient_authorized for value in self.corpus.development))
        self.assertTrue(all(not value.gradient_authorized for value in self.corpus.final))

    def test_build_is_exactly_reproducible_and_small(self):
        rebuilt = build_outcome_aware_apprenticeship_v5()
        self.assertEqual(self.corpus, rebuilt)
        self.assertLess(self.corpus.learner_payload_size_bytes(), 2_000_000)


if __name__ == "__main__":
    unittest.main()
