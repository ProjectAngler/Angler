from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import unittest

from experiments.corpora import paired_latent_contingency_credit_v15 as v15
from experiments.corpora import structure_protected_oml_natural_trace_v13 as v13


_STEP = re.compile(r"(?:^|(?<=\s))Step [1-9][0-9]*: ")


def _sentences(trace: str) -> tuple[str, ...]:
    matches = tuple(_STEP.finditer(trace))
    if not matches or matches[0].start() != 0:
        raise AssertionError("trace is not Step-framed")
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


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def _public_traces(pairs):
    return {
        trace
        for pair in pairs
        for event in pair.first.public.events
        for trace in (event.reference_trace_text, event.attempt_trace_text)
    }


class PairedLatentContingencyCreditV15CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = v15.build_paired_latent_contingency_credit_v15()
        cls.open_corpus = v15.build_paired_latent_contingency_credit_v15(
            include_final=True
        )
        cls.partitions = {
            "train": cls.corpus.train,
            "development": cls.corpus.development,
            "final": cls.open_corpus.final,
        }

    def test_exact_counts_families_transitions_and_unique_identities(self):
        self.assertEqual(len(self.corpus.train), 384)
        self.assertEqual(len(self.corpus.development), 48)
        self.assertEqual(len(self.open_corpus.final), 48)
        expected_families = {"train": 48, "development": 12, "final": 12}
        expected_per_family = {"train": 8, "development": 4, "final": 4}
        for name, pairs in self.partitions.items():
            families = Counter(pair.first.metadata.generator_family for pair in pairs)
            transitions = Counter(pair.first.metadata.transition_group for pair in pairs)
            self.assertEqual(len(families), expected_families[name])
            self.assertEqual(set(families.values()), {expected_per_family[name]})
            self.assertEqual(set(transitions), {"insertion", "removal", "reorder", "replacement"})
            self.assertEqual(len(set(transitions.values())), 1)
            refs = [pair.pair_ref for pair in pairs]
            episode_refs = [episode.metadata.episode_ref for pair in pairs for episode in pair.episodes]
            self.assertEqual(len(refs), len(set(refs)))
            self.assertEqual(len(episode_refs), len(set(episode_refs)))

    def test_counterfactual_twins_have_identical_public_inputs_and_opposite_probe_labels(self):
        for pairs in self.partitions.values():
            for pair in pairs:
                self.assertIs(pair.first.public, pair.second.public)
                first_payload = pair.first.to_learner_payload()
                second_payload = pair.second.to_learner_payload()
                self.assertEqual(first_payload, second_payload)
                self.assertEqual(
                    json.dumps(first_payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(second_payload, sort_keys=True, separators=(",", ":")),
                )
                first_regime = pair.first.metadata.hidden_regime
                second_regime = pair.second.metadata.hidden_regime
                self.assertEqual(
                    second_regime,
                    (1 - first_regime[0], 1 - first_regime[1]),
                )
                for index in v15.PROBE_INDICES:
                    labels = {
                        pair.first.supervision.outcome_labels[index],
                        pair.second.supervision.outcome_labels[index],
                    }
                    self.assertEqual(labels, {"SUCCESS", "FAILURE"})

    def test_two_anchors_and_four_fresh_probes_have_counterbalanced_relations(self):
        for pairs in self.partitions.values():
            for pair in pairs:
                for episode in pair.episodes:
                    supervision = episode.supervision
                    self.assertEqual(supervision.acquisition_indices, (0, 1))
                    self.assertEqual(supervision.probe_indices, (2, 3, 4, 5))
                    self.assertEqual(
                        Counter(supervision.relation_classes[:2]),
                        Counter({"alpha": 1, "beta": 1}),
                    )
                    self.assertEqual(
                        Counter(supervision.relation_classes[2:]),
                        Counter({"alpha": 2, "beta": 2}),
                    )
                    public_pairs = {
                        (event.reference_trace_text, event.attempt_trace_text)
                        for event in episode.public.events
                    }
                    self.assertEqual(len(public_pairs), 6)
                    for event, relation in zip(
                        episode.public.events,
                        supervision.relation_classes,
                        strict=True,
                    ):
                        reference = _sentences(event.reference_trace_text)
                        attempt = _sentences(event.attempt_trace_text)
                        self.assertEqual(len(reference), len(attempt))
                        self.assertNotEqual(reference, attempt)
                        if relation == "alpha":
                            self.assertEqual(Counter(reference), Counter(attempt))
                        else:
                            self.assertNotEqual(Counter(reference), Counter(attempt))

    def test_regimes_relations_and_polarities_are_balanced_at_every_position_and_group(self):
        for pairs in self.partitions.values():
            regimes = Counter(
                episode.metadata.hidden_regime
                for pair in pairs
                for episode in pair.episodes
            )
            self.assertEqual(set(regimes), {(0, 0), (0, 1), (1, 0), (1, 1)})
            self.assertEqual(len(set(regimes.values())), 1)

            position_relations = defaultdict(Counter)
            position_outcomes = defaultdict(Counter)
            family_position_outcomes = defaultdict(Counter)
            transition_position_outcomes = defaultdict(Counter)
            for pair in pairs:
                for episode in pair.episodes:
                    for position, (relation, outcome) in enumerate(
                        zip(
                            episode.supervision.relation_classes,
                            episode.supervision.outcome_labels,
                            strict=True,
                        )
                    ):
                        position_relations[position][relation] += 1
                        position_outcomes[position][outcome] += 1
                        family_position_outcomes[
                            (episode.metadata.generator_family, position)
                        ][outcome] += 1
                        transition_position_outcomes[
                            (episode.metadata.transition_group, position)
                        ][outcome] += 1
            for position in range(6):
                self.assertEqual(len(set(position_relations[position].values())), 1)
                self.assertEqual(len(set(position_outcomes[position].values())), 1)
            for counts in family_position_outcomes.values():
                self.assertEqual(counts["SUCCESS"], counts["FAILURE"])
            for counts in transition_position_outcomes.values():
                self.assertEqual(counts["SUCCESS"], counts["FAILURE"])

    def test_deranged_anchor_history_is_fixed_point_free_and_globally_balanced(self):
        for pairs in self.partitions.values():
            position_counts = defaultdict(Counter)
            for pair in pairs:
                for episode in pair.episodes:
                    true = episode.supervision.acquisition_outcomes
                    deranged = pair.deranged_acquisition_outcomes(episode)
                    self.assertTrue(
                        all(
                            left != right
                            for left, right in zip(true, deranged, strict=True)
                        )
                    )
                    for position, outcome in enumerate(deranged):
                        position_counts[position][outcome] += 1
                outsider = self.corpus.train[0].first
                if outsider not in pair.episodes:
                    with self.assertRaises(ValueError):
                        pair.deranged_acquisition_outcomes(outsider)
            for counts in position_counts.values():
                self.assertEqual(counts["SUCCESS"], counts["FAILURE"])

    def test_public_payload_contains_no_outcome_regime_class_or_identity_feature(self):
        forbidden = {
            "outcome",
            "outcome_label",
            "outcome_labels",
            "relation_class",
            "relation_classes",
            "regime",
            "hidden_regime",
            "partition",
            "pair_ref",
            "episode_ref",
            "twin_ref",
            "generator_family",
            "transition_group",
            "structure_signature",
            "answer",
            "target",
            "seed",
            "phase",
        }
        for pairs in self.partitions.values():
            for pair in pairs:
                for episode in pair.episodes:
                    payload = episode.to_learner_payload()
                    self.assertEqual(set(payload), {"events"})
                    self.assertTrue(_recursive_keys(payload).isdisjoint(forbidden))
                    for event in payload["events"]:
                        self.assertEqual(
                            set(event),
                            {"reference_trace_text", "attempt_trace_text", "temporal"},
                        )
                        self.assertEqual(
                            set(event["temporal"]),
                            {"acquired_ordinal", "age", "landmark_relations"},
                        )
                    text = json.dumps(payload, ensure_ascii=False)
                    for hidden in (
                        episode.metadata.pair_ref,
                        episode.metadata.episode_ref,
                        episode.metadata.twin_ref,
                        episode.metadata.generator_family,
                        episode.metadata.transition_group,
                    ):
                        self.assertNotIn(hidden, text)
                    for action_key in v13.v12._ACTION_KEYS:
                        self.assertNotIn(action_key, text)

    def test_train_development_final_surfaces_and_topologies_are_disjoint(self):
        traces = {name: _public_traces(pairs) for name, pairs in self.partitions.items()}
        self.assertTrue(traces["train"].isdisjoint(traces["development"]))
        self.assertTrue(traces["train"].isdisjoint(traces["final"]))
        self.assertTrue(traces["development"].isdisjoint(traces["final"]))
        signatures = {
            name: {pair.first.metadata.structure_signature for pair in pairs}
            for name, pairs in self.partitions.items()
        }
        self.assertTrue(signatures["train"].isdisjoint(signatures["development"]))
        self.assertTrue(signatures["train"].isdisjoint(signatures["final"]))
        self.assertTrue(signatures["development"].isdisjoint(signatures["final"]))

    def test_frozen_schedule_exposes_every_pair_twice_in_96_by_8_batches(self):
        schedule = self.corpus.training_schedule()
        self.assertEqual(len(schedule), 96)
        self.assertTrue(all(len(indices) == 8 for indices in schedule))
        exposures = Counter(index for indices in schedule for index in indices)
        self.assertEqual(exposures, Counter({index: 2 for index in range(384)}))
        for update_index, indices in enumerate(schedule):
            self.assertEqual(
                self.corpus.train_pairs_for_update(update_index),
                tuple(self.corpus.train[index] for index in indices),
            )
        with self.assertRaises(ValueError):
            self.corpus.train_pairs_for_update(-1)
        with self.assertRaises(ValueError):
            self.corpus.train_pairs_for_update(96)

    def test_final_seal_gradient_boundary_and_reproducibility(self):
        self.assertFalse(self.corpus.final_is_open)
        with self.assertRaises(v15.FinalPartitionSealedError):
            _ = self.corpus.final
        with self.assertRaises(v15.FinalPartitionSealedError):
            self.corpus.partition("final")
        self.assertTrue(self.open_corpus.final_is_open)
        self.assertEqual(self.corpus.train, self.open_corpus.train)
        self.assertEqual(self.corpus.development, self.open_corpus.development)
        self.assertTrue(
            all(
                episode.gradient_authorized
                for pair in self.corpus.train
                for episode in pair.episodes
            )
        )
        self.assertTrue(
            all(
                not episode.gradient_authorized
                for pair in self.corpus.development
                for episode in pair.episodes
            )
        )
        self.assertEqual(
            self.corpus,
            v15.build_paired_latent_contingency_credit_v15(),
        )
        self.assertEqual(
            self.open_corpus,
            v15.build_paired_latent_contingency_credit_v15(include_final=True),
        )
        self.assertGreater(self.corpus.learner_payload_size_bytes(), 0)
        with self.assertRaises(TypeError):
            v15.build_paired_latent_contingency_credit_v15(include_final=1)


if __name__ == "__main__":
    unittest.main()
