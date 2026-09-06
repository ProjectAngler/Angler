from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import unittest

from experiments.corpora import structure_keyed_persistent_credit_v14 as v14
from experiments.corpora import structure_protected_oml_natural_trace_v13 as v13


_STEP = re.compile(r"(?:^|(?<=\s))Step [1-9][0-9]*: ")


def _sentences(trace: str) -> tuple[str, ...]:
    matches = tuple(_STEP.finditer(trace))
    if not matches or matches[0].start() != 0:
        raise AssertionError("trace is not strictly Step-framed")
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


class StructureKeyedPersistentCreditV14CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = v14.build_structure_keyed_persistent_credit_v14()
        cls.open_corpus = v14.build_structure_keyed_persistent_credit_v14(
            include_final=True
        )

    def test_exact_mechanism_event_counts_and_chronology(self):
        self.assertEqual(len(self.corpus.train), 384)
        self.assertEqual(len(self.corpus.development), 48)
        self.assertEqual(len(self.open_corpus.final), 48)
        self.assertEqual(len(self.corpus.train_events), 2304)
        self.assertEqual(len(self.corpus.development_events), 288)
        self.assertEqual(
            tuple(row.temporal.acquired_ordinal for row in self.corpus.train_events),
            tuple(range(2304)),
        )
        self.assertEqual(
            tuple(
                row.temporal.acquired_ordinal
                for row in self.corpus.development_events
            ),
            tuple(range(2_000_000, 2_000_288)),
        )
        for mechanisms in (self.corpus.train, self.corpus.development):
            for mechanism in mechanisms:
                self.assertEqual(len(mechanism.public.episodes), 6)
                self.assertEqual(
                    tuple(row.temporal.age for row in mechanism.public.episodes),
                    (5, 4, 3, 2, 1, 0),
                )

    def test_inherits_only_the_public_v13_training_and_development_sequences(self):
        source = v13.build_structure_protected_oml_natural_trace_v13()
        self.assertEqual(self.corpus.train, source.train_inner)
        self.assertEqual(self.corpus.development, source.development)
        self.assertIs(self.corpus.train_mechanisms, self.corpus.train)
        self.assertIs(self.corpus.development_mechanisms, self.corpus.development)
        self.assertTrue(all(row.gradient_authorized for row in self.corpus.train))
        self.assertTrue(
            all(not row.gradient_authorized for row in self.corpus.development)
        )

    def test_true_and_shuffled_controls_are_balanced_and_change_every_association(self):
        for mechanism in (*self.corpus.train, *self.corpus.development):
            true = self.corpus.true_outcomes(mechanism)
            shuffled = self.corpus.shuffled_outcomes(mechanism)
            expected = Counter({"SUCCESS": 3, "FAILURE": 3})
            self.assertEqual(Counter(true), expected)
            self.assertEqual(Counter(shuffled), expected)
            self.assertTrue(
                all(left != right for left, right in zip(true, shuffled, strict=True))
            )
            self.assertEqual(
                shuffled,
                tuple(true[index] for index in v14.SHUFFLE_PERMUTATION),
            )

    def test_replication_panel_has_six_rows_per_exact_cross_cell(self):
        panel = self.corpus.structural_replication
        self.assertEqual(len(panel), 240)
        cross = Counter(
            (row.metadata.renderer_pair, row.metadata.target_position)
            for row in panel
        )
        self.assertEqual(len(cross), 8 * 5)
        self.assertEqual(set(cross.values()), {6})
        renderer = Counter(row.metadata.renderer_pair for row in panel)
        positions = Counter(row.metadata.target_position for row in panel)
        self.assertEqual(set(renderer.values()), {30})
        self.assertEqual(set(positions.values()), {48})

        families = Counter(row.metadata.generator_family for row in panel)
        transitions = Counter(row.metadata.transition_group for row in panel)
        self.assertEqual(len(families), 12)
        self.assertEqual(set(families.values()), {20})
        self.assertEqual(
            transitions,
            Counter(
                {
                    "insertion": 60,
                    "removal": 60,
                    "reorder": 60,
                    "replacement": 60,
                }
            ),
        )
        self.assertEqual(
            len({row.metadata.row_ref for row in panel}),
            len(panel),
        )

    def test_replication_explicitly_covers_each_corruption_and_procedure_relation(self):
        corruption_position_counts = defaultdict(Counter)
        for row in self.corpus.structural_replication:
            public = row.public
            supervision = row.supervision
            target = supervision.target_candidate_index
            self.assertEqual(target, row.metadata.target_position)
            self.assertEqual(
                len(set(public.candidate_attempt_trace_texts)), 5
            )
            reference = _sentences(public.reference_trace_text)
            paraphrase = _sentences(public.candidate_attempt_trace_texts[target])
            self.assertEqual(len(reference), len(paraphrase))
            self.assertTrue(
                all(left != right for left, right in zip(reference, paraphrase))
            )

            for corruption in ("reorder", "replacement", "omission", "insertion"):
                candidate_index = supervision.candidate_index_for(corruption)
                corruption_position_counts[corruption][candidate_index] += 1
                candidate = _sentences(
                    public.candidate_attempt_trace_texts[candidate_index]
                )
                if corruption == "reorder":
                    self.assertEqual(Counter(reference), Counter(candidate))
                    self.assertNotEqual(reference, candidate)
                elif corruption == "replacement":
                    self.assertEqual(len(reference), len(candidate))
                    self.assertNotEqual(reference, candidate)
                elif corruption == "omission":
                    self.assertEqual(len(candidate), len(reference) - 1)
                else:
                    self.assertEqual(len(candidate), len(reference) + 1)

        # Corruption serialization is rotated; no corruption acts as a fixed
        # answer-position cue in the fresh panel.
        for counts in corruption_position_counts.values():
            self.assertEqual(set(counts), set(range(5)))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)

    def test_replication_payload_excludes_answers_renderer_and_generator_metadata(self):
        forbidden = {
            "target_candidate_index",
            "candidate_corruption_types",
            "corruption_candidate_indices",
            "row_ref",
            "generator_family",
            "transition_group",
            "renderer_pair",
            "target_position",
            "challenge_structure_signature",
            "corruption",
            "relation",
            "seed",
        }
        for row in self.corpus.structural_replication:
            payload = row.to_encoder_payload()
            self.assertEqual(
                set(payload),
                {"reference_trace_text", "candidate_attempt_trace_texts"},
            )
            self.assertTrue(_recursive_keys(payload).isdisjoint(forbidden))
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(row.metadata.row_ref, serialized)
            self.assertNotIn(row.metadata.generator_family, serialized)
            self.assertNotIn(row.metadata.transition_group, serialized)
            for action_key in v13.v12._ACTION_KEYS:
                self.assertNotIn(action_key, serialized)

    def test_replication_surfaces_are_fresh_and_unique(self):
        source_traces = {
            trace
            for mechanism in self.corpus.development
            for episode in mechanism.public.episodes
            for trace in (
                episode.reference_trace_text,
                episode.attempt_trace_text,
            )
        }
        replication_references = {
            row.public.reference_trace_text
            for row in self.corpus.structural_replication
        }
        replication_candidates = {
            candidate
            for row in self.corpus.structural_replication
            for candidate in row.public.candidate_attempt_trace_texts
        }
        self.assertEqual(len(replication_references), 240)
        self.assertEqual(len(replication_candidates), 1200)
        self.assertTrue(replication_references.isdisjoint(source_traces))
        self.assertTrue(replication_candidates.isdisjoint(source_traces))

    def test_identities_partitions_seal_and_reproducibility(self):
        partitions = (
            self.corpus.train,
            self.corpus.development,
            self.open_corpus.final,
        )
        refs = [
            mechanism.metadata.mechanism_ref
            for mechanisms in partitions
            for mechanism in mechanisms
        ]
        event_refs = [
            (mechanism.metadata.mechanism_ref, event_index)
            for mechanisms in partitions
            for mechanism in mechanisms
            for event_index in range(6)
        ]
        self.assertEqual(len(refs), len(set(refs)))
        self.assertEqual(len(event_refs), len(set(event_refs)))
        self.assertFalse(self.corpus.final_is_open)
        with self.assertRaises(v13.FinalPartitionSealedError):
            _ = self.corpus.final
        with self.assertRaises(v13.FinalPartitionSealedError):
            self.corpus.partition("final")
        self.assertTrue(self.open_corpus.final_is_open)
        self.assertEqual(self.corpus.train, self.open_corpus.train)
        self.assertEqual(self.corpus.development, self.open_corpus.development)
        self.assertEqual(
            self.corpus.structural_replication,
            self.open_corpus.structural_replication,
        )
        self.assertEqual(
            self.corpus,
            v14.build_structure_keyed_persistent_credit_v14(),
        )
        self.assertEqual(
            self.open_corpus,
            v14.build_structure_keyed_persistent_credit_v14(
                include_final=True
            ),
        )
        with self.assertRaises(TypeError):
            v14.build_structure_keyed_persistent_credit_v14(include_final=1)


if __name__ == "__main__":
    unittest.main()
