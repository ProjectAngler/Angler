"""Random-experience invariants for causal operator induction."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from experiments.runners.causal_operator_experience import (  # noqa: E402
    build_causal_operator_experience,
)
from angler.procedures.alignment import find_structural_isomorphisms  # noqa: E402
from angler.procedures.operators import Constant, TypedVariable  # noqa: E402
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402


_EXECUTORS = {
    tokens.NAMESPACE: tokens.execute_token_action,
    files.NAMESPACE: files.execute_file_action,
    boxes.NAMESPACE: boxes.execute_box_action,
}


def _all_observed_arguments(experience):
    for corpus in experience.corpora:
        for trace in corpus.traces:
            for record in trace.initial.records:
                yield from record.arguments
            for transition in trace.transitions:
                yield from transition.action.arguments
                for record in transition.after.records:
                    yield from record.arguments
        for candidate in corpus.candidates:
            for exemplar in candidate.operator.exemplars:
                reconstruction = exemplar.reconstruction
                for _, value in reconstruction.variable_bindings:
                    yield value
                yield from reconstruction.constant_values
                for record in reconstruction.start_records + reconstruction.end_records:
                    yield from record.arguments
                for action in reconstruction.actions:
                    yield from action.arguments


class CausalOperatorExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # This is the advertised default call and therefore a regression guard
        # for support sparsity or structural-selection drift.
        cls.experience = build_causal_operator_experience(seed=42_017)

    def test_default_generation_is_deterministic_and_selects_two_step_trio(self) -> None:
        replay = build_causal_operator_experience(seed=42_017)

        self.assertEqual(self.experience, replay)
        self.assertEqual(self.experience.digest, replay.digest)
        self.assertEqual(len(self.experience.corpora), 3)
        self.assertEqual(
            tuple(item.body_steps for item in self.experience.support),
            (2, 2, 2),
        )
        self.assertTrue(
            all(len(item.operator.body) == 2 for item in self.experience.selected_candidates)
        )

    def test_every_retained_transition_replays_exactly_through_its_executor(self) -> None:
        for corpus in self.experience.corpora:
            executor = _EXECUTORS[corpus.namespace]
            self.assertGreater(corpus.blocked_attempts, 0)
            for trace in corpus.traces:
                self.assertIsNone(trace.goal)
                state = trace.initial
                for declared in trace.transitions:
                    observed = executor(state, declared.action)
                    self.assertEqual(observed, declared)
                    self.assertTrue(observed.applied)
                    state = observed.after
                self.assertEqual(state, trace.final_state)

    def test_training_corpus_excludes_heldout_entity_prefix(self) -> None:
        prefix = self.experience.heldout_entity_prefix
        arguments = tuple(_all_observed_arguments(self.experience))

        self.assertTrue(arguments)
        self.assertFalse(any(value.startswith(prefix) for value in arguments))
        self.assertTrue(any(value.startswith("train_") for value in arguments))

    def test_retains_reconstructable_multi_step_operator_for_unseen_bindings(self) -> None:
        for candidate in self.experience.selected_candidates:
            operator = candidate.operator
            self.assertEqual(len(operator.body), 2)
            self.assertGreaterEqual(len(operator.exemplars), 2)
            self.assertEqual(
                len(operator.exemplars),
                len({item.reconstruction.digest for item in operator.exemplars}),
            )
            self.assertGreaterEqual(
                len(
                    {
                        item.reconstruction.variable_bindings
                        for item in operator.exemplars
                    }
                ),
                2,
            )
            for exemplar in operator.exemplars:
                reconstruction = exemplar.reconstruction
                self.assertEqual(len(reconstruction.actions), 2)
                self.assertEqual(
                    tuple(item.digest for item in reconstruction.actions),
                    exemplar.action_digests,
                )

            # The retained body is a typed template, so a novel entity binding
            # can be reconstructed without copying any exemplar's concrete ID.
            unseen = {
                variable.name: f"heldout_probe_{index}"
                for index, variable in enumerate(operator.variables)
            }
            grounded = tuple(
                action.schema.ground(
                    *(
                        unseen[term.name]
                        if isinstance(term, TypedVariable)
                        else term.value
                        for term in action.arguments
                    )
                )
                for action in operator.body
            )
            self.assertEqual(len(grounded), 2)
            self.assertTrue(
                any(
                    value.startswith("heldout_probe_")
                    for action in grounded
                    for value in action.arguments
                )
            )
            self.assertFalse(
                any(
                    value.startswith("train_")
                    for action in grounded
                    for value in action.arguments
                )
            )
            self.assertTrue(
                all(
                    isinstance(term, (TypedVariable, Constant))
                    for action in operator.body
                    for term in action.arguments
                )
            )

    def test_selected_trio_has_pairwise_name_independent_structural_candidates(self) -> None:
        selected = self.experience.selected_candidates
        pairs = tuple(itertools.combinations(selected, 2))

        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(self.experience.pairwise_alignments), 3)
        for source, target in pairs:
            self.assertNotEqual(source.operator.name, target.operator.name)
            matches = find_structural_isomorphisms(
                source.operator,
                target.operator,
            )
            self.assertTrue(matches)
            self.assertTrue(
                any(item.coverage.matched_effects >= 2 for item in matches)
            )


if __name__ == "__main__":
    unittest.main()
