"""Tests for structural operator alignment and external evidence gating."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.alignment import (  # noqa: E402
    AliasTable,
    AlignmentError,
    CounterfactualExecutionCertificate,
    MergeProposal,
    MergeResult,
    VerifiedAliasEntry,
    authorize_merge,
    find_structural_isomorphisms,
)
from angler.procedures.operators import (  # noqa: E402
    ActionPattern,
    Effect,
    LearnedOperator,
    OperatorExemplar,
    ReconstructionExemplar,
    RecordPattern,
    TypedVariable,
)
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    Parameter,
    Record,
    State,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def learned_move(
    namespace: str,
    *,
    operator_local: str,
    action_local: str,
    at_local: str,
    available_local: str,
    variable_names: tuple[str, str, str],
    description: str,
    repeated_destination: bool = False,
) -> LearnedOperator:
    entity_type = namespace + ".object"
    location_type = namespace + ".place"
    entity = TypedVariable(variable_names[0], entity_type)
    source = TypedVariable(variable_names[1], location_type)
    destination = TypedVariable(variable_names[2], location_type)
    schema = ActionSchema(
        namespace + "." + action_local,
        (
            Parameter("object", entity_type),
            Parameter("origin", location_type),
            Parameter("destination", location_type),
        ),
        description,
    )
    body_destination = source if repeated_destination else destination
    ground_action = schema.ground("object_1", "site_a", "site_b")
    reconstruction = ReconstructionExemplar(
        namespace=namespace,
        start_records=tuple(
            sorted(
                (
                    Record(namespace + "." + at_local, ("object_1", "site_a")),
                    Record(namespace + "." + available_local, ("site_b",)),
                )
            )
        ),
        variable_bindings=tuple(
            sorted(
                (
                    (entity.name, "object_1"),
                    (source.name, "site_a"),
                    (destination.name, "site_b"),
                )
            )
        ),
        constant_values=(),
        actions=(ground_action,),
        end_records=(
            Record(namespace + "." + at_local, ("object_1", "site_b")),
        ),
    )
    return LearnedOperator(
        name=namespace + "." + operator_local,
        namespace=namespace,
        variables=(entity, source, destination),
        preconditions=(
            RecordPattern(namespace + "." + at_local, (entity, source)),
            RecordPattern(namespace + "." + available_local, (destination,)),
        ),
        effects=(
            Effect(
                "delete",
                RecordPattern(namespace + "." + at_local, (entity, source)),
            ),
            Effect(
                "add",
                RecordPattern(namespace + "." + at_local, (entity, destination)),
            ),
        ),
        body=(ActionPattern(schema, (entity, source, body_destination)),),
        exemplars=(
            OperatorExemplar(
                trace_digest=digest("a" if namespace.startswith("alpha") else "b"),
                start_index=0,
                stop_index=1,
                before_state_digest=digest("c"),
                after_state_digest=digest("d"),
                action_digests=(ground_action.digest,),
                reconstruction=reconstruction,
            ),
        ),
    )


class StructuralAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = learned_move(
            "alpha.world",
            operator_local="transport",
            action_local="carry",
            at_local="occupies",
            available_local="vacant",
            variable_names=("actor", "from_place", "to_place"),
            description="Carry an object into a vacant site.",
        )
        self.target = learned_move(
            "omega.realm",
            operator_local="relocate",
            action_local="shift",
            at_local="located",
            available_local="open",
            variable_names=("item", "old_site", "new_site"),
            description="Unrelated wording intentionally ignored by alignment.",
        )

    def test_isomorphism_uses_structure_not_names_or_descriptions(self) -> None:
        candidates = find_structural_isomorphisms(self.source, self.target)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        aliases = {(item.kind, item.source, item.target) for item in candidate.aliases}
        self.assertIn(
            ("operator", "alpha.world.transport", "omega.realm.relocate"),
            aliases,
        )
        self.assertIn(("action", "alpha.world.carry", "omega.realm.shift"), aliases)
        self.assertIn(
            ("predicate", "alpha.world.occupies", "omega.realm.located"),
            aliases,
        )
        self.assertIn(
            ("predicate", "alpha.world.vacant", "omega.realm.open"),
            aliases,
        )
        self.assertRegex(candidate.digest, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(FrozenInstanceError):
            candidate.aliases = ()  # type: ignore[misc]

    def test_different_coreference_topology_is_not_isomorphic(self) -> None:
        nonisomorphic = learned_move(
            "omega.realm",
            operator_local="relocate",
            action_local="shift",
            at_local="located",
            available_local="open",
            variable_names=("item", "old_site", "new_site"),
            description="Same arities, different role reuse.",
            repeated_destination=True,
        )
        self.assertEqual(
            find_structural_isomorphisms(self.source, nonisomorphic),
            (),
        )

    def test_maximal_common_core_retains_domain_specific_residuals(self) -> None:
        source = replace(
            self.source,
            preconditions=self.source.preconditions
            + (
                RecordPattern(
                    "alpha.world.adjacent",
                    (self.source.variables[1], self.source.variables[2]),
                ),
            ),
            effects=self.source.effects
            + (
                Effect(
                    "delete",
                    RecordPattern(
                        "alpha.world.free_slot",
                        (self.source.variables[1], self.source.variables[2]),
                    ),
                ),
                Effect(
                    "add",
                    RecordPattern(
                        "alpha.world.free_slot",
                        (self.source.variables[2], self.source.variables[1]),
                    ),
                ),
            ),
        )
        target = replace(
            self.target,
            preconditions=self.target.preconditions
            + (
                RecordPattern(
                    "omega.realm.capacity_ok",
                    (self.target.variables[2],),
                ),
            ),
        )
        candidates = find_structural_isomorphisms(source, target)
        self.assertTrue(candidates)
        candidate = max(candidates, key=lambda item: item.coverage.matched_preconditions)
        self.assertEqual(candidate.coverage.matched_effects, 2)
        self.assertEqual(candidate.coverage.source_effects, 4)
        self.assertEqual(candidate.coverage.target_effects, 2)
        self.assertEqual(len(candidate.residuals.source_effects), 2)
        self.assertEqual(len(candidate.residuals.target_effects), 0)
        self.assertEqual(len(candidate.residuals.source_preconditions), 1)
        self.assertEqual(len(candidate.residuals.target_preconditions), 1)

        one_effect_source = replace(source, effects=(source.effects[0],))
        one_effect_target = replace(target, effects=(target.effects[0],))
        self.assertEqual(
            find_structural_isomorphisms(one_effect_source, one_effect_target),
            (),
        )


class CertificateAndMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        source = learned_move(
            "alpha.world",
            operator_local="transport",
            action_local="carry",
            at_local="occupies",
            available_local="vacant",
            variable_names=("actor", "from_place", "to_place"),
            description="source",
        )
        target = learned_move(
            "omega.realm",
            operator_local="relocate",
            action_local="shift",
            at_local="located",
            available_local="open",
            variable_names=("item", "old_site", "new_site"),
            description="target",
        )
        self.source = source
        self.target = target
        self.candidate = find_structural_isomorphisms(source, target)[0]
        self.passing = CounterfactualExecutionCertificate(
            candidate_digest=self.candidate.digest,
            execution_digest=digest("1"),
            result_digest=digest("2"),
            result="pass",
            issued_by="external.counterfactual-evaluator",
        )
        self.failing = CounterfactualExecutionCertificate(
            candidate_digest=self.candidate.digest,
            execution_digest=digest("3"),
            result_digest=digest("4"),
            result="fail",
            issued_by="external.counterfactual-evaluator",
        )

    def test_alias_table_requires_external_passing_evidence(self) -> None:
        empty = AliasTable()
        with self.assertRaises(AlignmentError):
            empty.with_certificate(self.candidate, self.failing)
        with self.assertRaises(AlignmentError):
            VerifiedAliasEntry(
                self.candidate.digest,
                self.candidate.source_operator_digest,
                self.candidate.target_operator_digest,
                self.candidate.aliases,
                self.candidate.variable_map,
                self.failing,
            )

        table = empty.with_certificate(self.candidate, self.passing)
        self.assertEqual(empty.entries, ())
        self.assertEqual(len(table.entries), 1)
        self.assertEqual(
            table.entry_for(self.candidate.digest).certificate,
            self.passing,
        )

        wrong_binding = CounterfactualExecutionCertificate(
            candidate_digest=digest("9"),
            execution_digest=digest("1"),
            result_digest=digest("2"),
            result="pass",
            issued_by="external.counterfactual-evaluator",
        )
        with self.assertRaises(AlignmentError):
            empty.with_certificate(self.candidate, wrong_binding)

        source_action = next(
            item for item in self.candidate.aliases if item.kind == "action"
        )
        self.assertEqual(
            table.canonical_symbol("action", source_action.target),
            source_action.source,
        )
        self.assertEqual(
            table.canonicalize_operator(self.source),
            table.canonicalize_operator(self.target),
        )

    def test_certified_aliases_normalize_concrete_state_and_goal_records(self) -> None:
        table = AliasTable().with_certificate(self.candidate, self.passing)
        source_state = State.from_records(
            "alpha.world",
            (
                Record("alpha.world.occupies", ("object_1", "site_a")),
                Record("alpha.world.vacant", ("site_b",)),
            ),
        )
        target_state = State.from_records(
            "omega.realm",
            (
                Record("omega.realm.located", ("object_1", "site_a")),
                Record("omega.realm.open", ("site_b",)),
            ),
        )
        self.assertEqual(
            table.canonicalize_state(source_state).records,
            table.canonicalize_state(target_state).records,
        )
        self.assertNotEqual(
            AliasTable().canonicalize_state(source_state).records,
            AliasTable().canonicalize_state(target_state).records,
        )

        source_goal = Goal.from_records(
            "alpha.world",
            (Record("alpha.world.occupies", ("object_1", "site_b")),),
        )
        target_goal = Goal.from_records(
            "omega.realm",
            (Record("omega.realm.located", ("object_1", "site_b")),),
        )
        self.assertEqual(
            table.canonicalize_goal(source_goal).required,
            table.canonicalize_goal(target_goal).required,
        )

    def test_merge_cannot_authorize_without_admitted_matching_certificate(self) -> None:
        proposal = MergeProposal(
            self.candidate,
            self.candidate.source_operator_digest,
        )
        rejected = authorize_merge(proposal, AliasTable(), self.passing)
        self.assertEqual(rejected.status, "rejected")

        table = AliasTable().with_certificate(self.candidate, self.passing)
        authorized = authorize_merge(proposal, table, self.passing)
        self.assertEqual(authorized.status, "authorized")
        self.assertEqual(authorized.certificate, self.passing)
        self.assertEqual(proposal.retired_operator_digest, self.candidate.target_operator_digest)

        with self.assertRaises(AlignmentError):
            MergeResult(
                proposal_digest=proposal.digest,
                candidate_digest=self.candidate.digest,
                status="authorized",
                alias_table_digest=table.digest,
                certificate=None,
                reason="would_self_certify",
            )


if __name__ == "__main__":
    unittest.main()
