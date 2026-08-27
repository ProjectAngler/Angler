"""Tests for canonical symbolic operator mirrors and trace induction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.induction import (  # noqa: E402
    InductionError,
    MDLOperatorInducer,
    TransitionDelta,
    anti_unify_entities,
    cluster_subsegments,
    extract_subsegment_delta,
    extract_trace_subsegments,
)
from angler.procedures.operators import (  # noqa: E402
    ActionPattern,
    Constant,
    TypedVariable,
)
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Parameter,
    Record,
    State,
    Trace,
    Transition,
)


NAMESPACE = "demo.world"
ENTITY = "demo.world.entity"
LOCATION = "demo.world.location"
MOVE = ActionSchema(
    "demo.world.move",
    (
        Parameter("entity", ENTITY),
        Parameter("source", LOCATION),
        Parameter("destination", LOCATION),
    ),
    description="Move one entity between observed locations.",
)


def state(entity: str, occupied: str, clear: str) -> State:
    return State.from_records(
        NAMESPACE,
        (
            Record("demo.world.at", (entity, occupied)),
            Record("demo.world.clear", (clear,)),
        ),
    )


def move_trace(entity: str, source: str, destination: str) -> Trace:
    before = state(entity, source, destination)
    after = state(entity, destination, source)
    return Trace(
        initial=before,
        transitions=(
            Transition(
                before=before,
                action=MOVE.ground(entity, source, destination),
                after=after,
                applied=True,
                outcome="demo.world.applied",
            ),
        ),
    )


class DeltaExtractionTests(unittest.TestCase):
    def test_delta_is_exact_and_does_not_depend_on_a_goal(self) -> None:
        trace = move_trace("robot_1", "alpha", "beta")
        segment = extract_subsegment_delta(trace, 0, 1)

        self.assertEqual(segment.before, trace.initial)
        self.assertEqual(segment.after, trace.final_state)
        self.assertEqual(
            segment.delta,
            TransitionDelta.between(trace.initial, trace.final_state),
        )
        self.assertEqual(
            set(segment.delta.deleted),
            {
                Record("demo.world.at", ("robot_1", "alpha")),
                Record("demo.world.clear", ("beta",)),
            },
        )
        self.assertEqual(
            set(segment.delta.added),
            {
                Record("demo.world.at", ("robot_1", "beta")),
                Record("demo.world.clear", ("alpha",)),
            },
        )
        self.assertEqual(segment.to_canonical()["trace_digest"], trace.digest)

    def test_every_bounded_contiguous_subsegment_is_extracted(self) -> None:
        first = move_trace("robot_1", "alpha", "beta").transitions[0]
        middle = first.after
        final = state("robot_1", "gamma", "beta")
        second = Transition(
            before=middle,
            action=MOVE.ground("robot_1", "beta", "gamma"),
            after=final,
            applied=True,
            outcome="demo.world.applied",
        )
        trace = Trace(first.before, (first, second))

        segments = extract_trace_subsegments(trace, maximum_length=2)
        self.assertEqual(
            {(item.start_index, item.stop_index) for item in segments},
            {(0, 1), (0, 2), (1, 2)},
        )
        with self.assertRaises(InductionError):
            extract_subsegment_delta(trace, 1, 1)

    def test_minimal_precondition_basis_excludes_global_invariants(self) -> None:
        before = State.from_records(
            NAMESPACE,
            (
                Record("demo.world.at", ("robot", "alpha")),
                Record("demo.world.clear", ("beta",)),
                Record("demo.world.link", ("alpha", "beta")),
                Record("demo.world.link", ("beta", "gamma")),
                Record("demo.world.region", ("alpha", "global_region")),
            ),
        )
        after = State.from_records(
            NAMESPACE,
            (
                Record("demo.world.at", ("robot", "beta")),
                Record("demo.world.clear", ("alpha",)),
                Record("demo.world.link", ("alpha", "beta")),
                Record("demo.world.link", ("beta", "gamma")),
                Record("demo.world.region", ("alpha", "global_region")),
            ),
        )
        trace = Trace(
            before,
            (
                Transition(
                    before,
                    MOVE.ground("robot", "alpha", "beta"),
                    after,
                    True,
                    "demo.world.applied",
                ),
            ),
        )

        segment = extract_subsegment_delta(trace, 0, 1)

        self.assertEqual(
            set(segment.relevant_preconditions),
            {
                Record("demo.world.at", ("robot", "alpha")),
                Record("demo.world.clear", ("beta",)),
            },
        )


class AntiUnificationTests(unittest.TestCase):
    def test_constants_and_cross_position_coreference_are_preserved(self) -> None:
        result = anti_unify_entities(
            (
                ("alice", "shared_box", "alice"),
                ("bob", "shared_box", "bob"),
            ),
            type_rows=(
                (ENTITY, "demo.world.container", ENTITY),
                (ENTITY, "demo.world.container", ENTITY),
            ),
            fallback_type="demo.world.untyped",
        )

        self.assertIsInstance(result.terms[0], TypedVariable)
        self.assertEqual(result.terms[0], result.terms[2])
        self.assertEqual(
            result.terms[1],
            Constant("shared_box", "demo.world.container"),
        )
        self.assertEqual(
            dict(result.substitutions[0])[result.terms[0].name],
            "alice",
        )

    def test_incompatible_type_evidence_is_rejected(self) -> None:
        with self.assertRaises(InductionError):
            anti_unify_entities(
                (("a",), ("b",)),
                type_rows=((ENTITY,), (LOCATION,)),
                fallback_type=ENTITY,
            )


class OperatorInductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = tuple(
            extract_subsegment_delta(trace, 0, 1)
            for trace in (
                move_trace("robot_1", "alpha", "beta"),
                move_trace("robot_2", "gamma", "delta"),
                move_trace("robot_3", "epsilon", "zeta"),
            )
        )
        self.inducer = MDLOperatorInducer(minimum_support=2, minimum_savings=1)

    def test_within_domain_cluster_allocates_canonical_typed_mirror(self) -> None:
        cluster = cluster_subsegments(reversed(self.segments[:2]))[0]
        candidate = self.inducer.allocate(cluster)
        self.assertIsNotNone(candidate)
        assert candidate is not None

        operator = candidate.operator
        self.assertEqual(operator.namespace, NAMESPACE)
        self.assertTrue(operator.name.startswith(NAMESPACE + ".learned_"))
        self.assertEqual(len(operator.exemplars), 2)
        reconstruction = operator.exemplars[0].reconstruction
        self.assertEqual(reconstruction.actions[0].schema, MOVE)
        self.assertEqual(
            {name for name, _ in reconstruction.variable_bindings},
            {item.name for item in operator.variables},
        )
        self.assertLessEqual(len(reconstruction.start_records), 32)
        self.assertLessEqual(len(reconstruction.end_records), 32)
        self.assertEqual(operator.body[0].schema, MOVE)
        self.assertEqual(
            tuple(term.type_name for term in operator.body[0].arguments),
            (ENTITY, LOCATION, LOCATION),
        )
        self.assertEqual({effect.kind for effect in operator.effects}, {"add", "delete"})
        self.assertGreater(candidate.score.savings, 0)
        self.assertRegex(operator.digest, r"^sha256:[0-9a-f]{64}$")

        reordered = cluster_subsegments(self.segments[:2])[0]
        repeated = self.inducer.allocate(reordered)
        self.assertIsNotNone(repeated)
        assert repeated is not None
        self.assertEqual(repeated.operator.digest, operator.digest)
        with self.assertRaises(FrozenInstanceError):
            operator.revision = 9  # type: ignore[misc]

    def test_schema_description_is_bound_into_action_pattern_identity(self) -> None:
        terms = (
            TypedVariable("v0", ENTITY),
            TypedVariable("v1", LOCATION),
            TypedVariable("v2", LOCATION),
        )
        first = ActionPattern(MOVE, terms)
        changed_schema = ActionSchema(MOVE.name, MOVE.parameters, "Different semantics.")
        second = ActionPattern(changed_schema, terms)
        self.assertNotEqual(
            first.to_canonical()["schema"]["digest"],
            second.to_canonical()["schema"]["digest"],
        )

    def test_refinement_is_append_only_and_parent_bound(self) -> None:
        initial = self.inducer.allocate(cluster_subsegments(self.segments[:2])[0])
        self.assertIsNotNone(initial)
        assert initial is not None
        refined = self.inducer.refine(initial, (self.segments[2],))
        self.assertIsNotNone(refined)
        assert refined is not None

        self.assertEqual(initial.operator.revision, 1)
        self.assertEqual(len(initial.operator.exemplars), 2)
        self.assertEqual(refined.operator.revision, 2)
        self.assertEqual(refined.operator.parent_digest, initial.operator.digest)
        self.assertEqual(len(refined.operator.exemplars), 3)
        self.assertNotEqual(refined.operator.digest, initial.operator.digest)

    def test_support_threshold_and_cross_domain_partitioning(self) -> None:
        only = cluster_subsegments((self.segments[0],))[0]
        self.assertIsNone(self.inducer.allocate(only))

        other_namespace = "other.world"
        other_schema = ActionSchema(
            "other.world.move",
            (
                Parameter("entity", "other.world.entity"),
                Parameter("source", "other.world.location"),
                Parameter("destination", "other.world.location"),
            ),
        )
        before = State.from_records(
            other_namespace,
            (Record("other.world.at", ("unit", "left")),),
        )
        after = State.from_records(
            other_namespace,
            (Record("other.world.at", ("unit", "right")),),
        )
        other = Trace(
            before,
            (
                Transition(
                    before,
                    other_schema.ground("unit", "left", "right"),
                    after,
                    True,
                    "other.world.applied",
                ),
            ),
        )
        clusters = cluster_subsegments(
            (self.segments[0], extract_subsegment_delta(other, 0, 1))
        )
        self.assertEqual({item.namespace for item in clusters}, {NAMESPACE, other_namespace})

    def test_cluster_signature_separates_different_entity_coreference(self) -> None:
        def history_trace(*, reuse_entity: bool) -> Trace:
            initial = State.from_records(NAMESPACE, ())
            first_entity = "robot_1"
            second_entity = first_entity if reuse_entity else "robot_2"
            first_after = State.from_records(
                NAMESPACE,
                (Record("demo.world.visited", (first_entity, "beta")),),
            )
            final = State.from_records(
                NAMESPACE,
                (
                    Record("demo.world.visited", (first_entity, "beta")),
                    Record("demo.world.visited", (second_entity, "delta")),
                ),
            )
            first = Transition(
                initial,
                MOVE.ground(first_entity, "alpha", "beta"),
                first_after,
                True,
                "demo.world.applied",
            )
            second = Transition(
                first_after,
                MOVE.ground(second_entity, "gamma", "delta"),
                final,
                True,
                "demo.world.applied",
            )
            return Trace(initial, (first, second))

        same_entity = extract_subsegment_delta(history_trace(reuse_entity=True), 0, 2)
        different_entities = extract_subsegment_delta(
            history_trace(reuse_entity=False),
            0,
            2,
        )
        clusters = cluster_subsegments((same_entity, different_entities))
        self.assertEqual(len(clusters), 2)


if __name__ == "__main__":
    unittest.main()
