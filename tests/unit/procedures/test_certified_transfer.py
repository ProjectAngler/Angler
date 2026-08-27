"""Focused tests for certificate-gated transfer adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.alignment import (  # noqa: E402
    AliasTable,
    CounterfactualExecutionCertificate,
    VerifiedAliasEntry,
    find_structural_isomorphisms,
)
from angler.procedures.grounding import (  # noqa: E402
    StateBindingAssignment,
    StateOperatorBinding,
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
from angler.procedures.transfer import (  # noqa: E402
    CertifiedActionAdapter,
    CertifiedPredicateProjector,
    CertifiedTransferError,
    certified_transfer_binding,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _make_operator(
    namespace: str,
    names: tuple[str, str, str],
    *,
    residual_name: str | None = None,
) -> LearnedOperator:
    entity_name, source_name, destination_name = names
    entity = TypedVariable(entity_name, f"{namespace}.entity")
    source = TypedVariable(source_name, f"{namespace}.location")
    destination = TypedVariable(destination_name, f"{namespace}.location")
    variables = [entity, source, destination]
    schema = ActionSchema(
        f"{namespace}.move",
        (
            Parameter("entity", entity.type_name),
            Parameter("source", source.type_name),
            Parameter("destination", destination.type_name),
        ),
        description=f"Observed relocation in {namespace}.",
    )
    preconditions = [RecordPattern(f"{namespace}.at", (entity, source))]
    concrete = {
        entity.name: f"{namespace}_object",
        source.name: f"{namespace}_a",
        destination.name: f"{namespace}_b",
    }
    start_records = [
        Record(f"{namespace}.at", (concrete[entity.name], concrete[source.name]))
    ]
    end_records = [
        Record(
            f"{namespace}.at",
            (concrete[entity.name], concrete[destination.name]),
        )
    ]
    if residual_name is not None:
        residual = TypedVariable(residual_name, f"{namespace}.context")
        variables.append(residual)
        concrete[residual.name] = f"{namespace}_context"
        preconditions.append(
            RecordPattern(f"{namespace}.guard", (residual, source))
        )
        guard = Record(
            f"{namespace}.guard",
            (concrete[residual.name], concrete[source.name]),
        )
        start_records.append(guard)
        end_records.append(guard)
    action = schema.ground(
        concrete[entity.name],
        concrete[source.name],
        concrete[destination.name],
    )
    reconstruction = ReconstructionExemplar(
        namespace=namespace,
        start_records=tuple(sorted(start_records)),
        variable_bindings=tuple(sorted(concrete.items())),
        constant_values=(),
        actions=(action,),
        end_records=tuple(sorted(end_records)),
    )
    exemplar = OperatorExemplar(
        trace_digest=_sha("1"),
        start_index=0,
        stop_index=1,
        before_state_digest=_sha("2"),
        after_state_digest=_sha("3"),
        action_digests=(action.digest,),
        reconstruction=reconstruction,
    )
    return LearnedOperator(
        name=f"{namespace}.learned_move",
        namespace=namespace,
        variables=tuple(variables),
        preconditions=tuple(preconditions),
        effects=(
            Effect(
                "delete",
                RecordPattern(f"{namespace}.at", (entity, source)),
            ),
            Effect(
                "add",
                RecordPattern(f"{namespace}.at", (entity, destination)),
            ),
        ),
        body=(ActionPattern(schema, (entity, source, destination)),),
        exemplars=(exemplar,),
    )


def _admit(alias_table: AliasTable, source: LearnedOperator, target: LearnedOperator, index: int):
    candidate = find_structural_isomorphisms(source, target)[0]
    certificate = CounterfactualExecutionCertificate(
        candidate_digest=candidate.digest,
        execution_digest=_sha(str(4 + index)),
        result_digest=_sha(str(6 + index)),
        result="pass",
        issued_by="external.test.evaluator",
    )
    return alias_table.with_certificate(candidate, certificate), candidate


@dataclass(frozen=True)
class _EntityCandidate:
    value: str
    type_name: str


@dataclass(frozen=True)
class _Assignment:
    variable: TypedVariable
    entity: _EntityCandidate

    def __post_init__(self) -> None:
        if self.variable.type_name != self.entity.type_name:
            raise ValueError("type mismatch")


@dataclass(frozen=True)
class _ExecutionBinding:
    operator: LearnedOperator
    assignments: tuple[_Assignment, ...]

    def __post_init__(self) -> None:
        if tuple(item.variable for item in self.assignments) != self.operator.variables:
            raise ValueError("execution binding must cover the operator")


def _execution_stub() -> ModuleType:
    module = ModuleType("angler.procedures.execution")
    module.BindingAssignment = _Assignment
    module.OperatorBinding = _ExecutionBinding
    module.TypedEntityCandidate = _EntityCandidate
    return module


class CertifiedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _make_operator(
            "transfer.source",
            ("entity", "source", "destination"),
        )
        cls.bridge = _make_operator(
            "transfer.bridge",
            ("object", "origin", "endpoint"),
        )
        cls.target = _make_operator(
            "transfer.target",
            ("item", "from_location", "to_location"),
            residual_name="local_context",
        )
        table, cls.source_bridge = _admit(
            AliasTable(), cls.source, cls.bridge, 0
        )
        cls.table, cls.bridge_target = _admit(
            table, cls.bridge, cls.target, 1
        )
        values = {
            "item": "novel_item",
            "from_location": "novel_origin",
            "to_location": "novel_destination",
            "local_context": "target_only_context",
        }
        cls.target_binding = StateOperatorBinding(
            operator_digest=cls.target.digest,
            namespace=cls.target.namespace,
            assignments=tuple(
                StateBindingAssignment(variable, values[variable.name])
                for variable in cls.target.variables
            ),
        )

    def test_chained_certificates_transfer_only_executable_core_assignments(self) -> None:
        with patch.dict(
            sys.modules,
            {"angler.procedures.execution": _execution_stub()},
        ):
            result = certified_transfer_binding(
                self.table,
                self.source,
                self.target,
                self.target_binding,
            )

        self.assertIsInstance(result, _ExecutionBinding)
        self.assertEqual(result.operator, self.source)
        self.assertEqual(
            {item.variable.name: item.entity.value for item in result.assignments},
            {
                "destination": "novel_destination",
                "entity": "novel_item",
                "source": "novel_origin",
            },
        )
        self.assertNotIn(
            "target_only_context",
            {item.entity.value for item in result.assignments},
        )

    def test_binding_transfer_fails_without_certified_chain_or_when_ambiguous(self) -> None:
        with self.assertRaisesRegex(CertifiedTransferError, "certified chain"):
            certified_transfer_binding(
                AliasTable(),
                self.source,
                self.target,
                self.target_binding,
            )

        normal = dict(self.bridge_target.variable_map)
        bridge_origin = dict(self.source_bridge.variable_map)["source"]
        bridge_endpoint = dict(self.source_bridge.variable_map)["destination"]
        target_origin = normal[bridge_origin]
        target_endpoint = normal[bridge_endpoint]
        rogue_map = tuple(
            sorted(
                (
                    source,
                    target_endpoint
                    if source == bridge_origin
                    else target_origin
                    if source == bridge_endpoint
                    else target,
                )
                for source, target in self.bridge_target.variable_map
            )
        )
        rogue_digest = _sha("a")
        rogue = VerifiedAliasEntry(
            candidate_digest=rogue_digest,
            source_operator_digest=self.bridge.digest,
            target_operator_digest=self.target.digest,
            aliases=self.bridge_target.aliases,
            variable_map=rogue_map,
            certificate=CounterfactualExecutionCertificate(
                candidate_digest=rogue_digest,
                execution_digest=_sha("b"),
                result_digest=_sha("c"),
                result="pass",
                issued_by="external.test.evaluator",
            ),
        )
        ambiguous = AliasTable(self.table.entries + (rogue,))
        with self.assertRaisesRegex(CertifiedTransferError, "ambiguous"):
            certified_transfer_binding(
                ambiguous,
                self.source,
                self.target,
                self.target_binding,
            )

    def test_predicate_projection_shares_certified_symbols_and_isolates_residuals(self) -> None:
        projector = CertifiedPredicateProjector(
            self.table,
            canonical_namespace="transfer.canonical",
            canonical_source_namespace=self.source.namespace,
        )
        source_at = Record(
            f"{self.source.namespace}.at",
            ("source_item", "source_place"),
        )
        target_at = Record(
            f"{self.target.namespace}.at",
            ("target_item", "target_place"),
        )
        source_extra = Record(
            f"{self.source.namespace}.extra",
            ("source_only",),
        )
        target_extra = Record(
            f"{self.target.namespace}.extra",
            ("target_only",),
        )

        self.assertEqual(
            projector.project_record(source_at).predicate,
            projector.project_record(target_at).predicate,
        )
        self.assertNotEqual(
            projector.project_record(source_extra).predicate,
            projector.project_record(target_extra).predicate,
        )
        self.assertIn(
            ".residual_",
            projector.project_record(target_extra).predicate,
        )
        state = State.from_records(
            self.target.namespace,
            (target_at, target_extra),
        )
        projected_state = projector.project_state(state)
        self.assertEqual(projected_state.namespace, "transfer.canonical")
        self.assertTrue(
            all(item.arguments in (target_at.arguments, target_extra.arguments) for item in projected_state.records)
        )
        goal = Goal.from_records(
            self.target.namespace,
            (target_at,),
            forbidden=(target_extra,),
        )
        projected_goal = projector.project_goal(goal)
        self.assertEqual(projected_goal.namespace, "transfer.canonical")
        self.assertFalse(projected_goal.exact)
        self.assertEqual(projected_goal.required[0].arguments, target_at.arguments)
        self.assertEqual(projected_goal.forbidden[0].arguments, target_extra.arguments)

    def test_source_projection_is_zero_hop_but_target_requires_certificate(self) -> None:
        projector = CertifiedPredicateProjector(
            AliasTable(),
            canonical_namespace="transfer.canonical",
            canonical_source_namespace=self.source.namespace,
        )
        source_state = State.from_records(
            self.source.namespace,
            (Record(f"{self.source.namespace}.at", ("item", "place")),),
        )
        projected = projector.project_state(source_state)
        self.assertEqual(projected.namespace, "transfer.canonical")
        admitted = CertifiedPredicateProjector(
            self.table,
            canonical_namespace="transfer.canonical",
            canonical_source_namespace=self.source.namespace,
        )
        self.assertEqual(
            projected.records[0].predicate,
            admitted.project_state(source_state).records[0].predicate,
        )

        target_state = State.from_records(
            self.target.namespace,
            (Record(f"{self.target.namespace}.at", ("item", "place")),),
        )
        with self.assertRaisesRegex(CertifiedTransferError, "certified chain"):
            projector.project_state(target_state)

    def test_action_projection_round_trips_exact_schema_order_and_arguments(self) -> None:
        source_schema = self.source.body[0].schema
        target_schema = self.target.body[0].schema
        adapter = CertifiedActionAdapter(
            self.table,
            local_schema=target_schema,
            canonical_schema=source_schema,
        )
        local = target_schema.ground("fresh", "left", "right")
        canonical = adapter.project_action(local)

        self.assertEqual(adapter.project_schema(target_schema), source_schema)
        self.assertEqual(canonical.schema, source_schema)
        self.assertEqual(canonical.arguments, local.arguments)
        self.assertEqual(adapter.reverse_action(canonical), local)
        self.assertEqual(adapter.reverse_schema(source_schema), target_schema)

        identity = CertifiedActionAdapter(AliasTable(), source_schema, source_schema)
        self.assertEqual(identity.project_action(source_schema.ground("x", "y", "z")).arguments, ("x", "y", "z"))
        with self.assertRaisesRegex(CertifiedTransferError, "certified alias chain"):
            CertifiedActionAdapter(AliasTable(), target_schema, source_schema)


if __name__ == "__main__":
    unittest.main()
