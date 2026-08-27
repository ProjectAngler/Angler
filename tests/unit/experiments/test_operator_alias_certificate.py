from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from angler.procedures.alignment import find_structural_isomorphisms  # noqa: E402
from angler.procedures.grounding import (  # noqa: E402
    StateBindingAssignment,
    StateOperatorBinding,
    instantiate_operator,
)
from angler.procedures.operators import (  # noqa: E402
    ActionPattern,
    Effect,
    LearnedOperator,
    OperatorExemplar,
    RecordPattern,
    ReconstructionExemplar,
    TypedVariable,
)
from angler.procedures.records import Record, State  # noqa: E402
from experiments.evaluators import operator_alias_certificate as certificate_module  # noqa: E402
from experiments.evaluators.causal_operator_suite import (  # noqa: E402
    OperatorChallenge,
    commit_action_sequence,
    make_heldout_operator_suite,
)
from experiments.evaluators.operator_alias_certificate import (  # noqa: E402
    AliasCertificateEvaluationError,
    EVALUATOR_ISSUER,
    certify_counterfactual_alignment,
)


def _placement_predicate(challenge: OperatorChallenge) -> str:
    suffix = {
        "tokens": "token_in",
        "files": "file_at",
    }[challenge.domain]
    return f"{challenge.origin.namespace}.{suffix}"


def _challenge_values(challenge: OperatorChallenge) -> dict[str, str]:
    predicate = _placement_predicate(challenge)
    by_position = {
        record.arguments[1]: record.arguments[0]
        for record in challenge.origin.records
        if record.predicate == predicate
    }
    positions = sorted(
        {
            record.arguments[1]
            for record in challenge.goal.required
            if record.predicate == predicate
        }
        | set(by_position)
    )
    return {
        "entity0": by_position[positions[0]],
        "entity1": by_position[positions[1]],
        "place0": positions[0],
        "place1": positions[1],
        "place2": positions[2],
    }


def _operator_and_binding(
    challenge: OperatorChallenge,
    *,
    complete_effects: bool = True,
    binding_mode: str = "valid",
) -> tuple[LearnedOperator, StateOperatorBinding]:
    schema = challenge.allowed_action_schemas[0]
    predicate = _placement_predicate(challenge)
    entity_type = schema.parameters[0].type_name
    place_type = schema.parameters[1].type_name
    variables = {
        "entity0": TypedVariable("entity0", entity_type),
        "entity1": TypedVariable("entity1", entity_type),
        "place0": TypedVariable("place0", place_type),
        "place1": TypedVariable("place1", place_type),
        "place2": TypedVariable("place2", place_type),
    }
    values = _challenge_values(challenge)
    actions = (
        schema.ground(values["entity1"], values["place1"], values["place2"]),
        schema.ground(values["entity0"], values["place0"], values["place1"]),
    )
    start_records = tuple(
        sorted(
            (
                Record(predicate, (values["entity0"], values["place0"])),
                Record(predicate, (values["entity1"], values["place1"])),
            )
        )
    )
    end_records = tuple(
        sorted(
            (
                Record(predicate, (values["entity0"], values["place1"])),
                Record(predicate, (values["entity1"], values["place2"])),
            )
        )
    )
    reconstruction = ReconstructionExemplar(
        namespace=challenge.origin.namespace,
        start_records=start_records,
        variable_bindings=tuple(sorted(values.items())),
        constant_values=(),
        actions=actions,
        end_records=end_records,
    )
    effects = [
        Effect(
            "delete",
            RecordPattern(
                predicate,
                (variables["entity1"], variables["place1"]),
            ),
        ),
        Effect(
            "add",
            RecordPattern(
                predicate,
                (variables["entity1"], variables["place2"]),
            ),
        ),
    ]
    if complete_effects:
        effects.extend(
            (
                Effect(
                    "delete",
                    RecordPattern(
                        predicate,
                        (variables["entity0"], variables["place0"]),
                    ),
                ),
                Effect(
                    "add",
                    RecordPattern(
                        predicate,
                        (variables["entity0"], variables["place1"]),
                    ),
                ),
            )
        )
    target_state = State(challenge.goal.namespace, challenge.goal.required)
    operator = LearnedOperator(
        name=f"{challenge.origin.namespace}.certificate_relocation",
        namespace=challenge.origin.namespace,
        variables=tuple(variables.values()),
        preconditions=(
            RecordPattern(
                predicate,
                (variables["entity0"], variables["place0"]),
            ),
            RecordPattern(
                predicate,
                (variables["entity1"], variables["place1"]),
            ),
        ),
        effects=tuple(effects),
        body=(
            ActionPattern(
                schema,
                (
                    variables["entity1"],
                    variables["place1"],
                    variables["place2"],
                ),
            ),
            ActionPattern(
                schema,
                (
                    variables["entity0"],
                    variables["place0"],
                    variables["place1"],
                ),
            ),
        ),
        exemplars=(
            OperatorExemplar(
                trace_digest=challenge.case_id,
                start_index=0,
                stop_index=2,
                before_state_digest=challenge.origin.digest,
                after_state_digest=target_state.digest,
                action_digests=tuple(item.digest for item in actions),
                reconstruction=reconstruction,
            ),
        ),
    )
    bound_values = dict(values)
    if binding_mode == "partial":
        bound_values["entity0"] = values["entity0"] + "_missing"
    elif binding_mode == "blocked":
        bound_values["entity0"] = values["entity1"]
        bound_values["entity1"] = values["entity0"]
    elif binding_mode != "valid":
        raise ValueError("unsupported test binding mode")
    by_name = {item.name: item for item in operator.variables}
    binding = StateOperatorBinding(
        operator_digest=operator.digest,
        namespace=operator.namespace,
        assignments=tuple(
            StateBindingAssignment(variable, bound_values[variable.name])
            for variable in operator.variables
        ),
    )
    self_check = {item.variable.name for item in binding.assignments}
    if self_check != set(by_name):
        raise AssertionError("test binding did not cover the operator")
    return operator, binding


class OperatorAliasCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        suite = make_heldout_operator_suite(91_733)
        self.source_challenge = next(
            item
            for item in suite
            if item.domain == "tokens" and item.maximum_steps == 2
        )
        self.target_challenge = next(
            item
            for item in suite
            if item.domain == "files" and item.maximum_steps == 2
        )

    def _case(
        self,
        *,
        complete_effects: bool = True,
        binding_mode: str = "valid",
    ):
        source_operator, source_binding = _operator_and_binding(
            self.source_challenge,
            complete_effects=complete_effects,
            binding_mode=binding_mode,
        )
        target_operator, target_binding = _operator_and_binding(
            self.target_challenge,
            complete_effects=complete_effects,
            binding_mode=binding_mode,
        )
        candidate = find_structural_isomorphisms(
            source_operator,
            target_operator,
        )[0]
        source_prediction = instantiate_operator(source_operator, source_binding)
        target_prediction = instantiate_operator(target_operator, target_binding)
        source_commitment = commit_action_sequence(
            self.source_challenge,
            source_prediction.actions,
        )
        target_commitment = commit_action_sequence(
            self.target_challenge,
            target_prediction.actions,
        )
        return (
            candidate,
            source_operator,
            target_operator,
            source_binding,
            target_binding,
            source_commitment,
            target_commitment,
        )

    def _certify(self, case):
        return certify_counterfactual_alignment(
            case[0],
            case[1],
            case[2],
            case[3],
            case[4],
            self.source_challenge,
            self.target_challenge,
            case[5],
            case[6],
        )

    def test_passing_certificate_is_derived_after_exactly_two_executions(self) -> None:
        case = self._case()
        with patch.object(
            certificate_module,
            "evaluate_committed_sequence",
            wraps=certificate_module.evaluate_committed_sequence,
        ) as execute:
            certificate = self._certify(case)

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(certificate.result, "pass")
        self.assertEqual(certificate.candidate_digest, case[0].digest)
        self.assertEqual(certificate.issued_by, EVALUATOR_ISSUER)
        self.assertRegex(certificate.execution_digest, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(certificate.result_digest, r"^sha256:[0-9a-f]{64}$")

    def test_tampered_commitment_is_rejected_before_execution(self) -> None:
        case = list(self._case())
        case[5] = commit_action_sequence(
            self.source_challenge,
            tuple(reversed(case[5].actions)),
        )
        with patch.object(
            certificate_module,
            "evaluate_committed_sequence",
            wraps=certificate_module.evaluate_committed_sequence,
        ) as execute:
            with self.assertRaisesRegex(
                AliasCertificateEvaluationError,
                "instantiated mirror actions",
            ):
                self._certify(tuple(case))
        self.assertEqual(execute.call_count, 0)

    def test_partially_applied_bodies_receive_fail_certificate(self) -> None:
        case = self._case(binding_mode="partial")
        certificate = self._certify(case)
        self.assertEqual(certificate.result, "fail")

    def test_fully_blocked_bodies_receive_fail_certificate(self) -> None:
        case = self._case(binding_mode="blocked")
        certificate = self._certify(case)
        self.assertEqual(certificate.result, "fail")

    def test_successful_actions_with_incomplete_mirror_delta_fail(self) -> None:
        case = self._case(complete_effects=False)
        certificate = self._certify(case)
        self.assertEqual(certificate.result, "fail")

    def test_evaluator_has_no_learner_teacher_or_runner_import(self) -> None:
        path = ROOT / "experiments" / "evaluators" / "operator_alias_certificate.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(
                "learning" in name
                or "expert_iteration" in name
                or name.startswith("experiments.runners")
                for name in imports
            )
        )


if __name__ == "__main__":
    unittest.main()
