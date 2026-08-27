"""Tests for bounded callback-verified early-teacher search."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures import expert_iteration  # noqa: E402
from angler.procedures.expert_iteration import (  # noqa: E402
    ExpertIterationError,
    TrialEvidence,
    search_teacher_plan,
)
from angler.procedures.grounding import (  # noqa: E402
    GroundingError,
    enumerate_operator_bindings,
    instantiate_operator,
)
from angler.procedures.induction import (  # noqa: E402
    MDLOperatorInducer,
    cluster_subsegments,
    extract_subsegment_delta,
)
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    Parameter,
    Record,
    State,
    Trace,
    Transition,
)


NAMESPACE = "teacher.demo"
ENTITY = f"{NAMESPACE}.entity"
LOCATION = f"{NAMESPACE}.location"
AT = f"{NAMESPACE}.at"
CLEAR = f"{NAMESPACE}.clear"
MOVE = ActionSchema(
    f"{NAMESPACE}.move",
    (
        Parameter("entity", ENTITY),
        Parameter("source", LOCATION),
        Parameter("destination", LOCATION),
    ),
    description="Move one entity from its location into a clear location.",
)


def _state(entity: str, occupied: str, clears: tuple[str, ...]) -> State:
    return State.from_records(
        NAMESPACE,
        (
            Record(AT, (entity, occupied)),
            *(Record(CLEAR, (place,)) for place in clears),
        ),
    )


def _training_trace(entity: str, source: str, destination: str) -> Trace:
    before = _state(entity, source, (destination,))
    after = _state(entity, destination, (source,))
    transition = Transition(
        before,
        MOVE.ground(entity, source, destination),
        after,
        True,
        f"{NAMESPACE}.applied",
    )
    return Trace(before, (transition,))


def _operator():
    segments = tuple(
        extract_subsegment_delta(trace, 0, 1)
        for trace in (
            _training_trace("robot_1", "alpha", "beta"),
            _training_trace("robot_2", "gamma", "delta"),
        )
    )
    candidate = MDLOperatorInducer(
        minimum_support=2,
        minimum_savings=-100,
    ).allocate(cluster_subsegments(segments)[0])
    if candidate is None:
        raise AssertionError("test traces must induce an operator")
    return candidate.operator


def _satisfies(state: State, goal: Goal) -> bool:
    records = set(state.records)
    if goal.exact:
        return state.records == goal.required
    return set(goal.required) <= records and not (set(goal.forbidden) & records)


class GraphTrialBoundary:
    """Test-only external physics; the teacher never imports this behavior."""

    def __init__(self) -> None:
        self.edges = {("a", "b"), ("b", "c")}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if len(request.actions) != 1:
            raise AssertionError("test operator must propose one primitive")
        entity, source, destination = request.actions[0].arguments
        records = set(request.origin.records)
        located = Record(AT, (entity, source))
        clear = Record(CLEAR, (destination,))
        applicable = (
            (source, destination) in self.edges
            and located in records
            and clear in records
        )
        if applicable:
            records.remove(located)
            records.remove(clear)
            records.add(Record(AT, (entity, destination)))
            records.add(Record(CLEAR, (source,)))
            observed = State.from_records(NAMESPACE, records)
            applied = 1
            cost = 3
        else:
            observed = request.origin
            applied = 0
            cost = 7
        return TrialEvidence(
            request_digest=request.digest,
            observed_state=observed,
            success=_satisfies(observed, request.goal),
            applied_actions=applied,
            cost=cost,
        )


class ExpertIterationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.operator = _operator()
        self.initial = _state("unseen_unit", "a", ("b", "c"))
        self.goal = Goal.from_records(
            NAMESPACE,
            (Record(AT, ("unseen_unit", "c")),),
        )

    def test_verified_plan_retains_exact_chain_and_failed_trial_accounting(self) -> None:
        boundary = GraphTrialBoundary()

        result = search_teacher_plan(
            self.initial,
            self.goal,
            (self.operator,),
            boundary,
            maximum_operator_depth=2,
            maximum_expansions=8,
        )

        self.assertEqual(result.termination, "verified")
        self.assertIsNotNone(result.plan)
        assert result.plan is not None
        self.assertEqual(result.expansions, 3)
        self.assertEqual(len(boundary.requests), 3)
        self.assertEqual(len({item.digest for item in boundary.requests}), 3)
        self.assertFalse(result.trials[0].evidence.success)
        self.assertEqual(result.trials[0].evidence.applied_actions, 0)
        self.assertNotIn(result.trials[0], result.plan.chain)
        self.assertEqual(
            tuple(trial.request.actions[0].arguments[1:] for trial in result.plan.chain),
            (("a", "b"), ("b", "c")),
        )
        self.assertEqual(
            tuple(trial.request.operator_depth for trial in result.plan.chain),
            (1, 2),
        )
        self.assertEqual(result.plan.accounting, result.trials)
        self.assertEqual(result.total_cost, 13)
        self.assertEqual(result.total_applied_actions, 2)
        self.assertTrue(result.plan.chain[-1].evidence.success)
        self.assertRegex(result.plan.digest, r"^sha256:[0-9a-f]{64}$")

    def test_expansion_budget_stops_without_hidden_retry(self) -> None:
        boundary = GraphTrialBoundary()
        result = search_teacher_plan(
            self.initial,
            self.goal,
            (self.operator,),
            boundary,
            maximum_operator_depth=2,
            maximum_expansions=1,
        )

        self.assertIsNone(result.plan)
        self.assertEqual(result.termination, "expansion_limit")
        self.assertEqual(result.expansions, 1)
        self.assertEqual(len(boundary.requests), 1)
        self.assertEqual(result.trials[0].request.actions[0].arguments[2], "c")

    def test_depth_budget_does_not_continue_from_observed_progress(self) -> None:
        boundary = GraphTrialBoundary()
        result = search_teacher_plan(
            self.initial,
            self.goal,
            (self.operator,),
            boundary,
            maximum_operator_depth=1,
            maximum_expansions=8,
        )

        self.assertIsNone(result.plan)
        self.assertEqual(result.termination, "exhausted")
        self.assertEqual(result.expansions, 2)
        self.assertEqual(len(boundary.requests), 2)
        self.assertTrue(any(item.evidence.applied_actions for item in result.trials))

    def test_goal_effect_ordering_can_be_disabled_for_ablation(self) -> None:
        raw_bindings = enumerate_operator_bindings(self.operator, self.initial)
        expected_first = instantiate_operator(
            self.operator,
            raw_bindings[0],
        ).actions
        disabled_requests = []

        def always_fail(request):
            disabled_requests.append(request)
            return TrialEvidence(
                request.digest,
                request.origin,
                False,
                0,
                1,
            )

        disabled = search_teacher_plan(
            self.initial,
            self.goal,
            (self.operator,),
            always_fail,
            maximum_operator_depth=1,
            maximum_expansions=1,
            order_by_goal_effect_overlap=False,
        )
        self.assertEqual(disabled_requests[0].actions, expected_first)
        self.assertIsNone(disabled_requests[0].goal_effect_overlap)
        self.assertEqual(disabled.termination, "expansion_limit")

        enabled_requests = []

        def ordered_fail(request):
            enabled_requests.append(request)
            return TrialEvidence(request.digest, request.origin, False, 0, 1)

        search_teacher_plan(
            self.initial,
            self.goal,
            (self.operator,),
            ordered_fail,
            maximum_operator_depth=1,
            maximum_expansions=1,
            order_by_goal_effect_overlap=True,
        )
        self.assertEqual(enabled_requests[0].actions[0].arguments[2], "c")
        self.assertEqual(enabled_requests[0].goal_effect_overlap, 1)

    def test_contradictory_coreference_is_skipped_without_aborting_search(self) -> None:
        initial = _state("unit", "a", ("a", "b"))
        goal = Goal.from_records(
            NAMESPACE,
            (Record(AT, ("unit", "b")),),
        )
        bindings = enumerate_operator_bindings(self.operator, initial)
        valid_actions = []
        contradictions = 0
        for binding in bindings:
            try:
                valid_actions.append(
                    instantiate_operator(self.operator, binding).actions
                )
            except GroundingError:
                contradictions += 1
        self.assertEqual(contradictions, 1)
        self.assertEqual(valid_actions, [(MOVE.ground("unit", "a", "b"),)])

        requests = []

        def verify_valid_binding(request):
            requests.append(request)
            self.assertEqual(request.actions, (MOVE.ground("unit", "a", "b"),))
            observed = _state("unit", "b", ("a",))
            return TrialEvidence(request.digest, observed, True, 1, 1)

        result = search_teacher_plan(
            initial,
            goal,
            (self.operator,),
            verify_valid_binding,
            maximum_operator_depth=1,
            maximum_expansions=4,
        )

        self.assertEqual(result.termination, "verified")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.expansions, 1)
        self.assertEqual(len(requests), 1)

    def test_success_claim_without_goal_observation_is_rejected(self) -> None:
        def false_claim(request):
            return TrialEvidence(
                request.digest,
                request.origin,
                True,
                0,
                1,
            )

        with self.assertRaisesRegex(ExpertIterationError, "claimed success"):
            search_teacher_plan(
                self.initial,
                self.goal,
                (self.operator,),
                false_claim,
                maximum_expansions=1,
            )

    def test_evidence_must_bind_the_exact_request(self) -> None:
        def mismatched(request):
            return TrialEvidence(
                "sha256:" + "0" * 64,
                request.origin,
                False,
                0,
                1,
            )

        with self.assertRaisesRegex(ExpertIterationError, "another request"):
            search_teacher_plan(
                self.initial,
                self.goal,
                (self.operator,),
                mismatched,
                maximum_expansions=1,
            )

    def test_module_has_no_world_evaluator_or_transition_engine_import(self) -> None:
        source_path = SRC / "angler" / "procedures" / "expert_iteration.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(
            any(
                name.startswith("angler.worlds")
                or name.startswith("experiments.evaluators")
                for name in imported
            )
        )
        public_names = set(expert_iteration.__all__)
        forbidden = ("world", "evaluator", "solver", "route", "shortest")
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in public_names
                for fragment in forbidden
            )
        )


if __name__ == "__main__":
    unittest.main()
