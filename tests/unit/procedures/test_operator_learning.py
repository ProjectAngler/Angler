from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures.execution import (  # noqa: E402
    BindingAssignment,
    OperatorBinding,
    SharedPrimitiveSequenceDecoder,
    TypedEntityCandidate,
)
from angler.procedures.learning import (  # noqa: E402
    CandidateEvidenceFusion,
    CompositeOperatorLearner,
    VerifiedOperatorExample,
    VerifiedOperatorTrajectory,
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
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    Parameter,
    Record,
    State,
)
from angler.procedures.trunk import NeuralOperatorCore  # noqa: E402


NAMESPACE = "test.learning"
ZERO = "sha256:" + "0" * 64
ONE = "sha256:" + "1" * 64


def _record(predicate: str, entity: str) -> Record:
    return Record(f"{NAMESPACE}.{predicate}", (entity,))


def _fixtures() -> tuple[
    LearnedOperator,
    ActionSchema,
    tuple[TypedEntityCandidate, ...],
    tuple[OperatorBinding, ...],
]:
    type_name = f"{NAMESPACE}.entity"
    variable = TypedVariable("entity", type_name)
    inspect = ActionSchema(
        f"{NAMESPACE}.inspect",
        (Parameter("entity", type_name),),
        description="Inspect the selected entity.",
    )
    observed_action = inspect.ground("box-a")
    reconstruction = ReconstructionExemplar(
        namespace=NAMESPACE,
        start_records=(_record("ready", "box-a"),),
        variable_bindings=(("entity", "box-a"),),
        constant_values=(),
        actions=(observed_action,),
        end_records=(_record("marked", "box-a"),),
    )
    operator = LearnedOperator(
        name=f"{NAMESPACE}.inspect_then_mark",
        namespace=NAMESPACE,
        variables=(variable,),
        preconditions=(RecordPattern(f"{NAMESPACE}.ready", (variable,)),),
        effects=(Effect("add", RecordPattern(f"{NAMESPACE}.marked", (variable,))),),
        body=(ActionPattern(inspect, (variable,)),),
        exemplars=(
            OperatorExemplar(
                trace_digest=ZERO,
                start_index=0,
                stop_index=1,
                before_state_digest=ZERO,
                after_state_digest=ONE,
                action_digests=(observed_action.digest,),
                reconstruction=reconstruction,
            ),
        ),
    )
    entities = (
        TypedEntityCandidate("box-a", type_name),
        TypedEntityCandidate("box-b", type_name),
    )
    bindings = tuple(
        OperatorBinding(
            operator,
            (BindingAssignment(variable, entity),),
        )
        for entity in entities
    )
    return operator, inspect, entities, bindings


def _example(positive: int) -> VerifiedOperatorExample:
    _, inspect, entities, bindings = _fixtures()
    entity = entities[positive].value
    before = State.from_records(
        NAMESPACE,
        (_record("ready", "box-a"), _record("ready", "box-b")),
    )
    after = State.from_records(NAMESPACE, (_record("marked", entity),))
    goal = Goal.from_records(NAMESPACE, after.records, exact=True)
    labels = (positive == 0, positive == 1)
    return VerifiedOperatorExample(
        before=before,
        after=after,
        goal=goal,
        positive_binding=bindings[positive],
        candidate_bindings=bindings,
        applicability_labels=labels,
        verified_primitives=(inspect.ground(entity),),
        allowed_schemas=(inspect,),
        entity_candidates=entities,
    )


def _trajectory() -> VerifiedOperatorTrajectory:
    _, inspect, entities, bindings = _fixtures()
    start = State.from_records(
        NAMESPACE,
        (_record("ready", "box-a"), _record("ready", "box-b")),
    )
    middle = State.from_records(
        NAMESPACE,
        (_record("marked", "box-a"), _record("ready", "box-b")),
    )
    final = State.from_records(
        NAMESPACE,
        (_record("marked", "box-a"), _record("marked", "box-b")),
    )
    goal = Goal.from_records(NAMESPACE, final.records, exact=True)

    def step(
        before: State,
        after: State,
        positive: int,
        labels: tuple[bool, bool],
    ) -> VerifiedOperatorExample:
        entity = entities[positive].value
        return VerifiedOperatorExample(
            before=before,
            after=after,
            goal=goal,
            positive_binding=bindings[positive],
            candidate_bindings=bindings,
            applicability_labels=labels,
            verified_primitives=(inspect.ground(entity),),
            allowed_schemas=(inspect,),
            entity_candidates=entities,
        )

    return VerifiedOperatorTrajectory(
        (
            step(start, middle, 0, (True, True)),
            step(middle, final, 1, (False, True)),
        )
    )


class CompositeOperatorLearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2301)
        torch.set_num_threads(1)
        core = NeuralOperatorCore(width=24, hidden_width=40, schema_hash_width=64)
        decoder = SharedPrimitiveSequenceDecoder(
            width=24,
            hidden_width=40,
            hash_width=64,
            maximum_steps=4,
        )
        self.learner = CompositeOperatorLearner(
            core,
            decoder,
            binding_hash_width=64,
            hidden_width=40,
        )

    def test_horizon_agnostic_join_uses_learned_termination_gate(self) -> None:
        forward = torch.randn(3, self.learner.core.width)
        backward = torch.randn(3, self.learner.core.width)
        goal = torch.randn(self.learner.core.width)
        mask = torch.tensor((True, True, True))
        direct = -((forward - goal.unsqueeze(0)) ** 2).mean(dim=-1)

        with patch.object(
            self.learner.core,
            "termination_logits",
            return_value=torch.full((3,), 30.0),
        ):
            final_scores = self.learner.horizon_agnostic_join_scores(
                forward,
                backward,
                goal,
                mask,
            )
        with patch.object(
            self.learner.core,
            "termination_logits",
            return_value=torch.full((3,), -30.0),
        ):
            bridge_scores = self.learner.horizon_agnostic_join_scores(
                forward,
                backward,
                goal,
                mask,
            )

        self.assertTrue(torch.allclose(final_scores, direct, atol=1e-5))
        self.assertTrue(bool(torch.isfinite(bridge_scores).all().item()))
        self.assertFalse(torch.allclose(bridge_scores, direct))

    def test_horizon_agnostic_join_is_local_finite_and_differentiable(self) -> None:
        forward = torch.randn(3, self.learner.core.width, requires_grad=True)
        backward = torch.randn(3, self.learner.core.width, requires_grad=True)
        goal = torch.randn(self.learner.core.width, requires_grad=True)
        mask = torch.tensor((True, True, True))
        scores = self.learner.horizon_agnostic_join_scores(
            forward,
            backward,
            goal,
            mask,
        )
        permutation = torch.tensor((2, 0, 1))
        permuted = self.learner.horizon_agnostic_join_scores(
            forward[permutation],
            backward[permutation],
            goal,
            mask,
        )

        self.assertTrue(torch.allclose(permuted, scores[permutation]))
        scores.sum().backward()
        for tensor in (forward, backward, goal):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all().item()))
            self.assertTrue(bool((tensor.grad != 0).any().item()))

        singleton_mask = torch.tensor((False, True, False))
        singleton = self.learner.horizon_agnostic_join_scores(
            forward.detach(),
            backward.detach(),
            goal.detach(),
            singleton_mask,
        )
        direct = -(
            (forward.detach()[1] - goal.detach()) ** 2
        ).mean()
        self.assertTrue(torch.allclose(singleton[1], direct))
        self.assertTrue(bool(torch.isneginf(singleton[~singleton_mask]).all().item()))

        invalid = forward.detach().clone()
        invalid[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.learner.horizon_agnostic_join_scores(
                invalid,
                backward.detach(),
                goal.detach(),
                mask,
            )
        with self.assertRaisesRegex(ValueError, "requires one active candidate"):
            self.learner.horizon_agnostic_join_scores(
                forward.detach(),
                backward.detach(),
                goal.detach(),
                torch.zeros(3, dtype=torch.bool),
            )

    def test_named_losses_are_finite_and_gradients_reach_every_subsystem(self) -> None:
        losses = self.learner((_example(0),))

        self.assertEqual(losses.positive_binding_indices, (0,))
        self.assertEqual(
            set(losses.as_dict()),
            {
                "effect",
                "predecessor",
                "initiation",
                "termination",
                "proposer",
                "primitive_action",
                "primitive_argument",
                "total",
            },
        )
        for name, loss in losses.as_dict().items():
            self.assertEqual(loss.shape, (), name)
            self.assertTrue(bool(torch.isfinite(loss).item()), name)

        losses.total.backward()
        for prefix in (
            "heads.core.state_encoder",
            "heads.core.goal_encoder",
            "heads.core.schema_encoder",
            "heads.core.initiation_head",
            "heads.core.effect_head",
            "heads.core.termination_head",
            "heads.binding_encoder",
            "heads.binding_proposer.composer",
            "heads.binding_proposer.query",
            "heads.binding_proposer.keys",
            "decoder.schema_encoder",
            "decoder.binding_encoder",
            "decoder.action_query",
            "decoder.argument_query",
            "decoder.choice_projection",
            "decoder.history_cell",
        ):
            gradients = [
                parameter.grad
                for name, parameter in self.learner.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(
                any(
                    gradient is not None
                    and bool(torch.isfinite(gradient).all().item())
                    and bool((gradient != 0).any().item())
                    for gradient in gradients
                ),
                prefix,
            )

    def test_swapping_positive_binding_changes_imitation_target(self) -> None:
        first = _example(0)
        second = _example(1)

        losses = self.learner((first, second))

        self.assertEqual(first.positive_index, 0)
        self.assertEqual(second.positive_index, 1)
        self.assertEqual(losses.positive_binding_indices, (0, 1))
        self.assertEqual(
            first.verified_primitives[0].arguments,
            ("box-a",),
        )
        self.assertEqual(
            second.verified_primitives[0].arguments,
            ("box-b",),
        )

    def test_termination_learns_from_observed_and_predicted_successors(self) -> None:
        observed_batches: list[int] = []
        handle = self.learner.core.termination_head.register_forward_pre_hook(
            lambda _module, inputs: observed_batches.append(inputs[0].shape[0])
        )
        try:
            self.learner((_example(0),))
        finally:
            handle.remove()

        self.assertEqual(observed_batches, [3])

    def test_zero_teacher_forcing_rolls_its_own_latents_both_directions(self) -> None:
        trajectory = _trajectory()
        observed_middle = self.learner.core.encode_states(
            (trajectory.steps[0].after,)
        ).detach()
        goal_state = self.learner.core.encode_goal_states((trajectory.goal,)).detach()
        calls: list[
            tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]
        ] = []
        real_predict = self.learner.core.predict_effects

        def observed_predict(
            states: torch.Tensor,
            operators: torch.Tensor,
            *,
            reverse: bool = False,
        ) -> torch.Tensor:
            result = real_predict(states, operators, reverse=reverse)
            calls.append((states, operators, reverse, result))
            return result

        with patch.object(
            self.learner.core,
            "predict_effects",
            side_effect=observed_predict,
        ):
            self.learner.trajectory_losses(
                (trajectory,),
                teacher_forcing_ratio=0.0,
            )

        forward_calls = [item for item in calls if not item[2]]
        reverse_rollout_calls = [
            item for item in calls if item[2] and item[1].shape[0] == 1
        ]
        self.assertEqual(len(forward_calls), 2)
        self.assertEqual(len(reverse_rollout_calls), 2)
        first_forward = forward_calls[0][3][
            :,
            trajectory.steps[0].positive_index,
            :,
        ]
        self.assertTrue(torch.equal(forward_calls[1][0], first_forward))
        self.assertIsNotNone(forward_calls[1][0].grad_fn)
        self.assertFalse(
            torch.equal(forward_calls[1][0].detach(), observed_middle)
        )

        self.assertTrue(
            torch.equal(reverse_rollout_calls[0][0].detach(), goal_state)
        )
        first_reverse = reverse_rollout_calls[0][3][:, 0, :]
        self.assertTrue(torch.equal(reverse_rollout_calls[1][0], first_reverse))
        self.assertIsNotNone(reverse_rollout_calls[1][0].grad_fn)
        self.assertFalse(
            torch.equal(reverse_rollout_calls[1][0].detach(), observed_middle)
        )

        last_set = self.learner.heads.encode_candidates(
            trajectory.steps[1].candidate_bindings
        )
        expected_last = last_set[
            trajectory.steps[1].positive_index : trajectory.steps[1].positive_index + 1
        ]
        first_set = self.learner.heads.encode_candidates(
            trajectory.steps[0].candidate_bindings
        )
        expected_first = first_set[
            trajectory.steps[0].positive_index : trajectory.steps[0].positive_index + 1
        ]
        self.assertTrue(torch.equal(reverse_rollout_calls[0][1], expected_last))
        self.assertTrue(torch.equal(reverse_rollout_calls[1][1], expected_first))

    def test_trajectory_termination_labels_intermediate_false_and_final_true(self) -> None:
        captured_targets: list[torch.Tensor] = []
        real_bce = torch.nn.functional.binary_cross_entropy_with_logits

        def observed_bce(
            logits: torch.Tensor,
            targets: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> torch.Tensor:
            captured_targets.append(targets.detach().clone())
            return real_bce(logits, targets, *args, **kwargs)

        with patch(
            "angler.procedures.learning.F.binary_cross_entropy_with_logits",
            side_effect=observed_bce,
        ):
            losses = self.learner.trajectory_losses(
                (_trajectory(),),
                teacher_forcing_ratio=0.0,
            )

        # Initiation uses BCE at both stages; the final BCE call is the
        # dedicated trajectory termination target.
        self.assertEqual(len(captured_targets), 3)
        self.assertTrue(
            torch.equal(captured_targets[-1], torch.tensor((0.0, 1.0)))
        )
        for name, loss in losses.as_dict().items():
            self.assertTrue(bool(torch.isfinite(loss).item()), name)

    def test_trajectory_loss_reaches_dynamics_termination_binding_and_state(self) -> None:
        losses = self.learner.trajectory_losses(
            (_trajectory(),),
            teacher_forcing_ratio=0.0,
        )
        losses.total.backward()

        for prefix in (
            "heads.core.state_encoder",
            "heads.core.effect_head",
            "heads.core.termination_head",
            "heads.binding_encoder",
            "heads.binding_proposer.query",
            "decoder.action_query",
            "candidate_fusion",
        ):
            gradients = [
                parameter.grad
                for name, parameter in self.learner.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(
                any(
                    gradient is not None
                    and bool(torch.isfinite(gradient).all().item())
                    and bool((gradient != 0).any().item())
                    for gradient in gradients
                ),
                prefix,
            )

    def test_same_encoder_dynamics_targets_receive_gradients(self) -> None:
        captured: list[torch.Tensor] = []
        real_encode = self.learner.core.encode_states

        def observed_encode(states: tuple[State, ...]) -> torch.Tensor:
            result = real_encode(states)
            result.retain_grad()
            captured.append(result)
            return result

        with patch.object(
            self.learner.core,
            "encode_states",
            side_effect=observed_encode,
        ):
            losses = self.learner((_example(0),))
            losses.effect.backward()

        # The second state-encoder call is the observed successor target.
        self.assertGreaterEqual(len(captured), 2)
        successor_gradient = captured[1].grad
        self.assertIsNotNone(successor_gradient)
        self.assertTrue(bool(torch.isfinite(successor_gradient).all().item()))
        self.assertTrue(bool((successor_gradient != 0).any().item()))

    def test_selection_plasticity_freezes_acquired_dynamics_and_decoder(self) -> None:
        enabled = self.learner.configure_plasticity("selection")
        self.assertTrue(enabled)
        self.assertTrue(
            all(
                name.startswith(self.learner._SELECTION_PLASTIC_PREFIXES)
                for name in enabled
            )
        )
        self.assertFalse(self.learner.core.effect_head[1].weight.requires_grad)
        self.assertFalse(self.learner.core.termination_head[1].weight.requires_grad)
        self.assertFalse(self.learner.decoder.action_query.weight.requires_grad)

        atomic = self.learner((_example(0), _example(1)))
        trajectory = self.learner.trajectory_losses(
            (_trajectory(),),
            teacher_forcing_ratio=0.0,
        )
        (atomic.total + trajectory.total).backward()
        for name, parameter in self.learner.named_parameters():
            if name in enabled:
                self.assertTrue(parameter.requires_grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                and bool((parameter.grad != 0).any().item())
                for name, parameter in self.learner.named_parameters()
                if name in enabled
            )
        )

        restored = self.learner.configure_plasticity("full")
        self.assertEqual(len(restored), sum(1 for _ in self.learner.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.learner.parameters()))
        with self.assertRaisesRegex(ValueError, "full.*selection"):
            self.learner.configure_plasticity("unknown")

    def test_trajectory_validation_rejects_short_discontinuous_or_false_terminal(self) -> None:
        valid = _trajectory()
        with self.assertRaisesRegex(ValueError, "at least two"):
            VerifiedOperatorTrajectory((valid.steps[0],))

        discontinuous_second = VerifiedOperatorExample(
            before=valid.steps[0].before,
            after=valid.steps[1].after,
            goal=valid.goal,
            positive_binding=valid.steps[1].positive_binding,
            candidate_bindings=valid.steps[1].candidate_bindings,
            applicability_labels=valid.steps[1].applicability_labels,
            verified_primitives=valid.steps[1].verified_primitives,
            allowed_schemas=valid.steps[1].allowed_schemas,
            entity_candidates=valid.steps[1].entity_candidates,
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            VerifiedOperatorTrajectory((valid.steps[0], discontinuous_second))

        unreachable = Goal.from_records(
            NAMESPACE,
            (_record("marked", "box-a"),),
            exact=True,
        )
        wrong_steps = tuple(
            VerifiedOperatorExample(
                before=step.before,
                after=step.after,
                goal=unreachable,
                positive_binding=step.positive_binding,
                candidate_bindings=step.candidate_bindings,
                applicability_labels=step.applicability_labels,
                verified_primitives=step.verified_primitives,
                allowed_schemas=step.allowed_schemas,
                entity_candidates=step.entity_candidates,
            )
            for step in valid.steps
        )
        with self.assertRaisesRegex(ValueError, "final trajectory endpoint"):
            VerifiedOperatorTrajectory(wrong_steps)

    def test_partial_goal_and_unverified_positive_label_fail_closed(self) -> None:
        valid = _example(0)
        partial = Goal.from_records(NAMESPACE, valid.after.records, exact=False)
        with self.assertRaisesRegex(ValueError, "exact goal"):
            VerifiedOperatorExample(
                before=valid.before,
                after=valid.after,
                goal=partial,
                positive_binding=valid.positive_binding,
                candidate_bindings=valid.candidate_bindings,
                applicability_labels=valid.applicability_labels,
                verified_primitives=valid.verified_primitives,
                allowed_schemas=valid.allowed_schemas,
                entity_candidates=valid.entity_candidates,
            )
        with self.assertRaisesRegex(ValueError, "externally applicable"):
            VerifiedOperatorExample(
                before=valid.before,
                after=valid.after,
                goal=valid.goal,
                positive_binding=valid.positive_binding,
                candidate_bindings=valid.candidate_bindings,
                applicability_labels=(False, False),
                verified_primitives=valid.verified_primitives,
                allowed_schemas=valid.allowed_schemas,
                entity_candidates=valid.entity_candidates,
            )


class CandidateEvidenceFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = CandidateEvidenceFusion()

    def test_is_candidate_local_and_permutation_equivariant(self) -> None:
        initiation = torch.tensor((0.5, -0.5, 1.0))
        proposer = torch.tensor((1.5, 0.25, -1.0))
        join = torch.tensor((-0.1, -2.0, -0.5))
        mask = torch.tensor((True, True, False))
        original = self.fusion(initiation, proposer, join, mask)

        permutation = torch.tensor((1, 0, 2))
        permuted = self.fusion(
            initiation[permutation],
            proposer[permutation],
            join[permutation],
            mask[permutation],
        )
        self.assertTrue(torch.allclose(permuted, original[permutation]))

        extended = self.fusion(
            torch.cat((initiation, torch.tensor((10.0,)))),
            torch.cat((proposer, torch.tensor((-10.0,)))),
            torch.cat((join, torch.tensor((-100.0,)))),
            torch.tensor((True, True, False, True)),
        )
        self.assertTrue(torch.allclose(extended[:3], original))
        self.assertTrue(bool(torch.isneginf(original[-1]).item()))

    def test_enabled_sources_are_monotone_and_proposer_can_be_ablated(self) -> None:
        base = torch.zeros(2)
        mask = torch.ones(2, dtype=torch.bool)
        initial = self.fusion(base, base, base, mask)
        for channel in range(3):
            values = [base.clone(), base.clone(), base.clone()]
            values[channel][0] = 1.0
            increased = self.fusion(*values, mask)
            self.assertGreaterEqual(
                float(increased[0].detach()),
                float(initial[0].detach()),
            )

        without_first = self.fusion(
            base,
            torch.tensor((-100.0, 100.0)),
            base,
            mask,
            include_proposer=False,
        )
        without_second = self.fusion(
            base,
            torch.tensor((100.0, -100.0)),
            base,
            mask,
            include_proposer=False,
        )
        self.assertTrue(torch.equal(without_first, without_second))

    def test_gradients_are_finite_for_parameters_and_all_sources(self) -> None:
        initiation = torch.tensor((0.2, -0.4), requires_grad=True)
        proposer = torch.tensor((0.6, -0.1), requires_grad=True)
        join = torch.tensor((-0.2, -0.8), requires_grad=True)
        logits = self.fusion(
            initiation,
            proposer,
            join,
            torch.ones(2, dtype=torch.bool),
        )
        torch.nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor((0,)),
        ).backward()

        for source in (initiation, proposer, join):
            self.assertIsNotNone(source.grad)
            self.assertTrue(bool(torch.isfinite(source.grad).all().item()))
            self.assertTrue(bool((source.grad != 0).any().item()))
        for parameter in self.fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all().item()))


if __name__ == "__main__":
    unittest.main()
