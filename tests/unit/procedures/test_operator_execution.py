from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures.execution import (  # noqa: E402
    BindingAssignment,
    BindingConditionedOperatorHeads,
    BindingEncoder,
    OperatorBinding,
    PrimitiveStepScores,
    SharedPrimitiveSequenceDecoder,
    TypedEntityCandidate,
    canonicalize_binding_context,
    enumerate_operator_bindings,
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


NAMESPACE = "test.world"
_ZERO_DIGEST = "sha256:" + "0" * 64
_ONE_DIGEST = "sha256:" + "1" * 64


def _operator() -> tuple[LearnedOperator, ActionSchema]:
    entity_type = f"{NAMESPACE}.entity"
    variable = TypedVariable("entity", entity_type)
    inspect_schema = ActionSchema(
        f"{NAMESPACE}.inspect",
        (Parameter("entity", entity_type),),
        description="Inspect one entity.",
    )
    grounded = inspect_schema.ground("box-a")
    reconstruction = ReconstructionExemplar(
        namespace=NAMESPACE,
        start_records=(),
        variable_bindings=(("entity", "box-a"),),
        constant_values=(),
        actions=(grounded,),
        end_records=(Record(f"{NAMESPACE}.marked", ("box-a",)),),
    )
    operator = LearnedOperator(
        name=f"{NAMESPACE}.inspect_then_mark",
        namespace=NAMESPACE,
        variables=(variable,),
        preconditions=(),
        effects=(
            Effect(
                "add",
                RecordPattern(f"{NAMESPACE}.marked", (variable,)),
            ),
        ),
        body=(ActionPattern(inspect_schema, (variable,)),),
        exemplars=(
            OperatorExemplar(
                trace_digest=_ZERO_DIGEST,
                start_index=0,
                stop_index=1,
                before_state_digest=_ZERO_DIGEST,
                after_state_digest=_ONE_DIGEST,
                action_digests=(grounded.digest,),
                reconstruction=reconstruction,
            ),
        ),
    )
    return operator, inspect_schema


def _candidates() -> tuple[TypedEntityCandidate, ...]:
    return (
        TypedEntityCandidate("box-a", f"{NAMESPACE}.entity"),
        TypedEntityCandidate("box-b", f"{NAMESPACE}.entity"),
        TypedEntityCandidate("left", f"{NAMESPACE}.place"),
        TypedEntityCandidate("right", f"{NAMESPACE}.place"),
    )


class OperatorBindingTests(unittest.TestCase):
    def test_binding_is_complete_typed_canonical_and_digest_stable(self) -> None:
        operator, _ = _operator()
        variable = operator.variables[0]
        entity = _candidates()[0]
        binding = OperatorBinding(
            operator,
            (BindingAssignment(variable, entity),),
        )
        replay = OperatorBinding(
            operator,
            (BindingAssignment(variable, entity),),
        )

        self.assertEqual(binding.value_for("entity"), "box-a")
        self.assertEqual(binding.digest, replay.digest)
        self.assertTrue(binding.digest.startswith("sha256:"))

        with self.assertRaisesRegex(ValueError, "cover every"):
            OperatorBinding(operator, ())
        with self.assertRaisesRegex(ValueError, "type"):
            BindingAssignment(variable, _candidates()[2])

    def test_domain_neutral_enumeration_is_bounded_and_supports_unseen_values(self) -> None:
        operator, _ = _operator()
        bindings = enumerate_operator_bindings(
            operator,
            _candidates(),
            maximum_bindings=4,
        )

        self.assertEqual(len(bindings), 2)
        self.assertEqual(
            tuple(binding.value_for("entity") for binding in bindings),
            ("box-a", "box-b"),
        )
        fresh = TypedEntityCandidate("never-seen-before", f"{NAMESPACE}.entity")
        fresh_binding = enumerate_operator_bindings(operator, (fresh,))[0]
        self.assertEqual(fresh_binding.value_for("entity"), fresh.value)

        with self.assertRaisesRegex(ValueError, "maximum_bindings"):
            enumerate_operator_bindings(
                operator,
                _candidates(),
                maximum_bindings=1,
            )

    def test_binding_encoder_has_no_entity_or_operator_lookup_table(self) -> None:
        operator, _ = _operator()
        bindings = enumerate_operator_bindings(operator, _candidates())
        encoder = BindingEncoder(hash_width=64, hidden_width=40, output_width=24)
        encoded = encoder(bindings)

        self.assertEqual(encoded.shape, (2, 24))
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in encoder.modules()))

    def test_binding_conditioned_heads_distinguish_same_operator_bindings(self) -> None:
        operator, _ = _operator()
        bindings = enumerate_operator_bindings(operator, _candidates())
        core = NeuralOperatorCore(width=24, hidden_width=40, schema_hash_width=64)
        heads = BindingConditionedOperatorHeads(
            core,
            binding_hash_width=64,
            hidden_width=40,
        )
        states = torch.randn(1, 24)
        goals = torch.randn(1, 24)

        output = heads(states, goals, bindings)

        self.assertEqual(output.candidate_embeddings.shape, (2, 24))
        self.assertEqual(output.initiation_logits.shape, (1, 2))
        self.assertEqual(output.effect_embeddings.shape, (1, 2, 24))
        self.assertEqual(output.predecessor_embeddings.shape, (1, 2, 24))
        self.assertEqual(output.termination_logits.shape, (1, 2))
        self.assertEqual(output.proposer_logits.shape, (1, 2))
        self.assertFalse(
            torch.equal(
                output.candidate_embeddings[0],
                output.candidate_embeddings[1],
            )
        )
        loss = (
            output.initiation_logits.square().mean()
            + output.effect_embeddings.square().mean()
            + output.proposer_logits.square().mean()
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in heads.parameters()
                if parameter.requires_grad
            )
        )

    def test_binding_context_is_invariant_to_concrete_entity_renaming(self) -> None:
        operator, _ = _operator()
        bindings = enumerate_operator_bindings(operator, _candidates())
        first_state = State.from_records(
            NAMESPACE,
            (Record(f"{NAMESPACE}.ready", ("box-a",)),),
        )
        first_goal = Goal.from_records(
            NAMESPACE,
            (Record(f"{NAMESPACE}.marked", ("box-a",)),),
            exact=True,
        )
        second_state = State.from_records(
            NAMESPACE,
            (Record(f"{NAMESPACE}.ready", ("box-b",)),),
        )
        second_goal = Goal.from_records(
            NAMESPACE,
            (Record(f"{NAMESPACE}.marked", ("box-b",)),),
            exact=True,
        )

        first = canonicalize_binding_context(
            first_state,
            first_goal,
            bindings[0],
        )
        renamed = canonicalize_binding_context(
            second_state,
            second_goal,
            bindings[1],
        )
        mismatched = canonicalize_binding_context(
            first_state,
            first_goal,
            bindings[1],
        )

        self.assertEqual(first, renamed)
        self.assertNotEqual(first, mismatched)


class PrimitiveSequenceDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1702)
        torch.set_num_threads(1)
        self.operator, self.inspect = _operator()
        self.move = ActionSchema(
            f"{NAMESPACE}.move",
            (
                Parameter("entity", f"{NAMESPACE}.entity"),
                Parameter("destination", f"{NAMESPACE}.place"),
            ),
            description="Move one entity to one destination.",
        )
        self.schemas = (self.move, self.inspect)
        self.candidates = _candidates()
        self.bindings = enumerate_operator_bindings(
            self.operator,
            self.candidates,
        )
        self.decoder = SharedPrimitiveSequenceDecoder(
            width=24,
            hidden_width=40,
            hash_width=64,
            maximum_steps=6,
        )

    def test_entity_pointer_features_expose_selected_binding_roles(self) -> None:
        features = self.decoder._binding_entity_features(
            self.bindings,
            self.candidates,
        )

        self.assertEqual(features.shape, (2, 4, 24))
        # box-a occupies the learned role only in the first binding.
        self.assertFalse(torch.equal(features[0, 0], features[1, 0]))
        # Unbound place candidates retain identical structural features.
        self.assertTrue(torch.equal(features[0, 2], features[1, 2]))

    def test_decoder_scores_actions_stop_and_each_typed_argument_pointer(self) -> None:
        states = torch.randn(2, 24)
        goals = torch.randn(2, 24)
        scores = self.decoder(
            states,
            goals,
            self.bindings,
            self.schemas,
            self.candidates,
            torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertEqual(scores.action_logits.shape, (2, 3))
        self.assertEqual(scores.stop_index, 2)
        self.assertEqual(scores.argument_logits.shape, (2, 2, 2, 4))
        self.assertEqual(scores.argument_mask.shape, (2, 2, 4))
        # move(entity, destination)
        self.assertEqual(
            scores.argument_mask[0, 0].tolist(),
            [True, True, False, False],
        )
        self.assertEqual(
            scores.argument_mask[0, 1].tolist(),
            [False, False, True, True],
        )
        # inspect(entity) has no second argument row.
        self.assertEqual(
            scores.argument_mask[1, 0].tolist(),
            [True, True, False, False],
        )
        self.assertFalse(bool(scores.argument_mask[1, 1].any().item()))
        self.assertTrue(bool(torch.isfinite(scores.action_logits).all().item()))
        self.assertFalse(
            any(isinstance(module, nn.Embedding) for module in self.decoder.modules())
        )

        valid = scores.argument_logits.masked_select(
            scores.argument_mask.unsqueeze(0)
        )
        loss = scores.action_logits.square().mean() + valid.square().mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.decoder.parameters()
                if parameter.requires_grad
            )
        )

    def test_greedy_selection_emits_ground_action_or_explicit_stop_only(self) -> None:
        mask = self.decoder._argument_mask(
            self.schemas,
            self.candidates,
            torch.device("cpu"),
        )
        argument_logits = torch.full((1, 2, 2, 4), -torch.inf)
        argument_logits[0, 0, 0, 1] = 5.0  # entity box-b
        argument_logits[0, 0, 1, 3] = 7.0  # place right
        move_scores = PrimitiveStepScores(
            action_logits=torch.tensor([[9.0, 1.0, 0.0]]),
            argument_logits=argument_logits,
            argument_mask=mask,
            stop_index=2,
        )
        action = self.decoder.select_greedy(
            move_scores,
            self.schemas,
            self.candidates,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.schema, self.move)
        self.assertEqual(action.arguments, ("box-b", "right"))

        stop_scores = PrimitiveStepScores(
            action_logits=torch.tensor([[0.0, 1.0, 9.0]]),
            argument_logits=argument_logits,
            argument_mask=mask,
            stop_index=2,
        )
        self.assertIsNone(
            self.decoder.select_greedy(
                stop_scores,
                self.schemas,
                self.candidates,
            )
        )


if __name__ == "__main__":
    unittest.main()
