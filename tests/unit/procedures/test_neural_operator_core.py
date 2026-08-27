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

from angler.procedures.planner import (  # noqa: E402
    BidirectionalOperatorPlanner,
    PrimitiveChoice,
    SearchBudget,
)
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    Parameter,
    Record,
    State,
)
from angler.procedures.trunk import (  # noqa: E402
    FrozenHashTextEncoder,
    NeuralOperatorCore,
    PermutationInvariantSetEncoder,
    PrimitiveDecoder,
    RecordEncoder,
    SchemaEncoder,
    canonical_schema_text,
)


NAMESPACE = "test.world"


def _record(predicate: str, *arguments: str) -> Record:
    return Record(f"{NAMESPACE}.{predicate}", tuple(arguments))


def _state(*records: Record) -> State:
    return State.from_records(NAMESPACE, records)


def _goal(*records: Record, exact: bool = False) -> Goal:
    return Goal.from_records(NAMESPACE, records, exact=exact)


def _schemas() -> tuple[ActionSchema, ...]:
    entity = Parameter("entity", f"{NAMESPACE}.entity")
    place = Parameter("place", f"{NAMESPACE}.place")
    return (
        ActionSchema(
            f"{NAMESPACE}.move",
            (entity, place),
            description="Move one entity to a destination place.",
        ),
        ActionSchema(
            f"{NAMESPACE}.inspect",
            (entity,),
            description="Observe the current location of one entity.",
        ),
    )


class FrozenAndSetEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1401)
        torch.set_num_threads(1)

    def test_hash_encoder_is_deterministic_frozen_and_protocol_compatible(self) -> None:
        encoder = FrozenHashTextEncoder(64)
        records = (_record("at", "box", "left"), _record("clear", "right"))

        first = encoder.encode_records(records)
        replay = encoder.encode_records(records)
        changed = encoder.encode_records((records[1], records[0]))

        self.assertIsInstance(encoder, RecordEncoder)
        self.assertEqual(sum(parameter.numel() for parameter in encoder.parameters()), 0)
        self.assertTrue(torch.equal(first, replay))
        # The record encoder preserves rows; set invariance belongs to the
        # trainable set encoder rather than being hidden in hashing.
        self.assertTrue(torch.equal(first, changed.flip(0)))
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_hash_features_align_shared_atoms_across_serializations(self) -> None:
        encoder = FrozenHashTextEncoder(512)
        values = encoder.encode_texts(
            (
                "state entity box-a location left",
                "binding role entity value box-a",
                "unrelated alpha omega zeta",
            )
        )
        shared = torch.nn.functional.cosine_similarity(
            values[0].unsqueeze(0),
            values[1].unsqueeze(0),
        ).item()
        unrelated = torch.nn.functional.cosine_similarity(
            values[0].unsqueeze(0),
            values[2].unsqueeze(0),
        ).item()

        self.assertGreater(shared, unrelated)
        shared_overlap = bool(((values[0] != 0) & (values[1] != 0)).any().item())
        self.assertTrue(shared_overlap)

    def test_cached_hash_rows_are_not_exposed_for_mutation(self) -> None:
        encoder = FrozenHashTextEncoder(64)
        first = encoder.encode_texts(("stable structural feature",))
        expected = first.clone()

        first.fill_(99.0)
        replay = encoder.encode_texts(("stable structural feature",))

        self.assertTrue(torch.equal(replay, expected))

    def test_set_encoder_is_permutation_invariant_and_trainable(self) -> None:
        records = (
            _record("at", "box", "left"),
            _record("clear", "right"),
            _record("linked", "left", "right"),
        )
        encoder = PermutationInvariantSetEncoder(
            FrozenHashTextEncoder(48),
            hidden_width=40,
            output_width=32,
        )

        encoded = encoder.encode_record_sets(
            (records, tuple(reversed(records))),
            namespaces=(NAMESPACE, NAMESPACE),
        )

        self.assertTrue(torch.allclose(encoded[0], encoded[1], atol=1e-6))
        encoded.sum().backward()
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(gradients)
        self.assertTrue(any(gradient is not None for gradient in gradients))


class SchemaAndCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1402)
        torch.set_num_threads(1)

    def test_schema_features_include_description_and_exclude_provenance(self) -> None:
        first, _ = _schemas()
        changed_description = ActionSchema(
            first.name,
            first.parameters,
            description="Carry the entity into the named destination.",
        )
        encoder = SchemaEncoder(hash_width=96, hidden_width=48, output_width=32)
        embeddings = encoder((first, changed_description))

        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in encoder.modules()))

        class _CanonicalOperator:
            def __init__(self, trace: str) -> None:
                self.trace = trace

            def to_canonical(self) -> dict[str, object]:
                return {
                    "name": "test.world.transport",
                    "variables": [{"name": "x", "type_name": "test.world.entity"}],
                    "effects": [{"kind": "add", "record": "test.world.at"}],
                    "exemplars": [{"trace_digest": self.trace}],
                    "revision": 7,
                    "parent_digest": self.trace,
                }

        zeros = "sha256:" + "0" * 64
        ones = "sha256:" + "1" * 64
        self.assertEqual(
            canonical_schema_text(_CanonicalOperator(zeros)),
            canonical_schema_text(_CanonicalOperator(ones)),
        )

    def test_core_heads_have_expected_shapes_and_receive_gradients(self) -> None:
        core = NeuralOperatorCore(width=32, hidden_width=48, schema_hash_width=80)
        states = (
            _state(_record("at", "box", "left"), _record("clear", "right")),
            _state(_record("at", "box", "right"), _record("clear", "left")),
        )
        goals = (
            _goal(_record("at", "box", "right")),
            _goal(_record("at", "box", "left")),
        )
        output = core(states, goals, _schemas())

        self.assertEqual(output.state_embeddings.shape, (2, 32))
        self.assertEqual(output.goal_embeddings.shape, (2, 32))
        self.assertEqual(output.operator_embeddings.shape, (2, 32))
        self.assertEqual(output.initiation_logits.shape, (2, 2))
        self.assertEqual(output.effect_embeddings.shape, (2, 2, 32))
        self.assertEqual(output.predecessor_embeddings.shape, (2, 2, 32))
        self.assertEqual(output.termination_logits.shape, (2, 2))
        self.assertEqual(output.proposer_logits.shape, (2, 2))
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in core.modules()))

        loss = (
            output.initiation_logits.square().mean()
            + output.effect_embeddings.square().mean()
            + output.predecessor_embeddings.square().mean()
            + output.termination_logits.square().mean()
            + output.proposer_logits.square().mean()
        )
        loss.backward()
        for prefix in (
            "initiation_head",
            "effect_head",
            "termination_head",
            "proposer",
        ):
            gradients = [
                parameter.grad
                for name, parameter in core.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(any(gradient is not None for gradient in gradients), prefix)

    def test_primitive_decoder_pointer_scores_and_masks_candidates(self) -> None:
        decoder = PrimitiveDecoder(24)
        states = torch.randn(2, 24)
        goals = torch.randn(2, 24)
        operators = torch.randn(2, 24)
        actions = torch.randn(3, 24)
        arguments = torch.randn(3, 4, 24)
        action_mask = torch.tensor([True, False, True])
        argument_mask = torch.tensor(
            [
                [True, True, False, False],
                [False, False, False, False],
                [True, False, True, False],
            ]
        )

        scores = decoder(
            states,
            goals,
            operators,
            actions,
            arguments,
            action_mask=action_mask,
            argument_mask=argument_mask,
        )

        self.assertEqual(scores.action_logits.shape, (2, 3))
        self.assertEqual(scores.argument_logits.shape, (2, 3, 4))
        self.assertTrue(bool(torch.isneginf(scores.action_logits[:, 1]).all().item()))
        self.assertTrue(
            bool(torch.isneginf(scores.argument_logits[:, 0, 2:]).all().item())
        )
        self.assertTrue(
            bool(torch.isneginf(scores.argument_logits[:, 1]).all().item())
        )


class _LinearPlanningModel(nn.Module):
    """Tensor-only learned-boundary stand-in; domain logic is not in planner."""

    width = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _expand(operators: torch.Tensor, batch: int) -> torch.Tensor:
        if operators.ndim == 2:
            return operators.unsqueeze(0).expand(batch, -1, -1)
        return operators

    def proposer_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        operators: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidates = self._expand(operators, states.shape[0])
        remaining = (goals - states).abs().unsqueeze(1)
        return -(remaining - candidates.abs()).abs().squeeze(-1) + self.anchor * 0.0

    def initiation_logits(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        candidates = self._expand(operators, states.shape[0])
        return torch.full(
            candidates.shape[:2],
            8.0,
            device=states.device,
            dtype=states.dtype,
        ) + self.anchor * 0.0

    def predict_effects(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        candidates = self._expand(operators, states.shape[0])
        sign = -1.0 if reverse else 1.0
        return states.unsqueeze(1) + sign * candidates + self.anchor * 0.0

    def termination_logits(
        self,
        candidate_states: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        distance = (candidate_states - goals.unsqueeze(1)).abs().sum(dim=-1)
        return 10.0 - distance * 20.0 + self.anchor * 0.0


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.model = _LinearPlanningModel()
        self.planner = BidirectionalOperatorPlanner(self.model)
        self.origin = torch.tensor([0.0])
        self.goal = torch.tensor([4.0])
        self.operators = torch.tensor([[1.0], [2.0]])

    def test_bidirectional_planner_uses_proposer_and_accounts_exactly(self) -> None:
        # A threshold of one disables the learned-termination shortcut so this
        # test specifically exercises a forward/backward frontier join.
        class _Decoder:
            def decode(self, **values: object) -> tuple[PrimitiveChoice, ...]:
                return (
                    PrimitiveChoice(
                        action_identity=f"primitive:{values['operator_identity']}",
                        arguments=("fresh-binding",),
                        score=0.75,
                    ),
                )

        plan = self.planner.plan_embeddings(
            self.origin,
            self.goal,
            self.operators,
            SearchBudget(
                maximum_expansions=4,
                maximum_depth=4,
                proposals_per_state=1,
                join_tolerance=0.0,
                termination_probability=1.0,
            ),
            operator_identities=("step-one", "step-two"),
            primitive_decoder=_Decoder(),
        )

        self.assertTrue(plan.found, plan)
        self.assertEqual(plan.reason, "latent_frontier_join")
        self.assertEqual(plan.operator_indices, (1, 1))
        self.assertEqual(plan.operator_identities, ("step-two", "step-two"))
        self.assertTrue(plan.is_primitive_bound)
        self.assertEqual(
            tuple(choice[0].arguments for choice in plan.primitive_choices or ()),
            (("fresh-binding",), ("fresh-binding",)),
        )
        self.assertEqual(plan.forward_depth, 1)
        self.assertEqual(plan.backward_depth, 1)
        self.assertEqual(plan.accounting.forward_expansions, 1)
        self.assertEqual(plan.accounting.backward_expansions, 1)
        self.assertEqual(plan.accounting.total_expansions, 2)
        self.assertEqual(plan.accounting.proposer_calls, 2)
        self.assertEqual(plan.accounting.operator_scores, 4)
        self.assertTrue(self.model.training)

    def test_expansion_budget_is_hard_and_planner_does_not_update_model(self) -> None:
        before = self.model.anchor.detach().clone()
        plan = self.planner.plan_embeddings(
            self.origin,
            self.goal,
            self.operators,
            SearchBudget(
                maximum_expansions=1,
                maximum_depth=4,
                proposals_per_state=1,
                join_tolerance=0.0,
                termination_probability=1.0,
            ),
        )

        self.assertFalse(plan.found)
        self.assertEqual(plan.reason, "expansion_budget_exhausted")
        self.assertEqual(plan.accounting.total_expansions, 1)
        self.assertTrue(torch.equal(self.model.anchor.detach(), before))
        self.assertIsNone(self.model.anchor.grad)

    def test_high_level_bidirectional_planning_rejects_partial_goal(self) -> None:
        partial = _goal(_record("at", "box", "right"), exact=False)
        with self.assertRaisesRegex(ValueError, "exact Goal"):
            self.planner.plan(
                _state(_record("at", "box", "left")),
                partial,
                _schemas(),
                SearchBudget(4, 3, 1),
            )


if __name__ == "__main__":
    unittest.main()
