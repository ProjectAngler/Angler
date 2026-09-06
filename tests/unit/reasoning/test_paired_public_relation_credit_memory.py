from __future__ import annotations

from dataclasses import replace
import inspect
import json
import unittest

import torch

from angler.reasoning.paired_public_relation_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    RELATION_WIDTH,
    PairedPublicRelationCreditEvent,
    PairedPublicRelationCreditMemoryCore,
    PairedPublicRelationCreditState,
)
from angler.reasoning.shared_semantic_metric_credit_memory import (
    SharedSemanticMetricCreditMemoryCore,
)
from angler.reasoning.strict_key_value_credit_memory import StrictKeyValueCreditState


def _observation(seed: int, *, batch: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, RELATION_WIDTH, generator=generator),
        torch.randn(batch, 4, generator=generator),
        torch.randn(batch, generator=generator),
    )


def _write(
    core: PairedPublicRelationCreditMemoryCore,
    state: PairedPublicRelationCreditState,
    seed: int,
    outcome: float,
    *,
    detach_state: bool = True,
    pair_residual_enabled: bool | None = None,
) -> tuple[PairedPublicRelationCreditState, PairedPublicRelationCreditEvent]:
    output = core.predict(
        *_observation(seed),
        state=state,
        pair_residual_enabled=pair_residual_enabled,
    )
    return core.apply_feedback(
        state,
        output,
        torch.tensor([outcome]),
        evidence_refs=f"evidence:{seed}",
        detach_state=detach_state,
    )


class PairedPublicRelationCreditMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026083119)
        self.core = PairedPublicRelationCreditMemoryCore(temporal_width=4)

    def _occupied_state(
        self, *, count: int = 4, detach_state: bool = True
    ) -> PairedPublicRelationCreditState:
        state = self.core.initial_state()
        for index in range(count):
            state, _ = _write(
                self.core,
                state,
                100 + index,
                1.0 if index % 2 else -1.0,
                detach_state=detach_state,
            )
        return state

    def test_exact_architecture_state_shape_and_parameter_ownership(self) -> None:
        state = self.core.initial_state()
        self.assertEqual(state.keys.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.values.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.public_relations.shape, (MEMORY_SLOTS, RELATION_WIDTH))
        self.assertTrue(torch.equal(state.public_relations, torch.zeros_like(state.public_relations)))
        expected_bytes = (
            MEMORY_SLOTS * MEMORY_RANK * 2
            + MEMORY_SLOTS * 2
            + MEMORY_SLOTS * RELATION_WIDTH
        ) * 4
        self.assertEqual(state.bytes, expected_bytes)

        scorer = self.core.paired_relation_scorer
        self.assertIsInstance(scorer[0], torch.nn.LayerNorm)
        self.assertEqual(scorer[0].normalized_shape, (RELATION_WIDTH * 3,))
        self.assertEqual((scorer[1].in_features, scorer[1].out_features), (192, 64))
        self.assertIsInstance(scorer[2], torch.nn.SiLU)
        self.assertEqual((scorer[3].in_features, scorer[3].out_features), (64, 1))
        self.assertIsNone(scorer[3].bias)
        self.assertTrue(torch.equal(scorer[3].weight, torch.zeros_like(scorer[3].weight)))

        groups = self.core.parameter_groups()
        self.assertIn("paired_public_relation", groups)
        owned = tuple(name for names in groups.values() for name in names)
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), {name for name, _ in self.core.named_parameters()})
        self.assertEqual(tuple(self.core.named_buffers()), ())
        report = self.core.architecture_report()
        self.assertEqual(report["public_sidecar_write"], "exact_detached_winner_overwrite")
        self.assertFalse(report["outcome_affects_address_or_sidecar"])
        self.assertEqual(report["semantic_metric_shared_for"], ("read_query", "write_key"))
        self.assertEqual(report["semantic_metric_inputs"], ("detached_relation",))
        self.assertFalse(report["independent_query_or_key_networks"])
        self.assertFalse(report["temporal_affects_semantic_code_or_decoder"])
        self.assertFalse(report["outcome_affects_semantic_code_or_gates"])
        self.assertEqual(report["residual_decoder_inputs"], ("retrieved_value",))
        self.assertFalse(report["metadata_inputs"])
        self.assertEqual(
            tuple(inspect.signature(self.core.predict).parameters)[-2:],
            ("pair_residual_enabled", "pair_query_features"),
        )

    def test_feedback_exact_overwrites_sidecar_independent_of_outcome_and_gates(self) -> None:
        state = self.core.initial_state()
        relation, temporal, base = _observation(1)
        output = self.core.predict(relation, temporal, base, state=state)
        low = replace(
            output,
            write_strengths=torch.full_like(output.write_strengths, 0.1),
            erase_gates=torch.full_like(output.erase_gates, 0.2),
        )
        high = replace(
            output,
            write_strengths=torch.full_like(output.write_strengths, 0.9),
            erase_gates=torch.full_like(output.erase_gates, 0.8),
        )
        positive, positive_event = self.core.apply_feedback(
            state, low, torch.ones(1), evidence_refs="positive"
        )
        negative, negative_event = self.core.apply_feedback(
            state, high, -torch.ones(1), evidence_refs="negative"
        )
        self.assertTrue(torch.equal(positive.public_relations[0], relation[0]))
        self.assertTrue(torch.equal(negative.public_relations[0], relation[0]))
        self.assertTrue(torch.equal(positive_event.public_relation, relation))
        self.assertTrue(torch.equal(positive_event.public_relation, negative_event.public_relation))
        self.assertFalse(torch.equal(positive.values, negative.values))
        self.assertTrue(torch.equal(positive.public_relations[1:], torch.zeros_like(positive.public_relations[1:])))
        self.assertFalse(positive.public_relations.requires_grad)

    def test_pair_scorer_is_symmetric_and_sees_only_public_pair(self) -> None:
        state = self._occupied_state(count=3)
        query = _observation(20, batch=2)[0]
        raw, residual = self.core.paired_address_residuals(query, state)
        self.assertEqual(raw.shape, (2, MEMORY_SLOTS))
        self.assertEqual(residual.shape, (2, MEMORY_SLOTS))
        self.assertTrue(torch.equal(residual[:, 3:], torch.zeros_like(residual[:, 3:])))

        # Swap query/stored roles for one occupied row; the declared features
        # are symmetric, so the scorer result must be exact.
        stored = state.public_relations[1:2]
        forward_raw, _ = self.core.paired_address_residuals(query[:1], state)
        swapped_state = replace(
            state,
            public_relations=torch.cat(
                (
                    query[:1],
                    state.public_relations[1:],
                ),
                dim=0,
            ),
        )
        swapped_raw, _ = self.core.paired_address_residuals(stored, swapped_state)
        self.assertTrue(torch.equal(forward_raw[:, 1], swapped_raw[:, 0]))

        changed_nontarget = replace(
            state,
            keys=torch.flip(state.keys, dims=(0,)),
            values=-state.values,
            usage=torch.where(state.usage > 0, torch.full_like(state.usage, 0.75), state.usage),
            acquisition=torch.flip(state.acquisition, dims=(0,)),
        )
        changed_raw, changed_residual = self.core.paired_address_residuals(query, changed_nontarget)
        self.assertTrue(torch.equal(raw, changed_raw))
        self.assertTrue(torch.equal(residual, changed_residual))

    def test_residual_disabled_is_exact_v17_and_pair_query_is_isolated(self) -> None:
        torch.manual_seed(77)
        v17 = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        torch.manual_seed(77)
        v19 = PairedPublicRelationCreditMemoryCore(
            temporal_width=4, pair_residual_enabled_by_default=False
        )
        common = v17.state_dict()
        missing, unexpected = v19.load_state_dict(common, strict=False)
        self.assertTrue(all(name.startswith("paired_relation_scorer.") for name in missing))
        self.assertEqual(unexpected, [])

        state19 = v19.initial_state()
        for index in range(3):
            output19 = v19.predict(*_observation(30 + index), state=state19)
            state19, _ = v19.apply_feedback(
                state19,
                output19,
                torch.tensor([1.0 if index % 2 else -1.0]),
                evidence_refs=f"v19:{index}",
            )
        state17 = StrictKeyValueCreditState(
            state19.keys,
            state19.values,
            state19.usage,
            state19.acquisition,
            state19.step,
        )
        relation, temporal, base = _observation(40, batch=2)
        expected = v17.predict(relation, temporal, base, state=state17)
        actual = v19.predict(relation, temporal, base, state=state19)
        for name in (
            "logits",
            "residuals",
            "read_queries",
            "read_weights",
            "read_values",
            "write_keys",
            "read_strengths",
            "write_strengths",
            "erase_gates",
        ):
            self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)), name)
        self.assertTrue(torch.equal(actual.pair_address_residuals, torch.zeros_like(actual.pair_address_residuals)))

        changed_pair = torch.flip(relation, dims=(0,))
        isolated = v19.predict(
            relation,
            temporal,
            base,
            state=state19,
            pair_query_features=changed_pair,
        )
        self.assertTrue(torch.equal(isolated.logits, actual.logits))
        self.assertTrue(torch.equal(isolated.read_weights, actual.read_weights))
        self.assertTrue(torch.equal(isolated.pair_query_relations, changed_pair))

    def test_joint_slot_permutation_is_equivariant(self) -> None:
        # Open the zero-initialized head without selecting a deterministic slot.
        with torch.no_grad():
            self.core.paired_relation_scorer[-1].weight.copy_(
                torch.linspace(-0.1, 0.1, RELATION_WIDTH).unsqueeze(0)
            )
        state = self._occupied_state(count=5)
        relation, temporal, base = _observation(50, batch=3)
        original = self.core.predict(relation, temporal, base, state=state)
        permutation = torch.randperm(MEMORY_SLOTS, generator=torch.Generator().manual_seed(51))
        permuted = PairedPublicRelationCreditState(
            state.keys[permutation],
            state.values[permutation],
            state.usage[permutation],
            state.acquisition[permutation],
            state.public_relations[permutation],
            state.step,
        )
        moved = self.core.predict(relation, temporal, base, state=permuted)
        self.assertTrue(torch.allclose(moved.logits, original.logits, rtol=0.0, atol=1.0e-6))
        self.assertTrue(
            torch.allclose(
                moved.read_weights,
                original.read_weights[:, permutation],
                rtol=0.0,
                atol=1.0e-6,
            )
        )

    def test_derangement_changes_only_sidecar_and_has_no_fixed_rows(self) -> None:
        state = self._occupied_state(count=4)
        before_non_target = self.core.state_digest_without_public_relations(state)
        deranged = self.core.deranged_public_sidecar_state(state)
        self.assertEqual(
            self.core.state_digest_without_public_relations(deranged), before_non_target
        )
        self.assertNotEqual(self.core.state_digest(deranged), self.core.state_digest(state))
        occupied = state.usage.gt(0)
        self.assertTrue(
            (deranged.public_relations[occupied] != state.public_relations[occupied])
            .any(dim=-1)
            .all()
        )
        self.assertTrue(torch.equal(deranged.public_relations[4:], state.public_relations[4:]))

    def test_capture_restore_zero_json_replay_and_row_are_exact(self) -> None:
        state = self.core.initial_state()
        events = []
        logits = []
        for index in range(5):
            output = self.core.predict(*_observation(60 + index), state=state)
            self.assertTrue(torch.equal(output.row(0).pair_query_relations, output.pair_query_relations))
            logits.append(output.logits.detach().clone())
            state, event = self.core.apply_feedback(
                state,
                output,
                torch.tensor([1.0 if index % 2 else -1.0]),
                evidence_refs=f"event:{index}",
            )
            events.append(PairedPublicRelationCreditEvent.from_record(json.loads(json.dumps(event.to_record()))))
        digest = self.core.state_digest(state)
        restored = self.core.restore_state(self.core.capture_state(state))
        self.assertEqual(self.core.state_digest(restored), digest)
        replayed, outputs = self.core.replay(events)
        self.assertEqual(self.core.state_digest(replayed), digest)
        for expected, actual in zip(logits, outputs, strict=True):
            self.assertTrue(torch.equal(expected, actual.logits))
        zero = self.core.zero_state_like(state)
        self.assertEqual(zero.step, 0)
        self.assertTrue(torch.equal(zero.public_relations, torch.zeros_like(zero.public_relations)))
        self.assertNotEqual(self.core.state_digest(zero), digest)

    def test_pair_inputs_are_detached_and_malformed_masks_fail_closed(self) -> None:
        state = self._occupied_state(count=2)
        relation, temporal, base = _observation(70)
        pair = torch.randn_like(relation, requires_grad=True)
        relation.requires_grad_()
        temporal.requires_grad_()
        base.requires_grad_()
        output = self.core.predict(
            relation,
            temporal,
            base,
            state=state,
            pair_query_features=pair,
        )
        loss = output.pair_address_residuals.square().sum() + output.residuals.square().sum()
        observed = torch.autograd.grad(loss, (relation, temporal, base, pair), allow_unused=True)
        self.assertEqual(observed, (None, None, None, None))

        invalid_sidecar = state.public_relations.clone()
        invalid_sidecar[2] = 1.0
        invalid = replace(state, public_relations=invalid_sidecar)
        with self.assertRaisesRegex(ValueError, "unused.*exact zero"):
            self.core.validate_state(invalid)
        differentiable_source = torch.zeros_like(state.public_relations, requires_grad=True)
        differentiable = replace(
            state,
            public_relations=differentiable_source + 0.0,
        )
        self.assertTrue(differentiable.public_relations.requires_grad)
        self.assertIsNotNone(differentiable.public_relations.grad_fn)
        with self.assertRaisesRegex(ValueError, "sidecar must be detached"):
            self.core.validate_state(differentiable)
        mismatched_step = replace(state, step=state.step + 1)
        with self.assertRaisesRegex(ValueError, "occupied slot count must equal step"):
            self.core.validate_state(mismatched_step)
        with self.assertRaises(ValueError):
            self.core.predict(
                relation.detach(),
                temporal.detach(),
                base.detach(),
                state=state,
                pair_query_features=torch.full_like(pair.detach(), float("nan")),
            )

    def test_replay_rejects_v19_pair_query_mode_and_sidecar_mutations(self) -> None:
        state = self.core.initial_state()
        output = self.core.predict(*_observation(75), state=state)
        _, event = self.core.apply_feedback(
            state,
            output,
            torch.ones(1),
            evidence_refs="event:75",
        )
        changed_query = replace(
            event,
            pair_query_features=event.pair_query_features + 0.25,
        )
        with self.assertRaisesRegex(RuntimeError, "replay diverged"):
            self.core.replay((changed_query,))
        changed_mode = replace(
            event,
            pair_residual_enabled=not event.pair_residual_enabled,
        )
        with self.assertRaisesRegex(RuntimeError, "replay diverged"):
            self.core.replay((changed_mode,))
        inconsistent_sidecar = replace(
            event,
            public_relation=event.public_relation + 0.5,
        )
        with self.assertRaisesRegex(ValueError, "sidecar must equal"):
            self.core.replay((inconsistent_sidecar,))

    def test_full_gradient_opens_head_then_all_scorer_parameters_and_unary_bypasses(self) -> None:
        state = self._occupied_state(count=6)
        optimizer = torch.optim.SGD(self.core.parameters(), lr=0.05)
        scorer_names = tuple(
            name for name, _ in self.core.named_parameters() if name.startswith("paired_relation_scorer.")
        )
        self.assertGreater(len(scorer_names), 1)

        first = self.core.predict(*_observation(80), state=state)
        first_loss = self.core.outcome_loss(first, torch.ones(1))
        optimizer.zero_grad(set_to_none=True)
        first_loss.backward()
        gradients = dict(self.core.named_parameters())
        self.assertGreater(float(gradients["paired_relation_scorer.3.weight"].grad.abs().sum()), 0.0)
        optimizer.step()

        second = self.core.predict(*_observation(81), state=state)
        second_loss = self.core.outcome_loss(second, -torch.ones(1))
        optimizer.zero_grad(set_to_none=True)
        second_loss.backward()
        parameters = dict(self.core.named_parameters())
        for name in scorer_names:
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

        unary = PairedPublicRelationCreditMemoryCore(
            temporal_width=4, pair_residual_enabled_by_default=False
        )
        unary.load_state_dict(self.core.state_dict())
        unary_state = unary.restore_state(self.core.capture_state(state))
        unary_output = unary.predict(*_observation(82), state=unary_state)
        unary.outcome_loss(unary_output, torch.ones(1)).backward()
        for name, parameter in unary.named_parameters():
            if name.startswith("paired_relation_scorer."):
                self.assertIsNone(parameter.grad, name)


if __name__ == "__main__":
    unittest.main()
