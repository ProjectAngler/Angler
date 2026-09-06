from __future__ import annotations

import json
import unittest

import torch

from angler.reasoning.shared_semantic_metric_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    SharedSemanticMetricCreditEvent,
    SharedSemanticMetricCreditMemoryCore,
    SharedSemanticMetricCreditOutput,
    StrictKeyValueCreditMemoryCore,
)


def _observation(seed: int, *, batch: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    relation = torch.randn(batch, 64, generator=generator)
    temporal = torch.randn(batch, 4, generator=generator)
    base = torch.randn(batch, generator=generator)
    return relation, temporal, base


def _write(
    core: SharedSemanticMetricCreditMemoryCore,
    state: object,
    seed: int,
    outcome: float,
    *,
    detach_state: bool = True,
) -> tuple[object, SharedSemanticMetricCreditEvent]:
    output = core.predict(*_observation(seed), state=state)
    return core.apply_feedback(
        state,
        output,
        torch.tensor([outcome]),
        evidence_refs=f"semantic:{seed}",
        detach_state=detach_state,
    )


class SharedSemanticMetricCreditMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026083117)
        self.core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)

    def test_is_a_small_v16_successor_with_identical_state_contract(self) -> None:
        self.assertIsInstance(self.core, StrictKeyValueCreditMemoryCore)
        state = self.core.initial_state()
        self.assertEqual(state.keys.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.values.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.step, 0)
        output = self.core.predict(*_observation(1), state=state)
        self.assertIsInstance(output, SharedSemanticMetricCreditOutput)
        self.assertEqual(output.read_weights.shape, (1, MEMORY_SLOTS))

    def test_architecture_enumerates_one_shared_relation_only_metric(self) -> None:
        report = self.core.architecture_report()
        self.assertEqual(report["semantic_metric_inputs"], ("detached_relation",))
        self.assertEqual(
            report["semantic_metric_shared_for"],
            ("read_query", "write_key"),
        )
        self.assertTrue(report["semantic_metric_normalized"])
        self.assertFalse(report["independent_query_or_key_networks"])
        self.assertFalse(report["temporal_affects_semantic_code_or_decoder"])
        self.assertEqual(
            report["public_temporal_roles"],
            ("read_strength", "write_strength", "inactive_fresh_slot_erase"),
        )
        self.assertEqual(report["state_coordinate_roles"], ("slot_recency",))
        self.assertEqual(report["stored_value_inputs"], ("transformed_outcome_embedding",))
        self.assertEqual(report["residual_decoder_inputs"], ("retrieved_value",))
        names = {name for name, _ in self.core.named_parameters()}
        self.assertTrue(any(name.startswith("semantic_metric_network.") for name in names))
        self.assertFalse(any(name.startswith("query_network.") for name in names))
        self.assertFalse(any(name.startswith("key_network.") for name in names))
        groups = report["parameter_groups"]
        owned = [name for values in groups.values() for name in values]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), names)

    def test_same_relation_has_exact_shared_code_independent_of_time(self) -> None:
        relation, temporal, base = _observation(2, batch=4)
        shifted_temporal = temporal.flip(0) + 10.0
        left = self.core.predict(relation, temporal, base)
        right = self.core.predict(relation, shifted_temporal, base)
        self.assertTrue(torch.equal(left.read_queries, left.write_keys))
        self.assertEqual(left.read_queries.data_ptr(), left.write_keys.data_ptr())
        self.assertTrue(torch.equal(left.read_queries, right.read_queries))
        norms = left.read_queries.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=0.0))
        direct = self.core.semantic_codes(relation)
        self.assertTrue(torch.equal(direct, left.read_queries))

    def test_different_relations_change_semantic_code_without_metadata(self) -> None:
        left, _, _ = _observation(3)
        right, _, _ = _observation(4)
        left_code = self.core.semantic_codes(left)
        right_code = self.core.semantic_codes(right)
        self.assertFalse(torch.equal(left_code, right_code))
        self.assertFalse(self.core.architecture_report()["metadata_inputs"])

    def test_outcome_still_changes_only_strict_value_content(self) -> None:
        state = self.core.initial_state()
        output = self.core.predict(*_observation(5), state=state)
        positive, pos_event = self.core.apply_feedback(
            state, output, torch.ones(1), evidence_refs="positive"
        )
        negative, neg_event = self.core.apply_feedback(
            state, output, -torch.ones(1), evidence_refs="negative"
        )
        self.assertTrue(torch.equal(positive.keys, negative.keys))
        self.assertTrue(torch.equal(positive.usage, negative.usage))
        self.assertTrue(torch.equal(positive.acquisition, negative.acquisition))
        self.assertEqual(pos_event.allocation_index, neg_event.allocation_index)
        self.assertEqual(pos_event.read_strength, neg_event.read_strength)
        self.assertEqual(pos_event.write_strength, neg_event.write_strength)
        self.assertEqual(pos_event.erase_gate, neg_event.erase_gate)
        self.assertTrue(torch.equal(pos_event.write_key, neg_event.write_key))
        self.assertFalse(torch.equal(pos_event.write_value, neg_event.write_value))
        self.assertFalse(torch.equal(positive.values, negative.values))

    def test_downstream_credit_reaches_shared_metric_and_strict_paths(self) -> None:
        state = self.core.initial_state()
        state, _ = _write(self.core, state, 10, 1.0, detach_state=False)
        state, _ = _write(self.core, state, 11, -1.0, detach_state=False)
        relation, temporal, base = _observation(10)
        output = self.core.predict(relation + 0.01, temporal, base, state=state)
        loss = self.core.outcome_loss(output, torch.ones(1))
        parameters = dict(self.core.named_parameters())
        required = (
            "semantic_metric_network.1.weight",
            "semantic_metric_network.3.weight",
            "temporal_network.1.weight",
            "read_strength_network.weight",
            "write_strength_network.weight",
            "slot_temporal_network.0.weight",
            "outcome_embedding.weight",
            "outcome_value_network.1.weight",
            "residual_network.0.weight",
        )
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters[name] for name in required),
            allow_unused=True,
        )
        for name, gradient in zip(required, gradients, strict=True):
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_observations_are_detached_and_zero_read_remains_exact(self) -> None:
        state = self.core.initial_state()
        state, _ = _write(self.core, state, 12, 1.0, detach_state=False)
        state, _ = _write(self.core, state, 13, -1.0, detach_state=False)
        relation, temporal, base = _observation(14)
        relation.requires_grad_()
        temporal.requires_grad_()
        base.requires_grad_()
        output = self.core.predict(relation, temporal, base, state=state)
        observed = torch.autograd.grad(
            output.residuals.square().sum(),
            (relation, temporal, base),
            allow_unused=True,
        )
        self.assertEqual(observed, (None, None, None))
        disabled = self.core.predict(
            relation, temporal, base, state=state, read_enabled=False
        )
        self.assertTrue(torch.equal(disabled.residuals, torch.zeros_like(disabled.residuals)))
        self.assertTrue(torch.equal(disabled.logits, base.detach()))

    def test_inherited_snapshot_digest_json_replay_and_bounds_are_exact(self) -> None:
        state = self.core.initial_state()
        events = []
        expected_logits = []
        for index in range(8):
            output = self.core.predict(*_observation(100 + index), state=state)
            expected_logits.append(output.logits.detach().clone())
            state, event = self.core.apply_feedback(
                state,
                output,
                torch.tensor([1.0 if index % 2 else -1.0]),
                evidence_refs=f"event:{index}",
            )
            events.append(
                SharedSemanticMetricCreditEvent.from_record(
                    json.loads(json.dumps(event.to_record()))
                )
            )
        digest = self.core.state_digest(state)
        restored = self.core.restore_state(self.core.capture_state(state))
        replayed, outputs = self.core.replay(events)
        self.assertEqual(self.core.state_digest(restored), digest)
        self.assertEqual(self.core.state_digest(replayed), digest)
        self.assertEqual([event.allocation_index for event in events], list(range(8)))
        for expected, actual in zip(expected_logits, outputs, strict=True):
            self.assertTrue(torch.equal(expected, actual.logits))
        self.assertLessEqual(float(state.keys.abs().max()), 1.0)
        self.assertLessEqual(float(state.values.abs().max()), 1.0)
        self.assertTrue(bool(((state.usage >= 0) & (state.usage <= 1)).all()))
        self.assertTrue(bool(((state.acquisition >= 0) & (state.acquisition <= 1)).all()))


if __name__ == "__main__":
    unittest.main()
