from __future__ import annotations

import inspect
import json
import unittest

import torch

from angler.reasoning.strict_key_value_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    StrictKeyValueCreditEvent,
    StrictKeyValueCreditMemoryCore,
    StrictKeyValueCreditState,
)


def _observation(seed: int, *, batch: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    relation = torch.randn(batch, 64, generator=generator)
    temporal = torch.randn(batch, 4, generator=generator)
    base = torch.randn(batch, generator=generator)
    return relation, temporal, base


def _write(
    core: StrictKeyValueCreditMemoryCore,
    state: StrictKeyValueCreditState,
    seed: int,
    outcome: float,
    *,
    detach_state: bool = True,
) -> tuple[StrictKeyValueCreditState, StrictKeyValueCreditEvent]:
    output = core.predict(*_observation(seed), state=state)
    return core.apply_feedback(
        state,
        output,
        torch.tensor([outcome]),
        evidence_refs=f"evidence:{seed}",
        detach_state=detach_state,
    )


class StrictKeyValueCreditMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026083116)
        self.core = StrictKeyValueCreditMemoryCore(temporal_width=4)

    def test_state_shape_bounds_and_strict_architecture_enumeration(self) -> None:
        state = self.core.initial_state()
        self.assertEqual(state.keys.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.values.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.usage.shape, (MEMORY_SLOTS,))
        self.assertEqual(state.acquisition.shape, (MEMORY_SLOTS,))
        self.assertEqual(state.step, 0)
        self.assertEqual(state.bytes, (MEMORY_SLOTS * MEMORY_RANK * 2 + MEMORY_SLOTS * 2) * 4)

        report = self.core.architecture_report()
        self.assertEqual(report["stored_value_inputs"], ("transformed_outcome_embedding",))
        self.assertEqual(report["residual_decoder_inputs"], ("retrieved_value",))
        self.assertFalse(report["outcome_affects_address_or_gates"])
        self.assertFalse(report["query_affects_value_content_or_decoder"])
        self.assertEqual(
            report["erase_gate_role"],
            "fresh_slot_noop_under_stable_least_unused",
        )
        self.assertFalse(report["erase_gate_active_before_capacity"])
        self.assertFalse(report["learned_retention_evidence_claimed"])
        groups = report["parameter_groups"]
        owned = [name for names in groups.values() for name in names]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), {name for name, _ in self.core.named_parameters()})
        self.assertEqual(tuple(inspect.signature(self.core.decode_retrieved_values).parameters), ("read_values",))
        self.assertEqual(self.core.residual_network[0].in_features, MEMORY_RANK)
        self.assertIsNone(self.core.residual_network[0].bias)
        self.assertIsNone(self.core.residual_network[-1].bias)

    def test_zero_or_disabled_state_is_exact_zero_residual(self) -> None:
        relation, temporal, base = _observation(1, batch=3)
        output = self.core.predict(relation, temporal, base)
        self.assertTrue(torch.equal(output.read_weights, torch.zeros_like(output.read_weights)))
        self.assertTrue(torch.equal(output.read_values, torch.zeros_like(output.read_values)))
        self.assertTrue(torch.equal(output.residuals, torch.zeros_like(output.residuals)))
        self.assertTrue(torch.equal(output.logits, base))

        state, _ = _write(self.core, self.core.initial_state(), 2, 1.0)
        disabled = self.core.predict(relation, temporal, base, state=state, read_enabled=False)
        self.assertTrue(torch.equal(disabled.residuals, torch.zeros_like(disabled.residuals)))
        self.assertTrue(torch.equal(disabled.logits, base))
        decoded = self.core.decode_retrieved_values(torch.zeros(3, MEMORY_RANK))
        self.assertTrue(torch.equal(decoded, torch.zeros_like(decoded)))

    def test_outcome_changes_only_value_content_not_address_or_gates(self) -> None:
        state = self.core.initial_state()
        output = self.core.predict(*_observation(3), state=state)
        positive, positive_event = self.core.apply_feedback(
            state, output, torch.ones(1), evidence_refs="positive"
        )
        negative, negative_event = self.core.apply_feedback(
            state, output, -torch.ones(1), evidence_refs="negative"
        )
        self.assertTrue(torch.equal(positive.keys, negative.keys))
        self.assertTrue(torch.equal(positive.usage, negative.usage))
        self.assertTrue(torch.equal(positive.acquisition, negative.acquisition))
        self.assertEqual(positive_event.allocation_index, negative_event.allocation_index)
        self.assertEqual(positive_event.read_strength, negative_event.read_strength)
        self.assertEqual(positive_event.write_strength, negative_event.write_strength)
        self.assertEqual(positive_event.erase_gate, negative_event.erase_gate)
        self.assertTrue(torch.equal(positive_event.write_key, negative_event.write_key))
        self.assertFalse(torch.equal(positive_event.write_value, negative_event.write_value))
        self.assertFalse(torch.equal(positive.values, negative.values))

    def test_identical_outcomes_have_identical_prestrength_value_content(self) -> None:
        left_output = self.core.predict(*_observation(4))
        right_output = self.core.predict(*_observation(5))
        _, left = self.core.apply_feedback(
            self.core.initial_state(), left_output, torch.ones(1), evidence_refs="left"
        )
        _, right = self.core.apply_feedback(
            self.core.initial_state(), right_output, torch.ones(1), evidence_refs="right"
        )
        self.assertTrue(torch.equal(left.write_value, right.write_value))
        self.assertFalse(torch.equal(left.write_key, right.write_key))
        repeated = self.core.outcome_values(torch.tensor([1.0, 1.0]))
        self.assertTrue(torch.equal(repeated[0], repeated[1]))

    def test_postwrite_loss_reaches_address_outcome_value_and_readout(self) -> None:
        state = self.core.initial_state()
        state, _ = _write(self.core, state, 10, 1.0, detach_state=False)
        state, _ = _write(self.core, state, 11, -1.0, detach_state=False)
        relation, temporal, base = _observation(10)
        output = self.core.predict(relation + 0.01, temporal, base, state=state)
        loss = self.core.outcome_loss(output, torch.ones(1))
        parameters = dict(self.core.named_parameters())
        required = (
            "temporal_network.1.weight",
            "query_network.1.weight",
            "key_network.1.weight",
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

    def test_frozen_observations_do_not_receive_gradients(self) -> None:
        state = self.core.initial_state()
        state, _ = _write(self.core, state, 12, 1.0, detach_state=False)
        state, _ = _write(self.core, state, 13, -1.0, detach_state=False)
        relation, temporal, base = _observation(14)
        relation.requires_grad_()
        temporal.requires_grad_()
        base.requires_grad_()
        output = self.core.predict(relation, temporal, base, state=state)
        loss = output.residuals.square().sum()
        observed = torch.autograd.grad(
            loss,
            (relation, temporal, base),
            allow_unused=True,
        )
        self.assertEqual(observed, (None, None, None))

    def test_functional_history_is_bounded_and_uses_distinct_slots(self) -> None:
        state = self.core.initial_state()
        allocations = []
        for index in range(64):
            state, event = _write(
                self.core,
                state,
                100 + index,
                1.0 if index % 2 else -1.0,
            )
            allocations.append(event.allocation_index)
        self.assertEqual(allocations, list(range(64)))
        self.assertEqual(state.step, 64)
        self.assertEqual(int(state.usage.gt(0).sum()), 64)
        self.assertTrue(torch.isfinite(state.keys).all())
        self.assertTrue(torch.isfinite(state.values).all())
        self.assertLessEqual(float(state.keys.abs().max()), 1.0)
        self.assertLessEqual(float(state.values.abs().max()), 1.0)
        self.assertTrue(bool(((state.usage >= 0) & (state.usage <= 1)).all()))
        self.assertTrue(bool(((state.acquisition >= 0) & (state.acquisition <= 1)).all()))

    def test_capture_restore_digest_zero_and_json_replay_are_exact(self) -> None:
        state = self.core.initial_state()
        events = []
        original_logits = []
        for index in range(5):
            output = self.core.predict(*_observation(200 + index), state=state)
            original_logits.append(output.logits.detach().clone())
            state, event = self.core.apply_feedback(
                state,
                output,
                torch.tensor([1.0 if index % 2 else -1.0]),
                evidence_refs=f"event:{index}",
            )
            record = json.loads(json.dumps(event.to_record()))
            events.append(StrictKeyValueCreditEvent.from_record(record))
        digest = self.core.state_digest(state)
        restored = self.core.restore_state(self.core.capture_state(state))
        self.assertEqual(self.core.state_digest(restored), digest)
        replayed, outputs = self.core.replay(events)
        self.assertEqual(self.core.state_digest(replayed), digest)
        self.assertEqual(len(outputs), len(original_logits))
        for expected, actual in zip(original_logits, outputs, strict=True):
            self.assertTrue(torch.equal(expected, actual.logits))
        zero = self.core.zero_state_like(state)
        self.assertEqual(zero.step, 0)
        self.assertTrue(torch.equal(zero.keys, torch.zeros_like(zero.keys)))
        self.assertNotEqual(self.core.state_digest(zero), digest)

    def test_stale_batch_capacity_and_malformed_state_fail_closed(self) -> None:
        state = self.core.initial_state()
        stale = self.core.predict(*_observation(300), state=state)
        advanced, _ = _write(self.core, state, 301, 1.0)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.core.apply_feedback(
                advanced, stale, torch.ones(1), evidence_refs="stale"
            )
        batch = self.core.predict(*_observation(302, batch=2), state=state)
        with self.assertRaisesRegex(ValueError, "one event"):
            self.core.apply_feedback(
                state, batch, torch.ones(2), evidence_refs=("a", "b")
            )
        full = StrictKeyValueCreditState(
            keys=torch.zeros(MEMORY_SLOTS, MEMORY_RANK),
            values=torch.zeros(MEMORY_SLOTS, MEMORY_RANK),
            usage=torch.ones(MEMORY_SLOTS),
            acquisition=torch.ones(MEMORY_SLOTS),
            step=MEMORY_SLOTS,
        )
        full_output = self.core.predict(*_observation(303), state=full)
        with self.assertRaises(OverflowError):
            self.core.apply_feedback(
                full, full_output, torch.ones(1), evidence_refs="full"
            )
        invalid = StrictKeyValueCreditState(
            keys=torch.full((MEMORY_SLOTS, MEMORY_RANK), 2.0),
            values=state.values,
            usage=state.usage,
            acquisition=state.acquisition,
            step=0,
        )
        with self.assertRaisesRegex(ValueError, "keys"):
            self.core.validate_state(invalid)


if __name__ == "__main__":
    unittest.main()
