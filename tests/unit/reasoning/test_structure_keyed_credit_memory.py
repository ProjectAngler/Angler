from __future__ import annotations

import inspect
import json
import unittest

import torch

from angler.reasoning.structure_keyed_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    StructureKeyedCreditEvent,
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditState,
)


def _observation(
    token: int,
    *,
    batch: int = 1,
    temporal_width: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_260_831_140 + token)
    relation = torch.randn(batch, 64, generator=generator)
    temporal = torch.randn(batch, temporal_width, generator=generator)
    base = torch.randn(batch, generator=generator) * 0.1
    return relation, temporal, base


def _write(
    core: StructureKeyedCreditMemoryCore,
    state: StructureKeyedCreditState,
    token: int,
    outcome: float,
    *,
    detach_state: bool = True,
):
    relation, temporal, base = _observation(token)
    output = core.predict(relation, temporal, base, state=state)
    updated, event = core.apply_feedback(
        state,
        output,
        torch.tensor([outcome]),
        evidence_refs=f"evidence://synthetic/{token}",
        detach_state=detach_state,
    )
    return updated, output, event


class StructureKeyedCreditMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20_260_831_14)
        self.core = StructureKeyedCreditMemoryCore(temporal_width=4)

    def test_initial_state_shape_bounds_and_owner_report(self) -> None:
        state = self.core.initial_state()
        self.assertEqual(state.keys.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.values.shape, (MEMORY_SLOTS, MEMORY_RANK))
        self.assertEqual(state.usage.shape, (MEMORY_SLOTS,))
        self.assertEqual(state.acquisition.shape, (MEMORY_SLOTS,))
        self.assertEqual(state.step, 0)
        self.assertEqual(state.bytes, (MEMORY_SLOTS * MEMORY_RANK * 2 + MEMORY_SLOTS * 2) * 4)
        self.core.validate_state(state)
        report = self.core.parameter_report()
        self.assertEqual(report["memory_slots"], 512)
        self.assertEqual(report["rank"], 32)
        self.assertEqual(report["relation_width"], 64)
        self.assertFalse(report["v13_parameters_owned"])
        self.assertFalse(report["outcome_prediction_inputs"])

    def test_predict_is_outcome_free_and_zero_state_is_exact_no_read(self) -> None:
        relation, temporal, base = _observation(0, batch=3)
        signature = inspect.signature(self.core.predict).parameters
        self.assertNotIn("outcomes", signature)
        output = self.core.predict(relation, temporal, base)
        self.assertEqual(output.logits.shape, (3,))
        self.assertEqual(output.read_weights.shape, (3, 512))
        self.assertEqual(output.read_values.shape, (3, 32))
        self.assertEqual(output.write_keys.shape, (3, 32))
        self.assertEqual(output.state_step, 0)
        self.assertTrue(torch.equal(output.residuals, torch.zeros_like(base)))
        self.assertTrue(torch.equal(output.logits, base))
        no_read = self.core.predict(
            relation, temporal, base, read_enabled=False
        )
        self.assertTrue(torch.equal(no_read.read_weights, torch.zeros_like(no_read.read_weights)))
        self.assertTrue(torch.equal(no_read.logits, base))
        self.assertEqual(output.row(1).logits.shape, (1,))

    def test_predict_before_write_and_counterfactual_outcomes_diverge_state(self) -> None:
        initial = self.core.initial_state()
        relation, temporal, base = _observation(1)
        prediction = self.core.predict(relation, temporal, base, state=initial)
        positive, positive_event = self.core.apply_feedback(
            initial,
            prediction,
            torch.ones(1),
            evidence_refs="evidence://positive",
        )
        negative, negative_event = self.core.apply_feedback(
            initial,
            prediction,
            -torch.ones(1),
            evidence_refs="evidence://negative",
        )
        self.assertEqual(positive.step, 1)
        self.assertEqual(negative.step, 1)
        self.assertNotEqual(self.core.state_digest(positive), self.core.state_digest(negative))
        self.assertFalse(torch.equal(positive.values, negative.values))
        self.assertTrue(torch.equal(positive.keys, negative.keys))
        self.assertEqual(positive_event.allocation_index, 0)
        self.assertEqual(negative_event.allocation_index, 0)
        self.assertTrue(torch.equal(prediction.logits, base))

    def test_one_chronological_write_is_bounded_and_evidence_bound(self) -> None:
        initial = self.core.initial_state()
        updated, _, event = _write(self.core, initial, 2, 1.0)
        self.core.validate_state(updated)
        self.assertEqual(updated.step, 1)
        self.assertEqual(event.evidence_refs, ("evidence://synthetic/2",))
        self.assertEqual(event.allocation_index, 0)
        self.assertGreater(float(updated.usage[0].item()), 0.0)
        self.assertEqual(int(torch.count_nonzero(updated.usage).item()), 1)
        self.assertTrue(bool((updated.keys.abs() <= 1.0).all().item()))
        self.assertTrue(bool((updated.values.abs() <= 1.0).all().item()))
        self.assertTrue(bool(((updated.acquisition >= 0) & (updated.acquisition <= 1)).all().item()))
        with self.assertRaises(ValueError):
            self.core.apply_feedback(
                initial,
                self.core.predict(*_observation(3), state=initial),
                torch.ones(1),
                evidence_refs="",
            )

    def test_learned_chronological_credit_reaches_all_required_networks(self) -> None:
        state = self.core.initial_state()
        state, _, _ = _write(
            self.core, state, 4, 1.0, detach_state=False
        )
        relation, temporal, base = _observation(4)
        later = self.core.predict(relation, temporal, base, state=state)
        loss = self.core.outcome_loss(later, torch.ones(1))
        named = dict(self.core.named_parameters())
        gradients = torch.autograd.grad(
            loss,
            tuple(named.values()),
            allow_unused=True,
        )
        by_name = dict(zip(named, gradients, strict=True))
        for prefix in (
            "key_network.",
            "query_network.",
            "value_network.",
            "temporal_network.",
            "residual_network.",
            "outcome_embedding.",
        ):
            relevant = [
                value for name, value in by_name.items() if name.startswith(prefix)
            ]
            self.assertTrue(relevant, prefix)
            self.assertTrue(
                any(
                    value is not None and float(value.abs().sum().item()) > 0.0
                    for value in relevant
                ),
                prefix,
            )

    def test_observations_are_detached_from_frozen_v13_boundary(self) -> None:
        relation, temporal, base = _observation(5)
        relation.requires_grad_(True)
        temporal.requires_grad_(True)
        base.requires_grad_(True)
        output = self.core.predict(relation, temporal, base)
        parameters = tuple(self.core.parameters())
        gradient = torch.autograd.grad(
            output.write_keys.square().sum(),
            (*parameters, relation, temporal, base),
            allow_unused=True,
        )
        self.assertTrue(any(value is not None for value in gradient[: len(parameters)]))
        self.assertTrue(all(value is None for value in gradient[-3:]))

    def test_capture_restore_digest_zero_and_swap_compatibility(self) -> None:
        state_a, _, _ = _write(self.core, self.core.initial_state(), 6, 1.0)
        state_b, _, _ = _write(self.core, self.core.initial_state(), 7, -1.0)
        snapshot = self.core.capture_state(state_a)
        restored = self.core.restore_state(snapshot)
        self.assertEqual(self.core.state_digest(state_a), self.core.state_digest(restored))
        self.core.validate_state(state_b)
        relation, temporal, base = _observation(8)
        read_a = self.core.predict(relation, temporal, base, state=state_a)
        read_b = self.core.predict(relation, temporal, base, state=state_b)
        self.assertFalse(torch.equal(read_a.read_values, read_b.read_values))
        zero = self.core.zero_state_like(state_a)
        zero_output = self.core.predict(relation, temporal, base, state=zero)
        self.assertEqual(zero.step, 0)
        self.assertTrue(torch.equal(zero_output.logits, base))

    def test_event_json_roundtrip_and_exact_replay(self) -> None:
        state = self.core.initial_state()
        events = []
        original_outputs = []
        for token, outcome in ((9, 1.0), (10, -1.0), (11, 1.0)):
            state, output, event = _write(self.core, state, token, outcome)
            original_outputs.append(output)
            record = json.loads(json.dumps(event.to_record()))
            events.append(StructureKeyedCreditEvent.from_record(record))
        replayed, replay_outputs = self.core.replay(events)
        self.assertEqual(self.core.state_digest(state), self.core.state_digest(replayed))
        self.assertEqual(len(replay_outputs), 3)
        for original, reconstructed in zip(original_outputs, replay_outputs, strict=True):
            torch.testing.assert_close(
                original.logits,
                reconstructed.logits,
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_generic_least_unused_allocation_uses_distinct_slots(self) -> None:
        state = self.core.initial_state()
        indices = []
        for token in range(12, 28):
            state, _, event = _write(
                self.core,
                state,
                token,
                1.0 if token % 2 else -1.0,
            )
            indices.append(event.allocation_index)
        self.assertEqual(indices, list(range(16)))
        self.assertEqual(state.step, 16)
        self.assertEqual(int(torch.count_nonzero(state.usage).item()), 16)

    def test_long_functional_history_stays_finite_and_bounded(self) -> None:
        state = self.core.initial_state()
        for token in range(28, 92):
            state, _, _ = _write(
                self.core,
                state,
                token,
                1.0 if token % 3 else -1.0,
            )
        self.assertEqual(state.step, 64)
        self.core.validate_state(state)
        self.assertTrue(bool(torch.isfinite(state.keys).all().item()))
        self.assertTrue(bool(torch.isfinite(state.values).all().item()))
        self.assertTrue(bool((state.keys.abs() <= 1.0 + 1.0e-6).all().item()))
        self.assertTrue(bool((state.values.abs() <= 1.0 + 1.0e-6).all().item()))

    def test_stale_output_batch_write_and_capacity_fail_closed(self) -> None:
        initial = self.core.initial_state()
        output = self.core.predict(*_observation(92), state=initial)
        advanced, _, _ = _write(self.core, initial, 93, 1.0)
        with self.assertRaises(ValueError):
            self.core.apply_feedback(
                advanced,
                output,
                torch.ones(1),
                evidence_refs="evidence://stale",
            )
        batch_output = self.core.predict(*_observation(94, batch=2), state=initial)
        with self.assertRaises(ValueError):
            self.core.apply_feedback(
                initial,
                batch_output,
                torch.ones(2),
                evidence_refs=("evidence://0", "evidence://1"),
            )
        full = StructureKeyedCreditState(
            keys=initial.keys,
            values=initial.values,
            usage=torch.ones_like(initial.usage),
            acquisition=torch.ones_like(initial.acquisition),
            step=MEMORY_SLOTS,
        )
        full_output = self.core.predict(*_observation(95), state=full)
        with self.assertRaises(OverflowError):
            self.core.apply_feedback(
                full,
                full_output,
                torch.ones(1),
                evidence_refs="evidence://overflow",
            )


if __name__ == "__main__":
    unittest.main()
