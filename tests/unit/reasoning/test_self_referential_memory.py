from __future__ import annotations

from dataclasses import fields
import copy
import unittest

import torch

from angler.reasoning.self_referential_memory import (
    SelfReferentialMemory,
    SelfReferentialState,
    detach_self_referential_state,
    restore_self_referential_state,
    self_referential_state_digest,
    snapshot_self_referential_state,
)


class SelfReferentialMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8301)
        self.memory = SelfReferentialMemory(width=8, heads=2)

    @staticmethod
    def assert_state_equal(
        left: SelfReferentialState,
        right: SelfReferentialState,
    ) -> None:
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            if not torch.equal(getattr(left, name), getattr(right, name)):
                raise AssertionError(f"state field differs: {name}")

    def test_state_capacity_is_fixed_independent_of_stream_length(self) -> None:
        initial = self.memory.initial_state(1)
        short_inputs = torch.randn(1, 1, 8)
        long_inputs = torch.randn(1, 19, 8)

        _, short_state = self.memory.write(short_inputs, initial)
        _, long_state = self.memory.write(long_inputs, initial)

        expected = 2 * (3 * 4 * 4 + 4 * 4)
        self.assertEqual(initial.numel(), expected)
        self.assertEqual(short_state.numel(), expected)
        self.assertEqual(long_state.numel(), expected)
        self.assertEqual(self.memory.state_numel(), expected)
        self.assertEqual(self.memory.state_numel(3), expected * 3)
        self.assertEqual(short_state.delta_y.shape, (1, 2, 4, 4))
        self.assertEqual(short_state.delta_beta.shape, (1, 2, 4, 4))

    def test_one_step_matches_transposed_donor_block_equations(self) -> None:
        memory = SelfReferentialMemory(
            width=2,
            heads=1,
            input_softmax=True,
        )
        with torch.no_grad():
            memory.base_y.copy_(
                torch.tensor([[[[0.2, -0.1], [0.4, 0.3]]]])
            )
            memory.base_q.copy_(
                torch.tensor([[[[0.5, -0.2], [-0.4, 0.1]]]])
            )
            memory.base_k.copy_(
                torch.tensor([[[[-0.3, 0.2], [0.6, -0.5]]]])
            )
            memory.base_beta.copy_(
                torch.tensor(
                    [[[[ -0.7, -0.2, 0.3, 0.8],
                       [ 0.1,  0.4, -0.6, 0.2]]]]
                )
            )

        inputs = torch.tensor([[[0.2, -0.4]]])
        state = memory.initial_state(1)
        output, new_state = memory.write(inputs, state)

        event = torch.softmax(inputs[0, 0], dim=-1)
        weights_y = memory.base_y[0, 0]
        weights_q = memory.base_q[0, 0]
        weights_k = memory.base_k[0, 0]
        weights_beta = memory.base_beta[0, 0]
        expected_output = event @ weights_y
        query = torch.softmax(event @ weights_q, dim=-1)
        key = torch.softmax(event @ weights_k, dim=-1)
        rates = torch.sigmoid(event @ weights_beta)
        expected_y = rates[0] * torch.outer(
            key,
            query @ weights_y - key @ weights_y,
        )
        expected_q = rates[1] * torch.outer(
            key,
            query @ weights_q - key @ weights_q,
        )
        expected_k = rates[2] * torch.outer(
            key,
            query @ weights_k - key @ weights_k,
        )
        expected_beta = rates[3] * torch.outer(
            key,
            query @ weights_beta - key @ weights_beta,
        )

        self.assertTrue(torch.allclose(output[0, 0], expected_output))
        self.assertTrue(torch.allclose(new_state.delta_y[0, 0], expected_y))
        self.assertTrue(torch.allclose(new_state.delta_q[0, 0], expected_q))
        self.assertTrue(torch.allclose(new_state.delta_k[0, 0], expected_k))
        self.assertTrue(
            torch.allclose(new_state.delta_beta[0, 0], expected_beta)
        )

    def test_read_is_pure_and_write_updates_every_presented_token(self) -> None:
        initial = self.memory.initial_state(1)
        initial_snapshot = snapshot_self_referential_state(initial)
        query = torch.randn(1, 3, 8)

        first_read = self.memory.read(query, initial)
        second_read = self.memory(query, initial)
        self.assertTrue(torch.equal(first_read, second_read))
        self.assert_state_equal(
            initial,
            restore_self_referential_state(initial_snapshot),
        )

        whole_output, whole_state = self.memory.write(query[:, :2], initial)
        first_output, first_state = self.memory.write(query[:, :1], initial)
        second_output, sequential_state = self.memory.write(
            query[:, 1:2],
            first_state,
        )

        self.assertTrue(
            torch.allclose(
                whole_output,
                torch.cat((first_output, second_output), dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            self.assertTrue(
                torch.allclose(
                    getattr(whole_state, name),
                    getattr(sequential_state, name),
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
            self.assertTrue(torch.equal(getattr(initial, name), torch.zeros_like(getattr(initial, name))))
            self.assertNotEqual(
                getattr(whole_state, name).data_ptr(),
                getattr(initial, name).data_ptr(),
            )

    def test_experience_changes_later_behavior_reset_erases_and_swap_transfers(self) -> None:
        initial = self.memory.initial_state(1)
        experience = torch.randn(1, 4, 8)
        fresh_query = torch.randn(1, 2, 8)
        slow_before = {
            name: value.detach().clone()
            for name, value in self.memory.state_dict().items()
        }

        baseline = self.memory.read(fresh_query, initial)
        _, experienced_state = self.memory.write(experience, initial)
        adapted = self.memory.read(fresh_query, experienced_state)

        self.assertFalse(torch.equal(adapted, baseline))
        self.assertTrue(
            any(
                bool((getattr(experienced_state, name) != 0).any().item())
                for name in ("delta_y", "delta_q", "delta_k", "delta_beta")
            )
        )
        reset = self.memory.read(fresh_query, self.memory.initial_state(1))
        self.assertTrue(torch.equal(reset, baseline))

        receiver = copy.deepcopy(self.memory)
        receiver_reset = receiver.read(fresh_query, receiver.initial_state(1))
        receiver_swapped = receiver.read(fresh_query, experienced_state)
        self.assertTrue(torch.equal(receiver_reset, baseline))
        self.assertTrue(torch.equal(receiver_swapped, adapted))
        for name, value in self.memory.state_dict().items():
            self.assertTrue(torch.equal(value, slow_before[name]))

    def test_snapshot_restore_and_detach_are_exact_and_independent(self) -> None:
        _, live_state = self.memory.write(
            torch.randn(1, 5, 8),
            self.memory.initial_state(1),
        )
        detached = detach_self_referential_state(live_state)
        snapshot = snapshot_self_referential_state(detached)
        restored = restore_self_referential_state(snapshot)

        self.assert_state_equal(detached, restored)
        self.assertEqual(
            self_referential_state_digest(detached),
            self_referential_state_digest(restored),
        )
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            self.assertFalse(getattr(detached, name).requires_grad)
            self.assertFalse(getattr(restored, name).requires_grad)
            self.assertNotEqual(
                getattr(detached, name).data_ptr(),
                getattr(restored, name).data_ptr(),
            )
            self.assertNotEqual(
                snapshot[name].data_ptr(),
                getattr(restored, name).data_ptr(),
            )

        with torch.no_grad():
            detached.delta_y.add_(1.0)
        self.assertTrue(torch.equal(restored.delta_y, snapshot["delta_y"]))
        self.assertFalse(torch.equal(restored.delta_y, detached.delta_y))
        self.assertNotEqual(
            self_referential_state_digest(detached),
            self_referential_state_digest(restored),
        )

    def test_later_query_loss_meta_trains_the_earlier_update_dynamics(self) -> None:
        memory = SelfReferentialMemory(width=8, heads=2).double()
        experience = torch.randn(1, 3, 8, dtype=torch.float64)
        fresh_query = torch.randn(1, 2, 8, dtype=torch.float64)
        _, changed_state = memory.write(
            experience,
            memory.initial_state(1),
        )

        later_output = memory.read(fresh_query, changed_state)
        coefficients = torch.linspace(
            0.2,
            1.1,
            later_output.numel(),
            dtype=later_output.dtype,
        ).reshape_as(later_output)
        outer_loss = (later_output * coefficients).sum()
        gradients = torch.autograd.grad(
            outer_loss,
            (memory.base_q, memory.base_k, memory.base_beta),
        )

        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().sum().item()), 1e-12)

        severed_state = detach_self_referential_state(changed_state)
        severed_output = memory.read(fresh_query, severed_state)
        severed_loss = (severed_output * coefficients).sum()
        severed_gradients = torch.autograd.grad(
            severed_loss,
            (memory.base_q, memory.base_k, memory.base_beta),
            allow_unused=True,
        )
        self.assertEqual(severed_gradients, (None, None, None))

    def test_public_state_has_only_shared_numeric_offsets(self) -> None:
        self.assertEqual(
            [field.name for field in fields(SelfReferentialState)],
            ["delta_y", "delta_q", "delta_k", "delta_beta"],
        )
        self.assertEqual(
            set(self.memory.state_dict()),
            {"base_y", "base_q", "base_k", "base_beta"},
        )

    def test_invalid_topology_nonfinite_input_and_snapshot_are_rejected(self) -> None:
        state = self.memory.initial_state(1)
        with self.assertRaisesRegex(ValueError, "width"):
            self.memory.read(torch.randn(1, 2, 7), state)
        invalid = torch.randn(1, 2, 8)
        invalid[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.memory.read(invalid, state)
        wrong_state = SelfReferentialState(
            delta_y=state.delta_y,
            delta_q=state.delta_q,
            delta_k=state.delta_k,
            delta_beta=torch.zeros(1, 2, 4, 3),
        )
        with self.assertRaisesRegex(ValueError, "beta state"):
            self.memory.read(torch.randn(1, 2, 8), wrong_state)
        snapshot = snapshot_self_referential_state(state)
        snapshot["unexpected"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "snapshot keys"):
            restore_self_referential_state(snapshot)


if __name__ == "__main__":
    unittest.main()
