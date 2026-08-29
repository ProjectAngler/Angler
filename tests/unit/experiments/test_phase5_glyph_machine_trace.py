from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from experiments.evaluators.glyph_machine_trace_suite import (
    glyph_machine_mechanism_partition,
    judge_glyph_procedure_attempt,
    make_glyph_machine_trace_stream,
)
from experiments.runners import phase5_glyph_machine_trace as runner
from experiments.runners.phase5_glyph_machine_trace import (
    acquire_public_traces,
    acquire_and_score_public_traces,
    apply_scalar_procedure_feedback,
    build_glyph_machine_controller,
    build_glyph_machine_evaluation_arms,
    centered_trajectory_preference_loss,
    default_glyph_machine_experiment_config,
    glyph_associative_state_digest,
    glyph_machine_parameter_report,
    load_glyph_checkpoint,
    rollout_glyph_procedure,
    run_glyph_machine_experiment,
    save_glyph_checkpoint,
    scalar_outcome_loss,
    snapshot_glyph_state,
    restore_glyph_state,
)


@lru_cache(maxsize=1)
def _tasks_by_public_shape() -> dict[tuple[int, int], object]:
    selected: dict[tuple[int, int], object] = {}
    stream_seed = 107_001
    for partition in ("train", "development"):
        for offset, commitment in enumerate(
            glyph_machine_mechanism_partition(partition)
        ):
            stream = make_glyph_machine_trace_stream(
                stream_seed + offset,
                surface_seed=207_001 + offset,
                supports=1,
                queries=1,
                observations_per_support=2,
                mechanism_commitment=commitment,
                mechanism_partition=partition,
            )
            task = stream.supports[0].learner
            selected.setdefault((len(task.states), len(task.actions)), task)
        stream_seed += 10_000
    return selected


def _task_with(*, states: int | None = None, actions: int | None = None):
    for (state_count, action_count), task in _tasks_by_public_shape().items():
        if (states is None or states == state_count) and (
            actions is None or actions == action_count
        ):
            return task
    raise AssertionError(
        f"no public GlyphMachine task has shape states={states}, actions={actions}"
    )


def _zero_action_heads(controller: runner.GlyphMachineController) -> None:
    with torch.no_grad():
        for module in (
            controller.successor_query,
            controller.procedure_reasoner.edge_gate,
        ):
            for parameter in module.parameters():
                parameter.zero_()


def _maximum_off_diagonal_cosine(rows: torch.Tensor) -> float:
    if rows.ndim != 2 or len(rows) < 2:
        raise ValueError("cosine separation requires at least two rows")
    normalized = F.normalize(rows, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(
        len(rows),
        device=rows.device,
        dtype=torch.bool,
    )
    return float(similarities[off_diagonal].detach().max().item())


def _event_keys_for_state(
    controller: runner.GlyphMachineController,
    task,
    encoded: runner.GlyphTaskEncoding,
    state_index: int,
) -> torch.Tensor:
    goal_index = next(
        index
        for index, value in enumerate(task.states)
        if value.records == task.goal.required
    )
    belief = F.one_hot(
        torch.tensor(state_index, device=encoded.state_embeddings.device),
        len(task.states),
    ).to(dtype=encoded.state_embeddings.dtype)
    return controller.event_query_keys(encoded, belief, goal_index)


def _reasoner_fixture(
    reasoner: runner.GlyphBackwardProcedureReasoner,
    action_rows: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, runner.GlyphTransitionLattice]:
    """Build explicit public-belief tensors without an evaluator-owned table."""

    device = reasoner.goal_tokens.device
    dtype = reasoner.goal_tokens.dtype
    state_count = len(action_rows[0])
    action_count = len(action_rows)
    states = torch.zeros(
        state_count,
        reasoner.width,
        device=device,
        dtype=dtype,
    )
    states[:, :state_count] = torch.eye(state_count, device=device, dtype=dtype)
    probabilities = torch.zeros(
        state_count,
        action_count,
        state_count,
        device=device,
        dtype=dtype,
    )
    for action_index, row in enumerate(action_rows):
        for source_index, target_index in enumerate(row):
            probabilities[source_index, action_index, target_index] = 1.0
    logits = torch.log(
        probabilities.clamp_min(torch.finfo(dtype).tiny)
    )
    predicted = torch.einsum("sat,tw->saw", probabilities, states)
    contexts = torch.zeros_like(predicted)
    return states, runner.GlyphTransitionLattice(
        successor_state_logits=logits,
        successor_probabilities=probabilities,
        associative_recall_logits=torch.zeros_like(logits),
        predicted_successors=predicted,
        raw_reversible_successors=predicted.clone(),
        trace_contexts=contexts,
        outcome_contexts=contexts.clone(),
    )


class GlyphMachineControllerShapeTests(unittest.TestCase):
    def test_dynamic_public_shapes_produce_factorized_actions_plus_stop(self) -> None:
        torch.manual_seed(108_001)
        controller = build_glyph_machine_controller("smoke")
        state = controller.initial_state()
        tasks = _tasks_by_public_shape()

        self.assertEqual({key[0] for key in tasks}, {2, 3, 4})
        self.assertEqual({key[1] for key in tasks}, {1, 2, 3})
        for (state_count, action_count), task in tasks.items():
            scores = controller.score_actions(task, state)
            self.assertEqual(scores.logits.shape, (action_count + 1,))
            self.assertEqual(scores.action_logits.shape, (action_count,))
            self.assertEqual(
                scores.successor_state_logits.shape,
                (action_count, state_count),
            )
            self.assertEqual(
                scores.predicted_successors.shape,
                (action_count, controller.profile.width),
            )
            self.assertEqual(
                scores.transition_lattice_logits.shape,
                (state_count, action_count, state_count),
            )
            self.assertEqual(
                scores.reasoning_node_codes.shape,
                (state_count, controller.profile.width),
            )
            self.assertEqual(scores.current_state_belief.shape, (state_count,))
            self.assertEqual(scores.reasoning_steps, task.max_steps)
            self.assertTrue(torch.isfinite(scores.logits).all())
            self.assertTrue(
                torch.equal(scores.action_logits, scores.reasoning_action_logits)
            )

    def test_anchored_encoder_is_set_permutation_equivariant_on_one_device(self) -> None:
        torch.manual_seed(108_002)
        requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = build_glyph_machine_controller("smoke", device=requested_device)
        device = next(controller.parameters()).device
        task = _task_with(states=4, actions=3)
        state_order = (2, 0, 3, 1)
        action_order = (1, 2, 0)
        permuted = replace(
            task,
            states=tuple(task.states[index] for index in state_order),
            actions=tuple(task.actions[index] for index in action_order),
        )

        original = controller.encode_task(task)
        changed = controller.encode_task(permuted)
        self.assertTrue(
            torch.allclose(
                changed.state_embeddings,
                original.state_embeddings[list(state_order)],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                changed.action_embeddings,
                original.action_embeddings[list(action_order)],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                changed.state_address_anchors,
                original.state_address_anchors[list(state_order)],
            )
        )
        self.assertTrue(
            torch.equal(
                changed.action_address_anchors,
                original.action_address_anchors[list(action_order)],
            )
        )
        self.assertTrue(
            torch.equal(
                changed.pair_key_anchors,
                original.pair_key_anchors[list(state_order)][
                    :, list(action_order)
                ],
            )
        )
        self.assertTrue(
            torch.equal(
                changed.stop_key_anchors,
                original.stop_key_anchors[list(state_order)][
                    :, list(state_order)
                ],
            )
        )
        original_keys = controller.transition_event_keys(original)
        changed_keys = controller.transition_event_keys(changed)
        self.assertTrue(
            torch.allclose(
                changed_keys,
                original_keys[list(state_order)][:, list(action_order)],
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        )
        self.assertEqual(changed.state_embeddings.device, device)
        self.assertEqual(changed.action_embeddings.device, device)
        self.assertEqual(
            controller.graph_encoder.public_identity_features._device_anchor.device,
            device,
        )

    def test_pair_anchor_uses_complete_action_structure_without_parameters(self) -> None:
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=3, actions=2)
        changed_action = replace(
            task.actions[0],
            description="public structural description",
        )
        changed_task = replace(
            task,
            actions=(changed_action, *task.actions[1:]),
            observations=(),
        )
        original = controller.encode_task(replace(task, observations=()))
        changed = controller.encode_task(changed_task)

        self.assertFalse(
            torch.equal(
                original.pair_key_anchors[:, 0],
                changed.pair_key_anchors[:, 0],
            )
        )
        self.assertTrue(
            torch.equal(
                original.pair_key_anchors[:, 1:],
                changed.pair_key_anchors[:, 1:],
            )
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in controller.public_address_features.parameters()),
            0,
        )

    def test_rollout_uses_only_declared_actions_and_honors_stop_and_budget(self) -> None:
        torch.manual_seed(108_003)
        task = _task_with(actions=3)
        controller = build_glyph_machine_controller("smoke")
        state = controller.initial_state()
        _zero_action_heads(controller)

        with torch.no_grad():
            controller.stop_head[-1].weight.zero_()
            controller.stop_head[-1].bias.fill_(100.0)
        stopped = rollout_glyph_procedure(controller, task, state)
        self.assertTrue(stopped.procedure.stopped)
        self.assertEqual(stopped.procedure.actions, ())
        self.assertEqual(stopped.selected_indices, (len(task.actions),))

        with torch.no_grad():
            controller.stop_head[-1].bias.fill_(-100.0)
        exhausted = rollout_glyph_procedure(controller, task, state)
        self.assertFalse(exhausted.procedure.stopped)
        self.assertEqual(len(exhausted.procedure.actions), task.max_steps)
        self.assertEqual(len(exhausted.step_logits), task.max_steps)
        declared = set(task.actions)
        self.assertTrue(
            all(action.schema in declared for action in exhausted.procedure.actions)
        )


class GlyphMachineMultiStepReasonerTests(unittest.TestCase):
    def test_learned_backward_messages_use_a_deep_edge_not_one_step_goal_mass(self) -> None:
        torch.manual_seed(108_051)
        profile = runner.GLYPH_MACHINE_PROFILES["smoke"]
        reasoner = runner.GlyphBackwardProcedureReasoner(profile)
        # Public reversible edges: a takes 0->1; b then takes 1->2->3.
        # At the origin neither action has any one-step probability on goal 3.
        correct_rows = (
            (1, 0, 2, 3),
            (0, 2, 3, 1),
        )
        # The complete origin row and b's 1->2 edge are unchanged.  Only the
        # deeper b edge at state 2 no longer reaches the public destination.
        broken_rows = (
            (1, 0, 2, 3),
            (0, 2, 1, 3),
        )
        removed_rows = (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
        )
        states, correct = _reasoner_fixture(reasoner, correct_rows)
        _, broken = _reasoner_fixture(reasoner, broken_rows)
        _, removed = _reasoner_fixture(reasoner, removed_rows)
        self.assertTrue(
            torch.equal(
                correct.successor_probabilities[0, :, 3],
                torch.zeros(2),
            )
        )

        optimizer = torch.optim.Adam(reasoner.parameters(), lr=2.0e-2)
        public_suffix_targets = ((2, 1, 1), (1, 2, 1), (0, 3, 0))
        for _ in range(120):
            losses = []
            for source_index, horizon, action_index in public_suffix_targets:
                belief = F.one_hot(
                    torch.tensor(source_index),
                    4,
                ).to(dtype=states.dtype)
                logits, _ = reasoner(
                    states,
                    3,
                    correct,
                    belief,
                    steps_remaining=horizon,
                )
                losses.append(
                    torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor((action_index,)),
                    )
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        origin = torch.tensor((1.0, 0.0, 0.0, 0.0))
        with torch.no_grad():
            deep_correct, _ = reasoner(
                states, 3, correct, origin, steps_remaining=3
            )
            deep_broken, _ = reasoner(
                states, 3, broken, origin, steps_remaining=3
            )
            shallow_correct, _ = reasoner(
                states, 3, correct, origin, steps_remaining=1
            )
            shallow_broken, _ = reasoner(
                states, 3, broken, origin, steps_remaining=1
            )
            removed_logits, _ = reasoner(
                states, 3, removed, origin, steps_remaining=3
            )
        self.assertTrue(
            torch.equal(shallow_correct, shallow_broken),
            msg="one-step receptive field changed despite an identical origin row",
        )
        deep_margin = float((deep_correct[0] - deep_correct[1]).item())
        shallow_margin = float((shallow_correct[0] - shallow_correct[1]).item())
        self.assertGreater(deep_margin, 0.5)
        self.assertGreater(deep_margin, shallow_margin + 0.25)
        self.assertGreater(
            float((deep_correct[0] - deep_broken[0]).item()),
            0.1,
        )
        self.assertAlmostEqual(
            float((removed_logits[0] - removed_logits[1]).item()),
            0.0,
            places=6,
        )

        reasoner.zero_grad(set_to_none=True)
        logits, _ = reasoner(states, 3, correct, origin, steps_remaining=3)
        torch.nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor((0,)),
        ).backward()
        for label, module in (
            ("edge", reasoner.edge_encoder),
            ("gate", reasoner.edge_gate),
            ("recurrent", reasoner.state_cell),
        ):
            gradient = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, msg=label)

    def test_reasoner_is_simultaneously_state_and_action_permutation_equivariant(self) -> None:
        torch.manual_seed(108_053)
        reasoner = runner.GlyphBackwardProcedureReasoner(
            runner.GLYPH_MACHINE_PROFILES["smoke"]
        )
        states, lattice = _reasoner_fixture(
            reasoner,
            ((1, 0, 2, 3), (0, 2, 3, 1)),
        )
        belief = torch.tensor((1.0, 0.0, 0.0, 0.0))
        original, _ = reasoner(states, 3, lattice, belief, steps_remaining=3)

        state_order = torch.tensor((2, 0, 3, 1))
        action_order = torch.tensor((1, 0))
        permuted_states = states[state_order]
        probabilities = lattice.successor_probabilities[state_order][
            :, action_order
        ][:, :, state_order]
        predicted = torch.einsum("sat,tw->saw", probabilities, permuted_states)
        permuted = runner.GlyphTransitionLattice(
            successor_state_logits=lattice.successor_state_logits[state_order][
                :, action_order
            ][:, :, state_order],
            successor_probabilities=probabilities,
            associative_recall_logits=lattice.associative_recall_logits[
                state_order
            ][:, action_order][:, :, state_order],
            predicted_successors=predicted,
            raw_reversible_successors=predicted.clone(),
            trace_contexts=lattice.trace_contexts[state_order][:, action_order],
            outcome_contexts=lattice.outcome_contexts[state_order][:, action_order],
        )
        permuted_goal = int((state_order == 3).nonzero(as_tuple=False)[0].item())
        permuted_belief = belief[state_order]
        changed, _ = reasoner(
            permuted_states,
            permuted_goal,
            permuted,
            permuted_belief,
            steps_remaining=3,
        )
        self.assertTrue(
            torch.allclose(changed, original[action_order], atol=1.0e-6, rtol=1.0e-6)
        )

    def test_controller_reasoning_and_plastic_state_share_one_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = build_glyph_machine_controller("smoke", device=device)
        device = next(controller.parameters()).device
        task = _task_with(states=4, actions=2)
        state = controller.initial_state()
        scores = controller.score_actions(task, state)
        self.assertEqual(state.keys.device, device)
        for value in (
            scores.logits,
            scores.transition_lattice_logits,
            scores.reasoning_node_codes,
            scores.current_state_belief,
        ):
            self.assertEqual(value.device, device)
        optimizer = torch.optim.Adam(controller.parameters(), lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        scores.logits.square().mean().backward()
        optimizer.step()
        moment_devices = {
            value.device
            for parameter_state in optimizer.state.values()
            for name, value in parameter_state.items()
            if name in ("exp_avg", "exp_avg_sq")
            and isinstance(value, torch.Tensor)
        }
        self.assertEqual(moment_devices, {device})


class GlyphMachineMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(108_101)
        self.controller = build_glyph_machine_controller("smoke")
        self.task = _task_with(states=4, actions=3)

    def test_public_trace_acquisition_has_fixed_capacity(self) -> None:
        state = self.controller.initial_state()
        expected = state.numel()
        observed = 0
        accepted = 0
        with torch.no_grad():
            for _ in range(3):
                acquisition = acquire_public_traces(
                    self.controller,
                    self.task,
                    state,
                )
                state = acquisition.state
                observed += acquisition.public_transitions
                accepted += acquisition.accepted_writes

        self.assertGreater(observed, 0)
        self.assertGreater(accepted, 0)
        self.assertEqual(state.numel(), expected)
        self.assertEqual(expected, self.controller.memory.state_numel(1))
        self.assertEqual(
            int(state.occupied.sum().item()),
            self.controller.memory.trace_slot_count,
        )
        self.assertLessEqual(
            int(state.write_counts.sum().item()),
            self.controller.memory.trace_slot_count,
        )
        self.assertGreater(accepted, self.controller.memory.trace_slot_count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_anchored_write_read_recall_and_residual_gradients_remain_on_cuda(self) -> None:
        controller = build_glyph_machine_controller("smoke", device="cuda")
        task = self.task
        encoding = controller.encode_task(task)
        acquisition, objective, _, _ = acquire_and_score_public_traces(
            controller,
            task,
            controller.initial_state(),
        )
        keys = controller.transition_event_keys(encoding)
        read = controller.memory.read(
            keys.reshape(-1, controller.profile.width),
            acquisition.state,
            lane="trace",
        )
        lattice = controller.transition_lattice(encoding, acquisition.state)
        objective.backward()

        for value in (
            encoding.pair_key_anchors,
            keys,
            acquisition.state.keys,
            acquisition.state.values,
            read.attention_weights,
            lattice.associative_recall_logits,
            controller.memory._device_dtype_anchor,
            controller.public_address_features._device_anchor,
        ):
            self.assertEqual(value.device.type, "cuda")
        for module in (
            controller.event_key_encoder,
            controller.trace_value_encoder,
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(value.device.type == "cuda" for value in gradients))
            self.assertGreater(
                sum(float(value.detach().abs().sum().item()) for value in gradients),
                0.0,
            )

    def test_scalar_attempt_is_one_transaction_and_rejection_restores_exactly(self) -> None:
        initial = self.controller.initial_state()
        with torch.no_grad():
            acquired = acquire_public_traces(
                self.controller,
                self.task,
                initial,
            ).state
        trace_slots = self.controller.memory.trace_slot_count
        self.assertFalse(acquired.occupied[0, trace_slots:].any())
        trace_snapshot = {
            name: value.clone()
            for name, value in (
                ("keys", acquired.keys[:, :trace_slots]),
                ("values", acquired.values[:, :trace_slots]),
                ("occupied", acquired.occupied[:, :trace_slots]),
                ("counts", acquired.write_counts[:, :trace_slots]),
                (
                    "source_action",
                    acquired.public_source_action_ids[:, :trace_slots],
                ),
                ("successor", acquired.public_successor_ids[:, :trace_slots]),
                ("cursor", acquired.trace_cursor),
            )
        }
        with torch.no_grad():
            # Keep this binding test independent of the random policy length.
            # A single explicit STOP leaves a fresh outcome slot for rebound.
            self.controller.stop_head[-1].weight.zero_()
            self.controller.stop_head[-1].bias.fill_(100.0)
            rollout = rollout_glyph_procedure(
                self.controller,
                self.task,
                acquired,
            )
        before = glyph_associative_state_digest(acquired)
        rejected = apply_scalar_procedure_feedback(
            self.controller,
            self.task,
            rollout,
            1.0,
            acquired,
            minimum_effect=1.0e30,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.scalar_observations, 1)
        self.assertEqual(glyph_associative_state_digest(rejected.state), before)

        accepted = apply_scalar_procedure_feedback(
            self.controller,
            self.task,
            rollout,
            1.0,
            acquired,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.scalar_observations, 1)
        self.assertEqual(accepted.state.numel(), acquired.numel())
        self.assertEqual(len(accepted.write_slots), len(rollout.step_logits))
        self.assertTrue(accepted.state.occupied[0, trace_slots:].any())
        for name, value in (
            ("keys", accepted.state.keys[:, :trace_slots]),
            ("values", accepted.state.values[:, :trace_slots]),
            ("occupied", accepted.state.occupied[:, :trace_slots]),
            ("counts", accepted.state.write_counts[:, :trace_slots]),
            (
                "source_action",
                accepted.state.public_source_action_ids[:, :trace_slots],
            ),
            ("successor", accepted.state.public_successor_ids[:, :trace_slots]),
            ("cursor", accepted.state.trace_cursor),
        ):
            self.assertTrue(torch.equal(value, trace_snapshot[name]), msg=name)
        with self.assertRaises(ValueError):
            apply_scalar_procedure_feedback(
                self.controller,
                self.task,
                rollout,
                1.0,
                accepted.state,
            )
        rebound = apply_scalar_procedure_feedback(
            self.controller,
            self.task,
            rollout,
            1.0,
            accepted.state,
            binding_state=acquired,
        )
        self.assertTrue(rebound.accepted)

        failed = apply_scalar_procedure_feedback(
            self.controller,
            self.task,
            rollout,
            0.0,
            acquired,
        )
        self.assertTrue(failed.accepted)
        self.assertNotEqual(
            glyph_associative_state_digest(failed.state),
            glyph_associative_state_digest(accepted.state),
        )

    def test_outcome_lane_wraparound_cannot_evict_trace_evidence(self) -> None:
        with torch.no_grad():
            acquired = acquire_public_traces(
                self.controller,
                self.task,
                self.controller.initial_state(),
            ).state
            before = snapshot_glyph_state(acquired)
            event_count = 3 * self.controller.memory.outcome_slot_count
            write = self.controller.memory.write_events(
                torch.randn(event_count, self.controller.profile.width),
                torch.randn(event_count, self.controller.profile.width),
                acquired,
                lane="outcome",
            )
        self.assertTrue(write.accepted)
        trace_slots = self.controller.memory.trace_slot_count
        for name in (
            "keys",
            "values",
            "occupied",
            "write_counts",
            "public_source_action_ids",
            "public_successor_ids",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(write.state, name)[:, :trace_slots],
                    before[name][:, :trace_slots],
                ),
                msg=name,
            )
        self.assertTrue(torch.equal(write.state.trace_cursor, before["trace_cursor"]))

    def test_snapshot_restore_and_reversible_transition_round_trip(self) -> None:
        state = self.controller.initial_state()
        with torch.no_grad():
            state = acquire_public_traces(
                self.controller,
                self.task,
                state,
            ).state
        snapshot = snapshot_glyph_state(state)
        restored = restore_glyph_state(snapshot)
        self.assertEqual(
            glyph_associative_state_digest(restored),
            glyph_associative_state_digest(state),
        )
        self.assertTrue(
            all(
                value.data_ptr() != snapshot[name].data_ptr()
                for name, value in snapshot_glyph_state(restored).items()
            )
        )

        width = self.controller.profile.width
        source = torch.randn(3, width)
        condition = torch.randn(3, 2 * width)
        transitioned = self.controller.causal_transition(source, condition)
        recovered = self.controller.causal_transition(
            transitioned,
            condition,
            reverse=True,
        )
        self.assertTrue(torch.allclose(source, recovered, atol=1e-6, rtol=1e-6))

    def test_soft_action_conditioned_read_is_row_equivariant_and_empty_exact_zero(self) -> None:
        empty = self.controller.initial_state()
        encoded = self.controller.encode_task(self.task)
        origin_index = next(
            index
            for index, value in enumerate(self.task.states)
            if value.digest == self.task.origin.digest
        )
        keys = _event_keys_for_state(
            self.controller, self.task, encoded, origin_index
        )
        empty_read = self.controller.memory.read(keys, empty)
        self.assertTrue(torch.equal(empty_read.contexts, torch.zeros_like(empty_read.contexts)))
        self.assertTrue(
            torch.equal(
                empty_read.attention_weights,
                torch.zeros_like(empty_read.attention_weights),
            )
        )

        acquired = empty
        for _ in range(3):
            acquired = acquire_public_traces(
                self.controller, self.task, acquired
            ).state
        ordinary = self.controller.memory.read(keys, acquired)
        permutation = torch.arange(keys.shape[0] - 1, -1, -1)
        permuted = self.controller.memory.read(keys[permutation], acquired)
        self.assertTrue(
            torch.allclose(
                permuted.contexts,
                ordinary.contexts[:, permutation],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        occupied = acquired.occupied[0]
        self.assertGreater(
            int(occupied.sum().item()),
            self.controller.profile.memory_read_top_k,
        )
        nonzero = ordinary.attention_weights.count_nonzero(dim=-1)
        expected_nonzero = min(
            self.controller.profile.memory_read_top_k,
            int(occupied.sum().item()),
        )
        self.assertTrue(torch.equal(nonzero, nonzero.new_full(nonzero.shape, expected_nonzero)))
        self.assertTrue(
            torch.equal(
                ordinary.attention_weights[..., ~occupied],
                torch.zeros_like(ordinary.attention_weights[..., ~occupied]),
            )
        )
        action_contexts = ordinary.contexts[0, : len(self.task.actions)]
        self.assertGreater(
            float((action_contexts[0] - action_contexts[-1]).abs().max().item()),
            0.0,
        )

    def test_matching_transition_recall_survives_independent_surface_renaming(self) -> None:
        commitment = glyph_machine_mechanism_partition("development")[0]
        context_norms = []
        for surface_seed in (308_101, 408_101):
            stream = make_glyph_machine_trace_stream(
                208_101,
                surface_seed=surface_seed,
                supports=1,
                queries=1,
                observations_per_support=2,
                mechanism_commitment=commitment,
                mechanism_partition="development",
            )
            task = stream.supports[0].learner
            state = acquire_public_traces(
                self.controller,
                task,
                self.controller.initial_state(),
            ).state
            transition = task.observations[0].transitions[0]
            encoded = self.controller.encode_task(task)
            before_index = next(
                index
                for index, value in enumerate(task.states)
                if value.digest == transition.before.digest
            )
            action_index = next(
                index
                for index, value in enumerate(task.actions)
                if value.digest == transition.action.schema.digest
            )
            keys = _event_keys_for_state(
                self.controller, task, encoded, before_index
            )
            read = self.controller.memory.read(keys, state)
            context_norms.append(float(read.contexts[0, action_index].norm().item()))
        self.assertTrue(all(value > 0.0 for value in context_norms))

    def test_nonempty_read_below_top_k_uses_only_occupied_slots(self) -> None:
        observation = self.task.observations[0]
        single = replace(
            self.task,
            observations=(
                replace(observation, transitions=(observation.transitions[0],)),
            ),
        )
        state = acquire_public_traces(
            self.controller,
            single,
            self.controller.initial_state(),
        ).state
        encoded = self.controller.encode_task(single)
        origin_index = next(
            index
            for index, value in enumerate(single.states)
            if value.digest == single.origin.digest
        )
        keys = _event_keys_for_state(
            self.controller, single, encoded, origin_index
        )
        read = self.controller.memory.read(keys, state, lane="trace")
        self.assertEqual(int(state.occupied.sum().item()), 1)
        self.assertTrue(
            torch.equal(
                read.attention_weights.count_nonzero(dim=-1),
                torch.ones_like(read.attention_weights.count_nonzero(dim=-1)),
            )
        )

    def test_large_shared_learned_residual_cannot_collapse_pair_or_value_anchors(self) -> None:
        encoded = self.controller.encode_task(self.task)

        def huge(rows: torch.Tensor) -> torch.Tensor:
            return rows.new_full((*rows.shape[:-1], self.controller.profile.width), 1.0e20)

        with mock.patch.object(
            self.controller.event_key_encoder,
            "forward",
            side_effect=huge,
        ):
            keys = self.controller.transition_event_keys(encoded)
            source_index = 0
            goal_index = 1
            belief = F.one_hot(
                torch.tensor(source_index),
                len(self.task.states),
            ).to(dtype=encoded.state_embeddings.dtype)
            stop_key = self.controller.event_query_keys(
                encoded,
                belief,
                goal_index,
            )[-1]

        key_delta = keys - encoded.pair_key_anchors
        self.assertLessEqual(float(key_delta.norm(dim=-1).max().item()), 0.250001)
        self.assertLess(
            float((key_delta * encoded.pair_key_anchors).sum(dim=-1).abs().max().item()),
            1.0e-5,
        )
        self.assertLess(
            _maximum_off_diagonal_cosine(keys.flatten(0, 1)),
            0.98,
        )
        stop_anchor = encoded.stop_key_anchors[source_index, goal_index]
        stop_delta = stop_key - stop_anchor
        self.assertLessEqual(float(stop_delta.norm().item()), 0.250001)
        self.assertLess(float((stop_delta * stop_anchor).sum().abs().item()), 1.0e-5)

        before = encoded.state_embeddings[0]
        action = encoded.action_embeddings[0]
        successor = encoded.state_embeddings[1]
        with mock.patch.object(
            self.controller.trace_value_encoder,
            "forward",
            side_effect=huge,
        ):
            value = self.controller.trace_event_value(before, action, successor)
        value_delta = value - successor
        self.assertLessEqual(float(value_delta.norm().item()), 0.250001)
        self.assertLess(float((value_delta * successor).sum().abs().item()), 1.0e-4)

    def test_same_public_pair_has_one_address_but_successor_specific_values(self) -> None:
        transition = runner._unique_public_transitions(self.task)[0]
        alternative = next(
            value
            for value in self.task.states
            if value.digest != transition.after.digest
        )
        changed = replace(transition, after=alternative)
        first = runner._acquire_transition_sequence(
            self.controller,
            self.task,
            self.controller.initial_state(),
            (transition,),
        ).state
        second = runner._acquire_transition_sequence(
            self.controller,
            self.task,
            self.controller.initial_state(),
            (changed,),
        ).state

        self.assertTrue(torch.equal(first.keys[:, 0], second.keys[:, 0]))
        self.assertFalse(torch.equal(first.values[:, 0], second.values[:, 0]))
        self.assertTrue(
            torch.equal(
                first.public_source_action_ids[:, 0],
                second.public_source_action_ids[:, 0],
            )
        )
        self.assertFalse(
            torch.equal(
                first.public_successor_ids[:, 0],
                second.public_successor_ids[:, 0],
            )
        )

    def test_direct_successor_recall_preserves_the_stored_state_anchor(self) -> None:
        transition = runner._unique_public_transitions(self.task)[0]
        encoded = self.controller.encode_task(self.task)
        acquired = runner._acquire_transition_sequence(
            self.controller,
            self.task,
            self.controller.initial_state(),
            (transition,),
            encoding=encoded,
        ).state
        lattice = self.controller.transition_lattice(encoded, acquired)
        width = self.controller.profile.width
        expected = torch.einsum(
            "saw,tw->sat",
            lattice.trace_contexts,
            encoded.state_embeddings,
        ) / (width ** 0.5)
        self.assertTrue(
            torch.allclose(
                lattice.associative_recall_logits,
                expected,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        )
        state_indices = {
            value.digest: index for index, value in enumerate(self.task.states)
        }
        action_indices = {
            value.digest: index for index, value in enumerate(self.task.actions)
        }
        row = lattice.associative_recall_logits[
            state_indices[transition.before.digest],
            action_indices[transition.action.schema.digest],
        ]
        target = state_indices[transition.after.digest]
        self.assertEqual(int(row.argmax().item()), target)
        self.assertGreater(float(torch.softmax(row, dim=-1)[target].item()), 0.40)
        self.assertFalse(hasattr(self.controller, "associative_recall_query"))

    def test_memory_uses_direct_final_keys_and_nonpersistent_device_buffer(self) -> None:
        self.assertFalse(hasattr(self.controller.memory, "query_projection"))
        self.assertFalse(hasattr(self.controller.memory, "key_projection"))
        self.assertEqual(sum(p.numel() for p in self.controller.memory.parameters()), 0)
        self.assertNotIn("_device_dtype_anchor", self.controller.memory.state_dict())
        state = self.controller.initial_state()
        self.assertEqual(
            self.controller.memory._device_dtype_anchor.device,
            state.keys.device,
        )
        self.assertEqual(
            self.controller.memory._device_dtype_anchor.dtype,
            state.keys.dtype,
        )

    def test_transition_ablation_removes_every_direct_memory_to_action_path(self) -> None:
        initial = self.controller.initial_state()
        with torch.no_grad():
            initial_scores = self.controller.score_actions(self.task, initial)
            acquired = acquire_public_traces(
                self.controller,
                self.task,
                initial,
            ).state
            empty_scores = self.controller.score_actions(
                self.task,
                initial,
                include_reversible_transition=False,
            ).logits
            acquired_scores = self.controller.score_actions(
                self.task,
                acquired,
                include_reversible_transition=False,
            ).logits
            acquired_full = self.controller.score_actions(self.task, acquired)
        self.assertTrue(
            torch.equal(
                initial_scores.associative_recall_logits,
                torch.zeros_like(initial_scores.associative_recall_logits),
            )
        )
        self.assertGreater(
            float(acquired_full.associative_recall_logits.abs().max().item()),
            0.0,
        )
        self.assertTrue(torch.equal(empty_scores, acquired_scores))


class GlyphMachineLearningAndCheckpointTests(unittest.TestCase):
    def test_single_unique_trace_still_has_post_write_learning_term(self) -> None:
        commitment = glyph_machine_mechanism_partition("development")[0]
        task = make_glyph_machine_trace_stream(
            108_181,
            surface_seed=208_181,
            supports=1,
            queries=1,
            observations_per_support=1,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        ).supports[0].learner
        observation = task.observations[0]
        task = replace(
            task,
            observations=(
                replace(observation, transitions=(observation.transitions[0],)),
            ),
        )
        controller = build_glyph_machine_controller("smoke")
        acquisition, objective, unique_events, identifiable_events = (
            acquire_and_score_public_traces(
                controller,
                task,
                controller.initial_state(),
            )
        )
        self.assertEqual(unique_events, 1)
        self.assertEqual(identifiable_events, 0)
        self.assertTrue(objective.requires_grad)
        self.assertEqual(acquisition.public_transitions, 1)

    def test_trace_objective_weights_u_and_identifiable_i_per_event(self) -> None:
        torch.manual_seed(108_191)
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=4, actions=2)
        unique = runner._unique_public_transitions(task)
        self.assertGreater(len(unique), 1)
        initial = controller.initial_state()
        _, _, unprimed_u, unprimed_i = acquire_and_score_public_traces(
            controller,
            task,
            initial,
        )
        self.assertEqual(unprimed_u, len(unique))
        self.assertEqual(unprimed_i, 0)

        # A previous public pass makes each exact source/action successor
        # identifiable without consulting reversibility or generated internals.
        history = acquire_public_traces(controller, task, initial).state
        acquired_calls: list[tuple[tuple[str, str, str], ...]] = []
        loss_calls: list[tuple[tuple[str, str, str], ...]] = []
        original_acquire = runner._acquire_transition_sequence
        post_values = torch.arange(
            1,
            len(unique) + 1,
            dtype=history.keys.dtype,
            device=history.keys.device,
            requires_grad=True,
        )
        loo_values = tuple(
            post_values.new_tensor((10.0 + index,), requires_grad=True)
            for index in range(len(unique))
        )
        scripted_losses = iter((post_values, *loo_values))

        def observed_acquire(*args, **kwargs):
            transitions = tuple(args[3])
            acquired_calls.append(
                tuple(
                    (
                        value.before.digest,
                        value.action.schema.digest,
                        value.after.digest,
                    )
                    for value in transitions
                )
            )
            return original_acquire(*args, **kwargs)

        def observed_loss(*args, **kwargs):
            transitions = tuple(kwargs["transitions"])
            loss_calls.append(
                tuple(
                    (
                        value.before.digest,
                        value.action.schema.digest,
                        value.after.digest,
                    )
                    for value in transitions
                )
            )
            return next(scripted_losses)

        with (
            mock.patch.object(
                runner,
                "_acquire_transition_sequence",
                side_effect=observed_acquire,
            ),
            mock.patch.object(
                controller,
                "public_trace_losses",
                side_effect=observed_loss,
            ),
        ):
            _, objective, unique_events, identifiable_events = (
                acquire_and_score_public_traces(
                controller,
                task,
                history,
                )
            )
        self.assertTrue(objective.requires_grad)
        self.assertEqual(unique_events, len(unique))
        self.assertEqual(identifiable_events, len(unique))
        expected = (post_values.sum() + sum(value[0] for value in loo_values)) / (
            unique_events + identifiable_events
        )
        self.assertTrue(torch.equal(objective, expected))
        self.assertEqual(len(acquired_calls[0]), len(unique))
        for retained, target in zip(
            acquired_calls[1:], loss_calls[1:], strict=True
        ):
            self.assertEqual(len(target), 1)
            self.assertNotIn(target[0], retained)

    def test_public_identifiability_rejects_conflicting_successors(self) -> None:
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=4, actions=2)
        target = next(
            value
            for value in runner._unique_public_transitions(task)
            if value.applied
        )
        alternative_after = next(
            value
            for value in task.states
            if value.digest not in (target.before.digest, target.after.digest)
        )
        conflicting = replace(target, after=alternative_after)
        consistent_state = runner._acquire_transition_sequence(
            controller,
            task,
            controller.initial_state(),
            (target,),
        ).state
        conflicting_state = runner._acquire_transition_sequence(
            controller,
            task,
            consistent_state,
            (conflicting,),
        ).state
        source_action_id, successor_id = runner._public_transition_ids(
            target,
            device=consistent_state.keys.device,
        )
        self.assertTrue(
            controller.memory.public_trace_is_identifiable(
                consistent_state,
                source_action_id,
                successor_id,
            )
        )
        self.assertFalse(
            controller.memory.public_trace_is_identifiable(
                conflicting_state,
                source_action_id,
                successor_id,
            )
        )

    def test_public_trace_and_scalar_losses_have_differentiable_neural_paths(self) -> None:
        torch.manual_seed(108_201)
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=4, actions=2)
        state = controller.initial_state()

        acquisition, trace_loss, unique_events, identifiable_events = (
            acquire_and_score_public_traces(controller, task, state)
        )
        self.assertTrue(trace_loss.requires_grad)
        self.assertGreater(unique_events, 0)
        self.assertEqual(identifiable_events, 0)
        trace_loss.backward()
        modules = {
            "key": controller.event_key_encoder,
            "value": controller.trace_value_encoder,
            "condition": controller.causal_transition.condition_gate,
            "transition": controller.causal_transition,
        }
        for label, module in modules.items():
            gradient = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, msg=label)

        controller.zero_grad(set_to_none=True)
        policy_state = restore_glyph_state(snapshot_glyph_state(acquisition.state))
        torch.manual_seed(108_211)
        first = rollout_glyph_procedure(
            controller, task, policy_state, greedy=False
        )
        torch.manual_seed(108_213)
        second = rollout_glyph_procedure(
            controller, task, policy_state, greedy=False
        )
        attempt_loss = centered_trajectory_preference_loss(
            (first, second), (1.0, 0.0)
        )
        self.assertTrue(attempt_loss.requires_grad)
        attempt_loss.backward()
        policy_gradient = sum(
            float(parameter.grad.detach().abs().sum().item())
            for module in (
                controller.procedure_reasoner,
                controller.stop_head,
            )
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(policy_gradient, 0.0)

    def test_public_suffix_loss_uses_production_action_stop_logits_and_one_action(self) -> None:
        torch.manual_seed(108_205)
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(actions=1)
        state = acquire_public_traces(
            controller,
            task,
            controller.initial_state(),
        ).state
        production = controller._score_actions_from_lattice
        with mock.patch.object(
            controller,
            "_score_actions_from_lattice",
            wraps=production,
        ) as shared_scores:
            group_losses = controller.public_backward_reasoning_losses(task, state)

        # Prefixes and endpoints are separately averaged, even with only one
        # declared action.  Both compete against the production STOP logit.
        self.assertEqual(group_losses.shape, (2,))
        self.assertGreater(shared_scores.call_count, len(task.observations))
        self.assertTrue(group_losses.requires_grad)
        controller.zero_grad(set_to_none=True)
        group_losses.mean().backward()
        stop_gradient = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in controller.stop_head.parameters()
            if parameter.grad is not None
        )
        action_gradient = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in controller.procedure_reasoner.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(stop_gradient, 0.0)
        self.assertGreater(action_gradient, 0.0)

    def test_tiny_public_fit_learns_nonterminal_action_and_endpoint_stop(self) -> None:
        torch.manual_seed(108_502)
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=3, actions=2)
        optimizer = torch.optim.Adam(controller.parameters(), lr=5.0e-3)
        for _ in range(12):
            state = acquire_public_traces(
                controller,
                task,
                controller.initial_state(),
            ).state
            objective = controller.public_backward_reasoning_loss(task, state)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()

        state = acquire_public_traces(
            controller,
            task,
            controller.initial_state(),
        ).state
        encoded = controller.encode_task(task)
        state_indices = {
            value.digest: index for index, value in enumerate(task.states)
        }
        action_indices = {
            value.digest: index for index, value in enumerate(task.actions)
        }
        prefix_margins = []
        endpoint_margins = []
        with torch.no_grad():
            for observation in task.observations:
                transitions = observation.transitions
                suffix_start = max(0, len(transitions) - runner._MAX_REASONING_STEPS)
                goal_index = state_indices[transitions[-1].after.digest]
                for start_index in range(suffix_start, len(transitions)):
                    transition = transitions[start_index]
                    before_index = state_indices[transition.before.digest]
                    if before_index == goal_index:
                        continue
                    belief = F.one_hot(
                        torch.tensor(before_index),
                        len(task.states),
                    ).to(dtype=encoded.state_embeddings.dtype)
                    scores = controller.score_actions(
                        task,
                        state,
                        current_state_belief=belief,
                        goal_state_index=goal_index,
                        steps_remaining=len(transitions) - start_index,
                        encoding=encoded,
                    )
                    action_index = action_indices[transition.action.schema.digest]
                    prefix_margins.append(
                        float((scores.action_logits[action_index] - scores.stop_logit).item())
                    )
                endpoint = F.one_hot(
                    torch.tensor(goal_index),
                    len(task.states),
                ).to(dtype=encoded.state_embeddings.dtype)
                scores = controller.score_actions(
                    task,
                    state,
                    current_state_belief=endpoint,
                    goal_state_index=goal_index,
                    steps_remaining=1,
                    encoding=encoded,
                )
                endpoint_margins.append(
                    float((scores.stop_logit - scores.action_logits.max()).item())
                )

        self.assertTrue(prefix_margins)
        self.assertTrue(endpoint_margins)
        self.assertGreater(min(prefix_margins), 1.0)
        self.assertGreater(min(endpoint_margins), 1.0)

    def test_resource_local_fit_cannot_collapse_public_symbol_identity(self) -> None:
        torch.manual_seed(108_701)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = build_glyph_machine_controller("resource_graph", device=device)
        task = _task_with(states=3, actions=3)
        optimizer = torch.optim.Adam(controller.parameters(), lr=5.0e-3)

        before = controller.encode_task(task)
        before_state_cosine = _maximum_off_diagonal_cosine(before.state_embeddings)
        before_action_cosine = _maximum_off_diagonal_cosine(before.action_embeddings)
        for _ in range(4):
            acquisition, trace_objective, _, _ = acquire_and_score_public_traces(
                controller,
                task,
                controller.initial_state(),
            )
            objective = trace_objective + controller.public_backward_reasoning_loss(
                task,
                acquisition.state,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()

        after = controller.encode_task(task)
        after_state_cosine = _maximum_off_diagonal_cosine(after.state_embeddings)
        after_action_cosine = _maximum_off_diagonal_cosine(after.action_embeddings)
        self.assertLess(before_state_cosine, 0.9)
        self.assertLess(before_action_cosine, 0.9)
        self.assertLess(after_state_cosine, 0.9)
        self.assertLess(after_action_cosine, 0.9)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in (
                    controller.graph_encoder.public_identity_features.parameters()
                )
            ),
            0,
        )

    def test_resource_pair_retrieval_has_decisive_rank_margin_and_mass(self) -> None:
        torch.manual_seed(108_731)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = build_glyph_machine_controller("resource_graph", device=device)
        task = _task_with(states=4, actions=3)
        encoded = controller.encode_task(task)
        transitions = runner._unique_public_transitions(task)
        self.assertLessEqual(len(transitions), controller.memory.trace_slot_count)
        acquired = runner._acquire_transition_sequence(
            controller,
            task,
            controller.initial_state(),
            transitions,
            encoding=encoded,
        ).state
        pair_keys = controller.transition_event_keys(encoded)
        read = controller.memory.read(
            pair_keys.reshape(-1, controller.profile.width),
            acquired,
            lane="trace",
        )
        lattice = controller.transition_lattice(encoded, acquired)
        state_indices = {
            value.digest: index for index, value in enumerate(task.states)
        }
        action_indices = {
            value.digest: index for index, value in enumerate(task.actions)
        }
        occupied = acquired.occupied[0, : controller.memory.trace_slot_count]
        retained_keys = F.normalize(
            acquired.keys[0, : controller.memory.trace_slot_count][occupied],
            dim=-1,
        )
        margins = []
        masses = []
        successor_masses = []
        for slot, transition in enumerate(transitions):
            state_index = state_indices[transition.before.digest]
            action_index = action_indices[transition.action.schema.digest]
            flat_index = state_index * len(task.actions) + action_index
            similarities = (
                F.normalize(pair_keys[state_index, action_index], dim=-1)
                @ retained_keys.transpose(0, 1)
            )
            alternatives = torch.cat(
                (similarities[:slot], similarities[slot + 1 :])
            )
            if alternatives.numel():
                margins.append(float((similarities[slot] - alternatives.max()).item()))
            masses.append(
                float(read.attention_weights[0, flat_index, slot].item())
            )
            successor = state_indices[transition.after.digest]
            successor_masses.append(
                float(
                    torch.softmax(
                        lattice.associative_recall_logits[state_index, action_index],
                        dim=-1,
                    )[successor].item()
                )
            )

        self.assertGreaterEqual(min(margins, default=1.0), 0.10)
        self.assertGreaterEqual(min(masses), 0.90)
        self.assertGreaterEqual(min(successor_masses), 0.75)

    def test_preference_rollouts_share_incoming_state_and_defer_feedback(self) -> None:
        torch.manual_seed(108_217)
        commitment = glyph_machine_mechanism_partition("train")[0]
        pair = make_glyph_machine_trace_stream(
            108_217,
            surface_seed=208_217,
            supports=1,
            queries=1,
            observations_per_support=2,
            mechanism_commitment=commitment,
            mechanism_partition="train",
        ).supports[0]
        controller = build_glyph_machine_controller("smoke")
        state = acquire_public_traces(
            controller,
            pair.learner,
            controller.initial_state(),
        ).state
        incoming_digest = glyph_associative_state_digest(state)
        events: list[str] = []
        rollout_digests: list[str] = []
        feedback_state_digests: list[str] = []
        feedback_bindings: list[str] = []
        rewards = iter((1.0, 0.0))
        original_rollout = runner.rollout_glyph_procedure
        original_feedback = runner.apply_scalar_procedure_feedback

        def observed_rollout(*args, **kwargs):
            result = original_rollout(*args, **kwargs)
            events.append("rollout")
            rollout_digests.append(result.incoming_state_digest)
            return result

        def observed_judge(_pair, _procedure):
            events.append("judge")
            return next(rewards)

        def observed_feedback(*args, **kwargs):
            events.append("feedback")
            feedback_state_digests.append(glyph_associative_state_digest(args[4]))
            feedback_bindings.append(
                glyph_associative_state_digest(kwargs["binding_state"])
            )
            return original_feedback(*args, **kwargs)

        ledger = runner._ScalarJudgeLedger(observed_judge)
        with (
            mock.patch.object(
                runner,
                "rollout_glyph_procedure",
                side_effect=observed_rollout,
            ),
            mock.patch.object(
                runner,
                "apply_scalar_procedure_feedback",
                side_effect=observed_feedback,
            ),
        ):
            updated, preference, accepted = runner._sample_training_preferences(
                controller,
                pair,
                state,
                ledger,
                default_glyph_machine_experiment_config("smoke"),
            )
        self.assertEqual(ledger.calls, 2)
        self.assertEqual(rollout_digests, [incoming_digest, incoming_digest])
        self.assertEqual(
            events,
            ["rollout", "judge", "rollout", "judge", "feedback", "feedback"],
        )
        self.assertEqual(feedback_bindings, [incoming_digest, incoming_digest])
        self.assertEqual(feedback_state_digests[0], incoming_digest)
        self.assertNotEqual(feedback_state_digests[1], incoming_digest)
        self.assertEqual(accepted, 2)
        self.assertTrue(preference.requires_grad)
        self.assertNotEqual(glyph_associative_state_digest(updated), incoming_digest)

    def test_tiny_trace_fit_distinguishes_correct_absent_and_wrong_transition_beliefs(self) -> None:
        torch.manual_seed(108_221)
        controller = build_glyph_machine_controller("smoke")
        commitment = glyph_machine_mechanism_partition("development")[0]
        stream = make_glyph_machine_trace_stream(
            108_221,
            surface_seed=208_221,
            supports=1,
            queries=1,
            observations_per_support=2,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )
        optimizer = torch.optim.Adam(controller.parameters(), lr=5.0e-3)
        for _ in range(12):
            acquisition, objective, _, _ = acquire_and_score_public_traces(
                controller,
                stream.supports[0].learner,
                controller.initial_state(),
            )
            reasoning = controller.public_backward_reasoning_losses(
                stream.supports[0].learner,
                acquisition.state,
            )
            if reasoning.numel():
                objective = objective + reasoning.mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()

        arms = build_glyph_machine_evaluation_arms(stream)
        states = {}
        with torch.no_grad():
            for arm, (controlled, include_transition) in arms.items():
                state = controller.initial_state()
                for pair in controlled.supports:
                    state = acquire_public_traces(
                        controller, pair.learner, state
                    ).state
                states[arm] = (state, include_transition)

            task = stream.supports[0].learner
            action_indices = {
                value.digest: index for index, value in enumerate(task.actions)
            }
            successor_score_rows = {
                arm: []
                for arm in (
                    "correct",
                    "no_trace",
                    "wrong_trace",
                    "reversible_removed",
                )
            }
            for transition in runner._unique_public_transitions(task):
                target_task = replace(
                    task,
                    origin=transition.before,
                    goal=replace(task.goal, required=transition.after.records),
                )
                encoding = controller.encode_task(target_task)
                action_index = action_indices[transition.action.schema.digest]
                goal_index = next(
                    index
                    for index, value in enumerate(target_task.states)
                    if value.digest == transition.after.digest
                )
                successor_scores = {}
                for arm in (
                    "correct",
                    "no_trace",
                    "wrong_trace",
                    "reversible_removed",
                ):
                    state, include_transition = states[arm]
                    successor_scores[arm] = float(
                        controller.score_actions(
                            target_task,
                            state,
                            encoding=encoding,
                            include_reversible_transition=include_transition,
                        ).associative_recall_logits[action_index, goal_index].item()
                    )
                    successor_score_rows[arm].append(successor_scores[arm])
            means = {
                arm: sum(values) / len(values)
                for arm, values in successor_score_rows.items()
            }
            self.assertGreater(means["correct"], means["no_trace"])
            # The wrong arm is never a training target, so this narrow local
            # fit proves content sensitivity but does not pre-judge the
            # experiment-level directional control comparison.
            self.assertNotAlmostEqual(
                means["correct"],
                means["wrong_trace"],
                places=5,
            )
            self.assertGreater(means["correct"], means["reversible_removed"])

    def test_checkpoint_reload_is_strict_and_restores_exact_competence(self) -> None:
        torch.manual_seed(108_203)
        controller = build_glyph_machine_controller("smoke")
        task = _task_with(states=3)
        with torch.no_grad():
            state = acquire_public_traces(
                controller,
                task,
                controller.initial_state(),
            ).state
            expected_logits = controller.score_actions(task, state).logits.clone()
        expected_digest = glyph_associative_state_digest(state)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "glyph-machine.pt"
            save_glyph_checkpoint(checkpoint, controller, state)
            loaded_controller, loaded_state = load_glyph_checkpoint(checkpoint)

        self.assertEqual(glyph_associative_state_digest(loaded_state), expected_digest)
        for expected, actual in zip(
            controller.state_dict().values(),
            loaded_controller.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected.cpu(), actual.cpu()))
        with torch.no_grad():
            actual_logits = loaded_controller.score_actions(task, loaded_state).logits
        self.assertTrue(
            torch.allclose(expected_logits, actual_logits, atol=1e-6, rtol=1e-6)
        )

    def test_v32_checkpoint_rejects_v31_identity(self) -> None:
        controller = build_glyph_machine_controller("smoke")
        self.assertEqual(
            runner._RESULT_VERSION,
            "angler.phase5-glyph-machine-experiment.v3.2",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "glyph-machine.pt"
            save_glyph_checkpoint(checkpoint, controller, controller.initial_state())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(
                payload["version"],
                "angler.phase5-glyph-machine-trace.v3.2",
            )
            payload["version"] = "angler.phase5-glyph-machine-trace.v3.1"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "identity is invalid"):
                load_glyph_checkpoint(checkpoint)

    def test_profiles_share_architecture_and_graph_parameter_band(self) -> None:
        smoke = build_glyph_machine_controller("smoke")
        graph = build_glyph_machine_controller("resource_graph")
        smoke_report = glyph_machine_parameter_report(smoke)
        graph_report = glyph_machine_parameter_report(graph)
        smoke_config = default_glyph_machine_experiment_config("smoke")
        graph_config = default_glyph_machine_experiment_config("resource_graph")

        self.assertEqual(type(smoke), type(graph))
        self.assertEqual(
            (
                smoke_config.train_mechanisms,
                smoke_config.development_mechanisms,
                smoke_config.final_mechanisms,
            ),
            (8, 4, 4),
        )
        self.assertEqual(
            (
                graph_config.train_mechanisms,
                graph_config.development_mechanisms,
                graph_config.final_mechanisms,
            ),
            (64, 16, 16),
        )
        self.assertLess(smoke_report["trainable_parameters"], 1_000_000)
        self.assertGreaterEqual(graph_report["trainable_parameters"], 20_000_000)
        self.assertLessEqual(graph_report["trainable_parameters"], 30_000_000)
        self.assertGreater(
            graph_report["fixed_competence_state_elements"],
            smoke_report["fixed_competence_state_elements"],
        )


class GlyphMachineEndToEndHarnessTests(unittest.TestCase):
    def test_control_arms_match_every_public_field_except_support_evidence(self) -> None:
        commitment = glyph_machine_mechanism_partition("development")[0]
        stream = make_glyph_machine_trace_stream(
            108_301,
            surface_seed=208_301,
            supports=1,
            queries=1,
            observations_per_support=1,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )
        arms = build_glyph_machine_evaluation_arms(stream)

        self.assertEqual(
            set(arms),
            {"correct", "no_trace", "wrong_trace", "reversible_removed"},
        )
        correct_stream, correct_transition = arms["correct"]
        removed_stream, removed_transition = arms["reversible_removed"]
        self.assertIs(correct_stream, removed_stream)
        self.assertTrue(correct_transition)
        self.assertFalse(removed_transition)
        expected_queries = tuple(
            pair.learner.to_canonical() for pair in stream.queries
        )
        for arm, (candidate, _) in arms.items():
            self.assertEqual(candidate.mechanism_commitment, stream.mechanism_commitment)
            self.assertEqual(
                tuple(pair.learner.to_canonical() for pair in candidate.queries),
                expected_queries,
                msg=arm,
            )
            for original, controlled in zip(
                stream.supports,
                candidate.supports,
                strict=True,
            ):
                original_public = original.learner.to_canonical()
                controlled_public = controlled.learner.to_canonical()
                original_public.pop("observations")
                controlled_public.pop("observations")
                self.assertEqual(original_public, controlled_public, msg=arm)

    def test_pair_addresses_do_not_change_with_goal_or_successor_evidence_arm(self) -> None:
        commitment = glyph_machine_mechanism_partition("development")[1]
        stream = make_glyph_machine_trace_stream(
            108_302,
            surface_seed=208_302,
            supports=1,
            queries=1,
            observations_per_support=2,
            mechanism_commitment=commitment,
            mechanism_partition="development",
        )
        controller = build_glyph_machine_controller("smoke")
        arms = build_glyph_machine_evaluation_arms(stream)
        encodings = {
            arm: controller.encode_task(controlled.supports[0].learner)
            for arm, (controlled, _) in arms.items()
        }
        reference = encodings["correct"]
        reference_keys = controller.transition_event_keys(reference)
        for arm, encoding in encodings.items():
            self.assertTrue(
                torch.equal(encoding.pair_key_anchors, reference.pair_key_anchors),
                msg=arm,
            )
            self.assertTrue(
                torch.equal(encoding.stop_key_anchors, reference.stop_key_anchors),
                msg=arm,
            )
            self.assertTrue(
                torch.allclose(
                    controller.transition_event_keys(encoding),
                    reference_keys,
                    atol=1.0e-7,
                    rtol=1.0e-7,
                ),
                msg=arm,
            )

        alternate_goal = next(
            value
            for value in stream.supports[0].learner.states
            if value.records != stream.supports[0].learner.goal.required
        )
        goal_changed = replace(
            stream.supports[0].learner,
            goal=replace(
                stream.supports[0].learner.goal,
                required=alternate_goal.records,
            ),
        )
        changed = controller.encode_task(goal_changed)
        self.assertTrue(
            torch.equal(changed.pair_key_anchors, reference.pair_key_anchors)
        )
        self.assertTrue(
            torch.allclose(
                controller.transition_event_keys(changed),
                reference_keys,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        )

    def test_micro_run_accounts_every_scalar_and_reloads_before_final(self) -> None:
        config = replace(
            default_glyph_machine_experiment_config("smoke", seed=108_303),
            train_mechanisms=1,
            development_mechanisms=1,
            final_mechanisms=1,
            supports_per_mechanism=1,
            queries_per_mechanism=1,
            observations_per_support=1,
        )
        external_calls = 0
        events: list[str] = []
        original_load = runner.load_glyph_checkpoint
        original_evaluate = runner.evaluate_glyph_machine_partition

        def counted_judge(pair, procedure):
            nonlocal external_calls
            external_calls += 1
            return judge_glyph_procedure_attempt(pair, procedure)

        def observed_load(*args, **kwargs):
            events.append("reload")
            return original_load(*args, **kwargs)

        def observed_evaluate(*args, **kwargs):
            partition = kwargs["partition"]
            events.append(partition)
            if partition == "final":
                self.assertIn("reload", events)
            return original_evaluate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            checkpoint_path = Path(directory) / "checkpoint.pt"
            with (
                mock.patch.object(
                    runner,
                    "load_glyph_checkpoint",
                    side_effect=observed_load,
                ),
                mock.patch.object(
                    runner,
                    "evaluate_glyph_machine_partition",
                    side_effect=observed_evaluate,
                ),
            ):
                result = run_glyph_machine_experiment(
                    config,
                    result_path=result_path,
                    checkpoint_path=checkpoint_path,
                    judge=counted_judge,
                )
            serialized = json.loads(result_path.read_text(encoding="utf-8"))

        expected_calls = 28
        self.assertEqual(external_calls, expected_calls)
        self.assertEqual(result["total_scalar_judge_calls"], expected_calls)
        self.assertEqual(result["expected_scalar_judge_calls"], expected_calls)
        self.assertEqual(serialized["total_scalar_judge_calls"], expected_calls)
        self.assertTrue(result["checkpoint_reload_before_final"])
        self.assertTrue(result["final_slow_weights_frozen"])
        self.assertLess(events.index("development"), events.index("reload"))
        self.assertLess(events.index("reload"), events.index("final"))
        self.assertEqual(len(result["development"]["rows"]), 1)
        self.assertEqual(len(result["final"]["rows"]), 1)
        self.assertEqual(result["training"]["policy_attempts"], 4)
        self.assertEqual(result["development"]["trace_only"]["scalar_judge_calls"], 4)
        self.assertEqual(
            result["development"]["sequential_adaptation"]["scalar_judge_calls"],
            8,
        )
        self.assertEqual(
            result["development"]["rows"],
            result["development"]["trace_only"]["rows"],
        )
        self.assertEqual(
            set(result["final"]["rows"][0]),
            {
                "mechanism_commitment",
                "correct",
                "no_trace",
                "wrong_trace",
                "reversible_removed",
                "correct_over_no_trace",
                "correct_over_wrong_trace",
                "reversible_contribution",
            },
        )

    def test_trace_only_mode_never_applies_scalar_feedback(self) -> None:
        config = replace(
            default_glyph_machine_experiment_config("smoke", seed=108_307),
            development_mechanisms=1,
            supports_per_mechanism=1,
            queries_per_mechanism=1,
            observations_per_support=1,
        )
        controller = build_glyph_machine_controller("smoke")
        controller.requires_grad_(False)
        original_feedback = runner.apply_scalar_procedure_feedback
        with mock.patch.object(
            runner,
            "apply_scalar_procedure_feedback",
            wraps=original_feedback,
        ) as feedback:
            result = runner.evaluate_glyph_machine_partition(
                controller,
                config,
                partition="development",
                mechanism_count=1,
            )
        self.assertEqual(result["scalar_judge_calls"], 12)
        self.assertEqual(result["trace_only"]["scalar_judge_calls"], 4)
        self.assertEqual(
            result["sequential_adaptation"]["scalar_judge_calls"], 8
        )
        self.assertEqual(feedback.call_count, 8)


class GlyphMachineBoundaryTests(unittest.TestCase):
    def test_runner_ast_has_no_hidden_access_or_sequence_search_surface(self) -> None:
        source_path = Path(runner.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        forbidden_identifiers = {
            "hidden",
            "transition_rows",
            "bfs",
            "shortest_path",
            "permutations",
            "combinations",
            "cartesian_product",
            "target_procedure",
            "solve",
            "solver",
        }
        identifiers = {
            node.id.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }

        self.assertTrue(forbidden_identifiers.isdisjoint(identifiers))
        self.assertTrue(forbidden_identifiers.isdisjoint(attributes))
        self.assertNotIn("_PERMUTATIONS", source)
        self.assertNotIn("score_glyph_procedure", source)
        self.assertNotIn("query_projection", source)
        self.assertNotIn("key_projection", source)
        self.assertNotIn("associative_recall_query", source)

        memory_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GlyphAssociativeMemory"
        )
        memory_read = next(
            node
            for node in memory_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "read"
        )
        controller_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GlyphMachineController"
        )
        address_methods = [
            memory_read,
            *(
                node
                for node in controller_class.body
                if isinstance(node, ast.FunctionDef)
                and node.name in ("transition_event_keys", "event_query_keys")
            ),
        ]
        forbidden_address_names = {
            "digest",
            "metadata",
            "equal",
            "argmax",
            "search",
            "query_projection",
            "key_projection",
        }
        for method in address_methods:
            method_names = {
                node.id.lower()
                for node in ast.walk(method)
                if isinstance(node, ast.Name)
            }
            method_attributes = {
                node.attr.lower()
                for node in ast.walk(method)
                if isinstance(node, ast.Attribute)
            }
            self.assertTrue(
                forbidden_address_names.isdisjoint(
                    method_names | method_attributes
                ),
                msg=method.name,
            )


if __name__ == "__main__":
    unittest.main()
