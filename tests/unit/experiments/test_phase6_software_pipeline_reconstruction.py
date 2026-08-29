from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
import math
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

import torch

from experiments.evaluators.software_pipeline_reconstruction_suite import (
    make_software_pipeline_stream,
    software_pipeline_mechanism_partition,
)
from experiments.runners import phase6_software_pipeline_reconstruction as runner


def _stream(
    *,
    seed: int = 73_001,
    surface_seed: int | None = 83_001,
    supports_per_motif: int = 1,
):
    commitment = software_pipeline_mechanism_partition("development")[2]
    return make_software_pipeline_stream(
        seed,
        surface_seed=surface_seed,
        supports_per_motif=supports_per_motif,
        queries=1,
        maximum_steps=4,
        mechanism_commitment=commitment,
        mechanism_partition="development",
    )


def _train_stream(
    *,
    seed: int = 91_001,
    surface_seed: int = 92_001,
):
    return make_software_pipeline_stream(
        seed,
        surface_seed=surface_seed,
        supports_per_motif=2,
        queries=1,
        maximum_steps=4,
        mechanism_commitment=software_pipeline_mechanism_partition("train")[0],
        mechanism_partition="train",
    )


def _acquire_supports(
    controller: runner.SoftwarePipelineController,
    stream,
) -> runner.SoftwareReconstructionState:
    state = controller.initial_state()
    for pair in stream.supports:
        state = runner.acquire_public_pipeline_traces(
            controller, pair.learner, state
        ).state
    return state


def _fake_v9_stage_report(stage: str) -> dict[str, object]:
    updates = runner._RELATION_CREDIT_STAGE_UPDATES[stage]
    stream_losses = (0.20,) * 8
    stream_weights = (0.125,) * 8
    row_losses = ((0.20,) * 4,) * 8
    row_weights = ((0.25,) * 4,) * 8
    row_scalars = (0.20,) * 8
    row_effective = (4.0,) * 8
    robust_rows = stage in ("relation", "joint")
    context_history = (0.50,) * updates if stage == "context" else ()
    return {
        "stage": stage,
        "optimizer_steps": updates,
        "streams": updates * 8,
        "rows": updates * 32,
        "first_loss": 0.20,
        "last_loss": 0.20,
        "losses": (0.20,) * updates,
        "stream_losses": (stream_losses,) * updates,
        "stream_gradient_weights": (stream_weights,) * updates,
        "flat_mean_losses": (0.20,) * updates,
        "entropic_terms": (0.20,) * updates,
        "effective_stream_counts": (8.0,) * updates,
        "row_losses": ((row_losses,) * updates if robust_rows else ()),
        "row_gradient_weights": ((row_weights,) * updates if robust_rows else ()),
        "row_flat_mean_losses": ((row_scalars,) * updates if robust_rows else ()),
        "row_entropic_terms": ((row_scalars,) * updates if robust_rows else ()),
        "effective_row_counts": ((row_effective,) * updates if robust_rows else ()),
        "gradient_norms": (0.50,) * updates,
        "mean_gradient_norm": 0.50,
        "context_supported_rows_per_stream": (
            ((4,) * 8,) * updates if stage == "context" else ()
        ),
        "context_supported_rows": (
            (32,) * updates if stage == "context" else ()
        ),
        "context_responsibility_valid_set_mass": context_history,
        "context_responsibility_argmax_in_valid_fraction": context_history,
        "context_null_mass": context_history,
        "context_valid_set_mass": context_history,
        "context_valid_set_real_normalized_mass": context_history,
        "context_valid_set_top_one_fraction": context_history,
        "robust_stream_objective_applied": stage in ("relation", "joint"),
        "robust_row_objective_applied": robust_rows,
        "frozen_parameters_unchanged": True,
    }


def _fake_v12_stage_report(stage: str) -> dict[str, object]:
    updates = runner._RELATION_CREDIT_STAGE_UPDATES[stage]
    block_count = 4 if stage == "relation" else 5
    block_rows = tuple((0.125,) * 8 for _ in range(block_count))
    block_scalars = tuple(0.0 for _ in range(block_count))
    block_geometry = tuple(
        tuple(tuple(0.0 for _ in range(8)) for _ in range(8))
        for _ in range(block_count)
    )
    legacy_start = stage == "relation"
    parameter_blocks = {
        f"block-{index}": (f"parameter-{index}",)
        for index in range(block_count)
    }
    return {
        "stage": stage,
        "optimizer_steps": updates,
        "streams": updates * 8,
        "rows": updates * 32,
        "reference_losses": (0.20,) * updates,
        "meta_losses": (0.10,) * updates,
        "meta_flat_penalties": (0.10,) * updates,
        "meta_robust_penalties": (0.10,) * updates,
        "meta_mean_kl_from_existing_weights": (0.0,) * updates,
        "existing_stream_weights": ((0.125,) * 8,) * updates,
        "applied_block_weights": (block_rows,) * updates,
        "residual_logits": (tuple((0.0,) * 8 for _ in range(block_count)),) * updates,
        "withheld_alignments": (tuple((0.1,) * 8 for _ in range(block_count)),) * updates,
        "block_gradient_norms": (block_rows,) * updates,
        "block_cosine_grams": (block_geometry,) * updates,
        "legacy_negative_alignment_fractions": (block_scalars,) * updates,
        "applied_negative_alignment_fractions": (block_scalars,) * updates,
        "legacy_cancellation_ratios": (tuple(0.5 for _ in range(block_count)),) * updates,
        "applied_cancellation_ratios": (tuple(0.6 for _ in range(block_count)),) * updates,
        "legacy_direction_norms": (block_scalars,) * updates,
        "applied_direction_norms": (block_scalars,) * updates,
        "legacy_direction_digests": ("sha256:legacy",) * updates,
        "applied_direction_digests": ("sha256:applied",) * updates,
        "post_first_existing_weight_trace_digest": "sha256:existing-weights",
        "post_first_applied_weight_trace_digest": "sha256:applied-weights",
        "parameter_blocks": parameter_blocks,
        "trainable_parameter_names": tuple(
            name for names in parameter_blocks.values() for name in names
        ),
        "legacy_first_update_required": legacy_start,
        "first_update_used_legacy_weights": legacy_start,
        "mixer_parameters_changed": True,
        "mixer_initial_digest": "sha256:initial",
        "mixer_terminal_digest": "sha256:terminal",
        "frozen_parameters_unchanged": True,
        "controller_step_mixer_unchanged": True,
        "mixer_step_controller_unchanged": True,
        "public_leave_one_out_folds_per_update": 8,
        "stream_identity_input": False,
        "task_identity_input": False,
        "deterministic_gradient_projection": False,
    }


def _component_row_key(task, candidate):
    component = next(
        value for value in task.components if value.schema == candidate.schema
    )
    edges = [
        (record.arguments[1], record.arguments[2])
        for record in component.incidence
        if record.predicate.endswith(".relates")
    ]
    nodes = sorted({value for edge in edges for value in edge})
    labels = {
        node: (
            sum(target == node for _, target in edges),
            sum(source == node for source, _ in edges),
        )
        for node in nodes
    }
    for _ in range(len(nodes)):
        rows = {
            node: (
                labels[node],
                tuple(sorted(labels[source] for source, target in edges if target == node)),
                tuple(sorted(labels[target] for source, target in edges if source == node)),
            )
            for node in nodes
        }
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(set(rows.values()), key=repr))
        }
        labels = {node: vocabulary[value] for node, value in rows.items()}
    return (
        tuple(sorted(labels.values())),
        tuple(sorted((labels[source], labels[target]) for source, target in edges)),
    )


class SoftwarePipelineControllerShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.stream = _stream()
        self.controller = runner.build_software_pipeline_controller("smoke")

    def test_profiles_and_default_dimensions_are_bounded(self) -> None:
        smoke = runner.default_software_pipeline_experiment_config("smoke")
        resource = runner.default_software_pipeline_experiment_config(
            "resource_graph"
        )

        self.assertEqual(smoke.maximum_steps, 4)
        self.assertEqual(smoke.train_mechanisms, 8)
        self.assertEqual(resource.train_mechanisms, 64)
        self.assertEqual(resource.development_mechanisms, 16)
        self.assertEqual(resource.final_mechanisms, 16)
        self.assertEqual(resource.supports_per_motif, 4)
        self.assertEqual(resource.queries_per_mechanism, 8)
        with self.assertRaises(ValueError):
            replace(smoke, maximum_steps=5)

    def test_dynamic_task_encoding_and_action_stop_shapes(self) -> None:
        support = self.stream.supports[0].learner
        query = self.stream.queries[0].learner
        state = self.controller.initial_state()

        support_encoding = self.controller.encode_task(support)
        query_encoding = self.controller.encode_task(query)
        support_scores = self.controller.score_actions(support, state)
        query_scores = self.controller.score_actions(query, state)

        self.assertEqual(len(support.states), 3)
        self.assertEqual(len(query.states), 9)
        self.assertEqual(
            support_encoding.role_pair_keys.shape,
            (len(support.states), len(support.grounded_candidates), 32),
        )
        self.assertEqual(
            query_encoding.role_pair_keys.shape,
            (len(query.states), len(query.grounded_candidates), 32),
        )
        self.assertEqual(
            support_encoding.relation_context_embeddings.shape,
            (len(support.grounded_candidates), 32),
        )
        self.assertEqual(
            support_encoding.relation_component_embeddings.shape,
            (len(support.grounded_candidates), 32),
        )
        self.assertEqual(
            query_encoding.relation_context_embeddings.shape,
            (len(query.grounded_candidates), 32),
        )
        self.assertEqual(
            query_encoding.relation_component_embeddings.shape,
            (len(query.grounded_candidates), 32),
        )
        self.assertEqual(
            support_scores.logits.shape,
            (len(support.grounded_candidates) + 1,),
        )
        self.assertEqual(
            query_scores.logits.shape,
            (len(query.grounded_candidates) + 1,),
        )
        self.assertTrue(torch.isfinite(query_scores.logits).all())

    def test_two_lanes_are_bounded_and_device_resident(self) -> None:
        state = self.controller.initial_state()
        report = runner.software_pipeline_parameter_report(self.controller)

        self.assertEqual(state.pointer.width, 32)
        self.assertEqual(state.role.width, 32)
        self.assertEqual(state.pointer.keys.device, next(self.controller.parameters()).device)
        self.assertEqual(state.role.keys.device, next(self.controller.parameters()).device)
        self.assertGreater(report["pointer_state_elements"], 0)
        self.assertGreater(report["role_state_elements"], 0)
        self.assertEqual(
            state.context_trace_keys.shape,
            state.role.keys.shape,
        )
        self.assertEqual(
            state.relation_trace_values.shape,
            state.role.keys.shape,
        )
        self.assertEqual(
            report["context_trace_state_elements"],
            state.context_trace_keys.numel(),
        )
        self.assertEqual(
            report["relation_trace_value_state_elements"],
            state.relation_trace_values.numel(),
        )
        self.assertEqual(
            report["factorized_relation_state_elements"],
            state.context_trace_keys.numel() + state.relation_trace_values.numel(),
        )
        self.assertEqual(float(state.context_trace_keys.abs().sum()), 0.0)
        self.assertEqual(float(state.relation_trace_values.abs().sum()), 0.0)
        self.assertEqual(report["complete_pipeline_candidates"], 0)
        self.assertEqual(report["total_parameters"], 265_606)

    def test_evidence_head_preserves_prior_shape_and_rng_consumption(self) -> None:
        head = self.controller.evidence_action_head
        self.assertEqual(tuple(head[0].weight.shape), (64, 1))
        self.assertEqual(tuple(head[0].bias.shape), (64,))
        self.assertIsInstance(head[1], torch.nn.Softplus)
        self.assertEqual(tuple(head[2].weight.shape), (1, 64))
        self.assertIsNone(head[2].bias)
        gate = torch.nn.functional.softplus(
            self.controller.evidence_action_log_gate
        )
        self.assertGreater(float(gate.detach()), 0.0)
        self.assertAlmostEqual(
            float(gate.detach()),
            runner._INITIAL_EVIDENCE_ACTION_GATE,
            places=6,
        )

        with torch.random.fork_rng():
            torch.manual_seed(20_260_827)
            monotone = runner.build_software_pipeline_controller("smoke")
            torch.manual_seed(20_260_827)
            with mock.patch.object(runner.nn, "Softplus", runner.nn.SiLU):
                prior_construction = runner.build_software_pipeline_controller(
                    "smoke"
                )
        monotone_parameters = dict(monotone.named_parameters())
        prior_parameters = dict(prior_construction.named_parameters())
        self.assertEqual(monotone_parameters.keys(), prior_parameters.keys())
        for name, parameter in monotone_parameters.items():
            if name.startswith("evidence_action_head."):
                continue
            self.assertTrue(
                torch.equal(parameter, prior_parameters[name]),
                msg=name,
            )

        parameter_names = tuple(name for name, _ in monotone.named_parameters())
        old_tail = parameter_names.index("procedure_start")
        relation_head = min(
            index
            for index, name in enumerate(parameter_names)
            if name.startswith("relation_")
        )
        self.assertLess(old_tail, relation_head)
        evidence_pair = min(
            index
            for index, name in enumerate(parameter_names)
            if name.startswith("evidence_pair_encoder.")
        )
        self.assertLess(relation_head, evidence_pair)

    def test_incidence_residual_is_appended_after_legacy_evidence_parameters(self) -> None:
        class EmptyEvidenceEncoder(torch.nn.Module):
            def __init__(self, _profile) -> None:
                super().__init__()

        with torch.random.fork_rng():
            torch.manual_seed(20_260_827)
            controller = runner.build_software_pipeline_controller("smoke")
            torch.manual_seed(20_260_827)
            with mock.patch.object(
                runner,
                "EvidenceOrderedPairEncoder",
                EmptyEvidenceEncoder,
            ):
                without_dedicated = runner.build_software_pipeline_controller(
                    "smoke"
                )
        actual = dict(controller.named_parameters())
        baseline = dict(without_dedicated.named_parameters())
        self.assertTrue(
            set(actual) - set(baseline)
            and all(
                name.startswith(
                    ("evidence_pair_encoder.", "evidence_context_encoder.")
                )
                for name in set(actual) - set(baseline)
            )
        )
        for name, parameter in baseline.items():
            if not name.startswith("relation_incidence_"):
                self.assertTrue(torch.equal(parameter, actual[name]), msg=name)
        parameter_names = tuple(actual)
        self.assertLess(
            max(
                index
                for index, name in enumerate(parameter_names)
                if name.startswith(
                    ("evidence_pair_encoder.", "evidence_context_encoder.")
                )
            ),
            min(
                index
                for index, name in enumerate(parameter_names)
                if name.startswith("relation_incidence_")
            ),
        )

    def test_incidence_append_does_not_perturb_legacy_initialization(self) -> None:
        class EmptyIncidenceReadout(torch.nn.Module):
            def __init__(self, _profile) -> None:
                super().__init__()

        with torch.random.fork_rng():
            torch.manual_seed(20_260_828)
            controller = runner.build_software_pipeline_controller("smoke")
            torch.manual_seed(20_260_828)
            with mock.patch.object(
                runner,
                "RelationAxisSetReadout",
                EmptyIncidenceReadout,
            ):
                without_incidence = runner.build_software_pipeline_controller(
                    "smoke"
                )
        actual = dict(controller.named_parameters())
        baseline = dict(without_incidence.named_parameters())
        self.assertTrue(set(actual) - set(baseline))
        self.assertTrue(
            all(
                name.startswith("relation_incidence_readout.")
                for name in set(actual) - set(baseline)
            )
        )
        for name, parameter in baseline.items():
            self.assertTrue(torch.equal(parameter, actual[name]), msg=name)

    def test_factorized_trace_state_rejects_misalignment_and_nonfinite_values(self) -> None:
        state = self.controller.initial_state()
        for field in ("context_trace_keys", "relation_trace_values"):
            with self.subTest(field=field, case="shape"):
                with self.assertRaises(ValueError):
                    replace(
                        state,
                        **{
                            field: torch.zeros(
                                1, 1, self.controller.profile.width
                            )
                        },
                    )
            nonfinite = getattr(state, field).clone()
            nonfinite[0, 0, 0] = float("nan")
            with self.subTest(field=field, case="nonfinite"):
                with self.assertRaises(ValueError):
                    replace(state, **{field: nonfinite})
            unoccupied = getattr(state, field).clone()
            unoccupied[0, 0, 0] = 1.0
            with self.subTest(field=field, case="unoccupied"):
                with self.assertRaises(ValueError):
                    replace(state, **{field: unoccupied})
        occupied_role = replace(
            state.role,
            occupied=state.role.occupied.clone(),
        )
        occupied_role.occupied[0, self.controller.role_memory.trace_slot_count] = True
        for field in ("context_trace_keys", "relation_trace_values"):
            outcome = getattr(state, field).clone()
            outcome[0, self.controller.role_memory.trace_slot_count, 0] = 1.0
            arguments = {
                "pointer": state.pointer,
                "role": occupied_role,
                "context_trace_keys": state.context_trace_keys,
                "relation_trace_values": state.relation_trace_values,
            }
            arguments[field] = outcome
            with self.subTest(field=field, case="outcome"):
                with self.assertRaises(ValueError):
                    runner.SoftwareReconstructionState(**arguments)

    def test_cuda_state_and_gradients_stay_on_device_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        controller = runner.build_software_pipeline_controller(
            "smoke", device="cuda"
        )
        state = controller.initial_state()
        task = self.stream.supports[0].learner
        loss = controller.public_trace_losses(task, state).mean()
        loss.backward()

        self.assertEqual(state.pointer.keys.device.type, "cuda")
        self.assertEqual(state.role.keys.device.type, "cuda")
        self.assertEqual(state.context_trace_keys.device.type, "cuda")
        self.assertEqual(state.relation_trace_values.device.type, "cuda")
        self.assertEqual(loss.device.type, "cuda")
        self.assertTrue(
            all(
                parameter.grad is None or parameter.grad.device.type == "cuda"
                for parameter in controller.parameters()
            )
        )


class SoftwarePipelineRoleAndPointerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.stream = _stream(supports_per_motif=1)
        self.controller = runner.build_software_pipeline_controller("smoke")

    def test_exact_pointer_lane_is_local_and_role_lane_crosses_packages(self) -> None:
        support = self.stream.supports[0].learner
        query = self.stream.queries[0].learner
        acquisition = runner.acquire_public_pipeline_traces(
            self.controller, support, self.controller.initial_state()
        )
        support_scores = self.controller.score_actions(support, acquisition.state)
        query_scores = self.controller.score_actions(query, acquisition.state)

        self.assertGreater(float(support_scores.pointer_contexts.norm().detach()), 0.0)
        self.assertEqual(float(query_scores.pointer_contexts.norm().detach()), 0.0)
        self.assertGreater(float(query_scores.role_contexts.norm().detach()), 0.0)

    def test_factorized_relation_codes_follow_role_trace_slots_and_wrap(self) -> None:
        first_task = self.stream.supports[0].learner
        wrap_task = self.stream.supports[1].learner
        state = self.controller.initial_state()
        expected_contexts = state.context_trace_keys.clone()
        expected_relations = state.relation_trace_values.clone()
        trace_slots = self.controller.role_memory.trace_slot_count
        tasks = (first_task,) * 4 + (wrap_task,)
        for iteration, task in enumerate(tasks):
            cursor = int(state.role.trace_cursor[0].item())
            encoding = self.controller.encode_task(task)
            transitions = runner._public_transitions(task)
            event_contexts = torch.stack(
                tuple(
                    encoding.relation_context_embeddings[
                        task.grounded_candidates.index(transition.action)
                    ]
                    for transition in transitions
                )
            )
            event_relations = torch.stack(
                tuple(
                    encoding.relation_component_embeddings[
                        task.grounded_candidates.index(transition.action)
                    ]
                    for transition in transitions
                )
            )
            write_slots = torch.tensor(
                tuple((cursor + offset) % trace_slots for offset in range(len(transitions))),
                device=state.context_trace_keys.device,
                dtype=torch.long,
            )
            if iteration == 4:
                self.assertEqual(cursor, 0)
                self.assertFalse(
                    torch.equal(state.context_trace_keys[0, write_slots], event_contexts)
                )
                self.assertFalse(
                    torch.equal(
                        state.relation_trace_values[0, write_slots],
                        event_relations,
                    )
                )
            for offset, transition in enumerate(transitions):
                action_index = task.grounded_candidates.index(transition.action)
                expected_contexts[0, (cursor + offset) % trace_slots] = (
                    encoding.relation_context_embeddings[action_index]
                )
                expected_relations[0, (cursor + offset) % trace_slots] = (
                    encoding.relation_component_embeddings[action_index]
                )
            acquired = runner.acquire_public_pipeline_traces(
                self.controller,
                task,
                state,
            )
            self.assertEqual(acquired.role_writes, len(transitions))
            state = acquired.state
            self.assertTrue(
                torch.equal(state.context_trace_keys, expected_contexts)
            )
            self.assertTrue(
                torch.equal(state.relation_trace_values, expected_relations)
            )
            self.assertEqual(
                int(state.role.trace_cursor[0].item()),
                (cursor + len(transitions)) % trace_slots,
            )
        self.assertTrue(state.context_trace_keys.requires_grad)
        self.assertTrue(state.relation_trace_values.requires_grad)
        self.assertFalse(hasattr(state, "context_trace_cursor"))
        self.assertFalse(hasattr(state, "relation_trace_cursor"))
        self.assertEqual(
            tuple(field.name for field in fields(runner.SoftwareReconstructionState)),
            (
                "pointer",
                "role",
                "context_trace_keys",
                "relation_trace_values",
            ),
        )

    def test_failed_acquisition_restores_relation_state_atomically(self) -> None:
        task = self.stream.supports[0].learner
        state = self.controller.initial_state()
        digest = runner.software_reconstruction_state_digest(state)
        with mock.patch.object(
            self.controller,
            "score_actions",
            side_effect=RuntimeError("forced post-write rejection"),
        ):
            acquired = runner.acquire_public_pipeline_traces(
                self.controller,
                task,
                state,
            )
        self.assertEqual(acquired.pointer_writes, 0)
        self.assertEqual(acquired.role_writes, 0)
        self.assertIs(acquired.state, state)
        self.assertEqual(
            runner.software_reconstruction_state_digest(acquired.state),
            digest,
        )

    def test_dense_role_null_is_exact_zero_for_empty_memory(self) -> None:
        queries = torch.randn(5, self.controller.profile.width)
        read = self.controller._dense_role_trace_read(
            queries,
            self.controller.initial_state().role,
        )

        self.assertEqual(float(read.contexts.abs().sum()), 0.0)
        self.assertEqual(float(read.attention_weights.abs().sum()), 0.0)
        self.assertEqual(float(read.evidence_probabilities.abs().sum()), 0.0)
        self.assertEqual(float(read.evidence_logits.abs().sum()), 0.0)
        torch.testing.assert_close(read.null_weights, torch.ones(5))

    def test_dense_role_read_covers_all_slots_and_is_order_invariant(self) -> None:
        stream = _stream(supports_per_motif=2)
        acquired = _acquire_supports(self.controller, stream)
        trace_slots = self.controller.role_memory.trace_slot_count
        self.assertTrue(bool(acquired.role.occupied[0, :trace_slots].all()))
        encoding = self.controller.encode_task(stream.queries[0].learner)
        queries = encoding.role_pair_keys.reshape(-1, self.controller.profile.width)[:4]

        keys = acquired.role.keys.detach().clone().requires_grad_(True)
        values = acquired.role.values.detach().clone().requires_grad_(True)
        differentiable = replace(acquired.role, keys=keys, values=values)
        read = self.controller._dense_role_trace_read(queries, differentiable)
        (read.contexts.sum() + read.evidence_logits.sum()).backward()
        self.assertTrue(bool((keys.grad[0, :trace_slots].norm(dim=-1) > 0.0).all()))
        self.assertTrue(bool((values.grad[0, :trace_slots].norm(dim=-1) > 0.0).all()))

        order = torch.tensor((3, 0, 6, 1, 7, 2, 5, 4))

        def permute_trace(tensor: torch.Tensor) -> torch.Tensor:
            candidate = tensor.detach().clone()
            candidate[:, :trace_slots] = tensor.detach()[:, order]
            return candidate

        permuted = replace(
            acquired.role,
            keys=permute_trace(acquired.role.keys),
            values=permute_trace(acquired.role.values),
            occupied=permute_trace(acquired.role.occupied),
            write_counts=permute_trace(acquired.role.write_counts),
            public_source_action_ids=permute_trace(
                acquired.role.public_source_action_ids
            ),
            public_successor_ids=permute_trace(acquired.role.public_successor_ids),
        )
        original_read = self.controller._dense_role_trace_read(queries, acquired.role)
        permuted_read = self.controller._dense_role_trace_read(queries, permuted)
        torch.testing.assert_close(original_read.contexts, permuted_read.contexts)
        torch.testing.assert_close(
            original_read.evidence_logits,
            permuted_read.evidence_logits,
        )

    def test_dense_role_partial_occupancy_partitions_attention_with_null(self) -> None:
        acquired = runner.acquire_public_pipeline_traces(
            self.controller,
            self.stream.supports[0].learner,
            self.controller.initial_state(),
        ).state
        trace_slots = self.controller.role_memory.trace_slot_count
        occupied = acquired.role.occupied[0, :trace_slots]
        self.assertGreater(int(occupied.sum()), 0)
        self.assertLess(int(occupied.sum()), trace_slots)
        encoding = self.controller.encode_task(self.stream.queries[0].learner)
        queries = encoding.role_pair_keys.reshape(
            -1, self.controller.profile.width
        )[:5]
        read = self.controller._dense_role_trace_read(queries, acquired.role)

        self.assertTrue(bool((read.attention_weights >= 0.0).all()))
        self.assertTrue(bool((read.attention_weights <= 1.0).all()))
        self.assertTrue(bool((read.null_weights > 0.0).all()))
        self.assertTrue(bool((read.null_weights < 1.0).all()))
        self.assertEqual(
            float(read.attention_weights[:, ~occupied].detach().abs().sum()),
            0.0,
        )
        torch.testing.assert_close(
            read.attention_weights.sum(dim=-1) + read.null_weights,
            torch.ones_like(read.null_weights),
        )

    def test_monotone_evidence_contribution_and_role_memory_ablation(self) -> None:
        evidence = torch.linspace(-20.0, 20.0, 257, requires_grad=True)
        contribution = self.controller._evidence_action_contribution(evidence)
        derivative = torch.autograd.grad(contribution.sum(), evidence)[0]

        self.assertTrue(bool((derivative > 0.0).all()))
        self.assertTrue(bool((contribution[1:] > contribution[:-1]).all()))
        self.assertEqual(
            float(
                self.controller._evidence_action_contribution(
                    torch.zeros(7)
                ).detach().abs().sum()
            ),
            0.0,
        )

        query = self.stream.queries[0].learner
        acquired = _acquire_supports(self.controller, self.stream)
        present = self.controller.score_actions(query, acquired)
        removed = self.controller.score_actions(
            query,
            acquired,
            include_role_memory=False,
        )
        self.assertGreater(
            float(present.evidence_match_scores.detach().abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(removed.evidence_match_scores.detach().abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(
                self.controller._evidence_action_contribution(
                    removed.evidence_match_scores
                ).detach().abs().sum()
            ),
            0.0,
        )

    def test_detached_evidence_action_input_is_forward_byte_exact(self) -> None:
        stream = _stream(supports_per_motif=2)
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller,
                evidence.learner,
                state,
            ).state
        masked = replace(heldout, observations=())
        default = self.controller.score_actions(masked, state)
        joint = self.controller.score_actions(
            masked,
            state,
            detach_evidence_action_input=False,
        )
        detached = self.controller.score_actions(
            masked,
            state,
            detach_evidence_action_input=True,
        )
        for name in (
            "logits",
            "action_logits",
            "stop_logit",
            "successor_state_logits",
            "pointer_contexts",
            "role_contexts",
            "outcome_contexts",
            "evidence_match_scores",
            "reasoning_node_codes",
            "current_state_belief",
        ):
            self.assertTrue(
                torch.equal(getattr(default, name), getattr(joint, name)),
                msg=name,
            )
            self.assertTrue(
                torch.equal(getattr(joint, name), getattr(detached, name)),
                msg=name,
            )
        default_losses = self.controller.public_heldout_production_losses(
            heldout,
            state,
        )
        detached_losses = self.controller.public_heldout_production_losses(
            heldout,
            state,
            detach_evidence_action_input=True,
        )
        self.assertTrue(torch.equal(default_losses, detached_losses))

    def test_detached_action_ce_stops_only_direct_evidence_gradient(self) -> None:
        stream = _stream(supports_per_motif=2)
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller,
                evidence.learner,
                state,
            ).state
        masked = replace(heldout, observations=())
        transition = heldout.observations[0].transitions[-1]
        target_index = masked.grounded_candidates.index(transition.action)

        joint = self.controller.score_actions(
            masked,
            state,
            detach_evidence_action_input=False,
        )
        joint_loss = torch.nn.functional.cross_entropy(
            joint.logits.unsqueeze(0),
            torch.tensor((target_index,), dtype=torch.long),
        )
        joint_evidence_gradient = torch.autograd.grad(
            joint_loss,
            joint.evidence_match_scores,
            retain_graph=True,
        )[0]
        self.assertGreater(float(joint_evidence_gradient.abs().sum()), 0.0)

        self.controller.zero_grad(set_to_none=True)
        detached = self.controller.score_actions(
            masked,
            state,
            detach_evidence_action_input=True,
        )
        detached_loss = torch.nn.functional.cross_entropy(
            detached.logits.unsqueeze(0),
            torch.tensor((target_index,), dtype=torch.long),
        )
        detached_evidence_gradient = torch.autograd.grad(
            detached_loss,
            detached.evidence_match_scores,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNone(detached_evidence_gradient)
        detached_loss.backward()
        head_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_action_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(head_gradient, 0.0)
        self.assertIsNotNone(self.controller.evidence_action_log_gate.grad)
        self.assertGreater(
            float(self.controller.evidence_action_log_gate.grad.abs()),
            0.0,
        )

    def test_detached_action_mode_leaves_retrieval_key_gradient_live(self) -> None:
        stream = _stream(supports_per_motif=2)
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller,
                evidence.learner,
                state,
            ).state
        masked = replace(heldout, observations=())
        transition = heldout.observations[0].transitions[-1]
        target_index = masked.grounded_candidates.index(transition.action)
        scores = self.controller.score_actions(
            masked,
            state,
            detach_evidence_action_input=True,
            use_legacy_evidence=True,
        )
        retrieval_ce, retrieval_margin = (
            self.controller._public_retrieval_contrast_losses(
                scores.evidence_match_scores,
                target_index,
            )
        )
        (
            retrieval_ce
            + runner._PUBLIC_RETRIEVAL_MARGIN_WEIGHT * retrieval_margin
        ).backward()
        pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        key_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.local_role_key_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(pair_gradient, 0.0)
        self.assertGreater(key_gradient, 0.0)

    def test_full_scores_are_invariant_to_reversed_support_acquisition(self) -> None:
        stream = _stream(supports_per_motif=2)
        forward = _acquire_supports(self.controller, stream)
        reversed_state = self.controller.initial_state()
        for pair in reversed(stream.supports):
            reversed_state = runner.acquire_public_pipeline_traces(
                self.controller,
                pair.learner,
                reversed_state,
            ).state
        query = stream.queries[0].learner
        forward_scores = self.controller.score_actions(query, forward)
        reversed_scores = self.controller.score_actions(query, reversed_state)

        for name in (
            "logits",
            "action_logits",
            "stop_logit",
            "successor_state_logits",
            "pointer_contexts",
            "role_contexts",
            "outcome_contexts",
            "evidence_match_scores",
            "reasoning_node_codes",
            "current_state_belief",
        ):
            torch.testing.assert_close(
                getattr(forward_scores, name),
                getattr(reversed_scores, name),
                atol=2.0e-5,
                rtol=2.0e-5,
            )

    def test_role_removal_erases_cross_package_state_exactly(self) -> None:
        query = self.stream.queries[0].learner
        acquired = _acquire_supports(self.controller, self.stream)
        empty = self.controller.initial_state()

        removed = self.controller.score_actions(
            query,
            acquired,
            include_role_memory=False,
        )
        empty_scores = self.controller.score_actions(
            query,
            empty,
            include_role_memory=False,
        )

        torch.testing.assert_close(removed.logits, empty_scores.logits)
        self.assertEqual(float(removed.pointer_contexts.norm()), 0.0)
        self.assertEqual(float(removed.role_contexts.norm()), 0.0)

    def test_role_features_are_alpha_rename_invariant(self) -> None:
        rerendered = _stream(seed=73_001, surface_seed=83_002)
        left = self.stream.supports[0].learner
        right = rerendered.supports[0].learner
        left_encoding = self.controller.encode_task(left)
        right_encoding = self.controller.encode_task(right)

        left_state_order = sorted(
            range(len(left.states)), key=lambda index: len(left.states[index].records)
        )
        right_state_order = sorted(
            range(len(right.states)), key=lambda index: len(right.states[index].records)
        )
        left_component_order = sorted(
            range(len(left.grounded_candidates)),
            key=lambda index: _component_row_key(
                left, left.grounded_candidates[index]
            ),
        )
        right_component_order = sorted(
            range(len(right.grounded_candidates)),
            key=lambda index: _component_row_key(
                right, right.grounded_candidates[index]
            ),
        )

        torch.testing.assert_close(
            left_encoding.role_state_embeddings[left_state_order],
            right_encoding.role_state_embeddings[right_state_order],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            left_encoding.role_component_embeddings[left_component_order],
            right_encoding.role_component_embeddings[right_component_order],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            left_encoding.relation_component_embeddings[left_component_order],
            right_encoding.relation_component_embeddings[right_component_order],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            left_encoding.relation_context_embeddings[left_component_order],
            right_encoding.relation_context_embeddings[right_component_order],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        self.assertFalse(
            torch.equal(
                left_encoding.pointer_state_embeddings[left_state_order],
                right_encoding.pointer_state_embeddings[right_state_order],
            )
        )

    def test_role_features_separate_counterfactual_incidence_topology(self) -> None:
        task = self.stream.supports[0].learner
        components = runner._components_in_candidate_order(task)
        features, _ = runner._component_role_features(
            components, task.grounded_candidates
        )
        twin_pairs = []
        for left_index, left in enumerate(components):
            for right_index, right in enumerate(components[left_index + 1 :], left_index + 1):
                if (
                    left.input_type == right.input_type
                    and left.output_type == right.output_type
                    and left.error_type == right.error_type
                    and left.state_reads == right.state_reads
                    and left.state_writes == right.state_writes
                ):
                    twin_pairs.append((left_index, right_index))
        self.assertEqual(len(twin_pairs), 1)
        left_index, right_index = twin_pairs[0]
        # Scalar aggregates are deliberately identical; only learned
        # relational message passing may distinguish the anonymous graphs.
        self.assertEqual(features[left_index], features[right_index])
        encoding = self.controller.encode_task(task)
        self.assertFalse(
            torch.allclose(
                encoding.role_component_embeddings[left_index],
                encoding.role_component_embeddings[right_index],
            )
        )

    def test_multiplex_encoder_learns_over_shared_nodes_without_pair_features(self) -> None:
        task = self.stream.queries[0].learner
        components = runner._components_in_candidate_order(task)
        reference = self.controller.role_encoder.type_codes
        relations = self.controller.role_encoder._multiplex_relation_embeddings(
            components,
            reference,
        )
        output_types = {component.output_type for component in components}
        first_indices = [
            index
            for index, component in enumerate(components)
            if component.input_type not in output_types
        ]
        completion_indices = [
            index for index in range(len(components)) if index not in first_indices
        ]

        self.assertTrue(torch.equal(relations[first_indices], torch.zeros_like(relations[first_indices])))
        self.assertTrue(torch.all(relations[completion_indices].norm(dim=-1) > 0.0))
        for first_index in first_indices:
            twins = [
                index
                for index, component in enumerate(components)
                if component.input_type == components[first_index].output_type
            ]
            self.assertEqual(len(twins), 2)
            self.assertFalse(torch.allclose(relations[twins[0]], relations[twins[1]]))

    def test_ordered_pair_multiplex_is_permutation_invariant_dynamic_and_trainable(self) -> None:
        encoder = self.controller.role_encoder
        predecessor = torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        candidate = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        order = torch.tensor((2, 0, 3, 1))
        encoded = encoder._ordered_pair_multiplex_embedding(
            predecessor, candidate
        )
        permuted = encoder._ordered_pair_multiplex_embedding(
            predecessor[order][:, order],
            candidate[order][:, order],
        )
        five = torch.zeros((5, 5))
        five[torch.arange(5), torch.tensor((1, 2, 3, 4, 0))] = 1.0
        dynamic = encoder._ordered_pair_multiplex_embedding(five, five.T)

        torch.testing.assert_close(encoded, permuted, atol=2.0e-5, rtol=2.0e-5)
        self.assertEqual(dynamic.shape, (self.controller.profile.width,))
        contrasted = encoder._ordered_pair_multiplex_embedding(
            predecessor, candidate.T
        )
        separation = (encoded - contrasted).square().sum()
        self.assertGreater(float(separation.detach()), 1.0e-8)
        separation.backward()
        pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        self.assertGreater(pair_gradient, 0.0)

    def test_evidence_pair_encoder_is_equivariant_invariant_and_directed(self) -> None:
        encoder = self.controller.evidence_pair_encoder
        predecessor = torch.tensor(
            (
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        )
        candidate = torch.tensor(
            (
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        )
        order = torch.tensor((2, 0, 3, 1))
        self.assertEqual(len(encoder.pair_updates), 4)
        pair_states = encoder(predecessor, candidate)
        permuted = encoder(
            predecessor[order][:, order],
            candidate[order][:, order],
        )
        swapped = encoder(candidate, predecessor)

        torch.testing.assert_close(
            permuted,
            pair_states[order][:, order],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        relation_code = self.controller._pool_relation_tensor(pair_states)
        permuted_code = self.controller._pool_relation_tensor(permuted)
        torch.testing.assert_close(
            relation_code,
            permuted_code,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        self.assertFalse(torch.allclose(pair_states, swapped))
        self.assertFalse(
            torch.allclose(
                pair_states,
                self.controller.role_encoder._ordered_pair_multiplex_tensor(
                    predecessor,
                    candidate,
                ),
            )
        )
        self.assertFalse(
            torch.allclose(
                relation_code,
                self.controller._pool_relation_tensor(swapped),
            )
        )

        context = encoder(predecessor, torch.zeros_like(predecessor))
        context_permuted = encoder(
            predecessor[order][:, order],
            torch.zeros_like(predecessor),
        )
        torch.testing.assert_close(
            context_permuted,
            context[order][:, order],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            self.controller._pool_context_tensor(context),
            self.controller._pool_context_tensor(context_permuted),
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_hybrid_relation_readout_is_legacy_exact_and_incidence_sensitive(self) -> None:
        width = self.controller.profile.width
        pair_states = torch.randn(4, 4, width, requires_grad=True)
        node_order = torch.tensor((2, 0, 3, 1))

        def legacy_code(value):
            cells = value.reshape(-1, width)
            attention = torch.softmax(
                self.controller.relation_pool_attention(cells).transpose(0, 1),
                dim=-1,
            )
            features = torch.cat(
                (
                    (attention @ cells).reshape(-1),
                    cells.mean(dim=0),
                    cells.amax(dim=0),
                )
            )
            return torch.nn.functional.normalize(
                self.controller.relation_pool_projection(features),
                dim=-1,
                eps=1.0e-8,
            )

        self.assertTrue(
            torch.equal(
                self.controller.relation_incidence_projection.weight,
                torch.zeros_like(
                    self.controller.relation_incidence_projection.weight
                ),
            )
        )
        relation_code = self.controller._pool_relation_tensor(pair_states)
        self.assertTrue(torch.equal(relation_code, legacy_code(pair_states)))
        permuted_code = self.controller._pool_relation_tensor(
            pair_states[node_order][:, node_order]
        )
        torch.testing.assert_close(
            relation_code,
            permuted_code,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        cell_order = torch.tensor(
            (0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14)
        )
        rearranged = pair_states.reshape(16, width)[cell_order].reshape(
            4,
            4,
            width,
        )
        torch.testing.assert_close(
            legacy_code(pair_states),
            legacy_code(rearranged),
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        incidence_left = self.controller.relation_incidence_readout(pair_states)
        incidence_right = self.controller.relation_incidence_readout(rearranged)
        incidence_contrast = (incidence_left - incidence_right).square().sum()
        self.assertGreater(float(incidence_contrast.detach()), 1.0e-8)

    def test_zero_incidence_residual_opens_then_reaches_axis_features(self) -> None:
        width = self.controller.profile.width
        pair_states = torch.randn(4, 4, width, requires_grad=True)
        weights = torch.linspace(-1.0, 1.0, width)
        relation_code = self.controller._pool_relation_tensor(pair_states)
        weighted_code = (
            relation_code
            * weights.to(device=relation_code.device, dtype=relation_code.dtype)
        ).sum()
        weighted_code.backward()
        self.assertIsNotNone(pair_states.grad)
        self.assertTrue(torch.isfinite(pair_states.grad).all())
        self.assertGreater(float(pair_states.grad.abs().sum()), 0.0)
        incidence_projection_gradient = float(
            self.controller.relation_incidence_projection.weight.grad.abs().sum()
        )
        self.assertGreater(incidence_projection_gradient, 0.0)
        first_readout_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.relation_incidence_readout.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(first_readout_gradient, 0.0)

        with torch.no_grad():
            self.controller.relation_incidence_projection.weight.add_(
                -0.1
                * self.controller.relation_incidence_projection.weight.grad
            )
        self.controller.zero_grad(set_to_none=True)
        pair_states.grad = None
        second_code = self.controller._pool_relation_tensor(pair_states)
        (second_code * weights).sum().backward()
        second_readout_gradients = {
            name: float(parameter.grad.abs().sum())
            for name, parameter in self.controller.relation_incidence_readout.named_parameters()
            if parameter.grad is not None
        }
        for prefix in (
            "row_attention.",
            "column_attention.",
            "node_projection.",
            "node_pool_attention.",
        ):
            self.assertGreater(
                sum(
                    value
                    for name, value in second_readout_gradients.items()
                    if name.startswith(prefix)
                ),
                0.0,
                msg=prefix,
            )

    def test_relation_axis_set_readout_accepts_dynamic_node_sets(self) -> None:
        for node_count in (3, 5, 9):
            pair_states = torch.randn(
                node_count,
                node_count,
                self.controller.profile.width,
            )
            code = self.controller._pool_relation_tensor(pair_states)
            self.assertEqual(code.shape, (self.controller.profile.width,))
            self.assertTrue(torch.isfinite(code).all())
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(code).detach()),
                1.0,
                places=5,
            )

    def test_full_pair_refactor_preserves_old_pool_and_new_pool_is_invariant(self) -> None:
        encoder = self.controller.role_encoder
        predecessor = torch.tensor(
            (
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        )
        candidate = torch.tensor(
            (
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        )
        order = torch.tensor((2, 0, 3, 1))
        pair_states = encoder._ordered_pair_multiplex_tensor(
            predecessor,
            candidate,
        )
        diagonal = pair_states.diagonal(dim1=0, dim2=1).transpose(0, 1)
        prior_pool = encoder.multiplex_pair_pool(
            torch.cat(
                (
                    pair_states.mean(dim=(0, 1)),
                    pair_states.amax(dim=(0, 1)),
                    diagonal.mean(dim=0),
                    diagonal.amax(dim=0),
                )
            )
        )
        refactored = encoder._ordered_pair_multiplex_embedding(
            predecessor,
            candidate,
        )
        permuted_states = encoder._ordered_pair_multiplex_tensor(
            predecessor[order][:, order],
            candidate[order][:, order],
        )

        self.assertTrue(torch.equal(prior_pool, refactored))
        torch.testing.assert_close(
            permuted_states,
            pair_states[order][:, order],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            self.controller._pool_relation_tensor(pair_states),
            self.controller._pool_relation_tensor(permuted_states),
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_relation_pool_trains_dedicated_pair_tensor_not_shared_encoder(self) -> None:
        encoder = self.controller.evidence_pair_encoder
        adjacency = torch.zeros((5, 5))
        adjacency[torch.arange(5), torch.tensor((1, 2, 3, 4, 0))] = 1.0
        pair_states = encoder(
            adjacency,
            adjacency.T,
        )
        self.controller._pool_relation_tensor(pair_states).sum().backward()
        dedicated_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in encoder.parameters()
            if parameter.grad is not None
        )
        shared_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        new_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.named_parameters()
            if name.startswith("relation_pool") and parameter.grad is not None
        )
        self.assertGreater(dedicated_gradient, 0.0)
        self.assertEqual(shared_gradient, 0.0)
        self.assertGreater(new_gradient, 0.0)

    def test_nonzero_operator_direction_changes_factorized_role_key(self) -> None:
        width = self.controller.profile.width
        local = torch.randn(3, 2, width)
        operators = torch.zeros(2, width)
        operators[1] = torch.linspace(-1.0, 1.0, width)
        baseline = self.controller._factorized_role_keys(local, operators)
        changed_operators = operators.clone()
        changed_operators[1] = operators[1].roll(5)
        changed = self.controller._factorized_role_keys(local, changed_operators)

        torch.testing.assert_close(baseline[:, 0], changed[:, 0])
        self.assertFalse(torch.allclose(baseline[:, 1], changed[:, 1]))
        self.assertTrue(bool((baseline.norm(dim=-1) > 0.999).all()))
        self.assertTrue(bool((changed.norm(dim=-1) > 0.999).all()))

    def test_factorized_role_key_retains_centered_operator_anchor(self) -> None:
        width = self.controller.profile.width
        local = torch.randn(5, 3, width)
        operators = torch.randn(3, width)
        keys = self.controller._factorized_role_keys(local, operators)
        centered = torch.nn.functional.normalize(
            torch.nn.functional.layer_norm(operators, (width,)),
            dim=-1,
            eps=1.0e-8,
        ).unsqueeze(0).expand_as(keys)
        cosine = torch.nn.functional.cosine_similarity(keys, centered, dim=-1)
        minimum = 1.0 / (
            1.0 + runner._ROLE_RESIDUAL_LIMIT ** 2
        ) ** 0.5

        self.assertGreaterEqual(
            float(cosine.detach().min()),
            minimum - 2.0e-6,
        )

    def test_local_keys_effects_and_stop_relations_are_alpha_invariant(self) -> None:
        rerendered = _stream(seed=73_001, surface_seed=83_002)
        left = self.stream.supports[0].learner
        right = rerendered.supports[0].learner
        left_encoding = self.controller.encode_task(left)
        right_encoding = self.controller.encode_task(right)

        for left_transition, right_transition in zip(
            left.observations[0].transitions,
            right.observations[0].transitions,
            strict=True,
        ):
            left_before = left.states.index(left_transition.before)
            right_before = right.states.index(right_transition.before)
            left_after = left.states.index(left_transition.after)
            right_after = right.states.index(right_transition.after)
            left_action = left.grounded_candidates.index(left_transition.action)
            right_action = right.grounded_candidates.index(right_transition.action)
            torch.testing.assert_close(
                left_encoding.role_pair_keys[left_before, left_action],
                right_encoding.role_pair_keys[right_before, right_action],
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                left_encoding.relative_effect_embeddings[
                    left_before, left_action, left_after
                ],
                right_encoding.relative_effect_embeddings[
                    right_before, right_action, right_after
                ],
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        left_state_order = sorted(
            range(len(left.states)), key=lambda index: len(left.states[index].records)
        )
        right_state_order = sorted(
            range(len(right.states)), key=lambda index: len(right.states[index].records)
        )
        torch.testing.assert_close(
            left_encoding.stop_relation_embeddings[left_state_order],
            right_encoding.stop_relation_embeddings[right_state_order],
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_alpha_sibling_memory_prefers_matching_role_and_transfers_effect(self) -> None:
        rerendered = _stream(seed=73_001, surface_seed=83_002)
        source = self.stream.supports[0].learner
        sibling = rerendered.supports[0].learner
        acquired = runner.acquire_public_pipeline_traces(
            self.controller,
            source,
            self.controller.initial_state(),
        ).state
        transition = sibling.observations[0].transitions[-1]
        encoding = self.controller.encode_task(sibling)
        components = runner._components_in_candidate_order(sibling)
        correct_index = sibling.grounded_candidates.index(transition.action)
        correct_component = components[correct_index]
        wrong_index = next(
            index
            for index, component in enumerate(components)
            if index != correct_index
            and component.input_type == correct_component.input_type
            and component.output_type == correct_component.output_type
        )
        before_index = sibling.states.index(transition.before)
        after_index = sibling.states.index(transition.after)
        stored_completion_key = acquired.role.keys[0, 1]
        correct_similarity = torch.nn.functional.cosine_similarity(
            encoding.role_pair_keys[before_index, correct_index].unsqueeze(0),
            stored_completion_key.unsqueeze(0),
        )
        wrong_similarity = torch.nn.functional.cosine_similarity(
            encoding.role_pair_keys[before_index, wrong_index].unsqueeze(0),
            stored_completion_key.unsqueeze(0),
        )
        self.assertGreater(
            float((correct_similarity - wrong_similarity).detach()),
            0.0,
        )

        stored_effect = acquired.role.values[0, 1]
        candidate_effects = encoding.relative_effect_embeddings[
            before_index, correct_index
        ]
        similarities = torch.nn.functional.cosine_similarity(
            stored_effect.unsqueeze(0), candidate_effects, dim=-1
        )
        self.assertEqual(int(similarities.argmax().item()), after_index)
        sibling_scores = self.controller.score_actions(sibling, acquired)
        self.assertEqual(float(sibling_scores.pointer_contexts.norm()), 0.0)

    def test_raw_sum_graph_messages_separate_fresh_transforms(self) -> None:
        task = self.stream.queries[0].learner
        components = runner._components_in_candidate_order(task)
        topology = self.controller.role_encoder._incidence_topology_embeddings(
            components,
            self.controller.role_encoder.type_codes,
        )
        output_types = {component.output_type for component in components}
        for first_index, first in enumerate(components):
            if first.input_type in output_types:
                continue
            twins = [
                index
                for index, component in enumerate(components)
                if component.input_type == first.output_type
            ]
            self.assertEqual(len(twins), 2)
            self.assertFalse(torch.allclose(topology[first_index], topology[twins[0]]))
            self.assertFalse(torch.allclose(topology[first_index], topology[twins[1]]))
            self.assertFalse(torch.allclose(topology[twins[0]], topology[twins[1]]))

    def test_presentation_permutation_is_equivariant(self) -> None:
        task = self.stream.queries[0].learner
        reordered = replace(
            task,
            components=tuple(reversed(task.components)),
            grounded_candidates=tuple(reversed(task.grounded_candidates)),
            states=tuple(reversed(task.states)),
        )
        state = self.controller.initial_state()
        original = self.controller.score_actions(task, state)
        permuted = self.controller.score_actions(reordered, state)
        original_by_action = {
            action: original.action_logits[index]
            for index, action in enumerate(task.grounded_candidates)
        }
        permuted_rows = torch.stack(
            [
                permuted.action_logits[
                    reordered.grounded_candidates.index(action)
                ]
                for action in task.grounded_candidates
            ]
        )

        torch.testing.assert_close(
            torch.stack(
                [original_by_action[action] for action in task.grounded_candidates]
            ),
            permuted_rows,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            original.stop_logit,
            permuted.stop_logit,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_permuting_role_values_changes_transition_beliefs(self) -> None:
        query = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        encoding = self.controller.encode_task(query)
        original = self.controller.transition_lattice(encoding, state)
        snapshot = runner.snapshot_software_reconstruction_state(state)
        occupied = snapshot["role.occupied"][0].nonzero().flatten()
        trace_occupied = occupied[
            occupied < self.controller.role_memory.trace_slot_count
        ]
        self.assertGreaterEqual(len(trace_occupied), 2)
        snapshot["role.values"][:, trace_occupied] = snapshot[
            "role.values"
        ][:, trace_occupied.flip(0)]
        permuted_state = runner.restore_software_reconstruction_state(snapshot)
        changed = self.controller.transition_lattice(encoding, permuted_state)

        self.assertFalse(
            torch.allclose(
                original.successor_state_logits,
                changed.successor_state_logits,
            )
        )


class SoftwarePipelineRelationMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.stream = _train_stream()
        self.controller = runner.build_software_pipeline_controller("smoke")

    def test_fixed_fit_and_gate_exposure_plan_is_train_only(self) -> None:
        plan = runner.public_relation_fit_plan()
        self.assertEqual(plan["protocol_id"], runner._RELATION_PROTOCOL_ID)
        self.assertEqual(plan["protocol_id"], "phase6.public-relation-credit.v11")
        self.assertEqual(plan["partition"], "train")
        self.assertEqual(plan["initialization_seed"], 2_026_082_891)
        self.assertEqual(len(plan["commitments"]), 8)
        self.assertEqual(
            plan["stage_updates"],
            {"relation": 80, "context": 25, "joint": 35},
        )
        self.assertEqual(plan["streams_per_update"], 8)
        batches = plan["stage_seed_batches"]
        self.assertEqual(tuple(batches), ("relation", "context", "joint"))
        self.assertEqual(tuple(len(batches[name]) for name in batches), (80, 25, 35))
        self.assertEqual(plan["relation_terminal_gate_update"], 80)
        self.assertEqual(plan["relation_intermediate_selection"], "none")
        self.assertFalse(plan["relation_replay"])
        self.assertTrue(
            all(len(batch) == 8 for stage in batches.values() for batch in stage)
        )
        update_offset = 0
        for stage_name, stage_batches in batches.items():
            for stage_update, batch in enumerate(stage_batches):
                global_update = update_offset + stage_update
                for commitment_index, pair in enumerate(batch):
                    offset = 100_000 * global_update + 1_000 * commitment_index
                    self.assertEqual(
                        pair,
                        (1_901_000_001 + offset, 2_001_000_001 + offset),
                    )
            update_offset += plan["stage_updates"][stage_name]
        training_pairs = {
            pair
            for stage in batches.values()
            for batch in stage
            for pair in batch
        }
        self.assertEqual(len(training_pairs), 1_120)
        relation_panel = set(plan["relation_context_panel_seed_pairs"])
        final_panel = set(plan["final_panel_seed_pairs"])
        self.assertEqual(len(relation_panel), 8)
        self.assertEqual(len(final_panel), 8)
        self.assertEqual(
            plan["relation_context_panel_seed_pairs"],
            tuple(
                (2_021_000_001 + 1_000 * index, 2_031_000_001 + 1_000 * index)
                for index in range(8)
            ),
        )
        self.assertEqual(
            plan["final_panel_seed_pairs"],
            tuple(
                (2_041_000_001 + 1_000 * index, 2_051_000_001 + 1_000 * index)
                for index in range(8)
            ),
        )
        self.assertFalse(training_pairs & relation_panel)
        self.assertFalse(training_pairs & final_panel)
        self.assertFalse(relation_panel & final_panel)
        historical_train_intervals = (
            (221_000_001, 230_907_001),
            (321_000_001, 330_907_001),
            (241_000_001, 250_907_001),
            (341_000_001, 350_907_001),
            (261_000_001, 270_907_001),
            (361_000_001, 370_907_001),
            (281_000_001, 294_907_001),
            (381_000_001, 394_907_001),
            (301_000_001, 314_907_001),
            (401_000_001, 414_907_001),
            (70_000_000, 89_999_999),
            (900_000_000, 999_999_999),
            (1_101_000_001, 1_114_907_001),
            (1_201_000_001, 1_214_907_001),
            (1_501_000_001, 1_514_907_001),
            (1_601_000_001, 1_614_907_001),
        )
        historical_panel_scalars = {
            base + 1_000 * index
            for base in (
                421_000_001,
                431_000_001,
                521_000_001,
                531_000_001,
                441_000_001,
                451_000_001,
                541_000_001,
                551_000_001,
                461_000_001,
                471_000_001,
                561_000_001,
                571_000_001,
                481_000_001,
                491_000_001,
                581_000_001,
                591_000_001,
                501_000_001,
                511_000_001,
                601_000_001,
                611_000_001,
                1_301_000_001,
                1_311_000_001,
                1_401_000_001,
                1_411_000_001,
                1_701_000_001,
                1_711_000_001,
                1_801_000_001,
                1_811_000_001,
            )
            for index in range(8)
        }
        current_scalars = {
            seed
            for pair in training_pairs | relation_panel | final_panel
            for seed in pair
        }
        for seed in current_scalars:
            self.assertNotIn(seed, historical_panel_scalars)
            self.assertFalse(
                any(start <= seed <= end for start, end in historical_train_intervals),
                msg=seed,
            )
        self.assertEqual(
            plan["stream_objective"],
            {
                "relation": "anonymous_entropic_worst_stream",
                "context": "supported_valid_set_row_mean",
                "joint": "anonymous_entropic_worst_stream",
                "temperature": 0.05,
                "mean_weight": 0.5,
                "robust_weight": 0.5,
                "minimum_robust_gradient_weight_per_stream": 1.0 / 16.0,
                "context_stream_weighting": "supported_row_count_fraction",
            },
        )
        self.assertEqual(
            plan["row_objective"],
            {
                "relation": "anonymous_entropic_worst_row_within_stream",
                "context": "supported_valid_set_rows_only",
                "joint": "anonymous_entropic_worst_row_within_stream",
                "temperature": 0.05,
                "mean_weight": 0.5,
                "robust_weight": 0.5,
                "minimum_gradient_weight_per_row": 0.125,
            },
        )
        self.assertEqual(
            plan["valid_witness_set"]["context_denominator"],
            "supported_rows_only",
        )
        self.assertEqual(
            plan["valid_witness_set"]["context_mass"],
            "sum_all_valid_real_slots",
        )
        self.assertTrue(
            plan["valid_witness_set"]["same_slot_conjunction"]
        )
        self.assertEqual(
            plan["valid_witness_set"]["context_training_loss"],
            "negative_log_valid_set_mass",
        )
        self.assertEqual(
            plan["valid_witness_set"]["context_training_rows"],
            "supported_rows_only",
        )
        self.assertEqual(
            plan["valid_witness_set"]["context_training_runtime_effect"],
            "none",
        )
        first_batch = runner._relation_credit_stream_batches(
            plan["commitments"],
            plan["stage_seed_batches"]["relation"][:1],
        )
        self.assertEqual(len(first_batch), 1)
        self.assertEqual(len(first_batch[0]), 8)
        self.assertEqual(
            tuple(stream.mechanism_commitment for stream in first_batch[0]),
            plan["commitments"],
        )
        duplicate_pair = (plan["stage_seed_batches"]["relation"][0][0],) * 8
        with self.assertRaises(ValueError):
            runner._relation_credit_stream_batches(
                plan["commitments"],
                (duplicate_pair,),
            )
        legacy_plan = runner._legacy_public_relation_fit_plan()
        with self.assertRaises(RuntimeError):
            runner._calibrate_public_relation_actions_v2_audit_only(
                self.controller,
                {"raw_gate_passed": False, "plan": legacy_plan},
            )
        with self.assertRaises(RuntimeError):
            runner._calibrate_public_relation_actions_v2_audit_only(
                self.controller,
                {
                    "raw_gate_passed": True,
                    "plan": {**legacy_plan, "fit_rows": 1},
                },
            )

    def test_v9_orchestrator_enforces_stage_order_and_uses_no_control_stream(self) -> None:
        controller = runner.build_public_relation_credit_controller()
        stages = []

        def fake_fit(_controller, _batches, *, stage, **_kwargs):
            stages.append(stage)
            return _fake_v9_stage_report(stage)

        relation_panel = {
            "streams": 8,
            "rows": 32,
            "relation_supported_rows": 24,
            "streams_with_three_supported_rows": 6,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 4, 0, 0),
            "valid_slot_count_histogram": (8, 24, 0, 0),
            "context_valid_set_top_one_fraction_supported": 0.80,
            "context_valid_set_mass_mean_supported": 0.60,
        }
        final_panel = {
            **relation_panel,
            "positive_margin_mean": 0.10,
            "negative_margin_mean": -0.10,
            "separation_mean": 0.20,
            "signed_rows": 26,
            "streams_with_three_signed_rows": 7,
        }
        invariants = {
            "permutation_covariant": True,
            "empty_memory_zero_exact": True,
            "permutation_max_delta": 0.0,
        }
        with (
            mock.patch.object(
                runner,
                "_relation_credit_stream_batches",
                return_value=((self.stream,),),
            ),
            mock.patch.object(
                runner,
                "_relation_credit_panel_streams",
                return_value=(self.stream,) * 8,
            ),
            mock.patch.object(
                runner,
                "_fit_public_relation_credit_batches",
                side_effect=fake_fit,
            ),
            mock.patch.object(
                runner,
                "evaluate_public_relation_credit_panel",
                side_effect=(relation_panel, relation_panel, final_panel),
            ),
            mock.patch.object(
                runner,
                "_evaluate_public_relation_credit_invariants",
                return_value=invariants,
            ),
            mock.patch.object(
                runner,
                "make_software_pipeline_control_stream",
                side_effect=AssertionError("control stream entered v9 training"),
            ),
        ):
            report = runner.fit_public_relation_matcher(controller)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(stages, ["relation", "context", "joint"])
        self.assertEqual(report["wrong_evidence_training_streams"], 0)
        self.assertFalse(report["development_or_final_access"])

    def test_v9_clean_start_rejects_dirty_weights_and_wrong_profile(self) -> None:
        torch.manual_seed(98_765)
        rng_before = torch.get_rng_state().clone()
        clean = runner.build_public_relation_credit_controller()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(runner._public_relation_credit_controller_is_fresh(clean))

        dirty = runner.build_public_relation_credit_controller()
        with torch.no_grad():
            next(dirty.parameters()).view(-1)[0].add_(0.125)
        self.assertFalse(runner._public_relation_credit_controller_is_fresh(dirty))

        wrong_profile = replace(clean.profile, name="same-shape-wrong-profile")
        wrong = runner.SoftwarePipelineController(wrong_profile)
        wrong.load_state_dict(clean.state_dict(), strict=True)
        self.assertFalse(runner._public_relation_credit_controller_is_fresh(wrong))

    def test_v9_valid_set_requires_same_slot_and_sums_every_valid_slot(self) -> None:
        dtype = torch.float64
        empty = runner._relation_valid_set_metrics(
            torch.tensor((0.06, 0.04, 0.20), dtype=dtype),
            torch.tensor((-0.04, -0.06, 0.05), dtype=dtype),
            torch.tensor((0.20, 0.20, 0.20), dtype=dtype),
            torch.tensor(0.40, dtype=dtype),
        )
        self.assertFalse(empty["relation_supported"])
        self.assertEqual(empty["valid_slot_count"], 0)
        self.assertEqual(empty["context_valid_set_mass"], 0.0)
        self.assertFalse(empty["context_valid_set_top_one"])

        multiple = runner._relation_valid_set_metrics(
            torch.tensor((0.05, 0.20, 0.04), dtype=dtype),
            torch.tensor((-0.05, -0.10, -0.20), dtype=dtype),
            torch.tensor((0.32, 0.28, 0.10), dtype=dtype),
            torch.tensor(0.30, dtype=dtype),
        )
        self.assertEqual(multiple["valid_mask"].tolist(), [True, True, False])
        self.assertEqual(multiple["valid_slot_count"], 2)
        self.assertTrue(multiple["relation_supported"])
        self.assertAlmostEqual(multiple["context_valid_set_mass"], 0.60)
        self.assertAlmostEqual(multiple["context_null_mass"], 0.30)
        self.assertTrue(multiple["context_valid_set_top_one"])

    def test_v9_valid_set_boundaries_ties_null_and_permutation(self) -> None:
        dtype = torch.float64
        exact = runner._relation_valid_set_metrics(
            torch.tensor((0.05,), dtype=dtype),
            torch.tensor((-0.05,), dtype=dtype),
            torch.tensor((0.51,), dtype=dtype),
            torch.tensor(0.49, dtype=dtype),
        )
        self.assertTrue(exact["relation_supported"])
        self.assertTrue(exact["context_valid_set_top_one"])

        below_positive = runner._relation_valid_set_metrics(
            torch.tensor((math.nextafter(0.05, -math.inf),), dtype=dtype),
            torch.tensor((-0.05,), dtype=dtype),
            torch.tensor((0.51,), dtype=dtype),
            torch.tensor(0.49, dtype=dtype),
        )
        above_negative = runner._relation_valid_set_metrics(
            torch.tensor((0.05,), dtype=dtype),
            torch.tensor((math.nextafter(-0.05, math.inf),), dtype=dtype),
            torch.tensor((0.51,), dtype=dtype),
            torch.tensor(0.49, dtype=dtype),
        )
        self.assertFalse(below_positive["relation_supported"])
        self.assertFalse(above_negative["relation_supported"])

        valid_invalid_tie = runner._relation_valid_set_metrics(
            torch.tensor((0.10, 0.04), dtype=dtype),
            torch.tensor((-0.10, -0.10), dtype=dtype),
            torch.tensor((0.40, 0.40), dtype=dtype),
            torch.tensor(0.20, dtype=dtype),
        )
        valid_null_tie = runner._relation_valid_set_metrics(
            torch.tensor((0.10, 0.04), dtype=dtype),
            torch.tensor((-0.10, -0.10), dtype=dtype),
            torch.tensor((0.375, 0.25), dtype=dtype),
            torch.tensor(0.375, dtype=dtype),
        )
        null_dominant = runner._relation_valid_set_metrics(
            torch.tensor((0.10, 0.04), dtype=dtype),
            torch.tensor((-0.10, -0.10), dtype=dtype),
            torch.tensor((0.30, 0.20), dtype=dtype),
            torch.tensor(0.50, dtype=dtype),
        )
        self.assertFalse(valid_invalid_tie["context_valid_set_top_one"])
        self.assertFalse(valid_null_tie["context_valid_set_top_one"])
        self.assertFalse(null_dominant["context_valid_set_top_one"])

        positive = torch.tensor((0.20, 0.05, 0.04), dtype=dtype)
        negative = torch.tensor((-0.10, -0.05, -0.20), dtype=dtype)
        weights = torch.tensor((0.28, 0.32, 0.10), dtype=dtype)
        null_weight = torch.tensor(0.30, dtype=dtype)
        original = runner._relation_valid_set_metrics(
            positive,
            negative,
            weights,
            null_weight,
        )
        order = torch.tensor((2, 0, 1))
        permuted = runner._relation_valid_set_metrics(
            positive[order],
            negative[order],
            weights[order],
            null_weight,
        )
        self.assertEqual(
            original["valid_slot_count"],
            permuted["valid_slot_count"],
        )
        self.assertEqual(
            original["context_valid_set_top_one"],
            permuted["context_valid_set_top_one"],
        )
        self.assertAlmostEqual(
            original["context_valid_set_mass"],
            permuted["context_valid_set_mass"],
        )
        self.assertAlmostEqual(
            original["context_null_mass"],
            permuted["context_null_mass"],
        )

    def test_v9_equal_valid_witnesses_do_not_require_a_unique_winner(self) -> None:
        metrics = runner._relation_valid_set_metrics(
            torch.tensor((0.10, 0.10, 0.04), dtype=torch.float64),
            torch.tensor((-0.10, -0.10, -0.20), dtype=torch.float64),
            torch.tensor((0.34, 0.34, 0.00), dtype=torch.float64),
            torch.tensor(0.32, dtype=torch.float64),
        )
        self.assertEqual(metrics["valid_slot_count"], 2)
        self.assertTrue(metrics["relation_supported"])
        self.assertAlmostEqual(metrics["context_valid_set_mass"], 0.68)
        self.assertTrue(metrics["context_valid_set_top_one"])

    def test_v9_context_training_uses_detached_valid_set_mass_only(self) -> None:
        dtype = torch.float64
        weights = torch.tensor(
            (0.20, 0.30, 0.10),
            dtype=dtype,
            requires_grad=True,
        )
        scalar = torch.tensor(0.10, dtype=dtype)
        positive_margins = torch.tensor(
            (0.10, 0.10, 0.04),
            dtype=dtype,
            requires_grad=True,
        )
        negative_margins = torch.tensor(
            (-0.10, -0.10, -0.20),
            dtype=dtype,
            requires_grad=True,
        )
        row = runner.PublicRelationCreditRow(
            heldout_index=0,
            transition_index=0,
            positive_index=0,
            negative_index=1,
            positive_margin=scalar,
            negative_margin=-scalar,
            instance_loss=scalar,
            context_loss=scalar,
            separation_loss=scalar,
            joint_loss=scalar,
            slot_losses=torch.tensor((0.0, 0.0, 1.0), dtype=dtype),
            slot_positive_margins=positive_margins,
            slot_negative_margins=negative_margins,
            responsibilities=torch.tensor((0.10, 0.80, 0.10), dtype=dtype),
            context_weights=weights,
            context_null_weight=torch.tensor(0.40, dtype=dtype),
        )
        loss, diagnostics = runner._context_valid_set_training_term(row)
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss.detach()), -math.log(0.50))
        self.assertEqual(diagnostics["valid_slot_count"], 2)
        self.assertAlmostEqual(
            diagnostics["responsibility_valid_set_mass"],
            0.90,
        )
        self.assertTrue(diagnostics["responsibility_argmax_in_valid_set"])
        self.assertAlmostEqual(diagnostics["context_valid_set_mass"], 0.50)
        self.assertAlmostEqual(
            diagnostics["context_valid_set_real_normalized_mass"],
            5.0 / 6.0,
        )
        self.assertFalse(diagnostics["context_valid_set_top_one"])
        loss.backward()
        torch.testing.assert_close(
            weights.grad,
            torch.tensor((-2.0, -2.0, 0.0), dtype=dtype),
        )
        self.assertIsNone(positive_margins.grad)
        self.assertIsNone(negative_margins.grad)

        order = torch.tensor((2, 0, 1))
        permuted_row = replace(
            row,
            slot_losses=row.slot_losses[order],
            slot_positive_margins=row.slot_positive_margins[order],
            slot_negative_margins=row.slot_negative_margins[order],
            responsibilities=row.responsibilities[order],
            context_weights=row.context_weights.detach()[order],
        )
        permuted_loss, permuted_diagnostics = (
            runner._context_valid_set_training_term(permuted_row)
        )
        torch.testing.assert_close(permuted_loss, loss.detach())
        self.assertEqual(
            permuted_diagnostics["valid_slot_count"],
            diagnostics["valid_slot_count"],
        )
        self.assertAlmostEqual(
            permuted_diagnostics["context_valid_set_mass"],
            diagnostics["context_valid_set_mass"],
        )

        unsupported = replace(
            row,
            slot_positive_margins=torch.full((3,), 0.04, dtype=dtype),
            slot_negative_margins=torch.full((3,), -0.04, dtype=dtype),
        )
        unsupported_loss, unsupported_diagnostics = (
            runner._context_valid_set_training_term(unsupported)
        )
        self.assertIsNone(unsupported_loss)
        self.assertFalse(unsupported_diagnostics["supported"])

    def test_v9_real_context_fit_skips_unsupported_rows_and_reports_counts(self) -> None:
        original_rows = runner.public_relation_credit_rows

        def public_rows(controller, stream, **kwargs):
            rows = original_rows(controller, stream, **kwargs)
            rewritten = []
            for index, row in enumerate(rows):
                if index < 3:
                    positive = row.slot_positive_margins.new_tensor(
                        (0.10, 0.04, 0.04)
                    )
                    negative = row.slot_negative_margins.new_tensor(
                        (-0.10, -0.04, -0.04)
                    )
                else:
                    positive = row.slot_positive_margins.new_full((3,), 0.04)
                    negative = row.slot_negative_margins.new_full((3,), -0.04)
                rewritten.append(
                    replace(
                        row,
                        slot_positive_margins=positive,
                        slot_negative_margins=negative,
                    )
                )
            return tuple(rewritten)

        with mock.patch.object(
            runner,
            "public_relation_credit_rows",
            side_effect=public_rows,
        ):
            report = runner._fit_public_relation_credit_batches(
                self.controller,
                ((self.stream,),),
                stage="context",
            )
        self.assertEqual(report["context_supported_rows_per_stream"], ((3,),))
        self.assertEqual(report["context_supported_rows"], (3,))
        self.assertEqual(report["stream_gradient_weights"], ((1.0,),))
        self.assertTrue(math.isfinite(report["first_loss"]))
        self.assertTrue(report["frozen_parameters_unchanged"])

        unsupported_controller = runner.build_public_relation_credit_controller()

        def unsupported_rows(controller, stream, **kwargs):
            rows = original_rows(controller, stream, **kwargs)
            return tuple(
                replace(
                    row,
                    slot_positive_margins=row.slot_positive_margins.new_full(
                        (3,),
                        0.04,
                    ),
                    slot_negative_margins=row.slot_negative_margins.new_full(
                        (3,),
                        -0.04,
                    ),
                )
                for row in rows
            )

        with mock.patch.object(
            runner,
            "public_relation_credit_rows",
            side_effect=unsupported_rows,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "context aggregation requires positive aligned row counts",
            ):
                runner._fit_public_relation_credit_batches(
                    unsupported_controller,
                    ((self.stream,),),
                    stage="context",
                )

    def test_v9_entropic_stream_objective_matches_formula_and_gradient(self) -> None:
        equal_losses = torch.full(
            (8,),
            0.375,
            dtype=torch.float64,
            requires_grad=True,
        )
        objective, flat, entropic, weights, effective = (
            runner._anonymous_entropic_stream_objective(equal_losses)
        )
        objective.backward()
        self.assertAlmostEqual(float(objective.item()), 0.375)
        self.assertAlmostEqual(float(flat.item()), 0.375)
        self.assertAlmostEqual(float(entropic.item()), 0.375)
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 8.0))
        torch.testing.assert_close(equal_losses.grad, weights)
        self.assertAlmostEqual(float(effective.item()), 8.0)

        losses = torch.tensor(
            (0.0, 0.1, -0.2, 0.7, 0.3, -0.1, 0.2, 0.4),
            dtype=torch.float64,
            requires_grad=True,
        )
        objective, flat, entropic, weights, effective = (
            runner._anonymous_entropic_stream_objective(losses)
        )
        expected_entropic = 0.05 * (
            torch.logsumexp(losses / 0.05, dim=0) - math.log(8)
        )
        expected = 0.5 * losses.mean() + 0.5 * expected_entropic
        torch.testing.assert_close(objective, expected)
        torch.testing.assert_close(entropic, expected_entropic)
        objective.backward()
        torch.testing.assert_close(losses.grad, weights)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0)
        self.assertGreaterEqual(float(weights.min().item()), 1.0 / 16.0)
        self.assertEqual(int(torch.argmax(weights).item()), 3)
        self.assertTrue(torch.isfinite(effective))

        order = torch.tensor((7, 3, 1, 5, 0, 6, 2, 4))
        permuted = runner._anonymous_entropic_stream_objective(
            losses.detach()[order]
        )
        torch.testing.assert_close(permuted[0], objective.detach())
        torch.testing.assert_close(permuted[3], weights[order])
        torch.testing.assert_close(permuted[4], effective)

    def test_v10_entropic_row_objective_is_smooth_anonymous_upper_tail(self) -> None:
        equal_losses = torch.full(
            (4,),
            0.20,
            dtype=torch.float64,
            requires_grad=True,
        )
        objective, flat, entropic, weights, effective = (
            runner._anonymous_entropic_row_objective(equal_losses)
        )
        objective.backward()
        self.assertAlmostEqual(float(objective.item()), 0.20)
        self.assertAlmostEqual(float(flat.item()), 0.20)
        self.assertAlmostEqual(float(entropic.item()), 0.20)
        torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
        torch.testing.assert_close(equal_losses.grad, weights)
        self.assertAlmostEqual(float(effective.item()), 4.0)

        losses = torch.tensor(
            (0.02, 0.02, 0.20, 0.20),
            dtype=torch.float64,
            requires_grad=True,
        )
        objective, flat, entropic, weights, effective = (
            runner._anonymous_entropic_row_objective(losses)
        )
        expected_entropic = 0.05 * (
            torch.logsumexp(losses / 0.05, dim=0) - math.log(4)
        )
        torch.testing.assert_close(
            objective,
            0.5 * losses.mean() + 0.5 * expected_entropic,
        )
        losses.grad = None
        objective.backward()
        torch.testing.assert_close(losses.grad, weights)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0)
        self.assertGreaterEqual(float(weights.min().item()), 0.125)
        self.assertGreater(float(weights[2].item()), float(weights[0].item()))
        self.assertTrue(torch.isfinite(effective))

        uniform_same_mean = runner._anonymous_entropic_row_objective(
            torch.full((4,), 0.11, dtype=torch.float64)
        )
        self.assertAlmostEqual(float(flat.item()), float(uniform_same_mean[1].item()))
        self.assertGreater(float(objective.detach().item()), float(uniform_same_mean[0].item()))

        order = torch.tensor((2, 0, 3, 1))
        permuted = runner._anonymous_entropic_row_objective(
            losses.detach()[order]
        )
        torch.testing.assert_close(permuted[0], objective.detach())
        torch.testing.assert_close(permuted[3], weights[order])
        torch.testing.assert_close(permuted[4], effective)
        with self.assertRaises(ValueError):
            runner._anonymous_entropic_row_objective(torch.zeros(3))

    def test_v9_stage_aggregation_is_robust_only_for_relation_and_joint(self) -> None:
        losses = torch.tensor(
            (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.80),
            dtype=torch.float64,
        )
        counts = torch.tensor(
            (4.0, 4.0, 2.0, 0.0, 1.0, 3.0, 4.0, 2.0),
            dtype=torch.float64,
        )
        relation = runner._relation_credit_stream_objective(
            losses,
            stage="relation",
        )
        context = runner._relation_credit_stream_objective(
            losses,
            stage="context",
            stream_row_counts=counts,
        )
        joint = runner._relation_credit_stream_objective(
            losses,
            stage="joint",
        )
        self.assertTrue(relation[5])
        self.assertFalse(context[5])
        self.assertTrue(joint[5])
        torch.testing.assert_close(relation[0], joint[0])
        torch.testing.assert_close(relation[3], joint[3])
        self.assertGreater(float(relation[3].max()), float(relation[3].min()))
        torch.testing.assert_close(
            context[3],
            counts / counts.sum(),
        )
        torch.testing.assert_close(
            context[0],
            (losses * counts).sum() / counts.sum(),
        )
        self.assertAlmostEqual(
            float(context[4]),
            float((counts / counts.sum()).square().sum().reciprocal()),
        )
        with self.assertRaisesRegex(
            ValueError,
            "context aggregation requires positive aligned row counts",
        ):
            runner._relation_credit_stream_objective(
                losses,
                stage="context",
                stream_row_counts=torch.zeros_like(counts),
            )
        reconstructed = 0.5 * relation[1] + 0.5 * relation[2]
        torch.testing.assert_close(relation[0], reconstructed)

        extreme = runner._anonymous_entropic_stream_objective(
            torch.tensor((100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0))
        )
        self.assertTrue(torch.isfinite(extreme[0]))
        self.assertLessEqual(float(extreme[3].max()), 9.0 / 16.0)
        self.assertGreaterEqual(float(extreme[4]), 32.0 / 11.0 - 1.0e-5)

    def test_v9_panel_uses_supported_only_context_denominator(self) -> None:
        dtype = torch.float64

        def make_row(*, supported: bool) -> runner.PublicRelationCreditRow:
            if supported:
                slot_positive = torch.tensor((0.10, 0.04, 0.04), dtype=dtype)
                slot_negative = torch.tensor((-0.10, -0.04, -0.04), dtype=dtype)
                context = torch.tensor((0.70, 0.10, 0.05), dtype=dtype)
                null = torch.tensor(0.15, dtype=dtype)
            else:
                slot_positive = torch.tensor((0.04, 0.04, 0.04), dtype=dtype)
                slot_negative = torch.tensor((-0.04, -0.04, -0.04), dtype=dtype)
                context = torch.tensor((0.20, 0.20, 0.20), dtype=dtype)
                null = torch.tensor(0.40, dtype=dtype)
            scalar = torch.tensor(0.10, dtype=dtype)
            return runner.PublicRelationCreditRow(
                heldout_index=0,
                transition_index=0,
                positive_index=0,
                negative_index=1,
                positive_margin=scalar,
                negative_margin=-scalar,
                instance_loss=scalar,
                context_loss=scalar,
                separation_loss=scalar,
                joint_loss=scalar,
                slot_losses=torch.tensor((0.0, 1.0, 2.0), dtype=dtype),
                slot_positive_margins=slot_positive,
                slot_negative_margins=slot_negative,
                responsibilities=torch.full((3,), 1.0 / 3.0, dtype=dtype),
                context_weights=context,
                context_null_weight=null,
            )

        supported_group = tuple(make_row(supported=True) for _ in range(4))
        empty_group = tuple(make_row(supported=False) for _ in range(4))
        groups = (supported_group,) * 6 + (empty_group,) * 2
        with mock.patch.object(
            runner,
            "public_relation_credit_rows",
            side_effect=groups,
        ):
            panel = runner.evaluate_public_relation_credit_panel(
                self.controller,
                (self.stream,) * 8,
            )
        self.assertEqual(panel["relation_supported_rows"], 24)
        self.assertEqual(panel["streams_with_three_supported_rows"], 6)
        self.assertEqual(
            panel["supported_rows_per_stream"],
            (4, 4, 4, 4, 4, 4, 0, 0),
        )
        self.assertEqual(panel["valid_slot_count_histogram"], (8, 24, 0, 0))
        self.assertAlmostEqual(
            panel["context_valid_set_mass_mean_supported"],
            0.70,
        )
        self.assertAlmostEqual(
            panel["context_valid_set_mass_mean_all_rows"],
            0.525,
        )
        self.assertEqual(
            panel["context_valid_set_top_one_fraction_supported"],
            1.0,
        )

    def test_v9_gate_boundaries_and_aggregate_result_remain_independent(self) -> None:
        invariants = {
            "permutation_covariant": True,
            "empty_memory_zero_exact": True,
        }
        relation_panel = {
            "streams": 8,
            "rows": 32,
            "relation_supported_rows": 24,
            "streams_with_three_supported_rows": 6,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 4, 0, 0),
            "valid_slot_count_histogram": (8, 24, 0, 0),
        }
        self.assertTrue(
            runner._relation_credit_relation_gate(
                relation_panel,
                invariants,
            )["passed"]
        )
        below_rows = {
            **relation_panel,
            "relation_supported_rows": 23,
            "supported_rows_per_stream": (4, 4, 3, 3, 3, 3, 2, 1),
            "valid_slot_count_histogram": (9, 23, 0, 0),
        }
        below_streams = {
            **relation_panel,
            "streams_with_three_supported_rows": 5,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 2, 1, 1),
        }
        for below in (below_rows, below_streams):
            self.assertFalse(
                runner._relation_credit_relation_gate(below, invariants)["passed"]
            )

        inconsistent = {
            **relation_panel,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 3, 1, 1),
        }
        gate = runner._relation_credit_relation_gate(inconsistent, invariants)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["per_stream_counts_consistent"])

        perfect_set_panel = {
            **relation_panel,
            "context_valid_set_top_one_fraction_supported": 1.0,
            "context_valid_set_mass_mean_supported": 1.0,
            "positive_margin_mean": 0.099,
            "negative_margin_mean": -0.20,
            "separation_mean": 0.30,
            "signed_rows": 32,
            "streams_with_three_signed_rows": 8,
        }
        final = runner._relation_credit_final_gate(
            perfect_set_panel,
            invariants,
            shared_parameters_bit_exact=True,
        )
        self.assertFalse(final["passed"])
        self.assertFalse(final["checks"]["positive_margin_mean"])

    def test_v9_orchestrator_stops_before_context_when_relation_gate_fails(self) -> None:
        controller = runner.build_public_relation_credit_controller()
        calls = []

        def fake_fit(_controller, _batches, *, stage, **_kwargs):
            calls.append(stage)
            return _fake_v9_stage_report(stage)

        with (
            mock.patch.object(
                runner,
                "_relation_credit_stream_batches",
                return_value=((self.stream,),),
            ),
            mock.patch.object(
                runner,
                "_relation_credit_panel_streams",
                return_value=(self.stream,) * 8,
            ),
            mock.patch.object(
                runner,
                "_fit_public_relation_credit_batches",
                side_effect=fake_fit,
            ),
            mock.patch.object(
                runner,
                "evaluate_public_relation_credit_panel",
                return_value={
                    "streams": 8,
                    "rows": 32,
                    "relation_supported_rows": 23,
                    "streams_with_three_supported_rows": 6,
                    "supported_rows_per_stream": (4, 4, 3, 3, 3, 3, 2, 1),
                    "valid_slot_count_histogram": (9, 23, 0, 0),
                },
            ),
            mock.patch.object(
                runner,
                "_evaluate_public_relation_credit_invariants",
                return_value={
                    "permutation_covariant": True,
                    "empty_memory_zero_exact": True,
                },
            ),
        ):
            report = runner.fit_public_relation_matcher(controller)
        self.assertEqual(report["status"], "STOPPED_AFTER_RELATION_GATE")
        self.assertEqual(calls, ["relation"])
        self.assertNotIn("context", report["stage_reports"])
        self.assertNotIn("joint", report["stage_reports"])

    def test_v9_orchestrator_rejects_incomplete_stage_history(self) -> None:
        malformed_reports = []
        missing_stream_loss = _fake_v9_stage_report("relation")
        missing_stream_loss["stream_losses"] = missing_stream_loss[
            "stream_losses"
        ][:-1]
        malformed_reports.append(missing_stream_loss)
        missing_gradient = _fake_v9_stage_report("relation")
        missing_gradient["gradient_norms"] = missing_gradient["gradient_norms"][
            :-1
        ]
        malformed_reports.append(missing_gradient)
        invalid_mean = _fake_v9_stage_report("relation")
        invalid_mean["mean_gradient_norm"] = math.inf
        malformed_reports.append(invalid_mean)
        invalid_endpoint = _fake_v9_stage_report("relation")
        invalid_endpoint["first_loss"] = 0.21
        malformed_reports.append(invalid_endpoint)
        invalid_frozen = _fake_v9_stage_report("relation")
        invalid_frozen["frozen_parameters_unchanged"] = False
        malformed_reports.append(invalid_frozen)

        false_row_weights = _fake_v9_stage_report("relation")
        row_weights = list(false_row_weights["row_gradient_weights"])
        first_row_weight_groups = list(row_weights[0])
        first_row_weight_groups[0] = (0.40, 0.20, 0.20, 0.20)
        row_weights[0] = tuple(first_row_weight_groups)
        false_row_weights["row_gradient_weights"] = tuple(row_weights)
        row_effective = list(false_row_weights["effective_row_counts"])
        first_row_effective = list(row_effective[0])
        first_row_effective[0] = 1.0 / (0.40**2 + 3 * 0.20**2)
        row_effective[0] = tuple(first_row_effective)
        false_row_weights["effective_row_counts"] = tuple(row_effective)
        malformed_reports.append(false_row_weights)

        false_outer_weights = _fake_v9_stage_report("relation")
        stream_weights = list(false_outer_weights["stream_gradient_weights"])
        stream_weights[0] = (0.30,) + (0.10,) * 7
        false_outer_weights["stream_gradient_weights"] = tuple(stream_weights)
        stream_effective = list(false_outer_weights["effective_stream_counts"])
        stream_effective[0] = 1.0 / (0.30**2 + 7 * 0.10**2)
        false_outer_weights["effective_stream_counts"] = tuple(stream_effective)
        malformed_reports.append(false_outer_weights)

        false_outer_entropic = _fake_v9_stage_report("relation")
        entropic_terms = list(false_outer_entropic["entropic_terms"])
        entropic_terms[0] = 0.30
        false_outer_entropic["entropic_terms"] = tuple(entropic_terms)
        objectives = list(false_outer_entropic["losses"])
        objectives[0] = 0.25
        false_outer_entropic["losses"] = tuple(objectives)
        false_outer_entropic["first_loss"] = 0.25
        malformed_reports.append(false_outer_entropic)

        for malformed in malformed_reports:
            controller = runner.build_public_relation_credit_controller()
            with (
                mock.patch.object(
                    runner,
                    "_relation_credit_stream_batches",
                    return_value=((self.stream,),),
                ),
                mock.patch.object(
                    runner,
                    "_fit_public_relation_credit_batches",
                    return_value=malformed,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "relation stage exposure accounting changed",
                ):
                    runner.fit_public_relation_matcher(controller)

    def test_v9_orchestrator_rejects_false_context_history(self) -> None:
        updates = runner._RELATION_CREDIT_STAGE_UPDATES["context"]
        malformed_reports = []

        false_objective = _fake_v9_stage_report("context")
        false_objective["first_loss"] = 0.21
        false_objective["last_loss"] = 0.21
        false_objective["losses"] = (0.21,) * updates
        false_objective["flat_mean_losses"] = (0.21,) * updates
        malformed_reports.append(false_objective)

        float_totals = _fake_v9_stage_report("context")
        float_totals["context_supported_rows"] = (32.0,) * updates
        malformed_reports.append(float_totals)

        nonzero_empty_streams = _fake_v9_stage_report("context")
        nonzero_empty_streams["context_supported_rows_per_stream"] = (
            ((4, 4, 4, 4, 4, 4, 0, 0),) * updates
        )
        nonzero_empty_streams["context_supported_rows"] = (24,) * updates
        nonzero_empty_streams["stream_gradient_weights"] = (
            ((1.0 / 6.0,) * 6 + (0.0, 0.0),) * updates
        )
        malformed_reports.append(nonzero_empty_streams)

        relation_panel = {
            "streams": 8,
            "rows": 32,
            "relation_supported_rows": 24,
            "streams_with_three_supported_rows": 6,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 4, 0, 0),
            "valid_slot_count_histogram": (8, 24, 0, 0),
        }
        invariants = {
            "permutation_covariant": True,
            "empty_memory_zero_exact": True,
        }
        for malformed in malformed_reports:
            controller = runner.build_public_relation_credit_controller()

            def fake_fit(_controller, _batches, *, stage, **_kwargs):
                if stage == "relation":
                    return _fake_v9_stage_report("relation")
                return malformed

            with (
                mock.patch.object(
                    runner,
                    "_relation_credit_stream_batches",
                    return_value=((self.stream,),),
                ),
                mock.patch.object(
                    runner,
                    "_relation_credit_panel_streams",
                    return_value=(self.stream,) * 8,
                ),
                mock.patch.object(
                    runner,
                    "_fit_public_relation_credit_batches",
                    side_effect=fake_fit,
                ),
                mock.patch.object(
                    runner,
                    "evaluate_public_relation_credit_panel",
                    return_value=relation_panel,
                ),
                mock.patch.object(
                    runner,
                    "_evaluate_public_relation_credit_invariants",
                    return_value=invariants,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "context stage exposure accounting changed",
                ):
                    runner.fit_public_relation_matcher(controller)

    def test_v9_orchestrator_stops_at_context_and_reports_final_failure(self) -> None:
        relation_panel = {
            "streams": 8,
            "rows": 32,
            "relation_supported_rows": 24,
            "streams_with_three_supported_rows": 6,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 4, 0, 0),
            "valid_slot_count_histogram": (8, 24, 0, 0),
            "context_valid_set_top_one_fraction_supported": 0.80,
            "context_valid_set_mass_mean_supported": 0.60,
        }
        invariants = {
            "permutation_covariant": True,
            "empty_memory_zero_exact": True,
        }

        def execute(panel_reports):
            controller = runner.build_public_relation_credit_controller()
            stages = []

            def fake_fit(_controller, _batches, *, stage, **_kwargs):
                stages.append(stage)
                return _fake_v9_stage_report(stage)

            with (
                mock.patch.object(
                    runner,
                    "_relation_credit_stream_batches",
                    return_value=((self.stream,),),
                ),
                mock.patch.object(
                    runner,
                    "_relation_credit_panel_streams",
                    return_value=(self.stream,) * 8,
                ),
                mock.patch.object(
                    runner,
                    "_fit_public_relation_credit_batches",
                    side_effect=fake_fit,
                ),
                mock.patch.object(
                    runner,
                    "evaluate_public_relation_credit_panel",
                    side_effect=panel_reports,
                ),
                mock.patch.object(
                    runner,
                    "_evaluate_public_relation_credit_invariants",
                    return_value=invariants,
                ),
            ):
                return runner.fit_public_relation_matcher(controller), stages

        context_failure = {
            **relation_panel,
            "context_valid_set_top_one_fraction_supported": 0.79,
        }
        report, stages = execute((relation_panel, context_failure))
        self.assertEqual(report["status"], "STOPPED_AFTER_CONTEXT_GATE")
        self.assertEqual(stages, ["relation", "context"])
        self.assertNotIn("joint", report["stage_reports"])

        final_failure = {
            **relation_panel,
            "positive_margin_mean": 0.10,
            "negative_margin_mean": -0.10,
            "separation_mean": 0.20,
            "signed_rows": 25,
            "streams_with_three_signed_rows": 7,
        }
        report, stages = execute((relation_panel, relation_panel, final_failure))
        self.assertEqual(report["status"], "FAILED_FINAL_GATE")
        self.assertFalse(report["passed"])
        self.assertEqual(stages, ["relation", "context", "joint"])

    def test_whole_fold_pair_is_public_aligned_and_masked_task_equal(self) -> None:
        wrong = runner.make_software_pipeline_control_stream(
            self.stream,
            "wrong_evidence",
        )
        for positive, negative in zip(
            self.stream.supports,
            wrong.supports,
            strict=True,
        ):
            self.assertEqual(
                replace(positive.learner, observations=()),
                replace(negative.learner, observations=()),
            )
        folds = runner._paired_public_relation_folds(
            self.controller,
            self.stream,
        )
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertNotEqual(fold.positive_action, fold.negative_action)
            self.assertEqual(fold.masked_task.observations, ())
            encoding = self.controller.encode_task(fold.masked_task)
            positive_index = fold.masked_task.grounded_candidates.index(
                fold.positive_action
            )
            negative_index = fold.masked_task.grounded_candidates.index(
                fold.negative_action
            )
            self.assertFalse(
                torch.allclose(
                    encoding.relation_component_embeddings[positive_index],
                    encoding.relation_component_embeddings[negative_index],
                )
            )
            self.assertEqual(
                int(fold.positive_state.role.occupied[0].sum().item()),
                6,
            )
            self.assertEqual(
                int(fold.negative_state.role.occupied[0].sum().item()),
                6,
            )

    def test_twins_share_exact_predecessor_context_but_not_relation_value(self) -> None:
        task = replace(self.stream.supports[0].learner, observations=())
        encoding = self.controller.encode_task(task)
        components = runner._components_in_candidate_order(task)
        twin_groups = {}
        for index, component in enumerate(components):
            predecessors = tuple(
                predecessor
                for predecessor in components
                if predecessor.output_type == component.input_type
            )
            if not predecessors:
                continue
            key = (
                component.input_type,
                component.output_type,
                component.error_type,
                component.state_reads,
                component.state_writes,
            )
            twin_groups.setdefault(key, []).append(index)
        twins = next(indices for indices in twin_groups.values() if len(indices) == 2)
        left, right = twins
        self.assertTrue(
            torch.equal(
                encoding.relation_context_embeddings[left],
                encoding.relation_context_embeddings[right],
            )
        )
        self.assertFalse(
            torch.allclose(
                encoding.relation_component_embeddings[left],
                encoding.relation_component_embeddings[right],
            )
        )

    def test_correct_and_wrong_arms_share_context_keys_only(self) -> None:
        folds = runner._paired_public_relation_folds(
            self.controller,
            self.stream,
        )
        for fold in folds:
            self.assertTrue(
                torch.equal(
                    fold.positive_state.context_trace_keys,
                    fold.negative_state.context_trace_keys,
                )
            )
            occupied = fold.positive_state.role.occupied & fold.negative_state.role.occupied
            difference = (
                fold.positive_state.relation_trace_values
                - fold.negative_state.relation_trace_values
            ).abs()
            self.assertGreater(
                float(
                    difference.detach()
                    .masked_select(occupied.unsqueeze(-1))
                    .sum()
                ),
                0.0,
            )
            self.assertTrue(
                torch.equal(
                    fold.positive_state.role.trace_cursor,
                    fold.negative_state.role.trace_cursor,
                )
            )

    def test_paired_loss_has_opposite_gradients_and_no_arm_bias(self) -> None:
        positive = torch.tensor(0.0, requires_grad=True)
        negative = torch.tensor(0.0, requires_grad=True)
        loss = runner._paired_relation_margin_loss(positive, negative)
        positive_gradient, negative_gradient = torch.autograd.grad(
            loss,
            (positive, negative),
        )
        self.assertLess(float(positive_gradient), 0.0)
        self.assertGreater(float(negative_gradient), 0.0)
        swapped = runner._paired_relation_margin_loss(-negative, -positive)
        self.assertEqual(float(loss.detach()), float(swapped.detach()))

    def test_public_credit_uses_only_observed_actions_and_declared_alternatives(self) -> None:
        with mock.patch.object(
            runner,
            "make_software_pipeline_control_stream",
            side_effect=AssertionError("control stream is not a training input"),
        ):
            rows = runner.public_relation_credit_rows(
                self.controller,
                self.stream,
            )
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(tuple(row.slot_losses.shape), (3,))
            self.assertEqual(tuple(row.slot_positive_margins.shape), (3,))
            self.assertEqual(tuple(row.slot_negative_margins.shape), (3,))
            self.assertEqual(
                int(torch.argmin(row.slot_losses).item()),
                int(torch.argmax(row.responsibilities).item()),
            )
            self.assertAlmostEqual(
                float(row.responsibilities.sum()),
                1.0,
                places=6,
            )
            self.assertGreater(float(row.context_weights.detach().sum()), 0.0)
            self.assertLess(float(row.context_weights.detach().sum()), 1.0)
            self.assertGreaterEqual(float(row.instance_loss.detach()), 0.0)
            self.assertTrue(torch.isfinite(row.joint_loss))

        fit_controller = runner.build_public_relation_credit_controller()
        with mock.patch.object(
            runner,
            "make_software_pipeline_control_stream",
            side_effect=AssertionError("control stream entered the real fit path"),
        ):
            report = runner._fit_public_relation_credit_batches(
                fit_controller,
                ((self.stream,),),
                stage="relation",
            )
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["rows"], 4)

    def test_relation_and_context_credit_have_separate_gradient_paths(self) -> None:
        rows = runner.public_relation_credit_rows(self.controller, self.stream)
        torch.stack(tuple(row.instance_loss for row in rows)).mean().backward()
        relation_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_pair_encoder.parameters()
            if parameter.grad is not None
        )
        context_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_context_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(relation_gradient, 0.0)
        self.assertEqual(context_gradient, 0.0)

        self.controller.zero_grad(set_to_none=True)
        rows = runner.public_relation_credit_rows(self.controller, self.stream)
        torch.stack(tuple(row.context_loss for row in rows)).mean().backward()
        relation_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_pair_encoder.parameters()
            if parameter.grad is not None
        )
        context_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_context_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(relation_gradient, 0.0)
        self.assertGreater(context_gradient, 0.0)

    def test_public_credit_is_presentation_and_evidence_order_covariant(self) -> None:
        ordinary = runner.public_relation_credit_rows(self.controller, self.stream)
        for reverse_evidence, reverse_presentation in (
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                reverse_evidence=reverse_evidence,
                reverse_presentation=reverse_presentation,
            ):
                transformed = runner.public_relation_credit_rows(
                    self.controller,
                    self.stream,
                    reverse_evidence_order=reverse_evidence,
                    reverse_public_presentation=reverse_presentation,
                )
                for left, right in zip(ordinary, transformed, strict=True):
                    for field in (
                        "positive_margin",
                        "negative_margin",
                        "instance_loss",
                        "context_loss",
                        "separation_loss",
                        "joint_loss",
                        "context_null_weight",
                    ):
                        torch.testing.assert_close(
                            getattr(left, field),
                            getattr(right, field),
                            atol=1.0e-6,
                            rtol=0.0,
                        )
                    for field in (
                        "slot_losses",
                        "slot_positive_margins",
                        "slot_negative_margins",
                        "responsibilities",
                        "context_weights",
                    ):
                        right_value = getattr(right, field)
                        if reverse_evidence:
                            right_value = right_value.flip(0)
                        torch.testing.assert_close(
                            getattr(left, field),
                            right_value,
                            atol=1.0e-6,
                            rtol=0.0,
                        )

    def test_public_credit_invariant_gate_checks_each_axis_and_empty_memory(self) -> None:
        report = runner._evaluate_public_relation_credit_invariants(
            self.controller,
            (self.stream,) * 8,
        )
        self.assertTrue(report["permutation_covariant"])
        self.assertTrue(report["valid_set_covariant"])
        self.assertTrue(report["empty_memory_zero_exact"])
        for field in (
            "evidence_order_max_delta",
            "public_presentation_max_delta",
            "combined_max_delta",
            "permutation_max_delta",
        ):
            self.assertLessEqual(
                report[field],
                runner._RELATION_GATE_PERMUTATION_TOLERANCE,
            )

    def test_public_credit_positive_margin_matches_runtime_evidence_score(self) -> None:
        rows = runner.public_relation_credit_rows(self.controller, self.stream)
        for heldout_index, row in enumerate(rows):
            evidence_tasks = tuple(
                pair.learner
                for index, pair in enumerate(self.stream.supports)
                if index != heldout_index
            )
            state = runner._acquire_public_task_set(
                self.controller,
                evidence_tasks,
            )
            expected_contexts = []
            expected_relations = []
            for evidence_task in evidence_tasks:
                (
                    evidence_transitions,
                    evidence_observed,
                    _,
                    evidence_context_codes,
                    evidence_relation_codes,
                ) = runner._relation_credit_task(self.controller, evidence_task)
                self.assertEqual(len(evidence_transitions), len(evidence_observed))
                expected_contexts.extend(
                    evidence_context_codes[index] for index in evidence_observed
                )
                expected_relations.extend(
                    evidence_relation_codes[index] for index in evidence_observed
                )
            occupied = state.role.occupied[0, : self.controller.role_memory.trace_slot_count]
            torch.testing.assert_close(
                state.context_trace_keys[0, : self.controller.role_memory.trace_slot_count][occupied],
                torch.stack(expected_contexts),
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                state.relation_trace_values[0, : self.controller.role_memory.trace_slot_count][occupied],
                torch.stack(expected_relations),
                atol=0.0,
                rtol=0.0,
            )
            task = replace(
                self.stream.supports[heldout_index].learner,
                observations=(),
            )
            observed_task = self.stream.supports[heldout_index].learner
            encoding = self.controller.encode_task(task)
            (
                _,
                _,
                _,
                direct_context_codes,
                direct_relation_codes,
            ) = runner._relation_credit_task(self.controller, observed_task)
            torch.testing.assert_close(
                direct_context_codes,
                encoding.relation_context_embeddings,
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                direct_relation_codes,
                encoding.relation_component_embeddings,
                atol=0.0,
                rtol=0.0,
            )
            scores = self.controller._relation_evidence_scores(
                encoding.relation_context_embeddings,
                encoding.relation_component_embeddings,
                state,
            )
            trace_contexts = state.context_trace_keys[
                0, : self.controller.role_memory.trace_slot_count
            ]
            trace_relations = state.relation_trace_values[
                0, : self.controller.role_memory.trace_slot_count
            ]
            runtime_present = (
                occupied
                & (trace_contexts.norm(dim=-1) > 1.0e-8)
                & (trace_relations.norm(dim=-1) > 1.0e-8)
            )
            runtime_contexts = trace_contexts[runtime_present]
            runtime_relations = trace_relations[runtime_present]
            runtime_scores, runtime_weights, runtime_null, runtime_logits = (
                self.controller._relation_evidence_read(
                    encoding.relation_context_embeddings,
                    encoding.relation_component_embeddings,
                    runtime_contexts,
                    runtime_relations,
                )
            )
            query_present = (
                (encoding.relation_context_embeddings.norm(dim=-1) > 1.0e-8)
                & (encoding.relation_component_embeddings.norm(dim=-1) > 1.0e-8)
            )
            torch.testing.assert_close(
                scores,
                torch.where(
                    query_present,
                    runtime_scores,
                    torch.zeros_like(runtime_scores),
                ),
                atol=0.0,
                rtol=0.0,
            )
            self.assertEqual(runtime_null.shape, (runtime_weights.shape[0],))
            torch.testing.assert_close(
                runtime_weights.sum(dim=-1) + runtime_null,
                torch.ones_like(runtime_null),
                atol=1.0e-6,
                rtol=0.0,
            )
            transitions = runner._public_transitions(observed_task)
            components = runner._components_in_candidate_order(observed_task)
            transition = transitions[row.transition_index]
            positive_index = runner._action_index(
                observed_task.grounded_candidates,
                transition.action,
            )
            negative_index = runner._same_contract_alternative_index(
                components,
                positive_index,
            )
            self.assertIsNotNone(negative_index)
            self.assertEqual(row.positive_index, positive_index)
            self.assertEqual(row.negative_index, negative_index)
            torch.testing.assert_close(
                row.context_weights,
                runtime_weights[positive_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                row.context_null_weight,
                runtime_null[positive_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                row.slot_positive_margins,
                runtime_logits[positive_index] - runtime_logits[negative_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                row.positive_margin,
                (row.context_weights * row.slot_positive_margins).sum(),
                atol=1.0e-8,
                rtol=0.0,
            )
            torch.testing.assert_close(
                row.positive_margin,
                scores[positive_index] - scores[negative_index],
                atol=1.0e-6,
                rtol=0.0,
            )

    def test_public_credit_negative_margin_matches_wrong_evidence_audit_arm(self) -> None:
        rows = runner.public_relation_credit_rows(self.controller, self.stream)
        folds = runner._paired_public_relation_folds(self.controller, self.stream)
        self.assertEqual(len(rows), len(folds))
        for row, fold in zip(rows, folds, strict=True):
            encoding = self.controller.encode_task(fold.masked_task)
            scores = self.controller._relation_evidence_scores(
                encoding.relation_context_embeddings,
                encoding.relation_component_embeddings,
                fold.negative_state,
            )
            positive_index = runner._action_index(
                fold.masked_task.grounded_candidates,
                fold.positive_action,
            )
            negative_index = runner._action_index(
                fold.masked_task.grounded_candidates,
                fold.negative_action,
            )
            self.assertEqual(row.positive_index, positive_index)
            self.assertEqual(row.negative_index, negative_index)
            torch.testing.assert_close(
                row.negative_margin,
                scores[positive_index] - scores[negative_index],
                atol=1.0e-6,
                rtol=0.0,
            )

    def test_credit_comparators_are_bounded(self) -> None:
        rows = runner.public_relation_credit_rows(self.controller, self.stream)
        for row in rows:
            self.assertLessEqual(
                float(row.slot_positive_margins.detach().abs().max()),
                2.0,
            )
            self.assertLessEqual(
                float(row.slot_negative_margins.detach().abs().max()),
                2.0,
            )

    def test_staged_credit_fit_respects_tower_boundaries_and_reports_panel(self) -> None:
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
        }
        relation = runner._fit_public_relation_credit_batches(
            self.controller,
            ((self.stream,),),
            stage="relation",
        )
        self.assertEqual(relation["optimizer_steps"], 1)
        self.assertEqual(relation["streams"], 1)
        self.assertEqual(relation["rows"], 4)
        self.assertTrue(relation["robust_stream_objective_applied"])
        self.assertTrue(relation["robust_row_objective_applied"])
        self.assertEqual(relation["stream_gradient_weights"], ((1.0,),))
        self.assertEqual(len(relation["row_losses"]), 1)
        self.assertEqual(len(relation["row_losses"][0]), 1)
        self.assertEqual(len(relation["row_losses"][0][0]), 4)
        self.assertEqual(len(relation["row_gradient_weights"][0][0]), 4)
        self.assertAlmostEqual(
            sum(relation["row_gradient_weights"][0][0]),
            1.0,
        )
        after_relation = dict(self.controller.named_parameters())
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in after_relation.items()
                if name.startswith("evidence_pair_encoder.")
            )
        )
        for name, parameter in after_relation.items():
            if name.startswith("evidence_context_encoder."):
                self.assertTrue(torch.equal(before[name], parameter), msg=name)

        relation_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
            if name.startswith(
                (
                    "evidence_pair_encoder.",
                    "relation_pool_attention.",
                    "relation_pool_projection.",
                    "relation_incidence_readout.",
                    "relation_incidence_projection.",
                    "relation_comparator.",
                )
            )
        }
        def context_term(row):
            return row.context_loss, {
                "supported": True,
                "valid_slot_count": 1,
                "responsibility_valid_set_mass": 0.50,
                "responsibility_argmax_in_valid_set": True,
                "context_null_mass": 0.25,
                "context_valid_set_mass": 0.50,
                "context_valid_set_real_normalized_mass": 2.0 / 3.0,
                "context_valid_set_top_one": True,
            }

        with mock.patch.object(
            runner,
            "_context_valid_set_training_term",
            side_effect=context_term,
        ):
            context = runner._fit_public_relation_credit_batches(
                self.controller,
                ((self.stream,),),
                stage="context",
            )
        self.assertEqual(context["rows"], 4)
        self.assertFalse(context["robust_stream_objective_applied"])
        self.assertFalse(context["robust_row_objective_applied"])
        self.assertEqual(context["row_losses"], ())
        self.assertEqual(context["stream_gradient_weights"], ((1.0,),))
        self.assertEqual(context["context_supported_rows_per_stream"], ((4,),))
        self.assertEqual(context["context_supported_rows"], (4,))
        after_context = dict(self.controller.named_parameters())
        for name, value in relation_snapshot.items():
            self.assertTrue(torch.equal(value, after_context[name]), msg=name)
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in after_context.items()
                if name.startswith("evidence_context_encoder.")
            )
        )
        panel = runner.evaluate_public_relation_credit_panel(
            self.controller,
            (self.stream,),
        )
        self.assertEqual(panel["streams"], 1)
        self.assertEqual(panel["rows"], 4)
        self.assertTrue(math.isfinite(panel["target_witness_mean"]))
        self.assertTrue(math.isfinite(panel["raw_best_witness_mean"]))
        self.assertTrue(math.isfinite(panel["target_loss_mean"]))
        self.assertTrue(math.isfinite(panel["target_loss_gap_mean"]))
        self.assertTrue(math.isfinite(panel["target_responsibility_mean"]))
        self.assertTrue(
            math.isfinite(panel["context_valid_set_mass_mean_supported"])
        )
        self.assertEqual(len(panel["row_reports"]), 4)
        expected_row_fields = {
            "stream_index",
            "heldout_index",
            "transition_index",
            "target_slot",
            "target_positive_margin",
            "target_negative_margin",
            "target_witness",
            "target_loss",
            "target_loss_gap",
            "target_responsibility",
            "context_target_mass",
            "context_top_one",
            "valid_slots",
            "valid_slot_count",
            "relation_supported",
            "context_null_mass",
            "context_valid_set_mass",
            "context_valid_set_top_one",
            "raw_slot",
            "raw_positive_margin",
            "raw_negative_margin",
            "raw_witness",
            "slot_positive_margins",
            "slot_negative_margins",
            "slot_losses",
            "responsibilities",
            "context_weights",
            "positive_margin",
            "negative_margin",
            "unique_loss_selected_confident",
            "signed",
        }
        for row_report in panel["row_reports"]:
            self.assertEqual(set(row_report), expected_row_fields)
            for field in (
                "slot_positive_margins",
                "slot_negative_margins",
                "slot_losses",
                "responsibilities",
                "context_weights",
            ):
                self.assertEqual(len(row_report[field]), 3)

        before_joint = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
        }
        joint = runner._fit_public_relation_credit_batches(
            self.controller,
            ((self.stream,),),
            stage="joint",
        )
        self.assertEqual(joint["stage"], "joint")
        self.assertEqual(joint["optimizer_steps"], 1)
        self.assertEqual(joint["streams"], 1)
        self.assertEqual(joint["rows"], 4)
        self.assertTrue(joint["robust_stream_objective_applied"])
        self.assertTrue(joint["robust_row_objective_applied"])
        self.assertEqual(joint["stream_gradient_weights"], ((1.0,),))
        self.assertEqual(len(joint["row_losses"][0][0]), 4)
        after_joint = dict(self.controller.named_parameters())
        relation_prefixes = (
            "evidence_pair_encoder.",
            "relation_pool_attention.",
            "relation_pool_projection.",
            "relation_incidence_readout.",
            "relation_incidence_projection.",
            "relation_comparator.",
        )
        context_prefixes = (
            "evidence_context_encoder.",
            "relation_context_pool_attention.",
            "relation_context_pool_projection.",
            "relation_context_comparator.",
        )
        self.assertTrue(
            any(
                not torch.equal(before_joint[name], parameter)
                for name, parameter in after_joint.items()
                if name.startswith(relation_prefixes)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before_joint[name], parameter)
                for name, parameter in after_joint.items()
                if name.startswith(context_prefixes)
            )
        )
        for name, parameter in after_joint.items():
            if not name.startswith(relation_prefixes + context_prefixes):
                self.assertTrue(
                    torch.equal(before_joint[name], parameter),
                    msg=name,
                )

    def test_relation_rows_train_dedicated_encoder_and_matcher_only(self) -> None:
        rows = runner.public_paired_relation_fit_rows(
            self.controller,
            self.stream,
        )
        self.assertEqual(len(rows), 4)
        torch.stack(tuple(row.loss for row in rows)).mean().backward()
        relation_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.named_parameters()
            if name.startswith("relation_") and parameter.grad is not None
        )
        base_pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        self.assertGreater(relation_gradient, 0.0)
        self.assertEqual(base_pair_gradient, 0.0)
        dedicated_parameters = dict(
            self.controller.evidence_pair_encoder.named_parameters()
        )
        self.assertTrue(dedicated_parameters)
        for name, parameter in dedicated_parameters.items():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, msg=name)
        context_parameters = dict(
            self.controller.evidence_context_encoder.named_parameters()
        )
        self.assertTrue(context_parameters)
        for name, parameter in context_parameters.items():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, msg=name)
        gradients = {
            name: float(parameter.grad.abs().sum())
            for name, parameter in self.controller.named_parameters()
            if name.startswith("relation_") and parameter.grad is not None
        }
        for prefix in (
            "relation_pool_attention.",
            "relation_pool_projection.",
            "relation_incidence_projection.",
            "relation_comparator.",
            "relation_context_pool_attention.",
            "relation_context_pool_projection.",
            "relation_context_comparator.",
        ):
            self.assertGreater(
                sum(value for name, value in gradients.items() if name.startswith(prefix)),
                0.0,
            )

    def test_public_relation_objective_opens_incidence_trunk_on_second_step(self) -> None:
        def relation_loss():
            rows = runner.public_relation_credit_rows(
                self.controller,
                self.stream,
            )
            return torch.stack(
                tuple(
                    row.instance_loss
                    + runner._RELATION_CREDIT_SEPARATION_WEIGHT
                    * row.separation_loss
                    for row in rows
                )
            ).mean()

        relation_loss().backward()
        projection = self.controller.relation_incidence_projection.weight
        self.assertIsNotNone(projection.grad)
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)
        first_trunk_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.relation_incidence_readout.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(first_trunk_gradient, 0.0)

        with torch.no_grad():
            projection.add_(-1.0e-3 * projection.grad)
        self.controller.zero_grad(set_to_none=True)
        relation_loss().backward()
        second_trunk_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.relation_incidence_readout.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(second_trunk_gradient, 0.0)

    def test_relation_score_is_symmetric_empty_exact_and_order_covariant(self) -> None:
        width = self.controller.profile.width
        left = torch.randn(3, width)
        right = torch.randn(3, width)
        first_features = torch.cat(
            (left * right, (left - right).abs(), (left + right) * 0.5),
            dim=-1,
        )
        second_features = torch.cat(
            (right * left, (right - left).abs(), (right + left) * 0.5),
            dim=-1,
        )
        self.assertTrue(torch.equal(first_features, second_features))

        encoded_query = self.controller.encode_task(
            replace(self.stream.supports[0].learner, observations=())
        )
        empty_scores = self.controller._relation_evidence_scores(
            encoded_query.relation_context_embeddings,
            encoded_query.relation_component_embeddings,
            self.controller.initial_state(),
        )
        self.assertTrue(torch.equal(empty_scores, torch.zeros_like(empty_scores)))
        nonempty = _acquire_supports(self.controller, self.stream)
        no_role = self.controller.score_actions(
            replace(self.stream.supports[0].learner, observations=()),
            nonempty,
            include_pointer_memory=False,
            include_role_memory=False,
        )
        self.assertTrue(
            torch.equal(
                no_role.evidence_match_scores,
                torch.zeros_like(no_role.evidence_match_scores),
            )
        )

        ordinary = runner.public_paired_relation_fit_rows(
            self.controller,
            self.stream,
        )
        reordered = runner.public_paired_relation_fit_rows(
            self.controller,
            self.stream,
            reverse_evidence_order=True,
            reverse_public_presentation=True,
        )
        for left_row, right_row in zip(ordinary, reordered, strict=True):
            torch.testing.assert_close(
                left_row.positive_margin,
                right_row.positive_margin,
                atol=1.0e-6,
                rtol=0.0,
            )
            torch.testing.assert_close(
                left_row.negative_margin,
                right_row.negative_margin,
                atol=1.0e-6,
                rtol=0.0,
            )

    def test_unit_fit_freezes_every_old_parameter_and_counts_exposure(self) -> None:
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
        }
        fold = runner._paired_public_relation_folds(
            self.controller,
            self.stream,
        )[0]
        encoding = self.controller.encode_task(fold.masked_task)
        legacy_before = self.controller.score_actions(
            fold.masked_task,
            fold.positive_state,
            encoding=encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        no_memory_before = self.controller.score_actions(
            fold.masked_task,
            fold.positive_state,
            encoding=encoding,
            include_pointer_memory=False,
            include_role_memory=False,
        )
        report = runner._fit_public_relation_matcher_streams(
            self.controller,
            (self.stream,),
            learning_rate=1.0e-3,
        )
        selected = set(report["trainable_parameter_names"])
        self.assertTrue(
            any(name.startswith("evidence_pair_encoder.") for name in selected)
        )
        self.assertFalse(
            any(name.startswith("role_encoder.multiplex_pair") for name in selected)
        )
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["directional_arms"], 8)
        self.assertTrue(report["fresh_fast_state_after_every_update"])
        after = dict(self.controller.named_parameters())
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in selected)
        )
        for name in before.keys() - selected:
            self.assertTrue(torch.equal(before[name], after[name]), msg=name)
        legacy_after = self.controller.score_actions(
            fold.masked_task,
            fold.positive_state,
            encoding=encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        no_memory_after = self.controller.score_actions(
            fold.masked_task,
            fold.positive_state,
            encoding=encoding,
            include_pointer_memory=False,
            include_role_memory=False,
        )
        self.assertTrue(torch.equal(legacy_before.logits, legacy_after.logits))
        self.assertTrue(
            torch.equal(
                legacy_before.successor_state_logits,
                legacy_after.successor_state_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                legacy_before.reasoning_node_codes,
                legacy_after.reasoning_node_codes,
            )
        )
        self.assertTrue(
            torch.equal(legacy_before.stop_logit, legacy_after.stop_logit)
        )
        self.assertTrue(torch.equal(no_memory_before.logits, no_memory_after.logits))

    def test_unit_relation_action_calibration_freezes_matcher_and_base(self) -> None:
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
        }
        with self.assertRaises(RuntimeError):
            runner._calibrate_public_relation_action_streams(
                self.controller,
                (self.stream,),
                raw_gate_passed=False,
                learning_rate=1.0e-3,
            )
        report = runner._calibrate_public_relation_action_streams(
            self.controller,
            (self.stream,),
            raw_gate_passed=True,
            learning_rate=1.0e-3,
        )
        selected = set(report["trainable_parameter_names"])
        after = dict(self.controller.named_parameters())
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["directional_arms"], 8)
        self.assertTrue(report["relation_matcher_frozen"])
        self.assertTrue(report["no_memory_logits_exact"])
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in selected)
        )
        for name in before.keys() - selected:
            self.assertTrue(torch.equal(before[name], after[name]), msg=name)

    def test_relation_action_calibration_uses_one_identical_role_off_base(self) -> None:
        with mock.patch.object(
            self.controller,
            "score_actions",
            wraps=self.controller.score_actions,
        ) as scored:
            loss, arms = runner._paired_public_relation_action_loss(
                self.controller,
                self.stream,
            )
        self.assertEqual(arms, 8)
        calibration_calls = tuple(
            call
            for call in scored.call_args_list
            if "include_role_memory" in call.kwargs
        )
        self.assertEqual(len(calibration_calls), 4)
        for call in calibration_calls:
            self.assertFalse(call.kwargs["include_pointer_memory"])
            self.assertFalse(call.kwargs["include_role_memory"])
        loss.backward()
        head_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.evidence_action_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(head_gradient, 0.0)
        self.assertIsNotNone(self.controller.evidence_action_log_gate.grad)
        self.assertGreater(
            float(self.controller.evidence_action_log_gate.grad.abs()),
            0.0,
        )

    def test_relation_slot_permutation_moves_keys_with_role_trace(self) -> None:
        state = _acquire_supports(self.controller, self.stream)
        query = self.stream.queries[0].learner
        original = self.controller.score_actions(query, state)
        snapshot = runner.snapshot_software_reconstruction_state(state)
        trace_slots = self.controller.role_memory.trace_slot_count
        order = torch.arange(trace_slots - 1, -1, -1)
        for name in (
            "keys",
            "values",
            "occupied",
            "write_counts",
            "public_source_action_ids",
            "public_successor_ids",
        ):
            snapshot[f"role.{name}"][:, :trace_slots] = snapshot[
                f"role.{name}"
            ][:, order]
        for name in ("context_trace_keys", "relation_trace_values"):
            snapshot[name][:, :trace_slots] = snapshot[name][:, order]
        permuted = runner.restore_software_reconstruction_state(snapshot)
        changed = self.controller.score_actions(query, permuted)
        torch.testing.assert_close(
            original.evidence_match_scores,
            changed.evidence_match_scores,
            atol=1.0e-6,
            rtol=0.0,
        )

        mismatched_snapshot = runner.snapshot_software_reconstruction_state(state)
        mismatched_snapshot["relation_trace_values"][:, :trace_slots] = (
            mismatched_snapshot["relation_trace_values"][:, order]
        )
        mismatched = runner.restore_software_reconstruction_state(
            mismatched_snapshot
        )
        mismatched_scores = self.controller.score_actions(query, mismatched)
        self.assertFalse(
            torch.allclose(
                original.evidence_match_scores,
                mismatched_scores.evidence_match_scores,
            )
        )


class SoftwarePipelineRolloutAndLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.stream = _stream(supports_per_motif=1)
        self.controller = runner.build_software_pipeline_controller("smoke")

    def test_rollout_chooses_only_declared_components_and_obeys_budget(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        rollout = runner.rollout_software_pipeline(
            self.controller, task, state
        )

        self.assertLessEqual(len(rollout.pipeline.actions), task.max_steps)
        self.assertTrue(
            set(rollout.pipeline.actions) <= set(task.grounded_candidates)
        )
        self.assertLessEqual(len(rollout.step_logits), task.max_steps)
        self.assertTrue(
            all(
                logits.shape == (len(task.grounded_candidates) + 1,)
                for logits in rollout.step_logits
            )
        )

    def test_stop_is_an_autoregressive_decision(self) -> None:
        task = self.stream.queries[0].learner
        with torch.no_grad():
            for parameter in self.controller.stop_head.parameters():
                parameter.zero_()
            self.controller.stop_head[-1].bias.fill_(100.0)
        rollout = runner.rollout_software_pipeline(
            self.controller, task, self.controller.initial_state()
        )

        self.assertTrue(rollout.pipeline.stopped)
        self.assertEqual(rollout.pipeline.actions, ())
        self.assertEqual(len(rollout.step_logits), 1)
        self.assertEqual(
            rollout.selected_indices,
            (len(task.grounded_candidates),),
        )

    def test_cross_package_trace_and_reasoning_losses_backpropagate(self) -> None:
        first, second = self.stream.supports
        state = runner.acquire_public_pipeline_traces(
            self.controller,
            first.learner,
            self.controller.initial_state(),
        ).state
        trace = self.controller.public_trace_losses(second.learner, state).mean()
        reasoning = self.controller.public_backward_reasoning_losses(
            second.learner, state
        ).mean()
        loss = trace + reasoning
        loss.backward()

        role_key_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.local_role_key_encoder.parameters()
            if parameter.grad is not None
        )
        role_value_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.role_value_encoder.parameters()
            if parameter.grad is not None
        )
        reasoner_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.backward_reasoner.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(role_key_gradient, 0.0)
        self.assertGreater(role_value_gradient, 0.0)
        self.assertGreater(reasoner_gradient, 0.0)

    def test_public_leave_one_package_out_reaches_pair_key_effect_and_reasoner(self) -> None:
        stream = _stream(supports_per_motif=2)
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller, evidence.learner, state
            ).state
        loss = self.controller.public_heldout_production_losses(
            heldout, state
        ).mean()
        loss.backward()

        def gradient_sum(module: torch.nn.Module) -> float:
            return sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )

        pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        self.assertGreater(pair_gradient, 0.0)
        self.assertGreater(gradient_sum(self.controller.local_role_key_encoder), 0.0)
        self.assertGreater(
            gradient_sum(self.controller.role_encoder.relative_effect_projection),
            0.0,
        )
        self.assertGreater(gradient_sum(self.controller.backward_reasoner), 0.0)
        self.assertGreater(
            gradient_sum(self.controller.role_encoder.stop_relation_projection),
            0.0,
        )

    def test_public_selected_action_retrieval_contrast_reaches_pair_and_key(self) -> None:
        stream = _stream(supports_per_motif=2)
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller, evidence.learner, state
            ).state
        masked = replace(heldout, observations=())
        encoding = self.controller.encode_task(masked)
        transition = heldout.observations[0].transitions[-1]
        before_index = masked.states.index(transition.before)
        action_index = masked.grounded_candidates.index(transition.action)
        belief = torch.nn.functional.one_hot(
            torch.tensor(before_index),
            len(masked.states),
        ).to(dtype=encoding.role_state_embeddings.dtype)
        scores = self.controller.score_actions(
            masked,
            state,
            current_state_belief=belief,
            steps_remaining=1,
            encoding=encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        contrast = torch.nn.functional.cross_entropy(
            scores.evidence_match_scores.unsqueeze(0),
            torch.tensor((action_index,), dtype=torch.long),
        )
        contrast.backward()
        pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        key_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.local_role_key_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(pair_gradient, 0.0)
        self.assertGreater(key_gradient, 0.0)

    def test_public_retrieval_margin_clears_only_past_hardest_negative(self) -> None:
        clears = torch.tensor((0.00, 0.21, 0.08), requires_grad=True)
        _, clear_margin = self.controller._public_retrieval_contrast_losses(
            clears,
            1,
        )
        self.assertEqual(float(clear_margin.detach()), 0.0)

        misses = torch.tensor((0.00, 0.17, 0.08), requires_grad=True)
        _, missed_margin = self.controller._public_retrieval_contrast_losses(
            misses,
            1,
        )
        self.assertGreater(float(missed_margin.detach()), 0.0)
        expected = runner._PUBLIC_RETRIEVAL_MARGIN - (0.17 - 0.08)
        self.assertAlmostEqual(float(missed_margin.detach()), expected, places=6)

    def test_public_role_causal_hinge_is_class_local_and_detaches_off_arm(self) -> None:
        task = self.stream.supports[0].learner
        transition = task.observations[0].transitions[-1]
        target_index = task.grounded_candidates.index(transition.action)
        components = runner._components_in_candidate_order(task)
        target_key = runner._public_effect_equivalence_key(
            components[target_index]
        )
        twin_index = next(
            index
            for index, component in enumerate(components)
            if index != target_index
            and runner._public_effect_equivalence_key(component) == target_key
        )
        outside_index = next(
            index
            for index in range(len(components))
            if index not in (target_index, twin_index)
        )
        role_on = torch.zeros(len(components), requires_grad=True)
        role_off = torch.zeros(len(components), requires_grad=True)
        baseline = self.controller._public_role_memory_causal_hinge(
            task,
            role_on,
            role_off,
            target_index,
        )
        self.assertIsNotNone(baseline)

        outside_changed = role_on.detach().clone()
        outside_changed[outside_index] = 100.0
        common_shift = role_on.detach().clone()
        common_shift[target_index] = 17.0
        common_shift[twin_index] = 17.0
        target_boost = role_on.detach().clone()
        target_boost[target_index] = 0.2
        twin_boost = role_on.detach().clone()
        twin_boost[twin_index] = 0.2
        outside_loss = self.controller._public_role_memory_causal_hinge(
            task, outside_changed, role_off, target_index
        )
        shifted_loss = self.controller._public_role_memory_causal_hinge(
            task, common_shift, role_off, target_index
        )
        target_loss = self.controller._public_role_memory_causal_hinge(
            task, target_boost, role_off, target_index
        )
        twin_loss = self.controller._public_role_memory_causal_hinge(
            task, twin_boost, role_off, target_index
        )
        torch.testing.assert_close(outside_loss, baseline)
        torch.testing.assert_close(shifted_loss, baseline)
        self.assertLess(
            float(target_loss.detach()),
            float(baseline.detach()),
        )
        self.assertGreater(
            float(twin_loss.detach()),
            float(baseline.detach()),
        )

        baseline.backward()
        self.assertIsNotNone(role_on.grad)
        self.assertIsNone(role_off.grad)

    def test_public_role_causal_hinge_skips_root_singleton_and_stop(self) -> None:
        task = self.stream.supports[0].learner
        root = task.observations[0].transitions[0]
        root_index = task.grounded_candidates.index(root.action)
        logits = torch.zeros(len(task.grounded_candidates))

        self.assertIsNone(
            self.controller._public_role_memory_causal_hinge(
                task,
                logits,
                logits,
                root_index,
            )
        )
        self.assertEqual(runner._public_role_causal_target_count(task), 1)
        state = runner.acquire_public_pipeline_traces(
            self.controller,
            self.stream.supports[1].learner,
            self.controller.initial_state(),
        ).state
        ordinary = self.controller.public_heldout_production_losses(task, state)
        calibrated = self.controller.public_heldout_production_losses(
            task,
            state,
            include_role_memory_causal_hinge=True,
        )
        self.assertEqual(ordinary.numel(), 9)
        self.assertEqual(calibrated.numel(), 10)
        with self.assertRaisesRegex(ValueError, "target index"):
            self.controller._public_role_memory_causal_hinge(
                task,
                logits,
                logits,
                len(task.grounded_candidates),
            )

    def test_public_role_causal_hinge_is_presentation_and_alpha_covariant(self) -> None:
        def measured(task, *, select_twin: bool = False):
            transition = task.observations[0].transitions[-1]
            observed_index = task.grounded_candidates.index(transition.action)
            components = runner._components_in_candidate_order(task)
            target_key = runner._public_effect_equivalence_key(
                components[observed_index]
            )
            twin_index = next(
                index
                for index, component in enumerate(components)
                if index != observed_index
                and runner._public_effect_equivalence_key(component) == target_key
            )
            target_index = twin_index if select_twin else observed_index
            other_index = observed_index if select_twin else twin_index
            role_on = torch.zeros(len(components))
            role_on[target_index] = 0.08
            role_on[other_index] = -0.03
            role_off = torch.zeros(len(components))
            return self.controller._public_role_memory_causal_hinge(
                task,
                role_on,
                role_off,
                target_index,
            )

        original = self.stream.supports[0].learner
        reordered = replace(
            original,
            components=tuple(reversed(original.components)),
            grounded_candidates=tuple(reversed(original.grounded_candidates)),
        )
        rerendered = _stream(
            seed=73_001,
            surface_seed=83_002,
        ).supports[0].learner
        reference = measured(original)
        torch.testing.assert_close(measured(reordered), reference)
        torch.testing.assert_close(measured(rerendered), reference)
        # Swapping the publicly selected twin and its corresponding score is
        # the counterfactual covariance check; no private variant is read.
        torch.testing.assert_close(measured(original, select_twin=True), reference)

    def test_active_causal_hinge_pushes_positive_evidence_gate_up(self) -> None:
        task = self.stream.supports[0].learner
        transition = task.observations[0].transitions[-1]
        target_index = task.grounded_candidates.index(transition.action)
        components = runner._components_in_candidate_order(task)
        target_key = runner._public_effect_equivalence_key(
            components[target_index]
        )
        twin_index = next(
            index
            for index, component in enumerate(components)
            if index != target_index
            and runner._public_effect_equivalence_key(component) == target_key
        )
        evidence = torch.zeros(len(components))
        evidence[target_index] = 0.01
        evidence[twin_index] = 0.0
        role_on = self.controller._evidence_action_contribution(evidence)
        role_off = torch.zeros_like(role_on)
        loss = self.controller._public_role_memory_causal_hinge(
            task,
            role_on,
            role_off,
            target_index,
        )
        self.assertIsNotNone(loss)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(self.controller.evidence_action_log_gate.grad)
        self.assertLess(
            float(self.controller.evidence_action_log_gate.grad.detach()),
            0.0,
        )

    def test_collapsed_role_keys_still_backpropagate_retrieval_margin(self) -> None:
        stream = _stream(supports_per_motif=2)
        with torch.no_grad():
            output = self.controller.role_encoder.multiplex_pair_pool[-1]
            output.weight.mul_(1.0e-7)
            output.bias.copy_(
                torch.linspace(-0.2, 0.2, self.controller.profile.width)
            )
        heldout = stream.supports[0].learner
        state = self.controller.initial_state()
        for evidence in stream.supports[1:]:
            state = runner.acquire_public_pipeline_traces(
                self.controller,
                evidence.learner,
                state,
            ).state
        masked = replace(heldout, observations=())
        encoding = self.controller.encode_task(masked)
        transition = heldout.observations[0].transitions[-1]
        before_index = masked.states.index(transition.before)
        action_index = masked.grounded_candidates.index(transition.action)
        components = runner._components_in_candidate_order(masked)
        selected = components[action_index]
        twin_index = next(
            index
            for index, component in enumerate(components)
            if index != action_index
            and component.input_type == selected.input_type
            and component.output_type == selected.output_type
        )
        similarity = torch.nn.functional.cosine_similarity(
            encoding.role_pair_keys[before_index, action_index],
            encoding.role_pair_keys[before_index, twin_index],
            dim=0,
        )
        self.assertGreater(float(similarity.detach()), 0.999)
        belief = torch.nn.functional.one_hot(
            torch.tensor(before_index),
            len(masked.states),
        ).to(dtype=encoding.role_state_embeddings.dtype)
        scores = self.controller.score_actions(
            masked,
            state,
            current_state_belief=belief,
            steps_remaining=1,
            encoding=encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        retrieval_ce, retrieval_margin = (
            self.controller._public_retrieval_contrast_losses(
                scores.evidence_match_scores,
                action_index,
            )
        )
        self.assertGreater(float(retrieval_margin.detach()), 0.0)
        loss = (
            retrieval_ce
            + runner._PUBLIC_RETRIEVAL_MARGIN_WEIGHT * retrieval_margin
        )
        loss.backward()

        pair_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in self.controller.role_encoder.named_parameters()
            if name.startswith("multiplex_pair") and parameter.grad is not None
        )
        operator_gradient = float(
            self.controller.role_encoder.multiplex_pair_pool[-1]
            .weight.grad.abs().sum()
        )
        key_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.controller.local_role_key_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(pair_gradient, 0.0)
        self.assertGreater(operator_gradient, 0.0)
        self.assertGreater(key_gradient, 0.0)

    def test_backward_reasoning_is_a_causal_ablation(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        enabled = self.controller.score_actions(task, state)
        removed = self.controller.score_actions(
            task, state, include_backward_reasoning=False
        )

        self.assertFalse(torch.allclose(enabled.action_logits, removed.action_logits))
        self.assertEqual(float(removed.reasoning_node_codes.norm()), 0.0)

    def test_scalar_feedback_is_one_bounded_transaction(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        rollout = runner.rollout_software_pipeline(self.controller, task, state)
        before = runner.snapshot_software_reconstruction_state(state)
        feedback = runner.apply_scalar_pipeline_feedback(
            self.controller, task, rollout, 1.0, state
        )

        self.assertTrue(feedback.accepted)
        self.assertEqual(feedback.scalar_observations, 1)
        self.assertEqual(
            runner.snapshot_software_reconstruction_state(feedback.state)[
                "pointer.keys"
            ].tolist(),
            before["pointer.keys"].tolist(),
        )
        for field in ("context_trace_keys", "relation_trace_values"):
            self.assertTrue(
                torch.equal(getattr(feedback.state, field), getattr(state, field))
            )
        rejected = runner.apply_scalar_pipeline_feedback(
            self.controller,
            task,
            rollout,
            0.0,
            state,
            minimum_effect=1.0e9,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            runner.software_reconstruction_state_digest(rejected.state),
            runner.software_reconstruction_state_digest(state),
        )
        for field in ("context_trace_keys", "relation_trace_values"):
            self.assertTrue(
                torch.equal(getattr(rejected.state, field), getattr(state, field))
            )

    def test_scalar_feedback_rebinding_detects_changed_factorized_trace(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        rollout = runner.rollout_software_pipeline(self.controller, task, state)
        occupied = state.role.occupied[0, : self.controller.role_memory.trace_slot_count]
        slot = int(occupied.nonzero().flatten()[-1].item())
        for field in ("context_trace_keys", "relation_trace_values"):
            changed_values = getattr(state, field).detach().clone()
            changed_values[0, slot, 0] += 0.125
            changed = replace(state, **{field: changed_values})
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "retained public traces"):
                    runner.apply_scalar_pipeline_feedback(
                        self.controller,
                        task,
                        rollout,
                        1.0,
                        changed,
                        binding_state=state,
                    )

    def test_scalar_feedback_rejects_stale_state_binding(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        rollout = runner.rollout_software_pipeline(self.controller, task, state)
        updated = runner.apply_scalar_pipeline_feedback(
            self.controller, task, rollout, 0.0, state
        ).state
        with self.assertRaisesRegex(ValueError, "stale"):
            runner.apply_scalar_pipeline_feedback(
                self.controller, task, rollout, 1.0, updated
            )

    def test_centered_preference_uses_repeated_complete_attempts(self) -> None:
        task = self.stream.queries[0].learner
        state = _acquire_supports(self.controller, self.stream)
        first = runner.rollout_software_pipeline(
            self.controller, task, state, greedy=False
        )
        second = runner.rollout_software_pipeline(
            self.controller, task, state, greedy=False
        )
        loss = runner.centered_pipeline_preference_loss(
            (first, second), (0.0, 1.0)
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(first.incoming_state_digest, second.incoming_state_digest)


class SoftwarePipelineEvidenceDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.stream = _stream(supports_per_motif=2)
        self.controller = runner.build_software_pipeline_controller("smoke")

    @staticmethod
    def _summary_values(report) -> dict[tuple[str, str], float | None]:
        return {
            (group, name): value
            for group, metrics in report["summary"].items()
            for name, value in metrics.items()
            if name not in ("individual_count", "all_count")
        }

    def assert_summary_close(self, left, right) -> None:
        left_values = self._summary_values(left)
        right_values = self._summary_values(right)
        self.assertEqual(left_values.keys(), right_values.keys())
        for key, left_value in left_values.items():
            right_value = right_values[key]
            if left_value is None or right_value is None:
                self.assertIsNone(left_value, msg=key)
                self.assertIsNone(right_value, msg=key)
            else:
                tolerance = (
                    max(1.0e-5, 0.05 * max(abs(left_value), abs(right_value)))
                    if key[1] == "all_vs_best_single_retention_mean"
                    else 1.0e-5
                )
                self.assertAlmostEqual(
                    left_value,
                    right_value,
                    delta=tolerance,
                    msg=key,
                )

    def test_public_lopo_report_measures_individual_and_combined_margins(self) -> None:
        tasks = tuple(pair.learner for pair in self.stream.supports)
        report = runner.diagnose_public_lopo_evidence_margins(
            self.controller,
            tasks,
        )

        self.assertEqual(report["support_packages"], 4)
        self.assertEqual(report["heldout_transitions"], 8)
        self.assertTrue(report["public_observations_only"])
        self.assertTrue(report["fresh_fast_state_per_comparison"])
        hardest = report["summary"]["target_vs_hardest"]
        siblings = report["summary"]["effect_equivalent_target_vs_sibling"]
        self.assertEqual(hardest["individual_count"], 24)
        self.assertEqual(hardest["all_count"], 8)
        self.assertEqual(siblings["individual_count"], 12)
        self.assertEqual(siblings["all_count"], 4)
        for metrics in (hardest, siblings):
            self.assertGreaterEqual(metrics["individual_positive_fraction"], 0.0)
            self.assertLessEqual(metrics["individual_positive_fraction"], 1.0)
            self.assertGreaterEqual(metrics["all_positive_fraction"], 0.0)
            self.assertLessEqual(metrics["all_positive_fraction"], 1.0)
            self.assertIn("all_minus_best_single_mean", metrics)
            self.assertIn("all_vs_best_single_retention_mean", metrics)
        class_rows = [
            row for row in report["rows"] if row["effect_class_size"] >= 2
        ]
        singleton_rows = [
            row for row in report["rows"] if row["effect_class_size"] == 1
        ]
        self.assertEqual(len(class_rows), 4)
        self.assertEqual(len(singleton_rows), 4)
        self.assertTrue(
            all(
                row["all_others"]["target_vs_class_sibling"] is not None
                for row in class_rows
            )
        )
        self.assertTrue(
            all(
                row["all_others"]["target_vs_class_sibling"] is None
                for row in singleton_rows
            )
        )

    def test_public_lopo_summary_is_order_and_alpha_covariant(self) -> None:
        tasks = tuple(pair.learner for pair in self.stream.supports)
        baseline = runner.diagnose_public_lopo_evidence_margins(
            self.controller,
            tasks,
        )
        reordered_tasks = tuple(
            replace(
                task,
                components=tuple(reversed(task.components)),
                grounded_candidates=tuple(reversed(task.grounded_candidates)),
                states=tuple(reversed(task.states)),
            )
            for task in tasks
        )
        reordered = runner.diagnose_public_lopo_evidence_margins(
            self.controller,
            tuple(reversed(reordered_tasks)),
        )
        rerendered_stream = _stream(
            seed=73_001,
            surface_seed=83_002,
            supports_per_motif=2,
        )
        rerendered = runner.diagnose_public_lopo_evidence_margins(
            self.controller,
            tuple(pair.learner for pair in rerendered_stream.supports),
        )

        self.assert_summary_close(baseline, reordered)
        self.assert_summary_close(baseline, rerendered)

    def test_public_lopo_diagnostic_has_no_private_dependency(self) -> None:
        source = inspect.getsource(runner.diagnose_public_lopo_evidence_margins)
        self.assertNotIn(".hidden", source)
        self.assertNotIn("judge_software_pipeline_attempt", source)
        self.assertNotIn("mechanism_commitment", source)
        self.assertNotIn("package_commitment", source)


class SoftwarePipelineStateAndHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.stream = _stream(supports_per_motif=1)
        self.controller = runner.build_software_pipeline_controller("smoke")

    def test_snapshot_restore_is_exact_and_snapshot_is_detached(self) -> None:
        state = _acquire_supports(self.controller, self.stream)
        snapshot = runner.snapshot_software_reconstruction_state(state)
        restored = runner.restore_software_reconstruction_state(snapshot)
        digest = runner.software_reconstruction_state_digest(state)

        self.assertEqual(
            runner.software_reconstruction_state_digest(restored), digest
        )
        snapshot["role.keys"].zero_()
        self.assertEqual(runner.software_reconstruction_state_digest(state), digest)
        occupied = state.role.occupied[0, : self.controller.role_memory.trace_slot_count]
        slot = int(occupied.nonzero().flatten()[-1].item())
        for field in ("context_trace_keys", "relation_trace_values"):
            snapshot[field].zero_()
            self.assertEqual(
                runner.software_reconstruction_state_digest(state), digest
            )
            changed_values = getattr(state, field).detach().clone()
            changed_values[0, slot, 0] += 0.25
            changed = replace(state, **{field: changed_values})
            self.assertNotEqual(
                runner.software_reconstruction_state_digest(changed),
                digest,
            )
            incomplete = runner.snapshot_software_reconstruction_state(state)
            del incomplete[field]
            with self.assertRaises(ValueError):
                runner.restore_software_reconstruction_state(incomplete)

    def test_model_digest_is_scalar_safe_and_parameter_sensitive(self) -> None:
        before = runner.software_pipeline_model_digest(self.controller)
        scalar = self.controller.role_match_scale
        saved = scalar.detach().clone()
        with torch.no_grad():
            scalar.add_(1.0)
        changed = runner.software_pipeline_model_digest(self.controller)
        with torch.no_grad():
            scalar.copy_(saved)
        restored = runner.software_pipeline_model_digest(self.controller)
        self.assertRegex(before, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(changed, before)
        self.assertEqual(restored, before)
        with self.assertRaises(TypeError):
            runner.software_pipeline_model_digest(object())

    def test_checkpoint_reloads_weights_and_both_lanes(self) -> None:
        state = _acquire_supports(self.controller, self.stream)
        query = self.stream.queries[0].learner
        expected = self.controller.score_actions(query, state).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "software-pipeline.pt"
            runner.save_software_pipeline_checkpoint(path, self.controller, state)
            loaded, loaded_state = runner.load_software_pipeline_checkpoint(path)
        actual = loaded.score_actions(query, loaded_state).logits.detach()

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            runner.software_reconstruction_state_digest(loaded_state),
            runner.software_reconstruction_state_digest(state),
        )
        for field in ("context_trace_keys", "relation_trace_values"):
            self.assertTrue(
                torch.equal(getattr(loaded_state, field), getattr(state, field))
            )

    def test_checkpoint_rejects_prior_relation_architecture_versions(self) -> None:
        state = _acquire_supports(self.controller, self.stream)
        for version in (
            "angler.phase6-software-pipeline.v2",
            "angler.phase6-software-pipeline.v3",
            "angler.phase6-software-pipeline.v4",
            "angler.phase6-software-pipeline.v5",
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "software-pipeline.pt"
                runner.save_software_pipeline_checkpoint(path, self.controller, state)
                payload = torch.load(path, weights_only=True)
                payload["version"] = version
                torch.save(payload, path)
                with self.assertRaises(RuntimeError):
                    runner.load_software_pipeline_checkpoint(path)

    def test_checkpoint_digest_rejects_either_factor_tamper(self) -> None:
        state = _acquire_supports(self.controller, self.stream)
        trace_slots = self.controller.role_memory.trace_slot_count
        occupied = state.role.occupied[0, :trace_slots].nonzero().flatten()
        slot = int(occupied[-1].item())
        for field in ("context_trace_keys", "relation_trace_values"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "software-pipeline.pt"
                runner.save_software_pipeline_checkpoint(path, self.controller, state)
                payload = torch.load(path, weights_only=True)
                payload["competence_state"][field][0, slot, 0] += 0.125
                torch.save(payload, path)
                with self.assertRaises(RuntimeError):
                    runner.load_software_pipeline_checkpoint(path)

    def test_smoke_training_uses_one_scalar_per_committed_attempt(self) -> None:
        config = replace(
            runner.default_software_pipeline_experiment_config("smoke", seed=47),
            train_mechanisms=1,
            supports_per_motif=2,
            queries_per_mechanism=1,
        )
        evidence_encoder_before = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.evidence_pair_encoder.named_parameters()
        }
        report = runner.train_software_pipeline_controller(
            self.controller,
            config,
            judge=lambda pair, pipeline: 0.0,
        )
        evidence_encoder_after = dict(
            self.controller.evidence_pair_encoder.named_parameters()
        )

        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["scalar_judge_calls"], 0)
        self.assertEqual(report["expected_scalar_judge_calls"], 0)
        self.assertGreater(report["public_trace_terms"], 0)
        self.assertGreater(report["public_retrieval_terms"], 0)
        self.assertGreater(report["public_causal_terms"], 0)
        self.assertGreater(report["public_transfer_terms"], 0)
        self.assertGreater(report["public_reasoning_terms"], 0)
        self.assertEqual(
            report["public_transfer_terms"],
            report["public_trace_terms"]
            + report["public_retrieval_terms"]
            + report["public_reasoning_terms"],
        )
        calibration = report["evidence_calibration"]
        self.assertEqual(calibration["optimizer_steps"], 1)
        self.assertEqual(
            calibration["public_causal_terms"],
            report["public_causal_terms"],
        )
        self.assertTrue(calibration["frozen_parameters_unchanged"])
        self.assertTrue(calibration["no_memory_logits_exact"])
        self.assertEqual(calibration["no_memory_max_delta"], 0.0)
        self.assertTrue(report["main_evidence_action_input_detached"])
        self.assertTrue(report["main_legacy_evidence_path"])
        self.assertTrue(report["main_relation_matcher_excluded"])
        for name, before in evidence_encoder_before.items():
            self.assertTrue(
                torch.equal(before, evidence_encoder_after[name]),
                msg=name,
            )
        self.assertFalse(calibration["evidence_action_input_detached"])
        self.assertTrue(report["fresh_fast_state_per_fold"])
        self.assertEqual(report["complete_pipeline_candidates"], 0)

    def test_evidence_calibration_preserves_base_and_no_memory_logits(self) -> None:
        config = replace(
            runner.default_software_pipeline_experiment_config("smoke", seed=49),
            train_mechanisms=1,
            supports_per_motif=2,
            queries_per_mechanism=1,
        )
        commitment = software_pipeline_mechanism_partition("train")[:1]
        calibration_names = {
            "evidence_action_log_gate",
            "evidence_action_head.0.weight",
            "evidence_action_head.0.bias",
            "evidence_action_head.2.weight",
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.controller.named_parameters()
        }
        report = runner._calibrate_public_evidence_path(
            self.controller,
            config,
            commitment,
        )
        after = dict(self.controller.named_parameters())

        self.assertEqual(report["optimizer_steps"], 1)
        self.assertGreater(report["public_causal_terms"], 0)
        self.assertTrue(report["frozen_parameters_unchanged"])
        self.assertTrue(report["no_memory_logits_exact"])
        self.assertTrue(
            any(
                not torch.equal(before[name], after[name])
                for name in calibration_names
            )
        )
        for name in before.keys() - calibration_names:
            self.assertTrue(torch.equal(before[name], after[name]), msg=name)

    def test_development_harness_runs_all_fixed_arms_once_per_query(self) -> None:
        config = replace(
            runner.default_software_pipeline_experiment_config("smoke", seed=53),
            development_mechanisms=1,
            supports_per_motif=1,
            queries_per_mechanism=1,
        )
        self.controller.requires_grad_(False)
        result = runner.evaluate_software_pipeline_partition(
            self.controller,
            config,
            partition="development",
            mechanism_count=1,
            judge=lambda pair, pipeline: 0.0,
        )

        self.assertEqual(result["scalar_judge_calls"], 11)
        self.assertEqual(result["expected_scalar_judge_calls"], 11)
        row = result["rows"][0]
        for name in (
            "correct",
            "no_evidence",
            "wrong_evidence",
            "shuffled_outcome",
            "pointer_only",
            "a_only",
            "b_only",
            "role_memory_removed",
            "backward_reasoning_removed",
            "episodic_retrieval",
            "state_swap",
            "alpha_rerender",
        ):
            self.assertIn(name, row)
        counts = row["support_acquisition_counts"]
        self.assertEqual(counts["correct"], {"packages": 2, "public_transitions": 4})
        self.assertEqual(counts["no_evidence"], {"packages": 2, "public_transitions": 0})
        self.assertEqual(counts["a_only"], {"packages": 1, "public_transitions": 2})
        self.assertEqual(counts["b_only"], {"packages": 1, "public_transitions": 2})
        self.assertFalse(result["arm_matching"]["same_evidence_work"])
        self.assertTrue(result["arm_matching"]["same_query_rollout_compute"])

    def test_a_and_b_arms_use_evaluator_owned_motif_pure_supports(self) -> None:
        arms = runner.build_software_pipeline_evaluation_arms(self.stream)
        query_motifs = self.stream.queries[0].hidden.required_motifs

        self.assertEqual(
            {pair.hidden.required_motifs for pair in arms["a_only"].supports},
            {(query_motifs[0],)},
        )
        self.assertEqual(
            {pair.hidden.required_motifs for pair in arms["b_only"].supports},
            {(query_motifs[1],)},
        )


class SoftwarePipelineBoundaryTests(unittest.TestCase):
    def test_runner_ast_has_no_private_evaluator_access_or_sequence_constructor(self) -> None:
        source_path = Path(runner.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        forbidden = {
            "hidden",
            "implementations",
            "integration_inputs",
            "required_motifs",
            "reference_pipeline",
            "target_pipeline",
            "bfs",
            "dfs",
            "shortest_path",
            "permutations",
            "combinations",
            "product",
            "planner",
            "solver",
        }
        identifiers = {
            node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(identifiers | attributes))
        self.assertNotIn("BidirectionalOperatorPlanner", source)
        self.assertNotIn("search_teacher_plan", source)
        self.assertNotIn("complete_pipeline_candidates =", source)
        self.assertNotIn("midpoint = len(arms", source)
        self.assertNotIn(
            'row["episodic_retrieval"] = row["backward_reasoning_removed"]',
            source,
        )

        evaluator_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "experiments.evaluators.software_pipeline_reconstruction_suite"
        ]
        self.assertEqual(len(evaluator_imports), 1)
        self.assertTrue(
            all(not alias.name.startswith("_") for alias in evaluator_imports[0].names)
        )

    def test_relation_matcher_dependency_closure_has_no_identity_or_solver_input(self) -> None:
        values = (
            runner.EvidenceOrderedPairEncoder,
            runner.RelationAxisSetReadout,
            runner.SoftwarePipelineController._pool_context_tensor,
            runner.SoftwarePipelineController._pool_relation_tensor,
            runner.SoftwarePipelineController._context_pair_logits,
            runner.SoftwarePipelineController._relation_pair_logits,
            runner.SoftwarePipelineController._relation_evidence_read,
            runner.SoftwarePipelineController._factorized_relation_embeddings,
            runner.SoftwarePipelineController._relation_evidence_scores,
            runner._same_contract_alternative_index,
            runner._relation_credit_task,
            runner._relation_instance_losses,
            runner._relation_valid_set_metrics,
            runner._context_valid_set_training_term,
            runner._anonymous_entropic_stream_objective,
            runner._anonymous_entropic_row_objective,
            runner._relation_credit_stream_objective,
            runner.public_relation_credit_rows,
            runner._fit_public_relation_credit_batches,
            runner.public_paired_relation_fit_rows,
            runner._paired_public_relation_action_loss,
        )
        trees = tuple(
            ast.parse(textwrap.dedent(inspect.getsource(value)))
            for value in values
        )
        identifiers = {
            node.id.lower()
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr.lower()
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        forbidden = {
            "hidden",
            "judge_software_pipeline_attempt",
            "mechanism_commitment",
            "package_commitment",
            "digest",
            "pointer_pair_ids",
            "public_source_action_ids",
            "public_successor_ids",
            "_public_effect_equivalence_key",
            "topology_signature",
            "canonicalize",
            "transform_rule",
            "reference_pipeline",
            "variant",
            "motif",
            "planner",
            "solver",
        }
        self.assertTrue(forbidden.isdisjoint(identifiers | attributes))
        self.assertNotIn(
            "make_software_pipeline_control_stream",
            inspect.getsource(runner.public_relation_credit_rows),
        )
        self.assertNotIn("context_trace_cursor", Path(runner.__file__).read_text())
        self.assertNotIn("relation_trace_cursor", Path(runner.__file__).read_text())

    def test_role_encoder_dependency_closure_excludes_identity_material(self) -> None:
        source_path = Path(runner.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        role_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RenameInvariantRoleEncoder"
        )
        role_helpers = {
            "_state_role_features",
            "_component_role_features",
            "_incidence_graph",
            "_shared_incidence_graphs",
            "_relational_edges",
            "_adjacency_for_nodes",
            "_normalized_relation_pool",
            "_state_arguments",
            "_local_state_component_features",
            "_relative_effect_candidate_features",
            "_local_state_goal_features",
        }
        selected = [
            role_class,
            *(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in role_helpers
            ),
        ]
        forbidden = {
            "digest",
            "mechanism_commitment",
            "mechanism_partition",
            "to_canonical",
            "pointer_features",
            "pointer_pair_ids",
        }
        for node in selected:
            names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            attrs = {
                child.attr.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            self.assertTrue(forbidden.isdisjoint(names | attrs), msg=getattr(node, "name", ""))

    def test_role_encoder_uses_raw_adjacency_not_handcrafted_topology_counts(self) -> None:
        source_path = Path(runner.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        incidence = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_incidence_graph"
        )
        names = {
            child.id.lower()
            for child in ast.walk(incidence)
            if isinstance(child, ast.Name)
        }
        self.assertTrue(
            {"reciprocal", "two_step_out", "node_count", "edge_count"}.isdisjoint(
                names
            )
        )
        lowered = source.lower()
        self.assertNotIn("valid_transition", lowered)
        self.assertNotIn("topology_signature", lowered)
        self.assertNotIn("canonicalize", lowered)

    def test_episodic_control_is_outside_controller_and_training_closure(self) -> None:
        source_path = Path(runner.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        protected = {
            "SoftwarePipelineController",
            "train_software_pipeline_controller",
            "rollout_software_pipeline",
        }
        selected = [
            node
            for node in tree.body
            if getattr(node, "name", None) in protected
        ]
        for node in selected:
            calls = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            self.assertFalse(
                {"_commit_episodic_retrieval", "_episodic_topology_key"} & calls
            )


class SoftwarePipelineConflictReconcilerTests(unittest.TestCase):
    def test_v12_plan_is_fresh_disjoint_and_identity_free(self) -> None:
        plan = runner.public_relation_conflict_fit_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-conflict-reconcile.single.v12",
        )
        self.assertEqual(plan["initialization_seed"], 2_026_082_931)
        self.assertEqual(plan["mixer_initialization_seed"], 2_026_082_932)
        self.assertEqual(plan["stage_updates"], {"relation": 80, "context": 25, "joint": 35})
        training = {
            pair
            for batches in plan["stage_seed_batches"].values()
            for batch in batches
            for pair in batch
        }
        relation_panel = set(plan["relation_context_panel_seed_pairs"])
        final_panel = set(plan["final_panel_seed_pairs"])
        self.assertEqual(len(training), 1_120)
        self.assertEqual(len(relation_panel), 8)
        self.assertEqual(len(final_panel), 8)
        self.assertFalse(training & relation_panel)
        self.assertFalse(training & final_panel)
        self.assertFalse(relation_panel & final_panel)
        self.assertTrue(all(seed >= 3_301_000_001 for pair in training for seed in pair))
        rule = plan["update_rule"]
        self.assertEqual(
            rule["name"],
            "anonymous_learned_blockwise_conflict_reconciliation",
        )
        self.assertEqual(rule["withheld_folds_per_update"], 8)
        self.assertEqual(
            rule["relation_parameter_blocks"],
            ("pair_encoder", "global_readout", "incidence_readout", "comparator"),
        )
        self.assertEqual(
            rule["joint_parameter_blocks"],
            (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "comparator",
                "context",
            ),
        )
        self.assertFalse(rule["stream_identity_input"])
        self.assertFalse(rule["task_identity_input"])
        self.assertFalse(rule["deterministic_gradient_projection"])
        self.assertFalse(
            plan["cluster_pilot_rule"][
                "relation_or_final_gate_used_as_go_threshold"
            ]
        )

    def test_cluster_pilot_runtime_license_requires_causal_direction_change(self) -> None:
        report = _fake_v12_stage_report("relation")
        report["applied_direction_digests"] = report["legacy_direction_digests"]
        assessment = runner._conflict_cluster_pilot_runtime_assessment(report)
        self.assertFalse(assessment["runtime_preconditions_passed"])
        self.assertFalse(
            assessment["runtime_observations"][
                "at_least_one_post_first_applied_direction_differs_from_legacy"
            ]
        )

        changed = list(report["applied_direction_digests"])
        changed[-1] = "sha256:causally-different-direction"
        report["applied_direction_digests"] = tuple(changed)
        assessment = runner._conflict_cluster_pilot_runtime_assessment(report)
        self.assertTrue(assessment["runtime_preconditions_passed"])

    def test_conflict_mixer_is_stream_permutation_equivariant_and_zero_safe(self) -> None:
        torch.manual_seed(551)
        mixer = runner.AnonymousConflictMixer().double()
        losses = torch.tensor(
            (0.01, 0.20, 0.05, 0.40, 0.12, 0.08, 0.31, 0.17),
            dtype=torch.float64,
        )
        base = torch.tensor(
            (0.07, 0.14, 0.08, 0.23, 0.10, 0.09, 0.18, 0.11),
            dtype=torch.float64,
        )
        norms = torch.tensor(
            (
                (0.0, 1.0, 0.5, 2.0, 0.3, 0.8, 1.5, 0.2),
                (0.4, 0.2, 1.2, 0.6, 0.0, 1.1, 0.7, 0.9),
            ),
            dtype=torch.float64,
        )
        gram = torch.eye(8, dtype=torch.float64).repeat(2, 1, 1)
        gram[:, 0, 1] = gram[:, 1, 0] = -0.5
        gram[:, 3, 6] = gram[:, 6, 3] = 0.4
        weights, logits, features = mixer(losses, base, norms, gram)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, dtype=torch.float64))
        self.assertGreater(float(weights.detach().min()), 0.0)
        torch.testing.assert_close(logits, torch.zeros_like(logits))
        torch.testing.assert_close(weights, base.unsqueeze(0).expand(2, -1))

        with torch.no_grad():
            final_weight = mixer.residual_scorer[-1].weight
            final_weight.copy_(
                torch.linspace(
                    -0.20,
                    0.20,
                    final_weight.numel(),
                    dtype=final_weight.dtype,
                ).reshape_as(final_weight)
            )
        weights, logits, features = mixer(losses, base, norms, gram)
        self.assertGreater(float(logits.detach().abs().max()), 0.0)
        order = torch.tensor((7, 3, 1, 5, 0, 6, 2, 4))
        permuted = mixer(
            losses[order],
            base[order],
            norms[:, order],
            gram[:, order][:, :, order],
        )
        torch.testing.assert_close(permuted[0], weights[:, order])
        torch.testing.assert_close(permuted[1], logits[:, order])
        torch.testing.assert_close(permuted[2], features[:, order])

    def test_conflict_meta_loss_reaches_mixer_through_all_withheld_folds(self) -> None:
        torch.manual_seed(552)
        mixer = runner.AnonymousConflictMixer().double()
        losses = torch.tensor(
            (0.03, 0.15, 0.07, 0.31, 0.12, 0.19, 0.23, 0.09),
            dtype=torch.float64,
        )
        base = torch.softmax(losses / 0.10, dim=0)
        norms = torch.tensor(
            (
                (0.4, 1.0, 0.7, 1.8, 0.9, 1.2, 1.5, 0.6),
                (1.1, 0.5, 1.4, 0.8, 1.7, 0.4, 1.0, 1.3),
            ),
            dtype=torch.float64,
        )
        vectors = torch.tensor(
            (
                ((1.0, 0.0), (-0.8, 0.6), (0.9, 0.2), (-0.7, -0.7), (0.2, 1.0), (0.6, -0.5), (-0.4, 0.9), (0.8, 0.3)),
                ((0.3, 1.0), (0.9, -0.4), (-0.6, 0.8), (1.0, 0.1), (-0.9, -0.2), (0.4, 0.7), (0.7, -0.6), (-0.3, 0.95)),
            ),
            dtype=torch.float64,
        )
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        gram = vectors @ vectors.transpose(1, 2)
        objective, diagnostics = runner._conflict_leave_one_out_meta_objective(
            mixer,
            losses,
            base,
            norms,
            gram,
        )
        self.assertEqual(diagnostics["withheld_alignments"].shape, (2, 8))
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        final_gradient = mixer.residual_scorer[-1].weight.grad
        self.assertIsNotNone(final_gradient)
        self.assertGreater(float(final_gradient.abs().sum()), 0.0)

    def test_conflict_meta_gradient_and_update_are_stream_permutation_invariant(self) -> None:
        torch.manual_seed(553)
        mixer = runner.AnonymousConflictMixer().double()
        with torch.no_grad():
            final_weight = mixer.residual_scorer[-1].weight
            final_weight.copy_(
                torch.linspace(
                    -0.15,
                    0.15,
                    final_weight.numel(),
                    dtype=final_weight.dtype,
                ).reshape_as(final_weight)
            )
        permuted_mixer = runner.AnonymousConflictMixer().double()
        permuted_mixer.load_state_dict(mixer.state_dict(), strict=True)
        losses = torch.tensor(
            (0.03, 0.15, 0.07, 0.31, 0.12, 0.19, 0.23, 0.09),
            dtype=torch.float64,
        )
        base = torch.softmax(losses / 0.10, dim=0)
        norms = torch.tensor(
            (
                (0.4, 1.0, 0.7, 1.8, 0.9, 1.2, 1.5, 0.6),
                (1.1, 0.5, 1.4, 0.8, 1.7, 0.4, 1.0, 1.3),
            ),
            dtype=torch.float64,
        )
        vectors = torch.tensor(
            (
                ((1.0, 0.0), (-0.8, 0.6), (0.9, 0.2), (-0.7, -0.7), (0.2, 1.0), (0.6, -0.5), (-0.4, 0.9), (0.8, 0.3)),
                ((0.3, 1.0), (0.9, -0.4), (-0.6, 0.8), (1.0, 0.1), (-0.9, -0.2), (0.4, 0.7), (0.7, -0.6), (-0.3, 0.95)),
            ),
            dtype=torch.float64,
        )
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        gram = vectors @ vectors.transpose(1, 2)
        order = torch.tensor((7, 3, 1, 5, 0, 6, 2, 4))
        objective, diagnostics = runner._conflict_leave_one_out_meta_objective(
            mixer,
            losses,
            base,
            norms,
            gram,
        )
        permuted_objective, permuted_diagnostics = (
            runner._conflict_leave_one_out_meta_objective(
                permuted_mixer,
                losses[order],
                base[order],
                norms[:, order],
                gram[:, order][:, :, order],
            )
        )
        torch.testing.assert_close(objective, permuted_objective, rtol=1.0e-12, atol=1.0e-12)
        for field in ("flat_penalty", "robust_penalty", "mean_kl_from_existing_weights"):
            torch.testing.assert_close(
                diagnostics[field],
                permuted_diagnostics[field],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        for field in ("withheld_alignments", "alignment_penalties"):
            torch.testing.assert_close(
                diagnostics[field][:, order],
                permuted_diagnostics[field],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        objective.backward()
        permuted_objective.backward()
        gradient_total = 0.0
        for (name, parameter), (other_name, other_parameter) in zip(
            mixer.named_parameters(),
            permuted_mixer.named_parameters(),
            strict=True,
        ):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad)
            self.assertIsNotNone(other_parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(
                parameter.grad,
                other_parameter.grad,
                rtol=1.0e-11,
                atol=1.0e-12,
            )
            gradient_total += float(parameter.grad.detach().abs().sum())
        self.assertGreater(gradient_total, 0.0)
        optimizer = torch.optim.AdamW(mixer.parameters(), lr=1.0e-3, weight_decay=0.0)
        permuted_optimizer = torch.optim.AdamW(
            permuted_mixer.parameters(),
            lr=1.0e-3,
            weight_decay=0.0,
        )
        optimizer.step()
        permuted_optimizer.step()
        for name, value in mixer.state_dict().items():
            torch.testing.assert_close(
                value,
                permuted_mixer.state_dict()[name],
                rtol=1.0e-11,
                atol=1.0e-12,
            )

    def test_conflict_blocks_exactly_partition_mutable_relation_parameters(self) -> None:
        controller, mixer = runner.build_public_relation_conflict_system()
        self.assertTrue(runner._public_relation_conflict_system_is_fresh(controller, mixer))
        expected = {
            "relation": (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "comparator",
            ),
            "joint": (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "comparator",
                "context",
            ),
        }
        for stage, expected_names in expected.items():
            blocks = runner._conflict_parameter_blocks(controller, stage)
            self.assertEqual(tuple(blocks), expected_names)
            flattened = tuple(name for names in blocks.values() for name in names)
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(
                set(flattened),
                set(runner._relation_credit_parameter_names(controller, stage)),
            )

    def test_conflict_mixer_digest_detects_and_restores_real_parameter_change(self) -> None:
        _, mixer = runner.build_public_relation_conflict_system()
        before = runner.anonymous_conflict_mixer_digest(mixer)
        parameter = next(mixer.parameters())
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.reshape(-1)[0].add_(0.125)
        changed = runner.anonymous_conflict_mixer_digest(mixer)
        self.assertNotEqual(before, changed)
        with torch.no_grad():
            parameter.copy_(original)
        self.assertEqual(before, runner.anonymous_conflict_mixer_digest(mixer))

        config_variant = runner.AnonymousConflictMixer(anchor_weight=0.25)
        config_variant.load_state_dict(mixer.state_dict(), strict=True)
        self.assertNotEqual(
            before,
            runner.anonymous_conflict_mixer_digest(config_variant),
        )

    def test_one_real_conflict_update_starts_legacy_then_learns_mixer(self) -> None:
        controller, mixer = runner.build_public_relation_conflict_system()
        plan = runner.public_relation_conflict_fit_plan()
        test_seed_batch = tuple(
            (3_701_000_001 + 1_000 * index, 3_801_000_001 + 1_000 * index)
            for index in range(8)
        )
        batches = runner._relation_credit_stream_batches(
            plan["commitments"],
            (test_seed_batch,),
        )
        report = runner._fit_public_relation_conflict_batches(
            controller,
            mixer,
            batches,
            stage="relation",
            require_legacy_first_update=True,
        )
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["streams"], 8)
        self.assertEqual(report["rows"], 32)
        self.assertTrue(report["legacy_first_update_required"])
        self.assertTrue(report["first_update_used_legacy_weights"])
        self.assertTrue(report["mixer_parameters_changed"])
        self.assertGreater(report["mixer_parameter_delta_l2"], 0.0)
        self.assertTrue(report["frozen_parameters_unchanged"])
        self.assertTrue(report["controller_step_mixer_unchanged"])
        self.assertTrue(report["mixer_step_controller_unchanged"])
        self.assertEqual(len(report["withheld_alignments"][0]), 4)
        self.assertTrue(all(len(row) == 8 for row in report["withheld_alignments"][0]))
        self.assertEqual(len(report["block_cosine_grams"][0]), 4)
        self.assertTrue(report["legacy_direction_digests"][0].startswith("sha256:"))
        self.assertTrue(report["applied_direction_digests"][0].startswith("sha256:"))

    def test_first_conflict_controller_update_numerically_matches_legacy_twin(self) -> None:
        legacy, _ = runner.build_public_relation_conflict_system()
        conflict, mixer = runner.build_public_relation_conflict_system()
        plan = runner.public_relation_conflict_fit_plan()
        test_seed_batch = tuple(
            (3_711_000_001 + 1_000 * index, 3_811_000_001 + 1_000 * index)
            for index in range(8)
        )
        batches = runner._relation_credit_stream_batches(
            plan["commitments"],
            (test_seed_batch,),
        )
        legacy_report = runner._fit_public_relation_credit_batches(
            legacy,
            batches,
            stage="relation",
        )
        conflict_report = runner._fit_public_relation_conflict_batches(
            conflict,
            mixer,
            batches,
            stage="relation",
            require_legacy_first_update=True,
        )
        legacy_weights = legacy_report["stream_gradient_weights"][0]
        for block_weights in conflict_report["applied_block_weights"][0]:
            self.assertEqual(tuple(block_weights), tuple(legacy_weights))
        mutable = set(runner._relation_credit_parameter_names(conflict, "relation"))
        legacy_named = dict(legacy.named_parameters())
        conflict_named = dict(conflict.named_parameters())
        maximum_parameter_delta = 0.0
        maximum_gradient_delta = 0.0
        for name, legacy_value in legacy.state_dict().items():
            conflict_value = conflict.state_dict()[name]
            if name in mutable:
                maximum_parameter_delta = max(
                    maximum_parameter_delta,
                    float((legacy_value - conflict_value).detach().abs().max()),
                )
                torch.testing.assert_close(
                    legacy_value,
                    conflict_value,
                    rtol=2.0e-5,
                    atol=2.0e-6,
                )
                legacy_gradient = legacy_named[name].grad
                conflict_gradient = conflict_named[name].grad
                self.assertIsNotNone(legacy_gradient)
                self.assertIsNotNone(conflict_gradient)
                maximum_gradient_delta = max(
                    maximum_gradient_delta,
                    float(
                        (legacy_gradient - conflict_gradient)
                        .detach()
                        .abs()
                        .max()
                    ),
                )
                torch.testing.assert_close(
                    legacy_gradient,
                    conflict_gradient,
                    rtol=2.0e-5,
                    atol=2.0e-6,
                )
            else:
                self.assertTrue(torch.equal(legacy_value, conflict_value), msg=name)
        self.assertLessEqual(maximum_parameter_delta, 2.0e-6)
        self.assertLessEqual(maximum_gradient_delta, 2.0e-6)

    def test_trained_relation_mixer_continues_into_joint_without_rezeroing(self) -> None:
        controller, mixer = runner.build_public_relation_conflict_system()
        plan = runner.public_relation_conflict_fit_plan()
        relation_pairs = tuple(
            (3_721_000_001 + 1_000 * index, 3_821_000_001 + 1_000 * index)
            for index in range(8)
        )
        joint_pairs = tuple(
            (3_731_000_001 + 1_000 * index, 3_831_000_001 + 1_000 * index)
            for index in range(8)
        )
        relation_batches = runner._relation_credit_stream_batches(
            plan["commitments"],
            (relation_pairs,),
        )
        joint_batches = runner._relation_credit_stream_batches(
            plan["commitments"],
            (joint_pairs,),
        )
        relation_report = runner._fit_public_relation_conflict_batches(
            controller,
            mixer,
            relation_batches,
            stage="relation",
            require_legacy_first_update=True,
        )
        self.assertTrue(relation_report["mixer_parameters_changed"])
        relation_terminal_digest = runner.anonymous_conflict_mixer_digest(mixer)
        joint_report = runner._fit_public_relation_conflict_batches(
            controller,
            mixer,
            joint_batches,
            stage="joint",
            require_legacy_first_update=False,
        )
        self.assertFalse(joint_report["legacy_first_update_required"])
        self.assertFalse(joint_report["first_update_used_legacy_weights"])
        self.assertEqual(joint_report["mixer_initial_digest"], relation_terminal_digest)
        self.assertNotEqual(
            joint_report["mixer_terminal_digest"],
            relation_terminal_digest,
        )

    def test_conflict_checkpoint_roundtrip_binds_controller_mixer_and_competence(self) -> None:
        controller, mixer = runner.build_public_relation_conflict_system()
        state = _acquire_supports(controller, _train_stream(seed=3_741_000_001, surface_seed=3_841_000_001))
        with torch.no_grad():
            mixer.residual_scorer[-1].weight.fill_(0.0125)
        controller_digest = runner.software_pipeline_model_digest(controller)
        mixer_digest = runner.anonymous_conflict_mixer_digest(mixer)
        competence_digest = runner.software_reconstruction_state_digest(state)
        system_digest = runner.public_relation_conflict_system_digest(
            controller,
            mixer,
            state,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "conflict.pt"
            runner.save_public_relation_conflict_checkpoint(
                checkpoint,
                controller,
                mixer,
                state,
            )
            restored_controller, restored_mixer, restored_state = (
                runner.load_public_relation_conflict_checkpoint(checkpoint)
            )
            self.assertEqual(
                controller_digest,
                runner.software_pipeline_model_digest(restored_controller),
            )
            self.assertEqual(
                mixer_digest,
                runner.anonymous_conflict_mixer_digest(restored_mixer),
            )
            self.assertEqual(
                competence_digest,
                runner.software_reconstruction_state_digest(restored_state),
            )
            self.assertEqual(
                system_digest,
                runner.public_relation_conflict_system_digest(
                    restored_controller,
                    restored_mixer,
                    restored_state,
                ),
            )
            self.assertEqual(
                runner.public_relation_conflict_parameter_report(controller, mixer),
                runner.public_relation_conflict_parameter_report(
                    restored_controller,
                    restored_mixer,
                ),
            )

            tamper_cases = (
                ("model_state", next(iter(controller.state_dict()))),
                ("mixer_state", next(iter(mixer.state_dict()))),
                ("competence_state", "context_trace_keys"),
            )
            for index, (section, key) in enumerate(tamper_cases):
                payload = torch.load(checkpoint, weights_only=True)
                payload[section][key].reshape(-1)[0].add_(1)
                tampered = Path(directory) / f"tampered-{index}.pt"
                torch.save(payload, tampered)
                with self.assertRaises(RuntimeError):
                    runner.load_public_relation_conflict_checkpoint(tampered)
            payload = torch.load(checkpoint, weights_only=True)
            payload["mixer_config"]["anchor_weight"] = 0.25
            tampered = Path(directory) / "tampered-config.pt"
            torch.save(payload, tampered)
            with self.assertRaises(RuntimeError):
                runner.load_public_relation_conflict_checkpoint(tampered)

    def test_v12_orchestrator_preserves_stage_order_and_never_opens_controls(self) -> None:
        controller, mixer = runner.build_public_relation_conflict_system()
        stream = _train_stream()
        conflict_stages = []
        ordinary_stages = []

        def fake_conflict(
            _controller,
            _mixer,
            _batches,
            *,
            stage,
            require_legacy_first_update,
            **_kwargs,
        ):
            conflict_stages.append((stage, require_legacy_first_update))
            return _fake_v12_stage_report(stage)

        def fake_ordinary(_controller, _batches, *, stage, **_kwargs):
            ordinary_stages.append(stage)
            return _fake_v9_stage_report(stage)

        relation_panel = {
            "streams": 8,
            "rows": 32,
            "relation_supported_rows": 24,
            "streams_with_three_supported_rows": 6,
            "supported_rows_per_stream": (4, 4, 4, 4, 4, 4, 0, 0),
            "valid_slot_count_histogram": (8, 24, 0, 0),
            "context_valid_set_top_one_fraction_supported": 0.80,
            "context_valid_set_mass_mean_supported": 0.60,
        }
        final_panel = {
            **relation_panel,
            "positive_margin_mean": 0.10,
            "negative_margin_mean": -0.10,
            "separation_mean": 0.20,
            "signed_rows": 26,
            "streams_with_three_signed_rows": 7,
        }
        invariants = {
            "permutation_covariant": True,
            "empty_memory_zero_exact": True,
            "permutation_max_delta": 0.0,
        }
        with (
            mock.patch.object(
                runner,
                "_relation_credit_stream_batches",
                return_value=((stream,),),
            ),
            mock.patch.object(
                runner,
                "_relation_credit_panel_streams",
                return_value=(stream,) * 8,
            ),
            mock.patch.object(
                runner,
                "_fit_public_relation_conflict_batches",
                side_effect=fake_conflict,
            ),
            mock.patch.object(
                runner,
                "_fit_public_relation_credit_batches",
                side_effect=fake_ordinary,
            ),
            mock.patch.object(
                runner,
                "evaluate_public_relation_credit_panel",
                side_effect=(relation_panel, relation_panel, final_panel),
            ),
            mock.patch.object(
                runner,
                "_evaluate_public_relation_credit_invariants",
                return_value=invariants,
            ),
            mock.patch.object(
                runner,
                "make_software_pipeline_control_stream",
                side_effect=AssertionError("control stream entered v12"),
            ),
        ):
            report = runner.fit_public_relation_conflict_matcher(controller, mixer)
        self.assertTrue(report["passed"])
        self.assertEqual(conflict_stages, [("relation", True), ("joint", False)])
        self.assertEqual(ordinary_stages, ["context"])
        self.assertFalse(report["development_or_final_access"])
        self.assertEqual(report["wrong_evidence_training_streams"], 0)
        self.assertEqual(report["scalar_judge_calls"], 0)

    def test_conflict_mixer_dependency_closure_has_no_identity_or_solver_input(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        protected = {
            "AnonymousConflictMixer",
            "_conflict_parameter_blocks",
            "_conflict_gradient_geometry",
            "_conflict_direction_diagnostics",
            "_conflict_leave_one_out_meta_objective",
            "_assign_conflict_block_gradients",
            "_conflict_direction_digest",
            "_fit_public_relation_conflict_batches",
            "_conflict_cluster_pilot_runtime_assessment",
            "fit_public_relation_conflict_matcher",
        }
        selected = [
            node for node in tree.body if getattr(node, "name", None) in protected
        ]
        forbidden = {
            "mechanism_commitment",
            "mechanism_partition",
            "motif",
            "topology_seed",
            "surface_seed",
            "judge_software_pipeline_attempt",
            "search_teacher_plan",
            "bidirectionaloperatorplanner",
        }
        for node in selected:
            names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            attrs = {
                child.attr.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            self.assertTrue(forbidden.isdisjoint(names | attrs), msg=getattr(node, "name", ""))


class CapacityMatchedAnonymousClusterTests(unittest.TestCase):
    def test_v13_plan_is_fresh_fixed_and_exactly_paired(self) -> None:
        plan = runner.capacity_matched_relation_cluster_fit_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-anonymous-cluster.paired.v13",
        )
        self.assertEqual(plan["replicate_count"], 3)
        self.assertEqual(plan["updates_per_arm_per_replicate"], 80)
        self.assertEqual(plan["streams_per_arm_per_replicate"], 640)
        self.assertEqual(plan["rows_per_arm_per_replicate"], 2_560)
        self.assertFalse(plan["context_or_joint_training"])
        self.assertFalse(plan["early_stopping"])
        self.assertFalse(plan["adaptive_rerun"])
        self.assertFalse(plan["v12_checkpoint_reuse"])
        self.assertFalse(plan["stream_sharding"])
        self.assertFalse(plan["fixed_cell_roles"])
        self.assertFalse(plan["voting"])
        all_pairs = set()
        for replicate in plan["replicates"]:
            training = {
                pair for batch in replicate["train_seed_batches"] for pair in batch
            }
            panels = (
                set(replicate["panel_a_seed_pairs"])
                | set(replicate["panel_a_rerender_seed_pairs"])
                | set(replicate["panel_b_seed_pairs"])
            )
            self.assertEqual(len(training), 640)
            self.assertEqual(len(panels), 24)
            self.assertFalse(training & panels)
            self.assertFalse(all_pairs & (training | panels))
            all_pairs |= training | panels
            self.assertEqual(
                replicate["monolith_stream_binding_digest"],
                replicate["cluster_stream_binding_digest"],
            )
        v12 = runner.public_relation_conflict_fit_plan()
        v12_pairs = {
            pair
            for batches in v12["stage_seed_batches"].values()
            for batch in batches
            for pair in batch
        } | set(v12["relation_context_panel_seed_pairs"]) | set(
            v12["final_panel_seed_pairs"]
        )
        self.assertFalse(all_pairs & v12_pairs)

    def test_v13_capacity_parity_partition_and_shared_initialization(self) -> None:
        monolith, cluster, monolith_mixer, cluster_mixer = (
            runner.build_capacity_matched_relation_cluster_pair(0)
        )
        report = runner.capacity_matched_relation_cluster_parameter_report(
            monolith,
            cluster,
            monolith_mixer,
            cluster_mixer,
        )
        self.assertEqual(report["monolith_complete_parameters"], 269_010)
        self.assertEqual(report["cluster_complete_parameters"], 269_031)
        self.assertEqual(
            report["cluster_minus_monolith_complete_parameters"],
            21,
        )
        self.assertLessEqual(
            report["absolute_complete_fractional_difference"],
            0.001,
        )
        self.assertEqual(report["monolith_active_trainable_parameters"], 65_366)
        self.assertEqual(report["cluster_active_trainable_parameters"], 65_387)
        self.assertEqual(report["cluster_minus_monolith_active_parameters"], 21)
        self.assertLessEqual(
            report["absolute_active_fractional_difference"],
            0.001,
        )
        self.assertEqual(report["cell_parameters"], (13_850,) * 4)
        self.assertEqual(report["composer_parameters"], 6_583)
        self.assertEqual(report["monolith_mixer_parameters"], 3_404)
        self.assertEqual(report["inert_padding_parameters"], 0)
        self.assertTrue(report["shared_non_relation_parameters_bit_exact"])
        self.assertEqual(
            runner.anonymous_conflict_mixer_digest(monolith_mixer),
            runner.anonymous_conflict_mixer_digest(cluster_mixer),
        )
        self.assertEqual(
            tuple(runner._conflict_parameter_blocks(monolith, "relation")),
            (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "incidence_projection",
                "comparator",
            ),
        )
        self.assertEqual(
            tuple(runner._conflict_parameter_blocks(cluster, "relation")),
            ("cell_0", "cell_1", "cell_2", "cell_3", "composer"),
        )
        pointers = [
            next(cell.parameters()).data_ptr() for cell in cluster.relation_cells
        ]
        self.assertEqual(len(set(pointers)), 4)

    def test_v13_composer_is_cell_permutation_equivariant_and_all_active(self) -> None:
        torch.manual_seed(901)
        composer = runner.AnonymousAllActiveRelationComposer().double()
        with torch.no_grad():
            final = composer.residual_scorer[-1].weight
            final.copy_(
                torch.linspace(-0.3, 0.3, final.numel(), dtype=final.dtype).reshape_as(final)
            )
        query = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        stored = torch.randn(3, 4, 16, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        fused, weights, _, _ = composer(query, stored, logits)
        self.assertGreaterEqual(float(weights.detach().min()), 0.125)
        self.assertLessEqual(float(weights.detach().max()), 0.625)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3, dtype=torch.float64))
        torch.testing.assert_close(fused, (weights * logits).sum(dim=-1))
        fused.square().mean().backward()
        query_gradient = query.grad.detach().clone()
        parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in composer.named_parameters()
        }

        order = torch.tensor((2, 0, 3, 1))
        clone = runner.AnonymousAllActiveRelationComposer().double()
        clone.load_state_dict(composer.state_dict())
        permuted_query = query.detach()[:, order].clone().requires_grad_(True)
        permuted_stored = stored.detach()[:, order].clone().requires_grad_(True)
        permuted_logits = logits.detach()[..., order].clone().requires_grad_(True)
        permuted_fused, permuted_weights, _, _ = clone(
            permuted_query,
            permuted_stored,
            permuted_logits,
        )
        torch.testing.assert_close(permuted_fused, fused.detach())
        torch.testing.assert_close(permuted_weights, weights.detach()[..., order])
        permuted_fused.square().mean().backward()
        torch.testing.assert_close(permuted_query.grad, query_gradient[:, order])
        for name, parameter in clone.named_parameters():
            torch.testing.assert_close(parameter.grad, parameter_gradients[name])

    def test_v13_one_real_update_uses_every_cell_and_matches_ordinary_twin(self) -> None:
        plan = runner.capacity_matched_relation_cluster_fit_plan()
        commitments = plan["commitments"]
        seed_batch = plan["replicates"][0]["train_seed_batches"][0]
        batches = runner._relation_credit_stream_batches(
            commitments,
            (seed_batch,),
        )
        _, learned, _, learned_mixer = (
            runner.build_capacity_matched_relation_cluster_pair(0)
        )
        _, ordinary, _, _ = runner.build_capacity_matched_relation_cluster_pair(0)
        initial_cells = tuple(
            runner._learned_module_digest(cell) for cell in learned.relation_cells
        )
        initial_composer = runner._learned_module_digest(learned.relation_composer)
        ordinary_report = runner._fit_public_relation_credit_batches(
            ordinary,
            batches,
            stage="relation",
        )
        learned_report = runner._fit_public_relation_conflict_batches(
            learned,
            learned_mixer,
            batches,
            stage="relation",
            require_legacy_first_update=True,
        )
        self.assertEqual(ordinary_report["optimizer_steps"], 1)
        self.assertTrue(learned_report["first_update_used_legacy_weights"])
        self.assertEqual(
            tuple(learned_report["parameter_blocks"]),
            ("cell_0", "cell_1", "cell_2", "cell_3", "composer"),
        )
        self.assertTrue(
            all(
                any(float(value) > 0.0 for value in block)
                for block in learned_report["block_gradient_norms"][0]
            )
        )
        selected = runner._relation_credit_parameter_names(learned, "relation")
        learned_parameters = dict(learned.named_parameters())
        ordinary_parameters = dict(ordinary.named_parameters())
        for name in selected:
            torch.testing.assert_close(
                learned_parameters[name],
                ordinary_parameters[name],
                atol=2.0e-6,
                rtol=2.0e-5,
            )
        self.assertTrue(
            all(
                before != runner._learned_module_digest(cell)
                for before, cell in zip(
                    initial_cells,
                    learned.relation_cells,
                    strict=True,
                )
            )
        )
        self.assertNotEqual(
            initial_composer,
            runner._learned_module_digest(learned.relation_composer),
        )

    def test_v13_diagnostic_lesions_are_causal_and_parameter_preserving(self) -> None:
        _, cluster, _, _ = runner.build_capacity_matched_relation_cluster_pair(1)
        cluster.eval()
        with torch.no_grad():
            final = cluster.relation_composer.residual_scorer[-1].weight
            final.copy_(
                torch.linspace(-0.4, 0.4, final.numel()).reshape_as(final)
            )
        query = torch.randn(3, cluster.clustered_relation_width)
        stored = torch.randn(5, cluster.clustered_relation_width)
        before = runner.software_pipeline_model_digest(cluster)
        with torch.no_grad():
            cluster.set_relation_diagnostic_lesion(None)
            learned = cluster._relation_pair_logits(query, stored)
            outputs = {}
            for kind, index in (("uniform", None),) + tuple(
                ("single", value) for value in range(4)
            ) + tuple(("drop", value) for value in range(4)):
                cluster.set_relation_diagnostic_lesion(kind, index)
                outputs[(kind, index)] = cluster._relation_pair_logits(query, stored)
            cluster.set_relation_diagnostic_lesion(None)
        self.assertTrue(any(not torch.equal(learned, value) for value in outputs.values()))
        for index in range(4):
            self.assertFalse(torch.equal(learned, outputs[("drop", index)]))
        self.assertEqual(before, runner.software_pipeline_model_digest(cluster))

    def test_v13_checkpoint_roundtrip_binds_complete_lineage(self) -> None:
        systems = tuple(
            runner.build_capacity_matched_relation_cluster_pair(index)
            for index in range(3)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.pt"
            runner.save_capacity_matched_relation_cluster_checkpoint(path, systems)
            restored = runner.load_capacity_matched_relation_cluster_checkpoint(path)
            self.assertEqual(len(restored), 3)
            for replicate, (original, loaded) in enumerate(zip(systems, restored, strict=True)):
                self.assertEqual(
                    runner.capacity_matched_relation_cluster_system_digest(
                        *original,
                        replicate,
                    ),
                    runner.capacity_matched_relation_cluster_system_digest(
                        *loaded,
                        replicate,
                    ),
                )
            payload = torch.load(path, weights_only=True)
            payload["plan_digest"] = "sha256:tampered"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "identity or seed plan"):
                runner.load_capacity_matched_relation_cluster_checkpoint(path)

    def test_v13_learning_closure_has_no_roles_routing_replay_or_solver(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        protected = {
            "AnonymousRelationCell",
            "AnonymousAllActiveRelationComposer",
            "CapacityMatchedClusterController",
            "_fit_public_relation_conflict_batches",
            "fit_capacity_matched_relation_cluster_pilot",
        }
        selected = [
            node for node in tree.body if getattr(node, "name", None) in protected
        ]
        forbidden = {
            "mechanism_commitment",
            "motif",
            "judge_software_pipeline_attempt",
            "make_software_pipeline_control_stream",
            "rollout_software_pipeline",
            "acquire_public_pipeline_traces",
            "search_teacher_plan",
            "bidirectionaloperatorplanner",
            "vote",
            "replay",
        }
        for node in selected:
            names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            attrs = {
                child.attr.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            self.assertTrue(
                forbidden.isdisjoint(names | attrs),
                msg=getattr(node, "name", ""),
            )


if __name__ == "__main__":
    unittest.main()
