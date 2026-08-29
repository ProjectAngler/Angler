from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


# V19 preflight is deliberately CPU-only and is not a semantic evaluation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.nn import functional as F

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


_SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)


def _load_system() -> v19.V12ChampionPairedGraphContextSystem:
    return v19.load_v12_champion_paired_graph_context_source(_SOURCE_CHECKPOINT)


def _training_batches(count: int = 2):
    plan = v19.v12_champion_paired_graph_context_plan()
    return v12._relation_credit_stream_batches(
        plan["commitments"], plan["training_seed_batches"][:count]
    )


def _first_stream():
    plan = v19.v12_champion_paired_graph_context_plan()
    return v12._relation_credit_panel_streams(
        plan["commitments"], plan["panel_seed_pairs"][0]
    )[0]


def _directed_graph(node_count: int, capacity: int, positions=None):
    if positions is None:
        positions = tuple(range(node_count))
    graph = torch.zeros((capacity, capacity), dtype=torch.bool)
    mask = torch.zeros((capacity,), dtype=torch.bool)
    mask[list(positions)] = True
    for index in range(node_count):
        graph[positions[index], positions[(index + 1) % node_count]] = True
    if node_count >= 4:
        graph[positions[0], positions[2]] = True
        graph[positions[3], positions[1]] = True
    return graph, mask


def _credit_row(logits: torch.Tensor, valid: tuple[bool, ...]):
    probabilities = torch.softmax(
        torch.cat((logits / 0.25, logits.new_zeros(1))), dim=0
    )
    valid_mask = torch.tensor(valid, device=logits.device, dtype=torch.bool)
    zero = logits.sum() * 0.0
    return v19.V19PairedGraphCreditRow(
        heldout_index=0,
        transition_index=0,
        positive_index=0,
        negative_index=1,
        positive_margin=zero,
        negative_margin=zero,
        slot_positive_margins=torch.zeros_like(logits),
        slot_negative_margins=torch.zeros_like(logits),
        context_weights=probabilities[:-1],
        context_null_weight=probabilities[-1],
        context_real_logits=logits,
        valid_mask=valid_mask,
    )


class Phase6V12ChampionPairedGraphContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if not _SOURCE_CHECKPOINT.is_file():
            raise RuntimeError("the frozen terminal V12 checkpoint is required")

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_sources_plan_architecture_and_exclusions(self) -> None:
        root = Path(__file__).resolve().parents[3]
        observed = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            for name in v19.FROZEN_DEPENDENCY_HASHES
        }
        self.assertEqual(observed, v19.FROZEN_DEPENDENCY_HASHES)
        self.assertEqual(v19.frozen_dependency_hashes(), observed)
        plan = v19.v12_champion_paired_graph_context_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-v12-champion-paired-graph-context.v19",
        )
        self.assertEqual(plan["plan_digest"], "sha256:e66d9e4e90e4c3b2ccb704144c7a591009cde57b6367c3e1cc0b9dd64b8d40d5")
        self.assertEqual(plan["context_updates"], 512)
        self.assertEqual((len(plan["training_seed_batches"]), len(plan["panel_seed_pairs"])), (512, 4))
        self.assertEqual(plan["training_seed_batches"][0][0], (9_401_000_001, 9_501_000_001))
        self.assertEqual(
            plan["training_seed_batches"][511][7],
            (9_401_000_001 + 51_100_000 + 7_000, 9_501_000_001 + 51_100_000 + 7_000),
        )
        train = {pair for batch in plan["training_seed_batches"] for pair in batch}
        panels = {pair for panel in plan["panel_seed_pairs"] for pair in panel}
        self.assertEqual((len(train), len(panels), len(train & panels)), (4096, 32, 0))
        system = _load_system()
        report = v19.paired_graph_parameter_report(system.controller, system.mixer)
        self.assertEqual(
            (report["new_trainable_tensors"], report["new_trainable_parameters"]),
            (21, 34_048),
        )
        self.assertEqual(report["complete_learned_system_parameters"], 303_058)
        self.assertEqual(
            tuple(
                name
                for name, parameter in system.controller.named_parameters()
                if parameter.requires_grad
            ),
            v19.MUTABLE_PARAMETER_NAMES,
        )
        source = inspect.getsource(v19)
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any("context_incidence" in value or "context_residual" in value for value in imports))
        self.assertNotIn("self_referential_memory", source)
        self.assertNotIn("responsibilities", source)

    def test_migration_step_zero_and_same_object_primary_lesion_are_exact_v12(self) -> None:
        rng = torch.get_rng_state().clone()
        system = _load_system()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        base, _, _ = v12.load_public_relation_conflict_checkpoint(_SOURCE_CHECKPOINT)
        stream = _first_stream()
        base_state = base.initial_state()
        v19_state = system.controller.initial_state()
        for pair in stream.supports[:3]:
            base_state = v12.acquire_public_pipeline_traces(
                base, pair.learner, base_state
            ).state
            v19_state = v19.acquire_v19_public_pipeline_traces(
                system.controller, pair.learner, v19_state
            ).state
        self.assertEqual(type(v19_state), v19.V19SoftwareReconstructionState)
        expected_snapshot = v12.snapshot_software_reconstruction_state(base_state)
        actual_snapshot = v12.snapshot_software_reconstruction_state(v19_state)
        self.assertEqual(expected_snapshot.keys(), actual_snapshot.keys())
        for name in expected_snapshot:
            self.assertTrue(torch.equal(expected_snapshot[name], actual_snapshot[name]), name)
        task = stream.supports[3].learner
        common_encoding = system.controller.encode_task(task)
        expected = base.score_actions(task, base_state, encoding=common_encoding)
        step_zero = system.controller.score_actions(
            task, v19_state, encoding=common_encoding
        )
        for field in v12.SoftwareStepScores.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(expected, field), getattr(step_zero, field)), field)
        with torch.no_grad():
            system.controller.paired_graph_scorer[-1].weight.copy_(
                torch.linspace(-0.1, 0.1, 32).reshape(1, 32)
            )
        active_before = system.controller.score_actions(
            task, v19_state, encoding=common_encoding
        )
        with system.controller.paired_graph_lesion("zero_residual"):
            lesioned = system.controller.score_actions(
                task, v19_state, encoding=common_encoding
            )
            for field in v12.SoftwareStepScores.__dataclass_fields__:
                self.assertTrue(torch.equal(getattr(expected, field), getattr(lesioned, field)), field)
        self.assertFalse(
            torch.equal(
                active_before.evidence_match_scores, lesioned.evidence_match_scores
            )
        )
        self.assertFalse(torch.equal(active_before.action_logits, lesioned.action_logits))
        self.assertTrue(torch.equal(active_before.stop_logit, lesioned.stop_logit))
        for field in (
            "successor_state_logits",
            "pointer_contexts",
            "role_contexts",
            "outcome_contexts",
            "reasoning_node_codes",
            "current_state_belief",
        ):
            self.assertTrue(torch.equal(getattr(active_before, field), getattr(lesioned, field)), field)
        active_after = system.controller.score_actions(
            task, v19_state, encoding=common_encoding
        )
        for field in v12.SoftwareStepScores.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(active_before, field), getattr(active_after, field)), field)
        with self.assertRaises(TypeError):
            system.controller.score_actions(task, base_state)
        with self.assertRaises(TypeError):
            system.controller.score_actions(task, v19_state, encoding=base.encode_task(task))

        one_slot_snapshot = v19.snapshot_v19_reconstruction_state(v19_state)
        trace_slots = system.controller.role_memory.trace_slot_count
        occupied_indices = one_slot_snapshot["role.occupied"][0, :trace_slots].nonzero().flatten()
        self.assertGreaterEqual(occupied_indices.numel(), 2)
        keep = int(occupied_indices[0].item())
        for slot in range(trace_slots):
            if slot == keep:
                continue
            for name in (
                "role.keys",
                "role.values",
                "role.write_counts",
                "role.public_source_action_ids",
                "role.public_successor_ids",
                "context_trace_keys",
                "relation_trace_values",
                "context_trace_graphs",
                "context_trace_graph_masks",
            ):
                one_slot_snapshot[name][0, slot].zero_()
            one_slot_snapshot["role.occupied"][0, slot] = False
        one_slot = v19.restore_v19_reconstruction_state(one_slot_snapshot)
        one_slot_base = v12.restore_software_reconstruction_state(
            {
                name: value
                for name, value in one_slot_snapshot.items()
                if name not in {"context_trace_graphs", "context_trace_graph_masks"}
            }
        )
        expected_one = base.score_actions(task, one_slot_base, encoding=common_encoding)
        actual_one = system.controller.score_actions(task, one_slot, encoding=common_encoding)
        for field in v12.SoftwareStepScores.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(expected_one, field), getattr(actual_one, field)), field)
        encoded = system.controller.encode_task(task)
        components = v12._components_in_candidate_order(task)
        for action_index, candidate in enumerate(components):
            predecessors = tuple(
                predecessor
                for predecessor in components
                if predecessor.output_type == candidate.input_type
            )
            self.assertLessEqual(len(predecessors), 1)
            mask = encoded.context_graph_masks[action_index]
            if not predecessors:
                self.assertFalse(bool(mask.any().item()))
                self.assertEqual(
                    int(torch.count_nonzero(encoded.context_graph_adjacencies[action_index]).item()),
                    0,
                )
                self.assertEqual(
                    float(encoded.relation_context_embeddings[action_index].norm().item()),
                    0.0,
                )
                continue
            _, adjacency = v12._incidence_graph(predecessors[0])
            node_count = len(adjacency)
            expected_graph = torch.tensor(adjacency, dtype=torch.bool)
            self.assertTrue(
                torch.equal(
                    encoded.context_graph_adjacencies[action_index, :node_count, :node_count],
                    expected_graph,
                )
            )
            self.assertTrue(bool(mask[:node_count].all().item()))
            self.assertFalse(bool(mask[node_count:].any().item()))

    def test_unsupported_query_has_zero_sidecar_skips_matcher_and_keeps_v12_row(self) -> None:
        controller = _load_system().controller
        torch.manual_seed(8_019)
        query_codes = F.normalize(torch.randn(2, 32), dim=-1)
        stored_codes = F.normalize(torch.randn(3, 32), dim=-1)
        supported_graph, supported_mask = _directed_graph(5, 32)
        query_graphs = torch.stack((torch.zeros_like(supported_graph), supported_graph))
        query_masks = torch.stack((torch.zeros_like(supported_mask), supported_mask))
        stored_graphs = torch.stack(tuple(_directed_graph(6 + index, 32)[0] for index in range(3)))
        stored_masks = torch.stack(tuple(_directed_graph(6 + index, 32)[1] for index in range(3)))
        with mock.patch.object(
            controller,
            "_paired_graph_raw_residual",
            wraps=controller._paired_graph_raw_residual,
        ) as spy:
            adjusted = controller._paired_graph_context_logits(
                query_codes,
                query_graphs,
                query_masks,
                stored_codes,
                stored_graphs,
                stored_masks,
            )
        inherited = controller._context_pair_logits(query_codes, stored_codes)
        self.assertTrue(torch.equal(adjusted[0], inherited[0]))
        self.assertEqual(spy.call_count, 3)

    def test_temperature_log_partition_preserves_null_and_rejects_arithmetic_center(self) -> None:
        controller = _load_system().controller
        torch.manual_seed(4_919)
        query_codes = F.normalize(torch.randn(2, 32), dim=-1)
        stored_codes = F.normalize(torch.randn(3, 32), dim=-1)
        graphs = []
        masks = []
        for count in (5, 6, 7):
            graph, mask = _directed_graph(count, 32)
            graphs.append(graph)
            masks.append(mask)
        query_graphs = torch.stack(graphs[:2])
        query_masks = torch.stack(masks[:2])
        stored_graphs = torch.stack(graphs)
        stored_masks = torch.stack(masks)
        raw_values = iter((0.8, -0.4, 0.1, -0.2, 0.7, -0.6))

        def fake_raw(*_args):
            return query_codes.new_tensor(next(raw_values))

        inherited = controller._context_pair_logits(query_codes, stored_codes)
        with mock.patch.object(controller, "_paired_graph_raw_residual", side_effect=fake_raw):
            adjusted = controller._paired_graph_context_logits(
                query_codes,
                query_graphs,
                query_masks,
                stored_codes,
                stored_graphs,
                stored_masks,
            )
        self.assertTrue(
            torch.allclose(
                torch.logsumexp(adjusted.double() / 0.25, dim=-1),
                torch.logsumexp(inherited.double() / 0.25, dim=-1),
                atol=2.0e-7,
                rtol=0.0,
            )
        )
        old_weights = torch.softmax(
            torch.cat((inherited / 0.25, inherited.new_zeros(2, 1)), dim=-1), dim=-1
        )
        new_weights = torch.softmax(
            torch.cat((adjusted / 0.25, adjusted.new_zeros(2, 1)), dim=-1), dim=-1
        )
        self.assertTrue(torch.allclose(old_weights[:, -1], new_weights[:, -1], atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(old_weights[:, :-1].sum(-1), new_weights[:, :-1].sum(-1), atol=1.0e-7, rtol=0.0))
        residual = 0.5 * torch.tanh(torch.tensor(((0.8, -0.4, 0.1), (-0.2, 0.7, -0.6))))
        arithmetic = inherited + residual - residual.mean(dim=-1, keepdim=True)
        self.assertGreater(
            float(
                (
                    torch.logsumexp(arithmetic.double() / 0.25, dim=-1)
                    - torch.logsumexp(inherited.double() / 0.25, dim=-1)
                ).abs().max().item()
            ),
            1.0e-3,
        )
        one = controller._paired_graph_context_logits(
            query_codes,
            query_graphs,
            query_masks,
            stored_codes[:1],
            stored_graphs[:1],
            stored_masks[:1],
        )
        self.assertTrue(torch.equal(one, controller._context_pair_logits(query_codes, stored_codes[:1])))

    def test_crop_before_encoder_padding_independent_permutations_and_symmetry(self) -> None:
        controller = _load_system().controller
        with torch.no_grad():
            controller.paired_graph_scorer[-1].weight.copy_(
                torch.linspace(-0.2, 0.2, 32).reshape(1, 32)
            )
        compact_q, compact_q_mask = _directed_graph(5, 5)
        compact_s, compact_s_mask = _directed_graph(7, 7)
        q_positions = (1, 4, 8, 13, 19)
        s_positions = (0, 3, 9, 14, 18, 24, 30)
        padded_q, padded_q_mask = _directed_graph(5, 32, q_positions)
        padded_s, padded_s_mask = _directed_graph(7, 32, s_positions)
        poisoned_q = padded_q.clone()
        poisoned_s = padded_s.clone()
        poisoned_q[31, 30] = True
        poisoned_q[30, q_positions[0]] = True
        poisoned_s[31, 29] = True
        expected = controller._paired_graph_raw_residual(
            compact_q, compact_q_mask, compact_s, compact_s_mask
        )
        observed = controller._paired_graph_raw_residual(
            poisoned_q, padded_q_mask, poisoned_s, padded_s_mask
        )
        self.assertTrue(torch.allclose(expected, observed, atol=1.0e-6, rtol=0.0))
        graph23_q, mask23_q = _directed_graph(5, 23)
        graph23_s, mask23_s = _directed_graph(7, 23)
        result23 = controller._paired_graph_raw_residual(graph23_q, mask23_q, graph23_s, mask23_s)
        result32 = controller._paired_graph_raw_residual(padded_q, padded_q_mask, padded_s, padded_s_mask)
        self.assertTrue(torch.allclose(result23, result32, atol=1.0e-6, rtol=0.0))
        q_perm = torch.tensor((3, 0, 4, 1, 2))
        s_perm = torch.tensor((6, 2, 0, 4, 1, 5, 3))
        permuted_q = compact_q.index_select(0, q_perm).index_select(1, q_perm)
        permuted_s = compact_s.index_select(0, s_perm).index_select(1, s_perm)
        independently_permuted = controller._paired_graph_raw_residual(
            permuted_q, compact_q_mask, permuted_s, compact_s_mask
        )
        self.assertTrue(torch.allclose(expected, independently_permuted, atol=2.0e-6, rtol=0.0))
        reversed_pair = controller._paired_graph_raw_residual(
            compact_s, compact_s_mask, compact_q, compact_q_mask
        )
        self.assertTrue(torch.allclose(expected, reversed_pair, atol=2.0e-6, rtol=0.0))
        reversed_edges = controller._paired_graph_raw_residual(
            compact_q.transpose(0, 1), compact_q_mask, compact_s, compact_s_mask
        )
        self.assertGreater(float((expected - reversed_edges).abs().item()), 1.0e-8)
        with self.assertRaises(ValueError):
            controller._graph_node_tokens(
                torch.zeros((33, 33), dtype=torch.bool), torch.ones(33, dtype=torch.bool)
            )

    def test_objective_uses_only_informative_rows_and_exact_list_pair_terms(self) -> None:
        logits = torch.tensor((0.3, -0.2, 0.1), requires_grad=True)
        informative = _credit_row(logits, (True, False, False))
        unsupported = _credit_row(logits, (False, False, False))
        all_valid = _credit_row(logits, (True, True, True))
        loss, diagnostic = v19._paired_graph_row_objective(informative)
        self.assertIsNotNone(loss)
        probabilities = torch.softmax(logits / 0.25, dim=0)
        expected_list = -torch.log(probabilities[0])
        differences = logits[0] - logits[1:]
        expected_pair = 0.05 * torch.nn.functional.softplus((0.10 - differences) / 0.05).mean()
        expected = 0.5 * expected_list + 0.5 * expected_pair
        self.assertTrue(torch.allclose(loss, expected, atol=1.0e-7, rtol=0.0))
        self.assertAlmostEqual(diagnostic["list_loss"], float(expected_list.detach()), places=7)
        self.assertAlmostEqual(diagnostic["pair_loss"], float(expected_pair.detach()), places=7)
        self.assertEqual(v19._paired_graph_row_objective(unsupported)[0], None)
        self.assertEqual(v19._paired_graph_row_objective(all_valid)[0], None)
        combined, summary = v19._paired_graph_objective(((informative, unsupported, all_valid),))
        self.assertTrue(torch.equal(combined, loss))
        self.assertEqual((summary["informative_rows"], summary["excluded_rows"]), (1, 2))
        combined.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(int(torch.count_nonzero(logits.grad).item()), 0)

    def test_inherited_relation_qualifying_stream_boundary_is_three_of_four(self) -> None:
        logits = torch.tensor((0.3, -0.2, 0.1))
        supported = _credit_row(logits, (True, False, False))
        unsupported = _credit_row(logits, (False, False, False))
        three_of_four = v19._credit_rows_metrics(
            ((supported, supported, supported, unsupported),)
        )
        two_of_four = v19._credit_rows_metrics(
            ((supported, supported, unsupported, unsupported),)
        )
        self.assertEqual(
            (three_of_four["supported_rows"], three_of_four["qualifying_streams"]),
            (3, 1),
        )
        self.assertEqual(
            (two_of_four["supported_rows"], two_of_four["qualifying_streams"]),
            (2, 0),
        )
        thirty_two_streams = v19._credit_rows_metrics(
            tuple(
                (supported, supported, supported, unsupported)
                for _ in range(32)
            )
        )
        self.assertEqual(
            (
                thirty_two_streams["supported_rows"],
                thirty_two_streams["qualifying_streams"],
            ),
            (96, 32),
        )

    def test_raw_bool_state_atomic_alignment_rollback_and_live_reencoding(self) -> None:
        system = _load_system()
        controller = system.controller
        stream = _first_stream()
        state = controller.initial_state()
        for pair in stream.supports[:3]:
            acquired = v19.acquire_v19_public_pipeline_traces(
                controller, pair.learner, state
            )
            self.assertGreater(acquired.role_writes, 0)
            state = acquired.state
        self.assertEqual(state.context_trace_graphs.dtype, torch.bool)
        self.assertEqual(state.context_trace_graph_masks.dtype, torch.bool)
        trace_slots = controller.role_memory.trace_slot_count
        present = state.context_trace_graph_masks[0, :trace_slots].any(dim=-1)
        expected = (
            state.role.occupied[0, :trace_slots]
            & (state.context_trace_keys[0, :trace_slots].norm(dim=-1) > 1.0e-8)
            & (state.relation_trace_values[0, :trace_slots].norm(dim=-1) > 1.0e-8)
        )
        self.assertTrue(torch.equal(present, expected))
        before_state_digest = v19.v19_reconstruction_state_digest(state)
        task = stream.supports[3].learner
        with torch.no_grad():
            controller.paired_graph_scorer[-1].weight.copy_(
                torch.linspace(-0.1, 0.1, 32).reshape(1, 32)
            )
        before = controller.score_actions(task, state).evidence_match_scores
        with torch.no_grad():
            controller.paired_graph_node_encoder.node_projection[1].weight.add_(0.05)
        after = controller.score_actions(task, state).evidence_match_scores
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(v19.v19_reconstruction_state_digest(state), before_state_digest)
        snapshot = v19.snapshot_v19_reconstruction_state(state)
        with mock.patch.object(controller, "score_actions", side_effect=RuntimeError("reject")):
            rejected = v19.acquire_v19_public_pipeline_traces(
                controller, stream.supports[3].learner, state
            )
        self.assertEqual(rejected.role_writes, 0)
        self.assertIs(rejected.state, state)
        for name, value in snapshot.items():
            self.assertTrue(torch.equal(value, v19.snapshot_v19_reconstruction_state(state)[name]), name)

    def test_common_matcher_reached_by_score_credit_and_fit_with_staged_gradients(self) -> None:
        stream = _first_stream()
        score_system = _load_system()
        controller = score_system.controller
        state = controller.initial_state()
        for pair in stream.supports[:3]:
            state = v19.acquire_v19_public_pipeline_traces(
                controller, pair.learner, state
            ).state
        with torch.no_grad():
            controller.paired_graph_scorer[-1].weight.fill_(0.01)
        with mock.patch.object(
            controller,
            "_paired_graph_evidence_read",
            wraps=controller._paired_graph_evidence_read,
        ) as spy:
            controller.score_actions(stream.supports[3].learner, state)
            score_calls = spy.call_count
            v19.public_paired_graph_credit_rows(controller, stream)
            credit_calls = spy.call_count - score_calls
        self.assertGreater(score_calls, 0)
        self.assertGreater(credit_calls, 0)

        fit_system = _load_system()
        with mock.patch.object(
            fit_system.controller,
            "_paired_graph_evidence_read",
            wraps=fit_system.controller._paired_graph_evidence_read,
        ) as fit_spy, mock.patch.object(
            v12,
            "public_relation_credit_rows",
            side_effect=AssertionError("base V12 credit builder is forbidden"),
        ):
            fit = v19._fit_paired_graph_batches(fit_system, _training_batches(2))
        self.assertGreater(fit_spy.call_count, 0)
        self.assertTrue(fit["first_head_gradient_nonzero"])
        self.assertTrue(fit["first_upstream_gradients_exact_zero"])
        self.assertTrue(fit["later_upstream_gradient_reached"])
        self.assertEqual(fit["optimizer_steps"], 2)
        self.assertEqual(
            set(fit["gradient_reached_parameter_names"]),
            set(v19.MUTABLE_PARAMETER_NAMES),
        )
        self.assertEqual(
            set(fit["changed_parameter_names"]),
            set(v19.MUTABLE_PARAMETER_NAMES),
        )
        self.assertEqual(set(fit_system.optimizer_state["state"]), set(v19.MUTABLE_PARAMETER_NAMES))
        optimizer = v19.restore_paired_graph_optimizer(fit_system)
        self.assertEqual(
            tuple(float(group["lr"]) for group in optimizer.param_groups),
            (3.0e-4, 1.0e-3),
        )
        for group in optimizer.param_groups:
            self.assertEqual(group["weight_decay"], 0.0)
            self.assertEqual(group["betas"], (0.9, 0.999))
            self.assertEqual(group["eps"], 1.0e-8)
            self.assertFalse(group["amsgrad"])
            self.assertFalse(group["maximize"])
            self.assertFalse(group["foreach"])
            self.assertFalse(group["fused"])
            self.assertFalse(group["capturable"])
            self.assertFalse(group["differentiable"])

    def test_slot_candidate_covariance_and_checkpoint_split_continuation(self) -> None:
        system = _load_system()
        stream = _first_stream()
        ordinary = v19.public_paired_graph_credit_rows(system.controller, stream)
        reversed_slots = v19.public_paired_graph_credit_rows(
            system.controller, stream, reverse_evidence_order=True
        )
        for left, right in zip(ordinary, reversed_slots, strict=True):
            self.assertTrue(torch.equal(left.context_weights, right.context_weights.flip(0)))
            self.assertTrue(torch.equal(left.context_real_logits, right.context_real_logits.flip(0)))
            self.assertTrue(torch.equal(left.valid_mask, right.valid_mask.flip(0)))
        reversed_candidates = v19.public_paired_graph_credit_rows(
            system.controller, stream, reverse_public_presentation=True
        )
        for left, right in zip(ordinary, reversed_candidates, strict=True):
            self.assertEqual(int(left.valid_mask.sum()), int(right.valid_mask.sum()))
            self.assertTrue(
                torch.allclose(
                    left.context_weights,
                    right.context_weights,
                    atol=2.0e-6,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    left.context_real_logits,
                    right.context_real_logits,
                    atol=2.0e-6,
                    rtol=0.0,
                )
            )
            self.assertTrue(torch.equal(left.valid_mask, right.valid_mask))
        state = system.controller.initial_state()
        for pair in stream.supports[:3]:
            state = v19.acquire_v19_public_pipeline_traces(
                system.controller, pair.learner, state
            ).state
        original_task = stream.supports[3].learner
        reversed_task = replace(
            original_task,
            components=tuple(reversed(original_task.components)),
            grounded_candidates=tuple(reversed(original_task.grounded_candidates)),
            states=tuple(reversed(original_task.states)),
        )
        with torch.no_grad():
            system.controller.paired_graph_scorer[-1].weight.fill_(0.01)
        original_scores = system.controller.score_actions(original_task, state)
        reversed_scores = system.controller.score_actions(reversed_task, state)
        reindex = torch.tensor(
            tuple(
                reversed_task.grounded_candidates.index(action)
                for action in original_task.grounded_candidates
            ),
            dtype=torch.long,
        )
        self.assertTrue(
            torch.allclose(
                original_scores.evidence_match_scores,
                reversed_scores.evidence_match_scores.index_select(0, reindex),
                atol=2.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                original_scores.action_logits,
                reversed_scores.action_logits.index_select(0, reindex),
                atol=2.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.equal(original_scores.stop_logit, reversed_scores.stop_logit))

        continuous = _load_system()
        split = _load_system()
        batches = _training_batches(2)
        v19._fit_paired_graph_batches(continuous, batches)
        v19._fit_paired_graph_batches(split, batches[:1])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v19.pt"
            v19.save_v12_champion_paired_graph_context_checkpoint(checkpoint, split)
            resumed = v19.load_v12_champion_paired_graph_context_checkpoint(checkpoint)
            v19._fit_paired_graph_batches(resumed, batches[1:])
            self.assertEqual(
                v19.paired_graph_mutable_digest(continuous.controller),
                v19.paired_graph_mutable_digest(resumed.controller),
            )
            self.assertEqual(
                v19.paired_graph_optimizer_digest(continuous.optimizer_state),
                v19.paired_graph_optimizer_digest(resumed.optimizer_state),
            )
            self.assertEqual(continuous.context_updates, resumed.context_updates)
            self.assertEqual(
                v19.paired_graph_system_digest(continuous),
                v19.paired_graph_system_digest(resumed),
            )
            original_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            tamperings = []
            payload = copy.deepcopy(original_payload)
            payload["model_state"][v19.MUTABLE_PARAMETER_NAMES[0]].reshape(-1)[0] += 1.0
            tamperings.append(payload)
            payload = copy.deepcopy(original_payload)
            payload["optimizer_state"]["state"][v19.MUTABLE_PARAMETER_NAMES[0]]["exp_avg"].reshape(-1)[0] += 1.0
            tamperings.append(payload)
            payload = copy.deepcopy(original_payload)
            payload["competence_state"]["context_trace_graphs"][0, 0, 0, 0] = True
            tamperings.append(payload)
            payload = copy.deepcopy(original_payload)
            payload["context_updates"] = 2
            tamperings.append(payload)
            for index, payload in enumerate(tamperings):
                tampered = Path(directory) / f"tampered-{index}.pt"
                torch.save(payload, tampered)
                with self.assertRaises(RuntimeError):
                    v19.load_v12_champion_paired_graph_context_checkpoint(tampered)


if __name__ == "__main__":
    unittest.main()
