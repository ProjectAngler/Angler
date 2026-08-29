from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
import inspect
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from angler.procedures.skill_memory import (  # noqa: E402
    procedural_skill_state_digest,
    restore_procedural_skill_state,
    snapshot_procedural_skill_state,
)
from experiments.evaluators import skill_memory_suite as suite  # noqa: E402
from experiments.runners import phase5_skill_memory_stream as runner  # noqa: E402


def _is_reversible_state_key(name: str) -> bool:
    return name == "reversible_transition_mode" or name.startswith(
        "reversible_procedure_transition."
    )


class Phase5SkillMemoryStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(85_101)
        torch.set_num_threads(1)
        self.profile = runner._PROFILES["smoke"]
        self.policy = runner.SkillMemoryPolicy(self.profile)
        self.partition = suite.make_skill_memory_partition(
            "train", 85_101, instances_per_program=4
        )

    def _task_for_expression(
        self,
        expression: suite.PublicSkillExpression,
    ) -> suite.PublicSkillMemoryTask:
        template = self.partition.tasks[0].learner
        return suite.PublicSkillMemoryTask(
            template.items,
            template.public_flag,
            expression,
        )

    def _acquire_expression(
        self,
        expression: suite.PublicSkillExpression,
        state,
    ):
        public_task = self._task_for_expression(expression)
        proposal = runner.propose_task(
            self.policy,
            public_task,
            state,
            greedy=False,
            temperature=1.0,
        )
        staged = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            1.0,
            state,
        )
        self.assertIsNot(staged.candidate_state, state)
        return staged.candidate_state

    def test_transition_only_composition_validates_and_preserves_leaf_policy(
        self,
    ) -> None:
        leaf = next(
            pair.learner.request
            for pair in self.partition.tasks
            if not pair.learner.request.children
        )
        public_task = self._task_for_expression(leaf)
        state = self._acquire_expression(leaf, self.policy.initial_state(1))

        implicit_legacy = self.policy.score_task(public_task, state)
        explicit_legacy = self.policy.score_task(
            public_task,
            state,
            transition_only_composition=False,
        )
        transition_only = self.policy.score_task(
            public_task,
            state,
            transition_only_composition=True,
        )

        for observed in (explicit_legacy, transition_only):
            self.assertTrue(torch.equal(observed.logits, implicit_legacy.logits))
            self.assertTrue(
                torch.equal(observed.memory_bias, implicit_legacy.memory_bias)
            )
            self.assertTrue(
                torch.equal(
                    observed.composition_logits,
                    implicit_legacy.composition_logits,
                )
            )
        with self.assertRaisesRegex(
            TypeError,
            "transition_only_composition must be boolean",
        ):
            self.policy.score_task(
                public_task,
                state,
                transition_only_composition=1,
            )

    def test_transition_only_composition_removes_unary_decoder_bias(self) -> None:
        unary = next(
            pair.learner.request
            for pair in self.partition.tasks
            if len(pair.learner.request.children) == 1
        )
        state = self.policy.initial_state(1)
        state = self._acquire_expression(unary.children[0], state)
        state = self._acquire_expression(unary, state)
        public_task = self._task_for_expression(unary)

        legacy = self.policy.score_task(public_task, state)
        transition_only = self.policy.score_task(
            public_task,
            state,
            transition_only_composition=True,
        )

        self.assertGreater(float(legacy.memory_bias.abs().sum().item()), 0.0)
        self.assertTrue(
            torch.equal(
                transition_only.memory_bias,
                torch.zeros_like(transition_only.memory_bias),
            )
        )
        self.assertTrue(
            torch.equal(
                transition_only.composition_logits,
                legacy.composition_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                transition_only.logits,
                transition_only.composition_logits,
            )
        )
        self.assertFalse(torch.equal(transition_only.logits, legacy.logits))

    def test_transition_only_binary_composes_unary_transitions_and_ablates(self) -> None:
        unaries = []
        binary_symbol = None
        for pair in self.partition.tasks:
            request = pair.learner.request
            if len(request.children) == 1 and request not in unaries:
                unaries.append(request)
            elif len(request.children) == 2 and binary_symbol is None:
                binary_symbol = request.symbol
        self.assertGreaterEqual(len(unaries), 2)
        self.assertIsNotNone(binary_symbol)
        first, second = unaries[:2]
        root = suite.PublicSkillExpression(
            binary_symbol,
            (first, second),
        )
        state = self.policy.initial_state(1)
        acquired = set()
        for expression in (
            first.children[0],
            first,
            second.children[0],
            second,
            root,
        ):
            if expression not in acquired:
                state = self._acquire_expression(expression, state)
                acquired.add(expression)

        with torch.no_grad():
            self.policy.reversible_transition_mode.fill_(True)
            self.policy.reversible_procedure_transition.first_up.weight.normal_(
                mean=0.0,
                std=0.05,
            )
            self.policy.reversible_procedure_transition.second_up.weight.normal_(
                mean=0.0,
                std=0.05,
            )
        full = self.policy.score_task(
            self._task_for_expression(root),
            state,
            transition_only_composition=True,
        )
        child_policies = torch.stack(
            tuple(
                self.policy.score_task(
                    self._task_for_expression(child),
                    state,
                    transition_only_composition=True,
                ).logits
                for child in (first, second)
            ),
            dim=1,
        )
        expected = (
            full.root.executed_branch_weights.unsqueeze(-1) * child_policies
        ).sum(dim=1) * full.root_available.unsqueeze(-1)

        self.assertGreater(float(full.logits.abs().sum().item()), 0.0)
        self.assertTrue(
            torch.allclose(
                full.root.child_candidate_scores,
                child_policies,
                atol=1.0e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                full.binary_policy_logits,
                expected,
                atol=1.0e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.equal(full.logits, full.binary_policy_logits))

        removed = self.policy.score_task(
            self._task_for_expression(root),
            state,
            include_reversible_transition=False,
            transition_only_composition=True,
        )
        for value in (
            removed.logits,
            removed.composition_logits,
            removed.binary_policy_logits,
            removed.root.child_candidate_scores,
        ):
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_fast_procedural_adapter_is_neutral_and_zero_code_is_identity(self) -> None:
        adapter = runner.CodeConditionedLowRankTransition(64, rank=8)
        source = torch.randn(120, 64)
        code = torch.randn(1, 64)
        zero = torch.zeros_like(code)

        self.assertTrue(
            torch.equal(adapter(source, code), torch.zeros_like(source))
        )
        self.assertTrue(
            torch.equal(
                adapter(source, code, reverse=True),
                torch.zeros_like(source),
            )
        )
        with torch.no_grad():
            adapter.forward_up.weight.normal_()
            adapter.reverse_up.weight.normal_()
        for reverse in (False, True):
            self.assertTrue(
                torch.equal(
                    adapter(source, zero, reverse=reverse),
                    torch.zeros_like(source),
                )
            )
            self.assertGreater(
                float(adapter(source, code, reverse=reverse).abs().sum().item()),
                0.0,
            )

    def test_reversible_transition_is_exactly_invertible_and_equivariant(self) -> None:
        transition = runner.ConditionalReversibleTransition(64, 8)
        source = torch.randn(120, 64)
        condition = torch.randn(1, 128)

        self.assertTrue(torch.equal(transition(source, condition), source))
        with torch.no_grad():
            transition.first_up.weight.normal_(mean=0.0, std=0.05)
            transition.second_up.weight.normal_(mean=0.0, std=0.05)

        advanced = transition(source, condition)
        recovered = transition(advanced, condition, reverse=True)
        self.assertTrue(torch.allclose(recovered, source, atol=1.0e-6, rtol=1.0e-6))
        zero_gate = torch.zeros(1, 8)
        self.assertTrue(
            torch.equal(
                transition(
                    source,
                    condition,
                    post_tanh_gate_residual=zero_gate,
                ),
                advanced,
            )
        )
        public_gate = 0.25 * torch.randn(1, 8)
        public_advanced = transition(
            source,
            condition,
            post_tanh_gate_residual=public_gate,
        )
        self.assertFalse(torch.equal(public_advanced, advanced))
        public_recovered = transition(
            public_advanced,
            condition,
            reverse=True,
            post_tanh_gate_residual=public_gate,
        )
        self.assertTrue(
            torch.allclose(public_recovered, source, atol=1.0e-6, rtol=1.0e-6)
        )
        with self.assertRaisesRegex(ValueError, "rank-width"):
            transition(
                source,
                condition,
                post_tanh_gate_residual=torch.zeros(1, 7),
            )
        zero_condition = torch.zeros_like(condition)
        self.assertTrue(torch.equal(transition(source, zero_condition), source))
        self.assertTrue(
            torch.equal(
                transition(source, zero_condition, reverse=True),
                source,
            )
        )
        permutation = torch.randperm(len(source))
        self.assertTrue(
            torch.allclose(
                transition(source[permutation], condition),
                advanced[permutation],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )

    def test_reversible_stage_trains_only_codes_and_one_procedure_map(self) -> None:
        trainable = runner._configure_stage_trainability(
            self.policy,
            "reversible_transition_acquisition",
        )
        expected = tuple(
            name
            for name, _ in self.policy.named_parameters()
            if runner._is_reversible_transition_acquisition_state(name)
        )
        self.assertEqual(trainable, expected)
        self.assertIn("memory.feedback_direction_encoder.3.weight", trainable)
        self.assertIn(
            "composition_memory.feedback_direction_encoder.3.weight",
            trainable,
        )
        self.assertTrue(
            any(
                name.startswith("reversible_procedure_transition.")
                for name in trainable
            )
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in self.policy.parameters()
                if parameter.requires_grad
            ),
            4_112,
        )
        for frozen_prefix in (
            "stable_compiler.",
            "compiler_source_bridge.",
            "compiler_operator_bridge.",
            "compiler_successor_bridge.",
            "procedural_fast_adapter.",
            "procedural_goal_projection.",
            "phase4_direction_mixer.",
            "phase4_reliability_gate.",
        ):
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for name, parameter in self.policy.named_parameters()
                    if name.startswith(frozen_prefix)
                ),
                frozen_prefix,
            )

    def test_procedural_adapter_stage_trains_only_forward_fast_weights(self) -> None:
        trainable = runner._configure_stage_trainability(
            self.policy,
            "procedural_adapter",
        )
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(runner._PROCEDURAL_ADAPTER_TRAINABLE_PREFIXES)
                for name in trainable
            )
        )
        self.assertFalse(self.policy.procedural_fast_adapter.reverse_up.weight.requires_grad)
        self.assertTrue(self.policy.procedural_fast_adapter.forward_up.weight.requires_grad)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.policy.compiler_operator_bridge.parameters()
            )
        )

    def test_pairwise_adapter_feedback_scores_two_attempts_once_without_targets(self) -> None:
        pair = self.partition.tasks[0]
        candidate_pair = (7, 83)
        judge = mock.Mock(side_effect=(0.9, 0.2))
        with (
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("target index leaked into preference"),
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=AssertionError("target utilities leaked into preference"),
            ),
        ):
            scores = runner._scalar_pairwise_scores(
                pair,
                candidate_pair,
                judge,
            )
            logits = torch.zeros(120, requires_grad=True)
            loss, separation = runner._scalar_pairwise_preference_loss(
                logits,
                candidate_pair,
                scores,
            )
            loss.backward()

        self.assertEqual(judge.call_count, 2)
        self.assertEqual(scores, (0.9, 0.2))
        self.assertAlmostEqual(separation, 0.7)
        self.assertLess(float(logits.grad[candidate_pair[0]].item()), 0.0)
        self.assertGreater(float(logits.grad[candidate_pair[1]].item()), 0.0)
        untouched = logits.grad.detach().clone()
        untouched[list(candidate_pair)] = 0.0
        self.assertTrue(torch.equal(untouched, torch.zeros_like(untouched)))

    def test_procedural_adapter_training_changes_only_feedback_executor(self) -> None:
        profile = replace(
            runner._PROFILES["composition"],
            meta_steps=2,
            meta_instances_per_program=16,
        )
        policy = runner.SkillMemoryPolicy(profile)
        before_source = (
            policy.procedural_fast_adapter.source_down.weight.detach().clone()
        )
        before_gate = policy.procedural_fast_adapter.code_gate.weight.detach().clone()
        with (
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("target index leaked into adapter training"),
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=AssertionError("target utilities leaked into adapter training"),
            ),
        ):
            report = runner._train_procedural_adapter(
                policy,
                profile,
                85_111,
            )

        self.assertEqual(report["outer_steps"], 2)
        self.assertEqual(report["fresh_opaque_mappings"], 2)
        self.assertEqual(report["support_presentations_per_mapping"], 40)
        self.assertEqual(report["query_presentations_per_mapping"], 32)
        self.assertEqual(report["total_query_presentations"], 64)
        self.assertEqual(report["cohort_case_counts"]["binary_root"], 16)
        self.assertEqual(
            report["outside_adapter_fingerprint_before"],
            report["outside_adapter_fingerprint_after"],
        )
        self.assertEqual(
            report["reverse_adapter_fingerprint_before"],
            report["reverse_adapter_fingerprint_after"],
        )
        self.assertNotEqual(
            report["adapter_fingerprint_before"],
            report["adapter_fingerprint_after"],
        )
        self.assertFalse(
            torch.equal(
                before_source,
                policy.procedural_fast_adapter.source_down.weight,
            )
        )
        self.assertFalse(
            torch.equal(
                before_gate,
                policy.procedural_fast_adapter.code_gate.weight,
            )
        )
        self.assertTrue(report["target_permutations_used_for_training"] is False)
        self.assertTrue(
            report["candidate_utility_vectors_used_for_training"] is False
        )

    def test_goal_projection_is_neutral_code_gated_and_candidate_equivariant(self) -> None:
        projection = runner.CandidateEquivariantGoalProjection(64, rank=8)
        source = torch.randn(1, 64)
        candidates = torch.randn(120, 64)
        query = torch.randn(1, 8)
        zero_query = torch.zeros_like(query)

        neutral_delta, neutral_energy = projection(source, candidates, query)
        self.assertTrue(torch.equal(neutral_delta, torch.zeros_like(neutral_delta)))
        self.assertTrue(
            torch.equal(neutral_energy, torch.zeros_like(neutral_energy))
        )

        with torch.no_grad():
            projection.candidate_down.weight.normal_(mean=0.0, std=0.2)
        zero_delta, zero_energy = projection(source, candidates, zero_query)
        self.assertTrue(torch.equal(zero_delta, torch.zeros_like(zero_delta)))
        self.assertTrue(torch.equal(zero_energy, torch.zeros_like(zero_energy)))

        ordinary_delta, ordinary_energy = projection(source, candidates, query)
        permutation = torch.randperm(len(candidates))
        permuted_delta, permuted_energy = projection(
            source,
            candidates[permutation],
            query,
        )
        self.assertTrue(
            torch.allclose(permuted_delta, ordinary_delta, atol=1.0e-6, rtol=1.0e-5)
        )
        self.assertTrue(
            torch.allclose(
                permuted_energy,
                ordinary_energy[permutation],
                atol=1.0e-6,
                rtol=1.0e-5,
            )
        )

    def test_reverse_construction_stage_trains_only_joint_procedural_state(self) -> None:
        trainable = runner._configure_stage_trainability(
            self.policy,
            "reverse_construction",
        )
        expected = tuple(
            name
            for name, _ in self.policy.named_parameters()
            if runner._is_reverse_construction_state(name)
        )
        self.assertEqual(trainable, expected)
        self.assertIn(
            "memory.feedback_direction_encoder.3.weight",
            trainable,
        )
        self.assertIn(
            "composition_memory.feedback_direction_encoder.3.weight",
            trainable,
        )
        self.assertTrue(self.policy.procedural_fast_adapter.reverse_up.weight.requires_grad)
        self.assertTrue(
            self.policy.procedural_goal_projection.candidate_down.weight.requires_grad
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.policy.stable_compiler.parameters()
            )
        )

    def test_reverse_harmonization_stage_trains_only_existing_arbitration(self) -> None:
        trainable = runner._configure_stage_trainability(
            self.policy,
            "reverse_harmonization",
        )
        expected = tuple(
            name
            for name, _ in self.policy.named_parameters()
            if name.startswith(runner._REVERSE_HARMONIZATION_TRAINABLE_PREFIXES)
        )
        self.assertEqual(trainable, expected)
        self.assertTrue(
            any(name.startswith("phase4_direction_mixer.") for name in trainable)
        )
        self.assertTrue(
            any(name.startswith("phase4_reliability_gate.") for name in trainable)
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                == name.startswith(runner._REVERSE_HARMONIZATION_TRAINABLE_PREFIXES)
                for name, parameter in self.policy.named_parameters()
            )
        )

    def test_procedural_coadaptation_trains_only_complete_procedural_seam(self) -> None:
        profile = runner._PROFILES["composition"]
        policy = runner.SkillMemoryPolicy(profile)
        trainable = runner._configure_stage_trainability(
            policy,
            "procedural_coadaptation",
        )
        expected = tuple(
            name
            for name, _ in policy.named_parameters()
            if runner._is_reverse_construction_state(name)
            or runner._is_reverse_harmonization_state(name)
        )
        self.assertEqual(trainable, expected)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in policy.parameters()
                if parameter.requires_grad
            ),
            11_749,
        )
        self.assertTrue(
            all(
                parameter.requires_grad == (name in expected)
                for name, parameter in policy.named_parameters()
            )
        )

    def test_on_policy_reward_attempts_include_deployed_choice_and_full_softmax(self) -> None:
        logits = torch.linspace(-1.0, 1.0, len(runner._PERMUTATIONS))
        attempts = runner._on_policy_reward_candidate_set(logits, 3, 7)
        self.assertEqual(attempts[0], int(logits.argmax().item()))
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(set(attempts)), 4)
        self.assertEqual(
            attempts,
            runner._on_policy_reward_candidate_set(logits, 3, 7),
        )

        learnable = torch.zeros(len(runner._PERMUTATIONS), requires_grad=True)
        loss = runner._scalar_on_policy_reward_loss(
            learnable,
            attempts,
            (0.9, 0.6, 0.3, 0.1),
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(learnable.grad).all().item()))
        self.assertLess(float(learnable.grad[attempts[0]].item()), 0.0)
        self.assertGreater(float(learnable.grad[attempts[-1]].item()), 0.0)
        unattempted = learnable.grad.detach().clone()
        unattempted[list(attempts)] = 0.0
        self.assertTrue(torch.allclose(unattempted, torch.zeros_like(unattempted)))

        shifted = torch.zeros_like(learnable, requires_grad=True)
        shifted_loss = runner._scalar_on_policy_reward_loss(
            shifted,
            attempts,
            (0.95, 0.65, 0.35, 0.15),
        )
        shifted_loss.backward()
        self.assertTrue(
            torch.allclose(shifted.grad, learnable.grad, atol=1.0e-7, rtol=1.0e-6)
        )

    def test_reverse_harmonization_uses_only_deployed_scalar_preferences(self) -> None:
        torch.manual_seed(85_101)
        profile = replace(
            runner._PROFILES["composition"],
            meta_steps=1,
            meta_instances_per_program=16,
        )
        policy = runner.SkillMemoryPolicy(profile)
        outside_before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
            if not runner._is_reverse_harmonization_state(name)
        }
        deployed_logits: list[torch.Tensor] = []
        loss_inputs: list[torch.Tensor] = []
        real_score_task = policy.score_task
        real_preference_loss = runner._scalar_multi_preference_loss

        def score_spy(*args: object, **kwargs: object) -> runner.PolicyScores:
            scores = real_score_task(*args, **kwargs)
            if torch.is_grad_enabled():
                deployed_logits.append(scores.logits)
            return scores

        def loss_spy(
            logits: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> tuple[torch.Tensor, int]:
            self.assertIs(logits, deployed_logits[len(loss_inputs)])
            loss_inputs.append(logits)
            return real_preference_loss(logits, *args, **kwargs)

        with (
            mock.patch.object(policy, "score_task", side_effect=score_spy),
            mock.patch.object(
                runner,
                "_scalar_multi_preference_loss",
                side_effect=loss_spy,
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("target index leaked into harmonization"),
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=AssertionError("target utilities leaked into harmonization"),
            ),
            mock.patch.object(
                runner,
                "_root_reverse_construction_energies",
                side_effect=AssertionError("proxy energy leaked into harmonization"),
            ),
            mock.patch.object(
                runner,
                "_same_arity_root_codes",
                side_effect=AssertionError("specificity control leaked into harmonization"),
            ),
        ):
            report = runner._train_reverse_harmonization(
                policy,
                profile,
                85_119,
            )

        self.assertEqual(report["training_stage"], "reverse_harmonization")
        self.assertEqual(report["outer_steps"], 1)
        self.assertEqual(report["total_support_presentations"], 40)
        self.assertEqual(report["total_query_presentations"], 32)
        self.assertEqual(report["total_scored_query_attempts"], 128)
        self.assertEqual(len(deployed_logits), 32)
        self.assertEqual(len(loss_inputs), 32)
        self.assertGreater(report["total_observed_preference_edges"], 0)
        self.assertEqual(
            report["cohort_case_counts"],
            {
                "unary_depth2": 16,
                "unary_depth3": 4,
                "unary_direct_binary_child": 4,
                "binary_root": 8,
            },
        )
        self.assertEqual(
            report["outside_harmonizer_fingerprint_before"],
            report["outside_harmonizer_fingerprint_after"],
        )
        self.assertNotEqual(
            report["direction_mixer_fingerprint_before"],
            report["direction_mixer_fingerprint_after"],
        )
        self.assertNotEqual(
            report["reliability_gate_fingerprint_before"],
            report["reliability_gate_fingerprint_after"],
        )
        self.assertTrue(
            report["deployed_preference_gradient_reached_direction_mixer"]
        )
        self.assertTrue(
            report["deployed_preference_gradient_reached_reliability_gate"]
        )
        self.assertTrue(
            report["auxiliary_ranking_objectives_used_for_training"] is False
        )
        for name, expected in outside_before.items():
            self.assertTrue(torch.equal(policy.state_dict()[name], expected), name)

        lineage_report = copy.deepcopy(report)
        lineage_steps = runner._REVERSE_HARMONIZATION_OUTER_STEPS
        lineage_report.update(
            {
                "outer_steps": lineage_steps,
                "fresh_opaque_mappings": lineage_steps,
                "optimizer_steps": lineage_steps,
                "total_support_presentations": 40 * lineage_steps,
                "total_query_presentations": 32 * lineage_steps,
                "total_scored_query_attempts": 128 * lineage_steps,
                "cohort_case_counts": {
                    "unary_depth2": 16 * lineage_steps,
                    "unary_depth3": 4 * lineage_steps,
                    "unary_direct_binary_child": 4 * lineage_steps,
                    "binary_root": 8 * lineage_steps,
                },
            }
        )
        initialization = {
            "source_runner": runner._REPORT_VERSION,
            "source_stage": "reverse_harmonization",
            "fresh_parameter_keys": [],
            "source_initialization": {
                "sha256": runner._REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256,
                "source_runner": "angler.phase5-skill-memory-stream.v19",
                "source_stage": "reverse_construction",
                "fresh_parameter_keys": [],
            },
            "source_training": lineage_report,
        }
        self.assertEqual(
            runner._validate_operator_audit_checkpoint_lineage(
                policy,
                initialization,
            ),
            "reverse_harmonization",
        )
        tampered = copy.deepcopy(initialization)
        tampered["source_training"][
            "outside_harmonizer_fingerprint_after"
        ] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact v47 lineage"):
            runner._validate_operator_audit_checkpoint_lineage(policy, tampered)

    def test_procedural_coadaptation_uses_only_on_policy_scalar_reward(self) -> None:
        torch.manual_seed(85_101)
        profile = replace(
            runner._PROFILES["composition"],
            meta_steps=1,
            meta_instances_per_program=16,
        )
        policy = runner.SkillMemoryPolicy(profile)
        outside_before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
            if not runner._is_procedural_coadaptation_state(name)
        }
        real_score_task = policy.score_task
        real_reward_loss = runner._scalar_on_policy_reward_loss
        latest_logits: list[torch.Tensor | None] = [None]
        objective_calls: list[tuple[int, ...]] = []

        def score_spy(*args: object, **kwargs: object) -> runner.PolicyScores:
            scores = real_score_task(*args, **kwargs)
            latest_logits[0] = scores.logits
            return scores

        def reward_loss_spy(
            logits: torch.Tensor,
            candidate_indices: object,
            scalar_scores: object,
        ) -> torch.Tensor:
            self.assertIs(logits, latest_logits[0])
            indices = tuple(candidate_indices)
            self.assertEqual(indices[0], int(logits.detach().argmax().item()))
            self.assertEqual(len(indices), 4)
            self.assertEqual(len(tuple(scalar_scores)), 4)
            objective_calls.append(indices)
            return real_reward_loss(logits, indices, scalar_scores)

        forbidden = AssertionError("proxy, target, or pairwise surrogate was used")
        with (
            mock.patch.object(policy, "score_task", side_effect=score_spy),
            mock.patch.object(
                runner,
                "_scalar_on_policy_reward_loss",
                side_effect=reward_loss_spy,
            ),
            mock.patch.object(
                runner,
                "_scalar_multi_preference_loss",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_root_reverse_construction_energies",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_same_arity_root_codes",
                side_effect=forbidden,
            ),
        ):
            report = runner._train_procedural_coadaptation(
                policy,
                profile,
                85_121,
            )

        self.assertEqual(len(objective_calls), 32)
        self.assertEqual(report["training_stage"], "procedural_coadaptation")
        self.assertEqual(report["total_support_presentations"], 40)
        self.assertEqual(report["total_query_presentations"], 32)
        self.assertEqual(report["total_scored_query_attempts"], 128)
        self.assertEqual(report["training_objective"], "on_policy_reward")
        self.assertTrue(report["current_deployed_greedy_attempted_per_query"])
        self.assertTrue(report["complete_action_softmax_used_for_training"])
        self.assertFalse(report["support_graph_detached"])
        self.assertEqual(
            report["cohort_case_counts"],
            {
                "unary_depth2": 16,
                "unary_depth3": 4,
                "unary_direct_binary_child": 4,
                "binary_root": 8,
            },
        )
        self.assertEqual(
            report["outside_harmonizer_fingerprint_before"],
            report["outside_harmonizer_fingerprint_after"],
        )
        self.assertEqual(
            report["deployed_preference_gradient_reached_groups"],
            {
                "leaf_code_acquisition": True,
                "composition_code_acquisition": True,
                "fast_adapter": True,
                "goal_projection": True,
                "direction_mixer": True,
                "reliability_gate": True,
            },
        )
        for name, expected in outside_before.items():
            self.assertTrue(torch.equal(policy.state_dict()[name], expected), name)

        lineage_report = copy.deepcopy(report)
        lineage_steps = runner._PROCEDURAL_COADAPTATION_OUTER_STEPS
        lineage_report.update(
            {
                "outer_steps": lineage_steps,
                "fresh_opaque_mappings": lineage_steps,
                "optimizer_steps": lineage_steps,
                "total_support_presentations": 40 * lineage_steps,
                "total_query_presentations": 32 * lineage_steps,
                "total_scored_query_attempts": 128 * lineage_steps,
                "cohort_case_counts": {
                    "unary_depth2": 16 * lineage_steps,
                    "unary_depth3": 4 * lineage_steps,
                    "unary_direct_binary_child": 4 * lineage_steps,
                    "binary_root": 8 * lineage_steps,
                },
            }
        )
        initialization = {
            "source_runner": runner._REPORT_VERSION,
            "source_stage": "procedural_coadaptation",
            "fresh_parameter_keys": [],
            "source_initialization": {
                "sha256": runner._PROCEDURAL_COADAPTATION_SOURCE_CHECKPOINT_SHA256,
                "source_runner": "angler.phase5-skill-memory-stream.v20",
                "source_stage": "reverse_harmonization",
                "fresh_parameter_keys": [],
                "source_initialization": {
                    "sha256": runner._REVERSE_HARMONIZATION_SOURCE_CHECKPOINT_SHA256,
                    "source_runner": "angler.phase5-skill-memory-stream.v19",
                    "source_stage": "reverse_construction",
                    "fresh_parameter_keys": [],
                },
                "source_training": {
                    "training_stage": "reverse_harmonization",
                    "outer_steps": runner._REVERSE_HARMONIZATION_OUTER_STEPS,
                    "support_graph_detached": True,
                    "auxiliary_ranking_objectives_used_for_training": False,
                },
            },
            "source_training": lineage_report,
        }
        self.assertEqual(
            runner._validate_operator_audit_checkpoint_lineage(
                policy,
                initialization,
            ),
            "procedural_coadaptation",
        )
        tampered = copy.deepcopy(initialization)
        tampered["source_training"]["training_objective"] = "pairwise_preference"
        with self.assertRaisesRegex(RuntimeError, "exact v48 lineage"):
            runner._validate_operator_audit_checkpoint_lineage(policy, tampered)

    def test_reversible_acquisition_gets_end_to_end_scalar_feedback(self) -> None:
        torch.manual_seed(85_101)
        profile = replace(
            runner._PROFILES["composition"],
            meta_steps=2,
            meta_instances_per_program=16,
        )
        policy = runner.SkillMemoryPolicy(profile)
        forbidden = AssertionError("legacy gate, proxy, target, or surrogate was used")
        with (
            mock.patch.object(
                policy.procedural_fast_adapter,
                "forward",
                side_effect=forbidden,
            ),
            mock.patch.object(
                policy.procedural_goal_projection,
                "forward",
                side_effect=forbidden,
            ),
            mock.patch.object(
                policy.phase4_direction_mixer,
                "forward",
                side_effect=forbidden,
            ),
            mock.patch.object(
                policy.phase4_reliability_gate,
                "forward",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_scalar_multi_preference_loss",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_root_reverse_construction_energies",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_same_arity_root_codes",
                side_effect=forbidden,
            ),
            mock.patch.object(
                runner,
                "_soft_reanchor_intermediate",
                side_effect=forbidden,
            ),
        ):
            report = runner._train_reversible_transition_acquisition(
                policy,
                profile,
                85_123,
            )

        self.assertTrue(bool(policy.reversible_transition_mode.item()))
        self.assertEqual(
            report["deployed_preference_gradient_reached_groups"],
            {
                "leaf_code_acquisition": True,
                "composition_code_acquisition": True,
                "reversible_transition": True,
            },
        )
        self.assertEqual(report["training_stage"], "reversible_transition_acquisition")
        self.assertEqual(report["outer_steps"], 2)
        self.assertEqual(report["fresh_opaque_mappings"], 2)
        self.assertEqual(report["total_support_presentations"], 80)
        self.assertEqual(report["total_query_presentations"], 64)
        self.assertEqual(report["total_scored_query_attempts"], 256)
        self.assertEqual(report["training_objective"], "on_policy_reward")
        self.assertFalse(report["support_graph_detached"])
        self.assertEqual(
            report["outside_harmonizer_fingerprint_before"],
            report["outside_harmonizer_fingerprint_after"],
        )
        self.assertNotEqual(
            report["harmonizer_fingerprint_before"],
            report["harmonizer_fingerprint_after"],
        )

    def test_reverse_attempts_use_four_scalar_judgments_and_all_preferences(self) -> None:
        pair = self.partition.tasks[0]
        candidates = runner._reverse_construction_candidate_set(3, 7)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(set(candidates)), 4)
        self.assertEqual(
            candidates,
            runner._reverse_construction_candidate_set(3, 7),
        )
        judge = mock.Mock(side_effect=(1.0, 0.75, 0.25, 0.0))
        with (
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("target index leaked into reverse learning"),
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=AssertionError("target utilities leaked into reverse learning"),
            ),
        ):
            scalar_scores = runner._scalar_attempt_scores(
                pair,
                candidates,
                judge,
            )
            logits = torch.zeros(120, requires_grad=True)
            loss, edge_count = runner._scalar_multi_preference_loss(
                logits,
                candidates,
                scalar_scores,
            )
            loss.backward()

        self.assertEqual(judge.call_count, 4)
        self.assertEqual(edge_count, 6)
        self.assertTrue(bool(torch.isfinite(logits.grad).all().item()))
        untouched = logits.grad.detach().clone()
        untouched[list(candidates)] = 0.0
        self.assertTrue(torch.equal(untouched, torch.zeros_like(untouched)))

    def test_neutral_candidate_energy_has_finite_preference_gradient(self) -> None:
        energy = torch.zeros(120, requires_grad=True)
        standardized = runner._standardize_candidate_energy(energy)
        loss, edges = runner._scalar_multi_preference_loss(
            standardized,
            (3, 17, 61, 109),
            (1.0, 0.7, 0.3, 0.0),
        )
        loss.backward()
        self.assertEqual(edges, 6)
        self.assertTrue(bool(torch.isfinite(energy.grad).all().item()))
        self.assertGreater(float(energy.grad.abs().sum().item()), 0.0)

    def test_reverse_construction_training_completes_joint_meta_gradient(self) -> None:
        torch.manual_seed(85_101)
        profile = replace(
            runner._PROFILES["composition"],
            meta_steps=1,
            meta_instances_per_program=16,
        )
        policy = runner.SkillMemoryPolicy(profile)
        with (
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("target index leaked into reverse training"),
            ),
            mock.patch.object(
                runner,
                "_outer_target_candidate_utilities",
                side_effect=AssertionError("target utilities leaked into reverse training"),
            ),
        ):
            report = runner._train_reverse_construction(
                policy,
                profile,
                85_117,
            )

        self.assertEqual(report["training_stage"], "reverse_construction")
        self.assertEqual(report["outer_steps"], 1)
        self.assertEqual(report["total_support_presentations"], 40)
        self.assertEqual(report["total_query_presentations"], 32)
        self.assertEqual(report["total_scored_query_attempts"], 128)
        self.assertEqual(
            report["outside_learned_fingerprint_before"],
            report["outside_learned_fingerprint_after"],
        )
        for prefix in (
            "learned_state",
            "code_acquisition",
            "fast_adapter",
            "goal_projection",
        ):
            self.assertNotEqual(
                report[f"{prefix}_fingerprint_before"],
                report[f"{prefix}_fingerprint_after"],
            )
        self.assertTrue(report["support_graph_detached"] is False)
        self.assertTrue(report["target_permutations_used_for_training"] is False)
        self.assertTrue(
            report["candidate_utility_vectors_used_for_training"] is False
        )
        expected_fresh = sorted(
            name
            for name in policy.state_dict()
            if name.startswith(
                (
                    "phase4_direction_mixer.",
                    "procedural_fast_adapter.",
                    "procedural_goal_projection.",
                    "reversible_procedure_transition.",
                    "reversible_transition_mode",
                )
            )
        )
        initialization = {
            "source_runner": runner._REPORT_VERSION,
            "source_stage": "reverse_construction",
            "fresh_parameter_keys": [],
            "source_initialization": {
                "sha256": runner._PROCEDURAL_ADAPTER_SOURCE_CHECKPOINT_SHA256,
                "source_runner": "angler.phase5-skill-memory-stream.v13",
                "source_stage": "relational_acquisition",
                "fresh_parameter_keys": expected_fresh,
            },
            "source_training": report,
        }
        self.assertEqual(
            runner._validate_operator_audit_checkpoint_lineage(
                policy,
                initialization,
            ),
            "reverse_construction",
        )
        tampered = copy.deepcopy(initialization)
        tampered["source_training"]["outside_learned_fingerprint_after"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "exact v41 lineage"):
            runner._validate_operator_audit_checkpoint_lineage(policy, tampered)

    def test_empty_memory_is_exactly_uniform_and_read_only(self) -> None:
        state = self.policy.initial_state(1)
        before = procedural_skill_state_digest(state)
        scores = self.policy.score_task(self.partition.tasks[0].learner, state)

        self.assertEqual(scores.logits.shape, (1, 120))
        self.assertTrue(torch.equal(scores.logits, torch.zeros_like(scores.logits)))
        self.assertTrue(
            torch.equal(
                torch.softmax(scores.logits, dim=-1),
                torch.full_like(scores.logits, 1.0 / 120.0),
            )
        )
        self.assertTrue(
            torch.equal(scores.root_context, torch.zeros_like(scores.root_context))
        )
        self.assertEqual(procedural_skill_state_digest(state), before)

    def test_public_proposal_cannot_call_judge_or_outer_target(self) -> None:
        pair = self.partition.tasks[0]
        state = self.policy.initial_state(1)
        with (
            mock.patch.object(
                runner,
                "_outer_target_candidate_index",
                side_effect=AssertionError("outer target leaked into proposal"),
            ),
            mock.patch.object(
                suite,
                "score_skill_memory_answer",
                side_effect=AssertionError("judge leaked into proposal"),
            ),
        ):
            proposal = runner.propose_task(self.policy, pair.learner, state)

        self.assertEqual(len(proposal.answer), 5)
        self.assertEqual(set(proposal.answer), {item.symbol for item in pair.learner.items})
        self.assertEqual(
            {field.name for field in fields(runner.TaskProposal)},
            {
                "answer",
                "candidate_index",
                "scores",
                "behavior_probabilities",
                "competence_digest",
                "public_task_digest",
            },
        )
        forbidden = {
            "hidden",
            "target",
            "solution",
            "program",
            "mechanism",
            "domain_id",
            "task_id",
            "family_id",
            "partition",
            "namespace",
        }
        for function in (
            runner.propose_task,
            runner.propose_differentiable_feedback,
            runner.apply_transactional_feedback,
        ):
            self.assertFalse(
                set(inspect.signature(function).parameters) & forbidden,
                function.__name__,
            )

    def test_scalar_feedback_candidate_carries_later_query_meta_gradient(self) -> None:
        groups = runner._group_evaluator_pairs(self.partition.tasks)
        support, query = groups[0][:2]
        state = self.policy.initial_state(1)
        proposal = runner._proposal_for_candidate(
            self.policy,
            support.learner,
            state,
            runner._outer_target_candidate_index(support),
            include_compiler=False,
        )
        reward = suite.score_skill_memory_answer(
            support.learner, support.hidden, proposal.answer
        )
        staged = runner.propose_differentiable_feedback(
            self.policy, proposal, reward, state
        )
        loss = runner._outer_query_loss(self.policy, staged.candidate_state, query)
        gradients = torch.autograd.grad(
            loss,
            (
                self.policy.memory.feedback_encoder[-1].weight,
                self.policy.memory.feedback_direction_encoder[-1].weight,
                self.policy.memory.utility_decoder[-1].weight,
                self.policy.item_encoder[0].weight,
            ),
        )

        self.assertNotEqual(
            procedural_skill_state_digest(staged.candidate_state),
            procedural_skill_state_digest(state),
        )
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_scalar_feedback_is_calibrated_candidate_utility(self) -> None:
        neutral = torch.zeros((1, 120), requires_grad=True)
        neutral_loss = runner._scalar_feedback_tensor(neutral, 17, 0.5)
        neutral_gradient = torch.autograd.grad(neutral_loss, neutral)[0]
        self.assertEqual(float(neutral_gradient.abs().sum().item()), 0.0)

        gradients = []
        for reward in (1.0, 0.0):
            logits = torch.zeros((1, 120), requires_grad=True)
            loss = runner._scalar_feedback_tensor(logits, 17, reward)
            gradient = torch.autograd.grad(loss, logits)[0]
            self.assertEqual(
                int(torch.count_nonzero(gradient).item()),
                1,
            )
            gradients.append(float(gradient[0, 17].item()))
        self.assertLess(gradients[0], 0.0)
        self.assertGreater(gradients[1], 0.0)

        short = torch.zeros((1, 2))
        long = torch.zeros((1, 120))
        self.assertEqual(
            float(runner._scalar_feedback_tensor(short, 0, 0.8).item()),
            float(runner._scalar_feedback_tensor(long, 0, 0.8).item()),
        )

    def test_outer_utility_vector_matches_pairwise_metric(self) -> None:
        pair = self.partition.tasks[0]
        reference = torch.zeros((1, 120))
        utilities = runner._outer_target_candidate_utilities(pair, reference)
        target_index = runner._outer_target_candidate_index(pair)
        reverse_index = runner._PERMUTATION_TO_INDEX[
            tuple(reversed(runner._PERMUTATIONS[target_index]))
        ]

        self.assertEqual(utilities.shape, (1, 120))
        self.assertEqual(float(utilities[0, target_index].item()), 1.0)
        self.assertEqual(float(utilities[0, reverse_index].item()), 0.0)
        self.assertAlmostEqual(float(utilities.mean().item()), 0.5, places=6)

        counterfactual = 1.0 - utilities
        self.assertEqual(float(counterfactual[0, target_index].item()), 0.0)
        self.assertEqual(float(counterfactual[0, reverse_index].item()), 1.0)
        ordinary_logits = torch.zeros_like(reference)
        ordinary_logits[0, target_index] = 4.0
        ordinary_logits[0, reverse_index] = -4.0
        reversed_logits = -ordinary_logits
        self.assertLess(
            float(
                runner._outer_utility_loss(
                    ordinary_logits, utilities, target_index
                ).item()
            ),
            float(
                runner._outer_utility_loss(
                    reversed_logits, utilities, target_index
                ).item()
            ),
        )
        self.assertLess(
            float(
                runner._outer_utility_loss(
                    reversed_logits, counterfactual, reverse_index
                ).item()
            ),
            float(
                runner._outer_utility_loss(
                    ordinary_logits, counterfactual, reverse_index
                ).item()
            ),
        )

    def test_oracle_latent_gate_bypasses_only_acquisition_and_is_non_persistent(self) -> None:
        pair = next(
            pair for pair in self.partition.tasks if pair.hidden.program.depth == 0
        )
        self.policy.requires_grad_(False)
        before = {
            name: value.detach().clone()
            for name, value in self.policy.state_dict().items()
        }
        raw_code = torch.randn(self.profile.width, requires_grad=True)
        dense_state = runner._oracle_latent_state(
            self.policy, pair.learner, raw_code, keyed=False
        )
        keyed_state = runner._oracle_latent_state(
            self.policy, pair.learner, raw_code, keyed=True
        )
        dense = self.policy.score_task(
            pair.learner, dense_state, include_compiler=False
        )
        keyed = self.policy.score_task(
            pair.learner, keyed_state, include_compiler=False
        )

        self.assertTrue(
            torch.allclose(
                dense.root.memory_read.plastic_context,
                keyed.root.memory_read.plastic_context,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(dense.logits, keyed.logits, atol=1e-6, rtol=1e-6)
        )
        (gradient,) = torch.autograd.grad(keyed.logits.square().mean(), (raw_code,))
        self.assertTrue(bool(torch.isfinite(gradient).all().item()))
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        for name, value in self.policy.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_compatible_checkpoint_restores_slow_model_but_no_online_state(self) -> None:
        source_digest = runner.reasoning_state_digest(self.policy)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "leaf.pt"
            torch.save(
                {
                    "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                    "model": self.policy.state_dict(),
                    "profile": asdict(self.profile),
                    "result_digest": "sha256:" + "a" * 64,
                    "runner": runner._REPORT_VERSION,
                    "seed": 1,
                    "stage": "leaf_core",
                },
                checkpoint,
            )
            target = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                target,
                checkpoint,
                self.profile,
            )

        self.assertEqual(runner.reasoning_state_digest(target), source_digest)
        self.assertTrue(record["slow_model_state_restored"])
        self.assertFalse(record["online_state_restored"])
        self.assertEqual(record["source_stage"], "leaf_core")
        self.assertEqual(record["fresh_parameter_keys"], [])
        self.assertEqual(record["fresh_cloned_prefixes"], [])

    def test_pre_bridge_checkpoint_migrates_only_declared_adapters(self) -> None:
        old_state = {
            key: value
            for key, value in self.policy.state_dict().items()
            if not key.startswith(runner._PERMITTED_CHECKPOINT_MIGRATION_PREFIXES)
            and key != runner._CONDITION_AXIS_KEY
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pre-bridge.pt"
            torch.save(
                {
                    "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                    "model": old_state,
                    "profile": asdict(self.profile),
                    "result_digest": "sha256:" + "b" * 64,
                    "runner": "angler.phase5-skill-memory-stream.v8",
                    "stage": "leaf_core",
                },
                checkpoint,
            )
            target = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                target,
                checkpoint,
                self.profile,
            )

        expected = sorted(
            key
            for key in target.state_dict()
            if key.startswith(runner._PERMITTED_CHECKPOINT_MIGRATION_PREFIXES)
            or key == runner._CONDITION_AXIS_KEY
        )
        self.assertEqual(record["fresh_parameter_keys"], expected)
        self.assertTrue(expected)
        for name, value in target.memory.state_dict().items():
            self.assertTrue(
                torch.equal(value, target.composition_memory.state_dict()[name]),
                name,
            )
        for composition_module, leaf_module in (
            (target.composition_item_encoder, target.item_encoder),
            (target.composition_candidate_encoder, target.candidate_encoder),
            (target.composition_state_encoder, target.state_encoder),
            (target.composition_goal_encoder, target.goal_encoder),
        ):
            for name, value in leaf_module.state_dict().items():
                self.assertTrue(
                    torch.equal(value, composition_module.state_dict()[name]),
                    name,
                )

    def test_reliability_gate_checkpoint_migration_is_atomic(self) -> None:
        full_state = self.policy.state_dict()
        reversible_keys = sorted(
            key for key in full_state if _is_reversible_state_key(key)
        )
        adapter_keys = sorted(
            key for key in full_state if key.startswith("procedural_fast_adapter.")
        )
        goal_keys = sorted(
            key for key in full_state if key.startswith("procedural_goal_projection.")
        )
        gate_keys = sorted(
            key
            for key in full_state
            if key.startswith("phase4_reliability_gate.")
        )
        self.assertTrue(gate_keys)

        def payload(
            model,
            *,
            runner_identity="angler.phase5-skill-memory-stream.v14",
        ):
            return {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": model,
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "c" * 64,
                "runner": runner_identity,
                "stage": "integrated",
            }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            gate_missing = directory_path / "gate-missing.pt"
            torch.save(
                payload(
                    {
                        key: value
                        for key, value in full_state.items()
                        if key not in gate_keys
                        and key not in adapter_keys
                        and key not in goal_keys
                        and key not in reversible_keys
                    }
                ),
                gate_missing,
            )
            target = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                target,
                gate_missing,
                self.profile,
            )
            self.assertEqual(
                record["fresh_parameter_keys"],
                sorted(gate_keys + adapter_keys + goal_keys + reversible_keys),
            )

            partial = directory_path / "gate-partial.pt"
            torch.save(
                payload(
                    {
                        key: value
                        for key, value in full_state.items()
                        if key != gate_keys[0]
                        and key not in adapter_keys
                        and key not in goal_keys
                        and key not in reversible_keys
                    }
                ),
                partial,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    partial,
                    self.profile,
                )

    def test_direction_mixer_checkpoint_migration_is_atomic_and_neutral(self) -> None:
        full_state = self.policy.state_dict()
        reversible_keys = sorted(
            key for key in full_state if _is_reversible_state_key(key)
        )
        adapter_keys = sorted(
            key for key in full_state if key.startswith("procedural_fast_adapter.")
        )
        goal_keys = sorted(
            key for key in full_state if key.startswith("procedural_goal_projection.")
        )
        mixer_keys = sorted(
            key for key in full_state if key.startswith("phase4_direction_mixer.")
        )
        self.assertTrue(mixer_keys)

        def payload(model):
            return {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": model,
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "e" * 64,
                "runner": "angler.phase5-skill-memory-stream.v14",
                "stage": "relational_acquisition",
            }

        legacy_state = {
            key: value.detach().clone()
            for key, value in full_state.items()
            if key not in mixer_keys
            and key not in adapter_keys
            and key not in goal_keys
            and key not in reversible_keys
        }
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            missing = directory_path / "v14-missing-mixer.pt"
            torch.save(payload(legacy_state), missing)
            target = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                target,
                missing,
                self.profile,
            )
            self.assertEqual(
                record["fresh_parameter_keys"],
                sorted(mixer_keys + adapter_keys + goal_keys + reversible_keys),
            )
            for key, value in legacy_state.items():
                self.assertTrue(torch.equal(target.state_dict()[key], value), key)
            self.assertTrue(
                torch.equal(
                    target.phase4_direction_mixer[-1].weight,
                    torch.zeros_like(target.phase4_direction_mixer[-1].weight),
                )
            )
            self.assertTrue(
                torch.equal(
                    target.phase4_direction_mixer[-1].bias,
                    torch.zeros_like(target.phase4_direction_mixer[-1].bias),
                )
            )

            partial = directory_path / "v14-partial-mixer.pt"
            partial_state = {
                key: value
                for key, value in full_state.items()
                if key not in adapter_keys
                and key not in goal_keys
                and key not in reversible_keys
            }
            partial_state.pop(mixer_keys[0])
            torch.save(payload(partial_state), partial)
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    partial,
                    self.profile,
                )

            current_missing = directory_path / "v19-missing-mixer.pt"
            current_without_mixer = {
                key: value
                for key, value in full_state.items()
                if key not in mixer_keys
            }
            torch.save(
                {
                    **payload(current_without_mixer),
                    "runner": runner._REPORT_VERSION,
                },
                current_missing,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    current_missing,
                    self.profile,
                )

            introduced_missing = directory_path / "v15-missing-mixer.pt"
            torch.save(
                {
                    **payload(legacy_state),
                    "runner": "angler.phase5-skill-memory-stream.v15",
                },
                introduced_missing,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    introduced_missing,
                    self.profile,
                )

    def test_fast_adapter_checkpoint_migration_is_atomic_and_versioned(self) -> None:
        full_state = self.policy.state_dict()
        reversible_keys = sorted(
            key for key in full_state if _is_reversible_state_key(key)
        )
        adapter_keys = sorted(
            key for key in full_state if key.startswith("procedural_fast_adapter.")
        )
        goal_keys = sorted(
            key for key in full_state if key.startswith("procedural_goal_projection.")
        )
        self.assertTrue(adapter_keys)

        def payload(model, runner_identity: str):
            return {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": model,
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "9" * 64,
                "runner": runner_identity,
                "stage": "relational_acquisition",
            }

        without_adapter = {
            key: value.detach().clone()
            for key, value in full_state.items()
            if key not in adapter_keys
            and key not in goal_keys
            and key not in reversible_keys
        }
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            legacy = directory_path / "v16-without-adapter.pt"
            torch.save(
                payload(
                    without_adapter,
                    "angler.phase5-skill-memory-stream.v16",
                ),
                legacy,
            )
            target = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                target,
                legacy,
                self.profile,
            )
            self.assertEqual(
                record["fresh_parameter_keys"],
                sorted(adapter_keys + goal_keys + reversible_keys),
            )
            self.assertTrue(
                torch.equal(
                    target.procedural_fast_adapter.forward_up.weight,
                    torch.zeros_like(
                        target.procedural_fast_adapter.forward_up.weight
                    ),
                )
            )
            self.assertTrue(
                torch.equal(
                    target.procedural_fast_adapter.reverse_up.weight,
                    torch.zeros_like(
                        target.procedural_fast_adapter.reverse_up.weight
                    ),
                )
            )

            partial = directory_path / "v16-partial-adapter.pt"
            partial_state = dict(without_adapter)
            partial_state[adapter_keys[0]] = full_state[adapter_keys[0]]
            torch.save(
                payload(
                    partial_state,
                    "angler.phase5-skill-memory-stream.v16",
                ),
                partial,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared procedural"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    partial,
                    self.profile,
                )

            current = directory_path / "v19-without-adapter.pt"
            current_without_adapter = {
                key: value
                for key, value in full_state.items()
                if key not in adapter_keys
            }
            torch.save(
                payload(current_without_adapter, runner._REPORT_VERSION),
                current,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    current,
                    self.profile,
                )

    def test_goal_projection_checkpoint_migration_is_atomic_and_versioned(self) -> None:
        full_state = self.policy.state_dict()
        reversible_keys = sorted(
            key for key in full_state if _is_reversible_state_key(key)
        )
        goal_keys = sorted(
            key for key in full_state if key.startswith("procedural_goal_projection.")
        )
        self.assertTrue(goal_keys)
        without_goal = {
            key: value.detach().clone()
            for key, value in full_state.items()
            if key not in goal_keys and key not in reversible_keys
        }

        def payload(model, runner_identity: str):
            return {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": model,
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "8" * 64,
                "runner": runner_identity,
                "stage": "procedural_adapter",
            }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            legacy = directory_path / "v18-without-goal.pt"
            torch.save(
                payload(without_goal, "angler.phase5-skill-memory-stream.v18"),
                legacy,
            )
            migrated = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                migrated,
                legacy,
                self.profile,
            )
            self.assertEqual(
                record["fresh_parameter_keys"],
                sorted(goal_keys + reversible_keys),
            )
            for key, value in without_goal.items():
                self.assertTrue(torch.equal(migrated.state_dict()[key], value), key)
            self.assertTrue(
                torch.equal(
                    migrated.procedural_goal_projection.candidate_down.weight,
                    torch.zeros_like(
                        migrated.procedural_goal_projection.candidate_down.weight
                    ),
                )
            )

            contaminated = directory_path / "v18-partial-goal.pt"
            partial_goal = dict(without_goal)
            partial_goal[goal_keys[0]] = full_state[goal_keys[0]]
            torch.save(
                payload(partial_goal, "angler.phase5-skill-memory-stream.v18"),
                contaminated,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "undeclared procedural goal projection",
            ):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    contaminated,
                    self.profile,
                )

            current = directory_path / "v19-without-goal.pt"
            torch.save(payload(without_goal, runner._REPORT_VERSION), current)
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    current,
                    self.profile,
                )

    def test_direction_mixer_is_neutral_bounded_and_candidate_equivariant(self) -> None:
        base = torch.randn(1, 120)
        forward = torch.randn(120)
        reverse = torch.randn(120)
        score_limit = self.policy.memory.score_limit
        bridge, forward_evidence, reverse_evidence, gains = (
            self.policy._phase4_directional_evidence(
                base,
                forward,
                reverse,
                score_limit,
            )
        )
        legacy_combined = forward + reverse
        legacy_bridge = score_limit * torch.tanh(
            legacy_combined - legacy_combined.mean()
        ).unsqueeze(0)
        legacy_forward = score_limit * torch.tanh(
            forward - forward.mean()
        ).unsqueeze(0)
        legacy_reverse = score_limit * torch.tanh(
            reverse - reverse.mean()
        ).unsqueeze(0)
        self.assertTrue(torch.equal(gains, torch.ones_like(gains)))
        self.assertTrue(torch.equal(bridge, legacy_bridge))
        self.assertTrue(torch.equal(forward_evidence, legacy_forward))
        self.assertTrue(torch.equal(reverse_evidence, legacy_reverse))

        with torch.no_grad():
            for parameter in self.policy.phase4_direction_mixer.parameters():
                parameter.normal_(mean=0.0, std=0.2)
        ordinary = self.policy._phase4_directional_evidence(
            base,
            forward,
            reverse,
            score_limit,
        )
        self.assertFalse(torch.allclose(ordinary[0], legacy_bridge))
        self.assertTrue(torch.equal(ordinary[1], legacy_forward))
        self.assertTrue(torch.equal(ordinary[2], legacy_reverse))
        zero_direction = self.policy._phase4_directional_evidence(
            base,
            torch.zeros_like(forward),
            torch.zeros_like(reverse),
            score_limit,
        )
        for evidence in zero_direction[:3]:
            self.assertTrue(torch.equal(evidence, torch.zeros_like(evidence)))
        permutation = torch.randperm(120)
        permuted = self.policy._phase4_directional_evidence(
            base[:, permutation],
            forward[permutation],
            reverse[permutation],
            score_limit,
        )
        for ordinary_value, permuted_value in zip(ordinary, permuted, strict=True):
            expected = (
                ordinary_value[:, permutation, :]
                if ordinary_value.ndim == 3
                else ordinary_value[:, permutation]
            )
            self.assertTrue(
                torch.allclose(permuted_value, expected, atol=1e-6, rtol=1e-5)
            )
        learned_gains = ordinary[-1]
        self.assertTrue(bool(torch.isfinite(learned_gains).all().item()))
        self.assertTrue(bool((learned_gains >= 0.0).all().item()))
        self.assertTrue(bool((learned_gains <= 2.0).all().item()))
        for evidence in ordinary[:3]:
            self.assertTrue(bool((evidence.abs() <= score_limit).all().item()))

    def test_condition_axis_migration_is_zero_versioned_and_atomic(self) -> None:
        full_state = self.policy.state_dict()
        reversible_keys = {
            key for key in full_state if _is_reversible_state_key(key)
        }
        adapter_keys = {
            key for key in full_state if key.startswith("procedural_fast_adapter.")
        }
        goal_keys = {
            key for key in full_state if key.startswith("procedural_goal_projection.")
        }
        legacy_state = {
            key: value
            for key, value in full_state.items()
            if key != runner._CONDITION_AXIS_KEY
            and key not in adapter_keys
            and key not in goal_keys
            and key not in reversible_keys
        }

        def payload(model, runner_identity=runner._REPORT_VERSION):
            return {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": model,
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "d" * 64,
                "runner": runner_identity,
                "stage": "integrated",
            }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            v10 = directory_path / "v10.pt"
            torch.save(
                payload(legacy_state, "angler.phase5-skill-memory-stream.v10"),
                v10,
            )
            migrated = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                migrated,
                v10,
                self.profile,
            )
            self.assertEqual(
                record["fresh_parameter_keys"],
                sorted(
                    {
                        runner._CONDITION_AXIS_KEY,
                        *adapter_keys,
                        *goal_keys,
                        *reversible_keys,
                    }
                ),
            )
            self.assertTrue(
                torch.equal(
                    migrated.relational_branch_router.condition_axis,
                    torch.zeros_like(
                        migrated.relational_branch_router.condition_axis
                    ),
                )
            )
            for key, value in legacy_state.items():
                self.assertTrue(torch.equal(migrated.state_dict()[key], value), key)

            v12_state = {
                key: value
                for key, value in full_state.items()
                if key not in adapter_keys
                and key not in goal_keys
                and key not in reversible_keys
            }
            v12_state[runner._CONDITION_AXIS_KEY] = torch.full_like(
                v12_state[runner._CONDITION_AXIS_KEY],
                0.75,
            )
            v12 = directory_path / "v12.pt"
            torch.save(
                payload(v12_state, "angler.phase5-skill-memory-stream.v12"),
                v12,
            )
            reset_v12 = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            reset_record = runner._load_initial_policy_checkpoint(
                reset_v12,
                v12,
                self.profile,
            )
            self.assertEqual(
                reset_record["reset_parameter_keys"],
                [runner._CONDITION_AXIS_KEY],
            )
            self.assertEqual(
                reset_record["fresh_parameter_keys"],
                sorted(
                    {
                        runner._CONDITION_AXIS_KEY,
                        *adapter_keys,
                        *goal_keys,
                        *reversible_keys,
                    }
                ),
            )
            self.assertTrue(
                torch.equal(
                    reset_v12.relational_branch_router.condition_axis,
                    torch.zeros_like(
                        reset_v12.relational_branch_router.condition_axis
                    ),
                )
            )
            for key, value in v12_state.items():
                if key != runner._CONDITION_AXIS_KEY:
                    self.assertTrue(
                        torch.equal(reset_v12.state_dict()[key], value),
                        key,
                    )

            # v13 introduced the canonical left-versus-right outcome sign used
            # by v15. Recursive execution and directional calibration do not
            # reinterpret it. Every existing v13 tensor is retained exactly;
            # only the genuinely new neutral mixer may be fresh.
            mixer_keys = {
                key
                for key in full_state
                if key.startswith("phase4_direction_mixer.")
            }
            v13_state = {
                key: value.detach().clone()
                for key, value in full_state.items()
                if key not in mixer_keys
                and key not in adapter_keys
                and key not in goal_keys
                and key not in reversible_keys
            }
            v13_state[runner._CONDITION_AXIS_KEY] = torch.linspace(
                -0.75,
                0.75,
                v13_state[runner._CONDITION_AXIS_KEY].numel(),
            )
            v13 = directory_path / "v13.pt"
            torch.save(
                payload(v13_state, "angler.phase5-skill-memory-stream.v13"),
                v13,
            )
            retained_v13 = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            retained_record = runner._load_initial_policy_checkpoint(
                retained_v13,
                v13,
                self.profile,
            )
            self.assertEqual(retained_record["reset_parameter_keys"], [])
            self.assertEqual(
                retained_record["fresh_parameter_keys"],
                sorted(mixer_keys | adapter_keys | goal_keys | reversible_keys),
            )
            for key, value in v13_state.items():
                self.assertTrue(
                    torch.equal(retained_v13.state_dict()[key], value),
                    key,
                )

            current_missing = directory_path / "current-missing.pt"
            current_without_axis = dict(full_state)
            current_without_axis.pop(runner._CONDITION_AXIS_KEY)
            torch.save(
                payload(current_without_axis, runner._REPORT_VERSION),
                current_missing,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    current_missing,
                    self.profile,
                )

            legacy_router = dict(legacy_state)
            legacy_router["recursive_branch_router.0.weight"] = torch.ones(1)
            legacy_router["recursive_branch_router.3.bias"] = torch.zeros(2)
            partial = directory_path / "legacy-router.pt"
            torch.save(
                payload(legacy_router, "angler.phase5-skill-memory-stream.v10"),
                partial,
            )
            migrated_legacy = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            legacy_record = runner._load_initial_policy_checkpoint(
                migrated_legacy,
                partial,
                self.profile,
            )
            self.assertEqual(
                legacy_record["dropped_parameter_keys"],
                [
                    "recursive_branch_router.0.weight",
                    "recursive_branch_router.3.bias",
                ],
            )

            contaminated = directory_path / "contaminated-v10.pt"
            contaminated_v10 = dict(legacy_state)
            contaminated_v10[runner._CONDITION_AXIS_KEY] = full_state[
                runner._CONDITION_AXIS_KEY
            ]
            torch.save(
                payload(
                    contaminated_v10,
                    "angler.phase5-skill-memory-stream.v10",
                ),
                contaminated,
            )
            with self.assertRaisesRegex(RuntimeError, "undeclared condition axis"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    contaminated,
                    self.profile,
                )

            contaminated_v2 = directory_path / "folded-v2.pt"
            torch.save(
                payload(
                    legacy_state,
                    runner_identity="angler.phase5-skill-memory-stream.v2",
                ),
                contaminated_v2,
            )
            with self.assertRaisesRegex(RuntimeError, "folded-seam state"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    contaminated_v2,
                    self.profile,
                )

            unexpected = directory_path / "deprecated-fusion.pt"
            deprecated_state = dict(full_state)
            deprecated_state["composition_fusion.deprecated"] = torch.zeros(1)
            torch.save(payload(deprecated_state), unexpected)
            migrated = runner.SkillMemoryPolicy(
                self.profile,
                copy.deepcopy(self.policy.stable_compiler),
            )
            record = runner._load_initial_policy_checkpoint(
                migrated,
                unexpected,
                self.profile,
            )
            self.assertEqual(
                record["dropped_parameter_keys"],
                ["composition_fusion.deprecated"],
            )

            unknown = directory_path / "unknown-key.pt"
            unknown_state = dict(full_state)
            unknown_state["unknown_module.weight"] = torch.zeros(1)
            torch.save(payload(unknown_state), unknown)
            with self.assertRaisesRegex(RuntimeError, "undeclared state migration"):
                runner._load_initial_policy_checkpoint(
                    runner.SkillMemoryPolicy(
                        self.profile,
                        copy.deepcopy(self.policy.stable_compiler),
                    ),
                    unknown,
                    self.profile,
                )

    def test_relational_acquisition_stage_is_bounded(self) -> None:
        trainable = runner._configure_stage_trainability(
            self.policy,
            "relational_acquisition",
        )
        expected = tuple(
            name
            for name, _ in self.policy.named_parameters()
            if name.startswith(runner._RELATIONAL_ACQUISITION_PREFIXES)
        )

        self.assertEqual(trainable, expected)
        self.assertTrue(trainable)
        for name, parameter in self.policy.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)

    def test_harmonization_stage_is_bounded(self) -> None:
        self.assertEqual(
            runner._HARMONIZATION_TRAINABLE_PREFIXES,
            ("phase4_direction_mixer.",),
        )
        trainable = runner._configure_stage_trainability(
            self.policy,
            "harmonization",
        )
        expected = tuple(
            name
            for name, _ in self.policy.named_parameters()
            if name.startswith(runner._HARMONIZATION_TRAINABLE_PREFIXES)
        )

        self.assertEqual(trainable, expected)
        self.assertTrue(trainable)
        self.assertNotIn(runner._CONDITION_AXIS_KEY, trainable)
        expected_count = sum(
            parameter.numel()
            for parameter in self.policy.phase4_direction_mixer.parameters()
        )
        actual_count = sum(
            parameter.numel()
            for name, parameter in self.policy.named_parameters()
            if name in trainable
        )
        self.assertEqual(actual_count, expected_count)
        for name, parameter in self.policy.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)

    def test_condition_relation_is_zero_preserving_and_bilinear(self) -> None:
        router = self.policy.relational_branch_router
        context = torch.randn(3, self.profile.width)

        self.assertTrue(
            torch.equal(router(context, False), torch.zeros(3, 2))
        )
        self.assertTrue(
            torch.equal(router(context, True), torch.zeros(3, 2))
        )
        self.assertTrue(
            torch.equal(
                router(torch.zeros_like(context), True),
                torch.zeros(3, 2),
            )
        )

        with torch.no_grad():
            router.condition_axis.copy_(
                torch.linspace(-1.0, 1.0, router.condition_axis.numel())
            )
        true_logits = router(context, True)
        false_logits = router(context, False)
        reversed_context = router(-context, True)
        self.assertTrue(torch.allclose(false_logits, -true_logits))
        self.assertTrue(
            torch.allclose(
                torch.softmax(false_logits, dim=-1),
                torch.flip(torch.softmax(true_logits, dim=-1), dims=(-1,)),
            )
        )
        self.assertTrue(torch.allclose(reversed_context, -true_logits))
        self.assertTrue(
            torch.equal(
                true_logits.sum(dim=-1),
                torch.zeros(true_logits.shape[0]),
            )
        )
        content_only = context.clone()
        content_only[:, router.outcome_start :] = 0.0
        self.assertTrue(
            torch.equal(
                router(content_only, True),
                torch.zeros_like(true_logits),
            )
        )

    def test_binary_branch_summary_rejects_fixed_side_and_ties(self) -> None:
        cells = (
            ("IF_FLAG", False, 0),
            ("IF_FLAG", True, 1),
            ("IF_NOT_FLAG", False, 1),
            ("IF_NOT_FLAG", True, 0),
        )

        def rows_for(weights_by_cell):
            rows = []
            for index, (operator, public_flag, expected) in enumerate(cells):
                weights = weights_by_cell[index]
                execution_tied = weights[0] == weights[1]
                chosen = 0 if weights[0] >= weights[1] else 1
                executed = (
                    list(weights)
                    if execution_tied
                    else [float(chosen == 0), float(chosen == 1)]
                )
                rows.append(
                    {
                    "operator": operator,
                    "public_flag": public_flag,
                    "expected_branch": expected,
                    "eligible": True,
                    "branch_weights": weights,
                    "executed_branch_weights": executed,
                    "execution_tied": execution_tied,
                }
                )
            return rows

        perfect = runner._summarize_binary_branch_choices(
            rows_for(([0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]))
        )
        fixed_right = runner._summarize_binary_branch_choices(
            rows_for(([0.1, 0.9],) * 4)
        )
        tied = runner._summarize_binary_branch_choices(
            rows_for(([0.5, 0.5],) * 4)
        )

        self.assertEqual(perfect["hard_accuracy"], 1.0)
        self.assertEqual(perfect["tie_rate"], 0.0)
        self.assertEqual(fixed_right["hard_accuracy"], 0.5)
        self.assertEqual(
            [
                fixed_right["cells"][f"{operator}:{int(public_flag)}"][
                    "hard_accuracy"
                ]
                for operator, public_flag, _ in cells
            ],
            [0.0, 1.0, 1.0, 0.0],
        )
        self.assertEqual(tied["hard_accuracy"], 0.0)
        self.assertEqual(tied["tie_rate"], 1.0)

    def test_binary_execution_uses_one_scale_aware_tie_predicate(self) -> None:
        epsilon = torch.finfo(torch.float32).eps
        logits = torch.tensor(
            (
                (0.0, 0.0),
                (1.0, 1.0 + 16.0 * epsilon),
                (1.0, 1.0 + 64.0 * epsilon),
            ),
            requires_grad=True,
        )

        soft, executed, tied = runner._execute_binary_branch(logits)

        self.assertEqual(tied.tolist(), [True, True, False])
        self.assertTrue(torch.equal(executed[:2], soft[:2]))
        self.assertTrue(torch.equal(executed[2], torch.tensor((0.0, 1.0))))
        executed[:, 1].sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all().item()))
        self.assertGreater(float(logits.grad.abs().sum().item()), 0.0)

    def test_condition_relation_receives_gradient_without_unfreezing_old_state(self) -> None:
        runner._configure_stage_trainability(
            self.policy,
            "relational_acquisition",
        )
        router = self.policy.relational_branch_router
        context = torch.randn(2, self.profile.width)
        loss = router(context, True)[:, 1].sum()
        loss.backward()

        self.assertIsNotNone(router.condition_axis.grad)
        self.assertTrue(bool(torch.isfinite(router.condition_axis.grad).all().item()))
        self.assertGreater(float(router.condition_axis.grad.abs().sum().item()), 0.0)
        for name, parameter in self.policy.named_parameters():
            if not name.startswith(runner._RELATIONAL_ACQUISITION_PREFIXES):
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)

    def test_meta_stream_queries_heldout_contexts_and_private_compositions(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_105,
            instances_per_program=8,
        )
        groups = runner._group_evaluator_pairs(partition.tasks)
        by_root = runner._meta_variants_by_root(groups)
        roots = runner._ordered_meta_roots(by_root)
        varied_index = next(
            index for index, root in enumerate(roots) if len(by_root[root]) > 1
        )
        local_sequences = runner._meta_local_sequences(groups, varied_index)
        self.assertEqual(len(local_sequences), 4)
        self.assertTrue(
            all(
                not sequence[0][0].learner.request.children
                for sequence in local_sequences
            )
        )
        local = local_sequences[0]

        self.assertTrue(local)
        for support, query in local:
            self.assertEqual(
                support.learner.request.symbol,
                query.learner.request.symbol,
            )
            self.assertNotEqual(
                support.learner.to_canonical(),
                query.learner.to_canonical(),
            )
        parity_sequences = (
            runner._meta_local_sequences(groups, 0),
            runner._meta_local_sequences(groups, 1),
        )
        self.assertEqual(
            {
                support.learner.public_flag
                for sequences in parity_sequences
                for sequence in sequences
                for support, _ in sequence
            },
            {False, True},
        )
        self.assertEqual(
            {
                query.learner.public_flag
                for sequences in parity_sequences
                for sequence in sequences
                for _, query in sequence
            },
            {False, True},
        )
        interference = runner._meta_episode_sequence(groups, 0)
        self.assertEqual(len(interference), 40)
        self.assertEqual(
            len({pair[0].learner.request.symbol for pair in interference}),
            10,
        )
        root_counts = {}
        for support, _ in interference:
            root_counts[support.learner.request.symbol] = (
                root_counts.get(support.learner.request.symbol, 0) + 1
            )
        self.assertEqual(set(root_counts.values()), {4})
        for symbol in root_counts:
            selected = tuple(
                pair
                for pair in interference
                if pair[0].learner.request.symbol == symbol
            )
            self.assertEqual(
                [
                    support.learner.public_flag
                    for support, _ in selected
                ].count(False),
                2,
            )
            self.assertEqual(
                [
                    support.learner.public_flag
                    for support, _ in selected
                ].count(True),
                2,
            )
            self.assertEqual(
                [query.learner.public_flag for _, query in selected].count(False),
                2,
            )
            self.assertEqual(
                [query.learner.public_flag for _, query in selected].count(True),
                2,
            )
        interference_pair = (
            runner._meta_episode_sequence(groups, 0),
            runner._meta_episode_sequence(groups, 1),
        )
        for operator in ("IF_FLAG", "IF_NOT_FLAG"):
            selected = tuple(
                pair
                for episode in interference_pair
                for pair in episode
                if pair[0].hidden.program.operator == operator
            )
            self.assertTrue(selected)
            self.assertEqual(
                {support.learner.public_flag for support, _ in selected},
                {False, True},
            )
            self.assertEqual(
                {query.learner.public_flag for _, query in selected},
                {False, True},
            )
            self.assertTrue(
                all(
                    support.learner.public_flag != query.learner.public_flag
                    for support, query in selected
                )
            )
        composition = runner._meta_composition_queries(groups, 0)
        self.assertEqual(
            {pair.learner.request.depth for pair in composition},
            {2, 3},
        )
        composition_pair = composition + runner._meta_composition_queries(groups, 1)
        self.assertEqual(
            {pair.learner.public_flag for pair in composition_pair},
            {False, True},
        )
        self.assertEqual(
            {
                pair.hidden.program.operator
                for pair in composition_pair
                if pair.hidden.program.operator in {"IF_FLAG", "IF_NOT_FLAG"}
            },
            {"IF_FLAG", "IF_NOT_FLAG"},
        )

    def test_leaf_support_interventions_are_varied_and_task_independent(self) -> None:
        first = tuple(
            runner._stratified_leaf_candidate_index(0, root, support)
            for root in range(4)
            for support in range(4)
        )
        second = tuple(
            runner._stratified_leaf_candidate_index(1, root, support)
            for root in range(4)
            for support in range(4)
        )

        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(second), len(set(second)))
        self.assertNotEqual(first, second)
        self.assertTrue(all(0 <= value < len(runner._PERMUTATIONS) for value in first))

    def test_matched_descendant_loss_uses_complete_child_delta_and_is_read_only(self) -> None:
        pair = suite.make_skill_memory_meta_matched_queries(85_101)[0]
        state = self.policy.initial_state(1)
        with self.assertRaisesRegex(RuntimeError, "complete trees"):
            runner._outer_matched_descendant_loss(
                self.policy,
                state,
                pair,
                margin=0.10,
            )

        expressions = {}

        def collect(expression) -> None:
            expressions[expression.symbol] = expression
            for child in expression.children:
                collect(child)

        collect(pair.left.learner.request)
        collect(pair.right.learner.request)
        for expression in expressions.values():
            public_task = suite.PublicSkillMemoryTask(
                pair.left.learner.items,
                pair.left.learner.public_flag,
                expression,
            )
            proposal = runner.propose_task(
                self.policy,
                public_task,
                state,
                greedy=False,
                temperature=1.0,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                1.0,
                state,
            ).candidate_state
        before = procedural_skill_state_digest(state)
        loss, cross, paired_delta = runner._outer_matched_descendant_loss(
            self.policy,
            state,
            pair,
            margin=0.10,
        )

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertTrue(bool(torch.isfinite(cross).item()))
        self.assertTrue(bool(torch.isfinite(paired_delta).item()))
        self.assertGreaterEqual(float(cross.item()), 0.0)
        self.assertGreaterEqual(float(paired_delta.item()), 0.0)
        left_root_only = self.policy.score_task(
            pair.left.learner,
            state,
            include_descendants=False,
        )
        right_root_only = self.policy.score_task(
            pair.right.learner,
            state,
            include_descendants=False,
        )
        self.assertTrue(torch.equal(left_root_only.logits, right_root_only.logits))
        self.assertEqual(float(left_root_only.root_available.item()), 1.0)
        self.assertEqual(float(right_root_only.root_available.item()), 1.0)
        self.assertEqual(procedural_skill_state_digest(state), before)

    def test_neutral_feedback_is_retained_as_procedural_evidence(self) -> None:
        pair = next(
            pair
            for pair in self.partition.tasks
            if not pair.learner.request.children
        )
        state = self.policy.initial_state(1)
        proposal = runner.propose_task(self.policy, pair.learner, state)
        staged = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            0.5,
            state,
        )

        self.assertNotEqual(
            procedural_skill_state_digest(staged.candidate_state),
            procedural_skill_state_digest(state),
        )
        self.assertGreater(staged.delta_norm, 0.0)
        self.assertEqual(int(staged.candidate_state.write_counts.sum().item()), 1)
        self.assertEqual(int(staged.candidate_state.occupied.sum().item()), 1)

    def test_validated_state_digest_matches_legacy_and_rejects_stale(self) -> None:
        pair = next(
            pair for pair in self.partition.tasks if not pair.learner.request.children
        )
        state = self.policy.initial_state(1)
        proposal = runner.propose_task(self.policy, pair.learner, state)
        state_digest = procedural_skill_state_digest(state)

        legacy = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            1.0,
            state,
        )
        with mock.patch.object(
            runner,
            "procedural_skill_state_digest",
            side_effect=AssertionError("validated digest was recomputed"),
        ):
            optimized = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                1.0,
                state,
                validated_state_digest=state_digest,
            )

        self.assertEqual(optimized.write_slot, legacy.write_slot)
        self.assertEqual(optimized.delta_norm, legacy.delta_norm)
        self.assertTrue(
            torch.equal(
                optimized.route_probabilities,
                legacy.route_probabilities,
            )
        )
        self.assertEqual(
            procedural_skill_state_digest(optimized.candidate_state),
            procedural_skill_state_digest(legacy.candidate_state),
        )

        changed_digest = procedural_skill_state_digest(legacy.candidate_state)
        self.assertNotEqual(changed_digest, state_digest)
        with self.assertRaisesRegex(ValueError, "bound"):
            runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                1.0,
                legacy.candidate_state,
                validated_state_digest=changed_digest,
            )

    def test_meta_and_online_use_the_same_bounded_core_state(self) -> None:
        pair = self.partition.tasks[0]
        state = self.policy.initial_state(1)
        proposal = runner.propose_task(self.policy, pair.learner, state)
        meta = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            1.0,
            state,
        )
        online = runner.apply_transactional_feedback(
            self.policy,
            pair.learner,
            proposal,
            1.0,
            state,
        )

        self.assertEqual(
            procedural_skill_state_digest(meta.candidate_state),
            procedural_skill_state_digest(online.state),
        )
        self.assertEqual(meta.delta_norm, online.delta_norm)

    def test_online_no_effect_write_rejects_with_exact_rollback(self) -> None:
        pair = next(
            pair for pair in self.partition.tasks if not pair.learner.request.children
        )
        state = self.policy.initial_state(1)
        before = procedural_skill_state_digest(state)
        proposal = runner.propose_task(self.policy, pair.learner, state)
        reward = 1.0
        root = proposal.scores.root
        staged = self.policy.memory.propose_feedback(
            root.state_embedding,
            root.goal_embedding,
            root.candidate_embeddings,
            torch.tensor((proposal.candidate_index,), dtype=torch.long),
            proposal.scores.logits.new_tensor((reward,)),
            proposal.scores.logits - root.memory_read.score_bias,
            state=state,
            structural_context=root.recursive_predecessor,
        )
        rejected = self.policy.memory.commit_bounded_feedback(
            staged,
            minimum_effect=float(staged.delta_norm.item()) + 1.0,
        )

        self.assertFalse(bool(rejected.accepted.item()))
        self.assertIs(rejected.state, state)
        self.assertEqual(procedural_skill_state_digest(rejected.state), before)
        self.assertEqual(float(rejected.delta_norm.item()), 0.0)
        mismatched_snapshot = snapshot_procedural_skill_state(state)
        mismatched_snapshot["write_counts"][0, 0] = 1
        mismatched_snapshot["occupied"][0, 0] = True
        mismatched_state = restore_procedural_skill_state(mismatched_snapshot)
        with self.assertRaisesRegex(ValueError, "bound"):
            runner.apply_transactional_feedback(
                self.policy,
                pair.learner,
                proposal,
                reward,
                mismatched_state,
            )
        other = self.partition.tasks[1]
        with self.assertRaisesRegex(ValueError, "public task"):
            runner.apply_transactional_feedback(
                self.policy,
                other.learner,
                proposal,
                reward,
                state,
            )

    def test_non_finite_post_write_rescore_rolls_back_only_that_failure(self) -> None:
        pair = next(
            pair for pair in self.partition.tasks if not pair.learner.request.children
        )
        state = self.policy.initial_state(1)
        before = procedural_skill_state_digest(state)
        proposal = runner.propose_task(self.policy, pair.learner, state)

        with mock.patch.object(
            self.policy,
            "score_task",
            side_effect=runner._NonFinitePolicyScoresError(
                "policy produced non-finite candidate scores"
            ),
        ) as scorer:
            result = runner.apply_transactional_feedback(
                self.policy,
                pair.learner,
                proposal,
                1.0,
                state,
            )

        scorer.assert_called_once()
        self.assertTrue(result.core_accepted)
        self.assertFalse(result.accepted)
        self.assertIs(result.state, state)
        self.assertEqual(procedural_skill_state_digest(result.state), before)
        self.assertEqual(result.delta_norm, 0.0)
        self.assertEqual(result.after_loss, result.before_loss)

        with mock.patch.object(
            self.policy,
            "score_task",
            side_effect=RuntimeError("unexpected scorer failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected scorer failure"):
                runner.apply_transactional_feedback(
                    self.policy,
                    pair.learner,
                    proposal,
                    1.0,
                    state,
                )

    def test_entity_rename_and_display_shuffle_preserve_semantic_logits(self) -> None:
        pair = self.partition.tasks[0]
        renamed = suite.make_renamed_skill_variant(pair, seed=85_102)
        state = self.policy.initial_state(1)
        proposal = runner.propose_task(self.policy, pair.learner, state)
        reward = suite.score_skill_memory_answer(
            pair.learner, pair.hidden, proposal.answer
        )
        populated = runner.propose_differentiable_feedback(
            self.policy, proposal, reward, state
        ).candidate_state
        original = self.policy.score_task(pair.learner, populated).logits[0]
        changed = self.policy.score_task(renamed.learner, populated).logits[0]

        def semantic_map(public_task, logits):
            result = {}
            for index, permutation in enumerate(runner._PERMUTATIONS):
                result[
                    tuple(
                        (
                            public_task.items[item_index].rank_a,
                            public_task.items[item_index].rank_b,
                            public_task.items[item_index].group,
                            public_task.items[item_index].marked,
                        )
                        for item_index in permutation
                    )
                ] = logits[index]
            return result

        left = semantic_map(pair.learner, original)
        right = semantic_map(renamed.learner, changed)
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.allclose(left[key], right[key], atol=1e-6, rtol=1e-5))

    def test_binary_feedback_waits_for_two_distinct_acquired_children(self) -> None:
        binary = next(
            pair
            for pair in self.partition.tasks
            if len(pair.learner.request.children) == 2
        )
        state = self.policy.initial_state(1)
        before = procedural_skill_state_digest(state)
        proposal = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        reward = suite.score_skill_memory_answer(
            binary.learner,
            binary.hidden,
            proposal.answer,
        )
        self.assertFalse(bool(proposal.scores.root.feedback_available.item()))
        self.assertTrue(
            torch.equal(
                proposal.scores.root.feedback_context,
                torch.zeros_like(proposal.scores.root.feedback_context),
            )
        )

        with mock.patch.object(
            self.policy.composition_memory,
            "propose_feedback",
            wraps=self.policy.composition_memory.propose_feedback,
        ) as feedback_writer:
            meta = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                reward,
                state,
            )
            online = runner.apply_transactional_feedback(
                self.policy,
                binary.learner,
                proposal,
                reward,
                state,
            )
        feedback_writer.assert_not_called()
        self.assertIs(meta.candidate_state, state)
        self.assertEqual(
            meta.write_slot,
            int(proposal.scores.root.memory_read.write_slots.item()),
        )
        self.assertEqual(meta.delta_norm, 0.0)
        self.assertIs(online.state, state)
        self.assertFalse(online.accepted)
        self.assertFalse(online.core_accepted)
        self.assertEqual(online.write_slot, meta.write_slot)
        self.assertEqual(online.after_loss, online.before_loss)
        self.assertEqual(procedural_skill_state_digest(state), before)

        leaf_by_symbol = {
            pair.learner.request.symbol: pair
            for pair in self.partition.tasks
            if not pair.learner.request.children
        }
        first_child = leaf_by_symbol[binary.learner.request.children[0].symbol]
        child_proposal = runner.propose_task(
            self.policy,
            first_child.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        child_reward = suite.score_skill_memory_answer(
            first_child.learner,
            first_child.hidden,
            child_proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            child_proposal,
            child_reward,
            state,
        ).candidate_state
        one_child_proposal = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        self.assertFalse(
            bool(one_child_proposal.scores.root.feedback_available.item())
        )
        repeated_request = suite.PublicSkillExpression(
            binary.learner.request.symbol,
            (
                binary.learner.request.children[0],
                binary.learner.request.children[0],
            ),
        )
        repeated_task = suite.PublicSkillMemoryTask(
            binary.learner.items,
            binary.learner.public_flag,
            repeated_request,
        )
        repeated_proposal = runner.propose_task(
            self.policy,
            repeated_task,
            state,
            greedy=False,
            temperature=1.0,
        )
        self.assertFalse(
            bool(repeated_proposal.scores.root.feedback_available.item())
        )
        self.assertTrue(
            torch.allclose(
                repeated_proposal.scores.root.feedback_context,
                torch.zeros_like(
                    repeated_proposal.scores.root.feedback_context
                ),
                atol=1e-6,
                rtol=0.0,
            )
        )
        acquired_digest = procedural_skill_state_digest(state)
        with mock.patch.object(
            self.policy.composition_memory,
            "propose_feedback",
            wraps=self.policy.composition_memory.propose_feedback,
        ) as feedback_writer:
            one_child_write = runner.propose_differentiable_feedback(
                self.policy,
                one_child_proposal,
                1.0,
                state,
            )
            repeated_write = runner.propose_differentiable_feedback(
                self.policy,
                repeated_proposal,
                1.0,
                state,
            )
        feedback_writer.assert_not_called()
        self.assertIs(one_child_write.candidate_state, state)
        self.assertIs(repeated_write.candidate_state, state)
        self.assertEqual(procedural_skill_state_digest(state), acquired_digest)

    def test_binary_credit_uses_child_policies_and_attempted_action(self) -> None:
        binary = next(
            pair
            for pair in self.partition.tasks
            if len(pair.learner.request.children) == 2
        )
        leaf_by_symbol = {
            pair.learner.request.symbol: pair
            for pair in self.partition.tasks
            if not pair.learner.request.children
        }
        state = self.policy.initial_state(1)
        for child in binary.learner.request.children:
            support = leaf_by_symbol[child.symbol]
            support_proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            support_reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                support_proposal.answer,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                support_proposal,
                support_reward,
                state,
            ).candidate_state

        proposal = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        root = proposal.scores.root
        self.assertTrue(bool(root.feedback_available.item()))
        self.assertTrue(root.child_candidate_scores.requires_grad)
        self.assertFalse(root.conditioned_child_candidate_scores.requires_grad)
        self.assertFalse(root.candidate_branch_advantages.requires_grad)
        self.assertTrue(
            torch.allclose(
                proposal.behavior_probabilities,
                torch.softmax(proposal.scores.logits[0], dim=-1),
                atol=1e-7,
                rtol=0.0,
            )
        )

        child_nodes = {
            node.path: node
            for node in proposal.scores.nodes
            if len(node.path) == 1
        }
        signed_flag = 1.0 if binary.learner.public_flag else -1.0
        expected_child_scores = torch.stack(
            (
                child_nodes[(0,)].memory_read.score_bias,
                child_nodes[(1,)].memory_read.score_bias,
            ),
            dim=1,
        )
        self.assertTrue(
            torch.allclose(
                root.child_candidate_scores,
                expected_child_scores,
                atol=1e-7,
                rtol=0.0,
            )
        )
        expected_advantages = signed_flag * (
            root.child_candidate_scores[:, 1].detach()
            - root.child_candidate_scores[:, 0].detach()
        )
        self.assertTrue(
            torch.allclose(
                root.candidate_branch_advantages,
                expected_advantages,
                atol=1e-7,
                rtol=0.0,
            )
        )

        swapped_request = suite.PublicSkillExpression(
            binary.learner.request.symbol,
            tuple(reversed(binary.learner.request.children)),
        )
        swapped_task = suite.PublicSkillMemoryTask(
            binary.learner.items,
            binary.learner.public_flag,
            swapped_request,
        )
        swapped = self.policy.score_task(swapped_task, state)
        self.assertTrue(
            torch.allclose(
                swapped.root.candidate_branch_advantages,
                -root.candidate_branch_advantages,
                atol=1e-7,
                rtol=0.0,
            )
        )

        basis = runner._canonical_binary_outcome_basis(
            proposal,
            self.policy.composition_memory,
        )
        self.assertIsNotNone(basis)
        assert basis is not None
        self.assertFalse(basis.requires_grad)
        probabilities = proposal.behavior_probabilities.unsqueeze(0)
        child_scores = root.conditioned_child_candidate_scores
        child_probabilities = probabilities.unsqueeze(1)
        centered_children = child_scores - (
            child_probabilities * child_scores
        ).sum(dim=-1, keepdim=True)
        child_variances = (
            child_probabilities * centered_children.square()
        ).sum(dim=-1, keepdim=True)
        normalized_children = centered_children / child_variances.sqrt()
        normalized_advantages = (
            normalized_children[:, 1] - normalized_children[:, 0]
        )
        mean = (probabilities * normalized_advantages).sum(
            dim=-1,
            keepdim=True,
        )
        centered = normalized_advantages - mean
        variance = (probabilities * centered.square()).sum(
            dim=-1,
            keepdim=True,
        )
        bounded = torch.tanh(centered / variance.sqrt())
        bounded = bounded - (probabilities * bounded).sum(
            dim=-1,
            keepdim=True,
        )
        canonical_actions = bounded / bounded.abs().amax(dim=-1, keepdim=True)
        expected_scalar = canonical_actions[
            :, proposal.candidate_index
        ].unsqueeze(-1)
        self.assertTrue(
            torch.allclose(
                basis,
                expected_scalar.expand_as(basis),
                atol=1e-7,
                rtol=0.0,
            )
        )

        greedy = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=True,
        )
        self.assertEqual(float(greedy.behavior_probabilities.sum().item()), 1.0)
        self.assertEqual(
            int(torch.count_nonzero(greedy.behavior_probabilities).item()),
            1,
        )
        self.assertIsNone(
            runner._canonical_binary_outcome_basis(
                greedy,
                self.policy.composition_memory,
            )
        )
        greedy_write = runner.propose_differentiable_feedback(
            self.policy,
            greedy,
            1.0,
            state,
        )
        self.assertIs(greedy_write.candidate_state, state)
        self.assertEqual(greedy_write.delta_norm, 0.0)

        with torch.no_grad():
            self.policy.relational_branch_router.condition_axis.fill_(1.0)
        positive_state = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            1.0,
            state,
        ).candidate_state
        inverted_state = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            0.0,
            state,
        ).candidate_state
        neutral_state = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            0.5,
            state,
        ).candidate_state
        positive_weights = self.policy.score_task(
            binary.learner,
            positive_state,
        ).root.branch_weights
        inverted_weights = self.policy.score_task(
            binary.learner,
            inverted_state,
        ).root.branch_weights
        neutral_weights = self.policy.score_task(
            binary.learner,
            neutral_state,
        ).root.branch_weights
        self.assertTrue(
            torch.allclose(
                positive_weights,
                inverted_weights.flip(dims=(-1,)),
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.equal(
                neutral_weights,
                torch.full_like(neutral_weights, 0.5),
            )
        )

        skewed_probabilities = torch.full_like(
            proposal.behavior_probabilities,
            0.2 / (proposal.behavior_probabilities.numel() - 1),
        )
        skewed_probabilities[0] = 0.8
        skewed = replace(
            proposal,
            behavior_probabilities=skewed_probabilities,
        )
        action_features = []
        for candidate_index in range(skewed_probabilities.numel()):
            candidate_basis = runner._canonical_binary_outcome_basis(
                replace(skewed, candidate_index=candidate_index),
                self.policy.composition_memory,
            )
            self.assertIsNotNone(candidate_basis)
            assert candidate_basis is not None
            action_features.append(candidate_basis[0, 0])
        action_features = torch.stack(action_features)
        self.assertLessEqual(float(action_features.abs().max().item()), 1.0)
        self.assertAlmostEqual(
            float((skewed_probabilities * action_features).sum().item()),
            0.0,
            places=6,
        )

    def test_acquired_children_enter_frozen_compiler_and_preserve_order(self) -> None:
        binary = next(
            pair
            for pair in self.partition.tasks
            if len(pair.learner.request.children) == 2
        )
        leaf_by_symbol = {
            pair.learner.request.symbol: pair
            for pair in self.partition.tasks
            if not pair.learner.request.children
        }
        state = self.policy.initial_state(1)
        for child in binary.learner.request.children:
            support = leaf_by_symbol[child.symbol]
            proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                proposal.answer,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                reward,
                state,
            ).candidate_state

        children_only = self.policy.score_task(binary.learner, state)
        self.assertTrue(
            torch.equal(children_only.logits, torch.zeros_like(children_only.logits))
        )
        root_proposal = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        observed_root_reward = suite.score_skill_memory_answer(
            binary.learner,
            binary.hidden,
            root_proposal.answer,
        )
        # This structural-gradient test needs an identified non-neutral
        # outcome coordinate.  Binarize the ordinary bounded judge score; no
        # target or solution route enters the learner.
        root_reward = 1.0 if observed_root_reward >= 0.5 else 0.0
        with mock.patch.object(
            self.policy.composition_memory,
            "propose_feedback",
            wraps=self.policy.composition_memory.propose_feedback,
        ) as feedback_writer:
            state = runner.propose_differentiable_feedback(
                self.policy,
                root_proposal,
                root_reward,
                state,
            ).candidate_state
        self.assertTrue(bool(root_proposal.scores.root.feedback_available.item()))
        written_context = feedback_writer.call_args.kwargs[
            "structural_context"
        ]
        self.assertTrue(
            torch.equal(
                written_context,
                torch.zeros_like(root_proposal.scores.root.feedback_context),
            )
        )
        written_basis = feedback_writer.call_args.kwargs[
            "outcome_direction_basis"
        ]
        self.assertEqual(
            written_basis.shape,
            (1, self.policy.composition_memory.evidence_outcome_width),
        )
        self.assertTrue(bool(torch.isfinite(written_basis).all().item()))
        self.assertTrue(bool((written_basis.abs() <= 1.0).all().item()))
        self.assertTrue(
            torch.equal(
                written_basis,
                written_basis[:, :1].expand_as(written_basis),
            )
        )
        proposal_children = {
            node.path: node
            for node in root_proposal.scores.nodes
            if len(node.path) == 1
        }
        signed_flag = 1.0 if binary.learner.public_flag else -1.0
        self.assertTrue(
            torch.allclose(
                root_proposal.scores.root.feedback_context,
                signed_flag
                * (
                    proposal_children[(1,)].subtree_context
                    - proposal_children[(0,)].subtree_context
                ),
            )
        )
        self.assertGreater(
            float(root_proposal.scores.root.feedback_context.abs().sum().item()),
            0.0,
        )

        with mock.patch.object(
            self.policy.stable_compiler.core,
            "predict_effects",
            wraps=self.policy.stable_compiler.core.predict_effects,
        ) as effect_model:
            ordinary = self.policy.score_task(binary.learner, state)
        self.assertEqual(effect_model.call_count, 8)
        self.assertTrue(
            torch.equal(
                ordinary.memory_bias,
                torch.zeros_like(ordinary.memory_bias),
            )
        )
        self.assertGreater(float(ordinary.composition_logits.abs().sum().item()), 0.0)
        self.assertGreater(float(ordinary.phase4_bridge_logits.abs().sum().item()), 0.0)
        self.assertTrue(
            torch.equal(
                ordinary.composition_logits,
                ordinary.binary_policy_logits,
            )
        )
        self.assertEqual(ordinary.phase4_reliability.shape, (1, 120))
        self.assertEqual(ordinary.phase4_direction_gains.shape, (1, 120, 2))
        self.assertTrue(
            torch.equal(
                ordinary.phase4_direction_gains,
                torch.ones_like(ordinary.phase4_direction_gains),
            )
        )
        self.assertTrue(
            torch.equal(
                ordinary.phase4_reliability,
                torch.ones_like(ordinary.phase4_reliability),
            )
        )
        compiler_source = self.policy.compiler_source_bridge(
            ordinary.root.state_embedding
        )
        for call in effect_model.call_args_list[:4]:
            self.assertFalse(call.kwargs["reverse"])
            self.assertTrue(
                torch.allclose(
                    call.args[0],
                    compiler_source,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        successor_candidates = self.policy.compiler_successor_bridge(
            ordinary.root.candidate_embeddings[0]
        )
        for call in effect_model.call_args_list[4:]:
            self.assertTrue(call.kwargs["reverse"])
            self.assertTrue(
                torch.allclose(
                    call.args[0],
                    successor_candidates,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        self.assertGreater(
            float(ordinary.phase4_reverse_evidence.abs().sum().item()),
            0.0,
        )
        child_nodes = {
            node.path: node for node in ordinary.nodes if len(node.path) == 1
        }
        branch_weights = torch.softmax(
            self.policy.relational_branch_router(
                ordinary.root.memory_read.plastic_context,
                binary.learner.public_flag,
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(ordinary.root.branch_weights, branch_weights))
        self.assertTrue(
            torch.equal(
                ordinary.root.executed_branch_weights,
                branch_weights,
            )
        )
        left_successor = compiler_source + child_nodes[(0,)].subtree_context
        right_successor = compiler_source + child_nodes[(1,)].subtree_context
        expected_predecessor = left_successor + (
            ordinary.root.executed_branch_weights[:, 1:2]
            * (right_successor - left_successor)
        )
        self.assertTrue(
            torch.allclose(
                ordinary.root.recursive_predecessor,
                expected_predecessor,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                ordinary.root.subtree_context,
                ordinary.root.recursive_predecessor - compiler_source,
                atol=1e-6,
                rtol=1e-5,
            )
        )

        swapped_request = suite.PublicSkillExpression(
            binary.learner.request.symbol,
            tuple(reversed(binary.learner.request.children)),
        )
        swapped_task = suite.PublicSkillMemoryTask(
            binary.learner.items,
            binary.learner.public_flag,
            swapped_request,
        )
        swapped = self.policy.score_task(swapped_task, state)
        removed = self.policy.score_task(
            binary.learner,
            state,
            include_compiler=False,
        )
        phase4_removed = self.policy.score_task(
            binary.learner,
            state,
            include_phase4_bridge=False,
        )
        # With a fresh zero axis, no acquired reward covariance yet prefers a
        # side, so swapping alternatives is correctly symmetric.  The router
        # unit test separately proves signed order sensitivity once polarity
        # evidence is nonzero.
        self.assertTrue(torch.allclose(ordinary.logits, swapped.logits))
        self.assertFalse(torch.allclose(ordinary.logits, removed.logits))
        self.assertTrue(
            torch.allclose(
                ordinary.logits,
                removed.logits + ordinary.composition_logits,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        # Removing Phase-4 evidence must not remove the independently learned
        # binary selector or its acquired leaf policies.
        self.assertTrue(torch.allclose(phase4_removed.logits, ordinary.logits))
        self.assertTrue(
            torch.equal(
                phase4_removed.root.branch_weights,
                ordinary.root.branch_weights,
            )
        )
        self.assertTrue(
            torch.equal(
                phase4_removed.root.executed_branch_weights,
                ordinary.root.executed_branch_weights,
            )
        )
        self.assertTrue(
            torch.equal(
                phase4_removed.root.child_candidate_scores,
                ordinary.root.child_candidate_scores,
            )
        )
        self.assertTrue(
            torch.equal(
                phase4_removed.binary_policy_logits,
                ordinary.binary_policy_logits,
            )
        )
        for evidence in (
            phase4_removed.phase4_bridge_logits,
            phase4_removed.phase4_forward_evidence,
            phase4_removed.phase4_reverse_evidence,
        ):
            self.assertTrue(torch.equal(evidence, torch.zeros_like(evidence)))
        self.assertTrue(
            torch.equal(
                phase4_removed.phase4_direction_gains,
                torch.ones_like(phase4_removed.phase4_direction_gains),
            )
        )

        compiler_before = runner.reasoning_state_digest(
            self.policy.stable_compiler
        )
        self.policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.policy.relational_branch_router.condition_axis.fill_(0.1)
        detached = runner._detached_state(state)
        trained_scores = self.policy.score_task(binary.learner, detached)
        self.assertFalse(bool(trained_scores.root.execution_tied.item()))
        child_contrast = (
            trained_scores.root.child_candidate_scores[:, 1]
            - trained_scores.root.child_candidate_scores[:, 0]
        ).abs()
        contrast_index = int(child_contrast.argmax(dim=-1).item())
        loss = runner._outer_logits_loss(
            trained_scores.logits,
            binary,
        ) + runner._outer_top_target_loss(
            trained_scores.phase4_bridge_logits,
            binary,
        ) + 0.01 * trained_scores.binary_policy_logits[0, contrast_index]
        loss.backward()
        self.assertIsNotNone(
            self.policy.relational_branch_router.condition_axis.grad
        )
        self.assertGreater(
            float(
                self.policy.relational_branch_router.condition_axis.grad.abs()
                .sum()
                .item()
            ),
            0.0,
        )
        for label, module in (
            ("direction_mixer", self.policy.phase4_direction_mixer),
            ("source_bridge", self.policy.compiler_source_bridge),
            ("operator_bridge", self.policy.compiler_operator_bridge),
            ("successor_bridge", self.policy.compiler_successor_bridge),
            ("composition_items", self.policy.composition_item_encoder),
            ("composition_candidates", self.policy.composition_candidate_encoder),
            ("composition_state", self.policy.composition_state_encoder),
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            with self.subTest(module=label):
                self.assertTrue(gradients)
                self.assertTrue(
                    any(
                        gradient is not None
                        and bool(torch.isfinite(gradient).all().item())
                        and float(gradient.abs().sum().item()) > 0.0
                        for gradient in gradients
                    )
                )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in self.policy.composition_goal_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad and parameter.grad is None
                for parameter in self.policy.stable_compiler.parameters()
            )
        )
        self.assertEqual(
            runner.reasoning_state_digest(self.policy.stable_compiler),
            compiler_before,
        )
        self.assertTrue(torch.equal(removed.logits, torch.zeros_like(removed.logits)))

        zeroed_state = runner.zero_procedural_skill_content(state)
        zeroed_scores = self.policy.score_task(binary.learner, zeroed_state)
        self.assertTrue(
            torch.equal(
                zeroed_scores.root_context,
                torch.zeros_like(zeroed_scores.root_context),
            )
        )
        self.assertTrue(
            torch.equal(
                zeroed_scores.logits,
                torch.zeros_like(zeroed_scores.logits),
            )
        )

        # A learned reliability value can only scale Phase-4 evidence.  Even
        # an adversarial non-neutral gate cannot emit its own candidate score.
        with torch.no_grad():
            self.policy.phase4_reliability_gate[-1].bias.fill_(20.0)
        amplified = self.policy.score_task(binary.learner, state)
        no_phase4 = self.policy.score_task(
            binary.learner,
            state,
            include_phase4_bridge=False,
        )
        self.assertTrue(
            bool((amplified.phase4_reliability >= 0.0).all().item())
        )
        self.assertTrue(
            bool((amplified.phase4_reliability <= 2.0).all().item())
        )
        self.assertTrue(
            torch.allclose(
                amplified.logits,
                amplified.binary_policy_logits,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(no_phase4.logits, no_phase4.binary_policy_logits)
        )
        self.assertTrue(
            torch.equal(
                no_phase4.composition_logits,
                no_phase4.binary_policy_logits,
            )
        )
        self.assertGreater(float(no_phase4.logits.abs().sum().item()), 0.0)

    def test_atomic_bridge_probe_aligns_geometry_without_changing_leaf_policy(self) -> None:
        pair = next(
            pair for pair in self.partition.tasks if not pair.learner.request.children
        )
        state = self.policy.initial_state(1)
        proposal = runner.propose_task(
            self.policy,
            pair.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        reward = suite.score_skill_memory_answer(
            pair.learner,
            pair.hidden,
            proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            proposal,
            reward,
            state,
        ).candidate_state

        ordinary = self.policy.score_task(pair.learner, state)
        probe = self.policy.score_task(
            pair.learner,
            state,
            probe_leaf_bridge=True,
        )

        self.assertTrue(torch.equal(ordinary.logits, probe.logits))
        self.assertTrue(torch.equal(probe.logits, probe.memory_bias))
        self.assertTrue(
            torch.equal(
                ordinary.phase4_bridge_logits,
                torch.zeros_like(ordinary.phase4_bridge_logits),
            )
        )
        self.assertGreater(float(probe.phase4_bridge_logits.abs().sum().item()), 0.0)
        self.assertGreater(float(probe.phase4_forward_evidence.abs().sum().item()), 0.0)
        self.assertGreater(float(probe.phase4_reverse_evidence.abs().sum().item()), 0.0)
        self.assertEqual(float(probe.root_available.item()), 1.0)

        erased = runner.zero_procedural_skill_content(state)
        erased_probe = self.policy.score_task(
            pair.learner,
            erased,
            probe_leaf_bridge=True,
        )
        self.assertTrue(
            torch.equal(
                erased_probe.phase4_forward_evidence,
                torch.zeros_like(erased_probe.phase4_forward_evidence),
            )
        )
        self.assertTrue(
            torch.equal(
                erased_probe.phase4_reverse_evidence,
                torch.zeros_like(erased_probe.phase4_reverse_evidence),
            )
        )
        self.assertTrue(
            torch.equal(
                erased_probe.phase4_bridge_logits,
                torch.zeros_like(erased_probe.phase4_bridge_logits),
            )
        )
        self.assertTrue(
            torch.equal(
                erased_probe.composition_logits,
                torch.zeros_like(erased_probe.composition_logits),
            )
        )

    def test_recursive_unary_hands_each_successor_to_its_parent(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_107,
            instances_per_program=8,
        )
        by_program = {}
        for pair in partition.tasks:
            by_program.setdefault(pair.hidden.program.canonical, pair)
        target = by_program["GROUP_01(ROTATE(A_DESC))"]
        state = self.policy.initial_state(1)
        for canonical in (
            "A_DESC",
            "ROTATE(A_DESC)",
            "GROUP_01(ROTATE(A_DESC))",
        ):
            support = by_program[canonical]
            proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                proposal.answer,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                reward,
                state,
            ).candidate_state

        with mock.patch.object(
            self.policy.stable_compiler.core,
            "predict_effects",
            wraps=self.policy.stable_compiler.core.predict_effects,
        ) as effect_model:
            scores = self.policy.score_task(target.learner, state)
        self.assertEqual(effect_model.call_count, 12)
        nodes = {node.path: node for node in scores.nodes}
        source = self.policy.compiler_source_bridge(scores.root.state_embedding)
        leaf_successor = source + nodes[(0, 0)].subtree_context
        middle_successor = source + nodes[(0,)].subtree_context
        anchored_leaf = nodes[(0,)].recursive_predecessor
        anchored_middle = scores.root.recursive_predecessor
        self.assertTrue(bool(torch.isfinite(anchored_leaf).all().item()))
        self.assertTrue(bool(torch.isfinite(anchored_middle).all().item()))
        self.assertFalse(torch.allclose(anchored_leaf, leaf_successor))
        self.assertFalse(torch.allclose(anchored_middle, middle_successor))
        expected_sources = (
            source,
            source,
            anchored_leaf,
            anchored_leaf,
            anchored_middle,
            anchored_middle,
        )
        for call, expected in zip(
            effect_model.call_args_list[:6],
            expected_sources,
            strict=True,
        ):
            self.assertFalse(call.kwargs["reverse"])
            self.assertTrue(
                torch.allclose(call.args[0], expected, atol=1e-6, rtol=1e-5)
            )
        reverse_calls = effect_model.call_args_list[6:]
        self.assertTrue(all(call.kwargs["reverse"] for call in reverse_calls))
        successor_candidates = self.policy.compiler_successor_bridge(
            scores.root.candidate_embeddings[0]
        )
        self.assertTrue(
            torch.allclose(
                reverse_calls[0].args[0],
                successor_candidates,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                reverse_calls[1].args[0],
                successor_candidates,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        for call_index, path in zip((0, 2, 4), ((), (0,), (0, 0)), strict=True):
            expected_operator = self.policy.compiler_operator_bridge(
                nodes[path].memory_read.plastic_context
            )
            self.assertTrue(
                torch.allclose(
                    reverse_calls[call_index].args[1],
                    expected_operator,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        self.assertGreater(
            float(scores.phase4_reverse_evidence.abs().sum().item()),
            0.0,
        )

        # The bounded harmonization optimizer reaches both calibration groups
        # through an actually executed depth-2 policy and nothing consolidated.
        self.policy.zero_grad(set_to_none=True)
        runner._configure_stage_trainability(self.policy, "harmonization")
        trained_scores = self.policy.score_task(
            target.learner,
            runner._detached_state(state),
        )
        loss = runner._outer_logits_loss(
            trained_scores.logits,
            target,
        ) + runner._outer_top_target_loss(
            trained_scores.phase4_bridge_logits,
            target,
        )
        loss.backward()
        for label, module in (
            ("direction_mixer", self.policy.phase4_direction_mixer),
        ):
            gradients = [parameter.grad for parameter in module.parameters()]
            with self.subTest(trainable=label):
                self.assertTrue(
                    any(
                        gradient is not None
                        and bool(torch.isfinite(gradient).all().item())
                        and float(gradient.abs().sum().item()) > 0.0
                        for gradient in gradients
                    )
                )
        for label, module in (
            ("compiler", self.policy.stable_compiler),
            ("source_bridge", self.policy.compiler_source_bridge),
            ("operator_bridge", self.policy.compiler_operator_bridge),
            ("successor_bridge", self.policy.compiler_successor_bridge),
            ("reliability", self.policy.phase4_reliability_gate),
            ("leaf_memory", self.policy.memory),
            ("composition_memory", self.policy.composition_memory),
            ("relational_axis", self.policy.relational_branch_router),
        ):
            with self.subTest(frozen=label):
                self.assertTrue(
                    all(parameter.grad is None for parameter in module.parameters())
                )

    def test_operator_audit_reconstructs_live_unary_forward_evidence(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_107,
            instances_per_program=8,
        )
        by_program = {}
        for pair in partition.tasks:
            by_program.setdefault(pair.hidden.program.canonical, pair)
        target = by_program["GROUP_01(ROTATE(A_DESC))"]
        state = self.policy.initial_state(1)
        for canonical in (
            "A_DESC",
            "ROTATE(A_DESC)",
            "GROUP_01(ROTATE(A_DESC))",
        ):
            support = by_program[canonical]
            proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                proposal.answer,
            )
            state = runner._detached_state(
                runner.propose_differentiable_feedback(
                    self.policy,
                    proposal,
                    reward,
                    state,
                ).candidate_state
            )

        scores = self.policy.score_task(target.learner, state)
        frozen_only = self.policy.score_task(
            target.learner,
            state,
            include_fast_adapter=False,
        )
        reconstructed = runner._root_operator_forward_evidence(
            self.policy,
            scores,
            scores.root.memory_read.plastic_context,
        )

        self.assertEqual(float(scores.root_available.item()), 1.0)
        for left, right in (
            (scores.logits, frozen_only.logits),
            (scores.phase4_bridge_logits, frozen_only.phase4_bridge_logits),
            (scores.phase4_forward_evidence, frozen_only.phase4_forward_evidence),
            (scores.phase4_reverse_evidence, frozen_only.phase4_reverse_evidence),
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            torch.allclose(
                reconstructed.unsqueeze(0),
                scores.phase4_forward_evidence,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_operator_audit_cohorts_and_statistics_are_predeclared(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_131,
            instances_per_program=16,
        )
        groups = runner._group_evaluator_pairs(partition.tasks)
        queries = runner._operator_audit_queries(groups)
        cohorts = {
            cohort
            for pair in queries
            if (cohort := runner._operator_audit_cohort(pair)) is not None
        }
        self.assertEqual(cohorts, set(runner._OPERATOR_AUDIT_COHORTS))
        self.assertTrue(
            all(
                runner._operator_audit_cohort(pair) is None
                for pair in queries
                if len(pair.hidden.program.children) == 2
            )
        )

        evidence = torch.linspace(-1.0, 1.0, 120)
        utilities = torch.linspace(0.0, 1.0, 120).unsqueeze(0)
        aligned = runner._operator_audit_alignment(evidence, utilities)
        inverted = runner._operator_audit_alignment(-evidence, utilities)
        degenerate = runner._operator_audit_alignment(
            torch.zeros_like(evidence), utilities
        )
        self.assertIsNotNone(aligned)
        self.assertIsNotNone(inverted)
        assert aligned is not None and inverted is not None
        self.assertAlmostEqual(aligned["correlation"], 1.0, places=12)
        self.assertAlmostEqual(inverted["correlation"], -1.0, places=12)
        self.assertGreater(aligned["covariance"], 0.0)
        self.assertIsNone(degenerate)

    def test_operator_audit_gradient_summary_has_fixed_eight_seed_gate(self) -> None:
        seeds = range(8)
        coherent = {
            seed: torch.tensor((1.0, 0.25, -0.5)) for seed in seeds
        }
        report = runner._bridge_gradient_summary(coherent)
        self.assertTrue(report["passed"])
        self.assertEqual(report["pairwise_count"], 28)
        self.assertEqual(report["positive_pairwise_count"], 28)
        self.assertEqual(report["positive_leave_one_seed_out_count"], 8)

        incoherent = {
            seed: torch.eye(8, dtype=torch.float32)[seed] for seed in seeds
        }
        failed = runner._bridge_gradient_summary(incoherent)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["positive_pairwise_count"], 0)

    def test_binary_parent_executes_complete_unary_child_policies(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_107,
            instances_per_program=8,
        )
        by_program = {}
        for pair in partition.tasks:
            by_program.setdefault(pair.hidden.program.canonical, pair)
        target = by_program[
            "IF_FLAG(GROUP_01(A_ASC),ROTATE(B_ASC))"
        ]
        support_names = (
            "A_ASC",
            "B_ASC",
            "GROUP_01(A_ASC)",
            "ROTATE(B_ASC)",
            "IF_FLAG(GROUP_01(A_ASC),ROTATE(B_ASC))",
        )
        state = self.policy.initial_state(1)
        for canonical in support_names:
            support = by_program[canonical]
            proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                proposal.answer,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                reward,
                state,
            ).candidate_state

        scores = self.policy.score_task(target.learner, state)
        self.assertEqual(float(scores.root_available.item()), 1.0)
        child_nodes = {
            node.path: node for node in scores.nodes if len(node.path) == 1
        }
        child_standalones = []
        child_standalones_without_phase4 = []
        for index, child in enumerate(target.learner.request.children):
            child_task = suite.PublicSkillMemoryTask(
                target.learner.items,
                target.learner.public_flag,
                child,
            )
            standalone = self.policy.score_task(child_task, state)
            child_standalones.append(standalone)
            self.assertGreater(
                float(standalone.phase4_bridge_logits.abs().sum().item()),
                0.0,
            )
            self.assertTrue(
                torch.allclose(
                    scores.root.child_candidate_scores[:, index],
                    standalone.logits,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
            without_phase4 = self.policy.score_task(
                child_task,
                state,
                include_phase4_bridge=False,
            )
            child_standalones_without_phase4.append(without_phase4)

        self.assertTrue(
            any(
                not torch.allclose(
                    scores.root.child_candidate_scores[:, index],
                    child_nodes[(index,)].memory_read.score_bias,
                )
                for index in range(2)
            )
        )

        without_phase4 = self.policy.score_task(
            target.learner,
            state,
            include_phase4_bridge=False,
        )
        self.assertTrue(
            torch.equal(
                without_phase4.root.branch_weights,
                scores.root.branch_weights,
            )
        )
        self.assertTrue(
            torch.equal(
                without_phase4.root.executed_branch_weights,
                scores.root.executed_branch_weights,
            )
        )
        for index, standalone in enumerate(child_standalones_without_phase4):
            self.assertTrue(
                torch.allclose(
                    without_phase4.root.child_candidate_scores[:, index],
                    standalone.logits,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        self.assertGreater(
            float(without_phase4.binary_policy_logits.abs().sum().item()),
            0.0,
        )
        for evidence in (
            without_phase4.phase4_bridge_logits,
            without_phase4.phase4_forward_evidence,
            without_phase4.phase4_reverse_evidence,
        ):
            self.assertTrue(torch.equal(evidence, torch.zeros_like(evidence)))

    def test_nested_binary_child_never_reopens_its_generic_decoder(self) -> None:
        partition = suite.make_skill_memory_meta_partition(
            85_107,
            instances_per_program=8,
        )
        by_program = {}
        for pair in partition.tasks:
            by_program.setdefault(pair.hidden.program.canonical, pair)
        outer_template = by_program["IF_FLAG(A_ASC,B_ASC)"]
        inner = by_program["IF_NOT_FLAG(B_ASC,A_ASC)"]
        state = self.policy.initial_state(1)
        for canonical in (
            "A_ASC",
            "B_ASC",
            "A_DESC",
            "IF_NOT_FLAG(B_ASC,A_ASC)",
            "IF_FLAG(A_ASC,B_ASC)",
        ):
            support = by_program[canonical]
            proposal = runner.propose_task(
                self.policy,
                support.learner,
                state,
                greedy=False,
                temperature=1.0,
            )
            reward = suite.score_skill_memory_answer(
                support.learner,
                support.hidden,
                proposal.answer,
            )
            state = runner.propose_differentiable_feedback(
                self.policy,
                proposal,
                reward,
                state,
            ).candidate_state

        leaf = by_program["A_ASC"].learner.request
        outer_request = suite.PublicSkillExpression(
            outer_template.learner.request.symbol,
            (inner.learner.request, leaf),
        )
        outer_task = suite.PublicSkillMemoryTask(
            outer_template.learner.items,
            outer_template.learner.public_flag,
            outer_request,
        )
        standalone_inner_task = suite.PublicSkillMemoryTask(
            outer_task.items,
            outer_task.public_flag,
            inner.learner.request,
        )
        outer = self.policy.score_task(outer_task, state)
        standalone_inner = self.policy.score_task(standalone_inner_task, state)
        nested_binary = next(node for node in outer.nodes if node.path == (0,))

        self.assertEqual(float(outer.root_available.item()), 1.0)
        self.assertTrue(
            torch.allclose(
                outer.root.child_candidate_scores[:, 0],
                standalone_inner.logits,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(
                standalone_inner.memory_bias,
                torch.zeros_like(standalone_inner.memory_bias),
            )
        )
        self.assertTrue(
            torch.equal(
                standalone_inner.logits,
                standalone_inner.binary_policy_logits,
            )
        )
        self.assertFalse(
            torch.allclose(
                outer.root.child_candidate_scores[:, 0],
                nested_binary.memory_read.score_bias,
            )
        )
        self.assertGreater(
            float(nested_binary.child_candidate_scores.abs().sum().item()),
            0.0,
        )

        swapped_task = suite.PublicSkillMemoryTask(
            outer_task.items,
            outer_task.public_flag,
            suite.PublicSkillExpression(
                outer_request.symbol,
                tuple(reversed(outer_request.children)),
            ),
        )
        swapped = self.policy.score_task(swapped_task, state)
        self.assertTrue(
            torch.allclose(
                swapped.root.child_candidate_scores,
                torch.flip(outer.root.child_candidate_scores, dims=(1,)),
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(bool(outer.root.execution_tied.item()))
        self.assertTrue(bool(swapped.root.execution_tied.item()))
        self.assertTrue(
            torch.allclose(outer.logits, swapped.logits, atol=1e-6, rtol=1e-5)
        )

    def test_intermediate_reanchoring_learns_with_shared_candidate_geometry(self) -> None:
        source = torch.randn(1, self.profile.width, requires_grad=True)
        successor = torch.randn(1, self.profile.width, requires_grad=True)
        candidate_states = torch.randn(
            len(runner._PERMUTATIONS),
            self.profile.width,
            requires_grad=True,
        )

        anchored = runner._soft_reanchor_intermediate(
            source,
            successor,
            candidate_states,
        )
        anchored.square().mean().backward()

        self.assertEqual(anchored.shape, successor.shape)
        self.assertTrue(bool(torch.isfinite(anchored).all().item()))
        self.assertIsNotNone(successor.grad)
        self.assertGreater(float(successor.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(candidate_states.grad)
        self.assertGreater(float(candidate_states.grad.abs().sum().item()), 0.0)

        unchanged = runner._soft_reanchor_intermediate(
            source.detach(),
            source.detach(),
            candidate_states,
        )
        self.assertTrue(torch.equal(unchanged, source.detach()))

    def test_evaluation_stage_seed_is_independent_of_prior_rng_use(self) -> None:
        device = torch.device("cpu")
        first_seed = runner._seed_reproducible_stage(85_108, "online", device)
        first = (random.random(), torch.rand(4))
        random.random()
        torch.rand(17)

        second_seed = runner._seed_reproducible_stage(85_108, "online", device)
        second = (random.random(), torch.rand(4))
        other_seed = runner._seed_reproducible_stage(85_108, "leaf", device)

        self.assertEqual(first_seed, second_seed)
        self.assertEqual(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertNotEqual(first_seed, other_seed)

    def test_composition_requires_root_and_every_child(self) -> None:
        binary = next(
            pair
            for pair in self.partition.tasks
            if len(pair.learner.request.children) == 2
        )
        leaf_by_symbol = {
            pair.learner.request.symbol: pair
            for pair in self.partition.tasks
            if not pair.learner.request.children
        }
        state = self.policy.initial_state(1)
        initial_digest = procedural_skill_state_digest(state)

        root_proposal = runner.propose_task(self.policy, binary.learner, state)
        root_reward = suite.score_skill_memory_answer(
            binary.learner,
            binary.hidden,
            root_proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            root_proposal,
            root_reward,
            state,
        ).candidate_state
        self.assertEqual(procedural_skill_state_digest(state), initial_digest)
        root_only = self.policy.score_task(binary.learner, state)
        root_only_ablated = self.policy.score_task(
            binary.learner,
            state,
            include_descendants=False,
        )
        self.assertEqual(float(root_only.root_available.item()), 0.0)
        self.assertEqual(float(root_only_ablated.root_available.item()), 0.0)
        self.assertTrue(
            torch.equal(
                root_only.composition_logits,
                torch.zeros_like(root_only.composition_logits),
            )
        )

        first_child = leaf_by_symbol[binary.learner.request.children[0].symbol]
        child_proposal = runner.propose_task(self.policy, first_child.learner, state)
        child_reward = suite.score_skill_memory_answer(
            first_child.learner,
            first_child.hidden,
            child_proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            child_proposal,
            child_reward,
            state,
        ).candidate_state
        one_child = self.policy.score_task(binary.learner, state)
        self.assertEqual(float(one_child.root_available.item()), 0.0)
        self.assertTrue(
            torch.equal(
                one_child.composition_logits,
                torch.zeros_like(one_child.composition_logits),
            )
        )

        second_child = leaf_by_symbol[binary.learner.request.children[1].symbol]
        child_proposal = runner.propose_task(self.policy, second_child.learner, state)
        child_reward = suite.score_skill_memory_answer(
            second_child.learner,
            second_child.hidden,
            child_proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            child_proposal,
            child_reward,
            state,
        ).candidate_state
        children_ready = self.policy.score_task(binary.learner, state)
        self.assertEqual(float(children_ready.root_available.item()), 0.0)
        self.assertTrue(bool(children_ready.root.feedback_available.item()))
        root_proposal = runner.propose_task(
            self.policy,
            binary.learner,
            state,
            greedy=False,
            temperature=1.0,
        )
        root_reward = suite.score_skill_memory_answer(
            binary.learner,
            binary.hidden,
            root_proposal.answer,
        )
        state = runner.propose_differentiable_feedback(
            self.policy,
            root_proposal,
            root_reward,
            state,
        ).candidate_state
        complete = self.policy.score_task(binary.learner, state)
        ablated = self.policy.score_task(
            binary.learner,
            state,
            include_descendants=False,
        )
        self.assertEqual(float(complete.root_available.item()), 1.0)
        self.assertEqual(float(ablated.root_available.item()), 1.0)
        self.assertGreater(float(complete.composition_logits.abs().sum().item()), 0.0)
        self.assertTrue(
            torch.allclose(
                ablated.root.recursive_predecessor,
                self.policy.compiler_source_bridge(ablated.root.state_embedding),
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            all(
                torch.equal(
                    node.subtree_context,
                    torch.zeros_like(node.subtree_context),
                )
                for node in ablated.nodes
                if node.path
            )
        )

    def test_final_loader_is_blocked_until_every_slow_weight_is_frozen(self) -> None:
        with mock.patch.object(
            runner,
            "_load_final_curriculum",
            side_effect=AssertionError("final evaluator loaded early"),
        ) as loader:
            with self.assertRaisesRegex(RuntimeError, "freeze"):
                runner._evaluate(self.policy, self.profile, 85_103)
            loader.assert_not_called()

    def test_leaf_core_stage_never_loads_integrated_curriculum(self) -> None:
        with mock.patch.object(
            runner,
            "_load_final_curriculum",
            side_effect=AssertionError("integrated evaluator loaded in leaf stage"),
        ) as loader:
            result = runner.run(
                "smoke",
                seed=85_103,
                device="cpu",
                stage="leaf_core",
            )

        loader.assert_not_called()
        self.assertEqual(result["stage"], "leaf_core")
        self.assertEqual(result["training"]["training_stage"], "leaf_core")
        self.assertEqual(result["online"]["stage"], "leaf_core")
        self.assertFalse(result["online"]["final_partition_loaded_after_freeze"])
        self.assertEqual(set(result["online"]["roots"]), {"A", "B", "C", "D"})

    def test_relational_acquisition_run_preserves_all_preexisting_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial checkpoint"):
            runner.run(
                "smoke",
                seed=85_109,
                device="cpu",
                stage="relational_acquisition",
            )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "integrated.pt"
            stable_compiler, _ = runner._load_phase4_compiler(
                runner._PHASE4_CHECKPOINT
            )
            source = runner.SkillMemoryPolicy(self.profile, stable_compiler)
            payload = {
                "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                "model": source.state_dict(),
                "profile": asdict(self.profile),
                "result_digest": "sha256:" + "e" * 64,
                "runner": runner._REPORT_VERSION,
                "stage": "leaf_core",
            }
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "integrated"):
                runner.run(
                    "smoke",
                    seed=85_109,
                    device="cpu",
                    stage="relational_acquisition",
                    initial_checkpoint=checkpoint,
                )
            payload["stage"] = "integrated"
            torch.save(payload, checkpoint)
            result = runner.run(
                "smoke",
                seed=85_109,
                device="cpu",
                stage="relational_acquisition",
                initial_checkpoint=checkpoint,
            )
        training = result["training"]

        self.assertEqual(training["training_stage"], "relational_acquisition")
        self.assertTrue(training["preexisting_state_consolidated"])
        self.assertEqual(
            training["preexisting_state_fingerprint_before"],
            training["preexisting_state_fingerprint_after"],
        )
        self.assertNotEqual(
            training["relational_acquisition_fingerprint_before"],
            training["relational_acquisition_fingerprint_after"],
        )
        self.assertTrue(training["trainable_parameter_names"])
        self.assertTrue(
            all(
                name.startswith(runner._RELATIONAL_ACQUISITION_PREFIXES)
                for name in training["trainable_parameter_names"]
            )
        )

    def test_harmonization_run_preserves_axis_memory_and_all_outside_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial checkpoint"):
            runner.run(
                "smoke",
                seed=85_110,
                device="cpu",
                stage="harmonization",
            )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "relational.pt"
            stable_compiler, _ = runner._load_phase4_compiler(
                runner._PHASE4_CHECKPOINT
            )
            source = runner.SkillMemoryPolicy(self.profile, stable_compiler)
            source_state = {
                key: value
                for key, value in source.state_dict().items()
                if not key.startswith("phase4_direction_mixer.")
                and not key.startswith("procedural_fast_adapter.")
                and not key.startswith("procedural_goal_projection.")
                and not _is_reversible_state_key(key)
            }
            torch.save(
                {
                    "compiler_checkpoint_sha256": runner._PHASE4_CHECKPOINT_SHA256,
                    "model": source_state,
                    "profile": asdict(self.profile),
                    "result_digest": "sha256:" + "f" * 64,
                    "runner": "angler.phase5-skill-memory-stream.v13",
                    "stage": "relational_acquisition",
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "exact retained v41"):
                runner.run(
                    "smoke",
                    seed=85_110,
                    device="cpu",
                    stage="harmonization",
                    initial_checkpoint=checkpoint,
                )
            checkpoint_digest = runner.hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            with (
                mock.patch.object(
                    runner,
                    "_HARMONIZATION_SOURCE_CHECKPOINT_SHA256",
                    checkpoint_digest,
                ),
                mock.patch.object(
                    runner,
                    "_outer_matched_descendant_loss",
                    wraps=runner._outer_matched_descendant_loss,
                ) as matched_loss,
            ):
                result = runner.run(
                    "smoke",
                    seed=85_110,
                    device="cpu",
                    stage="harmonization",
                    initial_checkpoint=checkpoint,
                )
            self.assertTrue(matched_loss.call_args_list)
            self.assertTrue(
                all(
                    call.kwargs["include_evidence_delta"] is False
                    for call in matched_loss.call_args_list
                )
            )
        training = result["training"]

        self.assertEqual(training["training_stage"], "harmonization")
        self.assertEqual(training["outer_steps"], 1)
        self.assertEqual(training["phase4_direction_weight"], 0.0)
        self.assertEqual(training["local_phase4_alignment_weight"], 0.0)
        self.assertEqual(training["feedback_causal_weight"], 0.0)
        self.assertEqual(training["support_consistency_weight"], 0.0)
        self.assertEqual(training["route_balance_weight"], 0.0)
        self.assertEqual(training["phase4_residual_weight"], 0.5)
        self.assertEqual(training["phase4_residual_root_arities"], [1])
        self.assertFalse(training["matched_evidence_delta_is_objective"])
        self.assertEqual(
            training["outside_harmonization_fingerprint_before"],
            training["outside_harmonization_fingerprint_after"],
        )
        self.assertNotEqual(
            training["harmonization_fingerprint_before"],
            training["harmonization_fingerprint_after"],
        )
        self.assertEqual(
            training["condition_axis_fingerprint_before"],
            training["condition_axis_fingerprint_after"],
        )
        self.assertTrue(training["trainable_parameter_names"])
        self.assertTrue(
            all(
                name.startswith(runner._HARMONIZATION_TRAINABLE_PREFIXES)
                for name in training["trainable_parameter_names"]
            )
        )

    def test_smoke_run_reports_frozen_online_stream_and_controls(self) -> None:
        result = runner.run("smoke", seed=85_104, device="cpu")
        online = result["online"]

        self.assertEqual(result["candidate_count"], 120)
        self.assertEqual(
            result["compiler_checkpoint"]["sha256"],
            runner._PHASE4_CHECKPOINT_SHA256,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.policy.stable_compiler.parameters()
            )
        )
        self.assertEqual(online["phase_order"], ["A", "B", "C", "A", "D", "B"])
        self.assertEqual(online["ordinary_unique_presentations"], 20)
        self.assertEqual(online["disjoint_component_probe_count"], 20)
        self.assertEqual(online["transactions"], 20)
        self.assertEqual(online["online_replay_reads"], 0)
        self.assertEqual(online["history_retrievals"], 0)
        self.assertFalse(online["optimizer_reachable_online"])
        self.assertEqual(
            online["slow_fingerprint_before"], online["slow_fingerprint_after"]
        )
        self.assertEqual(
            online["slow_parameter_identity_before"],
            online["slow_parameter_identity_after"],
        )
        self.assertEqual(online["state_numel_initial"], online["state_numel_final"])
        self.assertEqual(online["composition"]["feedback_writes"], 0)
        self.assertEqual(
            online["composition"]["state_digest_before"],
            online["composition"]["state_digest_after"],
        )
        self.assertIn("cross_mechanism_collision_rate", online["slot_collisions"])
        self.assertEqual(
            result["training"]["compiler_fingerprint_before"],
            result["training"]["compiler_fingerprint_after"],
        )
        self.assertTrue(result["training"]["leaf_substrate_consolidated"])
        self.assertEqual(
            result["training"]["leaf_substrate_fingerprint_before"],
            result["training"]["leaf_substrate_fingerprint_after"],
        )
        self.assertTrue(result["training"]["trainable_parameter_names"])
        self.assertTrue(
            all(
                name.startswith(runner._COMPOSITION_TRAINABLE_PREFIXES)
                for name in result["training"]["trainable_parameter_names"]
            )
        )
        self.assertIn("composition_removed_mean", online["composition"])
        self.assertIn("phase4_removed_mean", online["composition"])
        self.assertIn("reverse_removed_mean", online["composition"])
        self.assertIn("root_only_mean", online["composition"])
        self.assertIn("full_root_availability", online["composition"])
        self.assertIn("root_only_availability", online["composition"])
        self.assertIn("equal_complete_availability", online["composition"])
        self.assertEqual(
            set(online["composition"]["by_depth"]),
            {"2", "3"},
        )
        self.assertEqual(
            set(online["composition"]["by_root_arity"]),
            {"1", "2"},
        )
        self.assertEqual(
            set(online["composition"]["binary_by_public_flag"]),
            {"0", "1"},
        )
        self.assertEqual(
            set(online["composition"]["binary_by_hidden_operator"]),
            {"IF_FLAG", "IF_NOT_FLAG"},
        )
        self.assertEqual(
            set(online["composition"]["binary_by_operator_and_flag"]),
            {
                "IF_FLAG:0",
                "IF_FLAG:1",
                "IF_NOT_FLAG:0",
                "IF_NOT_FLAG:1",
            },
        )
        branch_choice = online["composition"]["binary_branch_choice"]
        self.assertEqual(
            set(branch_choice["cells"]),
            {
                "IF_FLAG:0",
                "IF_FLAG:1",
                "IF_NOT_FLAG:0",
                "IF_NOT_FLAG:1",
            },
        )
        self.assertEqual(branch_choice["count"], 8)
        self.assertEqual(online["matched_binary_branch_probe_count"], 8)
        self.assertEqual(branch_choice["feedback_writes"], 0)
        self.assertEqual(
            branch_choice["state_digest_before"],
            branch_choice["state_digest_after"],
        )
        self.assertEqual(
            branch_choice["slow_fingerprint_before"],
            branch_choice["slow_fingerprint_after"],
        )
        self.assertGreaterEqual(branch_choice["hard_accuracy"], 0.0)
        self.assertLessEqual(branch_choice["hard_accuracy"], 1.0)
        for slices in (
            online["composition"]["by_depth"],
            online["composition"]["by_root_arity"],
        ):
            for metrics in slices.values():
                self.assertIn("phase4_gain", metrics)
                self.assertIn("descendant_gain", metrics)
        self.assertIn("criteria", online)
        self.assertGreaterEqual(online["rejection_rate"], 0.0)
        self.assertLessEqual(online["rejection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
