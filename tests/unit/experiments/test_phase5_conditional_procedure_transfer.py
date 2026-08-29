from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.evaluators.conditional_symbolic_procedure_transfer_suite import (
    conditional_mechanism_partition,
    make_conditional_procedure_transfer_stream,
)
from experiments.runners import phase5_conditional_procedure_transfer as runner
from experiments.runners import phase5_skill_memory_stream as phase5


class Phase5ConditionalProcedureTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_conditional_procedure_transfer_stream(
            97_101,
            supports_per_flag=1,
            queries_per_flag=1,
            mechanism_pair=conditional_mechanism_partition("train")[0],
            mechanism_partition="train",
        )

    def _reversible_policy(self) -> phase5.SkillMemoryPolicy:
        torch.manual_seed(97_104)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        policy.public_fact_adapter = runner.demonstration.TypedPublicFactPorts(
            runner.demonstration.SharedPublicFactAdapter(),
            runner.demonstration.SymbolicDemonstrationAdapter(),
        )
        reader = runner.demonstration._attach_public_evidence_reader(policy)
        with torch.no_grad():
            policy.reversible_transition_mode.fill_(True)
            policy.reversible_procedure_transition.first_up.weight.normal_(
                mean=0.0,
                std=0.05,
            )
            policy.reversible_procedure_transition.second_up.weight.normal_(
                mean=0.0,
                std=0.05,
            )
            reader.output.weight.normal_(mean=0.0, std=0.05)
            reader.transition_output.weight.normal_(mean=0.0, std=0.05)
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(
                runner.demonstration._is_demonstration_trainable(name)
            )
        return policy

    def _assert_skill_states_close(
        self,
        actual: object,
        expected: object,
        *,
        atol: float = 2.0e-3,
        rtol: float = 1.0e-4,
    ) -> None:
        self.assertIsInstance(actual, runner.ProceduralSkillState)
        self.assertIsInstance(expected, runner.ProceduralSkillState)
        actual_state = actual
        expected_state = expected
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            torch.testing.assert_close(
                getattr(actual_state.fast_weights, name),
                getattr(expected_state.fast_weights, name),
                atol=atol,
                rtol=rtol,
            )
        for name in ("slot_latents", "key_offsets"):
            torch.testing.assert_close(
                getattr(actual_state, name),
                getattr(expected_state, name),
                atol=atol,
                rtol=rtol,
            )
        self.assertTrue(torch.equal(actual_state.occupied, expected_state.occupied))
        self.assertTrue(
            torch.equal(actual_state.write_counts, expected_state.write_counts)
        )

    @staticmethod
    def _state_has_autograd(state: object) -> bool:
        if not isinstance(state, runner.ProceduralSkillState):
            return False
        values = (
            state.fast_weights.delta_y,
            state.fast_weights.delta_q,
            state.fast_weights.delta_k,
            state.fast_weights.delta_beta,
            state.slot_latents,
            state.key_offsets,
        )
        return any(value.requires_grad for value in values)

    def test_public_controls_change_only_demonstrations(self) -> None:
        task = self.stream.binding_supports[0].learner
        absent = runner._no_demonstration_task(task)
        wrong = runner._wrong_demonstration_task(task)

        for control in (absent, wrong):
            self.assertEqual(control.items, task.items)
            self.assertEqual(control.public_flag, task.public_flag)
            self.assertEqual(control.request, task.request)
        self.assertEqual(absent.demonstrations, ())
        self.assertNotEqual(wrong.demonstrations, task.demonstrations)
        for original, changed in zip(
            task.demonstrations,
            wrong.demonstrations,
            strict=True,
        ):
            self.assertEqual(changed.input_symbols, original.input_symbols)
            self.assertEqual(
                changed.output_symbols,
                original.output_symbols[1:] + original.output_symbols[:1],
            )

    def test_default_training_is_one_diverse_partition_pass(self) -> None:
        train = conditional_mechanism_partition("train")
        default = inspect.signature(runner.run).parameters["meta_steps"].default

        self.assertEqual(len(train), 512)
        self.assertEqual(default, len(train))
        self.assertEqual(runner._DEFAULT_META_STEPS, len(train))

    def test_every_proposal_requests_transition_only_composition(self) -> None:
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        state = policy.initial_state(1)
        task = self.stream.queries[0].learner
        original = policy.score_task
        calls: list[dict[str, object]] = []

        def recording_score(*args: object, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return original(*args, **kwargs)

        with patch.object(policy, "score_task", side_effect=recording_score):
            proposal = runner._transition_proposal(
                policy,
                task,
                state,
                greedy=True,
            )
            runner._transition_proposal(
                policy,
                task,
                state,
                greedy=True,
                include_reversible_transition=False,
            )

        self.assertEqual(len(proposal.answer), 5)
        self.assertTrue(all(call["transition_only_composition"] for call in calls))
        self.assertTrue(calls[0]["include_reversible_transition"])
        self.assertFalse(calls[1]["include_reversible_transition"])

    def test_validated_digest_path_is_identical_and_rejects_stale(self) -> None:
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        state = policy.initial_state(1)
        task = self.stream.anchor_supports[0].learner
        state_digest = runner.procedural_skill_state_digest(state)
        torch.manual_seed(97_102)
        sampling_state = torch.random.get_rng_state()

        legacy_proposal = runner._transition_proposal(
            policy,
            task,
            state,
            greedy=False,
        )
        torch.random.set_rng_state(sampling_state)
        optimized_proposal = runner._transition_proposal(
            policy,
            task,
            state,
            greedy=False,
            validated_state_digest=state_digest,
        )
        self.assertEqual(optimized_proposal.answer, legacy_proposal.answer)
        self.assertEqual(
            optimized_proposal.candidate_index,
            legacy_proposal.candidate_index,
        )
        self.assertEqual(
            optimized_proposal.competence_digest,
            legacy_proposal.competence_digest,
        )
        self.assertTrue(
            torch.equal(
                optimized_proposal.scores.logits,
                legacy_proposal.scores.logits,
            )
        )
        self.assertTrue(
            torch.equal(
                optimized_proposal.behavior_probabilities,
                legacy_proposal.behavior_probabilities,
            )
        )

        legacy_state, legacy_record = runner._commit_transition_feedback(
            policy,
            task,
            legacy_proposal,
            1.0,
            state,
        )
        optimized_state, optimized_record = runner._commit_transition_feedback(
            policy,
            task,
            optimized_proposal,
            1.0,
            state,
            validated_state_digest=state_digest,
        )
        self.assertEqual(optimized_record, legacy_record)
        self.assertEqual(
            runner.procedural_skill_state_digest(optimized_state),
            runner.procedural_skill_state_digest(legacy_state),
        )

        stale_digest = runner.procedural_skill_state_digest(optimized_state)
        self.assertNotEqual(stale_digest, state_digest)
        with self.assertRaisesRegex(ValueError, "bound"):
            runner._commit_transition_feedback(
                policy,
                task,
                optimized_proposal,
                1.0,
                optimized_state,
                validated_state_digest=stale_digest,
            )

    def test_matched_stage_calls_scalar_scorer_once_per_public_attempt(self) -> None:
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        states = (
            policy.initial_state(1),
            policy.initial_state(1),
            policy.initial_state(1),
        )
        observed: list[object] = []

        def scalar_only(public: object, hidden: object, answer: object) -> float:
            observed.append(hidden)
            return 0.5

        _, record = runner._acquire_matched_stage(
            policy,
            states,
            self.stream.anchor_supports,
            judge=scalar_only,
        )

        self.assertEqual(len(observed), len(self.stream.anchor_supports))
        self.assertEqual(record["scalar_evaluator_calls"], len(observed))
        self.assertTrue(record["matched_candidate"])
        self.assertTrue(record["matched_scalar_reused_across_controls"])

    def test_matched_acquisition_scores_match_leaf_unary_and_binary_rows(
        self,
    ) -> None:
        policy = self._reversible_policy()
        states = tuple(policy.initial_state(1) for _ in range(3))
        stages = (
            ("leaf", self.stream.anchor_supports[0]),
            ("unary", self.stream.component_supports[0]),
            ("binary", self.stream.binding_supports[0]),
        )
        compared_fields = (
            "logits",
            "composition_logits",
            "phase4_bridge_logits",
            "phase4_forward_evidence",
            "phase4_reverse_evidence",
            "binary_policy_logits",
            "root_context",
            "root_available",
            "public_feedback_evidence",
        )

        for stage, pair in stages:
            if stage == "unary":
                torch.manual_seed(97_106)
                states, _ = runner._acquire_matched_stage(
                    policy,
                    states,
                    self.stream.anchor_supports,
                )
            elif stage == "binary":
                torch.manual_seed(97_107)
                states, _ = runner._acquire_matched_stage(
                    policy,
                    states,
                    self.stream.component_supports,
                )
            tasks = (
                pair.learner,
                runner._no_demonstration_task(pair.learner),
                runner._wrong_demonstration_task(pair.learner),
            )
            scalar = tuple(
                policy.score_task(
                    task,
                    state,
                    transition_only_composition=True,
                )
                for task, state in zip(tasks, states, strict=True)
            )
            batched = runner._matched_acquisition_scores(policy, tasks, states)
            with self.subTest(stage=stage):
                for row, reference in enumerate(scalar):
                    for field in compared_fields:
                        torch.testing.assert_close(
                            getattr(batched, field)[row : row + 1],
                            getattr(reference, field),
                            atol=1.0e-6,
                            rtol=1.0e-5,
                        )
                    torch.testing.assert_close(
                        batched.root.conditioned_child_candidate_scores[
                            row : row + 1
                        ],
                        reference.root.conditioned_child_candidate_scores,
                        atol=1.0e-6,
                        rtol=1.0e-5,
                    )
                    self.assertTrue(
                        torch.equal(
                            batched.root.memory_read.write_slots[row : row + 1],
                            reference.root.memory_read.write_slots,
                        )
                    )

        changed = policy.initial_state(1)
        isolation_task = self.stream.component_supports[0].learner
        tasks = (
            isolation_task,
            runner._no_demonstration_task(isolation_task),
            runner._wrong_demonstration_task(isolation_task),
        )
        original = runner._matched_acquisition_scores(policy, tasks, states)
        perturbed = runner._matched_acquisition_scores(
            policy,
            tasks,
            (states[0], states[1], changed),
        )
        self.assertTrue(torch.equal(perturbed.logits[:2], original.logits[:2]))
        self.assertFalse(torch.equal(perturbed.logits[2], original.logits[2]))

    def test_batched_acquisition_matches_scalar_trace_state_objective_and_gradient(
        self,
    ) -> None:
        policy = self._reversible_policy()
        scalar_trace: list[tuple[tuple[str, ...], float]] = []
        batched_trace: list[tuple[tuple[str, ...], float]] = []

        def recording_judge(
            trace: list[tuple[tuple[str, ...], float]],
        ) -> object:
            def judge(public: object, hidden: object, answer: object) -> float:
                value = runner.score_conditional_procedure_answer(
                    public,  # type: ignore[arg-type]
                    hidden,  # type: ignore[arg-type]
                    answer,  # type: ignore[arg-type]
                )
                trace.append((tuple(answer), value))  # type: ignore[arg-type]
                return value

            return judge

        torch.manual_seed(97_108)
        random.seed(97_108)
        torch_rng = torch.random.get_rng_state()
        python_rng = random.getstate()
        scalar_states, scalar_records = runner._acquire_stream_arms(
            policy,
            self.stream,
            judge=recording_judge(scalar_trace),  # type: ignore[arg-type]
            matched_arm_batch=False,
        )
        scalar_torch_after = torch.random.get_rng_state()
        scalar_python_after = random.getstate()

        torch.random.set_rng_state(torch_rng)
        random.setstate(python_rng)
        batched_states, batched_records = runner._acquire_stream_arms(
            policy,
            self.stream,
            judge=recording_judge(batched_trace),  # type: ignore[arg-type]
            matched_arm_batch=True,
        )
        self.assertEqual(batched_trace, scalar_trace)
        self.assertEqual(batched_records, scalar_records)
        self.assertTrue(
            torch.equal(torch.random.get_rng_state(), scalar_torch_after)
        )
        self.assertEqual(random.getstate(), scalar_python_after)
        self.assertEqual(
            len(batched_trace),
            sum(
                len(pairs)
                for pairs in (
                    self.stream.anchor_supports,
                    self.stream.component_supports,
                    self.stream.binding_supports,
                )
            ),
        )
        for actual, expected in zip(
            batched_states,
            scalar_states,
            strict=True,
        ):
            self._assert_skill_states_close(actual, expected)
        self.assertTrue(self._state_has_autograd(batched_states[0]))
        self.assertFalse(self._state_has_autograd(batched_states[1]))
        self.assertFalse(self._state_has_autograd(batched_states[2]))

        pair = self.stream.queries[0]

        def objective(
            states: tuple[object, object, object],
        ) -> tuple[torch.Tensor, dict[str, int], tuple[int, ...], tuple[float, ...]]:
            logits = runner._matched_query_arm_logits(
                policy,
                pair.learner,
                states,  # type: ignore[arg-type]
            )
            candidates = phase5._on_policy_reward_candidate_set(
                logits[0],
                2,
                3,
            )
            rewards = runner._score_public_candidates(pair, candidates)
            loss, edges = runner._matched_multi_candidate_objective(
                *logits,
                candidates,
                rewards,
            )
            return loss, edges, candidates, rewards

        scalar_loss, scalar_edges, scalar_candidates, scalar_rewards = objective(
            scalar_states  # type: ignore[arg-type]
        )
        batched_loss, batched_edges, batched_candidates, batched_rewards = objective(
            batched_states  # type: ignore[arg-type]
        )
        self.assertEqual(batched_candidates, scalar_candidates)
        self.assertEqual(batched_rewards, scalar_rewards)
        self.assertEqual(batched_edges, scalar_edges)
        torch.testing.assert_close(
            batched_loss,
            scalar_loss,
            atol=1.0e-6,
            rtol=1.0e-5,
        )

        trainable = tuple(
            parameter for parameter in policy.parameters() if parameter.requires_grad
        )
        scalar_gradients = torch.autograd.grad(
            scalar_loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        batched_gradients = torch.autograd.grad(
            batched_loss,
            trainable,
            allow_unused=True,
        )
        for actual, expected in zip(
            batched_gradients,
            scalar_gradients,
            strict=True,
        ):
            self.assertEqual(actual is None, expected is None)
            if actual is not None and expected is not None:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=2.0e-6,
                    rtol=2.0e-5,
                )

    def test_training_query_scores_four_distinct_public_attempts_once(self) -> None:
        pair = self.stream.queries[0]
        logits = torch.linspace(-1.0, 1.0, len(phase5._PERMUTATIONS))
        candidates = phase5._on_policy_reward_candidate_set(logits, 2, 3)
        observed: list[tuple[str, ...]] = []

        def scalar_only(public: object, hidden: object, answer: object) -> float:
            del public, hidden
            observed.append(tuple(answer))  # type: ignore[arg-type]
            return float(len(observed)) / 4.0

        rewards = runner._score_public_candidates(
            pair,
            candidates,
            judge=scalar_only,
        )

        self.assertEqual(len(candidates), runner._TRAINING_ATTEMPTS_PER_QUERY)
        self.assertEqual(len(set(candidates)), len(candidates))
        self.assertEqual(len(observed), len(candidates))
        self.assertEqual(rewards, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(len(set(observed)), len(observed))

    def test_matched_query_batch_matches_scalar_objective_rng_and_gradients(
        self,
    ) -> None:
        policy = self._reversible_policy()
        torch.manual_seed(97_105)
        states, _ = runner._acquire_stream_arms(policy, self.stream)
        task = self.stream.queries[0].learner
        pair = self.stream.queries[0]

        scalar_scores = tuple(
            policy.score_task(
                task,
                state,
                transition_only_composition=True,
            )
            for state in states
        )
        scalar_again = policy.score_task(
            task,
            states[0],
            transition_only_composition=True,
        )
        self.assertTrue(torch.equal(scalar_again.logits, scalar_scores[0].logits))

        stacked = runner._stack_matched_query_states(states)
        self.assertEqual(stacked.batch_size, 3)
        batched_scores = policy.score_task(
            task,
            stacked,
            transition_only_composition=True,
        )
        compared_fields = (
            "logits",
            "composition_logits",
            "phase4_bridge_logits",
            "phase4_forward_evidence",
            "phase4_reverse_evidence",
            "binary_policy_logits",
            "root_context",
            "root_available",
            "public_feedback_evidence",
        )
        for row, scalar in enumerate(scalar_scores):
            for field in compared_fields:
                torch.testing.assert_close(
                    getattr(batched_scores, field)[row : row + 1],
                    getattr(scalar, field),
                    atol=1.0e-6,
                    rtol=1.0e-5,
                )

        perturbation = torch.linspace(
            -0.2,
            0.2,
            states[2].slot_latents.numel(),
            dtype=states[2].slot_latents.dtype,
            device=states[2].slot_latents.device,
        ).reshape_as(states[2].slot_latents)
        changed_third = replace(
            states[2],
            slot_latents=states[2].slot_latents + perturbation,
        )
        changed_scores = policy.score_task(
            task,
            runner._stack_matched_query_states(
                (states[0], states[1], changed_third)
            ),
            transition_only_composition=True,
        )
        self.assertTrue(
            torch.equal(changed_scores.logits[:2], batched_scores.logits[:2])
        )
        self.assertFalse(
            torch.equal(changed_scores.logits[2], batched_scores.logits[2])
        )

        reference_removed = policy.score_task(
            task,
            states[0],
            include_reversible_transition=False,
            transition_only_composition=True,
        ).logits.detach()
        (
            correct_logits,
            no_logits,
            wrong_logits,
            removed_logits,
        ) = runner._matched_query_arm_logits(policy, task, states)
        torch.testing.assert_close(
            correct_logits,
            scalar_scores[0].logits,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            no_logits,
            scalar_scores[1].logits,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            wrong_logits,
            scalar_scores[2].logits,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
        self.assertTrue(torch.equal(removed_logits, reference_removed))
        self.assertTrue(correct_logits.requires_grad)
        self.assertFalse(no_logits.requires_grad)
        self.assertFalse(wrong_logits.requires_grad)
        self.assertFalse(removed_logits.requires_grad)

        python_rng = random.getstate()
        torch_rng = torch.random.get_rng_state()
        reference_candidates = phase5._on_policy_reward_candidate_set(
            scalar_scores[0].logits,
            2,
            3,
        )
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))
        batched_candidates = phase5._on_policy_reward_candidate_set(
            correct_logits,
            2,
            3,
        )
        self.assertEqual(batched_candidates, reference_candidates)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))

        reference_trace: list[tuple[str, ...]] = []
        batched_trace: list[tuple[str, ...]] = []

        def recording_judge(
            trace: list[tuple[str, ...]],
        ) -> object:
            def judge(public: object, hidden: object, answer: object) -> float:
                trace.append(tuple(answer))  # type: ignore[arg-type]
                return runner.score_conditional_procedure_answer(
                    public,  # type: ignore[arg-type]
                    hidden,  # type: ignore[arg-type]
                    answer,  # type: ignore[arg-type]
                )

            return judge

        reference_rewards = runner._score_public_candidates(
            pair,
            reference_candidates,
            judge=recording_judge(reference_trace),  # type: ignore[arg-type]
        )
        batched_rewards = runner._score_public_candidates(
            pair,
            batched_candidates,
            judge=recording_judge(batched_trace),  # type: ignore[arg-type]
        )
        self.assertEqual(reference_trace, batched_trace)
        self.assertEqual(len(reference_trace), runner._TRAINING_ATTEMPTS_PER_QUERY)
        self.assertEqual(reference_rewards, batched_rewards)

        reference_loss, reference_edges = runner._matched_multi_candidate_objective(
            scalar_scores[0].logits,
            scalar_scores[1].logits.detach(),
            scalar_scores[2].logits.detach(),
            reference_removed,
            reference_candidates,
            reference_rewards,
        )
        batched_loss, batched_edges = runner._matched_multi_candidate_objective(
            correct_logits,
            no_logits,
            wrong_logits,
            removed_logits,
            batched_candidates,
            batched_rewards,
        )
        self.assertEqual(batched_edges, reference_edges)
        torch.testing.assert_close(
            batched_loss,
            reference_loss,
            atol=1.0e-6,
            rtol=1.0e-5,
        )

        def active_control_margins(
            correct: torch.Tensor,
            controls: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[bool, bool, bool]:
            correct_loss, _ = phase5._scalar_multi_preference_loss(
                correct,
                reference_candidates,
                reference_rewards,
            )
            return tuple(
                bool(
                    (
                        correct_loss
                        - phase5._scalar_multi_preference_loss(
                            control,
                            reference_candidates,
                            reference_rewards,
                        )[0]
                        + runner._CONTROL_MARGIN
                    ).item()
                    > 0.0
                )
                for control in controls
            )  # type: ignore[return-value]

        self.assertEqual(
            active_control_margins(
                scalar_scores[0].logits,
                (
                    scalar_scores[1].logits.detach(),
                    scalar_scores[2].logits.detach(),
                    reference_removed,
                ),
            ),
            active_control_margins(
                correct_logits,
                (no_logits, wrong_logits, removed_logits),
            ),
        )

        trainable = tuple(
            parameter for parameter in policy.parameters() if parameter.requires_grad
        )
        reference_gradients = torch.autograd.grad(
            reference_loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        batched_gradients = torch.autograd.grad(
            batched_loss,
            trainable,
            allow_unused=True,
        )
        for reference_gradient, batched_gradient in zip(
            reference_gradients,
            batched_gradients,
            strict=True,
        ):
            self.assertEqual(reference_gradient is None, batched_gradient is None)
            if reference_gradient is not None and batched_gradient is not None:
                torch.testing.assert_close(
                    batched_gradient,
                    reference_gradient,
                    atol=2.0e-6,
                    rtol=2.0e-5,
                )

    def test_matched_query_batch_rejects_every_unsupported_mode(self) -> None:
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        query = self.stream.queries[0].learner
        state_three = policy.initial_state(3)
        with self.assertRaisesRegex(ValueError, "reversible transition mode"):
            policy.score_task(
                query,
                state_three,
                transition_only_composition=True,
            )
        with torch.no_grad():
            policy.reversible_transition_mode.fill_(True)
        with self.assertRaisesRegex(ValueError, "exactly three"):
            policy.score_task(
                query,
                policy.initial_state(2),
                transition_only_composition=True,
            )
        with self.assertRaisesRegex(ValueError, "transition-only composition"):
            policy.score_task(query, state_three)
        with self.assertRaisesRegex(ValueError, "transition-enabled"):
            policy.score_task(
                query,
                state_three,
                include_reversible_transition=False,
                transition_only_composition=True,
            )
        with self.assertRaisesRegex(ValueError, "default compiler path"):
            policy.score_task(
                query,
                state_three,
                include_reverse_bridge=False,
                transition_only_composition=True,
            )
        with self.assertRaisesRegex(ValueError, "transition-only composition"):
            policy.score_task(
                self.stream.anchor_supports[0].learner,
                state_three,
                transition_only_composition=True,
            )
        with self.assertRaisesRegex(ValueError, "demonstration-free"):
            policy.score_task(
                self.stream.binding_supports[0].learner,
                state_three,
                transition_only_composition=True,
            )
        with self.assertRaisesRegex(ValueError, "exactly three states"):
            runner._stack_matched_query_states(  # type: ignore[arg-type]
                (policy.initial_state(1), policy.initial_state(1))
            )
        with self.assertRaisesRegex(ValueError, "singleton"):
            runner._stack_matched_query_states(
                (
                    policy.initial_state(1),
                    policy.initial_state(1),
                    policy.initial_state(2),
                )
            )

    def test_matched_acquisition_rejects_unsupported_modes_and_stale_rows(
        self,
    ) -> None:
        policy = self._reversible_policy()
        states = tuple(policy.initial_state(1) for _ in range(3))
        pair = self.stream.anchor_supports[0]
        tasks = (
            pair.learner,
            runner._no_demonstration_task(pair.learner),
            runner._wrong_demonstration_task(pair.learner),
        )
        demonstrated = self.stream.binding_supports[0].learner
        demonstrated_tasks = (
            demonstrated,
            runner._no_demonstration_task(demonstrated),
            runner._wrong_demonstration_task(demonstrated),
        )
        digests = tuple(
            runner.procedural_skill_state_digest(state) for state in states
        )

        legacy_policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        with self.assertRaisesRegex(ValueError, "reversible transition mode"):
            runner._acquire_matched_stage(
                legacy_policy,
                tuple(legacy_policy.initial_state(1) for _ in range(3)),
                (pair,),
                matched_arm_batch=True,
            )
        with self.assertRaisesRegex(TypeError, "matched_arm_batch"):
            runner._acquire_matched_stage(
                policy,
                states,
                (pair,),
                matched_arm_batch=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "row one"):
            runner._matched_acquisition_scores(
                policy,
                (
                    demonstrated_tasks[0],
                    demonstrated_tasks[2],
                    demonstrated_tasks[1],
                ),
                states,
            )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            policy.score_task(
                pair.learner,
                policy.initial_state(1),
                transition_only_composition=True,
                matched_acquisition_batch=True,
            )
        with self.assertRaisesRegex(ValueError, "demonstration-free"):
            policy.score_task(
                self.stream.binding_supports[0].learner,
                runner._stack_matched_query_states(states),
                transition_only_composition=True,
                matched_acquisition_batch=True,
            )
        with self.assertRaisesRegex(ValueError, "exactly three rows"):
            runner._split_matched_arm_state(policy.initial_state(1))

        scores = policy.score_task(
            pair.learner,
            states[0],
            transition_only_composition=True,
        )
        probabilities = torch.softmax(scores.logits[0] / 1.25, dim=-1)
        with self.assertRaisesRegex(ValueError, "three state rows"):
            phase5.propose_matched_differentiable_feedback(
                policy,
                scores,
                0,
                probabilities,
                0.5,
                states[0],
            )

        proposal = runner._matched_transition_proposal(
            policy,
            tasks,
            states,
            validated_state_digests=digests,  # type: ignore[arg-type]
        )
        stale = ("sha256:stale", digests[1], digests[2])
        with self.assertRaisesRegex(ValueError, "stale"):
            runner._commit_matched_transition_feedback(
                policy,
                tasks,
                proposal,
                0.5,
                states,
                validated_state_digests=stale,
            )

    def test_multi_candidate_objective_learns_preference_not_one_action_bce(self) -> None:
        candidates = (3, 17, 61, 109)
        rewards = (1.0, 0.7, 0.3, 0.0)
        correct = torch.zeros(
            (1, len(phase5._PERMUTATIONS)),
            requires_grad=True,
        )
        no_demo = torch.zeros_like(correct)
        wrong = torch.zeros_like(correct, requires_grad=True)
        removed = torch.zeros_like(correct)

        loss, edges = runner._matched_multi_candidate_objective(
            correct,
            no_demo,
            wrong,
            removed,
            candidates,
            rewards,
        )
        loss.backward()

        self.assertEqual(
            edges,
            {
                "correct": 6,
                "no_demonstration": 6,
                "wrong_demonstration": 6,
                "reversible_removed": 6,
            },
        )
        self.assertLess(float(correct.grad[0, candidates[0]].item()), 0.0)
        self.assertGreater(float(correct.grad[0, candidates[-1]].item()), 0.0)
        untouched = correct.grad.detach().clone()
        untouched[0, list(candidates)] = 0.0
        self.assertTrue(torch.equal(untouched, torch.zeros_like(untouched)))
        self.assertIsNone(wrong.grad)

    def test_runner_never_inspects_hidden_solution_fields(self) -> None:
        source_path = Path(runner.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {
            "position_permutation",
            "target_order",
            "instance_id",
            "source_instance_id",
            "generator_seed",
            "surface_seed",
        }
        observed = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(observed & forbidden)

        hidden_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "hidden"
        ]
        self.assertEqual(len(hidden_reads), 1)
        scorer = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_score_attempt"
        )
        self.assertIn(hidden_reads[0], list(ast.walk(scorer)))

    def test_training_opts_into_batch_and_steps_once_per_mechanism(self) -> None:
        policy = self._reversible_policy()
        batch_flags: list[object] = []
        optimizer_steps: list[int] = []
        evaluator_calls: list[tuple[str, ...]] = []
        original_acquire = runner._acquire_stream_arms
        original_step = torch.optim.AdamW.step

        def recording_acquire(*args: object, **kwargs: object) -> object:
            batch_flags.append(kwargs.get("matched_arm_batch"))
            return original_acquire(*args, **kwargs)  # type: ignore[arg-type]

        def recording_step(
            optimizer: torch.optim.AdamW,
            *args: object,
            **kwargs: object,
        ) -> object:
            optimizer_steps.append(1)
            return original_step(optimizer, *args, **kwargs)  # type: ignore[arg-type]

        def recording_judge(
            public: object,
            hidden: object,
            answer: object,
        ) -> float:
            evaluator_calls.append(tuple(answer))  # type: ignore[arg-type]
            return runner.score_conditional_procedure_answer(
                public,  # type: ignore[arg-type]
                hidden,  # type: ignore[arg-type]
                answer,  # type: ignore[arg-type]
            )

        with (
            patch.object(
                runner,
                "_acquire_stream_arms",
                side_effect=recording_acquire,
            ),
            patch.object(torch.optim.AdamW, "step", new=recording_step),
        ):
            report = runner._train_conditional_interface(
                policy,
                seed=97_109,
                meta_steps=1,
                supports_per_flag=1,
                queries_per_flag=1,
                learning_rate=4.0e-4,
                judge=recording_judge,
            )

        self.assertEqual(batch_flags, [True])
        self.assertEqual(len(optimizer_steps), report["meta_steps"])
        self.assertEqual(report["scalar_evaluator_calls"], 16)
        self.assertEqual(len(evaluator_calls), 16)
        self.assertEqual(report["total_scored_query_attempts"], 8)

    def test_run_reloads_train_checkpoint_before_opening_final(self) -> None:
        events: list[str] = []
        fake_policy = SimpleNamespace()
        runtime = runner._Runtime(
            policy=fake_policy,  # type: ignore[arg-type]
            base_record={"sha256": "base"},
            precedence_record={"sha256": "precedence"},
            source_interface_record={"sha256": "source"},
            compiler_record={"sha256": "compiler"},
        )
        training = {"status": "trained"}
        saved = {"sha256": "frozen", "stage": "train_only_before_final"}

        with (
            patch.object(runner, "_load_runtime", return_value=runtime),
            patch.object(
                runner,
                "_train_conditional_interface",
                side_effect=lambda *args, **kwargs: events.append("train") or training,
            ),
            patch.object(
                runner,
                "_save_train_only_checkpoint",
                side_effect=lambda *args, **kwargs: events.append("save") or saved,
            ),
            patch.object(
                runner,
                "_load_train_only_checkpoint",
                side_effect=lambda *args, **kwargs: (
                    events.append("reload") or (training, saved)
                ),
            ),
            patch.object(
                runner,
                "_evaluate_final_panel",
                side_effect=lambda *args, **kwargs: events.append("final")
                or {"partition": "final", "mechanisms": 20},
            ),
        ):
            result = runner.run(
                seed=97_103,
                device="cpu",
                initial_checkpoint="base.pt",
                precedence_adapter_checkpoint="precedence.pt",
                demonstration_adapter_checkpoint="source.pt",
                meta_steps=1,
                meta_supports_per_flag=1,
                meta_queries_per_flag=1,
                final_supports_per_flag=1,
                final_queries_per_flag=1,
                checkpoint="frozen.pt",
            )

        self.assertEqual(events, ["train", "save", "reload", "final"])
        self.assertTrue(
            result["claims"]["final_opened_after_train_checkpoint_reload"]
        )


if __name__ == "__main__":
    unittest.main()
