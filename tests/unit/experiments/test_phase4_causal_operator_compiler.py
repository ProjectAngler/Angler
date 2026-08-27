"""Integrity boundaries for the causal-operator experiment runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.alignment import AliasTable  # noqa: E402
from angler.procedures.learning import _meet_in_middle_scores  # noqa: E402
from experiments.evaluators.causal_operator_suite import (  # noqa: E402
    make_heldout_operator_suite,
)
from experiments.runners.causal_operator_experience import (  # noqa: E402
    build_causal_operator_experience,
)
from experiments.runners import phase4_causal_operator_compiler as runner  # noqa: E402


def _competence_summary(two_step: float, four_step: float) -> dict[str, object]:
    def group(rate: float) -> dict[str, object]:
        return {"attempts": 10, "success_rate": rate}

    return {
        "overall": group((two_step + four_step) / 2.0),
        "by_step_ceiling": {"2": group(two_step), "4": group(four_step)},
    }


class CausalOperatorRunnerIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experience = build_causal_operator_experience(seed=42_017)
        cls.selected = runner._selected_candidates(cls.experience)

    def test_candidate_enumeration_never_executes_the_world(self) -> None:
        canonical = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            canonical,
            AliasTable(),
        )
        challenge = next(
            item
            for item in make_heldout_operator_suite(91_001)
            if item.domain == "tokens"
        )

        with patch.object(
            runner,
            "_executor",
            side_effect=AssertionError("proposal attempted world execution"),
        ):
            candidates, witnesses = runner._enumerate_semantic_candidates(
                context,
                challenge.origin,
            )

        self.assertTrue(candidates)
        self.assertEqual({item.digest for item in candidates}, set(witnesses))

    def test_atomic_experience_terminates_at_its_observed_endpoint(self) -> None:
        operator = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            operator,
            AliasTable(),
        )

        examples = runner._experience_examples(context)

        self.assertTrue(examples)
        self.assertTrue(
            all(example.after.records == example.goal.required for example in examples)
        )
        self.assertTrue(
            all(example.before.records != example.goal.required for example in examples)
        )

    def test_teacher_preserves_verified_multi_step_plan_as_one_trajectory(self) -> None:
        canonical = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            canonical,
            AliasTable(),
        )
        training_cases = runner._suite_cases((52_017,), cases_per_domain=2)

        examples, trajectories = runner._training_data_for_domain(
            context,
            training_cases,
        )

        self.assertTrue(examples)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(len(trajectories[0].steps), 2)
        self.assertEqual(
            trajectories[0].steps[0].after,
            trajectories[0].steps[1].before,
        )
        self.assertEqual(
            trajectories[0].steps[-1].after.records,
            trajectories[0].goal.required,
        )
        self.assertEqual(
            tuple(
                item.digest
                for item in trajectories[0].steps[0].candidate_bindings
            ),
            tuple(
                item.digest
                for item in trajectories[0].steps[1].candidate_bindings
            ),
        )

    def test_teacher_forcing_schedule_finishes_with_pure_self_rollout(self) -> None:
        values = tuple(
            runner._trajectory_teacher_forcing_ratio(step, 8)
            for step in range(8)
        )
        scales = tuple(
            runner._trajectory_loss_scale(step, 8)
            for step in range(8)
        )

        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[-1], 0.0)
        self.assertEqual(values[-2], 0.0)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertTrue(any(0.0 < value < 1.0 for value in values))
        self.assertEqual(scales[0], 0.0)
        self.assertEqual(scales[1], 0.0)
        self.assertEqual(scales[-1], 1.0)
        self.assertTrue(all(left <= right for left, right in zip(scales, scales[1:])))
        self.assertTrue(any(0.0 < value < 1.0 for value in scales))

    def test_competence_requires_every_observed_horizon(self) -> None:
        self.assertFalse(
            runner._meets_competence(
                _competence_summary(1.0, 0.8),
                threshold=0.9,
            )
        )
        self.assertTrue(
            runner._meets_competence(
                _competence_summary(1.0, 0.9),
                threshold=0.9,
            )
        )

    def test_competent_increment_skips_every_update(self) -> None:
        learner = runner._make_learner(
            runner.PROFILES["smoke"],
            seed=92_010,
            device=torch.device("cpu"),
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in learner.named_parameters()
        }

        with (
            patch.object(
                runner,
                "_train_steps",
                side_effect=AssertionError("competent transfer trained"),
            ),
            patch.object(
                learner,
                "configure_plasticity",
                side_effect=AssertionError("competent transfer changed scope"),
            ),
        ):
            report = runner._adapt_incremental_domain(
                learner,
                "files",
                (object(),),
                (),
                (),
                (),
                {"files": object()},
                (),
                _competence_summary(1.0, 1.0),
                runner.PROFILES["smoke"],
                seed=92_011,
            )

        self.assertTrue(report["update_skipped"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["changed_parameter_names"], ())
        for name, parameter in learner.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter.detach()), name)

    def test_deficient_increment_updates_only_selection_and_stops_early(self) -> None:
        profile = replace(
            runner.PROFILES["smoke"],
            selection_optimizer_steps=120,
            progress_interval=60,
        )
        learner = runner._make_learner(
            profile,
            seed=92_012,
            device=torch.device("cpu"),
        )
        expected_trainable_ids: set[int] = set()

        def train_once(
            model: object,
            optimizer: torch.optim.Optimizer,
            *_args: object,
            **kwargs: object,
        ) -> dict[str, object]:
            actual_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertEqual(actual_ids, expected_trainable_ids)
            self.assertEqual(kwargs["trajectory_schedule"], "self_rollout")
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(0.25)
            return {"steps": kwargs["steps"], "trajectory_schedule": "self_rollout"}

        real_configure = learner.configure_plasticity

        def configure(scope: str) -> tuple[str, ...]:
            names = real_configure(scope)
            expected_trainable_ids.update(
                id(parameter)
                for parameter in learner.parameters()
                if parameter.requires_grad
            )
            return names

        with (
            patch.object(learner, "configure_plasticity", side_effect=configure),
            patch.object(runner, "_train_steps", side_effect=train_once) as train,
            patch.object(
                runner,
                "_evaluate_cases",
                return_value={
                    "summary": _competence_summary(1.0, 0.9),
                    "cases": [],
                },
            ),
        ):
            report = runner._adapt_incremental_domain(
                learner,
                "boxes",
                (object(),),
                (object(),),
                (),
                (),
                {"boxes": object()},
                (),
                _competence_summary(1.0, 0.2),
                profile,
                seed=92_013,
            )

        self.assertEqual(train.call_count, 1)
        self.assertFalse(report["update_skipped"])
        self.assertEqual(report["optimizer_steps"], 60)
        self.assertTrue(report["competent_after"])
        self.assertTrue(report["changed_parameter_names"])
        self.assertEqual(report["unexpected_changed_parameter_names"], ())
        self.assertTrue(
            set(report["changed_parameter_names"])
            <= set(report["enabled_parameter_names"])
        )
        for name, parameter in learner.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name in report["enabled_parameter_names"],
                name,
            )

    def test_self_rollout_training_bypasses_acquisition_schedule(self) -> None:
        profile = runner.PROFILES["smoke"]
        learner = runner._make_learner(
            profile,
            seed=92_014,
            device=torch.device("cpu"),
        )
        canonical = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            canonical,
            AliasTable(),
        )
        examples, trajectories = runner._training_data_for_domain(
            context,
            runner._suite_cases((92_015,), cases_per_domain=2),
        )
        self.assertTrue(trajectories)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-4)

        with (
            patch.object(
                runner,
                "_trajectory_teacher_forcing_ratio",
                side_effect=AssertionError("acquisition forcing was consulted"),
            ),
            patch.object(
                runner,
                "_trajectory_loss_scale",
                side_effect=AssertionError("acquisition scale was consulted"),
            ),
        ):
            receipt = runner._train_steps(
                learner,
                optimizer,
                examples[:1],
                (),
                new_trajectories=trajectories[:1],
                steps=1,
                batch_size=1,
                replay_ratio=0.0,
                seed=92_016,
                trajectory_schedule="self_rollout",
            )

        self.assertEqual(receipt["trajectory_schedule"], "self_rollout")
        self.assertEqual(receipt["trajectory_teacher_forcing_first"], 0.0)
        self.assertEqual(receipt["trajectory_teacher_forcing_last"], 0.0)
        self.assertEqual(receipt["trajectory_loss_scale_first"], 1.0)
        self.assertEqual(receipt["trajectory_loss_scale_last"], 1.0)

    def test_direct_proposal_has_no_teacher_or_world_execution_path(self) -> None:
        canonical = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            canonical,
            AliasTable(),
        )
        challenge = next(
            item
            for item in make_heldout_operator_suite(92_018, cases_per_domain=2)
            if item.domain == "tokens"
        )
        learner = runner._make_learner(
            runner.PROFILES["smoke"],
            seed=92_019,
            device=torch.device("cpu"),
        )
        _, trajectories = runner._training_data_for_domain(
            context,
            runner._suite_cases((52_019,), cases_per_domain=2),
        )
        self.assertTrue(trajectories)
        blocked = AssertionError("held-out proposal crossed the learning boundary")

        with patch.object(
            learner,
            "candidate_selection_logits",
            wraps=learner.candidate_selection_logits,
        ) as selection_path, patch.object(
            learner,
            "horizon_agnostic_join_scores",
            wraps=learner.horizon_agnostic_join_scores,
        ) as horizon_agnostic_join:
            learner.trajectory_losses(
                (trajectories[0],),
                teacher_forcing_ratio=0.0,
            )
            training_selection_calls = selection_path.call_count
            training_join_calls = horizon_agnostic_join.call_count
            with (
                patch.object(runner, "search_teacher_plan", side_effect=blocked),
                patch.object(runner, "_verified_teacher_plan", side_effect=blocked),
                patch.object(runner, "_teacher_training_data", side_effect=blocked),
                patch.object(runner, "_executor", side_effect=blocked),
                patch.object(runner, "evaluate_committed_sequence", side_effect=blocked),
            ):
                proposal = runner._direct_proposal(learner, context, challenge)

        self.assertIsInstance(proposal, runner.DirectProposal)
        self.assertGreater(training_selection_calls, 0)
        self.assertGreater(training_join_calls, 0)
        self.assertGreater(selection_path.call_count, training_selection_calls)
        self.assertGreater(horizon_agnostic_join.call_count, training_join_calls)

    def test_direct_proposal_does_not_treat_step_ceiling_as_solution_length(self) -> None:
        canonical = self.selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            self.selected["tokens"],
            canonical,
            AliasTable(),
        )
        challenge = next(
            item
            for item in make_heldout_operator_suite(92_020, cases_per_domain=2)
            if item.domain == "tokens" and item.maximum_steps == 2
        )
        slack = replace(challenge, maximum_steps=6)
        learner = runner._make_learner(
            runner.PROFILES["smoke"],
            seed=92_021,
            device=torch.device("cpu"),
        )

        def terminate(candidate_states: torch.Tensor, _goals: torch.Tensor) -> torch.Tensor:
            count = candidate_states.shape[0]
            return torch.full(
                (count,),
                30.0,
                device=candidate_states.device,
                dtype=candidate_states.dtype,
            )

        def decode(
            _state: torch.Tensor,
            _goal: torch.Tensor,
            binding: object,
            *_args: object,
            **_kwargs: object,
        ) -> SimpleNamespace:
            actions = []
            for pattern in canonical.body:
                arguments = tuple(
                    term.value
                    if isinstance(term, runner.Constant)
                    else binding.value_for(term.name)
                    for term in pattern.arguments
                )
                actions.append(pattern.schema.ground(*arguments))
            return SimpleNamespace(actions=tuple(actions), stopped=True)

        with (
            patch.object(learner.core, "termination_logits", side_effect=terminate),
            patch.object(learner.decoder, "decode_sequence_greedy", side_effect=decode),
        ):
            exact = runner._direct_proposal(learner, context, challenge)
            non_divisible = runner._direct_proposal(
                learner,
                context,
                replace(challenge, maximum_steps=3),
            )
            extra = runner._direct_proposal(learner, context, slack)

        self.assertEqual(exact, extra)
        self.assertEqual(exact, non_divisible)
        self.assertIsNone(exact.failure)
        self.assertEqual(len(exact.actions), len(canonical.body))

    def test_training_progress_and_final_case_identities_are_disjoint(self) -> None:
        training = runner._suite_cases((52_017, 52_018), cases_per_domain=4)
        progress = runner._suite_cases((92_018,), cases_per_domain=2)
        prior_final = runner._suite_cases((72_017, 72_018), cases_per_domain=2)
        version_two_final = runner._suite_cases(
            (102_017, 102_018),
            cases_per_domain=2,
        )
        version_three_final = runner._suite_cases(
            (132_017, 132_018),
            cases_per_domain=2,
        )
        version_four_final = runner._suite_cases(
            (172_017, 172_018),
            cases_per_domain=2,
        )
        version_five_incremental = runner._suite_cases(
            (352_017, 352_018),
            cases_per_domain=4,
        )
        version_five_adaptation = runner._suite_cases(
            (372_017, 372_018),
            cases_per_domain=2,
        )
        version_five_final = runner._suite_cases(
            (412_017, 412_018),
            cases_per_domain=2,
        )
        identities = [
            {challenge.case_id for _, challenge in cases}
            for cases in (
                training,
                progress,
                prior_final,
                version_two_final,
                version_three_final,
                version_four_final,
                version_five_incremental,
                version_five_adaptation,
                version_five_final,
            )
        ]

        for left in range(len(identities)):
            for right in range(left + 1, len(identities)):
                self.assertFalse(identities[left] & identities[right])

    def test_failed_partial_proposal_commits_no_actions(self) -> None:
        challenge = make_heldout_operator_suite(91_002)[0]
        schema = challenge.allowed_action_schemas[0]
        partial = schema.ground("uncommitted", "position_0", "position_1")
        proposal = runner.DirectProposal(
            actions=(partial,),
            candidate_counts=(3,),
            score_margins=(0.5,),
            decoder_stopped=(True,),
            failure="later chunk failed",
        )

        with patch.object(runner, "_direct_proposal", return_value=proposal):
            evaluation = runner._evaluate_cases(
                object(),
                {challenge.domain: object()},
                ((91_002, challenge),),
            )

        row = evaluation["cases"][0]
        self.assertEqual(row["tool_calls"], 0)
        self.assertEqual(row["applied_actions"], 0)
        self.assertEqual(row["rejected_action_count"], 1)
        self.assertFalse(row["success"])

    def test_masked_candidate_does_not_create_infinite_margin(self) -> None:
        self.assertIsNone(
            runner._score_margin(torch.tensor([1.0, -torch.inf]))
        )
        margin = runner._score_margin(torch.tensor([2.0, 0.5, -torch.inf]))
        self.assertIsNotNone(margin)
        self.assertTrue(math.isfinite(margin))
        self.assertAlmostEqual(margin, 1.5)

    def test_meet_in_middle_prefers_matching_forward_reverse_frontiers(self) -> None:
        forward = torch.tensor([[1.0, 0.0], [4.0, 0.0], [9.0, 0.0]])
        backward = torch.tensor([[8.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        mask = torch.tensor([True, True, True])

        scores = _meet_in_middle_scores(forward, backward, mask)

        # Candidate zero meets the distinct reverse candidate one exactly.
        self.assertEqual(int(torch.argmax(scores).item()), 0)
        self.assertEqual(float(scores[0]), 0.0)

    def test_result_digest_rejects_non_finite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            runner._digest({"invalid": math.inf})


if __name__ == "__main__":
    unittest.main()
