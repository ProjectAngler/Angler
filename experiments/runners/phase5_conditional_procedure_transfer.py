"""Train and evaluate Angler's first conditional demonstration transfer path.

The evaluator supplies an explicit public curriculum::

    A -> P0(A), P1(A) -> C(P0(A), P1(A)) -> held-out queries

All composed proposals run with ``transition_only_composition=True``.  Thus a
unary procedure contributes through the retained reversible transition rather
than its direct utility decoder, while the binary root remains a learned
candidate-blind selector.  Training sees only public tasks, attempted public
orderings, and one scalar evaluator result per attempt.  Hidden evaluator
objects are passed opaquely to the scalar scorer and are never inspected here.

This runner deliberately freezes and reloads a train-only interface checkpoint
before it asks the evaluator for the final mechanism partition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.skill_memory import (
    ProceduralSkillState,
    procedural_skill_state_digest,
)
from experiments.evaluators.conditional_symbolic_procedure_transfer_suite import (
    ConditionalProcedureTransferStream,
    GeneratedConditionalProcedureTask,
    conditional_mechanism_partition,
    make_conditional_procedure_transfer_stream,
    score_conditional_procedure_answer,
)
from experiments.evaluators.symbolic_procedure_transfer_suite import (
    PublicDemonstrationProcedureTask,
    PublicSymbolicDemonstration,
)
from experiments.runners import phase5_demonstration_transfer as demonstration
from experiments.runners import phase5_skill_memory_stream as phase5
from experiments.runners.phase5_cross_family_transfer import (
    _state_element_count,
    _summary,
)


_REPORT_VERSION = "angler.phase5-conditional-procedure-transfer.v1"
_TRANSITION_ONLY = True
_CONTROL_MARGIN = 0.03
_TRAINING_ATTEMPTS_PER_QUERY = 4
_DEFAULT_META_STEPS = 512

ScalarJudge = Callable[[Any, Any, tuple[str, ...]], float]


@dataclass(frozen=True, slots=True)
class _Runtime:
    policy: phase5.SkillMemoryPolicy
    base_record: dict[str, Any]
    precedence_record: dict[str, Any]
    source_interface_record: dict[str, Any]
    compiler_record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MatchedTransitionProposal:
    """One sampled public attempt bound to three matched competence rows."""

    answer: tuple[str, ...]
    candidate_index: int
    scores: phase5.PolicyScores
    behavior_probabilities: torch.Tensor
    competence_digests: tuple[str, str, str]
    public_task_digests: tuple[str, str, str]


def _no_demonstration_task(
    task: PublicDemonstrationProcedureTask,
) -> PublicDemonstrationProcedureTask:
    """Remove only learner-visible examples from one public task."""

    if not isinstance(task, PublicDemonstrationProcedureTask):
        raise TypeError("control task must use the demonstration schema")
    return replace(task, demonstrations=())


def _wrong_demonstration_task(
    task: PublicDemonstrationProcedureTask,
) -> PublicDemonstrationProcedureTask:
    """Apply one fixed public output rotation without evaluator information."""

    if not isinstance(task, PublicDemonstrationProcedureTask):
        raise TypeError("control task must use the demonstration schema")
    if not task.demonstrations:
        return task
    return replace(
        task,
        demonstrations=tuple(
            PublicSymbolicDemonstration(
                value.input_symbols,
                value.output_symbols[1:] + value.output_symbols[:1],
            )
            for value in task.demonstrations
        ),
    )


def _transition_proposal(
    policy: phase5.SkillMemoryPolicy,
    public_task: PublicDemonstrationProcedureTask,
    state: ProceduralSkillState,
    *,
    greedy: bool,
    temperature: float = 1.25,
    candidate_index: int | None = None,
    behavior_probabilities: torch.Tensor | None = None,
    include_reversible_transition: bool = True,
    validated_state_digest: str | None = None,
) -> phase5.TaskProposal:
    """Make an ordinary public proposal through transition-only composition.

    A supplied digest must already have been computed from this exact current
    state, which remains immutable for the proposal/commit transaction.
    """

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scores = policy.score_task(
        public_task,
        state,
        include_reversible_transition=include_reversible_transition,
        transition_only_composition=_TRANSITION_ONLY,
    )
    probabilities = torch.softmax(scores.logits[0] / temperature, dim=-1)
    if candidate_index is None:
        if greedy:
            selected = int(scores.logits.argmax(dim=-1).item())
            behavior = F.one_hot(
                torch.tensor(selected, device=scores.logits.device),
                num_classes=len(phase5._PERMUTATIONS),
            ).to(dtype=scores.logits.dtype)
        else:
            selected = int(torch.multinomial(probabilities, 1).item())
            behavior = probabilities.detach()
    else:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(phase5._PERMUTATIONS)
        ):
            raise ValueError("candidate_index is outside the public candidate set")
        selected = candidate_index
        if behavior_probabilities is None:
            behavior = probabilities.detach()
        else:
            behavior = behavior_probabilities.to(
                device=scores.logits.device,
                dtype=scores.logits.dtype,
            )
            if behavior.shape != probabilities.shape or not torch.allclose(
                behavior.sum(),
                behavior.new_tensor(1.0),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError("matched behavior probabilities are invalid")
    permutation = phase5._PERMUTATIONS[selected]
    answer = tuple(public_task.items[index].symbol for index in permutation)
    if validated_state_digest is None:
        competence_digest = procedural_skill_state_digest(state)
    else:
        if not isinstance(validated_state_digest, str):
            raise TypeError("validated_state_digest must be text")
        competence_digest = validated_state_digest
    return phase5.TaskProposal(
        answer=answer,
        candidate_index=selected,
        scores=scores,
        behavior_probabilities=behavior,
        competence_digest=competence_digest,
        public_task_digest=phase5._public_task_digest(public_task),
    )


def _score_attempt(
    pair: GeneratedConditionalProcedureTask,
    answer: tuple[str, ...],
    judge: ScalarJudge = score_conditional_procedure_answer,
) -> float:
    """Pass evaluator state opaquely into its scalar-only scoring boundary."""

    value = float(judge(pair.learner, pair.hidden, answer))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("conditional evaluator returned an invalid scalar")
    return value


def _validate_matched_acquisition_tasks(
    tasks: tuple[
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
    ],
) -> tuple[
    PublicDemonstrationProcedureTask,
    PublicDemonstrationProcedureTask,
    PublicDemonstrationProcedureTask,
]:
    """Require the exact correct/no/wrong task order and shared geometry."""

    if type(tasks) is not tuple or len(tasks) != 3:
        raise ValueError("matched acquisition requires exactly three public tasks")
    if any(not isinstance(task, PublicDemonstrationProcedureTask) for task in tasks):
        raise TypeError("matched acquisition tasks use the wrong public schema")
    correct, absent, wrong = tasks
    if absent != _no_demonstration_task(correct):
        raise ValueError("matched acquisition row one must remove demonstrations")
    if wrong != _wrong_demonstration_task(correct):
        raise ValueError("matched acquisition row two must rotate demonstrations")
    return correct, absent, wrong


def _matched_acquisition_scores(
    policy: phase5.SkillMemoryPolicy,
    tasks: tuple[
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
    ],
    states: tuple[
        ProceduralSkillState,
        ProceduralSkillState,
        ProceduralSkillState,
    ],
) -> phase5.PolicyScores:
    """Score shared public geometry once, then attach row-specific evidence."""

    correct, absent, wrong = _validate_matched_acquisition_tasks(tasks)
    scores = policy.score_task(
        absent,
        _stack_matched_query_states(states),
        transition_only_composition=_TRANSITION_ONLY,
        matched_acquisition_batch=True,
    )
    ports = getattr(policy, "public_fact_adapter", None)
    if not isinstance(ports, demonstration.TypedPublicFactPorts):
        raise RuntimeError("typed public-fact ports are not attached")
    reference = scores.root.feedback_context
    if reference.shape != (3, policy.profile.width):
        raise RuntimeError("matched acquisition feedback context has the wrong shape")
    correct_evidence = ports.feedback_evidence(correct, reference[0:1])
    with torch.no_grad():
        absent_evidence = ports.feedback_evidence(absent, reference[1:2])
        wrong_evidence = ports.feedback_evidence(wrong, reference[2:3])
    public_evidence = torch.cat(
        (
            correct_evidence,
            absent_evidence.detach(),
            wrong_evidence.detach(),
        ),
        dim=0,
    )
    if public_evidence.shape != reference.shape or not bool(
        torch.isfinite(public_evidence).all().item()
    ):
        raise RuntimeError("matched acquisition evidence is invalid")
    return replace(scores, public_feedback_evidence=public_evidence)


def _matched_transition_proposal(
    policy: phase5.SkillMemoryPolicy,
    tasks: tuple[
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
    ],
    states: tuple[
        ProceduralSkillState,
        ProceduralSkillState,
        ProceduralSkillState,
    ],
    *,
    validated_state_digests: tuple[str, str, str],
    temperature: float = 1.25,
) -> _MatchedTransitionProposal:
    """Sample only the correct row and bind that attempt to all three arms."""

    tasks = _validate_matched_acquisition_tasks(tasks)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if (
        type(validated_state_digests) is not tuple
        or len(validated_state_digests) != 3
        or any(not isinstance(value, str) for value in validated_state_digests)
    ):
        raise ValueError("matched state digests must contain three strings")
    scores = _matched_acquisition_scores(policy, tasks, states)
    probabilities = torch.softmax(scores.logits[0] / temperature, dim=-1)
    selected = int(torch.multinomial(probabilities, 1).item())
    permutation = phase5._PERMUTATIONS[selected]
    answer = tuple(tasks[0].items[index].symbol for index in permutation)
    return _MatchedTransitionProposal(
        answer=answer,
        candidate_index=selected,
        scores=scores,
        behavior_probabilities=probabilities.detach(),
        competence_digests=validated_state_digests,
        public_task_digests=tuple(phase5._public_task_digest(task) for task in tasks),
    )


def _commit_matched_transition_feedback(
    policy: phase5.SkillMemoryPolicy,
    tasks: tuple[
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
        PublicDemonstrationProcedureTask,
    ],
    proposal: _MatchedTransitionProposal,
    reward: float,
    states: tuple[
        ProceduralSkillState,
        ProceduralSkillState,
        ProceduralSkillState,
    ],
    *,
    validated_state_digests: tuple[str, str, str],
) -> tuple[
    tuple[ProceduralSkillState, ProceduralSkillState, ProceduralSkillState],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]:
    """Commit one row-local write and preserve per-arm acceptance evidence."""

    tasks = _validate_matched_acquisition_tasks(tasks)
    if not isinstance(proposal, _MatchedTransitionProposal):
        raise TypeError("proposal must be a matched transition proposal")
    if proposal.competence_digests != validated_state_digests:
        raise ValueError("matched proposal is stale for the supplied states")
    current_task_digests = tuple(
        phase5._public_task_digest(task) for task in tasks
    )
    if proposal.public_task_digests != current_task_digests:
        raise ValueError("matched proposal and public tasks differ")
    incoming_elements = tuple(_state_element_count(state) for state in states)
    stacked = _stack_matched_query_states(states)
    write = phase5.propose_matched_differentiable_feedback(
        policy,
        proposal.scores,
        proposal.candidate_index,
        proposal.behavior_probabilities,
        reward,
        stacked,
    )
    with torch.no_grad():
        rescored = policy.score_task(
            tasks[1],
            write.candidate_state,
            transition_only_composition=_TRANSITION_ONLY,
            matched_acquisition_batch=True,
        )
    if not bool(torch.isfinite(rescored.logits).all().item()):
        raise RuntimeError("matched feedback produced non-finite policy scores")
    candidates = _split_matched_arm_state(write.candidate_state)
    records: list[dict[str, Any]] = []
    for row, (candidate, incoming_digest, incoming_count) in enumerate(
        zip(candidates, validated_state_digests, incoming_elements, strict=True)
    ):
        if _state_element_count(candidate) != incoming_count:
            raise RuntimeError("matched feedback changed fixed state capacity")
        outgoing_digest = procedural_skill_state_digest(candidate)
        accepted = outgoing_digest != incoming_digest
        delta_norm = float(write.delta_norms[row].item())
        if accepted != (delta_norm > 0.0):
            raise RuntimeError(
                "matched feedback acceptance evidence is inconsistent"
            )
        records.append(
            {
                "accepted": accepted,
                "write_slot": int(write.write_slots[row].item()),
                "delta_norm": delta_norm,
            }
        )
    return candidates, tuple(records)  # type: ignore[return-value]


def _commit_transition_feedback(
    policy: phase5.SkillMemoryPolicy,
    public_task: PublicDemonstrationProcedureTask,
    proposal: phase5.TaskProposal,
    reward: float,
    state: ProceduralSkillState,
    *,
    validated_state_digest: str | None = None,
) -> tuple[ProceduralSkillState, dict[str, Any]]:
    """Commit the ordinary bounded differentiable memory transaction.

    A supplied digest must already have been computed from this exact current
    state, which remains immutable for the proposal/commit transaction.
    """

    if proposal.public_task_digest != phase5._public_task_digest(public_task):
        raise ValueError("proposal and public task differ")
    if validated_state_digest is None:
        incoming_digest = procedural_skill_state_digest(state)
    else:
        if not isinstance(validated_state_digest, str):
            raise TypeError("validated_state_digest must be text")
        incoming_digest = validated_state_digest
    incoming_elements = _state_element_count(state)
    if validated_state_digest is None:
        write = phase5.propose_differentiable_feedback(
            policy,
            proposal,
            reward,
            state,
        )
    else:
        write = phase5.propose_differentiable_feedback(
            policy,
            proposal,
            reward,
            state,
            validated_state_digest=incoming_digest,
        )
    candidate = write.candidate_state
    if _state_element_count(candidate) != incoming_elements:
        raise RuntimeError("conditional feedback changed fixed state capacity")
    rescored = policy.score_task(
        public_task,
        candidate,
        transition_only_composition=_TRANSITION_ONLY,
    )
    if not bool(torch.isfinite(rescored.logits).all().item()):
        raise RuntimeError("conditional feedback produced non-finite policy scores")
    outgoing_digest = procedural_skill_state_digest(candidate)
    accepted = outgoing_digest != incoming_digest
    if accepted != (write.delta_norm > 0.0):
        raise RuntimeError("conditional feedback acceptance evidence is inconsistent")
    return candidate, {
        "accepted": accepted,
        "write_slot": write.write_slot,
        "delta_norm": write.delta_norm,
    }


def _acquire_matched_stage(
    policy: phase5.SkillMemoryPolicy,
    states: tuple[ProceduralSkillState, ProceduralSkillState, ProceduralSkillState],
    pairs: Sequence[GeneratedConditionalProcedureTask],
    *,
    judge: ScalarJudge = score_conditional_procedure_answer,
    matched_arm_batch: bool = False,
) -> tuple[
    tuple[ProceduralSkillState, ProceduralSkillState, ProceduralSkillState],
    dict[str, Any],
]:
    """Acquire correct/absent/wrong public evidence with one score per attempt."""

    if not pairs:
        raise ValueError("conditional acquisition stage must not be empty")
    if not isinstance(matched_arm_batch, bool):
        raise TypeError("matched_arm_batch must be boolean")
    if not matched_arm_batch:
        correct_state, absent_state, wrong_state = states
        accepted = [0, 0, 0]
        rewards: list[float] = []
        for pair in pairs:
            correct_task = pair.learner
            absent_task = _no_demonstration_task(correct_task)
            wrong_task = _wrong_demonstration_task(correct_task)
            correct_digest = procedural_skill_state_digest(correct_state)
            correct_proposal = _transition_proposal(
                policy,
                correct_task,
                correct_state,
                greedy=False,
                validated_state_digest=correct_digest,
            )
            reward = _score_attempt(pair, correct_proposal.answer, judge)
            correct_state, record = _commit_transition_feedback(
                policy,
                correct_task,
                correct_proposal,
                reward,
                correct_state,
                validated_state_digest=correct_digest,
            )
            accepted[0] += int(record["accepted"])

            with torch.no_grad():
                absent_digest = procedural_skill_state_digest(absent_state)
                absent_proposal = _transition_proposal(
                    policy,
                    absent_task,
                    absent_state,
                    greedy=False,
                    candidate_index=correct_proposal.candidate_index,
                    behavior_probabilities=(
                        correct_proposal.behavior_probabilities
                    ),
                    validated_state_digest=absent_digest,
                )
                absent_state, record = _commit_transition_feedback(
                    policy,
                    absent_task,
                    absent_proposal,
                    reward,
                    absent_state,
                    validated_state_digest=absent_digest,
                )
            accepted[1] += int(record["accepted"])

            wrong_digest = procedural_skill_state_digest(wrong_state)
            wrong_proposal = _transition_proposal(
                policy,
                wrong_task,
                wrong_state,
                greedy=False,
                candidate_index=correct_proposal.candidate_index,
                behavior_probabilities=correct_proposal.behavior_probabilities,
                validated_state_digest=wrong_digest,
            )
            wrong_state, record = _commit_transition_feedback(
                policy,
                wrong_task,
                wrong_proposal,
                reward,
                wrong_state,
                validated_state_digest=wrong_digest,
            )
            accepted[2] += int(record["accepted"])
            rewards.append(reward)
        return (correct_state, absent_state, wrong_state), {
            "presentations": len(pairs),
            "scalar_evaluator_calls": len(pairs),
            "correct_accepted": accepted[0],
            "no_demonstration_accepted": accepted[1],
            "wrong_demonstration_accepted": accepted[2],
            "mean_attempt_score": sum(rewards) / len(rewards),
            "matched_candidate": True,
            "matched_scalar_reused_across_controls": True,
        }

    accepted = [0, 0, 0]
    rewards: list[float] = []
    for pair in pairs:
        correct_task = pair.learner
        tasks = (
            correct_task,
            _no_demonstration_task(correct_task),
            _wrong_demonstration_task(correct_task),
        )
        state_digests = tuple(
            procedural_skill_state_digest(state) for state in states
        )
        proposal = _matched_transition_proposal(
            policy,
            tasks,
            states,
            validated_state_digests=state_digests,  # type: ignore[arg-type]
        )
        reward = _score_attempt(pair, proposal.answer, judge)
        states, records = _commit_matched_transition_feedback(
            policy,
            tasks,
            proposal,
            reward,
            states,
            validated_state_digests=state_digests,  # type: ignore[arg-type]
        )
        for row, record in enumerate(records):
            accepted[row] += int(record["accepted"])
        rewards.append(reward)
    return states, {
        "presentations": len(pairs),
        "scalar_evaluator_calls": len(pairs),
        "correct_accepted": accepted[0],
        "no_demonstration_accepted": accepted[1],
        "wrong_demonstration_accepted": accepted[2],
        "mean_attempt_score": sum(rewards) / len(rewards),
        "matched_candidate": True,
        "matched_scalar_reused_across_controls": True,
    }


def _acquire_stream_arms(
    policy: phase5.SkillMemoryPolicy,
    stream: ConditionalProcedureTransferStream,
    *,
    judge: ScalarJudge = score_conditional_procedure_answer,
    matched_arm_batch: bool = False,
) -> tuple[
    tuple[ProceduralSkillState, ProceduralSkillState, ProceduralSkillState],
    dict[str, Any],
]:
    """Run anchor, both components, and binding in declared stage order."""

    initial = (
        policy.initial_state(1),
        policy.initial_state(1),
        policy.initial_state(1),
    )
    states = initial
    records: dict[str, Any] = {}
    for name, pairs in (
        ("anchor", stream.anchor_supports),
        ("components", stream.component_supports),
        ("binding", stream.binding_supports),
    ):
        states, records[name] = _acquire_matched_stage(
            policy,
            states,
            pairs,
            judge=judge,
            matched_arm_batch=matched_arm_batch,
        )
    return states, records


def _stack_matched_query_states(
    states: tuple[
        ProceduralSkillState,
        ProceduralSkillState,
        ProceduralSkillState,
    ],
) -> ProceduralSkillState:
    """Stack correct/absent/wrong singleton rows without severing autograd."""

    if type(states) is not tuple or len(states) != 3:
        raise ValueError("matched query batching requires exactly three states")
    if any(
        not isinstance(state, ProceduralSkillState) or state.batch_size != 1
        for state in states
    ):
        raise ValueError("matched query batching requires singleton skill states")
    first = states[0]
    if any(
        state.slot_count != first.slot_count or state.width != first.width
        for state in states[1:]
    ):
        raise ValueError("matched query states must share one topology")

    fast_weights = replace(
        first.fast_weights,
        delta_y=torch.cat(
            tuple(state.fast_weights.delta_y for state in states),
            dim=0,
        ),
        delta_q=torch.cat(
            tuple(state.fast_weights.delta_q for state in states),
            dim=0,
        ),
        delta_k=torch.cat(
            tuple(state.fast_weights.delta_k for state in states),
            dim=0,
        ),
        delta_beta=torch.cat(
            tuple(state.fast_weights.delta_beta for state in states),
            dim=0,
        ),
    )
    return ProceduralSkillState(
        fast_weights=fast_weights,
        slot_latents=torch.cat(tuple(state.slot_latents for state in states), dim=0),
        key_offsets=torch.cat(tuple(state.key_offsets for state in states), dim=0),
        occupied=torch.cat(tuple(state.occupied for state in states), dim=0),
        write_counts=torch.cat(tuple(state.write_counts for state in states), dim=0),
    )


def _split_matched_arm_state(
    state: ProceduralSkillState,
) -> tuple[ProceduralSkillState, ProceduralSkillState, ProceduralSkillState]:
    """Split correct/no/wrong rows and retain autograd only for correct."""

    if not isinstance(state, ProceduralSkillState) or state.batch_size != 3:
        raise ValueError("matched state splitting requires exactly three rows")
    slots = state.slot_count
    rows: list[ProceduralSkillState] = []
    for row in range(3):
        fast_values: dict[str, torch.Tensor] = {}
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            value = getattr(state.fast_weights, name)
            selected = value.reshape(3, slots, *value.shape[1:])[row]
            fast_values[name] = selected if row == 0 else selected.detach()
        slot_latents = state.slot_latents[row : row + 1]
        key_offsets = state.key_offsets[row : row + 1]
        rows.append(
            ProceduralSkillState(
                fast_weights=replace(state.fast_weights, **fast_values),
                slot_latents=(
                    slot_latents if row == 0 else slot_latents.detach()
                ),
                key_offsets=(
                    key_offsets if row == 0 else key_offsets.detach()
                ),
                occupied=state.occupied[row : row + 1].detach(),
                write_counts=state.write_counts[row : row + 1].detach(),
            )
        )
    return tuple(rows)  # type: ignore[return-value]


def _matched_query_arm_logits(
    policy: phase5.SkillMemoryPolicy,
    public_task: PublicDemonstrationProcedureTask,
    states: tuple[
        ProceduralSkillState,
        ProceduralSkillState,
        ProceduralSkillState,
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score matched live arms together and the transition ablation separately."""

    correct_state, _, _ = states
    live = policy.score_task(
        public_task,
        _stack_matched_query_states(states),
        transition_only_composition=_TRANSITION_ONLY,
    ).logits
    expected = (3, len(phase5._PERMUTATIONS))
    if live.shape != expected:
        raise RuntimeError("matched query batch produced an invalid logit shape")
    with torch.no_grad():
        removed = policy.score_task(
            public_task,
            correct_state,
            include_reversible_transition=False,
            transition_only_composition=_TRANSITION_ONLY,
        ).logits
    if removed.shape != (1, len(phase5._PERMUTATIONS)):
        raise RuntimeError("reversible query control produced an invalid logit shape")
    return (
        live[0:1],
        live[1:2].detach(),
        live[2:3].detach(),
        removed.detach(),
    )


def _score_public_candidates(
    pair: GeneratedConditionalProcedureTask,
    candidate_indices: Sequence[int],
    judge: ScalarJudge = score_conditional_procedure_answer,
) -> tuple[float, ...]:
    """Score distinct public attempts without exposing evaluator internals."""

    return phase5._scalar_attempt_scores(pair, candidate_indices, judge)


def _matched_multi_candidate_objective(
    correct_logits: torch.Tensor,
    no_demonstration_logits: torch.Tensor,
    wrong_demonstration_logits: torch.Tensor,
    reversible_removed_logits: torch.Tensor,
    candidate_indices: Sequence[int],
    scalar_scores: Sequence[float],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Learn scalar preferences and require the correct public arm to win.

    Every arm is evaluated over the same public candidates and the same scalar
    observations.  This prevents a one-action calibration target from standing
    in for procedural ranking, while preserving causal correct/absent/wrong and
    reversible-transition controls.  Control losses are detached so the learner
    cannot satisfy a margin merely by making a control arm worse.
    """

    correct_loss, correct_edges = phase5._scalar_multi_preference_loss(
        correct_logits,
        candidate_indices,
        scalar_scores,
    )
    no_loss, no_edges = phase5._scalar_multi_preference_loss(
        no_demonstration_logits,
        candidate_indices,
        scalar_scores,
    )
    wrong_loss, wrong_edges = phase5._scalar_multi_preference_loss(
        wrong_demonstration_logits,
        candidate_indices,
        scalar_scores,
    )
    removed_loss, removed_edges = phase5._scalar_multi_preference_loss(
        reversible_removed_logits,
        candidate_indices,
        scalar_scores,
    )
    loss = (
        correct_loss
        + torch.relu(correct_loss - no_loss.detach() + _CONTROL_MARGIN)
        + torch.relu(correct_loss - wrong_loss.detach() + _CONTROL_MARGIN)
        + torch.relu(correct_loss - removed_loss.detach() + _CONTROL_MARGIN)
    )
    return loss, {
        "correct": correct_edges,
        "no_demonstration": no_edges,
        "wrong_demonstration": wrong_edges,
        "reversible_removed": removed_edges,
    }


def _train_conditional_interface(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
    meta_steps: int,
    supports_per_flag: int,
    queries_per_flag: int,
    learning_rate: float,
    judge: ScalarJudge = score_conditional_procedure_answer,
) -> dict[str, Any]:
    """Meta-train only the public sensory/latent interface on train pairs."""

    if isinstance(meta_steps, bool) or not isinstance(meta_steps, int) or meta_steps < 1:
        raise ValueError("meta_steps must be a positive integer")
    train_mechanisms = conditional_mechanism_partition("train")
    selected: list[Any] = []
    pass_index = 0
    while len(selected) < meta_steps:
        current = list(train_mechanisms)
        random.Random(seed + 1_000_003 * pass_index).shuffle(current)
        selected.extend(current[: meta_steps - len(selected)])
        pass_index += 1

    ports = getattr(policy, "public_fact_adapter", None)
    if not isinstance(ports, demonstration.TypedPublicFactPorts):
        raise RuntimeError("typed public-fact ports are not attached")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(demonstration._is_demonstration_trainable(name))
    trainable = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("conditional training selected no interface parameters")
    protected_before = demonstration._protected_state_fingerprint(policy)
    interface_before = phase5._named_state_fingerprint(
        policy,
        include=demonstration._is_demonstration_trainable,
        domain=b"project-angler.conditional-interface.v1",
    )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    gradients: list[float] = []
    evaluator_calls = 0
    query_attempts = 0
    observed_preference_edges = 0
    mechanism_commitments: list[str] = []
    ports.demonstration_adapter.train()
    policy.composition_memory.public_evidence_reader.train()

    for step, mechanism in enumerate(selected):
        episode_seed = seed + 100_003 * (step + 1)
        phase5._seed_reproducible_stage(
            episode_seed,
            "conditional-interface-training",
            next(policy.parameters()).device,
        )
        stream = make_conditional_procedure_transfer_stream(
            episode_seed,
            supports_per_flag=supports_per_flag,
            queries_per_flag=queries_per_flag,
            mechanism_pair=mechanism,
            mechanism_partition="train",
        )
        mechanism_commitments.append(stream.mechanism_commitment)
        states, acquisition = _acquire_stream_arms(
            policy,
            stream,
            judge=judge,
            matched_arm_batch=True,
        )
        evaluator_calls += sum(
            int(record["scalar_evaluator_calls"])
            for record in acquisition.values()
        )
        correct_state, absent_state, wrong_state = states
        query_losses: list[torch.Tensor] = []
        for query_index, pair in enumerate(stream.queries):
            (
                correct_logits,
                absent_logits,
                wrong_logits,
                removed_logits,
            ) = _matched_query_arm_logits(
                policy,
                pair.learner,
                (correct_state, absent_state, wrong_state),
            )
            candidates = phase5._on_policy_reward_candidate_set(
                correct_logits,
                step,
                query_index,
            )
            rewards = _score_public_candidates(pair, candidates, judge)
            evaluator_calls += len(candidates)
            query_attempts += len(candidates)
            query_loss, edges = _matched_multi_candidate_objective(
                correct_logits,
                absent_logits,
                wrong_logits,
                removed_logits,
                candidates,
                rewards,
            )
            query_losses.append(query_loss)
            observed_preference_edges += edges["correct"]
        loss = torch.stack(query_losses).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("conditional training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable,
            5.0,
            error_if_nonfinite=True,
        )
        if not any(
            parameter.grad is not None and bool(parameter.grad.count_nonzero())
            for parameter in trainable
        ):
            raise RuntimeError("conditional scalar outcomes reached no interface parameter")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradients.append(float(gradient.detach().item()))

    policy.requires_grad_(False)
    policy.eval()
    interface_after = phase5._named_state_fingerprint(
        policy,
        include=demonstration._is_demonstration_trainable,
        domain=b"project-angler.conditional-interface.v1",
    )
    protected_after = demonstration._protected_state_fingerprint(policy)
    if protected_after != protected_before:
        raise RuntimeError("conditional training changed protected prior capability")
    if interface_after == interface_before:
        raise RuntimeError("conditional interface did not change")
    return {
        "meta_steps": meta_steps,
        "mechanism_partition": "train",
        "mechanism_partition_size": len(train_mechanisms),
        "unique_mechanisms": len(set(mechanism_commitments)),
        "supports_per_flag": supports_per_flag,
        "queries_per_flag": queries_per_flag,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "mean_gradient_norm": sum(gradients) / len(gradients),
        "scalar_evaluator_calls": evaluator_calls,
        "total_scored_query_attempts": query_attempts,
        "attempts_per_training_query": _TRAINING_ATTEMPTS_PER_QUERY,
        "observed_preference_edges": observed_preference_edges,
        "one_scalar_per_attempt": True,
        "training_objective": "matched_multi_candidate_scalar_preference",
        "transition_only_composition": True,
        "target_or_normalized_procedure_read": False,
        "deterministic_solver_used": False,
        "protected_fingerprint_before": protected_before,
        "protected_fingerprint_after": protected_after,
        "interface_fingerprint_before": interface_before,
        "interface_fingerprint_after": interface_after,
    }


def _save_train_only_checkpoint(
    path: str | Path,
    *,
    policy: phase5.SkillMemoryPolicy,
    seed: int,
    base_sha256: str,
    precedence_sha256: str,
    source_interface_sha256: str,
    training: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path)
    if checkpoint.exists():
        raise FileExistsError(f"conditional checkpoint already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    adapter = policy.public_fact_adapter.demonstration_adapter
    training_digest = "sha256:" + hashlib.sha256(
        json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    torch.save(
        {
            "runner": _REPORT_VERSION,
            "stage": "train_only_before_final",
            "seed": seed,
            "base_checkpoint_sha256": base_sha256,
            "precedence_adapter_sha256": precedence_sha256,
            "source_interface_sha256": source_interface_sha256,
            "demonstration_adapter_model": adapter.state_dict(),
            "public_evidence_reader_model": (
                demonstration._public_evidence_reader_state(policy)
            ),
            "training": training,
            "training_digest": training_digest,
            "protected_fingerprint": (
                demonstration._protected_state_fingerprint(policy)
            ),
        },
        checkpoint,
    )
    return {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "runner": _REPORT_VERSION,
        "stage": "train_only_before_final",
        "training_digest": training_digest,
    }


def _load_train_only_checkpoint(
    path: str | Path,
    *,
    policy: phase5.SkillMemoryPolicy,
    base_sha256: str,
    precedence_sha256: str,
    source_interface_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = Path(path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "runner": _REPORT_VERSION,
        "stage": "train_only_before_final",
        "base_checkpoint_sha256": base_sha256,
        "precedence_adapter_sha256": precedence_sha256,
        "source_interface_sha256": source_interface_sha256,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError("conditional checkpoint binding is invalid")
    adapter_state = payload.get("demonstration_adapter_model")
    reader_state = payload.get("public_evidence_reader_model")
    training = payload.get("training")
    if not all(isinstance(value, dict) for value in (adapter_state, reader_state, training)):
        raise RuntimeError("conditional checkpoint payload is incomplete")
    policy.public_fact_adapter.demonstration_adapter.load_state_dict(
        adapter_state,
        strict=True,
    )
    demonstration._load_public_evidence_reader_state(policy, reader_state)
    policy.requires_grad_(False)
    policy.eval()
    if payload.get("protected_fingerprint") != (
        demonstration._protected_state_fingerprint(policy)
    ):
        raise RuntimeError("conditional checkpoint protected binding is invalid")
    return training, {
        "path": str(checkpoint),
        "sha256": digest,
        "runner": _REPORT_VERSION,
        "stage": "train_only_before_final",
        "training_digest": payload.get("training_digest"),
    }


def _evaluate_stream(
    policy: phase5.SkillMemoryPolicy,
    stream: ConditionalProcedureTransferStream,
    *,
    judge: ScalarJudge = score_conditional_procedure_answer,
) -> dict[str, Any]:
    """Evaluate acquisition, controls, and reversible ablation for one pair."""

    reset = policy.initial_state(1)
    states, acquisition = _acquire_stream_arms(policy, stream, judge=judge)
    correct_state, absent_state, wrong_state = states
    correct_digest = procedural_skill_state_digest(correct_state)
    query_arms = (
        ("correct", correct_state, True, correct_digest),
        (
            "no_demonstration",
            absent_state,
            True,
            procedural_skill_state_digest(absent_state),
        ),
        (
            "wrong_demonstration",
            wrong_state,
            True,
            procedural_skill_state_digest(wrong_state),
        ),
        ("reversible_removed", correct_state, False, correct_digest),
        ("reset", reset, True, procedural_skill_state_digest(reset)),
    )
    values = {
        "correct": [],
        "no_demonstration": [],
        "wrong_demonstration": [],
        "reversible_removed": [],
        "reset": [],
    }
    with torch.inference_mode():
        for pair in stream.queries:
            for name, state, include_reversible, state_digest in query_arms:
                proposal = _transition_proposal(
                    policy,
                    pair.learner,
                    state,
                    greedy=True,
                    include_reversible_transition=include_reversible,
                    validated_state_digest=state_digest,
                )
                values[name].append(_score_attempt(pair, proposal.answer, judge))
    summaries = {name: _summary(scores) for name, scores in values.items()}
    correct_mean = float(summaries["correct"]["mean"])
    return {
        "mechanism_commitment": stream.mechanism_commitment,
        "mechanism_partition": stream.mechanism_partition,
        "acquisition": acquisition,
        **summaries,
        "acquired_gain": correct_mean - float(summaries["reset"]["mean"]),
        "correct_gain_over_no_demonstration": (
            correct_mean - float(summaries["no_demonstration"]["mean"])
        ),
        "correct_gain_over_wrong_demonstration": (
            correct_mean - float(summaries["wrong_demonstration"]["mean"])
        ),
        "reversible_transition_gain": (
            correct_mean - float(summaries["reversible_removed"]["mean"])
        ),
    }


def _evaluate_final_panel(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
    supports_per_flag: int,
    queries_per_flag: int,
    judge: ScalarJudge = score_conditional_procedure_answer,
) -> dict[str, Any]:
    """Open and evaluate the predeclared final pairs only after freeze/reload."""

    records: list[dict[str, Any]] = []
    for index, mechanism in enumerate(conditional_mechanism_partition("final")):
        stream = make_conditional_procedure_transfer_stream(
            seed + 100_003 * (index + 1),
            supports_per_flag=supports_per_flag,
            queries_per_flag=queries_per_flag,
            mechanism_pair=mechanism,
            mechanism_partition="final",
        )
        records.append(_evaluate_stream(policy, stream, judge=judge))
    return {
        "partition": "final",
        "mechanisms": len(records),
        "mean_correct": sum(float(r["correct"]["mean"]) for r in records)
        / len(records),
        "mean_acquired_gain": sum(float(r["acquired_gain"]) for r in records)
        / len(records),
        "mean_correct_gain_over_no_demonstration": sum(
            float(r["correct_gain_over_no_demonstration"]) for r in records
        )
        / len(records),
        "mean_correct_gain_over_wrong_demonstration": sum(
            float(r["correct_gain_over_wrong_demonstration"]) for r in records
        )
        / len(records),
        "mean_reversible_transition_gain": sum(
            float(r["reversible_transition_gain"]) for r in records
        )
        / len(records),
        "records": records,
    }


def _load_runtime(
    *,
    device: torch.device,
    initial_checkpoint: str | Path,
    precedence_adapter_checkpoint: str | Path,
    demonstration_adapter_checkpoint: str | Path,
    compiler_checkpoint: str | Path,
) -> _Runtime:
    compiler, compiler_record = phase5._load_phase4_compiler(compiler_checkpoint)
    policy = phase5.SkillMemoryPolicy(
        phase5._PROFILES["composition"],
        compiler,
    ).to(device=device, dtype=torch.float32)
    base_record = phase5._load_initial_policy_checkpoint(
        policy,
        initial_checkpoint,
        phase5._PROFILES["composition"],
    )
    if not bool(policy.reversible_transition_mode.item()) or (
        base_record.get("source_stage") != "reversible_transition_acquisition"
    ):
        raise RuntimeError("conditional transfer requires the retained V51 core")
    precedence_adapter, precedence_record = demonstration._load_precedence_adapter(
        precedence_adapter_checkpoint,
        expected_base_sha256=base_record["sha256"],
    )
    (
        public_adapter,
        reader_state,
        _,
        source_interface_record,
    ) = demonstration._load_demonstration_adapter(
        demonstration_adapter_checkpoint,
        expected_base_sha256=base_record["sha256"],
        expected_precedence_sha256=precedence_record["sha256"],
    )
    policy.public_fact_adapter = demonstration.TypedPublicFactPorts(
        precedence_adapter,
        public_adapter,
    ).to(device=device, dtype=torch.float32)
    demonstration._attach_public_evidence_reader(policy)
    demonstration._load_public_evidence_reader_state(policy, reader_state)
    policy.requires_grad_(False)
    policy.eval()
    return _Runtime(
        policy=policy,
        base_record=base_record,
        precedence_record=precedence_record,
        source_interface_record=source_interface_record,
        compiler_record=compiler_record,
    )


def run(
    *,
    seed: int,
    device: str | torch.device,
    initial_checkpoint: str | Path,
    precedence_adapter_checkpoint: str | Path,
    demonstration_adapter_checkpoint: str | Path,
    compiler_checkpoint: str | Path = phase5._PHASE4_CHECKPOINT,
    meta_steps: int = _DEFAULT_META_STEPS,
    meta_supports_per_flag: int = 2,
    meta_queries_per_flag: int = 2,
    learning_rate: float = 4.0e-4,
    final_supports_per_flag: int = 16,
    final_queries_per_flag: int = 20,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Train, freeze/reload, then open the conditional final partition."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    target_device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    runtime = _load_runtime(
        device=target_device,
        initial_checkpoint=initial_checkpoint,
        precedence_adapter_checkpoint=precedence_adapter_checkpoint,
        demonstration_adapter_checkpoint=demonstration_adapter_checkpoint,
        compiler_checkpoint=compiler_checkpoint,
    )
    training = _train_conditional_interface(
        runtime.policy,
        seed=seed + 1_000_003,
        meta_steps=meta_steps,
        supports_per_flag=meta_supports_per_flag,
        queries_per_flag=meta_queries_per_flag,
        learning_rate=learning_rate,
    )
    saved = _save_train_only_checkpoint(
        checkpoint,
        policy=runtime.policy,
        seed=seed,
        base_sha256=runtime.base_record["sha256"],
        precedence_sha256=runtime.precedence_record["sha256"],
        source_interface_sha256=runtime.source_interface_record["sha256"],
        training=training,
    )
    loaded_training, loaded = _load_train_only_checkpoint(
        checkpoint,
        policy=runtime.policy,
        base_sha256=runtime.base_record["sha256"],
        precedence_sha256=runtime.precedence_record["sha256"],
        source_interface_sha256=runtime.source_interface_record["sha256"],
    )
    if loaded_training != training or loaded["sha256"] != saved["sha256"]:
        raise RuntimeError("conditional train-only checkpoint round trip changed")

    final_panel = _evaluate_final_panel(
        runtime.policy,
        seed=seed + 2_000_003,
        supports_per_flag=final_supports_per_flag,
        queries_per_flag=final_queries_per_flag,
    )
    result: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "seed": seed,
        "device": str(target_device),
        "base_checkpoint": runtime.base_record,
        "precedence_adapter_checkpoint": runtime.precedence_record,
        "source_demonstration_interface": runtime.source_interface_record,
        "compiler_checkpoint": runtime.compiler_record,
        "training": training,
        "train_only_checkpoint": loaded,
        "final_panel": final_panel,
        "claims": {
            "transition_only_composition": True,
            "ordinary_public_proposals": True,
            "one_scalar_evaluator_result_per_attempt": True,
            "hidden_solution_inspection": False,
            "deterministic_solver": False,
            "final_opened_after_train_checkpoint_reload": True,
            "broad_agi_claim": False,
        },
    }
    result["result_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=97_001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--precedence-adapter-checkpoint", required=True)
    parser.add_argument("--demonstration-adapter-checkpoint", required=True)
    parser.add_argument("--compiler-checkpoint", default=str(phase5._PHASE4_CHECKPOINT))
    parser.add_argument("--meta-steps", type=int, default=_DEFAULT_META_STEPS)
    parser.add_argument("--meta-supports-per-flag", type=int, default=2)
    parser.add_argument("--meta-queries-per-flag", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--final-supports-per-flag", type=int, default=16)
    parser.add_argument("--final-queries-per-flag", type=int, default=20)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            run(
                seed=args.seed,
                device=args.device,
                initial_checkpoint=args.initial_checkpoint,
                precedence_adapter_checkpoint=args.precedence_adapter_checkpoint,
                demonstration_adapter_checkpoint=(
                    args.demonstration_adapter_checkpoint
                ),
                compiler_checkpoint=args.compiler_checkpoint,
                meta_steps=args.meta_steps,
                meta_supports_per_flag=args.meta_supports_per_flag,
                meta_queries_per_flag=args.meta_queries_per_flag,
                learning_rate=args.learning_rate,
                final_supports_per_flag=args.final_supports_per_flag,
                final_queries_per_flag=args.final_queries_per_flag,
                checkpoint=args.checkpoint,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
