"""Train and evaluate Angler's first transferable causal operator compiler.

The runner learns from random interventional traces and externally executed
teacher trials.  Held-out action sequences are decoded and frozen before the
owning world executes them.  Search never runs inside held-out evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures.alignment import AliasTable  # noqa: E402
from angler.procedures.continual import ReservoirSampler, mix_replay  # noqa: E402
from angler.procedures.execution import (  # noqa: E402
    OperatorBinding,
    SharedPrimitiveSequenceDecoder,
    TypedEntityCandidate,
    canonicalize_binding_context,
)
from angler.procedures.expert_iteration import (  # noqa: E402
    TeacherPlan,
    TrialEvidence,
    TrialRequest,
    search_teacher_plan,
)
from angler.procedures.grounding import (  # noqa: E402
    GroundingError,
    GroundingLimitError,
    StateBindingAssignment,
    StateOperatorBinding,
    enumerate_operator_bindings,
    instantiate_operator,
)
from angler.procedures.induction import (  # noqa: E402
    OperatorCandidate,
    TraceSubsegment,
    TransitionDelta,
)
from angler.procedures.learning import (  # noqa: E402
    CompositeOperatorLearner,
    VerifiedOperatorExample,
    VerifiedOperatorTrajectory,
)
from angler.procedures.operators import Constant, LearnedOperator  # noqa: E402
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    Goal,
    GroundAction,
    State,
    Trace,
)
from angler.procedures.transfer import (  # noqa: E402
    CertifiedActionAdapter,
    CertifiedPredicateProjector,
    CertifiedTransferError,
    certified_transfer_binding,
)
from angler.procedures.trunk import NeuralOperatorCore  # noqa: E402
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402
from experiments.evaluators.causal_operator_suite import (  # noqa: E402
    OperatorCaseResult,
    OperatorChallenge,
    commit_action_sequence,
    evaluate_committed_sequence,
    make_heldout_operator_suite,
)
from experiments.evaluators.operator_alias_certificate import (  # noqa: E402
    certify_counterfactual_alignment,
)
from experiments.runners.causal_operator_experience import (  # noqa: E402
    CausalOperatorExperience,
    build_causal_operator_experience,
)


Domain = Literal["tokens", "files", "boxes"]
Ablation = Literal[
    "normal",
    "binding_permuted",
    "proposer_removed",
    "mirror_removed",
    "alias_removed",
    "operator_retired",
]
CANONICAL_DOMAIN: Domain = "tokens"
CANONICAL_NAMESPACE = tokens.NAMESPACE
_RUNNER_VERSION = "angler.phase4.causal-operator-compiler.v5"


@dataclass(frozen=True, slots=True)
class RunProfile:
    width: int
    hidden_width: int
    hash_width: int
    traces_per_domain: int
    training_seed_count: int
    training_cases_per_domain: int
    optimizer_steps_per_domain: int
    batch_size: int
    learning_rate: float
    replay_ratio: float
    reservoir_capacity: int
    progress_interval: int
    evaluation_seed_count: int
    evaluation_cases_per_domain: int
    selection_optimizer_steps: int
    selection_learning_rate: float
    competence_threshold: float
    incremental_training_cases_per_domain: int


PROFILES = {
    "smoke": RunProfile(
        width=64,
        hidden_width=96,
        hash_width=128,
        traces_per_domain=40,
        training_seed_count=2,
        training_cases_per_domain=4,
        optimizer_steps_per_domain=240,
        batch_size=8,
        learning_rate=2e-3,
        replay_ratio=0.25,
        reservoir_capacity=192,
        progress_interval=60,
        evaluation_seed_count=2,
        evaluation_cases_per_domain=2,
        selection_optimizer_steps=60,
        selection_learning_rate=5e-4,
        competence_threshold=0.90,
        incremental_training_cases_per_domain=4,
    ),
    "full": RunProfile(
        width=128,
        hidden_width=192,
        hash_width=192,
        traces_per_domain=64,
        training_seed_count=3,
        training_cases_per_domain=8,
        optimizer_steps_per_domain=720,
        batch_size=12,
        learning_rate=1e-3,
        replay_ratio=0.25,
        reservoir_capacity=512,
        progress_interval=90,
        evaluation_seed_count=3,
        evaluation_cases_per_domain=4,
        selection_optimizer_steps=360,
        selection_learning_rate=5e-4,
        competence_threshold=0.90,
        incremental_training_cases_per_domain=8,
    ),
    # Reproduces the protected sequence established during development:
    # broad initial acquisition, 40-case zero-shot/adaptation probes, and a
    # bounded 64-trajectory incremental stream.  It is opt-in so the smoke
    # profile remains inexpensive.
    "continual": RunProfile(
        width=64,
        hidden_width=96,
        hash_width=128,
        traces_per_domain=40,
        training_seed_count=2,
        training_cases_per_domain=64,
        optimizer_steps_per_domain=720,
        batch_size=8,
        learning_rate=2e-3,
        replay_ratio=0.25,
        reservoir_capacity=512,
        progress_interval=120,
        evaluation_seed_count=20,
        evaluation_cases_per_domain=2,
        selection_optimizer_steps=480,
        selection_learning_rate=5e-4,
        competence_threshold=0.90,
        incremental_training_cases_per_domain=64,
    ),
}


@dataclass(frozen=True, slots=True)
class DomainContext:
    domain: Domain
    candidate: OperatorCandidate
    canonical_operator: LearnedOperator
    projector: CertifiedPredicateProjector
    action_adapter: CertifiedActionAdapter

    @property
    def operator(self) -> LearnedOperator:
        return self.candidate.operator

    @property
    def canonical_schema(self) -> ActionSchema:
        return self.action_adapter.canonical_schema


@dataclass(frozen=True, slots=True)
class DirectProposal:
    actions: tuple[GroundAction, ...]
    candidate_counts: tuple[int, ...]
    score_margins: tuple[float | None, ...]
    decoder_stopped: tuple[bool, ...]
    failure: str | None = None


class AtomicTrialBoundary:
    """Execute a learned body, rolling back any partially applied proposal."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.traces: dict[str, Trace] = {}

    def __call__(self, request: TrialRequest) -> TrialEvidence:
        if request.origin.namespace != self.namespace:
            raise ValueError("trial crossed its configured world boundary")
        trace, fully_applied = _replay_actions(
            request.origin,
            request.actions,
            request.goal,
        )
        self.traces[request.digest] = trace
        observed = trace.final_state if fully_applied else request.origin
        return TrialEvidence(
            request_digest=request.digest,
            observed_state=observed,
            success=fully_applied and _satisfies(observed, request.goal),
            applied_actions=(len(request.actions) if fully_applied else 0),
            cost=len(request.actions),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=42_017)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _domain_for_namespace(namespace: str) -> Domain:
    mapping: dict[str, Domain] = {
        tokens.NAMESPACE: "tokens",
        files.NAMESPACE: "files",
        boxes.NAMESPACE: "boxes",
    }
    try:
        return mapping[namespace]
    except KeyError as error:
        raise ValueError("unsupported relational namespace") from error


def _executor(namespace: str):
    if namespace == tokens.NAMESPACE:
        return tokens.execute_token_action
    if namespace == files.NAMESPACE:
        return files.execute_file_action
    if namespace == boxes.NAMESPACE:
        return boxes.execute_box_action
    raise ValueError("unsupported relational namespace")


def _replay_actions(
    origin: State,
    actions: Sequence[GroundAction],
    goal: Goal | None = None,
) -> tuple[Trace, bool]:
    execute = _executor(origin.namespace)
    state = origin
    transitions = []
    for action in actions:
        transition = execute(state, action)
        transitions.append(transition)
        state = transition.after
    trace = Trace(origin, tuple(transitions), goal)
    return trace, bool(transitions) and all(item.applied for item in transitions)


def _satisfies(state: State, goal: Goal) -> bool:
    if state.namespace != goal.namespace:
        return False
    if goal.exact:
        return state.records == goal.required
    facts = set(state.records)
    return set(goal.required) <= facts and not (set(goal.forbidden) & facts)


def _selected_candidates(
    experience: CausalOperatorExperience,
) -> dict[Domain, OperatorCandidate]:
    return {
        _domain_for_namespace(item.operator.namespace): item
        for item in experience.selected_candidates
    }


def _make_context(
    domain: Domain,
    candidate: OperatorCandidate,
    canonical_operator: LearnedOperator,
    alias_table: AliasTable,
) -> DomainContext:
    return DomainContext(
        domain=domain,
        candidate=candidate,
        canonical_operator=canonical_operator,
        projector=CertifiedPredicateProjector(
            alias_table,
            canonical_operator.namespace,
            canonical_operator.namespace,
        ),
        action_adapter=CertifiedActionAdapter(
            alias_table,
            candidate.operator.body[0].schema,
            canonical_operator.body[0].schema,
        ),
    )


def _binding_from_exemplar(
    operator: LearnedOperator,
    segment: TraceSubsegment,
) -> StateOperatorBinding:
    exemplar = next(
        (
            item
            for item in operator.exemplars
            if item.trace_digest == segment.trace.digest
            and item.start_index == segment.start_index
            and item.stop_index == segment.stop_index
        ),
        None,
    )
    if exemplar is None or exemplar.reconstruction is None:
        raise ValueError("operator segment lacks exact reconstruction provenance")
    values = dict(exemplar.reconstruction.variable_bindings)
    if set(values) != {item.name for item in operator.variables}:
        raise ValueError("reconstruction does not bind every operator variable")
    return StateOperatorBinding(
        operator_digest=operator.digest,
        namespace=operator.namespace,
        assignments=tuple(
            StateBindingAssignment(variable, values[variable.name])
            for variable in operator.variables
        ),
    )


def _enumerate_semantic_candidates(
    context: DomainContext,
    state: State,
) -> tuple[
    tuple[OperatorBinding, ...],
    dict[str, StateOperatorBinding],
]:
    """Enumerate feature bindings without executing or judging any action."""

    grouped: dict[str, tuple[OperatorBinding, StateOperatorBinding]] = {}
    for local_binding in enumerate_operator_bindings(
        context.operator,
        state,
        maximum_bindings=512,
        maximum_match_attempts=50_000,
    ):
        try:
            instantiate_operator(context.operator, local_binding)
        except GroundingError:
            continue
        semantic = certified_transfer_binding(
            context.projector.alias_table,
            context.canonical_operator,
            context.operator,
            local_binding,
        )
        prior = grouped.get(semantic.digest)
        if prior is None or local_binding.digest < prior[1].digest:
            grouped[semantic.digest] = (semantic, local_binding)
    ordered = tuple(grouped[key] for key in sorted(grouped))
    return (
        tuple(item[0] for item in ordered),
        {item[0].digest: item[1] for item in ordered},
    )


def _labeled_semantic_candidates(
    context: DomainContext,
    state: State,
) -> tuple[
    tuple[OperatorBinding, ...],
    dict[str, StateOperatorBinding],
    dict[str, bool],
]:
    """Externally label training candidates, retaining an applicable witness."""

    grouped: dict[str, tuple[OperatorBinding, StateOperatorBinding, bool]] = {}
    for local_binding in enumerate_operator_bindings(
        context.operator,
        state,
        maximum_bindings=512,
        maximum_match_attempts=50_000,
    ):
        try:
            prediction = instantiate_operator(context.operator, local_binding)
        except GroundingError:
            continue
        _, fully_applied = _replay_actions(state, prediction.actions)
        semantic = certified_transfer_binding(
            context.projector.alias_table,
            context.canonical_operator,
            context.operator,
            local_binding,
        )
        prior = grouped.get(semantic.digest)
        if prior is None:
            grouped[semantic.digest] = (semantic, local_binding, fully_applied)
        elif fully_applied and not prior[2]:
            # Residual-only target variables can collapse to one semantic
            # binding.  Keep a real applicable witness when one exists.
            grouped[semantic.digest] = (prior[0], local_binding, True)
        elif fully_applied == prior[2] and local_binding.digest < prior[1].digest:
            grouped[semantic.digest] = (prior[0], local_binding, fully_applied)
    ordered = tuple(grouped[key] for key in sorted(grouped))
    return (
        tuple(item[0] for item in ordered),
        {item[0].digest: item[1] for item in ordered},
        {item[0].digest: item[2] for item in ordered},
    )


def _entity_candidates(
    bindings: Sequence[OperatorBinding],
    actions: Sequence[GroundAction],
) -> tuple[TypedEntityCandidate, ...]:
    values = {
        assignment.entity
        for binding in bindings
        for assignment in binding.assignments
    }
    for action in actions:
        values.update(
            TypedEntityCandidate(argument, parameter.type_name)
            for parameter, argument in zip(
                action.schema.parameters,
                action.arguments,
                strict=True,
            )
        )
    return tuple(sorted(values))


def _decoder_entities(
    bindings: Sequence[OperatorBinding],
    operator: LearnedOperator,
) -> tuple[TypedEntityCandidate, ...]:
    values = {
        assignment.entity
        for binding in bindings
        for assignment in binding.assignments
    }
    for pattern in operator.body:
        for parameter, term in zip(
            pattern.schema.parameters,
            pattern.arguments,
            strict=True,
        ):
            if isinstance(term, Constant):
                values.add(TypedEntityCandidate(term.value, parameter.type_name))
    return tuple(sorted(values))


def _make_verified_example(
    context: DomainContext,
    *,
    before: State,
    after: State,
    goal: Goal,
    positive_binding: StateOperatorBinding,
    verified_actions: tuple[GroundAction, ...],
    candidate_universe: Sequence[OperatorBinding] | None = None,
    candidate_witnesses: Mapping[str, StateOperatorBinding] | None = None,
) -> VerifiedOperatorExample:
    prediction = instantiate_operator(context.operator, positive_binding)
    if prediction.actions != verified_actions:
        raise ValueError("positive action range differs from binding provenance")
    replay, fully_applied = _replay_actions(before, verified_actions, goal)
    if not fully_applied or replay.final_state != after:
        raise ValueError("positive body did not replay to its full observed endpoint")

    if (candidate_universe is None) != (candidate_witnesses is None):
        raise ValueError("candidate universe and witnesses must be supplied together")
    if candidate_universe is None:
        semantic_bindings, _, labels = _labeled_semantic_candidates(context, before)
    else:
        semantic_bindings = tuple(candidate_universe)
        if not semantic_bindings:
            raise ValueError("candidate universe must not be empty")
        if len({item.digest for item in semantic_bindings}) != len(semantic_bindings):
            raise ValueError("candidate universe bindings must be unique")
        assert candidate_witnesses is not None
        if set(candidate_witnesses) != {item.digest for item in semantic_bindings}:
            raise ValueError("candidate witnesses must exactly cover the universe")
        labels = {}
        for semantic in semantic_bindings:
            local = candidate_witnesses[semantic.digest]
            try:
                candidate_prediction = instantiate_operator(context.operator, local)
                _, candidate_applied = _replay_actions(
                    before,
                    candidate_prediction.actions,
                )
            except (GroundingError, TypeError, ValueError):
                candidate_applied = False
            labels[semantic.digest] = candidate_applied
    positive = certified_transfer_binding(
        context.projector.alias_table,
        context.canonical_operator,
        context.operator,
        positive_binding,
    )
    semantic_by_digest = {item.digest: item for item in semantic_bindings}
    if positive.digest not in semantic_by_digest:
        raise ValueError("positive replay binding is absent from candidate enumeration")
    if not labels[positive.digest]:
        raise ValueError("positive binding is not externally applicable")
    ordered = tuple(semantic_by_digest[key] for key in sorted(semantic_by_digest))
    projected_actions = tuple(
        context.action_adapter.project_action(item) for item in verified_actions
    )
    return VerifiedOperatorExample(
        before=context.projector.project_state(before),
        after=context.projector.project_state(after),
        goal=context.projector.project_goal(goal),
        positive_binding=semantic_by_digest[positive.digest],
        candidate_bindings=ordered,
        applicability_labels=tuple(labels[item.digest] for item in ordered),
        verified_primitives=projected_actions,
        allowed_schemas=(context.canonical_schema,),
        entity_candidates=_entity_candidates(ordered, projected_actions),
    )


def _experience_examples(context: DomainContext) -> tuple[VerifiedOperatorExample, ...]:
    examples = []
    for segment in context.candidate.supporting_segments:
        binding = _binding_from_exemplar(context.operator, segment)
        # This training item is one complete observed operator application.
        # Its positive termination target is therefore the segment endpoint,
        # not the end of a longer trace that may contain later applications.
        # Sequence-level teacher examples below retain the negative
        # intermediate-state targets needed for multi-operator composition.
        goal = Goal.from_records(
            segment.namespace,
            segment.after.records,
            exact=True,
        )
        examples.append(
            _make_verified_example(
                context,
                before=segment.before,
                after=segment.after,
                goal=goal,
                positive_binding=binding,
                verified_actions=segment.actions,
            )
        )
    return tuple(examples)


def _verified_teacher_plan(
    challenge: OperatorChallenge,
    operator: LearnedOperator,
) -> tuple[TeacherPlan, AtomicTrialBoundary] | None:
    if challenge.maximum_steps % len(operator.body):
        return None
    boundary = AtomicTrialBoundary(challenge.origin.namespace)
    result = search_teacher_plan(
        challenge.origin,
        challenge.goal,
        (operator,),
        boundary,
        maximum_operator_depth=challenge.maximum_steps // len(operator.body),
        maximum_expansions=256,
        maximum_bindings_per_operator=512,
        maximum_match_attempts=50_000,
        order_by_goal_effect_overlap=True,
    )
    if result.plan is None:
        return None
    action_count = sum(len(item.request.actions) for item in result.plan.chain)
    if action_count > challenge.maximum_steps:
        return None
    for trial in result.plan.chain:
        trace = boundary.traces.get(trial.request.digest)
        if (
            trace is None
            or len(trace.transitions) != len(trial.request.actions)
            or not all(item.applied for item in trace.transitions)
            or trace.final_state != trial.evidence.observed_state
        ):
            return None
    actions = tuple(
        action
        for trial in result.plan.chain
        for action in trial.request.actions
    )
    independent = evaluate_committed_sequence(
        challenge,
        commit_action_sequence(challenge, actions),
    )
    if not independent.success or independent.applied_actions != len(actions):
        return None
    return result.plan, boundary


def _teacher_training_data(
    context: DomainContext,
    challenges: Iterable[OperatorChallenge],
) -> tuple[
    tuple[VerifiedOperatorExample, ...],
    tuple[VerifiedOperatorTrajectory, ...],
]:
    """Preserve verified plan grouping for multi-step latent supervision."""

    examples: list[VerifiedOperatorExample] = []
    trajectories: list[VerifiedOperatorTrajectory] = []
    for challenge in challenges:
        if challenge.domain != context.domain:
            continue
        verified = _verified_teacher_plan(challenge, context.operator)
        if verified is None:
            continue
        plan, boundary = verified
        origin_candidates, origin_witnesses = _enumerate_semantic_candidates(
            context,
            challenge.origin,
        )
        if not origin_candidates:
            raise RuntimeError("verified teacher plan has no origin candidate universe")
        trajectory_steps: list[VerifiedOperatorExample] = []
        for trial in plan.chain:
            trace = boundary.traces[trial.request.digest]
            example = _make_verified_example(
                context,
                before=trial.request.origin,
                after=trace.final_state,
                goal=trial.request.goal,
                positive_binding=trial.request.binding,
                verified_actions=trial.request.actions,
                candidate_universe=origin_candidates,
                candidate_witnesses=origin_witnesses,
            )
            examples.append(example)
            trajectory_steps.append(example)
        if len(trajectory_steps) >= 2:
            trajectories.append(VerifiedOperatorTrajectory(tuple(trajectory_steps)))
    return tuple(examples), tuple(trajectories)


def _teacher_examples(
    context: DomainContext,
    challenges: Iterable[OperatorChallenge],
) -> tuple[VerifiedOperatorExample, ...]:
    """Compatibility view over the atomic members of verified trajectories."""

    examples, _ = _teacher_training_data(context, challenges)
    return examples


def _certify_pair(
    experience: CausalOperatorExperience,
    source: DomainContext,
    target_candidate: OperatorCandidate,
    *,
    seed: int,
):
    candidate = next(
        item
        for item in experience.pairwise_alignments
        if item.source_operator_digest == source.operator.digest
        and item.target_operator_digest == target_candidate.operator.digest
    )
    suite = make_heldout_operator_suite(seed, cases_per_domain=2)
    source_challenge = next(
        item
        for item in suite
        if item.domain == source.domain and item.maximum_steps == 2
    )
    target_domain = _domain_for_namespace(target_candidate.operator.namespace)
    target_challenge = next(
        item
        for item in suite
        if item.domain == target_domain and item.maximum_steps == 2
    )
    source_verified = _verified_teacher_plan(source_challenge, source.operator)
    target_verified = _verified_teacher_plan(target_challenge, target_candidate.operator)
    if source_verified is None or target_verified is None:
        raise RuntimeError("counterfactual certificate worlds rejected an operator")
    source_plan, _ = source_verified
    target_plan, _ = target_verified
    if len(source_plan.chain) != 1 or len(target_plan.chain) != 1:
        raise RuntimeError("certificate requires one complete operator application")
    source_actions = source_plan.chain[0].request.actions
    target_actions = target_plan.chain[0].request.actions
    return certify_counterfactual_alignment(
        candidate,
        source.operator,
        target_candidate.operator,
        source_plan.chain[0].request.binding,
        target_plan.chain[0].request.binding,
        source_challenge,
        target_challenge,
        commit_action_sequence(source_challenge, source_actions),
        commit_action_sequence(target_challenge, target_actions),
    )


def _make_learner(
    profile: RunProfile,
    *,
    seed: int,
    device: torch.device,
) -> CompositeOperatorLearner:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    core = NeuralOperatorCore(
        width=profile.width,
        hidden_width=profile.hidden_width,
        schema_hash_width=profile.hash_width,
    )
    decoder = SharedPrimitiveSequenceDecoder(
        width=profile.width,
        hidden_width=profile.hidden_width,
        hash_width=profile.hash_width,
        maximum_steps=4,
    )
    return CompositeOperatorLearner(
        core,
        decoder,
        binding_hash_width=profile.hash_width,
        hidden_width=profile.hidden_width,
    ).to(device)


def _trajectory_teacher_forcing_ratio(step: int, total_steps: int) -> float:
    """Hold observed context early, then end with pure self-rollout."""

    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps <= 0
        or not 0 <= step < total_steps
    ):
        raise ValueError("trajectory schedule step must be inside total_steps")
    if total_steps == 1:
        return 0.0
    progress = step / (total_steps - 1)
    if progress <= 0.25:
        return 1.0
    if progress >= 0.75:
        return 0.0
    return 1.0 - ((progress - 0.25) / 0.5)


def _trajectory_loss_scale(step: int, total_steps: int) -> float:
    """Acquire atomic geometry before smoothly enabling trajectory pressure."""

    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps <= 0
        or not 0 <= step < total_steps
    ):
        raise ValueError("trajectory schedule step must be inside total_steps")
    if total_steps == 1:
        return 1.0
    progress = step / (total_steps - 1)
    if progress <= 0.25:
        return 0.0
    if progress >= 0.5:
        return 1.0
    return (progress - 0.25) / 0.25


def _train_steps(
    learner: CompositeOperatorLearner,
    optimizer: torch.optim.Optimizer,
    new_examples: Sequence[VerifiedOperatorExample],
    replay_examples: Sequence[VerifiedOperatorExample],
    *,
    new_trajectories: Sequence[VerifiedOperatorTrajectory] = (),
    replay_trajectories: Sequence[VerifiedOperatorTrajectory] = (),
    steps: int,
    batch_size: int,
    replay_ratio: float,
    seed: int,
    schedule_step_offset: int = 0,
    schedule_total_steps: int | None = None,
    trajectory_schedule: Literal["acquisition", "self_rollout"] = "acquisition",
) -> dict[str, Any]:
    if not new_examples:
        raise ValueError("training requires new externally verified examples")
    if schedule_total_steps is None:
        schedule_total_steps = steps
    if (
        schedule_step_offset < 0
        or schedule_total_steps <= 0
        or schedule_step_offset + steps > schedule_total_steps
    ):
        raise ValueError("trajectory schedule does not cover this optimizer range")
    if trajectory_schedule not in {"acquisition", "self_rollout"}:
        raise ValueError(
            "trajectory_schedule must be 'acquisition' or 'self_rollout'"
        )
    learner.train()
    losses = []
    atomic_losses = []
    trajectory_losses = []
    forcing_ratios = []
    trajectory_scales = []
    preclip_gradient_norms = []
    clip_coefficients = []
    started = time.perf_counter()
    for step in range(steps):
        ratio = replay_ratio if replay_examples else 0.0
        batch = mix_replay(
            new_examples,
            replay_examples,
            batch_size=batch_size,
            generated_ratio=ratio,
            seed=seed + step,
        )
        values = tuple(item.value for item in batch.items)
        trajectory_values: tuple[VerifiedOperatorTrajectory, ...] = ()
        if new_trajectories:
            trajectory_ratio = replay_ratio if replay_trajectories else 0.0
            trajectory_batch = mix_replay(
                new_trajectories,
                replay_trajectories,
                batch_size=max(1, batch_size // 2),
                generated_ratio=trajectory_ratio,
                seed=seed + 1_000_000 + step,
            )
            trajectory_values = tuple(
                item.value for item in trajectory_batch.items
            )
        elif replay_trajectories:
            trajectory_batch = mix_replay(
                replay_trajectories,
                (),
                batch_size=max(1, batch_size // 2),
                generated_ratio=0.0,
                seed=seed + 1_000_000 + step,
            )
            trajectory_values = tuple(
                item.value for item in trajectory_batch.items
            )

        optimizer.zero_grad(set_to_none=True)
        result = learner(values)
        total = result.total
        if trajectory_schedule == "self_rollout":
            forcing = 0.0
            trajectory_scale = 1.0
        else:
            forcing = _trajectory_teacher_forcing_ratio(
                schedule_step_offset + step,
                schedule_total_steps,
            )
            trajectory_scale = _trajectory_loss_scale(
                schedule_step_offset + step,
                schedule_total_steps,
            )
        if trajectory_values:
            forcing_ratios.append(forcing)
            trajectory_scales.append(trajectory_scale)
            if trajectory_scale > 0.0:
                trajectory_result = learner.trajectory_losses(
                    trajectory_values,
                    teacher_forcing_ratio=forcing,
                )
                total = total + trajectory_scale * trajectory_result.total
                trajectory_losses.append(
                    float(trajectory_result.total.detach().cpu().item())
                )
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError("operator learning produced a non-finite loss")
        total.backward()
        preclip = float(
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
            .detach()
            .cpu()
            .item()
        )
        if not math.isfinite(preclip):
            raise RuntimeError("operator learning produced a non-finite gradient")
        coefficient = min(1.0, 5.0 / max(preclip, 1e-12))
        preclip_gradient_norms.append(preclip)
        clip_coefficients.append(coefficient)
        if (
            len(clip_coefficients) >= 20
            and statistics.median(clip_coefficients[-20:]) < 0.1
        ):
            raise RuntimeError(
                "operator learning gradients remained over-clipped for 20 steps"
            )
        optimizer.step()
        losses.append(float(total.detach().cpu().item()))
        atomic_losses.append(float(result.total.detach().cpu().item()))
    return {
        "steps": steps,
        "trajectory_schedule": trajectory_schedule,
        "examples": len(new_examples),
        "replay_examples_available": len(replay_examples),
        "replay_ratio": replay_ratio if replay_examples else 0.0,
        "trajectories": len(new_trajectories),
        "replay_trajectories_available": len(replay_trajectories),
        "trajectory_teacher_forcing_first": (
            forcing_ratios[0] if forcing_ratios else None
        ),
        "trajectory_teacher_forcing_last": (
            forcing_ratios[-1] if forcing_ratios else None
        ),
        "trajectory_loss_scale_first": (
            trajectory_scales[0] if trajectory_scales else None
        ),
        "trajectory_loss_scale_last": (
            trajectory_scales[-1] if trajectory_scales else None
        ),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "tail_mean_loss": sum(losses[-min(20, len(losses)) :])
        / min(20, len(losses)),
        "first_atomic_loss": atomic_losses[0],
        "last_atomic_loss": atomic_losses[-1],
        "first_trajectory_loss": (
            trajectory_losses[0] if trajectory_losses else None
        ),
        "last_trajectory_loss": (
            trajectory_losses[-1] if trajectory_losses else None
        ),
        "first_preclip_gradient_norm": preclip_gradient_norms[0],
        "last_preclip_gradient_norm": preclip_gradient_norms[-1],
        "median_clip_coefficient": statistics.median(clip_coefficients),
        "minimum_clip_coefficient": min(clip_coefficients),
        "wall_seconds": time.perf_counter() - started,
    }


def _score_margin(logits: torch.Tensor) -> float | None:
    flattened = logits.detach().flatten()
    values = torch.sort(
        flattened[torch.isfinite(flattened)],
        descending=True,
    ).values
    if len(values) < 2:
        return None
    return float((values[0] - values[1]).cpu().item())


def _direct_proposal(
    learner: CompositeOperatorLearner,
    context: DomainContext,
    challenge: OperatorChallenge,
    *,
    ablation: Ablation = "normal",
) -> DirectProposal:
    if ablation == "mirror_removed" or (
        ablation == "operator_retired" and context.domain == "boxes"
    ):
        return DirectProposal((), (), (), (), "symbolic mirror unavailable")
    maximum_chunks = challenge.maximum_steps // len(context.operator.body)
    if maximum_chunks < 1:
        return DirectProposal((), (), (), (), "execution ceiling below operator body")

    learner.eval()
    actions: list[GroundAction] = []
    counts: list[int] = []
    margins: list[float | None] = []
    stopped: list[bool] = []
    with torch.no_grad():
        try:
            # This is the only symbolic operation in held-out proposal: a
            # finite candidate set from the public origin.  It predicts no
            # applicability or outcome and invokes no world executor.
            semantic, _ = _enumerate_semantic_candidates(
                context,
                challenge.origin,
            )
            if not semantic:
                return DirectProposal((), (), (), (), "no grounded candidate")
            feature_state = context.projector.project_state(challenge.origin)
            feature_goal = context.projector.project_goal(challenge.goal)
            state_embedding = learner.core.encode_states((feature_state,))[0]
            goal_embedding = learner.core.encode_goals((feature_goal,))[0]
            goal_state_embedding = learner.core.encode_goal_states((feature_goal,))[0]
            candidate_embeddings = learner.heads.encode_candidates(semantic)
            entities = _decoder_entities(semantic, context.canonical_operator)
            relative_contexts = tuple(
                canonicalize_binding_context(
                    feature_state,
                    feature_goal,
                    binding,
                )
                for binding in semantic
            )
            relative_states = learner.core.encode_states(
                tuple(item[0] for item in relative_contexts)
            )
            relative_goals = learner.core.encode_goals(
                tuple(item[1] for item in relative_contexts)
            )
        except (
            GroundingError,
            GroundingLimitError,
            CertifiedTransferError,
            TypeError,
            ValueError,
        ) as error:
            return DirectProposal((), (), (), (), str(error))

        used = torch.zeros(
            len(semantic),
            dtype=torch.bool,
            device=state_embedding.device,
        )
        task_terminated = False
        for chunk_index in range(maximum_chunks):
            counts.append(len(semantic) - int(used.sum().item()))
            mask = ~used
            if not bool(mask.any().item()):
                return DirectProposal(
                    (), tuple(counts), tuple(margins), tuple(stopped), "candidate set exhausted"
                )
            if chunk_index == 0:
                proposer = torch.diagonal(
                    learner.heads.binding_proposer.score_candidates(
                        relative_states,
                        relative_goals,
                        candidate_embeddings,
                    )
                )
                initiation = torch.diagonal(
                    learner.core.initiation_logits(
                        relative_states,
                        candidate_embeddings,
                    )
                )
            else:
                proposer = learner.heads.binding_proposer.score_candidates(
                    state_embedding.unsqueeze(0),
                    goal_embedding.unsqueeze(0),
                    candidate_embeddings,
                    mask=mask,
                )[0]
                initiation = learner.core.initiation_logits(
                    state_embedding.unsqueeze(0),
                    candidate_embeddings,
                )[0]
            forward_states = learner.core.predict_effects(
                state_embedding.unsqueeze(0),
                candidate_embeddings,
            )[0]
            backward_states = learner.core.predict_effects(
                goal_state_embedding.unsqueeze(0),
                candidate_embeddings,
                reverse=True,
            )[0]
            join = learner.horizon_agnostic_join_scores(
                forward_states,
                backward_states,
                goal_state_embedding,
                mask,
            )
            logits = learner.candidate_selection_logits(
                initiation,
                proposer,
                join,
                mask,
                include_proposer=ablation != "proposer_removed",
            )
            margins.append(_score_margin(logits))
            selected = int(torch.argmax(logits).item())
            available = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            if ablation == "binding_permuted" and len(available) > 1:
                position = available.index(selected)
                selected = int(available[(position + 1) % len(available)])
            semantic_binding = semantic[selected]
            try:
                decoded = learner.decoder.decode_sequence_greedy(
                    state_embedding,
                    goal_embedding,
                    semantic_binding,
                    (context.canonical_schema,),
                    entities,
                    maximum_steps=len(context.operator.body) + 1,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                return DirectProposal(
                    (), tuple(counts), tuple(margins), tuple(stopped), str(error)
                )
            stopped.append(decoded.stopped)
            if not decoded.stopped or len(decoded.actions) != len(context.operator.body):
                return DirectProposal(
                    (), tuple(counts), tuple(margins), tuple(stopped), "decoder body/STOP mismatch"
                )
            try:
                local_actions = tuple(
                    context.action_adapter.reverse_action(item)
                    for item in decoded.actions
                )
            except (CertifiedTransferError, TypeError, ValueError) as error:
                return DirectProposal(
                    (), tuple(counts), tuple(margins), tuple(stopped), str(error)
                )
            actions.extend(local_actions)
            state_embedding = forward_states[selected]
            used[selected] = True
            termination_logit = learner.core.termination_logits(
                state_embedding.unsqueeze(0),
                goal_state_embedding.unsqueeze(0),
            )[0]
            task_terminated = bool((termination_logit >= 0).item())
            if task_terminated:
                break
        if not task_terminated:
            return DirectProposal(
                (), tuple(counts), tuple(margins), tuple(stopped), "neural termination not reached"
            )
    return DirectProposal(tuple(actions), tuple(counts), tuple(margins), tuple(stopped))


def _evaluate_cases(
    learner: CompositeOperatorLearner,
    contexts: dict[Domain, DomainContext],
    cases: Sequence[tuple[int, OperatorChallenge]],
    *,
    ablation: Ablation = "normal",
    included_domains: set[Domain] | None = None,
) -> dict[str, Any]:
    rows = []
    for suite_seed, challenge in cases:
        if included_domains is not None and challenge.domain not in included_domains:
            continue
        context = contexts[challenge.domain]
        proposal_ablation = ablation
        if ablation == "alias_removed" and challenge.domain != CANONICAL_DOMAIN:
            context = _make_context(
                challenge.domain,
                context.candidate,
                context.operator,
                AliasTable(),
            )
            proposal_ablation = "normal"
        proposal = _direct_proposal(
            learner,
            context,
            challenge,
            ablation=proposal_ablation,
        )
        rejected_action_count = 0
        proposed_actions = proposal.actions
        if proposal.failure is not None or len(proposed_actions) > challenge.maximum_steps:
            rejected_action_count = len(proposed_actions)
            proposed_actions = ()
        result = evaluate_committed_sequence(
            challenge,
            commit_action_sequence(challenge, proposed_actions),
        )
        rows.append(
            {
                "seed": suite_seed,
                "case_id": challenge.case_id,
                "domain": challenge.domain,
                "step_ceiling": challenge.maximum_steps,
                "success": result.success,
                "tool_calls": result.tool_calls,
                "applied_actions": result.applied_actions,
                "blocked_actions": result.tool_calls - result.applied_actions,
                "proposal_failure": proposal.failure,
                "rejected_action_count": rejected_action_count,
                "candidate_counts": proposal.candidate_counts,
                "score_margins": proposal.score_margins,
                "decoder_stopped": proposal.decoder_stopped,
                "commitment_digest": result.commitment_digest,
                "trace_digest": result.trace_digest,
            }
        )
    return {"summary": _summarize_rows(rows), "cases": rows}


def _wilson(successes: int, attempts: int) -> tuple[float, float]:
    if attempts <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / attempts + z * z / (4 * attempts * attempts)
    ) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def one(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        attempts = len(values)
        successes = sum(bool(item["success"]) for item in values)
        calls = sum(int(item["tool_calls"]) for item in values)
        blocked = sum(int(item["blocked_actions"]) for item in values)
        return {
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts else 0.0,
            "case_level_wilson_95": _wilson(successes, attempts),
            "tool_calls": calls,
            "blocked_action_rate": blocked / calls if calls else 0.0,
            "proposal_time_world_executions": 0,
            "frozen_commitment_evaluations": attempts,
        }

    return {
        "overall": one(rows),
        "by_domain": {
            domain: one([item for item in rows if item["domain"] == domain])
            for domain in ("tokens", "files", "boxes")
            if any(item["domain"] == domain for item in rows)
        },
        "by_step_ceiling": {
            str(steps): one(
                [item for item in rows if item["step_ceiling"] == steps]
            )
            for steps in sorted({int(item["step_ceiling"]) for item in rows})
        },
        "by_seed": {
            str(seed): one([item for item in rows if item["seed"] == seed])
            for seed in sorted({int(item["seed"]) for item in rows})
        },
    }


def _suite_cases(
    seeds: Sequence[int],
    *,
    cases_per_domain: int,
) -> tuple[tuple[int, OperatorChallenge], ...]:
    return tuple(
        (seed, challenge)
        for seed in seeds
        for challenge in make_heldout_operator_suite(
            seed,
            cases_per_domain=cases_per_domain,
        )
    )


def _training_data_for_domain(
    context: DomainContext,
    training_cases: Sequence[tuple[int, OperatorChallenge]],
) -> tuple[
    tuple[VerifiedOperatorExample, ...],
    tuple[VerifiedOperatorTrajectory, ...],
]:
    experience = _experience_examples(context)
    teacher, trajectories = _teacher_training_data(
        context,
        (challenge for _, challenge in training_cases),
    )
    return experience + teacher, trajectories


def _examples_for_domain(
    context: DomainContext,
    training_cases: Sequence[tuple[int, OperatorChallenge]],
) -> tuple[VerifiedOperatorExample, ...]:
    """Compatibility view over all atomic domain examples."""

    examples, _ = _training_data_for_domain(context, training_cases)
    return examples


def _meets_competence(
    summary: Mapping[str, Any],
    *,
    threshold: float,
) -> bool:
    """Require demonstrated competence overall and at every tested horizon."""

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("competence threshold must be finite and inside [0, 1]")
    groups = (summary["overall"], *summary["by_step_ceiling"].values())
    return bool(groups) and all(
        int(group["attempts"]) > 0
        and float(group["success_rate"]) >= threshold
        for group in groups
    )


def _adapt_incremental_domain(
    learner: CompositeOperatorLearner,
    domain: Domain,
    examples: Sequence[VerifiedOperatorExample],
    replay: Sequence[VerifiedOperatorExample],
    trajectories: Sequence[VerifiedOperatorTrajectory],
    replay_trajectories: Sequence[VerifiedOperatorTrajectory],
    contexts: dict[Domain, DomainContext],
    adaptation_cases: Sequence[tuple[int, OperatorChallenge]],
    zero_shot_summary: Mapping[str, Any],
    profile: RunProfile,
    *,
    seed: int,
) -> dict[str, Any]:
    """Apply a bounded selection-only update only when transfer is deficient.

    The caller must perform the zero-shot evaluation before constructing the
    new domain's teacher examples.  This function receives that immutable
    summary, protects the acquired dynamics/decoder through the learner's
    selection plasticity scope, and reports prior-domain retention separately.
    """

    if domain not in contexts:
        raise ValueError("incremental domain is absent from its context map")
    prior_domains = set(contexts) - {domain}
    retention_before = (
        _evaluate_cases(
            learner,
            contexts,
            adaptation_cases,
            included_domains=prior_domains,
        )["summary"]
        if prior_domains
        else None
    )
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in learner.named_parameters()
    }
    curve: list[dict[str, Any]] = [
        {
            "optimizer_steps": 0,
            "evaluation": dict(zero_shot_summary),
            "competent": _meets_competence(
                zero_shot_summary,
                threshold=profile.competence_threshold,
            ),
        }
    ]
    receipts: list[dict[str, Any]] = []
    completed = 0
    enabled_names: tuple[str, ...] = ()

    if not curve[0]["competent"]:
        enabled_names = learner.configure_plasticity("selection")
        trainable = tuple(
            (name, parameter)
            for name, parameter in learner.named_parameters()
            if parameter.requires_grad
        )
        if tuple(name for name, _ in trainable) != enabled_names:
            raise RuntimeError(
                "selection scope and optimizer parameter identities diverged"
            )
        optimizer = torch.optim.AdamW(
            (parameter for _, parameter in trainable),
            lr=profile.selection_learning_rate,
            weight_decay=1e-4,
        )

    while completed < profile.selection_optimizer_steps and not curve[-1][
        "competent"
    ]:
        chunk = min(
            profile.progress_interval,
            profile.selection_optimizer_steps - completed,
        )
        receipts.append(
            _train_steps(
                learner,
                optimizer,
                examples,
                replay,
                new_trajectories=trajectories,
                replay_trajectories=replay_trajectories,
                steps=chunk,
                batch_size=profile.batch_size,
                replay_ratio=profile.replay_ratio,
                seed=seed + completed,
                schedule_step_offset=completed,
                schedule_total_steps=profile.selection_optimizer_steps,
                trajectory_schedule="self_rollout",
            )
        )
        completed += chunk
        evaluation = _evaluate_cases(
            learner,
            contexts,
            adaptation_cases,
            included_domains={domain},
        )["summary"]
        curve.append(
            {
                "optimizer_steps": completed,
                "evaluation": evaluation,
                "competent": _meets_competence(
                    evaluation,
                    threshold=profile.competence_threshold,
                ),
            }
        )

    changed_names = tuple(
        name
        for name, parameter in learner.named_parameters()
        if not torch.equal(before_parameters[name], parameter.detach())
    )
    unexpected = tuple(sorted(set(changed_names) - set(enabled_names)))
    if unexpected:
        raise RuntimeError(
            "selection-only adaptation changed protected parameters: "
            + ", ".join(unexpected)
        )
    retention_after = (
        retention_before
        if completed == 0
        else _evaluate_cases(
            learner,
            contexts,
            adaptation_cases,
            included_domains=prior_domains,
        )["summary"]
    )
    return {
        "zero_shot": dict(zero_shot_summary),
        "update_skipped": completed == 0,
        "decision": (
            "skipped_competent" if completed == 0 else "selection_adapted"
        ),
        "plasticity_scope": "none" if completed == 0 else "selection",
        "competence_threshold": profile.competence_threshold,
        "optimizer_steps": completed,
        "selection_learning_rate": (
            None if completed == 0 else profile.selection_learning_rate
        ),
        "enabled_parameter_names": enabled_names,
        "changed_parameter_names": changed_names,
        "unexpected_changed_parameter_names": unexpected,
        "curve": curve,
        "training_receipts": receipts,
        "after": curve[-1]["evaluation"],
        "competent_after": curve[-1]["competent"],
        "retention": {
            "domains": tuple(sorted(prior_domains)),
            "before": retention_before,
            "after": retention_after,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILES[args.profile]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.set_num_threads(1)

    started = time.perf_counter()
    experience = build_causal_operator_experience(
        seed=args.seed,
        traces_per_domain=profile.traces_per_domain,
    )
    selected = _selected_candidates(experience)
    canonical_operator = selected[CANONICAL_DOMAIN].operator
    alias_table = AliasTable()

    training_seeds = tuple(
        args.seed + 10_000 + index
        for index in range(profile.training_seed_count)
    )
    incremental_training_seeds = tuple(
        args.seed + 310_000 + index
        for index in range(profile.training_seed_count)
    )
    adaptation_seeds = tuple(
        args.seed + 330_000 + index
        for index in range(profile.evaluation_seed_count)
    )
    # Version 5 reserves fresh partitions for incremental experience,
    # adaptation decisions, and the final report.
    evaluation_seeds = tuple(
        args.seed + 370_000 + index
        for index in range(profile.evaluation_seed_count)
    )
    certificate_seeds = {
        "tokens_files": args.seed + 20_001,
        "files_boxes": args.seed + 20_002,
    }
    training_cases = _suite_cases(
        training_seeds,
        cases_per_domain=profile.training_cases_per_domain,
    )
    incremental_training_cases = _suite_cases(
        incremental_training_seeds,
        cases_per_domain=profile.incremental_training_cases_per_domain,
    )
    adaptation_cases = _suite_cases(
        adaptation_seeds,
        cases_per_domain=profile.evaluation_cases_per_domain,
    )
    final_cases = _suite_cases(
        evaluation_seeds,
        cases_per_domain=profile.evaluation_cases_per_domain,
    )
    partitions = {
        "initial_training": training_cases,
        "incremental_training": incremental_training_cases,
        "adaptation": adaptation_cases,
        "final": final_cases,
    }
    identities = {
        name: {challenge.case_id for _, challenge in cases}
        for name, cases in partitions.items()
    }
    for left_index, left in enumerate(identities):
        for right in tuple(identities)[left_index + 1 :]:
            if identities[left] & identities[right]:
                raise RuntimeError(
                    f"{left} and {right} case identities overlap"
                )

    learner = _make_learner(profile, seed=args.seed + 1, device=device)
    untrained = _make_learner(profile, seed=args.seed + 1, device=device)
    full_parameter_names = learner.configure_plasticity("full")
    full_parameters = tuple(
        parameter for parameter in learner.parameters() if parameter.requires_grad
    )
    if len(full_parameter_names) != len(full_parameters):
        raise RuntimeError("full plasticity did not expose every learner parameter")
    optimizer = torch.optim.AdamW(
        full_parameters,
        lr=profile.learning_rate,
        weight_decay=1e-4,
    )
    reservoir: ReservoirSampler[VerifiedOperatorExample] = ReservoirSampler(
        profile.reservoir_capacity,
        seed=args.seed + 2,
    )
    trajectory_reservoir: ReservoirSampler[VerifiedOperatorTrajectory] = (
        ReservoirSampler(
            profile.reservoir_capacity,
            seed=args.seed + 3,
        )
    )
    contexts: dict[Domain, DomainContext] = {}
    training_receipts: dict[str, Any] = {}
    incremental_results: dict[str, Any] = {}
    certificates = []

    token_context = _make_context(
        "tokens",
        selected["tokens"],
        canonical_operator,
        alias_table,
    )
    contexts["tokens"] = token_context
    token_examples, token_trajectories = _training_data_for_domain(
        token_context,
        training_cases,
    )
    training_receipts["tokens"] = _train_steps(
        learner,
        optimizer,
        token_examples,
        (),
        new_trajectories=token_trajectories,
        steps=profile.optimizer_steps_per_domain,
        batch_size=profile.batch_size,
        replay_ratio=0.0,
        seed=args.seed + 100,
    )
    reservoir.extend(token_examples)
    trajectory_reservoir.extend(token_trajectories)
    initial_acquisition_evaluation = _evaluate_cases(
        learner,
        contexts,
        adaptation_cases,
        included_domains={"tokens"},
    )["summary"]

    token_file_certificate = _certify_pair(
        experience,
        token_context,
        selected["files"],
        seed=certificate_seeds["tokens_files"],
    )
    if token_file_certificate.result != "pass":
        raise RuntimeError("Tokens-Files counterfactual certificate failed")
    token_file_alignment = next(
        item
        for item in experience.pairwise_alignments
        if item.source_operator_digest == selected["tokens"].operator.digest
        and item.target_operator_digest == selected["files"].operator.digest
    )
    alias_table = alias_table.with_certificate(
        token_file_alignment,
        token_file_certificate,
    )
    certificates.append(token_file_certificate.to_canonical())
    contexts["tokens"] = _make_context(
        "tokens", selected["tokens"], canonical_operator, alias_table
    )
    file_context = _make_context(
        "files", selected["files"], canonical_operator, alias_table
    )
    contexts["files"] = file_context
    file_zero_shot = _evaluate_cases(
        learner,
        contexts,
        adaptation_cases,
        included_domains={"files"},
    )["summary"]
    # Preserve the zero-shot boundary: no File teacher plan or verified File
    # example exists until after the transfer evaluation above is frozen.
    file_examples, file_trajectories = _training_data_for_domain(
        file_context,
        incremental_training_cases,
    )
    incremental_results["files"] = _adapt_incremental_domain(
        learner,
        "files",
        file_examples,
        reservoir.items,
        file_trajectories,
        trajectory_reservoir.items,
        contexts,
        adaptation_cases,
        file_zero_shot,
        profile,
        seed=args.seed + 200,
    )
    training_receipts["files"] = incremental_results["files"][
        "training_receipts"
    ]
    reservoir.extend(file_examples)
    trajectory_reservoir.extend(file_trajectories)

    file_box_certificate = _certify_pair(
        experience,
        file_context,
        selected["boxes"],
        seed=certificate_seeds["files_boxes"],
    )
    if file_box_certificate.result != "pass":
        raise RuntimeError("Files-Boxes counterfactual certificate failed")
    file_box_alignment = next(
        item
        for item in experience.pairwise_alignments
        if item.source_operator_digest == selected["files"].operator.digest
        and item.target_operator_digest == selected["boxes"].operator.digest
    )
    alias_table = alias_table.with_certificate(
        file_box_alignment,
        file_box_certificate,
    )
    certificates.append(file_box_certificate.to_canonical())
    contexts = {
        domain: _make_context(domain, selected[domain], canonical_operator, alias_table)
        for domain in ("tokens", "files")
    }
    contexts["boxes"] = _make_context(
        "boxes", selected["boxes"], canonical_operator, alias_table
    )
    box_zero_shot = _evaluate_cases(
        learner,
        contexts,
        adaptation_cases,
        included_domains={"boxes"},
    )["summary"]
    # As with Files, teacher evidence is constructed only after the zero-shot
    # result has determined whether an update is needed.
    box_examples, box_trajectories = _training_data_for_domain(
        contexts["boxes"],
        incremental_training_cases,
    )
    incremental_results["boxes"] = _adapt_incremental_domain(
        learner,
        "boxes",
        box_examples,
        reservoir.items,
        box_trajectories,
        trajectory_reservoir.items,
        contexts,
        adaptation_cases,
        box_zero_shot,
        profile,
        seed=args.seed + 300,
    )
    training_receipts["boxes"] = incremental_results["boxes"][
        "training_receipts"
    ]
    reservoir.extend(box_examples)
    trajectory_reservoir.extend(box_trajectories)

    final = _evaluate_cases(learner, contexts, final_cases)
    ablations = {
        name: _evaluate_cases(
            learner if name != "untrained" else untrained,
            contexts,
            final_cases,
            ablation=("normal" if name == "untrained" else name),
        )["summary"]
        for name in (
            "untrained",
            "binding_permuted",
            "proposer_removed",
            "mirror_removed",
            "alias_removed",
            "operator_retired",
        )
    }

    result = {
        "runner": _RUNNER_VERSION,
        "seed": args.seed,
        "profile": args.profile,
        "profile_values": asdict(profile),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "experience": experience.to_metadata(),
        "experience_digest": experience.digest,
        "selected_operator_digests": {
            domain: selected[domain].operator.digest
            for domain in ("tokens", "files", "boxes")
        },
        "certificates": certificates,
        "alias_table_digest": alias_table.digest,
        "training_examples": {
            "tokens": len(token_examples),
            "token_trajectories": len(token_trajectories),
            "files": len(file_examples),
            "file_trajectories": len(file_trajectories),
            "boxes": len(box_examples),
            "box_trajectories": len(box_trajectories),
            "reservoir_final": len(reservoir.items),
            "reservoir_seen": reservoir.seen_count,
            "trajectory_reservoir_final": len(trajectory_reservoir.items),
            "trajectory_reservoir_seen": trajectory_reservoir.seen_count,
        },
        "training": training_receipts,
        "initial_acquisition": {
            "domain": CANONICAL_DOMAIN,
            "plasticity_scope": "full",
            "enabled_parameter_names": full_parameter_names,
            "evaluation": initial_acquisition_evaluation,
        },
        "incremental": incremental_results,
        "retention": {
            domain: report["retention"]
            for domain, report in incremental_results.items()
        },
        "final": final,
        "ablations": ablations,
        "nonclaims": [
            "This is a bounded synthetic procedural-learning result, not AGI.",
            "No foundation-model weights, external data, or deployment were used.",
            "Symbolic mirrors only enumerate finite bindings; neural heads rank, roll latent state, terminate, and decode.",
            "Multi-step teacher plans supply training evidence only; held-out proposal receives no teacher route and runs learned self-rollouts.",
            "Competent transferred domains skip optimization; deficient domains expose only learned selection parameters.",
            "Teacher-search and certificate execution costs are training evidence and are not included in final tool-call totals.",
            "Removal ablations establish necessity in this slice, not sufficiency of any component.",
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    result["result_digest"] = _digest(result)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "alias_table": alias_table.to_canonical(),
                "model": learner.state_dict(),
                "profile": asdict(profile),
                "result_digest": result["result_digest"],
                "runner": _RUNNER_VERSION,
                "seed": args.seed,
            },
            args.checkpoint,
        )
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
