"""Fresh synthetic apprenticeship corpus for outcome-aware procedural learning.

The generator owns answers and structural identities, but the learner-facing
projection contains only natural-language experience, objective outcomes,
Moving-Origin coordinates, and a fresh challenge.  In particular, generator
families, mechanism references, topology signatures, and target candidate
indices are kept in separate metadata/supervision objects.

Deterministic generation is used only to construct and judge disposable
software-work episodes.  Nothing in this module selects a procedure for the
learner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal


CORPUS_ID = "angler.outcome-aware-apprenticeship.v5"
TRAIN_MECHANISMS = 32
DEVELOPMENT_MECHANISMS = 8
FINAL_MECHANISMS = 8
EPISODES_PER_MECHANISM = 6
CANDIDATES_PER_CHALLENGE = 4

Partition = Literal["train", "development", "final"]
OutcomeLabel = Literal["SUCCESS", "FAILURE"]
TemporalRelation = Literal["BEFORE", "AT", "AFTER"]

_GENERATION_SALT = b"project-angler.outcome-aware-apprenticeship.v5\x00"
_OUTCOME_PATTERNS: tuple[tuple[OutcomeLabel, ...], ...] = (
    ("FAILURE", "SUCCESS", "FAILURE", "SUCCESS", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "SUCCESS", "FAILURE", "FAILURE", "SUCCESS"),
    ("FAILURE", "SUCCESS", "SUCCESS", "FAILURE", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "FAILURE", "SUCCESS", "FAILURE", "SUCCESS"),
)


@dataclass(frozen=True, slots=True)
class MovingOriginCoordinates:
    """Stable acquisition coordinate resolved at the stream's query point."""

    acquired_ordinal: int
    age: int
    landmark_relations: tuple[tuple[str, TemporalRelation], ...]

    def __post_init__(self) -> None:
        if self.acquired_ordinal < 0 or self.age < 0:
            raise ValueError("Moving-Origin coordinates must be non-negative")
        if not self.landmark_relations:
            raise ValueError("at least one landmark relation is required")
        if any(
            type(name) is not str
            or not name
            or relation not in ("BEFORE", "AT", "AFTER")
            for name, relation in self.landmark_relations
        ):
            raise ValueError("landmark relations are invalid")


@dataclass(frozen=True, slots=True)
class OutcomeEpisode:
    """One chronological learner-visible task, trace, and objective outcome."""

    task_text: str
    request_text: str
    action_trace_text: str
    outcome_label: OutcomeLabel
    objective_diagnostic_text: str
    temporal: MovingOriginCoordinates

    def __post_init__(self) -> None:
        for label, value in (
            ("task_text", self.task_text),
            ("request_text", self.request_text),
            ("action_trace_text", self.action_trace_text),
            ("objective_diagnostic_text", self.objective_diagnostic_text),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be non-empty natural language")
        if self.outcome_label not in ("SUCCESS", "FAILURE"):
            raise ValueError("outcome_label is invalid")

    @property
    def outcome_value(self) -> float:
        return 1.0 if self.outcome_label == "SUCCESS" else -1.0

    def to_learner_payload(self) -> dict[str, object]:
        return {
            "task_text": self.task_text,
            "request_text": self.request_text,
            "action_trace_text": self.action_trace_text,
            "outcome_label": self.outcome_label,
            "objective_diagnostic_text": self.objective_diagnostic_text,
            "temporal": {
                "acquired_ordinal": self.temporal.acquired_ordinal,
                "age": self.temporal.age,
                "landmark_relations": [list(row) for row in self.temporal.landmark_relations],
            },
        }


@dataclass(frozen=True, slots=True)
class ApprenticeshipChallenge:
    """Fresh structural variant presented after the chronological episodes."""

    task_text: str
    request_text: str
    candidate_action_trace_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.task_text) is not str or not self.task_text.strip():
            raise ValueError("challenge task_text must be non-empty")
        if type(self.request_text) is not str or not self.request_text.strip():
            raise ValueError("challenge request_text must be non-empty")
        if (
            type(self.candidate_action_trace_texts) is not tuple
            or len(self.candidate_action_trace_texts) < CANDIDATES_PER_CHALLENGE
            or any(type(value) is not str or not value.strip() for value in self.candidate_action_trace_texts)
            or len(set(self.candidate_action_trace_texts)) != len(self.candidate_action_trace_texts)
        ):
            raise ValueError("challenge candidates must be unique non-empty traces")

    def to_learner_payload(self) -> dict[str, object]:
        return {
            "task_text": self.task_text,
            "request_text": self.request_text,
            "candidate_action_trace_texts": list(self.candidate_action_trace_texts),
        }


@dataclass(frozen=True, slots=True)
class PublicApprenticeshipStream:
    """The only portion of one mechanism that may become learner features."""

    episodes: tuple[OutcomeEpisode, ...]
    challenge: ApprenticeshipChallenge

    def __post_init__(self) -> None:
        if type(self.episodes) is not tuple or len(self.episodes) != EPISODES_PER_MECHANISM:
            raise ValueError("each stream must contain the frozen episode count")
        if any(not isinstance(value, OutcomeEpisode) for value in self.episodes):
            raise TypeError("stream episodes contain an invalid value")
        ordinals = tuple(value.temporal.acquired_ordinal for value in self.episodes)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise ValueError("stream chronology must be contiguous")
        ages = tuple(value.temporal.age for value in self.episodes)
        if ages != tuple(reversed(range(len(self.episodes)))):
            raise ValueError("episode ages must resolve at the immediate query point")
        outcomes = tuple(value.outcome_label for value in self.episodes)
        if outcomes.count("SUCCESS") != 3 or outcomes.count("FAILURE") != 3:
            raise ValueError("each stream must contain balanced mixed outcomes")
        if outcomes[:3] in (("SUCCESS",) * 3, ("FAILURE",) * 3):
            raise ValueError("chronology cannot group outcomes by label")

    def to_learner_payload(self) -> dict[str, object]:
        """Return JSON-ready data with no generator metadata or supervision."""

        return {
            "episodes": [value.to_learner_payload() for value in self.episodes],
            "challenge": self.challenge.to_learner_payload(),
        }


@dataclass(frozen=True, slots=True)
class MechanismSupervision:
    """Evaluator/outer-loss labels; never candidate input features."""

    target_successful_candidate_indices: tuple[int, ...]
    relevant_failed_candidate_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, values in (
            ("target_successful_candidate_indices", self.target_successful_candidate_indices),
            ("relevant_failed_candidate_indices", self.relevant_failed_candidate_indices),
        ):
            if type(values) is not tuple or not values:
                raise ValueError(f"{label} must be a non-empty tuple")
            if tuple(sorted(set(values))) != values or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise ValueError(f"{label} must contain unique sorted indices")
        if set(self.target_successful_candidate_indices) & set(
            self.relevant_failed_candidate_indices
        ):
            raise ValueError("successful and failed candidate labels must be disjoint")


@dataclass(frozen=True, slots=True)
class GeneratorMetadata:
    """Non-feature partition and structural provenance."""

    partition: Partition
    mechanism_ref: str
    generator_family: str
    support_structure_signature: str
    challenge_structure_signature: str
    heldout_variant: str

    def __post_init__(self) -> None:
        if self.partition not in ("train", "development", "final"):
            raise ValueError("partition is invalid")
        for value in (
            self.mechanism_ref,
            self.generator_family,
            self.support_structure_signature,
            self.challenge_structure_signature,
            self.heldout_variant,
        ):
            if type(value) is not str or not value:
                raise ValueError("generator metadata must be non-empty text")
        if self.support_structure_signature == self.challenge_structure_signature:
            raise ValueError("the challenge must be a held-out structural variant")


@dataclass(frozen=True, slots=True)
class CorpusMechanism:
    public: PublicApprenticeshipStream
    supervision: MechanismSupervision
    metadata: GeneratorMetadata

    def __post_init__(self) -> None:
        candidate_count = len(self.public.challenge.candidate_action_trace_texts)
        indices = (
            *self.supervision.target_successful_candidate_indices,
            *self.supervision.relevant_failed_candidate_indices,
        )
        if any(index >= candidate_count for index in indices):
            raise ValueError("supervision index exceeds the candidate count")

    @property
    def gradient_authorized(self) -> bool:
        return self.metadata.partition == "train"

    def to_learner_payload(self) -> dict[str, object]:
        return self.public.to_learner_payload()


@dataclass(frozen=True, slots=True)
class OutcomeAwareApprenticeshipCorpus:
    train: tuple[CorpusMechanism, ...]
    development: tuple[CorpusMechanism, ...]
    final: tuple[CorpusMechanism, ...]

    def __post_init__(self) -> None:
        expected = {
            "train": TRAIN_MECHANISMS,
            "development": DEVELOPMENT_MECHANISMS,
            "final": FINAL_MECHANISMS,
        }
        refs: set[str] = set()
        families: dict[str, set[str]] = {}
        signatures: dict[str, set[tuple[str, str]]] = {}
        for partition in ("train", "development", "final"):
            values = getattr(self, partition)
            if type(values) is not tuple or len(values) != expected[partition]:
                raise ValueError(f"{partition} has the wrong mechanism count")
            if any(value.metadata.partition != partition for value in values):
                raise ValueError("mechanism is stored in the wrong partition")
            local_refs = {value.metadata.mechanism_ref for value in values}
            if len(local_refs) != len(values) or refs & local_refs:
                raise ValueError("mechanism references must be globally disjoint")
            refs.update(local_refs)
            families[partition] = {value.metadata.generator_family for value in values}
            signatures[partition] = {
                (
                    value.metadata.support_structure_signature,
                    value.metadata.challenge_structure_signature,
                )
                for value in values
            }
        for left, right in (
            ("train", "development"),
            ("train", "final"),
            ("development", "final"),
        ):
            if families[left] & families[right] or signatures[left] & signatures[right]:
                raise ValueError("partitions must use generator-disjoint structures")

    def partition(self, name: Partition) -> tuple[CorpusMechanism, ...]:
        if name not in ("train", "development", "final"):
            raise ValueError("partition is invalid")
        return getattr(self, name)

    @property
    def all_mechanisms(self) -> tuple[CorpusMechanism, ...]:
        return (*self.train, *self.development, *self.final)

    def learner_payload_size_bytes(self) -> int:
        payload = [value.to_learner_payload() for value in self.all_mechanisms]
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _FamilySpec:
    key: str
    partition: Partition
    support_steps: tuple[str, ...]
    challenge_steps: tuple[str, ...]
    heldout_variant: str
    heldout_description: str

    @property
    def support_signature(self) -> str:
        return _structure_signature(self.support_steps)

    @property
    def challenge_signature(self) -> str:
        return _structure_signature(self.challenge_steps)


_FAMILIES: tuple[_FamilySpec, ...] = (
    _FamilySpec(
        "bounded_checkpoint",
        "train",
        ("inspect", "modify", "validate"),
        ("inspect", "checkpoint", "modify", "validate"),
        "inserted-checkpoint-boundary",
        "a reversible checkpoint is now required between inspection and modification",
    ),
    _FamilySpec(
        "dependency_refresh",
        "train",
        ("inspect", "refresh_dependency", "modify", "integration_test"),
        ("inspect", "isolate", "refresh_dependency", "modify", "integration_test"),
        "isolated-dependency-refresh",
        "dependency refresh must now occur inside an isolated workspace",
    ),
    _FamilySpec(
        "guarded_update",
        "train",
        ("inspect", "guard", "modify", "validate"),
        ("inspect", "guard", "modify", "observe", "validate"),
        "post-change-observation",
        "an observation barrier now separates the change from final validation",
    ),
    _FamilySpec(
        "staged_activation",
        "train",
        ("checkpoint", "stage", "validate", "activate"),
        ("checkpoint", "stage", "unit_test", "integration_test", "activate"),
        "split-verification-gates",
        "the former validation gate is split into unit and integration checks",
    ),
    _FamilySpec(
        "branch_reconciliation",
        "development",
        ("inspect", "branch_left", "branch_right", "reconcile", "validate"),
        ("checkpoint", "inspect", "branch_left", "branch_right", "reconcile", "validate"),
        "checkpointed-diamond-merge",
        "the independent branches now share a checkpointed reconciliation boundary",
    ),
    _FamilySpec(
        "cache_reconstruction",
        "development",
        ("inspect", "invalidate", "modify", "rebuild", "validate"),
        ("inspect", "invalidate", "modify", "rebuild", "observe", "validate"),
        "observed-cache-rebuild",
        "the rebuilt cache must now be observed before acceptance",
    ),
    _FamilySpec(
        "migration_with_rollback_probe",
        "final",
        ("checkpoint", "migrate", "audit", "activate"),
        ("checkpoint", "migrate", "audit", "rollback_probe", "activate"),
        "pre-activation-rollback-probe",
        "activation now requires a successful rollback probe after the audit",
    ),
    _FamilySpec(
        "quorum_publication",
        "final",
        ("inspect", "stage_left", "stage_right", "stage_aux", "quorum", "activate"),
        ("inspect", "stage_left", "stage_right", "stage_aux", "quorum", "audit", "activate"),
        "audited-quorum-publication",
        "the quorum decision now requires an audit before publication",
    ),
)

_SYSTEMS = (
    "catalog service",
    "event processor",
    "policy adapter",
    "document index",
    "queue consumer",
    "configuration compiler",
    "artifact registry",
    "workflow gateway",
    "schema translator",
    "test coordinator",
    "cache manager",
    "release planner",
)
_ARTIFACTS = (
    "routing manifest",
    "compatibility layer",
    "validation profile",
    "dependency lock",
    "state projection",
    "interface contract",
    "generated index",
    "migration map",
)
_CHANGES = (
    "support the revised record shape",
    "preserve a newly required output field",
    "adopt the updated dependency boundary",
    "repair stale state propagation",
    "accept the reordered public contract",
    "handle the new optional branch",
    "separate preparation from activation",
    "retain compatibility during a format transition",
)
_CONSTRAINTS = (
    "the existing rollback path",
    "unrelated public behavior",
    "the declared interface boundary",
    "previously accepted records",
    "the read-only source contract",
    "the bounded workspace scope",
    "the stable serialization format",
    "the existing error semantics",
)
_CHECKS = (
    "the contract suite",
    "the integration harness",
    "the replay check",
    "the compatibility test",
    "the state-transition verifier",
    "the package acceptance test",
    "the dependency audit",
    "the reconstruction check",
)


def _rng(*parts: object) -> random.Random:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    digest = hashlib.sha256(_GENERATION_SALT + payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _structure_signature(steps: tuple[str, ...]) -> str:
    edges = tuple(f"{left}>{right}" for left, right in zip(steps, steps[1:]))
    return "|".join((f"nodes:{','.join(steps)}", f"edges:{','.join(edges)}"))


def _context(family: _FamilySpec, mechanism_index: int, surface_index: int) -> dict[str, str]:
    rng = _rng(family.key, mechanism_index, surface_index, "surface")
    return {
        "system": _SYSTEMS[rng.randrange(len(_SYSTEMS))],
        "artifact": _ARTIFACTS[rng.randrange(len(_ARTIFACTS))],
        "change": _CHANGES[(mechanism_index + surface_index) % len(_CHANGES)],
        "constraint": _CONSTRAINTS[(2 * mechanism_index + surface_index) % len(_CONSTRAINTS)],
        "check": _CHECKS[(3 * mechanism_index + surface_index) % len(_CHECKS)],
    }


def _task_text(context: dict[str, str], *, heldout_description: str | None = None) -> str:
    text = (
        f"The synthetic {context['system']} workspace contains a {context['artifact']}. "
        f"It must {context['change']} while preserving {context['constraint']}."
    )
    if heldout_description is not None:
        text += f" In this fresh variant, {heldout_description}."
    return text


def _request_text(context: dict[str, str]) -> str:
    return (
        f"Complete the requested change and leave the workspace in a state where "
        f"{context['check']} passes without weakening {context['constraint']}."
    )


def _step_text(role: str, context: dict[str, str]) -> str:
    values = {
        "inspect": f"Inspect the {context['artifact']} and record its current dependency boundary.",
        "checkpoint": "Create a bounded reversible checkpoint before changing workspace state.",
        "modify": f"Update the {context['artifact']} to {context['change']}.",
        "validate": f"Run {context['check']} against the completed workspace.",
        "refresh_dependency": "Refresh the declared dependency projection from its authoritative source.",
        "isolate": "Create an isolated work area for the dependency-sensitive change.",
        "guard": f"Confirm that {context['constraint']} is protected by an explicit guard.",
        "observe": "Observe the resulting state and record the externally visible transition.",
        "stage": f"Stage the revised {context['artifact']} without activating it.",
        "unit_test": "Run the focused unit checks for the changed boundary.",
        "integration_test": f"Run {context['check']} across the integrated dependency path.",
        "branch_left": "Apply the first independent branch of the requested change.",
        "branch_right": "Apply the second independent branch of the requested change.",
        "reconcile": "Reconcile both branch results into one consistent workspace state.",
        "invalidate": "Invalidate the stale derived state before producing a replacement.",
        "rebuild": "Rebuild the derived state from the revised authoritative inputs.",
        "migrate": f"Migrate the {context['artifact']} to the revised representation.",
        "audit": f"Audit the result against {context['constraint']} and the declared request.",
        "rollback_probe": "Exercise the bounded rollback probe and restore the candidate state.",
        "stage_left": "Stage the first independent publication shard.",
        "stage_right": "Stage the second independent publication shard.",
        "stage_aux": "Stage the auxiliary publication shard.",
        "quorum": "Confirm that the staged shards agree on one publication state.",
        "activate": "Activate the already verified candidate state.",
        "blind_change": f"Change the {context['artifact']} immediately without inspecting its boundary.",
    }
    try:
        return values[role]
    except KeyError as exc:
        raise ValueError(f"unknown synthetic procedure role: {role}") from exc


def _trace_text(steps: tuple[str, ...], context: dict[str, str]) -> str:
    return " ".join(
        f"Step {index + 1}: {_step_text(role, context)}"
        for index, role in enumerate(steps)
    )


def _failed_steps(steps: tuple[str, ...], mode: str) -> tuple[str, ...]:
    if mode == "reordered":
        values = list(steps)
        left = 1 if len(values) > 3 else 0
        right = left + 1
        values[left], values[right] = values[right], values[left]
        return tuple(values)
    if mode == "omitted":
        return steps[:-1]
    if mode == "bypass":
        return ("blind_change", *steps[1:])
    raise ValueError("unknown failure construction")


def _diagnostic(context: dict[str, str], outcome: OutcomeLabel, failure_mode: str) -> str:
    if outcome == "SUCCESS":
        return (
            f"Objective verification passed: {context['check']} accepted the result and "
            f"{context['constraint']} remained intact."
        )
    messages = {
        "reordered": (
            f"Objective verification failed: a dependent operation observed stale state, "
            f"and {context['check']} rejected the workspace."
        ),
        "omitted": (
            "Objective verification failed: completion evidence was missing, so the "
            "workspace could not be accepted."
        ),
        "bypass": (
            f"Objective verification failed: the trace bypassed a declared boundary and "
            f"did not preserve {context['constraint']}."
        ),
    }
    return messages[failure_mode]


def _build_mechanism(
    family: _FamilySpec,
    mechanism_index: int,
    *,
    ordinal_start: int,
) -> CorpusMechanism:
    outcomes = _OUTCOME_PATTERNS[mechanism_index % len(_OUTCOME_PATTERNS)]
    failure_modes = ("reordered", "omitted", "bypass")
    failure_cursor = 0
    episodes = []
    for episode_index, outcome in enumerate(outcomes):
        context = _context(family, mechanism_index, episode_index)
        if outcome == "SUCCESS":
            steps = family.support_steps
            mode = "reordered"  # ignored by the successful diagnostic
        else:
            mode = failure_modes[failure_cursor % len(failure_modes)]
            failure_cursor += 1
            steps = _failed_steps(family.support_steps, mode)
        episodes.append(
            OutcomeEpisode(
                task_text=_task_text(context),
                request_text=_request_text(context),
                action_trace_text=_trace_text(steps, context),
                outcome_label=outcome,
                objective_diagnostic_text=_diagnostic(context, outcome, mode),
                temporal=MovingOriginCoordinates(
                    acquired_ordinal=ordinal_start + episode_index,
                    age=EPISODES_PER_MECHANISM - episode_index - 1,
                    landmark_relations=(
                        (
                            "stream_start",
                            "AT" if episode_index == 0 else "AFTER",
                        ),
                    ),
                ),
            )
        )

    challenge_context = _context(family, mechanism_index, EPISODES_PER_MECHANISM + 1)
    candidate_rows = [
        ("target", _trace_text(family.challenge_steps, challenge_context)),
        (
            "failed",
            _trace_text(_failed_steps(family.challenge_steps, "reordered"), challenge_context),
        ),
        (
            "failed",
            _trace_text(_failed_steps(family.challenge_steps, "omitted"), challenge_context),
        ),
        (
            "failed",
            _trace_text(_failed_steps(family.challenge_steps, "bypass"), challenge_context),
        ),
    ]
    _rng(family.key, mechanism_index, "candidate-order").shuffle(candidate_rows)
    target_indices = tuple(
        index for index, (kind, _) in enumerate(candidate_rows) if kind == "target"
    )
    failed_indices = tuple(
        index for index, (kind, _) in enumerate(candidate_rows) if kind == "failed"
    )
    mechanism_ref = f"{family.partition}-mechanism-{family.key}-{mechanism_index:02d}"
    return CorpusMechanism(
        public=PublicApprenticeshipStream(
            episodes=tuple(episodes),
            challenge=ApprenticeshipChallenge(
                task_text=_task_text(
                    challenge_context,
                    heldout_description=family.heldout_description,
                ),
                request_text=_request_text(challenge_context),
                candidate_action_trace_texts=tuple(text for _, text in candidate_rows),
            ),
        ),
        supervision=MechanismSupervision(
            target_successful_candidate_indices=target_indices,
            relevant_failed_candidate_indices=failed_indices,
        ),
        metadata=GeneratorMetadata(
            partition=family.partition,
            mechanism_ref=mechanism_ref,
            generator_family=family.key,
            support_structure_signature=family.support_signature,
            challenge_structure_signature=family.challenge_signature,
            heldout_variant=family.heldout_variant,
        ),
    )


def build_outcome_aware_apprenticeship_v5() -> OutcomeAwareApprenticeshipCorpus:
    """Build the frozen, compact, generator-disjoint V5 corpus."""

    by_partition: dict[str, list[CorpusMechanism]] = {
        "train": [],
        "development": [],
        "final": [],
    }
    ordinal = 0
    local_indices = {"train": 0, "development": 0, "final": 0}
    counts_per_family = {"train": 8, "development": 4, "final": 4}
    for family in _FAMILIES:
        for _ in range(counts_per_family[family.partition]):
            mechanism_index = local_indices[family.partition]
            local_indices[family.partition] += 1
            value = _build_mechanism(
                family,
                mechanism_index,
                ordinal_start=ordinal,
            )
            by_partition[family.partition].append(value)
            ordinal += EPISODES_PER_MECHANISM
    return OutcomeAwareApprenticeshipCorpus(
        train=tuple(by_partition["train"]),
        development=tuple(by_partition["development"]),
        final=tuple(by_partition["final"]),
    )


__all__ = [
    "ApprenticeshipChallenge",
    "CANDIDATES_PER_CHALLENGE",
    "CORPUS_ID",
    "CorpusMechanism",
    "DEVELOPMENT_MECHANISMS",
    "EPISODES_PER_MECHANISM",
    "FINAL_MECHANISMS",
    "GeneratorMetadata",
    "MechanismSupervision",
    "MovingOriginCoordinates",
    "OutcomeAwareApprenticeshipCorpus",
    "OutcomeEpisode",
    "PublicApprenticeshipStream",
    "TRAIN_MECHANISMS",
    "build_outcome_aware_apprenticeship_v5",
]
