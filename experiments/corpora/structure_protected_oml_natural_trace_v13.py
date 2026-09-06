"""Identifiable paired natural-trace corpus for structure-protected OML V13.

The generator may construct and label disposable software-procedure pairs.
The learner boundary is deliberately smaller: every chronological row exposes
only one public reference trace, one public attempted trace, and its observed
outcome.  Generator identities, renderer choices, edit relations, structure
signatures, target positions, and evaluator contrasts never enter
``to_learner_payload``.

V13 reuses V12's action vocabulary and partition-disjoint topology families
without changing V12.  The intended procedure is now observable through a
separately worded reference, so a fresh held-out ordering is identifiable
without a hand-coded comparison or solution rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import random
from typing import Literal

from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    MovingOriginCoordinates,
)
from experiments.corpora import scaled_oml_natural_trace_v12 as v12


CORPUS_ID = "angler.structure-protected-oml-natural-trace.v13"
TRAIN_FAMILIES = 48
DEVELOPMENT_FAMILIES = 12
FINAL_FAMILIES = 12
TRAIN_INNER_MECHANISMS = 384
TRAIN_OUTER_UPDATES = 192
TRAIN_OUTER_SLOTS = 8
TRAIN_OUTER_MECHANISMS = TRAIN_OUTER_UPDATES * TRAIN_OUTER_SLOTS
DEVELOPMENT_MECHANISMS = 48
FINAL_MECHANISMS = 48
EPISODES_PER_MECHANISM = 6
RETRIEVAL_CANDIDATES = 5

Partition = Literal["train", "development", "final"]
SurfaceScope = Literal["inner", "outer", "development", "final"]
OutcomeLabel = Literal["SUCCESS", "FAILURE"]
AttemptRelation = Literal[
    "paraphrase", "reorder", "replacement", "omission", "insertion"
]

_OUTCOME_PATTERNS: tuple[tuple[OutcomeLabel, ...], ...] = (
    ("FAILURE", "SUCCESS", "FAILURE", "SUCCESS", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "SUCCESS", "FAILURE", "FAILURE", "SUCCESS"),
    ("FAILURE", "SUCCESS", "SUCCESS", "FAILURE", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "FAILURE", "SUCCESS", "FAILURE", "SUCCESS"),
)
_FAILURE_RELATIONS: tuple[AttemptRelation, ...] = (
    "reorder",
    "replacement",
    "omission",
    "insertion",
)
_SCOPE_PARTITION: dict[SurfaceScope, Partition] = {
    "inner": "train",
    "outer": "train",
    "development": "development",
    "final": "final",
}
_SCOPE_CLASS: dict[SurfaceScope, int] = {
    "inner": 0,
    "outer": 1,
    "development": 2,
    "final": 3,
}
_MECHANISMS_PER_FAMILY: dict[SurfaceScope, int] = {
    "inner": 8,
    "outer": 32,
    "development": 4,
    "final": 4,
}
_CONTEXT_SPACE = 16 * 12 * 10 * 10
_CONTEXT_STRIDE = 7919
if math.gcd(_CONTEXT_SPACE, _CONTEXT_STRIDE) != 1:  # pragma: no cover
    raise RuntimeError("V13 context stride must be coprime to its space")


@dataclass(frozen=True, slots=True)
class PairedOutcomeRow:
    """One learner-visible reference, attempt, and chronological outcome."""

    reference_trace_text: str
    attempt_trace_text: str
    outcome_label: OutcomeLabel
    temporal: MovingOriginCoordinates

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_trace_text", self.reference_trace_text),
            ("attempt_trace_text", self.attempt_trace_text),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty public text")
        if self.outcome_label not in ("SUCCESS", "FAILURE"):
            raise ValueError("outcome_label is invalid")
        if not isinstance(self.temporal, MovingOriginCoordinates):
            raise TypeError("temporal must be MovingOriginCoordinates")

    @property
    def outcome_value(self) -> float:
        return 1.0 if self.outcome_label == "SUCCESS" else -1.0

    def to_learner_payload(self) -> dict[str, str]:
        """Return the exhaustive learner feature/label boundary for one row."""

        return {
            "reference_trace_text": self.reference_trace_text,
            "attempt_trace_text": self.attempt_trace_text,
            "outcome_label": self.outcome_label,
        }


@dataclass(frozen=True, slots=True)
class PublicPairedStream:
    """Six chronological paired rows; no evaluation sidecar or generator data."""

    episodes: tuple[PairedOutcomeRow, ...]

    def __post_init__(self) -> None:
        if type(self.episodes) is not tuple or len(self.episodes) != EPISODES_PER_MECHANISM:
            raise ValueError("each V13 stream must contain exactly six rows")
        if any(not isinstance(value, PairedOutcomeRow) for value in self.episodes):
            raise TypeError("V13 stream contains an invalid paired row")
        outcomes = tuple(value.outcome_label for value in self.episodes)
        if outcomes.count("SUCCESS") != 3 or outcomes.count("FAILURE") != 3:
            raise ValueError("each V13 stream must contain three outcomes of each sign")
        if outcomes[:3] in (("SUCCESS",) * 3, ("FAILURE",) * 3):
            raise ValueError("V13 outcomes may not be grouped by sign")
        ordinals = tuple(value.temporal.acquired_ordinal for value in self.episodes)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise ValueError("V13 chronology must be contiguous")
        if tuple(value.temporal.age for value in self.episodes) != tuple(
            reversed(range(EPISODES_PER_MECHANISM))
        ):
            raise ValueError("V13 ages must resolve at the immediate query point")

    def to_learner_payload(self) -> dict[str, list[dict[str, str]]]:
        return {"episodes": [value.to_learner_payload() for value in self.episodes]}


@dataclass(frozen=True, slots=True)
class StructuralContrastSet:
    """Evaluator-only public trace texts, never a learner payload."""

    reference_trace_text: str
    paraphrase_attempt_trace_text: str
    reordered_attempt_trace_text: str
    replacement_attempt_trace_text: str
    omitted_attempt_trace_text: str
    inserted_attempt_trace_text: str
    serialized_candidate_attempt_trace_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.reference_trace_text,
            self.paraphrase_attempt_trace_text,
            self.reordered_attempt_trace_text,
            self.replacement_attempt_trace_text,
            self.omitted_attempt_trace_text,
            self.inserted_attempt_trace_text,
        )
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("structural contrasts must be non-empty public traces")
        if (
            type(self.serialized_candidate_attempt_trace_texts) is not tuple
            or len(self.serialized_candidate_attempt_trace_texts) != RETRIEVAL_CANDIDATES
            or len(set(self.serialized_candidate_attempt_trace_texts))
            != RETRIEVAL_CANDIDATES
            or set(self.serialized_candidate_attempt_trace_texts) != set(values[1:])
        ):
            raise ValueError("serialized structural candidates are invalid")


@dataclass(frozen=True, slots=True)
class PairedMechanismSupervision:
    """Generator/evaluator relations that are excluded from learner features."""

    episode_attempt_relations: tuple[AttemptRelation, ...]
    same_order_candidate_index: int
    candidate_attempt_relations: tuple[AttemptRelation, ...]
    primary_length_matched_candidate_indices: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            type(self.episode_attempt_relations) is not tuple
            or len(self.episode_attempt_relations) != EPISODES_PER_MECHANISM
            or any(value not in ("paraphrase", *_FAILURE_RELATIONS) for value in self.episode_attempt_relations)
            or self.episode_attempt_relations.count("paraphrase") != 3
        ):
            raise ValueError("episode relation supervision is invalid")
        if (
            type(self.same_order_candidate_index) is not int
            or not 0 <= self.same_order_candidate_index < RETRIEVAL_CANDIDATES
            or type(self.candidate_attempt_relations) is not tuple
            or len(self.candidate_attempt_relations) != RETRIEVAL_CANDIDATES
            or Counter(self.candidate_attempt_relations)
            != Counter(("paraphrase", *_FAILURE_RELATIONS))
            or self.candidate_attempt_relations[self.same_order_candidate_index]
            != "paraphrase"
        ):
            raise ValueError("candidate relation supervision is invalid")
        expected_primary = tuple(
            index
            for index, value in enumerate(self.candidate_attempt_relations)
            if value in ("reorder", "replacement")
        )
        if self.primary_length_matched_candidate_indices != expected_primary:
            raise ValueError("primary same-length supervision is invalid")


@dataclass(frozen=True, slots=True)
class PairedGeneratorMetadata:
    """Non-feature provenance and renderer counterbalance evidence."""

    partition: Partition
    mechanism_ref: str
    generator_family: str
    support_structure_signature: str
    challenge_structure_signature: str
    heldout_variant: str
    episode_reference_plan_ids: tuple[int, ...]
    episode_attempt_plan_ids: tuple[int, ...]
    structural_reference_plan_id: int
    structural_attempt_plan_id: int

    def __post_init__(self) -> None:
        if self.partition not in ("train", "development", "final"):
            raise ValueError("metadata partition is invalid")
        for value in (
            self.mechanism_ref,
            self.generator_family,
            self.support_structure_signature,
            self.challenge_structure_signature,
            self.heldout_variant,
        ):
            if type(value) is not str or not value:
                raise ValueError("metadata text fields must be non-empty")
        for plans in (self.episode_reference_plan_ids, self.episode_attempt_plan_ids):
            if (
                type(plans) is not tuple
                or len(plans) != EPISODES_PER_MECHANISM
                or any(type(value) is not int or not 0 <= value < 8 for value in plans)
            ):
                raise ValueError("episode renderer plans are invalid")
        if not 0 <= self.structural_reference_plan_id < 8 or not 0 <= self.structural_attempt_plan_id < 8:
            raise ValueError("structural renderer plans are invalid")


@dataclass(frozen=True, slots=True)
class PairedCorpusMechanism:
    public: PublicPairedStream
    structural_contrasts: StructuralContrastSet
    supervision: PairedMechanismSupervision
    metadata: PairedGeneratorMetadata

    @property
    def gradient_authorized(self) -> bool:
        return self.metadata.partition == "train"

    def to_learner_payload(self) -> dict[str, list[dict[str, str]]]:
        return self.public.to_learner_payload()


def _families(partition: Partition) -> tuple[object, ...]:
    return tuple(value for value in v12._FAMILIES if value.partition == partition)


_FAMILIES_BY_PARTITION = {
    partition: _families(partition)
    for partition in ("train", "development", "final")
}


def _context(scope: SurfaceScope, local_ordinal: int) -> dict[str, str]:
    """Return a unique partition-disjoint combination of shared public atoms.

    Every sentence includes both ``change`` and ``constraint``.  Their modular
    relation identifies a disjoint combination class, while each individual
    atom appears in every scope.  The learner is never given the class or
    ordinal.
    """

    if type(local_ordinal) is not int or not 0 <= local_ordinal < _CONTEXT_SPACE:
        raise ValueError("V13 public context ordinal is outside its bounded space")
    value = (local_ordinal * _CONTEXT_STRIDE) % _CONTEXT_SPACE
    system_index = value % len(v12._SYSTEMS)
    value //= len(v12._SYSTEMS)
    artifact_index = value % len(v12._ARTIFACTS)
    value //= len(v12._ARTIFACTS)
    change_index = value % len(v12._CHANGES)
    value //= len(v12._CHANGES)
    check_index = value % len(v12._CHECKS)
    constraint_index = (change_index + _SCOPE_CLASS[scope]) % len(v12._CONSTRAINTS)
    return {
        "system": v12._SYSTEMS[system_index],
        "artifact": v12._ARTIFACTS[artifact_index],
        "change": v12._CHANGES[change_index],
        "constraint": v12._CONSTRAINTS[constraint_index],
        "check": v12._CHECKS[check_index],
        "scope": scope,
    }


def _renderer_pair(pair_ordinal: int) -> tuple[int, int]:
    if type(pair_ordinal) is not int or pair_ordinal < 0:
        raise ValueError("pair ordinal must be non-negative")
    reference = pair_ordinal % 8
    # A cyclic Latin-square derangement: every renderer atom appears equally
    # in each named role, and the two roles never share an atom on a
    # paraphrase row.  The exact-multiset reorder control intentionally reuses
    # the reference rendering after this assignment.
    return reference, (reference + 4) % 8


def _rendered_sentences(
    steps: tuple[str, ...],
    context: dict[str, str],
    *,
    plan: int,
) -> dict[str, str]:
    return {
        action: v12._render_action(
            action,
            context,
            surface_ordinal=0,
            plan_override=plan,
        )
        for action in steps
    }


def _trace_from_sentences(
    steps: tuple[str, ...], rendered: dict[str, str]
) -> str:
    return " ".join(
        f"Step {index + 1}: {rendered[action]}"
        for index, action in enumerate(steps)
    )


def _trace_text(
    steps: tuple[str, ...], context: dict[str, str], *, plan: int
) -> str:
    return _trace_from_sentences(steps, _rendered_sentences(steps, context, plan=plan))


def _different_action(steps: tuple[str, ...], token: int) -> str:
    available = tuple(value for value in v12._ACTION_KEYS if value not in steps)
    return available[token % len(available)]


def _changed_steps(
    steps: tuple[str, ...], relation: AttemptRelation, *, token: int
) -> tuple[str, ...]:
    values = list(steps)
    if relation == "reorder":
        left = token % (len(values) - 1)
        values[left], values[left + 1] = values[left + 1], values[left]
    elif relation == "replacement":
        index = token % len(values)
        values[index] = _different_action(steps, token)
    elif relation == "omission":
        values.pop(token % len(values))
    elif relation == "insertion":
        values.insert(token % (len(values) + 1), _different_action(steps, token))
    else:
        raise ValueError("paraphrase does not change the underlying procedure")
    result = tuple(values)
    if result == steps or len(set(result)) != len(result):
        raise RuntimeError("V13 corruption did not produce a distinct valid procedure")
    return result


def _same_length_family_rank(
    family: object, scope: SurfaceScope, family_local_index: int
) -> tuple[int, int]:
    partition = _SCOPE_PARTITION[scope]
    matching = tuple(
        value
        for value in _FAMILIES_BY_PARTITION[partition]
        if len(value.support_steps) == len(family.support_steps)
    )
    family_rank = matching.index(family)
    mechanisms_per_family = _MECHANISMS_PER_FAMILY[scope]
    return (
        family_rank * mechanisms_per_family + family_local_index,
        len(matching) * mechanisms_per_family,
    )


@lru_cache(maxsize=None)
def _failure_plan(scope: SurfaceScope, support_length: int) -> tuple[AttemptRelation, ...]:
    partition = _SCOPE_PARTITION[scope]
    family_count = sum(
        len(value.support_steps) == support_length
        for value in _FAMILIES_BY_PARTITION[partition]
    )
    mechanism_count = family_count * _MECHANISMS_PER_FAMILY[scope]
    if mechanism_count % 2:
        raise RuntimeError("V13 length groups must contain an even mechanism count")
    if support_length == 3:
        counts = {"insertion": mechanism_count, "reorder": mechanism_count, "replacement": mechanism_count}
    elif support_length in (4, 5):
        counts = {
            "insertion": mechanism_count,
            "omission": mechanism_count,
            "reorder": mechanism_count // 2,
            "replacement": mechanism_count // 2,
        }
    elif support_length == 6:
        counts = {"omission": mechanism_count, "reorder": mechanism_count, "replacement": mechanism_count}
    else:  # pragma: no cover - inherited family design fixes 3..6
        raise ValueError("V13 support length is outside the frozen family range")
    values: list[AttemptRelation] = []
    for relation in _FAILURE_RELATIONS:
        values.extend((relation,) * counts.get(relation, 0))
    rng = random.Random(f"angler-v13-failure-plan:{scope}:{support_length}")
    rng.shuffle(values)
    if len(values) != mechanism_count * 3:
        raise RuntimeError("V13 failure plan has the wrong row count")
    return tuple(values)


def _episode_relations(
    family: object,
    scope: SurfaceScope,
    family_local_index: int,
    outcomes: tuple[OutcomeLabel, ...],
) -> tuple[AttemptRelation, ...]:
    rank, mechanism_count = _same_length_family_rank(
        family, scope, family_local_index
    )
    plan = _failure_plan(scope, len(family.support_steps))
    if len(plan) != mechanism_count * 3:
        raise RuntimeError("V13 failure plan/rank cardinality changed")
    failures = iter(plan[rank * 3 : rank * 3 + 3])
    result = tuple(
        "paraphrase" if outcome == "SUCCESS" else next(failures)
        for outcome in outcomes
    )
    if sum(value != "paraphrase" for value in result) != 3:
        raise RuntimeError("V13 mechanism did not receive three failure relations")
    return result


def _attempt_trace(
    reference_steps: tuple[str, ...],
    relation: AttemptRelation,
    context: dict[str, str],
    *,
    reference_plan: int,
    attempt_plan: int,
    token: int,
) -> tuple[str, str]:
    reference_sentences = _rendered_sentences(
        reference_steps, context, plan=reference_plan
    )
    reference = _trace_from_sentences(reference_steps, reference_sentences)
    if relation == "paraphrase":
        attempt = _trace_text(reference_steps, context, plan=attempt_plan)
    else:
        changed = _changed_steps(reference_steps, relation, token=token)
        if relation == "reorder":
            # This is the exact same rendered sentence multiset in a different
            # order.  It is the primary direction-only negative control.
            attempt = _trace_from_sentences(changed, reference_sentences)
        else:
            attempt = _trace_text(changed, context, plan=attempt_plan)
    if reference == attempt:
        raise RuntimeError("V13 reference and attempt must be distinct public text")
    return reference, attempt


def _structural_contrasts(
    family: object,
    *,
    scope: SurfaceScope,
    mechanism_index: int,
    family_local_index: int,
) -> tuple[
    StructuralContrastSet,
    tuple[AttemptRelation, ...],
    int,
    tuple[int, int],
    int,
    int,
]:
    context_local = mechanism_index * 7 + EPISODES_PER_MECHANISM
    context = _context(scope, context_local)
    reference_plan, attempt_plan = _renderer_pair(context_local)
    token = 100_000 + mechanism_index * 17 + family_local_index
    reference_steps = family.challenge_steps
    reference_sentences = _rendered_sentences(
        reference_steps, context, plan=reference_plan
    )
    reference = _trace_from_sentences(reference_steps, reference_sentences)
    paraphrase = _trace_text(reference_steps, context, plan=attempt_plan)
    changed = {
        relation: _changed_steps(reference_steps, relation, token=token + offset)
        for offset, relation in enumerate(_FAILURE_RELATIONS, start=1)
    }
    reordered = _trace_from_sentences(changed["reorder"], reference_sentences)
    replacement = _trace_text(changed["replacement"], context, plan=attempt_plan)
    omitted = _trace_text(changed["omission"], context, plan=attempt_plan)
    inserted = _trace_text(changed["insertion"], context, plan=attempt_plan)
    by_relation: dict[AttemptRelation, str] = {
        "paraphrase": paraphrase,
        "reorder": reordered,
        "replacement": replacement,
        "omission": omitted,
        "insertion": inserted,
    }
    relation_order = list(by_relation)
    # Five positions cannot divide every 48-row heldout panel exactly; this
    # rotation keeps every position within one observation of every other.
    target_position = mechanism_index % RETRIEVAL_CANDIDATES
    relation_order.remove("paraphrase")
    relation_order.insert(target_position, "paraphrase")
    candidates = tuple(by_relation[value] for value in relation_order)
    primary = tuple(
        index
        for index, value in enumerate(relation_order)
        if value in ("reorder", "replacement")
    )
    contrasts = StructuralContrastSet(
        reference_trace_text=reference,
        paraphrase_attempt_trace_text=paraphrase,
        reordered_attempt_trace_text=reordered,
        replacement_attempt_trace_text=replacement,
        omitted_attempt_trace_text=omitted,
        inserted_attempt_trace_text=inserted,
        serialized_candidate_attempt_trace_texts=candidates,
    )
    return (
        contrasts,
        tuple(relation_order),
        target_position,
        primary,
        reference_plan,
        attempt_plan,
    )


def _build_mechanism(
    family: object,
    *,
    scope: SurfaceScope,
    mechanism_index: int,
    family_local_index: int,
    ordinal_start: int,
) -> PairedCorpusMechanism:
    outcomes = _OUTCOME_PATTERNS[mechanism_index % len(_OUTCOME_PATTERNS)]
    relations = _episode_relations(
        family, scope, family_local_index, outcomes
    )
    rows = []
    reference_plans = []
    attempt_plans = []
    for episode_index, (outcome, relation) in enumerate(
        zip(outcomes, relations, strict=True)
    ):
        pair_ordinal = mechanism_index * EPISODES_PER_MECHANISM + episode_index
        reference_plan, attempt_plan = _renderer_pair(pair_ordinal)
        context = _context(scope, mechanism_index * 7 + episode_index)
        reference, attempt = _attempt_trace(
            family.support_steps,
            relation,
            context,
            reference_plan=reference_plan,
            attempt_plan=attempt_plan,
            token=mechanism_index * 31 + episode_index,
        )
        rows.append(
            PairedOutcomeRow(
                reference_trace_text=reference,
                attempt_trace_text=attempt,
                outcome_label=outcome,
                temporal=MovingOriginCoordinates(
                    acquired_ordinal=ordinal_start + episode_index,
                    age=EPISODES_PER_MECHANISM - episode_index - 1,
                    landmark_relations=((
                        "stream_start",
                        "AT" if episode_index == 0 else "AFTER",
                    ),),
                ),
            )
        )
        reference_plans.append(reference_plan)
        attempt_plans.append(reference_plan if relation == "reorder" else attempt_plan)

    (
        contrasts,
        candidate_relations,
        same_order_candidate_index,
        primary_indices,
        structural_reference_plan,
        structural_attempt_plan,
    ) = _structural_contrasts(
        family,
        scope=scope,
        mechanism_index=mechanism_index,
        family_local_index=family_local_index,
    )
    supervision = PairedMechanismSupervision(
        episode_attempt_relations=relations,
        same_order_candidate_index=same_order_candidate_index,
        candidate_attempt_relations=candidate_relations,
        primary_length_matched_candidate_indices=primary_indices,
    )
    partition = _SCOPE_PARTITION[scope]
    mechanism_ref = (
        f"{partition}-v13-{scope}-paired-mechanism-"
        f"{family.ordinal:02d}-{family_local_index:04d}"
    )
    return PairedCorpusMechanism(
        public=PublicPairedStream(tuple(rows)),
        structural_contrasts=contrasts,
        supervision=supervision,
        metadata=PairedGeneratorMetadata(
            partition=partition,
            mechanism_ref=mechanism_ref,
            generator_family=family.key,
            support_structure_signature=family.support_signature,
            challenge_structure_signature=family.challenge_signature,
            heldout_variant=family.transition,
            episode_reference_plan_ids=tuple(reference_plans),
            episode_attempt_plan_ids=tuple(attempt_plans),
            structural_reference_plan_id=structural_reference_plan,
            structural_attempt_plan_id=structural_attempt_plan,
        ),
    )


def _materialize_partition(
    families: tuple[object, ...],
    *,
    scope: SurfaceScope,
    mechanisms_per_family: int,
    ordinal_base: int,
) -> tuple[PairedCorpusMechanism, ...]:
    values = []
    mechanism_index = 0
    for family in families:
        for family_local_index in range(mechanisms_per_family):
            values.append(
                _build_mechanism(
                    family,
                    scope=scope,
                    mechanism_index=mechanism_index,
                    family_local_index=family_local_index,
                    ordinal_start=(
                        ordinal_base + mechanism_index * EPISODES_PER_MECHANISM
                    ),
                )
            )
            mechanism_index += 1
    return tuple(values)


@dataclass(frozen=True, slots=True)
class StructureProtectedOmlNaturalTraceCorpus:
    train_inner: tuple[PairedCorpusMechanism, ...]
    development: tuple[PairedCorpusMechanism, ...]
    _final: tuple[PairedCorpusMechanism, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.train_inner) != TRAIN_INNER_MECHANISMS:
            raise ValueError("train_inner has the wrong V13 mechanism count")
        if len(self.development) != DEVELOPMENT_MECHANISMS:
            raise ValueError("development has the wrong V13 mechanism count")
        if self._final is not None and len(self._final) != FINAL_MECHANISMS:
            raise ValueError("final has the wrong V13 mechanism count")
        if any(not value.gradient_authorized for value in self.train_inner):
            raise ValueError("all V13 inner rows must be gradient-authorized")
        if any(value.gradient_authorized for value in self.development):
            raise ValueError("V13 development may not authorize gradients")
        if self._final is not None and any(value.gradient_authorized for value in self._final):
            raise ValueError("V13 final may not authorize gradients")

    @property
    def final(self) -> tuple[PairedCorpusMechanism, ...]:
        if self._final is None:
            raise FinalPartitionSealedError(
                "final remains sealed until the V13 development gate authorizes it"
            )
        return self._final

    @property
    def final_is_open(self) -> bool:
        return self._final is not None

    def partition(self, name: str) -> tuple[PairedCorpusMechanism, ...]:
        if name == "train_inner":
            return self.train_inner
        if name == "development":
            return self.development
        if name == "final":
            return self.final
        raise ValueError("V13 partition name is invalid")

    def train_inner_for_update(
        self, update_index: int
    ) -> tuple[PairedCorpusMechanism, ...]:
        if type(update_index) is not int or not 0 <= update_index < TRAIN_OUTER_UPDATES:
            raise ValueError("update_index is outside the frozen V13 schedule")
        start = update_index * TRAIN_OUTER_SLOTS
        return tuple(
            self.train_inner[((start + slot) * 49) % TRAIN_INNER_MECHANISMS]
            for slot in range(TRAIN_OUTER_SLOTS)
        )

    def train_outer(
        self, update_index: int, slot_index: int
    ) -> PairedCorpusMechanism:
        if type(update_index) is not int or not 0 <= update_index < TRAIN_OUTER_UPDATES:
            raise ValueError("update_index is outside the frozen V13 schedule")
        if type(slot_index) is not int or not 0 <= slot_index < TRAIN_OUTER_SLOTS:
            raise ValueError("slot_index is outside the frozen V13 schedule")
        mechanism_index = update_index * TRAIN_OUTER_SLOTS + slot_index
        family = v12._TRAIN_FAMILY_VALUES[mechanism_index % TRAIN_FAMILIES]
        family_local_index = mechanism_index // TRAIN_FAMILIES
        return _build_mechanism(
            family,
            scope="outer",
            mechanism_index=mechanism_index,
            family_local_index=family_local_index,
            ordinal_start=(
                1_000_000 + mechanism_index * EPISODES_PER_MECHANISM
            ),
        )

    def learner_payload_size_bytes(self, *, include_outer: bool = False) -> int:
        payloads = [value.to_learner_payload() for value in self.train_inner]
        payloads.extend(value.to_learner_payload() for value in self.development)
        if self._final is not None:
            payloads.extend(value.to_learner_payload() for value in self._final)
        if include_outer:
            payloads.extend(
                self.train_outer(update, slot).to_learner_payload()
                for update in range(TRAIN_OUTER_UPDATES)
                for slot in range(TRAIN_OUTER_SLOTS)
            )
        return len(
            json.dumps(payloads, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )


def build_structure_protected_oml_natural_trace_v13(
    *, include_final: bool = False
) -> StructureProtectedOmlNaturalTraceCorpus:
    if type(include_final) is not bool:
        raise TypeError("include_final must be bool")
    train_inner = _materialize_partition(
        tuple(v12._TRAIN_FAMILY_VALUES),
        scope="inner",
        mechanisms_per_family=8,
        ordinal_base=0,
    )
    development = _materialize_partition(
        tuple(v12._DEVELOPMENT_FAMILY_VALUES),
        scope="development",
        mechanisms_per_family=4,
        ordinal_base=2_000_000,
    )
    final = (
        _materialize_partition(
            tuple(v12._FINAL_FAMILY_VALUES),
            scope="final",
            mechanisms_per_family=4,
            ordinal_base=3_000_000,
        )
        if include_final
        else None
    )
    return StructureProtectedOmlNaturalTraceCorpus(
        train_inner=train_inner,
        development=development,
        _final=final,
    )


__all__ = [
    "AttemptRelation",
    "CORPUS_ID",
    "DEVELOPMENT_FAMILIES",
    "DEVELOPMENT_MECHANISMS",
    "EPISODES_PER_MECHANISM",
    "FINAL_FAMILIES",
    "FINAL_MECHANISMS",
    "FinalPartitionSealedError",
    "PairedCorpusMechanism",
    "PairedGeneratorMetadata",
    "PairedMechanismSupervision",
    "PairedOutcomeRow",
    "PublicPairedStream",
    "RETRIEVAL_CANDIDATES",
    "StructuralContrastSet",
    "StructureProtectedOmlNaturalTraceCorpus",
    "TRAIN_FAMILIES",
    "TRAIN_INNER_MECHANISMS",
    "TRAIN_OUTER_MECHANISMS",
    "TRAIN_OUTER_SLOTS",
    "TRAIN_OUTER_UPDATES",
    "build_structure_protected_oml_natural_trace_v13",
]
