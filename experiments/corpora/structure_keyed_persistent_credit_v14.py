"""Leakage-bounded corpus and structural replication panel for V14.

V14 deliberately reuses V13's public chronological procedure rows.  The
learner receives those rows through the unchanged V13 public payload boundary;
generator identity, corruption type, renderer plan, target position, and the
replication answer remain evaluator-only.

The fresh structural panel contains six observations in every one of the
eight renderer-pair by five target-position cells.  The cross-cells are useful
diagnostics, while their adequately populated renderer and position marginals
support the preregistered replication decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from experiments.corpora import structure_protected_oml_natural_trace_v13 as v13


CORPUS_ID = "angler.structure-keyed-persistent-credit.v14"
TRAIN_MECHANISMS = 384
DEVELOPMENT_MECHANISMS = 48
FINAL_MECHANISMS = 48
EVENTS_PER_MECHANISM = 6
TRAIN_EVENTS = TRAIN_MECHANISMS * EVENTS_PER_MECHANISM
DEVELOPMENT_EVENTS = DEVELOPMENT_MECHANISMS * EVENTS_PER_MECHANISM
STRUCTURAL_REPLICATION_ROWS = 240
RENDERER_PAIRS = 8
TARGET_POSITIONS = 5
ROWS_PER_RENDERER_POSITION_CELL = 6
SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4)

OutcomeLabel = Literal["SUCCESS", "FAILURE"]
CorruptionType = Literal["reorder", "replacement", "omission", "insertion"]

_CORRUPTIONS: tuple[CorruptionType, ...] = (
    "reorder",
    "replacement",
    "omission",
    "insertion",
)
_RELATIONS = ("paraphrase", *_CORRUPTIONS)
_REPLICATION_CONTEXT_BASE = 10_000


@dataclass(frozen=True, slots=True)
class StructuralReplicationPublic:
    """The exhaustive model-visible boundary for one replication row."""

    reference_trace_text: str
    candidate_attempt_trace_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.reference_trace_text) is not str or not self.reference_trace_text.strip():
            raise ValueError("replication reference must be non-empty public text")
        if (
            type(self.candidate_attempt_trace_texts) is not tuple
            or len(self.candidate_attempt_trace_texts) != TARGET_POSITIONS
            or len(set(self.candidate_attempt_trace_texts)) != TARGET_POSITIONS
            or any(type(value) is not str or not value.strip() for value in self.candidate_attempt_trace_texts)
        ):
            raise ValueError("replication candidates must be five distinct public texts")

    def to_encoder_payload(self) -> dict[str, object]:
        return {
            "reference_trace_text": self.reference_trace_text,
            "candidate_attempt_trace_texts": list(self.candidate_attempt_trace_texts),
        }


@dataclass(frozen=True, slots=True)
class StructuralReplicationSupervision:
    """Evaluator-only answer and explicit per-corruption candidate mapping."""

    target_candidate_index: int
    candidate_corruption_types: tuple[str, ...]
    corruption_candidate_indices: tuple[tuple[CorruptionType, int], ...]

    def __post_init__(self) -> None:
        if type(self.target_candidate_index) is not int or not 0 <= self.target_candidate_index < TARGET_POSITIONS:
            raise ValueError("replication target position is invalid")
        if (
            type(self.candidate_corruption_types) is not tuple
            or len(self.candidate_corruption_types) != TARGET_POSITIONS
            or Counter(self.candidate_corruption_types) != Counter(_RELATIONS)
            or self.candidate_corruption_types[self.target_candidate_index] != "paraphrase"
        ):
            raise ValueError("replication candidate relations are invalid")
        expected = tuple(
            (relation, self.candidate_corruption_types.index(relation))
            for relation in _CORRUPTIONS
        )
        if self.corruption_candidate_indices != expected:
            raise ValueError("per-corruption candidate indices are inconsistent")

    def candidate_index_for(self, corruption: CorruptionType) -> int:
        if corruption not in _CORRUPTIONS:
            raise ValueError("unknown V14 corruption")
        return dict(self.corruption_candidate_indices)[corruption]


@dataclass(frozen=True, slots=True)
class StructuralReplicationMetadata:
    """Non-feature balance and provenance evidence for one replication row."""

    row_ref: str
    generator_family: str
    transition_group: str
    renderer_pair: tuple[int, int]
    target_position: int
    challenge_structure_signature: str

    def __post_init__(self) -> None:
        for value in (
            self.row_ref,
            self.generator_family,
            self.transition_group,
            self.challenge_structure_signature,
        ):
            if type(value) is not str or not value:
                raise ValueError("replication metadata text must be non-empty")
        if (
            type(self.renderer_pair) is not tuple
            or len(self.renderer_pair) != 2
            or any(type(value) is not int or not 0 <= value < RENDERER_PAIRS for value in self.renderer_pair)
            or self.renderer_pair[1] != (self.renderer_pair[0] + 4) % RENDERER_PAIRS
        ):
            raise ValueError("replication renderer pair is invalid")
        if self.target_position not in range(TARGET_POSITIONS):
            raise ValueError("replication metadata target position is invalid")


@dataclass(frozen=True, slots=True)
class StructuralReplicationRow:
    public: StructuralReplicationPublic
    supervision: StructuralReplicationSupervision
    metadata: StructuralReplicationMetadata

    def to_encoder_payload(self) -> dict[str, object]:
        """Serialize model inputs without evaluator labels or provenance."""

        return self.public.to_encoder_payload()


def _outcomes(
    mechanism: v13.PairedCorpusMechanism,
) -> tuple[OutcomeLabel, ...]:
    result = tuple(row.outcome_label for row in mechanism.public.episodes)
    if Counter(result) != Counter({"SUCCESS": 3, "FAILURE": 3}):
        raise RuntimeError("V14 inherited an unbalanced V13 outcome stream")
    return result


def _shuffled_outcomes(
    mechanism: v13.PairedCorpusMechanism,
) -> tuple[OutcomeLabel, ...]:
    true = _outcomes(mechanism)
    shuffled = tuple(true[index] for index in SHUFFLE_PERMUTATION)
    if Counter(shuffled) != Counter(true) or any(left == right for left, right in zip(true, shuffled, strict=True)):
        raise RuntimeError("V14 control must preserve balance and change every association")
    return shuffled


def _replication_row(
    row_index: int,
    renderer_index: int,
    target_position: int,
    replicate_index: int,
) -> StructuralReplicationRow:
    families = tuple(v13.v12._DEVELOPMENT_FAMILY_VALUES)
    family = families[row_index % len(families)]
    reference_plan, attempt_plan = v13._renderer_pair(renderer_index)
    context = v13._context("development", _REPLICATION_CONTEXT_BASE + row_index)
    steps = family.challenge_steps
    reference_sentences = v13._rendered_sentences(steps, context, plan=reference_plan)
    reference = v13._trace_from_sentences(steps, reference_sentences)
    by_relation = {
        "paraphrase": v13._trace_text(steps, context, plan=attempt_plan),
    }
    token = 900_000 + row_index * 19
    for offset, relation in enumerate(_CORRUPTIONS, start=1):
        changed = v13._changed_steps(steps, relation, token=token + offset)
        by_relation[relation] = (
            v13._trace_from_sentences(changed, reference_sentences)
            if relation == "reorder"
            else v13._trace_text(changed, context, plan=attempt_plan)
        )

    # Rotate negative candidate positions independently of the target so no
    # corruption obtains a fixed serialization position.
    rotation = (renderer_index + target_position + replicate_index) % len(_CORRUPTIONS)
    negatives = list(_CORRUPTIONS[rotation:] + _CORRUPTIONS[:rotation])
    relations = negatives
    relations.insert(target_position, "paraphrase")
    relation_tuple = tuple(relations)
    candidates = tuple(by_relation[relation] for relation in relation_tuple)
    corruption_indices = tuple(
        (relation, relation_tuple.index(relation)) for relation in _CORRUPTIONS
    )
    return StructuralReplicationRow(
        public=StructuralReplicationPublic(
            reference_trace_text=reference,
            candidate_attempt_trace_texts=candidates,
        ),
        supervision=StructuralReplicationSupervision(
            target_candidate_index=target_position,
            candidate_corruption_types=relation_tuple,
            corruption_candidate_indices=corruption_indices,
        ),
        metadata=StructuralReplicationMetadata(
            row_ref=f"development-v14-structural-replication-{row_index:03d}",
            generator_family=family.key,
            transition_group=family.transition,
            renderer_pair=(reference_plan, attempt_plan),
            target_position=target_position,
            challenge_structure_signature=family.challenge_signature,
        ),
    )


def _build_structural_replication() -> tuple[StructuralReplicationRow, ...]:
    rows = []
    for renderer_index in range(RENDERER_PAIRS):
        for target_position in range(TARGET_POSITIONS):
            for replicate_index in range(ROWS_PER_RENDERER_POSITION_CELL):
                row_index = len(rows)
                rows.append(
                    _replication_row(
                        row_index,
                        renderer_index,
                        target_position,
                        replicate_index,
                    )
                )
    result = tuple(rows)
    if len(result) != STRUCTURAL_REPLICATION_ROWS:
        raise RuntimeError("V14 structural replication cardinality changed")
    return result


@dataclass(frozen=True, slots=True)
class StructureKeyedPersistentCreditV14Corpus:
    train: tuple[v13.PairedCorpusMechanism, ...]
    development: tuple[v13.PairedCorpusMechanism, ...]
    structural_replication: tuple[StructuralReplicationRow, ...]
    _final: tuple[v13.PairedCorpusMechanism, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.train) != TRAIN_MECHANISMS:
            raise ValueError("V14 train mechanism count is invalid")
        if len(self.development) != DEVELOPMENT_MECHANISMS:
            raise ValueError("V14 development mechanism count is invalid")
        if len(self.structural_replication) != STRUCTURAL_REPLICATION_ROWS:
            raise ValueError("V14 structural replication count is invalid")
        if self._final is not None and len(self._final) != FINAL_MECHANISMS:
            raise ValueError("V14 final mechanism count is invalid")
        if any(not value.gradient_authorized for value in self.train):
            raise ValueError("V14 train mechanisms must authorize gradients")
        if any(value.gradient_authorized for value in self.development):
            raise ValueError("V14 development mechanisms may not authorize gradients")
        if self._final is not None and any(value.gradient_authorized for value in self._final):
            raise ValueError("V14 final mechanisms may not authorize gradients")

    @property
    def train_mechanisms(self) -> tuple[v13.PairedCorpusMechanism, ...]:
        return self.train

    @property
    def development_mechanisms(self) -> tuple[v13.PairedCorpusMechanism, ...]:
        return self.development

    @property
    def train_events(self) -> tuple[v13.PairedOutcomeRow, ...]:
        return tuple(row for mechanism in self.train for row in mechanism.public.episodes)

    @property
    def development_events(self) -> tuple[v13.PairedOutcomeRow, ...]:
        return tuple(row for mechanism in self.development for row in mechanism.public.episodes)

    @property
    def final(self) -> tuple[v13.PairedCorpusMechanism, ...]:
        if self._final is None:
            raise v13.FinalPartitionSealedError(
                "final remains sealed until the V14 development gate authorizes it"
            )
        return self._final

    @property
    def final_is_open(self) -> bool:
        return self._final is not None

    def partition(self, name: str) -> tuple[v13.PairedCorpusMechanism, ...]:
        if name in ("train", "train_inner"):
            return self.train
        if name == "development":
            return self.development
        if name == "final":
            return self.final
        raise ValueError("V14 partition name is invalid")

    @staticmethod
    def true_outcomes(
        mechanism: v13.PairedCorpusMechanism,
    ) -> tuple[OutcomeLabel, ...]:
        return _outcomes(mechanism)

    @staticmethod
    def shuffled_outcomes(
        mechanism: v13.PairedCorpusMechanism,
    ) -> tuple[OutcomeLabel, ...]:
        return _shuffled_outcomes(mechanism)


def build_structure_keyed_persistent_credit_v14(
    *, include_final: bool = False
) -> StructureKeyedPersistentCreditV14Corpus:
    if type(include_final) is not bool:
        raise TypeError("include_final must be bool")
    inherited = v13.build_structure_protected_oml_natural_trace_v13(
        include_final=include_final
    )
    return StructureKeyedPersistentCreditV14Corpus(
        train=inherited.train_inner,
        development=inherited.development,
        structural_replication=_build_structural_replication(),
        _final=inherited.final if include_final else None,
    )


__all__ = [
    "CORPUS_ID",
    "DEVELOPMENT_EVENTS",
    "DEVELOPMENT_MECHANISMS",
    "EVENTS_PER_MECHANISM",
    "FINAL_MECHANISMS",
    "RENDERER_PAIRS",
    "ROWS_PER_RENDERER_POSITION_CELL",
    "SHUFFLE_PERMUTATION",
    "STRUCTURAL_REPLICATION_ROWS",
    "StructuralReplicationMetadata",
    "StructuralReplicationPublic",
    "StructuralReplicationRow",
    "StructuralReplicationSupervision",
    "StructureKeyedPersistentCreditV14Corpus",
    "TARGET_POSITIONS",
    "TRAIN_EVENTS",
    "TRAIN_MECHANISMS",
    "build_structure_keyed_persistent_credit_v14",
]
