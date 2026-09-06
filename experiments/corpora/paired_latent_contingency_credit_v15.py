"""Counterfactual latent-contingency corpus for V15 procedural credit.

Each matched twin sees byte-identical public procedures and temporal
coordinates.  Its hidden two-bit regime is evaluator-only; the two acquisition
outcomes reveal the bits, and four fresh probes reuse the two abstract
relations on new surfaces.  Complementary twins therefore require opposite
probe outcomes for the same current input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Literal

from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    MovingOriginCoordinates,
)
from experiments.corpora import structure_protected_oml_natural_trace_v13 as v13


CORPUS_ID = "angler.paired-latent-contingency-credit.v15"
TRAIN_FAMILIES = 48
DEVELOPMENT_FAMILIES = 12
FINAL_FAMILIES = 12
TRAIN_PAIRS = 384
DEVELOPMENT_PAIRS = 48
FINAL_PAIRS = 48
EVENTS_PER_EPISODE = 6
ACQUISITION_INDICES = (0, 1)
PROBE_INDICES = (2, 3, 4, 5)
TRAIN_UPDATES = 96
PAIRS_PER_UPDATE = 8
SCHEDULE_MULTIPLIER = 49

Partition = Literal["train", "development", "final"]
SurfaceScope = Literal["inner", "development", "final"]
OutcomeLabel = Literal["SUCCESS", "FAILURE"]
RelationClass = Literal["alpha", "beta"]
HiddenRegime = tuple[int, int]

_REGIMES: tuple[HiddenRegime, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
_BASE_RELATION_CLASSES: tuple[RelationClass, ...] = (
    "alpha",
    "beta",
    "alpha",
    "beta",
    "alpha",
    "beta",
)
_SCOPE_PARTITION: dict[SurfaceScope, Partition] = {
    "inner": "train",
    "development": "development",
    "final": "final",
}
_MECHANISMS_PER_FAMILY: dict[SurfaceScope, int] = {
    "inner": 8,
    "development": 4,
    "final": 4,
}


def _label(bit: int) -> OutcomeLabel:
    if bit not in (0, 1):
        raise ValueError("latent contingency bit must be binary")
    return "SUCCESS" if bit else "FAILURE"


def _opposite(regime: HiddenRegime) -> HiddenRegime:
    return (1 - regime[0], 1 - regime[1])


@dataclass(frozen=True, slots=True)
class PublicContingencyEvent:
    """One prediction input; outcome and contingency remain unavailable."""

    reference_trace_text: str
    attempt_trace_text: str
    temporal: MovingOriginCoordinates

    def __post_init__(self) -> None:
        if type(self.reference_trace_text) is not str or not self.reference_trace_text.strip():
            raise ValueError("reference trace must be non-empty public text")
        if type(self.attempt_trace_text) is not str or not self.attempt_trace_text.strip():
            raise ValueError("attempt trace must be non-empty public text")
        if self.reference_trace_text == self.attempt_trace_text:
            raise ValueError("reference and attempt must be distinct")
        if not isinstance(self.temporal, MovingOriginCoordinates):
            raise TypeError("temporal must be MovingOriginCoordinates")

    def to_prediction_payload(self) -> dict[str, object]:
        return {
            "reference_trace_text": self.reference_trace_text,
            "attempt_trace_text": self.attempt_trace_text,
            "temporal": {
                "acquired_ordinal": self.temporal.acquired_ordinal,
                "age": self.temporal.age,
                "landmark_relations": [list(value) for value in self.temporal.landmark_relations],
            },
        }


@dataclass(frozen=True, slots=True)
class PublicContingencyEpisode:
    events: tuple[PublicContingencyEvent, ...]

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or len(self.events) != EVENTS_PER_EPISODE:
            raise ValueError("V15 public episode must contain six events")
        if any(not isinstance(value, PublicContingencyEvent) for value in self.events):
            raise TypeError("V15 public episode contains an invalid event")
        ordinals = tuple(value.temporal.acquired_ordinal for value in self.events)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + EVENTS_PER_EPISODE)):
            raise ValueError("V15 event chronology must be contiguous")
        if tuple(value.temporal.age for value in self.events) != tuple(reversed(range(EVENTS_PER_EPISODE))):
            raise ValueError("V15 event ages must resolve at the probe boundary")

    def to_prediction_payload(self) -> dict[str, object]:
        return {"events": [value.to_prediction_payload() for value in self.events]}


@dataclass(frozen=True, slots=True)
class ContingencyEpisodeSupervision:
    """Outcome and relation sidecar revealed only by the evaluator."""

    outcome_labels: tuple[OutcomeLabel, ...]
    relation_classes: tuple[RelationClass, ...]
    acquisition_indices: tuple[int, int] = ACQUISITION_INDICES
    probe_indices: tuple[int, int, int, int] = PROBE_INDICES

    def __post_init__(self) -> None:
        if (
            type(self.outcome_labels) is not tuple
            or len(self.outcome_labels) != EVENTS_PER_EPISODE
            or any(value not in ("SUCCESS", "FAILURE") for value in self.outcome_labels)
        ):
            raise ValueError("V15 outcome supervision is invalid")
        if (
            type(self.relation_classes) is not tuple
            or len(self.relation_classes) != EVENTS_PER_EPISODE
            or Counter(self.relation_classes[:2]) != Counter(("alpha", "beta"))
            or Counter(self.relation_classes[2:]) != Counter(("alpha", "alpha", "beta", "beta"))
        ):
            raise ValueError("V15 relation-class schedule is invalid")
        if self.acquisition_indices != ACQUISITION_INDICES or self.probe_indices != PROBE_INDICES:
            raise ValueError("V15 acquisition/probe boundary is invalid")

    @property
    def outcome_values(self) -> tuple[float, ...]:
        return tuple(1.0 if value == "SUCCESS" else -1.0 for value in self.outcome_labels)

    @property
    def acquisition_outcomes(self) -> tuple[OutcomeLabel, OutcomeLabel]:
        return tuple(self.outcome_labels[index] for index in self.acquisition_indices)  # type: ignore[return-value]

    @property
    def probe_outcomes(self) -> tuple[OutcomeLabel, OutcomeLabel, OutcomeLabel, OutcomeLabel]:
        return tuple(self.outcome_labels[index] for index in self.probe_indices)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ContingencyEpisodeMetadata:
    """Evaluator-only identity, balance, and hidden-regime evidence."""

    partition: Partition
    pair_ref: str
    episode_ref: str
    twin_ref: str
    generator_family: str
    transition_group: str
    hidden_regime: HiddenRegime
    structure_signature: str

    def __post_init__(self) -> None:
        if self.partition not in ("train", "development", "final"):
            raise ValueError("V15 metadata partition is invalid")
        for value in (
            self.pair_ref,
            self.episode_ref,
            self.twin_ref,
            self.generator_family,
            self.transition_group,
            self.structure_signature,
        ):
            if type(value) is not str or not value:
                raise ValueError("V15 metadata text must be non-empty")
        if self.episode_ref == self.twin_ref:
            raise ValueError("V15 twin identities must differ")
        if self.hidden_regime not in _REGIMES:
            raise ValueError("V15 hidden regime is invalid")


@dataclass(frozen=True, slots=True)
class ContingencyEpisode:
    public: PublicContingencyEpisode
    supervision: ContingencyEpisodeSupervision
    metadata: ContingencyEpisodeMetadata

    @property
    def gradient_authorized(self) -> bool:
        return self.metadata.partition == "train"

    def to_learner_payload(self) -> dict[str, object]:
        """Return public prediction inputs without outcomes or sidecars."""

        return self.public.to_prediction_payload()


@dataclass(frozen=True, slots=True)
class CounterfactualTwinPair:
    first: ContingencyEpisode
    second: ContingencyEpisode

    def __post_init__(self) -> None:
        if not isinstance(self.first, ContingencyEpisode) or not isinstance(self.second, ContingencyEpisode):
            raise TypeError("V15 pair members must be contingency episodes")
        if self.first.public != self.second.public:
            raise ValueError("counterfactual twins must have identical public inputs")
        if self.first.metadata.pair_ref != self.second.metadata.pair_ref:
            raise ValueError("counterfactual twins must share one pair identity")
        if self.first.metadata.twin_ref != self.second.metadata.episode_ref or self.second.metadata.twin_ref != self.first.metadata.episode_ref:
            raise ValueError("counterfactual twin references are inconsistent")
        if self.second.metadata.hidden_regime != _opposite(self.first.metadata.hidden_regime):
            raise ValueError("counterfactual twin regimes must be bitwise complements")
        if any(
            left == right
            for left, right in zip(
                self.first.supervision.probe_outcomes,
                self.second.supervision.probe_outcomes,
                strict=True,
            )
        ):
            raise ValueError("matched twin probes must have opposite outcomes")

    @property
    def pair_ref(self) -> str:
        return self.first.metadata.pair_ref

    @property
    def episodes(self) -> tuple[ContingencyEpisode, ContingencyEpisode]:
        return (self.first, self.second)

    def deranged_acquisition_outcomes(
        self, episode: ContingencyEpisode
    ) -> tuple[OutcomeLabel, OutcomeLabel]:
        """Return the opposite twin's anchors as a balanced wrong-history arm."""

        if episode == self.first:
            return self.second.supervision.acquisition_outcomes
        if episode == self.second:
            return self.first.supervision.acquisition_outcomes
        raise ValueError("episode does not belong to this counterfactual pair")


def _event_text(
    family: object,
    *,
    scope: SurfaceScope,
    pair_index: int,
    event_index: int,
    relation_class: RelationClass,
) -> tuple[str, str]:
    context_index = pair_index * EVENTS_PER_EPISODE + event_index
    context = v13._context(scope, context_index)
    reference_plan, attempt_plan = v13._renderer_pair(context_index)
    steps = family.support_steps
    reference_sentences = v13._rendered_sentences(steps, context, plan=reference_plan)
    reference = v13._trace_from_sentences(steps, reference_sentences)
    relation = "reorder" if relation_class == "alpha" else "replacement"
    changed = v13._changed_steps(
        steps,
        relation,
        token=700_000 + pair_index * 31 + event_index,
    )
    attempt = (
        v13._trace_from_sentences(changed, reference_sentences)
        if relation_class == "alpha"
        else v13._trace_text(changed, context, plan=attempt_plan)
    )
    return reference, attempt


def _relation_classes(pair_index: int) -> tuple[RelationClass, ...]:
    if type(pair_index) is not int or pair_index < 0:
        raise ValueError("pair index must be non-negative")
    if pair_index % 2 == 0:
        return _BASE_RELATION_CLASSES
    return tuple(
        "beta" if value == "alpha" else "alpha"
        for value in _BASE_RELATION_CLASSES
    )


def _outcomes(
    regime: HiddenRegime,
    relation_classes: tuple[RelationClass, ...],
) -> tuple[OutcomeLabel, ...]:
    return tuple(
        _label(regime[0] if relation == "alpha" else regime[1])
        for relation in relation_classes
    )


def _build_pair(
    family: object,
    *,
    scope: SurfaceScope,
    pair_index: int,
    family_local_index: int,
    ordinal_base: int,
) -> CounterfactualTwinPair:
    del family_local_index  # balance evidence is carried by deterministic loop position
    public_events = []
    relation_classes = _relation_classes(pair_index)
    for event_index, relation_class in enumerate(relation_classes):
        reference, attempt = _event_text(
            family,
            scope=scope,
            pair_index=pair_index,
            event_index=event_index,
            relation_class=relation_class,
        )
        public_events.append(
            PublicContingencyEvent(
                reference_trace_text=reference,
                attempt_trace_text=attempt,
                temporal=MovingOriginCoordinates(
                    acquired_ordinal=ordinal_base + pair_index * EVENTS_PER_EPISODE + event_index,
                    age=EVENTS_PER_EPISODE - event_index - 1,
                    landmark_relations=((
                        "episode_start",
                        "AT" if event_index == 0 else "AFTER",
                    ),),
                ),
            )
        )
    public = PublicContingencyEpisode(tuple(public_events))
    first_regime = _REGIMES[pair_index % len(_REGIMES)]
    second_regime = _opposite(first_regime)
    partition = _SCOPE_PARTITION[scope]
    pair_ref = f"{partition}-v15-counterfactual-pair-{family.ordinal:02d}-{pair_index:04d}"
    first_ref = pair_ref + "-a"
    second_ref = pair_ref + "-b"

    def episode(regime: HiddenRegime, episode_ref: str, twin_ref: str) -> ContingencyEpisode:
        return ContingencyEpisode(
            public=public,
            supervision=ContingencyEpisodeSupervision(
                outcome_labels=_outcomes(regime, relation_classes),
                relation_classes=relation_classes,
            ),
            metadata=ContingencyEpisodeMetadata(
                partition=partition,
                pair_ref=pair_ref,
                episode_ref=episode_ref,
                twin_ref=twin_ref,
                generator_family=family.key,
                transition_group=family.transition,
                hidden_regime=regime,
                structure_signature=family.support_signature,
            ),
        )

    return CounterfactualTwinPair(
        first=episode(first_regime, first_ref, second_ref),
        second=episode(second_regime, second_ref, first_ref),
    )


def _materialize_partition(
    families: tuple[object, ...],
    *,
    scope: SurfaceScope,
    ordinal_base: int,
) -> tuple[CounterfactualTwinPair, ...]:
    values = []
    pair_index = 0
    for family in families:
        for family_local_index in range(_MECHANISMS_PER_FAMILY[scope]):
            values.append(
                _build_pair(
                    family,
                    scope=scope,
                    pair_index=pair_index,
                    family_local_index=family_local_index,
                    ordinal_base=ordinal_base,
                )
            )
            pair_index += 1
    return tuple(values)


def _training_schedule() -> tuple[tuple[int, ...], ...]:
    if math.gcd(SCHEDULE_MULTIPLIER, TRAIN_PAIRS) != 1:
        raise RuntimeError("V15 schedule multiplier must be coprime")
    flat = tuple(
        (position * SCHEDULE_MULTIPLIER) % TRAIN_PAIRS
        for position in range(TRAIN_UPDATES * PAIRS_PER_UPDATE)
    )
    if Counter(flat) != Counter({index: 2 for index in range(TRAIN_PAIRS)}):
        raise RuntimeError("V15 schedule must expose every pair exactly twice")
    return tuple(
        flat[start : start + PAIRS_PER_UPDATE]
        for start in range(0, len(flat), PAIRS_PER_UPDATE)
    )


@dataclass(frozen=True, slots=True)
class PairedLatentContingencyCreditV15Corpus:
    train: tuple[CounterfactualTwinPair, ...]
    development: tuple[CounterfactualTwinPair, ...]
    _final: tuple[CounterfactualTwinPair, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.train) != TRAIN_PAIRS:
            raise ValueError("V15 train pair count is invalid")
        if len(self.development) != DEVELOPMENT_PAIRS:
            raise ValueError("V15 development pair count is invalid")
        if self._final is not None and len(self._final) != FINAL_PAIRS:
            raise ValueError("V15 final pair count is invalid")
        if any(not episode.gradient_authorized for pair in self.train for episode in pair.episodes):
            raise ValueError("V15 train episodes must authorize gradients")
        if any(episode.gradient_authorized for pair in self.development for episode in pair.episodes):
            raise ValueError("V15 development episodes may not authorize gradients")
        if self._final is not None and any(episode.gradient_authorized for pair in self._final for episode in pair.episodes):
            raise ValueError("V15 final episodes may not authorize gradients")

    @property
    def final(self) -> tuple[CounterfactualTwinPair, ...]:
        if self._final is None:
            raise FinalPartitionSealedError(
                "final remains sealed until the V15 development gate authorizes it"
            )
        return self._final

    @property
    def final_is_open(self) -> bool:
        return self._final is not None

    def partition(self, name: str) -> tuple[CounterfactualTwinPair, ...]:
        if name == "train":
            return self.train
        if name == "development":
            return self.development
        if name == "final":
            return self.final
        raise ValueError("V15 partition name is invalid")

    @staticmethod
    def training_schedule() -> tuple[tuple[int, ...], ...]:
        return _training_schedule()

    def train_pairs_for_update(self, update_index: int) -> tuple[CounterfactualTwinPair, ...]:
        if type(update_index) is not int or not 0 <= update_index < TRAIN_UPDATES:
            raise ValueError("V15 update index is outside the frozen schedule")
        return tuple(self.train[index] for index in _training_schedule()[update_index])

    def learner_payload_size_bytes(self) -> int:
        payload = [
            episode.to_learner_payload()
            for pairs in (self.train, self.development)
            for pair in pairs
            for episode in pair.episodes
        ]
        if self._final is not None:
            payload.extend(
                episode.to_learner_payload()
                for pair in self._final
                for episode in pair.episodes
            )
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def build_paired_latent_contingency_credit_v15(
    *, include_final: bool = False
) -> PairedLatentContingencyCreditV15Corpus:
    if type(include_final) is not bool:
        raise TypeError("include_final must be bool")
    train = _materialize_partition(
        tuple(v13.v12._TRAIN_FAMILY_VALUES),
        scope="inner",
        ordinal_base=0,
    )
    development = _materialize_partition(
        tuple(v13.v12._DEVELOPMENT_FAMILY_VALUES),
        scope="development",
        ordinal_base=2_000_000,
    )
    final = (
        _materialize_partition(
            tuple(v13.v12._FINAL_FAMILY_VALUES),
            scope="final",
            ordinal_base=3_000_000,
        )
        if include_final
        else None
    )
    return PairedLatentContingencyCreditV15Corpus(
        train=train,
        development=development,
        _final=final,
    )


__all__ = [
    "ACQUISITION_INDICES",
    "CORPUS_ID",
    "CounterfactualTwinPair",
    "ContingencyEpisode",
    "ContingencyEpisodeMetadata",
    "ContingencyEpisodeSupervision",
    "DEVELOPMENT_FAMILIES",
    "DEVELOPMENT_PAIRS",
    "EVENTS_PER_EPISODE",
    "FINAL_FAMILIES",
    "FINAL_PAIRS",
    "FinalPartitionSealedError",
    "PAIRS_PER_UPDATE",
    "PROBE_INDICES",
    "PairedLatentContingencyCreditV15Corpus",
    "PublicContingencyEpisode",
    "PublicContingencyEvent",
    "SCHEDULE_MULTIPLIER",
    "TRAIN_FAMILIES",
    "TRAIN_PAIRS",
    "TRAIN_UPDATES",
    "build_paired_latent_contingency_credit_v15",
]
