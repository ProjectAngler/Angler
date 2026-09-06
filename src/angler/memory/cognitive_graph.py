"""Typed, rebuildable cognitive-record projection and temporal rejoin.

Canonical episode bytes remain in the cognitive transaction store.  This
module derives one deterministic EPISODIC record per episode, lets an
untrusted backend index only a content-addressed projection, and reconstructs
every returned reference from the canonical episode before exposing it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    RelationType,
)
from angler.episodes.canonical import canonical_bytes, parse_json

from .moving_origin import MovingOriginIndex, TemporalPosition


_CONTRACT = "ANG-CTR-COGNITIVE-GRAPH-PROJECTION-001@0.1.0"
_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


def _ref(value: object, label: str) -> str:
    if type(value) is not str or _REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    return value.strip()


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{label} fields do not match the contract")


@dataclass(frozen=True, slots=True)
class CognitiveGraphProjection:
    """Strict content-addressed envelope accepted by reference backends."""

    source_episode_ref: str
    record: CognitiveMemoryRecord

    def __post_init__(self) -> None:
        _ref(self.source_episode_ref, "source_episode_ref")
        if not isinstance(self.record, CognitiveMemoryRecord):
            raise TypeError("record must be a CognitiveMemoryRecord")
        if self.source_episode_ref not in self.record.provenance_refs:
            raise ValueError("projection record does not cite its source episode")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": _CONTRACT,
            "record": self.record.to_payload(),
            "source_episode_ref": self.source_episode_ref,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @property
    def projection_ref(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CognitiveGraphProjection":
        _exact_keys(payload, {"contract", "record", "source_episode_ref"}, _CONTRACT)
        if payload["contract"] != _CONTRACT:
            raise ValueError("unsupported cognitive graph projection contract")
        record_payload = payload["record"]
        if not isinstance(record_payload, Mapping):
            raise TypeError("projection record must be an object")
        return cls(
            source_episode_ref=payload["source_episode_ref"],
            record=CognitiveMemoryRecord.from_payload(record_payload),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CognitiveGraphProjection":
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        if not isinstance(encoded, bytes):
            raise TypeError("projection JSON must be text or bytes")
        payload = parse_json(encoded)
        if not isinstance(payload, dict) or canonical_bytes(payload) != encoded:
            raise ValueError("projection JSON is not canonical")
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class CognitiveReferenceHit:
    """Untrusted backend candidate containing references, never record content."""

    source_episode_ref: str
    record_ref: str
    adjacent_record_refs: tuple[str, ...] = ()
    score: float | None = None
    backend_ref: str | None = None

    def __post_init__(self) -> None:
        _ref(self.source_episode_ref, "source_episode_ref")
        _ref(self.record_ref, "record_ref")
        if type(self.adjacent_record_refs) is not tuple:
            raise TypeError("adjacent_record_refs must be a tuple")
        for item in self.adjacent_record_refs:
            _ref(item, "adjacent record reference")
        if len(set(self.adjacent_record_refs)) != len(self.adjacent_record_refs):
            raise ValueError("adjacent_record_refs must be unique")
        if self.score is not None and (
            type(self.score) not in (int, float) or not math.isfinite(float(self.score))
        ):
            raise ValueError("score must be finite or None")
        if self.backend_ref is not None:
            _text(self.backend_ref, "backend_ref", 512)


@runtime_checkable
class CognitiveReferenceBackend(Protocol):
    async def project(self, projection: CognitiveGraphProjection) -> str | None: ...

    async def search(self, query: str, *, limit: int) -> Sequence[CognitiveReferenceHit]: ...

    async def forget_namespace(self) -> None: ...


@runtime_checkable
class CanonicalEpisodeSource(Protocol):
    def get_episode_item(self, episode_ref: str) -> object: ...

    def episode_items(
        self, after_sequence: int = 0, limit: int = 128
    ) -> Sequence[object]: ...


def _item_parts(item: object) -> tuple[int, str, CognitiveEpisode]:
    try:
        sequence = getattr(item, "sequence")
        episode_ref = getattr(item, "episode_ref")
        episode = getattr(item, "episode")
    except AttributeError as exc:
        raise TypeError("episode item does not implement the canonical item contract") from exc
    if type(sequence) is not int or sequence < 1:
        raise ValueError("episode sequence must be a positive integer")
    _ref(episode_ref, "episode_ref")
    if not isinstance(episode, CognitiveEpisode) or episode.episode_ref != episode_ref:
        raise ValueError("episode item identity does not match canonical episode bytes")
    return sequence, episode_ref, episode


def _bounded_content(episode: CognitiveEpisode) -> str:
    return (
        "Request:\n"
        + episode.request[:4_096]
        + "\nSelected procedure:\n"
        + episode.proposals[episode.selected_index][:4_096]
        + "\nResponse:\n"
        + episode.response[:4_096]
        + "\nObjective outcome:\n"
        + episode.outcome
        + "\nFeedback:\n"
        + episode.feedback_text[:2_048]
    )


def episode_record(item: object) -> CognitiveMemoryRecord:
    """Derive the sole typed EPISODIC view without inference or model calls."""

    sequence, episode_ref, episode = _item_parts(item)
    relations = [
        CognitiveRelation(RelationType.ACTUAL_OF, episode.commitment.commitment_ref)
    ]
    about_refs = set(episode.recalled_refs).union(episode.supporting_evidence_refs)
    relations.extend(
        CognitiveRelation(RelationType.ABOUT, target_ref)
        for target_ref in sorted(about_refs)
    )
    relations.sort(key=lambda value: (value.relation_type.value, value.target_ref))
    return CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content=_bounded_content(episode),
        provenance_refs=tuple(sorted({episode_ref, episode.feedback_source_ref})),
        visibility=episode.visibility,
        producer_id="angler.cognitive-cycle",
        producer_checkpoint_ref=episode.model_ref,
        competence_ref=episode.child_state_digest,
        acquired_ordinal=sequence - 1,
        world_valid_from=None,
        world_valid_until=None,
        relations=tuple(relations),
    )


@dataclass(frozen=True, slots=True)
class TypedCognitiveRecall:
    record: CognitiveMemoryRecord
    source_episode_ref: str
    position: TemporalPosition
    adjacent_records: tuple[CognitiveMemoryRecord, ...]
    backend_score: float | None
    backend_ref: str | None
    world_valid_at_query: bool | None

    @property
    def artifact_ref(self) -> str:
        return self.record.record_ref

    @property
    def text(self) -> str:
        return self.record.content

    @property
    def source_ref(self) -> str:
        return self.source_episode_ref

    @property
    def context(self) -> tuple[tuple[str, str], ...]:
        """Existing recall surface without weakening the typed record."""

        return (
            ("epistemic_status", self.record.epistemic_status.value),
            ("kind", self.record.kind.value),
            ("producer_checkpoint_ref", self.record.producer_checkpoint_ref),
        )

    @property
    def visibility(self) -> str:
        return self.record.visibility

    @property
    def acquired_ordinal(self) -> int:
        return self.position.acquired_ordinal

    @property
    def age(self) -> int:
        return self.position.age

    @property
    def landmark_relations(self) -> tuple[tuple[str, str], ...]:
        return self.position.landmark_relations

    @property
    def world_valid_from(self) -> int | None:
        return self.position.world_valid_from

    @property
    def world_valid_until(self) -> int | None:
        return self.position.world_valid_until


@dataclass(frozen=True, slots=True)
class TypedCognitiveRecallBatch:
    items: tuple[TypedCognitiveRecall, ...]
    rejected: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypedProjectionReceipt:
    source_episode_ref: str
    record_ref: str
    projection_ref: str
    acquired_ordinal: int
    backend_ref: str | None


class TypedSituatedMemory:
    """Canonical-reference recall joined to one rebuildable temporal origin."""

    def __init__(
        self,
        *,
        source: CanonicalEpisodeSource,
        backend: CognitiveReferenceBackend,
        origin: MovingOriginIndex | None = None,
    ) -> None:
        if not isinstance(source, CanonicalEpisodeSource):
            raise TypeError("source does not implement CanonicalEpisodeSource")
        if not isinstance(backend, CognitiveReferenceBackend):
            raise TypeError("backend does not implement CognitiveReferenceBackend")
        self.source = source
        self.backend = backend
        self.origin = origin or MovingOriginIndex()
        # Disposable reverse index.  Values are canonical episode refs and are
        # rebuilt from the transaction store, never accepted from the backend.
        self._source_by_record_ref: dict[str, str] = {}

    @staticmethod
    def _synchronization_error(detail: str) -> ValueError:
        return ValueError(
            "typed memory is not synchronized with the canonical store; "
            f"rebuild_from_store is required ({detail})"
        )

    def assert_synchronized(
        self, expected_sequence: int, expected_episode_ref: str | None
    ) -> None:
        """Prove the disposable views match one canonical head in O(1).

        This check deliberately never pages history.  A failed assertion tells
        the caller to invoke the explicit rebuild operation before recall.
        """

        if type(expected_sequence) is not int or expected_sequence < 0:
            raise ValueError("expected_sequence must be a non-negative integer")
        if expected_sequence == 0:
            if expected_episode_ref is not None:
                raise self._synchronization_error(
                    "sequence zero cannot identify a head episode"
                )
            if self.origin.size != 0 or self._source_by_record_ref:
                raise self._synchronization_error(
                    "sequence zero requires empty temporal and reverse indexes"
                )
            return

        if expected_episode_ref is None:
            raise self._synchronization_error(
                "a nonzero sequence requires a canonical head episode"
            )
        expected_episode_ref = _ref(expected_episode_ref, "expected_episode_ref")
        if (
            self.origin.size != expected_sequence
            or len(self._source_by_record_ref) != expected_sequence
        ):
            raise self._synchronization_error(
                "temporal or reverse-index cardinality differs from the head sequence"
            )
        try:
            item = self.source.get_episode_item(expected_episode_ref)
            item_sequence, item_episode_ref, _episode = _item_parts(item)
            projection = self._projection(item)
            anchor = self.origin.anchor(projection.record.record_ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise self._synchronization_error(
                "canonical head cannot be rejoined to the typed projection"
            ) from exc
        if item_sequence != expected_sequence or item_episode_ref != expected_episode_ref:
            raise self._synchronization_error(
                "canonical head item does not match the expected sequence and reference"
            )
        if anchor.ordinal != expected_sequence - 1:
            raise self._synchronization_error(
                "canonical head is not the final Moving Origin anchor"
            )
        if not self._anchor_matches_projection(anchor, projection):
            raise self._synchronization_error(
                "canonical head anchor bytes or validity bounds differ"
            )
        if (
            self._source_by_record_ref.get(projection.record.record_ref)
            != expected_episode_ref
        ):
            raise self._synchronization_error(
                "canonical head reverse mapping differs"
            )

    @staticmethod
    def _projection(item: object) -> CognitiveGraphProjection:
        _sequence, episode_ref, _episode = _item_parts(item)
        return CognitiveGraphProjection(episode_ref, episode_record(item))

    @staticmethod
    def _anchor_matches_projection(anchor: object, projection: CognitiveGraphProjection) -> bool:
        record = projection.record
        return (
            getattr(anchor, "ordinal", None) == record.acquired_ordinal
            and getattr(anchor, "world_valid_from", None) == record.world_valid_from
            and getattr(anchor, "world_valid_until", None) == record.world_valid_until
            and hmac.compare_digest(
                getattr(anchor, "projection_id", ""), projection.projection_ref
            )
        )

    async def _project_with_origin(
        self,
        item: object,
        origin: MovingOriginIndex,
        source_by_record_ref: dict[str, str],
    ) -> TypedProjectionReceipt:
        projection = self._projection(item)
        record = projection.record
        try:
            anchor = origin.anchor(record.record_ref)
        except KeyError:
            if origin.size != record.acquired_ordinal:
                raise ValueError("record acquisition ordinal is not the next Moving Origin position")
            anchor = origin.append(
                record.record_ref,
                projection.projection_ref,
                world_valid_from=record.world_valid_from,
                world_valid_until=record.world_valid_until,
            )
        else:
            if not self._anchor_matches_projection(anchor, projection):
                raise ValueError("existing Moving Origin anchor does not match projection")
        previous_source = source_by_record_ref.get(record.record_ref)
        if previous_source is not None and previous_source != projection.source_episode_ref:
            raise ValueError("record reference maps to multiple canonical episodes")
        source_by_record_ref[record.record_ref] = projection.source_episode_ref
        backend_ref = await self.backend.project(projection)
        return TypedProjectionReceipt(
            source_episode_ref=projection.source_episode_ref,
            record_ref=record.record_ref,
            projection_ref=projection.projection_ref,
            acquired_ordinal=anchor.ordinal,
            backend_ref=backend_ref,
        )

    async def remember_episode_item(self, item: object) -> TypedProjectionReceipt:
        return await self._project_with_origin(
            item, self.origin, self._source_by_record_ref
        )

    async def reproject_episode(self, source_episode_ref: str) -> TypedProjectionReceipt:
        item = self.source.get_episode_item(_ref(source_episode_ref, "source_episode_ref"))
        projection = self._projection(item)
        try:
            anchor = self.origin.anchor(projection.record.record_ref)
        except KeyError as exc:
            raise ValueError("reprojection requires an existing Moving Origin anchor") from exc
        if not self._anchor_matches_projection(anchor, projection):
            raise ValueError("reprojection bytes do not match the existing anchor")
        previous_source = self._source_by_record_ref.get(projection.record.record_ref)
        if previous_source is not None and previous_source != projection.source_episode_ref:
            raise ValueError("record reference maps to multiple canonical episodes")
        self._source_by_record_ref[
            projection.record.record_ref
        ] = projection.source_episode_ref
        backend_ref = await self.backend.project(projection)
        return TypedProjectionReceipt(
            source_episode_ref=projection.source_episode_ref,
            record_ref=projection.record.record_ref,
            projection_ref=projection.projection_ref,
            acquired_ordinal=anchor.ordinal,
            backend_ref=backend_ref,
        )

    def _resolve_hit(
        self, hit: CognitiveReferenceHit, allowed_visibility: frozenset[str]
    ) -> tuple[CognitiveMemoryRecord, CognitiveGraphProjection] | str:
        try:
            item = self.source.get_episode_item(hit.source_episode_ref)
            projection = self._projection(item)
        except (KeyError, TypeError, ValueError):
            return "REFERENCE_UNKNOWN_OR_INVALID"
        record = projection.record
        if not hmac.compare_digest(record.record_ref, hit.record_ref):
            return "REFERENCE_IDENTITY_MISMATCH"
        if record.visibility not in allowed_visibility:
            return "REFERENCE_VISIBILITY_DENIED"
        try:
            anchor = self.origin.anchor(record.record_ref)
        except KeyError:
            return "REFERENCE_TEMPORAL_UNKNOWN"
        if not self._anchor_matches_projection(anchor, projection):
            return "REFERENCE_STALE_OR_TAMPERED"
        return record, projection

    def _resolve_record_ref(
        self, record_ref: str, allowed_visibility: frozenset[str]
    ) -> CognitiveMemoryRecord | str:
        """Resolve an adjacency independently of every backend seed hit."""

        source_episode_ref = self._source_by_record_ref.get(record_ref)
        if source_episode_ref is None:
            return "ADJACENT_REFERENCE_UNKNOWN"
        synthetic_hit = CognitiveReferenceHit(
            source_episode_ref=source_episode_ref,
            record_ref=record_ref,
        )
        resolved = self._resolve_hit(synthetic_hit, allowed_visibility)
        if isinstance(resolved, str):
            return "ADJACENT_" + resolved.removeprefix("REFERENCE_")
        record, _projection = resolved
        return record

    @staticmethod
    def _world_valid(
        position: TemporalPosition, world_time: int | None
    ) -> bool | None:
        if world_time is None:
            return None
        if position.world_valid_from is not None and world_time < position.world_valid_from:
            return False
        if position.world_valid_until is not None and world_time > position.world_valid_until:
            return False
        return True

    async def recall(
        self,
        query: str,
        *,
        limit: int = 15,
        allowed_visibility: Sequence[str] = ("LEARNER_VISIBLE",),
        world_time: int | None = None,
        frozen_origin: bool = False,
    ) -> TypedCognitiveRecallBatch:
        query = _text(query, "query", 16_384)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if world_time is not None and type(world_time) is not int:
            raise ValueError("world_time must be an integer or None")
        allowed = frozenset(allowed_visibility)
        backend_hits = await self.backend.search(query, limit=limit)
        if not isinstance(backend_hits, Sequence) or isinstance(
            backend_hits, (str, bytes)
        ):
            raise TypeError("reference backend search must return a sequence")
        if len(backend_hits) > limit:
            raise ValueError("reference backend returned more hits than requested")
        raw_hits = tuple(backend_hits)
        resolved: list[tuple[CognitiveReferenceHit, CognitiveMemoryRecord]] = []
        rejected: list[str] = []
        seen_record_refs: set[str] = set()
        for hit in raw_hits:
            if not isinstance(hit, CognitiveReferenceHit):
                rejected.append("REFERENCE_RESULT_INVALID")
                continue
            result = self._resolve_hit(hit, allowed)
            if isinstance(result, str):
                rejected.append(result)
                continue
            record, _projection = result
            if record.record_ref in seen_record_refs:
                rejected.append("REFERENCE_DUPLICATE")
                continue
            seen_record_refs.add(record.record_ref)
            resolved.append((hit, record))

        items = []
        for hit, record in resolved:
            adjacent = []
            for adjacent_ref in hit.adjacent_record_refs:
                candidate = self._resolve_record_ref(adjacent_ref, allowed)
                if isinstance(candidate, str):
                    rejected.append(candidate)
                    continue
                adjacent.append(candidate)
            position = (
                self.origin.frozen_position(record.record_ref)
                if frozen_origin
                else self.origin.position(record.record_ref)
            )
            items.append(
                TypedCognitiveRecall(
                    record=record,
                    source_episode_ref=hit.source_episode_ref,
                    position=position,
                    adjacent_records=tuple(adjacent),
                    backend_score=None if hit.score is None else float(hit.score),
                    backend_ref=hit.backend_ref,
                    world_valid_at_query=self._world_valid(position, world_time),
                )
            )
        return TypedCognitiveRecallBatch(tuple(items), tuple(rejected))

    async def rebuild_from_store(self, *, page_size: int = 128) -> tuple[str, ...]:
        if type(page_size) is not int or not 1 <= page_size <= 256:
            raise ValueError("page_size must be an integer from 1 through 256")
        await self.backend.forget_namespace()
        rebuilt_origin = MovingOriginIndex()
        rebuilt_sources: dict[str, str] = {}
        record_refs: list[str] = []
        after_sequence = 0
        while True:
            page = tuple(
                self.source.episode_items(
                    after_sequence=after_sequence,
                    limit=page_size,
                )
            )
            if not page:
                break
            expected = after_sequence + 1
            for item in page:
                sequence, _episode_ref, _episode = _item_parts(item)
                if sequence != expected:
                    raise ValueError("canonical episode page is not contiguous and ordered")
                receipt = await self._project_with_origin(
                    item, rebuilt_origin, rebuilt_sources
                )
                record_refs.append(receipt.record_ref)
                expected += 1
            after_sequence = expected - 1
            if len(page) < page_size:
                break
        self.origin = rebuilt_origin
        self._source_by_record_ref = rebuilt_sources
        return tuple(record_refs)


__all__ = [
    "CanonicalEpisodeSource",
    "CognitiveGraphProjection",
    "CognitiveReferenceBackend",
    "CognitiveReferenceHit",
    "TypedCognitiveRecall",
    "TypedCognitiveRecallBatch",
    "TypedProjectionReceipt",
    "TypedSituatedMemory",
    "episode_record",
]
