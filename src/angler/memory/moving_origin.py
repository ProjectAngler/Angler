"""An Angler-native moving-origin temporal index.

The design is informed by Moving Origin Research's E1 mechanism: one
monotonic autobiographical position makes self-relative age an O(1)
calculation, while landmark coordinates remain explicitly tied to designation
events.  Canonical evidence stays outside this index; this state is a
rebuildable temporal projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


BEFORE = "BEFORE"
AT = "AT"
AFTER = "AFTER"


@dataclass(frozen=True, slots=True)
class TemporalAnchor:
    event_ref: str
    ordinal: int
    projection_id: str
    previous_event_ref: str | None
    world_valid_from: int | None = None
    world_valid_until: int | None = None
    landmarks_at_encoding: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Landmark:
    name: str
    target_event_ref: str
    designation_event_ref: str
    designated_at: int


@dataclass(frozen=True, slots=True)
class TemporalPosition:
    acquired_ordinal: int
    age: int
    landmark_relations: tuple[tuple[str, str], ...]
    world_valid_from: int | None
    world_valid_until: int | None


@dataclass(frozen=True, slots=True)
class RecentResult:
    event_refs: tuple[str, ...]
    inspected: int


class MovingOriginIndex:
    """One monotonic autobiographical clock with constant-time re-resolution."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._anchors: dict[str, TemporalAnchor] = {}
        self._landmarks: dict[str, Landmark] = {}

    @property
    def now(self) -> int:
        return len(self._order) - 1

    @property
    def size(self) -> int:
        return len(self._order)

    def append(
        self,
        event_ref: str,
        projection_id: str,
        *,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> TemporalAnchor:
        if type(event_ref) is not str or not event_ref:
            raise ValueError("event_ref must be non-empty text")
        if type(projection_id) is not str or not projection_id:
            raise ValueError("projection_id must be non-empty text")
        if event_ref in self._anchors:
            raise ValueError("event_ref already has an autobiographical position")
        if (
            world_valid_from is not None
            and world_valid_until is not None
            and world_valid_until < world_valid_from
        ):
            raise ValueError("world_valid_until cannot precede world_valid_from")
        previous = self._order[-1] if self._order else None
        ordinal = len(self._order)
        anchor = TemporalAnchor(
            event_ref=event_ref,
            ordinal=ordinal,
            projection_id=projection_id,
            previous_event_ref=previous,
            world_valid_from=world_valid_from,
            world_valid_until=world_valid_until,
            landmarks_at_encoding=tuple(sorted(self._landmarks)),
        )
        self._anchors[event_ref] = anchor
        self._order.append(event_ref)
        return anchor

    def designate_landmark(
        self,
        name: str,
        *,
        target_event_ref: str,
        designation_event_ref: str,
        projection_id: str,
    ) -> Landmark:
        if type(name) is not str or not name or name in self._landmarks:
            raise ValueError("landmark name must be non-empty and unique")
        if target_event_ref not in self._anchors:
            raise KeyError("landmark target is not in the autobiography")
        designation = self.append(designation_event_ref, projection_id)
        landmark = Landmark(
            name=name,
            target_event_ref=target_event_ref,
            designation_event_ref=designation_event_ref,
            designated_at=designation.ordinal,
        )
        self._landmarks[name] = landmark
        return landmark

    def anchor(self, event_ref: str) -> TemporalAnchor:
        try:
            return self._anchors[event_ref]
        except KeyError as exc:
            raise KeyError("event is not in the autobiography") from exc

    @staticmethod
    def _relation(ordinal: int, designated_at: int) -> str:
        if ordinal < designated_at:
            return BEFORE
        if ordinal > designated_at:
            return AFTER
        return AT

    def position(self, event_ref: str) -> TemporalPosition:
        anchor = self.anchor(event_ref)
        relations = tuple(
            (name, self._relation(anchor.ordinal, landmark.designated_at))
            for name, landmark in sorted(self._landmarks.items())
        )
        return TemporalPosition(
            acquired_ordinal=anchor.ordinal,
            age=self.now - anchor.ordinal,
            landmark_relations=relations,
            world_valid_from=anchor.world_valid_from,
            world_valid_until=anchor.world_valid_until,
        )

    def frozen_position(self, event_ref: str) -> TemporalPosition:
        """Option-E-style control: retain content but never re-resolve origin."""

        anchor = self.anchor(event_ref)
        relations = tuple(
            (
                name,
                self._relation(anchor.ordinal, self._landmarks[name].designated_at),
            )
            for name in anchor.landmarks_at_encoding
        )
        return TemporalPosition(
            acquired_ordinal=anchor.ordinal,
            age=0,
            landmark_relations=relations,
            world_valid_from=anchor.world_valid_from,
            world_valid_until=anchor.world_valid_until,
        )

    def recent(self, limit: int) -> RecentResult:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        selected = tuple(reversed(self._order[-limit:]))
        return RecentResult(event_refs=selected, inspected=len(selected))

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": "angler.moving-origin.v1",
            "anchors": [
                {
                    "event_ref": anchor.event_ref,
                    "ordinal": anchor.ordinal,
                    "projection_id": anchor.projection_id,
                    "previous_event_ref": anchor.previous_event_ref,
                    "world_valid_from": anchor.world_valid_from,
                    "world_valid_until": anchor.world_valid_until,
                    "landmarks_at_encoding": list(anchor.landmarks_at_encoding),
                }
                for anchor in (self._anchors[event_ref] for event_ref in self._order)
            ],
            "landmarks": [
                {
                    "name": landmark.name,
                    "target_event_ref": landmark.target_event_ref,
                    "designation_event_ref": landmark.designation_event_ref,
                    "designated_at": landmark.designated_at,
                }
                for _, landmark in sorted(self._landmarks.items())
            ],
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "MovingOriginIndex":
        if snapshot.get("version") != "angler.moving-origin.v1":
            raise ValueError("unsupported moving-origin snapshot")
        restored = cls()
        expected_previous = None
        for expected_ordinal, item in enumerate(snapshot.get("anchors", [])):
            if item.get("ordinal") != expected_ordinal:
                raise ValueError("autobiographical ordinals must be contiguous")
            if item.get("previous_event_ref") != expected_previous:
                raise ValueError("autobiographical predecessor chain is invalid")
            anchor = TemporalAnchor(
                event_ref=item["event_ref"],
                ordinal=item["ordinal"],
                projection_id=item["projection_id"],
                previous_event_ref=item["previous_event_ref"],
                world_valid_from=item.get("world_valid_from"),
                world_valid_until=item.get("world_valid_until"),
                landmarks_at_encoding=tuple(item.get("landmarks_at_encoding", [])),
            )
            if anchor.event_ref in restored._anchors:
                raise ValueError("snapshot contains a duplicate event_ref")
            restored._anchors[anchor.event_ref] = anchor
            restored._order.append(anchor.event_ref)
            expected_previous = anchor.event_ref
        for item in snapshot.get("landmarks", []):
            landmark = Landmark(
                name=item["name"],
                target_event_ref=item["target_event_ref"],
                designation_event_ref=item["designation_event_ref"],
                designated_at=item["designated_at"],
            )
            if landmark.name in restored._landmarks:
                raise ValueError("snapshot contains a duplicate landmark")
            if landmark.target_event_ref not in restored._anchors:
                raise ValueError("landmark target is absent")
            designation = restored.anchor(landmark.designation_event_ref)
            if designation.ordinal != landmark.designated_at:
                raise ValueError("landmark designation position is invalid")
            restored._landmarks[landmark.name] = landmark
        known_landmarks = set(restored._landmarks)
        for anchor in restored._anchors.values():
            if not set(anchor.landmarks_at_encoding).issubset(known_landmarks):
                raise ValueError("anchor references an unknown landmark")
        return restored


class FairNaiveTemporalView:
    """Correct scan-based control with the same answers and explicit work."""

    def __init__(
        self,
        anchors: Iterable[TemporalAnchor],
        landmarks: Iterable[Landmark] = (),
    ) -> None:
        self._anchors = tuple(anchors)
        self._landmarks = tuple(landmarks)

    @classmethod
    def from_index(cls, index: MovingOriginIndex) -> "FairNaiveTemporalView":
        return cls(
            (index.anchor(event_ref) for event_ref in index._order),
            index._landmarks.values(),
        )

    def position(self, event_ref: str) -> tuple[TemporalPosition, int]:
        """Recompute one live position by scanning the complete history."""

        selected = None
        now = -1
        inspected = 0
        for anchor in self._anchors:
            inspected += 1
            now = max(now, anchor.ordinal)
            if anchor.event_ref == event_ref:
                selected = anchor
        if selected is None:
            raise KeyError("event is not in the autobiography")
        relations = tuple(
            (
                landmark.name,
                MovingOriginIndex._relation(selected.ordinal, landmark.designated_at),
            )
            for landmark in sorted(self._landmarks, key=lambda item: item.name)
        )
        return (
            TemporalPosition(
                acquired_ordinal=selected.ordinal,
                age=now - selected.ordinal,
                landmark_relations=relations,
                world_valid_from=selected.world_valid_from,
                world_valid_until=selected.world_valid_until,
            ),
            inspected,
        )

    def recent(self, limit: int) -> RecentResult:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        ordered = sorted(self._anchors, key=lambda item: item.ordinal, reverse=True)
        return RecentResult(
            event_refs=tuple(item.event_ref for item in ordered[:limit]),
            inspected=len(self._anchors),
        )


__all__ = [
    "AFTER",
    "AT",
    "BEFORE",
    "FairNaiveTemporalView",
    "Landmark",
    "MovingOriginIndex",
    "RecentResult",
    "TemporalAnchor",
    "TemporalPosition",
]
