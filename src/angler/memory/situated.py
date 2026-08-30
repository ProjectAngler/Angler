"""Compose Angler evidence, Moving Origin coordinates, and Cognee recall."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Iterable

from angler.episodes.canonical import canonical_bytes, parse_json

from .contracts import (
    MemoryProjection,
    ProjectionBackend,
    RecallBatch,
    SituatedRecall,
)
from .moving_origin import MovingOriginIndex


_BEGIN = "ANGLER_SITUATED_MEMORY_V1\n"
_END = "\nEND_ANGLER_SITUATED_MEMORY_V1"


def _payload(projection: MemoryProjection) -> dict[str, Any]:
    return {
        "version": "angler.situated-memory.v1",
        "artifact_ref": projection.artifact_ref,
        "text": projection.text,
        "source_ref": projection.source_ref,
        "visibility": projection.visibility,
        "context": [[key, value] for key, value in projection.context],
    }


def projection_id(projection: MemoryProjection) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(_payload(projection))).hexdigest()


def encode_projection(projection: MemoryProjection) -> str:
    body = dict(_payload(projection))
    body["projection_id"] = projection_id(projection)
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return _BEGIN + encoded + _END


def decode_projection(document: str) -> tuple[MemoryProjection, str]:
    if type(document) is not str:
        raise ValueError("projection document must be text")
    start = document.find(_BEGIN)
    end = document.find(_END, start + len(_BEGIN))
    if start < 0 or end < 0:
        raise ValueError("projection envelope is absent")
    raw = document[start + len(_BEGIN) : end]
    try:
        payload = parse_json(raw)
    except ValueError as exc:
        raise ValueError("projection JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("version") != "angler.situated-memory.v1":
        raise ValueError("projection version is unsupported")
    expected_keys = {
        "version",
        "artifact_ref",
        "text",
        "source_ref",
        "visibility",
        "context",
        "projection_id",
    }
    if set(payload) != expected_keys:
        raise ValueError("projection fields are invalid")
    declared = payload.pop("projection_id", None)
    context_raw = payload.get("context")
    if not isinstance(context_raw, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or not all(isinstance(value, str) for value in item)
        for item in context_raw
    ):
        raise ValueError("projection context is invalid")
    try:
        projection = MemoryProjection(
            artifact_ref=payload["artifact_ref"],
            text=payload["text"],
            source_ref=payload["source_ref"],
            visibility=payload["visibility"],
            context=tuple((item[0], item[1]) for item in context_raw),
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("projection fields are invalid") from exc
    expected = projection_id(projection)
    if not isinstance(declared, str) or not hmac.compare_digest(declared, expected):
        raise ValueError("projection identity does not match its contents")
    return projection, expected


def _world_valid(
    *,
    world_time: int | None,
    valid_from: int | None,
    valid_until: int | None,
) -> bool | None:
    if world_time is None:
        return None
    if valid_from is not None and world_time < valid_from:
        return False
    if valid_until is not None and world_time > valid_until:
        return False
    return True


class SituatedMemory:
    """Join an untrusted retrieval projection with trusted temporal anchors.

    Backend order is preserved.  No recency formula, answer synthesis, or
    procedure selection is hidden here; Angler's learned core receives the
    validated temporal features and must learn whether they matter.
    """

    def __init__(
        self,
        backend: ProjectionBackend,
        *,
        origin: MovingOriginIndex | None = None,
    ) -> None:
        if not isinstance(backend, ProjectionBackend):
            raise TypeError("backend must implement ProjectionBackend")
        self.backend = backend
        self.origin = origin or MovingOriginIndex()

    async def remember(
        self,
        projection: MemoryProjection,
        *,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> int:
        document = encode_projection(projection)
        identity = projection_id(projection)
        anchor = self.origin.append(
            projection.artifact_ref,
            identity,
            world_valid_from=world_valid_from,
            world_valid_until=world_valid_until,
        )
        # The canonical temporal event deliberately survives a projection
        # failure: Cognee is disposable and can be rebuilt from evidence.
        await self.backend.remember(document)
        return anchor.ordinal

    async def reproject(self, projection: MemoryProjection) -> str | None:
        """Retry or rebuild Cognee from an already registered evidence event."""

        identity = projection_id(projection)
        anchor = self.origin.anchor(projection.artifact_ref)
        if not hmac.compare_digest(anchor.projection_id, identity):
            raise ValueError("projection does not match the registered evidence event")
        return await self.backend.remember(encode_projection(projection))

    async def recall(
        self,
        query: str,
        *,
        limit: int = 15,
        allowed_visibility: Iterable[str] = ("LEARNER_VISIBLE",),
        world_time: int | None = None,
        frozen_origin: bool = False,
    ) -> RecallBatch:
        allowed = frozenset(allowed_visibility)
        hits = await self.backend.search(query, limit=limit)
        items: list[SituatedRecall] = []
        rejected: list[str] = []
        for hit in hits:
            try:
                projection, identity = decode_projection(hit.document)
            except ValueError:
                rejected.append("PROJECTION_INVALID")
                continue
            try:
                anchor = self.origin.anchor(projection.artifact_ref)
            except KeyError:
                rejected.append("PROJECTION_UNKNOWN")
                continue
            if not hmac.compare_digest(anchor.projection_id, identity):
                rejected.append("PROJECTION_STALE_OR_TAMPERED")
                continue
            if projection.visibility not in allowed:
                rejected.append("PROJECTION_VISIBILITY_DENIED")
                continue
            position = (
                self.origin.frozen_position(projection.artifact_ref)
                if frozen_origin
                else self.origin.position(projection.artifact_ref)
            )
            items.append(
                SituatedRecall(
                    artifact_ref=projection.artifact_ref,
                    text=projection.text,
                    source_ref=projection.source_ref,
                    context=projection.context,
                    visibility=projection.visibility,
                    acquired_ordinal=position.acquired_ordinal,
                    age=position.age,
                    landmark_relations=position.landmark_relations,
                    world_valid_from=position.world_valid_from,
                    world_valid_until=position.world_valid_until,
                    world_valid_at_query=_world_valid(
                        world_time=world_time,
                        valid_from=position.world_valid_from,
                        valid_until=position.world_valid_until,
                    ),
                    backend_score=hit.score,
                    backend_ref=hit.backend_ref,
                )
            )
        return RecallBatch(items=tuple(items), rejected=tuple(rejected))

    async def forget_projection(self) -> None:
        """Delete only Cognee's rebuildable dataset; retain evidence/origin."""

        await self.backend.forget_dataset()


__all__ = [
    "SituatedMemory",
    "decode_projection",
    "encode_projection",
    "projection_id",
]
