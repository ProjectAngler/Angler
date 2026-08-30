"""Contracts for temporally situated, retrieval-assisted evidence.

These values carry evidence and temporal coordinates to Angler.  They do not
choose a procedure, prescribe an answer, or mutate procedural competence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


_VISIBILITY_CLASSES = frozenset(
    {
        "LEARNER_VISIBLE",
        "CONTROL_PLANE",
        "SEALED_EVALUATION",
        "HUMAN_AUTHORITY",
        "RESTRICTED_PERSONAL",
    }
)


def _bounded_text(value: str, label: str, *, maximum: int) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


@dataclass(frozen=True, slots=True)
class MemoryProjection:
    """A visibility-filtered projection of canonical Angler evidence.

    ``artifact_ref`` points back to the source of truth.  This projection is
    disposable and may be rebuilt; it is never itself procedural competence.
    """

    artifact_ref: str
    text: str
    source_ref: str
    visibility: str = "LEARNER_VISIBLE"
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _bounded_text(self.artifact_ref, "artifact_ref", maximum=256)
        # Keep the complete identity envelope inside one ordinary Cognee chunk;
        # larger evidence stays canonical and is projected as smaller records.
        _bounded_text(self.text, "text", maximum=4_096)
        _bounded_text(self.source_ref, "source_ref", maximum=512)
        if self.visibility not in _VISIBILITY_CLASSES:
            raise ValueError("visibility is not a registered Angler visibility class")
        if type(self.context) is not tuple:
            raise TypeError("context must be an immutable tuple")
        previous = None
        for key, value in self.context:
            _bounded_text(key, "context key", maximum=128)
            _bounded_text(value, "context value", maximum=1_024)
            if previous is not None and key <= previous:
                raise ValueError("context keys must be unique and canonically sorted")
            previous = key

    @classmethod
    def from_mapping(
        cls,
        *,
        artifact_ref: str,
        text: str,
        source_ref: str,
        visibility: str = "LEARNER_VISIBLE",
        context: dict[str, str] | None = None,
    ) -> "MemoryProjection":
        pairs = tuple(sorted((context or {}).items()))
        return cls(
            artifact_ref=artifact_ref,
            text=text,
            source_ref=source_ref,
            visibility=visibility,
            context=pairs,
        )


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One raw candidate returned by a replaceable retrieval backend."""

    document: str
    score: float | None = None
    backend_ref: str | None = None


@runtime_checkable
class ProjectionBackend(Protocol):
    """Minimal boundary implemented by Cognee or an in-memory test double."""

    async def remember(self, document: str) -> str | None:
        """Index one disposable projection and return an optional backend id."""

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        """Return candidate projections in backend order."""

    async def forget_dataset(self) -> None:
        """Delete this adapter's rebuildable dataset, not canonical evidence."""


@dataclass(frozen=True, slots=True)
class SituatedRecall:
    """A retrieval candidate augmented with Moving Origin coordinates."""

    artifact_ref: str
    text: str
    source_ref: str
    context: tuple[tuple[str, str], ...]
    visibility: str
    acquired_ordinal: int
    age: int
    landmark_relations: tuple[tuple[str, str], ...]
    world_valid_from: int | None
    world_valid_until: int | None
    world_valid_at_query: bool | None
    backend_score: float | None
    backend_ref: str | None


@dataclass(frozen=True, slots=True)
class RecallBatch:
    """Validated recall output plus non-payload-bearing rejection reasons."""

    items: tuple[SituatedRecall, ...]
    rejected: tuple[str, ...] = ()


__all__ = [
    "MemoryHit",
    "MemoryProjection",
    "ProjectionBackend",
    "RecallBatch",
    "SituatedRecall",
]
