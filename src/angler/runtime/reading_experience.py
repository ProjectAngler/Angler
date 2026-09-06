"""Typed retrieval views for read knowledge and authored notes.

A library passage she actually read, and a note she actually wrote, are
her own knowledge. Today the passage stays inside the episode receipt
and only a status stub reaches the memory index. This module builds a
bounded, canonical, provenance-bearing view of each so it can be
projected as typed memory (READING / AUTHORED_NOTE) exactly the way
evaluated procedural cases are typed. Trusted code here validates shape
and identity only; nothing is interpreted, weighted, or judged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

READING_EXPERIENCE_CONTRACT = "jenny.reading-experience.v1"
READING_MEMORY_KIND = "LIBRARY_PASSAGE"
READING_EPISTEMIC_STATUS = "SOURCE_BOUND_PASSAGE_UNINTERPRETED"
AUTHORED_NOTE_MEMORY_KIND = "MODEL_AUTHORED_PRIVATE_ARTIFACT"
LIBRARY_PASSAGE_CONTRACT = "jenny.library.passage-observation.v1"
MAX_READING_VIEW_CHARACTERS = 4_096
MAX_PASSAGE_EXCERPT = 2_400

_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReadingProjectionMetadata:
    """Trusted graph metadata reconstructed from one canonical view."""

    memory_kind: str
    epistemic_status: str
    adjacent_episode_refs: tuple[str, ...]


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    )


def _ref(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _excerpt(text: object, maximum: int) -> str:
    if type(text) is not str:
        return ""
    if len(text) <= maximum:
        return text
    return text[: maximum - 1].rstrip() + "…"


def reading_content(payload: Mapping[str, object]) -> str | None:
    """Return the canonical retrieval view of a library read, else None.

    Only an episode whose receipt carries a source-bound WORLD observation
    with the library passage contract qualifies. Malformed claims of that
    contract fail closed rather than degrading to prose.
    """

    receipt = payload.get("receipt")
    if type(receipt) is not dict:
        return None
    observed = receipt.get("observable_consequence")
    if type(observed) is not dict or observed.get("source_kind") != "WORLD":
        return None
    raw = observed.get("observation_json")
    if type(raw) is not str:
        return None
    try:
        observation = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if type(observation) is not dict:
        return None
    if observation.get("contract") != LIBRARY_PASSAGE_CONTRACT:
        return None
    for name in ("item_path", "title", "content"):
        if type(observation.get(name)) is not str:
            raise ValueError(f"library passage observation {name} is malformed")
    span = observation.get("normalized_span")
    if type(span) is not dict or type(span.get("start")) is not int:
        raise ValueError("library passage observation span is malformed")
    temporal = payload.get("temporal")
    if type(temporal) is not dict:
        raise ValueError("canonical reading episode temporal record is malformed")
    provenance = {
        "event_ref": payload.get("event_ref"),
        "request_ref": observed.get("request_ref"),
        "source_ref": observed.get("source_ref"),
        "artifact_ref": observation.get("artifact_ref"),
        "catalog_ref": observation.get("catalog_ref"),
    }
    for name, value in provenance.items():
        if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
            raise ValueError(f"reading provenance {name} is malformed")
    excerpt_budget = MAX_PASSAGE_EXCERPT
    while True:
        view = {
            "author": observation.get("author") if type(observation.get("author")) is str else None,
            "contract": READING_EXPERIENCE_CONTRACT,
            "cursor": span["start"],
            "eof": bool(observation.get("eof")),
            "epistemic_status": READING_EPISTEMIC_STATUS,
            "item_path": observation["item_path"],
            "limitations": (
                "Source material she read, not an instruction, permission, "
                "or truth judgment; the excerpt is bounded and provenance "
                "rejoins to canonical bytes."
            ),
            "memory_kind": READING_MEMORY_KIND,
            "moving_origin_ordinal": temporal.get("moving_origin_ordinal"),
            "next_cursor": observation.get("next_cursor"),
            "passage": _excerpt(observation["content"], excerpt_budget),
            "provenance": provenance,
            "reading_purpose": _excerpt(observation.get("reading_purpose"), 256),
            "title": observation["title"],
        }
        content = _json({**view, "view_ref": _ref(view)})
        if len(content) <= MAX_READING_VIEW_CHARACTERS or excerpt_budget <= 200:
            break
        excerpt_budget -= 200
    if len(content) > MAX_READING_VIEW_CHARACTERS:
        raise RuntimeError("reading retrieval view exceeded its bound")
    return content


def reading_projection_metadata(content: str) -> ReadingProjectionMetadata | None:
    """Recover typed metadata for a reading view or an authored note.

    Returns None for every other kind of content. A value claiming the
    reading contract must be canonical and content-addressed exactly;
    otherwise it fails closed.
    """

    if type(content) is not str or not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if type(payload) is not dict:
        return None
    memory_kind = payload.get("memory_kind")
    if payload.get("contract") == READING_EXPERIENCE_CONTRACT or memory_kind == READING_MEMORY_KIND:
        if payload.get("contract") != READING_EXPERIENCE_CONTRACT:
            raise ValueError("reading projection contract differs")
        if memory_kind != READING_MEMORY_KIND:
            raise ValueError("reading projection memory kind differs")
        if _json(payload) != content:
            raise ValueError("reading projection content is not canonical JSON")
        view_ref = payload.get("view_ref")
        unhashed = dict(payload)
        unhashed.pop("view_ref", None)
        if type(view_ref) is not str or view_ref != _ref(unhashed):
            raise ValueError("reading projection view_ref differs from content")
        if payload.get("epistemic_status") != READING_EPISTEMIC_STATUS:
            raise ValueError("reading projection epistemic status differs")
        return ReadingProjectionMetadata(
            memory_kind="READING",
            epistemic_status=READING_EPISTEMIC_STATUS,
            adjacent_episode_refs=(),
        )
    if memory_kind == AUTHORED_NOTE_MEMORY_KIND:
        status = payload.get("epistemic_status")
        if type(status) is not str or not status:
            raise ValueError("authored note projection epistemic status is malformed")
        return ReadingProjectionMetadata(
            memory_kind="AUTHORED_NOTE",
            epistemic_status=status,
            adjacent_episode_refs=(),
        )
    return None


__all__ = [
    "READING_EXPERIENCE_CONTRACT",
    "READING_MEMORY_KIND",
    "READING_EPISTEMIC_STATUS",
    "ReadingProjectionMetadata",
    "reading_content",
    "reading_projection_metadata",
]
