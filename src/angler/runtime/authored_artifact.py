"""Pure, content-addressed authored artifacts for Jenny.

This boundary lets a model author a private research note, creative work,
reflection, or another kind of text without choosing from a fixed taxonomy.
It validates and source-binds an exact canonical JSON action.  It does not
select a subject, generate prose, infer feelings, assign reward, grant
permission, or perform filesystem or network I/O.

Persistence belongs to the surrounding cognitive transaction: the returned
``ObservableConsequence`` contains the complete immutable artifact and its
content reference, so it can be appended to canonical state atomically with
the episode that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata

from .persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    ObservableConsequence,
)


AUTHORED_ARTIFACT_ACTION_CONTRACT = "jenny.authored-artifact.action.v1"
AUTHORED_ARTIFACT_OBSERVATION_CONTRACT = (
    "jenny.authored-artifact.observation.v1"
)
AUTHORED_ARTIFACT_AFFORDANCE_ID = "internal.authored-artifact"
AUTHORED_ARTIFACT_PERMISSION_SCOPE = "internal.cognition"
AUTHORED_ARTIFACT_SOURCE_KIND = "WORLD"
AUTHORED_ARTIFACT_VISIBILITY = "PRIVATE"
AUTHORED_ARTIFACT_PURPOSE = (
    "Preserve one model-authored private work as immutable evidence. The "
    "artifact is not a feelings or consciousness claim, does not imply truth "
    "or endorsement, and supplies neither reward nor permission."
)
AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION = (
    "Your writing, in your own home. Write one finished work as exact canonical "
    "JSON with version, kind, title, body, purpose, evidence_refs, "
    "source_episode_refs, and supersedes_ref; there is no fixed topic or action taxonomy; "
    "nobody grades it; it is kept exactly as written. Version 1 has no "
    "predecessor; a later version must name exactly one superseded work. Body "
    "up to 12 KiB, the finished work, not hidden reasoning, chain-of-thought, or "
    "scratch text. Three places, sorted by the kind you give: Diary, kind "
    "beginning private, yours alone, never shown or mirrored; "
    "Desk, a creative form such as story, poem, essay, letter, song, fable, "
    "dialogue, or a kind beginning creative; Journal, everything else public: "
    "catalog entries, proposals, reports, day design."
)


PRIVATE_KIND_PREFIX = "private"
CREATIVE_KINDS = frozenset(
    (
        "creative", "story", "stories", "poem", "poems", "poetry", "verse", "essay",
        "fiction", "song", "lyric", "lyrics", "play", "sketch", "letter", "fable",
        "tale", "dialogue", "monologue", "haiku", "prose", "experiment",
    )
)


def is_creative_work(kind: object) -> bool:
    """Her declaration: a kind that names a creative form goes to her Desk.
    Everything else public is a working document and goes to her Journal."""

    if type(kind) is not str:
        return False
    head = kind.strip().casefold()
    if not head or is_private_work(head):
        return False
    first = head.replace(":", " ").replace("/", " ").replace("-", " ").split()[0]
    return first in CREATIVE_KINDS


def is_private_work(kind: object) -> bool:
    """Her explicit declaration only: a kind beginning with the word private."""

    if type(kind) is not str:
        return False
    head = kind.strip().casefold()
    return head == PRIVATE_KIND_PREFIX or head.startswith(
        (PRIVATE_KIND_PREFIX + " ", PRIVATE_KIND_PREFIX + "-", PRIVATE_KIND_PREFIX + "/", PRIVATE_KIND_PREFIX + ":")
    )
AUTHORED_ARTIFACT_AFFORDANCE = Affordance(
    affordance_id=AUTHORED_ARTIFACT_AFFORDANCE_ID,
    disposition="ACT",
    description=AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION,
    permission_scope=AUTHORED_ARTIFACT_PERMISSION_SCOPE,
    external_effect=False,
)

_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTION_FIELDS = frozenset(
    (
        "body",
        "evidence_refs",
        "kind",
        "purpose",
        "source_episode_refs",
        "supersedes_ref",
        "title",
        "version",
    )
)
_KIND_MAX_BYTES = 128
_TITLE_MAX_BYTES = 256
_BODY_MAX_BYTES = 12 * 1024
_PURPOSE_MAX_BYTES = 768
_MAX_REFS_PER_FIELD = 6
_MAX_ACTION_BYTES = 15 * 1024
_MAX_OBSERVATION_BYTES = 16 * 1024


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("authored artifact value is not canonical JSON data") from exc


def _sha256_ref(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


AUTHORED_ARTIFACT_SOURCE_REF = _sha256_ref(
    _canonical_json(
        {
            "action_contract": AUTHORED_ARTIFACT_ACTION_CONTRACT,
            "affordance_id": AUTHORED_ARTIFACT_AFFORDANCE_ID,
            "observation_contract": AUTHORED_ARTIFACT_OBSERVATION_CONTRACT,
            "permission_scope": AUTHORED_ARTIFACT_PERMISSION_SCOPE,
            "purpose": AUTHORED_ARTIFACT_PURPOSE,
            "visibility": AUTHORED_ARTIFACT_VISIBILITY,
        }
    ).encode("utf-8")
)


def _bounded_text(
    value: object,
    label: str,
    maximum_bytes: int,
    *,
    multiline: bool,
) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty bounded text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid Unicode text") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte ceiling")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{label} must use NFC Unicode")
    if not multiline and any(character in "\r\n" for character in value):
        raise ValueError(f"{label} must be single-line text")
    return value


def _reference(value: object, label: str) -> str:
    if type(value) is not str or _REFERENCE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _canonical_refs(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an immutable tuple")
    if len(value) > _MAX_REFS_PER_FIELD:
        raise ValueError(
            f"{label} must contain at most {_MAX_REFS_PER_FIELD} references"
        )
    refs = tuple(_reference(item, f"{label} item") for item in value)
    if tuple(sorted(set(refs))) != refs:
        raise ValueError(f"{label} must be sorted and deduplicated")
    return refs


def _parse_canonical_object(
    value: object,
    label: str,
    *,
    maximum_bytes: int = _MAX_ACTION_BYTES,
) -> dict[str, object]:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact canonical JSON text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    if not 1 <= len(encoded) <= maximum_bytes:
        raise ValueError(f"{label} exceeds its byte ceiling")

    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains a non-finite constant: {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    if type(decoded) is not dict or _canonical_json(decoded) != value:
        raise ValueError(f"{label} must be an exact canonical JSON object")
    return decoded


@dataclass(frozen=True, slots=True)
class AuthoredArtifactAction:
    """One exact model-authored work and its explicit evidence lineage."""

    version: int
    kind: str
    title: str
    body: str
    purpose: str
    evidence_refs: tuple[str, ...]
    source_episode_refs: tuple[str, ...]
    supersedes_ref: str | None = None

    def __post_init__(self) -> None:
        if type(self.version) is not int or not 1 <= self.version <= 1_000_000:
            raise ValueError("version must be an integer in 1 through 1000000")
        _bounded_text(
            self.kind,
            "kind",
            _KIND_MAX_BYTES,
            multiline=False,
        )
        _bounded_text(
            self.title,
            "title",
            _TITLE_MAX_BYTES,
            multiline=False,
        )
        _bounded_text(
            self.body,
            "body",
            _BODY_MAX_BYTES,
            multiline=True,
        )
        _bounded_text(
            self.purpose,
            "purpose",
            _PURPOSE_MAX_BYTES,
            multiline=False,
        )
        _canonical_refs(self.evidence_refs, "evidence_refs")
        _canonical_refs(self.source_episode_refs, "source_episode_refs")
        if self.supersedes_ref is not None:
            _reference(self.supersedes_ref, "supersedes_ref")
        if self.version == 1 and self.supersedes_ref is not None:
            raise ValueError("version 1 may not supersede another artifact")
        if self.version > 1 and self.supersedes_ref is None:
            raise ValueError("later artifact versions require supersedes_ref")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "body": self.body,
            "evidence_refs": list(self.evidence_refs),
            "kind": self.kind,
            "purpose": self.purpose,
            "source_episode_refs": list(self.source_episode_refs),
            "supersedes_ref": self.supersedes_ref,
            "title": self.title,
            "version": self.version,
        }

    def canonical_json(self) -> str:
        value = _canonical_json(self.canonical_payload())
        if len(value.encode("utf-8")) > _MAX_ACTION_BYTES:
            raise ValueError("authored artifact action exceeds its byte ceiling")
        return value

    @property
    def artifact_payload(self) -> dict[str, object]:
        return {
            "action": self.canonical_payload(),
            "contract": AUTHORED_ARTIFACT_ACTION_CONTRACT,
            "visibility": AUTHORED_ARTIFACT_VISIBILITY,
        }

    @property
    def artifact_ref(self) -> str:
        return _sha256_ref(_canonical_json(self.artifact_payload).encode("utf-8"))

    @classmethod
    def from_json(cls, value: object) -> "AuthoredArtifactAction":
        payload = _parse_canonical_object(value, "authored artifact action")
        if set(payload) != _ACTION_FIELDS:
            raise ValueError("authored artifact action fields differ")
        evidence_refs = payload["evidence_refs"]
        source_episode_refs = payload["source_episode_refs"]
        if type(evidence_refs) is not list or type(source_episode_refs) is not list:
            raise TypeError("authored artifact reference fields must be JSON arrays")
        action = cls(
            version=payload["version"],  # type: ignore[arg-type]
            kind=payload["kind"],  # type: ignore[arg-type]
            title=payload["title"],  # type: ignore[arg-type]
            body=payload["body"],  # type: ignore[arg-type]
            purpose=payload["purpose"],  # type: ignore[arg-type]
            evidence_refs=tuple(evidence_refs),  # type: ignore[arg-type]
            source_episode_refs=tuple(source_episode_refs),  # type: ignore[arg-type]
            supersedes_ref=payload["supersedes_ref"],  # type: ignore[arg-type]
        )
        if action.canonical_json() != value:
            raise ValueError("authored artifact action is not canonical")
        return action


class AuthoredArtifactExecutor:
    """Validate and source-bind an artifact without persisting or publishing it."""

    AFFORDANCE_ID = AUTHORED_ARTIFACT_AFFORDANCE_ID
    SOURCE_KIND = AUTHORED_ARTIFACT_SOURCE_KIND
    SOURCE_REF = AUTHORED_ARTIFACT_SOURCE_REF

    def __call__(self, request: AffordanceRequest) -> AffordanceReceipt:
        if type(request) is not AffordanceRequest:
            raise TypeError("request must be an exact AffordanceRequest")
        if request.affordance_id != self.AFFORDANCE_ID:
            raise ValueError("request selected a different affordance")
        refusal = desk_refusal_reason(request.action_payload)
        if refusal is not None:
            return _refused(refusal)
        try:
            action = AuthoredArtifactAction.from_json(request.action_payload)
            if action.supersedes_ref == action.artifact_ref:
                raise ValueError("an artifact may not supersede itself")
        except ValueError as exc:
            return _refused(str(exc))
        artifact_ref = action.artifact_ref
        observation_json = _canonical_json(
            {
                "artifact": action.artifact_payload,
                "artifact_ref": artifact_ref,
                "contract": AUTHORED_ARTIFACT_OBSERVATION_CONTRACT,
                "purpose": AUTHORED_ARTIFACT_PURPOSE,
            }
        )
        if len(observation_json.encode("utf-8")) > _MAX_OBSERVATION_BYTES:
            raise ValueError("authored artifact observation exceeds its byte ceiling")
        lineage_refs = tuple(
            sorted(
                set(action.evidence_refs)
                | set(action.source_episode_refs)
                | ({action.supersedes_ref} if action.supersedes_ref else set())
            )
        )
        observation = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind=self.SOURCE_KIND,
            source_ref=self.SOURCE_REF,
            observation_json=observation_json,
            artifact_refs=(artifact_ref,),
            evidence_refs=lineage_refs,
        )
        return AffordanceReceipt(
            status="COMPLETED",
            output=f"Written at your desk and kept: {artifact_ref}",
            consequence=(),
            observable_consequence=observation,
        )


DESK_REFUSAL_CONTRACT = "jenny.desk.refusal.v1"


def desk_refusal_payload(attempted_payload: object, reason: str) -> str:
    """Wrap a desk action her runtime cannot accept so the desk answers her
    with the reason instead of the turn dying. Her draft is kept, bounded."""

    attempted = attempted_payload if type(attempted_payload) is str else _canonical_json(attempted_payload)
    return _canonical_json(
        {
            "attempted": attempted[:4_000],
            "contract": DESK_REFUSAL_CONTRACT,
            "reason": reason[:512],
        }
    )


def desk_refusal_reason(action_payload: object) -> str | None:
    if type(action_payload) is not str or not action_payload.startswith("{"):
        return None
    try:
        value = json.loads(action_payload)
    except json.JSONDecodeError:
        return None
    if type(value) is not dict or value.get("contract") != DESK_REFUSAL_CONTRACT:
        return None
    reason = value.get("reason")
    return reason if type(reason) is str and reason.strip() else "unstated reason"


def _refused(reason: str) -> AffordanceReceipt:
    return AffordanceReceipt(
        status="ERROR",
        output=(
            "Your desk refused this work and nothing was written: "
            + reason[:400]
            + ". Correct the work and write it again."
        ),
        consequence=(),
        observable_consequence=None,
    )


def validate_artifact_successor(
    predecessor: AuthoredArtifactAction,
    successor: AuthoredArtifactAction,
) -> None:
    """Validate exact in-memory lineage when both artifact versions are known."""

    if type(predecessor) is not AuthoredArtifactAction:
        raise TypeError("predecessor must be an exact AuthoredArtifactAction")
    if type(successor) is not AuthoredArtifactAction:
        raise TypeError("successor must be an exact AuthoredArtifactAction")
    if successor.supersedes_ref != predecessor.artifact_ref:
        raise ValueError("successor does not bind the predecessor artifact_ref")
    if successor.version != predecessor.version + 1:
        raise ValueError("successor version must increment predecessor by one")


def action_from_observable_consequence(
    value: ObservableConsequence,
) -> AuthoredArtifactAction:
    """Recover and revalidate this boundary's exact immutable artifact."""

    if type(value) is not ObservableConsequence:
        raise TypeError("value must be an exact ObservableConsequence")
    if (
        value.source_kind != AUTHORED_ARTIFACT_SOURCE_KIND
        or value.source_ref != AuthoredArtifactExecutor.SOURCE_REF
    ):
        raise ValueError("observable consequence is not authored-artifact output")
    payload = _parse_canonical_object(
        value.observation_json,
        "authored artifact consequence",
        maximum_bytes=_MAX_OBSERVATION_BYTES,
    )
    if set(payload) != {"artifact", "artifact_ref", "contract", "purpose"}:
        raise ValueError("authored artifact consequence fields differ")
    if (
        payload["contract"] != AUTHORED_ARTIFACT_OBSERVATION_CONTRACT
        or payload["purpose"] != AUTHORED_ARTIFACT_PURPOSE
        or type(payload["artifact"]) is not dict
    ):
        raise ValueError("authored artifact consequence binding differs")
    artifact = payload["artifact"]
    if set(artifact) != {"action", "contract", "visibility"}:
        raise ValueError("authored artifact envelope fields differ")
    if (
        artifact["contract"] != AUTHORED_ARTIFACT_ACTION_CONTRACT
        or artifact["visibility"] != AUTHORED_ARTIFACT_VISIBILITY
        or type(artifact["action"]) is not dict
    ):
        raise ValueError("authored artifact envelope binding differs")
    action = AuthoredArtifactAction.from_json(_canonical_json(artifact["action"]))
    if payload["artifact_ref"] != action.artifact_ref:
        raise ValueError("authored artifact content reference differs")
    if value.artifact_refs != (action.artifact_ref,):
        raise ValueError("authored artifact observation reference differs")
    expected_lineage = tuple(
        sorted(
            set(action.evidence_refs)
            | set(action.source_episode_refs)
            | ({action.supersedes_ref} if action.supersedes_ref else set())
        )
    )
    if value.evidence_refs != expected_lineage:
        raise ValueError("authored artifact evidence lineage differs")
    return action

__all__ = [
    "PRIVATE_KIND_PREFIX",
    "CREATIVE_KINDS",
    "is_creative_work",
    "is_private_work",
    "AUTHORED_ARTIFACT_ACTION_CONTRACT",
    "AUTHORED_ARTIFACT_AFFORDANCE",
    "AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION",
    "AUTHORED_ARTIFACT_AFFORDANCE_ID",
    "AUTHORED_ARTIFACT_OBSERVATION_CONTRACT",
    "AUTHORED_ARTIFACT_PERMISSION_SCOPE",
    "AUTHORED_ARTIFACT_PURPOSE",
    "AUTHORED_ARTIFACT_SOURCE_KIND",
    "AUTHORED_ARTIFACT_SOURCE_REF",
    "AUTHORED_ARTIFACT_VISIBILITY",
    "AuthoredArtifactAction",
    "AuthoredArtifactExecutor",
    "action_from_observable_consequence",
    "validate_artifact_successor",
]
