"""Immutable contracts for Angler's canonical cognitive transaction spine.

These values describe evidence, predictions, and completed turns.  They do
not select a procedure, establish truth, or mutate learned competence.
Identity is the SHA-256 of strict canonical JSON under the existing Angler
canonicalization profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
import unicodedata
from typing import Any, ClassVar, Mapping, Self

from angler.episodes.canonical import canonical_bytes, parse_json


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
_VISIBILITIES = frozenset(
    {
        "LEARNER_VISIBLE",
        "CONTROL_PLANE",
        "SEALED_EVALUATION",
        "HUMAN_AUTHORITY",
        "RESTRICTED_PERSONAL",
    }
)


class CognitiveMemoryKind(str, Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    CAUSAL = "CAUSAL"
    SELF = "SELF"
    COUNTERFACTUAL = "COUNTERFACTUAL"


class EpistemicStatus(str, Enum):
    OBSERVED = "OBSERVED"
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    RETRACTED = "RETRACTED"


class RelationType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    VERIFIES = "VERIFIES"
    SUPERSEDES = "SUPERSEDES"
    TOMBSTONES = "TOMBSTONES"
    CAUSES = "CAUSES"
    PREDICTS = "PREDICTS"
    ACTUAL_OF = "ACTUAL_OF"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    ABOUT = "ABOUT"
    IMPLEMENTS = "IMPLEMENTS"


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not value and not allow_empty):
        qualifier = "text" if allow_empty else "non-empty text"
        raise ValueError(f"{label} must be {qualifier} of at most {maximum} characters")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must be Unicode NFC-normalized")
    return value


def _identity(value: Any, label: str) -> str:
    _text(value, label, 256)
    if not _IDENTITY.fullmatch(value):
        raise ValueError(f"{label} has an invalid identity form")
    return value


def _digest(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _enum(value: Any, enum_type: type[Enum], label: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not registered") from exc


def _tuple(value: Any, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an immutable tuple")
    return value


def _digest_tuple(
    value: Any,
    label: str,
    *,
    sorted_required: bool,
    nonempty: bool = False,
) -> tuple[str, ...]:
    values = _tuple(value, label)
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    for item in values:
        _digest(item, f"{label} item")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    if sorted_required and values != tuple(sorted(values)):
        raise ValueError(f"{label} must be canonically sorted")
    return values


def _content_ref(text: str) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes({"content": text})).hexdigest()


class _CanonicalContract:
    """Shared strict canonical JSON behavior for immutable contracts."""

    CONTRACT: ClassVar[str]
    REF_ATTRIBUTE: ClassVar[str]

    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @property
    def content_ref(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        raise NotImplementedError

    @classmethod
    def from_json(cls, raw: str | bytes) -> Self:
        parsed = parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("canonical contract payload must be an object")
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        if canonical_bytes(parsed) != encoded:
            raise ValueError("contract JSON is not in canonical byte form")
        return cls.from_payload(parsed)


@dataclass(frozen=True, slots=True)
class CognitiveRelation:
    relation_type: RelationType
    target_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_type",
            _enum(self.relation_type, RelationType, "relation_type"),
        )
        _digest(self.target_ref, "relation target_ref")

    def to_payload(self) -> dict[str, str]:
        return {"relation_type": self.relation_type.value, "target_ref": self.target_ref}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CognitiveRelation":
        _exact_keys(payload, {"relation_type", "target_ref"}, "relation")
        return cls(
            relation_type=_enum(payload["relation_type"], RelationType, "relation_type"),
            target_ref=payload["target_ref"],
        )


@dataclass(frozen=True, slots=True)
class CognitiveMemoryRecord(_CanonicalContract):
    """One canonical typed memory record; Cognee may only project this value."""

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-MEMORY-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "record_ref"

    kind: CognitiveMemoryKind
    epistemic_status: EpistemicStatus
    content: str
    provenance_refs: tuple[str, ...]
    visibility: str
    producer_id: str
    producer_checkpoint_ref: str
    competence_ref: str
    acquired_ordinal: int
    world_valid_from: int | None = None
    world_valid_until: int | None = None
    relations: tuple[CognitiveRelation, ...] = ()
    supersedes_refs: tuple[str, ...] = ()
    tombstone_refs: tuple[str, ...] = ()
    confidence_ppm: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CognitiveMemoryKind, "kind"))
        object.__setattr__(
            self,
            "epistemic_status",
            _enum(self.epistemic_status, EpistemicStatus, "epistemic_status"),
        )
        _text(self.content, "content", 16_384)
        _digest_tuple(
            self.provenance_refs,
            "provenance_refs",
            sorted_required=True,
            nonempty=True,
        )
        if self.visibility not in _VISIBILITIES:
            raise ValueError("visibility is not a registered Angler visibility class")
        _identity(self.producer_id, "producer_id")
        _digest(self.producer_checkpoint_ref, "producer_checkpoint_ref")
        _digest(self.competence_ref, "competence_ref")
        _integer(self.acquired_ordinal, "acquired_ordinal")
        if self.world_valid_from is not None:
            _integer(self.world_valid_from, "world_valid_from")
        if self.world_valid_until is not None:
            _integer(self.world_valid_until, "world_valid_until")
        if (
            self.world_valid_from is not None
            and self.world_valid_until is not None
            and self.world_valid_until < self.world_valid_from
        ):
            raise ValueError("world_valid_until cannot precede world_valid_from")
        relations = _tuple(self.relations, "relations")
        if any(not isinstance(item, CognitiveRelation) for item in relations):
            raise TypeError("relations must contain CognitiveRelation values")
        relation_keys = tuple(
            (item.relation_type.value, item.target_ref) for item in relations
        )
        if relation_keys != tuple(sorted(relation_keys)) or len(set(relation_keys)) != len(
            relation_keys
        ):
            raise ValueError("relations must be unique and canonically sorted")
        if any(
            item.relation_type in {RelationType.SUPERSEDES, RelationType.TOMBSTONES}
            for item in relations
        ):
            raise ValueError(
                "SUPERSEDES and TOMBSTONES must use their dedicated canonical ref fields"
            )
        _digest_tuple(self.supersedes_refs, "supersedes_refs", sorted_required=True)
        _digest_tuple(self.tombstone_refs, "tombstone_refs", sorted_required=True)
        if set(self.supersedes_refs) & set(self.tombstone_refs):
            raise ValueError("one record cannot both supersede and tombstone the same ref")
        if self.epistemic_status is EpistemicStatus.RETRACTED:
            if not self.tombstone_refs:
                raise ValueError("a RETRACTED record must identify tombstoned records")
        elif self.tombstone_refs:
            raise ValueError("only a RETRACTED record may carry tombstone_refs")
        if self.confidence_ppm is not None:
            _integer(self.confidence_ppm, "confidence_ppm")
            if self.confidence_ppm > 1_000_000:
                raise ValueError("confidence_ppm must be <= 1000000")

    @property
    def record_ref(self) -> str:
        return self.content_ref

    def to_payload(self) -> dict[str, Any]:
        return {
            "acquired_ordinal": self.acquired_ordinal,
            "competence_ref": self.competence_ref,
            "confidence_ppm": self.confidence_ppm,
            "content": self.content,
            "contract": self.CONTRACT,
            "epistemic_status": self.epistemic_status.value,
            "kind": self.kind.value,
            "producer_checkpoint_ref": self.producer_checkpoint_ref,
            "producer_id": self.producer_id,
            "provenance_refs": list(self.provenance_refs),
            "relations": [item.to_payload() for item in self.relations],
            "supersedes_refs": list(self.supersedes_refs),
            "tombstone_refs": list(self.tombstone_refs),
            "visibility": self.visibility,
            "world_valid_from": self.world_valid_from,
            "world_valid_until": self.world_valid_until,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CognitiveMemoryRecord":
        keys = {
            "acquired_ordinal", "competence_ref", "confidence_ppm", "content",
            "contract", "epistemic_status", "kind", "producer_checkpoint_ref",
            "producer_id", "provenance_refs", "relations", "supersedes_refs",
            "tombstone_refs", "visibility", "world_valid_from", "world_valid_until",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            kind=payload["kind"],
            epistemic_status=payload["epistemic_status"],
            content=payload["content"],
            provenance_refs=_payload_tuple(payload["provenance_refs"], "provenance_refs"),
            visibility=payload["visibility"],
            producer_id=payload["producer_id"],
            producer_checkpoint_ref=payload["producer_checkpoint_ref"],
            competence_ref=payload["competence_ref"],
            acquired_ordinal=payload["acquired_ordinal"],
            world_valid_from=payload["world_valid_from"],
            world_valid_until=payload["world_valid_until"],
            relations=tuple(
                CognitiveRelation.from_payload(item)
                for item in _payload_list(payload["relations"], "relations")
            ),
            supersedes_refs=_payload_tuple(payload["supersedes_refs"], "supersedes_refs"),
            tombstone_refs=_payload_tuple(payload["tombstone_refs"], "tombstone_refs"),
            confidence_ppm=payload["confidence_ppm"],
        )


@dataclass(frozen=True, slots=True)
class ProspectiveCommitment(_CanonicalContract):
    """An immutable prediction committed before the selected action executes."""

    CONTRACT: ClassVar[str] = "ANG-CTR-PROSPECTIVE-COMMITMENT-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "commitment_ref"

    parent_event_ref: str | None
    task_id: str
    candidate_index: int
    candidate_trace: str
    predicted_score: float
    uncertainty: float | None
    horizon: int
    competence_state_digest: str

    def __post_init__(self) -> None:
        _digest(self.parent_event_ref, "parent_event_ref", optional=True)
        _identity(self.task_id, "task_id")
        _integer(self.candidate_index, "candidate_index")
        _text(self.candidate_trace, "candidate_trace", 8_192)
        if type(self.predicted_score) is not float or not math.isfinite(self.predicted_score):
            raise ValueError("predicted_score must be a finite float")
        if self.uncertainty is not None and (
            type(self.uncertainty) is not float
            or not math.isfinite(self.uncertainty)
            or self.uncertainty < 0.0
        ):
            raise ValueError("uncertainty must be a finite non-negative float or None")
        _integer(self.horizon, "horizon", minimum=1)
        _digest(self.competence_state_digest, "competence_state_digest")

    @property
    def commitment_ref(self) -> str:
        return self.content_ref

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "candidate_trace": self.candidate_trace,
            "competence_state_digest": self.competence_state_digest,
            "contract": self.CONTRACT,
            "horizon": self.horizon,
            "parent_event_ref": self.parent_event_ref,
            "predicted_score_hex": self.predicted_score.hex(),
            "task_id": self.task_id,
            "uncertainty_hex": None if self.uncertainty is None else self.uncertainty.hex(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProspectiveCommitment":
        keys = {
            "candidate_index", "candidate_trace", "competence_state_digest", "contract", "horizon",
            "parent_event_ref", "predicted_score_hex", "task_id", "uncertainty_hex",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        score = _hex_float(payload["predicted_score_hex"], "predicted_score_hex")
        uncertainty = payload["uncertainty_hex"]
        if uncertainty is not None:
            uncertainty = _hex_float(uncertainty, "uncertainty_hex")
        return cls(
            parent_event_ref=payload["parent_event_ref"],
            task_id=payload["task_id"],
            candidate_index=payload["candidate_index"],
            candidate_trace=payload["candidate_trace"],
            predicted_score=score,
            uncertainty=uncertainty,
            horizon=payload["horizon"],
            competence_state_digest=payload["competence_state_digest"],
        )


@dataclass(frozen=True, slots=True)
class CognitiveEpisode(_CanonicalContract):
    """One complete pre-decision, action, feedback, and state-transition turn."""

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-EPISODE-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "episode_ref"

    task_id: str
    request: str
    recalled_refs: tuple[str, ...]
    proposals: tuple[str, ...]
    selected_index: int
    commitment: ProspectiveCommitment
    response: str
    observations: tuple[str, ...]
    outcome: str
    feedback_text: str
    feedback_source_ref: str
    parent_state_digest: str
    child_state_digest: str
    model_ref: str
    encoder_ref: str
    supporting_evidence_refs: tuple[str, ...]
    visibility: str = "LEARNER_VISIBLE"

    def __post_init__(self) -> None:
        _identity(self.task_id, "task_id")
        _text(self.request, "request", 16_384)
        _digest_tuple(self.recalled_refs, "recalled_refs", sorted_required=False)
        proposals = _tuple(self.proposals, "proposals")
        if not 2 <= len(proposals) <= 64:
            raise ValueError("proposals must contain 2 through 64 public proposals")
        for proposal in proposals:
            _text(proposal, "proposal", 8_192)
        if len(set(proposals)) != len(proposals):
            raise ValueError("public proposals must be distinct")
        _integer(self.selected_index, "selected_index")
        if self.selected_index >= len(proposals):
            raise ValueError("selected_index is outside the public proposals")
        if not isinstance(self.commitment, ProspectiveCommitment):
            raise TypeError("commitment must be a ProspectiveCommitment")
        if self.commitment.task_id != self.task_id:
            raise ValueError("commitment task_id does not match the episode")
        if self.commitment.candidate_index != self.selected_index:
            raise ValueError("commitment candidate_index does not match selected_index")
        if self.commitment.candidate_trace != proposals[self.selected_index]:
            raise ValueError("commitment does not bind the selected public proposal")
        _text(self.response, "response", 16_384, allow_empty=True)
        observations = _tuple(self.observations, "observations")
        for observation in observations:
            _text(observation, "observation", 8_192)
        if self.outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        _text(self.feedback_text, "feedback_text", 8_192)
        for label, value in (
            ("feedback_source_ref", self.feedback_source_ref),
            ("parent_state_digest", self.parent_state_digest),
            ("child_state_digest", self.child_state_digest),
            ("model_ref", self.model_ref),
            ("encoder_ref", self.encoder_ref),
        ):
            _digest(value, label)
        if self.commitment.competence_state_digest != self.parent_state_digest:
            raise ValueError("commitment competence state does not match parent_state_digest")
        _digest_tuple(
            self.supporting_evidence_refs,
            "supporting_evidence_refs",
            sorted_required=True,
        )
        if not set(self.supporting_evidence_refs).issubset(self.recalled_refs):
            raise ValueError("supporting_evidence_refs must be a subset of recalled_refs")
        if self.visibility not in _VISIBILITIES:
            raise ValueError("visibility is not a registered Angler visibility class")

    @staticmethod
    def proposal_ref(proposal: str) -> str:
        _text(proposal, "proposal", 8_192)
        return _content_ref(proposal)

    @property
    def episode_ref(self) -> str:
        return self.content_ref

    def to_payload(self) -> dict[str, Any]:
        return {
            "child_state_digest": self.child_state_digest,
            "commitment": self.commitment.to_payload(),
            "contract": self.CONTRACT,
            "encoder_ref": self.encoder_ref,
            "feedback_source_ref": self.feedback_source_ref,
            "feedback_text": self.feedback_text,
            "model_ref": self.model_ref,
            "observations": list(self.observations),
            "outcome": self.outcome,
            "parent_state_digest": self.parent_state_digest,
            "proposals": list(self.proposals),
            "recalled_refs": list(self.recalled_refs),
            "request": self.request,
            "response": self.response,
            "selected_index": self.selected_index,
            "supporting_evidence_refs": list(self.supporting_evidence_refs),
            "task_id": self.task_id,
            "visibility": self.visibility,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CognitiveEpisode":
        keys = {
            "child_state_digest", "commitment", "contract",
            "encoder_ref", "feedback_source_ref", "feedback_text", "model_ref",
            "observations", "outcome", "parent_state_digest",
            "proposals", "recalled_refs", "request", "response", "selected_index",
            "supporting_evidence_refs", "task_id", "visibility",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            task_id=payload["task_id"],
            request=payload["request"],
            recalled_refs=_payload_tuple(payload["recalled_refs"], "recalled_refs"),
            proposals=_payload_tuple(payload["proposals"], "proposals"),
            selected_index=payload["selected_index"],
            commitment=ProspectiveCommitment.from_payload(payload["commitment"]),
            response=payload["response"],
            observations=_payload_tuple(payload["observations"], "observations"),
            outcome=payload["outcome"],
            feedback_text=payload["feedback_text"],
            feedback_source_ref=payload["feedback_source_ref"],
            parent_state_digest=payload["parent_state_digest"],
            child_state_digest=payload["child_state_digest"],
            model_ref=payload["model_ref"],
            encoder_ref=payload["encoder_ref"],
            supporting_evidence_refs=_payload_tuple(
                payload["supporting_evidence_refs"], "supporting_evidence_refs"
            ),
            visibility=payload["visibility"],
        )


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{label} payload fields do not match the contract")


def _exact_contract(
    payload: Mapping[str, Any], expected: set[str], contract: str
) -> None:
    _exact_keys(payload, expected, contract)
    if payload.get("contract") != contract:
        raise ValueError("unsupported contract version")


def _payload_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _payload_tuple(value: Any, label: str) -> tuple[Any, ...]:
    return tuple(_payload_list(value, label))


def _hex_float(value: Any, label: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{label} must be a hexadecimal float string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a hexadecimal float string") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{label} is not a canonical finite float")
    return result


__all__ = [
    "CognitiveEpisode",
    "CognitiveMemoryKind",
    "CognitiveMemoryRecord",
    "CognitiveRelation",
    "EpistemicStatus",
    "ProspectiveCommitment",
    "RelationType",
]
