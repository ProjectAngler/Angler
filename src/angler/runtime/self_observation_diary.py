"""Model-authored, non-authoritative self-observation diary boundary.

The model may propose its own label and a functional description of an
observable pattern.  This module validates and source-binds that proposal; it
does not score it, infer private phenomenology, grant permission, or write any
canonical or learned state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from .persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    ObservableConsequence,
)


SELF_OBSERVATION_DIARY_CONTRACT = (
    "ANG-CTR-JENNY-SELF-OBSERVATION-DIARY-001@0.1.0"
)
SELF_OBSERVATION_DIARY_PURPOSE = (
    "Longitudinal evidence comparison and revision only; estimated_strength "
    "is a model-authored estimate, not a measurement; this is not proof of "
    "feelings or consciousness and supplies neither reward nor permission."
)
SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS = (
    "MODEL_SELF_OBSERVATION_HYPOTHESIS_UNVERIFIED"
)
SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS = "UNESTABLISHED"
SELF_OBSERVATION_DIARY_STRENGTH_STATUS = "MODEL_ESTIMATE_UNCALIBRATED"
SELF_OBSERVATION_DIARY_LABEL_STATUS = "MODEL_CHOSEN_PROVISIONAL"
SELF_OBSERVATION_DIARY_APPRAISAL_ROLE = (
    "PERSISTENT_EVIDENCE_GROUNDED_FUNCTIONAL_APPRAISAL_HYPOTHESES"
)
SELF_OBSERVATION_DIARY_AFFORDANCE_ID = "internal.self-observation-diary"
SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION = (
    "Propose one longitudinal self-observation hypothesis as exact canonical "
    "JSON with candidate_label, functional_description, estimated_strength, "
    "uncertainty, observable_signals, alternative_explanations, and optional "
    "revision_target_ref, and optional resolution_reason. Invent a context-appropriate "
    "candidate_label; do not select from a fixed emotion taxonomy. Strength is an "
    "explicitly uncalibrated model estimate, not a measured level; strength and "
    "uncertainty are finite 0..1 values, and each array contains 0..6 short "
    "observations. resolution_reason is free-form and may appear only with a "
    "revision_target_ref when evidence no longer warrants keeping that target active. "
    "Do not supply epistemic or phenomenology status."
)
SELF_OBSERVATION_DIARY_SOURCE_KIND = "WORLD"

_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BASE_FIELDS = frozenset(
    (
        "alternative_explanations",
        "candidate_label",
        "estimated_strength",
        "functional_description",
        "observable_signals",
        "uncertainty",
    )
)
_OPTIONAL_FIELDS = frozenset(("revision_target_ref", "resolution_reason"))
_LABEL_MAX_BYTES = 256
_DESCRIPTION_MAX_BYTES = 4_096
_SIGNAL_MAX_BYTES = 512
_MAX_SIGNALS = 6
_MAX_PROPOSAL_BYTES = 16_384


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
        raise ValueError("self-observation value is not canonical JSON data") from exc


def _source_ref() -> str:
    material = _canonical_json(
        {
            "affordance_id": SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
            "contract": SELF_OBSERVATION_DIARY_CONTRACT,
            "purpose": SELF_OBSERVATION_DIARY_PURPOSE,
            "source_kind": SELF_OBSERVATION_DIARY_SOURCE_KIND,
        }
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


SELF_OBSERVATION_DIARY_SOURCE_REF = _source_ref()


def _bounded_text(value: object, label: str, maximum_bytes: int) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty bounded text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid Unicode text") from exc
    if size > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte ceiling")
    return value


def _unit_estimate(value: object, label: str) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be a JSON number")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be finite in [0, 1]")
    return value


def _reference(value: object, label: str) -> str:
    if type(value) is not str or _REFERENCE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _short_text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) > _MAX_SIGNALS:
        raise ValueError(f"{label} must be an immutable tuple of 0 through 6 items")
    return tuple(
        _bounded_text(item, f"{label} item", _SIGNAL_MAX_BYTES) for item in value
    )


def _parse_canonical_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact canonical JSON text")
    try:
        if not 1 <= len(value.encode("utf-8", errors="strict")) <= _MAX_PROPOSAL_BYTES:
            raise ValueError(f"{label} exceeds its byte ceiling")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid Unicode text") from exc

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
class SelfObservationDiaryProposal:
    """One model-authored hypothesis, not a measurement or diary commit."""

    candidate_label: str
    functional_description: str
    estimated_strength: int | float
    uncertainty: int | float
    observable_signals: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    revision_target_ref: str | None = None
    resolution_reason: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.candidate_label, "candidate_label", _LABEL_MAX_BYTES)
        _bounded_text(
            self.functional_description,
            "functional_description",
            _DESCRIPTION_MAX_BYTES,
        )
        _unit_estimate(self.estimated_strength, "estimated_strength")
        _unit_estimate(self.uncertainty, "uncertainty")
        _short_text_tuple(self.observable_signals, "observable_signals")
        _short_text_tuple(
            self.alternative_explanations,
            "alternative_explanations",
        )
        if self.revision_target_ref is not None:
            _reference(self.revision_target_ref, "revision_target_ref")
        if self.resolution_reason is not None:
            _bounded_text(
                self.resolution_reason,
                "resolution_reason",
                _DESCRIPTION_MAX_BYTES,
            )
            if self.revision_target_ref is None:
                raise ValueError(
                    "resolution_reason requires a revision_target_ref"
                )

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "alternative_explanations": list(self.alternative_explanations),
            "candidate_label": self.candidate_label,
            "estimated_strength": self.estimated_strength,
            "functional_description": self.functional_description,
            "observable_signals": list(self.observable_signals),
            "uncertainty": self.uncertainty,
        }
        if self.revision_target_ref is not None:
            payload["revision_target_ref"] = self.revision_target_ref
        if self.resolution_reason is not None:
            payload["resolution_reason"] = self.resolution_reason
        return payload

    def canonical_json(self) -> str:
        value = _canonical_json(self.canonical_payload())
        if len(value.encode("utf-8")) > _MAX_PROPOSAL_BYTES:
            raise ValueError("self-observation proposal exceeds its byte ceiling")
        return value

    @classmethod
    def from_json(cls, value: object) -> "SelfObservationDiaryProposal":
        payload = _parse_canonical_object(value, "self-observation proposal")
        fields = frozenset(payload)
        valid_fields = (
            _BASE_FIELDS,
            _BASE_FIELDS | {"revision_target_ref"},
            _BASE_FIELDS | _OPTIONAL_FIELDS,
        )
        if fields not in valid_fields:
            raise ValueError("self-observation proposal fields differ")
        raw_signals = payload["observable_signals"]
        raw_alternatives = payload["alternative_explanations"]
        if type(raw_signals) is not list or type(raw_alternatives) is not list:
            raise TypeError("self-observation signal fields must be JSON arrays")
        proposal = cls(
            candidate_label=payload["candidate_label"],  # type: ignore[arg-type]
            functional_description=payload[  # type: ignore[arg-type]
                "functional_description"
            ],
            estimated_strength=payload["estimated_strength"],  # type: ignore[arg-type]
            uncertainty=payload["uncertainty"],  # type: ignore[arg-type]
            observable_signals=tuple(raw_signals),  # type: ignore[arg-type]
            alternative_explanations=tuple(raw_alternatives),  # type: ignore[arg-type]
            revision_target_ref=payload.get(  # type: ignore[arg-type]
                "revision_target_ref"
            ),
            resolution_reason=payload.get("resolution_reason"),  # type: ignore[arg-type]
        )
        if proposal.canonical_json() != value:
            raise ValueError("self-observation proposal is not canonical")
        return proposal


SELF_OBSERVATION_DIARY_AFFORDANCE = Affordance(
    affordance_id=SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
    disposition="ACT",
    description=SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION,
    permission_scope="internal.cognition",
    external_effect=False,
)


class SelfObservationDiaryExecutor:
    """Validate and source-bind one proposal without writing any state."""

    SOURCE_REF = SELF_OBSERVATION_DIARY_SOURCE_REF
    SOURCE_KIND = SELF_OBSERVATION_DIARY_SOURCE_KIND
    AFFORDANCE_ID = SELF_OBSERVATION_DIARY_AFFORDANCE_ID

    def __call__(self, request: AffordanceRequest) -> AffordanceReceipt:
        if type(request) is not AffordanceRequest:
            raise TypeError("request must be an exact AffordanceRequest")
        if request.affordance_id != self.AFFORDANCE_ID:
            raise ValueError("request selected a different affordance")
        proposal = SelfObservationDiaryProposal.from_json(request.action_payload)
        observation_json = _canonical_json(
            {
                "contract": SELF_OBSERVATION_DIARY_CONTRACT,
                "proposal": proposal.canonical_payload(),
                "purpose": SELF_OBSERVATION_DIARY_PURPOSE,
            }
        )
        observation = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind=self.SOURCE_KIND,
            source_ref=self.SOURCE_REF,
            observation_json=observation_json,
        )
        return AffordanceReceipt(
            status="COMPLETED",
            output=(
                "Self-observation hypothesis validated without reward or permission."
            ),
            consequence=(),
            observable_consequence=observation,
        )


def proposal_from_observable_consequence(
    value: ObservableConsequence,
) -> SelfObservationDiaryProposal:
    """Recognize and recover this module's exact source-bound proposal."""

    if type(value) is not ObservableConsequence:
        raise TypeError("value must be an exact ObservableConsequence")
    if (
        value.source_kind != SELF_OBSERVATION_DIARY_SOURCE_KIND
        or value.source_ref != SELF_OBSERVATION_DIARY_SOURCE_REF
    ):
        raise ValueError("observable consequence is not self-observation diary output")
    payload = _parse_canonical_object(
        value.observation_json,
        "self-observation consequence",
    )
    if set(payload) != {"contract", "proposal", "purpose"}:
        raise ValueError("self-observation consequence fields differ")
    if (
        payload["contract"] != SELF_OBSERVATION_DIARY_CONTRACT
        or payload["purpose"] != SELF_OBSERVATION_DIARY_PURPOSE
        or type(payload["proposal"]) is not dict
    ):
        raise ValueError("self-observation consequence binding differs")
    return SelfObservationDiaryProposal.from_json(
        _canonical_json(payload["proposal"])
    )


__all__ = [
    "SELF_OBSERVATION_DIARY_AFFORDANCE",
    "SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION",
    "SELF_OBSERVATION_DIARY_AFFORDANCE_ID",
    "SELF_OBSERVATION_DIARY_APPRAISAL_ROLE",
    "SELF_OBSERVATION_DIARY_CONTRACT",
    "SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS",
    "SELF_OBSERVATION_DIARY_LABEL_STATUS",
    "SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS",
    "SELF_OBSERVATION_DIARY_PURPOSE",
    "SELF_OBSERVATION_DIARY_SOURCE_KIND",
    "SELF_OBSERVATION_DIARY_SOURCE_REF",
    "SELF_OBSERVATION_DIARY_STRENGTH_STATUS",
    "SelfObservationDiaryExecutor",
    "SelfObservationDiaryProposal",
    "proposal_from_observable_consequence",
]
