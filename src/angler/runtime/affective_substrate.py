"""Unlabelled event substrate for later learned affect dynamics.

This module performs mechanical projection only.  It does not infer emotion,
valence, arousal, importance, reward, social sentiment, or action priority.
Every retained number keeps the provenance it had in the canonical episode;
model estimates and receipt-declared consequence are not relabelled as
measurements or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping


EVENT_SUBSTRATE_CONTRACT = "jenny.event-substrate-observation.v1"
EVENT_SUBSTRATE_EPISTEMIC_STATUS = (
    "MECHANICAL_PROJECTION_FROM_CANONICAL_EPISODE_NO_AFFECT_SEMANTICS"
)
_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _reference(value: object, label: str) -> str:
    if type(value) is not str or _REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _optional_model_uncertainty(value: object, label: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _choice_context(choice: Mapping[str, object]) -> dict[str, object]:
    raw = choice.get("context_json", "{}")
    if type(raw) is not str:
        raise ValueError("choice context_json must be text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("choice context_json must be JSON") from exc
    if type(value) is not dict:
        raise ValueError("choice context_json must encode an object")
    return value


def _retrieval_observations(context: Mapping[str, object]) -> list[dict[str, object]]:
    raw = context.get("memories", [])
    if type(raw) is not list:
        raise ValueError("choice memories must be a list")
    observations: list[dict[str, object]] = []
    for item in raw:
        if type(item) is not dict:
            raise ValueError("choice memory must be an object")
        reference = item.get("record_ref")
        distance = item.get("semantic_distance")
        if reference is None or distance is None:
            continue
        observations.append(
            {
                "record_ref": _reference(reference, "memory record_ref"),
                "semantic_distance": _finite(
                    distance, "memory semantic_distance"
                ),
            }
        )
    return observations


def _runtime_observations(context: Mapping[str, object]) -> dict[str, float]:
    raw = context.get("phase_timings_ms", {})
    if type(raw) is not dict:
        raise ValueError("phase timings must be an object")
    observations: dict[str, float] = {}
    for name, value in raw.items():
        if type(name) is not str or not name:
            raise ValueError("phase timing name must be text")
        measured = _finite(value, "phase timing")
        if measured < 0.0:
            raise ValueError("phase timing must be non-negative")
        observations[name] = measured
    return dict(sorted(observations.items()))


def _temporal_observation(temporal: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "acquired_time_utc": temporal.get("acquired_time_utc"),
        "clock_jump_detected": temporal.get("clock_jump_detected"),
        "event_time_utc": temporal.get("event_time_utc"),
        "precision_ms": temporal.get("precision_ms"),
        "recorded_time_utc": temporal.get("recorded_time_utc"),
        "timezone": temporal.get("timezone"),
        "uncertainty_ms": temporal.get("uncertainty_ms"),
        "valid_from_utc": temporal.get("valid_from_utc"),
        "valid_until_utc": temporal.get("valid_until_utc"),
        "verified_time_utc": temporal.get("verified_time_utc"),
    }
    for name in (
        "acquired_time_utc",
        "event_time_utc",
        "recorded_time_utc",
        "timezone",
        "valid_from_utc",
        "valid_until_utc",
        "verified_time_utc",
    ):
        value = result[name]
        if value is not None and type(value) is not str:
            raise ValueError(f"temporal {name} must be text or null")
    for name in ("precision_ms", "uncertainty_ms"):
        value = result[name]
        if value is not None:
            measured = _finite(value, f"temporal {name}")
            if measured < 0.0:
                raise ValueError(f"temporal {name} must be non-negative")
            result[name] = measured
    if type(result["clock_jump_detected"]) is not bool:
        raise ValueError("temporal clock_jump_detected must be boolean")
    result["value_kind"] = "TRUSTED_CLOCK_MEASUREMENT"
    return result


def _commitment_observation(
    context: Mapping[str, object],
) -> dict[str, object] | None:
    initiative = context.get("autonomous_initiative")
    if initiative is None:
        return None
    if type(initiative) is not dict:
        raise ValueError("autonomous initiative must be an object")
    formation = initiative.get("formation")
    if type(formation) is not dict:
        return None
    proposal = formation.get("proposal")
    if type(proposal) is not dict:
        return None
    commitment = proposal.get("commitment")
    if commitment is None:
        return None
    if type(commitment) is not dict:
        raise ValueError("autonomous commitment must be an object")
    status = commitment.get("status")
    statement = commitment.get("statement")
    evidence = commitment.get("evidence_keys", [])
    if (
        status not in {"ACTIVE", "NONE"}
        or type(statement) is not str
        or type(evidence) is not list
        or any(type(item) is not str for item in evidence)
    ):
        raise ValueError("autonomous commitment observation is malformed")
    return {
        "commitment_ref": _digest(commitment),
        "evidence_count": len(evidence),
        "model_authored_uncertainty": _optional_model_uncertainty(
            commitment.get("uncertainty"), "commitment uncertainty"
        ),
        "statement_ref": _digest(statement),
        "status": status,
        "value_kind": "MODEL_AUTHORED_COMMITMENT_ESTIMATE",
    }


def _prediction_observation(
    choice: Mapping[str, object], context: Mapping[str, object]
) -> dict[str, object]:
    prediction = choice.get("prediction")
    rankings = choice.get("rankings")
    if type(prediction) is not str or type(rankings) is not list:
        raise ValueError("canonical choice prediction or rankings are malformed")
    ranking_observations: list[dict[str, object]] = []
    for item in rankings:
        if (
            type(item) is not dict
            or type(item.get("affordance_id")) is not str
        ):
            raise ValueError("canonical choice ranking is malformed")
        ranking_observations.append(
            {
                "affordance_id": item["affordance_id"],
                "declared_score": _finite(
                    item.get("score"), "choice ranking score"
                ),
            }
        )
    intent = context.get("intent_proposal")
    intent_uncertainty = None
    alternative_count = 0
    if intent is not None:
        if type(intent) is not dict:
            raise ValueError("intent proposal must be an object")
        intent_uncertainty = _optional_model_uncertainty(
            intent.get("uncertainty"), "intent uncertainty"
        )
        alternatives = intent.get("alternatives", [])
        if type(alternatives) is not list:
            raise ValueError("intent alternatives must be a list")
        alternative_count = len(alternatives)
    return {
        "alternative_count": alternative_count,
        "choice_prediction_ref": _digest(prediction),
        "choice_uncertainty": _optional_model_uncertainty(
            choice.get("uncertainty"), "choice uncertainty"
        ),
        "intent_uncertainty": intent_uncertainty,
        "rankings": ranking_observations,
        "value_kind": "MODEL_AUTHORED_OR_CONTROLLER_DECLARATION",
    }


def _closure_observation(receipt: Mapping[str, object]) -> dict[str, object]:
    status = receipt.get("status")
    consequence = receipt.get("consequence")
    if type(status) is not str or type(consequence) not in (list, tuple):
        raise ValueError("canonical receipt status or consequence is malformed")
    declared: list[list[object]] = []
    for item in consequence:
        if (
            type(item) not in (list, tuple)
            or len(item) != 2
            or type(item[0]) is not str
        ):
            raise ValueError("declared consequence row is malformed")
        declared.append([item[0], _finite(item[1], "declared consequence")])
    observable = receipt.get("observable_consequence")
    observable_ref = None
    source_kind = None
    source_ref = None
    if observable is not None:
        if type(observable) is not dict:
            raise ValueError("observable consequence must be an object")
        observable_ref = _digest(observable)
        source_kind = observable.get("source_kind")
        source_ref = observable.get("source_ref")
        if type(source_kind) is not str:
            raise ValueError("observable source_kind must be text")
        _reference(source_ref, "observable source_ref")
    return {
        "closure_kind": "CANONICAL_TRANSACTION_CLOSED",
        "declared_consequence": declared,
        "declared_consequence_kind": (
            "RECEIPT_DECLARED_DIRECT_OR_SOURCE_BOUND_MAPPER_OUTPUT"
        ),
        "observable_consequence_ref": observable_ref,
        "observable_source_kind": source_kind,
        "observable_source_ref": source_ref,
        "receipt_ref": _digest(receipt),
        "receipt_status": status,
    }


@dataclass(frozen=True, slots=True)
class EventSubstrateObservation:
    """One content-addressed, text-free observation of a closed event."""

    episode_ref: str
    event_ref: str
    ordinal: int
    sample_ref: str
    payload_json: str

    def __post_init__(self) -> None:
        _reference(self.episode_ref, "episode_ref")
        _reference(self.event_ref, "event_ref")
        _reference(self.sample_ref, "sample_ref")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if type(self.payload_json) is not str or len(self.payload_json) > 65_536:
            raise ValueError("payload_json must be bounded text")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json must be JSON") from exc
        if type(payload) is not dict:
            raise ValueError("payload_json must encode an object")
        if (
            payload.get("contract") != EVENT_SUBSTRATE_CONTRACT
            or payload.get("epistemic_status")
            != EVENT_SUBSTRATE_EPISTEMIC_STATUS
            or payload.get("episode_ref") != self.episode_ref
            or payload.get("event_ref") != self.event_ref
            or payload.get("moving_origin_ordinal") != self.ordinal
        ):
            raise ValueError("event substrate identity differs")
        without_ref = dict(payload)
        if without_ref.pop("sample_ref", None) != self.sample_ref:
            raise ValueError("event substrate sample_ref differs")
        if _digest(without_ref) != self.sample_ref:
            raise ValueError("event substrate content digest differs")
        if _canonical(payload).decode("utf-8") != self.payload_json:
            raise ValueError("event substrate payload must be canonical JSON")


def project_canonical_episode(
    *, episode_ref: str, episode_payload: Mapping[str, object]
) -> EventSubstrateObservation:
    """Project one canonical episode without assigning affective semantics."""

    _reference(episode_ref, "episode_ref")
    event_ref = _reference(episode_payload.get("event_ref"), "event_ref")
    temporal = episode_payload.get("temporal")
    observation = episode_payload.get("observation")
    choice = episode_payload.get("choice")
    receipt = episode_payload.get("receipt")
    if not all(type(item) is dict for item in (temporal, observation, choice, receipt)):
        raise ValueError("canonical episode fields are malformed")
    assert isinstance(temporal, dict)
    assert isinstance(observation, dict)
    assert isinstance(choice, dict)
    assert isinstance(receipt, dict)
    ordinal = temporal.get("moving_origin_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("canonical episode ordinal is malformed")
    observation_ref = _digest(observation)
    choice_ref = _digest(choice)
    temporal_ref = _digest(temporal)
    context = _choice_context(choice)
    parent_state_ref = _reference(
        episode_payload.get("parent_state_ref"), "parent_state_ref"
    )
    child_state_ref = _reference(
        episode_payload.get("child_state_ref"), "child_state_ref"
    )
    payload: dict[str, object] = {
        "choice_ref": choice_ref,
        "closure": _closure_observation(receipt),
        "commitment": _commitment_observation(context),
        "contract": EVENT_SUBSTRATE_CONTRACT,
        "epistemic_status": EVENT_SUBSTRATE_EPISTEMIC_STATUS,
        "episode_ref": episode_ref,
        "event_ref": event_ref,
        "learned_state_transition": {
            "changed": parent_state_ref != child_state_ref,
            "child_state_ref": child_state_ref,
            "parent_state_ref": parent_state_ref,
        },
        "moving_origin_ordinal": ordinal,
        "observation": {
            "observation_ref": observation_ref,
            "source": observation.get("source"),
        },
        "prediction": _prediction_observation(choice, context),
        "retrieval": {
            "observations": _retrieval_observations(context),
            "value_kind": "RECORDED_SEMANTIC_DISTANCE",
        },
        "runtime": {
            "phase_timings_ms": _runtime_observations(context),
            "value_kind": "MEASURED_RUNTIME_DURATION",
        },
        "selected_affordance_id": choice.get("selected_affordance_id"),
        "temporal": _temporal_observation(temporal),
        "temporal_ref": temporal_ref,
    }
    sample_ref = _digest(payload)
    payload["sample_ref"] = sample_ref
    return EventSubstrateObservation(
        episode_ref=episode_ref,
        event_ref=event_ref,
        ordinal=ordinal,
        sample_ref=sample_ref,
        payload_json=_canonical(payload).decode("utf-8"),
    )


__all__ = [
    "EVENT_SUBSTRATE_CONTRACT",
    "EVENT_SUBSTRATE_EPISTEMIC_STATUS",
    "EventSubstrateObservation",
    "project_canonical_episode",
]
