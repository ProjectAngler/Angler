"""Bounded online appraisal dynamics learned from grounded event signals.

This module deliberately does not assign emotions, valence, reward, or action
policy.  It learns each numeric signal's empirical baseline and the covariance
between signals that were observed together.  The resulting innovation and
relation view is evidence a model may interpret; it is never a scripted cause
of behaviour and makes no phenomenology claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping, MutableMapping, Sequence


GROUNDED_APPRAISAL_CONTRACT = "jenny.grounded-appraisal-dynamics.v1"
GROUNDED_APPRAISAL_STATE_KEY = "grounded_appraisal_dynamics"
GROUNDED_APPRAISAL_EPISTEMIC_STATUS = (
    "EMPIRICAL_SIGNAL_DYNAMICS_WITHOUT_EMOTION_SEMANTICS"
)
GROUNDED_APPRAISAL_PHENOMENOLOGY_STATUS = "NO_SUBJECTIVE_EXPERIENCE_CLAIM"
MAX_APPRAISAL_FEATURES = 64
MAX_APPRAISAL_CONTEXT_CHARACTERS = 2_048
MAX_CONTEXT_INNOVATIONS = 4
MAX_CONTEXT_RELATIONS = 4

_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNAL = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SOURCES = frozenset({"SCHEDULER", "HUMAN", "AGENT", "CONTINUATION"})
_STATUSES = frozenset({"COMPLETED", "COMPLETED_UNEVALUATED", "DENIED", "ERROR"})


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _optional_unit(value: object, label: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _signal_name(value: object) -> str:
    if type(value) is not str or _SIGNAL.fullmatch(value) is None:
        raise ValueError("appraisal signal name is invalid")
    return value


def _reference(value: object, label: str) -> str:
    if type(value) is not str or _REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def event_appraisal_signals(
    *,
    choice: Mapping[str, object],
    receipt: Mapping[str, object],
    context: Mapping[str, object],
    temporal: Mapping[str, object],
) -> dict[str, float]:
    """Project numeric observations without inventing affective semantics."""

    choice_uncertainty = _optional_unit(
        choice.get("uncertainty"), "choice uncertainty"
    )
    if choice_uncertainty is None:
        raise ValueError("choice uncertainty is required")
    signals: dict[str, float] = {
        "choice.uncertainty": choice_uncertainty,
    }
    rankings = choice.get("rankings", [])
    if type(rankings) not in (list, tuple) or not rankings:
        raise ValueError("choice rankings must be a non-empty sequence")
    ranking_values: list[float] = []
    selected_id = choice.get("selected_affordance_id")
    selected_score: float | None = None
    for row in rankings:
        item = _mapping(row, "choice ranking")
        affordance_id = item.get("affordance_id")
        if type(affordance_id) is not str or not affordance_id:
            raise ValueError("choice ranking affordance is invalid")
        score = _finite(item.get("score"), "choice ranking score")
        ranking_values.append(score)
        if affordance_id == selected_id:
            selected_score = score
    if selected_score is None:
        raise ValueError("selected affordance is absent from rankings")
    signals.update(
        {
            "choice.ranking_count": float(len(ranking_values)),
            "choice.selected_score": selected_score,
            "choice.score_spread": max(ranking_values) - min(ranking_values),
        }
    )

    intent = context.get("intent_proposal")
    if intent is not None:
        intent = _mapping(intent, "intent proposal")
        uncertainty = _optional_unit(intent.get("uncertainty"), "intent uncertainty")
        if uncertainty is not None:
            signals["intent.uncertainty"] = uncertainty
        alternatives = intent.get("alternatives", [])
        if type(alternatives) is not list:
            raise ValueError("intent alternatives must be a list")
        signals["intent.alternative_count"] = float(len(alternatives))

    memories = context.get("memories", [])
    if type(memories) is not list:
        raise ValueError("context memories must be a list")
    distances: list[float] = []
    for row in memories:
        item = _mapping(row, "context memory")
        if item.get("semantic_distance") is not None:
            distances.append(
                _finite(item["semantic_distance"], "memory semantic distance")
            )
    signals["retrieval.count"] = float(len(distances))
    if distances:
        signals.update(
            {
                "retrieval.nearest_distance": min(distances),
                "retrieval.mean_distance": sum(distances) / len(distances),
                "retrieval.distance_spread": max(distances) - min(distances),
            }
        )

    initiative = context.get("autonomous_initiative")
    if initiative is not None:
        formation = _mapping(initiative, "autonomous initiative").get("formation")
        if formation is not None:
            proposal = _mapping(formation, "autonomous formation").get("proposal")
            if proposal is not None:
                commitment = _mapping(proposal, "autonomous proposal").get(
                    "commitment"
                )
                if commitment is not None:
                    commitment = _mapping(commitment, "autonomous commitment")
                    uncertainty = _optional_unit(
                        commitment.get("uncertainty"), "commitment uncertainty"
                    )
                    if uncertainty is not None:
                        signals["commitment.uncertainty"] = uncertainty
                    evidence = commitment.get("evidence_keys", [])
                    if type(evidence) is not list:
                        raise ValueError("commitment evidence keys must be a list")
                    signals["commitment.evidence_count"] = float(len(evidence))

    consequence = receipt.get("consequence", [])
    if type(consequence) not in (list, tuple) or len(consequence) > 32:
        raise ValueError("receipt consequence must be a bounded sequence")
    for row in consequence:
        if type(row) not in (list, tuple) or len(row) != 2:
            raise ValueError("receipt consequence row is malformed")
        name = _signal_name("consequence." + str(row[0]))
        if name in signals:
            raise ValueError("receipt consequence names must be unique")
        signals[name] = _finite(row[1], "receipt consequence")

    clock_uncertainty = temporal.get("uncertainty_ms")
    if clock_uncertainty is not None:
        value = _finite(clock_uncertainty, "temporal uncertainty")
        if value < 0.0:
            raise ValueError("temporal uncertainty must be non-negative")
        signals["temporal.clock_uncertainty_ms"] = value
    if len(signals) > MAX_APPRAISAL_FEATURES:
        raise ValueError("event exceeds the appraisal feature ceiling")
    return dict(sorted(signals.items()))


def _updated_moment(previous: Mapping[str, object] | None, value: float) -> dict[str, object]:
    if previous is None:
        return {"count": 1, "mean": value, "m2": 0.0, "minimum": value, "maximum": value}
    count = previous.get("count")
    if type(count) is not int or count < 1:
        raise ValueError("appraisal feature count is invalid")
    mean = _finite(previous.get("mean"), "appraisal feature mean")
    m2 = _finite(previous.get("m2"), "appraisal feature m2")
    minimum = _finite(previous.get("minimum"), "appraisal feature minimum")
    maximum = _finite(previous.get("maximum"), "appraisal feature maximum")
    if m2 < 0.0:
        raise ValueError("appraisal feature m2 must be non-negative")
    next_count = count + 1
    delta = value - mean
    next_mean = mean + delta / next_count
    next_m2 = m2 + delta * (value - next_mean)
    return {
        "count": next_count,
        "mean": next_mean,
        "m2": max(0.0, next_m2),
        "minimum": min(minimum, value),
        "maximum": max(maximum, value),
    }


def _innovation(previous: Mapping[str, object] | None, value: float) -> dict[str, object]:
    if previous is None:
        return {
            "basis_count": 0,
            "kind": "NEW_SIGNAL",
            "prior_mean": None,
            "standardized_innovation": None,
            "value": value,
        }
    count = previous.get("count")
    if type(count) is not int or count < 1:
        raise ValueError("appraisal innovation basis count is invalid")
    mean = _finite(previous.get("mean"), "appraisal innovation mean")
    m2 = _finite(previous.get("m2"), "appraisal innovation m2")
    if count < 2:
        standardized = None
        kind = "INSUFFICIENT_VARIANCE_HISTORY"
    else:
        variance = m2 / (count - 1)
        if variance > 0.0:
            standardized = (value - mean) / math.sqrt(variance)
            kind = "STANDARDIZED_FROM_PRIOR_HISTORY"
        elif value == mean:
            standardized = 0.0
            kind = "MATCHES_PRIOR_CONSTANT"
        else:
            standardized = None
            kind = "DIFFERS_FROM_PRIOR_CONSTANT"
    return {
        "basis_count": count,
        "kind": kind,
        "prior_mean": mean,
        "standardized_innovation": standardized,
        "value": value,
    }


def _relation_key(left: str, right: str) -> str:
    return _canonical([left, right])


def _updated_relation(
    previous: Mapping[str, object] | None, left: float, right: float
) -> dict[str, object]:
    if previous is None:
        return {
            "count": 1,
            "left_mean": left,
            "right_mean": right,
            "left_m2": 0.0,
            "right_m2": 0.0,
            "co_m2": 0.0,
        }
    count = previous.get("count")
    if type(count) is not int or count < 1:
        raise ValueError("appraisal relation count is invalid")
    left_mean = _finite(previous.get("left_mean"), "relation left mean")
    right_mean = _finite(previous.get("right_mean"), "relation right mean")
    left_m2 = _finite(previous.get("left_m2"), "relation left m2")
    right_m2 = _finite(previous.get("right_m2"), "relation right m2")
    co_m2 = _finite(previous.get("co_m2"), "relation co-moment")
    if left_m2 < 0.0 or right_m2 < 0.0:
        raise ValueError("appraisal relation m2 must be non-negative")
    next_count = count + 1
    delta_left = left - left_mean
    delta_right = right - right_mean
    next_left_mean = left_mean + delta_left / next_count
    next_right_mean = right_mean + delta_right / next_count
    return {
        "count": next_count,
        "left_mean": next_left_mean,
        "right_mean": next_right_mean,
        "left_m2": max(0.0, left_m2 + delta_left * (left - next_left_mean)),
        "right_m2": max(0.0, right_m2 + delta_right * (right - next_right_mean)),
        "co_m2": co_m2 + delta_left * (right - next_right_mean),
    }


def _strongest_relations(
    relations: Mapping[str, object], *, maximum: int = MAX_CONTEXT_RELATIONS
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for key, raw in relations.items():
        if type(key) is not str or type(raw) is not dict:
            raise ValueError("appraisal relation store is malformed")
        try:
            names = json.loads(key)
        except json.JSONDecodeError as exc:
            raise ValueError("appraisal relation identity is malformed") from exc
        if (
            type(names) is not list
            or len(names) != 2
            or any(type(item) is not str for item in names)
        ):
            raise ValueError("appraisal relation identity is malformed")
        count = raw.get("count")
        if type(count) is not int or count < 1:
            raise ValueError("appraisal relation count is invalid")
        left_m2 = _finite(raw.get("left_m2"), "relation left m2")
        right_m2 = _finite(raw.get("right_m2"), "relation right m2")
        co_m2 = _finite(raw.get("co_m2"), "relation co-moment")
        if count < 3 or left_m2 <= 0.0 or right_m2 <= 0.0:
            continue
        correlation = co_m2 / math.sqrt(left_m2 * right_m2)
        candidates.append(
            {
                "coobservations": count,
                "correlation": max(-1.0, min(1.0, correlation)),
                "left_signal": names[0],
                "right_signal": names[1],
            }
        )
    candidates.sort(
        key=lambda item: (
            -abs(float(item["correlation"])),
            str(item["left_signal"]),
            str(item["right_signal"]),
        )
    )
    return candidates[:maximum]


def update_grounded_appraisal(
    state_payload: MutableMapping[str, object],
    *,
    choice: Mapping[str, object],
    receipt: Mapping[str, object],
    context: Mapping[str, object],
    temporal: Mapping[str, object],
    observation_source: str,
    choice_ref: str,
    receipt_ref: str,
) -> dict[str, object]:
    """Apply one forward-only online update to the bounded appraisal graph."""

    if observation_source not in _SOURCES:
        raise ValueError("observation source is unsupported")
    status = receipt.get("status")
    if status not in _STATUSES:
        raise ValueError("receipt status is unsupported")
    choice_ref = _reference(choice_ref, "choice_ref")
    receipt_ref = _reference(receipt_ref, "receipt_ref")
    ordinal = temporal.get("moving_origin_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("moving origin ordinal is invalid")
    signals = event_appraisal_signals(
        choice=choice, receipt=receipt, context=context, temporal=temporal
    )

    raw_state = state_payload.get(GROUNDED_APPRAISAL_STATE_KEY)
    if raw_state is None:
        appraisal: dict[str, object] = {
            "contract": GROUNDED_APPRAISAL_CONTRACT,
            "epistemic_status": GROUNDED_APPRAISAL_EPISTEMIC_STATUS,
            "phenomenology_status": GROUNDED_APPRAISAL_PHENOMENOLOGY_STATUS,
            "behavior_role": "ADVISORY_EVIDENCE_NO_DIRECT_ACTION_AUTHORITY",
            "observation_count": 0,
            "feature_moments": {},
            "relation_moments": {},
            "latest": None,
        }
    elif type(raw_state) is dict:
        appraisal = dict(raw_state)
        if (
            appraisal.get("contract") != GROUNDED_APPRAISAL_CONTRACT
            or appraisal.get("epistemic_status")
            != GROUNDED_APPRAISAL_EPISTEMIC_STATUS
            or appraisal.get("phenomenology_status")
            != GROUNDED_APPRAISAL_PHENOMENOLOGY_STATUS
            or appraisal.get("behavior_role")
            != "ADVISORY_EVIDENCE_NO_DIRECT_ACTION_AUTHORITY"
        ):
            raise ValueError("grounded appraisal state identity differs")
    else:
        raise ValueError("grounded appraisal state must be an object")

    count = appraisal.get("observation_count")
    moments = appraisal.get("feature_moments")
    relations = appraisal.get("relation_moments")
    if (
        type(count) is not int
        or count < 0
        or type(moments) is not dict
        or type(relations) is not dict
    ):
        raise ValueError("grounded appraisal state is malformed")
    known = set(moments)
    novel = set(signals) - known
    if len(known | novel) > MAX_APPRAISAL_FEATURES:
        raise ValueError("grounded appraisal lifetime feature ceiling exceeded")

    innovations: dict[str, object] = {}
    for name, value in signals.items():
        _signal_name(name)
        previous = moments.get(name)
        if previous is not None and type(previous) is not dict:
            raise ValueError("appraisal feature moment is malformed")
        innovations[name] = _innovation(previous, value)
        moments[name] = _updated_moment(previous, value)

    names = sorted(signals)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            key = _relation_key(left_name, right_name)
            previous = relations.get(key)
            if previous is not None and type(previous) is not dict:
                raise ValueError("appraisal relation moment is malformed")
            relations[key] = _updated_relation(
                previous, signals[left_name], signals[right_name]
            )

    metadata = {
        "choice_ref": choice_ref,
        "commitment_status": None,
        "event_signal_ref": _digest(signals),
        "moving_origin_ordinal": ordinal,
        "observation_source": observation_source,
        "observable_consequence_present": receipt.get("observable_consequence")
        is not None,
        "receipt_ref": receipt_ref,
        "receipt_status": status,
        "selected_affordance_id": choice.get("selected_affordance_id"),
    }
    initiative = context.get("autonomous_initiative")
    if type(initiative) is dict:
        formation = initiative.get("formation")
        if type(formation) is dict:
            proposal = formation.get("proposal")
            if type(proposal) is dict and type(proposal.get("commitment")) is dict:
                metadata["commitment_status"] = proposal["commitment"].get("status")
    appraisal["observation_count"] = count + 1
    appraisal["feature_moments"] = moments
    appraisal["relation_moments"] = relations
    appraisal["latest"] = {
        "innovations": innovations,
        "metadata": metadata,
        "signal_values": signals,
    }
    state_payload[GROUNDED_APPRAISAL_STATE_KEY] = appraisal
    return appraisal


def grounded_appraisal_context(
    state_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the bounded current evidence view supplied to existing model calls."""

    raw = state_payload.get(GROUNDED_APPRAISAL_STATE_KEY)
    if raw is None:
        return None
    appraisal = _mapping(raw, "grounded appraisal state")
    if appraisal.get("contract") != GROUNDED_APPRAISAL_CONTRACT:
        raise ValueError("grounded appraisal context contract differs")
    latest = _mapping(appraisal.get("latest"), "grounded appraisal latest")
    innovations = _mapping(latest.get("innovations"), "appraisal innovations")
    standardized: list[tuple[float, str, Mapping[str, object]]] = []
    unstandardized: list[tuple[str, Mapping[str, object]]] = []
    for name, raw_innovation in innovations.items():
        _signal_name(name)
        innovation = _mapping(raw_innovation, "appraisal innovation")
        value = innovation.get("standardized_innovation")
        if value is None:
            unstandardized.append((name, innovation))
        else:
            measured = _finite(value, "standardized appraisal innovation")
            standardized.append((abs(measured), name, innovation))
    standardized.sort(key=lambda item: (-item[0], item[1]))
    unstandardized.sort(key=lambda item: item[0])
    selected = [
        {"signal": name, **dict(innovation)}
        for _, name, innovation in standardized[:MAX_CONTEXT_INNOVATIONS]
    ]
    remaining = MAX_CONTEXT_INNOVATIONS - len(selected)
    if remaining:
        selected.extend(
            {"signal": name, **dict(innovation)}
            for name, innovation in unstandardized[:remaining]
        )
    relations = _mapping(
        appraisal.get("relation_moments"), "appraisal relation moments"
    )
    metadata = _mapping(latest.get("metadata"), "appraisal latest metadata")
    result: dict[str, object] = {
        "behavior_role": appraisal.get("behavior_role"),
        "contract": GROUNDED_APPRAISAL_CONTRACT,
        "epistemic_status": GROUNDED_APPRAISAL_EPISTEMIC_STATUS,
        "interpretation_constraint": (
            "Empirical deviations and correlations are fallible evidence, not "
            "emotion labels, causal proof, reward, or commands."
        ),
        "latest": {
            "innovations": selected,
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "commitment_status",
                    "event_signal_ref",
                    "moving_origin_ordinal",
                    "observation_source",
                    "observable_consequence_present",
                    "receipt_status",
                    "selected_affordance_id",
                )
            },
        },
        "observation_count": appraisal.get("observation_count"),
        "phenomenology_status": GROUNDED_APPRAISAL_PHENOMENOLOGY_STATUS,
        "strongest_empirical_relations": _strongest_relations(relations),
    }
    encoded = _canonical(result)
    while len(encoded) > MAX_APPRAISAL_CONTEXT_CHARACTERS:
        innovations_view = result["latest"]["innovations"]
        relations_view = result["strongest_empirical_relations"]
        if len(innovations_view) > 1 and len(innovations_view) >= len(relations_view):
            innovations_view.pop()
        elif relations_view:
            relations_view.pop()
        elif len(innovations_view) > 1:
            innovations_view.pop()
        else:
            break
        encoded = _canonical(result)
    if len(encoded) > MAX_APPRAISAL_CONTEXT_CHARACTERS:
        raise RuntimeError("grounded appraisal context exceeded its bound")
    return result


__all__ = [
    "GROUNDED_APPRAISAL_CONTRACT",
    "GROUNDED_APPRAISAL_EPISTEMIC_STATUS",
    "GROUNDED_APPRAISAL_PHENOMENOLOGY_STATUS",
    "GROUNDED_APPRAISAL_STATE_KEY",
    "MAX_APPRAISAL_CONTEXT_CHARACTERS",
    "event_appraisal_signals",
    "grounded_appraisal_context",
    "update_grounded_appraisal",
]
