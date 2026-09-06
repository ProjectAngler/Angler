"""Fast persistent procedural learning from canonical Jenny episodes.

This module does not solve tasks or assign semantic value.  It converts an
already observed action/outcome into a compact reusable case and maintains
exact online moments for each procedure's observed consequence dimensions.
Canonical episodes remain the complete history; these structures are bounded
working views that can be rebuilt from that history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, MutableMapping, Sequence


PROCEDURAL_EXPERIENCE_CONTRACT = "jenny.procedural-experience.v1"
PROCEDURAL_OUTCOME_PROFILE_CONTRACT = "jenny.procedural-outcome-profile.v1"
PROCEDURAL_OUTCOME_PROFILE_STATE_KEY = "procedural_outcome_profiles"
MAX_PROCEDURAL_MEMORY_CHARACTERS = 4_096

_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}")
_RECEIPT_STATUSES = {
    "COMPLETED",
    "COMPLETED_UNEVALUATED",
    "DENIED",
    "ERROR",
}
_PROCEDURAL_EPISTEMIC_STATUSES = {
    "OBSERVED_PROCEDURAL_OUTCOME",
    "OBSERVED_PROCEDURAL_FAILURE",
}


@dataclass(frozen=True, slots=True)
class ProceduralProjectionMetadata:
    """Trusted graph metadata reconstructed from one canonical case."""

    memory_kind: str
    epistemic_status: str
    adjacent_episode_refs: tuple[str, ...]


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _ref(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: object, *, label: str, maximum: int) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be text")
    if len(value) <= maximum:
        return value
    tail = min(96, maximum // 4)
    head = maximum - tail - len("\n[bounded]\n")
    return value[:head] + "\n[bounded]\n" + value[-tail:]


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _choice_context(choice: Mapping[str, object]) -> dict[str, object]:
    raw = choice.get("context_json")
    if type(raw) is not str:
        raise ValueError("episode choice context must be JSON text")
    try:
        context = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("episode choice context must be JSON") from exc
    return _mapping(context, label="episode choice context")


def _selected_capabilities(context: Mapping[str, object]) -> list[dict[str, object]]:
    intent = _mapping(context.get("intent_proposal"), label="intent proposal")
    cognitive = _mapping(
        context.get("cognitive_state", {}), label="choice cognitive state"
    )
    keys = intent.get("selected_capability_keys", [])
    modules = cognitive.get("capability_modules", [])
    if (
        type(keys) is not list
        or len(keys) > 16
        or len(keys) != len(set(keys))
        or any(type(item) is not str or not item for item in keys)
        or type(modules) is not list
        or any(type(item) is not dict for item in modules)
    ):
        raise ValueError("selected capability context is malformed")
    selected: list[dict[str, object]] = []
    for module in modules:
        key = module.get("capability_key")
        reference = module.get("active_capability_ref")
        revision = module.get("revision_index")
        if (
            type(key) is not str
            or type(reference) is not str
            or _SHA256_REF.fullmatch(reference) is None
            or type(revision) is not int
            or revision < 1
        ):
            raise ValueError("selected capability identity is malformed")
        selected.append(
            {
                "capability_key": key,
                "capability_ref": reference,
                "revision_index": revision,
            }
        )
    if [item["capability_key"] for item in selected] != keys:
        raise ValueError("selected capability modules differ from intent")
    return selected


def _consequence_vector(value: object) -> dict[str, float]:
    if type(value) is not list or len(value) > 32:
        raise ValueError("episode consequence must be a bounded list")
    result: dict[str, float] = {}
    for row in value:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) not in (int, float)
            or not math.isfinite(float(row[1]))
            or row[0] in result
        ):
            raise ValueError("episode consequence row is malformed")
        result[row[0]] = float(row[1])
    return result


def _bounded_observation(value: Mapping[str, object]) -> dict[str, object]:
    raw = value.get("observation_json")
    if type(raw) is not str:
        raise ValueError("observable consequence has no observation JSON")
    try:
        observation = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("observable consequence observation is not JSON") from exc
    if type(observation) is not dict or not observation:
        raise ValueError("observable consequence observation must be an object")
    canonical = _json(observation)
    if len(canonical) <= 448:
        return observation
    return {
        "content_ref": _ref(observation),
        "excerpt": _bounded_text(
            canonical, label="observable consequence", maximum=384
        ),
        "representation": "BOUNDED_CANONICAL_EXCERPT",
    }


def procedural_case_from_episode(
    payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Derive one reusable case only after an outcome actually exists.

    `COMPLETED_UNEVALUATED` model text is deliberately excluded. A legacy
    `COMPLETED` episode with neither an observation nor a consequence is also
    excluded because completion alone is not evidence that a procedure worked.
    """

    receipt = _mapping(payload.get("receipt"), label="episode receipt")
    status = receipt.get("status")
    if status not in _RECEIPT_STATUSES:
        raise ValueError("episode receipt status is unsupported")
    if status == "COMPLETED_UNEVALUATED":
        return None
    consequence = _consequence_vector(receipt.get("consequence", []))
    raw_observed = receipt.get("observable_consequence")
    if raw_observed is not None and type(raw_observed) is not dict:
        raise ValueError("observable consequence must be an object")
    if not consequence and raw_observed is None and status == "COMPLETED":
        return None

    choice = _mapping(payload.get("choice"), label="episode choice")
    observation = _mapping(payload.get("observation"), label="episode observation")
    temporal = _mapping(payload.get("temporal"), label="episode temporal record")
    context = _choice_context(choice)
    late_feedback = context.get("late_feedback")
    if late_feedback is not None and type(late_feedback) is not dict:
        raise ValueError("late feedback must be an object")
    request = context.get("request") or observation.get("content")
    intent = _mapping(context.get("intent_proposal"), label="intent proposal")
    affordance_id = choice.get("selected_affordance_id")
    action_payload = choice.get("action_payload", "")
    if type(affordance_id) is not str or not affordance_id:
        raise ValueError("selected affordance id is malformed")
    if type(action_payload) is not str:
        raise ValueError("action payload must be text")

    epistemic_status = (
        "OBSERVED_PROCEDURAL_OUTCOME"
        if status == "COMPLETED"
        else "OBSERVED_PROCEDURAL_FAILURE"
    )
    observable: dict[str, object] | None = None
    if raw_observed is not None:
        observable = {
            "observation_ref": _ref(raw_observed),
            "source_kind": raw_observed.get("source_kind"),
            "source_ref": raw_observed.get("source_ref"),
            "result": _bounded_observation(raw_observed),
        }

    feedback: dict[str, object] | None = None
    if late_feedback is not None:
        feedback = {
            "feedback_source_ref": late_feedback.get("feedback_source_ref"),
            "feedback_text": _bounded_text(
                late_feedback.get("feedback_text"),
                label="feedback text",
                maximum=384,
            ),
            "target_episode_ref": late_feedback.get("target_episode_ref"),
            "target_choice_ref": late_feedback.get("target_choice_ref"),
            "target_receipt_ref": late_feedback.get("target_receipt_ref"),
        }

    core: dict[str, object] = {
        "contract": PROCEDURAL_EXPERIENCE_CONTRACT,
        "memory_kind": "PROCEDURAL_EXPERIENCE",
        "epistemic_status": epistemic_status,
        "condition": _bounded_text(request, label="procedure request", maximum=512),
        "goal": _bounded_text(
            intent.get("resolution_target", ""),
            label="procedure goal",
            maximum=320,
        ),
        "procedure": {
            "selected_affordance_id": affordance_id,
            "action_payload": _bounded_text(
                action_payload, label="procedure action", maximum=512
            ),
            "selected_capabilities": _selected_capabilities(context),
        },
        "prediction": _bounded_text(
            intent.get("predicted_consequence", choice.get("prediction", "")),
            label="procedure prediction",
            maximum=320,
        ),
        "outcome": {
            "receipt_status": status,
            "output": _bounded_text(
                receipt.get("output", ""), label="procedure output", maximum=448
            ),
            "consequence_vector": consequence,
            "observable": observable,
        },
        "feedback": feedback,
        "temporal": {
            "moving_origin_ordinal": temporal.get("moving_origin_ordinal"),
            "event_time_utc": temporal.get("event_time_utc"),
            "acquired_time_utc": temporal.get("acquired_time_utc"),
            "recorded_time_utc": temporal.get("recorded_time_utc"),
            "verified_time_utc": temporal.get("verified_time_utc"),
            "valid_from_utc": temporal.get("valid_from_utc"),
            "valid_until_utc": temporal.get("valid_until_utc"),
            "timezone": temporal.get("timezone"),
        },
        "provenance": {
            "event_ref": payload.get("event_ref"),
            "choice_ref": _ref(choice),
            "receipt_ref": _ref(receipt),
            "parent_state_ref": payload.get("parent_state_ref"),
            "child_state_ref": payload.get("child_state_ref"),
        },
    }
    case = {**core, "case_ref": _ref(core)}
    content = _json(case)
    if len(content) > MAX_PROCEDURAL_MEMORY_CHARACTERS:
        # Preserve identities and grounded outcome vectors while shortening
        # only prose already retained in full by the canonical episode.
        case["condition"] = _bounded_text(
            case["condition"], label="procedure request", maximum=256
        )
        case["goal"] = _bounded_text(
            case["goal"], label="procedure goal", maximum=192
        )
        case["prediction"] = _bounded_text(
            case["prediction"], label="procedure prediction", maximum=192
        )
        procedure = _mapping(case["procedure"], label="procedure")
        procedure["action_payload"] = _bounded_text(
            procedure["action_payload"], label="procedure action", maximum=256
        )
        outcome = _mapping(case["outcome"], label="procedure outcome")
        output = outcome.get("output", "")
        outcome["output"] = _bounded_text(
            output, label="procedure output", maximum=192
        )
        observable = outcome.get("observable")
        if type(observable) is dict and "result" in observable:
            observable["result"] = {
                "content_ref": _ref(observable["result"]),
                "representation": "CONTENT_REFERENCE_ONLY",
            }
        case.pop("case_ref")
        case["case_ref"] = _ref(case)
        content = _json(case)
    if len(content) > MAX_PROCEDURAL_MEMORY_CHARACTERS:
        raise ValueError("procedural experience exceeds its retrieval bound")
    return case


def procedural_case_content(payload: Mapping[str, object]) -> str | None:
    case = procedural_case_from_episode(payload)
    return None if case is None else _json(case)


def procedural_projection_metadata(
    content: str,
) -> ProceduralProjectionMetadata | None:
    """Recover typed graph metadata without trusting an index or model output.

    Non-procedural canonical memory remains eligible for the ordinary episodic
    projection. A value claiming this contract must, however, be canonical and
    content-addressed exactly; malformed procedural claims fail closed instead
    of being silently downgraded to episodic prose.
    """

    if type(content) is not str or not content:
        raise ValueError("procedural projection content must be non-empty text")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("procedural projection contains a duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError:
        return None
    if type(payload) is not dict:
        return None
    contract = payload.get("contract")
    memory_kind = payload.get("memory_kind")
    if contract != PROCEDURAL_EXPERIENCE_CONTRACT:
        if memory_kind == "PROCEDURAL_EXPERIENCE":
            raise ValueError("procedural projection contract differs")
        return None
    expected = {
        "case_ref",
        "condition",
        "contract",
        "epistemic_status",
        "feedback",
        "goal",
        "memory_kind",
        "outcome",
        "prediction",
        "procedure",
        "provenance",
        "temporal",
    }
    if set(payload) != expected or memory_kind != "PROCEDURAL_EXPERIENCE":
        raise ValueError("procedural projection fields differ")
    if _json(payload) != content:
        raise ValueError("procedural projection content is not canonical JSON")
    case_ref = payload.get("case_ref")
    unhashed = dict(payload)
    unhashed.pop("case_ref")
    if type(case_ref) is not str or case_ref != _ref(unhashed):
        raise ValueError("procedural projection case_ref differs from content")
    epistemic_status = payload.get("epistemic_status")
    if epistemic_status not in _PROCEDURAL_EPISTEMIC_STATUSES:
        raise ValueError("procedural projection epistemic status differs")
    provenance = _mapping(payload.get("provenance"), label="procedure provenance")
    for name in (
        "event_ref",
        "choice_ref",
        "receipt_ref",
        "parent_state_ref",
        "child_state_ref",
    ):
        if type(provenance.get(name)) is not str or _SHA256_REF.fullmatch(
            provenance[name]
        ) is None:
            raise ValueError(f"procedural projection {name} is malformed")
    adjacent: set[str] = set()
    feedback = payload.get("feedback")
    if feedback is not None:
        feedback = _mapping(feedback, label="procedure feedback")
        target_episode_ref = feedback.get("target_episode_ref")
        if type(target_episode_ref) is not str or _SHA256_REF.fullmatch(
            target_episode_ref
        ) is None:
            raise ValueError("procedural feedback target episode is malformed")
        adjacent.add(target_episode_ref)
    return ProceduralProjectionMetadata(
        memory_kind="PROCEDURAL",
        epistemic_status=str(epistemic_status),
        adjacent_episode_refs=tuple(sorted(adjacent)),
    )


def record_procedural_outcome(
    state: MutableMapping[str, object],
    *,
    procedure_ids: Sequence[str],
    consequence: Mapping[str, float],
    choice_ref: str,
    receipt_ref: str,
    moving_origin_ordinal: int,
) -> None:
    """Update exact vector moments online without replay or scalarization."""

    if (
        type(procedure_ids) not in (list, tuple)
        or not procedure_ids
        or len(procedure_ids) > 17
        or len(procedure_ids) != len(set(procedure_ids))
        or any(type(item) is not str or not item for item in procedure_ids)
    ):
        raise ValueError("procedure_ids must be bounded unique text")
    for reference, label in ((choice_ref, "choice_ref"), (receipt_ref, "receipt_ref")):
        if type(reference) is not str or _SHA256_REF.fullmatch(reference) is None:
            raise ValueError(f"{label} must be a sha256 reference")
    if type(moving_origin_ordinal) is not int or moving_origin_ordinal < 0:
        raise ValueError("moving_origin_ordinal must be non-negative")
    if not consequence:
        raise ValueError("procedural outcome requires at least one dimension")
    observed: dict[str, float] = {}
    for name, value in consequence.items():
        if (
            type(name) is not str
            or not name
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
        ):
            raise ValueError("procedural consequence dimensions must be finite")
        observed[name] = float(value)

    raw_profiles = state.get(PROCEDURAL_OUTCOME_PROFILE_STATE_KEY, {})
    if type(raw_profiles) is not dict:
        raise ValueError("procedural outcome profiles must be an object")
    profiles = dict(raw_profiles)
    for procedure_id in procedure_ids:
        previous = profiles.get(procedure_id)
        if previous is None:
            count = 0
            mean = {name: 0.0 for name in observed}
            m2 = {name: 0.0 for name in observed}
            evidence_refs: list[str] = []
        else:
            if type(previous) is not dict:
                raise ValueError("procedural outcome profile must be an object")
            if (
                previous.get("contract") != PROCEDURAL_OUTCOME_PROFILE_CONTRACT
                or previous.get("procedure_id") != procedure_id
                or type(previous.get("count")) is not int
                or previous["count"] < 1
                or type(previous.get("mean")) is not dict
                or type(previous.get("m2")) is not dict
                or set(previous["mean"]) != set(observed)
                or set(previous["m2"]) != set(observed)
                or type(previous.get("recent_evidence_refs")) is not list
            ):
                raise ValueError("procedural outcome profile schema differs")
            count = previous["count"]
            mean = {name: float(previous["mean"][name]) for name in observed}
            m2 = {name: float(previous["m2"][name]) for name in observed}
            evidence_refs = list(previous["recent_evidence_refs"])
            if not all(math.isfinite(value) for value in (*mean.values(), *m2.values())):
                raise ValueError("procedural outcome profile contains non-finite moments")
        next_count = count + 1
        for name, value in observed.items():
            delta = value - mean[name]
            mean[name] += delta / next_count
            m2[name] += delta * (value - mean[name])
        profile = {
            "contract": PROCEDURAL_OUTCOME_PROFILE_CONTRACT,
            "procedure_id": procedure_id,
            "count": next_count,
            "mean": mean,
            "m2": m2,
            "latest_moving_origin_ordinal": moving_origin_ordinal,
            "latest_choice_ref": choice_ref,
            "latest_receipt_ref": receipt_ref,
            "recent_evidence_refs": [*evidence_refs, receipt_ref][-8:],
        }
        profiles[procedure_id] = profile
    state[PROCEDURAL_OUTCOME_PROFILE_STATE_KEY] = profiles


def procedural_outcome_profile(
    state: Mapping[str, object], procedure_id: str
) -> dict[str, object] | None:
    profiles = state.get(PROCEDURAL_OUTCOME_PROFILE_STATE_KEY, {})
    if type(profiles) is not dict:
        raise ValueError("procedural outcome profiles must be an object")
    value = profiles.get(procedure_id)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("procedural outcome profile must be an object")
    return dict(value)
