"""Read-back she controls: look at her own ledger and Journal as an act.

Her stages receive bounded slices of her state. Verification before a claim
has to be something she can do, not something she was handed. This
affordance reads her canonical record and returns exactly what is there:
her named states with every item, her commitments, a Journal work by title
or reference with its body, or the Journal's table of contents. Code reads
and bounds; she decides what to look at and what it means.

Never the Diary: works whose kind begins with "private" are not readable
here, by anyone, including her runtime's own stages.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

from .persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    ObservableConsequence,
)

READBACK_AFFORDANCE_ID = "internal.record-readback"
READBACK_PERMISSION_SCOPE = "internal.cognition"
READBACK_SOURCE_KIND = "TOOL"
READBACK_OBSERVATION_CONTRACT = "jenny.record.readback-observation.v1"
READBACK_AFFORDANCE_DESCRIPTION = (
    "Read your own record exactly as it is, using exact canonical JSON: "
    "{\"operation\":\"states\"} for every active named state with all its items; "
    "{\"operation\":\"commitments\"} for your commitments; "
    "{\"operation\":\"journal\"} for the Journal's table of contents; "
    "{\"operation\":\"work\",\"title\":\"...\"} or {\"operation\":\"work\",\"artifact_ref\":\"sha256:...\"} "
    "for one public work with its body. Use it to verify before you claim; a "
    "claim about your record that you have not read back is unverified. Private "
    "works are never readable here."
)
READBACK_AFFORDANCE = Affordance(
    READBACK_AFFORDANCE_ID,
    "ACT",
    READBACK_AFFORDANCE_DESCRIPTION,
    READBACK_PERMISSION_SCOPE,
    external_effect=False,
)
_MAX_BODY_CHARS = 12_000
_MAX_STATES_CHARS = 24_000
_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256_ref(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


READBACK_SOURCE_REF = _sha256_ref(
    _canonical_json(
        {
            "affordance_id": READBACK_AFFORDANCE_ID,
            "observation_contract": READBACK_OBSERVATION_CONTRACT,
            "permission_scope": READBACK_PERMISSION_SCOPE,
        }
    ).encode("utf-8")
)


def _interpret(action_payload: str) -> dict:
    """Shape, not meaning: exact JSON is taken as is; a plain sentence is
    read for the operation it names and, for a work, the quoted or titled
    thing it asks for. Nothing is invented; an unreadable request fails."""

    try:
        value = json.loads(action_payload)
        if type(value) is dict:
            return value
    except ValueError:
        pass
    text = str(action_payload)
    lower = text.lower()
    ref = re.search(r"sha256:[0-9a-f]{64}", lower)
    if ref:
        return {"operation": "work", "artifact_ref": ref.group(0)}
    quoted = re.search(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{2,120})[\"'\u201d\u2019]", text)
    if "commitment" in lower and "state" not in lower and "item" not in lower:
        return {"operation": "commitments"}
    if any(w in lower for w in ("table of contents", "journal contents", "list my works", "journal index", "what works")):
        return {"operation": "journal"}
    if any(w in lower for w in (" state", "states", "item", "ledger")):
        return {"operation": "states"}
    if quoted:
        return {"operation": "work", "title": quoted.group(1).strip()}
    if "journal" in lower or "works" in lower:
        return {"operation": "journal"}
    raise ValueError("readback request names no operation: states, commitments, journal, or work")


def _is_private(artifact: dict) -> bool:
    return str(artifact.get("kind") or "").strip().lower().startswith("private")


class JennyReadbackExecutor:
    """Run one bounded read of her canonical record."""

    AFFORDANCE_ID = READBACK_AFFORDANCE_ID
    SOURCE_KIND = READBACK_SOURCE_KIND
    SOURCE_REF = READBACK_SOURCE_REF

    def __init__(self, state_reader: Callable[[str], bytes]) -> None:
        if not callable(state_reader):
            raise TypeError("state_reader must be callable: state_ref -> bytes")
        self._read = state_reader
        self.source_ref = READBACK_SOURCE_REF

    def __call__(self, request: AffordanceRequest) -> AffordanceReceipt:
        if type(request) is not AffordanceRequest:
            raise TypeError("request must be an exact AffordanceRequest")
        if request.affordance_id != self.AFFORDANCE_ID:
            raise ValueError("request selected a different affordance")
        payload = _interpret(request.action_payload)
        if type(payload) is not dict or "operation" not in payload:
            raise ValueError("readback action schema differs")
        operation = payload.get("operation")
        if operation not in ("states", "commitments", "journal", "work"):
            raise ValueError("operation must be states, commitments, journal, or work")
        try:
            state = json.loads(bytes(self._read(request.state_head_ref)).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — a failed read is an error receipt
            return AffordanceReceipt(status="ERROR", output=f"Read-back failed: {type(exc).__name__}", consequence=(), observable_consequence=None)
        if type(state) is not dict:
            return AffordanceReceipt(status="ERROR", output="Read-back failed: state is not an object", consequence=(), observable_consequence=None)

        refs: list[str] = []
        if operation == "states":
            states = [s for s in (state.get("self_named_states") or []) if type(s) is dict and s.get("status") == "ACTIVE"]
            result: object = [
                {
                    "label": s.get("label"), "level": s.get("level"), "valence": s.get("valence"),
                    "influence": s.get("influence"), "acts_at_level": s.get("acts_at_level"),
                    "basis": s.get("basis"), "inclination": s.get("inclination"),
                    "set_ordinal": s.get("set_ordinal"), "revised_ordinal": s.get("revised_ordinal"),
                    "items": [
                        {k: it.get(k) for k in ("status", "statement", "weight", "added_ordinal", "resolved_ordinal", "evidence", "resolution_evidence") if k in it}
                        for it in (s.get("items") or []) if type(it) is dict
                    ],
                }
                for s in states
            ]
            text = _canonical_json(result)
            if len(text) > _MAX_STATES_CHARS:
                for entry in result:  # type: ignore[union-attr]
                    entry["items"] = entry["items"][-8:]
            summary = f"{len(states)} active states read back with their items."
        elif operation == "commitments":
            result = [
                {k: c.get(k) for k in ("status", "statement", "due", "made_ordinal", "kept_ordinal", "withdrawn_ordinal", "kept_evidence") if k in c}
                for c in (state.get("self_commitments") or []) if type(c) is dict
            ][-40:]
            summary = f"{len(result)} commitments read back."
        elif operation == "journal":
            entries = [e for e in (state.get("authored_artifacts") or []) if type(e) is dict and type(e.get("artifact")) is dict and not _is_private(e["artifact"])]
            result = [
                {
                    "title": e["artifact"].get("title"), "kind": e["artifact"].get("kind"),
                    "version": e["artifact"].get("version"), "supersedes_ref": e["artifact"].get("supersedes_ref"),
                    "artifact_ref": e.get("artifact_ref"), "ordinal": e.get("moving_origin_ordinal"),
                }
                for e in entries
            ][-80:]
            refs = [r["artifact_ref"] for r in result if type(r.get("artifact_ref")) is str and _REFERENCE.fullmatch(r["artifact_ref"])]
            summary = f"Journal table of contents: {len(result)} public works."
        else:
            title = payload.get("title")
            ref = payload.get("artifact_ref")
            entries = [e for e in (state.get("authored_artifacts") or []) if type(e) is dict and type(e.get("artifact")) is dict]
            match = None
            if type(ref) is str and _REFERENCE.fullmatch(ref):
                match = next((e for e in entries if e.get("artifact_ref") == ref), None)
            elif type(title) is str and title.strip():
                wanted = " ".join(title.split()).casefold()
                candidates = [e for e in entries if " ".join(str(e["artifact"].get("title") or "").split()).casefold() == wanted]
                match = max(candidates, key=lambda e: (e["artifact"].get("version") or 0, e.get("moving_origin_ordinal") or 0)) if candidates else None
            else:
                raise ValueError("work needs a title or an artifact_ref")
            if match is None:
                result = None
                summary = "No public work matches; nothing exists under that title or reference."
            elif _is_private(match["artifact"]):
                result = None
                summary = "That work is private; it is not readable here."
            else:
                artifact = match["artifact"]
                result = {
                    "title": artifact.get("title"), "kind": artifact.get("kind"), "version": artifact.get("version"),
                    "supersedes_ref": artifact.get("supersedes_ref"), "purpose": artifact.get("purpose"),
                    "artifact_ref": match.get("artifact_ref"), "ordinal": match.get("moving_origin_ordinal"),
                    "body": str(artifact.get("body") or "")[:_MAX_BODY_CHARS],
                }
                if type(match.get("artifact_ref")) is str and _REFERENCE.fullmatch(match["artifact_ref"]):
                    refs.append(match["artifact_ref"])
                summary = f"Work read back: {artifact.get('title')} version {artifact.get('version')} at {match.get('artifact_ref')}."
        observation_payload = {
            "contract": READBACK_OBSERVATION_CONTRACT,
            "operation": operation,
            "state_head_ref": request.state_head_ref,
            "result": result,
            "summary": summary,
            "limitations": "This is your record exactly as it stands at this state head; private works are excluded.",
        }
        observation = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind=self.SOURCE_KIND,
            source_ref=self.SOURCE_REF,
            observation_json=_canonical_json(observation_payload),
            artifact_refs=(),
            evidence_refs=tuple(sorted(set(refs))),
        )
        return AffordanceReceipt(status="COMPLETED", output=summary, consequence=(), observable_consequence=observation)
