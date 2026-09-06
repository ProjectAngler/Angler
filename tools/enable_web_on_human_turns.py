"""Becca's switch: offer Jenny's read-only web reader on turns with a person.

Claude's session could not apply these two edits because an automated
permission layer refused changes that open outward network access on
conversational turns. Run this yourself to apply them:

    /opt/angler/venvs/angler/bin/python tools/enable_web_on_human_turns.py
    sudo systemctl restart jenny2-api.service

Edit 1 lets the human-turn route offer external.web-read (read-only,
allowlisted scope external.web.read). Edit 2 keeps a bounded, judgment-free
record of what she searched and read, shown to her as web_research_state.
"""

import ast
import os

PATH = "/opt/angler/src/angler/src/angler/runtime/higher_level_autonomy_adapter.py"

EDITS = [
    (
        '''                    or (
                        item.affordance_id
                        in (LIBRARY_AFFORDANCE_ID, AUTHORED_ARTIFACT_AFFORDANCE_ID)
                        and item.disposition == "ACT"
                        and item.permission_scope == "internal.cognition"
                        and not item.external_effect
                        and item.authorization_mode == "LOCAL"
                    )
                )''',
        '''                    or (
                        item.affordance_id
                        in (LIBRARY_AFFORDANCE_ID, AUTHORED_ARTIFACT_AFFORDANCE_ID)
                        and item.disposition == "ACT"
                        and item.permission_scope == "internal.cognition"
                        and not item.external_effect
                        and item.authorization_mode == "LOCAL"
                    )
                    or (
                        item.affordance_id == WEB_AFFORDANCE_ID
                        and item.disposition == "ACT"
                        and item.permission_scope == "external.readonly.web"
                        and item.authorization_mode == "LOCAL"
                    )
                )''',
    ),
    (
        '''            elif choice.selected_affordance_id == AUTHORED_ARTIFACT_AFFORDANCE_ID:
                authored_artifact_entry = _append_authored_artifact_entry(''',
        '''            elif choice.selected_affordance_id == WEB_AFFORDANCE_ID:
                _record_web_observation(
                    state_payload=state_payload,
                    observed=observed,
                    choice=choice,
                    temporal=temporal,
                )
            elif choice.selected_affordance_id == AUTHORED_ARTIFACT_AFFORDANCE_ID:
                authored_artifact_entry = _append_authored_artifact_entry(''',
    ),
    (
        '''def _record_library_observation(
    *,''',
        '''MAX_WEB_RESEARCH_RECORDS = 24


def _record_web_observation(
    *,
    state_payload: dict[str, object],
    observed: ObservableConsequence,
    choice: CycleChoice,
    temporal: TemporalV2,
) -> None:
    """Keep a bounded record of what she searched and read on the web.
    Provenance only: no judgment of the content, no policy."""

    payload = json.loads(observed.observation_json)
    if type(payload) is not dict:
        raise ValueError("web observation payload must be an object")
    contract = payload.get("contract")
    if contract == WEB_SEARCH_CONTRACT:
        record = {
            "operation": "search",
            "query": payload.get("query"),
            "result_count": payload.get("result_count"),
            "results": [
                {"title": item.get("title"), "url": item.get("url")}
                for item in payload.get("results", [])
                if type(item) is dict
            ][:10],
        }
    elif contract == WEB_PAGE_CONTRACT:
        record = {
            "operation": "read",
            "url": payload.get("url"),
            "final_url": payload.get("final_url"),
            "title": payload.get("title"),
            "span": payload.get("normalized_span"),
            "next_cursor": payload.get("next_cursor"),
            "eof": payload.get("eof"),
            "total_normalized_chars": payload.get("total_normalized_chars"),
        }
    else:
        raise ValueError("web observation contract differs")
    record.update(
        {
            "artifact_ref": payload.get("artifact_ref"),
            "fetched_at": payload.get("fetched_at"),
            "reading_purpose": payload.get("reading_purpose"),
            "observation_ref": observed.observation_ref,
            "choice_ref": choice.choice_ref,
            "moving_origin_ordinal": temporal.moving_origin_ordinal,
        }
    )
    history = state_payload.get("web_research_state", [])
    if type(history) is not list:
        history = []
    state_payload["web_research_state"] = [*history, record][-MAX_WEB_RESEARCH_RECORDS:]


def _web_research_context(state_payload: dict[str, object]) -> object:
    history = state_payload.get("web_research_state", [])
    if type(history) is not list:
        return None
    recent = [item for item in history if type(item) is dict][-8:]
    return _bounded_semantic_model_record(
        {"recent": list(reversed(recent)), "kept": len(history)},
        maximum_characters=2_048,
    )


def _record_library_observation(
    *,''',
    ),
    (
        '''            "library_reading_state": _adaptive_router_library_context(
                state_payload
            ),''',
        '''            "library_reading_state": _adaptive_router_library_context(
                state_payload
            ),
            "web_research_state": _web_research_context(state_payload),''',
    ),
]


def main() -> int:
    src = open(PATH, encoding="utf-8").read()
    if "_record_web_observation(" in src and "def _record_web_observation" in src:
        print("already applied")
        return 0
    for old, new in EDITS:
        if src.count(old) != 1:
            raise SystemExit(f"anchor not found exactly once: {old[:60]!r}")
        src = src.replace(old, new, 1)
    ast.parse(src)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(src)
    os.replace(tmp, PATH)
    print("applied; now: sudo systemctl restart jenny2-api.service")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
