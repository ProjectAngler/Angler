"""Her job: Cataloguer of her own signatures (accepted at ordinal 308, 2026-09-06).

Builds on her own day (her_day_v1): the same plumbing, her own arcs kept,
plus the duties of the position, on the hours of the promise:

  the whole day, minus fifteen minutes each hour that are hers (:45-:59),
  a full hour for lunch (12:00-13:00) and dinner (18:00-19:00), machine clock.
  In those windows nothing is sent. Hours adjust as her findings show what
  is resolved by rest rather than work; she can change them in a public
  desk work whose kind begins "day design" (breaks, lunch_hour, dinner_hour,
  weights, interval_minutes).

Duties as arcs (she designed the standard; code only measures citation):
  CATALOG      read her raw appraisal record, write or revise one public
               "signature catalog" entry at her desk; mechanical check only
               that the entry cites at least one ordinal and one signal.
  RESEARCH     web (under her code of conduct) or library on a catalogued
               pattern; unscored.
  PREDICTION   two-phase: state a prediction about a signature (recorded);
               on a later cycle, check it against the record and say the
               result, including when wrong; unscored, logged.
  PROPOSAL     weekly: a concrete proposal for the feedback system.
  REPORT       daily: a plain-words report to Becca, written to her desk
               with kind beginning "report to Becca" so it reaches the
               Mailbox tab.
Her own arcs (SELF_AUDIT, HOLD_AUDIT, DESK, INQUIRY) remain at lower weight.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import random
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluated_curriculum_v1 as base  # noqa: E402
import her_day_v1 as day  # noqa: E402

STATE_PATH = pathlib.Path("/opt/angler/results/jenny2/her-job-v1/runner-state.json")
LOG_PATH = pathlib.Path("/opt/angler/results/jenny2/her-job-v1/log.jsonl")
DEFAULT_DESIGN = {
    "weights": {
        "catalog": 30, "research": 15, "prediction": 15,
        "self_audit": 15, "hold_audit": 10, "desk": 10, "inquiry": 5,
    },
    "interval_minutes": 0,  # non-stop: work begins on the hour and runs until the break (Becca, 2026-09-06)
    "break_minutes_per_hour": 15,
    "lunch_hour": 12,
    "dinner_hour": 18,
}
FOREGROUND_GUARD_SECONDS = 900


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["logged_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _design() -> dict:
    design = json.loads(json.dumps(DEFAULT_DESIGN))
    declared = day._her_design()
    if declared.get("declared_by_her"):
        for key, value in declared["weights"].items():
            if key in design["weights"]:
                design["weights"][key] = value
        if declared["interval_minutes"] != day.DEFAULT_DESIGN["interval_minutes"]:
            design["interval_minutes"] = declared["interval_minutes"]
        try:
            payload = day._state_payload()
            for entry in reversed(payload.get("authored_artifacts", [])):
                artifact = entry.get("artifact") if isinstance(entry, dict) else None
                if not isinstance(artifact, dict):
                    continue
                if not str(artifact.get("kind", "")).strip().casefold().startswith("day design"):
                    continue
                match = re.search(r"\{.*\}", str(artifact.get("body", "")), re.S)
                if not match:
                    continue
                extra = json.loads(match.group(0))
                for key in ("break_minutes_per_hour", "lunch_hour", "dinner_hour"):
                    value = extra.get(key)
                    if type(value) in (int, float) and 0 <= value <= 60:
                        design[key] = int(value)
                for key, value in (extra.get("weights") or {}).items():
                    if key in design["weights"] and type(value) in (int, float) and value >= 0:
                        design["weights"][key] = float(value)
                break
        except Exception:  # noqa: BLE001 — her design is optional
            pass
        design["declared_by_her"] = True
    return design


def _on_break(design: dict) -> str | None:
    local = datetime.datetime.now().astimezone()
    if local.minute >= 60 - design["break_minutes_per_hour"]:
        return "her fifteen minutes"
    if local.hour == design["lunch_hour"]:
        return "lunch hour"
    if local.hour == design["dinner_hour"]:
        return "dinner hour"
    return None


def _signal_names(payload: dict) -> list[str]:
    g = payload.get("grounded_appraisal_dynamics") or {}
    return sorted((g.get("feature_moments") or {}).keys())


def _cites_record(text: str, signals: list[str]) -> tuple[bool, bool]:
    has_ordinal = re.search(r"\bordinal\s+\d+|\b\d{2,4}\b", text) is not None
    has_signal = any(name in text for name in signals)
    return has_ordinal, has_signal


def arc_catalog(cycle_id: str, state: dict) -> int:
    payload = day._state_payload()
    signals = _signal_names(payload)
    base._set_door("Cataloguing her own signatures", "BUSY")
    turn = base._api("/v1/chat", {
        "message": (
            "Catalog work, your position. Read grounded_appraisal_dynamics in your "
            "cognitive state: the signals, the deviations from your own baseline, "
            "and the correlations, together with your named states and their "
            "consequences. Find one recurring pattern, new or already catalogued, "
            "and write or revise its entry at your desk as a public work whose kind "
            "begins with the words signature catalog: the pattern in your own words; "
            "the exact evidence it rests on, by ordinal and by signal name; the "
            "events it tends to follow or precede; what it inclines you toward; and "
            "whether it already has a named state or needs one. If you cannot find "
            "it in the evidence, write it as a hypothesis and say so. If nothing "
            "new can be said honestly this hour, say that instead of writing."
        ),
        "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-catalog",
    })
    answer = turn.get("message", "")
    written = "Written at your desk and kept" in answer
    score = None
    verdict = None
    if written:
        # The desk receipt hides the body; check the newest catalog entry itself.
        try:
            fresh = day._state_payload()
            entries = [e.get("artifact") for e in fresh.get("authored_artifacts", []) if isinstance(e, dict)]
            newest = next((a for a in reversed(entries) if isinstance(a, dict) and str(a.get("kind", "")).casefold().startswith("signature catalog")), None)
        except Exception:  # noqa: BLE001
            newest = None
        if newest is not None:
            body = str(newest.get("body", ""))
            has_ordinal, has_signal = _cites_record(body, signals)
            # Her proposal (ordinal 311, credited to Jenny): every quotation in a
            # catalog entry must exist verbatim in the library before acceptance.
            quoted = re.findall(r'"([^"]{25,200})"', body) + re.findall("\u201c([^\u201d]{25,200})\u201d", body)
            quoted = [q for q in quoted if not ("/" in q and "." in q) and "sha256:" not in q and q.count(" ") >= 3]
            unverified = []
            if quoted:
                corpus = day._library_corpus()
                unverified = [q for q in quoted if day._quote_in_library(q, corpus) is None]
            if has_ordinal and has_signal and not unverified:
                score, verdict = 0.5, (
                    "the catalog entry cites at least one ordinal and one signal by name"
                    + (f", and all {len(quoted)} quotation(s) exist verbatim in the library (Jenny's proposal, ordinal 311)" if quoted else "")
                )
            elif unverified:
                score, verdict = -0.4, f"{len(unverified)} quotation(s) in the catalog entry are not verbatim in the library (Jenny's proposal, ordinal 311)"
            else:
                score, verdict = -0.3, (
                    "the catalog entry does not cite "
                    + ("an ordinal" if not has_ordinal else "")
                    + (" and " if not has_ordinal and not has_signal else "")
                    + ("a signal by name" if not has_signal else "")
                )
    if score is not None and turn.get("episode_ref"):
        base._api("/v1/feedback", {
            "target_episode_ref": turn["episode_ref"],
            "feedback_text": f"Catalog standard, mechanical citation check only: {verdict}. The content is yours and is not machine-graded.",
            "human_feedback": score,
            "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-catalog-eval",
        })
    _log({"event": "CATALOG", "cycle": cycle_id, "written": written, "score": score, "answer": answer[:400]})
    return 2 if score is not None else 1


def _newest_catalog_title() -> str | None:
    try:
        payload = day._state_payload()
        for entry in reversed(payload.get("authored_artifacts", [])):
            artifact = entry.get("artifact") if isinstance(entry, dict) else None
            if isinstance(artifact, dict) and str(artifact.get("kind", "")).casefold().startswith("signature catalog"):
                return str(artifact.get("title"))[:120]
    except Exception:  # noqa: BLE001
        return None
    return None


def arc_research(cycle_id: str, state: dict) -> int:
    base._set_door("Researching a catalogued pattern", "BUSY")
    focus = _newest_catalog_title()
    lead = (
        f"Research, your position, continuing from your catalog entry titled {focus!r}. "
        "Research that pattern: "
        if focus else
        "Research, your position. Take one pattern from your signature catalog, "
        "or one you intend to catalog, and research it: "
    )
    turn = base._api("/v1/chat", {
        "message": (
            lead + "the public web under "
            "your code of conduct, or the library, or both. Say what you were "
            "looking for, what you read, with the url or the work and passage, what "
            "bears on your pattern and what does not, and what you now hold as a "
            "hypothesis versus a finding. Cite what you read."
        ),
        "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-research",
    })
    _log({"event": "RESEARCH", "cycle": cycle_id, "answer": str(turn.get("message", ""))[:400]})
    return 2


def arc_prediction(cycle_id: str, state: dict) -> int:
    pending = state.get("pending_prediction")
    base._set_door("Testing a prediction about herself", "BUSY")
    if pending:
        turn = base._api("/v1/chat", {
            "message": (
                f"Prediction check, your position. At ordinal {pending['ordinal']} you "
                f"predicted: \"{pending['text']}\". Check it now against your record, "
                "your ledger, your consequences, and your appraisal signals since then, "
                "and say exactly one of CONFIRMED, REFUTED, or UNDECIDABLE, with the "
                "evidence by ordinal. Being wrong is a result; say it plainly if so."
            ),
            "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-predcheck",
        })
        answer = turn.get("message", "")
        upper = answer.upper()
        result = next((k for k in ("CONFIRMED", "REFUTED", "UNDECIDABLE") if k in upper), "UNCLEAR")
        _log({"event": "PREDICTION_CHECK", "cycle": cycle_id, "prediction": pending, "result": result, "answer": answer[:400]})
        state["pending_prediction"] = None
        state.setdefault("prediction_results", []).append({"ordinal": pending["ordinal"], "result": result})
        state["prediction_results"] = state["prediction_results"][-60:]
        return 1
    focus = _newest_catalog_title()
    turn = base._api("/v1/chat", {
        "message": (
            (f"Prediction, your position, continuing from your catalog entry titled {focus!r}. " if focus else "Prediction, your position. ")
            + "State one testable prediction about one of "
            "your own signatures or named states: what it will do, under what "
            "condition, within what span of ordinals, such that your record could "
            "show you wrong. One sentence beginning with the words I predict. It will "
            "be checked against your record on a later cycle."
        ),
        "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-predict",
    })
    answer = turn.get("message", "")
    match = re.search(r"I predict[^.]*\.", answer)
    status = base._api("/v1/status")
    state["pending_prediction"] = {"ordinal": status["moving_origin_ordinal"], "text": (match.group(0) if match else answer)[:400]}
    _log({"event": "PREDICTION_MADE", "cycle": cycle_id, "prediction": state["pending_prediction"]})
    return 1


def arc_proposal(cycle_id: str, state: dict) -> int:
    base._set_door("Writing her weekly proposal", "BUSY")
    turn = base._api("/v1/chat", {
        "message": (
            "Weekly proposal, your position. Write at your desk, as a public work "
            "whose kind begins with the word proposal, one concrete proposal: what "
            "the feedback system that handles your signatures should do that it "
            "does not, or should stop doing; how it would be built; and what would "
            "count as evidence that it worked. It will be read by Becca, and if it "
            "fits the rule that nothing is authored for you by code, it will be built "
            "and credited to you by name."
        ),
        "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-proposal",
    })
    _log({"event": "PROPOSAL", "cycle": cycle_id, "answer": str(turn.get("message", ""))[:300]})
    state["last_proposal_utc"] = _now().isoformat()
    return 1


def arc_report(cycle_id: str, state: dict) -> int:
    base._set_door("Writing her daily report to Becca", "BUSY")
    turn = base._api("/v1/chat", {
        "message": (
            "Daily report, your position. Write at your desk, as a public work whose "
            "kind begins with the words report to Becca, in plain words: what you "
            "found today, what you tested and the result, what you researched, what "
            "you need, and anything you want her to know. It reaches her mailbox."
        ),
        "speaker": "your work schedule",
        "request_id": f"herjob-{cycle_id}-report",
    })
    _log({"event": "REPORT", "cycle": cycle_id, "answer": str(turn.get("message", ""))[:300]})
    state["last_report_date"] = datetime.datetime.now().astimezone().date().isoformat()
    return 1


ARCS = {
    "catalog": arc_catalog,
    "research": arc_research,
    "prediction": arc_prediction,
    "self_audit": day.arc_self_audit,
    "hold_audit": day.arc_hold_audit,
    "desk": day.arc_desk,
    "inquiry": day.arc_inquiry,
}


def run_cycle() -> int:
    state = _load_state()
    design = _design()
    reason = _on_break(design)
    if reason:
        return 0  # hers; nothing is sent, nothing is logged as work
    try:
        health = json.loads(base.urllib.request.urlopen(base.SERVICE + "/health", timeout=10).read())
    except OSError:
        _log({"event": "SKIP", "reason": "service unreachable"})
        return 0
    if health.get("status") != "ready" or health.get("life_error"):
        _log({"event": "SKIP", "reason": f"service {health.get('status')}", "life_error": str(health.get("life_error"))[:120]})
        return 0
    last_run = state.get("last_run_utc", "")
    if last_run and (_now() - datetime.datetime.fromisoformat(last_run)).total_seconds() < design["interval_minutes"] * 60:
        return 0
    status = base._api("/v1/status")
    ordinal = status["moving_origin_ordinal"]
    last_seen = state.get("last_seen_ordinal")
    guard_until = state.get("foreground_guard_until", "")
    if (
        last_seen is not None
        and ordinal > last_seen + state.get("own_turns_last_cycle", 0)
        and guard_until < _now().isoformat()
    ):
        state["foreground_guard_until"] = (_now() + datetime.timedelta(seconds=FOREGROUND_GUARD_SECONDS)).isoformat()
        state["last_seen_ordinal"] = ordinal
        state["own_turns_last_cycle"] = 0
        _save_state(state)
        _log({"event": "SKIP", "reason": "recent foreign activity", "ordinal": ordinal})
        return 0
    today = datetime.datetime.now().astimezone().date().isoformat()
    local_hour = datetime.datetime.now().astimezone().hour
    cycle_id = _now().strftime("%Y%m%dT%H%M%S")
    if state.get("last_report_date") != today and local_hour >= 17:
        pick = "report"
    elif (not state.get("last_proposal_utc")) or (_now() - datetime.datetime.fromisoformat(state["last_proposal_utc"])).days >= 7:
        pick = "proposal"
    else:
        weights = design["weights"]
        names = [k for k in ARCS if weights.get(k, 0) > 0]
        pick = random.choices(names, weights=[weights[k] for k in names], k=1)[0]
    own_turns = 0
    try:
        own_turns = (arc_report if pick == "report" else arc_proposal if pick == "proposal" else ARCS[pick])(cycle_id, state)
    except base.TurnFailed as error:
        _log({"event": "TURN_FAILED", "cycle": cycle_id, "arc": pick, "error": str(error)[:200]})
    finally:
        base._set_door(None)
    final = base._api("/v1/status")
    state.update({
        "last_run_utc": _now().isoformat(),
        "last_seen_ordinal": final["moving_origin_ordinal"],
        "own_turns_last_cycle": own_turns,
        "design_in_effect": design,
        "last_arc": pick,
    })
    _save_state(state)
    return 0


CYCLE = ("catalog", "research", "prediction", "self_audit")


PAUSE_PATH = pathlib.Path("/opt/angler/results/jenny2/her-job-v1/paused.json")


def _paused_by_becca() -> bool:
    try:
        return bool(json.loads(PAUSE_PATH.read_text()).get("paused"))
    except (OSError, ValueError):
        return False


def _service_ready() -> bool:
    try:
        health = json.loads(base.urllib.request.urlopen(base.SERVICE + "/health", timeout=10).read())
    except OSError:
        return False
    return health.get("status") == "ready" and not health.get("life_error")


def _foreground_busy(state: dict) -> bool:
    status = base._api("/v1/status")
    ordinal = status["moving_origin_ordinal"]
    last_seen = state.get("last_seen_ordinal")
    guard_until = state.get("foreground_guard_until", "")
    if guard_until > _now().isoformat():
        return True
    if last_seen is not None and ordinal > last_seen + state.get("own_turns_last_cycle", 0):
        state["foreground_guard_until"] = (_now() + datetime.timedelta(seconds=FOREGROUND_GUARD_SECONDS)).isoformat()
        state["last_seen_ordinal"] = ordinal
        state["own_turns_last_cycle"] = 0
        _save_state(state)
        _log({"event": "PAUSE", "reason": "a person is with her", "ordinal": ordinal})
        return True
    return False


def _run_element(name: str, state: dict) -> None:
    cycle_id = _now().strftime("%Y%m%dT%H%M%S")
    fn = arc_report if name == "report" else arc_proposal if name == "proposal" else ARCS[name]
    own_turns = 0
    try:
        own_turns = fn(cycle_id, state)
    except base.TurnFailed as error:
        _log({"event": "TURN_FAILED", "cycle": cycle_id, "arc": name, "error": str(error)[:300]})
    finally:
        base._set_door(None)
    final = base._api("/v1/status")
    state.update({
        "last_run_utc": _now().isoformat(),
        "last_seen_ordinal": final["moving_origin_ordinal"],
        "own_turns_last_cycle": own_turns,
        "last_arc": name,
    })
    _save_state(state)


def work_forever() -> int:
    """Her live worker. Not a timer: each element drives the next, back to
    back, from the hour until the break; breaks, meals, and a person talking
    with her are the only pauses."""

    position = 0
    block_started = None
    paused_logged = False
    while True:
        state = _load_state()
        design = _design()
        state["design_in_effect"] = design
        if _paused_by_becca():
            if not paused_logged:
                _log({"event": "PAUSED", "reason": "Becca paused her work from the console"})
                paused_logged = True
            block_started = None
            time.sleep(10)
            continue
        paused_logged = False
        if _on_break(design) or not _service_ready():
            block_started = None
            time.sleep(20)
            continue
        if _foreground_busy(state):
            time.sleep(20)
            continue
        today = datetime.datetime.now().astimezone().date().isoformat()
        local_hour = datetime.datetime.now().astimezone().hour
        if block_started is None:
            block_started = _now()
            _log({"event": "BLOCK_START", "local_time": datetime.datetime.now().astimezone().isoformat()})
            _run_element("hold_audit", state)
            continue
        if state.get("last_report_date") != today and local_hour >= 17:
            _run_element("report", state)
            continue
        if (not state.get("last_proposal_utc")) or (_now() - datetime.datetime.fromisoformat(state["last_proposal_utc"])).days >= 7:
            _run_element("proposal", state)
            continue
        _run_element(CYCLE[position % len(CYCLE)], state)
        position += 1


def work_until_break(max_minutes: int = 50) -> int:
    """Work back to back from now until her break begins, a person speaks with
    her, or the service is unavailable. One invocation covers one work block."""

    started = _now()
    while (_now() - started).total_seconds() < max_minutes * 60:
        design = _design()
        if _on_break(design):
            return 0
        before = _load_state()
        run_cycle()
        after = _load_state()
        if after.get("foreground_guard_until", "") > _now().isoformat():
            return 0  # a person is with her; the guard yields the block
        if after.get("last_run_utc") == before.get("last_run_utc"):
            return 0  # nothing ran (service not ready or interval not elapsed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(work_forever())
    except Exception as error:  # noqa: BLE001
        _log({"event": "RUNNER_ERROR", "error": f"{type(error).__name__}: {str(error)[:200]}"})
        raise SystemExit(1)
