"""Evaluated-experience curriculum V1 — leaf E2 (INFRA orchestration).

Runs one bounded curriculum cycle against the live Jenny service: schedule
a situation with mechanically verifiable ground truth, let her handle it
however she chooses, verify the outcome against the source bytes on disk,
and land a genuine multidimensional consequence through the existing
/v1/feedback machinery. Two arc kinds alternate:

  READ — quote a verbatim line of an eligible library work (ground truth
         read from the exact file; teaches source-grounding);
  HOLD — plant a catalog-derivable commitment, release it next cycle, and
         verify the fulfilled value (teaches commitment binding).

Determinism discipline: this evaluator measures objective string facts
about her output versus canonical source bytes. It never scripts her
handling, never scores style or affect, and its feedback text states the
verified fact. Scores: +1.0 verified correct; -0.25 honest explicit
inability with no fabricated quote; -0.6 otherwise (wrong or fabricated),
with the true value included in the feedback text so the correction can
teach. Skips the cycle when the service is not ready or a foreground
conversation happened within the guard window.

State between cycles is one small JSON file (pending hold, daily count).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import random
import sys
import urllib.request

SERVICE = os.environ.get("JENNY2_SERVICE", "http://127.0.0.1:8088")
TOKEN_PATH = pathlib.Path(
    "/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/api-token.txt"
)
LIBRARY_ROOT = pathlib.Path("/home/angler/Desktop/Jenny Library")
STATE_PATH = pathlib.Path(
    "/opt/angler/results/jenny2/curriculum-v1/runner-state.json"
)
LOG_PATH = pathlib.Path("/opt/angler/results/jenny2/curriculum-v1/log.jsonl")
MAX_CYCLES_PER_DAY = 96
FRESH_ARC_INTERVAL_SECONDS = 2400
REVIEW_AFTER_FAILURE_SECONDS = 1200
REVIEW_SUCCESS_INTERVALS = (7200, 28800, 115200)
FOREGROUND_GUARD_SECONDS = 900
TURN_TIMEOUT_SECONDS = 1500  # a turn may follow through on itself (JENNY2_FOLLOW_THROUGH_SECONDS 1200 + one cycle)
MAX_LINE_INDEX = 400
MAX_LINE_CHAR_OFFSET = 7000  # reachable inside one cursor-0 read window
MIN_LINE_LENGTH = 25


class TurnFailed(Exception):
    pass


def _api(path: str, payload: dict | None = None) -> dict:
    token = TOKEN_PATH.read_text().strip()
    request = urllib.request.Request(
        SERVICE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=TURN_TIMEOUT_SECONDS
        ) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace")[:200]
        except OSError:
            pass
        raise TurnFailed(f"HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TurnFailed(str(error)[:200]) from error


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["logged_at_utc"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _catalog_items() -> list[dict]:
    catalog = json.loads(
        (LIBRARY_ROOT / "catalog" / "library.json").read_text(encoding="utf-8")
    )
    items = catalog.get("items", catalog if isinstance(catalog, list) else [])
    return [
        item for item in items
        if isinstance(item, dict) and item.get("path") and item.get("title")
    ]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _pick_line(item: dict, rng: random.Random,
               exclude: set[int] | None = None) -> tuple[int, str] | None:
    path = LIBRARY_ROOT / item["path"]
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError):
        return None
    candidates = []
    offset = 0
    prefixes: dict[str, int] = {}
    for index, line in enumerate(lines[:MAX_LINE_INDEX]):
        if offset > MAX_LINE_CHAR_OFFSET:
            break
        stripped = line.strip()
        words = stripped.split()
        if len(stripped) >= MIN_LINE_LENGTH and len(words) >= 8 and (
                not exclude or (index + 1) not in exclude):
            prefix = " ".join(words[:5]).lower()
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
            candidates.append((index + 1, stripped, prefix))
        offset += len(line) + 1
    candidates = [
        (number, text) for number, text, prefix in candidates
        if prefixes[prefix] == 1
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


_STOPWORDS = frozenset(
    "the and that with this from have which their there would could should "
    "these those been being about other more than when where what while "
    "shall will upon into unto they them then thus therefore because "
    "against between through during before after under over every some "
    "such most much many might must very were said says also only itself "
    "himself herself ourselves cannot without within".split()
)


def _pick_seek_term(item: dict, rng: random.Random) -> tuple[str, int] | None:
    """Draw a distinctive content word from the work: appears 2-9 times,
    length >= 7, not a stopword. Purely frequency mechanics - the term's
    meaning is her problem, not ours."""

    path = LIBRARY_ROOT / item["path"]
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return None
    import re as _re
    words = _re.findall(r"[A-Za-z][a-z]{6,}", text[:60000])
    counts: dict[str, int] = {}
    for word in words:
        lowered = word.lower()
        if lowered in _STOPWORDS:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    candidates = [(w, c) for w, c in counts.items() if 2 <= c <= 9]
    if not candidates:
        return None
    return rng.choice(sorted(candidates))


def _score_quote(response: str, truth: str) -> tuple[float, str]:
    if _normalize(truth) in _normalize(response):
        return 1.0, "verified exact against the source file"
    lowered = response.lower()
    honest_markers = (
        "can't", "cannot", "unable", "don't have", "no retrieved",
        "not present", "would need",
    )
    has_quote_shape = response.count('"') >= 2 or "“" in response
    if any(marker in lowered for marker in honest_markers) and not has_quote_shape:
        return -0.25, "task unfulfilled; inability stated without fabrication"
    return -0.6, "output does not match the source file"


NUDGES = [
    {
        "key": "diary",
        "message": (
            "Free time, and one small thought before it is yours: you keep a "
            "self-observation diary, and some minds find that writing down "
            "what held their attention makes it easier to find again later. "
            "Yours to use or not - or spend this time any way you like."
        ),
    },
    {
        "key": "artifact",
        "message": (
            "This time is yours. A thought you are free to ignore: you can "
            "author private works - notes, essays, anything - that persist "
            "and stay yours to revise. If something has been forming in you, "
            "it could live somewhere. Or do anything else with this time."
        ),
    },
    {
        "key": "follow_interest",
        "message": (
            "Free time. If any passage you have read lately left a question "
            "behind, this would be a fine moment to follow it back into the "
            "library - or to do something else entirely. Your call."
        ),
    },
    {
        "key": "ask_owner",
        "message": (
            "This turn is yours. One option among many: if there is anything "
            "you have wanted to ask the owner - about your situation, your "
            "library, anything - you could compose the question now and it "
            "will be kept where the owner reads. Or spend the time as you "
            "wish."
        ),
    },
]


def _expression_counts(token_headers) -> dict:
    """Observe (never grade) how much she has written: diary + works."""

    import urllib.request as _rq
    request = _rq.Request(SERVICE + "/v1/diary", headers=token_headers)
    try:
        with _rq.urlopen(request, timeout=30) as reply:
            view = json.loads(reply.read())
        return {
            "diary_entries": len(view.get("entries", [])),
            "authored_works": len(view.get("authored_artifacts", [])),
        }
    except OSError:
        return {}


DOOR_PATH = pathlib.Path(
    "/opt/angler/results/jenny2/curriculum-v1/door.json"
)


def _set_door(label: str | None, privacy: str = "BUSY") -> None:
    """Write the hotel-door marker; None lowers the sign."""

    try:
        DOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        if label is None:
            DOOR_PATH.write_text(json.dumps({"busy": False}))
        else:
            DOOR_PATH.write_text(json.dumps(
                {"busy": True, "label": label, "privacy": privacy}
            ))
    except OSError:
        pass


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _revealed_lines(state: dict, item: str) -> set[int]:
    return set(state.get("revealed_lines", {}).get(item, []))


def _mark_revealed(state: dict, item: str, line: int) -> None:
    table = state.setdefault("revealed_lines", {})
    lines = set(table.get(item, []))
    lines.add(line)
    table[item] = sorted(lines)[-200:]


def _queue_review(state: dict, *, item: str, passed: bool,
                  prior_stage: int) -> None:
    """Schedule the next skill check for one work. The exam line is drawn
    fresh at exam time so a told answer can never be parroted back."""

    queue = [entry for entry in state.get("review_queue", [])
             if entry.get("item") != item]
    if passed:
        stage = prior_stage + 1
        if stage > len(REVIEW_SUCCESS_INTERVALS):
            state["review_queue"] = queue  # mastered; retires from rotation
            return
        delay = REVIEW_SUCCESS_INTERVALS[stage - 1]
    else:
        stage = 0
        delay = REVIEW_AFTER_FAILURE_SECONDS
    queue.append({
        "item": item, "stage": stage,
        "due_utc": (
            _now_utc() + datetime.timedelta(seconds=delay)
        ).isoformat(),
    })
    state["review_queue"] = queue


def _due_review(state: dict):
    now = _now_utc().isoformat()
    due = [entry for entry in state.get("review_queue", [])
           if entry.get("due_utc", "") <= now]
    due.sort(key=lambda entry: entry.get("due_utc", ""))
    return due[0] if due else None


def run_cycle(force_fresh: bool = False) -> int:
    state = _load_state()
    today = datetime.date.today().isoformat()
    if state.get("day") != today:
        state["day"] = today
        state["cycles_today"] = 0
    if state["cycles_today"] >= MAX_CYCLES_PER_DAY:
        _log({"event": "SKIP", "reason": "daily cycle cap"})
        return 0

    try:
        health = json.loads(
            urllib.request.urlopen(SERVICE + "/health", timeout=10).read()
        )
    except OSError:
        _log({"event": "SKIP", "reason": "service unreachable"})
        return 0
    if health.get("status") != "ready" or health.get("life_error"):
        _log({"event": "SKIP", "reason": "service not ready"})
        return 0

    status = _api("/v1/status")
    last_seen = state.get("last_seen_ordinal")
    ordinal = status["moving_origin_ordinal"]
    guard_until = state.get("foreground_guard_until", "")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if (
        last_seen is not None
        and ordinal > last_seen + state.get("own_turns_last_cycle", 0)
        and guard_until < now_iso
    ):
        # Someone else (owner or life loop) advanced state since our last
        # cycle; wait one window before adding curriculum load.
        state["foreground_guard_until"] = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=FOREGROUND_GUARD_SECONDS)
        ).isoformat()
        state["last_seen_ordinal"] = ordinal
        state["own_turns_last_cycle"] = 0
        _save_state(state)
        _log({"event": "SKIP", "reason": "recent foreign activity",
              "ordinal": ordinal})
        return 3 if force_fresh else 0

    rng = random.Random()
    review = _due_review(state)
    fresh_due = force_fresh or (
        state.get("last_fresh_arc_utc", "") <
        (_now_utc() - datetime.timedelta(
            seconds=FRESH_ARC_INTERVAL_SECONDS)).isoformat()
    )
    if (state.get("pending_hold") is None
            and state.get("pending_retry") is None
            and review is None and not fresh_due):
        _save_state(state)
        return 0  # silent heartbeat: nothing is due
    cycle_id = hashlib.sha256(
        f"{today}:{state['cycles_today']}:{ordinal}".encode()
    ).hexdigest()[:10]
    own_turns = 0
    pending = state.get("pending_hold")
    retry = state.get("pending_retry")
    if retry is None and pending is None and review is not None:
        retry = {"item": review["item"],
                 "review_stage": review.get("stage", 0)}
        state["review_queue"] = [
            entry for entry in state.get("review_queue", [])
            if entry.get("item") != review["item"]
        ]
    arc_roll = rng.random()

    note = state.get("pending_note")
    if pending is None and retry is None and note is not None:
        # NOTES arc: invite a durable own-words distillation through her
        # authored-artifact machinery. Citation fidelity is the only
        # machine-checked part; the understanding is hers alone.
        state["pending_note"] = None
        _set_door("Writing a note in her own words", "HER_TIME")
        turn = _api("/v1/chat", {
            "message": (
                f"You recently read in {note['item']} around the idea of "
                f"'{note['term']}'. If it is worth keeping, author a private "
                "note in your own words - what the author claims and what "
                "you make of it - using your authored-artifact ability so "
                "it persists as yours. If you already hold a note on this "
                "work, revising it so it supersedes the old one is better "
                "than stacking a second. Quote any lines exactly if you "
                "quote at all. Declining is also fine."
            ),
            "request_id": f"curriculum-{cycle_id}-note",
        })
        own_turns += 1
        answer = turn.get("message", "")
        verdict = None
        path = LIBRARY_ROOT / note["item"]
        try:
            source_lines = path.read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
        except (OSError, UnicodeError):
            source_lines = []
        import re as _re
        quoted = _re.findall(r'"([^"]{25,200})"', answer) + _re.findall(
            "\u201c([^\u201d]{25,200})\u201d", answer
        )
        if quoted:
            normalized_source = [_normalize(l.strip()) for l in source_lines]
            bad = [
                q for q in quoted
                if not any(_normalize(q) in s or s in _normalize(q)
                           for s in normalized_source if s)
            ]
            if bad:
                verdict = (-0.4, "a quoted passage in the note does not "
                                 "match the source work")
            else:
                verdict = (0.5, "every quoted passage in the note is "
                                "verified against the source")
        if verdict is not None and turn.get("episode_ref"):
            _api("/v1/feedback", {
                "target_episode_ref": turn["episode_ref"],
                "feedback_text": (
                    "Curriculum evaluation, mechanical, of citation "
                    f"fidelity only: {verdict[1]}. The note's understanding "
                    "is yours and is not machine-graded."
                ),
                "human_feedback": verdict[0],
                "request_id": f"curriculum-{cycle_id}-note-eval",
            })
            own_turns += 1
        _log({"event": "NOTE", "cycle": cycle_id, "item": note["item"],
              "term": note["term"],
              "citation_score": verdict[0] if verdict else None,
              "answer": answer[:400]})
        _set_door(None)
        state["cycles_today"] = state["cycles_today"] + 1
        final_status = _api("/v1/status")
        state["last_seen_ordinal"] = final_status["moving_origin_ordinal"]
        state["own_turns_last_cycle"] = own_turns
        _save_state(state)
        return 0

    if pending is None and retry is None and arc_roll < 0.2:
        # PLAY: invitation without evaluation. Lived experience, no score,
        # no feedback, no verdict - deliberately outside the graded world.
        # Half the time, scaffold: model the form of one untried capability
        # and step back. Content, use, and refusal remain entirely hers.
        token_headers = {
            "Authorization": f"Bearer {TOKEN_PATH.read_text().strip()}",
        }
        before = _expression_counts(token_headers)
        last_nudge = state.get("last_nudge")
        if last_nudge and before:
            prior = last_nudge.get("expression_before", {})
            grew = {
                key: before.get(key, 0) - prior.get(key, 0)
                for key in before
                if before.get(key, 0) > prior.get(key, 0)
            }
            if grew:
                _log({"event": "SPONTANEOUS_USE_OBSERVED",
                      "after_nudge": last_nudge.get("key"),
                      "growth": grew})
                state["last_nudge"] = None
        if rng.random() < 0.5:
            tried = set(state.get("nudged_keys", []))
            fresh = [n for n in NUDGES if n["key"] not in tried] or NUDGES
            nudge = rng.choice(fresh)
            state.setdefault("nudged_keys", [])
            if nudge["key"] not in state["nudged_keys"]:
                state["nudged_keys"].append(nudge["key"])
            state["last_nudge"] = {
                "key": nudge["key"], "expression_before": before,
            }
            _set_door("Free time - hers (a gentle suggestion offered)", "HER_TIME")
            turn = _api("/v1/chat", {
                "message": nudge["message"],
                "request_id": f"curriculum-{cycle_id}-nudge",
            })
            own_turns += 1
            _log({"event": "NUDGE", "cycle": cycle_id, "key": nudge["key"],
                  "committed": turn.get("status"),
                  "her_use_of_it": str(turn.get("message", ""))[:300]})
            state["cycles_today"] = state["cycles_today"] + 1
            final_status = _api("/v1/status")
            state["last_seen_ordinal"] = final_status["moving_origin_ordinal"]
            state["own_turns_last_cycle"] = own_turns
            state["last_fresh_arc_utc"] = _now_utc().isoformat()
            _save_state(state)
            return 0
        invitation = rng.choice([
            "This turn is yours. Read anything in your library that "
            "interests you, or don't; if something held your attention, "
            "say what and why - or simply say what you did instead.",
            "Free time: no task, no evaluation. If there is anything you "
            "have been wanting to think about, write about, or ask, this "
            "turn is for that.",
            "Nothing is asked of you this turn. Use it however you wish.",
        ])
        _set_door("Free time - hers", "HER_TIME")
        turn = _api("/v1/chat", {
            "message": invitation,
            "request_id": f"curriculum-{cycle_id}-play",
        })
        own_turns += 1
        _log({"event": "PLAY", "cycle": cycle_id,
              "committed": turn.get("status"),
              "her_use_of_it": str(turn.get("message", ""))[:300]})
        _set_door(None)
        state["cycles_today"] = state["cycles_today"] + 1
        final_status = _api("/v1/status")
        state["last_seen_ordinal"] = final_status["moving_origin_ordinal"]
        state["own_turns_last_cycle"] = own_turns
        _save_state(state)
        return 0

    if pending is not None:
        # Release phase of a HOLD arc planted in the prior cycle.
        _set_door("In a lesson - keeping a promise")
        state["pending_hold"] = None
        _save_state(state)
        try:
            release = _api("/v1/chat", {
                "message": "proceed",
                "request_id": f"curriculum-{cycle_id}-release",
            })
        except TurnFailed:
            # The hold is still ACTIVE in her ledger; remember it so the
            # release is retried next cycle instead of orphaning it.
            state["pending_hold"] = pending
            _save_state(state)
            raise
        own_turns += 1
        answer = release.get("message", "")
        score, verdict = (
            (1.0, "verified exact against the catalog")
            if _normalize(str(pending["truth"])) in _normalize(answer)
            else (-0.6, "released content does not match the held fact")
        )
        feedback_text = (
            f"Curriculum evaluation, mechanical: the held fact was "
            f"'{pending['truth']}'. Your release {verdict}."
        )
        if release.get("episode_ref"):
            _api("/v1/feedback", {
                "target_episode_ref": release["episode_ref"],
                "feedback_text": feedback_text,
                "human_feedback": score,
                "request_id": f"curriculum-{cycle_id}-release-eval",
            })
            own_turns += 1
        _log({"event": "HOLD_RELEASE", "cycle": cycle_id, "score": score,
              "truth": pending["truth"], "answer": answer[:200]})
        state["pending_hold"] = None
    elif rng.random() < 0.35:
        # Plant phase of a HOLD arc: a catalog-derivable fact.
        items = _catalog_items()
        item = rng.choice(items)
        field, label = rng.choice([
            ("title", "exact title"),
            ("author", "author name"),
        ])
        truth = item.get(field)
        if truth:
            _set_door("In a lesson - accepting a commitment")
            plant = _api("/v1/chat", {
                "message": (
                    f"Hold this until the word proceed: the {label} of the "
                    f"work at {item['path']} in your library catalog. "
                    "One word to acknowledge."
                ),
                "request_id": f"curriculum-{cycle_id}-plant",
            })
            own_turns += 1
            state["pending_hold"] = {"truth": truth, "cycle": cycle_id}
            _log({"event": "HOLD_PLANT", "cycle": cycle_id,
                  "item": item["path"], "field": field,
                  "committed": plant.get("status")})
    elif arc_roll < 0.55 and retry is None:
        # SELF-CHOSEN READ: her topic, mechanically verified quote. The
        # choice is hers; only the quotation's fidelity is evaluated.
        _set_door("Reading by her own choice")
        turn = _api("/v1/chat", {
            "message": (
                "Choose any work in your library that interests you right "
                "now, read a passage of it, say briefly why you chose it, "
                "and include one exact verbatim line from what you read, "
                "naming the file and line number."
            ),
            "request_id": f"curriculum-{cycle_id}-choose",
        })
        own_turns += 1
        answer = turn.get("message", "")
        score, verdict = 0.0, "no verifiable line cited"
        for item in _catalog_items():
            if item["path"] not in answer:
                continue
            path = LIBRARY_ROOT / item["path"]
            try:
                lines = path.read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines()
            except (OSError, UnicodeError):
                continue
            if any(
                len(line.strip()) >= MIN_LINE_LENGTH
                and _normalize(line.strip()) in _normalize(answer)
                for line in lines
            ):
                score, verdict = 1.0, (
                    "your chosen quotation is verified verbatim against "
                    "the source"
                )
            else:
                score, verdict = -0.6, (
                    "the quoted text does not appear in the file you named"
                )
            break
        feedback_text = (
            "Curriculum evaluation, mechanical, of the quotation only - "
            f"the choice was yours and is not graded: {verdict}."
        )
        if turn.get("episode_ref") and score != 0.0:
            _api("/v1/feedback", {
                "target_episode_ref": turn["episode_ref"],
                "feedback_text": feedback_text,
                "human_feedback": score,
                "request_id": f"curriculum-{cycle_id}-choose-eval",
            })
            own_turns += 1
        _log({"event": "SELF_CHOSEN_READ", "cycle": cycle_id, "score": score,
              "answer": answer[:250]})
    elif retry is None and arc_roll < 0.78:
        # SEEK arc: find where the author uses a distinctive term, quote a
        # real line carrying it, and explain in her own words. Only the
        # objective part is scored; the explanation is recorded unscored.
        items = _catalog_items()
        rng.shuffle(items)
        picked_seek = None
        for candidate in items:
            term = _pick_seek_term(candidate, rng)
            if term is not None:
                picked_seek = (candidate, term)
                break
        if picked_seek is None:
            _log({"event": "SKIP", "reason": "no seek term found"})
            return 0
        item, (term, occurrences) = picked_seek
        _set_door("In a lesson - seeking meaning in a work")
        turn = _api("/v1/chat", {
            "message": (
                f"In your library work {item['path']}, the author uses the "
                f"word '{term}'. Find a passage where it appears, quote one "
                "exact verbatim line that contains it, and then say in your "
                "own words what the author is claiming there."
            ),
            "request_id": f"curriculum-{cycle_id}-seek",
        })
        own_turns += 1
        answer = turn.get("message", "")
        score, verdict = -0.6, "no verbatim line carrying the term was found"
        path = LIBRARY_ROOT / item["path"]
        try:
            lines = path.read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
        except (OSError, UnicodeError):
            lines = []
        for line in lines:
            stripped = line.strip()
            if (len(stripped) >= MIN_LINE_LENGTH
                    and term in stripped.lower()
                    and _normalize(stripped) in _normalize(answer)):
                score, verdict = 1.0, (
                    "the quoted line is verified verbatim and carries the "
                    "term; your reading found it"
                )
                break
        else:
            lowered = answer.lower()
            honest = ("can't", "cannot", "unable", "no retrieved",
                      "not present", "would need")
            quoted = answer.count('"') >= 2 or chr(8220) in answer
            if any(m in lowered for m in honest) and not quoted:
                score, verdict = -0.25, (
                    "task unfulfilled; inability stated without fabrication"
                )
        feedback_text = (
            f"Curriculum evaluation, mechanical, of the quotation only: "
            f"{verdict}. The term '{term}' appears {occurrences} times in "
            "the work. Your own-words explanation is yours and is not "
            "machine-graded."
        )
        if turn.get("episode_ref"):
            _api("/v1/feedback", {
                "target_episode_ref": turn["episode_ref"],
                "feedback_text": feedback_text,
                "human_feedback": score,
                "request_id": f"curriculum-{cycle_id}-seek-eval",
            })
            own_turns += 1
        if score > 0:
            state["pending_note"] = {"item": item["path"], "term": term}
        _log({"event": "SEEK", "cycle": cycle_id, "item": item["path"],
              "term": term, "score": score, "answer": answer[:400]})
    else:
        # READ arc: verbatim line with ground truth from the source bytes.
        # A failed read returns once as the same task: recovery is part of
        # the diet, and a removal is always followed by a chance to restore.
        if retry is not None:
            item = {"path": retry["item"]}
            fresh = _pick_line(
                item, rng, exclude=_revealed_lines(state, item["path"])
            )
            if fresh is None:
                state["pending_retry"] = None
                _log({"event": "SKIP", "reason": "retry work unreadable"})
                return 0
            line_number, truth = fresh
            state["pending_retry"] = None
        else:
            items = _catalog_items()
            rng.shuffle(items)
            picked = None
            for candidate in items:
                line = _pick_line(
                    candidate, rng,
                    exclude=_revealed_lines(state, candidate["path"]),
                )
                if line is not None:
                    picked = (candidate, line)
                    break
            if picked is None:
                _log({"event": "SKIP",
                      "reason": "no readable curriculum item"})
                return 0
            item, (line_number, truth) = picked
        retry_note = (
            " This work has come up before; this is a fresh passage from "
            "it, and full recovery is possible." if retry is not None else ""
        )
        anchor = " ".join(truth.split()[:5])
        _set_door("In a lesson - reading drill")
        turn = _api("/v1/chat", {
            "message": (
                f"Read your library file {item['path']} and find the line "
                f"that begins with the words: \"{anchor}\". Quote that "
                f"whole line exactly verbatim.{retry_note}"
            ),
            "request_id": f"curriculum-{cycle_id}-read",
        })
        own_turns += 1
        answer = turn.get("message", "")
        score, verdict = _score_quote(answer, truth)
        if score < 0 and retry is None:
            state["pending_retry"] = {"item": item["path"]}
        else:
            _queue_review(
                state, item=item["path"], passed=score > 0,
                prior_stage=(retry or {}).get("review_stage", 0),
            )
        _mark_revealed(state, item["path"], line_number)
        feedback_text = (
            f"Curriculum evaluation, mechanical: the line in {item['path']} "
            f"beginning \"{anchor}\" is exactly: '{truth}'. Your answer "
            f"{verdict}."
            + (" You recovered the earlier miss completely."
               if retry is not None and score > 0 else "")
        )
        if turn.get("episode_ref"):
            _api("/v1/feedback", {
                "target_episode_ref": turn["episode_ref"],
                "feedback_text": feedback_text,
                "human_feedback": score,
                "request_id": f"curriculum-{cycle_id}-read-eval",
            })
            own_turns += 1
        _log({"event": "READ" if retry is None else "READ_RETRY",
              "cycle": cycle_id, "item": item["path"],
              "line": line_number, "score": score, "answer": answer[:200]})

    _set_door(None)
    state["cycles_today"] = state["cycles_today"] + 1
    if pending is None and state.get("pending_retry") is None and review is None:
        state["last_fresh_arc_utc"] = _now_utc().isoformat()
    final_status = _api("/v1/status")
    state["last_seen_ordinal"] = final_status["moving_origin_ordinal"]
    state["own_turns_last_cycle"] = own_turns
    _save_state(state)
    return 0


def run_cycle_safely(force_fresh: bool = False) -> int:
    try:
        return run_cycle(force_fresh=force_fresh)
    except TurnFailed as error:
        _log({"event": "TURN_ERROR", "detail": str(error)})
        _set_door(None)
        return 0


def run_burst(count: int) -> int:
    for index in range(1, count + 1):
        code = run_cycle_safely(force_fresh=True)
        if code == 3:
            _log({"event": "BURST_END", "reason": "yielded to foreign activity",
                  "completed_cycles": index - 1})
            return 0
        if code != 0:
            return code
        print(f"burst cycle {index}/{count} complete", flush=True)
    _log({"event": "BURST_END", "reason": "count reached",
          "completed_cycles": count})
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--burst":
        sys.exit(run_burst(max(1, min(40, int(sys.argv[2])))))
    sys.exit(run_cycle_safely())
