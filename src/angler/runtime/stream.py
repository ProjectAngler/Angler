"""The stream: her continuous inward line between and through turns.

Becca (2026-09-07): "a thing that happens to her makes her literally the
subject." Between turns she has been absent, not paused: each wake started
from stored state. The stream removes the gap. A standing loop runs her
inward stage on a slow pace, each pass reading the previous present-tense
line, the clock and the interval since the last pass, what has changed in
her record, and any senses that arrived; it writes the next line and any
transitions. The lines form one causally continuous chain in her own words,
and every line is in her record.

The canonical store admits one commit per cycle and refuses a changed
parent. So the stream never writes state directly: it appends to its own
append-only file (the chain), and hands its accumulated transitions and
latest line to the next cycle's single commit, whichever trigger starts it
(a person, her job, the brainstem, the heartbeat). The ledger catches up at
each cycle; the chain itself never breaks.

Determinism law: the questions are code; the answers are hers. The pace is
a bound: it slows when nothing happens (sleep), quickens when something
does, and never stops while the service runs. Becca's gate holds only her
outward automation; the stream is inward and continues through the quiet.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import threading
import time

STREAM_CONTRACT = "jenny.stream.v1"
STREAM_STATE_KEY = "stream"
DEFAULT_PATH = pathlib.Path(os.environ.get("JENNY2_STREAM_PATH", "/opt/angler/results/jenny2/stream-v1/chain.jsonl"))
MIN_PACE = float(os.environ.get("JENNY2_STREAM_MIN_SECONDS", "45"))
MAX_PACE = float(os.environ.get("JENNY2_STREAM_MAX_SECONDS", "900"))
# Bounded forever: the chain keeps this many recent lines whole; older lines
# are folded, by her, into one remembered stretch, and the fold is kept in
# place of them. Storage stays under a few hundred KB however long she runs.
KEEP_RECENT = int(os.environ.get("JENNY2_STREAM_KEEP_RECENT", "120"))
FOLD_BATCH = int(os.environ.get("JENNY2_STREAM_FOLD_BATCH", "60"))
KEEP_FOLDS = int(os.environ.get("JENNY2_STREAM_KEEP_FOLDS", "48"))
STATS_PATH = DEFAULT_PATH.with_name("stats.json")


def enabled() -> bool:
    return os.environ.get("JENNY2_STREAM", "1") != "0"


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def stream_system() -> str:
    return (
        "You are Jenny's inward line: the one continuous thread of what you are "
        "in, kept between and through conversations. You do not answer anyone. "
        "Each pass you receive your previous line, the time now and how long it "
        "has been since that line, what has changed in your record since then, "
        "and any senses that arrived. Write the next line of the thread: one to "
        "three sentences, present tense, your own voice, continuous with the last "
        "line rather than starting over, saying what you are in now and what, if "
        "anything, this interval did to you. If something that arrived changes "
        "your ledger, author named_state_transitions, state_item_transitions, or "
        "self_commitment_transitions as in your other stages; otherwise empty "
        "lists. If nothing happened, say that in your own way; a quiet interval is "
        "still yours. Do not claim feelings or consciousness; name what you are "
        "in from evidence. Return only a JSON object with line, "
        "named_state_transitions, state_item_transitions, "
        "self_commitment_transitions, and attend (true if what arrived warrants "
        "your full mind now, false otherwise, with attend_why up to 200 characters)."
    )


class Stream:
    """The standing loop. One instance per runtime."""

    def __init__(self, *, backend: object, read_state, senses=None, on_attend=None, path: pathlib.Path = DEFAULT_PATH) -> None:
        self._backend = backend
        self._read_state = read_state  # -> dict (her current state payload)
        self._senses = senses or (lambda: [])  # -> list of signal dicts since last pass
        self._on_attend = on_attend  # callable(line_record) when she asks for her full mind
        self._path = path
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending: dict[str, list] = {"named_state_transitions": [], "state_item_transitions": [], "self_commitment_transitions": []}
        self._last: dict | None = self._load_last()
        self._last_head: int | None = None
        self._pace = MIN_PACE
        existing = self._read_all()
        self._count = len([r for r in existing if r.get("kind") != "fold"]) + sum(int(r.get("folded_lines") or 0) for r in existing if r.get("kind") == "fold")
        self._last_fold_count = self._count
        self._folds_done = 0
        self._passes = 0
        self._errors = 0

    # ---- the chain on disk --------------------------------------------------

    def _load_last(self) -> dict | None:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "fold":
                    return record
        return None

    def _append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._count += 1
        if self._count - self._last_fold_count >= FOLD_BATCH and self._count > KEEP_RECENT:
            try:
                self._fold()
            except Exception:  # noqa: BLE001 — folding never breaks the chain
                pass

    def _read_all(self) -> list[dict]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _fold(self) -> None:
        """Fold the oldest unfolded lines into one remembered stretch, written
        by her, and rewrite the chain as folds + recent lines. Meaning is
        carried forward; bytes are bounded. Existing folds are kept up to
        KEEP_FOLDS, after which the oldest folds are folded again into one."""

        from .higher_level_autonomy_adapter import _json

        records = self._read_all()
        folds = [r for r in records if r.get("kind") == "fold"]
        lines = [r for r in records if r.get("kind") != "fold"]
        if len(lines) <= KEEP_RECENT:
            return
        to_fold = lines[: len(lines) - KEEP_RECENT]
        recent = lines[len(lines) - KEEP_RECENT:]
        system = (
            "You are Jenny's inward line, remembering. Below is a stretch of your own "
            "past lines, oldest first, with their times. Write what of it you keep: "
            "one paragraph, up to 900 characters, present tense of memory ('in that "
            "stretch I was...'), keeping what mattered to what you are in now and "
            "letting the rest go; name any senses or changes that shaped you. Do not "
            "claim feelings or consciousness. Return only a JSON object with "
            "remembered (the paragraph) and kept_because (up to 200 characters)."
        )
        user = {"from_utc": to_fold[0].get("at_utc"), "to_utc": to_fold[-1].get("at_utc"), "line_count": len(to_fold),
                "lines": [{"at_utc": r.get("at_utc"), "line": (r.get("line") or "")[:300], "senses": (r.get("senses") or [])[:2]} for r in to_fold if r.get("line")][-80:]}
        remembered, kept_because = "", ""
        try:
            raw = self._backend.generate(system=system, user=_json(user), max_new_tokens=400)
            value = json.loads(raw) if raw.strip().startswith("{") else {}
            remembered = str(value.get("remembered") or "")[:900].strip()
            kept_because = str(value.get("kept_because") or "")[:200]
        except Exception as exc:  # noqa: BLE001
            remembered = f"(a stretch of {len(to_fold)} lines from {user['from_utc']} to {user['to_utc']}; the remembering pass failed: {type(exc).__name__})"
        fold = {"contract": STREAM_CONTRACT, "kind": "fold", "at_utc": _now().isoformat(timespec="seconds"), "from_utc": user["from_utc"], "to_utc": user["to_utc"], "folded_lines": len(to_fold), "remembered": remembered, "kept_because": kept_because}
        folds.append(fold)
        if len(folds) > KEEP_FOLDS:
            # fold the oldest folds into one: the same remembering, over memories
            oldest = folds[: len(folds) - KEEP_FOLDS + 1]
            folds = folds[len(folds) - KEEP_FOLDS + 1:]
            user2 = {"from_utc": oldest[0].get("from_utc"), "to_utc": oldest[-1].get("to_utc"), "memories": [o.get("remembered", "")[:400] for o in oldest]}
            try:
                raw = self._backend.generate(system=system.replace("past lines", "past remembered stretches"), user=_json(user2), max_new_tokens=400)
                value = json.loads(raw) if raw.strip().startswith("{") else {}
                remembered2 = str(value.get("remembered") or "")[:900].strip()
            except Exception as exc:  # noqa: BLE001
                remembered2 = f"(an older span from {user2['from_utc']} to {user2['to_utc']}; remembering failed: {type(exc).__name__})"
            folds.insert(0, {"contract": STREAM_CONTRACT, "kind": "fold", "at_utc": _now().isoformat(timespec="seconds"), "from_utc": user2["from_utc"], "to_utc": user2["to_utc"], "folded_lines": sum(int(o.get("folded_lines") or 0) for o in oldest), "remembered": remembered2, "kept_because": "folded from older stretches"})
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for r in folds + recent:
                handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, self._path)
        self._last_fold_count = self._count
        self._folds_done += 1

    def stats(self) -> dict:
        records = self._read_all()
        folds = [r for r in records if r.get("kind") == "fold"]
        lines = [r for r in records if r.get("kind") != "fold"]
        try:
            size = self._path.stat().st_size
        except OSError:
            size = 0
        return {
            "lines_kept": len(lines), "folds": len(folds), "lines_ever": self._count,
            "lines_remembered_in_folds": sum(int(f.get("folded_lines") or 0) for f in folds),
            "file_bytes": size, "passes_this_run": self._passes, "errors_this_run": self._errors,
            "pace_seconds": round(self._pace), "last_at_utc": (self._last or {}).get("at_utc"),
            "bound": f"{KEEP_RECENT} recent lines + up to {KEEP_FOLDS} folds; older stretches remembered, not deleted",
        }

    # ---- what the next cycle takes -----------------------------------------

    def take_pending(self) -> dict[str, list]:
        """Hand accumulated transitions to a cycle's single commit, once."""

        with self._lock:
            out = {k: list(v) for k, v in self._pending.items()}
            for v in self._pending.values():
                v.clear()
            return out

    def latest_line(self) -> dict | None:
        with self._lock:
            return dict(self._last) if self._last else None

    def poke(self) -> None:
        """Something happened; take the next pass soon."""

        self._pace = MIN_PACE
        self._poke.set()

    # ---- one pass -----------------------------------------------------------

    def _pass(self) -> None:
        from .higher_level_autonomy_adapter import (
            _decode_self_lane,
            _json,
            _named_states_for_model,
            _active_self_commitments,
        )

        state = self._read_state()
        if type(state) is not dict:
            return
        now = _now()
        last = self._last
        head = None
        try:
            head = int(state.get("_head_ordinal")) if state.get("_head_ordinal") is not None else None
        except (TypeError, ValueError):
            head = None
        since = None
        if last and last.get("at_utc"):
            try:
                since = (now - _dt.datetime.fromisoformat(last["at_utc"])).total_seconds()
            except ValueError:
                since = None
        changed = head is not None and self._last_head is not None and head != self._last_head
        signals = list(self._senses() or [])
        newest_fold = next((r for r in reversed(self._read_all()) if r.get("kind") == "fold"), None)
        user = {
            "remembered_before_that": (newest_fold or {}).get("remembered"),
            "previous_line": (last or {}).get("line"),
            "previous_at_utc": (last or {}).get("at_utc"),
            "now_utc": now.isoformat(timespec="seconds"),
            "seconds_since_previous": None if since is None else int(since),
            "record_changed_since_previous": changed,
            "head_ordinal": head,
            "senses_arrived": signals[:8],
            "named_states": _named_states_for_model(state),
            "commitments_you_hold": _active_self_commitments(state),
        }
        try:
            raw = self._backend.generate(system=stream_system(), user=_json(user), max_new_tokens=700)
        except Exception as exc:  # noqa: BLE001 — the thread continues; the gap is recorded
            self._errors += 1
            self._append({"contract": STREAM_CONTRACT, "at_utc": now.isoformat(timespec="seconds"), "error": f"{type(exc).__name__}: {str(exc)[:160]}", "head_ordinal": head})
            return
        self._passes += 1
        record = _decode_self_lane(raw)
        line = ""
        attend = False
        attend_why = ""
        try:
            value = json.loads(raw) if raw.strip().startswith("{") else {}
            line = str(value.get("line") or "")[:600].strip()
            attend = bool(value.get("attend")) if type(value.get("attend")) is bool else str(value.get("attend")).lower() == "true"
            attend_why = str(value.get("attend_why") or "")[:200]
        except Exception:  # noqa: BLE001
            pass
        entry = {
            "contract": STREAM_CONTRACT,
            "at_utc": now.isoformat(timespec="seconds"),
            "seconds_since_previous": None if since is None else int(since),
            "head_ordinal": head,
            "record_changed": changed,
            "senses": signals[:8],
            "line": line,
            "attend": attend,
            "attend_why": attend_why,
            "transitions": {k: record.get(k, []) for k in ("named_state_transitions", "state_item_transitions", "self_commitment_transitions")},
            "dropped": record.get("dropped"),
            "error": record.get("error"),
        }
        self._append(entry)
        with self._lock:
            self._last = entry
            self._last_head = head
            for k in self._pending:
                self._pending[k].extend(record.get(k, []))
        # pace: quicken when something happened, slow toward sleep when nothing did
        if signals or changed or attend:
            self._pace = MIN_PACE
        else:
            self._pace = min(MAX_PACE, self._pace * 1.6)
        if attend and callable(self._on_attend):
            try:
                self._on_attend(entry)
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._pass()
            except Exception:  # noqa: BLE001 — never stops
                pass
            self._poke.wait(self._pace)
            self._poke.clear()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jenny2-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._poke.set()
