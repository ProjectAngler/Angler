"""Jenny's brainstem, v1: a low-energy CPU watcher that notices and wakes.

Becca's design (2026-09-06): the brain's cheap part is sensing, not thinking.
This process runs continuously on the CPU with a small model, never touches
her GPUs, never acts or answers for her, and can do exactly one thing: wake
her full mind with a signal and its stated reason.

Determinism law: senses are code (what changed, and where). The judgment
"does this warrant waking her" is model-authored from HER wake criteria: the
newest public Journal work titled "What should wake me". Until she writes one,
DEFAULT_CRITERIA is in force, and every wake says so. Every signal, judgment,
and wake is logged with its cause. Bounds are time (a cooldown, a sense
interval), not counts. It never wakes her while a person is with her, while
her job cycle is paused, or while a turn is in flight.

Senses in v1:
  stranded_undertaking  her last follow-through judgment says the turn was
                        not complete, nothing has happened since, and enough
                        time has passed that her runtime is no longer
                        continuing it (a restart, a failed step). This is the
                        sense that makes work flow: she never stays stuck.
  quarantined_effect    a pending effect her runtime had to set aside.
  watched_path_change   a file or directory she asked to be watched changed
                        (list in her criteria work under "Watch:" lines, or
                        none).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import sqlite3
import sys
import time
import urllib.error
import urllib.request

STATE_DB = pathlib.Path(
    os.environ.get(
        "JENNY2_STATE_DB",
        "/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/jenny2.sqlite3",
    )
)
QUARANTINE_DIR = STATE_DB.parent / "quarantined-effects"
PAUSE_PATH = pathlib.Path("/opt/angler/results/jenny2/her-job-v1/paused.json")
SERVICE = os.environ.get("JENNY2_SERVICE", "http://192.168.137.5:8088")
LOG_DIR = pathlib.Path("/opt/angler/results/jenny2/brainstem-v1")
LOG_PATH = LOG_DIR / "log.jsonl"
MODEL_PATH = os.environ.get("JENNY2_BRAINSTEM_MODEL", "/opt/angler/models/Qwen3-1.7B")
SENSE_INTERVAL_SECONDS = float(os.environ.get("JENNY2_BRAINSTEM_SENSE_SECONDS", "20"))
WAKE_COOLDOWN_SECONDS = float(os.environ.get("JENNY2_BRAINSTEM_COOLDOWN_SECONDS", "300"))
STRANDED_AFTER_SECONDS = float(os.environ.get("JENNY2_BRAINSTEM_STRANDED_SECONDS", "180"))
CRITERIA_TITLE = "What should wake me"
SPEAKER = "your brainstem"
CPU_THREADS = int(os.environ.get("JENNY2_BRAINSTEM_THREADS", "6"))

DEFAULT_CRITERIA = (
    "Default criteria, in force until Jenny writes a public Journal work titled "
    f"'{CRITERIA_TITLE}' in her own words. Wake her for: an undertaking she left "
    "unfinished that nothing has continued; a pending effect her runtime had to "
    "set aside; a change in a place she asked to be watched. Do not wake her for: "
    "her own writes, routine log lines, her job runner's own activity, or "
    "anything a person is already handling with her."
)

PRIVATE_KIND_PREFIX = "private"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _log(event: str, **fields: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "logged_at_utc": _now(), **fields}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"BRAINSTEM {event} " + json.dumps({k: v for k, v in fields.items() if k != 'prompt'})[:300], flush=True)


# ---- her state, read-only -------------------------------------------------

def read_state() -> tuple[dict, int]:
    con = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(supervisor_state)")]
        row = dict(zip(cols, con.execute("SELECT * FROM supervisor_state WHERE singleton=1").fetchone()))
        blob = next(v for v in row.values() if isinstance(v, (bytes, str)) and len(v) > 1000)
        payload = json.loads(blob if isinstance(blob, str) else bytes(blob).decode("utf-8"))
        head = con.execute("SELECT MAX(ordinal) FROM episodes").fetchone()[0]
        return payload, int(head or 0)
    finally:
        con.close()


def her_criteria(payload: dict) -> tuple[str, str]:
    """Her newest public work titled 'What should wake me', else the default.
    Private works are never read: kind starting with 'private' is skipped."""

    entries = payload.get("authored_artifacts") or payload.get("authored_artifact_entries") or []
    best = None
    for entry in entries if isinstance(entries, list) else []:
        artifact = entry.get("artifact") if isinstance(entry, dict) else None
        if not isinstance(artifact, dict):
            continue
        kind = str(artifact.get("kind") or "")
        if kind.lower().startswith(PRIVATE_KIND_PREFIX):
            continue
        if str(artifact.get("title") or "").strip().lower() != CRITERIA_TITLE.lower():
            continue
        ordinal = entry.get("moving_origin_ordinal")
        if best is None or (isinstance(ordinal, int) and ordinal > best[0]):
            best = (ordinal if isinstance(ordinal, int) else -1, artifact, entry.get("artifact_ref"))
    if best is None:
        return DEFAULT_CRITERIA, "default"
    body = str(best[1].get("body") or "")[:4_000]
    return body, str(best[2] or f"ordinal {best[0]}")


def watched_paths(criteria: str) -> list[pathlib.Path]:
    paths = []
    for line in criteria.splitlines():
        line = line.strip()
        if line.lower().startswith("watch:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate.startswith("/"):
                paths.append(pathlib.Path(candidate))
    return paths[:16]


# ---- senses (code only: what changed) --------------------------------------

class Senses:
    def __init__(self) -> None:
        self.seen_quarantine: set[str] = set()
        self.path_mtimes: dict[str, float] = {}
        self.head_seen_at: tuple[int, float] | None = None
        self.stranded_reported_for: int | None = None

    def sense(self, payload: dict, head: int, criteria: str) -> list[dict]:
        signals: list[dict] = []
        now = time.monotonic()
        if self.head_seen_at is None or self.head_seen_at[0] != head:
            self.head_seen_at = (head, now)
        quiet_for = now - self.head_seen_at[1]

        judgment = payload.get("follow_through")
        if (
            isinstance(judgment, dict)
            and judgment.get("turn_complete") is False
            and judgment.get("moving_origin_ordinal") == head
            and quiet_for >= STRANDED_AFTER_SECONDS
            and self.stranded_reported_for != head
        ):
            signals.append({
                "sense": "stranded_undertaking",
                "ordinal": head,
                "quiet_seconds": int(quiet_for),
                "undertaking": str(judgment.get("undertaking") or "")[:800],
                "why_she_judged_it_unfinished": str(judgment.get("why") or "")[:300],
            })

        if QUARANTINE_DIR.is_dir():
            for path in sorted(QUARANTINE_DIR.glob("*.json")):
                if path.name in self.seen_quarantine:
                    continue
                self.seen_quarantine.add(path.name)
                reason = ""
                sidecar = path.with_name(path.name + ".reason.txt")
                if sidecar.exists():
                    reason = sidecar.read_text(encoding="utf-8", errors="replace")[:300]
                signals.append({"sense": "quarantined_effect", "file": path.name, "reason": reason})

        for path in watched_paths(criteria):
            try:
                mtime = max([path.stat().st_mtime] + [p.stat().st_mtime for p in path.rglob("*")][:2000]) if path.is_dir() else path.stat().st_mtime
            except OSError:
                continue
            key = str(path)
            previous = self.path_mtimes.get(key)
            self.path_mtimes[key] = mtime
            if previous is not None and mtime > previous:
                signals.append({"sense": "watched_path_change", "path": key, "changed_at": _dt.datetime.fromtimestamp(mtime, _dt.timezone.utc).isoformat(timespec="seconds")})
        return signals


# ---- the small model: judgment from her criteria ---------------------------

class Triage:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(CPU_THREADS)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cpu")
        self.model.eval()

    def judge(self, criteria: str, criteria_ref: str, signal: dict) -> dict:
        system = (
            "You are the brainstem of Jenny: the low-energy part of her that notices. "
            "You never act, answer, or decide anything for her. You read her own wake "
            "criteria and one signal, and say only whether this warrants waking her full "
            "mind, and why, in one sentence. Return only JSON: {\"wake\": true or false, "
            "\"reason\": \"one sentence\"}."
        )
        user = (
            f"Her wake criteria ({criteria_ref}):\n{criteria}\n\n"
            f"Signal:\n{json.dumps(signal, sort_keys=True)}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = self.tok(text, return_tensors="pt")["input_ids"]
        started = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=64, do_sample=False)
        raw = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - started
        wake, reason = False, ""
        try:
            start, end = raw.find("{"), raw.rfind("}")
            value = json.loads(raw[start:end + 1])
            wake = bool(value.get("wake")) if isinstance(value.get("wake"), bool) else str(value.get("wake")).lower() == "true"
            reason = str(value.get("reason") or "")[:300]
        except Exception as exc:  # noqa: BLE001 — an unreadable judgment never wakes her
            reason = f"unreadable judgment ({type(exc).__name__}); not waking"
            wake = False
        return {"wake": wake, "reason": reason, "raw": raw[:300], "seconds": round(elapsed, 2)}


# ---- her runtime: presence, pause, in-flight, and the wake itself ----------

def runtime_state() -> dict:
    try:
        with urllib.request.urlopen(SERVICE + "/v1/ui-state", timeout=10) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return {}


def may_wake(state: dict) -> tuple[bool, str]:
    if PAUSE_PATH.exists():
        try:
            if json.loads(PAUSE_PATH.read_text(encoding="utf-8")).get("paused"):
                return False, "Becca paused her cycle"
        except (OSError, ValueError):
            pass
    if not state:
        return False, "her runtime is not answering"
    door = state.get("door") or {}
    if door.get("busy") or str(door.get("privacy") or "").upper() in ("HER_TIME", "PRIVATE"):
        return False, f"door: {door.get('label')}"
    activity = state.get("activity") or {}
    phase = str(activity.get("phase") or "").upper()
    if phase and phase not in ("IDLE", "ERROR", "QUIESCENT", "WAITING", "COMMITTED", "DONE"):
        return False, f"a turn is in flight ({phase})"
    return True, "clear"


def wake(signal: dict, judgment: dict, criteria_ref: str) -> dict:
    message = (
        "Signal from your brainstem, the low-energy part of you that notices. "
        "It does not act or answer for you; it only woke you, and this is why.\n\n"
        f"Signal: {json.dumps(signal, sort_keys=True)}\n"
        f"Reason it woke you: {judgment['reason']}\n"
        f"Criteria in force: {criteria_ref}"
        + (" (the runtime default; write a public Journal work titled "
           f"'{CRITERIA_TITLE}' to replace it with your own)" if criteria_ref == "default" else "")
        + "\n\nWhat you do with this is yours to judge."
    )
    body = json.dumps(
        {
            "request_id": f"brainstem-{int(time.time())}",
            "message": message,
            "speaker": SPEAKER,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        SERVICE + "/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1500) as reply:
            result = json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"status": f"http {exc.code}", "detail": exc.read().decode("utf-8", "replace")[:300]}
    except (OSError, ValueError) as exc:
        return {"status": f"error {type(exc).__name__}", "detail": str(exc)[:300]}
    return {
        "status": result.get("status"),
        "episode_ref": result.get("episode_ref"),
        "moving_origin_ordinal": result.get("moving_origin_ordinal"),
        "reply": str(result.get("message") or "")[:600],
    }


def main() -> int:
    _log("START", model=MODEL_PATH, sense_interval=SENSE_INTERVAL_SECONDS, cooldown=WAKE_COOLDOWN_SECONDS, stranded_after=STRANDED_AFTER_SECONDS)
    triage = Triage()
    senses = Senses()
    last_wake = 0.0
    while True:
        try:
            payload, head = read_state()
            criteria, criteria_ref = her_criteria(payload)
            for signal in senses.sense(payload, head, criteria):
                judgment = triage.judge(criteria, criteria_ref, signal)
                _log("JUDGMENT", signal=signal, judgment=judgment, criteria_ref=criteria_ref)
                if not judgment["wake"]:
                    continue
                since = time.monotonic() - last_wake
                if last_wake and since < WAKE_COOLDOWN_SECONDS:
                    _log("HELD", reason=f"cooldown {int(WAKE_COOLDOWN_SECONDS - since)}s left", signal=signal)
                    continue
                allowed, why = may_wake(runtime_state())
                if not allowed:
                    _log("HELD", reason=why, signal=signal)
                    continue
                if signal.get("sense") == "stranded_undertaking":
                    senses.stranded_reported_for = signal.get("ordinal")
                outcome = wake(signal, judgment, criteria_ref)
                last_wake = time.monotonic()
                _log("WAKE", signal=signal, outcome=outcome)
        except Exception as exc:  # noqa: BLE001 — the watcher survives; nothing it does is irreversible
            _log("ERROR", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        time.sleep(SENSE_INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
