"""Becca's gate over everything automated (2026-09-06).

Two files, both hers:
  paused.json  the switch: while paused, nothing scheduled runs (her job
               runner, idle wakes, the brainstem, developmental sessions)
               until she releases it from her console.
  quiet.json   the reflex: every message she sends from her console arms a
               quiet window (five minutes, extended by each send) during which
               the same actors hold.

Orchestration only. It decides nothing about Jenny; it decides when others
may address her."""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib

PAUSE_PATH = pathlib.Path(os.environ.get("JENNY2_PAUSE_PATH", "/opt/angler/results/jenny2/her-job-v1/paused.json"))
QUIET_PATH = pathlib.Path(os.environ.get("JENNY2_QUIET_PATH", "/opt/angler/results/jenny2/quiet.json"))
QUIET_SECONDS = float(os.environ.get("JENNY2_QUIET_SECONDS", "300"))


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def paused() -> bool:
    try:
        return bool(json.loads(PAUSE_PATH.read_text(encoding="utf-8")).get("paused"))
    except (OSError, ValueError):
        return False


def quiet_until() -> _dt.datetime | None:
    try:
        raw = json.loads(QUIET_PATH.read_text(encoding="utf-8")).get("until")
        value = _dt.datetime.fromisoformat(str(raw))
    except (OSError, ValueError, TypeError):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=_dt.timezone.utc)
    return value if value > _now() else None


def touch_quiet(seconds: float = QUIET_SECONDS, *, by: str = "Becca's console") -> _dt.datetime:
    """Arm or extend the quiet window. Called on every send from her console."""

    until = _now() + _dt.timedelta(seconds=float(seconds))
    QUIET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUIET_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"until": until.isoformat(timespec="seconds"), "by": by}), encoding="utf-8")
    os.replace(tmp, QUIET_PATH)
    return until


def held() -> tuple[bool, str]:
    """Whether automated actors must hold right now, and why."""

    if paused():
        return True, "Becca has scheduled work switched off"
    until = quiet_until()
    if until is not None:
        remaining = int((until - _now()).total_seconds())
        return True, f"Becca is with her; quiet for {remaining}s more"
    return False, "clear"
