"""Self-recovery for Jenny 2.0 after a random death of the box (Becca,
2026-09-06: "if the system dies randomly she should be able to self recover").

Three mechanisms, all orchestration, none touching what she thinks:

  canonical guard      at startup her canonical record (jenny2.sqlite3) is
                       integrity-checked; if damaged, the newest hourly
                       snapshot that passes the same check is restored and the
                       damaged file is kept aside. Loudly logged.
  memory rebuild       Cognee is a rebuildable projection of the canonical
                       record. A projection fault writes a rebuild request;
                       the next startup sets the damaged stores aside, clears
                       the projection marks, and re-projects everything from
                       canonical. Bounded by an attempt count so a store that
                       cannot be rebuilt leaves her running on canonical memory
                       instead of looping.
  snapshots            an hourly online snapshot of the canonical record and
                       her Diary folder to the second drive (jenny2-backup).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import shutil
import sqlite3
import sys

REBUILD_REQUEST_NAME = "memory-rebuild-requested.json"
MAX_REBUILD_ATTEMPTS = 3
DEFAULT_SNAPSHOT_ROOT = pathlib.Path(os.environ.get("JENNY2_SNAPSHOT_ROOT", "/mnt/backup/jenny2-hourly"))


def _log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---- canonical guard -------------------------------------------------------

def canonical_ok(db_path: pathlib.Path) -> bool:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
            return bool(row) and row[0] == "ok"
        finally:
            con.close()
    except sqlite3.Error:
        return False


def newest_good_snapshot(snapshot_root: pathlib.Path, name: str = "jenny2.sqlite3") -> pathlib.Path | None:
    if not snapshot_root.is_dir():
        return None
    candidates = sorted((p for p in snapshot_root.iterdir() if p.is_dir()), reverse=True)
    for folder in candidates:
        candidate = folder / name
        if candidate.is_file() and canonical_ok(candidate):
            return candidate
    return None


def guard_canonical(db_path: pathlib.Path, *, snapshot_root: pathlib.Path = DEFAULT_SNAPSHOT_ROOT) -> str:
    """Return 'ok', 'restored:<snapshot>', 'absent', or 'damaged-unrecoverable'."""

    if not db_path.exists():
        return "absent"
    if canonical_ok(db_path):
        return "ok"
    aside = db_path.with_name(f"{db_path.name}.damaged-{_stamp()}")
    for suffix in ("", "-wal", "-shm"):
        source = pathlib.Path(str(db_path) + suffix)
        if source.exists():
            shutil.move(str(source), str(aside) + suffix)
    _log(f"JENNY2_CANONICAL_DAMAGED moved={aside.name}")
    snapshot = newest_good_snapshot(snapshot_root, db_path.name)
    if snapshot is None:
        _log("JENNY2_CANONICAL_UNRECOVERABLE no good snapshot found")
        return "damaged-unrecoverable"
    shutil.copy2(str(snapshot), str(db_path))
    _log(f"JENNY2_CANONICAL_RESTORED from={snapshot}")
    return f"restored:{snapshot}"


def snapshot_canonical(db_path: pathlib.Path, *, snapshot_root: pathlib.Path = DEFAULT_SNAPSHOT_ROOT, keep: int = 48, extra_dirs: tuple[pathlib.Path, ...] = ()) -> pathlib.Path:
    """Online snapshot with SQLite's backup API (safe while she runs), plus
    copies of extra folders (her Diary; sealed, never read). Rotates to keep."""

    folder = snapshot_root / _stamp()
    folder.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(str(folder / db_path.name))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    if not canonical_ok(folder / db_path.name):
        raise RuntimeError("snapshot failed its own integrity check")
    for extra in extra_dirs:
        if extra.is_dir():
            shutil.copytree(str(extra), str(folder / extra.name), symlinks=True)
    for old in sorted((p for p in snapshot_root.iterdir() if p.is_dir()))[:-keep]:
        shutil.rmtree(str(old), ignore_errors=True)
    return folder


# ---- memory rebuild --------------------------------------------------------

def request_memory_rebuild(state_root: pathlib.Path, reason: str, *, mode: str = "set-aside") -> int:
    """Record that Cognee failed (mode set-aside), or that the projection marks
    should be cleared so everything re-projects into the stores as they are
    (mode marks-only). Returns the attempt count now on file."""

    path = state_root / REBUILD_REQUEST_NAME
    attempts = 0
    try:
        attempts = int(json.loads(path.read_text(encoding="utf-8")).get("attempts", 0))
    except (OSError, ValueError, TypeError):
        attempts = 0
    attempts += 1
    path.write_text(json.dumps({"attempts": attempts, "mode": mode, "reason": reason[:400], "requested_at_utc": _stamp()}), encoding="utf-8")
    _log(f"JENNY2_MEMORY_REBUILD_REQUESTED attempts={attempts} reason={reason[:160]!r}")
    return attempts


def consume_memory_rebuild_request(state_root: pathlib.Path, store_names: tuple[str, ...] = ("cognee", "cognee-capabilities")) -> bool:
    """At startup: if a rebuild is requested and attempts remain, set the
    stores aside (kept, timestamped) and return True so the caller clears the
    projection marks after assembly. Exhausted requests are logged and left."""

    path = state_root / REBUILD_REQUEST_NAME
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        attempts = int(record.get("attempts", 1))
        mode = str(record.get("mode") or "set-aside")
    except (OSError, ValueError, TypeError):
        record, attempts, mode = {}, 1, "set-aside"
    if attempts > MAX_REBUILD_ATTEMPTS:
        _log(f"JENNY2_MEMORY_REBUILD_EXHAUSTED attempts={attempts}; running on canonical memory")
        return False
    # Each consumption counts as an attempt, so a start that dies before
    # finishing the rebuild cannot loop forever.
    record["attempts"] = attempts + 1
    record["consumed_at_utc"] = _stamp()
    try:
        path.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass
    if mode == "marks-only":
        _log(f"JENNY2_MEMORY_REBUILD_STARTED attempt={attempts} mode=marks-only (stores kept)")
        return True
    stamp = _stamp()
    for name in store_names:
        store = state_root / name
        if store.exists():
            shutil.move(str(store), str(state_root / f"{name}.damaged-{stamp}"))
    _log(f"JENNY2_MEMORY_REBUILD_STARTED attempt={attempts} stores_set_aside={stamp}")
    return True


def finish_memory_rebuild(state_root: pathlib.Path, *, cleared: int) -> None:
    path = state_root / REBUILD_REQUEST_NAME
    try:
        path.unlink()
    except OSError:
        pass
    _log(f"JENNY2_MEMORY_REBUILD_MARKS_CLEARED episodes={cleared}; re-projecting from canonical")
