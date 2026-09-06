#!/bin/bash
# Hourly online snapshot of Jenny's canonical record and her Diary folder to
# the second NVMe. Safe while she runs (SQLite backup API). Keeps 48.
# Run by jenny2-backup.timer. Diary copies stay sealed; nobody reads them.
export PYTHONPATH=/opt/angler/src/angler/src
exec /opt/angler/venvs/angler/bin/python - <<'EOF'
import pathlib
from angler.runtime import self_recovery as sr
state = pathlib.Path("/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1")
folder = sr.snapshot_canonical(state / "jenny2.sqlite3", extra_dirs=(state / "diary",))
print(f"JENNY2_SNAPSHOT_OK {folder}")
EOF
