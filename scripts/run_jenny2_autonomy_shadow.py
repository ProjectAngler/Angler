#!/usr/bin/env python3
"""Run one effectless real-model autonomy decision in shadow mode."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.jenny2_runtime import assemble_nvfp4_autonomous_jenny2_with_cognee


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=args.existing)
    scope = CogneeWorkerScope(
        dataset_name=args.identity,
        tenant_name=f"{args.identity}-tenant",
        node_set_name=f"{args.identity}-records",
        state_root=str(root / "cognee"),
    )
    runtime = assemble_nvfp4_autonomous_jenny2_with_cognee(
        root / "jenny2.sqlite3",
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=scope,
    )
    try:
        before = runtime.status()
        result = runtime.supervisor.scheduler_tick("autonomy:shadow:zero")
        after = runtime.status()
        shadow = runtime.supervisor.shadow_bytes()
        payload = {
            "before": asdict(before),
            "result": asdict(result),
            "after": asdict(after),
            "shadow": None if shadow is None else json.loads(shadow),
        }
        print("JENNY2_AUTONOMY_SHADOW=" + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if result.status == "SHADOW" else 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
