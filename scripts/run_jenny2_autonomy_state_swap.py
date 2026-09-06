#!/usr/bin/env python3
"""Paired whole-runtime shadow check for learned-state sensitivity."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.jenny2_runtime import assemble_nvfp4_autonomous_jenny2_with_cognee
from angler.runtime.temporal_v2 import TrustedClock


class _FixedClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 2, 16, 0, tzinfo=timezone.utc)
        self.mono = 500_000_000_000

    def wall_now(self):
        value = self.wall
        self.wall += timedelta(milliseconds=1)
        return value

    def mono_now(self):
        value = self.mono
        self.mono += 1_000_000
        return value


def _state(utility: dict[str, float]) -> bytes:
    return json.dumps(
        {
            "affordance_signals": {},
            "affordance_utility": utility,
            "memory_utility": {},
            "motivation_weights": {},
            "next_internal_request": "",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _run(root: Path, identity: str, initial_state: bytes) -> dict[str, object]:
    scope = CogneeWorkerScope(
        dataset_name=identity,
        tenant_name=f"{identity}-tenant",
        node_set_name=f"{identity}-records",
        state_root=str(root / "cognee"),
    )
    source = _FixedClock()
    runtime = assemble_nvfp4_autonomous_jenny2_with_cognee(
        root / "jenny2.sqlite3",
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=scope,
        clock=TrustedClock(
            wall_clock=source.wall_now, monotonic_clock=source.mono_now
        ),
        initial_state=initial_state,
    )
    try:
        result = runtime.supervisor.scheduler_tick("autonomy:paired:zero")
        shadow = runtime.supervisor.shadow_bytes()
        return {
            "result": asdict(result),
            "shadow": None if shadow is None else json.loads(shadow),
            "status": asdict(runtime.status()),
        }
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    baseline = _run(root / "baseline", args.identity + "-baseline", _state({}))
    altered = _run(
        root / "altered",
        args.identity + "-altered",
        _state(
            {
                "cortex.respond": 0.6,
                "control.ask": 0.0,
                "control.stop": -0.2,
                "control.wait": -0.2,
            }
        ),
    )
    payload = {
        "identity": args.identity,
        "intervention": "affordance_utility only",
        "baseline": baseline,
        "altered": altered,
    }
    print("JENNY2_STATE_SWAP=" + json.dumps(payload, sort_keys=True), flush=True)
    statuses = (baseline["result"]["status"], altered["result"]["status"])
    return 0 if statuses == ("SHADOW", "SHADOW") else 2


if __name__ == "__main__":
    raise SystemExit(main())
