#!/usr/bin/env python3
"""Run and restart one complete effectless Jenny 2.0 Qwen3.8 cycle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.higher_level_experience_cycle import ConsequenceVector
from angler.runtime.jenny2_runtime import assemble_qwen38_jenny2_with_cognee


def _scope(root: Path, identity: str) -> CogneeWorkerScope:
    return CogneeWorkerScope(
        dataset_name=identity,
        tenant_name=f"{identity}-tenant",
        node_set_name=f"{identity}-records",
        state_root=str(root / "cognee"),
    )


def _objective(receipt: object) -> ConsequenceVector:
    response = getattr(receipt, "response", "").strip()
    exact = response == "ORCHID-742"
    return ConsequenceVector(
        1.0 if exact else -0.5,
        1.0 if exact else -1.0,
        0.0 if exact else 1.0,
        0.2,
        1.0,
        0.1,
        0.1,
        1.0,
        0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--runtime-ref", required=True)
    parser.add_argument("--qualification-ref", required=True)
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    database = root / "jenny2.sqlite3"
    common = dict(
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=_scope(root, args.identity),
        runtime_ref=args.runtime_ref,
        consequence_evaluator=_objective,
        selector_qualification_ref=args.qualification_ref,
    )
    runtime = assemble_qwen38_jenny2_with_cognee(database, **common)
    try:
        before = runtime.status()
        result = runtime.chat(
            f"{args.identity}:human:one",
            "Return exactly ORCHID-742 and nothing else.",
        )
        after = runtime.status()
        output, receipt_status = runtime.turn_output(result.episode_ref)
        episode_count = len(runtime.supervisor.episode_items(limit=256))
    finally:
        runtime.close()
    restarted = assemble_qwen38_jenny2_with_cognee(database, **common)
    try:
        resumed = restarted.status()
        payload = {
            "identity": args.identity,
            "before": asdict(before),
            "result": asdict(result),
            "after": asdict(after),
            "resumed": asdict(resumed),
            "output": output,
            "receipt_status": receipt_status,
            "episode_count": episode_count,
            "state_exact_after_restart": resumed.state_ref == after.state_ref,
            "ordinal_exact_after_restart": (
                resumed.moving_origin_ordinal == after.moving_origin_ordinal
            ),
        }
        passed = (
            result.status == "COMMITTED"
            and output == "ORCHID-742"
            and receipt_status == "COMPLETED"
            and episode_count == 1
            and after.moving_origin_ordinal == 0
            and before.state_ref != after.state_ref
            and after.pending_projections == 0
            and not after.external_effects_enabled
            and payload["state_exact_after_restart"]
            and payload["ordinal_exact_after_restart"]
        )
        payload["passed"] = passed
        print("JENNY2_QWEN38_FUNCTIONAL=" + json.dumps(payload, sort_keys=True), flush=True)
        (root / "functional-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0 if passed else 2
    finally:
        restarted.close()


if __name__ == "__main__":
    raise SystemExit(main())
