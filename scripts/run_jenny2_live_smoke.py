#!/usr/bin/env python3
"""Run one bounded, whole-runtime Jenny 2.0 functional smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.jenny2_runtime import assemble_nvfp4_jenny2_with_cognee


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = args.identity
    scope = CogneeWorkerScope(
        dataset_name=dataset,
        tenant_name=f"{dataset}-tenant",
        node_set_name=f"{dataset}-records",
        state_root=str(root / "cognee"),
    )
    started = monotonic()
    runtime = None
    try:
        runtime = assemble_nvfp4_jenny2_with_cognee(
            root / "jenny2.sqlite3",
            genesis_created_at_utc="2026-09-02T00:00:00Z",
            cognee_scope=scope,
            selector_qualification_ref=_ref(args.identity + ":functional-smoke-only"),
        )
        before = runtime.status()
        print("JENNY2_BEFORE=" + json.dumps(asdict(before), sort_keys=True), flush=True)
        result = runtime.chat(
            _ref(args.identity + ":trigger"),
            (
                "Explain why preserving the failed first attempt while retrying on "
                "fresh state improves scientific integrity. Give a concise answer."
            ),
        )
        after = runtime.status()
        payload = {
            "identity": args.identity,
            "elapsed_seconds": monotonic() - started,
            "result": asdict(result),
            "before": asdict(before),
            "after": asdict(after),
            "episode_count": len(runtime.supervisor.episode_items(limit=256)),
            "pending_projection_count": len(
                runtime.supervisor.pending_projections(limit=256)
            ),
        }
        print("JENNY2_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
        (root / "live-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0 if result.status == "COMMITTED" else 2
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
