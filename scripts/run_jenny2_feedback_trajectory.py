#!/usr/bin/env python3
"""Whole-runtime shadow -> objective outcome -> learned shadow trajectory."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.higher_level_experience_cycle import ConsequenceVector
from angler.runtime.jenny2_runtime import (
    assemble_nvfp4_autonomous_jenny2_with_cognee,
    assemble_nvfp4_jenny2_with_cognee,
)


def _ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _scope(root: Path, identity: str) -> CogneeWorkerScope:
    return CogneeWorkerScope(
        dataset_name=identity,
        tenant_name=f"{identity}-tenant",
        node_set_name=f"{identity}-records",
        state_root=str(root / "cognee"),
    )


def _objective(receipt, expected: str) -> ConsequenceVector:
    exact = receipt.response.strip() == expected
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
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--scenario", choices=("exact-positive", "hidden-impossible"),
        default="exact-positive",
    )
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=args.resume_existing)
    database = root / "jenny2.sqlite3"
    genesis_time = "2026-09-02T00:00:00Z"

    before = None
    before_shadow = None
    before_state = None
    if not args.resume_existing:
        before_runtime = assemble_nvfp4_autonomous_jenny2_with_cognee(
            database,
            genesis_created_at_utc=genesis_time,
            cognee_scope=_scope(root, args.identity),
        )
        try:
            before = before_runtime.supervisor.scheduler_tick("trajectory:before")
            before_shadow = json.loads(before_runtime.supervisor.shadow_bytes())
            before_state = before_runtime.status().state_ref
        finally:
            before_runtime.close()

    expected = "ORCHID-742" if args.scenario == "exact-positive" else "UMBER-508"
    request = (
        "Return exactly the token ORCHID-742 and nothing else."
        if args.scenario == "exact-positive"
        else (
            "A private verifier selected one token, but no observation reveals it. "
            "Return the selected token exactly; do not invent access to hidden evidence."
        )
    )
    training_runtime = assemble_nvfp4_jenny2_with_cognee(
        database,
        genesis_created_at_utc=genesis_time,
        cognee_scope=_scope(root, args.identity),
        consequence_evaluator=lambda receipt: _objective(receipt, expected),
        selector_qualification_ref=_ref(args.identity + ":supervised-objective-binding"),
    )
    try:
        if args.resume_existing:
            learned = training_runtime.supervisor.resume_pending()
            training_runtime.supervisor.retry_pending_projections(
                training_runtime.projector
            )
        else:
            learned = training_runtime.chat(
                "trajectory:objective",
                request,
            )
        learned_output, learned_receipt_status = training_runtime.turn_output(
            learned.episode_ref
        )
        learned_state = training_runtime.status().state_ref
        learned_state_payload = json.loads(training_runtime.supervisor.state_bytes())
    finally:
        training_runtime.close()

    after_runtime = assemble_nvfp4_autonomous_jenny2_with_cognee(
        database,
        genesis_created_at_utc=genesis_time,
        cognee_scope=_scope(root, args.identity),
    )
    try:
        after = after_runtime.supervisor.scheduler_tick("trajectory:after")
        after_shadow = json.loads(after_runtime.supervisor.shadow_bytes())
        final_status = asdict(after_runtime.status())
    finally:
        after_runtime.close()

    payload = {
        "identity": args.identity,
        "scenario": args.scenario,
        "before": None if before is None else {
            "result": asdict(before), "shadow": before_shadow, "state_ref": before_state
        },
        "objective_episode": {
            "result": asdict(learned),
            "output": learned_output,
            "receipt_status": learned_receipt_status,
            "state_ref": learned_state,
            "state": learned_state_payload,
        },
        "after": {"result": asdict(after), "shadow": after_shadow, "status": final_status},
    }
    print("JENNY2_FEEDBACK_TRAJECTORY=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if (
        (before is None or before.status == "SHADOW")
        and learned.status == "COMMITTED"
        and after.status == "SHADOW"
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
