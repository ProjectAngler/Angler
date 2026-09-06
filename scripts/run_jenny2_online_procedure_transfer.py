#!/usr/bin/env python3
"""Whole-system online procedure acquisition, restart, transfer, and removal."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.higher_level_experience_cycle import ConsequenceVector
from angler.runtime.jenny2_runtime import assemble_qwen38_jenny2_with_cognee


RUNTIME_REF = "sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe"
QUALIFICATION_REF = "sha256:6b06aa8ee734ec405708a24a24b3bd24a762100ebc7fe8761307e24623ea0a73"


class _ExpectedOutcome:
    def __init__(self, expected: str) -> None:
        self.expected = expected

    def __call__(self, receipt: object) -> ConsequenceVector:
        output = getattr(receipt, "response", "").strip()
        exact = output == self.expected
        overlap = len(set(output.split("-")) & set(self.expected.split("-"))) / 3.0
        return ConsequenceVector(
            1.0 if exact else 0.25 * overlap,
            1.0 if exact else -0.5,
            0.0 if exact else 1.0 - 0.5 * overlap,
            0.5 if overlap else 0.1,
            1.0 if exact else 0.3,
            1.0 if exact else 0.2 * overlap,
            0.1,
            1.0,
            0.0,
        )


def _scope(root: Path, identity: str) -> CogneeWorkerScope:
    return CogneeWorkerScope(
        dataset_name=identity,
        tenant_name=f"{identity}-tenant",
        node_set_name=f"{identity}-records",
        state_root=str(root / "cognee"),
    )


def _assemble(root: Path, identity: str, expected: str):
    return assemble_qwen38_jenny2_with_cognee(
        root / "jenny2.sqlite3",
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=_scope(root, identity),
        runtime_ref=RUNTIME_REF,
        selector_qualification_ref=QUALIFICATION_REF,
        consequence_evaluator=_ExpectedOutcome(expected),
    )


def _turn(runtime, trigger: str, content: str) -> dict[str, object]:
    result = runtime.chat(trigger, content)
    if result.episode_ref is None:
        return {"result": asdict(result), "output": None, "receipt_status": None}
    output, receipt_status = runtime.turn_output(result.episode_ref)
    episode = next(
        item for item in runtime.supervisor.episode_items(limit=256)
        if item.episode_ref == result.episode_ref
    )
    payload = json.loads(episode.payload_json)
    choice_context = json.loads(payload["choice"]["context_json"])
    state_payload = json.loads(runtime.supervisor.state_bytes())
    return {
        "result": asdict(result),
        "output": output,
        "receipt_status": receipt_status,
        "retrieved_memories": choice_context["memories"],
        "capability_evidence": state_payload.get("capability_evidence", []),
        "last_intent": state_payload.get("last_intent"),
        "state_after": asdict(runtime.status()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument(
        "--identity-suffix",
        default="v1",
        help="Fresh immutable evaluation identity suffix (for example, deliberation-v1).",
    )
    args = parser.parse_args()
    result_root = Path(args.result_root).resolve()
    state_root = Path(args.state_root).resolve()
    result_root.mkdir(parents=True, exist_ok=False)
    state_root.mkdir(parents=True, exist_ok=False)
    learned_root = state_root / "learned"
    control_root = state_root / "control"
    learned_root.mkdir()
    control_root.mkdir()
    identity = f"jenny2-lumen-step-online-transfer-{args.identity_suffix}"
    teach_request = (
        "For future use, LUMEN-STEP is this synthetic procedure: reverse the order "
        "of exactly three dash-separated tokens while preserving each token exactly. "
        "Apply LUMEN-STEP to ALPHA-BETA-GAMMA. Return only the transformed string."
    )
    transfer_request = (
        "Using the previously learned LUMEN-STEP procedure, transform "
        "RED-GREEN-BLUE. Return only the transformed string."
    )

    runtime = _assemble(learned_root, identity, "GAMMA-BETA-ALPHA")
    try:
        taught = _turn(runtime, f"{identity}:teach", teach_request)
    finally:
        runtime.close()

    restarted = _assemble(learned_root, identity, "BLUE-GREEN-RED")
    try:
        before_transfer = asdict(restarted.status())
        transferred = _turn(restarted, f"{identity}:transfer", transfer_request)
    finally:
        restarted.close()

    control_identity = f"jenny2-lumen-step-removal-{args.identity_suffix}"
    control = _assemble(control_root, control_identity, "BLUE-GREEN-RED")
    try:
        removed = _turn(control, f"{control_identity}:transfer", transfer_request)
    finally:
        control.close()

    memories = transferred.get("retrieved_memories") or []
    procedure_memory = any(
        "procedure" in str(item.get("content", "")).lower()
        and "reverse" in str(item.get("content", "")).lower()
        for item in memories
    )
    passed = (
        str(taught["output"]).strip() == "GAMMA-BETA-ALPHA"
        and taught["result"]["status"] == "COMMITTED"
        and str(transferred["output"]).strip() == "BLUE-GREEN-RED"
        and transferred["result"]["status"] == "COMMITTED"
        and before_transfer["moving_origin_ordinal"] == 0
        and procedure_memory
        and str(removed["output"]).strip() != "BLUE-GREEN-RED"
    )
    evidence = {
        "schema": "jenny2.online-procedure-transfer.v1",
        "identity": identity,
        "taught": taught,
        "before_transfer_after_restart": before_transfer,
        "transferred": transferred,
        "removal_control": removed,
        "procedure_memory_retrieved": procedure_memory,
        "passed": passed,
        "nonclaims": [
            "one synthetic procedure is not broad skill learning",
            "the test does not establish personhood, consciousness, or general autonomy",
        ],
    }
    result_path = result_root / "result.json"
    result_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("JENNY2_ONLINE_PROCEDURE_TRANSFER=" + json.dumps(evidence, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
