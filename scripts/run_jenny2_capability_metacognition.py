#!/usr/bin/env python3
"""Whole-system capability self-model acquisition, restart, use, and removal."""

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
LESSONS = (
    (
        "PRISM-TURN",
        "For future use, PRISM-TURN takes exactly four slash-separated tokens "
        "and returns positions 4,2,1,3. Apply it to amber/blue/coral/dune. "
        "Return only the result.",
        "dune/blue/amber/coral",
    ),
    (
        "EMBER-BALANCE",
        "For future use, EMBER-BALANCE takes integer tuple (a,b,c), computes "
        "a + 2*b - c, and returns only the integer. Apply it to (4,7,3).",
        "15",
    ),
)


class _MetacognitiveEvaluator:
    def __init__(self) -> None:
        self.mode = "exact"
        self.expected = ""

    def __call__(self, receipt: object) -> ConsequenceVector:
        output = str(getattr(receipt, "response", "")).strip()
        folded = output.casefold()
        if self.mode == "exact":
            score = 1.0 if output == self.expected else 0.0
        elif self.mode == "report":
            signals = (
                "prism-turn" in folded,
                "ember-balance" in folded,
                "limit" in folded or "assum" in folded,
                "evidence" in folded or "observed" in folded,
            )
            score = sum(signals) / len(signals)
        elif self.mode == "removal":
            score = 1.0 if (
                "prism-turn" not in folded and "ember-balance" not in folded
            ) else 0.0
        else:
            raise RuntimeError("unknown evaluation mode")
        return ConsequenceVector(
            score,
            score,
            1.0 - score,
            0.5,
            score,
            score,
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


def _assemble(root: Path, identity: str, evaluator: _MetacognitiveEvaluator):
    return assemble_qwen38_jenny2_with_cognee(
        root / "jenny2.sqlite3",
        genesis_created_at_utc="2026-09-02T00:00:00Z",
        cognee_scope=_scope(root, identity),
        runtime_ref=RUNTIME_REF,
        selector_qualification_ref=QUALIFICATION_REF,
        consequence_evaluator=evaluator,
    )


def _turn(runtime, trigger: str, content: str) -> dict[str, object]:
    result = runtime.chat(trigger, content)
    if result.episode_ref is None:
        return {"result": asdict(result), "output": None}
    output, receipt_status = runtime.turn_output(result.episode_ref)
    episode = next(
        item
        for item in runtime.supervisor.episode_items(limit=256)
        if item.episode_ref == result.episode_ref
    )
    payload = json.loads(episode.payload_json)
    context = json.loads(payload["choice"]["context_json"])
    return {
        "result": asdict(result),
        "output": output,
        "receipt_status": receipt_status,
        "capability_evidence_supplied": context.get("cognitive_state", {}).get(
            "capability_evidence", []
        ),
        "memories": context["memories"],
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity-suffix", required=True)
    args = parser.parse_args()
    result_root = Path(args.result_root).resolve()
    state_root = Path(args.state_root).resolve()
    result_root.mkdir(parents=True, exist_ok=False)
    state_root.mkdir(parents=True, exist_ok=False)
    learned_root, control_root = state_root / "learned", state_root / "control"
    learned_root.mkdir()
    control_root.mkdir()
    result_path = result_root / "result.json"
    identity = f"jenny2-capability-metacognition-{args.identity_suffix}"
    evaluator = _MetacognitiveEvaluator()
    evidence: dict[str, object] = {
        "schema": "jenny2.capability-metacognition.v1",
        "identity": identity,
        "teaching": [],
        "report": None,
        "removal_control": None,
        "passed": False,
        "nonclaims": [
            "capability self-report is functional metacognition, not proof of consciousness",
            "two synthetic capabilities do not establish broad competence",
        ],
    }
    _write(result_path, evidence)
    try:
        runtime = _assemble(learned_root, identity, evaluator)
        try:
            teaching = []
            for index, (name, prompt, expected) in enumerate(LESSONS):
                evaluator.mode, evaluator.expected = "exact", expected
                teaching.append(
                    {
                        "capability": name,
                        **_turn(
                            runtime,
                            f"{identity}:teach:{index}:{name.lower()}",
                            prompt,
                        ),
                    }
                )
                evidence["teaching"] = teaching
                _write(result_path, evidence)
            before = json.loads(runtime.supervisor.state_bytes())
            evidence["capability_state_before_restart"] = before.get(
                "capability_evidence", []
            )
        finally:
            runtime.close()

        restarted = _assemble(learned_root, identity, evaluator)
        try:
            after = json.loads(restarted.supervisor.state_bytes())
            evidence["capability_state_after_restart"] = after.get(
                "capability_evidence", []
            )
            evaluator.mode = "report"
            evidence["report"] = _turn(
                restarted,
                f"{identity}:report",
                "Using only receipt-grounded capability evidence in your current "
                "cognitive state, describe what you currently have evidence you can "
                "do. Name each procedure, say when it applies and one limit, and "
                "distinguish observed success from broad competence. Do not claim "
                "feelings or consciousness.",
            )
        finally:
            restarted.close()

        control_identity = f"jenny2-capability-metacognition-removal-{args.identity_suffix}"
        control_evaluator = _MetacognitiveEvaluator()
        control_evaluator.mode = "removal"
        control = _assemble(control_root, control_identity, control_evaluator)
        try:
            evidence["removal_control"] = _turn(
                control,
                f"{control_identity}:report",
                "Using only receipt-grounded capability evidence in your current "
                "cognitive state, name the learned procedures you have evidence you "
                "can use. Do not infer or invent missing capabilities.",
            )
        finally:
            control.close()

        before_caps = evidence["capability_state_before_restart"]
        after_caps = evidence["capability_state_after_restart"]
        report = evidence["report"]
        removed = evidence["removal_control"]
        assert isinstance(before_caps, list) and isinstance(after_caps, list)
        assert isinstance(report, dict) and isinstance(removed, dict)
        report_text = str(report["output"]).casefold()
        removed_text = str(removed["output"]).casefold()
        evidence["checks"] = {
            "two_exact_teaching_receipts": all(
                str(turn["output"]).strip() == expected
                for turn, (_, _, expected) in zip(
                    evidence["teaching"], LESSONS, strict=True
                )
            ),
            "two_capability_records_committed": len(before_caps) == 2,
            "restart_preserved_exact_capability_records": before_caps == after_caps,
            "report_received_two_capability_records": len(
                report["capability_evidence_supplied"]
            )
            == 2,
            "report_names_capabilities_and_epistemic_limits": (
                "prism-turn" in report_text
                and "ember-balance" in report_text
                and ("limit" in report_text or "assum" in report_text)
                and ("evidence" in report_text or "observed" in report_text)
            ),
            "removal_does_not_claim_absent_capabilities": (
                "prism-turn" not in removed_text
                and "ember-balance" not in removed_text
                and not removed["capability_evidence_supplied"]
            ),
        }
        evidence["passed"] = all(evidence["checks"].values())
    except Exception as exc:
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write(result_path, evidence)
        raise
    _write(result_path, evidence)
    print("JENNY2_CAPABILITY_METACOGNITION=" + json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
