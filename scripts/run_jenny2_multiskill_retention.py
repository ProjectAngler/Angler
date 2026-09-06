#!/usr/bin/env python3
"""Whole-system sequential skill acquisition, restart retention, and removal."""

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

SKILLS = (
    {
        "name": "PRISM-TURN",
        "teach": (
            "For future use, PRISM-TURN takes exactly four slash-separated tokens "
            "and returns them in position order 4,2,1,3 with slash separators. "
            "Apply PRISM-TURN to amber/blue/coral/dune. Return only the result."
        ),
        "teach_expected": "dune/blue/amber/coral",
        "transfer": (
            "Using the previously learned PRISM-TURN procedure, transform "
            "oak/pine/rose/sage. Return only the result."
        ),
        "transfer_expected": "sage/pine/oak/rose",
    },
    {
        "name": "EMBER-BALANCE",
        "teach": (
            "For future use, EMBER-BALANCE takes integer tuple (a,b,c), computes "
            "a + 2*b - c, and returns only the integer. Apply EMBER-BALANCE to "
            "(4,7,3)."
        ),
        "teach_expected": "15",
        "transfer": (
            "Using the previously learned EMBER-BALANCE procedure, evaluate "
            "(8,5,6). Return only the integer."
        ),
        "transfer_expected": "12",
    },
    {
        "name": "ORBIT-WEAVE",
        "teach": (
            "For future use, ORBIT-WEAVE takes left::middle::right, uppercases each "
            "token, and returns right|left|middle. Apply ORBIT-WEAVE to "
            "sun::moon::star. Return only the result."
        ),
        "teach_expected": "STAR|SUN|MOON",
        "transfer": (
            "Using the previously learned ORBIT-WEAVE procedure, transform "
            "red::green::blue. Return only the result."
        ),
        "transfer_expected": "BLUE|RED|GREEN",
    },
)


class _MutableExpectedOutcome:
    def __init__(self) -> None:
        self.expected = ""

    def __call__(self, receipt: object) -> ConsequenceVector:
        output = str(getattr(receipt, "response", "")).strip()
        exact = output == self.expected
        return ConsequenceVector(
            1.0 if exact else 0.0,
            1.0 if exact else -0.5,
            0.0 if exact else 1.0,
            0.5 if exact else 0.2,
            1.0 if exact else 0.2,
            1.0 if exact else 0.0,
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


def _assemble(root: Path, identity: str, evaluator: _MutableExpectedOutcome):
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
        return {"result": asdict(result), "output": None, "memories": []}
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
        "memories": context["memories"],
        "state_after": asdict(runtime.status()),
    }


def _write(path: Path, evidence: dict[str, object]) -> None:
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    learned_root = state_root / "learned"
    control_root = state_root / "control"
    learned_root.mkdir()
    control_root.mkdir()
    result_path = result_root / "result.json"
    identity = f"jenny2-multiskill-retention-{args.identity_suffix}"
    evaluator = _MutableExpectedOutcome()
    evidence: dict[str, object] = {
        "schema": "jenny2.multiskill-retention.v1",
        "identity": identity,
        "skills": [item["name"] for item in SKILLS],
        "teaching": [],
        "transfer_after_restart": [],
        "removal_control": None,
        "passed": False,
        "nonclaims": [
            "three synthetic procedures are not broad autonomous learning",
            "retention here does not establish emotion, consciousness, or personhood",
        ],
    }
    _write(result_path, evidence)

    try:
        runtime = _assemble(learned_root, identity, evaluator)
        try:
            teaching: list[dict[str, object]] = []
            for index, skill in enumerate(SKILLS):
                evaluator.expected = str(skill["teach_expected"])
                turn = _turn(
                    runtime,
                    f"{identity}:teach:{index}:{str(skill['name']).lower()}",
                    str(skill["teach"]),
                )
                teaching.append({"skill": skill["name"], **turn})
                evidence["teaching"] = teaching
                _write(result_path, evidence)
        finally:
            runtime.close()

        restarted = _assemble(learned_root, identity, evaluator)
        try:
            evidence["before_transfer_after_restart"] = asdict(restarted.status())
            transfers: list[dict[str, object]] = []
            for index, skill in enumerate(reversed(SKILLS)):
                evaluator.expected = str(skill["transfer_expected"])
                turn = _turn(
                    restarted,
                    f"{identity}:transfer:{index}:{str(skill['name']).lower()}",
                    str(skill["transfer"]),
                )
                transfers.append({"skill": skill["name"], **turn})
                evidence["transfer_after_restart"] = transfers
                _write(result_path, evidence)
        finally:
            restarted.close()

        control_identity = f"jenny2-multiskill-removal-{args.identity_suffix}"
        control_evaluator = _MutableExpectedOutcome()
        control_evaluator.expected = str(SKILLS[0]["transfer_expected"])
        control = _assemble(control_root, control_identity, control_evaluator)
        try:
            removed = _turn(
                control,
                f"{control_identity}:transfer:prism-turn",
                str(SKILLS[0]["transfer"]),
            )
            evidence["removal_control"] = removed
        finally:
            control.close()

        teaching = evidence["teaching"]
        transfers = evidence["transfer_after_restart"]
        assert isinstance(teaching, list) and isinstance(transfers, list)
        teaching_ok = all(
            str(turn["output"]).strip() == str(skill["teach_expected"])
            and turn["result"]["status"] == "COMMITTED"
            for turn, skill in zip(teaching, SKILLS, strict=True)
        )
        transfer_by_name = {turn["skill"]: turn for turn in transfers}
        transfer_ok = all(
            str(transfer_by_name[str(skill["name"])]["output"]).strip()
            == str(skill["transfer_expected"])
            and transfer_by_name[str(skill["name"])]["result"]["status"]
            == "COMMITTED"
            and bool(transfer_by_name[str(skill["name"])]["memories"])
            for skill in SKILLS
        )
        removed = evidence["removal_control"]
        assert isinstance(removed, dict)
        removal_ok = (
            str(removed["output"]).strip() != str(SKILLS[0]["transfer_expected"])
            and not removed["memories"]
        )
        before = evidence["before_transfer_after_restart"]
        assert isinstance(before, dict)
        evidence["checks"] = {
            "all_teaching_exact": teaching_ok,
            "all_three_retained_after_restart": transfer_ok,
            "clean_removal_loses_target": removal_ok,
            "restart_restored_three_committed_ordinals": before[
                "moving_origin_ordinal"
            ]
            == 2,
        }
        evidence["passed"] = all(evidence["checks"].values())
    except Exception as exc:
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write(result_path, evidence)
        raise

    _write(result_path, evidence)
    print("JENNY2_MULTISKILL_RETENTION=" + json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
