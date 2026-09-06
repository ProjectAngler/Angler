"""Thinking-mode qualification V1 — offline, per-request, no serving change.

Leaf: ANG-WORK-RUNTIME-THINKING-MODE-QUALIFICATION-V1-001.

Question: does enabling the base model's native deliberation
(enable_thinking, per-request chat-template switch) improve the
affordance-controller's method choices on recorded decision points,
and at what latency cost?

Arms per decision point:
  CURRENT  — the live serving configuration (adapter identity,
             non-thinking), exactly as production runs it.
  THINKING — the unsuffixed base model with enable_thinking=True and
             reasoning_effort="low".

Replays recorded controller inputs (same reconstruction as the E1
removal harness). Records selection, structured-payload validity,
wall latency, and completion tokens. Mutates no state; reruns no live
identity; calls only the local endpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import time
import zlib

from angler.runtime.higher_level_autonomy_adapter import (
    ControllerOutputContractExhausted,
    FrozenModelAffordanceController,
    _normalize_json_object_text,
    _requires_canonical_object_action,
)
from angler.runtime.frozen_cognitive_models import (
    LocalOpenAICompatibleFrozenBackend,
)
from angler.runtime.higher_level_experience_cycle import (
    MemoryCandidate,
    StructuredExperience,
)
from angler.runtime.jenny_library import LIBRARY_AFFORDANCE
from angler.runtime.persistent_autonomy import Affordance, CycleObservation
from angler.runtime.temporal_v2 import TemporalNow

CORTEX_AFFORDANCE = Affordance(
    "cortex.respond",
    "ACT",
    "Investigate or resolve one bounded internal question with the frozen cortex.",
    "internal.cognition",
    external_effect=False,
)


def from_recorded(cls, data: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_state(blob: bytes) -> bytes:
    raw = bytes(blob)
    try:
        json.loads(raw)
        return raw
    except ValueError:
        return zlib.decompress(raw)


def build_controller(*, args, served_model: str, enable_thinking: bool,
                     reasoning_effort):
    runtime_ref = "sha256:" + hashlib.sha256(
        b"thinking-mode-qualification-v1"
    ).hexdigest()
    backend = LocalOpenAICompatibleFrozenBackend(
        endpoint=args.endpoint,
        served_model=served_model,
        model_path=args.model_path,
        revision=args.revision,
        runtime_ref=runtime_ref,
        maximum_input_characters=1_048_576,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        maximum_output_tokens=16_384 if enable_thinking else 4_096,
        timeout_seconds=420.0,
        trace_label="thinking-qualification",
    )
    return FrozenModelAffordanceController(
        backend,
        draft_backend=None,
        repair_backend=backend,
        qualification_ref=args.qualification_ref,
        maximum_output_tokens=16_384 if enable_thinking else 4_096,
        maximum_repair_output_tokens=4_096,
        require_intent_candidates=True,
    )


def payload_valid(selection) -> bool:
    if not _requires_canonical_object_action(selection.selected_affordance_id):
        return True
    try:
        _normalize_json_object_text(
            selection.action_payload, label="qualification payload"
        )
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--adapter-model", required=True)
    parser.add_argument("--base-model", default="jenny-qwen3.8-27b")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--qualification-ref", required=True)
    parser.add_argument("--ordinals", required=True,
                        help="comma-separated decision points; append :read "
                             "when the correct method is internal.library-read")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    controllers = {
        "CURRENT": build_controller(
            args=args, served_model=args.adapter_model,
            enable_thinking=False, reasoning_effort=None,
        ),
        "THINKING": build_controller(
            args=args, served_model=args.base_model,
            enable_thinking=True, reasoning_effort="low",
        ),
    }

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    results = []
    for token in args.ordinals.split(","):
        parts = token.split(":")
        ordinal = int(parts[0])
        expected = (
            "internal.library-read" if len(parts) > 1 and parts[1] == "read"
            else None
        )
        row = con.execute(
            "SELECT episode_json FROM episodes WHERE ordinal=?", (ordinal,)
        ).fetchone()
        episode = json.loads(bytes(row[0]))
        context = json.loads(episode["choice"]["context_json"])
        blob_row = con.execute(
            "SELECT child_state_blob FROM learned_updates WHERE child_state_ref=?",
            (episode["parent_state_ref"],),
        ).fetchone()
        state_bytes = load_state(blob_row[0])
        observation = from_recorded(CycleObservation, episode["observation"])
        temporal = from_recorded(TemporalNow, context["temporal_now"])
        memories = [
            from_recorded(MemoryCandidate, item)
            for item in context["memories"]
        ]
        experience = StructuredExperience.from_mapping(context["experience"])
        capability_modules = context.get("cognitive_state", {}).get(
            "capability_modules", []
        )
        lived = episode["choice"]["selected_affordance_id"]

        for arm, controller in controllers.items():
            for rep in range(1, args.reps + 1):
                record = {
                    "ordinal": ordinal, "arm": arm, "rep": rep,
                    "lived_selection": lived,
                    "expected_method": expected,
                }
                started = time.monotonic()
                try:
                    selection = controller.select(
                        observation=observation,
                        temporal=temporal,
                        affordances=(CORTEX_AFFORDANCE, LIBRARY_AFFORDANCE),
                        memories=memories,
                        experience=experience,
                        state=state_bytes,
                        capability_modules=capability_modules,
                        autonomous_formation=None,
                    )
                    record["selection"] = selection.selected_affordance_id
                    record["payload_valid"] = payload_valid(selection)
                    if expected is not None:
                        record["method_correct"] = (
                            selection.selected_affordance_id == expected
                            and record["payload_valid"]
                        )
                except ControllerOutputContractExhausted as exc:
                    record["selection"] = "CONTRACT_FAIL"
                    record["payload_valid"] = False
                    if expected is not None:
                        record["method_correct"] = False
                    record["error"] = str(exc)[:160]
                record["wall_seconds"] = round(time.monotonic() - started, 2)
                results.append(record)
                print(
                    f"ordinal={ordinal} arm={arm} rep={rep} "
                    f"sel={record['selection']} "
                    f"valid={record.get('payload_valid')} "
                    f"correct={record.get('method_correct', 'n/a')} "
                    f"t={record['wall_seconds']}s",
                    flush=True,
                )

    artifact = {
        "contract": "jenny.thinking-mode-qualification.v1",
        "leaf": "ANG-WORK-RUNTIME-THINKING-MODE-QUALIFICATION-V1-001",
        "epistemic_status": (
            "OFFLINE_PER_REQUEST_TEMPLATE_COMPARISON_NO_SERVING_CHANGE"
        ),
        "arms": {
            "CURRENT": "adapter identity, non-thinking (production config)",
            "THINKING": "base identity, enable_thinking, reasoning_effort=low",
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    print("artifact:", args.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
