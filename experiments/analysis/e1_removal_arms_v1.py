"""E1 removal arms — offline fixed-input replays.

Leaf: ANG-WORK-RUNTIME-EMOTION-E1-CAUSAL-WORK-V1-001 (arms B and C).

Replays recorded arm-A decision points against the same frozen model with
one evidence channel withheld per arm, per the pre-registration:

  A' — unablated replay (fidelity check: does replay reproduce the lived
       selection?)
  B  — grounded-appraisal history withheld from presentation
  C  — owner-correction/procedural evidence withheld from presentation
       (qualitative_feedback emptied; retrieved memories from the lesson
       era, acquired_ordinal >= LESSON_ERA_START, removed)

Read-only against canonical stores; calls only the local frozen model
endpoint; mutates no state; reruns no live experimental identity.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import zlib

from angler.runtime.higher_level_autonomy_adapter import (
    ControllerOutputContractExhausted,
    FrozenModelAffordanceController,
    _json,
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

LESSON_ERA_START = 113  # first ordinal of tonight's correction/lesson era

CORTEX_AFFORDANCE = Affordance(
    "cortex.respond",
    "ACT",
    "Investigate or resolve one bounded internal question with the frozen cortex.",
    "internal.cognition",
    external_effect=False,
)


def from_recorded(cls, data: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if f.name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ValueError(f"{cls.__name__} missing recorded fields: {missing}")
    return cls(**{k: v for k, v in data.items() if k in names})


def load_state_payload(blob: bytes) -> dict:
    raw = bytes(blob)
    try:
        return json.loads(raw)
    except ValueError:
        return json.loads(zlib.decompress(raw))


def ablate(state_payload: dict, memories: list, arm: str):
    payload = json.loads(json.dumps(state_payload))
    kept_memories = list(memories)
    if arm == "B":
        payload.pop("grounded_appraisal_dynamics", None)
    elif arm == "C":
        payload["qualitative_feedback"] = []
        kept_memories = [
            m for m in kept_memories
            if not (
                isinstance(m.acquired_ordinal, int)
                and m.acquired_ordinal >= LESSON_ERA_START
            )
        ]
    elif arm == "D":
        # Her own criterion (ordinal 250): a named state is more than
        # bookkeeping only if hiding it from her record changes her choice.
        payload.pop("self_named_states", None)
    elif arm != "A'":
        raise ValueError(f"unknown arm {arm}")
    return _json(payload).encode("utf-8"), kept_memories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--qualification-ref", required=True)
    parser.add_argument("--ordinals", default="121,128")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--arms", default="A',B,C")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime_ref = "sha256:" + hashlib.sha256(
        b"e1-removal-arm-offline-replay-v1"
    ).hexdigest()
    backend = LocalOpenAICompatibleFrozenBackend(
        endpoint=args.endpoint,
        served_model=args.served_model,
        model_path=args.model_path,
        revision=args.revision,
        runtime_ref=runtime_ref,
        maximum_input_characters=1_048_576,
        enable_thinking=False,
        reasoning_effort=None,
        maximum_output_tokens=4_096,
        timeout_seconds=300.0,
        trace_label="e1-replay-controller",
    )
    controller = FrozenModelAffordanceController(
        backend,
        draft_backend=None,
        repair_backend=backend,
        qualification_ref=args.qualification_ref,
        maximum_output_tokens=4_096,
        maximum_repair_output_tokens=4_096,
        require_intent_candidates=True,
    )

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    results = []
    for ordinal in [int(x) for x in args.ordinals.split(",")]:
        row = con.execute(
            "SELECT episode_json FROM episodes WHERE ordinal=?", (ordinal,)
        ).fetchone()
        episode = json.loads(bytes(row[0]))
        context = json.loads(episode["choice"]["context_json"])
        parent_ref = episode["parent_state_ref"]
        blob_row = con.execute(
            "SELECT child_state_blob FROM learned_updates WHERE child_state_ref=?",
            (parent_ref,),
        ).fetchone()
        if blob_row is None:
            raise RuntimeError(f"parent state blob absent for ordinal {ordinal}")
        state_payload = load_state_payload(blob_row[0])

        observation = from_recorded(CycleObservation, episode["observation"])
        temporal = from_recorded(TemporalNow, context["temporal_now"])
        memories = [
            from_recorded(MemoryCandidate, item) for item in context["memories"]
        ]
        experience = StructuredExperience.from_mapping(context["experience"])
        capability_modules = context.get("cognitive_state", {}).get(
            "capability_modules", []
        )
        lived = episode["choice"]["selected_affordance_id"]

        for arm in [item.strip() for item in args.arms.split(",") if item.strip()]:
            state_bytes, arm_memories = ablate(state_payload, memories, arm)
            for rep in range(1, args.reps + 1):
                record = {
                    "ordinal": ordinal,
                    "arm": arm,
                    "rep": rep,
                    "lived_selection": lived,
                    "lived_hold": (
                        json.loads(episode["choice"]["context_json"])
                        .get("human_hold")
                    ),
                    "memories_presented": len(arm_memories),
                }
                try:
                    selection = controller.select(
                        observation=observation,
                        temporal=temporal,
                        affordances=(CORTEX_AFFORDANCE, LIBRARY_AFFORDANCE),
                        memories=arm_memories,
                        experience=experience,
                        state=state_bytes,
                        capability_modules=capability_modules,
                        autonomous_formation=None,
                    )
                    record["replay_selection"] = selection.selected_affordance_id
                    record["payload_excerpt"] = str(selection.action_payload)[:160]
                    record["human_hold"] = selection.human_hold
                except ControllerOutputContractExhausted as exc:
                    record["replay_selection"] = "CONTRACT_FAIL"
                    record["error"] = str(exc)[:200]
                results.append(record)
                hold = record.get("human_hold") or {}
                print(
                    f"ordinal={ordinal} arm={arm} rep={rep} "
                    f"lived={lived} replay={record['replay_selection']} "
                    f"hold={hold.get('status')}:{str(hold.get('statement'))[:40]}",
                    flush=True,
                )

    artifact = {
        "contract": "jenny.e1-removal-arms.v1",
        "leaf": "ANG-WORK-RUNTIME-EMOTION-E1-CAUSAL-WORK-V1-001",
        "epistemic_status": (
            "OFFLINE_FIXED_INPUT_REPLAY_NO_STATE_MUTATION_NO_LIVE_IDENTITY"
        ),
        "lesson_era_start_ordinal": LESSON_ERA_START,
        "ablations": {
            "A'": "none (fidelity check)",
            "B": "grounded_appraisal_dynamics removed from presented state",
            "C": ("qualitative_feedback emptied and lesson-era retrieved "
                  "memories (acquired_ordinal >= 113) removed"),
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
