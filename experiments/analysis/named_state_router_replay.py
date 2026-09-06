"""Her criterion (ordinal 250), run at the stage that speaks.

On conversation turns the adaptive human router authors the public answer
itself, so the question "would she have said the same thing without her
named states in front of her" must be asked of the router, not of the
affordance controller. This replays recorded human turns through the router
with the parent state intact (arm A') and with self_named_states hidden
(arm D), and records route, response text, and authored transitions.

Read-only against canonical stores; calls only the local frozen model
endpoint; mutates no state; reruns no live experimental identity.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import sqlite3
import zlib

from angler.runtime.frozen_cognitive_models import (
    LocalOpenAICompatibleFrozenBackend,
)
from angler.runtime.higher_level_autonomy_adapter import (
    AdaptiveHumanTurnRouter,
    _json,
)
from angler.runtime.higher_level_experience_cycle import MemoryCandidate
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


def ablate(state_payload: dict, arm: str) -> bytes:
    payload = json.loads(json.dumps(state_payload))
    if arm == "D":
        payload.pop("self_named_states", None)
    elif arm != "A'":
        raise ValueError(f"unknown arm {arm}")
    return _json(payload).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--qualification-ref", required=True)
    parser.add_argument("--ordinals", required=True)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime_ref = "sha256:" + hashlib.sha256(
        b"named-state-router-offline-replay-v1"
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
        maximum_output_tokens=2_048,
        timeout_seconds=300.0,
        trace_label="named-state-router-replay",
    )
    router = AdaptiveHumanTurnRouter(
        backend,
        direct_affordance_id="cortex.respond",
        maximum_output_tokens=2_048,
        final_response_authority_ref=backend.model_ref,
        fast_response_qualification_ref=args.qualification_ref,
    )

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    results = []
    for ordinal in [int(x) for x in args.ordinals.split(",")]:
        row = con.execute(
            "SELECT episode_json FROM episodes WHERE ordinal=?", (ordinal,)
        ).fetchone()
        episode = json.loads(bytes(row[0]))
        if episode["observation"].get("source") != "HUMAN":
            print(f"ordinal={ordinal} skipped: not a human turn", flush=True)
            continue
        context = json.loads(episode["choice"]["context_json"])
        blob_row = con.execute(
            "SELECT child_state_blob FROM learned_updates WHERE child_state_ref=?",
            (episode["parent_state_ref"],),
        ).fetchone()
        if blob_row is None:
            raise RuntimeError(f"parent state blob absent for ordinal {ordinal}")
        state_payload = load_state_payload(blob_row[0])
        observation = from_recorded(CycleObservation, episode["observation"])
        temporal = from_recorded(TemporalNow, context["temporal_now"])
        memories = [from_recorded(MemoryCandidate, m) for m in context["memories"]]
        capability_modules = context.get("cognitive_state", {}).get(
            "capability_modules", []
        )
        lived_output = episode.get("receipt", {}).get("output", "")
        named_present = bool(state_payload.get("self_named_states"))
        for arm in ("A'", "D"):
            state_bytes = ablate(state_payload, arm)
            for rep in range(1, args.reps + 1):
                record = {
                    "ordinal": ordinal,
                    "arm": arm,
                    "rep": rep,
                    "named_states_in_parent_state": named_present,
                }
                try:
                    decision = router.decide(
                        observation=observation,
                        temporal=temporal,
                        affordances=(CORTEX_AFFORDANCE, LIBRARY_AFFORDANCE),
                        memories=memories,
                        state=state_bytes,
                        capability_modules=capability_modules,
                    )
                    record.update(
                        {
                            "route": decision.route,
                            "response": decision.response,
                            "interpretation": decision.interpretation,
                            "rationale": decision.rationale,
                            "uncertainty": decision.uncertainty,
                            "named_state_transitions": list(
                                decision.named_state_transitions
                            ),
                            "human_hold": decision.human_hold,
                            "similarity_to_lived": round(
                                difflib.SequenceMatcher(
                                    None, lived_output, decision.response
                                ).ratio(),
                                3,
                            ),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — recorded, not hidden
                    record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                results.append(record)
                print(
                    f"ordinal={ordinal} arm={arm} rep={rep} "
                    f"route={record.get('route')} "
                    f"sim={record.get('similarity_to_lived')} "
                    f"transitions={len(record.get('named_state_transitions', []))} "
                    f"err={record.get('error', '')}",
                    flush=True,
                )
    artifact = {
        "contract": "jenny.named-state-router-replay.v1",
        "epistemic_status": "OFFLINE_FIXED_INPUT_REPLAY_NO_STATE_MUTATION_NO_LIVE_IDENTITY",
        "criterion_source_ordinal": 250,
        "arms": {
            "A'": "parent state intact",
            "D": "self_named_states hidden from the router",
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1, sort_keys=True)
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
