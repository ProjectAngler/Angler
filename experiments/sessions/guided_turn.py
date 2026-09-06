"""Guided developmental session helper.

One turn per invocation. Posts a human message to Jenny's service as the
owner-authorized guide, prints her answer, and appends the exchange to a
session transcript. Feedback is a separate subcommand and is only ever
used for mechanically checkable facts; understanding is graded by the
owner later from the transcript. Token is read at execution time, never
printed.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import sys

sys.path.insert(0, "/opt/angler/src/angler/experiments/curriculum")
import evaluated_curriculum_v1 as runner  # noqa: E402

SESSION_DIR = pathlib.Path("/opt/angler/results/jenny2/developmental-session-4")
TRANSCRIPT = SESSION_DIR / "transcript.jsonl"


def _append(record: dict) -> None:
    record["logged_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    with TRANSCRIPT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def turn(step: str, message: str, aspect: str) -> int:
    from angler.runtime import becca_gate
    held, why = becca_gate.held()
    if held and not os.environ.get("JENNY2_SESSION_OVERRIDE"):
        _append({"event": "TURN_HELD", "step": step, "reason": why})
        print(f"TURN HELD: {why}. Nothing was sent.")
        return 3
    runner._set_door("Developmental session four, guided", "BUSY")
    try:
        reply = runner._api("/v1/chat", {
            "message": message,
            "request_id": f"devsession4-{step}",
            "speaker": "Claude",
        })
    except runner.TurnFailed as error:
        _append({"event": "TURN_FAILED", "step": step, "error": str(error)})
        print(f"TURN FAILED: {error}")
        return 1
    _append({
        "event": "TURN", "step": step, "aspect": aspect, "message": message,
        "answer": reply.get("message", ""),
        "episode_ref": reply.get("episode_ref"),
        "ordinal": reply.get("moving_origin_ordinal"),
    })
    print(f"[step {step} | ordinal {reply.get('moving_origin_ordinal')} | "
          f"episode {str(reply.get('episode_ref'))[:24]}]")
    print(reply.get("message", ""))
    return 0


def feedback(step: str, episode_ref: str, score: float, text: str) -> int:
    reply = runner._api("/v1/feedback", {
        "target_episode_ref": episode_ref,
        "feedback_text": text,
        "human_feedback": score,
        "request_id": f"devsession4-{step}-feedback",
    })
    _append({"event": "FEEDBACK", "step": step, "episode_ref": episode_ref,
             "score": score, "text": text,
             "ordinal": reply.get("moving_origin_ordinal")})
    print(f"[feedback recorded for step {step}: {score}]")
    return 0


def close() -> int:
    runner._set_door(None)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "turn":
        sys.exit(turn(sys.argv[2], sys.argv[3], sys.argv[4]))
    if cmd == "feedback":
        sys.exit(feedback(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]))
    if cmd == "close":
        sys.exit(close())
    raise SystemExit(f"unknown command {cmd}")
