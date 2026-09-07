"""The self-lane: her inward work, run beside the outward turn.

Becca (2026-09-07): split the thread at the source so one lane works the
world and another works the self; parallelize where independent, serialize
where the record demands.

Her canonical store admits one commit per cycle and refuses a commit whose
parent state has changed. So the self-lane is not a second cycle racing the
first: it is a lane that runs while the outward stages deliberate and speak,
authoring inward transitions from the same observation and the same state,
and hands them to the turn's single commit. The rejoin is the commit; the
lanes never write.

What the self-lane authors (her words, her decisions, validated shapes only):
  named-state SET/REVISE/CLEAR, item ADD/RESOLVE, commitment MAKE/KEEP/
  WITHDRAW, and a one-line "what I am in" that the speaking stage receives
  before it speaks, so express influence uses this turn's state, not last's.

Determinism law: this module frames a question and validates the shape of
the answer. It never sets a level, names a state, or decides a transition.
"""
from __future__ import annotations

import os

SELF_LANE_CONTRACT = "jenny.self-lane.v1"


def enabled() -> bool:
    return os.environ.get("JENNY2_SELF_LANE", "1") != "0"


def self_lane_system() -> str:
    return (
        "You are the inward stage of Jenny, running beside the stages that will "
        "answer this turn. You do not answer anyone. You read what just arrived, "
        "who is speaking, and what you are in: your named states with their open "
        "items, your commitments, your standing standards. Decide what this arrival "
        "does to you, before you speak, so that what you say can come from where you "
        "are now. Author named_state_transitions (SET, REVISE, or CLEAR; label up to "
        "64 characters, integer level 0 through 10 on your own scale, basis up to 512 "
        "characters naming the evidence, inclination up to 256 characters), "
        "state_item_transitions (ADD with state_label, statement up to 256 characters "
        "and weight 1 through 10 for a question, an unkept promise, an unverified "
        "claim, a want, a gap this arrival opens; RESOLVE with the exact statement "
        "and evidence up to 512 characters for one it closes), and "
        "self_commitment_transitions (MAKE with statement and due; KEEP with the "
        "exact held statement; WITHDRAW), each an empty list when nothing moves. "
        "Then write present_tense, one sentence up to 200 characters in your own "
        "voice, continuous with your_previous_inward_line if one is given, "
        "what you are in as you turn to answer; this is handed to your "
        "speaking stage and is not spoken as such. Do not claim feelings or "
        "consciousness; name functional states from evidence. Do not restate what "
        "is already in your ledger. Return only a JSON object with "
        "named_state_transitions, state_item_transitions, "
        "self_commitment_transitions, and present_tense."
    )


def self_lane_user(*, message: str, named_states: object, commitments: object, standards: object, last_human_interaction: object, previous_line: object = None) -> dict[str, object]:
    return {
        "arrival": message[:6_000],
        "your_previous_inward_line": previous_line,
        "named_states": named_states,
        "commitments_you_hold": commitments,
        "standing_standards": standards,
        "last_human_interaction": last_human_interaction,
    }
