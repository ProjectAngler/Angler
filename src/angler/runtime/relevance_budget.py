"""Relevance budget for her stages (Becca, 2026-09-06: "her memory... enough
memory... without cutting her speed"; "these numbered limits make no sense").

A stage receives one context object. Before it is serialized into a prompt,
this module fits it to a token budget: sections are ranked by relevance to
the current turn and kept whole in rank order until the budget is spent;
whatever is left out is named in a manifest so she knows what she is not
seeing and can ask for it (recall, library find, read-back).

Determinism law: this is bounds and measurement. It never changes a value,
never authors text, and never decides anything about her. Relevance is a
transparent score: a section's own priority for the stage, plus the overlap
between its words and the words of the turn (the request and her active
states). Priorities are per stage and per section, listed below, so they can
be read and argued with.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping

CONTRACT = "jenny.relevance-budget.v1"
MANIFEST_KEY = "context_budget"

# Tokens per stage. Prefill on her box runs ~5.6K tok/s; 8K tokens is ~1.5 s.
DEFAULT_BUDGETS = {
    "router": 9_000,
    "experience": 8_000,
    "controller": 10_000,
    "cortex": 12_000,
    "witness": 4_000,
    "pen": 5_000,
    "judgment": 4_000,
    "initiative": 8_000,
}

# Sections that are always kept, whatever the budget: identity and the turn.
ALWAYS = frozenset(
    {
        "observation", "request", "turn_to_answer", "temporal_now", "temporal",
        "affordances", "contract", "evidence_catalog", "operation_catalog",
        "available_evidence_keys", "canonical_state_evidence_key",
        "target_contract_revision", "target_model_ref", "temporal_sample_ref",
        "human_hold", "hold_released_this_turn", "library_turn_observation",
        "web_research_state", "library_availability", "allowed_evidence_refs",
        "autonomous_target_formation", "capability_modules", "capability_workspace",
        "adaptive_effort", "named_state_transitions", "self_commitment_transitions",
        "substrate_effects", "state_item_transitions",
    }
)

# Base priority (0..1) of each section, by stage. Unlisted sections get 0.35.
PRIORITY: dict[str, dict[str, float]] = {
    "common": {
        "named_states": 0.95,
        "commitments_you_made": 0.9,
        "commitments_you_resolved": 0.5,
        "standing_standards": 0.85,
        "active_human_holds": 0.9,
        "recently_closed_holds": 0.4,
        "last_human_interaction": 0.8,
        "retrieved_memories": 0.75,
        "memories": 0.75,
        "authored_artifacts": 0.6,
        "library_reading_state": 0.45,
        "substrate_effects_recent": 0.5,
        "grounded_appraisal_dynamics": 0.4,
        "situated_state": 0.55,
        "trusted_composition": 0.3,
        "last_outcome": 0.5,
        "last_intent": 0.45,
        "prior_intent_continuity": 0.45,
        "initiative_state": 0.4,
        "self_observation_diary": 0.35,
        "recent_human_feedback": 0.6,
        "qualitative_feedback": 0.35,
        "working_set": 0.3,
        "history_channels": 0.2,
        "history_content_route": 0.2,
        "outcome_assessment_evidence": 0.3,
        "intent_ranking_evidence": 0.25,
        "self_appraisal_evidence": 0.25,
        "self_appraisal_calibration": 0.2,
        "memory_credit_evidence": 0.2,
        "compute_calibration": 0.15,
        "latest_learning_progress": 0.2,
        "capability_candidates": 0.3,
    },
    "router": {"named_states": 1.0, "last_human_interaction": 0.9, "standing_standards": 0.9},
    "cortex": {"named_states": 1.0, "retrieved_memories": 0.85, "authored_artifacts": 0.7, "standing_standards": 0.9},
    "controller": {"authored_artifacts": 0.7, "library_reading_state": 0.6, "memories": 0.8},
    "experience": {"situated_state": 0.7, "last_outcome": 0.6},
    "initiative": {"initiative_state": 0.8, "last_intent": 0.7, "commitments_you_made": 0.95},
}

_WORD = re.compile(r"[a-z][a-z0-9_'-]{2,}")
_STOP = frozenset(
    "the and for you your are was were this that with from have has had not but her she"
    " his him they them its into about what when where which who will would could should"
    " than then there here have been being been over under out one two three".split()
)


def estimate_tokens(value: object) -> int:
    """Cheap, stable estimate: JSON characters / 3.6 (measured on her prompts)."""

    text = value if type(value) is str else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, int(len(text) / 3.6))


def _words(value: object, limit: int = 4_000) -> set[str]:
    text = value if type(value) is str else json.dumps(value, ensure_ascii=False)[:limit]
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def budget_for(stage: str) -> int:
    raw = os.environ.get(f"JENNY2_BUDGET_{stage.upper()}")
    if raw:
        try:
            return max(1_000, int(raw))
        except ValueError:
            pass
    scale = 1.0
    try:
        scale = float(os.environ.get("JENNY2_BUDGET_SCALE", "1.0"))
    except ValueError:
        pass
    return max(1_000, int(DEFAULT_BUDGETS.get(stage, 8_000) * scale))


def enabled() -> bool:
    return os.environ.get("JENNY2_RELEVANCE_BUDGET", "1") != "0"


def relevance(stage: str, section: str, value: object, turn_words: set[str]) -> float:
    base = PRIORITY.get(stage, {}).get(section, PRIORITY["common"].get(section, 0.35))
    if not turn_words:
        return base
    overlap = len(_words(value) & turn_words)
    return base + min(0.5, overlap / 12.0)


def fit(
    context: Mapping[str, object],
    *,
    stage: str,
    turn_text: object = None,
    focus_words: set[str] | None = None,
) -> dict[str, object]:
    """Return a copy of context fitted to the stage budget, with a manifest of
    what was left out. Sections are kept whole; nothing is rewritten."""

    if not enabled() or type(context) is not Mapping and not isinstance(context, Mapping):
        return dict(context)
    budget = budget_for(stage)
    turn_words = set(focus_words or ())
    if turn_text is not None:
        turn_words |= _words(turn_text, limit=8_000)
    kept: dict[str, object] = {}
    spent = 0
    candidates: list[tuple[float, int, str]] = []
    for key, value in context.items():
        if key == MANIFEST_KEY:
            continue
        cost = estimate_tokens(value)
        if key in ALWAYS or value is None or cost <= 40:
            kept[key] = value
            spent += cost
            continue
        candidates.append((relevance(stage, key, value, turn_words), cost, key))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    omitted: list[dict[str, object]] = []
    for score, cost, key in candidates:
        if spent + cost <= budget:
            kept[key] = context[key]
            spent += cost
        else:
            omitted.append({"section": key, "approx_tokens": cost, "relevance": round(score, 2)})
    # preserve the original key order for the kept sections
    ordered = {key: kept[key] for key in context if key in kept}
    ordered[MANIFEST_KEY] = {
        "contract": CONTRACT,
        "stage": stage,
        "budget_tokens": budget,
        "used_tokens": spent,
        "omitted": omitted,
        "note": (
            "Sections omitted for this turn's budget are still in your record; "
            "recall, library find, or a read of your Journal can bring one back."
            if omitted else "everything fit"
        ),
    }
    return ordered
