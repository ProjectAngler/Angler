"""Substrate atlas V1 — leaf E0 (read-only, offline, analysis only).

Reads canonical episode history plus the live grounded-appraisal state and
produces, without touching any runtime file, service, projection, or store:

  (a) an enumeration of every numeric event signal actually feeding
      grounded appraisal, with code provenance and observation counts;
  (b) per-signal distributions and a replayed innovation timeline;
  (c) recurring innovation signatures across episodes — noticed, not named;
  (d) a blind-spot report: life-event classes that move no signal.

Epistemic discipline: every quantity here is an empirical projection of
canonical bytes. Nothing in this file assigns emotion labels, valence,
reward, or behavior. Signature identifiers are content hashes, not names.
No episode text content is copied into the atlas — refs and numbers only.

Usage (from the repository root):
  python3 experiments/analysis/substrate_atlas_v1.py \
      --db /opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/jenny2.sqlite3 \
      --out artifacts/substrate-atlas-v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import statistics
import sys
import zlib

SIGNATURE_Z_THRESHOLD = 1.5
SIGNATURE_MIN_BASIS = 3

# Code provenance of each signal family, from src/angler/runtime/
# grounded_appraisal.py::event_appraisal_signals and its single live caller
# (higher_level_autonomy_adapter.py, update_grounded_appraisal call site).
SIGNAL_PROVENANCE = {
    "choice.uncertainty": "episode.choice.uncertainty (model-declared)",
    "choice.ranking_count": "len(episode.choice.rankings)",
    "choice.selected_score": "episode.choice.rankings[selected].score (model/controller-declared)",
    "choice.score_spread": "max-min over episode.choice.rankings scores",
    "intent.uncertainty": "choice.context_json.intent_proposal.uncertainty (model-declared)",
    "intent.alternative_count": "len(choice.context_json.intent_proposal.alternatives)",
    "retrieval.count": "len(choice.context_json.memories with semantic_distance)",
    "retrieval.nearest_distance": "min semantic_distance over context memories (Cognee-recorded)",
    "retrieval.mean_distance": "mean semantic_distance over context memories",
    "retrieval.distance_spread": "max-min semantic_distance over context memories",
    "commitment.uncertainty": "context.autonomous_initiative.formation.proposal.commitment.uncertainty",
    "commitment.evidence_count": "len(...commitment.evidence_keys)",
    "consequence.*": "episode.receipt.consequence rows (dynamic names; none observed to date)",
    "temporal.clock_uncertainty_ms": "episode.temporal.uncertainty_ms (trusted clock)",
}


def load_grounded_appraisal(repo_root: str):
    path = os.path.join(repo_root, "src", "angler", "runtime", "grounded_appraisal.py")
    spec = importlib.util.spec_from_file_location("ga_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_signal_inputs(episode: dict) -> dict:
    choice = episode["choice"]
    context = json.loads(choice.get("context_json") or "{}")
    return {
        "choice": choice,
        "receipt": episode["receipt"],
        "context": context,
        "temporal": episode["temporal"],
    }


def classify_episode(episode: dict) -> dict:
    receipt = episode.get("receipt", {})
    consequence = receipt.get("consequence") or []
    context = {}
    try:
        context = json.loads(episode.get("choice", {}).get("context_json") or "{}")
    except (TypeError, ValueError):
        pass
    initiative = context.get("autonomous_initiative")
    return {
        "source": episode.get("observation", {}).get("source", "UNKNOWN"),
        "receipt_status": receipt.get("status", "UNKNOWN"),
        "consequence_rows": len(consequence),
        "commitment_present": bool(initiative),
        "memories_offered": len(context.get("memories") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ga = load_grounded_appraisal(args.repo)
    con = sqlite3.connect(f"file:{os.path.abspath(args.db)}?mode=ro", uri=True)

    # Live aggregated state, for the consistency check.
    blob = con.execute("SELECT state_blob FROM supervisor_state").fetchone()[0]
    raw = bytes(blob)
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = json.loads(zlib.decompress(raw))
    live = payload.get(ga.GROUNDED_APPRAISAL_STATE_KEY)

    rows = con.execute(
        "SELECT ordinal, episode_ref, episode_json FROM episodes ORDER BY ordinal"
    ).fetchall()

    projected = []           # (ordinal, episode_ref, signals)
    projection_failures = [] # (ordinal, episode_ref, error class/message)
    classes = []             # per-episode class rows
    for ordinal, episode_ref, blob in rows:
        episode = json.loads(bytes(blob))
        cls = classify_episode(episode)
        cls.update({"ordinal": ordinal, "episode_ref": episode_ref})
        classes.append(cls)
        try:
            inputs = episode_signal_inputs(episode)
            signals = ga.event_appraisal_signals(**inputs)
            projected.append((ordinal, episode_ref, signals))
            cls["projects_signals"] = True
            cls["signal_count"] = len(signals)
        except (KeyError, TypeError, ValueError) as exc:
            projection_failures.append(
                {"ordinal": ordinal, "episode_ref": episode_ref,
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            cls["projects_signals"] = False
            cls["signal_count"] = 0

    # Replay the exact online update math over all projectable episodes to get
    # a per-episode innovation timeline (the live state only keeps aggregates).
    moments: dict = {}
    relations: dict = {}
    timeline = []
    per_signal_values: dict[str, list] = {}
    for ordinal, episode_ref, signals in projected:
        innovations = {}
        for name, value in signals.items():
            previous = moments.get(name)
            innovations[name] = ga._innovation(previous, value)
            moments[name] = ga._updated_moment(previous, value)
            per_signal_values.setdefault(name, []).append((ordinal, value))
        names = sorted(signals)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                key = ga._relation_key(left, right)
                relations[key] = ga._updated_relation(
                    relations.get(key), signals[left], signals[right]
                )
        timeline.append(
            {"ordinal": ordinal, "episode_ref": episode_ref,
             "innovations": innovations}
        )

    # Consistency check: does replaying the most recent K projectable episodes
    # from an empty state reproduce the live aggregated moments exactly?
    live_match = {"checked": False}
    if live and isinstance(live.get("feature_moments"), dict):
        target = live["feature_moments"]
        target_count = live.get("observation_count")
        live_match = {"checked": True, "live_observation_count": target_count,
                      "match_found": False}
        for k in range(1, len(projected) + 1):
            trial: dict = {}
            for _, _, signals in projected[-k:]:
                for name, value in signals.items():
                    trial[name] = ga._updated_moment(trial.get(name), value)
            if set(trial) != set(target):
                continue
            ok = all(
                trial[n]["count"] == target[n]["count"]
                and math.isclose(trial[n]["mean"], target[n]["mean"],
                                 rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(trial[n]["m2"], target[n]["m2"],
                                 rel_tol=1e-9, abs_tol=1e-9)
                for n in target
            )
            if ok:
                live_match.update(
                    {"match_found": True, "replayed_tail_episodes": k,
                     "first_feeding_ordinal": projected[-k][0]}
                )
                break

    # Distributions per signal (over every projectable episode).
    distributions = {}
    for name, pairs in sorted(per_signal_values.items()):
        values = [value for _, value in pairs]
        entry = {
            "observations": len(values),
            "first_ordinal": pairs[0][0],
            "last_ordinal": pairs[-1][0],
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
        }
        if len(values) > 1:
            entry["stdev"] = statistics.stdev(values)
            entry["median"] = statistics.median(values)
        constant = len(set(values)) == 1
        entry["constant_to_date"] = constant
        distributions[name] = entry

    # Recurring innovation signatures: exact sets of (signal, direction) with
    # |z| >= threshold and sufficient basis, recurring across episodes.
    # Identified by content hash only — noticed, not named.
    signature_members: dict[str, list] = {}
    signature_sets: dict[str, list] = {}
    for row in timeline:
        marked = []
        for name, innovation in row["innovations"].items():
            z = innovation.get("standardized_innovation")
            basis = innovation.get("basis_count", 0)
            if z is None or basis < SIGNATURE_MIN_BASIS:
                continue
            if abs(z) >= SIGNATURE_Z_THRESHOLD:
                marked.append([name, "high" if z > 0 else "low"])
        if not marked:
            continue
        marked.sort()
        encoded = json.dumps(marked, separators=(",", ":"))
        sig_id = "signature-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]
        signature_sets[sig_id] = marked
        signature_members.setdefault(sig_id, []).append(
            {"ordinal": row["ordinal"], "episode_ref": row["episode_ref"]}
        )
    recurring = {
        sig_id: {"pattern": signature_sets[sig_id], "members": members}
        for sig_id, members in sorted(signature_members.items())
        if len(members) >= 2
    }
    singleton = {
        sig_id: {"pattern": signature_sets[sig_id], "members": members}
        for sig_id, members in sorted(signature_members.items())
        if len(members) == 1
    }

    # Blind spots: life-event classes present in canon that move no signal.
    def class_key(c):
        return (c["source"], c["receipt_status"],
                "consequence" if c["consequence_rows"] else "no_consequence")

    class_counts: dict = {}
    class_projecting: dict = {}
    for c in classes:
        key = class_key(c)
        class_counts[key] = class_counts.get(key, 0) + 1
        if c["projects_signals"]:
            class_projecting[key] = class_projecting.get(key, 0) + 1
    event_classes = [
        {"source": key[0], "receipt_status": key[1], "consequence": key[2],
         "episodes": count, "episodes_projecting_signals":
             class_projecting.get(key, 0)}
        for key, count in sorted(class_counts.items())
    ]

    consequence_signals = [n for n in per_signal_values if n.startswith("consequence.")]
    blind_spots = {
        "consequence_channel": {
            "finding": "receipt.consequence rows have never produced a signal",
            "consequence_signal_names_observed": consequence_signals,
            "episodes_with_consequence_rows":
                sum(1 for c in classes if c["consequence_rows"]),
            "note": ("multidimensional observed_consequence vectors exist in "
                     "cognitive-state evidence channels (compute_calibration, "
                     "memory_credit_evidence, latest_learning_progress) but are "
                     "not routed through receipt.consequence, so grounded "
                     "appraisal never observes any evaluated consequence"),
        },
        "owner_relational_channel": {
            "finding": ("no signal distinguishes owner-interaction quality; "
                        "qualitative_feedback and FEEDBACK_ON lineage exist in "
                        "canon but feed no appraisal signal"),
            "signals_referencing_owner_behavior": [],
            "decision_required": ("per integration plan §E0 this is an "
                                  "owner-consent item, not an engineering "
                                  "default"),
        },
        "constant_signals": [n for n, d in distributions.items()
                             if d["constant_to_date"]],
        "episodes_not_feeding_substrate": {
            "count": len(projection_failures),
            "of_total": len(rows),
            "reason_sample": projection_failures[:3],
        },
    }

    atlas = {
        "contract": "jenny.substrate-atlas.v1",
        "leaf": "ANG-WORK-RUNTIME-EMOTION-E0-SUBSTRATE-ATLAS-V1-001",
        "epistemic_status": ("OFFLINE_MECHANICAL_PROJECTION_OF_CANONICAL_BYTES"
                             "_NO_AFFECT_SEMANTICS_NO_RUNTIME_EFFECT"),
        "inputs": {
            "database": os.path.abspath(args.db),
            "episodes_total": len(rows),
            "episodes_projecting_signals": len(projected),
        },
        "signal_enumeration": {
            "provenance": SIGNAL_PROVENANCE,
            "signals_observed_to_date": sorted(per_signal_values),
        },
        "live_state_consistency": live_match,
        "distributions": distributions,
        "innovation_timeline": timeline,
        "recurring_innovation_signatures": recurring,
        "singleton_innovation_signatures": singleton,
        "strongest_replayed_relations":
            ga._strongest_relations(relations, maximum=20),
        "event_classes": event_classes,
        "blind_spots": blind_spots,
        "thresholds": {
            "signature_abs_z": SIGNATURE_Z_THRESHOLD,
            "signature_min_basis_count": SIGNATURE_MIN_BASIS,
            "note": ("descriptive reporting thresholds for this atlas only; "
                     "declared before results were viewed and not used by any "
                     "runtime or acceptance decision"),
        },
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "atlas.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(atlas, handle, indent=2, sort_keys=True)
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()

    summary = {
        "episodes_total": len(rows),
        "episodes_projecting": len(projected),
        "signals": len(per_signal_values),
        "recurring_signatures": len(recurring),
        "singleton_signatures": len(singleton),
        "live_match": live_match,
        "atlas_sha256": digest,
        "atlas_path": out_path,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
