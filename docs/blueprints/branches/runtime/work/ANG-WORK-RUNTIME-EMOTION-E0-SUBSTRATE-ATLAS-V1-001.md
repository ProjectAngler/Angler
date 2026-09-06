# ANG-WORK-RUNTIME-EMOTION-E0-SUBSTRATE-ATLAS-V1-001

**Status:** COMPLETE (read-only; no runtime effect)
**Parent program:** JENNY_EMOTION_AGENCY_INTEGRATION_PLAN_V1.md, leaf E0,
merged with the takeover doc's suggested read-only audit
(`JENNY2-CLAUDE-TAKEOVER-2026-09-04` §13).
**Executor:** Claude (coordinator session, via SSH), 2026-09-04.
**Accountable outcome:** know exactly which numeric event signals feed
grounded appraisal, with provenance, distributions, recurring unnamed
signatures, and blind spots — before any question about what any of it means.

## Scope and non-goals

In scope: read-only audit of recorded closed-loop/latency evidence; offline
analysis tooling (`experiments/analysis/substrate_atlas_v1.py`) reading the
canonical SQLite store in `mode=ro`; rebuildable artifact
(`artifacts/substrate-atlas-v1/atlas.json`); append-only `AGENTS_SYNC.md`
reporting. Non-goals, honored: no runtime, service, model, permission,
canonical-store, or projection change; no restarts; no model calls; no naming
of clusters; no threshold tuning of anything live; no new substrate or
logger — `grounded_appraisal.py` is the substrate.

Owner directive recorded 2026-09-04: the `affective_substrate` event-substrate
corpus is a separate effort the owner has designated failed and to be ignored.
This atlas therefore reads only `episodes` and the live
`grounded_appraisal_dynamics` state. No change to that table or its writer was
made under this leaf.

## Audit results (recorded evidence vs. re-verification)

- Event-driven-wake receipts: `first.json` SHA-256 `187b7e…d57efd6f` and
  `second.json` `938f05…c5ed6d5` match the values in `AGENTS_SYNC.md` exactly.
- Focused procedural/Cognee/adapter/runtime suite: 91/91 PASS (5.081 s),
  matching the recorded 91/91.
- Complete runtime suite: 480/480 PASS (54.172 s), matching the recorded
  480/480.
- Service healthy throughout: ready, scheduler + life loop running, no
  pending operation/projection, no life/projection error. State advanced
  legitimately 106 → 112 between the handoff snapshot and this leaf's claim;
  ordinal 112 / `sha256:c6aa7d92…411614` held stable across the leaf.
- Latency witnesses consistent with recorded behavior: unchanged-head wake
  suppression observed live (ordinal stable across repeated checks);
  per-episode `phase_timings_ms` present in canon (e.g. ordinal 112
  `choose_total` 11 863 ms, one FAST_RESPONSE call).

No contradiction between recorded evidence and the live system was found.

## Atlas findings (from `artifacts/substrate-atlas-v1/atlas.json`)

1. **Provenance proof of substrate activation.** Replaying the exact online
   update math over the most recent 16 projectable episodes from an empty
   state reproduces the live `feature_moments` bit-for-bit (rel tol 1e-9)
   and the live `observation_count` of 16. Grounded appraisal has been live
   since ordinal 97. All 113 canonical episodes project valid signal sets
   under today's projection code.
2. **Signal enumeration.** 22 signal names observed across history; 13 in the
   live state. The 9 `consequence.*` dimensions
   (constraint_satisfaction, cost, evidence_quality, human_feedback,
   information_gain, objective_progress, prediction_error, reuse_value,
   safety) occur in exactly ONE episode in all of canon. Full per-signal
   provenance is recorded in the artifact.
3. **Redundancy.** `choice.ranking_count ~ choice.score_spread` r = 1.0
   (n = 113) and `choice.uncertainty ~ intent.uncertainty` r = 1.0 (n = 112):
   the effective sensed dimensionality is materially lower than the signal
   count. Whether intent uncertainty is independently authored or copied
   upstream deserves one focused look before E1 interprets either.
4. **Recurring unnamed signatures** (content-hash IDs, |z| ≥ 1.5, basis ≥ 3;
   descriptive thresholds declared before viewing results): 9 recurring
   patterns, including `signature-6c1315aad0f4` — nine consecutive episodes
   (ordinals 69–77) of simultaneously high ranking count, score spread, and
   intent alternatives — and `signature-bbf5fa68e1ff` (5 episodes: those plus
   high choice/intent uncertainty). Noticed, not named.
5. **Blind spots.**
   - Evaluated consequence starvation: 95/113 episodes are
     HUMAN + COMPLETED_UNEVALUATED with no consequence rows; 7 COMPLETED
     total; 1 consequence vector ever. Multidimensional
     `observed_consequence` vectors DO exist in cognitive-state evidence
     channels (`compute_calibration`, `memory_credit_evidence`,
     `latest_learning_progress`) but are not routed through
     `receipt.consequence`, so the substrate has effectively never observed
     an evaluated consequence. This is the concrete mechanism behind the
     plan's E2 premise.
   - Owner-relational channel: absent. `qualitative_feedback` and
     `FEEDBACK_ON` lineage exist in canon but feed no appraisal signal.
     Per plan §E0 any such signal is an owner-consent decision, not an
     engineering default. Left for the owner.
   - `temporal.clock_uncertainty_ms` is constant (1.0) to date —
     uninformative so far.

## Limitations

- Historical episodes are projected with the current
  `event_appraisal_signals` code; the atlas describes what the substrate
  would sense in today's terms, which for the pre-ordinal-97 era is a
  reconstruction, not lived appraisal history.
- Signature thresholds are descriptive reporting choices for this atlas
  only; nothing live consumes them.
- The atlas is rebuildable at any time by re-running the script; the JSON
  artifact is not canonical state.

## Next action

`ANG-WORK-RUNTIME-EMOTION-E1-CAUSAL-WORK-V1-001` drafted with pre-registered
success criteria and separate fixed-input removal controls. DRAFT status —
no run occurs before explicit owner review and go-ahead (program GATE 1).
