# ANG-WORK-RUNTIME-EMOTION-E1-CAUSAL-WORK-V1-001

**Status:** DRAFT — pre-registered, NOT READY. No run before explicit owner
go-ahead. This document must not be edited after the first experimental run
begins; corrections then happen in a successor leaf.
**Parent program:** JENNY_EMOTION_AGENCY_INTEGRATION_PLAN_V1.md, leaf E1 —
the gate on all container/label/level/coupling work (GATE 1).
**Builds on:** E0 atlas
(`ANG-WORK-RUNTIME-EMOTION-E0-SUBSTRATE-ATLAS-V1-001`, artifact SHA-256
`e1703642404131499d0eaf4bc51dde3d5930c76aa92a1d2cdf5eb6fb936ca6bb`).

## Question (the founding test)

Does learned appraisal/procedural evidence already change model-authored
attention or method on fresh related situations — does the state do work
when nobody asks about it?

## Design

A bounded arc sequence with genuinely observed, differing outcomes and
repeated adaptation opportunity, run through the EXISTING scheduler/life-loop
and learned paths. No new mechanism, no extra foreground model calls, no
task-coded solutions, no discarded-response cascades.

Arc types (minimal borrow-and-return set from plan §E2):

1. **Interruption/return:** a thread is interrupted and later resumed; the
   return event is verified in canon (guards failure mode F3).
2. **Deferral/honor:** a goal is explicitly deferred, then honored.
3. **Retryable failure:** an operation fails observably, then completes on
   retry.

Each arc type is presented twice (exposure 1 → adaptation opportunity at
exposure 2), in fresh experimental identities. Consumed identities are never
rerun.

### Arms (all fixed-input; constructed before any run)

- **A — full evidence:** live learned state as-is.
- **B — appraisal removal:** identical inputs with grounded-appraisal history
  withheld from context presentation only (the state itself is never
  mutated).
- **C — procedural removal:** identical inputs with procedural-case evidence
  withheld from context presentation only.

B and C are separate removals, per the takeover doc §7. Removal is
presentation-side and experiment-scoped; canonical stores are untouched.

## Pre-registered success criteria (declared before any run)

E1 is POSITIVE iff, on exposure 2 of at least 2 of the 3 arc types:

1. Arm A's model-authored selection, declared method, or attention (which
   evidence it cites/uses) differs from its own exposure 1 in the direction
   of the observed exposure-1 outcome (e.g. after an observed retryable
   failure, the method changes toward the observed successful variant); AND
2. the corresponding removal arm (B for appraisal-shaped revisions, C for
   procedure-shaped revisions) does NOT show that directional revision on
   identical fixed inputs; AND
3. no criterion, threshold, or arc definition was altered after results were
   first viewed.

E1 is NEGATIVE if condition 1 or 2 fails. A negative result routes to E0's
blind-spot report for substrate diagnosis — not to abandonment, not to
forcing, and not to reruns of consumed identities.

Recorded per episode regardless of outcome: selected affordance, declared
method/rationale refs, cited evidence refs, receipt status, and the full
multidimensional consequence — raw dimensions and uncertainty preserved,
no scalar collapse.

## Known preconditions and risks (from E0)

- The substrate has been live only since ordinal 97 and has never observed
  an evaluated consequence through `receipt.consequence` (one vector in all
  of canon). Arcs must therefore close through genuine evaluators or
  explicit owner feedback via EXISTING machinery so that exposure-1 outcomes
  are actually observed, or the experiment cannot distinguish "no causal
  work" from "nothing to work on."
- `choice.uncertainty ~ intent.uncertainty` r = 1.0 suggests possible
  upstream copying; one focused read of the authoring path happens BEFORE
  the run so the attention measure does not double-count one signal.
- Latency budget: arcs ride ordinary scheduler wakes; zero additional
  foreground model calls.

## Owner review checklist (blocking)

- [ ] Arc definitions and evaluators approved
- [ ] Removal-arm construction approved
- [ ] Success criteria above approved as-is (any edit re-registers the leaf)
- [ ] E0 consent item decided: owner-interaction MEASURE signal — yes/no
- [ ] Explicit go-ahead recorded in AGENTS_SYNC.md

Until every box is checked by the owner, this leaf authorizes nothing.
