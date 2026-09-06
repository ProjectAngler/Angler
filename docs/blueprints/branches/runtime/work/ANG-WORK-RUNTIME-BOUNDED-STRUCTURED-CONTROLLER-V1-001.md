---
work_id: ANG-WORK-RUNTIME-BOUNDED-STRUCTURED-CONTROLLER-V1-001
parent_id: ANG-BP-RUNTIME
status: complete
created_at: 2026-09-04
---

# Bounded structured controller V1

## Outcome

Make one learned affordance decision fit its declared inference boundary and
remain syntactically executable. Candidate generation stays variable and
model-authored; deterministic code may bound and validate the envelope but may
not select, score, rewrite, or complete an operation.

## Evidence prompting this leaf

The first live post-appraisal heartbeat left canonical state and the pending
slot unchanged, then exhausted three consecutive 4,096-token controller
outputs. The draft, deliberation, and repair inputs grew from roughly 25K to
32K tokens and the final JSON remained incomplete. This is a capacity/contract
failure upstream of the new grounded-appraisal update.

## Bounded scope

- `src/angler/runtime/frozen_cognitive_models.py`
- `src/angler/runtime/higher_level_autonomy_adapter.py`
- `src/angler/runtime/jenny2_runtime.py`
- focused runtime tests
- this leaf and append-only `AGENTS_SYNC.md`

No model weights, adapter bytes, canonical Jenny state, Cognee records, Moving
Origin, stored episode, threshold, permission, or external-effect boundary may
be changed.

## Acceptance

- A JSON-capable serving backend uses a strict schema for the learned choice.
- Intent candidates remain a model-chosen variable set from one through a
  declared capacity tied to the completion boundary; no padding or fixed count.
- The live assembly performs one primary means-selection pass. At most one
  bounded repair remains for a genuine contract miss; there is no automatic
  draft -> deliberation -> repair cascade.
- A scheduler target already authored from state, memory, and Moving Origin is
  reused as the structured experience instead of invoking the same model to
  restate it. When the controller selects private `cortex.respond`, its exact
  validated conclusion is the execution output; no second same-evidence cortex
  call is made.
- An unevaluated private action is retained without utility credit and without
  a model-authored self-appraisal pass. Later scalar human feedback immediately
  updates bounded persistent policy/outcome state without a synchronous model
  call; neural or capability consolidation remains explicitly deferred.
- Invalid or incomplete output never commits state.
- Focused and complete runtime tests pass.
- A live heartbeat commits exactly once, persists grounded-appraisal state,
  uses no more than target-formation plus means-selection model calls, and
  reports no controller or projection fault.

## Rollback

Restore only this leaf's code/test edits and restart `jenny2-api.service`; the
pre-existing canonical database is the unchanged rollback state.

## Result

The live production path now uses exactly two model calls: one tool-blind
target-formation call and one schema-constrained means-selection call. It
reuses the target as the structured experience, reuses a selected private
`cortex.respond` conclusion as the execution result, performs no same-evidence
self-appraisal call, and awards no utility or capability credit to an
unevaluated result.

The means-selection prompt now receives a bounded, hash-linked working set for
seven historical channels while the complete canonical records remain in state
and Cognee remains the semantic route to older relevant evidence. The live
ordinal-101 witness committed exactly once with no retry, pending operation,
life error, or projection error. Compared with the immediately preceding
two-call witness, aggregate prompt input fell from 39,362 to 25,141 tokens
(36.13%). Generation varied upward from 1,463 to 2,374 tokens, so observed wall
time moved only from 84.059853 to 81.201192 seconds; this separates a real
prefill reduction from stochastic completion-length variance.

Validation: focused adapter tests 60/60 PASS; complete runtime tests 475/475
PASS in 52.406 seconds; `git diff --check` PASS. The service restart preserved
ordinal 100 and state ref
`sha256:e75abe88238d93e616ec364e1fef6784d01dec1319eb1b1208414762030926b9`
before the witness. The witness advanced once to ordinal 101, episode
`sha256:86dae695d6f56e3b4b6a0546527c608e4c479a901f8267af68c8f6809ff4852d`,
and state ref
`sha256:6ff2fed4ca98e04996093f468d67fdd1f3fa5e437f3ea184b7255f9fdba1e6f9`.
Grounded appraisal advanced to five observations; the result remained
`COMPLETED_UNEVALUATED`, and neural consolidation remained deferred.

The remaining root inefficiency is no longer hidden: the current commitment
contract conflates “this remains important” with “perform an operation now.”
The model therefore emitted another response even while explicitly concluding
that no new evidence justified action. That semantic coupling belongs to the
event-driven successor leaf rather than another token-limit patch.
