---
work_id: ANG-WORK-RUNTIME-EVENT-DRIVEN-COMMITMENT-WAKE-V1-001
parent_id: ANG-BP-RUNTIME
status: complete
created_at: 2026-09-04
---

# Event-driven commitment wake V1

## Outcome

Separate a model-authored persistent commitment from a model-authored decision
that one concrete operation is justified on the present wake. A commitment may
remain unresolved while Jenny honestly waits for new evidence. The unchanged
state/event pair must then become quiescent without another controller call,
episode, semantic-time advance, or fabricated learning event.

## Evidence prompting this leaf

At ordinal 101 the target and controller both concluded that the current state
was stable, no novel affordance or evidence existed, and acting would carry
stale context forward. The contract nevertheless required every `ACTIVE`
commitment to carry a non-empty immediate request, and the live registry offered
no non-committing learned outcome. The system therefore spent two model calls
and committed another unevaluated response merely to express “wait.” This is a
semantic architecture defect, not a GPU-throughput defect.

## Bounded scope

- `src/angler/runtime/frozen_cognitive_models.py`
- `src/angler/runtime/higher_level_autonomy_adapter.py`
- focused target-formation, adapter, and persistent-autonomy tests
- this leaf and append-only `AGENTS_SYNC.md`

No model or adapter bytes, canonical episode, existing commitment, Cognee
record, Moving Origin event, permission, external effect, or evaluation
threshold may be rewritten.

## Design constraints

- The frozen target model—not trusted code—judges both present commitment and
  present actionability from state, memory, and time before seeing affordances.
- `ACTIVE` means the commitment remains supported now; it does not by itself
  mean an operation is justified on this wake.
- A bounded actionability field distinguishes `ACT_NOW` from `WAIT_FOR_CHANGE`;
  `NONE` remains the no-commitment state.
- `ACT_NOW` alone carries a non-empty internal request and may enter
  means-selection. `WAIT_FOR_CHANGE` carries no immediate request and cannot
  select, execute, learn, commit state, or advance Moving Origin.
- A successful wait inspection is checkpointed against the exact canonical
  state/event pair. Only a new event or state head invites another learned
  inspection.
- Existing active commitment evidence remains canonical while waiting. A
  transient formation result cannot erase, reinterpret, or promote it.
- Code validates and routes the model-authored status; it may not derive the
  status from keywords, scores, thresholds, or a fixed action rule.

## Acceptance

- Unit tests prove an active but presently unactionable commitment returns
  `QUIESCENT`, performs one target call and zero controller/executor calls,
  preserves state bytes and Moving Origin, and suppresses repeated calls for
  the unchanged head across restart.
- A later genuine state/event change invites exactly one fresh target judgment.
- An `ACT_NOW` target retains the current tool-blind lineage and normal
  selection path.
- Malformed or contradictory commitment/actionability combinations fail before
  selection or mutation.
- Focused and complete runtime tests pass.
- One bounded live wake over the current stable commitment becomes quiescent
  without a means-selection call or state advance; the immediately repeated
  wake uses zero model calls.

## Rollback

Restore only this leaf's source/test edits and restart `jenny2-api.service`.
The canonical database is not a rollback target because this leaf's live
acceptance must not mutate it.

## Result

The target-forming model now authors commitment status and present
actionability separately. `ACT_NOW` alone may enter means-selection;
`WAIT_FOR_CHANGE` preserves an evidence-grounded active commitment while
returning no immediate request; `NONE` represents no supported commitment.
Contradictory combinations fail before selection. The target prompt/model and
target-contract revisions advanced explicitly rather than silently changing a
consumed identity.

Focused target/adapter/supervisor tests passed 113/113. Complete runtime tests
passed 477/477 in 52.775 seconds, and `git diff --check` passed. The focused
restart test first creates a real committed active target, then proves that an
active `WAIT_FOR_CHANGE` formation invokes no controller or executor, mutates
no state or Moving-Origin value, deduplicates the unchanged wake across a
restart, and is invited exactly once after a new attributed event.

The live witness ran over Jenny's existing active commitment and exact ordinal
101 state. The first wake completed in 19.652227 seconds with one target-model
call, 7,783 prompt tokens, and 452 generated tokens. It returned `QUIESCENT`,
created no episode, selected no affordance, and preserved state ref
`sha256:6ff2fed4ca98e04996093f468d67fdd1f3fa5e437f3ea184b7255f9fdba1e6f9`
and ordinal 101. The immediate repeated wake completed in 0.006383 seconds with
zero model calls and the same state. Both ended with no pending operation,
life error, or projection error. Evidence files are
`/tmp/jenny-event-driven-wake.GRxB7Y/first.json` (SHA-256
`187b7edd9246a4e7078c8e8445eba2322ab1c68201ce3bde629aee95d57efd6f`)
and `/tmp/jenny-event-driven-wake.GRxB7Y/second.json` (SHA-256
`938f05b1e152914b7b5a1b95616d111e85d77d04a23ad403054915f80c5ed6d5`).

This is event-driven learned dormancy, not a claim that the model feels bored,
wants, or waits subjectively. It removes the demonstrated self-generated
inference loop while preserving model authority over whether current evidence
supports action.
