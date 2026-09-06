# ANG-WORK-RUNTIME-PROSPECTIVE-RESERVATION-V1-001

Status: ready for local component implementation

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decision: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`

Assurance: LOW. This leaf adds local transaction, recovery, serialization,
and fake-executor tests on the dedicated Ubuntu workstation. It authorizes no
model/GPU run, live Cognee service, network call, real-person data, deployment,
or external effect.

## Purpose

Close the unsafe interval between Angler choosing a procedure and recording
what an executor did. A selected commitment must be durable before execution,
the exact request must be claimed before any effect, a returned execution must
be recorded before feedback, and the episode plus competence child must
resolve the reservation atomically.

This is the first Prospective Origin leaf. It establishes continuity and
exactly-once recovery semantics; it does not add learned future prediction,
planning, curiosity, improved reasoning, or autonomous authority.

## Scope and ownership

Fresh files:

- `src/angler/cognition/prospective.py`
- `tests/unit/cognition/test_prospective.py`

Narrow edits:

- `src/angler/cognition/contracts.py`
- `tests/unit/cognition/test_contracts.py`
- `src/angler/cognition/__init__.py`
- `src/angler/runtime/cognitive_transaction_store.py`
- `tests/unit/runtime/test_cognitive_transaction_store.py`
- `src/angler/runtime/cognitive_cycle.py`
- `tests/unit/runtime/test_cognitive_cycle.py`
- `src/angler/runtime/durable_ability_bridge.py`
- `tests/unit/runtime/test_durable_ability_bridge.py`
- `src/angler/runtime/__init__.py`
- this leaf and append-only `AGENTS_SYNC.md`

Consumed experiments, checkpoints, results, Qwen modules, Cognee datasets,
Moving Origin history, and accepted cognitive-memory records are read-only.

## Canonical contracts

`ProspectiveTurnReservation@0.1.0` binds:

- the exact parent sequence, event reference, competence digest, and snapshot
  digest;
- frozen model and encoder references;
- the exact acting-agent reference and active-world reference as lowercase
  content references;
- task, request, ordered recalled references, a bounded variable set of 2–64
  distinct public proposals, selected index/trace, and one finite hexadecimal
  logit per proposal;
- the decision evidence and supporting references;
- the exact `ProspectiveCommitment`; and
- the SHA-256 of a bounded state-owner pending blob.

Every field cross-checks the commitment and decision. The reservation is
content addressed. `reservation_ref` is the sole external idempotency key;
the narrower `commitment_ref` identifies the selected prediction but omits
parts of the execution context and therefore may never deduplicate effects.
The resolved reservation preserves agent/world binding, but this leaf does not
yet add those identities to `CognitiveEpisode` or its Cognee projection. That
projection boundary is an explicit input to the successor multi-world leaf and
must not be described as complete here.
These are opaque per-turn bindings, not yet a canonical agent genesis or
semantic world registry; stable self-lineage lands with the additive world/self
contracts rather than being inferred from a digest label.

The generic contracts, cognitive cycle, `DurableAbilityLearner`, and legacy
bridge accept 2–64 proposals. The learner ranks the set in one neural batch and
recomputes only the winner as an exact single-row pending output so restart and
feedback replay retain byte-exact validation. This removes the artificial
four-candidate limit; relation width, memory rank/slots, and future-dynamics
topology remain explicit state-shape decisions for later migration.

`CognitiveExecutionRequest@0.1.0` binds the reservation, commitment,
idempotency key, task, request, selected trace, and bounded input
observations. It is durably claimed before the executor is called.

`CognitiveExecutionReceipt@0.1.0` binds the exact request and contains only a
registered status, executed trace, public response, and public output
observations. `ObjectiveFeedbackRecord@0.1.0` binds objective feedback to the
same commitment and task before any learned update occurs.

The pending blob is opaque to the store, limited to 16 MiB, and owned by the
active learner. `DurableAbilityLearner` must capture and restore it with exact
checkpoint, parent-state, dtype, shape, value, and public-decision checks. A
restart may recompute a pure prediction from canonical inputs, but it may not
substitute a different decision or accept an unsafe deserialization global.

## Durable lifecycle

The canonical store advances one active reservation through:

```text
RESERVED -> CLAIMED -> EXECUTION_RECORDED -> RESOLVED
```

- `RESERVED` contains the canonical reservation and pending learner bytes.
- `CLAIMED` additionally contains the exact execution request.
- `EXECUTION_RECORDED` additionally contains an exact execution receipt and
  may contain staged objective feedback.
- `RESOLVED` preserves compact reservation/request/receipt/feedback evidence,
  clears the large pending blob, and names either the atomically committed
  episode or a known non-completion/cancellation disposition.

At most one row is active. Exact duplicate transitions are idempotent;
same-identity/different-byte transitions fail. A reservation may be cancelled
only before claim. A recorded `CLARIFICATION_REQUIRED` or `ERROR` may resolve
as known non-completion without a learned update. `CLAIMED` never expires,
rolls back, or becomes safe to retry because of time alone.

SQLite schema v2 keeps the accepted v1 tables and episode bytes unchanged and
adds the reservation aggregate. An exact known-v1 fingerprint may migrate in
one `BEGIN IMMEDIATE` transaction. Counterfeit, partial, or unversioned stores
fail before repair. Ordinary paths validate only the indexed head and at most
one active reservation; full history remains an explicit audit.

## Executor and recovery boundary

An effect-capable executor accepts the canonical execution request and must
durably enforce its idempotency key. Before returning a successful call, it
must be able to recover the byte-equivalent canonical receipt for that exact
request; a same-trace reconstruction without the original receipt binding is
insufficient. It also exposes recovery for the exact request:

- `RECORDED`: return the prior exact receipt; do not execute again;
- `NOT_STARTED`: authoritative proof that mutation never began, permitting
  one call with the already-claimed exact request/key;
- `IN_PROGRESS` or `UNKNOWN`: block with reconciliation required.

The runtime may never infer effect status from files, timeouts, process death,
or workspace state. A nonrecoverable executor is rejected before an effectful
call. An exception after claim leaves the claim active and blocks new work; it
does not restore an apparently clean turn that could execute twice.

## Atomic feedback and restart

`begin_turn` persists `RESERVED` before returning the chosen turn.
`execute_turn` persists `CLAIMED` before calling the executor and persists the
receipt before exposing an executed turn. `record_outcome` stages exact
feedback, performs one learned update, and in one SQLite transaction appends
the episode/state/outbox and resolves the reservation.

If learning or the final transaction fails, the learner returns to the exact
parent and rehydrates the same pending decision from the reservation. The
execution receipt and staged feedback remain retryable; the effect is never
re-executed. Restart reconstructs `RESERVED`, `CLAIMED`, or
`EXECUTION_RECORDED` state from canonical bytes and blocks on any mismatch.

## Deterministic-code ceiling

Code may validate, hash, serialize, bind parents, reserve, claim, record,
recover, stage feedback, resolve, audit, and measure. It may not choose or
alter a procedure, infer whether an ambiguous effect happened, fabricate an
outcome, auto-retry unknown work, inspect hidden verifier data, convert
feedback into a plan, or encode task-family solution logic.

## Acceptance

1. Canonical contracts round-trip byte exactly and reject identity,
   cross-binding, bounds, float, order, and tamper violations.
2. Exact v1-to-v2 migration preserves all existing head/state/episode/outbox
   identities; counterfeit and interrupted migrations fail closed.
3. Reserve/claim/receipt/feedback transitions are atomic, one-active,
   exact-idempotent, and conflicting-byte rejecting, including two-store races.
   Distinct reservations that share a conceptual commitment but differ in
   request, agent, or world retain distinct aggregate and executor identities.
4. A crash at each lifecycle boundary restarts into the same state and public
   decision, agent, and world; pending learner bytes restore exactly. Restart
   under a different agent or world identity fails closed.
5. An unknown claimed effect invokes no executor, update, new turn, or timeout
   retry. `RECORDED` recovery performs no second mutation; `NOT_STARTED` uses
   the original request/key once.
6. Episode, child snapshot, canonical head, projection outbox, and reservation
   resolution commit together. Injected failure leaves the parent plus the
   recorded execution/feedback retryable.
7. Known non-completion resolves without competence update or episode;
   claimed work cannot be cancelled or silently discarded.
8. Ordinary operations remain bounded and never invoke the full-history audit.
9. Existing cognition, typed memory, Cognee adapter, transaction, cycle,
   durable-ability, and situated-memory suites remain green on Ubuntu.
10. Tests use only fake durable executors and synthetic public data; no real
    effect, model, GPU, service, network, or personal data occurs.

## Human impact and rollback

This leaf strengthens meaningful human control by preventing silent duplicate
effects and by requiring explicit reconciliation when reality is unknown. It
does not grant an action, permission, goal, or survival interest. The Human-
Flourishing Constitution remains outside learner-writable state.

Rollback restores the edited modules and removes the two fresh files only.
Existing canonical evidence is never deleted. A migrated test/store fixture
is retained or restored as a whole; no partial schema downgrade is attempted.

## Stop conditions

Stop for a need to wrap a non-idempotent external API as if it were safe,
deserialize untrusted arbitrary code, trust noncanonical pending state, scan
unbounded history during an ordinary turn, mutate an accepted experiment,
invoke a real effect/model/service, or weaken parent, visibility, provenance,
or Human-Flourishing constraints.

## Next

After acceptance, implement
`ANG-WORK-RUNTIME-PROSPECTIVE-DYNAMICS-V1-001`: a resource-bounded variable
batch of learned sibling futures for one actual-world state, an evidence-backed
self perspective, and counterfactual branches; one shared action-conditioned
latent/outcome/uncertainty head; a learned selection residual; and one
composite competence snapshot. A learned soft focus allocates bounded reads,
branches, and recurrent compute across the eligible active world set; it never
owns identity, permission, truth status, or deletion, and dormant worlds remain
recoverable. Reality mode, perspective, subject, and scope remain separate
dimensions, and all branches share the one Moving-Origin autobiographical
acquisition clock. Fresh state begins with an exact-zero
prospective residual; capability activation waits for a preregistered
scientific evaluation with removal controls.
