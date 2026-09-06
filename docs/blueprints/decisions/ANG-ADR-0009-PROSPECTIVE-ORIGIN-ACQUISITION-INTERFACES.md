# ANG-ADR-0009 — Prospective Origin and canonical acquisition interfaces

Status: accepted for local experimental contract construction; runtime/store
activation remains separately gated

Owner: ANG-AUTH-PROJECT-OWNER-001

Date: 2026-08-31

Supersedes: none. All existing `001@0.1.0` cognitive transaction and
episode-only projection identities remain immutable experimental history.

## Context

ADR-0007 requires prospective sibling futures to be committed before action,
later resolved without rewriting prediction, and situated on Moving Origin's
single autobiographical acquisition clock. The completed prospective-dynamics
component can maintain trainable internal world/self/focus state, but the
current persisted interfaces cannot represent its complete boundary:

- `ProspectiveCommitment@0.1.0` stores only the selected trace, score, scalar
  uncertainty, horizon, and parent competence digest;
- `ProspectiveTurnReservation@0.1.0` binds an agent and world but no explicit
  perspective/scope/state lineage or immutable sibling prediction batch;
- `CognitiveEpisode@0.1.0` has no agent/world lineage or resolution reference;
- the projection source and outbox are episode-only, so Moving Origin size is
  forced to equal episode/state-head sequence; and
- Cognee projection has no canonical lane for proposed/hypothetical records.

Reinterpreting those consumed bytes would invalidate content identities and
could silently upgrade a prediction into observed evidence. Adding hidden
sidecar semantics without a canonical reference would also break exact replay
and independent audit.

## Decision

### Preserve every existing identity

No `001@0.1.0` payload, parser, registry row, persisted byte sequence, or
scientific record is changed. Introduce seven explicit successors:

1. `ANG-CTR-SITUATED-STATE-LINEAGE-001@0.1.0`;
2. `ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0`;
3. `ANG-CTR-PROSPECTIVE-TURN-RESERVATION-002@0.1.0`;
4. `ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0`;
5. `ANG-CTR-COGNITIVE-EPISODE-002@0.1.0`;
6. `ANG-CTR-COGNITIVE-ACQUISITION-001@0.1.0`; and
7. `ANG-CTR-COGNITIVE-GRAPH-PROJECTION-002@0.1.0`.

All are experimental local interfaces. None is admissible as approved
`ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001`, and no
record conveys permission, truth, promotion, scientific acceptance, or a
Human-Flourishing gate result.

### Explicit situated state and sibling prediction

Situated state lineage keeps agent, world, reality mode, perspective, subject,
scope, checkpoint, competence digest, parent lineage, and cited evidence as
separate canonical fields. An agent reference is an operational lineage label,
not personhood or selfhood. A world reference is an external context binding,
not authority to create, select, or declare a world true.

A prospective batch embeds one exact parent lineage and all sibling branches
under one bounded resource envelope. Every branch binds public candidate/trace
identity, a `HYPOTHETICAL` context, predicted next latent, outcome logit,
uncertainty, horizon, focus allocation, and supporting evidence. Learned code
owns those predicted values and focus. Deterministic code only validates
declared eligibility/budgets, canonicalizes values, hashes identities, enforces
ordering, and computes versioned measurement arithmetic from supplied values.

The batch also binds an ordered nonempty context-ref set and one explicit
caller-supplied eligibility row per branch. Focus indices and weights may use
only the corresponding eligible row. Eligibility is outside learner control;
validation cannot infer it from content or prediction.

The only accepted perspective in this first version is `AGENT_SITUATED`.
Parent reality mode is `ACTUAL` or `SIMULATED`; branch reality mode is
`HYPOTHETICAL`; agent/subject, scope, world, and perspective cannot drift
between them. These are operational context labels, not personhood or truth
claims.

The batch and selected branch exist before executor claim. Capacity is
constructor/resource selected and bounded by the contract version; it is not a
four-candidate or permanent 64-wide architectural constant.

### Compatibility reservation without effect-identity drift

Reservation `002` embeds the exact legacy reservation and references the batch,
parent lineage, and selected branch. Its executor-facing `idempotency_key`
remains the embedded legacy reservation's `reservation_ref`. Therefore current
execution request, receipt, and objective-feedback `001` records remain exact
compatibility records and require no reinterpretation or successor in this
decision.

The legacy reservation proposals and logits must biject to all batch branches
with identical cardinality and index order. Each public proposal/trace and
logit binds its corresponding branch; checking only the selected branch is
insufficient. The batch schema may express configured capacity above 64, but a
reservation-`001` compatibility wrapper has 2 through 64 actual branches.

The future store must validate both envelopes atomically. A mismatch fails
before claim; batch or wrapper identity never becomes an alternate external
effect key.

### Append-only resolution and an acyclic content graph

Prospective resolution uses only `OBSERVED`, `COMPLETED_UNEVALUATED`,
`CANCELLED`, `CLARIFICATION_REQUIRED`, or `ERROR`. It references the batch,
selected branch, reservation wrapper, compatibility execution records when
they exist, exact parent/child lineages, and an exact objective-feedback record
for `OBSERVED`. It carries ordered branch lifecycle statuses covering the
whole batch and never edits or replaces the batch.

Only `OBSERVED` may carry a child lineage, observed next latent, outcome, or
prediction-error measurement. It requires a `COMPLETED` execution receipt and
the exact `ObjectiveFeedbackRecord`; that feedback record, never the receipt,
is the sole source of objective outcome truth. The selected branch status
equals the resolution disposition and every unexecuted sibling remains `OPEN`.
The record preserves predicted value, observed target, and versioned signed
error without deterministically declaring a prediction true, false, or
contradicted. Version `0.1.0` admits only
`binary-outcome-softplus-argument-v1`, storing exactly
`-label * selected_outcome_logit` for objective label `+1.0` success or `-1.0`
failure. `COMPLETED_UNEVALUATED` records only a completed receipt when objective
feedback is unavailable. It and cancellation, clarification, and error cannot
fabricate an objective outcome, observed world state, or learned child state.

Resolution contains no episode reference. Episode `002` is permitted only for
an `OBSERVED` resolution and embeds the exact legacy episode compatibility view
whose receipt-derived observations/response, objective feedback/outcome, and
parent/child competence digests match the resolution. Noncompletion
resolutions cannot produce an episode. Episode `002` references resolution.
This one-way content graph avoids an impossible resolution-ref/episode-ref hash
cycle. A future SQLite row may keep a relational foreign-key sidecar from
resolution to episode after both canonical identities exist; that sidecar is
integrity metadata, not part of either content identity.

### One canonical acquisition stream

Moving Origin remains unchanged and remains the only autobiographical
acquisition clock. A future known-fingerprint schema-v3 migration adds a
generic canonical acquisition stream and generic projection outbox:

1. Existing canonical episodes and their bytes/refs remain unchanged and are
   mapped in order to acquisition ordinals `0..N-1`; their derived record refs
   and projection acknowledgements remain byte-identical.
2. Reserving a new turn atomically appends the immutable batch plus one
   `COUNTERFACTUAL/PROPOSED` acquisition and its outbox row before claim.
3. A successful outcome atomically commits episode, competence state,
   resolution, and one later `EPISODIC/OBSERVED` acquisition/outbox row.
4. Known cancellation, clarification, or error appends a later observed
   lifecycle-resolution acquisition without objective outcome or child-update
   claims.
5. Claim, receipt, feedback staging, projection retry, and acknowledgment do
   not advance acquisition time.

Each acquisition embeds exactly one canonical `CognitiveMemoryRecord`, its
ordinal and predecessor reference, and exact source contract/ref. The contract
locally enforces ordinal/record equality and `(ordinal == 0) iff predecessor is
absent`; only the future schema-v3 transaction may establish global ordinal
uniqueness, predecessor adjacency, and one-clock contiguity. Moving Origin
continues to use `record_ref` as its event key. Historical anchors are never
deleted; Cognee namespaces remain disposable and rebuildable.

### Generic, distrustful projection

Projection `002` binds a generic acquisition, its source contract/ref, and the
exact canonical record. Backend hits identify only `record_ref`; source,
content, kind, epistemic status, visibility, relationships, and temporal
position are reconstructed from canonical bytes. Cognee-provided content or
truth labels are never trusted.

Default procedure recall admits only `OBSERVED` or `VALIDATED` records.
`PROPOSED`/counterfactual material requires a separate, frozen recall policy
and cannot be selected into ordinary evidence by query identity or learned
focus. Retractions/tombstones append history, filter canonical recall before
backend projection, and never remove Moving-Origin anchors.

Projection retries are oldest-first and bounded, reuse exact canonical record
refs/anchors/backend UUIDs, and acknowledge only after canonical revalidation
and successful backend projection. Frozen-origin and prospective-removal
controls reuse the same eligible canonical hits, content, ordering, limits, and
token budgets; only the declared component changes.

## Ownership and deterministic boundary

- RUNTIME owns transaction form, lifecycle, sequencing, idempotency, atomicity,
  restart, and outbox behavior.
- LEARNING owns predictions, focus, learned latent content, credit, update, and
  generalization.
- EVIDENCE owns canonical memory semantics, admissibility, provenance,
  epistemic status, supersession, and tombstones.
- WORLDS remains the owner of accepted Action and Feedback semantics.
- Moving Origin supplies acquisition position only; Cognee supplies untrusted
  retrieval candidates only; the frozen model supplies proposals/rendering.

No deterministic task rule, keyword-to-world mapping, answer lookup, candidate
slot policy, verifier inspection, hidden observation, or confidence-to-truth
promotion is permitted.

## Staged implementation and migration

The first authorized leaf implements only canonical records, registry rows,
exports, and synthetic contract tests. It may not edit the store, cycle,
projection runtime, or existing `001` classes.

A separate successor leaf must freeze the exact schema-v3 SQL/fingerprint,
known v2 fingerprint, migration transaction, acquisition APIs, outbox
transition, rollback points, and fault-injection tests before runtime changes.
Migration with an active v2 reservation must either be explicitly modeled and
tested or fail before mutation; it may not guess missing batch history.

Only after store/cycle/projection integration passes exact restart/rebuild may
the composite prospective core enter the active cycle. A later preregistered
training/evaluation identity must supply a trained checkpoint, held-out
prediction evidence, and fair removals before any useful-integration claim.

## Validation and acceptance

Contract construction requires byte-exact canonical round trips, bounded
2/7/64/above-64 branch cases and multiple latent widths, exhaustive cross-ref
tamper rejection, lifecycle truth-status tests, acyclic static inspection, and
unchanged existing contract suites. Registry inspection must find exactly one
row for each successor and no changed meaning for any existing row.

This ADR itself authorizes no model, GPU, Cognee service, network, external
effect, schema migration, experiment, threshold/result change, or milestone
claim. The Human-Flourishing Constitution remains supreme and outside all
learner-writable state.

## Rollback

Before runtime activation, rollback removes this ADR, its seven registry rows,
fresh contract modules/tests, exports, and the active leaf. Existing persisted
bytes, schema-v2 databases, Moving-Origin history, Cognee data, checkpoints,
and evidence remain untouched. After any future schema-v3 migration, rollback
must use that leaf's immutable pre-migration database copy and cannot rewrite
canonical history in place.
