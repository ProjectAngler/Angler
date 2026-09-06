# ANG-WORK-RUNTIME-PROSPECTIVE-ORIGIN-CONTRACTS-V1-001

Status: technical PASS — canonical successor vocabulary only; store/runtime
activation remains separately gated

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decisions: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`,
`ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES`, and accepted
`ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES`

Predecessor: `ANG-WORK-RUNTIME-PROSPECTIVE-DYNAMICS-V1-001`

Gate: `ANG-GATE-RUNTIME-PROSPECTIVE-ORIGIN-CONTRACTS-V1-001`

Assurance: LOW. This leaf adds canonical local contracts and synthetic CPU
tests. It authorizes no store migration, runtime activation, executor call,
model/GPU run, Cognee service, network use, real-person data, external effect,
scientific evaluation, or capability claim.

## Accountable outcome

Define the smallest immutable successor records required to carry prospective
world/self state through the existing reservation boundary without changing
or reinterpreting any consumed `001@0.1.0` bytes. The records must preserve all
sibling predictions before execution, append later resolution instead of
rewriting prediction, and provide one generic canonical acquisition envelope
for Moving Origin and rebuildable Cognee projection.

This is contract construction, not completed Prospective Origin. The current
prospective component remains untrained, and these records do not make its
hypotheses true, observed, authorized, calibrated, or useful.

## Exact outputs and write scope

Fresh files:

- `docs/blueprints/decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md`;
- `src/angler/cognition/prospective_origin.py`;
- `tests/unit/cognition/test_prospective_origin.py`;
- `src/angler/memory/cognitive_acquisition.py`;
- `tests/unit/memory/test_cognitive_acquisition.py`.

Narrow edits:

- `docs/blueprints/INTERFACE_REGISTRY.md`, only seven successor rows;
- `src/angler/cognition/__init__.py` and `src/angler/memory/__init__.py`, only
  exports required by the fresh modules;
- this leaf and append-only `AGENTS_SYNC.md`.

Read-only and out of scope:

- existing `001@0.1.0` contracts and their canonical bytes;
- `src/angler/runtime/cognitive_transaction_store.py` and its SQLite schema;
- `src/angler/runtime/cognitive_cycle.py` and all executor behavior;
- existing cognitive graph/Cognee/Moving-Origin implementation;
- prospective dynamics, model/Qwen, checkpoints, experiments, thresholds,
  results, and consumed identities.

Stop and split the leaf if an existing contract must change, if source and
episode identities become circular, or if a store/cycle/projection behavior
change is required to make a contract test pass.

## Successor identities

The leaf may introduce exactly:

1. `ANG-CTR-SITUATED-STATE-LINEAGE-001@0.1.0` for an evidence-bound agent,
   world, perspective, scope, reality mode, checkpoint, and competence state;
2. `ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0` for one parent lineage and
   a bounded immutable sibling set of learned predictions committed together;
3. `ANG-CTR-PROSPECTIVE-TURN-RESERVATION-002@0.1.0` as a compatibility wrapper
   that binds the batch and selected branch while retaining the exact legacy
   reservation as the sole executor idempotency key;
4. `ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0` for later observed execution,
   feedback, next-state evidence, and branch dispositions without episode ref;
5. `ANG-CTR-COGNITIVE-EPISODE-002@0.1.0` as a successor episode that references
   the resolution, avoiding a resolution/episode content-identity cycle;
6. `ANG-CTR-COGNITIVE-ACQUISITION-001@0.1.0` for one store-assigned contiguous
   acquisition and exactly one canonical `CognitiveMemoryRecord`; and
7. `ANG-CTR-COGNITIVE-GRAPH-PROJECTION-002@0.1.0` for a rebuildable projection
   bound to a generic canonical acquisition rather than an episode-only source.

Nested situated-context and branch values are owned by the batch contract and
do not receive additional `ANG-CTR-*` identities. No executor request, receipt,
or objective-feedback successor is introduced: reservation `002` embeds the
exact legacy reservation, and its `idempotency_key` remains that embedded
reservation's `reservation_ref`.

## Required semantics

All source records use strict canonical JSON, exact field sets, lowercase
SHA-256 references, finite hex-encoded floats, immutable tuples, bounded text,
and content-derived identities. Unknown versions and extra fields fail closed.

Situated context keeps `reality_mode`, `perspective`, `subject_ref`,
`scope_ref`, and `world_ref` separate. Parent lineage is `ACTUAL` or
`SIMULATED`; predicted branches are `HYPOTHETICAL`. Version `0.1.0` accepts
only the closed perspective value `AGENT_SITUATED`; parent and branch context
must keep the same agent/subject, scope, world, and perspective. These labels
never imply personhood, truth, permission, world creation, or an
authority-bearing self.
Self/agent claims require cited evidence and an exact checkpoint/state lineage;
tensor content cannot invent an agent or world identity.

One batch binds:

- its exact parent lineage and parent acquisition/event references;
- a versioned branch/read/latent resource envelope;
- between 2 and the declared bounded branch budget, not a permanent four-way
  or 64-way architecture;
- an ordered nonempty context-ref set plus one caller-declared eligibility row
  per branch; learned focus indices must be a subset of that exact row and
  cannot make an ineligible context eligible;
- one row-local prediction per sibling: candidate/trace refs, hypothetical
  context, predicted next latent, outcome logit, uncertainty, horizon, focus
  allocation, and supporting evidence;
- one selected branch ref already present in the sibling set; and
- exact dynamics/checkpoint/config identities.

The batch branch set must have the same cardinality and index order as the
embedded legacy reservation's public proposals. Every proposal and logit is
bound one-to-one to its corresponding branch, not only the selected branch;
no legacy proposal may be omitted, duplicated, reordered, or supplemented.
The batch contract itself supports a resource configuration above 64, but a
batch wrapped by legacy reservation `001` necessarily contains 2 through 64
actual branches until a separately authorized execution-interface successor
exists.

The batch is immutable before claim. Learned code owns prediction content,
scores, uncertainty, focus, and generalization. Deterministic code may only
validate, bound, canonicalize, hash, order, and later compute a declared
prediction-error measurement from supplied prediction and observation values.

Resolution uses the closed lifecycle vocabulary `OBSERVED`,
`COMPLETED_UNEVALUATED`, `CANCELLED`, `CLARIFICATION_REQUIRED`, and `ERROR`.
It references the batch, selected branch, compatibility reservation,
execution request/receipt when those exist, and exact parent/child lineages.
`OBSERVED` is the only disposition
allowed to carry a child lineage, observed outcome, next-latent evidence, or
prediction-error measurement, and it requires an exact completed receipt and
exact `ObjectiveFeedbackRecord`; the feedback record is the sole source of
objective outcome truth. Every resolution carries a branch-status entry for
every batch branch in the same order. The selected branch status equals the
resolution disposition and every unexecuted sibling remains `OPEN`; neither a
receipt nor deterministic prediction-error arithmetic may label a prediction
true, false, or contradicted. The only prediction-error measurement identity
is `binary-outcome-softplus-argument-v1`: with objective label `+1.0` for
success and `-1.0` for failure, the stored signed value is exactly
`-label * selected_outcome_logit`, the argument to the learned component's
softplus outcome loss. Known completed-but-unevaluated, cancellation,
clarification, or error records lifecycle truth only and cannot fabricate
objective outcome truth or a child competence update.

Episode `002` exists only for an `OBSERVED` resolution with exact objective
feedback. It embeds the exact legacy episode as a compatibility view and
references the resolution. The legacy episode outcome, feedback, observations,
response, and parent/child competence digests must match that resolution's
bound receipt, feedback, and lineages. `CANCELLED`,
`COMPLETED_UNEVALUATED`, `CLARIFICATION_REQUIRED`, and `ERROR` resolutions
cannot produce Episode `002`. Resolution intentionally does not reference the episode. A future store
may add a relational sidecar/FK from resolution to the committed episode
without changing either content identity.

Each acquisition locally binds one non-negative ordinal, predecessor
acquisition ref, source contract/ref, and exact canonical memory record. The
ordinal equals the record's acquisition ordinal, and ordinal zero has no
predecessor while every nonzero ordinal requires one. Global uniqueness,
predecessor adjacency, and one-clock contiguity are explicitly deferred to the
future transactional schema-v3 leaf and cannot be claimed by this standalone
contract. Future-batch
acquisitions must be `COUNTERFACTUAL/PROPOSED`; successful post-action
resolution acquisitions must be `EPISODIC/OBSERVED`. Projection `002` embeds
the acquisition identity plus exact record and cites the generic source. It is
disposable and conveys no canonical truth or authority.

## Acceptance

1. Every new top-level record round-trips byte-exactly with stable content ref;
   extra/missing/unknown-version, noncanonical JSON, float, digest, ordering,
   duplicate, and cross-binding tampering fail closed.
2. Batch tests cover 2, 7, 64, and an unwrapped configured batch above 64, plus
   multiple latent widths, without candidate-axis state or
   task/answer/verifier fields; reservation compatibility remains 2 through
   64 actual proposals.
3. Context tests reject hypothetical parent lineage, non-hypothetical branches,
   subject/agent/scope drift, unsupported reality/perspective values, missing
   evidence, and state/checkpoint mismatch.
4. Reservation `002` binds the exact legacy reservation, batch, parent lineage,
   and selected branch while delegating the sole external idempotency key to
   the legacy reservation ref.
5. Resolution rejects batch/branch/reservation/request/receipt/feedback and
   parent/child lineage drift; `OBSERVED` requires exact completed receipt and
   objective feedback, while noncompletion cannot carry objective outcome or
   claim a child update; `COMPLETED_UNEVALUATED` records a completed receipt
   without outcome/update; the sole prediction-error metric and exact signed
   arithmetic are fixed; all branch statuses cover the batch in order and
   unexecuted siblings remain open.
6. Episode `002` accepts only an `OBSERVED` resolution with matching objective
   feedback and lineages, references resolution in one direction only, and
   static tests prove no resolution payload or class contains an episode ref.
7. Acquisition ordinal/record equality and zero/predecessor form are exact,
   one record is embedded, and record kind/status/source semantics match batch
   versus resolution source; no global contiguity claim is made before the
   store leaf.
8. Projection `002` round-trips and rejects any acquisition/source/record drift;
   it remains distinct from episode-only projection `001`.
9. Existing cognition/prospective/cognitive-graph contract suites remain green;
   static inspection finds exactly seven new registry rows and no existing row
   reinterpretation.
10. No store, cycle, Moving-Origin, Cognee, model, executor, authorization,
    experiment, result, or Human-Flourishing gate state changes.

Passing establishes canonical successor vocabulary only. It does not establish
runtime persistence, one-clock migration, learned prediction quality, improved
reasoning, temporal agency, selfhood, autonomy, AGI, or readiness.

## Validation commands

Run only CUDA-hidden CPU/static checks for the two fresh suites and unchanged
contract suites. Do not run Qwen, Cognee, GPU work, the expanded integration
suite, or any experiment.

## Validation result

The final CUDA-hidden focused run passed 54 tests and 105 parameterized
subtests in 0.22 seconds pytest time / 0.35 seconds wall time with 39,704 KB
maximum RSS. It covered the two fresh suites plus the unchanged cognition,
prospective-execution, and cognitive-graph contract suites. Byte-exact round
trips, 2/7/64/65 branch batches, multiple latent widths, full proposal/logit
bijection, eligibility escape rejection, feedback-bound observed resolution,
completed-but-unevaluated and noncompletion lifecycles, observed-only episode
construction, local acquisition form, source-owned memory semantics, and
generic projection rejoin all passed.

Static validation found exactly seven successor registry rows and seven ADR
links, no changed existing interface meaning, no resolution-to-episode back
reference, no backup/reject artifact, and no whitespace defect. Python compile
checks passed. Independent read-only audits first blocked ambiguous truth,
episode, eligibility, acquisition, and source-dispatch semantics; after the
corrections, both the cognition contract implementation and the exact-type
acquisition/projection boundary returned PASS with no remaining blocker.

Final SHA-256 values are:

- cognition contracts: `68A62B7533BE6918759D754455196ECA6783CAF61120B0E93C94C11B11F1F8C8`;
- cognition tests: `9E7160A665A2B76F727F326198A6433CDC1BB26DB9D4F1046E28CB411FDE94BD`;
- acquisition/projection contracts: `E93C9F181FE1938A4C033A55A88827BA3D0CA5B8B9F46CB1B5569BC4987D36E7`;
- acquisition/projection tests: `F98F7EB9C1E53778BD0BF4445E6A655E561BA6D29148E2685C82A6762A6FCA97`;
- cognition exports: `BFCADD49F6BE02A2FDFFF4919A00C4959AEE56F7BF8169795A40AB7CE671D33D`;
- memory exports: `1E38021ED5C28AEF060A54C48D8C4BD06756EFDDF206CCD382C3662006CD8B3A`;
- interface registry: `621D38DA041437127635AE71E1DD142A397DF207C01062A58551DB30AF168804`;
- ADR-0009: `B20CAABCB5A6C12ED97E6445026007998E767B0C15E3F6C87F0098A49F7FE1F2`.

No existing `001` bytes, store schema, runtime cycle, Moving Origin state,
Cognee data, model/checkpoint, experiment identity/result, authorization, or
external effect was invoked or changed.

## Human impact and rollback

Proportionate assessment for this exact local synthetic contract scope: impact
class `LOW`, disposition `ALLOW`, mapped to
`ANG-GATE-HUMAN-FLOURISHING-001`. The records describe hypotheses and observed
lifecycle facts only. They grant no permission, authority, identity
entitlement, survival interest, or power to change eligible worlds, tools,
budgets, truth status, or human control. There is no network, service, model,
GPU, person data, external effect, promoted-state mutation, or deployment. The
principal risks are truth-status confusion, interface drift, deterministic
solution encoding, and overclaim; strict source/status binding, preserved
legacy identities, learned-content ownership, narrow scope, and explicit
non-equivalence mitigate them. This assessment does not pass or waive the
Human-Flourishing gate for runtime activation or promotion. Any scope expansion
requires a successor assessment.

Rollback removes only the seven registry rows, ADR, fresh modules/tests, export
edits, and this leaf. Existing persisted bytes, schema-v2 stores, Moving-Origin
history, Cognee data, checkpoints, and scientific evidence remain untouched.

## Next

After this gate passes, expand a separate store/cycle leaf for schema v3,
atomic pre-action batch acquisition, atomic post-action resolution/episode/state
commit, generic projection outbox, known-fingerprint v2 migration, and exact
restart/rebuild. Only then wire the composite prospective core into the cycle.
