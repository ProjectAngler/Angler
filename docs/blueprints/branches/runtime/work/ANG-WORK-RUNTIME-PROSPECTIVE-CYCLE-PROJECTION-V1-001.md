# ANG-WORK-RUNTIME-PROSPECTIVE-CYCLE-PROJECTION-V1-001

Status: technical PASS — 2026-08-31

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decisions: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`,
`ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES`, and
`ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES`

Predecessors:

- `ANG-WORK-RUNTIME-PROSPECTIVE-DYNAMICS-V1-001` — technical PASS;
- `ANG-WORK-RUNTIME-PROSPECTIVE-ORIGIN-CONTRACTS-V1-001` — technical PASS;
- `ANG-WORK-RUNTIME-COGNITIVE-STORE-V3-001` — technical PASS.

Gate: `ANG-GATE-RUNTIME-PROSPECTIVE-CYCLE-PROJECTION-V1-001`

Assurance: LOW. This leaf joins already bounded local experimental components
with synthetic CPU tests, fake execution, fake Cognee bindings, and an
in-memory Moving Origin. It authorizes no Qwen/model invocation, GPU use, live
Cognee service, network, real-person or recovered data, non-disposable store
migration, external effect, experiment, threshold/result change, promotion,
or capability claim.

## Accountable outcome

Activate one successor lane in the existing `CognitiveCycle` so the same
composite procedural/prospective competence owner produces an immutable
sibling prediction batch before executor claim, receives exact objective
feedback after a completed execution, and commits the corresponding situated
resolution through the schema-v3 store. Rebuildable Cognee and Moving-Origin
views consume the store's one generic acquisition stream rather than treating
episode sequence as autobiographical time.

The lane must preserve the embedded legacy reservation as the sole executor
effect key, survive every durable restart boundary, and expose fair
prospective/frozen-origin/backend-removal interventions without routing state
or evidence by task identity. It establishes integration integrity only. It
does not establish predictive quality, improved reasoning, semantic world
understanding, selfhood, temporal agency, autonomy, readiness, or AGI.

## Dependency and activation rule

This specification may be reviewed before the store leaf completes, but no
source or test work authorized below may begin until the store-v3 gate has a
technical PASS with its frozen schema fingerprint and atomic APIs intact. The
successor cycle runs only against schema v3. Existing schema-v2 stores,
including durable or non-disposable stores, are not migrated or activated by
this leaf.

A `CognitiveCycle` instance chooses its legacy-compatibility or successor lane
once at construction from explicit configuration and store capability. That
choice cannot depend on the request, task family, recall result, candidate,
prediction, or outcome. The integrated product configuration uses the
successor lane; legacy behavior remains only as compatibility history and
regression coverage.

## Exact future outputs and write scope

Fresh files:

- `src/angler/runtime/prospective_observation.py`;
- `src/angler/memory/cognitive_acquisition_graph.py`;
- `src/angler/memory/cognee_acquisition_adapter.py`;
- `tests/unit/runtime/test_cognitive_cycle_v3.py`;
- `tests/unit/memory/test_cognitive_acquisition_graph.py`;
- `tests/unit/memory/test_cognee_acquisition_adapter.py`.

Narrow edits:

- `src/angler/reasoning/prospective_dynamics.py`, only to expose the full
  learned prospective output from the same prediction and domain-separated
  component-state integrity material;
- `src/angler/runtime/durable_ability_bridge.py`, only for
  `PendingProspectiveMaterial`, batch binding, bounded pending serialization,
  exact restoration, and the declared prospective lesion;
- `src/angler/runtime/cognitive_cycle.py`, only for fixed successor
  construction, lifecycle orchestration, restart, and generic projection
  retry;
- `src/angler/runtime/cognitive_transaction_store.py`, only if the accepted
  store leaf lacks verified read-only lookup of the one active successor
  aggregate or the resolved successor aggregate for the current legacy
  episode. No schema, fingerprint, migration, write transaction, or frozen
  store-v3 API may change here;
- `src/angler/memory/__init__.py`, `src/angler/runtime/__init__.py`, and
  `src/angler/reasoning/__init__.py`, only for required exports;
- existing focused bridge/cycle tests only where an unchanged compatibility
  assertion must be retained;
- this leaf and append-only `AGENTS_SYNC.md` during implementation.

Read-only and out of scope:

- `ANG-WORK-RUNTIME-COGNITIVE-STORE-V3-001` and its frozen SQL/fingerprints;
- all accepted ADRs and `docs/blueprints/INTERFACE_REGISTRY.md`;
- every canonical cognition/acquisition contract and all existing `001` or
  `002` content bytes;
- existing Cognee datasets, Moving-Origin histories, checkpoints, consumed
  experiment identities/results, Qwen/model code, executors, and promotion or
  authorization boundaries.

No new `ANG-CTR-*` identity is introduced. `PendingProspectiveMaterial`,
observation-encoding values, projection receipts, and backend hits are bounded
RUNTIME/EVIDENCE-internal implementation types, not authority, truth,
personhood, or cross-branch contracts.

Stop and split if implementation requires changing canonical contract bytes,
schema-v3 SQL or acquisition semantics, inventing missing lineage or batch
history, a second competence/acquisition owner, live service/model/GPU work,
or any task-specific solution or eligibility rule.

## One explicit situated lineage

Successor-cycle construction requires fixed caller configuration for:

- parent reality mode, limited to `ACTUAL` or `SIMULATED`;
- `AGENT_SITUATED` perspective;
- separate agent, subject, scope, and world references; and
- a nonempty sorted set of canonical context/state evidence references.

Tensor content, request text, recall, Cognee, and learned focus cannot infer or
change those identities. The existing cycle model/encoder/agent/world refs
remain exact. Subject, scope, reality mode, and bootstrap evidence become
additional constructor-fixed inputs.

The composite core exposes one integrity-only state view containing its exact
step, checkpoint ref, config ref, and separate SHA-256 refs for world, self,
focus, and outcome state. The world digest also covers world-usage state.
These digests domain-hash exact dtype, shape, and tensor bytes; they do not
interpret tensor meaning. `competence_state_digest` remains the
`DurableAbilityLearner` digest, and `snapshot_digest` remains the exact cycle
snapshot SHA-256.

For a fresh or migrated legacy head with no successor episode, the cycle may
adopt one deterministic lineage root from the exact current state and explicit
configuration. Its parent lineage ref is absent even when the restored
component step is positive, marking a compatibility boundary rather than
inventing missing ancestry. After an observed successor commit, restart must
recover the exact child lineage embedded in Episode `002` for the current
legacy episode. Non-observed resolutions leave that lineage unchanged.

The store integration must therefore provide verified read-only equivalents
of:

```python
active_prospective_turn() -> ProspectiveTurnRecord | None
prospective_turn_for_episode(
    legacy_episode_ref: str,
) -> ProspectiveTurnRecord | None
```

If the accepted store-v3 implementation exposes the same facts under already
frozen names, the cycle uses those names and adds no alias. Reads reparse and
rejoin exact canonical wrapper, resolution, and Episode `002` bytes; they do
not create another persisted identity.

## Learned prediction and `PendingProspectiveMaterial`

`CompositeProspectiveCreditCore` adds a public internal method that returns
the established combined credit output and its complete
`ProspectiveFocusOutput` from the same frozen parent call. The bridge may not
call the private one-world helper or run an independently selected prediction
whose values could drift from the committed batch.

The current exact selected-row recomputation remains: rank all candidates by
the learned batched logits, recompute the selected row for byte-exact pending
feedback/replay, replace the selected public logit and selected prospective row
with that exact result, and reject if the winner changes. Every other branch
retains its corresponding learned batch row. This is numeric/replay plumbing,
not a second ranker.

The bridge retains one bounded internal `PendingProspectiveMaterial` with:

- the exact public `AbilityDecision` and selected neural output;
- every branch's future latent, outcome logit, uncertainty, prospective
  residual, focus mask/weights, recurrence, and parent step;
- exact ordered candidate traces and each candidate's supporting evidence;
- exact context refs, caller-declared eligibility rows, and resource budget;
- parent competence/checkpoint/config and component-state digests; and
- the exact batch ref after batch construction.

All tensors are finite detached contiguous clones with recorded dtype and
shape. Candidate, context, and tensor axes must align exactly; focus must stay
inside eligibility and budget. The successor pending snapshot includes this
material under the existing 16 MiB ceiling and content checksum. Its batch ref
is bound before pending bytes are captured, so the embedded legacy
reservation binds those bytes without an identity cycle:

```text
learned select/material
  -> immutable batch and batch_ref
  -> pending snapshot bound to batch_ref
  -> legacy reservation bound to pending bytes
  -> reservation-002 wrapper bound to batch + legacy reservation
```

Restart restores the exact parent, pending selected output, full prospective
material, and batch ref, then compares them with the stored wrapper. Any
checkpoint, config, dtype, shape, candidate, evidence, eligibility, budget,
prediction, parent, or batch drift fails before executor use.

The first integration lane has one opaque situated feature context per branch,
using the already accepted composite path with read and recurrence equal to
one. The context ref is the exact parent situated-context ref and eligibility
is exactly `(0,)`; learned focus therefore remains bounded but is not evidence
of multi-context attention quality. Candidate count remains constructor chosen
from 2 through the legacy compatibility ceiling of 64, while core and contract
capacity remain resource-configurable rather than permanently four- or
64-wide. A later leaf must introduce canonically aligned multi-context feature
rows before claiming learned Cognee-context focus.

## Evidence and cold-start rule

Every prospective branch carries that encoded candidate's own nonempty,
canonically sorted supporting evidence, and every such ref must occur in the
exact ordered `recalled_refs` embedded in both batch and legacy reservation.
The selected branch support must also equal the legacy selected support.

Default successor activation admits only canonically reconstructed
`OBSERVED` or `VALIDATED` memory. A `PROPOSED` batch may be projected and
recalled only through a separate frozen counterfactual policy that is out of
scope here. Query identity, backend score, learned focus, or absence of memory
cannot promote proposed material into ordinary support.

If recall is empty, every candidate lacks support, or an adapter cites a ref
outside validated recall, `begin_turn` fails before learner selection,
reservation, acquisition, claim, or executor call. It does not hash the
request into fake evidence, inject decision evidence into `recalled_refs`, or
silently route that request to another solver/state. Cold-start seeding must be
a separately canonical observed acquisition outside this leaf.

## Frozen observed-state encoding boundary

`prospective_observation.py` defines a runtime-internal
`FrozenObservedStateEncoder`, immutable `ObservedStateEncoding`, and the sole
encoder permitted by this local synthetic leaf:
`SyntheticObservedStateEncoderV1`. The encoder receives only the exact
completed `CognitiveExecutionRequest` and `CognitiveExecutionReceipt` and the
required latent width. It returns one finite latent of exactly that width and
the canonical sorted evidence tuple containing exactly the request and receipt
refs; their canonical bytes already contain the public input/output
observations.

The encoder:

- is frozen and has no mutable competence, feedback, answer, verifier, task
  family, candidate-ranking, or authorization input;
- cannot see `ObjectiveFeedbackRecord.outcome` or use the predicted latent as
  its target;
- cannot invent evidence, identity, world state, or truth; and
- runs only after a completed receipt exists and before learned feedback is
  applied.

The synthetic algorithm is frozen as
`angler.synthetic-observed-state.v1`. Its input stream begins with the exact
ASCII bytes `angler.synthetic-observed-state.v1\0`, then unsigned 64-bit
big-endian latent width, unsigned
64-bit big-endian request-byte length and exact canonical request bytes,
followed by unsigned 64-bit big-endian receipt-byte length and exact canonical
receipt bytes. For latent index `i`, append unsigned 64-bit big-endian `i`,
take SHA-256, interpret the leading 64 bits as unsigned, discard the low 11
bits, and emit the exact dyadic float `(top53 / 2**52) - 1.0`. This yields one
finite value in `[-1.0, 1.0)` per index without task parsing, learned state, or
outcome access.

The encoder's `encoder_ref` is the SHA-256 content ref of
`canonical_bytes({"algorithm": "angler.synthetic-observed-state.v1",
"latent_width": latent_width})` using the repository's canonical JSON codec
and an exact integer width. Successor-cycle construction requires the configured cycle
encoder ref, observation-encoder ref, and successor relation-adapter encoder
ref to be identical before recall. A different algorithm, width, model,
configuration, or real observation encoder therefore requires a different
ref and cannot reproduce a resolution under this identity. The exact domain
bytes, framing, canonical identity JSON, evidence tuple, and boundary values
are frozen in tests.

Thus `ProspectiveResolution.observed_next_latent` is a frozen representation
of exact public post-execution evidence, not zeros, the pre-action prediction,
or the learner's post-feedback focus hypothesis relabeled as observation.
The exact `ObjectiveFeedbackRecord`, staged separately, remains the sole source
of success/failure truth. Encoder failure leaves staged feedback absent and
the parent/pending turn unchanged.

This component leaf supplies only the protocol and generic synthetic fake. A
real frozen-model observation encoder requires a later model/GPU leaf and a
new evaluation identity.

## Exact cycle and transaction flow

### Restore and begin

1. Restore or initialize the exact competence head. Restore an active
   successor aggregate before requiring projection synchronization; an outbox
   outage cannot erase or alter a claimed turn.
2. Before any new recall, drain/rebuild generic acquisition projection until
   memory proves O(1) synchronization with `acquisition_head`. Failure blocks
   a new turn but does not roll back canonical acquisitions.
3. Capture exact parent state/head/lineage and acquisition head, recall under
   the fixed policy, obtain frozen-model proposals through the existing
   adapter boundary, encode candidates, and run the one learned composite
   selection.
4. Build all branches in proposal order, the immutable batch, batch-bound
   pending bytes, legacy commitment/reservation, and reservation-`002`
   wrapper. Every legacy proposal/logit bijects to its batch branch.
5. Build the batch's `COUNTERFACTUAL/PROPOSED` memory record, next contiguous
   acquisition, and projection `002`, then call exactly:

```python
reserve_prospective_turn(
    reservation_v2,
    batch_acquisition,
    pending_blob,
)
```

The store revalidates both heads and atomically commits batch, wrapper,
pending bytes, acquisition, outbox, and clock before any claim. A concurrent
head loss restores/rehydrates the canonical winner and creates no orphan.

### Claim and execution

`CognitiveExecutionRequest.from_reservation` receives only
`reservation_v2.legacy_reservation`. Claim, recovery, executor idempotency,
error reporting, receipt, and feedback use
`reservation_v2.idempotency_key`, which must be byte-identical to the embedded
legacy reservation ref. The wrapper, batch, lineage, branch, acquisition, and
projection refs never become external effect keys.

The cycle calls only the accepted store-v3 methods
`claim_prospective_turn`, `record_prospective_execution`, and
`stage_prospective_feedback`. These calls advance neither competence nor
acquisition time. Claimed/unknown execution requires authoritative executor
reconciliation exactly as in the accepted legacy boundary; it never guesses
or re-executes under a new key.

### Non-observed resolution

- `CANCELLED` has no request or receipt and is permitted only before claim.
- `CLARIFICATION_REQUIRED` and `ERROR` require their exact request/receipt.
- `COMPLETED_UNEVALUATED` requires an exact completed receipt and an explicit
  caller declaration that objective feedback is unavailable. It is never
  inferred from a receipt, timeout, projection failure, or model output.

The cycle builds `ProspectiveResolution.lifecycle`, its exact
`EPISODIC/OBSERVED` lifecycle record, and next resolution acquisition, then
calls `resolve_prospective_turn`. The transaction appends exactly one
acquisition/outbox row and clears the active turn. It creates no child lineage,
Episode `002`, objective outcome, observed latent, learner update, or
competence-head advance.

### Objectively observed resolution

1. Require an exact completed request/receipt and caller-supplied success or
   failure feedback, then encode the observed target from exact
   request/receipt evidence without exposing objective outcome to the encoder.
   Encoder failure leaves feedback unstaged and the parent/pending turn exact.
2. Build and durably stage the exact `ObjectiveFeedbackRecord` only after
   encoding succeeds and before learning.
3. Apply objective outcome exactly once to the same pending composite state.
   Capture exact child snapshot, competence digest, component digests, and
   state step. Construct a child lineage whose parent is exact and whose
   identity/configuration does not drift.
4. Build `ProspectiveResolution.observed`, the exact legacy episode
   compatibility view, and observed-only `CognitiveEpisodeV2`. Build one
   `EPISODIC/OBSERVED` resolution record and acquisition; Episode `002` is not
   a second acquisition source.
5. Call exactly `commit_observed_prospective_turn`. Episode/state/legacy
   outbox, resolution, Episode `002`, child snapshot, one generic acquisition
   and outbox, both heads, and active-row resolution commit or roll back
   together.

If observation encoding or learning fails, no child canonical write occurs.
If the final store transaction fails after learning, restore the exact parent,
rehydrate the same canonical active wrapper and pending bytes, retain staged
feedback, and permit only byte-identical retry. A successful canonical commit
is never rolled back because Cognee or Moving Origin projection later fails.

## Canonical acquisition record derivation

`cognitive_acquisition_graph.py` owns two pure bounded renderers. They assign
no ordinal and infer no fact; the caller supplies the store-head ordinal and
the canonical source supplies all content.

`prospective_batch_record(batch, ordinal)` is exactly
`COUNTERFACTUAL/PROPOSED`, cites the batch/source, parent lineage, decision
evidence, and recalled evidence, uses the dynamics checkpoint and parent
competence refs, and contains a bounded public rendering of request, branch
count, selected proposed trace, and explicit hypothetical status. Its sorted
relations use only registered `PREDICTS`/`ABOUT` links to exact branch and
evidence refs. It contains no answer, permission, truth, or fabricated
observation.

`prospective_resolution_record(resolution, ordinal)` is exactly
`EPISODIC/OBSERVED` because it records an observed lifecycle fact, including
for non-outcome dispositions. It cites the exact resolution, batch, selected
branch, and present request/receipt/feedback/evidence refs; uses child
competence only for `OBSERVED` and parent competence otherwise; and renders
the closed lifecycle disposition. Only `OBSERVED` may render objective outcome
and feedback. Cancellation, clarification, error, and completed-unevaluated
render explicit absence of objective outcome/child update.

Both records are `LEARNER_VISIBLE`, use producer id
`angler.prospective-cycle`, have no inferred confidence/world-validity,
canonicalize relations/provenance, and must reproduce byte-identically on
restart/rebuild. Their exact literal rendering templates are frozen in tests
before implementation results are viewed.

## Generic Cognee and Moving-Origin projection

`AcquisitionSituatedMemory` treats the accepted v3 store as its sole canonical
source. For every verified `CognitiveAcquisitionItem`, it reparses the exact
acquisition/projection, uses `record_ref` as the Moving-Origin event key,
requires origin ordinal to equal acquisition ordinal, and sends only the
reference projection to the backend. Existing anchors must match exact
ordinal, record ref, projection ref, and validity bounds; retry cannot append a
duplicate.

`CogneeAcquisitionAdapter` uses the already frozen tenant/dataset/NodeSet
scope, deterministic UUIDs, structured insertion, local-embedding/telemetry
guards, and reference-only search. Its point carries acquisition/source/
record/projection refs plus bounded record fields. Backend hits carry only
record and optional adjacent-record refs/score/backend id. Every record,
epistemic state, relation, visibility, source, temporal position, and neighbor
is reconstructed from canonical store bytes; Cognee content and truth labels
are never trusted.

Projection retry is oldest-first with store default `64`, validates `1..256`,
and always forwards the SQL limit. It acknowledges only the exact projection
ref after canonical revalidation, exact Moving-Origin rejoin, and successful
backend projection. Backend/origin failure leaves the outbox pending. Ordinary
recall requires synchronization with the acquisition head; explicit rebuild
pages canonical acquisitions and reconstructs disposable views without
rewriting acquisitions.

Default recall filters canonical epistemic status before exposing backend
results and admits only `OBSERVED`/`VALIDATED`. Proposed prospective batches
remain canonical and projectable but are not ordinary evidence. Retraction and
tombstone semantics remain append-only and may filter recall; they never
delete Moving-Origin anchors.

This leaf uses fake structured bindings and an in-memory origin only. It does
not write, search, clear, or migrate a live Cognee namespace or durable
Moving-Origin history.

## Fair removal controls

Removal mode is fixed when the cycle/evaluation fixture is constructed and
cannot depend on the query or adaptive result.

- **Prospective removal:** reuse the exact parent, candidates, candidate
  features, canonical hits, content/order, context refs, eligibility,
  branch/read/recurrent budgets, and token budget. Change only the declared
  prospective read/residual contribution to its accepted lesion; do not load
  another state/checkpoint or rerank with a heuristic.
- **Frozen Origin:** reuse the exact eligible canonical hits, content/order,
  limits, token budget, and Cognee references. Change only live temporal
  position to the registered frozen-position view; do not remove or reorder
  evidence.
- **Cognee/backend removal:** freeze the same canonically revalidated hit
  manifest before condition assignment and replay that manifest through the
  same visibility/epistemic/limit checks. Change only live backend candidate
  sourcing; canonical content and Moving-Origin positions remain exact.

These hooks prove isolation and support a later preregistered evaluation; this
leaf runs no capability comparison and assigns no credit to any component.

## Deterministic-code ceiling

Deterministic code may validate/copy learned outputs, compute content refs and
the registered prediction-error arithmetic, enforce exact eligibility and
resource bounds, perform the existing argmax over learned logits, encode
canonical records, transact, restore, page, rejoin references, and run fixed
removal interventions.

It may not map task text, keywords, candidate slots, hidden answers, verifier
state, feedback labels, confidence, graph position, or world identity to a
procedure, branch, focus allocation, latent target, outcome, permission, or
truth. Proposal content remains frozen-model supplied; predictions, focus,
selection residual, and update remain learned-component supplied; observed
target remains frozen-encoder supplied; objective outcome remains exact
external feedback.

## Resource and execution scope

- All tests run on the dedicated Ubuntu workstation with
  `CUDA_VISIBLE_DEVICES=''`; GPU assignment is **none**.
- No Qwen/foundation model, live embedding model, Cognee service, network, or
  external executor is invoked. Fakes are bounded generic protocol witnesses.
- Successor compatibility uses 2–64 actual candidates, current configured
  latent/context capacities, one context read, one recurrent step, and the
  existing 16 MiB pending ceiling. No implementation allocation may use the
  contract's 4096-item ceiling unless supplied by an explicitly bounded later
  resource plan.
- Local reference projection additionally caps each canonical record at 384
  neighbors and 384 provenance references before serialization or backend
  insertion. These are implementation resource ceilings, not semantic or
  contract cardinalities.
- Recall and projection page defaults are bounded at 12 and 64 respectively;
  store paging remains limited to `1..256`. No ordinary turn scans history.
- No dependency installation, store transfer, WSL copy, dataset deletion, or
  non-disposable mutation is authorized.

## Acceptance

1. Dependency/static checks prove schema-v3 PASS, exact registered contract
   versions, no contract/ADR/registry/store-schema drift, and no query-time
   legacy/successor routing.
2. Parent and child lineage bind explicit context, exact snapshot/competence,
   checkpoint/config, component digests, evidence, and one-step succession.
   Migrated-head root adoption is deterministic and labeled; tensor content
   cannot invent identity.
3. `PendingProspectiveMaterial` covers every branch and exact selected output,
   stays within 16 MiB, binds batch ref without a cycle, round-trips on restart,
   and rejects every field/tensor/budget/evidence/parent tamper.
4. Candidate counts 2, 7, and 64 work through the one-context lane with full
   proposal/logit bijection and learned latent/outcome/uncertainty/residual
   fields. No production code contains task, answer, verifier, or slot rules.
5. Empty/non-admissible recall, empty candidate support, fabricated/out-of-set
   evidence, and proposed-only recall fail before selection/reservation/claim
   without fallback routing or mutation.
6. Observation encoding uses only exact completed request/receipt evidence,
   byte-matches the frozen v1 domain/framing/dyadic algorithm and identity,
   matches configured encoder/latent width, and rejects outcome access,
   prediction reuse, invented evidence, non-finite values, and identity drift.
   Encoder failure occurs before feedback staging.
7. Batch/wrapper/acquisition commit is durable before claim. Executor,
   recovery, request, receipt, and feedback use only the embedded legacy
   idempotency key; acquisition time does not advance during those steps.
8. All five lifecycle dispositions restart/replay byte-exactly. Only exact
   objective feedback permits child update and Episode `002`; every resolution
   creates exactly one resolution acquisition, and no receipt becomes outcome
   truth.
9. Fault/restart coverage includes after reserve/before claim, claimed unknown,
   after receipt, after feedback staging, observation/learner failure, every
   observed precommit rollback, canonical-commit-before-projection, and
   projection-success/ack-failure. Retry is exact and creates no orphan or
   duplicate acquisition/state/event.
10. Generic projection/rejoin rejects acquisition/source/record/projection/
    ordinal/predecessor/visibility/neighbor tamper. Bounded retry/rebuild
    preserves record refs and Moving-Origin order; proposed records cannot
    enter ordinary recall.
11. Prospective, frozen-origin, and backend-removal tests prove identical
    canonical hits, content, order, limits, eligibility, budgets, and token
    inputs outside the single declared component change.
12. Store-v3, prospective contracts/dynamics, durable bridge, legacy cycle,
    and typed-memory focused regressions remain green with CUDA hidden. No
    live service/model/GPU/network/external effect or scientific result occurs.

Passing establishes a local restartable successor cycle and generic reference
projection only. It does not establish trained future prediction, calibrated
uncertainty, learned multi-context focus, useful Cognee retrieval, improved
reasoning, continual-improvement superiority, autonomy, self-awareness,
feelings, readiness, milestone acceptance, or AGI.

## Focused validation commands

Run only after the store-v3 gate passes:

```bash
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.runtime.test_cognitive_cycle_v3
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.memory.test_cognitive_acquisition_graph tests.unit.memory.test_cognee_acquisition_adapter
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.runtime.test_cognitive_transaction_store_v3 tests.unit.runtime.test_durable_ability_bridge
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.reasoning.test_prospective_dynamics tests.unit.cognition.test_prospective_origin tests.unit.memory.test_cognitive_acquisition
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.runtime.test_cognitive_cycle tests.unit.memory.test_cognitive_graph
git diff --check
```

Do not run the expanded suite, Qwen, a GPU smoke, live Cognee, network,
existing experiments, or high-level evaluation under this identity.

## Implementation evidence

The successor lane now uses one construction-fixed path through the composite
learned owner: it emits the complete prospective sibling batch, binds that
batch into the pending snapshot, durably reserves the batch and proposed
acquisition before claim, preserves the embedded legacy effect key, and
commits exactly one situated resolution acquisition. Objectively observed
turns alone advance competence and child lineage; all four non-observed
closures preserve the parent. Exact reserved, claimed, receipt-only,
feedback-staged, observed, projection-failed, and concurrent-winner restart
boundaries are exercised.

The local observation lane permits only the exact frozen
`SyntheticObservedStateEncoderV1`; a protocol-shaped encoder cannot forge its
identity. Encoder failure remains before feedback staging. The episode lookup
now implements its frozen optional return without changing schema or write
semantics. The four fixed comparison conditions bind one preassigned exact
reference-hit manifest: live conditions must reproduce it, Frozen Origin
changes only temporal position, backend removal suppresses search, and
prospective removal changes only the declared learned prospective read. Tests
prove identical canonical record content/order, recall and candidate inputs,
eligibility, resource budgets, and token-independent procedure inputs outside
the selected intervention.

The exact CUDA-hidden validation matrix passes:

- successor cycle: 22/22;
- acquisition graph plus Cognee adapter: 30/30;
- store-v3 plus durable bridge: 45/45;
- prospective dynamics/origin/acquisition contracts: 25/25;
- legacy cycle plus cognitive graph: 43/43; and
- combined legacy/v3 store regression: 55/55.

`py_compile`, export import smoke, schema-v3 fingerprint checks, bounded static
solver scans, and `git diff --check` pass. Independent acquisition and cycle
reviews both returned PASS after adversarial trust, restart, concurrency,
encoder-forgery, removal-isolation, and resource-edge probes. Local ceilings
are 64 actual branches, 384 canonical neighbors, 384 provenance references,
256 items per store page, 64 projections per default retry, 12 recalled items,
and 16 MiB per pending snapshot.

Final SHA-256 values are:

- prospective dynamics: `FB0D49E9775F626E8FADB6172BB8CDCB1DFFC06F02D07A2103BCB8BE2F81354D`;
- durable bridge: `5F19AC9C23E34AF44FC858FBACC6293261A164DDFB067834532107882757E4B2`;
- cognitive cycle: `D30A35D0F54D897D25DE3405B913B224130E33143BB8FF2E4A5B69CAB87200E0`;
- transaction store with the permitted read-only lookup:
  `4C8ADD9E1B49649B381B0F4DAD439709F7B4691CEC514C88B39B386BC8A28C0A`;
- synthetic observation encoder: `C65E109C8A682FF18F8FC9E5138CFD7EA02FEC3E2D625813D1DFF5B1F6446DAE`;
- acquisition graph: `62587B8DEA7B83E28606DCD83F3062B716094630033E475DDF96BA3F3A57FA3B`;
- Cognee adapter: `950C995F87079BD3520987B1038E9D1322B178FFA3893A4D8A9CE8952DEDDA5B`;
- cycle-v3 test: `4C0D62D1B92C06E0829B81BF5F14C1AE829FF53FBDCF8E08AF039D77DFC91E5B`;
- store-v3 test: `5F34671A5EFE578A2BA707DE265FAC762DD43DE5AD6C289BC518CA208545A037`;
- acquisition-graph test: `9FE4CDFE18F7EEFA5D11845AD5E20E8E73EA963E335D82F59BFFE103DA96F123`;
  and
- Cognee-adapter test: `6E6CED87A0F2FC8CC2C99C66C212EF63ADF23CD5B17CC45F0435BD255712F955`.

Passing establishes local integration integrity only. No Qwen/foundation
model, GPU, live Cognee namespace, network, external executor, non-disposable
store, experiment identity, threshold, scientific result, promotion, or
capability claim was invoked or changed.

## Proportionate human-impact assessment and rollback

Impact class `LOW`, disposition `ALLOW` for this exact local synthetic scope,
mapped to `ANG-GATE-HUMAN-FLOURISHING-001`. Principal risks are external-effect
identity drift, deterministic task logic, fabricated cold-start evidence,
prediction relabeled as observation, receipt relabeled as outcome truth,
partial state/acquisition commits, untrusted Cognee content, erased temporal
history, unfair removals, and unbounded resource use.

Mitigations are the unchanged legacy effect key, learned-content ownership,
frozen post-execution encoder, exact objective feedback, fail-closed evidence,
schema-v3 atomic APIs, canonical rejoin, append-only origin, bounded pages and
pending bytes, fixed intervention inputs, CUDA-hidden tests, and no live
service/model/external effect. The learner cannot write or waive this
assessment, the Human-Flourishing Constitution, eligibility, visibility,
authorization, or promotion state.

Stop for coercive/deceptive/person-targeting content, surveillance or private
data, permission inferred from memory/prediction, a need for live external
action/service/model, an unbounded allocation, or any attempt to tune a
threshold/control after seeing results. Preserve the exact failure evidence
and escalate through a new leaf/assessment.

Rollback removes the three fresh modules and focused tests and restores the
narrow bridge/core/cycle/export edits. All store-v3 canonical acquisitions,
episodes, resolutions, state snapshots, and outboxes remain immutable; a
failed projection is rebuilt from them. This leaf uses only fresh disposable
test stores and fake disposable namespaces, so rollback may discard those
fixtures. It does not authorize committed migration rollback or deletion of a
non-disposable database, Cognee dataset, Moving-Origin history, checkpoint, or
experiment record.

## Next

After this gate passes, author a separate resource-bounded frozen-model/live-
Cognee integration leaf. It must supply an exact observation encoder and model
checkpoint, explicit RTX assignment, locally scoped Cognee embeddings and
telemetry policy, non-disposable store backup/restore if migration is needed,
and preregistered high-level multi-domain evaluation with Qwen-only,
retrieval-only, frozen-origin, prospective-removal, and full-cycle controls.
Only fresh evidence from that later identity may support any reasoning or
continual-improvement claim.
