# ANG-WORK-RUNTIME-UNIFIED-COGNITIVE-CYCLE-V1-001

Status: component implementation accepted on the dedicated Ubuntu workstation

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decision: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`

Assurance: LOW; local standard-library/PyTorch unit work with fakes and no live
model, Cognee service, GPU, network, external effect, or real-person data

## Purpose

Create the transactional nucleus that all later Angler conversation and
autonomous surfaces will share. The leaf joins one replaceable model adapter,
one situated-memory boundary, one temporal origin, one competence owner, one
executor, and one canonical cognitive event/state store. It does not create a
second learner or claim capability improvement.

## Frozen implementation scope

Fresh files:

- `src/angler/cognition/__init__.py`
- `src/angler/cognition/contracts.py`
- `src/angler/runtime/cognitive_transaction_store.py`
- `src/angler/runtime/cognitive_cycle.py`
- `tests/unit/cognition/test_contracts.py`
- `tests/unit/runtime/test_cognitive_transaction_store.py`
- `tests/unit/runtime/test_cognitive_cycle.py`

Narrow export edits may touch `src/angler/runtime/__init__.py` only after the
focused tests pass. Append-only coordination may touch `AGENTS_SYNC.md`.

Consumed V13–V20 files, current V20 implementation, old journals/checkpoints,
Cognee/Moving-Origin modules, foundation-model files, reports, and outputs are
read-only.

## Contracts

`CognitiveMemoryRecord` is immutable and content-addressed. It validates the
six record kinds, four epistemic states, typed relation references,
provenance/visibility, producer and competence identities, separate
world-validity bounds, supersession/tombstone references, and canonical JSON
round-trip.

`CognitiveEpisode` records one complete turn: task/request, recalled canonical
refs, all public proposals, selected proposal, a pre-action prospective
commitment, observable trajectory/result, external outcome/feedback, and exact
parent/child competence digests.

`CognitiveTransactionStore` uses one SQLite transaction to append the episode,
store the matching competence snapshot, advance the event/state head, and
enqueue rebuildable projections. Parent-head mismatch fails without mutation.
Projection acknowledgement is separate and cannot change canonical state.

`CognitiveCycle` reuses the existing frozen-model text adapter,
procedure-relation adapter, `DurableAbilityLearner`-compatible state owner, and
`SituatedMemory` recall contract. It allows exactly one pending turn. It scores
before feedback, executes only the selected public procedure, applies exactly
one learner update only after caller-supplied objective feedback, commits the
episode/state atomically, and then attempts rebuildable projection.

If projection fails after canonical commit, the committed outbox remains
pending and the caller receives a projection-pending result; competence is not
silently rolled back beneath a committed episode. Failures before canonical
commit restore the exact parent learner snapshot.

## Deterministic-code ceiling

The leaf may validate, hash, serialize, transact, execute a caller-selected
bounded procedure, and record external outcomes. It may not inspect task
families, expected answers, verifier internals, hidden labels, or solution
operators; generate a correction; hand-rank memory; or infer an emotion.

All fake test components must be generic protocol witnesses. Tests may assert
which candidate a deliberately configured fake learner selects, but production
orchestration contains no task-to-procedure mapping.

## Acceptance tests

1. Six memory kinds and all epistemic states round-trip byte-canonically;
   malformed visibility, provenance, validity, relation, and supersession
   structures fail closed.
2. A transaction stores one episode and exactly matching state snapshot/head;
   restart returns identical bytes, digests, record, and pending projection.
3. Optimistic parent mismatch, duplicate event, invalid snapshot, and injected
   precommit failure leave database bytes/logical head unchanged.
4. Projection acknowledgement is idempotent; a failed projection stays pending
   and cannot corrupt the canonical head.
5. The cycle accepts exactly four distinct proposals, cites only validated
   recalled evidence, and permits only one pending turn.
6. No learner update occurs before objective feedback or after clarification,
   executor error, or a feedback identity mismatch.
7. Successful and failed outcomes each cause exactly one post-outcome update,
   produce a typed episode, and atomically bind parent/child state digests.
8. Injected learner/store failure restores the exact parent state; injected
   projection failure preserves the committed episode/state and exposes its
   outbox item for rebuild.
9. Restart from the canonical store restores the same competence bytes and
   head; tombstone/rebuild hooks remain explicit and cannot be selected by
   query identity.
10. Existing focused memory/runtime tests remain green. No V20 runner, model,
    Cognee service, GPU, network, or experimental output is invoked.

## Stop conditions

Stop for any need to edit consumed experimental evidence, introduce a second
adaptive state, let Cognee establish truth/competence, expose hidden verifier
data, encode a task solution, weaken the Human-Flourishing Constitution, add a
dependency, or run a model/GPU/live service under this component identity.

## Next leaf

After component acceptance, add typed canonical memory plus Cognee
reference-only graph expansion. Then add Prospective Origin and a learned
action-conditioned world model inside the same competence envelope. Only a
fresh later identity may run Qwen/Cognee/GPU capability comparisons.

## Result

The contracts, SQLite transaction store, cognitive cycle, and narrow runtime
exports are implemented. Independent review confirmed canonical episode
revalidation, bounded current-tail corruption detection, and one-winner
same-parent concurrency. On `angler-workstation`, the focused contracts,
transaction, cycle, durable-ability, and situated-memory suite passed 61/61 in
1.672 seconds with CUDA hidden. This establishes the component spine only; it
does not claim improved reasoning or conversation readiness.
