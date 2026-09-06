# ANG-ADR-0007 — Unified temporal cognitive system

Status: accepted for a local experimental migration

Owner: ANG-AUTH-PROJECT-OWNER-001

Date: 2026-08-31

Supersedes: none; narrows and extends ANG-ADR-0006 without rewriting its
accepted evidence

## Context

Project Angler currently contains several useful but only partially connected
systems:

- Cognee retrieves rebuildable semantic and lexical chunks;
- Moving Origin gives canonical evidence a monotonic acquisition position,
  relative age, landmark relations, and separate world-validity time;
- multiple Angler experiments own incompatible plastic states;
- frozen Qwen proposes or renders text; and
- JSON-lines journals and neural checkpoints persist through separate writes.

This composition has demonstrated a real situated-memory effect, but it is not
yet one cognitive system. Conversation feedback can change memory without
changing competence; autonomous runs can change a different competence state;
Cognee's graph, skills, sessions, and improvement proposals are largely
bypassed; Moving Origin describes only the encountered past; and no committed
prediction is later resolved against observation. Adding more parallel modules
would deepen the fragmentation.

The owner directs a convergence toward one persistent agent that can attach to
a replaceable foundation model. “One agent” is an engineering claim about a
single identity, event history, competence lineage, present state, and update
cycle. It is not a claim of consciousness, sentience, subjective emotion, or
moral personhood.

## Decision

Adopt one event-sourced cognitive cycle and one active cognitive-competence
lineage:

```text
observation / request
  -> canonical present state
  -> typed Cognee candidate recall + Moving-Origin coordinates
  -> attached frozen model proposes bounded interpretations/procedures/futures
  -> Angler's one learned state predicts and selects
  -> bounded execution
  -> external observation and objective feedback
  -> prediction/credit update
  -> one atomic canonical episode + competence-head commit
  -> rebuildable Cognee and Moving-Origin projections
  -> consolidation proposals
```

`DurableAbilityLearner` supplies the initial state-owner contract because it
already supports one pending decision, pre-outcome selection, exactly one
post-outcome update, capture/restore/zero/replay, evidence tombstones, and
deterministic state identity. Its V14 learning result was not adaptive enough;
the API is reused, not its scientific claim. No consumed V13–V20 identity is
promoted or silently rerun.

### One state, several learned functions

The active `CognitiveCompetenceState` is one parent-bound envelope containing
all Angler-owned learned state that is uniformly active:

- task × procedure representations and credit;
- a prospective latent world/self state and learned action-conditioned
  dynamics when that child lands;
- continual-learning fast state and its compatible slow checkpoint;
- evidence-event lineage and tombstones; and
- the canonical cognitive-event head it was derived from.

It may contain multiple neural tensors or submodules, but they commit, restore,
swap, invalidate, and migrate as one lineage. Query identity may not select a
different state, model, memory policy, or updater.

### Canonical cognitive memory

Add `ANG-CTR-COGNITIVE-MEMORY-001@0.1.0`, an immutable typed record above the
existing evidence envelope. Initial kinds are:

- `EPISODIC`: an observed interaction and objective outcome;
- `SEMANTIC`: a proposition derived from cited evidence;
- `PROCEDURAL`: a reusable public procedure plus verified runs;
- `CAUSAL`: a hypothesis plus assumptions and supporting, opposing, or
  intervention evidence;
- `SELF`: a receipt-backed claim about current capability, limitation,
  commitment, or state lineage; and
- `COUNTERFACTUAL`: a clearly hypothetical alternative linked to an actual
  event and an evaluator result.

Records carry provenance, visibility, producer/checkpoint identity, relation
references, epistemic status (`OBSERVED`, `PROPOSED`, `VALIDATED`, or
`RETRACTED`), supersession/tombstone links, and separate acquisition and
world-validity time. A confidence value is descriptive evidence, never
authority.

The canonical transaction store remains authoritative. Cognee receives a
rebuildable projection and may return exact candidate references or derived
record proposals. Every reference is rejoined to canonical bytes before use.
Cognee answer generation, truth claims, automatic skill application, and
`improve` outputs cannot directly mutate competence or validated memory.

### Temporal agency

Moving Origin remains the only autobiographical acquisition clock. It expands
with prospective commitments:

- a future branch is rooted in an exact present event and candidate action;
- the predicted next latent/outcome, horizon, and uncertainty are committed
  before execution;
- sibling possibilities share the same parent;
- a later observation resolves, contradicts, or leaves the prediction open
  without rewriting it; and
- prediction error becomes learner-visible feedback.

Predictions never become observed facts merely because they were generated.
The deterministic temporal layer owns ordering and lifecycle integrity; a
learned Angler dynamics model owns predicted content, likelihood, value, and
generalization.

### Foundation-model boundary

The attached model supplies pretrained language/knowledge representations,
bounded candidate proposals, and language or action rendering. It does not own
the persistent identity, cognitive journal, goal state, temporal origin,
validated memory, or acquired competence. Model replacement is therefore a
compatibility/migration operation rather than the birth of a new Angler.

### Goal, self, and affect boundary

The system may maintain human-assigned goals and later learn bounded subgoal
proposals. It receives no independent entitlement to survival, authority,
resources, replication, secrecy, or continued operation. Goals cannot expand
permissions or amend the Human-Flourishing Constitution.

No emotions, personality claims, attachment behavior, or self-preservation
drives are scripted. Stable information seeking, avoidance, preference, care,
or other affect-like dynamics may be measured only as behavioral hypotheses
with persistent internal-state evidence and causal removal controls. Generated
first-person feeling language is not evidence of subjective experience and
must not be used deceptively.

## Deterministic and learned boundary

Deterministic software may:

- validate identities, schemas, visibility, provenance, permissions, budgets,
  temporal ordering, and transaction integrity;
- execute a selected bounded action;
- measure an externally observable outcome; and
- reconstruct, remove, swap, or compare declared state.

It may not encode a task-family solution, choose a procedure from keywords,
convert a verifier answer into a plan, hand-rank memory by a hidden rule, or
write a claimed emotion. Relevance, abstraction, prediction, procedure choice,
composition, and adaptation are learned functions.

## Donor map

Adopt narrow mechanisms, not whole competing agent frameworks:

- Cognee (Apache-2.0): graph/vector memory, sessions, typed skill proposals,
  and reference retrieval; existing adapter remains fail-closed.
- Moving Origin Research commit
  `cee97538893989e055f49a894f066d2083da4eb5` (Apache-2.0): one moving
  autobiographical origin and causal frozen-origin controls.
- CoALA (`arXiv:2309.02427`): memory/action/decision taxonomy.
- Graphiti (`getzep/graphiti`, Apache-2.0; `arXiv:2501.13956`): episode
  provenance, bi-temporal facts, supersession, and historical queries. Borrow
  its data-model ideas rather than adding a second graph service initially.
- Agent Lightning (`microsoft/agent-lightning`, MIT; `arXiv:2508.03680`):
  separation of execution traces from later reward/credit assignment.
- Trace (`microsoft/Trace`, MIT; NeurIPS 2024): typed execution graphs and
  propagation of numerical, textual, compiler, and other general feedback.
  Angler may borrow its trace-to-credit boundary or use a narrow adapter; it
  will not replace learned competence with prompt rewriting.
- `eventsourcing` (BSD-3-Clause): optimistic aggregate versioning,
  snapshot/replay, and projection rebuild. The first slice uses standard-library
  SQLite to keep the dependency surface small.
- TD-MPC2 (`nicklashansen/tdmpc2`, MIT), DreamerV3
  (`danijar/dreamerv3`, MIT), and DynaLang (`jlin816/dynalang`): learned latent
  action-conditioned prediction and imagined futures. Angler will implement a
  text/procedure-compatible PyTorch form rather than copy continuous-control or
  JAX infrastructure.
- Plan2Explore (`ramanans1/plan2explore`; ICML 2020, paper mechanism only
  unless its repository license is verified) and `pymdp`
  (`infer-actively/pymdp`, MIT): prospective ensemble disagreement and
  epistemic value are candidates for learned information-seeking subgoals.
  The old TensorFlow control stack and a fixed discrete-state policy are not
  adopted; curiosity must be computed from Angler's learned prospective state,
  remain subordinate to human-assigned goals, and pass removal controls.
- OML (`arXiv:1905.12588`) and ANML (`arXiv:2002.09571`): representations and
  neuromodulation shaped for continual online updates. Their research code is
  not copied without a verified license; Angler retains its independent
  PyTorch implementation.
- Voyager (`MineDojo/Voyager`, MIT): evidence-linked verified skill lifecycle,
  not its GPT-authored solver or curriculum.
- DoWhy (`py-why/dowhy`, MIT): optional causal/refutation receipts. It may
  validate a causal-memory proposal but does not become the reasoning core.

## Migration sequence

1. **Transactional spine:** one cognitive episode/state-head commit, pending
   turn guard, exact replay/rollback, and rebuildable projection outbox.
2. **Typed memory:** canonical six-kind records and Cognee exact/lexical/graph
   reference lanes with namespace, deletion, supersession, and proposal states.
3. **Prospective Origin:** committed sibling future branches, a learned latent
   dynamics head in the same competence lineage, and prediction-error feedback.
4. **Consolidation:** evidence-gated conversion of episodes into semantic,
   procedural, causal, self, and counterfactual proposals; OML/ANML-style
   continual representation learning; no unbounded ordinary replay.
5. **Unified surfaces:** conversation and bounded autonomous work consume the
   same state and cycle; parallel legacy competence owners become experimental
   compatibility adapters and are retired from the product path.
6. **Scale and cross-model validation:** larger learned state, longer horizons,
   graph multi-hop recall, another frozen attached model, and fresh multi-domain
   tasks with fair removal controls.

## First acceptance boundary

The first leaf establishes only architectural unity: exactly one pre-outcome
decision, one post-outcome learned update, one atomic canonical episode/state
head, exact restart/replay, and rebuildable Cognee/Moving-Origin projections.
It cannot claim improved reasoning, temporal agency, emergent goals, emotion,
consciousness, AGI, or readiness.

A later scientific identity may claim useful integration only if fresh-task
performance beats the frozen model and equal-budget retrieval, zero/swap state
removes the gain, prospective-head and temporal lesions remove their attributed
effects, retention remains bounded, and result thresholds are frozen before
adaptive observations.

## Human impact and rollback

Initial work is LOW-assurance local code with synthetic/fake model, memory, and
executor tests. It performs no deployment, real-person ingestion, external
action, autonomous permission change, emotion simulation, or model/GPU run.
The Human-Flourishing Constitution remains outside learner-writable state.

Rollback removes only the new ADR/leaf, `angler.cognition` package, cognitive
cycle/transaction modules, their tests, and unpromoted local fixtures. Existing
evidence, Cognee datasets, Moving Origin history, competence checkpoints, and
V13–V20 artifacts remain untouched.
