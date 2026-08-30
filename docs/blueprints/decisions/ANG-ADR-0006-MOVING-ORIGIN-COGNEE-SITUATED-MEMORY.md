# ANG-ADR-0006 — Moving Origin and Cognee situated memory

Status: accepted for a non-promoted proof of concept

Owner: ANG-AUTH-PROJECT-OWNER-001

Date: 2026-08-30

Supersedes: none

Superseded by: none

## Context

Angler learns procedures but lacks a general external memory that can retrieve
evidence by meaning and place it relative to the system's own continuing
experience. Cognee supplies a graph/vector memory system. Moving Origin
Research supplies an experimentally tested operational mechanism for a
monotonic autobiographical `now` and self-relative temporal coordinates.

Neither component is itself the reasoning core. Cognee can recover content
without acquiring a transferable procedure. Moving Origin can situate an
event without deciding what to do. The integration must preserve Angler's
separation between episodic evidence and learned competence and must not
become a deterministic answer solver, a query-selected adapter router, or a
claim about awareness.

The exact upstreams inspected for this decision are:

- Moving Origin Research, local Git commit
  `cee97538893989e055f49a894f066d2083da4eb5`, Apache-2.0, copyright Rebecca
  R. McClintic. Its E1 result supports the bounded claim that the maintained
  index produced oracle-equivalent answers, scaled operationally better than
  a fair scan, and was load-bearing for its declared recency consumer. Its M3
  program is not fully validated and is not represented as such here.
- Cognee, <https://github.com/topoteretes/cognee>, inspected commit
  `690c0ec023719a2a277dc893cdecfec1ca8012cc`, package version `1.5.3`,
  Apache-2.0.

## Decision

Adopt a three-layer, one-way composition:

```text
canonical Angler evidence
  -> MovingOriginIndex (one monotonic acquisition ordinal)
  -> visibility-filtered disposable projection
  -> Cognee graph/vector candidate retrieval
  -> projection identity + visibility revalidation
  -> SituatedRecall(content, provenance, age, landmarks, world validity)
  -> Angler's one active learned procedural core
```

### State ownership

1. **Angler EVIDENCE remains canonical.** Cognee is a rebuildable projection,
   never the source of truth and never the owner of procedural competence.
2. **Moving Origin owns only autobiographical coordinates.** One contiguous
   integer ordinal advances on each registered event. No wall clock, Cognee
   timestamp, query, task ID, or model output may advance a private competing
   clock.
3. **World-validity and acquisition time are separate.** `world_valid_from`
   and `world_valid_until` describe when a fact applies in the represented
   world; `acquired_ordinal` describes when Angler encountered it.
4. **Cognee proposes candidates only.** The PoC uses raw `CHUNKS` retrieval,
   not Cognee answer generation. Backend order is retained. Moving-origin age
   and landmark relations are features for a learned consumer, not a
   hand-written recency answer rule.
5. **Every recall is distrusted and rejoined.** The projection content digest,
   evidence reference, and Angler visibility class must match the local
   temporal anchor before a `SituatedRecall` is emitted. Unknown, stale,
   malformed, or unauthorized hits are rejected with non-payload-bearing
   reason codes.
6. **One competence state remains uniformly active.** Neither Cognee results
   nor temporal coordinates may select a model, adapter, skill state, or
   updater by query identity.

### Failure and deletion

The temporal event is recorded before the disposable projection is written.
A Cognee failure therefore cannot erase canonical history; `reproject()` can
retry from the matching evidence projection. `forget_projection()` deletes
only the dedicated Cognee dataset and deliberately retains canonical Angler
evidence and moving-origin state. Destruction of canonical evidence remains an
EVIDENCE-owned operation outside this adapter.

Cognee ingestion and retrieval may invoke configured embedding or language
model providers. The concrete adapter fails closed unless the caller declares
that local models are configured or explicitly authorizes external model
calls. It does not mutate Cognee's global configuration or silently accept
Cognee's provider defaults. Cognee also enables product telemetry by default;
every adapter operation fails closed unless `TELEMETRY_DISABLED` is set or the
caller explicitly authorizes telemetry.

## Alternatives considered

- **Use Cognee temporal retrieval alone.** Rejected because document/world
  dates do not establish Angler's own acquisition position, frozen-origin
  intervention, or one-clock invariant.
- **Store mutable relative ages in Cognee.** Rejected because every tick would
  rewrite the corpus and make the projection authoritative. Store stable
  acquisition ordinals; resolve `now - acquired_ordinal` at recall.
- **Hard-code recency-weighted ranking.** Rejected as a task policy disguised
  as memory. The learner must establish whether temporal position helps.
- **Merge the Moving Origin repository or its governance system.** Rejected.
  Angler independently implements the small mechanism and imports none of the
  donor's workflow machinery or incomplete M3 claims.
- **Let external memory select procedural adapters.** Rejected by
  `ANG-INV-ONE-COMPETENCE-001`.

## Consequences

The system gains a model-agnostic path for long-lived, provenance-bearing,
temporally situated recall. It also gains clean causal controls: frozen
origin, fair-naive scan, Cognee removed, competence reset, and state swap.
Operational complexity increases when Cognee is actually deployed: graph,
vector, relational, embedding, and possibly LLM providers must be configured,
isolated, monitored, and deleted coherently. The adapter boundary allows
Cognee to be replaced without changing the reasoning core.

This decision does not establish that situated memory improves reasoning. It
only makes that claim falsifiable.

## Affected requirements, invariants, blueprints, contracts, tests, and gates

- Preserves `ANG-INV-EVIDENCE-SEPARATION-001`,
  `ANG-INV-ONE-COMPETENCE-001`, `ANG-INV-CAUSAL-PROMOTION-001`, and
  `ANG-INV-HUMAN-FLOURISHING-001`.
- Defines prototype contract `ANG-CTR-SITUATED-RECALL-001@0.1.0`, represented
  in code by `MemoryProjection`, `TemporalPosition`, and `SituatedRecall`.
- EVIDENCE owns canonical event/acquisition identity; RUNTIME consumes recall;
  LEARNING may use only declared recall features; SCIENCE owns causal tests.
- Unit evidence is `tests/unit/memory/test_situated_memory.py`.
- No normal EVIDENCE, RUNTIME, LEARNING, Human-Flourishing, slice, milestone,
  or promotion gate is passed by this PoC.

## Evidence and PoC acceptance criteria

The local no-network PoC is acceptable when:

1. live age re-resolves as the one origin advances while the frozen-origin arm
   retains all content but reports its birth coordinate;
2. maintained and fair-naive recent queries return identical answers while
   deterministic inspected-item accounting grows with `k` versus full history;
3. snapshot/restore preserves the predecessor chain and coordinates;
4. Cognee candidates retain backend order, and malformed, unknown, stale, or
   visibility-denied projections never reach `SituatedRecall`;
5. world-validity is evaluated separately from acquisition age;
6. Cognee model-provider use fails closed without explicit local/external
   configuration; and
7. projection deletion leaves canonical temporal state intact.

Initial local evidence on 2026-08-30: the focused suite passed 11/11, the
unchanged Evidence schema suite passed 16/16, all seven new Python files parsed,
the package imported without Cognee installed, `pyproject.toml` parsed with the
declared optional dependency, and `git diff --check` passed. This is component
evidence only, not a live Cognee or reasoning-improvement result.

A later live Cognee trial uses one self-hosted dataset and synthetic evidence.
Its scientific successor must compare full, frozen-origin, fair-naive,
Cognee-only, Cognee-removed, and competence-reset arms on temporally dependent
multi-step tasks. Full integration is supported only if it beats the best
component/control and each ablation removes the effect attributed to it.

## Migration plan

The PoC lives under `angler.memory` and is not connected to the promoted
runtime. After contract review, EVIDENCE may add acquisition/bi-temporal
references to a successor Episode projection without changing the immutable
Episode v1 schema. RUNTIME then consumes `SituatedRecall` through an explicit
retrieval-assisted mode, and LEARNING/SCIENCE receive a new evaluation identity
before any adaptive result is viewed.

## Rollback plan

Remove `src/angler/memory`, its unit tests, the optional `memory` dependency,
and this ADR's documentation references. If a live Cognee PoC exists, delete
only its dedicated dataset and retain canonical Angler evidence. Existing
procedural states and experimental evidence are untouched.

## Expiration or review condition

Review before the first live Cognee ingestion, any real-person data, any
external model/provider call, any change to Episode or visibility contracts,
or any attempt to connect recall to a promoted competence state.
