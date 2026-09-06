# ANG-WORK-RUNTIME-TYPED-COGNITIVE-REFERENCE-PROJECTION-V2-001

Status: accepted — local component scope only

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decision: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`

Assurance: LOW; local deterministic projection code and fake-backend tests on
the dedicated Ubuntu workstation. No model, GPU, live Cognee service, network,
external action, or real-person data is authorized by this leaf.

## Purpose

Make the accepted cognitive episode store, Cognee memory, and Moving Origin
share one verifiable identity path. Each committed episode has one
deterministic typed `EPISODIC` memory view. Cognee may index and return exact
references, while every usable hit is reconstructed from canonical SQLite
episode bytes and joined to Moving Origin coordinates.

This leaf does not add another canonical database, learner, answer generator,
or truth source. It does not claim improved reasoning.

## Implementation scope

Fresh files:

- `src/angler/memory/cognitive_graph.py`
- `src/angler/memory/cognee_cognitive_adapter.py`
- `tests/unit/memory/test_cognitive_graph.py`
- `tests/unit/memory/test_cognee_cognitive_adapter.py`

Narrow edits:

- `src/angler/runtime/cognitive_transaction_store.py`
- `tests/unit/runtime/test_cognitive_transaction_store.py`
- `src/angler/runtime/cognitive_cycle.py`
- `tests/unit/runtime/test_cognitive_cycle.py`
- `src/angler/memory/__init__.py`
- `src/angler/runtime/__init__.py` only if an export is required
- this leaf and append-only `AGENTS_SYNC.md`

Consumed experimental files, checkpoints, results, legacy journals, Qwen
modules, and existing Cognee datasets are read-only.

## Contracts

The transaction store exposes verified bounded episode access as
`CognitiveEpisodeItem(sequence, episode_ref, episode)`, with exact lookup and
paged iteration. It parses through the existing canonical verification path;
it never returns unchecked database payloads.

`episode_record(item)` deterministically derives one
`CognitiveMemoryRecord`:

- kind `EPISODIC`, epistemic state `OBSERVED`;
- content limited to public request, selected procedure, response, objective
  outcome, and feedback;
- provenance binds the episode and feedback source;
- producer/checkpoint, competence, visibility, world-validity, and acquisition
  ordinal remain exact;
- typed `ACTUAL_OF` and `ABOUT` links identify the prospective commitment and
  cited evidence without rewriting their semantics.

`CognitiveGraphProjection` is a strict content-addressed envelope containing
the typed record and source episode reference. A reference backend may project,
search, and forget only its rebuildable namespace. Search output contains
source/record references and optional adjacent record references, never trusted
record content.

`TypedSituatedMemory` owns projection/retrieval/rebuild orchestration. It:

1. derives the record from a verified episode item;
2. appends exactly one Moving Origin anchor for the record reference;
3. projects the typed envelope to Cognee;
4. reconstructs every returned record and neighbor from the canonical episode
   store and compares exact content identities before exposing it; and
5. treats Cognee and Moving Origin failures as projection failures, leaving the
   accepted episode/state commit and outbox intact.

The Cognee adapter borrows Cognee's Apache-2.0 structured `DataPoint`, typed
relationship, semantic-index, and dataset/NodeSet scoping mechanisms. It uses
deterministic identifiers and direct structured insertion. LLM extraction,
graph completion, automatic feedback, `improve`, `memify`, and answer
generation are excluded.

Moving Origin remains the only acquisition clock. For the first derived record
its zero-based origin ordinal must equal store sequence minus one. Projection
retry reuses the existing anchor; rebuild pages canonical episodes and can
recreate both disposable views without rewriting history.

## Deterministic-code ceiling

Code may derive a typed public record, validate identities, serialize, page,
rejoin exact references, maintain temporal ordering, and measure removal
controls. It may not rank by task keywords, infer truth, solve a task, convert
feedback into a procedure, invent relations, or use graph position as hidden
supervision. Backend scores may seed candidates but cannot bypass canonical
validation.

## Acceptance

1. Episode item lookup and paging are bounded, ordered, canonical, and reject
   unknown or corrupted rows.
2. The same episode always produces byte-identical record and projection refs;
   tampering changes or invalidates identity.
3. All registered relation types present in records round-trip without
   semantic rewriting.
4. One committed episode creates exactly one record anchor; retry does not
   duplicate it.
5. Valid exact references recall canonical content with live Moving Origin
   coordinates; unknown, forged, stale, visibility-denied, or mismatched refs
   are rejected.
6. Every graph neighbor is independently revalidated; one poisoned neighbor
   cannot become trusted memory.
7. Backend failure after canonical commit leaves the episode/state intact and
   its projection outbox pending; retry acknowledges only after success.
8. Clearing the disposable backend and rebuilding from paged canonical
   episodes yields the same record refs and Moving Origin snapshot.
9. Fake structured-Cognee tests prove deterministic ids, dataset isolation,
   typed edges, and no LLM/improve path. Installed Cognee API compatibility is
   inspected separately before a live-service leaf.
10. Existing 61 focused cognitive/runtime/memory tests remain green on the
    dedicated workstation.

Active retraction/tombstone enforcement and non-episodic canonical append are
deferred to the immediately following leaf. This leaf preserves their typed
links but does not simulate deletion or truth revision inside a projection.

## Result — 2026-08-31

Accepted at component scope on the dedicated Ubuntu workstation. The final
focused suite passed 86/86 in 2.317 seconds with CUDA hidden, and the public
memory/runtime import surface passed. This includes canonical episode paging,
typed projection/rejoin, per-neighbor validation, visibility rejection,
Moving Origin sequencing and idempotent retry, explicit restart rebuild,
adapter enforcement of frozen Cognee tenant/dataset/NodeSet scope under fake
bindings, durable outbox recovery, and the existing
cognitive/durable-ability/situated-memory regressions.

The final hardening rejects backend responses larger than the requested bound
before resolving any content, requires exact booleans for Cognee authority
switches, and treats telemetry as disabled only for the exact value `1`.

Independent read-only review found no blocking discrepancy. Ordinary turns
validate the current typed-memory head in constant time; history is paged only
by explicit rebuild. Cognee returns candidates and references only: every
record exposed to cognition is reconstructed from canonical episode bytes.
The installed Cognee 1.5.3 structured graph compatibility seam was exercised
without storage, model, embedding, or service effects. Live Cognee
write/search/forget behavior and embedding isolation remain untested.

This result establishes a shared, recoverable identity and temporal path. It
does not establish improved reasoning, live-service readiness, active
non-episodic truth revision, autonomous action, or consciousness.

## Stop conditions

Stop for any need to trust Cognee-generated content, add a second canonical
store, mutate a consumed experiment, invoke a model/GPU/live service, weaken
visibility/provenance, scan unbounded history during an ordinary turn, or add
task-solution logic.

## Next

After component acceptance, add canonical non-episodic memory append with
proposal/validation/supersession semantics. Then implement Prospective Origin's
durable pre-action commitment and learned action-conditioned world model.
