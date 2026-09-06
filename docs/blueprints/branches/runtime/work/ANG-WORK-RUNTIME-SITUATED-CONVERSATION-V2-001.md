# ANG-WORK-RUNTIME-SITUATED-CONVERSATION-V2-001

Status: first valid live result preserved `NOT_READY`; successor required

Active node: `ANG-BP-RUNTIME` / local situated-conversation prototype

Assurance: LOW for local synthetic evaluation and owner-controlled terminal use; no deployment or promotion authority

## Accountable outcome

Turn the already-supported situated-memory and Qwen interface into one real,
multi-turn terminal system. The system must load frozen local Qwen3-4B and the
sealed V2 situated reader, use the actual self-hosted Cognee adapter for recall,
reconstruct trusted Moving Origin coordinates from a local canonical journal,
include bounded conversation history, and persist only feedback explicitly
supplied by the human operator.

## Frozen evidence and inputs

- situated reader checkpoint SHA-256
  `B94E27AD0A42E3F499B7DE5FF1B5217463DADD38E60934A7AF7F3EBD2DA4E3D4`;
- situated-memory V2 result SHA-256
  `2D726321C3D8136CECD8BEBBA5CFA611D18D565587FA0CC0F93614C89124207F`;
- live Cognee recall and evaluation SHA-256 values
  `C59083344F1ACC...5E8DD45` and `22FC22B839A1DE...54B43`;
- situated-Qwen V2 result SHA-256
  `A3E9E04C7A709C913E05682055C0317A2AD8C28EAAE30D370433F4BA686A1A42`;
- local model `/opt/angler/models/Qwen3-4B`, existing Angler and Cognee
  environments, and the existing `SituatedMemory`, `CogneeProjectionBackend`,
  `MovingOriginIndex`, `LearnedSituatedMemoryReader`, and `LocalQwenIO`
  boundaries.

The checkpoint, foundation model, old results, existing memory/reasoning code,
and V8–V10 artifacts remain immutable.

## Deliverables and literal write scope

- `src/angler/runtime/situated_conversation.py`;
- `src/angler/runtime/__init__.py` (exports only);
- `experiments/runners/situated_conversation_v2.py`;
- `tests/unit/runtime/test_situated_conversation.py`;
- this leaf and append-only `AGENTS_SYNC.md` entries;
- first remote result `/opt/angler/results/situated-conversation-v2-evaluation.json`;
- local workstation state only under `/opt/angler/state/situated-conversation-v2`.

No other source, blueprint, checkpoint, result, model, or remote repository is
in write scope.

## Required behavior

1. A canonical local JSONL journal is the persistent source of projection and
   feedback records. Cognee remains a disposable/rebuildable retrieval index.
2. Moving Origin is reconstructed deterministically from the journal, and its
   live coordinates—not stored stale coordinates—enter the sealed reader.
3. The current user query drives retrieval. Bounded prior turns enter only the
   visible Qwen prompt; no hidden task or answer identity routes memory.
4. Qwen and the reader remain frozen. Angler selects evidence and publishes
   attribution; Qwen alone generates the response.
5. Success/failure and the procedure note come only from the human operator.
   A feedback record is available to a later recall without updating neural
   weights. The system never grades itself.
6. The operator can inspect attribution, forget the disposable Cognee dataset,
   rebuild it from the journal, and clear only a named local conversation
   workspace.
7. Empty recall falls back visibly to Qwen-alone rather than fabricating
   evidence. Malformed journals, hash mismatches, non-local model requests,
   empty generation, and projection inconsistencies fail closed.
8. Conversation records use Cognee's supported chunk-only custom pipeline:
   classification, chunking, local embedding, vector storage, scoped search,
   and deletion remain active, while LLM graph extraction/summarization is
   omitted. The canonical record already carries Angler provenance, so a
   second model may not reinterpret it on the latency-critical write path.

## First live evaluation and frozen interpretation

Run one first-result evaluation over the existing eight public synthetic
families, using one global Cognee dataset and the same frozen Qwen responses.
Record:

- full Angler + Cognee + Moving Origin + Qwen accuracy;
- Qwen-alone and unordered fair-RAG accuracy;
- Cognee target coverage;
- coordinate-frozen/Moving-Origin-removed and reader-reset selections;
- a retrieval-removal control using a different query's recalled candidates;
- an explicit feedback record followed by a new query that recalls that exact
  record;
- a two-turn history prompt check;
- Qwen and reader pre/post digests, device placement, versions, latency, and
  peak memory.

The first valid result is accepted without tuning. `CONVERSATION_PROTOTYPE_READY`
requires all of: actual Cognee ingestion/recall succeeds; full Qwen accuracy is
at least `0.75`; full exceeds both Qwen-alone and fair-RAG by at least `0.25`;
target coverage is at least `0.90`; at least one of the Moving-Origin,
retrieval, or reader-reset removals reduces selector accuracy by at least
`0.20`; the external-feedback record is recalled on the next relevant turn;
two-turn history is visibly present; all focused tests pass; and frozen digests
remain exact. Failure of a causal arm prevents a readiness claim even if the
terminal interface launches.

A pass establishes only a useful bounded eight-family conversation prototype
and a real persistent feedback/memory loop. It does not establish arbitrary
conversation improvement, online neural learning, cross-domain reasoning,
AGI, consciousness, a normal RUNTIME/LEARNING gate, a slice, milestone,
promotion, or deployment readiness.

The preserved R2 attempt reached actual eight-family Cognee ingestion/recall,
Qwen generation, and canonical feedback append, then Cognee's full `remember`
pipeline generated more than 67,000 helper-model tokens without completing
its structured output and hit the exact 600-second timeout. No result was
created. That failure consumes the synchronous graph-enrichment strategy, not
the frozen evaluation. The successor attempt changes only Cognee projection
to its own `get_just_chunks_tasks` pipeline and records that mode in output;
all inputs, models, thresholds, controls, and interpretation remain frozen.

The first valid chunk-projection result is preserved at SHA-256
`EBC71D84468F52E415942C53899C246F0325B1C8CBB1C4285FB8FECA17816E42`.
It scored full Angler/Qwen `0.75`, Qwen-alone `0.125`, fair-RAG `0.375`,
selector full `0.75`, Moving-Origin removed `0.375`, reader reset `0.0`, and
retrieval removed `0.0`. Explicit feedback was recalled on the next turn,
two-turn history was visible, and Qwen/reader digests remained exact. The
classification is nevertheless `NOT_READY`: Cognee vector target coverage was
`0.875`, below the frozen `0.90` requirement. Instrument calibration was the
single coverage miss; the same preserved dataset's independent lexical search
returned that target at rank five. This result is final for V2 and will not be
tuned or rerun.

## Proportionate human-impact assessment

The implementation is a local terminal research prototype with no listening
network service, tool execution, autonomous action, external messaging,
foundation mutation, or self-authored feedback. Evaluation data is synthetic.
Interactive messages and feedback are owner-controlled local data and may be
deleted by clearing the named workspace; telemetry must be disabled. Main
risks are false capability claims, unintended retention, stale/tampered
memory, and accidental response-as-feedback. Mitigations are explicit
nonclaims, visible evidence attribution, human-only outcomes, canonical
journal validation, disposable Cognee storage, named deletion, local-only
paths, and frozen identity checks. Impact class: `LOW`; disposition for this
exact scope: `ALLOW`. Any network service, third-party model call, autonomous
tool use, automatic grading, real-person deployment, or promoted-state change
requires a successor assessment. This maps to
`ANG-GATE-HUMAN-FLOURISHING-001` and does not pass or waive it.

## Tests, failure, and rollback

Focused CPU tests must cover journal round-trip/tamper rejection, Moving Origin
reconstruction, bounded multi-turn prompt visibility, explicit feedback
persistence, Qwen-alone fallback, inspect/forget/rebuild, and unchanged frozen
parameters with injected doubles. The workstation run must exercise actual
Cognee and both GPUs under existing placement constraints.

Stop on identity mismatch, unexpected output occupation, non-local model
access, telemetry, non-finite values, empty generation, CUDA failure, Cognee
projection failure, malformed journal, or frozen-parameter mutation. Preserve
the first result or failure record. Source rollback removes only the fresh
module, runner, test, export lines, and this leaf; state rollback removes only
the named V2 workspace after exact-path verification. Never delete canonical
evidence or another Cognee dataset.
