# ANG-WORK-RUNTIME-HYBRID-RECALL-CONVERSATION-V3-001

Status: `CONVERSATION_PROTOTYPE_READY`; interactive launcher verified for owner-controlled terminal use

Active node: `ANG-BP-RUNTIME`

## Outcome

Repair the sole V2 readiness miss with Cognee's existing lexical retrieval,
without changing the learned reader, Moving Origin, Qwen, tasks, or thresholds.

## Exact mechanism

For each query, request Cognee `CHUNKS` top-six and `CHUNKS_LEXICAL` top-six
from the same dataset. Preserve vector order, append unseen lexical records,
deduplicate by the exact encoded projection, and expose at most twelve records
to the sealed reader. Fair-RAG receives that identical union. No target,
procedure label, evaluator field, answer, or family-specific rule participates
in retrieval or fusion.

The diagnosis is frozen: V2 result SHA-256
`EBC71D84468F52E415942C53899C246F0325B1C8CBB1C4285FB8FECA17816E42`
missed instrument-calibration target coverage in vector top-six; Cognee lexical
retrieval independently returned that target at rank five. V2 is not rerun.

## Scope and test

Write scope is this leaf, the existing situated-conversation runtime module and
exports, its focused test, the existing V2 runner used as the V3 successor
harness, append-only `AGENTS_SYNC.md`, the fresh workstation state root
`/opt/angler/state/situated-conversation-v3/evaluation-first`, and result
`/opt/angler/results/situated-conversation-v3-evaluation.json`.

Tests must prove deterministic union order, exact deduplication, result cap,
actual Cognee vector and lexical calls, existing journal/history/rebuild
behavior, both frozen digests, and all V2 removal controls. The V2 readiness
thresholds remain exact. The first valid V3 result is accepted without tuning.

This remains a local, synthetic, no-tools, no-service, human-feedback-only
prototype under the V2 LOW assessment and nonclaims. No training, autonomous
grading, graph-LLM enrichment, network deployment, external effect, promotion,
or remote push is authorized.

## First result

The first valid V3 result is preserved at SHA-256
`0F149262462293A686515E04168A520F59A53B691C266BD82EA13B900E25D493`.
It passed the frozen classification with full Angler/Qwen `0.875`, Qwen-alone
`0.125`, fair-RAG `0.375`, target coverage `1.0`, full selector `0.875`,
Moving-Origin removed `0.5`, reader reset `0.0`, and mismatched retrieval
`0.0`. Feedback recall, two-turn history, Qwen digest, and reader digest all
passed. Runtime was 25.324 seconds; Qwen used the RTX 5080 and the reader used
the RTX 5070. This supports only the bounded conversation prototype claim.

The final product smoke verified the exact readiness artifact, loaded both
frozen models, inspected an empty named workspace, visibly took Qwen-alone
fallback without querying uninitialized Cognee, generated `Hello!`, and exited
cleanly. Focused tests remain 9/9. The owner may now use the local terminal;
only explicit `/success` or `/failure` commands create feedback memory.
