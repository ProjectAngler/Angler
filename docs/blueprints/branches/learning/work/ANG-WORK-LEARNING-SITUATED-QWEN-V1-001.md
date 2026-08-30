# ANG-WORK-LEARNING-SITUATED-QWEN-V1-001

Status: complete — `NOT_SUPPORTED`

Assurance: bounded local synthetic GPU experiment; no promotion authority

## Question

Does the supported Angler + Moving Origin + Cognee mechanism improve the
observable output of the same frozen Qwen3-4B model when Angler-selected
experience is supplied as working context?

## Frozen comparison

The eight accepted live Cognee family recalls and sealed situated-reader V2
checkpoint are reused without training. The same frozen Qwen model receives:

1. the query alone;
2. all six semantically retrieved experiences, explicitly unordered and
   without Moving Origin coordinates (fair RAG);
3. exactly one experience chosen by Angler's learned attention over content
   plus live Moving Origin coordinates.

Qwen generates the final answer in every arm. Angler does not emit the final
answer, and deterministic code only builds prompts and parses the declared
four-word response vocabulary. Foundation weights, reader weights, recalls,
candidate order, and thresholds are immutable.

## First-result gate

`QWEN_GENERATION_BENEFIT_SUPPORTED` requires Angler-attached accuracy at least
0.75 and at least 0.25 above both Qwen-alone and fair-RAG accuracy. The selected
evidence target must remain present in at least 90% of rows. The runner records
every prompt mode, raw generation, parsed answer, device, versions, input
hashes, peak CUDA allocation, and wall time. Accept the first result without
tuning.

## Interpretation boundary

A pass demonstrates that the learned situated-memory mechanism can improve
Qwen's generated response on fresh instances of the same eight synthetic
mechanisms. It does not demonstrate arbitrary conversation improvement,
cross-mechanism transfer, lifelong learning, graph-reasoning benefit, AGI,
consciousness, or production readiness.

## First result

The identity completed once without tuning. Result SHA-256:
`4A795447A37F9CF528094FBE0C4FBFDA266C3F423F8C6E881BD7E5B58002340F`.

- Angler-selected Qwen: `4/8 = 0.50`
- Qwen alone: `2/8 = 0.25`
- unordered fair RAG: `0/8 = 0.00`
- target-memory coverage: `8/8 = 1.00`

The frozen gate failed because attached accuracy was below `0.75`. The result
does contain a bounded positive signal: attached Qwen gained exactly `0.25`
over Qwen alone and `0.50` over fair RAG. The sealed selector chose the
target-bearing experience in seven rows; Qwen returned the correct action in
four, returned a wrong action in one, and exhausted the 12-token generation
ceiling before emitting an action in three. Two of those truncated rows had a
correct selected experience. This is a failed first result, not permission to
reinterpret or tune it.
