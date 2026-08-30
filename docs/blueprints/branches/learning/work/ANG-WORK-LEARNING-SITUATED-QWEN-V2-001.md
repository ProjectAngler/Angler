# ANG-WORK-LEARNING-SITUATED-QWEN-V2-001

Status: complete — `QWEN_INTERFACE_BENEFIT_SUPPORTED`

Assurance: bounded local synthetic GPU integration test; no promotion authority

## Purpose

V1 preserved a positive but failed result because Qwen frequently started an
explanation and reached the 12-token ceiling before publishing the requested
action. V2 tests only the Qwen I/O contract: require the action as the first
word and allow 64 generated tokens. The model, reader, recalls, candidates,
prompt evidence, arm ordering, parser vocabulary, and scientific thresholds
remain unchanged.

This is an observed-row engineering regression, not a fresh learner-
generalization experiment. It may establish that the runtime can transmit the
already-demonstrated selector signal through Qwen, but cannot add independent
evidence that the selector generalizes.

## Frozen comparison and gate

The same frozen Qwen3-4B runs query-alone, unordered fair-RAG, and one
Angler-selected experience arms. Qwen generates every final answer. A pass
requires attached accuracy at least `0.75`, at least `0.25` above each control,
and target-memory coverage at least `0.90`. Accept the first V2 result without
tuning. Preserve V1 result SHA-256
`4A795447A37F9CF528094FBE0C4FBFDA266C3F423F8C6E881BD7E5B58002340F`.

## Constraints

No weight update, action lookup, answer emission by Angler, changed retrieval,
changed threshold, or result-dependent retry is allowed. Deterministic code may
only assemble the explicit prompt, invoke frozen generation, parse the four-
word response vocabulary, and record evidence.

## First result

The identity completed once without tuning. Result SHA-256:
`A3E9E04C7A709C913E05682055C0317A2AD8C28EAAE30D370433F4BA686A1A42`.

- Angler-selected Qwen: `7/8 = 0.875`
- Qwen alone: `1/8 = 0.125`
- unordered fair RAG: `3/8 = 0.375`
- target-memory coverage: `8/8 = 1.00`

Every frozen threshold passed. The sole attached error was the already known
reader-selection error. This establishes a working Qwen publication boundary
for the bounded eight-family prototype, not independent learner generalization
or general-purpose Qwen improvement because V2 deliberately reused V1's
observed rows to isolate the interface defect.
