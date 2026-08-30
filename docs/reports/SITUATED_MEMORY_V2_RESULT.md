# Situated Memory V2 Result

Date: 2026-08-30

Classification: `SUPPORTED_FOR_PROTOTYPE`

## What was tested

This experiment composed three replaceable roles without a deterministic
answer solver:

1. Cognee 1.5.3 persisted and semantically retrieved disposable evidence
   projections using only local services.
2. Moving Origin maintained monotonic autobiographical positions, live age,
   and a `current_regime` landmark outside Cognee.
3. A GPU-resident learned Angler reader combined frozen Qwen3-4B content
   representations with those temporal coordinates to select one of four
   procedures.

The query did not contain the answer, candidate order varied, targets varied,
and the reader received no family ID, recency rule, procedure table, or answer
lookup. V1's failed shortcut result was preserved; V2 corrected the evidence
confounds and accepted its first result without tuning.

## Causal synthetic result

| Arm | Accuracy |
|---|---:|
| Full Angler + live Moving Origin | 0.9951171875 |
| Held semantic families | 0.9920477137 |
| Frozen origin | 0.4731445312 |
| Coordinates removed | 0.4736328125 |
| Content removed | 0.2470703125 |
| Retrieved target replaced | 0.0009765625 |
| Learned competence reset | 0.2646484375 |

The fair-naive temporal reconstruction matched full accuracy but inspected 42
records per query versus 6 for the maintained Moving Origin index.

## Live self-hosted result

Cognee recalled all six valid projections for each of eight family queries;
the designated target was present in 8/8. Without training or changing the
sealed checkpoint, the reader selected the correct procedure in 7/8 queries
(0.875) and assigned mean attention 0.8751317176 to the target evidence.

The Cognee build/retrieve/forget stage took 395.08 seconds. Much of that time
was optional knowledge-graph extraction and structured-summary repair; the
actual Angler Qwen-embedding and frozen-reader evaluation took 3.66 seconds on
an NVIDIA GeForce RTX 5080. Peak CUDA allocation for that evaluation was
8,831,196,672 bytes. The learned reader itself has 1,917,979 parameters.

## Evidence identities

- V1 failed result: `4df3c604bea1116e8398312435c2b52481942894d75e55d8cdafa551e4dbd71d`
- V2 result: `2d726321c3d8136cecd8bebba5cfa611d18d565587fa0cc0f93614c89124207f`
- V2 reader checkpoint: `b94e27ad0a42e3f499b7de5ff1b5217463dadd38e60934a7af7f3ebd2da4e3d4`
- Live Cognee recall: `c59083344f1accd8d44bd002922138d26628f4e42895c806d61f360185e8dd45`
- Live sealed-reader evaluation: `22fc22b839a1de2629192da5d7272a7020ac1e2f97f30d63a0080ba476054b43`

## Interpretation

This is evidence that the components harmonize: Cognee finds relevant
experience, Moving Origin supplies efficient current context, and Angler has
learned a reusable procedure for combining content and temporal state. It is
not evidence that Cognee's graph enrichment improves the answer, nor proof of
arbitrary cross-domain reasoning, lifelong retention, AGI, consciousness, or
production readiness. The one live error and expensive graph-enrichment path
are the clearest next engineering targets.
