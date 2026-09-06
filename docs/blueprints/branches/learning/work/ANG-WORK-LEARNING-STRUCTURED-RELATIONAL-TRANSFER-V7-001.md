# ANG-WORK-LEARNING-STRUCTURED-RELATIONAL-TRANSFER-V7-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic procedural learning, no external effects

## Outcome

Test whether a learned typed relational representation restores the structural
distinctions erased by whole-row pooled Qwen embeddings and enables V6's
action-level procedural memory to transfer to held-out mechanisms.

## Frozen scope

- Parent V6 checkpoint SHA-256
  `08A966424CA678E9742A8D80B985BA48FC0C0FF8E3DF4BC86C0262B9C53FAD7A`.
- Qwen cache SHA-256
  `5031D348376622732FCB99A2AF44E9C4E4ED2E734D89758C80B92469B9028CE9`.
- Same corpus partitions, eight passes, thresholds, and first-result rule.
- The V6 core and decoder remain frozen. Only a fresh recurrent relational
  encoder and zero-initialized semantic-fusion residual train.
- Deterministic plumbing may parse public origin, goal, input/output,
  reads/writes, and candidate-local graph edges into typed incidence. It may
  not search, solve, repair, rank, use evaluator identities, or inspect hidden
  mechanism fields.
- The relation schema is informed by Cognee's typed graph/provenance boundary;
  Cognee remains replaceable retrieval and never becomes procedural
  competence. Neural processing reuses Angler's existing shared recurrent
  reasoning core.

## Causal controls

Full execution is compared with persistent reset, Moving Origin coordinates
removed, unrelated evidence, procedure slots removed, correspondence removed,
the structured residual removed from both support writes and queries,
topology edges removed while node/count plumbing remains, and foundation
action semantics removed. State-local and persistent-only arms remain visible.

## Gate

Full final execution at least `0.60`, development at least `0.60`, target
evidence attribution at least `0.75`, and full at least `0.10` above reset,
coordinates removed, unrelated evidence, procedure slots removed, and the
structured-removed arm. Accept the first valid result without tuning.

## Outputs

- `/opt/angler/results/compositional-procedure-v7-relational.json`
- `/opt/angler/results/compositional-procedure-v7-relational.pt`

## Final preflight

The focused relational/V6/recurrent-core suite passes `26/26`. Source
identities are:

- relational implementation
  `224B1704BDEC1BA235B0CD832BBEA335A98F5CEB10E58BA079D00F7A164BD131`;
- runner
  `C5312C4FA6391110648E02866D1713DFBB637206873D5DF32CAED79270E9BF31`;
- relational tests
  `103FA850D66308858DF4E7FA8F156AF9807AD1EF0604F915C9B7B1D92D2A17D4`;
- runner tests
  `CE1DF312EB36D1CDD224209BD6DF2F41111A6AD1EDF8C21647A47297B349B5E6`.

A zero-output pass loaded the exact parent/cache on `cuda:1`, wrote one
support trace through plastic memory, and backpropagated a composed query.
The zero-initialized fusion opened on update one; update two produced finite,
nonzero gradients in all `54` relational-encoder gradient tensors. The frozen
V6 core and decoder received no gradients. Peak allocation was `413,070,848`
bytes and both result paths remained absent.

## Frozen first result

The first valid V7 run completed in `126.499` seconds on the RTX 5070 and is
preserved `NOT_SUPPORTED` without tuning:

- result SHA-256
  `147833EE8C5861B3757207F501C2006AC66865B27FEA07F066AF547DF7C3A8C1`;
- checkpoint SHA-256
  `7281B0FC907704484C2C738242FF0BD7704B3AC7C845B52AAF301EB76A515FA9`;
- peak CUDA allocation `643,381,760` bytes;
- trainable parameters `2,082,249`;
- terminal support exactness `0.984375` and train composition `0.7109375`;
- development and every final/control execution arm `0.0`;
- target-evidence attribution approximately `0.999998`.

The structured path did not cause transfer: full versus structured-removed
differed on only `1/32` final sequences, full versus topology-removed differed
on `0/32`, and action-semantics removal changed `32/32`. A read-only checkpoint
diagnostic found candidate relational representations had mean pairwise cosine
approximately `0.99999996` on train, development, and final partitions. The
learned relational residual was only about `0.0574` of the semantic norm.

This falsifies the isolated per-candidate graph-summary implementation, not the
usefulness of public graph structure. The next product-facing successor must
reuse the already-supported V19/V20 paired whole-graph representation and its
preserved checkpoints through a narrow production bridge; it must not add
another isolated summary, a deterministic solver, or post-result V7 tuning.
