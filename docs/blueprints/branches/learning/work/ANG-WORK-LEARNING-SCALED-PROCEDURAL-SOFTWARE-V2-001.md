# ANG-WORK-LEARNING-SCALED-PROCEDURAL-SOFTWARE-V2-001

Status: complete — `NOT_SUPPORTED`

Active node: `ANG-BP-LEARNING`

Assurance: LOW; evaluation-only synthetic successor

## Accountable outcome

Complete the hidden execution evaluation of the consumed V1 core after
correcting the demonstrated publication-boundary defect. V1 result SHA-256
must equal `AE0523155E5B8C1428887767FD0EBBE8ACB2B0ACBDD3A3456A8469A531EF48E7`
and checkpoint SHA-256 must equal
`08CB21C1485CEE41565A79B28AFDABF1B14A3A379F0D8044CB06870A3D7A1DF2`.

## Frozen scope

There is no optimizer, training, update, threshold change, task change,
checkpoint mutation, parser change, arm change, or generation-budget change.
The same 16 final mechanisms, 32 composed queries, seven causal arms,
16-token ceiling, local labels, hidden judge, and V1 thresholds apply. The
only semantic change is that every publication prompt ends with a literal
`answer=` line after all task/evidence text. This prevents the causal model
from continuing the component or experience list instead of opening the
declared response field.

## Acceptance

The inherited V1 gate remains exact: full accuracy at least `0.60`; at least
`0.15` above Qwen alone and fair retrieval; at least `0.10` above
prefix-removed, reset-state, and coordinates-removed; evidence attribution at
least `0.75`; and the already recorded V1 loss reduction at least `0.20`.
Accept the first valid result without further interface edits.

## Result

The evaluation-only successor completed in 162.63 seconds. Result SHA-256:
`902A7EB87275EB7B125484F7393BFF1FB891AC6D56EE20F1DFB7011B8A3C6F4D`.
The literal answer boundary stopped list continuation, but all hidden execution
arms remained at zero. Full Angler sometimes emitted plausible local-label
sequences, while the dominant response was `1`; prefix-removed/fair retrieval
usually emitted `?`. No causal component claim passed. This consumes the
publication-only hypothesis. The next successor must add a learned structured
procedure decoder over Angler's latent slots and public candidate embeddings,
not another prompt edit or deterministic repair.
