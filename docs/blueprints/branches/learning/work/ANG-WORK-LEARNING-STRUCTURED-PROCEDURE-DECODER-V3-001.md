# ANG-WORK-LEARNING-STRUCTURED-PROCEDURE-DECODER-V3-001

Status: ready

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local synthetic learned decoder, no external effects

## Accountable outcome

Test whether the preserved V1 latent procedural core contains enough
generalizable information to drive an explicit learned action sequence, rather
than relying on frozen Qwen to infer an executable sequence from opaque soft
prefix tokens.

## Frozen mechanism

- parent core checkpoint SHA-256:
  `08CB21C1485CEE41565A79B28AFDABF1B14A3A379F0D8044CB06870A3D7A1DF2`;
- parent core and Qwen are immutable;
- decoder: candidate-conditioned recurrent pointer, hidden width `512`, eight
  attention heads, at most six task-local actions, four output steps plus a
  learned STOP choice;
- inputs: detached Qwen embeddings of each public candidate component and the
  preserved core's latent procedure slots;
- targets: only action sequences in public successful support traces;
- training: four fixed passes over all 256 training supports (`1,024` decoder
  updates), no hidden-query labels, no task/mechanism identity, no answer table,
  no solver, no post-result tuning;
- final: same 16 untouched final mechanisms and 32 composed queries as V1/V2.

## Literal execution outputs

- `/opt/angler/results/structured-procedure-decoder-v3.json`;
- `/opt/angler/results/structured-procedure-decoder-v3.pt`.

Both paths must be absent before execution. The JSON records the accepted first
result; the checkpoint contains only the trained decoder and its parent binding.
Neither parent artifact may be overwritten.

## Controls and gate

Record direct hidden execution for full live state, complete state reset,
Moving Origin coordinates removed, unrelated evidence, and latent procedure
slots zeroed. The decoder must reach at least `0.60` full accuracy and exceed
each causal removal by at least `0.10`. Development exact visible-sequence
accuracy must reach at least `0.75`; target-evidence attribution must remain at
least `0.75`. Compare transparently with preserved V2 Qwen-alone/fair-retrieval
accuracy (`0.0`) but do not treat the different publication path as an
equal-interface causal control.

The deterministic runner may pad candidates, translate local labels to
indices, commit the decoder's unmodified greedy sequence, and call the hidden
judge once. It may not repair, search, reorder, or score candidate procedures.

Preflight requires the focused unit suite for the knowledge encoder, decoder,
V1 corpus serialization, and V3 runner contract to pass. A failed run is
preserved and this experimental identity is not silently tuned or rerun.

## Interpretation

A pass establishes a usable learned planning interface inside this synthetic
world and justifies passing Angler's explicit plan to Qwen for conversational
publication. It does not establish unrestricted software reasoning,
cross-domain transfer, conversation readiness, AGI, or deployment readiness.
