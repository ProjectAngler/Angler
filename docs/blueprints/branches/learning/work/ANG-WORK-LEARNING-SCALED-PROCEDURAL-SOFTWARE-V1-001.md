# ANG-WORK-LEARNING-SCALED-PROCEDURAL-SOFTWARE-V1-001

Status: complete — `NOT_SUPPORTED`

Active node: `ANG-BP-LEARNING`

Assurance: LOW; contained local synthetic software reconstruction, frozen
foundation model, no external effects or promotion authority

## Accountable outcome

Determine whether the resource-scalable Angler core can learn reusable latent
procedures from public software-pipeline experience and improve a frozen
Qwen3-4B on untouched multi-step reconstruction tasks while Cognee-compatible
evidence, Moving Origin coordinates, bounded plastic memory, and learned
procedure-prefix tokens remain causally separable.

## Fixed experiment identity

- identity: `angler.scaled-procedural-software-reconstruction.v1`
- seed: `20260844`
- foundation: local frozen `/opt/angler/models/Qwen3-4B`
- initial training tier: `workstation` (47,846,425 parameters)
- train/development/final mechanism partitions: the existing public synthetic
  software-pipeline suite, with no evaluator-private fields in learner inputs
- training scale: all 64 training mechanisms / 256 visible support
  procedures; 32 differentiable within-mechanism meta episodes; every fourth
  slow or meta update also carries frozen-Qwen language consequence loss
- evaluation scale: all 16 development and all 16 final mechanisms, two
  untouched composed queries per mechanism
- maximum public pipeline length: four actions
- first valid result is accepted without threshold, seed, prompt, optimizer,
  architecture, or partition tuning

## Experience and learning boundary

Training examples come only from public successful support traces. For each
example, one support task is held out as the target while the remaining public
support traces are evidence. Local presentation labels replace opaque action
digests, but their mapping is rebuilt from each task's declared candidate
order; no label has global semantic meaning. The learner predicts the held
trace's visible action sequence through frozen-Qwen teacher forcing and learns
evidence attribution from the same public trace. It never receives a hidden
query solution, evaluator state, mechanism index, partition answer, or a
hand-written solver.

At evaluation, the core receives only the public composed query and public
support evidence. Qwen emits one immutable local-label sequence before the
hidden evaluator returns a single terminal `0.0` or `1.0`. Query outcomes may
update only the bounded plastic state after judgment; they do not train on the
same query before its score is recorded.

## Causal arms

Every final task uses the same frozen Qwen, prompt contract, candidate order,
and generation budget. Record:

1. full Angler + evidence + live Moving Origin + plastic state + prefix;
2. frozen Qwen alone;
3. frozen Qwen with all unordered evidence (fair retrieval);
4. full evidence selection with the learned procedure prefix removed;
5. plastic state reset immediately before the query;
6. Moving Origin coordinates replaced by zeros;
7. evidence content replaced by a matched unrelated set.

The full arm must score at least `0.60`, exceed Qwen alone and fair retrieval
by at least `0.15`, and exceed each of prefix-removed, reset-state, and
coordinates-removed by at least `0.10`. Target-evidence attribution must be at
least `0.75`. Training loss must be finite and fall by at least `20%`; frozen
Qwen parameters and tokenizer identity must remain byte-stable. These gates
are frozen before the first semantic result. A miss is preserved as evidence,
not tuned under this identity.

## Resource and stop conditions

Start with the workstation tier because gradients through a frozen 4B model's
inputs retain language-model activations. Use microbatch one, bounded sequence
length, gradient accumulation, mixed precision, and activation checkpointing
only when they preserve the exact objective. Record CUDA peak, wall time,
library versions, update count, and all arm metrics. Stop on non-finite values,
GPU out-of-memory after the predeclared memory fallback, frozen-model mutation,
partition overlap, hidden-field exposure, malformed evaluator calls, or an
unexpected existing output/checkpoint.

## Exact implementation scope

- `src/angler/reasoning/scalable_procedural_core.py`
- `src/angler/runtime/procedural_qwen.py`
- their existing exports and focused unit tests
- one fresh runner and focused runner test under `experiments/runners/` and
  `tests/unit/experiments/`
- this leaf plus one result report after the terminal artifact exists

No existing checkpoint/result may be overwritten. Deterministic code may
serialize public tasks, assign local labels, enforce masks, invoke the frozen
model, commit a proposed pipeline, run the hidden judge once, and measure the
result. It may not choose or repair the procedure.

## Interpretation

A pass would establish useful transfer across untouched mechanisms inside the
synthetic software-pipeline world and causal benefit from the integrated
components. It would not establish arbitrary software engineering,
conversation readiness, unrestricted cross-domain reasoning, consciousness,
AGI, production safety, or deployment readiness.

## First result

The frozen first run completed in 802.75 seconds. Result SHA-256:
`AE0523155E5B8C1428887767FD0EBBE8ACB2B0ACBDD3A3456A8469A531EF48E7`;
checkpoint SHA-256:
`08CB21C1485CEE41565A79B28AFDABF1B14A3A379F0D8044CB06870A3D7A1DF2`.

Training loss fell `75.32%`, development latent loss was `0.04866`,
development relevance mass was `0.99860`, and final target-evidence
attribution was `0.99981`. Nevertheless, every hidden execution arm scored
zero. The generated text localizes the immediate failure: evidence-bearing
prompts ended with `experience 4`, so Qwen continued with `experience 5`; the
query-only prompt ended with component `F`, so Qwen continued with a
nonexistent component `G`. No arm opened an explicit answer field. This V1 is
consumed and will not be rerun or tuned. A fresh evaluation-only successor may
reuse the checkpoint, arms, queries, and thresholds while changing only the
publication boundary to append a literal `answer=` cue.
