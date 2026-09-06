# ANG-WORK-LEARNING-COMPOSITIONAL-PROCEDURE-V4-001

Status: ready

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local synthetic procedural training, no external effects

## Accountable outcome

Test whether direct end-to-end sequence credit plus an actually demonstrated
composition curriculum makes Angler's latent core causally useful for explicit
multi-step procedure construction.

## Frozen mechanism and curriculum

- Start from the immutable 47.8M-parameter V1 core; Qwen remains frozen and is
  used only for detached semantic embeddings.
- Train the core, plastic write path, and candidate-conditioned decoder jointly.
- Each of 64 training mechanisms supplies four public two-action supports and
  two varied public four-action compositions. The evaluator may materialize
  correct traces only for the `train` partition; development and final targets
  remain hidden and are used only by the existing judge.
- Run eight fixed passes. One optimizer update consumes one complete mechanism
  episode so credit crosses its support sequence and plastic writes.
- Core learning rate is `0.00003`; decoder learning rate is `0.0003`; retrieval
  preservation weight is `0.1`; gradient norm ceiling is `1.0`.
- No task/mechanism identity, answer table, search, repair, final label, or
  deterministic solver enters learner input or inference.
- Accept the first result without post-result tuning.

## Outputs

- `/opt/angler/results/compositional-procedure-v4.json`;
- `/opt/angler/results/compositional-procedure-v4.pt`;
- `/opt/angler/results/compositional-procedure-v4-embeddings.pt`.

All three paths must be absent before execution. The embedding cache is a
content-bound runtime artifact retained to avoid repeatedly spending nineteen
minutes on identical frozen-Qwen encoding in downstream integration work.

## Gate

On the same untouched 32 final queries, full hidden execution must be at least
`0.60` and exceed state reset, coordinates removed, unrelated evidence, and
procedure slots removed by at least `0.10` each. Development composed execution
must be at least `0.60`, and target-evidence attribution at least `0.75`.

A pass supports this bounded compositional planning interface. It does not by
itself establish general conversation readiness; that requires the subsequent
Qwen + Cognee + Moving Origin multi-turn integration and removal test.
