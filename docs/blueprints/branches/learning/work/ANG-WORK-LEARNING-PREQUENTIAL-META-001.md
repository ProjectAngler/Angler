---
blueprint_id: ANG-WORK-LEARNING-PREQUENTIAL-META-001
parent_id: ANG-BP-META-UPDATER
revision: 1
tier: 4
design_status: approved
delivery_status: complete_failed
human_authority: project owner direction, 2026-08-25
human_impact: LOW; contained synthetic learning experiment
depends_on:
  - ANG-WORK-LEARNING-ADAPTIVE-CORE-001
  - ANG-WORK-WORLDS-LATENT-PROGRAM-STREAM-001
---

# Prequential changing-mechanism meta-learning run

## Objective

Meta-train the slow SRWM dynamics on unique one-pass experiences, then freeze
all slow weights and measure whether the carried numeric state improves from
scalar outcomes while ordering mechanisms change.  A static heldout score is
not evidence of this capability.  Primary evidence is within-Angler
prequential gain, adaptation speed, causal state dependence, and delayed
reacquisition savings as mechanisms change.  Structurally new evaluator
programs serve only as an anti-memorization check.

## Exact outputs

- `src/angler/reasoning/recurrent_core.py` and its unit test: generic
  evaluator-only scoring of a prescribed training order.
- `src/angler/reasoning/adaptive_core.py` and its unit test: score a fresh
  query through the same state-conditioned policy without writing state.
- `experiments/runners/phase2_prequential_srwm.py`: structured detached-input
  connector, meta-training stream, frozen-slow evaluation, controls, and JSON
  receipt.
- `experiments/evaluators/latent_order_suite.py`: evaluator-only mechanisms,
  imported after the candidate slow-state fingerprint is frozen.
- `tests/unit/experiments/test_phase2_prequential_srwm.py`: matched-RNG,
  no-write, flag-balance, and causal-status regression checks.
- this work leaf.

## Learning boundary

During meta-training only, visible training-world targets define the outer
loss that teaches the update dynamics.  Online support encounters still expose
only the attempted action and scalar pairwise reward.  Validation/test program
targets never update slow weights.  Final evaluation freezes the recurrent
core, SRWM bases, encoders, and decoder; only the fixed-size SRWM offsets change.

No problem/family/program/split ID, AST, target, violated pair, answer trace,
replay item, per-family state, router, or deterministic solver may enter the
learner.  Every public instance hash is unique and each support is presented
once within a learning branch.  Matched experimental control branches may use
the same instance from the exact same incoming state and random stream; this
counterfactual reuse is counted explicitly and is not training replay.  The
structured connector serializes public attributes but does not compute an
ordering; Qwen is neither loaded nor trained in this isolating run.

## Acceptance and claim limit

The run records every unique attempted solution before its scalar feedback
write, forming the primary online trajectory as mechanisms change.  Each
mechanism starts matched ordinary, no-write, and inverted-feedback branches
from the exact incoming state with a common Torch random stream.  Slow-weight
fingerprints must remain unchanged during evaluation, and later new instances
measure reacquisition from historical versus cold state.  Both public-flag
branches are balanced and reported separately; the worst branch determines
directional success.  A positive result requires improvement on multiple
changing mechanisms, advantage over causal controls, and reacquisition savings.
Failure is retained as design evidence, not patched with a solver.

Even a pass proves only online acquisition over new compositions built from
known primitives within one ordering action algebra.  It does not prove
universal adaptation to arbitrary unrelated problem types; later vertical
slices must widen mechanisms, observations, actions, tools, and domains.

## Result

The smoke and full runs both correctly rejected the candidate.  In the full
256-step run, outer loss fell from `0.958614` to `0.565344`, but ordinary,
no-write, and inverted-feedback branches remained behaviorally identical.
Worst-flag online gain was `-0.03125`; causal advantage and reacquisition
savings were both `0.0`.  The result was
`NO_CAUSAL_ADAPTIVE_EFFECT_OBSERVED`: the slow network optimized the training
objective while bypassing the changing SRWM state.

Preserved evidence:

- `work/experiments/prequential-srwm-full-6201.json` SHA-256
  `254A7E228FFEB30A45D0412AFCCC5461B1AF7BC873FA67813D9F85BEE4D48DF1`
- `work/experiments/prequential-srwm-full-6201.safetensors` SHA-256
  `F11795769558813810979146838FF6A2B90E12E490437E1437F2AE600ACFA534`
- corresponding smoke artifacts remain beside them.

This leaf is closed as negative evidence.  A successor must introduce a
qualitatively different procedural-learning mechanism rather than increasing
training or adding a task-specific solver.

## Effects and rollback

Local synthetic GPU training/inference inside Angler WSL2 only.  No network,
package, service, external effect, personal data, deployment, or Qwen/base-model
mutation.  Candidate state is experimental and reversible.  Code rollback is
the exact parent commit after the latent-world leaf.
