---
blueprint_id: ANG-WORK-LEARNING-SKILL-LOCAL-MEMORY-001
parent_id: ANG-BP-META-UPDATER
revision: 1
tier: 4
design_status: approved
delivery_status: in_progress
human_authority: project owner direction, 2026-08-27
human_impact: LOW; contained local synthetic procedural learning
depends_on:
  - ANG-WORK-LEARNING-CAUSAL-OPERATOR-COMPILER-001@2
substrate_commit: 52a8345f90a06a453d17522335b4f77840dec511
substrate_checkpoint_sha256: CBB470051BC56D84A128A163AD646F8FE83AB3D15335A8A9E5A53522036A2407
---

# Skill-local procedural memory

## Objective

Add one fixed-capacity, internally routed procedural-evidence memory around the
stable causal operator compiler.  Meta-training teaches a public-evidence
encoder and nonlinear conditional decoder; during online evaluation all slow
weights and the compiler remain frozen and only the selected summary slot and
its count may change.  This tests persistent ability acquisition without
replaying every old problem.

This is neural attention inside one model state, not selection among models or
per-task adapters.  The public learner receives structural state, goal,
operator/candidate features, its attempted action, and bounded external
outcome feedback.  It receives no namespace, domain, task, family, episode,
adapter, split, hidden-mechanism, answer, solution-route, or evaluator field.

## Exact increment

- `src/angler/procedures/skill_memory.py`: fixed-size state, learned structural
  routing, exchangeable running-mean evidence summaries, nonlinear conditional
  utility inference, exact snapshot/restore, and reversible transactions;
- `src/angler/procedures/learning.py` and `src/angler/procedures/__init__.py`:
  optional memory residual in the shared candidate-scoring path, with the
  no-memory path bit-identical;
- `experiments/evaluators/skill_memory_suite.py`: evaluator-only changing
  mechanisms and outcome scoring;
- `experiments/runners/phase5_skill_memory_stream.py`: disjoint meta-training
  and frozen-slow online A -> B -> C -> A -> D -> B stream;
- focused unit/runner tests and one compact tracked result manifest.

The initial state must contribute exactly zero so slow parameters cannot solve
the experiment while bypassing memory.  Each valid public observation,
including a score equal to the current prediction, is independently encoded
from public state, attempted candidate, and scalar outcome; the selected slot
stores their running mean and count.  The accumulator is order-invariant and
has no recurrent overwrite path.  Opaque procedure symbols address slots but
never enter their learned content.  Nonselected slots remain bit-identical.
Read-time top-k mixtures may compose retained skills.  Capacity is constant
with stream length and no online replay/history retrieval is permitted.  Slot
merging or trunk distillation is deferred until routed persistence itself
works.

## Predeclared evidence

Use unique varied instances throughout, at least 64 online feedback encounters
per mechanism, at least 40 disjoint final cases per mechanism, and novel
two-skill compositions.  Record state and slow-weight hashes, changed-slot
masks, route weights, accepted/rejected local transactions, experience counts,
all partition identities, proposal-time executor calls, online replay reads,
retention/reacquisition/composition results, and reset/zero/permuted-route
interventions.

A positive experimental result requires:

- fresh-variant first-to-last improvement of at least 15 percentage points in
  at least three of four initially deficient mechanisms;
- final old-skill retention within five points of its post-acquisition score;
- returned skills recover their prior score using at most 25 percent of their
  first-acquisition feedback count;
- at least 70 percent success on novel two-skill compositions;
- reset, memory-zero, and route permutation each remove at least 15 points of
  the corresponding gain.

Tests must also prove constant memory size, exact snapshot/restore, frozen
compiler and slow weights online, selected-slot-only mutation, retained neutral
evidence, evidence-order invariance, rollback of a rejected update, slot-order
permutation equivariance, surface rename invariance, and a later-query
meta-gradient through an earlier feedback write.  An evaluator-only oracle-code
gate must distinguish readout capacity from acquisition failure without making
oracle identities or codes available online.

Failure is retained as evidence and triggers architectural revision, never a
domain solver or hidden routing label.  Passing is a bounded experimental
result only: normal LEARNING, Human-Flourishing, Slice, and milestone gates
remain `NOT_RUN`/`NOT_PASSED`.

## Current experimental result — 2026-08-27

The v47 proxy-energy route was rejected because its auxiliary loss improved
while the deployed policy did not.  V48 then trained only the existing
direction/reliability arbitration from deployed scalar preferences; it raised
fresh sampled-candidate preference accuracy from `0.5610` to `0.6058`, but
that result did not transfer to the full online causal gate.

V49 corrected the objective and credit path: every query scored the current
deployed action plus three policy-sampled explorations, used the complete
120-action softmax, and retained the differentiable graph through all 40
support observations.  Exactly 11,749 existing procedural parameters adapted;
no foundation model, deterministic solver, proxy energy, hidden target, or
utility vector participated.  Novel online composition improved from V48's
`0.6082` four-seed mean to `0.6369`, and the primary run reached `0.6636`, but
the required `0.70` was not reached and fast/reverse causal contributions were
not stable across seeds.  V49 is therefore retained as evidence and is not a
promoted candidate.  Exact hashes and results are in
`outputs/phase5-skill-memory-v49-result-summary.json`.

The residual fast/reverse adapter line is closed against further gate stacking.
The next architectural revision is one canonical goal-conditioned reversible
transition model in the deployed path: predict an outcome, infer a procedure
backward, execute it forward, and learn from its scalar consequences while
retaining fixed-capacity skill-local memory.

V51 implements that revision as one rank-8 additive coupling whose forward and
reverse passes share the same 2,064 learned parameters and are algebraic
inverses.  It bypasses the fast adapter, goal projection, direction mixer,
reliability gate, soft intermediate projection, and direct composition-memory
bias.  Two feedback-to-code heads plus the coupling form a 10,256-parameter
trainable seam; every other slow tensor remains frozen.  Credit uses only four
attempted public outputs and their scalar outcomes, centered within each query.

Across three independent 64-mapping runs, novel composition averaged `0.6379`
(`0.0024` sample standard deviation).  Removing only the reversible transition
reduced the mean to `0.5621`, a causal contribution of `+0.0758`, positive on
every seed.  A 256-mapping run improved to `0.6568`; its transition ablation was
`0.5602` (`+0.0966`).  Trained forward/reverse round trips had maximum absolute
error `2.38e-7`, and zero code remained a bit-exact identity.  All 124 relevant
tests pass.  The `0.70` threshold remains unmet, so V51 is retained evidence,
not a promoted result.  Exact records are in
`outputs/phase5-skill-memory-v51-result-summary.json`.

### Cross-family continuation — 2026-08-27

A direct frozen-V51 transfer to visible precedence graphs first failed: the
raw lossless graph packing produced negative acquired-state and reversible-core
effects.  That failure is retained rather than hidden.  It localized the
problem to an observation coordinate mismatch, because old-family retention
and the frozen core remained exact.

The successor added one typed, learned 9,536-parameter sensory adapter for the
public precedence-edge schema.  It is permutation-equivariant, has no answer
head, solver, target, family/task/skill identity, or access to evaluator state;
it was trained across 64 fresh opaque mappings per seed using only four
attempted public permutations and centered scalar outcomes.  The V51
procedural core and every other retained parameter remained frozen.  Online
acquisition then used one sampled output and one scalar score per unique
support, with no history replay.

Across three independent seeds, forward-path performance averaged `0.5688`
with a `+0.0813` acquired-state gain, positive on every seed.  Reverse-path
composition averaged `0.6083`; disabling only the unchanged V51 reversible
core removed `0.1042` on average (`+0.0813` minimum contribution).  The old
latent-program family changed by exactly `0.0000` after sensory-adapter
training and again by exactly `0.0000` after new-family acquisition, without
old-task replay.  All 133 combined tests pass.  This is bounded evidence that
one frozen procedural core can consume a newly learned observation interface
and acquire a structurally different procedure without erasing the retained
family.  It is not broad-domain transfer or AGI, and both families still share
the five-item permutation action algebra.  Exact records and the retained
zero-shot failure are in
`outputs/phase5-cross-family-v3-result-summary.json`.

### Demonstration-conditioned composition continuation — 2026-08-27

The new public-demonstration reader first produced a weak but transferable
procedure-direction signal on 19 unseen development mechanisms: correct
evidence alignment was `0.527371`, wrong-evidence alignment was `0.470038`
(`+5.7333` percentage points).  Repeating only 80 mechanisms for 256 updates
made the training signal much stronger but reversed the unseen advantage to
`-0.6002` points.  That run is retained as overfitting evidence and was not
used as the general-learning result.

The successor trained the conditional interface across 512 distinct mechanism
pairs while the V51 procedural core remained frozen.  Each episode supplied an
identity anchor, two demonstrated component procedures, and a public flag that
selected their composition.  Evaluation used 20 untouched mechanism pairs and
reloaded the train-only checkpoint before any final query.  Correct evidence
scored `0.5525`, versus `0.4700` after reset, `0.4700` without demonstrations,
`0.4800` with wrong demonstrations, and `0.4700` when the reversible transition
was removed.  Thus the acquired-state and reversible-transition effects were
both `+0.0825`; correct evidence exceeded wrong evidence by `+0.0725`.

This is a single-seed, bounded synthetic result within a shared five-item
permutation action algebra.  It is evidence of learned conditional procedural
composition, not AGI or broad-domain reasoning.  Old-family and precedence
retention remain to be remeasured after this conditional-interface update.
The exact compact record is
`outputs/phase5-conditional-v1-result-summary.json`.

Next exact action: implement and shadow-test FP32 equivalence-preserving
acceleration, then replicate the conditional result.  If replication or wider
generalization is weak, add a bounded episode-local latent procedure refined
from support-only scalar feedback through the same candidate-blind reversible
gate; do not substitute replay or a deterministic solution path.

## Effects and rollback

Local source, synthetic data, and foreground WSL2 GPU work only.  No language
model, foundation-weight update, network, service, personal/recovered data,
deployment, external effect, or promotion.  An online candidate rejection
restores its exact incoming memory bytes.  Code rollback returns to
`52a8345` and removes only this additive increment; the Phase-4 checkpoint and
evidence remain unchanged.
