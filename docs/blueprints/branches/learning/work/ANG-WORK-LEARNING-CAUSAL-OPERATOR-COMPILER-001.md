---
blueprint_id: ANG-WORK-LEARNING-CAUSAL-OPERATOR-COMPILER-001
parent_id: ANG-BP-META-UPDATER
revision: 2
tier: 4
design_status: approved
delivery_status: in_progress
human_authority: project owner direction and two-model architecture council, 2026-08-26
human_impact: LOW; contained local synthetic procedural learning
depends_on:
  - ANG-WORK-LEARNING-BIDIRECTIONAL-PROCEDURE-001
---

# Self-distilling causal operator compiler

## Objective

Create the first transferable unit of Angler competence: a parameterized
operator induced from executed transitions, represented both by a learned
relational mirror and by a neural module that predicts and realizes its effect.
Verified bounded search is initially a teacher; a goal-conditioned proposer
must absorb successful operator chains so search becomes a fallback rather
than the source of the procedure.

The same mechanism is intended to become self-hosting.  After cross-domain
transfer is demonstrated, repository state, executable requirements, typed
development tools, and independently verified outcomes become an ordinary
Angler environment.

## Exact first slice

- immutable namespaced record, goal, action, transition, and trace contracts;
- three independently namespaced reversible/partially irreversible domains;
- trace-delta abstraction and parameterized operator induction without domain
  rules or solution paths;
- schema-derived operator embeddings, initiation/effect/termination models,
  context-conditioned decoding, and a goal-conditioned operator proposer;
- bounded proposer-prior planning with explicit budget-one measurement;
- unit tests and one executable sequential transfer/retention experiment.

Implementation lives under `src/angler/procedures/`, with domain adapters under
`src/angler/worlds/`, tests under `tests/unit/procedures/`, and the experiment
runner under `experiments/runners/`.

## Learned versus deterministic boundary

Deterministic code may canonicalize typed records, execute declared actions,
measure deltas, meter search, and judge externally observable outcomes.  It may
not encode domain solution routes, cross-domain operator equivalence, action
selection, subgoals, or expected answers.  Operator identity must be induced
from interventional traces.  Its neural executor must generalize to unseen
bindings, and its utility must disappear when the induced operator is removed
or permuted.

## Acceptance sequence

1. Reproduce the existing token-world capability through the general record
   interface without exposing a solver.
2. Induce at least one multi-step operator that executes its learned effect on
   unseen entity bindings and improves equal-budget planning.
3. Train the direct proposer from verified traces and show budget-one success
   rises while search remains fixed.
4. Stream a second and third namespaced domain through the same lineage;
   demonstrate faster acquisition from retained operators than a fresh
   lineage, retain prior competence, and solve unseen operator compositions.
5. Ablate the transferred operator, neural proposer, and symbolic mirror
   separately so neither deterministic search nor a frozen model can explain
   the effect.

A first-slice pass is not AGI.  It establishes only a reusable procedural unit
and a credible path to self-hosting.  Stop or redesign if operator count tracks
episodes, cross-domain removal has no effect, budget-one performance does not
rise, or the general interface requires a domain-specific solution feature.

## Integrated v5 result (2026-08-27)

The implemented core is a learned procedural model, not a token predictor and
not a foundation-model update.  It encodes current and desired relational
states, predicts forward and reverse latent transitions, ranks grounded
operator bindings, decodes primitive actions, predicts termination, and learns
from externally verified transition/trajectory evidence.

Before the clean run, a review found that held-out inference treated the public
execution ceiling as the exact solution horizon.  That oracle leak was removed.
Training and inference now share one horizon-agnostic learned scoring path:
each successor is interpreted as both a possible goal state and a possible
bridge to the reverse frontier, with the externally grounded learned
termination head selecting the blend.  The ceiling only bounds execution.

One clean run started from random weights with seed `642017`, used 64 varied
multi-step trajectories for full-plastic Tokens acquisition, then exposed only
22 procedure-selection tensors during incremental Files and Boxes experience.
Replay was bounded to 25 percent and incremental trajectories used pure
self-rollout.  The result was:

| Measurement | Result |
|---|---:|
| Initial Tokens acquisition | 38/40 (95%) |
| Files zero-shot -> after update | 34/40 -> 34/40 |
| Boxes zero-shot -> after update | 18/40 -> 33/40 |
| Prior-domain retention during Boxes update | Tokens 95% -> 95%; Files 85% -> 85% |
| Untouched final, all domains | 114/120 (95%) |
| Untouched final by domain | Tokens 38/40; Files 38/40; Boxes 38/40 |
| Untouched final four-step compositions | 55/60 (91.7%) |
| Proposal-time world executions | 0 |
| Full unit suite | 238/238 |

Causal controls on the same untouched partition were: untrained 0/120,
permuted learned binding 0/120, learned proposer removed 101/120, symbolic
mirror removed 0/120, transfer aliases removed 41/120, and the Boxes operator
retired 76/120.  These results show that learned selection, the induced
operator, and cross-domain alignment materially contribute.

The Files update did not improve its fixed adaptation partition despite
changing only the allowed 22 tensors, and neither incremental domain crossed
the predeclared 90-percent threshold on every adaptation stratum.  The 95%
untouched result is therefore a strong experimental result, not completion of
lifelong learning or evidence that every attempted update is beneficial.

Full generated evidence and the checkpoint are preserved locally under
`outputs/phase4-continual-v5-clean.{json,pt}`.  The compact tracked record is
`experiments/manifests/phase4-causal-operator-compiler-20260827.json`.

## Next exact build action

Keep this learned operator compiler as the stable procedural substrate.  Add
skill-local procedural memory and learned routing so new feedback updates only
the responsible skill state, no-op updates can be rejected, and useful skills
can be consolidated without replaying every historical problem.  Test
acquisition, transfer, composition, and retention across a changing mechanism
stream before connecting a foundation model as the language/knowledge cortex.

## Effects and rollback

Local source, synthetic data, and foreground WSL2 training only.  No external
effects, service, personal/recovered data, deployment, or foundation-model
update.  Rollback returns to commit `5137a15` and removes only this additive
successor slice.
