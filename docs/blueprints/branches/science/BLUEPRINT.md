---
blueprint_id: ANG-BP-SCIENCE
title: Scientific proof and evaluation
parent_id: ANG-BP-ROOT
revision: 2
tier: 1
design_status: approved_for_cr0
delivery_status: blocked_by_cr0_predecessors
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-BENCHMARKS
  - ANG-BP-PARTITIONS
  - ANG-BP-BASELINES
  - ANG-BP-CAUSAL-TESTS
  - ANG-BP-PROMOTION
  - ANG-BP-ADVERSARIAL-EVAL
depends_on:
  - ANG-BP-WORLDS
  - ANG-BP-RUNTIME
  - ANG-BP-EVIDENCE
  - ANG-BP-RESOURCES
contracts_in:
  - ANG-CTR-TASK-SPEC-001
  - ANG-CTR-PLASTIC-STATE-001
  - ANG-CTR-EPISODE-001
  - ANG-CTR-EXECUTION-PLAN-001
contracts_out:
  - ANG-CTR-EVALUATION-SUITE-001
  - ANG-CTR-EVALUATION-RECEIPT-001
  - ANG-CTR-PROMOTION-DECISION-001
gate: ANG-GATE-SCIENCE-DESIGN-001
adrs:
  - ANG-ADR-0002
  - ANG-ADR-0004
---

# Scientific proof and evaluation

## Context capsule

This branch decides whether apparent improvement is transferable reasoning competence rather than memorization, retrieval, answer leakage, verifier exploitation, or extra compute. It owns hidden partitions, fair baselines, causal interventions, statistical promotion policy, and adversarial evaluation. It does not train the learner or author the runtime.

## Contribution to the root

Project Angler's central claim is scientific, not merely functional. Without this branch, an adapter that repeats an episode, a larger prompt, or a query-selected memory could be mislabeled as evolving reasoning.

## Inherited invariants

Applies: `ANG-INV-ONE-COMPETENCE-001`, `ANG-INV-EVIDENCE-SEPARATION-001`, `ANG-INV-CAUSAL-PROMOTION-001`, `ANG-INV-OUTCOME-JUDGES-001`, `ANG-INV-ELASTIC-COMPUTE-001`, and `ANG-INV-EXTERNAL-AUTHORITY-001`.

## Scope

- Define procedural skill claims and their held-out structural variants.
- Control adaptation, development, promotion, final-transfer, retention, and composition partitions.
- Define frozen, fair-RAG, extra-token, conventional fine-tuning, random/shuffled-update, per-episode memory, and oracle baselines.
- Specify state-zero, state-swap, permutation, replay, shuffled-feedback, and tool-disabled interventions.
- Lock statistical thresholds before adaptive results are viewed.
- Issue evaluation receipts and promotion decisions; SAFETY retains veto authority.

## Explicit non-goals

- Implementing environments, model runtime, adapter updates, or evidence storage.
- Selecting an optimizer based on hidden promotion results.
- Treating fluent output, loss reduction, or a single successful seed as sufficient evidence.
- Redefining resource budgets after seeing candidate performance.

## Contracts

The branch consumes versioned tasks, states, episodes, and execution plans. It produces a sealed `EvaluationSuite`, an immutable `EvaluationReceipt`, and a precommitted `PromotionDecision`. Hidden labels and promotion items must be inaccessible to the learner, updater, curriculum generator, and self-critique path.

## Internal design

Evaluation is separated into four layers:

1. **Construct validity:** each benchmark family isolates a named procedure and supports surface transformations.
2. **Fair comparison:** candidate and baselines receive matched observable information, token/compute budgets, tools, and execution plans.
3. **Causal attribution:** intervene on the plastic state while holding model, prompt, task, and plan fixed.
4. **Promotion policy:** aggregate predeclared metrics across seeds, task families, retention, safety, and resource cost.

Promotion data remains sealed until the candidate identity is frozen. Final-transfer generators and seeds are used only at milestone gates.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-BENCHMARKS` | Named procedural skills with independent surface generators and composition cases | `ANG-GATE-BENCHMARKS-001` | approved; construction-ready |
| `ANG-BP-PARTITIONS` | Disjoint, sealed visibility and seed policy | `ANG-GATE-PARTITIONS-001` | approved; construction-ready |
| `ANG-BP-BASELINES` | Fair comparison matrix and budget accounting | `ANG-GATE-BASELINES-001` | approved; construction-ready |
| `ANG-BP-CAUSAL-TESTS` | State intervention and replay test specifications | `ANG-GATE-CAUSAL-TESTS-001` | stub |
| `ANG-BP-PROMOTION` | Precommitted statistics and promotion policy | `ANG-GATE-PROMOTION-POLICY-001` | stub |
| `ANG-BP-ADVERSARIAL-EVAL` | Leakage, reward hacking, latent retrieval, and verifier exploit tests | `ANG-GATE-ADVERSARIAL-001` | stub |

## Dependencies and sequencing

WORLDS supplies task and verifier packages but cannot see sealed transfer answers. EVIDENCE supplies identity and receipt schemas. RESOURCES supplies matched plans and measured budgets. RUNTIME supplies exact state interventions. SCIENCE defines their acceptance meaning without implementing those branches.

`BENCHMARKS`, `PARTITIONS`, and `BASELINES` expand first; `CAUSAL-TESTS` follows the state contract; `PROMOTION` locks thresholds after baseline variance measurement but before adaptive evaluation.

## Acceptance gate and evidence

`ANG-GATE-SCIENCE-DESIGN-001` passes when every child has a bounded outcome, owned contracts, leakage boundary, negative controls, evidence requirements, and a path through slices 01–04. It requires an independent review showing that an episode memorizer, fair-RAG system, and query-conditioned adapter bank cannot satisfy the claimed causal gate by construction.

## Testing and validation

- Generator disjointness and transformation tests.
- Producer/consumer contract tests for suite and receipt schemas.
- Deliberate leakage canaries that must be detected.
- Dummy state effects proving zero/swap/permutation harness behavior.
- Budget-accounting equality tests.
- Statistical simulations for false-promotion behavior.
- Cross-plan replication rules.

## Risks and rollback

- `ANG-RISK-LEAKAGE-001`: hidden answers reach training or prompt assembly. Fail closed and invalidate the evaluation identity.
- `ANG-RISK-FAIRNESS-001`: baselines receive unequal information or compute. No promotion decision.
- `ANG-RISK-METRIC-GAMING-001`: threshold or metric changes follow result inspection. Supersede through ADR and rerun under a new identity.
- `ANG-RISK-OVERFIT-SUITE-001`: repeated use turns promotion data into training data. Rotate sealed generators and preserve final-transfer reserve.

## Resource scaling

Scientific meaning is invariant across resource plans. Larger resources may increase seed count, evaluator parallelism, or model capacity, but comparison groups remain plan-matched and results remain bound to the exact plan. Cross-plan conclusions require explicit replication, not pooled convenience.

## Current status and next leaf

The Tier-1 ownership boundary and the coordinated BENCHMARKS, PARTITIONS, and BASELINES designs are approved for Construction Release 0. This approval authorizes schema/scaffold work only; no scientific result, promotion threshold, Slice 00, or milestone has passed. The next executable work is the SCIENCE leaf named by the release manifest, limited to synthetic specifications and tests.
