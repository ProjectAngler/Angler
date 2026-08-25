---
blueprint_id: ANG-BP-SCIENCE
blueprint_revision: 2
capsule_revision: 3
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# SCIENCE capsule

Mission: prove that a promoted change is transferable reasoning competence rather than memory, leakage, verifier exploitation, or extra compute.

Owns hidden partitions, fair baselines, causal interventions, statistics, adversarial evaluation, evaluation receipts, and promotion policy. Does not implement training, runtime, environments, evidence storage, or resource placement.

Consumes `TaskSpec`, `PlasticState`, `Episode`, and `ExecutionPlan`. Produces `EvaluationSuite`, `EvaluationReceipt`, and `PromotionDecision`.

Required children: BENCHMARKS, PARTITIONS, BASELINES, CAUSAL-TESTS, PROMOTION, ADVERSARIAL-EVAL. Initial expansion must bind benchmark claims, sealed visibility, and fair budgets together.

Non-negotiable tests: held-out structural transfer; equal-budget fair-RAG; state zero/swap/permutation; replay; retention/composition; shuffled feedback; tool-disabled control; leakage canaries; matched resources.

Initial construction decisions: symbolic rule induction and constraint decomposition are the first two benchmark families. Adaptation/development data are distinct from sealed promotion/final-transfer data. Candidate and baselines bind identical task visibility and predeclared token/tool/compute accounting.

Current gate: `ANG-GATE-SCIENCE-DESIGN-001`. Tier-1 and BENCHMARKS/PARTITIONS/BASELINES designs are approved for CR0, with detailed suite/receipt/decision contracts; delivery remains blocked by evidence, resources, and worlds predecessors. No scientific evidence or milestone pass is claimed.

Top risks: hidden-answer leakage, unfair budgets, threshold changes after results, and promotion-set overuse.

Next action: execute the release-listed SCIENCE schema leaf, then independently validate leakage canaries and budget equality before activating a runtime. Read root capsule, SCIENCE blueprint, the three active child capsules, ADR-0002, and the CR0 leaf.
