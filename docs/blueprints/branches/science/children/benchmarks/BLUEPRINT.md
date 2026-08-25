---
blueprint_id: ANG-BP-BENCHMARKS
title: Benchmark families
parent_id: ANG-BP-SCIENCE
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: blocked_by_cr0_predecessors
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-BP-ENVIRONMENT-PROTOCOL
contracts_in:
  - ANG-CTR-TASK-SPEC-001@1
contracts_out:
  - ANG-CTR-EVALUATION-SUITE-001@1
gates:
  - ANG-GATE-BENCHMARKS-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Benchmark families

## Outcome and scope

Own a versioned statement of each procedural skill, allowed transformations, success measure, known shortcuts, and generator/verifier separation. CR0 specifies synthetic fixtures only; it does not generate training data or run a model.

The first two families are:

1. **Symbolic rule induction:** infer a transformation from demonstrations and apply it to a new instance. Structural variants rename symbols, reorder demonstrations, change irrelevant formatting, and use independently sampled instances. Exact externally computed outputs determine success.
2. **Constraint decomposition:** produce a satisfying assignment or a valid unsatisfiability certificate for small compositional constraints. Variants rename/reorder variables and constraints, alter graph layout and irrelevant wording, and change instance size inside a declared envelope. A solver-independent checker validates only the outcome.

Family specifications contain `family_id`, semantic version, claimed procedure, instance grammar, transformation set, difficulty dimensions, solution/verifier identity, budget class, leakage hypotheses, and compatible partition policy. They never contain a preferred reasoning trace.

For CR0, these benchmark-family definitions are embedded as versioned sections of `schemas/control/v1/science/evaluation-suite.schema.json`. Their known-valid, transformed, leaky, and invalid cases live only in the release leaf's declared control and negative fixtures; CR0 authorizes no separate benchmark artifact or executable generator.

## Non-goals and boundaries

No chain-of-thought targets, answer-specific scripts, model calls, curriculum adaptation, hidden-seed access, or claims that these two families exhaust reasoning. WORLDS owns generators/verifiers; SCIENCE owns what transformations support the claim.

## Gate, tests, and rollback

`ANG-GATE-BENCHMARKS-001` requires a known-solution round trip, invariance under every declared surface transform, detection of a deliberately leaky cue, verifier mutation rejection, and independent mapping from each score to an observable outcome. Evidence is content-addressed. Failure retires the family version and all dependent evaluation identities; rollback restores the prior specification.

## Next leaf

Construct only the benchmark sections embedded in the release-listed evaluation-suite schema and their declared synthetic control/negative fixtures within the CR0 leaf's exact write scope.
