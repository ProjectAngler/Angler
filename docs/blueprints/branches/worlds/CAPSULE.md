---
blueprint_id: ANG-BP-WORLDS
blueprint_revision: 2
capsule_revision: 3
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# WORLDS capsule

Mission: provide controllable experience and outcome feedback that contains learnable procedures without prescribing them.

Owns environment protocol, fixed procedural families, outcome verifiers, environment validation, later curriculum scheduling, and much later generated worlds. SCIENCE independently owns partitions and claims.

Produces `TaskSpec`, `Observation`, transitions, and `Feedback`. Feedback may report observable failures or constraints allowed by contract but never hidden transfer answers or a target reasoning chain.

Required children: ENVIRONMENT-PROTOCOL, FIXED-FAMILIES, OUTCOME-VERIFIERS, ENVIRONMENT-VALIDATION, CURRICULUM, GENERATED-WORLDS.

Initial families: symbolic rule induction, constraint decomposition, counterexample search, causal intervention, program repair, and tool-choice reasoning. Start with two, each with independent surface generation.

The v1 protocol contracts separate immutable `TaskSpec`, learner-visible `Observation`, and outcome-only `Feedback`; every transition binds environment/version, instance, step, visibility, seed commitment, budgets, permissions, and evidence identity.

Current gate: `ANG-GATE-WORLDS-DESIGN-001`. Tier-1 and ENVIRONMENT-PROTOCOL designs, including the explicit Action/step boundary, are approved for CR0; delivery remains blocked by safety/evidence predecessors. No environment execution, model use, or milestone pass is authorized.

Top risks: solution-prescribing verifiers, task leakage, curriculum collusion, and non-replayable generation.

Next action: execute the release-listed protocol schema leaf, then expand the two selected fixed families with verifier and validator contracts. Read root capsule, WORLDS and ENVIRONMENT-PROTOCOL capsules, SCIENCE capsule, ADR-0002, and the CR0 leaf.
