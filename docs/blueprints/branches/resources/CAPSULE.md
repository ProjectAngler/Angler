---
blueprint_id: ANG-BP-RESOURCES
blueprint_revision: 2
capsule_revision: 3
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 750
---

# RESOURCES capsule

Mission: make Project Angler scale from constrained hardware through clusters without changing its scientific meaning or treating the current 16 GB GPU as a ceiling.

Owns capability inventory, bounded probes, execution planning, placement, safe replanning, competence migration, and scaling evaluation.

Produces `ResourceInventory`, `ExecutionPlan`, and `MigrationProposal`. Plans bind model/precision, adapter topology, budgets, placement, parallelism, headroom, constraints, objective, and probe evidence. Plan changes create new experiment identities.

Planning: discover → eliminate infeasible/unsafe plans → optimize feasible plans against user-selected quality, time, cost, energy, or latency. A future learned planner may predict outcomes but cannot redefine objectives or constraints.

Required children: CAPABILITY-INVENTORY, RESOURCE-PROBES, EXECUTION-PLANNER, PLACEMENT, SAFE-REPLANNING, COMPETENCE-MIGRATION, SCALING-EVAL.

No query-conditioned model routing. Compatible placement may reload state at a transaction boundary; incompatible model/topology changes require replay/distillation and full causal regating.

Inventory separates observed capacity, uncertainty, and administrative permission. Probes are separately authorized, bounded, idempotent, and disposable. Plans first eliminate violations, then optimize only among feasible candidates; every plan reserves evaluation, rollback, and host-stability headroom set by external policy.

Current gate: `ANG-GATE-RESOURCE-DESIGN-001`. Tier-1 and inventory/probe/planner designs are approved for CR0; schema/synthetic-test delivery remains blocked by EVIDENCE-SCHEMAS. No real probe, GPU/model use, or milestone pass is authorized.

Next action: execute the release-listed resource schema leaf, then seek a successor assessment before any real host/GPU probe. Read root capsule, RESOURCES plus three active child capsules, ADR-0002, and the CR0 leaf.
