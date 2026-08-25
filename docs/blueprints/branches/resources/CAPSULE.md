---
blueprint_id: ANG-BP-RESOURCES
blueprint_revision: 2
capsule_revision: 6
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

The Evidence predecessor is satisfied only for CR0 scaffolding by decision SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`; its normal gate remains `NOT_RUN`. Resources activation attempt 003 was `REJECTED` and is immutable history. Revalidation 004 later authorized the bounded synthetic leaf, independently `SCAFFOLD_ACCEPTED` at commit `7f383939d021c4bba9dd5af046ce0838b032ff02`, receipt SHA-256 `D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B`.

The accepted outputs are schemas, synthetic fixtures, a deterministic test, and handoff only. They contain no measured inventory and prove no host, GPU, placement, or model capability. The leaf is historical and must not be rerun. Normal Resource and Human-Flourishing gates remain `NOT_RUN`; Slice 00 and M0 remain `NOT_PASSED`. Any real probe or successor work requires a new manifest and authority after continuity reconciliation.
