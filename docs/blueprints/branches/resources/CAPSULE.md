---
blueprint_id: ANG-BP-RESOURCES
blueprint_revision: 2
capsule_revision: 4
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

The Evidence predecessor is satisfied only for CR0 scaffolding by `SCAFFOLD_ACCEPTED` decision SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`; the normal Evidence gate remains `NOT_RUN`. CR0 gate `ANG-GATE-CR0-RESOURCES-001@1` is specified, and the exact Resource leaf specification is ready, but revalidation `ANG-CR0-REVALIDATION-20260825-003` is PENDING/NON-AUTHORIZING. The normal Resource design gate, Human-Flourishing gate, Slice 00, and M0 remain unpassed. No real probe, GPU/model use, or milestone pass is authorized.

Next action: the bound independent revalidation reviewer inspects the packet and writes only its reserved decision. Do not execute the leaf/test or create any of its nine outputs. After an `APPROVED` decision and separate Manifest v2 PASS/authorization, execute only the frozen Resource leaf; any host/GPU probe still requires a successor assessment and leaf.
