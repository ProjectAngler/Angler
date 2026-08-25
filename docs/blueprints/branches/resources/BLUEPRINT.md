---
blueprint_id: ANG-BP-RESOURCES
title: Resource adaptation and portability
parent_id: ANG-BP-ROOT
revision: 2
tier: 1
design_status: approved_for_cr0
delivery_status: blocked_by_evidence_schemas
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-CAPABILITY-INVENTORY
  - ANG-BP-RESOURCE-PROBES
  - ANG-BP-EXECUTION-PLANNER
  - ANG-BP-PLACEMENT
  - ANG-BP-SAFE-REPLANNING
  - ANG-BP-COMPETENCE-MIGRATION
  - ANG-BP-SCALING-EVAL
depends_on:
  - ANG-BP-EVIDENCE
  - ANG-BP-SAFETY
contracts_out:
  - ANG-CTR-RESOURCE-INVENTORY-001
  - ANG-CTR-EXECUTION-PLAN-001
  - ANG-CTR-MIGRATION-PROPOSAL-001
gate: ANG-GATE-RESOURCE-DESIGN-001
adrs:
  - ANG-ADR-0002
  - ANG-ADR-0004
---

# Resource adaptation and portability

## Context capsule

This branch lets the same logical system operate efficiently on constrained devices, the current workstation, larger servers, or clusters. It profiles actual capability, runs bounded probes, selects a feasible plan against a user objective, places workloads, replans only at safe boundaries, and explicitly migrates competence across incompatible model/topology changes.

## Contribution to the root

No hardware tier is a permanent project limit. Resource elasticity allows the minimal learner to start locally without allowing today's machine to hard-code tomorrow's architecture or scientific claims.

## Inherited invariants

Applies: `ANG-INV-ONE-COMPETENCE-001`, `ANG-INV-CAUSAL-PROMOTION-001`, `ANG-INV-REVERSIBLE-UPDATES-001`, `ANG-INV-ELASTIC-COMPUTE-001`, and `ANG-INV-EXTERNAL-AUTHORITY-001`.

## Scope

- Inventory accelerator, CPU, RAM, storage, topology, kernels, precision support, and administrative budgets.
- Run bounded empirical probes for usable memory, throughput, offload, communication, and representative learner/evaluator operations.
- Filter infeasible plans and optimize feasible ones for declared priorities such as quality, time, cost, energy, or latency.
- Place model, updater, replay, environments, and evaluation across a device or cluster.
- Preserve evaluation/rollback headroom rather than filling all capacity with the model.
- Replan after resource changes only at transaction boundaries.
- Migrate accepted competence by replay/distillation and full regating across incompatible model or adapter topology.
- Measure quality/resource Pareto frontiers and cross-plan reproducibility.

## Explicit non-goals

- Choosing a different model or adapter per query.
- Changing the scientific objective, data visibility, or promotion threshold to fit hardware.
- Treating a planner prediction as proof that a plan is safe or optimal.
- Silent precision/topology changes inside a learning transaction.
- Direct raw-adapter loading across incompatible signatures.

## Contracts

RESOURCES emits a measured `ResourceInventory`, validated `ExecutionPlan`, placement handle details, and explicit `MigrationProposal`. A plan binds model/precision, adapter topology, context/replay budgets, optimizer state, placement, parallelism, reserved headroom, probes, constraints, and optimization objective.

Every consuming run records the plan identity. Changing a scientifically relevant plan field creates a new experiment identity.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-CAPABILITY-INVENTORY` | Stable inventory of hardware, software, topology, and administrative capacity | `ANG-GATE-INVENTORY-001` | approved; construction-ready |
| `ANG-BP-RESOURCE-PROBES` | Safe measured capabilities and headroom | `ANG-GATE-RESOURCE-PROBES-001` | design approved; delivery deferred to successor leaf |
| `ANG-BP-EXECUTION-PLANNER` | Feasible plan selection against explicit objectives/constraints | `ANG-GATE-EXECUTION-PLANNER-001` | approved; construction-ready |
| `ANG-BP-PLACEMENT` | Single-device, offload, sharded, parallel, and distributed materialization | `ANG-GATE-PLACEMENT-001` | stub |
| `ANG-BP-SAFE-REPLANNING` | Transaction-boundary shrink/expand behavior | `ANG-GATE-REPLANNING-001` | stub |
| `ANG-BP-COMPETENCE-MIGRATION` | Replay/distill/regate into a new compatible lineage | `ANG-GATE-MIGRATION-001` | deferred until promoted competence exists |
| `ANG-BP-SCALING-EVAL` | Quality/resource Pareto and cross-plan comparison | `ANG-GATE-SCALING-EVAL-001` | stub |

## Planning model

Planning has three stages:

1. **Discover:** inventory and bounded empirical probes.
2. **Constrain:** eliminate incompatibility, insufficient headroom, permission, time, energy, or cost violations.
3. **Optimize:** compare remaining plans using a declared objective and uncertainty bounds.

The first planner uses rules plus measurements. A later learned policy may predict plan performance from accumulated receipts, but the user objective and hard constraints remain external and plan validation remains independent.

## Dependencies and sequencing

EVIDENCE defines plan/inventory identity and receipts. SAFETY defines probe and resource limits. INVENTORY, PROBES, and PLANNER form the first package for slice 00. PLACEMENT integrates in slice 02. SAFE-REPLANNING follows transaction semantics. MIGRATION begins only after M3 and a materially different target exists.

## Acceptance gate and evidence

`ANG-GATE-RESOURCE-DESIGN-001` passes when every workload receives a declared resource responsibility, every plan reserves rollback/evaluation headroom, incompatible configurations fail before model mutation, and the same interfaces support constrained, workstation, server, and cluster profiles.

M1 later requires feasible plans for the current machine plus simulated materially smaller and larger topologies without changing scientific interfaces.

## Testing and validation

- Inventory stability and topology-change detection.
- Probe containment, timeout, and cleanup.
- Planner feasibility and headroom rejection tests.
- Objective trade-off and uncertainty tests.
- Simulated resource disappearance at every safe boundary.
- Plan identity and experiment-change tests.
- Placement contract tests across available backends.
- Migration causal/retention regate when activated.

## Risks and rollback

- `ANG-RISK-PLAN-OVERREACH-001`: model consumes evaluation/rollback capacity. Enforce reservations and fallback plan.
- `ANG-RISK-BAD-PROBE-001`: probe destabilizes host. Bound, isolate, time out, and clean up.
- `ANG-RISK-PLAN-DRIFT-001`: runtime diverges from recorded plan. Abort transaction and record mismatch.
- `ANG-RISK-SCALE-SEMANTICS-001`: precision/topology changes alter conclusions. New plan identity and cross-plan replication.
- `ANG-RISK-MIGRATION-LOSS-001`: competence fails to transfer. Preserve source lineage and reject target promotion.

## Current status and next leaf

The Tier-1 boundary and coordinated CAPABILITY-INVENTORY, RESOURCE-PROBES, and EXECUTION-PLANNER designs are approved for CR0. The CR0 resource leaf constructs inventory/plan schemas and synthetic smaller/observed/larger-profile feasibility tests; it reserves probe-provenance fields but does not implement ProbeSpec/result schemas. Probe delivery requires a successor exact leaf and assessment. CR0 cannot probe the host/GPU, load a model, install packages, or use the network. The current 16 GB workstation remains one future measured profile, not a baked-in limit.
