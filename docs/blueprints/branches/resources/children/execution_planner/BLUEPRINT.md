---
blueprint_id: ANG-BP-EXECUTION-PLANNER
title: Execution planner
parent_id: ANG-BP-RESOURCES
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: blocked_by_evidence_schemas
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-BP-CAPABILITY-INVENTORY
  - ANG-BP-RESOURCE-PROBES
contracts_in:
  - ANG-CTR-RESOURCE-INVENTORY-001@1
contracts_out:
  - ANG-CTR-EXECUTION-PLAN-001@1
gates:
  - ANG-GATE-EXECUTION-PLANNER-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Execution planner

## Outcome

Select a reproducible feasible plan without turning the current machine into an architectural limit. Planning is lexicographic: authenticate inventory/policy; enumerate candidate profiles; reject permission, compatibility, headroom, time/cost/energy, safety, or scientific-semantic violations; then rank feasible candidates by the externally supplied objective and tie-breaker.

A plan binds exact inventory/policy/objective, model/tokenizer/precision placeholder or admitted identity, adapter/optimizer shape, context/replay/batch/update budgets, placement/parallelism/offload, environment/evaluator concurrency, mandatory host/evaluation/rollback headroom, predicted uncertainty, probes relied on, fallback, validity window, and plan-drift checks. Unknown required capability is infeasible.

CR0 tests pure planning over synthetic constrained, workstation-class, larger-server, and cluster inventories plus missing-capability/headroom/permission/semantic-drift controls. It loads no model and uses no GPU. A materially changed plan creates a new experiment identity.

## Failure and rollback

No feasible candidate yields typed `NO_FEASIBLE_PLAN` with violated constraints and no mutation. Runtime drift aborts at the next safe boundary and preserves the last valid state. Next: build v1 schemas and pure synthetic feasibility tests under the release leaf.
