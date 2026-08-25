---
blueprint_id: ANG-BP-CAPABILITY-INVENTORY
title: Capability inventory
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
  - ANG-BP-EVIDENCE-SCHEMAS
contracts_out:
  - ANG-CTR-RESOURCE-INVENTORY-001@1
gates:
  - ANG-GATE-INVENTORY-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Capability inventory

## Outcome

Produce an immutable snapshot of discoverable hardware/software/topology plus a separate administrative-capacity ceiling. Each datum declares source (`OS`, `DRIVER`, `CONFIG`, `PROBE`, or `INFERRED`), collection time, units, uncertainty/freshness, visibility, and permission to use. Discovery never implies authority.

Inventory covers CPU/NUMA, RAM/swap, accelerators and memory/precision/kernel capabilities, storage capacity/performance class, interconnect/network topology, operating system/drivers/runtimes, process/container limits, energy/cost/time ceilings, and unavailable/unknown fields. Sensitive identifiers are redacted or separately protected.

## Tests, failure, rollback

CR0 uses synthetic inventories only. Schema tests cover constrained single-device, observed-class workstation, larger server, cluster, missing/unknown values, topology changes, stale data, permission below physical capacity, and sensitive-field redaction. Unknown capability prevents plans that require it; it is never guessed as available. Corrections create successor inventories.

Next: build the v1 schema and fixtures under the exact CR0 leaf.
