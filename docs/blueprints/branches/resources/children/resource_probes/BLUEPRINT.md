---
blueprint_id: ANG-BP-RESOURCE-PROBES
title: Bounded empirical probes
parent_id: ANG-BP-RESOURCES
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: deferred_to_successor_leaf
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-BP-CAPABILITY-INVENTORY
contracts_in:
  - ANG-CTR-RESOURCE-INVENTORY-001@1
contracts_out:
  - ANG-CTR-RESOURCE-INVENTORY-001@1
gates:
  - ANG-GATE-RESOURCE-PROBES-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Bounded empirical probes

## Outcome

Define disposable measurements that refine only named inventory fields. A ProbeSpec binds input inventory, exact operation, expected allocation, maximum duration/iterations/output/temp storage, permitted devices/processes, cleanup, abort signal, measurement/tolerance, idempotency key, and authorization. It cannot install dependencies, change system configuration, use a network, retain background processes, or mutate model/state.

Probe results are evidence with observed value, uncertainty, resource high-water marks, completion/abort reason, cleanup verification, and successor inventory link. Failure or timeout preserves the prior inventory and marks the probed capability unknown or bounded; it never expands a ceiling.

CR0 approves this design and reserves its provenance/result fields inside ResourceInventory, but does not build ProbeSpec/result schemas or execute probes. Delivery is deferred to a successor exact leaf after the inventory/plan scaffold passes. That leaf must declare literal schema/fixture outputs; a separate human-impact assessment must name any actual host command and ceiling.
