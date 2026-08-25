---
blueprint_id: ANG-BP-RESOURCE-PROBES
blueprint_revision: 1
capsule_revision: 2
freshness_date: 2026-08-25
parent_id: ANG-BP-RESOURCES
target_tokens: 500
---

# RESOURCE-PROBES capsule

Mission: refine named resource facts through separately authorized, bounded, abortable, cleanup-verified measurements.

A ProbeSpec freezes operation, allocation, time/iteration/output/temp ceilings, permitted devices/processes, measurement, tolerance, idempotency, abort, cleanup, and authorization. Failure never expands capacity and preserves the prior inventory.

Gate: `ANG-GATE-RESOURCE-PROBES-001`; design is approved for CR0, but delivery is deferred to a successor leaf. CR0 only reserves probe provenance/result fields in ResourceInventory—no ProbeSpec schema and no CPU/GPU/storage probe. Next after the CR0 inventory/plan gate: issue a separate leaf/assessment for synthetic probe schemas before considering any actual measurement.
