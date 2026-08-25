---
contract_id: ANG-CTR-RESOURCE-INVENTORY-001
version: 1
status: approved_design
owner: ANG-BP-RESOURCES
producer: ANG-BP-CAPABILITY-INVENTORY
consumers: [ANG-BP-RESOURCE-PROBES, ANG-BP-EXECUTION-PLANNER, ANG-BP-EVIDENCE]
---

# ResourceInventory v1

An immutable inventory contains content identity, host/topology pseudonymous identity, collection/freshness interval, collector build, operating system/runtime/driver facts, CPU/NUMA, RAM/swap, accelerator/memory/precision/kernel support, storage, interconnect/network, process/container constraints, administrative time/cost/energy/use ceilings, redactions, and shared evidence envelope.

Every capability records value/unit, source (`OS`, `DRIVER`, `CONFIG`, `PROBE`, `INFERRED`), confidence/uncertainty, observed_at, expires_at, visibility, and authorized ceiling. Physical capacity and use authority are different fields. Missing/stale/unknown values cannot satisfy a required capability.

Collection is read-only and idempotent for a collection identity; probes are separate authorized artifacts. Corrections produce successor inventories. Consumers reject unsupported major versions, identity/timestamp/unit errors, authority above physical capacity, and sensitive fields outside visibility policy.

## Operation and failure contract

Collection runs under an exact permission/profile/timeout ceiling and records start/end, partial sources, and cleanup. Typed failures are `SOURCE_UNAVAILABLE`, `PERMISSION_DENIED`, `TIMEOUT`, `UNIT_INVALID`, `TOPOLOGY_CHANGED`, `SENSITIVE_FIELD`, and `INVENTORY_INCOMPLETE`. A partial inventory may be recorded with explicit unknowns but cannot satisfy a plan requirement that depends on them. Permission/sensitive-field failure is not retryable without successor authorization; transient source/timeout failure may retry under a new attempt identity with the same ceiling. Inventory collection never invokes an empirical workload or mutates host configuration.

Unknown major versions reject. Minor additions preserve unknown optional data and cannot reinterpret units, authority, source, uncertainty, or freshness. Any changed observation, topology, authority, or policy yields a successor inventory/content identity; migration is explicit and originals remain evidence.

Producer/consumer tests cover deterministic synthetic serialization, units, source/uncertainty, stale/unknown fields, authority below physical capacity, permission denial, timeout/partial collection, topology change, sensitive-field redaction, unknown-major rejection, additive-minor preservation, idempotent duplicate admission, and evidence-envelope identity.
