---
blueprint_id: ANG-BP-ENVIRONMENT-PROTOCOL
blueprint_revision: 1
capsule_revision: 3
freshness_date: 2026-08-25
parent_id: ANG-BP-WORLDS
target_tokens: 550
---

# ENVIRONMENT-PROTOCOL capsule

Mission: make reset, action transitions, scoring, cleanup, and replay stable across task families without prescribing reasoning.

Inputs are an immutable TaskSpec, authorized seed commitment, execution-plan reference, and actions. Outputs are learner-visible Observations and outcome-only Feedback with shared evidence identities. WORLDS owns state/verifiers; RUNTIME acts; SCIENCE owns partitions/claims; EVIDENCE records.

Typed failures terminate without fabricated scores and clean resources. Action/step transitions are instance/attempt/prior-observation-bound, idempotent, ordered, and permission-limited. Current gate: `ANG-GATE-ENV-PROTOCOL-001`; design is approved for CR0, while delivery remains blocked by release predecessors. No executable environment or model is authorized.

Next: build the four v1 schemas (`TaskSpec`, `Observation`, `Action`, and `Feedback`) and synthetic lifecycle/negative-control fixtures in the named release leaf.
