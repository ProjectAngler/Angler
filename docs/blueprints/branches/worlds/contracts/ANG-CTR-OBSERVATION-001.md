---
contract_id: ANG-CTR-OBSERVATION-001
version: 1
status: approved_design
owner: ANG-BP-WORLDS
producer: environment instance
consumers: [ANG-BP-RUNTIME, ANG-BP-EVIDENCE]
---

# Observation v1

An Observation carries content identity, TaskSpec/attempt/instance identity, monotonic step index, learner-visible payload and payload schema, visibility/trust classification, provenance, remaining public budgets, terminal flag/reason when public, and shared evidence envelope.

It never carries hidden answers, raw sealed seeds, evaluator-only annotations, privileged verifier state, or a prescribed reasoning trace. Same inputs and state produce the same observation within the declared tolerance. Duplicate delivery is idempotent by observation identity.

Consumers reject wrong-attempt, nonmonotonic, over-budget, unauthorized-visibility, malformed, or identity-mismatched observations. A rejected observation terminates/quarantines the attempt and is recorded; it is not silently repaired.

## Failure, operation, compatibility, and tests

Typed failures are `SCHEMA_UNSUPPORTED`, `IDENTITY_MISMATCH`, `WRONG_INSTANCE`, `WRONG_ATTEMPT`, `NONMONOTONIC_STEP`, `VISIBILITY_DENIED`, `PAYLOAD_INVALID`, `BUDGET_INVALID`, and `OBSERVATION_TOO_LARGE`. The TaskSpec fixes observation payload/total size, step/deadline, and count limits. Production and validation are bounded by the attempt timeout; timeout terminates the attempt and cannot fabricate an Observation.

The Observation is persisted through Trajectory/EVIDENCE before it can support Feedback or evaluation. Exact duplicate delivery is idempotent. A transport retry may redeliver the exact identity; wrong/malformed/unauthorized observations are not repairable under that identity. Rollback terminates/quarantines the environment attempt, preserves the last valid observation/evidence, and never rewrites it.

Unknown major versions reject. Minor additions preserve unknown optional fields and cannot alter payload meaning, visibility, provenance, budgets, terminal semantics, instance/attempt/step identity, or tolerance. Material change requires a successor TaskSpec/Observation identity.

Producer/consumer tests cover valid/terminal observations, exact duplicate delivery, wrong instance/attempt/step, skipped/reordered/concurrent steps, identity/payload mutation, unauthorized visibility and constant-shape denial, hidden metadata leakage, invalid/over-limit payload, budget inconsistency, timeout, unknown-major/additive-minor behavior, persistence, and rollback to the last valid attempt state.
