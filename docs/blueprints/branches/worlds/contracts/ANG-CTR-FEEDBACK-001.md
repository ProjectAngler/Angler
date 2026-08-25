---
contract_id: ANG-CTR-FEEDBACK-001
version: 1
status: approved_design
owner: ANG-BP-WORLDS
producer: independently identified outcome verifier
consumers: [ANG-BP-LEARNING, ANG-BP-EVIDENCE, ANG-BP-SCIENCE]
---

# Feedback v1

Feedback binds TaskSpec, attempt, terminal trajectory, verifier/build, partition, and evidence identities. It contains disposition (`VALID_RESULT`, `INVALID_ATTEMPT`, or `VERIFIER_ERROR`), externally observable metric values with units/ranges, permitted failing cases or constraint violations, uncertainty/tolerance, verifier resource use, learner-visibility class, and integrity/provenance fields.

Feedback describes outcomes, not the preferred reasoning procedure. It excludes hidden transfer answers, raw seeds, target chain of thought, evaluator secrets, and information disallowed by the TaskSpec feedback policy. It is immutable and idempotent for one trajectory/verifier identity.

Schema/range/provenance/visibility failures reject the record. `VERIFIER_ERROR` never becomes a zero score or successful result. Corrections create linked successor records and invalidate dependent evaluation receipts under their policy.

## Failure, operation, compatibility, and tests

Typed failures are `SCHEMA_UNSUPPORTED`, `IDENTITY_MISMATCH`, `TRAJECTORY_INCOMPLETE`, `VERIFIER_MISMATCH`, `VERIFIER_TIMEOUT`, `VERIFIER_ERROR`, `METRIC_INVALID`, `VISIBILITY_DENIED`, `HIDDEN_DATA_LEAK`, and `INTEGRITY_FAILURE`. TaskSpec fixes verifier timeout, output-size/metric ranges, permitted projections, and retry policy. Feedback computation is deterministic/idempotent for the exact trajectory/verifier/build when declared; otherwise its seeded tolerance/attempt identity is explicit.

A transient verifier infrastructure failure may retry only if TaskSpec predeclares it, the trajectory/verifier identities remain frozen, no hidden payload was exposed, and a new verifier-attempt identity is recorded. Schema, visibility, identity, hidden-data, or integrity failure is not retryable under the same feedback identity. Rollback means no learner eligibility or evaluation use, quarantine the record, preserve the trajectory and failure evidence, and recompute only under an authorized successor attempt/record.

Unknown major versions reject. Minor additions cannot alter metric values/ranges, disposition, verifier identity, visibility, retry, eligibility, uncertainty, or task/trajectory bindings. Corrections are linked successor records; originals and dependent invalidation edges persist.

Producer/consumer tests cover valid/invalid-attempt/verifier-error dispositions, metric range/unit, wrong task/trajectory/verifier/partition, incomplete trajectory, timeout/retry/idempotency, zero-score versus verifier error, hidden answer/seed/reasoning-trace leakage, unauthorized projection, unknown-major/additive-minor behavior, correction/invalidation, and deterministic evidence reconstruction.
