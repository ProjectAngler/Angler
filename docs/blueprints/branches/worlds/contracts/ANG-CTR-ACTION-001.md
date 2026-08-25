---
contract_id: ANG-CTR-ACTION-001
version: 1.0.0
status: approved_for_cr0_design
owner: ANG-BP-WORLDS
producer: ANG-BP-RUNTIME
consumers: [ANG-BP-WORLDS, ANG-BP-EVIDENCE]
---

# Action and step-request contract v1

## Input and output

RUNTIME emits an immutable Action envelope containing evidence identity, TaskSpec/environment/instance/attempt identity, expected prior Observation identity, monotonic step index, action-schema/version, action payload, requested permitted tool (if any), idempotency key, creation/deadline, and learner-visible provenance. WORLDS consumes it and returns exactly one next Observation, terminal Observation plus separately computed Feedback, or typed failure record.

The action payload must validate against the TaskSpec-declared action schema and cannot request a permission, tool, resource, data visibility, or budget beyond that spec/ExecutionPlan/impact authorization.

## Behavior and mutation

An accepted action may mutate only the named environment instance. It cannot alter TaskSpec, verifier, partition, visibility, budgets, another instance, runtime/model state, or evidence history. The tuple `(instance, attempt, step, idempotency_key, prior_observation)` admits at most one transition; an exact duplicate returns the recorded result without a second mutation. Concurrent or out-of-order actions fail closed.

## Failure, timeout, and retry

Typed failures are `INVALID_ACTION`, `SCHEMA_UNSUPPORTED`, `WRONG_INSTANCE`, `WRONG_ATTEMPT`, `STALE_OBSERVATION`, `NONMONOTONIC_STEP`, `DUPLICATE_CONFLICT`, `PERMISSION_DENIED`, `BUDGET_EXCEEDED`, `DEADLINE_EXCEEDED`, `ENVIRONMENT_TERMINATED`, and `TRANSITION_FAILURE`. Invalid/permission/budget/order failures are not retryable without a corrected action/new identity. A declared transient transition failure may retry only when no mutation committed and the environment proves cleanup/idempotency.

## Compatibility and validation

Unknown major versions reject. Minor additions cannot change mutation, idempotency, permission, step-order, or budget semantics. TaskSpec pins compatible versions; any material action-schema change creates a successor TaskSpec.

Contract tests cover valid step, schema/type/range errors, wrong instance/attempt/prior observation, replayed exact duplicate, conflicting duplicate, out-of-order/concurrent steps, permission/tool/budget expansion, deadline, terminal-state action, partial failure/cleanup, unknown major, and evidence-envelope identity. CR0 uses only synthetic records and performs no environment/model execution.
