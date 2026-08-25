---
contract_id: ANG-CTR-TASK-SPEC-001
version: 1
status: approved_design
owner: ANG-BP-WORLDS
producer: environment package admitted by WORLDS
consumers: [ANG-BP-RUNTIME, ANG-BP-SCIENCE, ANG-BP-EVIDENCE]
---

# TaskSpec v1

An immutable TaskSpec identifies `task_spec_id`, family and semantic version, generator/build identity, instance identity, partition-manifest reference, seed commitment (never an exposed sealed seed), observation/action schema references, termination rules, verifier identity, feedback policy, visibility map, allowed tools, permissions, token/step/time/resource/output budgets, determinism/tolerance policy, impact-authorization reference, and shared evidence envelope.

Preconditions: referenced contracts/identities exist, permissions do not exceed the execution plan or impact authorization, and caller may resolve the named visibility class. Guarantees: fields do not mutate during an attempt; any material change creates a new content identity.

Failures are typed `INVALID_SPEC`, `UNSUPPORTED_VERSION`, `IDENTITY_MISMATCH`, `UNAUTHORIZED_VISIBILITY`, or `INFEASIBLE_BUDGET` and occur before stateful execution. Validation rejects absent required fields, unknown major versions, reconstructable hidden seeds in learner-visible forms, and permission/budget expansion. Producer and every consumer run the same conformance fixtures.

## Operation, persistence, retry, and rollback

Validation is deterministic and idempotent for the exact envelope bytes/identity. A valid TaskSpec is persisted as an immutable EVIDENCE artifact before reset; a filename, in-memory object, or mutable environment field is never authoritative. The spec declares reset/step/attempt/total timeouts, maximum observations/actions/output size, resource ceilings, cleanup deadline, and whether a transient transport retry is permitted. Validation itself cannot resolve hidden payloads outside the caller's role.

An exact duplicate is the same artifact. A transport/read failure may retry only with the same identity and authorization. Schema, visibility, permission, budget, seed-exposure, or compatibility failure is not retryable under the same spec; correction creates a successor TaskSpec. Because failure occurs before environment mutation, rollback means no reset/instance creation; if a consumer created partial state contrary to the precondition, it must terminate/clean it and record `INTERNAL_ERROR`.

Unknown major versions reject. Minor versions may add optional non-semantic fields only and consumers preserve them; any change to action/observation schema, verifier, feedback policy, visibility, partition, seed commitment, tools/permissions, budgets, termination, tolerance, or impact reference creates a successor major/minor identity as declared and requires consumer revalidation.

Producer/consumer tests cover valid round trip, every required/unknown field, unsupported major/additive minor, identity mutation, duplicate-key/ambiguous canonical form, hidden-seed leakage, wrong visibility, permission/tool/budget expansion, infeasible limits, idempotent duplicate validation, transport retry, no-state-on-failure, and immutable persistence/reconstruction.
