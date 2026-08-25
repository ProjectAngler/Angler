---
blueprint_id: ANG-BP-ENVIRONMENT-PROTOCOL
title: Environment protocol
parent_id: ANG-BP-WORLDS
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: blocked_by_cr0_predecessors
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-BP-EVIDENCE-SCHEMAS
  - ANG-BP-PARTITIONS
contracts_in:
  - ANG-CTR-EXECUTION-PLAN-001@1
contracts_out:
  - ANG-CTR-TASK-SPEC-001@1
  - ANG-CTR-OBSERVATION-001@1
  - ANG-CTR-ACTION-001@1.0.0
  - ANG-CTR-FEEDBACK-001@1
gates:
  - ANG-GATE-ENV-PROTOCOL-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Environment protocol

## Outcome

Provide implementation-independent lifecycle contracts for `reset`, `step`, `score`, `close`, and replay. A run begins from an immutable TaskSpec and authorized seed commitment, emits learner-visible Observations for accepted Actions, terminates on an explicit condition/budget/error, and emits externally computed Feedback. Every artifact uses the shared evidence envelope.

## State and authority

WORLDS owns environment state and verifier code. RUNTIME may submit actions but cannot alter task identity, budgets, verifier, visibility, or hidden payloads. SCIENCE owns partition assignment and what results support a claim. EVIDENCE persists records. `reset` is deterministic for an exact environment build, TaskSpec, and resolved seed inside declared tolerances; concurrent instance state never aliases.

## Failure semantics

Typed failures are `INVALID_SPEC`, `UNAUTHORIZED_VISIBILITY`, `INVALID_ACTION`, `BUDGET_EXCEEDED`, `RESOURCE_UNAVAILABLE`, `VERIFIER_FAILURE`, `NONDETERMINISM`, and `INTERNAL_ERROR`. Failure terminates the instance, emits no fabricated score, cleans bounded resources, and records evidence. Retry requires an idempotency key or a new attempt identity as specified.

## Gate and next leaf

`ANG-GATE-ENV-PROTOCOL-001` requires schema round trips; deterministic reset/replay; budget, visibility, invalid-action, timeout, cleanup, and verifier-failure controls; and proof that feedback lacks a solution trace or sealed answer. CR0 constructs schemas and synthetic fixtures only under its leaf.
