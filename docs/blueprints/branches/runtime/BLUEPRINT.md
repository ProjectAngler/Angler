---
blueprint_id: ANG-BP-RUNTIME
title: Runtime and neural state
parent_id: ANG-BP-ROOT
revision: 2
tier: 1
design_status: draft
delivery_status: not_started
accountable_owner: unassigned
execution_owner: unassigned
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-MODEL-BOUNDARY
  - ANG-BP-AGENT-RUNTIME
  - ANG-BP-PLASTIC-STATE
  - ANG-BP-STATE-LINEAGE
  - ANG-BP-LEARNING-TRANSACTION
  - ANG-BP-RUNTIME-COMPATIBILITY
depends_on:
  - ANG-BP-RESOURCES
  - ANG-BP-EVIDENCE
  - ANG-BP-SAFETY
contracts_in:
  - ANG-CTR-EXECUTION-PLAN-001
  - ANG-CTR-TASK-SPEC-001
  - ANG-CTR-UPDATE-PROPOSAL-001
  - ANG-CTR-EVALUATION-RECEIPT-001
  - ANG-CTR-PROMOTION-DECISION-001
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out:
  - ANG-CTR-TRAJECTORY-001
  - ANG-CTR-PLASTIC-STATE-001
  - ANG-CTR-TRANSACTION-RECEIPT-001
gate: ANG-GATE-RUNTIME-DESIGN-001
---

# Runtime and neural state

## Context capsule

This branch executes a stable foundation model with exactly one uniformly active plastic competence state. It owns model and adapter compatibility, action execution, state snapshots, atomic learning transactions, and mechanical enforcement that no candidate promotes without valid external scientific and human-flourishing authorization. It does not author either decision or compute candidate updates.

## Contribution to the root

The central causal claim requires a physically intervenable state whose parentage and influence are exact. RUNTIME makes `zero`, `swap`, `restore`, and replay meaningful and prevents query-conditioned state routing.

## Inherited invariants

Applies: `ANG-INV-STABLE-BASE-001`, `ANG-INV-ONE-COMPETENCE-001`, `ANG-INV-REVERSIBLE-UPDATES-001`, `ANG-INV-ELASTIC-COMPUTE-001`, and `ANG-INV-EXTERNAL-AUTHORITY-001`.

## Scope

- Load an immutable foundation model/tokenizer under a validated execution plan.
- Apply one active PEFT/LoRA state uniformly to learner invocations.
- Produce complete observable trajectories and expose bounded training losses/activations.
- Create, content-address, save, load, zero, swap, transfer, and restore compatible states.
- Maintain workspace, candidate, promoted, rejected, and rollback state identities.
- Coordinate atomic propose → evaluate → promote or restore transactions.
- Authenticate the exact artifact/scope human-impact `ALLOW` immediately before promotion; fail closed on every other state.
- Validate precision, kernel, backend, state, and model-plan compatibility.

## Explicit non-goals

- Choosing update gradients, replay samples, benchmarks, thresholds, or curriculum.
- Selecting a model or adapter based on the current query.
- Mutating foundation weights in early milestones.
- Treating backend-specific behavior as part of reasoning semantics.

## Contracts and state machine

RUNTIME consumes an `ExecutionPlan`, `TaskSpec`, `UpdateProposal`, `EvaluationReceipt`, and `PromotionDecision`; it emits a `Trajectory`, `PlasticStateRef`, and `TransactionReceipt`.

Promotion additionally consumes `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001`. This protected authorization cannot be supplied, written, or selected by the candidate or learner.

```text
PROMOTED(parent)
  → WORKSPACE
  → CANDIDATE(parent-bound, immutable)
      ├── science approved + valid human-impact ALLOW + authority valid
      │              → PROMOTED(new head)
      └── rejected/error → REJECTED + exact restore(parent)
```

Only the transaction coordinator may advance the promoted lineage. Evaluation receives isolated state references and cannot mutate them.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-MODEL-BOUNDARY` | Immutable model signature, tokenizer, generation, loss, and hidden-state interface | `ANG-GATE-MODEL-BOUNDARY-001` | stub |
| `ANG-BP-AGENT-RUNTIME` | Observation-to-action execution with trajectory and tool boundary | `ANG-GATE-AGENT-RUNTIME-001` | stub |
| `ANG-BP-PLASTIC-STATE` | LoRA topology, serialization, compatibility, hashing, and interventions | `ANG-GATE-PLASTIC-STATE-001` | stub |
| `ANG-BP-STATE-LINEAGE` | Workspace/candidate/promoted/rejected lineage and checkpoint store | `ANG-GATE-STATE-LINEAGE-001` | stub |
| `ANG-BP-LEARNING-TRANSACTION` | Atomic candidate evaluation and promote-or-restore behavior | `ANG-GATE-TRANSACTION-001` | stub |
| `ANG-BP-RUNTIME-COMPATIBILITY` | Backend/precision/kernel/model-plan validation | `ANG-GATE-RUNTIME-COMPAT-001` | stub |

## Dependencies and sequencing

RESOURCES produces the plan; SAFETY supplies runtime permissions and authority; EVIDENCE registers state and transaction identities. `MODEL-BOUNDARY`, `PLASTIC-STATE`, and `RUNTIME-COMPATIBILITY` stabilize first. `STATE-LINEAGE` and `AGENT-RUNTIME` follow. `LEARNING-TRANSACTION` integrates them with LEARNING and SCIENCE.

## Acceptance gate and evidence

`ANG-GATE-RUNTIME-DESIGN-001` passes when each child has unambiguous state ownership, public contracts, failure semantics, resource behavior, and slice 02–04 tests. The design must make it impossible for an ordinary runtime call to pick a model or adapter from query identity.

M2 evidence later requires immutable base hashes; deterministic state serialization; successful save/zero/swap/transfer/restore/replay; exact parent restoration after failure; no partial promotion; and mechanical rejection of every missing, denied, escalated, expired, tampered, or mismatched human-impact authorization.

## Testing and validation

- Model/tokenizer and adapter compatibility contracts.
- State serialization/hash round trips.
- State-zero, cross-runtime swap, and incompatible-state rejection.
- Transaction failure injection at every boundary.
- Human-impact authorization matrix: `ALLOW` for exact scope succeeds; all missing/non-allow/expired/tampered/mismatched cases reject and restore.
- Shutdown and rollback succeed even when they destroy candidate or promoted competence.
- Base-weight immutability checks.
- Backend equivalence within declared numeric tolerances.
- Resource-plan headroom and safe-replan integration tests.

## Risks and rollback

- `ANG-RISK-STATE-ROUTING-001`: query-conditioned state selection. Reject architecture.
- `ANG-RISK-PARTIAL-PROMOTION-001`: evidence/state lineage diverge. Fail closed and restore parent.
- `ANG-RISK-NUMERIC-DRIFT-001`: backend changes alter behavior beyond tolerance. New plan/evaluation identity.
- `ANG-RISK-BASE-MUTATION-001`: foundation parameters change. Invalidate transaction and reload signed base.
- `ANG-RISK-AUTHORIZATION-BYPASS-001`: a candidate promotes without valid human-impact authorization. Stop runtime, invalidate lineage, preserve evidence, and restore the last authorized head.

## Resource scaling

Logical runtime contracts remain stable across constrained, workstation, server, and cluster plans. A state is bound to its model and adapter topology. Topology changes require explicit competence migration and full regating; compatible placement changes may reload the same state at transaction boundaries.

## Current status and next leaf

Design revision 2 is draft. First expand `ANG-BP-MODEL-BOUNDARY`, `ANG-BP-PLASTIC-STATE`, and `ANG-BP-RUNTIME-COMPATIBILITY` as one substrate package for slices 02–03, with the human-impact authorization check included in the transaction contract before it becomes ready.
