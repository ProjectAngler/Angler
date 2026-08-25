---
adr_id: ANG-ADR-0004
title: Construction Release 0 shared-interface stabilization
status: accepted
owner: ANG-BP-ROOT
accepted_at: 2026-08-25
supersedes: none
superseded_by: none
---

# Construction Release 0 shared-interface stabilization

## Context

The conceptual registry was sufficient for architecture drafting but not for build leaves. CR0 needs explicit resource, environment-step, and scientific-evaluation semantics before schemas can claim conformance. These are cross-branch boundaries and therefore require a project ADR and consumer-impact review rather than an unrecorded registry edit.

## Decision

Accept these implementation-independent v1 designs:

- `ANG-CTR-RESOURCE-INVENTORY-001@1` and `ANG-CTR-EXECUTION-PLAN-001@1`;
- `ANG-CTR-TASK-SPEC-001@1`, `ANG-CTR-OBSERVATION-001@1`, `ANG-CTR-ACTION-001@1.0.0`, and `ANG-CTR-FEEDBACK-001@1`;
- `ANG-CTR-EVALUATION-SUITE-001@1.0.0`, `ANG-CTR-EVALUATION-RECEIPT-001@1.0.0`, and `ANG-CTR-PROMOTION-DECISION-001@1.0.0`.

`ANG-CTR-ACTION-001` is owned by WORLDS because it defines the accepted environment input and transition boundary; RUNTIME is its producer. An action is bound to TaskSpec, environment instance, attempt, prior Observation, step, permissions, budget, deadline, and idempotency key. It may mutate only that environment instance and returns one Observation/terminal result or typed failure.

Every accepted contract explicitly defines input/output, behavior/mutation, typed error, retryability, idempotency, timeout/limits, major/minor compatibility, migration/new-identity rules, producer tests, and consumer contract tests. Scientific thresholds and hidden payloads remain precommitted/sealed. Promotion inputs are conjunctive; performance cannot waive authorization, safety, evidence, lineage, or rollback.

## Consumer-impact review

| Consumer | Impact and disposition |
|---|---|
| EVIDENCE | Envelope/manifest schemas must represent these contract/version references; first schema leaf is compatible and does not implement payload semantics. |
| RUNTIME | Must later emit Action and consume plans/suites/decisions; branch remains draft/not-ready, so no implementation migration exists. Its future design gate must accept v1 contract tests. |
| LEARNING | Must later consume Feedback/Episode and plans; branch remains draft/not-ready and gains no CR0 authority. |
| SAFETY | Impact assessment/binding remains independent and conjunctive; no weakening or new authority. |
| SCIENCE | Owns suite/receipt/decision and accepts exact resource/environment identities; delivery remains predecessor-blocked. |
| RESOURCES | Owns inventory/plan; real collection/probes remain unauthorized. |
| WORLDS | Owns task/observation/action/feedback accepted semantics; executable environments remain unauthorized. |
| TOOLS | No CR0 use; later tool requests inside Action require separate tool contracts/permissions. |

There are no implemented consumers or accepted run artifacts to migrate. No branch is marked `needs_revision`; instead, every future implementation leaf must pin these versions and pass its producer/consumer tests. A consumer objection or material semantic change requires a successor contract/ADR before its leaf becomes ready.

## Non-equivalence

This ADR approves contract designs for bounded construction. It does not pass a contract delivery gate, branch gate, human-flourishing gate, Slice 00, M0, or scientific claim, and it grants no model/GPU/network/dependency/external authority.

## Alternatives rejected

- Infer actions from raw model text: loses instance, permission, ordering, budget, and idempotency guarantees.
- Leave error/retry/timeout behavior to implementations: creates incompatible consumers and unsafe silent retries.
- Defer evaluation receipt/promotion semantics until after results: permits post-result policy drift.
- Encode reasoning procedures in environment/verifier scripts: violates the outcome-judge invariant.

## Rollback

Before executable evidence exists, rollback retires these contract versions, returns registry rows to conceptual, blocks dependent leaves, and issues successor identities. Preserve this ADR and every review; never reinterpret artifacts created under one version as another.
