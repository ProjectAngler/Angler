---
blueprint_id: ANG-BP-EVIDENCE
title: Evidence and reproducibility
parent_id: ANG-BP-ROOT
revision: 3
tier: 1
design_status: approved_for_cr0
delivery_status: ready
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: human_directed_leaf_operator
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-EVIDENCE-SCHEMAS
  - ANG-BP-EVENT-STORE
  - ANG-BP-ARTIFACT-LINEAGE
  - ANG-BP-EXPERIMENT-RUNNER
  - ANG-BP-REPLAY-RECOVERY
  - ANG-BP-OBSERVABILITY
depends_on:
  - ANG-BP-SAFETY
contracts_in:
  - all registered lifecycle artifacts
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out:
  - ANG-CTR-EVIDENCE-ENVELOPE-001
  - ANG-CTR-EPISODE-001
  - ANG-CTR-EXPERIMENT-MANIFEST-001
  - ANG-CTR-ARTIFACT-LINEAGE-001
  - ANG-CTR-AUTHORIZATION-BINDING-001
gate: ANG-GATE-EVIDENCE-DESIGN-001
---

# Evidence and reproducibility

## Context capsule

EVIDENCE makes every experience, resource choice, update, rejection, promotion, migration, authorization, and scientific claim attributable and replayable. It owns immutable evidence envelopes, visibility semantics, content-addressed lineage, experiment manifests, recovery, and cross-cutting metrics. It records decisions made by their owning authorities; it does not make scientific, resource, or safety decisions.

## Contribution to the root

An adaptive system changes over time. Without stable identity, parentage, visibility, and correction rules, neither rollback nor causal attribution is credible. This branch supplies the evidence substrate that lets every branch exchange artifacts without relying on conversation history or mutable filenames.

## Inherited invariants

All root invariants apply. This branch strengthens:

- `ANG-INV-EVIDENCE-SEPARATION-001`: episode evidence and competence state remain distinct artifacts and visibility domains.
- `ANG-INV-REVERSIBLE-UPDATES-001`: every candidate and decision identifies an exact parent and rollback target.
- `ANG-INV-EXTERNAL-AUTHORITY-001`: learner-visible processes cannot write, select, replace, or validate authorization evidence.
- `ANG-INV-HUMAN-FLOURISHING-001`: evidence may not conceal affected people, material risk, dissent, revocation, or failed outcomes.

Construction is bounded by `ANG-ADR-0002` and its Release-0 ceiling. No design artifact here authorizes model execution, learning, network use, real-person data, external dependencies, or deployment.

## Scope

- Define the canonical evidence envelope, identity material, canonicalization profile, and typed references shared across lifecycle artifacts.
- Define visibility classes and consumer projections without making hidden payloads discoverable through names, hashes, errors, or metadata.
- Specify `Episode` and `ExperimentManifest` contracts.
- Specify content addressing, typed parentage, corrections, versions, invalidations, and reconstructable lineage.
- Bind human-impact assessments to exact subjects without an artifact/authorization hash cycle.
- Preserve `ALLOW`, `DENY`, `ESCALATE`, expiry, revocation, conditions, and authority identity without rewriting history.
- Later provide append-only storage, experiment execution records, replay/recovery, and observability through bounded children.

## Explicit non-goals

- Choosing architecture, promotion thresholds, safety policy, resource plans, or moral outcomes.
- Treating an `ALLOW` as scientific approval or an evaluation pass as safety approval.
- Storing credentials, secret keys, hidden answers, or recovered/private material in ordinary evidence payloads.
- Building a mutable central database as the sole source of truth; indexes are rebuildable projections.
- Claiming bitwise reproducibility where declared backend tolerances are the honest guarantee.
- Implementing code or claiming executable evidence in this design revision.

## Inputs, outputs, and contracts

### Inputs

- All registered lifecycle contracts and their producer-owned semantics.
- `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1` and `ANG-GATE-HUMAN-FLOURISHING-001@1`.
- `ANG-ADR-0002`, which authorizes only low-impact local construction through ready leaves.

### Outputs

- `ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0`: immutable identity and visibility wrapper.
- `ANG-CTR-EPISODE-001@1.0.0`: immutable task/trajectory/feedback join.
- `ANG-CTR-EXPERIMENT-MANIFEST-001@1.0.0`: precommitted run identity and intended claims.
- `ANG-CTR-ARTIFACT-LINEAGE-001@1.0.0`: typed graph, correction, and compatibility semantics.
- `ANG-CTR-AUTHORIZATION-BINDING-001@1.0.0`: post-identity binding between an exact artifact and impact authorization.

Cross-branch producers own payload semantics. EVIDENCE owns only the shared envelope, persistence identity, visibility-enforcement requirements, and relationship semantics.

## Internal design and state ownership

### Two identities, one immutable record

Every admissible artifact has:

1. a `content_id` or protected `payload_commitment` over canonical payload bytes; and
2. an `artifact_id` over canonical identity material containing contract/version, artifact type, payload commitment, producer/run identity, typed parents, visibility-policy identity, applicable assessment reference or requirement, and contract-declared semantic bindings.

Storage locations, access times, cache state, signatures, and mutable projections are excluded from identity. Signatures and attestations bind the resulting `artifact_id`. Committed payloads and envelopes never change; a changed byte or semantic field creates a new identity.

### Visibility classes

| Class | Payload resolvers | Learner behavior | Release-0 rule |
|---|---|---|---|
| `LEARNER_VISIBLE` | Authorized runtime and learner-facing components | May resolve only the declared projection | Synthetic data only |
| `CONTROL_PLANE` | Authorized non-learner services and humans | May receive a non-sensitive existence/status projection | Default for construction evidence |
| `SEALED_EVALUATION` | Evaluation authority and evidence custodian | Cannot resolve before an explicit release transition | Synthetic fixtures may test denial only |
| `HUMAN_AUTHORITY` | Named human authority, safety validator, and evidence custodian | Cannot read raw payload or write any form | Assessments, approvals, revocations, and credential references |
| `RESTRICTED_PERSONAL` | Explicitly authorized privacy-controlled consumers | No learner access without separately approved purpose | Prohibited in Release 0 |

Visibility is an access policy, not a sensitivity score. A consumer must be explicitly allowed; no class implies write permission. Combining artifacts takes the most restrictive applicable policy unless an independently authorized release creates a new derived artifact. Evidence stores credential references, never credential material.

Low-entropy sealed payloads may not expose a raw unsalted digest. They use an authority-held keyed commitment or a digest of authenticated ciphertext so an identifier cannot become an answer oracle.

### Parentage and correction

Typed edges point only to already committed identities. Derivation, version, correction, promotion, migration, and rollback edges are acyclic. Corrections are new artifacts with `CORRECTS` and, where appropriate, `SUPERSEDES` edges; the original remains discoverable and is never silently rewritten. Revocation and invalidation are new signed records, not mutable flags.

Unknown major contract versions fail closed. Minor additions are accepted only if mandatory semantics remain intact and unknown optional fields are preserved. A semantic change requires a new contract version and consumer revalidation.

### Authorization without a hash cycle

Authorization uses a staged subject tuple:

1. Canonical payload bytes establish the final `content_id` or protected commitment, plus proposed artifact type/version, parentage, scope, resource plan, and permissions.
2. SAFETY assesses and immutably binds that exact tuple; the assessment never depends on the future final `artifact_id`.
3. The final artifact envelope records the applicable assessment reference or explicit authorization requirement and receives its `artifact_id`.
4. A separately content-addressed authorization/promotion binding references both the final artifact envelope and the assessment, revalidates their subject tuple, and records gate/authority conditions.

This permits the artifact to record its assessment while avoiding an artifact↔assessment hash cycle. A current authentic `ALLOW` binding is necessary for eligible operations, but remains a separate authority artifact. `DENY`, `ESCALATE`, expired, revoked, wrong-subject, wrong-parent, wrong-plan, or condition-violating records remain preserved and are never eligible.

### Authoritative versus derived state

Immutable artifacts, relationship records, and signed authority records are authoritative. Search indexes, active-version pointers, lineage visualizations, and “current authorization” views are disposable projections reproducible from authoritative records.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-EVIDENCE-SCHEMAS` | Canonical envelope, visibility, `Episode`, and `ExperimentManifest` contracts | normal `ANG-GATE-EVIDENCE-SCHEMAS-001`; CR0-only `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` | design approved; first CR0 scaffold leaf ready; normal gate not run |
| `ANG-BP-EVENT-STORE` | Append-only, correction-preserving local record | `ANG-GATE-EVENT-STORE-001` | bounded stub; Slice 01 |
| `ANG-BP-ARTIFACT-LINEAGE` | Content addressing, typed parentage, correction, and authorization binding | `ANG-GATE-ARTIFACT-LINEAGE-001` | design approved; delivery blocked by schema gate |
| `ANG-BP-EXPERIMENT-RUNNER` | Frozen manifests, seeds, partitions, tolerances, and results | `ANG-GATE-EXPERIMENT-RUNNER-001` | bounded stub; Slice 01 |
| `ANG-BP-REPLAY-RECOVERY` | State reconstruction and interrupted-transaction recovery | `ANG-GATE-REPLAY-RECOVERY-001` | bounded stub; Slice 03 |
| `ANG-BP-OBSERVABILITY` | Cross-branch metrics with bounded cost and trust labels | `ANG-GATE-OBSERVABILITY-001` | bounded stub; incremental |

## Dependencies and sequencing

1. Approve canonical envelope/visibility and lineage/authorization contracts together; neither is safe in isolation.
2. Activate `ANG-WORK-EVIDENCE-SCHEMAS-001` under a current Release-0 work authorization.
3. After its contract tests pass, activate `ANG-WORK-ARTIFACT-LINEAGE-001`.
4. Require cross-branch producer/consumer review before changing registry rows from conceptual to detailed draft.
5. Expand EVENT-STORE and EXPERIMENT-RUNNER for Slice 01; expand replay only after runtime state identity exists.

## Acceptance gate and required evidence

`ANG-GATE-EVIDENCE-DESIGN-001` is the Tier-1 design gate. It may approve this design for bounded construction only when the two concrete child specifications, five contracts, visibility/identity decision, consumer-impact list, work-leaf boundaries, and failure semantics are independently reviewed. It does not pass the child executable gates or M0.

Every executable leaf still requires a proportionate impact assessment and `ANG-GATE-HUMAN-FLOURISHING-001` decision. Release-0 authorization cannot be inherited after any scope escalation.

## Testing and validation

- Canonicalization stability across key ordering, whitespace, timestamp formats, and duplicate-key rejection.
- Content and artifact identity round trips; any identity-material mutation changes the correct identity.
- Visibility-matrix denials, metadata-leak checks, and low-entropy sealed commitment tests.
- Unknown-major rejection and additive-minor compatibility.
- Parent existence, type, cardinality, cycle, and wrong-parent tests.
- Correction, supersession, invalidation, and revocation preserve original history.
- Human-impact binding rejects missing, wrong-subject, wrong-parent, wrong-plan, expired, revoked, tampered, `DENY`, and `ESCALATE` cases.
- Episode partition/learning-eligibility rules prevent held-out or sealed evidence from becoming learning input.
- Experiment thresholds and partitions cannot change under the same manifest/evaluation identity.
- Reconstruction does not depend on storage paths, caches, or conversation history.

## Risks, failure behavior, and rollback

- `ANG-RISK-EVIDENCE-LEAKAGE-001`: hidden data enters visible metadata or identifiers. Quarantine affected artifacts, invalidate dependent runs, and rotate sealed material.
- `ANG-RISK-LINEAGE-GAP-001`: an admissible artifact has a missing or incompatible parent. Reject or quarantine it; never infer parentage from filenames.
- `ANG-RISK-SCHEMA-DRIFT-001`: producers and consumers silently diverge. Reject unknown major versions and mark affected consumers `needs_revision`.
- `ANG-RISK-IMPACT-RECEIPT-SUBSTITUTION-001`: an authorization is reused for a different subject or scope. Reject the binding and all dependent promotion/use.
- `ANG-RISK-HASH-ORACLE-001`: a sealed low-entropy answer is recoverable from a digest. Use a protected commitment and invalidate exposed evaluation identities.
- `ANG-RISK-AUTHORIZATION-CYCLE-001`: assessment and artifact identities recursively depend on one another. Enforce the staged content/assessment/envelope/binding sequence.
- `ANG-RISK-EVIDENCE-OVERHEAD-001`: recording dominates work. Reduce payload duplication or stream payloads; never remove mandatory identity or authority fields.

Before implementation, each leaf records an exact baseline. Release-wide rollback is the archive identified by `ANG-ADR-0002`; leaf rollback restores only declared files and preserves failed evidence.

## Resource profiles and scaling

Identity and visibility semantics are invariant across resource tiers. Constrained runs may serialize writers and store payloads locally; clusters may partition payload storage but must preserve one logical lineage, stable commitments, consistent authorization checks, and rebuildable projections. Resource-plan identity is mandatory evidence, not optional telemetry.

## Decisions and active ADRs

- `ANG-ADR-0002`: accepted Release-0 construction boundary.
- `ANG-ADR-EVIDENCE-0001`: originating proposal accepted project-wide as `ANG-ADR-0003`.
- `ANG-ADR-0003`: accepted canonical identity, visibility, protected-commitment, and sidecar authorization-binding semantics.

## Current status and blockers

Revision 3 passed independent CR0 design review and is registered under `ANG-ADR-0003`. EVIDENCE-SCHEMAS has one exact LOW bootstrap scaffold leaf ready; its non-equivalent scaffold acceptance may support only later CR0 scaffold leaves. ARTIFACT-LINEAGE remains blocked until the normal `ANG-GATE-EVIDENCE-SCHEMAS-001` passes under ordinary prerequisites. No executable child gate, Slice-00, human-flourishing, or M0 claim exists.

## Parent roll-up and next executable leaf

EVIDENCE has approved CR0 designs for schemas and lineage. The exact first executable leaf is `ANG-WORK-EVIDENCE-SCHEMAS-001`, now ready under ADR-0002, the local-scaffold policy, the LOW bootstrap assessment, and baseline `ANG-BASELINE-EVIDENCE-SCHEMAS-001`. Execute no other EVIDENCE leaf first.
