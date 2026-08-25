---
adr_id: ANG-ADR-EVIDENCE-0001
title: Canonical evidence identity, visibility, and authorization binding
status: accepted_as_ANG-ADR-0003
owner: ANG-BP-EVIDENCE
proposed_at: 2026-08-25
accepted_at: 2026-08-25
global_decision: ANG-ADR-0003
affects:
  - all persistent and evaluative contracts
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
  - ANG-CTR-PROMOTION-DECISION-001
---

# Canonical evidence identity, visibility, and authorization binding

## Context

Cross-branch artifacts need one identity and visibility language. Making an assessment bind a final artifact identity that itself embeds the assessment would create a hash cycle. Publishing raw hashes for small sealed answer spaces could also leak answers by enumeration.

## Proposed decision

1. Use an immutable shared evidence envelope with separate `content_id`/protected commitment and `artifact_id`.
2. Compute identity from versioned canonical semantic material, never storage location, cache state, mutable status, future signature, or a circular future authorization.
3. Use the visibility classes `LEARNER_VISIBLE`, `CONTROL_PLANE`, `SEALED_EVALUATION`, `HUMAN_AUTHORITY`, and future-only `RESTRICTED_PERSONAL`.
4. Use keyed or ciphertext commitments for sealed low-entropy payloads.
5. Represent corrections, versions, invalidations, and revocations as new typed lineage records.
6. Establish final content commitment and the complete subject tuple first; make the immutable assessment bind that tuple second; create the final artifact envelope recording the assessment reference/requirement third; and create a separate authorization/promotion envelope binding the final artifact and assessment fourth.
7. Treat indexes and current-status views as rebuildable projections, never authoritative evidence.

## Alternatives rejected

- Mutable database rows: cannot prove correction history or exact rollback.
- Filename/path identity: changes under copying and storage replanning.
- One raw SHA-256 for every payload: enables dictionary attacks against small sealed spaces.
- Assessment binding a final artifact ID that embeds the assessment: creates circular identity dependencies.
- A single public/private bit: cannot express learner, evaluator, control-plane, human-authority, and personal-data boundaries.

## Affected consumers

- All registered artifact producers must emit or explicitly mark non-applicable canonical identity fields and select a visibility policy.
- RUNTIME must reject identity/parent/authorization mismatch before use or promotion.
- SCIENCE must keep sealed evaluation payloads unresolved outside evaluator authority and bind receipts to exact manifests and subjects.
- SAFETY must bind assessments to the complete pre-envelope subject tuple and issue signed authority records outside learner write control.
- RESOURCES must bind inventories/plans without making storage location part of semantic identity.
- WORLDS must mark task, observation, feedback, and partition visibility and prevent answer leakage through metadata.
- LEARNING may consume only artifacts explicitly eligible and visible to it.
- TOOLS must bind package/receipt provenance and permissions to exact identities.

## Consequences

The root interface registry must add the evidence-envelope, artifact-lineage, and authorization-binding contracts and clarify that authorization is verified through a sidecar binding. Every producer must populate or explicitly mark non-applicable identity fields. Runtime and promotion consumers must validate a current binding rather than trust an artifact-local boolean.

## Migration and rollback

No implementation or accepted evidence exists, so approval requires no data migration. If rejected, retain this proposed ADR and revise the child contracts under new revisions; do not silently reuse its semantics.
