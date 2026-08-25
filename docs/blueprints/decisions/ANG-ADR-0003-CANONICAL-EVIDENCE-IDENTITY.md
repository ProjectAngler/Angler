---
adr_id: ANG-ADR-0003
title: Canonical evidence identity, visibility, and authorization binding
status: accepted
owner: ANG-BP-ROOT
accepted_at: 2026-08-25
originating_proposal: ANG-ADR-EVIDENCE-0001
supersedes: none
superseded_by: none
---

# Canonical evidence identity, visibility, and authorization binding

## Context

Every branch needs one identity and visibility language. Mutable paths or database rows cannot support exact rollback. A raw digest may leak a small sealed answer space by enumeration. An assessment cannot bind an artifact identifier that recursively includes that same assessment.

## Decision

1. Every persistent or evaluative artifact uses `ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0` or a registered successor.
2. Identity is derived from versioned canonical semantic material and excludes storage location, caches, mutable status/projections, future signatures, and access time.
3. The initial visibility classes are `LEARNER_VISIBLE`, `CONTROL_PLANE`, `SEALED_EVALUATION`, `HUMAN_AUTHORITY`, and future-only `RESTRICTED_PERSONAL`. CR0 forbids the personal-data class.
4. Low-entropy sealed or authority-protected payloads use a keyed or authenticated-encryption commitment; a publicly enumerable raw hash is forbidden.
5. Corrections, versions, derivations, supersessions, invalidations, revocations, promotions, migrations, and rollbacks are new typed immutable lineage records. Nothing rewrites accepted history.
6. Authorization uses this non-circular sequence:
   - commit the final payload content and complete subject tuple;
   - issue an immutable human-impact assessment binding that tuple;
   - create the final artifact envelope with the assessment reference/requirement;
   - create a separate authorization binding that joins the final artifact and assessment;
   - let a separate promotion decision verify current authorization and scientific policy.
7. Indexes, active pointers, and current-status views are rebuildable projections, never authoritative evidence.

The complete assessment subject tuple includes artifact type/version, content commitment, parents, intended scope, resource plan, permissions, context, conditions, and duration. A different value requires a different assessment/binding.

## Authority and consumer effects

- EVIDENCE owns canonicalization, envelope, lineage, and structural validation; it records but does not make safety, scientific, resource, or promotion decisions.
- SAFETY owns assessment meaning, disposition, authority, conditions, expiry, and revocation.
- RUNTIME rejects missing, stale, revoked, or mismatched binding before use/promotion.
- SCIENCE keeps sealed payloads unresolved outside evaluator authority and binds receipts to exact manifests and subjects.
- WORLDS marks task/observation/feedback visibility and prevents answer leakage through metadata.
- RESOURCES keeps physical location out of semantic identity while binding inventory and plan semantics.
- LEARNING and TOOLS may consume only explicitly visible and eligible artifacts.

No envelope-local `authorized`, `safe`, `promoted`, or `current` boolean is authoritative.

## Alternatives rejected

- Mutable evidence rows or filename/path identity.
- One public raw SHA-256 rule for every payload.
- A public/private bit instead of role/purpose-bound projections.
- An assessment/final-artifact hash cycle.
- Treating an index or active pointer as evidence.

## Compatibility and rollout

CR0 may implement the canonical envelope, Episode, ExperimentManifest, and pure synthetic identity/visibility tests with the Python standard library. Cross-branch producers cannot claim admissible evidence until their contracts accept the envelope obligation. Unknown major versions fail closed. Any material semantic change requires a successor contract/ADR and consumer revalidation.

## Rollback

No prior executable evidence exists. If this decision is later superseded, preserve all artifacts and the originating proposal, stop admission of the affected contract version, create successor identities, and require explicit migrations. Never reinterpret old IDs under new semantics.
