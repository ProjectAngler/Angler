---
blueprint_id: ANG-BP-ARTIFACT-LINEAGE
blueprint_revision: 1
capsule_revision: 2
freshness_date: 2026-08-25
parent_id: ANG-BP-EVIDENCE
target_tokens: 650
---

# ARTIFACT-LINEAGE capsule

Outcome: an immutable, typed, reconstructable artifact graph with correction/version rules and exact impact-authorization binding.

Authoritative state is the set of canonical artifacts and relationship records. Indexes, active pointers, current-authorization views, and diagrams are rebuildable caches. Relationships reference exact artifact/content IDs, type, contract/version, and visibility; applicable edges are acyclic and parents must already exist.

Initial relations: `DERIVED_FROM`, `VERSION_OF`, `CORRECTS`, `SUPERSEDES`, `GENERATED_BY`, `EVALUATES`, `ASSESSED_BY`, `AUTHORIZES`, `PROMOTES`, `ROLLS_BACK_TO`, `MIGRATED_FROM`, `INVALIDATES`, and `REVOKES`. Corrections and revocations create new records and retain originals. Parent changes create new identities; no filename/timestamp inference.

Authorization sequence: compute final content commitment and subject tuple → SAFETY assessment binds it → final envelope records assessment reference/requirement → sidecar binding joins final artifact and assessment → promotion references binding. Binding validates content, parent, plan, permissions, scope, context, conditions, authority, disposition, expiry, and revocation. `ALLOW` is necessary, never sufficient.

Unknown major versions, missing parents, stale authority frontiers, and inconsistent projections fail closed. A non-genesis competence state has exactly one active competence parent.

Gate: `ANG-GATE-ARTIFACT-LINEAGE-001`, specified but not run. Design is approved for CR0, but `ANG-WORK-ARTIFACT-LINEAGE-001` remains blocked until schema delivery and a separate authorization/baseline. No code or executable evidence exists.
