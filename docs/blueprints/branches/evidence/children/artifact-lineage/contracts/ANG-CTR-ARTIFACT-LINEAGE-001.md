---
contract_id: ANG-CTR-ARTIFACT-LINEAGE-001
version: 1.0.0
owner: ANG-BP-ARTIFACT-LINEAGE
status: approved_for_cr0
producers:
  - all artifact producers
  - evidence relationship recorder
consumers:
  - all branches resolving provenance, compatibility, authorization, replay, or rollback
---

# Artifact lineage and relationship record

## Purpose

Define immutable, typed, content-addressed relationships that reconstruct provenance, versions, corrections, evaluations, authorizations, promotions, migrations, revocations, and rollback without relying on paths, mutable database rows, caches, or conversation history.

## Input

- A subject envelope conforming to `ANG-CTR-EVIDENCE-ENVELOPE-001@1.x`.
- Zero or more target envelopes already committed under compatible contracts.
- Relationship type, contract-specific cardinality/order, reason/evidence, producer authority, and visibility.
- For a lineage snapshot/projection: an authoritative record frontier and projection policy/version.

Preconditions: every reference carries exact artifact/content ID, artifact type, payload contract/version, and expected visibility; producer is authorized to assert the relationship; target existence can be verified even if payload visibility is denied.

## Output schema

A relationship record is itself a canonical evidence envelope whose payload contains:

```text
relationship_id (same as envelope artifact_id)
subject_ref
relation:
  DERIVED_FROM | VERSION_OF | CORRECTS | SUPERSEDES |
  GENERATED_BY | EVALUATES | ASSESSED_BY | AUTHORIZES |
  PROMOTES | ROLLS_BACK_TO | MIGRATED_FROM |
  INVALIDATES | REVOKES
target_ref
semantic_order | null
reason_code
reason_evidence_refs
asserting_component_and_authority_refs
effective_at_utc
scope_and_condition_refs
compatibility_assertion | null
projection_effect:
  NONE | PREFERRED_SUCCESSOR | INELIGIBLE | REVOKED
```

A lineage snapshot/projection contains only exact relationship/node references, authoritative frontier, projection policy/version, generated-at time, and reconstruction digest. It is a cache artifact, not a replacement for source records.

## Relationship behavior

- `DERIVED_FROM`: subject computation consumed target; default visibility is at least as restrictive as all inputs.
- `VERSION_OF`: subject continues target's logical artifact family; compatibility must be explicit.
- `CORRECTS`: subject corrects target; correction authority, reason, and corrected claim/field scope are required.
- `SUPERSEDES`: an approved projection may prefer subject; target remains valid historical evidence unless separately invalidated.
- `GENERATED_BY`: links artifact to exact producing run/transaction; cardinality is one where the payload contract requires it.
- `EVALUATES`: evaluation receipt binds exact subject, suite, manifest, and plan.
- `ASSESSED_BY`: links subject tuple to exact impact assessment, including non-`ALLOW` outcomes.
- `AUTHORIZES`: the authorization-binding artifact joins final subject envelope and exact assessment.
- `PROMOTES`: promotion decision selects exact candidate under exact policy, evaluation, binding, and parent.
- `ROLLS_BACK_TO`: transaction restores an exact previously admitted target.
- `MIGRATED_FROM`: subject was explicitly replayed/distilled from target under migration/regating.
- `INVALIDATES`: authorized decision makes target ineligible in stated scope; it does not erase target.
- `REVOKES`: authorized record ends an earlier grant from its effective time/frontier.

## Parentage and graph invariants

- All targets exist before an admissible relationship commits.
- Self-edges are invalid. Derivation/version/correction/supersession/promotion/rollback/migration subgraphs are directed acyclic graphs.
- Relationship records and typed parents are canonical identity material.
- Multiple identical edges are invalid unless the owning contract explicitly permits meaningful multiplicity.
- A non-genesis competence state has exactly one competence parent; no query-conditioned alternate lineage is represented as another parent.
- Producer contracts may require stricter type, cardinality, temporal, and compatibility rules.
- Sealed targets can be commitment-verified without payload resolution; denial of payload visibility is not permission to infer it.

## Correction, version, and revocation rules

Original records never change. A correction creates a new artifact plus `CORRECTS`; preference requires `SUPERSEDES`; blocking use requires `INVALIDATES`; ending authority requires `REVOKES`. These effects are separate so a correction cannot silently erase evidence or revoke authority.

Unknown major versions fail closed. A compatible minor version may add relationship metadata but cannot change direction, authority, visibility, or projection effect. A semantic relation change requires a new contract major or new relationship type and consumer review.

## Failure semantics

| Error | Meaning | Response |
|---|---|---|
| `LINEAGE_NODE_MISSING` | Referenced node cannot be verified at frontier | Reject/quarantine; do not infer |
| `LINEAGE_NODE_MISMATCH` | Artifact/content/type/version mismatch | Reject and preserve attempt |
| `LINEAGE_EDGE_INVALID` | Relation violates type/cardinality/order rules | Reject under same identity |
| `LINEAGE_CYCLE` | New edge creates forbidden cycle | Reject edge and dependents |
| `LINEAGE_AUTHORITY_DENIED` | Producer cannot assert relation/effect | Deny and escalate if attempted privilege expansion |
| `LINEAGE_VERSION_INCOMPATIBLE` | Consumer cannot interpret required semantics | Block until revalidated |
| `LINEAGE_FRONTIER_STALE` | Projection lacks records needed for current authority/use decision | Fail closed and refresh authoritative records |
| `LINEAGE_PROJECTION_MISMATCH` | Rebuild differs from cached projection | Discard cache and audit |

## Operation

Validation and projection are deterministic/idempotent for the same records, policy, fixed time, and frontier. Input order cannot change graph or authorization results. Persistence and atomic append belong to EVENT-STORE; a partially written record is never admissible.

## Compatibility

Consumers reject unknown major versions and unknown relation semantics. Minor optional metadata is preserved. New relation types require registry/consumer impact review even if serialized as an enum extension. Projection-policy changes produce new projection identities and never rewrite authoritative edges.

## Contract tests

- Valid DAG reconstructs identically from every declared input ordering.
- Missing, wrong-content, wrong-type, incompatible, self, duplicate, and cyclic edges reject.
- Changing relationship/target/order/authority changes identity.
- Relocating nodes or rebuilding index leaves authoritative identities unchanged.
- Correction, supersession, invalidation, and revocation have distinct effects and retain target.
- One-parent competence constraint rejects two active competence parents.
- Unknown major/relation rejects.
- Stale projection cannot authorize use; rebuild mismatch discards cache.
- Learner cannot assert authority-bearing relationships.
