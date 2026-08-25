---
blueprint_id: ANG-BP-ARTIFACT-LINEAGE
title: Artifact identity, lineage, correction, and authorization binding
parent_id: ANG-BP-EVIDENCE
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: blocked_by_evidence_schemas
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: unassigned_until_dependency_passes
updated_at: 2026-08-25
parent_revision: 3
required_children: []
work_leaves:
  - ANG-WORK-ARTIFACT-LINEAGE-001
depends_on:
  - ANG-BP-EVIDENCE-SCHEMAS
  - ANG-BP-SAFETY
contracts_in:
  - ANG-CTR-EVIDENCE-ENVELOPE-001
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
  - ANG-CTR-PROMOTION-DECISION-001
contracts_out:
  - ANG-CTR-ARTIFACT-LINEAGE-001
  - ANG-CTR-AUTHORIZATION-BINDING-001
gate: ANG-GATE-ARTIFACT-LINEAGE-001
---

# Artifact identity, lineage, correction, and authorization binding

## Context capsule

This node defines the immutable graph that explains where an artifact came from, which exact parent/version it replaces or corrects, which assessment applies, and why an authorization or promotion was eligible. It does not own payload semantics, storage engines, scientific decisions, or safety dispositions.

## Contribution to the parent

Content hashes without typed relationships cannot prove causality, rollback, correction history, or authorization scope. This node turns immutable evidence envelopes into a reconstructable graph while keeping mutable indexes non-authoritative.

## Inherited requirements and invariants

- All root and EVIDENCE invariants apply.
- One competence lineage, exact rollback, evidence separation, and external authority must remain visible in graph constraints.
- `ANG-ADR-0002` limits implementation to local synthetic control-plane work.
- `ANG-ADR-0003` defines the shared identity/visibility and staged authorization sequence.

## Scope

- Define authoritative node/reference and typed relationship semantics.
- Enforce parent existence, type/version compatibility, cardinality, ordering, and acyclicity.
- Define version, correction, supersession, derivation, invalidation, revocation, promotion, rollback, and migration relationships.
- Define immutable lineage snapshots/projections that are reproducible from authoritative records.
- Define the authorization binding that joins the final artifact envelope to its exact impact assessment without a hash cycle.
- Define freshness, expiry, revocation, wrong-scope, wrong-plan, wrong-parent, and condition validation.

## Explicit non-goals

- Implementing the append-only event store or distributed graph database.
- Selecting an active model/state, deciding safety, or deciding scientific promotion.
- Inferring missing relationships from filenames, timestamps, directory layout, or conversation.
- Allowing a mutable active pointer or cached “authorized” boolean to become evidence.
- Signing as human authority or storing authority credentials.
- Implementing code or claiming executable evidence in this task.

## Inputs, outputs, and contracts

Consumes canonical envelopes, impact assessments, and promotion decisions. Produces:

- `ANG-CTR-ARTIFACT-LINEAGE-001@1.0.0`;
- `ANG-CTR-AUTHORIZATION-BINDING-001@1.0.0`.

Every relationship record is itself a canonical evidence artifact. Producers remain responsible for payload truth; EVIDENCE validates structural and referential claims and preserves the signed authority source.

## Internal design and state ownership

### Authoritative graph

The authoritative graph is the set of immutable envelopes and typed relationship artifacts. There is no mutable master row. A relationship points to exact `artifact_id`, `content_id`, type, contract/version, and expected visibility. Relationships reference already committed nodes.

Allowed initial relations, directed from the new record or subject toward an existing target, are:

| Relation | Meaning | Core constraint |
|---|---|---|
| `DERIVED_FROM` | New payload was computed from target | Target exists; visibility propagates conservatively |
| `VERSION_OF` | New contract/artifact version continues target | Same logical type; compatibility declared |
| `CORRECTS` | New record corrects an error in target | Reason and correcting authority required |
| `SUPERSEDES` | New record should replace target in an approved projection | Does not delete or invalidate target |
| `GENERATED_BY` | Artifact was produced by run/transaction | Exactly one producing run where applicable |
| `EVALUATES` | Receipt evaluates target | Suite/manifest and subject exact-match required |
| `ASSESSED_BY` | Artifact subject tuple is covered by assessment | Assessment binds exact content/parent/scope tuple |
| `AUTHORIZES` | Binding joins final artifact and assessment | Current exact `ALLOW` and authority required |
| `PROMOTES` | Decision selects candidate | Binding, evaluation, policy, and parent match |
| `ROLLS_BACK_TO` | Transaction restores target | Exact previously promoted identity required |
| `MIGRATED_FROM` | Artifact transfers competence/topology | Migration proposal and re-gating required |
| `INVALIDATES` | Authority declares evidence unusable | Reason/scope/authority required; history retained |
| `REVOKES` | Authority ends a prior grant | Exact grant, effective time, and authority required |

Derivation, version, correction, supersession, promotion, rollback, and migration relations are acyclic. Contract-specific cardinality can strengthen these constraints; notably, a non-genesis competence state has exactly one active competence parent.

### Content addressing and parentage

Parent edges are identity material and canonically ordered by `(relation, artifact_type, artifact_id)` unless the producer contract declares semantically meaningful order. Adding, removing, or changing a parent produces a new `artifact_id`. Missing, mistyped, incompatible, self, or cyclic parents make the artifact inadmissible. Sealed parents may be existence/commitment-verified without payload resolution.

### Corrections and versions

An error never rewrites its source. The correction records target, corrected fields/claim, reason, evidence, authority, visibility, and effect on dependents. `CORRECTS` alone adds information; `SUPERSEDES` changes an approved projection; `INVALIDATES` blocks use; none removes historical identity. Unknown major versions reject; migration produces new artifacts and explicit edges.

If protected personal material is ever introduced under a future release, authorized payload withdrawal may replace accessible content with a non-sensitive tombstone and retain only the minimum protected commitment/audit fact. Release 0 forbids such data, so this is not an active capability.

### Authorization binding

The sequence is fixed:

1. Compute final payload `content_id`/commitment and complete subject tuple: type/version, parents, intended scope, plan, permissions, context, and duration.
2. SAFETY produces an immutable assessment binding that tuple and disposition.
3. Produce the final artifact envelope, recording the assessment reference or authorization requirement, and compute its `artifact_id`.
4. Produce an `AuthorizationBinding` envelope that references the final artifact, assessment, constitution/policy, gate, authority, conditions, issue/expiry, and exact subject tuple.
5. Promotion references the binding and independently checks evaluation/scientific policy.

The binding validator recomputes that the assessment subject tuple equals the final artifact’s content, parentage, plan, permissions, scope, and context. `ALLOW` is necessary, never sufficient. Revocation is a new `REVOKES` artifact; current status is a derived query over grants, expiry, conditions, and revocations.

### Projections and reconstruction

Indexes, reverse edges, active versions, current authorization, and lineage diagrams are caches. A projection records source frontier and policy version and must rebuild to the same result. If authoritative records are unavailable or disagree, fail closed rather than select by time or filename.

## Executable work package

`ANG-WORK-ARTIFACT-LINEAGE-001` implements pure local identity/relationship validation and synthetic graph fixtures. It remains `not_ready` until EVIDENCE-SCHEMAS passes, contract consumers accept the boundary, and a separate impact `ALLOW` and rollback baseline exist.

## Dependencies and sequencing

1. Approve this design together with EVIDENCE-SCHEMAS and the shared ADR.
2. Do not implement until the schema leaf gate passes at its exact version.
3. Implement immutable graph validation before EVENT-STORE chooses persistence mechanics.
4. Require RUNTIME, SCIENCE, and SAFETY contract tests before using bindings for real promotion.

## Acceptance gate and required evidence

`ANG-GATE-ARTIFACT-LINEAGE-001` requires deterministic graph, correction, version, authorization, revocation, and reconstruction tests under the Release-0 scope. Design review alone does not pass it.

## Testing and validation

- Stable identities independent of location/index; mutations affect expected identities.
- Missing, dangling, mistyped, incompatible, self, duplicate, and cyclic edges.
- Contract-specific parent cardinality, including one competence parent.
- Correction/supersession/invalidation/revocation retain original history.
- Projection rebuild equivalence from shuffled record order.
- Assessment/final-envelope subject-tuple equivalence.
- Missing, wrong-artifact, content, parent, plan, permission, scope, context, condition, time, policy, authority, gate, disposition, and revocation binding cases.
- Learner cannot author, select, overwrite, or validate authority records.

## Risks, failure behavior, and rollback

- Graph cycle or inferred parent: quarantine and reject dependents.
- Authorization substitution: deny all dependent use/promotion and preserve attempt evidence.
- Mutable projection trusted as truth: discard/rebuild projection and audit consumers.
- Revocation race: require authoritative frontier/freshness; stale view fails closed.
- Overbroad tombstone/privacy mechanism: deferred and prohibited in Release 0.

On failure, preserve receipts, restore exact leaf baseline, and do not publish a lineage version. A failed binding never mutates the subject artifact.

## Resource profiles and scaling

Release-0 graphs are small, in-memory, CPU-only, and synthetic. Later indexes may shard, but validation results, typed edges, authoritative frontier, and authorization freshness cannot depend on node placement or graph-store vendor.

## Decisions and active ADRs

- `ANG-ADR-0002` accepted Release-0 boundary.
- `ANG-ADR-EVIDENCE-0001` accepted through project decision `ANG-ADR-0003`.

## Current status and blockers

Design revision 1 is approved for CR0. Delivery remains blocked by `ANG-GATE-EVIDENCE-SCHEMAS-001`; its implementation leaf needs a successor exact baseline/assessment after that gate passes.

## Parent roll-up and next exact action

Two detailed contracts and one bounded implementation leaf are specified. Review now; after EVIDENCE-SCHEMAS passes, authorize `ANG-WORK-ARTIFACT-LINEAGE-001` without pulling EVENT-STORE or runtime behavior into scope.
