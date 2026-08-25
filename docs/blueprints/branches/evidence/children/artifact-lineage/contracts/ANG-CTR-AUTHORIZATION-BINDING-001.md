---
contract_id: ANG-CTR-AUTHORIZATION-BINDING-001
version: 1.0.0
owner: ANG-BP-ARTIFACT-LINEAGE
status: approved_for_cr0
producers:
  - independent safety authorization process
  - human-controlled promotion-envelope process
consumers:
  - ANG-BP-RUNTIME
  - ANG-BP-SCIENCE
  - ANG-BP-EVIDENCE
  - ANG-BP-SAFETY
  - human promotion authority
---

# Human-impact authorization binding

## Purpose

Create an immutable sidecar envelope proving that one exact final artifact matches the exact subject tuple reviewed by one human-impact assessment and remains within its constitution, policy, authority, scope, conditions, time window, and revocation frontier. This is a promotion/work prerequisite, not scientific approval or deployment authority.

## Input

- Final subject envelope and exact artifact/content identities.
- `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1` assessment.
- Subject tuple used by the assessment: content commitment, artifact type/version, parents, purpose/scope, resource plan, permissions, context, duration, and material conditions.
- Human-flourishing gate decision and active constitution/policy identities.
- Independent assessor and human-authority attestations.
- Fixed validation time and authoritative revocation/invalidation frontier.
- For a promotable artifact, evaluation and promotion-policy references remain external prerequisites.

Preconditions: assessment is complete/authentic; final artifact records the applicable assessment reference or explicit authorization requirement; authority records are outside learner/subject write control; no required tuple field is inferred.

## Output schema

The binding, wrapped by `ANG-CTR-EVIDENCE-ENVELOPE-001` with `HUMAN_AUTHORITY` visibility, contains:

```text
binding_id (same as envelope artifact_id)
authorization_kind:
  BOOTSTRAP_WORK | EXPERIMENT_EXECUTION | ARTIFACT_PROMOTION |
  TOOL_ACTIVATION | MIGRATION | DEPLOYMENT
subject_artifact_ref
subject_tuple:
  content_id
  artifact_type_and_contract_version
  parent_refs
  intended_purpose_and_scope_ref
  resource_plan_ref
  permission_set_ref
  deployment_or_execution_context_ref
  duration_or_expiry_basis
assessment_ref
assessment_subject_tuple_digest
constitution_ref
assessment_policy_ref
human_flourishing_gate_decision_ref
assessment_disposition: ALLOW
conditions_and_ceiling_ref
independent_assessor_ref
human_authority_ref
authority_attestation_refs
issued_at_utc
expires_at_utc
validated_at_utc
authoritative_frontier_ref
required_monitor_stop_rollback_refs
```

An assessment with `DENY` or `ESCALATE` remains linked by `ASSESSED_BY` evidence but cannot produce a valid authorization binding.

## Staged binding behavior

1. Compute final content commitment and complete proposed subject tuple.
2. Assessment immutably binds that tuple; it does not bind a future final artifact ID that depends on assessment reference.
3. Final artifact envelope records assessment reference/requirement and receives its artifact ID.
4. Binding references both and recomputes tuple equality.
5. Promotion or work execution references the binding and performs its own independent eligibility checks.

The following must match exactly unless the assessment explicitly authorizes a bounded set represented in its tuple: content, artifact type/version, parents, purpose, scope, plan, permissions, context, duration, constitution, policy, gate, and conditions. “Equivalent” filenames, latest versions, larger plans, subset guesses, or semantically inferred identities are forbidden.

## Eligibility behavior

A binding is current only if:

- assessment and authority attestations authenticate;
- disposition is `ALLOW`;
- exact tuple equality holds;
- final envelope assessment reference/requirement is consistent;
- validation time is inside issue/expiry window;
- required conditions, monitoring, stop, and rollback remain satisfiable;
- no applicable `REVOKES` or `INVALIDATES` record exists at the authoritative frontier;
- requested operation is no broader than `authorization_kind` and ceiling;
- contract/constitution/policy/gate revisions are supported.

Current status is derived; no stored `is_authorized` boolean is authoritative. `ALLOW` remains necessary, never sufficient: SCIENCE, RUNTIME, permissions, and human promotion/deployment controls may still reject.

## Mutation and authority rules

- Binding, assessment, and authority records are immutable.
- Revocation is a new signed `REVOKES` relationship; changed conditions or scope require a new assessment and binding.
- The learner, subject, assessed artifact, ordinary operator, and producer under assessment cannot author, approve, select, overwrite, or validate their own binding.
- Credentials/signature keys are never stored as evidence payloads.
- A larger resource plan or copied artifact never inherits authorization automatically.

## Failure semantics

| Error | Meaning | Response |
|---|---|---|
| `AUTH_BINDING_MISSING` | Operation requires no supplied binding | Block operation |
| `AUTH_BINDING_DISPOSITION` | Assessment is not `ALLOW` | Deny; preserve assessment |
| `AUTH_BINDING_SUBJECT_MISMATCH` | Content/artifact/type differs | Deny and preserve substitution attempt |
| `AUTH_BINDING_PARENT_MISMATCH` | Parentage differs | Deny; new assessment required |
| `AUTH_BINDING_SCOPE_MISMATCH` | Purpose/scope/context/operation exceeds grant | Deny or escalate |
| `AUTH_BINDING_PLAN_MISMATCH` | Resource plan or permissions differ | Deny; reassess |
| `AUTH_BINDING_CONDITION_FAILED` | Required condition/monitor/rollback absent | Deny or stop dependent operation |
| `AUTH_BINDING_EXPIRED` | Validation time outside window | Deny; successor assessment required |
| `AUTH_BINDING_REVOKED` | Applicable revocation exists | Stop dependent operation and rollback/quarantine |
| `AUTH_BINDING_FRONTIER_STALE` | Current revocation status cannot be established | Fail closed |
| `AUTH_BINDING_TAMPERED` | Identity/attestation/policy validation fails | Quarantine and notify authority |
| `AUTH_BINDING_AUTHORITY_DENIED` | Issuer/validator is not authorized/independent | Deny and escalate |

## Operation

Validation is deterministic/idempotent for the same final artifact, assessment, authority records, requested operation, fixed time, and authoritative frontier. It performs no promotion itself. Runtime must validate immediately before an authorized operation; cached success is invalid after frontier, condition, scope, or time changes.

## Compatibility

Unknown major contract, constitution, assessment-policy, or gate revisions reject. Minor additive fields are accepted only if mandatory tuple and authority semantics remain unchanged. Any weakening or broadening requires a successor contract, ADR, reassessment, and consumer revalidation.

## Contract tests

- Valid exact-scope `ALLOW` binding succeeds only for its named operation.
- Missing, `DENY`, `ESCALATE`, expired, revoked, tampered, stale-frontier, wrong-authority, or unsigned records reject.
- Same payload with wrong final envelope, parent, plan, permission, scope, context, or duration rejects.
- Final envelope assessment reference differing from binding rejects.
- Larger resource plan and copied artifact do not inherit authorization.
- Cached valid result fails after applicable revocation.
- Learner/subject cannot author, approve, select, overwrite, or validate binding.
- Promotion remains blocked without its independent scientific/promotion prerequisites.
- Replay reconstructs the same authorization relationship from immutable records and frontier.
