---
gate_id: ANG-GATE-EVIDENCE-SCHEMAS-001
version: 1
owner: ANG-BP-EVIDENCE-SCHEMAS
status: specified_not_run
independent_verifier: ANG-AUTH-SAFETY-APPROVER-001
human_impact_assessment: required
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Evidence-schema acceptance gate

This is the node's normal technical completion gate. It remains `specified_not_run` during CR0 bootstrap construction and cannot inherit or be satisfied by `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`.

## Claim being tested

The implemented Release-0 schemas and validator deterministically identify valid artifacts, enforce visibility and compatibility, reject ambiguity and leakage, and faithfully implement the three owned contracts.

## Entry criteria

- Node design and contracts are approved at exact revisions.
- Registry entries and affected consumers are synchronized.
- Work leaf has a current exact-scope `ALLOW`, owner, baseline, resource ceiling, and test manifest.
- Schema, validator, and synthetic fixture files have immutable identities.
- ARTIFACT-LINEAGE contract version consumed by reference checks is fixed.

## Procedure

1. Run deterministic schema/canonicalization unit tests twice from a clean local process.
2. Reorder object keys and transport whitespace; verify identical canonical bytes and identities.
3. Mutate each identity-relevant field; verify expected `content_id` or `artifact_id` changes.
4. Run every forbidden visibility pair and protected-error assertion.
5. Run episode partition/eligibility and manifest immutability cases.
6. Run version-compatibility and correction fixtures.
7. Inspect dependencies, paths, data provenance, and Release-0 boundary.

## Precommitted pass/fail thresholds

- 100% of declared positive fixtures validate identically on two runs.
- 100% of declared negative fixtures fail with the specified typed error.
- 100% of forbidden visibility requests deny without payload-derived error detail.
- Every identity-material mutation changes the expected identity; transport-only changes do not.
- Unknown major, duplicate key, non-finite number, raw low-entropy sealed digest, held-out learner eligibility, and manifest threshold mutation are always rejected.
- No network, package install, model/GPU use, real-person data, recovered material, or out-of-scope write occurs.

No flaky retry converts a failure to a pass.

## Required negative controls

- Duplicate JSON key with conflicting values.
- `1`, `1.0`, NaN, infinity, and locale-dependent decimal ambiguity.
- Unicode canonically equivalent but byte-different strings.
- Raw SHA-256 commitment for a four-choice sealed answer.
- Learner request for sealed or authority payload.
- Hidden label leaked in filename, error, size, or projection.
- `HELD_OUT` episode marked learner-eligible.
- Manifest threshold changed while reusing its identity.
- Unknown major version with otherwise familiar fields.
- Missing required parent, plan, assessment requirement, or visibility policy.

## Required child gates

None. This node owns one executable leaf.

## Evidence artifacts and identities

- Work manifest and source/fixture identities.
- Two-run test receipts and negative-control matrix.
- Dependency/path/network/resource inspection receipt.
- Independent review and human-impact authorization.

## Failure and rollback response

Do not publish schema versions or activate consumers. Preserve failed receipts, restore the exact leaf baseline, and revise under new code/schema identities. Leakage invalidates affected fixtures and dependent evaluation identities.

## Waiver policy

No producer, learner, or branch owner may waive an identity, visibility, version, or Release-0 negative control. Threshold changes require an ADR and new test identity before results are rerun.
