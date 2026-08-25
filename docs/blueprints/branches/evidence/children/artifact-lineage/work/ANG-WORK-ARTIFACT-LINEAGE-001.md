---
blueprint_id: ANG-WORK-ARTIFACT-LINEAGE-001
parent_id: ANG-BP-ARTIFACT-LINEAGE
revision: 1
tier: 4
design_status: draft
delivery_status: not_ready
accountable_owner: unassigned
execution_owner: unassigned
updated_at: 2026-08-25
authorized_write_scope:
  - src/angler/episodes/identity.py
  - src/angler/episodes/lineage.py
  - src/angler/episodes/authorization.py
  - tests/unit/evidence/test_artifact_lineage.py
  - tests/fixtures/artifact_lineage/**
  - artifacts/control-plane/artifact-lineage/**
prohibited_changes:
  - PROJECT_BLUEPRINT.md
  - AGENTS.md
  - docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md
  - docs/blueprints/decisions/**
  - docs/blueprints/branches/safety/**
  - outputs/**
  - schema-leaf outputs except through a successor leaf
  - all model, learner, runtime, environment, tool, event-store, and deployment code
contracts_in:
  - ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0
  - ANG-CTR-ARTIFACT-LINEAGE-001@1.0.0
  - ANG-CTR-AUTHORIZATION-BINDING-001@1.0.0
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
contracts_out: []
gate: ANG-GATE-ARTIFACT-LINEAGE-001
human_impact_assessment: required_before_ready
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
rollback_ref: required_before_ready
construction_release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
---

# Exact objective

Implement deterministic local validation of artifact identities, typed lineage, corrections/versions, projections, and exact human-impact authorization bindings using only synthetic records and the approved schema kernel.

## Required context

Read root/EVIDENCE/ARTIFACT-LINEAGE capsules, both lineage contracts and gate, accepted identity ADR successor, approved schema contracts/gate evidence, `ANG-ADR-0002`, and this leaf’s impact authorization. Do not load recovered outputs or unrelated branch internals.

## Inputs and preconditions

- `ANG-GATE-EVIDENCE-SCHEMAS-001` passed at every consumed version.
- Both lineage contracts and registry entries are approved.
- Named accountable/execution owners and independent verifier exist.
- A content-addressed pre-leaf baseline covers exact write paths.
- A current exact-scope `ALLOW` fixes paths, commands, synthetic-only data, no-network mode, time, resources, and rollback.
- Python 3.11 standard library is the only dependency; no package installation is allowed.

## Deliverables

- `identity.py`: immutable typed references and identity checks built on the approved canonical module.
- `lineage.py`: relation validation, DAG/cardinality checks, corrections/versions, and deterministic projections.
- `authorization.py`: pure binding/freshness/revocation validation with explicit fixed time/frontier inputs; no credential handling or signature implementation beyond fixture-level interface stubs.
- Synthetic positive/negative graph, assessment, binding, expiry, and revocation fixtures.
- One unit-test module and immutable local receipts under the declared artifact path.
- No persistence engine, model/runtime integration, real signature authority, promotion, or deployment behavior.

## Execution constraints

- Local-only, network-free, foreground, CPU-only execution.
- Standard-library imports only; no subprocess tree, background process, telemetry, package manager, model/GPU, authority credential, real-person data, recovered material, or out-of-scope file.
- Fixture authority/signature values are visibly synthetic and cannot be mistaken for real credentials.
- One command at a time; 120-second command timeout; all generated files/evidence remain below 20 MiB.
- Stop on schema-version mismatch, out-of-scope need, non-synthetic input, network/dependency request, authority ambiguity, or rollback uncertainty.

## Tests and evidence

Run twice from repository root:

```powershell
python -m unittest tests.unit.evidence.test_artifact_lineage
python -m unittest tests.unit.evidence.test_artifact_lineage
```

Both runs must produce identical case counts/results. Record source/fixture hashes, consumed schema evidence, Python version, commands, fixed test time, authoritative frontier, path/network/dependency observations, resource/output totals, and result.

The independent verifier inspects every gate negative control, including shuffled order, cycles, two-parent competence, correction retention, subject-tuple mismatch, expiry/revocation, stale cache/frontier, and learner-authored authority.

## Acceptance gate

Completion requires `ANG-GATE-ARTIFACT-LINEAGE-001` under current exact-scope human-impact `ALLOW`. Passing does not approve EVENT-STORE, runtime enforcement, real cryptographic authority, promotion, Slice 00, or M0.

## Failure and rollback

Stop, preserve failing receipts, verify rollback targets resolve beneath the repository and exact scope, restore modified files from `rollback_ref`, and remove only individually verified leaf-created files. Never use broad recursive deletion. Record rollback verification and keep lineage/binding contracts unpublished.

## Handoff requirement

Record exact revisions, files, commands, tests, fixed time/frontier, failures, effects, evidence, authorization, baseline, and next action. Explicitly state that real authority, persistence, runtime, and promotion integration remain unimplemented and separately gated.
