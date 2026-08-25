---
blueprint_id: ANG-WORK-EVIDENCE-SCHEMAS-001
parent_id: ANG-BP-EVIDENCE-SCHEMAS
revision: 1
tier: 4
design_status: approved
delivery_status: ready
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: ANG-EXEC-CODEX-ROOT-CR0-001
independent_validator: ANG-AUTH-VALIDATOR-001
independent_gate_authority: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-001
independent_gate_reviewer_session_ref: codex-subagent:/root/safety_change_map
test_id: ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001
updated_at: 2026-08-25
permitted_authoring_mechanism: Codex apply_patch limited to authorized_write_scope
permitted_command_classes:
  - read-only PowerShell Test-Path -LiteralPath on baseline targets
  - read-only PowerShell Get-FileHash -Algorithm SHA256 on the named baseline and leaf outputs
  - python -B -m unittest tests.unit.evidence.test_evidence_schemas
authorized_read_scope:
  - AGENTS.md
  - docs/blueprints/ROOT_CAPSULE.md
  - docs/blueprints/INTERFACE_REGISTRY.md
  - docs/blueprints/decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md
  - docs/blueprints/decisions/ANG-ADR-0003-CANONICAL-EVIDENCE-IDENTITY.md
  - docs/blueprints/branches/evidence/CAPSULE.md
  - docs/blueprints/branches/evidence/STATUS.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EVIDENCE-ENVELOPE-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EPISODE-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EXPERIMENT-MANIFEST-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-EVIDENCE-SCHEMAS-001.md
  - docs/blueprints/branches/evidence/children/artifact-lineage/CAPSULE.md
  - docs/blueprints/branches/evidence/children/artifact-lineage/contracts/ANG-CTR-ARTIFACT-LINEAGE-001.md
  - docs/blueprints/branches/evidence/children/artifact-lineage/contracts/ANG-CTR-AUTHORIZATION-BINDING-001.md
  - docs/blueprints/branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md
  - docs/blueprints/branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md
  - docs/blueprints/branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md
  - docs/blueprints/branches/safety/children/human-authority/BLUEPRINT.md
  - docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-EVIDENCE-SCHEMAS-001.json
authorized_write_scope_owner: ANG-EXEC-CODEX-ROOT-CR0-001
authorized_write_scope:
  - src/angler/episodes/__init__.py
  - src/angler/episodes/canonical.py
  - src/angler/episodes/schema_validation.py
  - src/angler/episodes/visibility.py
  - src/angler/episodes/schemas/evidence-envelope.v1.json
  - src/angler/episodes/schemas/episode.v1.json
  - src/angler/episodes/schemas/experiment-manifest.v1.json
  - tests/unit/evidence/test_evidence_schemas.py
  - tests/fixtures/evidence_schemas/valid-envelope.json
  - tests/fixtures/evidence_schemas/valid-episode.json
  - tests/fixtures/evidence_schemas/valid-experiment-manifest.json
  - tests/fixtures/evidence_schemas/invalid-cases.json
  - tests/fixtures/evidence_schemas/visibility-matrix.json
  - tests/fixtures/evidence_schemas/sealed-commitment-cases.json
  - artifacts/control-plane/evidence-schemas/test-receipt.json
  - artifacts/control-plane/evidence-schemas/effect-receipt.json
  - artifacts/control-plane/evidence-schemas/HANDOFF.md
executor_denied_write_scope:
  - artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json
independent_gate_read_scope:
  - docs/blueprints/decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md
  - docs/blueprints/branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md
  - docs/blueprints/branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md
  - docs/blueprints/branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md
  - docs/blueprints/branches/safety/children/human-authority/BLUEPRINT.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-EVIDENCE-SCHEMAS-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/work/ANG-WORK-EVIDENCE-SCHEMAS-001.md
  - docs/blueprints/releases/construction-0/MANIFEST.md
  - docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-EVIDENCE-SCHEMAS-001.json
  - src/angler/episodes/__init__.py
  - src/angler/episodes/canonical.py
  - src/angler/episodes/schema_validation.py
  - src/angler/episodes/visibility.py
  - src/angler/episodes/schemas/evidence-envelope.v1.json
  - src/angler/episodes/schemas/episode.v1.json
  - src/angler/episodes/schemas/experiment-manifest.v1.json
  - tests/unit/evidence/test_evidence_schemas.py
  - tests/fixtures/evidence_schemas/valid-envelope.json
  - tests/fixtures/evidence_schemas/valid-episode.json
  - tests/fixtures/evidence_schemas/valid-experiment-manifest.json
  - tests/fixtures/evidence_schemas/invalid-cases.json
  - tests/fixtures/evidence_schemas/visibility-matrix.json
  - tests/fixtures/evidence_schemas/sealed-commitment-cases.json
  - artifacts/control-plane/evidence-schemas/test-receipt.json
  - artifacts/control-plane/evidence-schemas/effect-receipt.json
  - artifacts/control-plane/evidence-schemas/HANDOFF.md
independent_gate_write_scope_owner: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_write_scope_instance: ANG-REVIEW-CODEX-SAFETY-CR0-001
independent_gate_write_scope:
  - artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json
prohibited_changes:
  - PROJECT_BLUEPRINT.md
  - AGENTS.md
  - docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md
  - docs/blueprints/decisions/**
  - docs/blueprints/branches/safety/**
  - docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-EVIDENCE-SCHEMAS-001.json
  - .git/**
  - outputs/**
  - every __pycache__ directory and *.pyc file
  - all model, learner, runtime, environment, tool, and deployment code
contracts_in:
  - ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0
  - ANG-CTR-EPISODE-001@1.0.0
  - ANG-CTR-EXPERIMENT-MANIFEST-001@1.0.0
  - ANG-CTR-ARTIFACT-LINEAGE-001@1.0.0
contracts_out: []
gate: ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001
normal_technical_gate: ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001
authorization_profile: ANG-POL-LOCAL-SCAFFOLD-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
rollback_ref: "ANG-BASELINE-EVIDENCE-SCHEMAS-001@sha256:F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F"
construction_release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
---

# Exact objective

Implement and locally validate the three approved EVIDENCE schema contracts with deterministic Python 3.11 standard-library code and synthetic fixtures only.

## Required context

Read root and EVIDENCE capsules, EVIDENCE-SCHEMAS blueprint/status, its three contracts and gate, ARTIFACT-LINEAGE capsule/contract, `ANG-ADR-0002`, accepted identity ADR successor, and the applicable impact authorization. Do not load recovered outputs or unrelated branch internals.

## Inputs and preconditions

- All consumed contract revisions are approved for CR0 and registered by `ANG-ADR-0003`.
- The executor, deterministic validator role, and concrete independent SAFETY reviewer instance/session are explicitly bound and distinct; the reviewer accepted the bounded role before release authorization, and the executor cannot issue its own disposition or relabel itself as reviewer.
- `ANG-BASELINE-EVIDENCE-SCHEMAS-001` records every target as absent and must be reverified immediately before the first write.
- `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001` plus the local-scaffold policy and this leaf form the exact-scope LOW `BOOTSTRAP_WORK` `ALLOW`.
- Python 3.11 and PowerShell are already present; no package installation or import outside the standard library is permitted.

## Deliverables

- Three versioned JSON schema documents at the exact paths in `authorized_write_scope`.
- Deterministic canonicalization, structural validation, and visibility-policy reference code.
- Positive and negative synthetic fixtures covering every gate case.
- One `unittest` module and immutable local test/effect receipts under the declared artifact path.
- One separately authored, immutable scaffold-gate decision at the independent gate path; the executor cannot create or edit it.
- No model, learner, event store, lineage persistence, or production service.

## Execution constraints

- Local-only, network-free, foreground execution.
- Executor file creation/editing uses only the Codex `apply_patch` operation and only the literal `authorized_write_scope` paths. The independent SAFETY gate phase may use `apply_patch` only for the one literal `independent_gate_write_scope` path after reviewing only `independent_gate_read_scope`. Shell redirection, ad-hoc writer scripts, bulk rewrites, cross-role writes, and any other authoring mechanism are unauthorized.
- Pre-start/read-only verification may use only PowerShell `Test-Path -LiteralPath` for the literal baseline targets and `Get-FileHash -Algorithm SHA256` for the named baseline/leaf outputs. Test execution uses only the exact `python -B -m unittest tests.unit.evidence.test_evidence_schemas` command below; `-B` prevents undeclared bytecode-cache writes.
- Standard-library imports only; permitted modules include `json`, `hashlib`, `hmac`, `unicodedata`, `decimal`, `datetime`, `pathlib`, and `unittest`.
- No model/GPU access, subprocess tree, background process, telemetry, package manager, credential access, real-person data, recovered material, or files outside declared paths.
- Python bytecode/cache/temp output is prohibited. Effect inspection must confirm that no `__pycache__` directory, `.pyc` file, or undeclared temporary file was created.
- Synthetic fixtures use no real name, account, conversation, repo content, or recoverable personal identifier.
- One foreground command at a time; no parallelism or subprocess creation; 60-second command timeout and 600-second aggregate active ceiling.
- Maximum 1 logical CPU core, 512 MiB working set, 25 MiB total new files, 5 MiB per file, and 1 MiB per fixture; network, GPU, packages, spend, and background work remain zero.
- Stop on any out-of-scope read/write need, dependency need, network need, scope ambiguity, protected-data discovery, or inability to restore the baseline.

## Tests and evidence

Run from repository root:

```powershell
python -B -m unittest tests.unit.evidence.test_evidence_schemas
python -B -m unittest tests.unit.evidence.test_evidence_schemas
```

Both runs must produce the same case count and results. Record source/schema/fixture hashes, Python version, command, start/end time, result, observed written paths, and resource/output totals. Independently inspect the negative-control matrix and confirm no out-of-scope filesystem/network/dependency effect.

Expected bootstrap requirements are exactly those in `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`; the technical contract cases remain those specified by `ANG-GATE-EVIDENCE-SCHEMAS-001`, but running them under CR0 does not run or pass that normal gate. No threshold may be changed after viewing results under either identity.

## Acceptance gate

The leaf's CR0 scaffold delivery completes only when the independently authored `scaffold-gate-decision.json` records `SCAFFOLD_ACCEPTED` for `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` under the current bootstrap `ALLOW` and binds the deterministic validation/effect receipts. `ANG-GATE-EVIDENCE-SCHEMAS-001` and `ANG-GATE-HUMAN-FLOURISHING-001` remain not run; scaffold acceptance does not complete the node or approve ARTIFACT-LINEAGE, EVENT-STORE, model execution, Slice 00, M0, or any normal technical/scientific gate.

## Failure and rollback

Stop immediately. Preserve immutable `test-receipt.json`, `effect-receipt.json`, `HANDOFF.md`, and any independent `scaffold-gate-decision.json` on failure, rejection, escalation, or rollback. For baseline targets classified `restore_or_remove` only, verify each path resolves beneath the repository and matches the declaration; restore a pre-existing file from the pre-leaf Git baseline or remove an absent-at-baseline file individually. Never delete or rewrite `preserve_on_failure` evidence, and do not use a broad or recursive delete. Record rollback verification and leave both normal gates not run.

## Handoff requirement

The executor writes `artifacts/control-plane/evidence-schemas/HANDOFF.md` recording the executor, validator, intended independent gate authority, pre-leaf Git/base snapshot, exact revisions, files, commands, elapsed/resource use, tests, failures, effects, evidence identities, authorization, baseline, rollback result, and next action. It must state explicitly that the independent decision is still pending, the normal EVIDENCE technical and Human-Flourishing gates remain not run, and ARTIFACT-LINEAGE delivery remains blocked pending the normal technical gate rather than the CR0 scaffold receipt. The SAFETY gate authority then writes only the separate decision artifact.
