---
blueprint_id: ANG-WORK-CR0-RESOURCES-001
parent_id: ANG-BP-RESOURCES
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 2
tier: 4
design_status: approved
delivery_status: ready
activation_state: unusable_pending_revalidation
execution_condition: ANG-CR0-REVALIDATION-20260825-003 is APPROVED and Manifest v2 is PASS/authorized
accountable_owner: ANG-BP-RESOURCES
execution_owner: ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001
independent_validator: ANG-AUTH-VALIDATOR-001
independent_gate_authority: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001
independent_gate_reviewer_session_ref: codex-subagent:/root/safety_change_map
independent_gate_reviewer_acceptance: ACCEPTED
independent_gate_reviewer_reachability: reachable
reviewer_vocabulary_ack: ACK_ACCEPTED
result_recorder: ANG-BP-ROOT
authorized_write_scope_owner: ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001
authorized_write_scope:
  - schemas/control/v1/resources/resource-inventory.schema.json
  - schemas/control/v1/resources/execution-plan.schema.json
  - tests/synthetic/slice00/resources/constrained.inventory.json
  - tests/synthetic/slice00/resources/workstation.inventory.json
  - tests/synthetic/slice00/resources/cluster.inventory.json
  - tests/synthetic/slice00/resources/invalid-overcommitted.plan.json
  - tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1
  - docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md
executor_denied_write_scope:
  - docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md
independent_gate_write_scope_owner: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_write_scope_instance: ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001
independent_gate_write_scope:
  - docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md
authorized_read_scope:
  - docs/blueprints/ROOT_CAPSULE.md
  - docs/blueprints/PROTOCOL.md
  - docs/blueprints/INTERFACE_REGISTRY.md
  - docs/blueprints/branches/resources/BLUEPRINT.md
  - docs/blueprints/branches/resources/CAPSULE.md
  - docs/blueprints/branches/resources/contracts/ANG-CTR-RESOURCE-INVENTORY-001.md
  - docs/blueprints/branches/resources/contracts/ANG-CTR-EXECUTION-PLAN-001.md
  - docs/blueprints/decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md
  - docs/blueprints/decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md
  - docs/blueprints/branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md
  - docs/blueprints/branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md
  - docs/blueprints/releases/construction-0/MANIFEST.md
  - docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-001.json
  - docs/blueprints/branches/resources/gates/ANG-GATE-CR0-RESOURCES-001.md
  - artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json
  - src/angler/episodes/schemas/evidence-envelope.v1.json
updated_at: 2026-08-25
gate: ANG-GATE-CR0-RESOURCES-001@1
normal_resource_design_gate: ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
rollback_ref: ANG-BASELINE-CR0-RESOURCES-001@sha256:EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569
---

# Exact objective

Create implementation-independent `ResourceInventory` and `ExecutionPlan` schemas plus synthetic constrained, workstation, and cluster cases with reserved probe-provenance reference fields. RESOURCE-PROBES delivery is deferred to a successor leaf; no probe schema, receipt, fixture, or real hardware probe is authorized.

## Required context and read set

Read only the literal `authorized_read_scope`. Do not inspect the host/GPU, enumerate files, follow reparse points, or read `outputs/**`.

## Versioned inputs and preconditions

- Base commit `903f9b9d5e58818d774604dbd6f4d89b2b4544e0`; `ANG-BP-ROOT@2`; `ANG-BP-RESOURCES@2` SHA-256 `796C1838973BB24B41416968D700DF2FD760A4BA7ECA854E7ECCB4E12B814F53`.
- Interface registry SHA-256 `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`; ADR-0004 SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6`.
- `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` identity `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, decision SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, disposition `SCAFFOLD_ACCEPTED`. `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`.
- Gate SHA-256 `AF52436A9E75850201622785206089B81570423B92A73E8C802B283E99F88E0B`; baseline SHA-256 `EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569`. All nine targets must remain absent before start.
- Revalidation 003 must be independently `APPROVED`, and Manifest v2 must separately be `PASS`/`authorized`. PENDING is NON-AUTHORIZING.

## Exact outputs and authorized write scope

- `schemas/control/v1/resources/resource-inventory.schema.json`
- `schemas/control/v1/resources/execution-plan.schema.json`
- `tests/synthetic/slice00/resources/constrained.inventory.json`
- `tests/synthetic/slice00/resources/workstation.inventory.json`
- `tests/synthetic/slice00/resources/cluster.inventory.json`
- `tests/synthetic/slice00/resources/invalid-overcommitted.plan.json`
- `tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1`
- `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md`

The executor owns only the eight `authorized_write_scope` paths and is denied the receipt. The accepted reviewer instance alone may create `RESOURCES.md` after the executor handoff; it is denied every executor/release-spec write.

## Non-goals and execution constraints

No `nvidia-smi`, WMI/CIM, benchmark, allocation, actual or synthetic probe execution/receipt, planner optimization, model/GPU call, resource mutation, network, package, external tool, background work, real/recovered data, deployment, status edit, or `outputs/**`. Reserved probe fields cannot assert that a probe ran. One foreground PowerShell process; no parallelism; 60 seconds per command, 10 minutes total, 1 logical CPU, 512 MiB working set, 25 MiB total new files, 5 MiB/file, 1 MiB/fixture; zero accelerator/network/package/spend/background use.

Author only through the host-provided Codex `apply_patch` primitive on the active role's literal scope. Shell permits read-only `Test-Path -LiteralPath`, `Get-FileHash -Algorithm SHA256`, and exact tests below. Shell redirection, ad-hoc writer scripts, bulk rewrites, cross-role writes, and undeclared commands are forbidden.

## Dependencies and status

The specification is `ready` but unusable while revalidation 003 is PENDING. Root alone records an `APPROVED` revalidation and changes the manifest to authorized; the executor cannot self-activate. Historical Evidence work is non-repeatable. Normal Evidence/Resource gates remain `NOT_RUN`.

## Predeclared tests and evidence

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Both Resource runs must produce identical case counts with zero failures and accept three synthetic tiers while rejecting overcommitment, missing headroom/identities, fabricated measurement/probe success, query-conditioned routing, and mid-transaction topology change. Evidence identity: `ANG-EVID-CR0-RESOURCES-<sha256>`.

## Human-impact mapping and acceptance gate

Bootstrap `ALLOW` covers these local schemas/cases only. Gate dispositions are exactly `SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`; generic aliases are invalid. The independent reviewer returned `ACK_ACCEPTED` for this final vocabulary and confirmed independence and reachability; this is a fulfilled precondition, not a gate disposition. Scaffold acceptance makes no normal Resource design, Human-Flourishing, Slice-00, M0, measured-plan, scientific, or deployment claim.

## Failure, rollback, and handoff

Stop on mismatch, missing/expired authority, undeclared path, collision, probe/host access, forbidden request, ceiling breach, policy drift, or any failed test. Revert only baseline `restore_or_remove` paths individually; preserve receipt/handoff, never broadly delete or unpack the archive. The handoff records the frozen executor, base, predecessor, files, hashes, commands, resources, results, limitations, and rollback, and must exist before independent review/decision.
