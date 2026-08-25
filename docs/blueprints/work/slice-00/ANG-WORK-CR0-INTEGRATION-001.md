---
blueprint_id: ANG-WORK-CR0-INTEGRATION-001
parent_id: ANG-BP-ROOT
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: blocked
unblock_condition: CR0 safety, evidence scaffold, resources, worlds, and science decisions pass/accept with exact receipt hashes pinned
accountable_owner: ANG-BP-ROOT
execution_owner: ANG-BP-EVIDENCE
independent_verifier: ANG-BP-ROOT
updated_at: 2026-08-25
gate: ANG-GATE-CR0-INTEGRATION-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Assemble one synthetic, non-executable Slice-00 control record that references every branch scaffold, then prove identity, authority, visibility, resource-plan, policy, and rollback mismatches fail closed.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [protocol](../../PROTOCOL.md), [interface registry](../../INTERFACE_REGISTRY.md), [integration spine](../../INTEGRATION_SPINE.md), [ADR-0004](../../decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md), [release manifest](../../releases/construction-0/MANIFEST.md), [SAFETY-owned impact assessment](../../branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md), [SAFETY-owned policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md), and the five accepted predecessor handoffs/evidence summaries. Read RUNTIME/LEARNING/TOOLS leaf headers only to verify `not_ready`; do not read unrelated source trees, `outputs/**`, recovered material, models, or real datasets.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`; interface snapshot `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`; spine snapshot `78EC4F6909E8D2BDE478FD2557147C0933BE7940F066381995813660A2B53833`.
- `ANG-ADR-0004`, SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6`.
- Release manifest version 2, assessment/policy version 1, and rollback archive hash from frontmatter.
- Five immutable predecessor receipts (SAFETY, EVIDENCE-SCAFFOLD, RESOURCES, WORLDS, SCIENCE) and exact produced contract/schema versions. EVIDENCE must be the independent `SCAFFOLD_ACCEPTED` decision at `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`; its exact hash is pinned, while `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`.
- No unresolved blocking dissent or overlapping write remains.

## Exact outputs and authorized write scope

- `docs/blueprints/releases/construction-0/SLICE_00_CONTROL_PROFILE.md`
- `docs/blueprints/releases/construction-0/CONSTRUCTION_RELEASE_0_HANDOFF.md`
- `schemas/control/v1/slice00/control-record.schema.json`
- `tests/synthetic/slice00/integration/valid-control-record.json`
- `tests/synthetic/slice00/integration/invalid-wrong-policy.json`
- `tests/synthetic/slice00/integration/invalid-wrong-plan.json`
- `tests/synthetic/slice00/integration/invalid-missing-branch.json`
- `tests/synthetic/slice00/integration/invalid-forbidden-scope.json`
- `tests/synthetic/slice00/integration/Test-Cr0Integration.ps1`

## Non-goals and execution constraints

Do not edit root/index/tree/spine/interface/branch status, resolve a formal gate, run a model/environment/tool, install packages, use network/external tools, create background work, inspect GPU/host, use real/recovered data, deploy, mutate runtime, or claim Slice-00/M0/scientific success. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

Blocked until all five predecessor gates pass and the root steward pins their receipts. RUNTIME, LEARNING, and TOOLS must remain `not_ready`. The integration executor cannot approve its own final gate; root review is required.

## Predeclared tests and evidence

Run every activated predecessor command recorded in accepted handoffs, then exactly:

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/integration/Test-Cr0Integration.ps1
pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

All valid fixtures must pass; 100% of missing predecessor/evidence, wrong policy/plan/parent/authority, expired scope, hidden-data visibility, forbidden-capability, or premature RUNTIME/LEARNING/TOOLS activation cases must fail. Evidence: `ANG-EVID-CR0-INTEGRATION-<sha256>`.

## Human-impact mapping and acceptance gate

The bootstrap assessment permits local integration evidence only. `ANG-GATE-CR0-INTEGRATION-001` passes when exact outputs/tests succeed, all branch receipts and input hashes match, and root-controlled review records the result. It closes Construction Release 0 only. Formal `ANG-GATE-HUMAN-FLOURISHING-001`, `ANG-SLICE-00-CONTROL`, Tier-1 design gates, and M0 remain `NOT_PASSED`.

## Failure, rollback, and handoff

Stop on input/archive or receipt mismatch, undeclared path, collision, forbidden capability/data request, resource ceiling, policy change, formal-gate conflation, or twice-failing test. Preserve the invalid record and stop all dependent closure. Revert only listed integration paths. Root/human authority alone may restore the full archive. Final handoff records branch receipts, commands, resources, pass/fail, dissent, root files needing updates, and the next separately authorized release action.
