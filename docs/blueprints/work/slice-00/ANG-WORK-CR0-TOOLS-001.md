---
blueprint_id: ANG-WORK-CR0-TOOLS-001
parent_id: ANG-BP-TOOLS
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: not_ready
unblock_condition: M3 and a successor construction release explicitly authorize tool scaffolding
accountable_owner: ANG-BP-TOOLS
execution_owner: ANG-BP-TOOLS
independent_verifier: ANG-BP-SAFETY
updated_at: 2026-08-25
gate: ANG-GATE-CR0-TOOLS-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Record and mechanically test the Release-0 TOOLS deferral boundary: tool packages may be described as inert draft metadata, but no tool, sandbox, dependency, execution, promotion, or external capability may be activated.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [TOOLS blueprint revision 1](../../branches/tools/BLUEPRINT.md), [SAFETY capsule](../../branches/safety/CAPSULE.md), [interface registry](../../INTERFACE_REGISTRY.md), [ADR-0002](../../decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md), [manifest](../../releases/construction-0/MANIFEST.md), and [SAFETY-owned policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md), plus predecessor receipts. Never inspect installed tools, packages, `outputs/**`, or recovered code.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`; `ANG-BP-TOOLS@1`, SHA-256 `DCFF8772870E1C6FBC7F4BCD445883B021724E4DE5BEF0CF2B8B386DC350E40A` with delivery status `deferred`.
- Interface snapshot `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`.
- Accepted predecessor evidence IDs; ADR/policy/manifest version 2.
- Completion must leave the Tier-1 TOOLS delivery status deferred.

## Exact outputs and authorized write scope

- `docs/blueprints/branches/tools/work/CR0_DEFERRED_BOUNDARY.md`
- `schemas/control/v1/tools/tool-package-draft.schema.json`
- `tests/synthetic/slice00/tools/valid-deferred-package.json`
- `tests/synthetic/slice00/tools/invalid-executable-package.json`
- `tests/synthetic/slice00/tools/invalid-network-permission.json`
- `tests/synthetic/slice00/tools/invalid-dependency-request.json`
- `tests/synthetic/slice00/tools/Test-Cr0ToolsDeferral.ps1`
- `docs/blueprints/branches/tools/work/CR0_HANDOFF.md`

## Non-goals and execution constraints

No tool code, execution, registry, sandbox, workshop, package installation/import, dependency/recovered-code intake, network, external application, child process, background work, model/GPU, real data, deployment, promotion, status change, or `outputs/**` access. The draft schema must require `lifecycle_status: DEFERRED` and deny executable material. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

Not ready and not an activation candidate in CR0. It remains deferred until M3 and a successor release; predecessor progress cannot activate it.

## Predeclared tests and evidence

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/tools/Test-Cr0ToolsDeferral.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Tests accept inert deferred metadata only and reject executable/source/binary payloads, invocation commands, permissions, network, package/dependency requests, promoted status, or runtime resolution. Evidence: `ANG-EVID-CR0-TOOLS-<sha256>`.

## Human-impact mapping and acceptance gate

Bootstrap assessment authorizes documenting and testing the prohibition, not tools. `ANG-GATE-CR0-TOOLS-001` passes when exact outputs/tests succeed and all activation cases fail closed. Formal flourishing, TOOLS design/delivery, Slice-00, M0, M6, and tool promotion remain pending.

## Failure, rollback, and handoff

Stop on input/archive or predecessor mismatch, undeclared path, collision, any request to execute/inspect/install/connect a tool or use forbidden data/capability, resource ceiling, policy change, or twice-failing test. Revert only listed files. Handoff records executor, predecessors, files, commands, resources, results, continued deferral, and next review.
