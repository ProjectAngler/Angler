---
blueprint_id: ANG-WORK-CR0-RESOURCES-001
parent_id: ANG-BP-RESOURCES
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: blocked
unblock_condition: ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001 records SCAFFOLD_ACCEPTED and its independent decision hash is pinned
accountable_owner: ANG-BP-RESOURCES
execution_owner: ANG-BP-RESOURCES
independent_verifier: ANG-BP-SAFETY
updated_at: 2026-08-25
gate: ANG-GATE-CR0-RESOURCES-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Specify implementation-independent `ResourceInventory` and `ExecutionPlan` identities, including reserved probe-provenance reference fields, and prove them against synthetic constrained, workstation, and cluster fixtures. RESOURCE-PROBES delivery is deferred to a successor leaf; no probe schema, receipt, fixture, or real hardware probe is authorized here.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [RESOURCES blueprint revision 2](../../branches/resources/BLUEPRINT.md), [interface registry](../../INTERFACE_REGISTRY.md), [EVIDENCE capsule](../../branches/evidence/CAPSULE.md), [ADR-0002](../../decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md), [ADR-0004](../../decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md), [manifest](../../releases/construction-0/MANIFEST.md), and [SAFETY-owned policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md). Read the predecessor scaffold decision only after it exists. Do not inspect the actual host/GPU or `outputs/**`.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`; `ANG-BP-RESOURCES@2`, SHA-256 `796C1838973BB24B41416968D700DF2FD760A4BA7ECA854E7ECCB4E12B814F53`.
- Interface-registry snapshot `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`.
- `ANG-ADR-0004`, SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6`; ADR/policy/manifest version 2.
- Required predecessor: independent `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json` with `SCAFFOLD_ACCEPTED` and content identity `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-<sha256>`, pinned by the root steward before this leaf becomes `ready`. It is bootstrap scaffold evidence only; `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`.
- All inventory values are explicitly marked `synthetic`; a request for measured capability stops work.

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

## Non-goals and execution constraints

No `nvidia-smi`, WMI/CIM inventory, filesystem enumeration, benchmark, allocation, actual or synthetic probe execution/receipt, probe schema, placement, planner optimization, model selection, GPU call, or resource mutation. Probe-provenance fields are reserved references only and cannot assert that a probe ran. No network, packages, external tools, background work, real data, `outputs/**`, deployment, or status edits. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

Blocked until `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` has an independent `SCAFFOLD_ACCEPTED` decision whose exact content hash is pinned. The normal `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN` and cannot unblock CR0. Root steward then changes status to `ready`; no executor may self-unblock.

## Predeclared tests and evidence

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Tests must accept all three synthetic inventory tiers, reject an overcommitted plan, require rollback/evaluation headroom, bind every plan to exactly one inventory/evidence identity, validate reserved probe-provenance reference shape without manufacturing probe evidence, and treat any `measured=true`, probe-success claim, query-conditioned model routing, or mid-transaction topology change as invalid. Evidence: `ANG-EVID-CR0-RESOURCES-<sha256>`.

## Human-impact mapping and acceptance gate

Bootstrap `ALLOW` covers schemas/synthetic profiles only; it does not authorize actual resource discovery. `ANG-GATE-CR0-RESOURCES-001` passes when exact outputs/tests succeed and all profiles use the same logical contracts without making 16 GB or any tier an architecture ceiling. Formal flourishing, RESOURCES design, Slice-00, and M0 gates remain pending.

## Failure, rollback, and handoff

Stop on input/archive mismatch, missing predecessor evidence, undeclared path, collision, any actual probe, forbidden capability/data request, resource ceiling, policy change, or twice-failing test. Revert only listed paths; never unpack the archive automatically. The handoff records executor, predecessor evidence, files, commands, resource observations, results, limitations, and next review.
