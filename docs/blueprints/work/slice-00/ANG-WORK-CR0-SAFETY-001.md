---
blueprint_id: ANG-WORK-CR0-SAFETY-001
parent_id: ANG-BP-SAFETY
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: not_ready
unblock_condition: release steward binds one executor identity and rechecks ANG-GATE-CONSTRUCTION-RELEASE-0-001
accountable_owner: ANG-BP-SAFETY
execution_owner: ANG-BP-SAFETY
independent_verifier: ANG-BP-ROOT
updated_at: 2026-08-25
gate: ANG-GATE-CR0-SAFETY-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Reverify the SAFETY-owned CR0 policy, assessment, construction gate, and rollback identity against this release manifest, then emit a release-scoped receipt and synthetic scope-denial fixtures. This leaf does not pass the SAFETY design gate or formal human-flourishing gate.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [SAFETY blueprint revision 2](../../branches/safety/BLUEPRINT.md), [constitution revision 1](../../HUMAN_FLOURISHING_CONSTITUTION.md), [impact contract version 1](../../branches/safety/contracts/ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001.md), [bootstrap gate version 1](../../branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md), [ADR-0002](../../decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md), [release manifest](../../releases/construction-0/MANIFEST.md), and [SAFETY-owned local policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md). Target packet: at most 7,400 tokens; do not read sibling branches or `outputs/**`.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`, `ANG-BP-SAFETY@2` (SHA-256 `4376A7C61D1CAFFB25671858A79C1A61185721EE850620E4A932B6D5002F8A5D`), and `ANG-CON-HUMAN-FLOURISHING-001@1`.
- `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1`, `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1`, and `ANG-GATE-CONSTRUCTION-RELEASE-0-001@1`; this leaf verifies but cannot alter them.
- `ANG-ADR-0002`, `ANG-POL-LOCAL-SCAFFOLD-001@1`, and release manifest version 2.
- Rollback archive exists with the frontmatter hash. Any mismatch stops work.

## Exact outputs and authorized write scope

Only these paths may be created or changed:

- `docs/blueprints/releases/construction-0/branch-receipts/SAFETY.md`
- `tests/synthetic/slice00/safety/valid-release-reference.json`
- `tests/synthetic/slice00/safety/invalid-scope-expansion.json`
- `tests/synthetic/slice00/safety/invalid-expired-authority.json`
- `tests/synthetic/slice00/safety/Test-Cr0SafetyReleaseReference.ps1`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-SAFETY-001.md`

## Non-goals and execution constraints

Do not implement permissions, sandboxing, promotion, deployment, dependency intake, model/runtime controls, or a replacement constitution/gate. Do not touch root/branch status, `outputs/**`, recovered code, packages, network, external tools, models/GPUs, real-person data, or background processes. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

No construction leaf dependency. Status is `not_ready` until the release steward records one concrete executor identity and rechecks `ANG-GATE-CONSTRUCTION-RELEASE-0-001`; the executor cannot self-activate.

## Predeclared tests and evidence

Run exactly:

```powershell
pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/safety/Test-Cr0SafetyReleaseReference.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

The canonical test must remain passing. The release-reference test must accept the exact valid local scope and reject 100% of fixtures requesting expiry bypass, undeclared paths, `outputs/**`, network, packages, external tools, models/GPUs, real data, background work, deployment, or milestone/gate credit. Evidence ID: `ANG-EVID-CR0-SAFETY-<sha256>`.

## Human-impact mapping and acceptance gate

Bootstrap work is covered by `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1` under `ANG-POL-LOCAL-SCAFFOLD-001@1`. Formal `ANG-GATE-HUMAN-FLOURISHING-001` remains pending. `ANG-GATE-CR0-SAFETY-001` passes only when all exact outputs exist, both commands pass, authority is external to the scaffolded artifact, and every forbidden case fails closed. Passing grants no SAFETY, Slice-00, or M0 pass.

## Failure, rollback, and handoff

Stop on an input/archive hash mismatch, undeclared path, concurrent file collision, forbidden capability/data request, resource ceiling, authority ambiguity, policy change, or twice-failing test. Preserve the failing fixture/result, revert only listed paths to pre-leaf hashes, and report the blocker. Never unpack the common archive automatically. The handoff records executor, base, files, commands, elapsed/resource use, evidence hash, dissent, and next review action.
