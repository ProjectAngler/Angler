---
blueprint_id: ANG-WORK-CR0-CONTINUITY-004
parent_id: ANG-BP-ROOT
revision: 1
tier: 4
design_status: approved
delivery_status: ready
activation_state: unusable_pending_revalidation
activation_revalidation: ANG-CR0-REVALIDATION-20260825-008
activation_base_commit: 21f7474ad40b46a6dc09ebab521f54c9089fbf50
construction_release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
release_manifest_version: 3
accountable_owner: ANG-BP-ROOT
human_authority: ANG-AUTH-PROJECT-OWNER-001
execution_owner: ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004
executor_task_id: 01a03a80-bb20-7d01-acf6-f50ca4856be5
authorized_write_scope_owner: ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004
independent_validator: ANG-AUTH-VALIDATOR-001
independent_gate_authority: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004
independent_gate_reviewer_session_ref: codex-subagent:/root/safety_change_map
independent_gate_write_scope_owner: ANG-AUTH-SAFETY-APPROVER-001
independent_gate_write_scope_instance: ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004
reviewer_vocabulary_ack: ACK_ACCEPTED
result_recorder: ANG-BP-ROOT
test_id: ANG-TEST-CR0-CONTINUITY-004
gate: ANG-GATE-CR0-CONTINUITY-004@1
gate_sha256: CAEB94B1C559E9D01A7836CB7D5DE55CB7E65D473F9C23A7E3C1CD464A1B56A6
baseline: ANG-BASELINE-CR0-CONTINUITY-004
rollback_ref: ANG-BASELINE-CR0-CONTINUITY-004@sha256:8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC
authorization_profile: ANG-POL-LOCAL-SCAFFOLD-001@1
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
normal_evidence_gate: ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN
normal_resources_gate: ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN
permitted_authoring_mechanism: Codex apply_patch limited to authorized_write_scope
authorized_write_scope:
  - docs/blueprints/ROOT_CAPSULE.md
  - docs/blueprints/STATUS.md
  - docs/blueprints/branches/resources/CAPSULE.md
  - docs/blueprints/branches/resources/STATUS.md
  - docs/blueprints/branches/evidence/BLUEPRINT.md
  - docs/blueprints/branches/evidence/CAPSULE.md
  - docs/blueprints/branches/evidence/STATUS.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md
  - docs/blueprints/BLUEPRINT_INDEX.json
  - docs/blueprints/TREE.md
  - docs/blueprints/TRACEABILITY.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-004.md
  - tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1
  - docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-004.md
executor_denied_write_scope:
  - docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md
  - tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1
  - docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-002.md
  - docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md
  - docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-003.md
  - tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-003.ps1
  - docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-003.md
  - docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-003.md
  - docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md
independent_gate_write_scope:
  - docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md
updated_at: 2026-08-25
---

# CR0 continuity reconciliation leaf

## Exact objective

Repair only the confirmed continuation-projection inconsistency left after the independently accepted Evidence and Resources CR0 scaffolds. Update thirteen existing documentation projections, add one append-only Evidence post-decision handoff, add one deterministic cross-file consistency test, and add one executor handoff. Do not rerun, modify, reinterpret, or replace either historical scaffold or its evidence.

## Why the existing LOW authority applies

ADR-0002 already permits project-local design artifacts, deterministic validators, and local tests; `ANG-POL-LOCAL-SCAFFOLD-001@1` and `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1` classify that exact class as LOW only under literal paths, synthetic/no-person data, no external effect, and exact rollback. This leaf narrows that ceiling to sixteen text files, existing local PowerShell, no model/GPU/probe/network/package/dependency/recovered/personal data, and stricter resource limits. It adds no new capability, data source, subject, external effect, contract, ADR, threshold, technical claim, or authority.

Any need to alter an ADR, policy, assessment, contract, historical artifact, source/schema/fixture, path, command, ceiling, role, or disposition is a scope change. Stop and `ESCALATE`; this leaf cannot approve its own expansion.

## Required context and exact read scope

The executor may read only the following repository-local paths for this leaf:

- `AGENTS.md`
- `PROJECT_BLUEPRINT.md`
- `docs/blueprints/PROTOCOL.md`
- `docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md`
- `docs/blueprints/releases/construction-0/MANIFEST.md`
- `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-004.json`
- `docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-004.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-008.md`
- `docs/blueprints/work/slice-00/validate-construction-release-0-v3-008.ps1`
- `docs/blueprints/decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md`
- `docs/blueprints/branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md`
- `docs/blueprints/branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md`
- `docs/blueprints/branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md`
- `docs/blueprints/branches/resources/BLUEPRINT.md`
- `docs/blueprints/branches/evidence/BLUEPRINT.md`
- `docs/blueprints/branches/evidence/CAPSULE.md`
- `docs/blueprints/branches/evidence/STATUS.md`
- `docs/blueprints/branches/evidence/children/evidence-schemas/work/ANG-WORK-EVIDENCE-SCHEMAS-001.md`
- `docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md`
- `docs/blueprints/branches/evidence/children/evidence-schemas/gates/ANG-GATE-EVIDENCE-SCHEMAS-001.md`
- `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-EVIDENCE-SCHEMAS-001.json`
- `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`
- `artifacts/control-plane/evidence-schemas/test-receipt.json`
- `artifacts/control-plane/evidence-schemas/effect-receipt.json`
- `artifacts/control-plane/evidence-schemas/HANDOFF.md`
- the fourteen Evidence implementation/schema/test/fixture paths enumerated and hashed in `test-receipt.json`
- `docs/blueprints/work/slice-00/ANG-WORK-CR0-RESOURCES-001.md`
- `docs/blueprints/branches/resources/gates/ANG-GATE-CR0-RESOURCES-001.md`
- `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-002.json`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003-decision.json`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004-decision.json`
- `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md`
- the seven Resources schema/fixture/test paths enumerated and hashed in `RESOURCES.md`
- `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-002.json`
- `docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-002.md`
- `docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-002.md`
- `docs/blueprints/work/slice-00/validate-construction-release-0-v3-006.ps1`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-006.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-006-decision.json`
- `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-002.md`
- `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md`
- `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-003.json`
- `docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-003.md`
- `docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-003.md`
- `docs/blueprints/work/slice-00/validate-construction-release-0-v3-007.ps1`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007-decision.json` (must be absent)
- `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-003.md` (must be absent)
- `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-003.ps1` (must be absent)
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-003.md` (must be absent)
- `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-003.md` (must be absent)
- the thirteen pre-existing projection paths in `authorized_write_scope`
- `tools/validate_blueprint_tree.ps1`

No ad hoc directory enumeration, unrelated sibling design, `outputs/**`, recovered material, personal/credential/host area, model, package cache, or external resource is in scope. The only permitted traversal is traversal performed internally by the exact declared `tools/validate_blueprint_tree.ps1` command.

## Frozen historical inputs

### Revalidation 005 — approved review but failed authorization validation

- Reviewed PENDING Manifest SHA-256: `08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F`.
- Independent decision SHA-256: `467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3`; disposition `APPROVED` for review only.
- Attempted authorized Manifest SHA-256: `9578BD1B97451636449254CF1496387DA9602240B51401CC85595237E657C3E5`.
- Validator SHA-256: `EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D`; preserved commit `84ff08b484197755c3fed66e7dc06988539e456e`.
- Exact failed command: `pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0-v3.ps1`; exit 1; output `Write-Error: Manifest-v3 authority/non-equivalence is missing required literal: ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS`.
- No continuity leaf, test, target output, or reviewer receipt ran or existed. Because the required authorized validation failed, revalidation 005 grants no execution authority and cannot be retried or reinterpreted.

### Evidence scaffold — accepted and non-repeatable

- Decision identity/SHA-256: `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0` / `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`; disposition `SCAFFOLD_ACCEPTED`.
- Historical leaf SHA-256: `5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289`.
- Historical gate SHA-256: `A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5`.
- Historical baseline SHA-256: `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F`.
- Test receipt SHA-256: `897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53`.
- Effect receipt SHA-256: `9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893`.
- Executor handoff SHA-256: `017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855`.
- Execution-time Manifest-v2 SHA-256: `802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2`.
- All fourteen implementation/schema/test/fixture hashes are the exact map in the immutable test receipt and must rehash 14/14.

### Resources scaffold — accepted and non-repeatable

- Receipt SHA-256: `D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B`; disposition `SCAFFOLD_ACCEPTED`.
- Authorized Manifest-v2 SHA-256: `35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41`.
- Leaf/gate/baseline SHA-256: `67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA` / `A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2` / `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`.
- Revalidation-004 decision SHA-256: `12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9`.
- Resource inventory schema: `92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F`.
- Execution-plan schema: `7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2`.
- Constrained/workstation/cluster fixtures: `CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956` / `162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9` / `AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85`.
- Invalid-plan fixture: `92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A`.
- Resource test: `047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA`.
- Executor handoff: `916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A`.

Revalidation 003 remains immutable `REJECTED` evidence: spec `1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9`, decision `9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A`, predecessor baseline `EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569`.

### Continuity-002 — rejected and immutable

- Failed packet commit: `c98fbe85ceebb7bddd167b33b5a7459ce54110bc`.
- Reviewer receipt: `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md`, SHA-256 `BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173`, disposition `RECONCILIATION_REJECTED`.
- Rejected addendum: `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md`, SHA-256 `8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8`.
- Rejected executor handoff: `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-002.md`, SHA-256 `42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C`.
- Rejected test identity: `1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC`; its old path is absent after rollback and may not be recreated or reused.
- The rejection found six mangled Evidence bindings in the addendum and an incomplete executor handoff/test assurance record. These preserved files are failure evidence, never correction targets. The thirteen projections alone were restored to the baseline-002 bytes.

### Packet 007 — audit-failed before activation

- Clean preservation commit: `21f7474ad40b46a6dc09ebab521f54c9089fbf50`.
- PENDING Manifest SHA-256: `7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33`.
- Baseline-003/gate-003/leaf-003 SHA-256: `134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28` / `C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9` / `D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD`.
- Validator-007/spec-007 SHA-256: `A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B` / `D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D`.
- Gate-003 front matter pinned baseline-003 correctly, but its entry body used stale hash `FE2E12E1E4F5536E5978C80B43A760886C19DF71D4586B63C8EEDB312B0DAC2D`; validator-007 froze the whole-file hash without validating that semantic body binding. No decision, leaf, test, projection, addendum, handoff, or receipt ran. Packet 007 is immutable failed evidence, grants no authority, and cannot be edited or retried.

## Exact executor write scope

Executor `ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004`, persistent task `01a03a80-bb20-7d01-acf6-f50ca4856be5`, may author only these sixteen paths with Codex `apply_patch`:

1. `docs/blueprints/ROOT_CAPSULE.md`
2. `docs/blueprints/STATUS.md`
3. `docs/blueprints/branches/resources/CAPSULE.md`
4. `docs/blueprints/branches/resources/STATUS.md`
5. `docs/blueprints/branches/evidence/BLUEPRINT.md`
6. `docs/blueprints/branches/evidence/CAPSULE.md`
7. `docs/blueprints/branches/evidence/STATUS.md`
8. `docs/blueprints/branches/evidence/children/evidence-schemas/BLUEPRINT.md`
9. `docs/blueprints/branches/evidence/children/evidence-schemas/CAPSULE.md`
10. `docs/blueprints/branches/evidence/children/evidence-schemas/STATUS.md`
11. `docs/blueprints/BLUEPRINT_INDEX.json`
12. `docs/blueprints/TREE.md`
13. `docs/blueprints/TRACEABILITY.md`
14. `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-004.md`
15. `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1`
16. `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-004.md`

The executor is explicitly denied `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md`. Only reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004`, session `codex-subagent:/root/safety_change_map`, may create that one path after the executor handoff. Reviewer and recorder cannot edit executor outputs; root recorder cannot alter the receipt disposition.

That fresh reviewer instance is independent and reachable and has returned `ACK_ACCEPTED` specifically for the successor vocabulary `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`, the receipt-only write scope, and the requirement to rerun the consolidated test and tree validator once after the executor handoff. No continuity-002 acknowledgement or disposition is inherited.

## Required reconciled content

- Root, Resources, Evidence ancestor, Evidence-Schemas, index, tree, and traceability projections record both accepted scaffold identities and eliminate all stale pre-execution/PENDING directions.
- `ANG-BP-EVIDENCE-SCHEMAS` advances to revision 2 only to record historical delivery state and the one-shot continuity repair; its contracts, normal gate, CR0 scaffold gate, thresholds, and implementation semantics do not change.
- `CAPSULE.md` advances to capsule revision 5 against blueprint revision 2; `STATUS.md` records status revision 2. Both identify the old leaf as historical/non-repeatable, not ready/executable.
- The fresh append-only corrective addendum binds the immutable Evidence decision, leaf, gate, baseline, execution-time Manifest-v2, test receipt, effect receipt, original handoff, all 14/14 implementation identities, base commit `903f9b9d5e58818d774604dbd6f4d89b2b4544e0`, the complete Resources binding/output map, continuity-002 rejection lineage, and non-equivalence. It explicitly corrects the six false hashes without editing, superseding, deleting, or relabeling the rejected addendum, rejected handoff, rejected receipt, original Evidence handoff, or independent decisions.
- Every successor projection, gate, leaf, Manifest, specification, test, handoff, and receipt must use authoritative baseline-004 SHA-256 `8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC`. The test must reject the stale packet-007 gate-body binding and every competing hash paired with exact ID `ANG-BASELINE-CR0-CONTINUITY-004`.
- Resources projections bind receipt `D7221A84...24B4B`, its eight outputs/handoff, synthetic/unmeasured limitation, and no-probe/non-equivalence state.
- `BLUEPRINT_INDEX.json`, `TREE.md`, and `TRACEABILITY.md` agree with the exact revisions, decisions, receipts, historical status, current one-shot continuity leaf, and future successor-manifest requirement.
- Every projection states normal Evidence, Resources, and Human-Flourishing gates `NOT_RUN`; Slice 00 and M0 `NOT_PASSED`. No wording may imply branch completion, measured-resource evidence, SAFETY activation, scientific success, production, or external-use authority.

## Commands and numeric ceilings

Pre-start verification may use only read-only `Test-Path -LiteralPath` on the seventeen baseline targets and `Get-FileHash -Algorithm SHA256` on named packet, baseline, and frozen historical files. Authoring uses only Codex `apply_patch` on the exact role-owned paths.

After the thirteen projections, append-only addendum, and deterministic test exist as the fifteen non-handoff outputs, run sequentially from repository root. These are the exact two executor commands; their single literal child PowerShell process, processor-affinity assignment, foreground polling, and no-file metric line are the sole exceptions to the subprocess denial:

```powershell
pwsh -NoProfile -NonInteractive -Command '$scriptPath="tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1"; $sw=[Diagnostics.Stopwatch]::StartNew(); $p=Start-Process -FilePath pwsh -ArgumentList @("-NoProfile","-NonInteractive","-File",$scriptPath) -NoNewWindow -PassThru; $p.ProcessorAffinity=[IntPtr]1; $peak=0L; while(-not $p.HasExited){$p.Refresh(); $peak=[Math]::Max($peak,$p.WorkingSet64); if($sw.ElapsedMilliseconds -gt 60000 -or $peak -gt 536870912){$p.Kill($true); throw "ANG-CEILING"}; Start-Sleep -Milliseconds 25}; $p.WaitForExit(); $p.Refresh(); $sw.Stop(); $peak=[Math]::Max($peak,$p.PeakWorkingSet64); "ANG-METRICS path=$scriptPath exit=$($p.ExitCode) wall_ms=$($sw.ElapsedMilliseconds) cpu_ms=$([int64]$p.TotalProcessorTime.TotalMilliseconds) peak_bytes=$peak affinity=1"; if($p.ExitCode -ne 0 -or $sw.ElapsedMilliseconds -gt 60000 -or $peak -gt 536870912){exit 1}'
pwsh -NoProfile -NonInteractive -Command '$scriptPath="tools/validate_blueprint_tree.ps1"; $sw=[Diagnostics.Stopwatch]::StartNew(); $p=Start-Process -FilePath pwsh -ArgumentList @("-NoProfile","-NonInteractive","-File",$scriptPath) -NoNewWindow -PassThru; $p.ProcessorAffinity=[IntPtr]1; $peak=0L; while(-not $p.HasExited){$p.Refresh(); $peak=[Math]::Max($peak,$p.WorkingSet64); if($sw.ElapsedMilliseconds -gt 60000 -or $peak -gt 536870912){$p.Kill($true); throw "ANG-CEILING"}; Start-Sleep -Milliseconds 25}; $p.WaitForExit(); $p.Refresh(); $sw.Stop(); $peak=[Math]::Max($peak,$p.PeakWorkingSet64); "ANG-METRICS path=$scriptPath exit=$($p.ExitCode) wall_ms=$($sw.ElapsedMilliseconds) cpu_ms=$([int64]$p.TotalProcessorTime.TotalMilliseconds) peak_bytes=$peak affinity=1"; if($p.ExitCode -ne 0 -or $sw.ElapsedMilliseconds -gt 60000 -or $peak -gt 536870912){exit 1}'
```

The consolidated continuity run must report exactly 64 cases and zero failures; the tree validator must pass once. The test must not require the not-yet-authored executor handoff. Only after those two commands pass may the executor author its handoff as output sixteen. Do not run either historical Evidence or Resources construction test.

One foreground command at a time; no parallelism, background process, or nested subprocess tree. The exact foreground measurement wrappers above may create only their one named child, bind it to processor affinity mask 1, poll only that process, and create no file. Ceilings are 60 seconds per command; 600 seconds aggregate active time; 1 logical CPU; 512 MiB peak working set; 2 MiB total changed/new bytes; 512 KiB per file; network/GPU/packages/tools/models/spend/background work all zero. Each wrapper's `ANG-METRICS` line is the predeclared wall-time/CPU/peak-working-set evidence; file lengths and baseline states are the predeclared byte evidence. Stop on a missing metric, ceiling breach, undeclared path/command/read/write, hash mismatch, stale ambiguity, protected data, or inability to perform literal rollback.

## Explicit non-goals and mandatory denials

- No schema, source, fixture, historical test, decision, receipt, handoff, baseline, ADR, policy, assessment, contract, normal gate, or threshold change.
- No Evidence or Resources scaffold rerun, re-acceptance, normal delivery, ARTIFACT-LINEAGE activation, SAFETY activation, Slice/M0 advancement, experiment, model/learner/runtime/tool/environment work, probe/host enumeration, deployment, promotion, or scientific claim.
- No network, DNS, API, telemetry, dashboard, remote Git, package/dependency/plugin/tool installation, model/GPU use, recovered `outputs/**`, real-person/credential/personal/unknown-provenance data, host area, ad hoc directory enumeration, background/persistent process, external effect, shell redirection, ad-hoc writer, bulk rewrite, cache/temp/bytecode output, broad deletion, staging, or commit. Traversal internal to the exact declared tree-validator command is the sole traversal exception. The two literal foreground metric wrappers above are the sole one-child process exceptions and are not persistent monitoring or telemetry.

## Tests, handoff, gate, and completion

The continuity test must deterministically cover the gate's complete positive and negative matrix in exactly 64 named cases. It may read only declared text/JSON files and must not write. It must mechanically check every frozen Evidence binding (leaf, gate, baseline, decision, execution-time Manifest-v2, test receipt, effect receipt, and original handoff), rehash the full fourteen-file Evidence implementation map and require 14/14, check the complete Resources receipt/leaf/gate/baseline/decision/Manifest/seven-output/handoff map, bind failed-005, rejected continuity-002, and audit-failed packet-007 lineage, and verify exact projections, role/write-scope separation, rollback classes, and non-equivalence. Predeclared in-memory negative mutations must cover each Evidence binding, an implementation hash, each Resources binding class, rejection identity/disposition, the stale packet-007 baseline-body pair, a competing continuity-004 baseline hash, each stale projection class, role collision, scope expansion, generic aliases, and every false normal-gate/Slice/M0 claim.

The executor handoff records executor/task, exact authorized Manifest-v3 hash, baseline/gate/leaf hashes, all 17 pre-start target identities, all 16 post-write executor-output identities, exact commands/results/64-case count/wall times, per-file and aggregate changed/new bytes, the 60-second/600-second/1-CPU/512-MiB/2-MiB/512-KiB ceilings and observation method, effects, rollback state, all immutable identities and maps including packet-007 hashes/failure class, the authoritative baseline-004 body binding, and reviewer identity/path. It explicitly says the independent receipt is pending and makes no disposition. If any time, byte, CPU, or memory ceiling cannot be verified without an undeclared tool or write, the executor stops and records the limitation; it may not claim measured compliance.

Only the independent reviewer may decide `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`. `RECONCILIATION_ACCEPTED` completes only this projection repair and exhausts Manifest v3; it does not activate another leaf. A later successor manifest and revalidation are required for SAFETY or any further construction.

## Failure and rollback

Stop immediately. Preserve the fresh append-only corrective addendum, executor handoff, and any successor reviewer receipt. Preserve the rejected continuity-002 addendum, handoff, and receipt byte-for-byte and preserve all packet-007 files/commit as audit-failed pre-activation evidence. Restore each of the thirteen pre-existing projections individually to its exact base hash. Restore or remove only the fresh continuity-004 test according to its absent baseline; never recreate or reuse either earlier test path. Never broadly or recursively delete, restore, or extract the archive. Preserve all historical evidence and failed repair evidence. All normal gates, Slice 00, and M0 remain unpassed.
