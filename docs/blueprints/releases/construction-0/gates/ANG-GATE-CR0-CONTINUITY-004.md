---
gate_id: ANG-GATE-CR0-CONTINUITY-004
version: 1
owner: ANG-BP-ROOT
status: specified
activation_state: unusable_pending_revalidation
activation_revalidation: ANG-CR0-REVALIDATION-20260825-008
gate_class: bootstrap_continuity_reconciliation
release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
release_manifest_version: 3
leaf: ANG-WORK-CR0-CONTINUITY-004@1
baseline: ANG-BASELINE-CR0-CONTINUITY-004
baseline_sha256: 8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC
executor: ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004
executor_task_id: 01a03a80-bb20-7d01-acf6-f50ca4856be5
independent_verifier: ANG-AUTH-VALIDATOR-001
independent_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
independent_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004
independent_reviewer_session_ref: codex-subagent:/root/safety_change_map
reviewer_vocabulary_ack: ACK_ACCEPTED
allowed_dispositions: RECONCILIATION_ACCEPTED|RECONCILIATION_REJECTED|ESCALATE
test_id: ANG-TEST-CR0-CONTINUITY-004
result_recorder: ANG-BP-ROOT
decision_writer: ANG-AUTH-SAFETY-APPROVER-001
decision_path: docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
normal_evidence_gate: ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN
normal_resources_gate: ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN
---

# CR0 continuity reconciliation gate

## Claim and strict non-equivalence

This gate tests only whether sixteen documentation/test outputs reconcile Project Angler's continuation projections with two already accepted, immutable CR0 scaffold decisions while preserving their exact historical identities and authority boundaries. It neither reruns nor re-decides either scaffold.

`RECONCILIATION_ACCEPTED` means only that the listed projections consistently record the historical Evidence decision and Resources receipt and that future workers will not be directed to rerun completed leaves. It is not `SCAFFOLD_ACCEPTED`; it does not pass the normal Evidence or Resources gate, `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a scientific gate, or any branch delivery gate. It creates no SAFETY-leaf, model, learner, probe, network, deployment, production, promotion, or external-use authority. A later successor manifest is mandatory before SAFETY or any other construction leaf can activate.

## Entry criteria

- Manifest v3 is `authorized` with revalidation `ANG-CR0-REVALIDATION-20260825-008` recorded `PASS`, and its separately authored decision exists with exactly `APPROVED` and its bound hash.
- `ANG-WORK-CR0-CONTINUITY-004@1` is the sole ready executable leaf in Manifest v3. Executor `ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-004` is prebound to persistent task `01a03a80-bb20-7d01-acf6-f50ca4856be5` in both leaf and manifest.
- `ANG-BASELINE-CR0-CONTINUITY-004` SHA-256 `8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC` is reverified immediately before the first write: thirteen targets exist with their exact hashes and four targets are absent.
- ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001@1`, and `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1` remain current. This leaf is a narrower documentation-only instance of their existing LOW local-scaffold ceiling and adds no capability, data source, effect, resource, or authority.
- The accepted Evidence decision, its bound leaf/gate/baseline/test/effect/handoff/execution-manifest identities, the accepted Resources receipt and its eight output/handoff identities, rejected revalidation 003, and approved revalidation 004 all match the immutable hashes in Manifest v3.
- Revalidation 005 remains immutable failed-authorization history: reviewed PENDING Manifest SHA-256 `08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F`, decision SHA-256 `467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3`, attempted authorized Manifest SHA-256 `9578BD1B97451636449254CF1496387DA9602240B51401CC85595237E657C3E5`, validator SHA-256 `EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D`, and preserved commit `84ff08b484197755c3fed66e7dc06988539e456e`. Its authorized validator exited 1 with `Write-Error: Manifest-v3 authority/non-equivalence is missing required literal: ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS`; no leaf, test, output, or receipt ran, so 005 grants no execution authority.
- Continuity-002 is immutable rejected construction evidence: failed packet commit `c98fbe85ceebb7bddd167b33b5a7459ce54110bc`, reviewer receipt SHA-256 `BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173`, rejected addendum SHA-256 `8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8`, rejected executor handoff SHA-256 `42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C`, and rejected test SHA-256 `1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC`. The receipt disposition is `RECONCILIATION_REJECTED`; the old test is absent after literal rollback, and all thirteen projections are restored byte-exact. The three preserved files must never be edited, deleted, relabeled, or treated as acceptance.
- Packet 007 is immutable audit-failed pre-activation evidence at commit `21f7474ad40b46a6dc09ebab521f54c9089fbf50`: PENDING Manifest `7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33`, baseline-003 `134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28`, gate-003 `C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9`, leaf-003 `D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD`, validator-007 `A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B`, and spec-007 `D388D741EC2B3EEBAF8CF71A593FEA27AE878A28AD3C3186A41ECEEBCB816C8D`. Gate-003's entry body named a non-authoritative predecessor baseline hash, and validator-007 failed to detect it. No decision, leaf, test, output, or receipt ran; 007 grants no authority and cannot be repaired or retried in place.
- Independent verifier `ANG-AUTH-VALIDATOR-001`, reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-004` at `codex-subagent:/root/safety_change_map`, and recorder `ANG-BP-ROOT` are distinct from the executor. The reviewer returned a fresh `ACK_ACCEPTED` specifically for `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`; no earlier vocabulary acknowledgement is reused or inferred. The reviewer is reachable. Only that reviewer may create the receipt; it may not edit the packet or executor outputs.

## Required child gates

No child delivery gate is passed by this bootstrap reconciliation. Entry depends only on the already accepted LOW assessment `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1`, ADR-0002, the default-deny local-scaffold policy, the authorized Manifest-v3 pre-start state, and the exact historical Evidence/Resources scaffold dispositions as immutable inputs. `ANG-GATE-EVIDENCE-SCHEMAS-001@1`, `ANG-GATE-RESOURCE-DESIGN-001@1`, `ANG-GATE-HUMAN-FLOURISHING-001@1`, Slice 00, and M0 remain respectively `NOT_RUN`, `NOT_RUN`, `NOT_RUN`, `NOT_PASSED`, and `NOT_PASSED` before and after this gate.

## Procedure and precommitted thresholds

1. Reverify the baseline and every immutable historical hash with read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256`. Stop on any mismatch.
2. The executor uses Codex `apply_patch` only to author or update the fifteen non-handoff executor outputs: thirteen pre-existing projections, the append-only addendum, and the deterministic test. It does not touch the executor handoff yet, the reviewer receipt, or any historical artifact.
3. Run, sequentially from the repository root, the exact consolidated continuity test once using the literal foreground measurement command declared in the leaf. That command must restrict the child PowerShell process to processor affinity mask `1`, poll only that literal child for wall time and peak working set, emit `ANG-METRICS`, enforce 60 seconds and 512 MiB, create no file, and invoke only `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1`.

4. The consolidated run must report exactly `64` named cases and zero failures. Then run the exact literal foreground measurement command declared in the leaf exactly once for `tools/validate_blueprint_tree.ps1`, under the same processor, wall-time, working-set, no-file, and single-child constraints.

5. The continuity test must confirm all thirteen repaired projections agree on accepted/historical state, all frozen identities, the active one-shot continuity leaf, and every non-equivalence state. It must mechanically check the exact Evidence leaf, gate, baseline, decision, execution-time Manifest-v2, test receipt, effect receipt, and original handoff bindings; rehash all fourteen receipt-bound Evidence implementation paths and require 14/14; check the accepted Resources receipt and its complete leaf/gate/baseline/decision/Manifest/output/handoff map; and check the failed-005 plus rejected continuity-002 lineage. Its predeclared negative mutations must cover every frozen Evidence binding, one Evidence implementation hash, every Resources binding class, rejection disposition/identity, projection state, role collision, scope expansion, generic disposition aliases, and each normal-gate/milestone false-positive. It does not require the not-yet-authored executor handoff during this run.
6. Only after the one consolidated continuity run and the tree validator pass, the executor writes its append-only handoff as output sixteen. The handoff must bind the exact authorized Manifest hash, leaf/gate/baseline hashes, executor/task, reviewer path, every pre-start target identity, every post-write target identity, exact commands/results/case count/wall time, per-file and aggregate changed/new bytes, the declared CPU/memory/time ceilings and how each was observed, effects, rollback state, immutable lineage, and the pending independent review. It must explicitly identify any ceiling that could not be independently instrumented; such an inability stops and cannot be represented as measured compliance. The handoff contains no gate disposition.
7. The bound SAFETY reviewer independently inspects the exact packet, outputs, tests, changed paths, immutable hashes, role separation, effects, rollback classes, and non-equivalence. For independent reproducibility after the executor handoff exists, it reruns the same consolidated 64-case test measurement command exactly once and the same tree-validator measurement command exactly once; this is the only gate-required repetition and is not executor self-repetition. It alone may then write `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY-004.md` with exactly `RECONCILIATION_ACCEPTED`, `RECONCILIATION_REJECTED`, or `ESCALATE`.
8. Root may mechanically record or consume the exact disposition but cannot alter it. Even after `RECONCILIATION_ACCEPTED`, Manifest v3 is exhausted and cannot activate another leaf.

Pass requires the executor's one consolidated test and the independent reviewer's one reproducibility run each to report exactly 64 named cases with zero failures, each role's tree validator to pass once, exact `ANG-METRICS` within all ceilings, exactly sixteen executor paths and one reviewer path, no immutable-byte change, and no forbidden effect. The gate does not demand an artificial duplicate run by either role. No retry may broaden scope, suppress a negative control, change a threshold, or relabel a disposition.

## Required negative controls

- A stale projection saying Evidence has no code/receipt/decision, its historical leaf is still executable, Resources revalidation 004 is still pending, Resources has no receipt/output, or either completed leaf should run next.
- Missing or mismatched Evidence decision/leaf/gate/baseline/test/effect/handoff/execution-manifest identity, Resources receipt/output/handoff identity, base commit, release archive, role, task, or reviewer acknowledgement.
- Any failure to prove 14/14 exact Evidence implementation identities, any incomplete Resources binding/output map, any mutation or omission of failed-005 or continuity-002 rejection lineage, or any attempt to overwrite the rejected corrective artifacts instead of authoring the fresh successor paths.
- Any omission, mutation, reinterpretation, or reuse of packet-007 identities; any stale or competing continuity-004 gate-body baseline hash; or any continuity-004 baseline hash other than authoritative `8BB2AC84692DE4F4BA55E90AF3C593506CB922149260DA569B1D2AE0359D45EC`.
- Treating `RECONCILIATION_ACCEPTED`, `SCAFFOLD_ACCEPTED`, bootstrap `ALLOW`, deterministic test success, or a tree-validation pass as a normal technical, Human-Flourishing, Slice-00, M0, scientific, deployment, production, or promotion pass.
- Unprefixed generic aliases such as ACCEPTED or REJECTED in place of the registered reconciliation dispositions.
- Executor self-approval, reviewer/executor identity collision, receipt written before the executor handoff, reviewer modification of packet/executor outputs, or root alteration of reviewer disposition.
- Any write outside the sixteen executor paths or one reviewer receipt; any mutation of historical evidence, decisions, receipts, handoffs, schemas, source, fixtures, tests, policy, assessment, ADR, or rollback archive.
- The executor is specifically denied the rejected addendum, rejected continuity-002 handoff, rejected `CONTINUITY.md` receipt, absent old test path, and fresh `CONTINUITY-004.md` receipt. Any attempt to edit, delete, recreate, or overwrite one of those paths fails closed.
- Network/DNS/socket/browser/API/telemetry/remote Git, model/GPU/probe, package/plugin/dependency/tool installation, recovered or real-person data, credentials, host enumeration, ad hoc directory enumeration, background process, deployment, external effect, bytecode/cache/temp output, broad deletion, or resource-ceiling expansion. The only permitted traversal is traversal performed internally by the exact declared `tools/validate_blueprint_tree.ps1` command.

## Required evidence and receipt identity

The executor evidence is:

- `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency-004.ps1`;
- `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted-corrective-004.md`;
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-004.md`;
- exact before/after hashes for every executor output and the test transcript recorded in the handoff.

The independent receipt identity is `ANG-EVID-CR0-CONTINUITY-004-<sha256-of-exact-utf8-receipt-bytes>`. The receipt must bind gate/version/disposition, Manifest-v3 hash, leaf/gate/baseline/test/handoff hashes, executor and persistent task, validator/test identity, reviewer role/instance/session, `ACK_ACCEPTED`, the exact sixteen-plus-one path partition, before/after hashes, commands/results/case count/wall time/byte totals, immutable Evidence and Resources identities and complete maps, continuity-002 rejection lineage, rollback classes, effect denials, and explicit `NOT_RUN`/`NOT_PASSED` values. The executor cannot create or edit it.

## Failure and rollback

On any failure, stop immediately and record `RECONCILIATION_REJECTED` or `ESCALATE`; do not reinterpret partial output as success. Preserve the fresh corrective-004 addendum, fresh executor handoff, and any fresh independent receipt. Separately preserve the rejected continuity-002 addendum, handoff, and receipt as immutable failure evidence. Restore each of the thirteen pre-existing projection targets individually to its exact baseline bytes and hash. Restore or remove only the fresh continuity-004 test according to its absent baseline; the old rejected test path remains absent. Never use broad, recursive, globbed, directory-level, or archive-wide deletion/restoration. Historical evidence is never changed or deleted.

Failure leaves all normal gates `NOT_RUN`, Slice 00 and M0 `NOT_PASSED`, every other construction leaf unauthorized, and a successor packet mandatory.

## Waiver policy

No waiver exists for an immutable hash, path boundary, role separation, test, negative control, rollback class, registered disposition, non-equivalence statement, or denied capability. Any ambiguity or mismatch yields `ESCALATE` and requires a successor ID/revalidation.
