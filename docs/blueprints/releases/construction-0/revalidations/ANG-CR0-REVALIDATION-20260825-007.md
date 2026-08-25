---
revalidation_id: ANG-CR0-REVALIDATION-20260825-007
status: PENDING
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
release_manifest_version: 3
base_commit: 8a62f9508c1e8f573a57a32fd99e880aefea7ae6
pending_manifest_sha256: 7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33
validator_sha256: A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B
baseline_sha256: 134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28
gate_sha256: C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9
leaf_sha256: D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD
reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-007
reviewer_session_ref: codex-subagent:/root/flourishing_red_team
reviewer_vocabulary_ack: ACK_ACCEPTED
decision_writer_role: ANG-AUTH-SAFETY-APPROVER-001
decision_path: docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007-decision.json
allowed_dispositions: APPROVED|REJECTED|ESCALATE
failed_predecessor_revalidation_id: ANG-CR0-REVALIDATION-20260825-005
rejected_predecessor_leaf: ANG-WORK-CR0-CONTINUITY-002@1
rejected_predecessor_receipt_sha256: BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173
rejected_predecessor_status: RECONCILIATION_REJECTED
result_recorder: ANG-BP-ROOT
created_at: 2026-08-25
---

# Revalidation 007 — corrective continuity-003 packet

## State and purpose

This specification is **PENDING / NON-AUTHORIZING**. It asks one fresh independent SAFETY reviewer to decide whether Manifest v3 may make the exact six-edit transition to a pre-start authorized state for `ANG-WORK-CR0-CONTINUITY-003@1`. It grants no present permission to run the leaf, its test, the tree validator as construction, or any output writer. Packet-authoring validation is review evidence only.

Continuity-002 is immutable rejected evidence. Revalidation 007 neither edits nor retries it. The successor exists only to correct its proven projection/addendum/test/handoff defects through fresh paths and identities while preserving every failed-005, rejected continuity-002, accepted Evidence, and accepted Resources byte.

## Immutable predecessor ledger

- Revalidation 005: reviewed PENDING Manifest `08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F`; decision `467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3`; attempted authorized Manifest `9578BD1B97451636449254CF1496387DA9602240B51401CC85595237E657C3E5`; validator `EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D`; preserved commit `84ff08b484197755c3fed66e7dc06988539e456e`. Exact command exited 1 with `Write-Error: Manifest-v3 authority/non-equivalence is missing required literal: ready but unusable until independent revalidation approval, Root authorization, and an authorized v3-validator PASS`. No leaf/test/output/receipt ran; 005 grants no authority.
- Revalidation 006: PENDING Manifest `E41000C14FE7BA0F88FED46DA29DBCFAF3D3DD15D2DB54E001A34A76CFCBCE02`; authorized Manifest `D10284F3A81B85DFBF1F342EFF213C5B2E67CABDB229552C6F11165C896159D7`; decision `C34C11606057FC6F22B7428D4DCE9F707B7A64C1D8038FE9F356DC0CBED98296`; validator `866AC1E579A4D38CA898B640AF2F2F8B464352BAB759ABC38F5D3C0A6D10D21A`. Its one-shot authority was consumed and exhausted by continuity-002.
- Continuity-002: failed packet commit `c98fbe85ceebb7bddd167b33b5a7459ce54110bc`; receipt `CONTINUITY.md` SHA-256 `BE8C5514CD43DB1EFA42C97BAC20188AC4AD993518FA5DF1D2EACBFB2F1AE173` with `RECONCILIATION_REJECTED`; rejected addendum SHA-256 `8A144F3C1F3614334FEED04DBB424BD79F5CBED98DFAEB675D4B6D56C84505E8`; rejected handoff SHA-256 `42924F9D75EC6459A591EF3503F8E53BD8E5652F31F32DD0B49879DE01273A2C`; rejected test SHA-256 `1211E0DE9B6792465F475077217501094F79B52B51582585D5E47C9D574882AC`, now absent after literal rollback. The thirteen projections are restored exactly.
- The receipt found six false Evidence hashes, an under-scoped 33-case test, and an incomplete handoff. The three preserved files are immutable failure evidence, and the old test path must remain absent.

## Exact packet under review

| Item | Path | SHA-256/state |
|---|---|---|
| Candidate Manifest v3 | `docs/blueprints/releases/construction-0/MANIFEST.md` | `7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33` |
| Baseline | `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-CONTINUITY-003.json` | `134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28` |
| Gate | `docs/blueprints/releases/construction-0/gates/ANG-GATE-CR0-CONTINUITY-003.md` | `C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9` |
| Leaf | `docs/blueprints/work/slice-00/ANG-WORK-CR0-CONTINUITY-003.md` | `D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD` |
| Validator | `docs/blueprints/work/slice-00/validate-construction-release-0-v3-007.ps1` | `A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B` |
| This specification | `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007.md` | reviewer computes exact hash; no self-hash claim |
| Reserved decision | `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-007-decision.json` | `ABSENT` while PENDING |

Hash direction is baseline-003 -> gate-003 -> leaf-003 -> validator-007 -> PENDING Manifest-v3 -> this specification -> future decision. Manifest names this specification only by ID/path. The future decision is the sole bridge to Root's frozen transition.

## Fresh roles and disjoint writes

Revalidation reviewer role `ANG-AUTH-SAFETY-APPROVER-001`, fresh instance `ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-007`, session `codex-subagent:/root/flourishing_red_team`, is independent/reachable and returned fresh `ACK_ACCEPTED` for `APPROVED|REJECTED|ESCALATE`. It may later write only the decision-007 JSON. It is barred from Manifest, packet files, executor outputs, old rejection evidence, and both receipt paths.

The future executor is fresh identity `ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-003` bound to persistent task `01a03a80-bb20-7d01-acf6-f50ca4856be5`. Gate reviewer role `ANG-AUTH-SAFETY-APPROVER-001`, fresh instance `ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-003`, session `codex-subagent:/root/safety_change_map`, returned fresh `ACK_ACCEPTED` for `RECONCILIATION_ACCEPTED|RECONCILIATION_REJECTED|ESCALATE` and may later write only `CONTINUITY-003.md`. Validator `ANG-AUTH-VALIDATOR-001` and recorder `ANG-BP-ROOT` are separate from both reviewers and executor. No role may borrow a predecessor acknowledgement or disposition.

## Required independent checks

The reviewer must verify without construction execution:

1. Base identity is `8a62f9508c1e8f573a57a32fd99e880aefea7ae6`; packet changes are only Manifest plus baseline/gate/leaf/validator/spec; decision-007 and all four fresh successor targets are absent.
2. Baseline has exactly 17 unique targets: thirteen exact present projections, fresh corrective addendum/test/handoff/receipt absent; rollback is 13 `restore_exact`, only test-003 `restore_or_remove`, and the other three fresh paths `preserve_on_failure`.
3. Old addendum, handoff, and receipt rehash exactly; old test remains absent. The validator freezes the complete failed commit/revalidation/receipt lineage.
4. Evidence decision is `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0` with exact leaf `5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289`, gate `A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5`, baseline `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F`, execution Manifest `802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2`, test receipt `897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53`, effect receipt `9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893`, original handoff `017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855`, and all 14/14 implementation hashes.
5. Resources receipt `D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B` and the complete Resources leaf/gate/baseline/decision/Manifest/seven-output/handoff map rehash exactly.
6. Executor owns exactly sixteen fresh/current output paths; gate reviewer owns only fresh receipt-003; revalidation reviewer owns only decision-007; denied old paths are explicit; roles are disjoint.
7. The successor test contract has exactly 64 named positive/negative cases covering every frozen binding, 14/14 Evidence implementation identities, complete Resources map, rejection lineage, projections, role/scope/non-equivalence, and all predeclared mutation classes.
8. The handoff contract requires authorized Manifest/leaf/gate/baseline hashes, all 17 pre-start and 16 post-write target identities, exact commands/results/64-case count/wall times, byte ceilings, metric evidence, effects, rollback, and reviewer path.
9. Normal Evidence, Resources, and Human-Flourishing gates remain `NOT_RUN`; Slice 00 and M0 remain `NOT_PASSED`; continuity acceptance cannot activate SAFETY or any other leaf.

Any mismatch requires `REJECTED` or `ESCALATE`; the reviewer may not repair, repin, execute, or reinterpret it.

## Packet-review commands

Run one consolidated PENDING validation round, sequentially from the repository root:

```powershell
pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1
pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0-v3-007.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Every command must exit zero. The v3 validator must say `PENDING / NON-AUTHORIZING` and `AUTHORIZED_TRANSITION_SELF_TEST PASS`. The self-test constructs a syntactically valid simulated decision and exercises the same authorized-state function as the real branch entirely in memory, then reverses exactly to the frozen PENDING hash. Do not execute the continuity leaf or test, create decision/receipt/output, stage, or commit. No second authoring round is required; the future gate separately requires one independent reviewer reproducibility run after execution because continuity-002's test assurance failed.

## Future decision and exact reversible transition

Only the bound revalidation reviewer may create decision-007. It must include:

- `revalidation_id`: `ANG-CR0-REVALIDATION-20260825-007`;
- `disposition`: exactly `APPROVED`, `REJECTED`, or `ESCALATE`;
- exact reviewer role/instance/session/`ACK_ACCEPTED`;
- bindings for PENDING Manifest `7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33`, this specification's computed hash, validator `A580A38909D9862609F578481548F3AD268D857C2A238E1FDA6B9D227227D00B`, baseline `134A6EDF9D608C5CDED1812C9CB4A0F63AEA99ACBCF814A79CF00EDF83492D28`, gate `C7FF7AD57661D7EF2303D628E6A69A0F54C3366FA163143BA7E35EBAE76479F9`, leaf `D4D15CC6C725416BB5FDCB8DC6833EB913B3A4EFD8B34303672D57D6DC072BCD`, and base commit;
- exact `NOT_RUN`/`NOT_PASSED` non-equivalence fields;
- JSON-false confirmations that leaf, test, outputs, and receipt were not executed/created during review.

`APPROVED` alone is not authority. Root may then make exactly six Manifest edits and no others:

1. `status: pending_revalidation` -> `status: authorized`.
2. `revalidation_status: PENDING` -> `revalidation_status: PASS`.
3. `revalidation_decision_status: ABSENT` -> `revalidation_decision_status: APPROVED`.
4. `revalidation_decision_sha256: ABSENT` -> the exact decision hash.
5. `pending_manifest_sha256: ABSENT_UNTIL_AUTHORIZED` -> `7346C520E70587A4C3E5C729F67B3AE8A2109F94012991C9B57AAFC984E91B33`.
6. Replace the exact PENDING phase line with:

```text
> **AUTHORIZED ONLY FOR `ANG-WORK-CR0-CONTINUITY-003@1`.** Revalidation 007 is independently `APPROVED`; authority exists only after this validator returns AUTHORIZED PASS.
```

The real authorized validator uses the same logic as the PENDING self-test, requires the exact decision bytes/bindings and unchanged 17-target pre-start state, reverses the six edits, and requires the reconstructed PENDING hash. Any extra Manifest edit fails. Only its AUTHORIZED PASS permits the prebound executor to begin.

## Stop, rollback, and denials

Stop on any undeclared difference, hash drift, missing negative mutation, role collision, old-path reuse, ceiling ambiguity, or broadened claim. Preserve fresh corrective addendum/handoff/receipt on any future failure, preserve all rejected continuity-002 evidence byte-exact, restore thirteen projections individually, and restore/remove only test-003. Never broadly delete/restore or extract the archive.

Network, DNS, browser/API/telemetry/dashboard, remote Git, package/plugin/dependency/tool installation, model/GPU/probe, recovered or real-person data, credentials, background work, deployment, promotion, production, external effect, staging, commit, and all construction execution are forbidden during revalidation. Normal Evidence/Resources/Human-Flourishing gates remain `NOT_RUN`; Slice 00/M0 remain `NOT_PASSED`. Even later `RECONCILIATION_ACCEPTED` requires a successor Manifest before SAFETY or any other leaf.
