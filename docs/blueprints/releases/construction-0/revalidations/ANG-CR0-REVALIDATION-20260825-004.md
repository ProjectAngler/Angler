---
revalidation_id: ANG-CR0-REVALIDATION-20260825-004
supersedes_revalidation: ANG-CR0-REVALIDATION-20260825-003
superseded_revalidation_disposition: REJECTED
superseded_revalidation_spec_sha256: 1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9
superseded_revalidation_decision_sha256: 9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
manifest_version: 2
status: PENDING
authorization_effect: NONE
base_commit: 7313a0d951c8f27af4c036e3b67059b7506cb3f1
accountable_owner: ANG-BP-ROOT
packet_writer: ANG-BP-ROOT
independent_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
independent_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-004
independent_reviewer_session_ref: codex-subagent:/root/flourishing_red_team
reviewer_acceptance: ACCEPTED
reviewer_independence: independent
reviewer_reachability: reachable
decision_path: docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004-decision.json
decision_status: ABSENT
pre_successor_manifest_sha256: 1B81B3A41B63B55A5000FFC4873BD8AC11D06B4537B7F2958EB14024D9628C55
candidate_manifest_sha256: A42B15F2296676838D650045C879BAB4A7123FBF7D6853B89CBB527C247ECF14
evidence_execution_manifest_sha256: 802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2
evidence_decision_sha256: 520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0
resources_gate_version: 2
resources_leaf_revision: 3
resources_baseline_id: ANG-BASELINE-CR0-RESOURCES-002
resources_reviewer_vocabulary_ack: ACK_ACCEPTED
resources_gate_sha256: A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2
resources_baseline_sha256: A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE
resources_leaf_sha256: 67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA
historical_resources_baseline_001_sha256: EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569
blueprint_index_sha256: 1A4EA38741243E24EF46B74DA78EDDE256BDE7E9BE7EDDDAAFA013CF7FF17015
root_capsule_sha256: E2DE70E2C118432A0B8B35D7F3B26E6DA116E99C5C5AC12B9192868366D00B72
root_status_sha256: CBA795AFF3CA2C24E1C9373192E792F5B108D7390DF59294D3C6FEF675A11A54
resources_capsule_sha256: 03DC17C85CF1076D8DB7482136A403D9EE4F7C804361C68210CC3232B2030175
resources_status_sha256: A1A221EF1FBB85EF8A41D6ECC627E8FECA2290073F849834891EDFCF92465DEA
evidence_capsule_sha256: 54860217CACB6DC5DEADA3763F8E35371C879BD8A0A792579211B3011EE70DD3
evidence_status_sha256: 118B3D928F93F3C53B299D352B6CC5A4F2D323A2CF1297B9A06C9B0CABE34DA5
validator_sha256: 50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94
issued_at: 2026-08-25
---

# Resources activation revalidation 004 successor work/spec

## PENDING / NON-AUTHORIZING boundary

This root-owned successor document specifies structural review only. It grants no execution authority, writes no independent disposition, and cannot make Manifest v2 usable. Do not execute the Resources leaf or Resource test, create any of its nine outputs, rerun historical Evidence work, rewrite rejected-003 evidence, or claim a normal Evidence/Resource, Human-Flourishing, Slice-00, M0, scientific, measured-plan, model, deployment, or external-use pass.

Revalidation `ANG-CR0-REVALIDATION-20260825-003` is immutable `REJECTED` history. Its specification is `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003.md`, SHA-256 `1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9`; its decision is `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-003-decision.json`, SHA-256 `9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A`. Revalidation 004 supersedes it only for a corrected future review and never alters that evidence.

The Evidence predecessor is historical decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, disposition `SCAFFOLD_ACCEPTED`. Its execution-time manifest binding remains `802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2`; the normal Evidence gate remains `NOT_RUN`.

## Exact packet paths and frozen hashes

| Path | Owner/write class | SHA-256 or state |
|---|---|---|
| `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004.md` | root successor work/spec; immutable during review | self-hash recorded only by future independent decision |
| `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004-decision.json` | independent revalidation reviewer sole future write scope | ABSENT |
| `docs/blueprints/releases/construction-0/MANIFEST.md` | root release ledger | `A42B15F2296676838D650045C879BAB4A7123FBF7D6853B89CBB527C247ECF14` |
| `docs/blueprints/branches/resources/gates/ANG-GATE-CR0-RESOURCES-001.md` | `ANG-GATE-CR0-RESOURCES-001@2` | `A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2` |
| `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-002.json` | root active rollback baseline for leaf revision 3 | `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE` |
| `docs/blueprints/work/slice-00/ANG-WORK-CR0-RESOURCES-001.md` | frozen `ANG-WORK-CR0-RESOURCES-001@3` specification | `67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA` |
| `docs/blueprints/releases/construction-0/baselines/ANG-BASELINE-CR0-RESOURCES-001.json` | immutable rejected-003 baseline evidence | `EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569` |
| `docs/blueprints/BLUEPRINT_INDEX.json` | unchanged root registry | `1A4EA38741243E24EF46B74DA78EDDE256BDE7E9BE7EDDDAAFA013CF7FF17015` |
| `docs/blueprints/ROOT_CAPSULE.md` | mandatory authority-state cache revision 6 | `E2DE70E2C118432A0B8B35D7F3B26E6DA116E99C5C5AC12B9192868366D00B72` |
| `docs/blueprints/STATUS.md` | mandatory authority-state cache | `CBA795AFF3CA2C24E1C9373192E792F5B108D7390DF59294D3C6FEF675A11A54` |
| `docs/blueprints/branches/resources/CAPSULE.md` | mandatory authority-state cache revision 5 | `03DC17C85CF1076D8DB7482136A403D9EE4F7C804361C68210CC3232B2030175` |
| `docs/blueprints/branches/resources/STATUS.md` | mandatory authority-state cache | `A1A221EF1FBB85EF8A41D6ECC627E8FECA2290073F849834891EDFCF92465DEA` |
| `docs/blueprints/branches/evidence/CAPSULE.md` | refreshed Evidence cache revision 8 | `54860217CACB6DC5DEADA3763F8E35371C879BD8A0A792579211B3011EE70DD3` |
| `docs/blueprints/branches/evidence/STATUS.md` | refreshed Evidence status | `118B3D928F93F3C53B299D352B6CC5A4F2D323A2CF1297B9A06C9B0CABE34DA5` |
| `docs/blueprints/work/slice-00/validate-construction-release-0.ps1` | deterministic structural validator | `50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94` |

All nine Resource targets and the 004 decision path must remain absent. No normative branch blueprint, shared contract, interface, ADR, policy, assessment, Evidence artifact, rejected-003 evidence, or Resources output is changed by this successor packet.

## Role and write separation

- Root steward `ANG-BP-ROOT` authored only the successor packet/spec/cache paths and may later record an independently issued decision without altering it.
- Resources executor `ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001` has no authority while PENDING and later owns only the leaf's eight literal executor outputs.
- Validator `ANG-AUTH-VALIDATOR-001` produces structural/test evidence only and cannot authorize.
- Resources reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001`, session `codex-subagent:/root/safety_change_map`, alone may later write `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md` after the executor handoff. Its dispositions are exactly `SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`; generic aliases are invalid. It returned `ACK_ACCEPTED` for successor gate version 2 and confirmed independence/reachability; this is a fulfilled precondition, not a gate disposition.
- Revalidation reviewer `ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-004`, session `codex-subagent:/root/flourishing_red_team`, explicitly ACCEPTED and is independent and reachable under `ANG-AUTH-SAFETY-APPROVER-001`. It is denied every packet, executor, and Resources-receipt write; its sole future write scope is the reserved 004 decision. That decision uses exactly `APPROVED`, `REJECTED`, or `ESCALATE`.

## Exact absence set

The following literal paths must be absent throughout PENDING review:

- `schemas/control/v1/resources/resource-inventory.schema.json`
- `schemas/control/v1/resources/execution-plan.schema.json`
- `tests/synthetic/slice00/resources/constrained.inventory.json`
- `tests/synthetic/slice00/resources/workstation.inventory.json`
- `tests/synthetic/slice00/resources/cluster.inventory.json`
- `tests/synthetic/slice00/resources/invalid-overcommitted.plan.json`
- `tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1`
- `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md`
- `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-004-decision.json`

## Static PENDING review commands

Run only from the repository root, sequentially. These are two identical structural rounds, not the Resources leaf/test.

Round 1:

```powershell
pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1
pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Round 2:

```powershell
pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1
pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

A successful CR0 result must explicitly say `PENDING / NON-AUTHORIZING` and grant no leaf permission. The validator must parse all mandatory authority-state caches, reject every Resources baseline hash except baseline-002 SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`, reject every active revalidation other than 004, and allow 003 only as explicit historical `REJECTED` evidence.

## Stop, rollback, and phase transition

Stop on any mismatch, ambiguity, denied capability/data/path, changed scope/command/ceiling/claim, unavailable reviewer, target collision, stale cache, nonzero check, or human revocation. Do not broaden or retry. Preserve accepted Evidence artifacts, baseline-001, rejected-003 spec/decision, and this successor packet. Active Resource targets are governed only by `ANG-BASELINE-CR0-RESOURCES-002`; preserve receipt/handoff on failure, revert only its seven `restore_or_remove` paths individually, never broadly or recursively delete, and never unpack the release archive automatically.

PENDING cannot transition itself. The independent reviewer first verifies exact bytes, absence, roles, commands, non-equivalence, mandatory cache consistency, rejected-003 preservation, and the fulfilled Resources `ACK_ACCEPTED`. Only that reviewer may create the reserved 004 decision. `REJECTED` or `ESCALATE` keeps the release pending/blocked. `APPROVED` is necessary but not sufficient: root must then perform a separate minimal manifest transition to `status: authorized` and `revalidation_status: PASS`, pin the decision hash, preserve every frozen identity, and run the authorized validator branch before any Resource write.

## Hash-direction rule and next action

No circular hash claim is authority. Baseline-002 precedes gate@2; gate/baseline precede leaf@3; caches and leaf precede validator; validator and frozen inputs precede the candidate manifest; this spec pins that manifest and validator. The manifest references this spec by ID/path without hashing it. The future independent decision must pin both spec and candidate-manifest hashes, allowing a later authorized manifest to pin the decision without reciprocal hashes.

Next action: `ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-004` performs read-only review and alone writes the reserved 004 decision. No other action is authorized.
