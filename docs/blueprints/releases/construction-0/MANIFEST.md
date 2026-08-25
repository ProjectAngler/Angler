---
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
version: 3
status: authorized
supersedes_version: 2
predecessor_manifest_v2_sha256: 35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41
revalidation_id: ANG-CR0-REVALIDATION-20260825-005
revalidation_status: PASS
revalidation_spec_path: docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005.md
revalidation_decision_path: docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json
revalidation_decision_status: APPROVED
revalidation_decision_sha256: 467A3EDFDD83049F8B7ECF1C371CEF62E79CC226264A66B4EE32CBF90D31CBA3
pending_manifest_sha256: 08186187A1BE7468488250EECA8B5E78D0CAF0F1450B4A10A3B0830AB801043F
activation_base_commit: 7f383939d021c4bba9dd5af046ce0838b032ff02
sole_ready_leaf: ANG-WORK-CR0-CONTINUITY-001@1
continuity_baseline: ANG-BASELINE-CR0-CONTINUITY-001
continuity_baseline_sha256: EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7
continuity_gate: ANG-GATE-CR0-CONTINUITY-001@1
continuity_gate_sha256: 85DAEB5E3FE72FE37EF40A141FB7DA8B2A133590C6520B2BC1BC55D0C94E20AC
continuity_leaf_sha256: 8175422997DAF86D287A776E45BED2D7F48F3BA79D3EBC65D42130A222419116
continuity_executor: ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001
continuity_executor_task_id: 01a03a80-bb20-7d01-acf6-f50ca4856be5
continuity_validator: ANG-AUTH-VALIDATOR-001
continuity_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
continuity_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001
continuity_reviewer_session_ref: codex-subagent:/root/safety_change_map
continuity_reviewer_vocabulary_ack: ACK_ACCEPTED
continuity_result_recorder: ANG-BP-ROOT
revalidation_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
revalidation_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005
revalidation_reviewer_session_ref: codex-subagent:/root/flourishing_red_team
revalidation_reviewer_vocabulary_ack: ACK_ACCEPTED
v3_validator_sha256: EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D
historical_v2_validator_sha256: 50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94
decision: ANG-ADR-0002
policy: ANG-POL-LOCAL-SCAFFOLD-001@1
bootstrap_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
bootstrap_gate: ANG-GATE-CONSTRUCTION-RELEASE-0-001@1
formal_human_flourishing_gate_status: NOT_RUN
normal_evidence_schema_gate_status: NOT_RUN
normal_resource_design_gate_status: NOT_RUN
slice_id: ANG-SLICE-00-CONTROL
slice_status: NOT_PASSED
milestone_id: ANG-M0-BLUEPRINT
milestone_status: NOT_PASSED
authority_owner: ANG-BP-ROOT
release_steward: ANG-BP-ROOT
issued_at: 2026-08-25
expires_at: 2026-09-24T23:59:59-04:00
rollback_ref: work/pre-construction-release-0-20260825.zip
rollback_sha256: 5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Construction Release 0 manifest v3

## Current authority state

> **AUTHORIZED ONLY FOR `ANG-WORK-CR0-CONTINUITY-001@1`.** Revalidation 005 is independently `APPROVED`; Manifest v3 authorizes only the frozen continuity leaf after this validator passes in AUTHORIZED mode.

Revalidation `ANG-CR0-REVALIDATION-20260825-005` is `PASS`; its separately authored decision is `APPROVED` and pinned in front matter. This transition grants no other authority and preserves every frozen scope, hash, denial, and non-equivalence state.

Manifest v3 supersedes v2 only for continuation-state authority. It preserves every v2 Evidence, Resources, and revalidation identity as immutable history. The proposed repair is an in-scope narrowing of ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001@1`, and LOW assessment `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1`: sixteen literal local documentation/test outputs, deterministic synthetic checks, no external effect, and stricter ceilings. It adds no contract, ADR, capability, data source, technical claim, or threshold and grants no model/GPU/probe, network/package, recovered/real-person data, deployment, promotion, production, or external-use authority.

## Frozen continuity packet and acyclic hash direction

| Component | Frozen identity |
|---|---|
| Base | clean commit `7f383939d021c4bba9dd5af046ce0838b032ff02` |
| Baseline | `ANG-BASELINE-CR0-CONTINUITY-001`; SHA-256 `EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7`; 13 present plus 4 absent targets |
| Gate | `ANG-GATE-CR0-CONTINUITY-001@1`; SHA-256 `85DAEB5E3FE72FE37EF40A141FB7DA8B2A133590C6520B2BC1BC55D0C94E20AC` |
| Leaf | `ANG-WORK-CR0-CONTINUITY-001@1`; SHA-256 `8175422997DAF86D287A776E45BED2D7F48F3BA79D3EBC65D42130A222419116` |
| Validator | `docs/blueprints/work/slice-00/validate-construction-release-0-v3.ps1`; SHA-256 `EE783A80C63AF12F03D829E0F8381C7C5A964BC46FB8772D2D22DCC833683F8D` |
| Revalidation specification | `ANG-CR0-REVALIDATION-20260825-005` at `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005.md`; named by ID/path, not reciprocal hash |
| Future decision | `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json`; `ABSENT` |

The hash direction is baseline -> gate -> leaf -> v3 validator -> this PENDING manifest -> revalidation specification. The future independent decision must bind the PENDING manifest, specification, validator, baseline, gate, leaf, and base hashes. Root may then record only that exact decision and the predeclared phase transition. No self-hash or reciprocal packet pin is authority.

## Concrete roles and disjoint writes

Executor `ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001` is prebound to persistent task `01a03a80-bb20-7d01-acf6-f50ca4856be5`. Validator `ANG-AUTH-VALIDATOR-001`, independent gate reviewer `ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001`, independent revalidation reviewer `ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-005`, and recorder `ANG-BP-ROOT` are distinct from that executor and from one another.

The gate reviewer, at `codex-subagent:/root/safety_change_map` under role `ANG-AUTH-SAFETY-APPROVER-001`, returned a fresh `ACK_ACCEPTED` for exactly `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`; no Resources acknowledgement is reused. Only that reviewer may later create `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md`, after the executor handoff. The executor and Root recorder are denied that reviewer-owned write, and the reviewer is denied packet and executor-output writes.

The revalidation reviewer, at `codex-subagent:/root/flourishing_red_team` under role `ANG-AUTH-SAFETY-APPROVER-001`, returned `ACK_ACCEPTED`, is independent and reachable, and alone may later create `docs/blueprints/releases/construction-0/revalidations/ANG-CR0-REVALIDATION-20260825-005-decision.json` with exactly `APPROVED`, `REJECTED`, or `ESCALATE`. It may not edit this manifest, specification, validator, baseline, gate, leaf, executor outputs, or continuity receipt. Root records results without altering dispositions.

The future executor owns exactly these sixteen outputs:

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
14. `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md`
15. `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1`
16. `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-001.md`

The 117 unique role-owned outputs in the complete CR0 ledger remain collision-free after adding these sixteen executor paths and one reviewer path. No undeclared output is implied.

## Immutable historical ledger

### Manifest v2 and same-version revalidations

- Authorized predecessor Manifest v2 is immutable at SHA-256 `35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41`. The Evidence execution-time Manifest-v2 identity remains `802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2`.
- Embedded revalidations 001 and 002 remain historical v2 records of the narrowed `apply_patch` authoring mechanism and prebound-executor/disposition correction. Manifest v3 neither repeats nor reinterprets their static passes.
- Revalidation 003 is immutable `REJECTED` history: specification SHA-256 `1AA06113B50B53327CCC79E8F06DC7F4E133AA1DA205BC762BAA677CD14F13F9`, decision SHA-256 `9C5FD13E5D7EB2B8256B703FC78F8BC2F190D2299C8523282757F7E4A559504A`, and predecessor baseline-001 SHA-256 `EF5387CDD4B652641E990DA1C1FF64B146B1D1598C9BAF9509D633A2FD1E2569`. Its stale-cache defect is not erased.
- Revalidation 004 is immutable `APPROVED` history: specification SHA-256 `50ED004747A4DABD9D306E8931D8BFD8A558F0CDE441596645A8E028F15CF9D2` and decision SHA-256 `12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9`. Its authority was consumed only by the accepted Resources scaffold.
- The historical v2 validator is immutable at SHA-256 `50C4700BE03F680DB229A325A2816DB6F0BD3EE2059507F12E159FC9EC431E94`; it is not the v3 validator and is not rerun for this packet.

### Evidence scaffold — accepted history

Evidence decision identity `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, exact decision SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`, records `SCAFFOLD_ACCEPTED`. It and its inputs are historical and non-repeatable:

| Bound item | SHA-256 |
|---|---|
| Leaf `ANG-WORK-EVIDENCE-SCHEMAS-001@1` | `5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289` |
| Gate `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001@1` | `A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5` |
| Baseline | `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F` |
| Test receipt | `897D57E64169654F41422D70A2DA87C3E15E22A521A77437A8E80FB12B3C3E53` |
| Effect receipt | `9808AFC82EF8B77DAABF8296E11A54C267E087566BA24ABA46999747A775C893` |
| Executor handoff | `017969B6E382D405EE32E73306808C82BB283A2B68E5CAC1129596439AC5C855` |
| Execution-time Manifest v2 | `802D152574ABD5771CF851293F9E2240039472F61DBC78472D2BF3638CE5E5D2` |

The immutable test receipt's fourteen implementation/schema/test/fixture hashes must rehash 14/14. No historical Evidence decision, receipt, handoff, schema, fixture, test, or implementation byte may change.

### Resources scaffold — accepted history

Resources independent receipt SHA-256 `D7221A84BE404A64487E79A93B5268E5E819E153B3F07E5647473F2855A24B4B` records `SCAFFOLD_ACCEPTED`. It is historical and non-repeatable and binds leaf SHA-256 `67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA`, gate SHA-256 `A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2`, and baseline-002 SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`.

Its exact seven scaffold outputs plus executor handoff are:

| Resource output | SHA-256 |
|---|---|
| Inventory schema | `92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F` |
| Execution-plan schema | `7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2` |
| Constrained fixture | `CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956` |
| Workstation fixture | `162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9` |
| Cluster fixture | `AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85` |
| Invalid-plan fixture | `92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A` |
| Deterministic Resource test | `047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA` |
| Executor handoff | `916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A` |

These are synthetic design scaffolds only. They contain no measured inventory or probe provenance, do not claim current host fitness, and cannot be rerun or promoted by v3.

## Leaf ledger and non-equivalence

| Leaf | v3 state | Executable now? |
|---|---|---|
| `ANG-WORK-EVIDENCE-SCHEMAS-001@1` | Evidence `SCAFFOLD_ACCEPTED`; historical and non-repeatable | no |
| `ANG-WORK-CR0-RESOURCES-001@3` | Resources `SCAFFOLD_ACCEPTED`; historical and non-repeatable | no |
| `ANG-WORK-CR0-CONTINUITY-001@1` | sole ready specification; PENDING revalidation 005 | no |
| `ANG-WORK-CR0-SAFETY-001@1` | blocked; requires a later successor manifest | no |
| WORLDS, SCIENCE, INTEGRATION | blocked | no |
| RUNTIME, LEARNING, TOOLS | not ready | no |

The normal Evidence, Resources, and Human-Flourishing gates remain `NOT_RUN`; Slice 00 and M0 remain `NOT_PASSED`. Neither `SCAFFOLD_ACCEPTED`, `RECONCILIATION_ACCEPTED`, bootstrap `ALLOW`, deterministic-test success, nor tree-validation success is a normal technical, Human-Flourishing, Slice-00, M0, scientific, deployment, production, promotion, or product pass.

Continuity acceptance repairs only stale authority projections. It cannot activate SAFETY, reopen either accepted scaffold, or consume any blocked leaf. A later successor manifest is required before SAFETY or any other leaf can activate.

## Future execution sequence and ceilings

If and only if revalidation 005 is independently `APPROVED`, Root performs the exact authorized/PASS transition, and the authorized v3 validator passes, the bound executor may use Codex `apply_patch` on the sixteen leaf-owned paths. It first authors or updates the thirteen projections, append-only addendum, and deterministic test as fifteen non-handoff outputs. It then runs the continuity test twice with identical case counts and zero failures, runs the tree validator once, and only then authors the executor handoff as output sixteen. The reviewer receipt follows the handoff and is reviewer-only.

One foreground command at a time; 60 seconds per command; 600 seconds aggregate active time; 1 logical CPU; 512 MiB working set; 2 MiB total changed/new bytes; 512 KiB per file; zero network, GPU, package/tool/model installation, spend, background work, probe, host enumeration, cache/temp output, staging, or commit. Only traversal performed internally by the exact declared tree-validator command is permitted; ad hoc directory enumeration is denied.

## Rollback and stop

Baseline `ANG-BASELINE-CR0-CONTINUITY-001` records 17 literal targets at base `7f383939d021c4bba9dd5af046ce0838b032ff02`: thirteen present projections with exact hashes and four absent future outputs. Rollback restores the thirteen pre-existing targets individually, restores or removes only the continuity test, and preserves the append-only addendum, executor handoff, and reviewer receipt on failure. The pre-release archive remains `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`. Broad, recursive, globbed, directory-level, or archive-wide deletion/restoration is forbidden.

Any mismatch in identity, baseline state, role, scope, command, ceiling, denial, result vocabulary, or non-equivalence stops the packet. No retry may silently refresh a pin or broaden authority.

## PENDING packet-review requirement

The packet author must run, sequentially, the existing CR0 safety validator, the frozen v3 validator, and the blueprint-tree validator for two identical PENDING rounds. Those static rounds must report PENDING / NON-AUTHORIZING, leave the revalidation decision and all four future outputs absent, and execute neither the continuity leaf nor its test. Their success is review evidence only, never execution permission.
