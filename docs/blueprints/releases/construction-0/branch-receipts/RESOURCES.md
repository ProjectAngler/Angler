---
receipt_type: ANG-EVID-CR0-RESOURCES
identity_rule: ANG-EVID-CR0-RESOURCES-<sha256-of-exact-utf8-receipt-bytes>
gate_id: ANG-GATE-CR0-RESOURCES-001
gate_version: 2
disposition: SCAFFOLD_ACCEPTED
reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001
reviewer_session_ref: codex-subagent:/root/safety_change_map
executor: ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001
independent_validator: ANG-AUTH-VALIDATOR-001
result_recorder: ANG-BP-ROOT
reviewed_on: 2026-08-25
---

# CR0 Resources scaffold independent decision

## Decision

`SCAFFOLD_ACCEPTED` applies only to the exact local, synthetic Resources scaffold bound below. The frozen authority, artifact hashes, role separation, repeated tests, tree validation, required negative controls, effects, ceilings, rollback classes, and non-equivalence statements satisfy `ANG-GATE-CR0-RESOURCES-001@2`.

## Frozen authority and evidence bindings

| Binding | Exact identity |
|---|---|
| Authorization checkpoint | `54bbbf9b5068880ba8b24315cd1c0e430b2f2d74` |
| Activation base commit | `7313a0d951c8f27af4c036e3b67059b7506cb3f1` |
| Release manifest v2 | SHA-256 `35936904E70CDB883ACF9A7235D943A94E5ED7EB2E3F7577654908E2E4BF4A41` |
| Work leaf | `ANG-WORK-CR0-RESOURCES-001@3`; SHA-256 `67BFE6FA3A9131D08293E8FB211504A9C9C7CCE5A4E9CF837DBBCA48AEFB7DBA` |
| Gate | `ANG-GATE-CR0-RESOURCES-001@2`; SHA-256 `A12017C6B84F0ED63C021482E94379F95D524825C4C0FE25C33027A6669246F2` |
| Baseline | `ANG-BASELINE-CR0-RESOURCES-002`; SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE` |
| Revalidation | `ANG-CR0-REVALIDATION-20260825-004`; independently `APPROVED`; decision SHA-256 `12EA8BB22C3F059985C1A9CEE3B90AA3469D31F2E4DD62E2F7649CA001D0CAA9` |
| Evidence predecessor | `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`; SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`; disposition `SCAFFOLD_ACCEPTED` |
| Executor handoff | `ANG-HANDOFF-CR0-RESOURCES-001`; SHA-256 `916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A` |

The executor, deterministic validator, reviewer, and root recorder identities are distinct. The executor-created handoff contains no gate disposition, and the reviewer-owned receipt path was absent immediately before this write.

## Exact paths and hashes

### Executor outputs

| Path | SHA-256 |
|---|---|
| `schemas/control/v1/resources/resource-inventory.schema.json` | `92055E5F07FF789FB22BC67043E3CD16BCE904AE472C2C62B80AAA571A70C98F` |
| `schemas/control/v1/resources/execution-plan.schema.json` | `7B5F6EC835B8A00B1891C98E86B8195672B553FBF732DB7CD3CA5160A57F3FE2` |
| `tests/synthetic/slice00/resources/constrained.inventory.json` | `CC37A0E7C6765D8B221A47F924E8413BF31E3AF0F9AE2F86D9EB741DBC9F5956` |
| `tests/synthetic/slice00/resources/workstation.inventory.json` | `162FE7BAF9403F4816446D11A318FA5DB8AE8908280DA76AD0BA5DFF8A1F12F9` |
| `tests/synthetic/slice00/resources/cluster.inventory.json` | `AE918EEED93A55F5C3D659AFCE6C204840D65A4D14A73852A502C9D7DA6F5F85` |
| `tests/synthetic/slice00/resources/invalid-overcommitted.plan.json` | `92AFD197394C6BDF31E65590BBC30915CC974BFB0F8795D617EA506E80119E9A` |
| `tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1` | `047CB4AED702AFA79CE9513230F670FAE872E285031B7EB8C6DE7CB0933076BA` |
| `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-RESOURCES-001.md` | `916C838696D18F9FA95E3BDBCE91C6A66174A87167D047F40FF2CF89617F264A` |

### Reviewer output

- `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md`

No other changed path is accepted or authorized by this decision.

## Independent commands and results

Commands ran sequentially from the repository root in separate foreground processes:

1. `pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1`
   - Exit code `0`; wall time `0.5997041` seconds.
   - `PASS - CR0 Resources scaffold: 16 cases, 0 failures; 3 synthetic tiers accepted and all declared negative controls rejected.`
2. `pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1`
   - Exit code `0`; wall time `0.5773962` seconds.
   - Output and case count were identical to run 1.
3. `pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1`
   - Exit code `0`; wall time `0.804641` seconds.
   - `Blueprint validation PASS: 21 concrete nodes, 38 declared child nodes, 140 Markdown files.`

Aggregate observed command wall time was `1.9817413` seconds. The repeated Resource results were identical, with 16 cases and zero failures each; the blueprint validator reported zero errors.

## Schema, fixture, negative-control, and effect findings

- The two schemas retain their exact v1 contract identities, require synthetic/unmeasured state, separate inventory and Evidence identities, reserve probe-provenance references without asserting probe success, require positive host/evaluation/rollback headroom, bind topology, prohibit query-conditioned routing and mid-transaction replanning, and encode fail-closed fallbacks.
- Constrained, workstation, and cluster fixtures use the same logical inventory contract, contain explicit synthetic markers and synthetic identifiers, and make no host-measurement claim.
- The deterministic test accepted all three tiers and a bounded plan, then rejected overcommitment, absent rollback headroom, missing inventory identity, multiple Evidence identities, fabricated measurement, fabricated probe success, query-conditioned routing, mid-transaction topology change, malformed probe provenance, and non-synthetic input. Static inspection confirms the shared validator paths also fail closed on the remaining symmetric identity/headroom and tier-invariance cases required by the gate.
- The reviewed script reads only its literal schemas and fixtures and contains no host/GPU probe or enumeration, WMI/CIM, `nvidia-smi`, model/GPU operation, network, package, background process, recovered/personal data, `outputs/**`, deployment, or write mechanism.
- Frozen hashes remained unchanged after the three independent commands. No forbidden effect, threshold change, authority broadening, executor self-approval, or reviewer/executor cross-role write was observed in the declared gate scope.

## Rollback identity and classes

- Release archive: `work/pre-construction-release-0-20260825.zip`; SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`.
- Baseline: `ANG-BASELINE-CR0-RESOURCES-002`; SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`.
- `restore_or_remove`: the two schemas, four synthetic fixtures, and Resource test script.
- `preserve_on_failure`: the executor handoff and this independent decision receipt.
- No rollback was performed because the gate evidence passed. Any later failure preserves the handoff and receipt and may revert only literal baseline `restore_or_remove` paths individually; broad or recursive deletion and automatic archive extraction remain forbidden.

## Limitations and discrepancies

- No host inventory, GPU probe, filesystem enumeration, continuous profiling, telemetry, third-party JSON-Schema engine, or exact byte-metering layer was used or authorized. This decision therefore accepts implementation-independent schema and deterministic synthetic-control behavior only, not measured feasibility or runtime efficiency.
- The independent tree validator counted `140` Markdown files while the executor handoff records `139`; the additional file is the required executor handoff, which was authored after the executor's tree-validation run and before this independent review. Both runs reported zero structural errors.
- Git staging and commit state were not inspected because no Git-enumeration command was authorized. This review binds the supplied authorization checkpoint and the exact frozen artifacts and hashes above; it performs no stage or commit.

## Non-equivalence

- `ANG-GATE-RESOURCE-DESIGN-001`: `NOT_RUN`.
- `ANG-GATE-EVIDENCE-SCHEMAS-001`: `NOT_RUN`.
- `ANG-GATE-HUMAN-FLOURISHING-001`: `NOT_RUN`.
- Slice 00: `NOT_PASSED`.
- M0: `NOT_PASSED`.
- Measured ResourceInventory or ExecutionPlan evidence: none.
- Resource probe, host/GPU discovery, model/learner operation, scientific success, deployment, production, promotion, and external-use authority: none.

This decision permits only manifest-listed CR0 consumers to use the exact content-addressed Resources scaffold. It makes no ordinary Resource design, Human-Flourishing, Slice 00, M0, measured-plan, scientific, model, learner, deployment, production, promotion, or external-use claim.
