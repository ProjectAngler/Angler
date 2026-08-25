---
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
version: 2
status: authorized
supersedes_version: 1
revalidation_id: ANG-CR0-REVALIDATION-20260825-002
revalidation_status: PASS
revalidated_at: 2026-08-25
release_class: local_control_plane_scaffold
decision: ANG-ADR-0002
policy: ANG-POL-LOCAL-SCAFFOLD-001@1
bootstrap_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
bootstrap_assessment_sha256: 181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC
bootstrap_gate: ANG-GATE-CONSTRUCTION-RELEASE-0-001@1
evidence_scaffold_gate: ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001@1
evidence_scaffold_disposition: NOT_RUN
normal_evidence_schema_gate_status: NOT_RUN
independent_safety_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
independent_safety_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-001
independent_safety_reviewer_session_ref: codex-subagent:/root/safety_change_map
first_leaf_executor: ANG-EXEC-CODEX-ROOT-CR0-001
formal_human_flourishing_gate_status: NOT_RUN
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

# Construction Release 0 manifest

## Release capsule

This release registers one branch-owned EVIDENCE Tier-4 leaf, seven release-scoped branch leaves, and one integration leaf. Only EVIDENCE-SCHEMAS, SAFETY, RESOURCES, WORLDS, SCIENCE, and INTEGRATION may become activation candidates in CR0. RUNTIME, LEARNING, and TOOLS remain `not_ready`. Work may create documentation, implementation-independent JSON schemas, foreground standard-library validators, and synthetic non-person fixtures only. It authorizes no model/GPU work, network, package, external tool, recovered material, background process, deployment, promoted runtime mutation, or scientific claim.

Bootstrap `ALLOW` under [the SAFETY-owned impact record](../../branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md) means only that these reversible construction tasks may start. EVIDENCE progression uses the separate scaffold-only `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`; even `SCAFFOLD_ACCEPTED` does not pass the normal `ANG-GATE-EVIDENCE-SCHEMAS-001`, which remains `NOT_RUN`. Neither bootstrap gate is `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, or product/runtime approval.

## Authority and governing artifacts

- [Accepted construction decision](../../decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md)
- [Accepted evidence identity decision](../../decisions/ANG-ADR-0003-CANONICAL-EVIDENCE-IDENTITY.md)
- [Accepted CR0 interface-stabilization decision](../../decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md)
- [Local scaffold policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md)
- [Human-impact assessment](../../branches/safety/assessments/ANG-ASSESS-CONSTRUCTION-RELEASE-0-001.md)
- [Bootstrap construction gate](../../branches/safety/gates/ANG-GATE-CONSTRUCTION-RELEASE-0-001.md)
- [Bootstrap safety-design evidence](../../branches/safety/evidence/ANG-EVID-CR0-SAFETY-DESIGN-001.md)
- [EVIDENCE scaffold acceptance gate](../../branches/evidence/children/evidence-schemas/gates/ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001.md)
- [Normal EVIDENCE schema gate—specified, not run](../../branches/evidence/children/evidence-schemas/gates/ANG-GATE-EVIDENCE-SCHEMAS-001.md)
- [Root capsule](../../ROOT_CAPSULE.md)
- [Recursive protocol](../../PROTOCOL.md)
- [Integration spine](../../INTEGRATION_SPINE.md)
- [Interface registry](../../INTERFACE_REGISTRY.md)

The project owner delegates bounded construction authority through the root coordinator. Before a leaf starts, its concrete executor identity must already be bound in that leaf's frozen front matter and in this manifest; an unassigned, branch-only, or mismatched executor keeps the leaf blocked. The handoff is an execution output, not a pre-start authority source: it must record the same frozen executor binding before independent review. Leaf executors cannot approve their own gate or broaden the release.

For the first leaf, deterministic verifier `ANG-AUTH-VALIDATOR-001` is distinct from executor `ANG-EXEC-CODEX-ROOT-CR0-001`. Independent SAFETY reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-001`, session `codex-subagent:/root/safety_change_map`, holds role `ANG-AUTH-SAFETY-APPROVER-001`; it accepted this bounded role and is barred from executor and release-spec writes. Only that bound reviewer may author `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`, whose required fields include `reviewer_role`, `reviewer_instance`, and `reviewer_session_ref`. The root recorder may copy/use but not alter its disposition.

## Frozen semantic inputs

| Input | Frozen identity |
|---|---|
| Tier-0 blueprint | `ANG-BP-ROOT@2`; SHA-256 `A573A281D9733587891AB64B019170B40D41ACC341BA446B11EF764A63A8CE13` |
| Root capsule | source/capsule revision 2; SHA-256 `67F60E24343F628A6C108EFFF642DAF7CEBD52AC4A7F90DB599E32F99EBA6090` |
| Constitution | `ANG-CON-HUMAN-FLOURISHING-001@1`; SHA-256 `A72B18C5B718829C030C33B7AFCA0F3F53A33232CE17358E6985159B04108EBA` |
| Blueprint index | SHA-256 `D04CB8C304F54A71C8444EF7124D5D63663F498E0432443913FC74E92857B896` |
| Protocol | SHA-256 `C922170BEDE154056E68732242DD489A738EF64E75453016B816F14D3E02C0CB` |
| Interface registry | SHA-256 `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7` |
| Dependency graph | SHA-256 `6F6079FF224247A1BFE3E4111785C450CD772DF57D11594672379FA78B15CA5D` |
| Integration spine | SHA-256 `78EC4F6909E8D2BDE478FD2557147C0933BE7940F066381995813660A2B53833` |
| Construction decision | `ANG-ADR-0002`; SHA-256 `C5B97294FD53AFA9F95E0C28AD6F36C9A7861DF07B50B714F489ED1F37873753` |
| Evidence identity decision | `ANG-ADR-0003`; SHA-256 `CBD8ACDB5EA5D0B217A047DFDC93BF36A36F60A40DE9E2BFE2C97250EF95E20F` |
| Interface stabilization decision | `ANG-ADR-0004`; SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6` |
| EVIDENCE design | `ANG-BP-EVIDENCE@3`; SHA-256 `56DCE997BF6F002BB9202C144913B90D2A28A64203885B8A3730671AFB16ED48` |
| EVIDENCE-SCHEMAS design | `ANG-BP-EVIDENCE-SCHEMAS@1`; SHA-256 `2E50B0BB3F016AC0FEA3B5E72FFCDE7E945BD1D82DDA553722986339FF1F93FB` |
| EVIDENCE schema work leaf | `ANG-WORK-EVIDENCE-SCHEMAS-001@1`; SHA-256 `5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289` |
| EVIDENCE schema baseline | `ANG-BASELINE-EVIDENCE-SCHEMAS-001`; SHA-256 `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F` |
| EVIDENCE scaffold gate | `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001@1`; SHA-256 `A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5` |
| Normal EVIDENCE schema gate | `ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN`; SHA-256 `CCDB0782B520328AA5B0A04C6684E16EB9390B8338B61FC6BBA1CB8913A49210` |
| SAFETY design | `ANG-BP-SAFETY@2`; SHA-256 `4376A7C61D1CAFFB25671858A79C1A61185721EE850620E4A932B6D5002F8A5D` |
| RESOURCES design | `ANG-BP-RESOURCES@2`; SHA-256 `796C1838973BB24B41416968D700DF2FD760A4BA7ECA854E7ECCB4E12B814F53` |
| WORLDS design | `ANG-BP-WORLDS@2`; SHA-256 `B79D0B407F9864AB7C1DB899BA4516507C775179B843B022D91861696A39DAEA` |
| SCIENCE design | `ANG-BP-SCIENCE@2`; SHA-256 `854DAB17C6F1CB29DF5466FB18F6D9DDFAAE72039CCC7529A7D86A4F15BF0A40` |
| Construction policy | `ANG-POL-LOCAL-SCAFFOLD-001@1`; SHA-256 `23D04D544208C7273BA6C7860CC788CDD81640C8DD8236FFD1FED1F2D77495C6` |
| Bootstrap assessment | `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1`; SHA-256 `181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC` |
| Bootstrap construction gate | `ANG-GATE-CONSTRUCTION-RELEASE-0-001@1`; SHA-256 `B768F7669241A0C3432E95E0DDB900AE5A007B2369AB3E4D073F473434DE8EEB` |
| Bootstrap safety-design evidence | `ANG-EVID-CR0-SAFETY-DESIGN-001`; SHA-256 `0BA8237AB624C980870F263B79EDC1F3974058E8C313F1C422D993FD99FB1F4A` |
| SAFETY work leaf | `ANG-WORK-CR0-SAFETY-001@1`; SHA-256 `A7D9C4FC82C9D7C8470B8085BB39F93D64A1B05A9C83614F9B47746860A193EE` |
| RESOURCES work leaf | `ANG-WORK-CR0-RESOURCES-001@1`; SHA-256 `19465FFFC741BFFB2F7E8BBDCA16CCA0F0132B906F8F3481F9CA2C85FA13EE6A` |
| WORLDS work leaf | `ANG-WORK-CR0-WORLDS-001@1`; SHA-256 `3ADCE6430F40D27AA2F5027393898BC448BA3616EE1BAC72E513E1F1AA34E704` |
| SCIENCE work leaf | `ANG-WORK-CR0-SCIENCE-001@1`; SHA-256 `363D4848173BE0D6EBAE840BB71D01712D8F4E08891ACAB3DA91F9C841C73798` |
| RUNTIME work leaf | `ANG-WORK-CR0-RUNTIME-001@1`; SHA-256 `A00FF073CB671DA626B991FBC00045AD24FC354E5C21B8B8ABD542367CBC355C` |
| LEARNING work leaf | `ANG-WORK-CR0-LEARNING-001@1`; SHA-256 `5D69A7B0DE3DF06E647ADF538C88E933514ED3F97855D2F10F242BFCEAE83E44` |
| TOOLS work leaf | `ANG-WORK-CR0-TOOLS-001@1`; SHA-256 `092796D9D02DE5E3A561C460257E77E65660D5338EE8E7A26261922BEDB6D551` |
| INTEGRATION work leaf | `ANG-WORK-CR0-INTEGRATION-001@1`; SHA-256 `6A8378F9D5241078F80FCEC8895391F1B88731A3C9AFC08AA41E3872DC92B472` |
| Rollback archive | SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3` |

Status files are continuation summaries rather than semantic inputs. Workers must read current status for blockers but do not invalidate a leaf merely because a roll-up was refreshed. Any change to a semantic input above stops all unopened leaves until this manifest is superseded or explicitly revalidated.

## Version 2 authoring-mechanism revalidation

Revalidation record `ANG-CR0-REVALIDATION-20260825-001`, dated 2026-08-25, treats the authoring-mechanism clarification as a material authority change. ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001@1`, the LOW bootstrap assessment, and the CR0 bootstrap gate now explicitly admit the host-provided Codex `apply_patch` primitive as the sole authoring mechanism and limit it to the active leaf's literal output paths. Shell redirection, ad-hoc writer scripts, bulk writers, and every undeclared path remain denied.

This change narrows and makes executable the already leaf-bounded local authority. It grants no external-tool status or authority, no network/package/model/GPU/background capability, no recovered or real-person data access, no deployment/promotion authority, and no human-flourishing, Slice-00, M0, scientific, or technical pass.

While version 2 remained pending, ADR-0004 froze the detailed RESOURCES/WORLDS/SCIENCE interfaces, and EVIDENCE bootstrap progression was separated from the normal technical gate. The executor may write only its 17 literal targets; the independently bound SAFETY reviewer alone may write the separately baselined `scaffold-gate-decision.json`. The exact command is `python -B -m unittest tests.unit.evidence.test_evidence_schemas`, and rollback preserves evidence classified `preserve_on_failure`. These additions narrow role, output, cache, and gate semantics; they create no new external or normal-gate authority.

| Revalidated input | Version 1 SHA-256 | Version 2 SHA-256 |
|---|---|---|
| ADR-0002 | `E881E4665A6E41ABAAD85961CA33A01E97111E7A2A6283A7B61C5A49AC6CA7B2` | `C5B97294FD53AFA9F95E0C28AD6F36C9A7861DF07B50B714F489ED1F37873753` |
| Local scaffold policy | `04E5DE3061E5D20305B9B846556971DB1600981EB44C7A2363E122D19B68DF79` | `23D04D544208C7273BA6C7860CC788CDD81640C8DD8236FFD1FED1F2D77495C6` |
| Bootstrap assessment | `F52324553A45B424201D4E96800A77BC18F8C593297A7B6406643CABE0EDA16A` | `181BAC18E5EA0711F22D54BF4DE49DDA33B4DCB09C708439FE4A641366A3D8CC` |
| Bootstrap gate | `8D0817BE278A7DF842D83A635B1DBC312E94934261B1EA58CED6FC4916886B5C` | `B768F7669241A0C3432E95E0DDB900AE5A007B2369AB3E4D073F473434DE8EEB` |
| SAFETY evidence manifest | `5451909BCBB30FCCBCD39EE2BC32739DD3803E1109C0BC60C996798691CA8DAF` | `0BA8237AB624C980870F263B79EDC1F3974058E8C313F1C422D993FD99FB1F4A` |

Revalidation status: **PASS** on 2026-08-25. Version 1 is superseded. Version 2 is authorized only for the exact leaf statuses, roles, paths, commands, gates, ceilings, dependencies, frozen identities, and prohibitions recorded here; any semantic-input or authority change stops unopened work and requires a successor revalidation.

| Pre-authorization check | Recorded result |
|---|---|
| `pwsh -NoProfile -NonInteractive -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1` | PASS — bootstrap non-equivalence, authority, ceilings, assessment, and rollback identity validated |
| `pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0.ps1` | PASS — 8 release leaves plus 1 branch-owned EVIDENCE leaf, 100 unique role-owned outputs, rollback and frozen inputs verified |
| `pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1` | PASS — 21 concrete nodes, 38 declared children, 136 Markdown files |

These were static revalidation checks; they executed no construction leaf and created no scaffold disposition. `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`, the normal EVIDENCE technical gate, the Human-Flourishing gate, Slice 00, and M0 remain unrun/unpassed as stated above.

## Same-version revalidation 002 — executor binding and disposition identity

Revalidation record `ANG-CR0-REVALIDATION-20260825-002`, dated 2026-08-25, is **PASS**. It corrects two authorization defects without executing a leaf:

1. Executor authority is prebound by `first_leaf_executor`, the frozen EVIDENCE leaf front matter, and that leaf's pinned hash. The absent pre-start handoff is not an authority source; the executor-created handoff must later record the identical binding before independent review.
2. The scaffold-gate claim now names only the registered disposition `SCAFFOLD_ACCEPTED`. Generic `ACCEPTED` is not an alias and is rejected.

The scaffold-gate hash changes from `382309358594D4E6EE5CA207EDCBD417956420FD9410DB6B4225A1AAB9BC915A` to `A883FE8799BF06390B9691F1F15F06763DAD3FB5A7210F958A2AADA7C68548F5`. The frozen leaf remains SHA-256 `5CF4A9DE5337D1C52F3D8E1CFEC6404425853808F65F31DBA6BB7284EDC53289`, with `execution_owner: ANG-EXEC-CODEX-ROOT-CR0-001`. All other frozen inputs remain unchanged.

No construction leaf or scaffold test has run, and no handoff or gate-decision artifact exists. Before authorization resumed, two consecutive pending-state rounds produced the same results:

| Check | Round 1 | Round 2 |
|---|---|---|
| SAFETY static design validation | PASS | PASS |
| CR0 release validation | PASS — 8 release leaves plus 1 branch-owned EVIDENCE leaf; 100 unique role-owned outputs | PASS — same counts and result |
| Whole blueprint tree validation | PASS — 21 concrete nodes, 38 declared children, 136 Markdown files | PASS — same counts and result |

Manifest v2 is therefore reauthorized only for its exact frozen scope. Any executor-binding, handoff-authority, disposition-vocabulary, gate, or other semantic change stops unopened work and requires another explicit revalidation.

## Authorized leaves and dependency order

| Leaf | Accountable owner | Dependencies | Status | CR0 activation candidate | Gate |
|---|---|---|---|---|---|
| [`ANG-WORK-CR0-SAFETY-001`](../../work/slice-00/ANG-WORK-CR0-SAFETY-001.md) | `ANG-BP-SAFETY` | concrete executor binding and gate recheck | not_ready | yes, after blocker | `ANG-GATE-CR0-SAFETY-001` |
| [`ANG-WORK-EVIDENCE-SCHEMAS-001`](../../branches/evidence/children/evidence-schemas/work/ANG-WORK-EVIDENCE-SCHEMAS-001.md) | `ANG-BP-EVIDENCE-SCHEMAS` | ADR-0003/0004, synchronized registry, exact baseline, distinct executor/validator/reviewer roles, current bootstrap authority | ready | yes | `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` |
| [`ANG-WORK-CR0-RESOURCES-001`](../../work/slice-00/ANG-WORK-CR0-RESOURCES-001.md) | `ANG-BP-RESOURCES` | independent EVIDENCE `SCAFFOLD_ACCEPTED` decision hash pinned; normal EVIDENCE gate stays `NOT_RUN` | blocked | yes, after dependency | `ANG-GATE-CR0-RESOURCES-001` |
| [`ANG-WORK-CR0-WORLDS-001`](../../work/slice-00/ANG-WORK-CR0-WORLDS-001.md) | `ANG-BP-WORLDS` | safety plus pinned independent EVIDENCE scaffold decision | blocked | yes, after dependencies | `ANG-GATE-CR0-WORLDS-001` |
| [`ANG-WORK-CR0-SCIENCE-001`](../../work/slice-00/ANG-WORK-CR0-SCIENCE-001.md) | `ANG-BP-SCIENCE` | safety, pinned independent EVIDENCE scaffold decision, resources, worlds | blocked | yes, after dependencies | `ANG-GATE-CR0-SCIENCE-001` |
| [`ANG-WORK-CR0-RUNTIME-001`](../../work/slice-00/ANG-WORK-CR0-RUNTIME-001.md) | `ANG-BP-RUNTIME` | successor release | not_ready | no | `ANG-GATE-CR0-RUNTIME-001` |
| [`ANG-WORK-CR0-LEARNING-001`](../../work/slice-00/ANG-WORK-CR0-LEARNING-001.md) | `ANG-BP-LEARNING` | successor release | not_ready | no | `ANG-GATE-CR0-LEARNING-001` |
| [`ANG-WORK-CR0-TOOLS-001`](../../work/slice-00/ANG-WORK-CR0-TOOLS-001.md) | `ANG-BP-TOOLS` | M3 and successor release | not_ready | no | `ANG-GATE-CR0-TOOLS-001` |
| [`ANG-WORK-CR0-INTEGRATION-001`](../../work/slice-00/ANG-WORK-CR0-INTEGRATION-001.md) | `ANG-BP-ROOT` | five exact predecessor decision/receipt hashes, including independent EVIDENCE scaffold decision | blocked | yes, after dependencies | `ANG-GATE-CR0-INTEGRATION-001` |

`blocked` and `not_ready` are not permission to begin. The release steward may change only a listed activation candidate to `ready`, and only after recording predecessor evidence and satisfying the canonical construction gate. Branch scaffolds are proposals until their normal design gates approve them; construction completion gives no milestone credit. RUNTIME, LEARNING, and TOOLS require a successor release and cannot be activated by CR0.

## Shared execution ceiling

All leaves inherit `ANG-POL-LOCAL-SCAFFOLD-001@1`: authoring only through host-provided Codex `apply_patch` on literal active-role outputs; one foreground PowerShell or Python 3.11 standard-library process, as explicitly named by the active leaf; no parallelism; 60 seconds per command; 600 seconds (10 minutes) active execution per leaf; 1 logical CPU core; 512 MiB working set; 25 MiB total new files; 5 MiB per file; 1 MiB per fixture; zero accelerator use, network, packages, spend, and background work. Shell redirection, ad-hoc writer scripts, bulk rewrites, cross-role writes, and undeclared commands are forbidden.

`outputs/**`, recovered code/data, models, adapters, tokenizers, updaters, actual GPU/host probes, external tools, real-person data, deployment, and promoted runtime mutation are forbidden. Each leaf narrows this policy further with literal output paths and stop conditions.

## Release-level predeclared tests

Run in the repository root, in this order:

```powershell
pwsh -NoProfile -NonInteractive -File docs/blueprints/work/slice-00/validate-construction-release-0.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

After activated leaf execution, the integration leaf additionally runs every exact branch-local test command listed in the six candidate receipts. No alternative test framework or downloaded validator may be substituted.

## Release gate: `ANG-GATE-CR0-INTEGRATION-001`

The construction release may close only when:

- every activated candidate leaf's literal outputs exist and no undeclared path changed;
- every predeclared local synthetic test passes within the ceiling;
- one synthetic Slice-00 record binds charter, contract, evidence, resource-plan, trust-boundary, policy, and rollback identities;
- missing, mismatched, forbidden-scope, expired, and non-synthetic fixtures fail closed;
- branch scaffolds contain no runtime implementation or scientific-success claim, and RUNTIME/LEARNING/TOOLS remain `not_ready`;
- root-controlled review records pass/fail and unresolved dissent.
- the independent EVIDENCE scaffold decision is pinned and never represented as `ANG-GATE-EVIDENCE-SCHEMAS-001`, which remains `NOT_RUN`.

Closing this gate means only that the construction package is internally coherent. `ANG-GATE-HUMAN-FLOURISHING-001`, `ANG-SLICE-00-CONTROL`, every Tier-1 design gate, and `ANG-M0-BLUEPRINT` remain separate and pending.

## Stop, revocation, and rollback

Any policy stop condition pauses the affected leaf and all dependents. A policy, assessment, semantic-input, authority, or archive mismatch pauses the whole release. The worker preserves a handoff and reverts only leaf-owned files. Only root/human authority may use the full rollback archive, after validating its hash and protecting unrelated later work.

## Required release handoff

Each leaf records executor identity, base snapshot, exact files, commands, elapsed time, approximate resource use, tests/evidence, deviations, blockers, and scoped rollback. The independent EVIDENCE decision additionally records `reviewer_role`, `reviewer_instance`, and `reviewer_session_ref`; the executor may not write it. The release steward rolls these up without changing root or branch status automatically.
