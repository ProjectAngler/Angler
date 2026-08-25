---
gate_id: ANG-GATE-CR0-CONTINUITY-001
version: 1
owner: ANG-BP-ROOT
status: specified
activation_state: unusable_pending_revalidation
activation_revalidation: ANG-CR0-REVALIDATION-20260825-005
gate_class: bootstrap_continuity_reconciliation
release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
release_manifest_version: 3
leaf: ANG-WORK-CR0-CONTINUITY-001@1
baseline: ANG-BASELINE-CR0-CONTINUITY-001
baseline_sha256: EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7
executor: ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001
executor_task_id: 01a03a80-bb20-7d01-acf6-f50ca4856be5
independent_verifier: ANG-AUTH-VALIDATOR-001
independent_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
independent_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001
independent_reviewer_session_ref: codex-subagent:/root/safety_change_map
reviewer_vocabulary_ack: ACK_ACCEPTED
allowed_dispositions: RECONCILIATION_ACCEPTED|RECONCILIATION_REJECTED|ESCALATE
test_id: ANG-TEST-CR0-CONTINUITY-001
result_recorder: ANG-BP-ROOT
decision_writer: ANG-AUTH-SAFETY-APPROVER-001
decision_path: docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md
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

- Manifest v3 is `authorized` with revalidation `ANG-CR0-REVALIDATION-20260825-005` recorded `PASS`, and its separately authored decision exists with exactly `APPROVED` and its bound hash.
- `ANG-WORK-CR0-CONTINUITY-001@1` is the sole ready executable leaf in Manifest v3. Executor `ANG-EXEC-CODEX-ROOT-CR0-CONTINUITY-001` is prebound to persistent task `01a03a80-bb20-7d01-acf6-f50ca4856be5` in both leaf and manifest.
- `ANG-BASELINE-CR0-CONTINUITY-001` SHA-256 `EA997CD8B78817514A906BECC85491D4ACF74BB650D0324006C6C806D7A5A9F7` is reverified immediately before the first write: thirteen targets exist with their exact hashes and four targets are absent.
- ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001@1`, and `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1` remain current. This leaf is a narrower documentation-only instance of their existing LOW local-scaffold ceiling and adds no capability, data source, effect, resource, or authority.
- The accepted Evidence decision, its bound leaf/gate/baseline/test/effect/handoff/execution-manifest identities, the accepted Resources receipt and its eight output/handoff identities, rejected revalidation 003, and approved revalidation 004 all match the immutable hashes in Manifest v3.
- Independent verifier `ANG-AUTH-VALIDATOR-001`, reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-CONTINUITY-001` at `codex-subagent:/root/safety_change_map`, and recorder `ANG-BP-ROOT` are distinct from the executor. The reviewer returned a fresh `ACK_ACCEPTED` specifically for `RECONCILIATION_ACCEPTED | RECONCILIATION_REJECTED | ESCALATE`; no earlier vocabulary acknowledgement is reused or inferred. The reviewer is reachable. Only that reviewer may create the receipt; it may not edit the packet or executor outputs.

## Procedure and precommitted thresholds

1. Reverify the baseline and every immutable historical hash with read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256`. Stop on any mismatch.
2. The executor uses Codex `apply_patch` only to author or update the fifteen non-handoff executor outputs: thirteen pre-existing projections, the append-only addendum, and the deterministic test. It does not touch the executor handoff yet, the reviewer receipt, or any historical artifact.
3. Run, sequentially from the repository root, the exact continuity test twice:

   `pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1`

4. Both runs must report the same case count, zero failures, and byte-identical success semantics. Then run exactly once:

   `pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1`

5. The continuity test must confirm all thirteen repaired projections agree on accepted/historical state, all frozen identities, the active one-shot continuity leaf, and every non-equivalence state. It must reject every predeclared stale or broadened-state mutation. It does not require the not-yet-authored executor handoff during these runs.
6. Only after both continuity runs and the tree validator pass, the executor writes its append-only handoff as output sixteen and records exact before/after hashes, commands, case counts, effects, ceilings, rollback state, and the pending independent review. The handoff contains no gate disposition.
7. The bound SAFETY reviewer independently inspects the exact packet, outputs, tests, changed paths, immutable hashes, role separation, effects, rollback classes, and non-equivalence. It alone may write `docs/blueprints/releases/construction-0/branch-receipts/CONTINUITY.md` with exactly `RECONCILIATION_ACCEPTED`, `RECONCILIATION_REJECTED`, or `ESCALATE`.
8. Root may mechanically record or consume the exact disposition but cannot alter it. Even after `RECONCILIATION_ACCEPTED`, Manifest v3 is exhausted and cannot activate another leaf.

Pass requires every declared assertion and negative control to pass twice identically, the tree validator to pass once, exactly sixteen executor paths and one reviewer path, no immutable-byte change, and no forbidden effect. No retry may broaden scope, suppress a negative control, change a threshold, or relabel a disposition.

## Required negative controls

- A stale projection saying Evidence has no code/receipt/decision, its historical leaf is still executable, Resources revalidation 004 is still pending, Resources has no receipt/output, or either completed leaf should run next.
- Missing or mismatched Evidence decision/leaf/gate/baseline/test/effect/handoff/execution-manifest identity, Resources receipt/output/handoff identity, base commit, release archive, role, task, or reviewer acknowledgement.
- Treating `RECONCILIATION_ACCEPTED`, `SCAFFOLD_ACCEPTED`, bootstrap `ALLOW`, deterministic test success, or a tree-validation pass as a normal technical, Human-Flourishing, Slice-00, M0, scientific, deployment, production, or promotion pass.
- Unprefixed generic aliases such as ACCEPTED or REJECTED in place of the registered reconciliation dispositions.
- Executor self-approval, reviewer/executor identity collision, receipt written before the executor handoff, reviewer modification of packet/executor outputs, or root alteration of reviewer disposition.
- Any write outside the sixteen executor paths or one reviewer receipt; any mutation of historical evidence, decisions, receipts, handoffs, schemas, source, fixtures, tests, policy, assessment, ADR, or rollback archive.
- Network/DNS/socket/browser/API/telemetry/remote Git, model/GPU/probe, package/plugin/dependency/tool installation, recovered or real-person data, credentials, host enumeration, ad hoc directory enumeration, background process, deployment, external effect, bytecode/cache/temp output, broad deletion, or resource-ceiling expansion. The only permitted traversal is traversal performed internally by the exact declared `tools/validate_blueprint_tree.ps1` command.

## Required evidence and receipt identity

The executor evidence is:

- `tests/synthetic/slice00/continuity/Test-Cr0ContinuationConsistency.ps1`;
- `docs/blueprints/branches/evidence/children/evidence-schemas/handoffs/2026-08-25-cr0-scaffold-accepted.md`;
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-CONTINUITY-001.md`;
- exact before/after hashes for every executor output and both test transcripts recorded in the handoff.

The independent receipt identity is `ANG-EVID-CR0-CONTINUITY-<sha256-of-exact-utf8-receipt-bytes>`. The receipt must bind gate/version/disposition, Manifest-v3 hash, leaf/gate/baseline/test/handoff hashes, executor and persistent task, validator/test identity, reviewer role/instance/session, `ACK_ACCEPTED`, the exact sixteen-plus-one path partition, before/after hashes, commands/results, immutable Evidence and Resources identities, rollback classes, effect denials, and explicit `NOT_RUN`/`NOT_PASSED` values. The executor cannot create or edit it.

## Failure and rollback

On any failure, stop immediately and record `RECONCILIATION_REJECTED` or `ESCALATE`; do not reinterpret partial output as success. Preserve the append-only post-scaffold addendum, executor handoff, and any independent receipt. Restore each of the thirteen pre-existing projection targets individually to its exact baseline bytes and hash. Restore or remove only the continuity test according to its absent baseline. Never use broad, recursive, globbed, directory-level, or archive-wide deletion/restoration. Historical evidence is never changed or deleted.

Failure leaves all normal gates `NOT_RUN`, Slice 00 and M0 `NOT_PASSED`, every other construction leaf unauthorized, and a successor packet mandatory.

## Waiver policy

No waiver exists for an immutable hash, path boundary, role separation, test, negative control, rollback class, registered disposition, non-equivalence statement, or denied capability. Any ambiguity or mismatch yields `ESCALATE` and requires a successor ID/revalidation.
