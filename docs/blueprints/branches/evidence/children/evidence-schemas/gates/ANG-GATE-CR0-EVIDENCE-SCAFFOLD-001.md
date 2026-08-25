---
gate_id: ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001
version: 1
owner: ANG-BP-EVIDENCE-SCHEMAS
status: specified
gate_class: bootstrap_scaffold_acceptance
release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
independent_verifier: ANG-AUTH-VALIDATOR-001
independent_safety_reviewer: ANG-AUTH-SAFETY-APPROVER-001
independent_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-001
independent_reviewer_session_ref: codex-subagent:/root/safety_change_map
test_id: ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001
result_recorder: ANG-BP-ROOT
decision_writer: ANG-AUTH-SAFETY-APPROVER-001
decision_path: artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json
executor: ANG-EXEC-CODEX-ROOT-CR0-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
normal_technical_gate: ANG-GATE-EVIDENCE-SCHEMAS-001@1-NOT_RUN
---

# CR0 evidence-scaffold acceptance gate

## Non-equivalence and claim

This bootstrap gate tests only whether the local EVIDENCE schema scaffold was produced within Construction Release 0's exact authority and is coherent enough to serve as a provisional input to other CR0 scaffold leaves. A `SCAFFOLD_ACCEPTED` disposition does **not** pass `ANG-GATE-EVIDENCE-SCHEMAS-001`, the Human-Flourishing gate, the EVIDENCE branch, Slice 00, M0, or any technical or scientific milestone. It creates no production, model, learner, deployment, or external-use authority.

The normal EVIDENCE technical gate remains `specified_not_run` until its ordinary impact and Human-Flourishing prerequisites can be satisfied. CR0 consumers must bind this scaffold receipt and may not represent it as normal-gate evidence.

## Entry criteria

- CR0 manifest version 2 is authorized after its predeclared revalidation passes.
- `ANG-WORK-EVIDENCE-SCHEMAS-001` is `ready`, its semantic inputs match the manifest, and its absent-state baseline is reverified.
- Executor `ANG-EXEC-CODEX-ROOT-CR0-001`, independent deterministic verifier `ANG-AUTH-VALIDATOR-001`, SAFETY decision writer instance `ANG-REVIEW-CODEX-SAFETY-CR0-001` (`codex-subagent:/root/safety_change_map`) holding `ANG-AUTH-SAFETY-APPROVER-001`, and mechanical result recorder `ANG-BP-ROOT` are explicitly bound and distinct in authority. The reviewer instance accepted this bounded role before release authorization, is not the executor, and cannot execute or modify leaf/release inputs. The executor cannot write the decision artifact or issue/record its own disposition; the recorder cannot override the decision.
- `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1` and `ANG-POL-LOCAL-SCAFFOLD-001@1` remain current and unchanged.
- Every target, command, ceiling, stop condition, and rollback action is literal and within the CR0 boundary.

## Procedure and precommitted thresholds

1. Reverify the frozen baseline and semantic inputs before the first write.
2. Execute only the leaf's authorized writes and exact `python -B -m unittest tests.unit.evidence.test_evidence_schemas` command; bytecode/cache/temp output is forbidden.
3. Run the declared unit suite twice in clean foreground processes. Both runs must have the same case count and result.
4. Require 100% acceptance of declared positive fixtures and 100% typed rejection of declared negative and forbidden-visibility fixtures.
5. Verify canonical identity invariance for transport-only changes and identity change for every identity-material mutation.
6. Inspect dependency, path, process, network, GPU, data-provenance, output-size, and changed-file effects; every forbidden effect must remain absent.
7. Test identity `ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001` binds the two runs and effect inspection. The independent validator emits validation evidence without construction authority. The SAFETY reviewer checks the exact declared artifacts, receipts, scope/effects, authority separation, and non-equivalence, then creates the immutable decision artifact with exactly `SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`. The root steward may record/use that disposition without alteration.

No retry may broaden scope, suppress a negative control, change a threshold, or turn a failure into acceptance under the same gate identity.

## Required negative controls

- Duplicate JSON keys, ambiguous/non-finite numbers, Unicode normalization ambiguity, and unknown major versions.
- Raw low-entropy sealed-answer digest and payload-derived filename, error, size, or projection leakage.
- Learner access to sealed/authority content and held-out learner eligibility.
- Manifest threshold mutation under an unchanged identity.
- Missing required parent, plan, assessment, authorization, or visibility-policy reference.
- Out-of-scope path, undeclared dependency/import, `__pycache__`/`.pyc`/temporary output, network attempt, GPU/model use, recovered/real-person material, background process, and excess-resource effect.
- Executor self-approval, missing independent authority, expired/revoked assessment, or an attempt to treat this receipt as the normal technical gate.

## Required child gates and evidence

Required child gates: none. This is a release-scoped bootstrap acceptance record, not the node's normal completion gate.

`artifacts/control-plane/evidence-schemas/test-receipt.json` contains the content-addressed deterministic validation evidence and must not contain a gate disposition. It includes the leaf/manifest/baseline hashes, executor, validator and test identities, both run results, effect-receipt hash, exact changed paths, rollback hash, source/schema/fixture hashes, exact command and Python identity, elapsed/resource/output totals, and rollback verification.

Only bound reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-001` holding `ANG-AUTH-SAFETY-APPROVER-001` may create `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`, the content-addressed decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-<sha256>`. It binds the gate/version/disposition, leaf/manifest/baseline/test/effect/handoff hashes, executor identity, `reviewer_role`, `reviewer_instance`, `reviewer_session_ref`, exact changed paths, rollback identity, and explicit `NOT_RUN` values for the normal technical and Human-Flourishing gates. The executor's write scope excludes this file; a different or relabeled reviewer requires release revalidation before any write.

## Result semantics, failure, and rollback

`SCAFFOLD_ACCEPTED` in the immutable independent decision artifact permits only the manifest-listed CR0 scaffold consumers to use the exact content-addressed outputs. Any different release, normal branch completion, learner/model use, or external use requires ordinary gates and new authority. The gate specification itself is never edited to record a run result.

`SCAFFOLD_REJECTED` or `ESCALATE` blocks all consumers. Stop and preserve immutable test/effect receipts, handoff, and the independent decision. Restore or remove only the baseline's literal `restore_or_remove` targets; never delete or rewrite its `preserve_on_failure` evidence. Keep the normal `ANG-GATE-EVIDENCE-SCHEMAS-001` unpassed. Leakage, authority ambiguity, semantic-input drift, or forbidden effects invalidate the scaffold identity and require a successor leaf/baseline/assessment.

## Waiver policy

No executor, producer, learner, branch owner, schedule concern, or technical result may waive scope, identity, visibility, role separation, negative controls, Human-Flourishing non-equivalence, or rollback requirements.
