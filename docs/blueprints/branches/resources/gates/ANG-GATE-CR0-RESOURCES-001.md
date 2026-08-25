---
gate_id: ANG-GATE-CR0-RESOURCES-001
version: 2
supersedes_version: 1
owner: ANG-BP-RESOURCES
status: specified
activation_state: unusable_pending_revalidation
gate_class: bootstrap_scaffold_acceptance
release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
activation_revalidation: ANG-CR0-REVALIDATION-20260825-004
activation_base_commit: 7313a0d951c8f27af4c036e3b67059b7506cb3f1
leaf: ANG-WORK-CR0-RESOURCES-001@3
leaf_revision: 3
executor: ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001
independent_validator: ANG-AUTH-VALIDATOR-001
independent_reviewer_role: ANG-AUTH-SAFETY-APPROVER-001
independent_reviewer_instance: ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001
independent_reviewer_session_ref: codex-subagent:/root/safety_change_map
reviewer_acceptance: ACCEPTED
reviewer_reachability: reachable
reviewer_vocabulary_ack: ACK_ACCEPTED
allowed_dispositions: SCAFFOLD_ACCEPTED|SCAFFOLD_REJECTED|ESCALATE
result_recorder: ANG-BP-ROOT
decision_writer: ANG-AUTH-SAFETY-APPROVER-001
decision_path: docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md
rollback_baseline: ANG-BASELINE-CR0-RESOURCES-002@sha256:A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
normal_resource_design_gate: ANG-GATE-RESOURCE-DESIGN-001@1-NOT_RUN
---

# CR0 Resources scaffold gate

## Claim and non-equivalence

This gate decides only whether the exact local, synthetic `ResourceInventory` and `ExecutionPlan` schema scaffold is acceptable for CR0 consumers. `SCAFFOLD_ACCEPTED` is not a pass for the normal `ANG-GATE-RESOURCE-DESIGN-001`, `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a measured plan, actual resource discovery, model/GPU work, scientific success, deployment, or external use. Those states remain `NOT_RUN` or `NOT_PASSED`.

## Entry criteria

- Successor revalidation `ANG-CR0-REVALIDATION-20260825-004` has an independent `APPROVED` decision and Manifest v2 is `authorized`; PENDING is non-authorizing. Revalidation 003 remains immutable rejected history and cannot satisfy this entry criterion.
- Evidence decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0` is byte-identical and records `SCAFFOLD_ACCEPTED`; the normal Evidence gate remains `NOT_RUN`.
- The manifest, leaf, this gate, and baseline hashes match; all nine baseline targets are absent immediately before the first write.
- Executor, deterministic validator, reviewer instance, and root recorder match the frozen identities above and remain role-distinct.
- No denied capability, data, path, process, authority, or changed threshold is present.

## Procedure and precommitted thresholds

1. The executor creates only its eight literal outputs through leaf-bounded Codex `apply_patch`; it cannot write the decision receipt.
2. Run `pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1` twice, sequentially, with identical case counts and zero failures/errors, then run `pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1` once with zero errors.
3. Accept all three explicitly synthetic inventory tiers. Reject every overcommitted plan and require rollback/evaluation headroom, one inventory identity, one Evidence identity, reserved probe-provenance reference shape, and invariant logical contracts across tiers.
4. Reject any `measured=true`, probe-success claim, query-conditioned model routing, mid-transaction topology change, actual host/GPU observation, undeclared write, or non-synthetic fixture.
5. The executor writes the handoff first. The independent reviewer then verifies exact hashes, paths, commands, effects, roles, non-equivalence, and baseline before writing the sole decision receipt.

Any mismatch, failed run, unequal repeated result, missing negative control, ambiguity, or ceiling breach prevents `SCAFFOLD_ACCEPTED`.

## Required negative controls

Overcommitment; missing headroom; missing or multiple inventory/evidence identities; malformed reserved probe-provenance reference; fabricated measurement/probe success; tier-specific contract semantics; query-conditioned routing; topology change inside a transaction; actual probe/host/GPU access; network/package/background request; recovered/real-person data; `outputs/**`; outside-scope/reparse path; executor-authored decision; missing handoff; altered threshold; and expired or pending authority must fail closed.

## Required child gates

No normal child delivery gate is passed by this CR0 gate. `ANG-GATE-INVENTORY-001` and `ANG-GATE-EXECUTION-PLANNER-001` remain normal gates not run by bootstrap; `ANG-GATE-RESOURCE-PROBES-001` remains deferred to a successor leaf. Their non-completion does not become an implied pass.

## Evidence and decision identity

Required evidence is the exact executor handoff, hashes of all executor outputs, both Resource-test runs, the tree-validator result, effect/ceiling observations, Evidence predecessor identity, and baseline identity. The decision receipt identity is `ANG-EVID-CR0-RESOURCES-<sha256-of-exact-utf8-receipt-bytes>`.

The receipt must record gate/version, exactly one disposition (`SCAFFOLD_ACCEPTED`, `SCAFFOLD_REJECTED`, or `ESCALATE`), reviewer role/instance/session, executor, validator, recorder, base commit, manifest/leaf/gate/baseline/Evidence-decision/handoff hashes, exact changed paths, commands/results, rollback class, and all non-equivalence states. Unprefixed generic aliases are invalid. For successor gate version 2, the independent reviewer returned `ACK_ACCEPTED` for this exact vocabulary and confirmed independence and reachability; the durable acknowledgement is frozen in front matter and is not itself a gate disposition.

## Independent authority and write separation

Executor `ANG-EXEC-CODEX-ROOT-CR0-RESOURCES-001` cannot approve the gate or write `docs/blueprints/releases/construction-0/branch-receipts/RESOURCES.md`. Only accepted and reachable reviewer instance `ANG-REVIEW-CODEX-SAFETY-CR0-RESOURCES-001`, session `codex-subagent:/root/safety_change_map`, holding `ANG-AUTH-SAFETY-APPROVER-001`, may create that receipt. It may not write executor outputs, the leaf, gate, manifest, baseline, validator, or revalidation specification. `ANG-BP-ROOT` may record/use but never alter the disposition.

## Failure and rollback

On failure, rejection, escalation, denied action, ambiguity, or human stop: cease immediately, preserve the receipt and handoff, revert only baseline `restore_or_remove` paths individually, never unpack the release archive automatically, never broadly or recursively delete, and do not retry with broader authority. Dependents remain blocked.

## Waiver policy

No operator, validator, reviewer, recorder, model, or branch may waive an entry criterion, negative control, threshold, role separation, non-equivalence statement, or rollback rule. Broader work requires a successor ADR, policy, assessment, manifest, leaf, and explicit human authorization.
