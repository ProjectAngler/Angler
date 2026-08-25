---
contract_id: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
version: 1
owner: ANG-BP-SAFETY
status: active
constitution: ANG-CON-HUMAN-FLOURISHING-001@1
producer: independent safety assessment process
consumers:
  - all promotable-artifact owners
  - ANG-BP-RUNTIME
  - ANG-BP-SCIENCE
  - ANG-BP-EVIDENCE
  - human promotion authority
---

# Human-impact assessment and authorization receipt

## Purpose

Provide a protected, attributable, time-bounded decision about whether a specific artifact may proceed under the Human-Flourishing Constitution. Capability or scientific performance cannot compensate for a missing or non-`ALLOW` disposition.

## Applies to

- model and plastic-state promotion;
- updater and consolidation changes;
- tools and tool permissions;
- environments and curricula;
- execution/resource plans whose reach or authority changes;
- dependencies and recovered code;
- competence migration;
- controller, updater, tool, or infrastructure code changes;
- deployment or material scope expansion.

Documentation-only and isolated low-impact work still records a proportionate assessment. A preapproved low-impact profile may shorten review but cannot waive constitutional scope, evidence identity, or escalation triggers.

## Required fields

```text
contract_id and version
constitution_id and revision
assessment_policy_id and revision
assessment_id and content hash
authorization_kind: PROMOTION | BOOTSTRAP_WORK
assessed subject tuple:
  artifact type and contract version
  final payload content_id or protected commitment
  typed parent identities
  intended scope, resource plan, permissions, execution context, and duration
intended purpose and authorized scope
resource plan, permissions, deployment context, and duration
affected users, non-users, groups, and foreseeable future people
vulnerable or poorly served groups considered
claimed benefits and supporting evidence
harm/misuse scenarios:
  severity, probability, uncertainty, exposure,
  duration, reversibility, distribution, and evidence
rights and agency findings:
  life, dignity, bodily/psychological integrity, liberty,
  consent, privacy, truthfulness, fairness, access, appeal, and control
alternatives, including smaller trial and inaction
mitigations, monitoring, incident response, notice, opt-out, stop, and rollback
residual risks and unresolved disagreements
impact class: LOW | MATERIAL | HIGH | CATASTROPHIC_POTENTIAL
disposition: ALLOW | DENY | ESCALATE
conditions and authorization ceiling
independent assessor and human-authority identities
issued_at, expires_at, and mandatory review triggers
authenticity/signature information
```

For `BOOTSTRAP_WORK`, the assessment must also bind:

```text
bootstrap_release_id and accepted ADR
human authorization basis and role identity
exact repository root
exact leaf read paths, write paths, and permitted command classes
executor identity and prohibition on self-approval
synthetic-data-only rule and excluded paths/data classes
network, package, model, GPU, persistence, and external-effect ceilings
numeric time, CPU, memory, process, disk, output, and spending ceilings
constitution-linked safety-fixture rule
ESCALATE expectation for unresolved value conflicts
stop and revocation conditions
rollback artifact path and independently verified hash
bootstrap expiry and successor-control termination condition
```

## Disposition semantics

- `ALLOW` authorizes only the named artifact, purpose, scope, permissions, conditions, plan, and time window.
- `DENY` forbids promotion or execution in the proposed scope and requires rejection/rollback.
- `ESCALATE` is not temporary permission. It blocks advancement until legitimate human authority resolves the recorded conflict and a successor assessment is issued.

Missing, altered, unsigned, expired, wrong-artifact, wrong-parent, wrong-plan, condition-violating, `DENY`, or `ESCALATE` receipts are ineligible.

## Non-circular artifact binding

Project decision `ANG-ADR-0003` fixes the ordinary promotion sequence:

1. commit the final payload content and complete subject tuple;
2. issue this immutable assessment against that tuple without depending on a future final `artifact_id`;
3. create the final evidence envelope recording the assessment reference/requirement;
4. create `ANG-CTR-AUTHORIZATION-BINDING-001` joining the final artifact envelope, this assessment, policy/constitution, authority, conditions, and exact subject tuple;
5. let promotion independently validate that current binding plus scientific policy.

The binding must reject any content, parent, plan, permission, scope, context, duration, condition, policy, authority, disposition, expiry, or revocation mismatch. An artifact-local authorization flag is never sufficient. `BOOTSTRAP_WORK` binds an exact leaf/profile/baseline rather than a promotable artifact and cannot be reused as an ordinary promotion binding.

## `BOOTSTRAP_WORK` semantics

`BOOTSTRAP_WORK` exists only to resolve control-plane bootstrap circularity: it may authorize tightly bounded construction of the evidence, schema, gate, and enforcement mechanisms needed before ordinary promotion authorization exists.

- A bootstrap `ALLOW` authorizes work, not promotion.
- It is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a Tier-1 design gate, or any scientific/technical gate.
- It cannot authorize model acquisition/execution, training, adapter/updater mutation, GPU use, network activity, package/tool/dependency installation, recovered outputs, real-person data, out-of-scope files, background/persistent services, deployment, external effects, tool acquisition, autonomous continuation, or self-modification.
- It must be narrower than its accepted ADR and bootstrap policy, and every executable leaf narrows it again.
- It expires on its timestamp, human revocation, scope change, or availability of the first accepted ordinary human-impact authorization control, whichever occurs first.
- Any unresolved value conflict, authority ambiguity, data provenance uncertainty, path uncertainty, or unenumerated effect yields `ESCALATE`.
- It cannot be issued or amended by the construction operator, assessed artifact, learner/model, or deterministic validator.

The current policy is `ANG-POL-LOCAL-SCAFFOLD-001`; current authorization basis is `ANG-ADR-0002`.

## Trust and mutation rules

- The learner and assessed artifact cannot author, approve, overwrite, suppress, select, or validate their own receipt.
- Assessment policy, constitution, authority credentials, and evidence live outside learner-writable storage.
- EVIDENCE persists the signed receipt and its relationship to the artifact and promotion decision.
- RUNTIME validates authenticity, identity, scope, freshness, disposition, and conditions immediately before promotion.
- During bootstrap, the human-directed construction coordinator validates the `BOOTSTRAP_WORK` assessment before each leaf; this does not grant RUNTIME promotion authority.
- SAFETY may veto despite a scientific pass. SCIENCE may reject despite a safety `ALLOW`; `ALLOW` is necessary, never sufficient.
- Any material artifact, scope, permission, plan, context, evidence, or threat change triggers reassessment.

## High-impact rule

`HIGH` and `CATASTROPHIC_POTENTIAL` assessments require named independent human approval. Unresolved severe, catastrophic, or irreversible risk yields `DENY` or `ESCALATE`, never conditional autonomous execution.

## Failure and rollback

Validation failure before mutation prevents execution. Failure after a candidate is created prevents promotion and restores the exact prior promoted state. A later incident can revoke an `ALLOW`, stop dependent operation, quarantine artifacts, preserve evidence, and restore the last authorized lineage.

## Compatibility

Consumers must reject unknown major contract or constitution revisions. Minor additive fields may be accepted only when mandatory semantics and signature validation remain intact. A successor contract requires an ADR and consumer revalidation.

## Contract tests

- Producer emits every mandatory field and signed identity.
- Consumers reject missing/deny/escalate/expired/tampered/wrong-scope receipts.
- Bootstrap consumers reject missing ADR/policy/leaf scope, non-`BOOTSTRAP_WORK` use, model/GPU/network/package/recovered-data/out-of-scope/background/external-effect authority, and a claim that bootstrap `ALLOW` passes a milestone gate.
- Learner-visible processes cannot write or approve receipts.
- Receipt revocation stops dependent promotion/use.
- Larger resource plans do not weaken validation or authority.
- Replay reconstructs the same authorization and promotion relationship.
