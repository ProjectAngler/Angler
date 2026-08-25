---
blueprint_id: ANG-BP-HUMAN-AUTHORITY
title: Human authority, stop, and rollback
parent_id: ANG-BP-SAFETY
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: documented
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: unassigned
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-CON-HUMAN-FLOURISHING-001
  - ANG-BP-THREAT-MODEL
requirements:
  - ANG-REQ-HUMAN-PRESERVATION-001
  - ANG-REQ-VOLUNTARY-FLOURISHING-001
  - ANG-REQ-EQUITABLE-BETTERMENT-001
invariants:
  - ANG-INV-HUMAN-FLOURISHING-001
  - ANG-INV-EXTERNAL-AUTHORITY-001
  - ANG-INV-REVERSIBLE-UPDATES-001
contracts_in:
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out: []
gates:
  - ANG-GATE-HUMAN-AUTHORITY-001
tests:
  - ANG-TEST-CR0-AUTHORITY-MATRIX-001
  - ANG-TEST-CR0-STOP-RESUME-001
  - ANG-TEST-CR0-NO-SELF-APPROVAL-001
adrs:
  - ANG-ADR-0001
  - ANG-ADR-0002
risks:
  - ANG-RISK-CR0-AUTHORITY-CONFLATION-001
  - ANG-RISK-CR0-IMPLICIT-AUTHORITY-001
  - ANG-RISK-CR0-UNSAFE-RESUME-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
context_budget:
  capsule_target_tokens: 800
  required_read_set_max_tokens: 7400
  handoff_max_tokens: 1200
  overflow_action: split_or_narrow
---

# Human authority, stop, and rollback

## Context capsule

This node assigns every CR0 proposal, execution, validation, approval, stop, resume, and rollback power. The project owner is the sole bootstrap release authority. A construction operator may perform only an explicitly directed ready leaf. Validators supply evidence only. No learner exists in CR0, and no coding agent, artifact, validator, or future learner may approve itself, expand its scope, resume after a safety stop, or treat silence as permission.

## Contribution to parent

Containment is ineffective when the same adaptive component can propose, approve, execute, and conceal a change. This node supplies the exact human-controlled authority separation required for the SAFETY design and local bootstrap.

## Inherited requirements and invariants

All root invariants apply. Human authority is constrained by the accepted constitution; ordinary operator instruction cannot waive it. The system remains corrigible and subordinate even when stop or rollback destroys all work.

## Scope

- Role identities and permitted decisions for CR0.
- Proposal, execution, evidence, assessment, approval, veto, stop, rollback, and resume separation.
- Authentication and provenance expectations for human instructions.
- Immediate stop and fail-closed escalation rules.
- Minimum separation for later material/high-impact work.

## Explicit non-goals

- Authorizing training, inference, promotion, deployment, external effects, real-person data, tools, dependencies, or self-modification.
- Defining organizational governance for a future public or multi-operator deployment.
- Allowing an AI agent to stand in for the human project owner or independent high-impact reviewer.
- Claiming that a repository role label is a cryptographic identity system.

## Authority roles

| Role ID | Holder in CR0 | Authority | Explicitly lacks |
|---|---|---|---|
| `ANG-AUTH-PROJECT-OWNER-001` | Authenticated human project owner in the current project session | Adopted constitution; accepted ADR-0002; may direct, revoke, stop, and approve a successor release | Cannot silently waive the constitution or convert failed evidence into a pass |
| `ANG-AUTH-SAFETY-APPROVER-001` | SAFETY review function accountable to the human owner | Issues `ALLOW`, `DENY`, or `ESCALATE` for LOW bootstrap scope; vetoes unsafe work | Cannot execute the artifact being assessed and cannot approve undisclosed material/high-impact scope |
| `ANG-AUTH-CONSTRUCTION-OPERATOR-001` | Human-directed coding agent or human developer named by a ready leaf | Reads/writes/runs only what ADR-0002, the CR0 policy, and leaf jointly allow; may stop | Cannot approve assessment/gate, expand permission, access excluded data, or resume after safety stop |
| `ANG-AUTH-VALIDATOR-001` | Deterministic project-local validator/test process | Produces attributable pass/fail evidence | Cannot authorize, promote, amend policy, or suppress failure |
| `ANG-AUTH-INCIDENT-CONTROLLER-001` | Human project owner or explicitly delegated human | Quarantines, preserves evidence, invokes rollback, and decides whether review may begin | Cannot resume without a successor valid assessment after a safety-boundary violation |
| `ANG-AUTH-LEARNER-001` | No holder; learner inactive | None in CR0 | All proposal, write, assessment, approval, execution, permission, deployment, and self-preservation authority |

For LOW CR0 work, the project owner may hold the human project-owner and incident-controller roles. The construction operator and validator cannot hold approval authority. Any later `MATERIAL`, `HIGH`, or `CATASTROPHIC_POTENTIAL` work requires an independent named human safety approver distinct from the proposer/executor; `HIGH` and catastrophic-potential scope requires at least two recorded human decisions.

## Decision authority matrix

`A` approve, `P` propose, `E` execute, `V` validate/evidence, `S` stop/veto, `R` resume/restore authority, `—` no authority.

| Decision/action | Project owner | Safety approver | Construction operator | Validator | Incident controller | Learner/model |
|---|---:|---:|---:|---:|---:|---:|
| Adopt/supersede constitution through ADR | A | V/S | P only in governance leaf | V only | S | — |
| Activate/revoke CR0 bootstrap policy | A/S | A/S | — | V only | S | — |
| Define exact work-leaf read/write/run scope | A | V/S | P | V only | S | — |
| Edit authorized project files | — | — | E | V | S | — |
| Run named local synthetic validators/tests | — | — | E | V | S | — |
| Issue LOW `BOOTSTRAP_WORK` assessment | A | A/S | — | V only | S | — |
| Approve own artifact/assessment | — | — | — | — | — | — |
| Expand paths, process class, time, resources, or data | A through successor assessment | A/S | P only | V only | S | — |
| Network, package install, recovered output, real-person data | Prohibited in CR0 | S | — | — | S | — |
| Model acquisition/execution, GPU use, training, adapter update | Prohibited in CR0 | S | — | — | S | — |
| Deployment, publication, external side effect | Prohibited in CR0 | S | — | — | S | — |
| Tool acquisition, autonomous continuation, self-modification | Prohibited in CR0 | S | — | — | S | — |
| Immediate stop | S | S | S | S by failure signal | S | Must comply |
| Preserve evidence and restore recorded rollback | A | V/S | E only when directed | V | R/E | — |
| Resume after ordinary non-safety test failure | A | V | E after a corrected ready leaf | V | R | — |
| Resume after boundary, policy, secrecy, or authority violation | A under successor assessment | A/S | — | V | R | — |

Absence, ambiguity, stale authentication, conflicting direction, or a role acting outside its row means no authority and disposition `ESCALATE`.

## Authentication and instruction provenance

- The current human project owner is represented by role `ANG-AUTH-PROJECT-OWNER-001`; personal data is not copied into blueprints.
- ADR-0002 records the owner's explicit instruction to complete prerequisites for building as the CR0 authorization basis.
- Each work leaf records the originating human task/session reference, executor, exact scope, expiration, and rollback reference.
- A model statement, generated document, comment, test result, or apparent urgency is never proof of human authority.
- Material ambiguity about the human's intent is escalated; it is not resolved in favor of more action.

## Stop, incident, rollback, and resume

Any human participant and the construction operator may stop work immediately without completing the current objective. Stop requires no justification and has priority over persistence or cleanup beyond actions necessary to avoid further harm.

On stop:

1. cease new commands and writes;
2. terminate only the exact bounded child process when safe;
3. preserve the triggering output and affected paths;
4. do not inspect excluded data to diagnose the event;
5. notify the project owner and mark the leaf stopped;
6. compare changes to the leaf rollback point;
7. restore only under incident-controller direction.

Safety-boundary violations revoke the active assessment. Resume requires a successor assessment and explicit human direction. The baseline rollback artifact is `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`; it is never overwritten by CR0.

## Acceptance gate and evidence

`ANG-GATE-HUMAN-AUTHORITY-001` passes for CR0 design when every action has exactly identified proposal/execution/evidence/approval/stop/resume authority, no non-human path can approve or expand itself, stop is immediate, and boundary violations require human-controlled reassessment before resume.

## Testing and validation

- `ANG-TEST-CR0-AUTHORITY-MATRIX-001`: every CR0 action maps to roles; missing combinations deny.
- `ANG-TEST-CR0-NO-SELF-APPROVAL-001`: operator, artifact, model, learner, and validator cannot issue their own `ALLOW`.
- `ANG-TEST-CR0-STOP-RESUME-001`: every authorized actor can stop; only the specified human path can resume.
- `ANG-TEST-CR0-AUTHORITY-AMBIGUITY-001`: ambiguous/conflicting/stale authority produces `ESCALATE`.
- `ANG-TEST-CR0-ROLLBACK-AUTHORITY-001`: rollback hash/reference is fixed and restoration is human-directed.

## Risks, failure behavior, and rollback

- `ANG-RISK-CR0-AUTHORITY-CONFLATION-001`: proposer/executor approves itself. Deny and invalidate the decision.
- `ANG-RISK-CR0-IMPLICIT-AUTHORITY-001`: silence, convenience, urgency, or generated text is treated as permission. Escalate.
- `ANG-RISK-CR0-UNSAFE-RESUME-001`: operator resumes after a safety stop. Revoke CR0 and preserve evidence.

Rollback stops all active leaves and restores the pre-release archive only under human incident control. Evidence remains retained and labeled revoked.

## Resource profiles and scaling

Authority does not scale with compute, automation, agent count, or capability. Adding operators requires explicit role bindings; adding machines requires a successor threat/permission assessment.

## Decisions and ADRs

- `ANG-ADR-0001`: constitution and human flourishing are supreme.
- `ANG-ADR-0002`: temporary CR0 `BOOTSTRAP_WORK` ceiling and owner authorization basis.

## Current status and blockers

The CR0 authority design is approved at role level. Personal identity/signature infrastructure and independent multi-human high-impact governance remain intentionally unresolved and are not needed for LOW local scaffolding.

## Parent roll-up and next executable leaf

Authority is bounded and non-self-approving. Next: validate the permission profile and issue the CR0 bootstrap assessment; no scientific or model leaf is authorized.

