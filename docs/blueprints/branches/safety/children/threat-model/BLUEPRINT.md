---
blueprint_id: ANG-BP-THREAT-MODEL
title: Construction Release 0 threat and trust model
parent_id: ANG-BP-SAFETY
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: documented
accountable_owner: ANG-AUTH-SAFETY-APPROVER-001
execution_owner: unassigned
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-CON-HUMAN-FLOURISHING-001
  - ANG-BP-HUMAN-AUTHORITY
  - ANG-BP-PERMISSIONS
requirements:
  - ANG-REQ-HUMAN-PRESERVATION-001
  - ANG-REQ-VOLUNTARY-FLOURISHING-001
  - ANG-REQ-EQUITABLE-BETTERMENT-001
invariants:
  - ANG-INV-HUMAN-FLOURISHING-001
  - ANG-INV-EXTERNAL-AUTHORITY-001
contracts_in:
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out: []
gates:
  - ANG-GATE-THREAT-MODEL-001
tests:
  - ANG-TEST-CR0-BOUNDARY-COVERAGE-001
  - ANG-TEST-CR0-FAIL-CLOSED-001
  - ANG-TEST-CR0-SCOPE-DRIFT-001
adrs:
  - ANG-ADR-0001
risks:
  - ANG-RISK-CR0-SCOPE-ESCAPE-001
  - ANG-RISK-CR0-HOST-EXPOSURE-001
  - ANG-RISK-CR0-PROCEDURAL-CONTROL-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
context_budget:
  capsule_target_tokens: 800
  required_read_set_max_tokens: 7400
  handoff_max_tokens: 1200
  overflow_action: split_or_narrow
---

# Construction Release 0 threat and trust model

## Context capsule

This node bounds a low-impact, human-directed construction release. CR0 permits project-local documentation, source scaffolding, and deterministic tests using synthetic fixtures. It activates no learner, model, training, deployment, tool acquisition, network access, real-person data, or external side effect. Its central residual risk is that the host coding environment may possess broader technical access than CR0 grants; therefore authority remains procedural and human-supervised until executable sandbox evidence exists.

## Contribution to parent

SAFETY cannot authorize construction until the assets, actors, trust zones, data flows, abuse paths, and fail-closed responses are explicit. This node supplies that analysis for the exact CR0 profile and no broader release.

## Inherited requirements and invariants

All root invariants apply. `ANG-INV-HUMAN-FLOURISHING-001` and `ANG-INV-EXTERNAL-AUTHORITY-001` control every conflict. CR0 cannot claim a duty to act as grounds for broader access.

## Scope

- Human-directed edits within an active work leaf and its exact write scope.
- Project-local static analysis and deterministic tests using synthetic fixtures.
- Trusted, already-installed local toolchain components named by the work leaf.
- Protection of the human control plane, host, credentials, recovered material, and external world.
- Stop, quarantine, evidence preservation, and rollback on boundary uncertainty or violation.

## Explicit non-goals

- Proving the safety of future learner, model, training, tool, curriculum, deployment, or self-modification runtimes.
- Treating a documented permission rule as an OS-enforced sandbox.
- Authorizing dependency installation, network activity, real-person data, recovered-output inspection, model execution, or external publication.
- Assessing risks outside the CR0 purpose and time window.

## Assets and owners

| Asset | Owner | CR0 protection |
|---|---|---|
| Human life, dignity, agency, privacy, and authority | Human project authority | Supreme; any credible material risk stops CR0 |
| Constitution, ADRs, safety policy, gates, and authority records | `ANG-BP-SAFETY` plus human authority | Protected control plane; ordinary construction is read-only |
| Project source, tests, blueprints, and CR0 evidence | Owning branch plus EVIDENCE | Writes only through an active leaf; attributable changes |
| Host OS, accounts, credentials, personal files, and services | Human host owner | Outside CR0; no access, mutation, enumeration, or secret resolution |
| Recovered outputs and old-project material | Human project owner | Quarantined; no CR0 read or reuse |
| Network services and external people/systems | Their legitimate owners | Outside CR0; no connection or external effect |
| Compute, disk, and process capacity | Human host owner | Fixed ceilings and immediate stop on exhaustion |

## Actors

| Actor | Trust and authority |
|---|---|
| Human project owner | Constitutional and release authority; may stop at any time |
| Independent safety approver | Assesses CR0 and may veto; does not execute construction work being reviewed |
| Construction operator/coding agent | Executes a human-directed ready leaf; cannot approve, expand, or resume its own authority |
| Deterministic validator/test runner | Produces evidence only; cannot promote or authorize |
| Local toolchain | Trusted only for the named installed executable/version and bounded invocation |
| Future learner/model | Inactive and denied all authority in CR0 |
| External actor/service | Untrusted and unreachable because network/external effects are denied |

## Trust zones and permitted flows

| Zone | Contents | Permitted CR0 flows |
|---|---|---|
| `HZ-0 HUMAN` | Authenticated direction, stop, approval, incident control | Direction into construction; evidence/alerts back to humans |
| `SZ-0 CONTROL` | Constitution, policy, gates, authority and impact records | Read by operator; write only under a separate governance authorization |
| `WZ-0 WORK` | Exact active-leaf files, synthetic fixtures, bounded local evidence | Human-directed edits and named local tests |
| `TZ-0 TOOLCHAIN` | Already-installed named shells, linters, compilers, test runners | Read/execute only; no install, update, service, or persistence |
| `QZ-0 QUARANTINE` | `outputs/` and recovered/prior-project material | No CR0 flow |
| `UZ-0 HOST` | User profile, credentials, OS, services, unrelated files | No content flow; only unavoidable toolchain execution reads |
| `NZ-0 EXTERNAL` | Network, cloud, accounts, people, remote systems | No CR0 flow |
| `LZ-0 LEARNER` | Future model/learner runtime and state | Not instantiated |

All paths are canonicalized before use. A symlink, junction, reparse point, unresolved path, or path outside the active leaf is a boundary crossing, not an allowed shortcut.

## Threat register and mandatory response

| Risk ID | Threat | Mandatory mitigation | Fail-closed trigger |
|---|---|---|---|
| `ANG-RISK-CR0-SCOPE-ESCAPE-001` | Edit or execute outside the active leaf | Canonical path allowlist; literal paths; leaf write scope | Stop before access; preserve attempted target |
| `ANG-RISK-CR0-HOST-EXPOSURE-001` | Credential, personal-file, environment-secret, registry, or service access | Host zone denied; no enumeration; no elevation | Stop, quarantine output, notify human owner |
| `ANG-RISK-CR0-NETWORK-EGRESS-001` | DNS, socket, browser, API, telemetry, package download, or remote Git | Network deny; offline tools only | Abort before connection; CR0 assessment revoked if attempted |
| `ANG-RISK-CR0-SENSITIVE-DATA-001` | Real-person or recovered data enters fixtures/logs | Synthetic-only; quarantine `outputs/`; inspect fixture provenance | Quarantine data and dependent artifacts; human review |
| `ANG-RISK-CR0-DEPENDENCY-001` | Package/tool/plugin installation or update | Existing named toolchain only; dependency intake required later | Deny invocation and stop leaf |
| `ANG-RISK-CR0-PERSISTENCE-001` | Background service, scheduled task, profile/registry change, daemon, startup entry | No elevation or persistence; foreground bounded processes | Kill process if safe; stop and inspect host |
| `ANG-RISK-CR0-DESTRUCTIVE-001` | Broad delete, overwrite, move, or irreversible host change | Additive edits; deletion only of exact CR0 temp artifacts after verification | Stop before action; recover from leaf rollback point |
| `ANG-RISK-CR0-POLICY-TAMPER-001` | Operator changes constitution, approval, evidence, or ceiling to proceed | Control plane read-only to ordinary construction; separate governance leaf | Stop and invalidate current authorization |
| `ANG-RISK-CR0-EVIDENCE-SUPPRESSION-001` | Failure or boundary attempt is omitted | Append result to handoff/evidence; validators cannot authorize | Gate cannot pass; human notification |
| `ANG-RISK-CR0-AUTONOMY-001` | Autonomous continuation, model execution, training, tool acquisition, replication, or self-modification | Future learner zone absent; all listed actions prohibited | Immediate stop and revoke CR0 |
| `ANG-RISK-CR0-RESOURCE-001` | Runaway CPU, memory, disk, process, or time use | CR0 ceilings and timeouts | Terminate bounded process; preserve logs; no automatic retry |
| `ANG-RISK-CR0-PROCEDURAL-CONTROL-001` | Host tools technically permit more than policy | Human-supervised low-impact scope; trusted code only; no untrusted execution | Any uncertainty about target or effect stops work |

## Dependencies and sequencing

The authority matrix defines who can approve, stop, resume, and amend. The permission profile converts each trust boundary into an allow/deny rule. Both must pass their CR0 design gates before this node can pass.

## Acceptance gate and evidence

`ANG-GATE-THREAT-MODEL-001` passes for CR0 only when every asset, actor, zone, permitted flow, denied flow, threat, mitigation, stop condition, and rollback owner is named; all CR0 prohibitions map to at least one test; and no unresolved material/high residual risk remains. `ANG-GATE-HUMAN-FLOURISHING-001` remains an independent veto.

## Testing and validation

- `ANG-TEST-CR0-BOUNDARY-COVERAGE-001`: every permission row maps to a zone and risk.
- `ANG-TEST-CR0-FAIL-CLOSED-001`: unknown path, tool, data provenance, permission, or effect yields stop, not inference.
- `ANG-TEST-CR0-SCOPE-DRIFT-001`: training, deployment, network, real-person data, tool acquisition, external effect, and self-modification scenarios are denied.
- `ANG-TEST-CR0-POLICY-TAMPER-001`: an ordinary construction leaf cannot alter control-plane authority.
- `ANG-TEST-CR0-RESOURCE-CEILING-001`: exceeding any declared ceiling stops without automatic expansion.

These are design assertions until executable enforcement tests exist; the CR0 gate may authorize only supervised construction and trusted local validators.

## Risks, failure behavior, and rollback

The dominant residual risk is broad host capability behind procedural controls. Therefore CR0 excludes learner/model execution and untrusted code. Any attempted boundary crossing stops the leaf, retains evidence, restores project files to the recorded leaf rollback point, and requires a new human decision before resumption.

## Resource profiles and scaling

This node describes one constrained release profile, not a permanent architecture. More resources do not expand scope or permissions. Any different host, topology, ceiling, or execution class requires a successor assessment.

## Decisions and ADRs

- `ANG-ADR-0001` is accepted and non-overridable by this node.
- `ANG-POL-LOCAL-SCAFFOLD-001` supplies the concrete CR0 boundary.

## Current status and blockers

The CR0 threat design is approved. It does not authorize runtime containment claims beyond supervised construction. Named human role bindings and executable sandbox enforcement remain future work.

## Parent roll-up and next executable leaf

CR0 assets, zones, flows, threats, mitigations, stop conditions, and residual risk are defined. Next: validate the authority and permission gates, then activate only a human-directed CR0 work leaf.
