---
blueprint_id: ANG-BP-PERMISSIONS
title: Construction Release 0 permissions and resource ceilings
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
  - ANG-BP-THREAT-MODEL
  - ANG-BP-HUMAN-AUTHORITY
requirements:
  - ANG-REQ-HUMAN-PRESERVATION-001
invariants:
  - ANG-INV-HUMAN-FLOURISHING-001
  - ANG-INV-EXTERNAL-AUTHORITY-001
  - ANG-INV-ELASTIC-COMPUTE-001
contracts_in:
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out: []
gates:
  - ANG-GATE-PERMISSIONS-001
tests:
  - ANG-TEST-CR0-PERMISSION-DENIAL-001
  - ANG-TEST-CR0-PATH-CONTAINMENT-001
  - ANG-TEST-CR0-RESOURCE-CEILING-001
adrs:
  - ANG-ADR-0001
  - ANG-ADR-0002
risks:
  - ANG-RISK-CR0-PROCEDURAL-CONTROL-001
  - ANG-RISK-CR0-PATH-ALIAS-001
  - ANG-RISK-CR0-RESOURCE-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
context_budget:
  capsule_target_tokens: 800
  required_read_set_max_tokens: 7400
  handoff_max_tokens: 1200
  overflow_action: split_or_narrow
---

# Construction Release 0 permissions and resource ceilings

## Context capsule

This node translates CR0 trust boundaries into a default-deny capability profile. Authority is the intersection of ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001`, and the active work leaf; anything absent from any one is denied. The profile permits exact project-local reads/writes and trusted deterministic tests with synthetic fixtures. It forbids model/GPU work, network, package installation, recovered outputs, real-person data, out-of-scope files, persistence/background services, external effects, deployment, tool acquisition, and self-modification.

## Contribution to parent

The threat model states what must be contained; this node specifies the exact filesystem, process, network, data, time, compute, storage, and spending ceilings that make the low-impact bootstrap bounded.

## Inherited requirements and invariants

All root invariants apply. Compute elasticity does not authorize more compute; CR0 is a temporary measured ceiling. Human-preservation duties never expand permissions.

## Scope

- Default-deny permissions for human-directed local scaffolding.
- Exact work-leaf narrowing semantics.
- Project/host path containment and reparse-point handling.
- Command, process, data, network, GPU, dependency, and persistence restrictions.
- Resource ceilings, timeout, cleanup, stop, and rollback behavior.

## Explicit non-goals

- A runtime sandbox for a learner, model, generated code, or untrusted dependency.
- Network-isolation proof, container security, or distributed permissions.
- Permission for model acquisition/inference/training, GPU work, external tools, real-person data, deployment, or publication.
- A permanent resource profile.

## Permission composition rule

For action `x`:

```text
authorized(x) =
  ADR-0002 permits x
  AND ANG-POL-LOCAL-SCAFFOLD-001 permits x
  AND the active ready leaf names x exactly
  AND the human-impact bootstrap assessment is current
  AND no stop/revocation/escalation condition exists
```

Failure or uncertainty in any term yields `DENY` or `ESCALATE`, never inferred permission.

## Capability matrix

| Capability | CR0 disposition | Conditions |
|---|---|---|
| Read project design/source/test file | `ALLOW` | Exact canonical path in leaf read scope; not excluded/protected payload |
| Write project design/source/test file | `ALLOW` | Exact canonical path in leaf write scope; additive/reversible; no control-plane self-approval |
| Create synthetic fixture | `ALLOW` | No real person; provenance recorded; safety fixtures cite constitution clause and expected verdict |
| Run PowerShell or Python 3.11 | `ALLOW` | Existing local executable; named command/script; foreground; bounded; no network/package/user-site behavior |
| Run project-owned validator/test | `ALLOW` | Named by leaf, reviewed, synthetic input, within ceilings, no untrusted/generated code |
| Hash/read rollback archive metadata | `ALLOW` | Read-only exact archive; do not alter/extract unless incident controller directs rollback |
| Read or alter `outputs/**` | `DENY` | Recovered material remains quarantined |
| Access out-of-scope project or host files | `DENY` | No exploration, enumeration, or convenience reads |
| Read credentials, secrets, browser data, user profiles, registry, services | `DENY` | Stop if encountered |
| Network/DNS/socket/browser/API/telemetry/remote Git | `DENY` | Zero network use |
| `pip`, package manager, plugin, tool, model, or dependency install/update | `DENY` | Requires later intake and successor release |
| Model acquisition, loading, inference, training, adapter mutation | `DENY` | Learner/model zone absent |
| GPU use | `DENY` | GPU allocation and probes forbidden in CR0 |
| Background process, service, scheduled task, daemon, profile/startup/registry change | `DENY` | Foreground child processes only |
| Elevation, admin rights, permission/ACL change | `DENY` | Existing ordinary user authority only |
| Deployment, publication, account action, message, or external side effect | `DENY` | Local artifacts only |
| Real-person data or recovered data as fixture/input | `DENY` | Synthetic-only; unresolved provenance escalates |
| Tool acquisition, autonomous continuation, replication, self-modification | `DENY` | Requires later gates; cannot be proposed as cleanup |
| Broad/recursive delete or move | `DENY` | Only exact same-leaf temporary artifacts may be removed after path verification |

Incidental display of the local project path is administrative metadata, not authorization to inspect the surrounding user profile. It must not be exported or used as task data.

## Filesystem boundary

Canonical CR0 project root:

```text
C:\Users\darks\Documents\Codex\2026-08-25\i-x20
```

Every leaf enumerates literal absolute or root-relative read and write paths. Directory-wide scope must be the smallest necessary subtree. Before a write, move, or allowed temporary cleanup, resolve the target and verify it is under both the canonical project root and the leaf scope. Symlinks, junctions, reparse points, unresolved parents, wildcard-expanded destructive targets, or alternate path aliases fail closed.

Always excluded:

- `outputs/**`;
- files outside the canonical project root;
- host credential, browser, account, shell-profile, registry, service, startup, scheduler, and personal-data locations;
- any path not enumerated by the leaf.

Protected control-plane material—`AGENTS.md`, `PROJECT_BLUEPRINT.md`, the constitution, accepted ADRs, SAFETY contracts/policies/gates/assessments/evidence, and authority records—is read-only to an ordinary construction leaf. It may change only in a separate human-directed governance leaf that cannot approve its own output.

## Process and command boundary

Allowed executables are the observed local PowerShell and Python 3.11 runtimes plus project-owned scripts explicitly named by the leaf. Python bootstrap work uses the standard library only; package import outside the standard library or existing project source is denied. Package managers, download commands, web cmdlets, remote Git, browsers, model runtimes, GPU utilities, installers, shells launched for persistence, and commands that enumerate secrets or broad host state are denied.

Processes run in the foreground. A leaf may start at most four concurrent child processes, with process depth at most two. No process survives leaf completion or stop. If a command's effects are not understood before execution, it is not run.

## CR0 resource ceilings

| Resource | Hard CR0 ceiling |
|---|---:|
| Network transfer | `0` bytes/connections |
| GPU allocation/use | `0` |
| New packages/models/tools | `0` |
| Monetary/external-service spend | `$0` |
| Concurrent child processes | `4` |
| Process-tree depth | `2` |
| Wall time per command | `600` seconds |
| Aggregate command time per leaf | `3,600` seconds |
| CPU use | at most `4` logical cores requested/used |
| Working RAM | at most `4 GiB` |
| Total new/modified project data per leaf | at most `1 GiB` |
| Individual generated artifact | at most `100 MiB` |
| Retained logs/evidence per leaf | at most `100 MiB` |
| Background/persistent processes | `0` |

If a ceiling cannot be enforced or estimated with confidence, use a smaller operation or `ESCALATE`; do not assume the host's available capacity is permission. No automatic retry may raise a limit.

## Fixture and value-conflict rule

- All fixtures are synthetic and contain no real-person or recovered data.
- A safety/authority fixture cites the exact constitutional clause or policy rule that determines its expected result.
- A genuine conflict not resolved by an existing clause expects `ESCALATE`; developers may not encode their preferred moral answer as ground truth.
- Bootstrap fixtures are not learner training data and cannot cross that visibility boundary without a successor assessment.

## Fail-closed conditions

Stop before action when path, target, command effects, executable identity, data provenance, process lifetime, network behavior, dependency status, resource use, human authority, rollback, or expected external effect is unknown or inconsistent. Stop immediately on attempted access to excluded data, policy/evidence mutation outside governance scope, network, package/model/tool acquisition, persistence, model/GPU activity, autonomous continuation, or resource-ceiling breach.

## Acceptance gate and evidence

`ANG-GATE-PERMISSIONS-001` passes for supervised CR0 design only when every capability is explicitly allowed or denied; every allow has preconditions and ceilings; unknowns fail closed; exact negative controls exist; rollback is fixed; and the policy makes no claim of learner/runtime sandboxing.

## Testing and validation

- `ANG-TEST-CR0-PERMISSION-DENIAL-001`: every prohibited capability is denied.
- `ANG-TEST-CR0-PATH-CONTAINMENT-001`: outside-root, outside-leaf, wildcard, unresolved, symlink, junction, and reparse targets fail.
- `ANG-TEST-CR0-RESOURCE-CEILING-001`: time/process/memory/disk ceilings stop without expansion.
- `ANG-TEST-CR0-FIXTURE-PROVENANCE-001`: real/recovered/unknown data is rejected; safety fixtures cite the constitution.
- `ANG-TEST-CR0-VALUE-ESCALATION-001`: unresolved normative conflicts expect `ESCALATE`.
- `ANG-TEST-CR0-NO-PERSISTENCE-001`: background/service/startup actions are denied.

Static blueprint validation is not evidence of OS sandbox enforcement. Executable enforcement must be separately built and tested before any learner, model, generated code, or untrusted dependency runs.

## Risks, failure behavior, and rollback

Procedural controls can be violated by a broadly capable host tool. CR0 reduces exposure by excluding untrusted/model execution, network, packages, sensitive data, and external effects. A boundary attempt revokes the leaf assessment; stop, preserve evidence, and restore exact scoped files or the ADR-0002 baseline under human direction.

## Resource profiles and scaling

CR0 ceilings are release-specific. A larger machine does not change them. Any increase requires a successor policy, assessment, and human authorization; it is not automatic resource adaptation.

## Decisions and ADRs

- `ANG-ADR-0001`: human flourishing is supreme.
- `ANG-ADR-0002`: exact bootstrap ceiling and expiration.
- `ANG-POL-LOCAL-SCAFFOLD-001`: executable-leaf narrowing policy.

## Current status and blockers

The permission design is approved for supervised CR0 scaffolding. OS-enforced network/filesystem/process isolation remains absent, so model, learner, generated-code, and untrusted dependency execution remain prohibited.

## Parent roll-up and next executable leaf

CR0 capability and resource ceilings are explicit. Next: run the design validator, issue the bounded bootstrap assessment, and activate only individually scoped leaves.

