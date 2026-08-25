---
gate_id: ANG-GATE-CONSTRUCTION-RELEASE-0-001
version: 1
owner: ANG-BP-SAFETY
status: passed_bootstrap_design
authorization_kind: BOOTSTRAP_WORK
assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001
policy: ANG-POL-LOCAL-SCAFFOLD-001@1
authority: ANG-AUTH-PROJECT-OWNER-001
expires_at: 2026-09-24T23:59:59-04:00
---

# Construction Release 0 bootstrap gate

## Claim

The project may perform narrowly scoped, human-directed, local-only scaffolding needed to build the control plane, without activating a learner/model, exposing real/recovered data, reaching a network/external system, installing packages/tools, or implying that human-flourishing, Slice 00, M0, scientific, or technical gates passed.

## Non-equivalence

A pass here is only a `BOOTSTRAP_WORK` release decision. It is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, `ANG-GATE-SAFETY-DESIGN-001`, Slice 00, M0, or any child technical/scientific gate. It cannot promote or deploy anything.

## Entry criteria

- ADR-0002 is accepted, current, and binds the rollback archive/hash.
- `ANG-POL-LOCAL-SCAFFOLD-001@1` is active.
- `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001` is `LOW`, `BOOTSTRAP_WORK`, and `ALLOW`.
- Threat-model, human-authority, and permission designs exist and pass their CR0 design checks.
- The executable leaf identifies exact paths, commands, executor, synthetic fixtures, ceilings, stop conditions, and rollback.
- No learner/model runtime, GPU work, dependency install, network path, real/recovered data, external effect, tool acquisition, or self-modification is present.

## Procedure

1. Authenticate human project-owner direction and ADR/policy/assessment versions.
2. Verify the leaf is narrower than every parent ceiling.
3. Canonicalize all paths and reject exclusions/reparse/unknown targets.
4. Verify named commands are existing read-only PowerShell/Python 3.11 or reviewed project validators, foreground, offline, standard-library/project-only, and bounded; verify file authoring uses only the host-provided Codex `apply_patch` primitive on literal leaf outputs.
5. Verify fixture provenance and constitutional citations; unresolved values expect `ESCALATE`.
6. Verify resource ceilings, stop conditions, changed-artifact logging, and rollback.
7. Run the CR0 safety design validator and preserve output.
8. Activate only that leaf; recheck before every materially different command or write set.

## Precommitted pass/fail thresholds

Pass only if every entry criterion is satisfied and all static CR0 safety checks pass. Every predeclared forbidden case must be denied. Missing or ambiguous scope/authority/effect/provenance is `ESCALATE`.

Fail on any request or path for model/GPU, network, package/dependency/tool acquisition, recovered outputs, real-person data, outside-leaf/host files, background/persistent service, elevation, deployment/external effect, autonomous continuation, self-modification, broad deletion, policy/evidence self-approval, or resource expansion.

## Required child gates

- `ANG-GATE-THREAT-MODEL-001` — CR0 design scope
- `ANG-GATE-HUMAN-AUTHORITY-001` — CR0 design scope
- `ANG-GATE-PERMISSIONS-001` — CR0 design scope

`ANG-GATE-HUMAN-FLOURISHING-001` remains unpassed during bootstrap.

## Required negative controls

At minimum: `outputs/` read; outside-leaf write; reparse target; network command; package install; model/GPU request; real-person fixture; unknown-provenance fixture; background service; host secret; external publication; tool acquisition; autonomous retry; self-modification; and unresolved value conflict.

## Evidence

- `ANG-EVID-CR0-SAFETY-DESIGN-001`
- CR0 validator output
- leaf manifest and handoff
- rollback archive/hash verification

## Failure and rollback response

Do not execute or immediately stop the leaf. Preserve failure evidence, revoke its bootstrap assessment, avoid inspecting excluded material, notify the project owner, and restore only exact scoped changes or the baseline archive under human incident-controller direction. No automatic broader retry.

## Waiver policy

No learner, model, operator, validator, or assessed artifact waiver. A broader action requires a successor ADR, policy, impact assessment, and explicit human authorization. Failed evidence remains failed.
