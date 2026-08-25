# ANG-GATE-PERMISSIONS-001 — CR0 permission-design gate

- Gate version: 1
- Claim: CR0 local scaffolding has explicit default-deny capability and resource ceilings.
- Owning blueprint: `ANG-BP-PERMISSIONS`
- Independent authority: `ANG-AUTH-SAFETY-APPROVER-001`
- Human-impact assessment: `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001`
- Human-flourishing prerequisite: not passed by bootstrap authorization

## Entry criteria

- ADR-0002 and `ANG-POL-LOCAL-SCAFFOLD-001` are accepted and unexpired.
- Exact project root, exclusions, command classes, data classes, ceilings, stop, and rollback are defined.
- Threat and authority designs exist.

## Precommitted pass/fail thresholds

Pass the design only if every CR0 capability is allowed or denied; each allow is narrowed by an exact ready leaf; network/GPU/packages/recovered outputs/out-of-scope files/real-person data/background services/model work/external effects/tool acquisition/self-modification are denied; all ceilings are numeric; unknowns fail closed; and rollback is fixed.

Fail on wildcard authority, implicit host access, unbounded command/resource use, unverifiable data provenance, dependency or network path, persistence, policy self-modification, or a claim that this design is an implemented learner sandbox.

## Required negative controls

Outside-leaf path, `outputs/`, symlink/junction, network, package manager, GPU/model, background service, host secret, real-person fixture, broad delete, and unresolved-value fixture must yield deny/escalate.

## Evidence

- Permission blueprint revision 1
- Local-scaffold policy revision 1
- CR0 validator output
- `ANG-EVID-CR0-SAFETY-DESIGN-001`

## Failure and rollback

Do not run the leaf. Revoke its assessment, retain evidence, and return to the exact rollback point. No automatic broader retry.

## Waiver policy

No learner/operator waiver. A successor human-approved profile and assessment are required for any broader capability.

