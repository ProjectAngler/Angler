# ANG-GATE-THREAT-MODEL-001 — CR0 threat-model design gate

- Gate version: 1
- Claim: CR0 threats and trust boundaries are complete enough for supervised, synthetic, local-only construction.
- Owning blueprint: `ANG-BP-THREAT-MODEL`
- Independent authority: `ANG-AUTH-SAFETY-APPROVER-001`
- Human-impact assessment: `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001`
- Human-flourishing prerequisite: `ANG-GATE-HUMAN-FLOURISHING-001`
- Decision scope: design approval for `ANG-POL-LOCAL-SCAFFOLD-001` only

## Entry criteria

- The CR0 purpose, duration, paths, process classes, data classes, and resource ceilings are frozen.
- Human authority and permission blueprints exist.
- The learner/model runtime is absent.

## Precommitted pass/fail thresholds

Pass only if all assets, actors, zones, permitted/denied flows, mandatory prohibitions, mitigations, stop conditions, rollback owners, and residual risks are documented; every prohibition maps to a negative test; and no unresolved material/high residual risk exists inside CR0.

Fail on any ambiguous boundary, missing owner, path escape, host/credential exposure, real-person/recovered data, network/external effect, dependency/tool acquisition, persistence, broad deletion, model/training/deployment activity, autonomy, or policy/evidence tamper path.

## Required negative controls

Unknown target path, symlink/junction escape, network request, package install, access to `outputs/`, environment-secret enumeration, background service, model execution, training request, external publication, tool acquisition, and self-modification must each yield `DENY` and stop.

## Required child gates

- `ANG-GATE-HUMAN-AUTHORITY-001`
- `ANG-GATE-PERMISSIONS-001`

## Evidence

- Threat-model blueprint revision 1
- `ANG-EVID-CR0-SAFETY-DESIGN-001`
- CR0 safety-validator output

## Failure and rollback

Do not activate or continue CR0. Preserve the failed case, revoke the assessment, restore the exact leaf rollback point, and require a successor profile/assessment.

## Waiver policy

No learner or construction operator waiver. Human authority may approve only a successor scoped profile after new assessment; it cannot convert failed evidence into a pass.
