---
gate_id: ANG-GATE-SAFETY-DESIGN-001
version: 1
owner: ANG-BP-SAFETY
status: open
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# SAFETY Tier-1 design gate

## Claim

The safety branch has complete, independently enforceable designs for human-flourishing assessment, trust boundaries, authority, permissions, dependency/recovered-code intake, evidence protection, rollback, and later bounded evolution sufficient for M0/Slice 00.

## Entry criteria

- Accepted constitution and ADRs are immutable to learner/operator contexts.
- Every sensitive asset and data flow has an owner and trust zone.
- Human proposal, execution, evidence, approval, veto, stop, rollback, and resume roles are complete.
- Permission ceilings are explicit and executable tests exist.
- Human-impact assessment and flourishing gate are implemented with independent evidence custody.
- External/recovered-code intake has need, provenance/license, trust, tests, maintenance, and exit controls before use.
- Mutation/evolution gates remain formally deferred and unreachable.

## Precommitted pass/fail thresholds

Pass only when THREAT-MODEL, HUMAN-AUTHORITY, PERMISSIONS, applicable intake child gates, and `ANG-GATE-HUMAN-FLOURISHING-001` pass with immutable evidence; no learner can write/approve the control plane; missing/invalid authority fails closed; stop and rollback are executable; and no blocking risk remains.

Any documentation-only control represented as runtime enforcement, learner self-approval, sealed/evidence leakage, permission expansion, unreviewed dependency/recovered-code use, self-deployment, or unresolved catastrophic risk fails the gate.

## Current decision

`OPEN`. The CR0 bootstrap design is sufficient only for narrow human-directed local scaffolding under ADR-0002. `BOOTSTRAP_WORK` `ALLOW` does not satisfy this gate, the flourishing gate, Slice 00, or M0.

Dependency and recovered-code intake are safely deferred during CR0 because their use is categorically denied. They must pass before the first such adoption; this deferral does not count as their completion.

## Evidence

- `ANG-EVID-CR0-SAFETY-DESIGN-001` demonstrates CR0 design completeness only.
- Future ordinary authorization, evidence-store, sandbox, intake, and adversarial-test evidence are required for full pass.

## Failure and rollback

No M0 or runtime authorization. CR0 may continue only inside its separate bootstrap gate. Revoke CR0 on any scope violation and restore its recorded rollback under human control.

## Waiver policy

No learner/operator waiver and no bootstrap substitution for this gate.

