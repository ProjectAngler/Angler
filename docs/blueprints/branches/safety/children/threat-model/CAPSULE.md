---
blueprint_id: ANG-BP-THREAT-MODEL
blueprint_revision: 1
capsule_revision: 1
freshness_date: 2026-08-25
parent_id: ANG-BP-SAFETY
target_tokens: 800
---

# THREAT-MODEL capsule

Mission: bound Construction Release 0 so project-local construction cannot silently become learner operation, host access, or an external action.

CR0 permits human-directed documentation, source scaffolding, and named deterministic local tests with synthetic fixtures. It prohibits network access, external side effects, real-person/recovered data, dependency or tool acquisition, model execution, training, deployment, autonomous continuation, replication, and self-modification.

Trust zones: HUMAN authority; protected SAFETY control plane; active-leaf WORK area; named installed TOOLCHAIN; quarantined recovered OUTPUTS; protected HOST; denied EXTERNAL network/world; inactive LEARNER.

Unknown paths, reparse points, tools, data provenance, permissions, effects, or authority fail closed. Any boundary attempt stops the leaf, preserves evidence, restores the leaf rollback point, and requires human review.

Current gate: `ANG-GATE-THREAT-MODEL-001`, approved for CR0 design only. It does not claim OS sandbox enforcement. The dominant residual risk is broad host capability behind procedural controls, controlled by human supervision, synthetic-only data, trusted validators, and the absence of model/untrusted-code execution.

Next action: combine the passed authority and permission design gates with `ANG-GATE-CONSTRUCTION-RELEASE-0-001` before activating a work leaf.

Mandatory reads: root capsule; safety capsule; this blueprint; human-flourishing constitution; CR0 policy; authority and permission capsules.

