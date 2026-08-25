---
blueprint_id: ANG-BP-PERMISSIONS
blueprint_revision: 1
capsule_revision: 1
freshness_date: 2026-08-25
parent_id: ANG-BP-SAFETY
target_tokens: 800
---

# PERMISSIONS capsule

Mission: enforce a default-deny CR0 profile whose authority is the intersection of ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001`, and an exact active work leaf.

Allowed: human-directed project-local documentation/source scaffolding, synthetic fixtures, and named foreground PowerShell/Python 3.11 validators/tests within exact paths and ceilings.

Denied: model/GPU activity; network; packages/models/tools/dependencies; `outputs/**`; real-person/recovered/unknown data; out-of-scope or host files; secrets; elevation; background services/persistence; external side effects; deployment; tool acquisition; autonomous continuation; replication; self-modification; broad deletion.

Ceilings: zero network/GPU/packages/spend/background work; 4 child processes; depth 2; 600 seconds/command and 3,600 seconds/leaf; 4 logical CPU cores; 4 GiB RAM; 1 GiB changed data; 100 MiB per artifact and retained logs.

Paths are canonicalized and must remain under both the project root and leaf scope. Unknowns, symlink/junction/reparse paths, unclear command effects, unresolved data provenance, or unenforceable ceilings stop or escalate.

Safety fixtures cite their constitutional basis; unresolved values expect `ESCALATE`. Fixtures are not learner training data.

Current gate: `ANG-GATE-PERMISSIONS-001`, approved for supervised scaffolding design only. No OS sandbox claim exists; learner/model/generated/untrusted execution remains prohibited.

Next action: bind a ready leaf to the profile, assessment, exact paths, commands, ceilings, and rollback before executing it.

