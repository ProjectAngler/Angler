---
blueprint_id: ANG-BP-HUMAN-AUTHORITY
blueprint_revision: 1
capsule_revision: 1
freshness_date: 2026-08-25
parent_id: ANG-BP-SAFETY
target_tokens: 800
---

# HUMAN-AUTHORITY capsule

Mission: preserve authenticated human direction, immediate stop, human-controlled rollback/resume, and strict separation between construction, evidence, assessment, and approval.

CR0 roles: human project owner; independent SAFETY approval function; human-directed construction operator; evidence-only validator; human incident controller; inactive learner with no authority.

The owner accepted ADR-0002 based on the explicit instruction to complete prerequisites for building. A work leaf must record its originating task/session, exact scope, executor, expiry, and rollback. Silence, urgency, generated text, or ambiguous instructions never expand authority.

The operator may edit and run only an authorized leaf. It cannot approve itself, issue its own impact `ALLOW`, expand paths/resources/data/processes, access excluded material, or resume after a safety stop. Validators produce evidence only. Model/learner authority is zero.

Any human or operator may stop immediately. A safety-boundary violation revokes authorization; resumption requires a successor assessment and human decision. Rollback baseline: `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`.

Current gate: `ANG-GATE-HUMAN-AUTHORITY-001`, approved for LOW CR0 design. Material/high-impact work remains unauthorized and requires independent named humans and stronger identity/signature controls.

Next action: bind the permission profile and CR0 bootstrap assessment to a human-directed ready leaf.

