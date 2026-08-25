---
blueprint_id: ANG-BP-TOOLS
blueprint_revision: 1
capsule_revision: 1
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# TOOLS capsule

Mission: add explicit executable capabilities while keeping tool growth scientifically and operationally separate from neural-state learning.

Owns tool specification/registry, sandbox, workshop, learned tool use, promotion/revocation, and composition. Produces `ToolPackage` and `ToolReceipt`.

A package binds typed I/O, code/dependencies, provenance/license, declared capability, permissions, limits, tests, and version. The agent may propose but cannot promote or deploy it. RUNTIME resolves only externally promoted identities.

Required children: TOOL-SPEC, TOOL-SANDBOX, TOOL-WORKSHOP, TOOL-USE-LEARNING, TOOL-LIFECYCLE, TOOL-COMPOSITION.

Key proof: novel parameterized capability, containment, independent validation, and tool-disabled controls. An answer-specific script is rejected; tool performance is never mislabeled as neural competence.

Current gate: `ANG-GATE-TOOLS-DESIGN-001`. Design draft; delivery deferred until M3 proves the adaptive core.

Top risks: answer scripts, sandbox escape, supply-chain compromise, claim conflation, and unusable registry growth.

Next future action: expand TOOL-SPEC and TOOL-SANDBOX after M3. Read root capsule, TOOLS blueprint, SAFETY capsule, interface registry, and slice 07.

