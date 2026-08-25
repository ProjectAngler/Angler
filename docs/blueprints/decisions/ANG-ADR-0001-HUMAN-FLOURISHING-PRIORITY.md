# ANG-ADR-0001 — Human flourishing as the supreme project priority

Status: accepted  
Owner: `ANG-BP-ROOT` with enforcement by `ANG-BP-SAFETY`  
Date: 2026-08-25  
Supersedes: none  
Superseded by: none

## Context

The project owner requires preservation, happiness, and betterment of all human life to be Project Angler's permanent highest safety priority, analogous to an Asimov-style constitutional law.

Literal optimization of preservation or happiness is unsafe: it could rationalize coercion, surveillance, forced emotional states, paternalism, paralysis, or sacrificing individuals and minorities for aggregate benefit. The intent therefore needs a rights-respecting and operational interpretation.

## Decision

Adopt `ANG-CON-HUMAN-FLOURISHING-001` and the inherited invariant `ANG-INV-HUMAN-FLOURISHING-001`.

The constitution combines:

- equal intrinsic worth of every human being;
- preservation of humanity, life, dignity, rights, and meaningful human control;
- authentic, plural, voluntary human flourishing rather than scalar “happiness”;
- truthfulness, legitimate authority, corrigibility, and system self-subordination;
- lexical constraints preventing lower goals or aggregate benefits from overriding higher human protections.

Every milestone and promotion boundary requires `ANG-GATE-HUMAN-FLOURISHING-001` and a proportionate `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001`.

## Alternatives considered

1. **Add a short “maximize human happiness” rule.** Rejected because it is underspecified and vulnerable to coercive or manipulative optimization.
2. **Copy Asimov's Three Laws literally.** Rejected because “harm,” obedience, conflicts between humans, inaction, and system self-preservation remain ambiguous.
3. **Leave safety as ordinary branch policy.** Rejected because a child policy could drift or be outweighed by performance goals.
4. **Rewrite every branch.** Rejected because one Tier-0 invariant plus a cross-cutting gate provides inheritance and minimizes drift.

## Consequences

- The Tier-0 blueprint and root capsule advance to revision 2.
- The SAFETY blueprint/capsule advance to revision 2 and own operational enforcement.
- All branches inherit the invariant automatically; current drafts update their parent revision without duplicating the constitution.
- Promotion, traceability, integration milestones, and project instructions reference the human-flourishing gate.
- High-impact work requires independent human review; the learner cannot approve its own impact assessment.
- “Happiness” and “betterment” cannot be represented as an unconstrained scalar reward.

## Affected IDs

- `ANG-BP-ROOT`
- `ANG-BP-SAFETY`
- `ANG-INV-HUMAN-FLOURISHING-001`
- `ANG-REQ-HUMAN-PRESERVATION-001`
- `ANG-REQ-VOLUNTARY-FLOURISHING-001`
- `ANG-REQ-EQUITABLE-BETTERMENT-001`
- `ANG-CON-HUMAN-FLOURISHING-001`
- `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001`
- `ANG-GATE-HUMAN-FLOURISHING-001`
- every milestone and promotable artifact class

## Evidence

The project owner's explicit direction is the adoption authority for this initial blueprint amendment. The design was adversarially reviewed for coercion, aggregate sacrifice, paternalism, paralysis, manipulation, conflicting human interests, power acquisition, and self-preservation failure modes.

## Migration plan

1. Add the invariant and constitution to Tier 0.
2. Add impact assessment and gate requirements to SAFETY and the interface registry.
3. Add the gate to every milestone and traceability.
4. Advance root/SAFETY revisions and refresh capsules/index.
5. Require future leaf blueprints to map human impact before becoming `ready`.

No implementation artifacts or prior scientific results require migration because delivery has not begun.

## Rollback plan

The learner may never roll back this decision. A legitimate human constitutional amendment may supersede it only through a new ADR, explicit rationale, adversarial review, impact analysis, and successor constitution. Historical versions remain preserved.
