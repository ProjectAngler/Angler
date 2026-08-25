---
blueprint_id: ANG-BP-EXAMPLE
title: Example branch
parent_id: ANG-BP-PARENT
revision: 1
tier: 2
design_status: stub
delivery_status: not_started
accountable_owner: unassigned
execution_owner: unassigned
updated_at: YYYY-MM-DD
parent_revision: 1
required_children: []
optional_children: []
depends_on: []
requirements: []
invariants: []
contracts_in: []
contracts_out: []
gates: []
tests: []
adrs: []
risks: []
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
context_budget:
  capsule_target_tokens: 1200
  required_read_set_max_tokens: 7400
  handoff_max_tokens: 1200
  overflow_action: split_or_narrow
---

# Outcome

State the one observable outcome this node owns.

## Context capsule

Summarize purpose, current design, current gate, exact interfaces, top risks, and next action. Keep this synchronized with `CAPSULE.md`.

## Contribution to parent

Explain why the parent cannot pass without this outcome.

## Inherited requirements and invariants

Reference IDs; do not copy and redefine their meaning.

## Scope

List responsibilities and state owned here.

## Explicit non-goals

List attractive adjacent work that belongs elsewhere or later.

## Inputs, outputs, and contracts

For each boundary state producer, consumer, version, preconditions, guarantees, failure, compatibility, and tests.

## Internal design and state ownership

Describe lifecycle, algorithms, components, and mutation authority.

## Child branch map

| Child ID | Outcome | Contracts | Gate | Status |
|---|---|---|---|---|

## Dependencies and sequencing

List dependency IDs and explain why they constrain work.

## Acceptance gate and evidence

Define entry criteria, procedure, precommitted thresholds, negative controls, evidence, verifier, and rollback.

Map the node's proportionate human-impact assessment and show how the independent human-flourishing gate can veto technical promotion.

## Testing and validation

Map unit, contract, integration, causal, regression, security, replay, resource, and reproducibility tests as applicable.

## Risks, failure behavior, and rollback

Use stable risk IDs. State fail-closed behavior and exact rollback boundary.

## Resource profiles and scaling

Explain constrained, workstation, server, and cluster behavior without redefining the scientific contract.

## Decisions and ADRs

List accepted, proposed, and superseded decision IDs.

## Current status and blockers

State design/delivery status, evidence already available, and blockers.

## Parent roll-up and next executable leaf

Give the bounded report to the parent and identify one exact next leaf.
