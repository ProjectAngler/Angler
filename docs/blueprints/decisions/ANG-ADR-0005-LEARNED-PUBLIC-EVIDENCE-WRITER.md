---
adr_id: ANG-ADR-0005
title: Bounded learned public-evidence residual writer
status: accepted
owner: ANG-BP-LEARNING
accepted_at: 2026-08-27
authority: project owner continuation direction
supersedes: none
superseded_by: none
---

# Bounded learned public-evidence residual writer

## Context

The third-family v4 experiment proved that adding a typed demonstration vector
to the frozen generic writer's structural context could change memory and
logits, but it collapsed correct and wrong demonstrations into the same
presence signal.  Changing the retained base writer would risk both acquired
families.  A typed public observation therefore needs a narrow learned write
interface that can preserve its content without becoming an answer decoder.

## Decision

`RoutedProceduralMemory.propose_feedback` and `incorporate_feedback` accept an
optional width-matched `public_evidence` tensor.  A memory tier may attach one
`PublicEvidenceResidualWriter`.  It combines that evidence with the frozen
base content and outcome-direction features and emits two independently
bounded residuals.  The content residual is stored only through the existing
transactional running-mean state update; the direction residual is multiplied
by the same externally supplied scalar outcome as the base writer.

The bridge starts as an exact no-op, is gated off for an all-zero evidence
tensor, cannot select a slot, and has no path to current logits or candidate
answers.  Its residual magnitude is capped at `0.25` per coordinate.  In the
active demonstration leaf, only the typed sensory encoder and the composition
tier's residual bridge may train.  The base writers, router, utility decoder,
reversible transition, compiler, leaf memory, precedence adapter, and
foundation remain frozen.  A support is judged once; matched causal arms reuse
that one scalar result.

## Required evidence

- Zero evidence and an attached zero bridge preserve the prior write and state
  bit-for-bit.
- Demonstration content cannot change current logits or its write address.
- A public alpha-rename remains closer than a fixed public output rotation in
  both sensory evidence and the resulting selected-slot state.
- Correct, absent, and rotated demonstrations use identical attempted actions
  and scalar feedback; later queries contain no demonstrations and perform no
  writes.
- Removing the acquired state or reversible transition removes the claimed
  gain, while earlier families remain unchanged without replay.

## Non-equivalence

This is a learned input-to-memory tool, not a deterministic permutation solver,
answer head, promotion, deployment, foundation update, or AGI claim.  No model,
network, real-data, tool-execution, or external-effect authority is added.

## Rollback

Remove the optional argument and unattached bridge, restore the v4 call path,
and discard only v5 interface checkpoints.  Existing V51 and precedence
checkpoints remain compatible because they contain no bridge parameters and
the module is attached only after those frozen states load.
