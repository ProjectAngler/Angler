---
blueprint_id: ANG-WORK-WORLDS-REVERSIBLE-TRANSITION-001
parent_id: ANG-BP-WORLDS
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-26
human_impact: LOW; contained local synthetic procedure-learning world
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
---

# Reversible primitive-transition world

## Objective

Provide a deterministic environment in which every observed action supplies a
forward and inverse transition example, while never exposing a shortest path or
prescribed solution.  This is the first causal laboratory for learning a world
model and constructing a procedure from an origin to a proposed destination.

## Exact outputs

- `src/angler/worlds/reversible_transition_world.py`
- `tests/unit/worlds/test_reversible_transition_world.py`
- this work leaf

The world displays a token permutation and accepts only one primitive adjacent
exchange per step.  A complete action sequence must be committed before the
independent executor runs it.  The executor reports the reached state and
whether it equals the supplied verification target; it does not expose a path,
next-action hint, distance, or intermediate reward.

## Acceptance

Tests must establish deterministic reset/replay, reversible primitive
round-trips, bounded and atomic execution, rejection of invalid actions and
over-budget procedures, unique public task identity, and absence of a public
solver/path interface.  Environment transition code is physics and judging,
not learner policy.

## Result

Implemented a six-token reversible world with five adjacent-exchange
primitives, immutable seeded task identity, frozen procedure commitment, and
atomic terminal execution.  The public API contains no solver, path, distance,
or next-action facility.  The focused world suite passes 12/12; the combined
world, core, and evaluator suite passes 23/23.

## Proportionate impact assessment

Impact class `LOW`, disposition `ALLOW` by the project owner for this exact
local synthetic experiment.  It uses no personal or recovered data, network,
service, external effect, deployment, autonomous tool use, or promoted-state
mutation.  It cannot pass a milestone or the Human-Flourishing gate.  Stop on
scope expansion; rollback removes only the three additive outputs above.
