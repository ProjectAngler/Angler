---
blueprint_id: ANG-WORK-WORLDS-LATENT-PROGRAM-STREAM-001
parent_id: ANG-BP-WORLDS
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-25
human_impact: LOW; contained synthetic evaluation world
---

# Latent ordering-program stream

## Objective

Create the replayable world primitive for a compact stream where the latent
solution mechanism changes.  Train on primitive and shallow ordering programs;
reserve deeper operator compositions for validation/test.  The learner sees
only public item attributes, its own attempted ordering, and one scalar
pairwise score after acting.  The successor runner owns stream scheduling,
global uniqueness, one-pass presentation, and replay-count enforcement.

## Exact outputs

- `src/angler/worlds/latent_order_programs.py`
- `src/angler/worlds/__init__.py`
- `experiments/evaluators/latent_order_suite.py`
- `tests/unit/worlds/test_latent_order_programs.py`
- this work leaf

The evaluator-side program algebra may sort by either of two public ranks,
partition by a public group bit, zigzag positions, rotate to a marked item, or
choose a branch using a public flag.  Complete test compositions are absent
from training, while every primitive meaning is represented during training.

## Boundaries

Program AST, split, family/type ID, target, violated pair, solution trace, and
generator seed never enter the learner projection.  Exact final evaluator ASTs
are outside the public `angler.worlds` API and are loaded only after a candidate
is frozen.  Distinct public contents have distinct identities, while
counterfactual programs over identical public content have the same learner
projection.  Deterministic code is limited to generating public worlds and
judging final outcomes; it is not part of Angler's policy.

## Acceptance

Tests must prove public/hidden separation, deterministic seeded generation,
unique identities for distinct public instances, query-dependent targets, disjoint complete program
structures, nontrivial deeper evaluator compositions, exact scalar scoring,
balanced controllable public flags, and renaming invariance.  Some conditional
branches intentionally reuse familiar subprograms, so results must be reported
per flag and by the worst flag.  This world supports a bounded claim about
online acquisition within one ordering action algebra.  It cannot establish
adaptation to arbitrary new semantics; later worlds must widen the action and
observation algebras.

## Effects and rollback

Local synthetic source/tests only.  No model, GPU, network, package, service,
external effect, personal data, or deployment.  Roll back to commit `60e6568`.

## Result

The evaluator-private program algebra, public/hidden separation, structural
and sampled semantic split checks, balanced flag control, independent operator
fixtures, and scalar judge passed 9/9 focused unit tests.  This world remains
valid as a diagnostic primitive; its successor learning candidate failed for
causal reasons recorded in `ANG-WORK-LEARNING-PREQUENTIAL-META-001`.
