---
blueprint_id: ANG-WORK-LEARNING-PERSISTENT-PLASTICITY-001
parent_id: ANG-BP-META-UPDATER
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-25
human_impact: LOW; contained local synthetic learning research
---

# Persistent learned-plasticity substrate

## Objective

Replace the fixed optimizer as Angler's adaptation mechanism with one shared,
fixed-size neural state whose update dynamics are themselves learned.  The
foundation model remains frozen and supplies only detached representations.

## Exact first increment

- Add an Angler-native, pure-PyTorch stateful self-referential fast-weight
  module adapted from the MIT-licensed ACL/SRWM donor design.
- The state is common to every problem.  It receives no task-family ID,
  adapter selection, oracle answer, hidden target, or replay buffer.
- Provide exact snapshot/restore and causal reset/swap behavior.
- Demonstrate that gradients from a later fresh query reach the slow update
  dynamics through an earlier state change.

Authorized paths for this increment:

- `src/angler/reasoning/self_referential_memory.py`
- `src/angler/reasoning/__init__.py`
- `tests/unit/reasoning/test_self_referential_memory.py`
- this work leaf

## Acceptance gate

Run the reasoning unit suite in the isolated WSL2 environment.  It must prove:

1. fixed state size independent of stream length;
2. deterministic exact snapshot/restore;
3. a presented experience changes the state and later behavior;
4. state reset erases and state swap transfers that behavioral change;
5. an outer loss on later unseen input trains the shared update dynamics;
6. no family/router/replay/oracle field or deterministic task solver exists.

This increment proves only the plastic substrate.  The next leaf must connect
attempted solutions and scalar feedback, meta-train it across changing
mechanisms, and use prequential fresh-query retention tests.  Same-generator
heldouts alone cannot satisfy that gate.

## Result

Implemented one 100,352-scalar state at production width 512 with 8 heads
(401,408 bytes in FP32).  Focused SRWM tests passed 8/8; the combined reasoning
suite passed 16/16; the remaining WSL unit modules passed 57/57; and the
standard-library Evidence suite passed 16/16 on the host.  The tests establish
the six substrate properties above, not cross-mechanism learning.

## Effects and rollback

Local source and tests only.  No network, package installation, service,
external effect, deployment, personal data, or foundation-weight mutation.
Rollback is the exact parent Git commit.  Failed experiments do not become the
active state.
