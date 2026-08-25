---
blueprint_id: ANG-BP-LEARNING
blueprint_revision: 1
capsule_revision: 1
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# LEARNING capsule

Mission: convert visible experience and bounded outcome feedback into transferable candidate competence, then eventually improve the update rule itself.

Owns feedback update v0, replay selection, meta-updater, consolidation, retention, and skill composition. Does not own state promotion, hidden evaluation, environment truth, or resource placement.

Consumes `Episode`, `Feedback`, parent `PlasticState`, safe replay, and `ExecutionPlan`. Produces one parent-bound `UpdateProposal`; only plastic parameters may change.

First mechanism: bounded FTTT-inspired LoRA gradient update. Later order: replay → learned updater → consolidation → lifelong retention/composition. Later systems may not be added to hide failure of the minimal learner.

Current gate: `ANG-GATE-LEARNING-DESIGN-001`. Design draft, delivery not started.

Core proof at M3: held-out structural gain above frozen/fair-RAG; zero erases, swap transfers, replay reconstructs, and old skills remain in bounds.

Top risks: episode memorization, evaluator leakage into gradients, interference, updater overfit, and accumulated state drift.

Next action: expand FEEDBACK-UPDATE after state, episode, feedback, plan, and evaluation interfaces stabilize. Read root capsule, LEARNING blueprint, interface registry, and slices 03–06.

