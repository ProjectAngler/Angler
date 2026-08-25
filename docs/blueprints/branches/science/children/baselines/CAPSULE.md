---
blueprint_id: ANG-BP-BASELINES
blueprint_revision: 1
capsule_revision: 3
freshness_date: 2026-08-25
parent_id: ANG-BP-SCIENCE
target_tokens: 500
---

# BASELINES capsule

Mission: compare adaptation against frozen and memory/compute explanations under matched observable information and declared budgets.

Required arms: frozen, fair-RAG, matched extra tokens/context, conventional bounded LoRA, shuffled/random update, and diagnostic query-routed memory. Shared fields include model, tasks, tools, token/attempt/time ceilings, accelerator accounting, and execution-plan class; arm-specific costs remain visible.

CR0 builds fair-budget sections embedded in the evaluation-suite schema and synthetic accounting cases only. Performance thresholds wait for baseline variance and freeze before adaptive results. Gate: `ANG-GATE-BASELINES-001`; design is approved for CR0, while delivery remains blocked by release predecessors. Any mismatch invalidates comparison rather than being normalized after the fact.
