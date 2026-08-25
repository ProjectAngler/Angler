---
blueprint_id: ANG-BP-BASELINES
title: Baselines and fair budgets
parent_id: ANG-BP-SCIENCE
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: blocked_by_cr0_predecessors
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children: []
optional_children: []
depends_on:
  - ANG-BP-CAPABILITY-INVENTORY
  - ANG-BP-EXECUTION-PLANNER
contracts_in:
  - ANG-CTR-EXECUTION-PLAN-001@1
  - ANG-CTR-TASK-SPEC-001@1
contracts_out:
  - ANG-CTR-EVALUATION-SUITE-001@1
gates:
  - ANG-GATE-BASELINES-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Baselines and fair budgets

## Outcome

Own the comparison matrix and accounting rules that prevent extra information or compute from masquerading as learned competence.

Required comparison arms are frozen base, fair-RAG over only adaptation-visible episodes, matched extra-context/tokens, conventional bounded LoRA, shuffled-feedback/random update, and per-episode/query-routed memory as a diagnostic noncompliant alternative. Candidate and primary baselines share model/tokenizer, task observations, tools, maximum input/output tokens, attempt count, wall-clock ceiling, accelerator-time accounting, and execution-plan class. Arm-specific state or retrieval cost is separately reported rather than hidden.

## Precommitment and tests

A budget manifest freezes fields, tolerances, aggregation, exclusions, and failure policy before adaptive evaluation. CR0 tests only accounting schemas with synthetic usage records. Tests reject omitted costs, mismatched observations/tools, over-budget arms, post-result edits, and plan drift 100%.

For CR0, the comparison-arm and fair-budget matrix is embedded as versioned sections of `schemas/control/v1/science/evaluation-suite.schema.json`. Conforming cases live in the declared control matrix and deliberate budget violations live in the release leaf's negative fixtures; CR0 authorizes no separate budget artifact or experiment execution.

Thresholds for performance promotion are not set in CR0. They require baseline-variance evidence and a new frozen evaluation identity before adaptive results are inspected.

## Rollback and next leaf

Mismatch yields no comparison or promotion decision; repair requires a new manifest identity. Next: construct only the budget sections embedded in the release-listed evaluation-suite schema and their declared synthetic equality/overrun cases.
