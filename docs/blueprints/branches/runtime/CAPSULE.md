---
blueprint_id: ANG-BP-RUNTIME
blueprint_revision: 2
capsule_revision: 2
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# RUNTIME capsule

Mission: execute one immutable foundation substrate with exactly one uniformly active, causally intervenable plastic competence state.

Owns model/tokenizer boundary, action runtime, LoRA state representation, state lineage, atomic learning transactions, backend compatibility, and mechanical promotion authorization enforcement. It does not compute updates or author scientific/human-impact decisions.

Consumes `ExecutionPlan`, `TaskSpec`, `UpdateProposal`, `EvaluationReceipt`, `PromotionDecision`, and protected `HumanImpactAssessment`. Produces `Trajectory`, `PlasticStateRef`, and `TransactionReceipt`.

State lifecycle: promoted parent → workspace → immutable candidate → promoted new head only with scientific approval plus an exact current external human-impact `ALLOW`; every missing/non-allow/tampered/mismatched case rejects and exactly restores the parent. Only the transaction coordinator advances lineage. No query-conditioned model/adapter selection.

Required children: MODEL-BOUNDARY, AGENT-RUNTIME, PLASTIC-STATE, STATE-LINEAGE, LEARNING-TRANSACTION, RUNTIME-COMPATIBILITY.

Current gate: `ANG-GATE-RUNTIME-DESIGN-001`. Design revision 2 draft, delivery not started.

Top risks: authorization bypass, query routing, partial promotion, base mutation, and undeclared backend numeric drift.

Next action: specify the model boundary, plastic-state format/interventions, and runtime compatibility together. Read root capsule, RUNTIME blueprint, interface registry, and slices 02–04.
