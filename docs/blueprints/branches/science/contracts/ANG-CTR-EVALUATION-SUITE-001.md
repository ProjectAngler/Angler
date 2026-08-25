---
contract_id: ANG-CTR-EVALUATION-SUITE-001
version: 1.0.0
status: approved_for_cr0_design
owner: ANG-BP-SCIENCE
producer: independent evaluation authority
consumers: [ANG-BP-RUNTIME, ANG-BP-EVIDENCE, ANG-BP-SCIENCE]
---

# EvaluationSuite v1

## Input and output

The producer consumes approved benchmark-family versions, partition commitments, TaskSpec contracts, baseline/budget policy, causal interventions, metrics/statistical policy, an execution-plan compatibility class, and human-impact requirements. It emits an immutable suite envelope containing:

- suite/contract identity, intended claim, benchmark transformations, and compatible TaskSpec versions;
- adaptation/development/promotion/final-transfer/retention/composition partition commitments and visibility roles;
- frozen comparison arms, observable-information/tool/token/attempt/accelerator/time accounting rules, tolerances, and exclusion policy;
- zero/swap/permutation/replay/shuffled-feedback/tool-disabled interventions applicable at the milestone;
- metrics, aggregation, seed/stratum policy, required negative controls, and precommitment state;
- protected generator/seed/verifier references, evaluator authority, stop/timeout policy, required receipts, and evidence envelope.

Sealed payloads use protected commitments and evaluator-only resolution. Suite identity never exposes raw hidden seeds, answers, labels, sizes that leak them, or preferred reasoning traces.

## Behavior and invariants

The suite is immutable. Candidate identity and all tuning choices freeze before sealed payload resolution. Performance thresholds freeze under a new evaluation identity after baseline variance is measured and before adaptive results are viewed. Candidate and baselines receive matched declared observations, tools, token/attempt ceilings, and execution-plan class; arm-specific costs remain visible. WORLDS validates outcomes, but cannot redefine the scientific claim or partition.

## Failure, operation, and compatibility

Typed failures are `UNSUPPORTED_VERSION`, `IDENTITY_MISMATCH`, `PARTITION_EXPOSED`, `BUDGET_POLICY_MISSING`, `PLAN_INCOMPATIBLE`, `THRESHOLD_NOT_PRECOMMITTED`, `UNAUTHORIZED_RESOLUTION`, and `SUITE_INVALID`. They fail before candidate evaluation and are not retryable under the same suite if hidden material or results were exposed. A resource/transient failure may retry only with the same frozen identities and an explicit attempt identity within the suite policy.

Each evaluation attempt declares per-task and total timeouts. Unknown major versions reject; additive minor versions require consumer conformance and cannot change visibility, metric, threshold, intervention, or budget semantics. Material changes create a successor suite/evaluation identity.

## Validation

Producer/consumer tests reject missing fields, deliberate partition overlap, reconstructable sealed seeds, post-result threshold edits, unequal information/tools/budgets, omitted arm cost, incompatible plans, and unauthorized payload resolution. Release 0 builds schemas and synthetic controls only; it cannot resolve real sealed data or run a model.
