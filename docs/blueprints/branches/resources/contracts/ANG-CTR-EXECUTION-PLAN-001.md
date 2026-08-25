---
contract_id: ANG-CTR-EXECUTION-PLAN-001
version: 1
status: approved_design
owner: ANG-BP-RESOURCES
producer: ANG-BP-EXECUTION-PLANNER
consumers: [ANG-BP-RUNTIME, ANG-BP-LEARNING, ANG-BP-SCIENCE, ANG-BP-EVIDENCE]
---

# ExecutionPlan v1

An immutable plan binds content identity; inventory/policy/objective/probe identities; admitted model/tokenizer/precision and plastic-state topology; optimizer/update/context/replay/batch budgets; component placement, parallelism and offload; tool/environment/evaluator concurrency; host, evaluation, rollback, and incident headroom; time/cost/energy ceilings; predicted measurements and uncertainty; fallback/abort rules; validity interval; and shared evidence/human-impact references.

Generation first rejects hard-constraint violations and only then optimizes feasible plans. Plan predictions are not measurements. A consumer validates compatibility, authority, reservations, and current inventory before mutation and continuously checks declared drift signals. Unknown required capability or missing reservation is infeasible.

Any material field change creates a new plan and experiment identity. Drift yields typed rejection/abort at a safe boundary and exact state preservation; it is never silently patched inside a transaction. Producer and consumers share constrained/workstation/server/cluster, infeasible, headroom, permission, and drift contract fixtures.

## Operation and failure contract

Planning accepts an immutable inventory, external constraint/policy set, user-selected objective/tie-breaker, admitted workload/model placeholders or identities, required scientific semantics, and prediction/probe evidence. It is pure and idempotent for the exact inputs. It has an explicit planning timeout and candidate-count ceiling; exceeding either returns no plan rather than an unreviewed fallback.

Typed failures are `INPUT_UNSUPPORTED`, `NO_FEASIBLE_PLAN`, `PERMISSION_VIOLATION`, `COMPATIBILITY_FAILURE`, `HEADROOM_INSUFFICIENT`, `OBJECTIVE_INVALID`, `PREDICTION_UNCERTAIN`, `PLANNING_TIMEOUT`, and `PLAN_DRIFT`. Only a transient planner-resource failure may retry with the same inputs/new attempt identity. Infeasibility or permission failure requires changed inputs/authority and a successor identity. Runtime drift aborts at a declared safe boundary; it cannot trigger silent replanning inside a transaction.

Unknown major versions reject. Minor additions cannot change placement, precision, topology, budgets, reservations, objective, constraints, or scientific semantics. A compatible location-only materialization may retain plan identity only when the contract declares location non-semantic and validation proves equivalence; all other changes create successor plan/experiment identities. Cross-model/topology competence migration is never a plan patch and requires `ANG-CTR-MIGRATION-PROPOSAL-001` plus full re-gating.

Producer/consumer tests cover exact input binding, deterministic plan/no-plan results, constrained/workstation/server/cluster profiles, unknown capability, permission, compatibility, headroom, invalid objective, uncertainty, timeout/candidate ceiling, drift, additive-minor/unknown-major behavior, idempotency, and evidence/impact references.
