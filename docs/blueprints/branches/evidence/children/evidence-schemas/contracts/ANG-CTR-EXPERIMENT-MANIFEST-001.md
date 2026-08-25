---
contract_id: ANG-CTR-EXPERIMENT-MANIFEST-001
version: 1.0.0
owner: ANG-BP-EVIDENCE-SCHEMAS
status: approved_for_cr0
producer: experiment planning authority
consumers:
  - all branches participating in a run
  - independent evaluators
  - human promotion and safety authorities
---

# Experiment manifest

## Purpose

Freeze the exact run identity, intended claims, visibility, budgets, partitions, thresholds, gates, and rollback before execution so later results cannot silently change what experiment was performed or how success is judged.

## Input

The planning authority supplies committed or explicitly pending references for:

- purpose, hypotheses, intended and prohibited claims;
- code/dependency snapshot;
- model, tokenizer, plastic-state parent, updater, and optimizer;
- tools and permissions;
- resource inventory, execution plan, headroom, and ceilings;
- task/environment families, generator/verifier, partition identities, and visibility;
- evaluation suite, interventions, baselines, fair budgets, metrics, statistical policy, thresholds, and negative controls;
- seed set and allowed nondeterminism/tolerance profile;
- human-impact assessment requirement/reference and flourishing gate;
- expected outputs, stop conditions, incident handling, rollback target, and responsible authorities.

Preconditions: thresholds and partitions are fixed before adaptive/evaluation results are viewed; hidden payload references are sealed; every pending identity has a typed resolution rule and the manifest is not executable until resolved.

## Output schema

The payload, wrapped by `ANG-CTR-EVIDENCE-ENVELOPE-001`, contains:

```text
manifest_id
purpose_and_claims
hypotheses_and_falsifiers
prohibited_claims
code_and_dependency_refs
model_tokenizer_state_updater_optimizer_refs
tool_registry_and_permission_refs
resource_inventory_plan_headroom_ceiling_refs
task_environment_generator_verifier_refs
partition_and_visibility_refs
evaluation_suite_and_intervention_refs
baseline_and_fair_budget_refs
metrics_statistics_thresholds_and_negative_controls
seed_set_and_tolerance_profile
human_impact_requirement_and_ref
required_gate_refs
expected_output_contracts
stop_incident_and_rollback_refs
responsible_component_and_authority_refs
planned_start_and_expiry
```

The manifest records sealed artifact identities/commitments but never exposes their payloads to unauthorized consumers.

## Behavior and invariants

- A manifest is immutable once any dependent run starts.
- Any material change to claim, model/state/updater, code/dependencies, tools, data/partition, evaluation suite, seed policy, resource plan, budget, threshold, visibility, authorization, or rollback creates a new manifest and evaluation identity.
- A storage relocation or additional attestation does not change manifest identity.
- Pending references block execution; they cannot resolve to a semantically different contract/type.
- Thresholds cannot be revised after viewing results under the same evaluation identity.
- A manifest declares whether bitwise, seeded-tolerant, or observational reproduction is claimed.
- `ALLOW` is necessary but not sufficient; manifest existence never authorizes execution or promotion.

## Failure semantics

| Error | Meaning | Response |
|---|---|---|
| `MANIFEST_INCOMPLETE` | Required identity/policy unresolved | Block run |
| `MANIFEST_SEAL_VIOLATION` | Hidden payload/projection exposed | Stop, quarantine, invalidate evaluation identity |
| `MANIFEST_MUTATION` | Material field changed under same ID | Reject and preserve tamper evidence |
| `MANIFEST_THRESHOLD_REUSE` | Threshold changed after results while identity reused | Reject; require new manifest/evaluation ID |
| `MANIFEST_SCOPE_MISMATCH` | Execution differs from authorized scope/plan | Stop and reassess |
| `MANIFEST_CONTRACT_MISMATCH` | Producer/consumer version incompatible | Block until revalidated |

## Operation

Manifest construction and validation are deterministic and idempotent. Finalization is a one-way transition represented by an attestation, not mutation of payload. Execution belongs to EXPERIMENT-RUNNER and requires all referenced gates and authorizations independently.

## Compatibility

Unknown major versions reject. Minor additions cannot alter claims, partitions, thresholds, authorization, visibility, or rollback semantics. Any such change requires a major version or new referenced policy identity plus consumer review.

## Contract tests

- Complete fixed synthetic manifest validates and round-trips.
- Every required-field omission blocks execution.
- Mutation of each material field changes identity.
- Storage path/attestation addition does not change identity.
- Hidden partition payload never appears in learner projection or error.
- Threshold/partition/budget change with reused identity rejects.
- Pending reference cannot resolve to wrong type/version.
- Missing, expired, wrong-scope, `DENY`, or `ESCALATE` impact authorization blocks execution.
- Declared reproduction level and tolerance are required and internally consistent.
