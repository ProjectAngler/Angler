---
blueprint_id: ANG-BP-PARTITIONS
title: Partitioning and sealing
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
  - ANG-BP-EVIDENCE-SCHEMAS
contracts_in:
  - ANG-CTR-EXPERIMENT-MANIFEST-001@1
contracts_out:
  - ANG-CTR-EVALUATION-SUITE-001@1
gates:
  - ANG-GATE-PARTITIONS-001
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Partitioning and sealing

## Outcome

Own disjoint adaptation, development, promotion, final-transfer, retention, and composition identities plus their visibility policy. A manifest records generator/version, selection rule, count, transformation strata, seed commitment, payload locator, authorized roles, exposure events, and retirement state.

The learner/updater may see adaptation observations and permitted feedback. Developers may inspect development fixtures. Promotion and final-transfer payloads, answer keys, raw seeds, and verifier secrets are evaluator-only. Candidate identity and all tuning choices freeze before resolving promotion material. Final-transfer material is one-use per milestone identity; exposure retires it.

For CR0, partition definitions and visibility roles are embedded as versioned sections of `schemas/control/v1/science/evaluation-suite.schema.json`. Valid commitments live in the declared control matrix, while deliberate overlap and visibility violations live in the release leaf's negative fixtures; CR0 authorizes no sealed payload or separate partition artifact.

## Tests and failure

Predeclared tests prove no instance/content identity overlaps across disallowed partitions, public manifests reveal commitments but not reconstructable secrets, unauthorized roles cannot resolve payloads, exposure is appended and causes retirement, and fixed manifests replay. Deliberate overlap and leaked-seed canaries must be rejected 100%.

Failure quarantines the suite, rotates sealed material, and invalidates every dependent receipt. No threshold or partition may be relabeled after results are viewed.

## Next leaf

Construct only the partition and visibility sections embedded in the release-listed evaluation-suite schema and their declared synthetic access/overlap cases under CR0.
