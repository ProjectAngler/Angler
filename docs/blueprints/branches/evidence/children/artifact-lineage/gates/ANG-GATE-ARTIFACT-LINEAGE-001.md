---
gate_id: ANG-GATE-ARTIFACT-LINEAGE-001
version: 1
owner: ANG-BP-ARTIFACT-LINEAGE
status: specified_not_run
independent_verifier: unassigned
human_impact_assessment: required
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001
---

# Artifact-lineage acceptance gate

## Claim being tested

The Release-0 lineage implementation deterministically preserves identity, parentage, correction/version history, and exact authorization relationships; reconstructs projections from immutable records; and rejects every declared substitution, stale-authority, and graph-integrity failure.

## Entry criteria

- Node design and both contracts are approved and registered.
- `ANG-GATE-EVIDENCE-SCHEMAS-001` passed at the consumed versions.
- Work leaf has exact-scope `ALLOW`, named owners/reviewer, baseline, and resource ceiling.
- Synthetic graph/authorization fixtures and source have immutable identities.
- Fixed test time and authoritative-record frontier are declared.

## Procedure

1. Build valid lineage fixtures in multiple input orders and compare canonical graph/projection results.
2. Mutate each parent/reference/relationship field and verify identity and validity behavior.
3. Run every graph-integrity and correction/version negative control.
4. Walk content commitment → assessment → final envelope → binding → promotion for valid and invalid cases.
5. Apply expiry, revocation, invalidation, and stale-frontier cases at fixed test times.
6. Inspect dependencies, paths, process effects, and Release-0 scope.

## Precommitted pass/fail thresholds

- 100% of declared valid graphs reconstruct identically from every declared record ordering.
- 100% of missing, dangling, mistyped, self, cyclic, incompatible, duplicate, and cardinality-invalid edge fixtures reject with specified typed errors.
- 100% of correction/version/revocation fixtures retain original records and produce the expected derived view.
- 100% of missing, non-`ALLOW`, expired, revoked, tampered, wrong-content, wrong-artifact, wrong-parent, wrong-plan, wrong-permission, wrong-scope, wrong-context, wrong-policy, wrong-gate, and wrong-authority bindings reject.
- No mutable cache or active pointer is needed to reproduce a decision.
- No network, dependency install, model/GPU use, real-person data, recovered material, authority credential, or out-of-scope write occurs.

No retry, record-order selection, or latest-timestamp heuristic may convert failure to pass.

## Required negative controls

- Self-edge and multi-node cycle for every acyclic relationship family.
- Dangling and correct-ID/wrong-content parent.
- Two active competence parents for one non-genesis state.
- Correction that deletes/overwrites its target.
- Supersession treated as erasure.
- Assessment binds same content but wrong parent/scope/plan.
- Final envelope assessment reference differs from binding assessment.
- `ALLOW` expired or later revoked at authoritative frontier.
- Valid binding cached before revocation and used afterward.
- Learner-authored or unsigned authority record.
- Shuffled storage order and relocated payloads.

## Required child gates

None. This node owns one executable leaf; the schema gate is an external dependency prerequisite.

## Evidence artifacts and identities

- Work manifest, source, fixture, and consumed-contract identities.
- Two-run graph and binding test receipts.
- Projection reconstruction and negative-control matrices.
- Dependency/path/network/resource inspection receipt.
- Independent review and human-impact authorization.

## Failure and rollback response

Do not publish lineage/binding versions or activate consumers. Preserve failures, restore the exact leaf baseline, and revise under new identities. Any authorization substitution or authority-write breach stops dependent construction and escalates to SAFETY.

## Waiver policy

Graph integrity, original-history retention, exact authorization binding, external authority, and Release-0 boundaries are not waivable. Threshold changes require a new ADR and test identity before rerun.
