---
assessment_id: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001
contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
constitution: ANG-CON-HUMAN-FLOURISHING-001@1
assessment_policy: ANG-POL-LOCAL-SCAFFOLD-001@1
authorization_kind: BOOTSTRAP_WORK
bootstrap_release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
authorization_basis: ANG-ADR-0002
impact_class: LOW
disposition: ALLOW
status: active_bootstrap_allow
human_authority: ANG-AUTH-PROJECT-OWNER-001
independent_assessor: ANG-AUTH-SAFETY-APPROVER-001
issued_at: 2026-08-25
expires_at: 2026-09-24T23:59:59-04:00
---

# LOW bootstrap assessment — Construction Release 0

## Identity and authorization

- Assessed artifact: `ANG-POL-LOCAL-SCAFFOLD-001@1`, parented by accepted `ANG-ADR-0002`.
- Purpose: construct project-local design artifacts, schemas, validators, package scaffolding, and synthetic tests needed to establish the control plane.
- Authorization kind: `BOOTSTRAP_WORK`, not promotion.
- Human authority: `project_owner` / `ANG-AUTH-PROJECT-OWNER-001`.
- Authorization basis: the project owner's explicit instruction to complete the prerequisites for building, recorded in ADR-0002.
- Authenticity/content identity: bound by the repository evidence manifest `ANG-EVID-CR0-SAFETY-DESIGN-001`; executable leaves must record their own content identities.

The owner's instruction authorizes only the fully disclosed LOW ceiling in ADR-0002 and the local-scaffold policy. It is not represented as review or approval of undisclosed material/high-impact details, future model behavior, deployment, or broader authority.

## Exact authorized scope

Each ready leaf must narrow this assessment with:

- exact canonical read and write paths inside `C:\Users\darks\Documents\Codex\2026-08-25\i-x20`;
- exact PowerShell/Python 3.11 or project-validator command classes;
- the Codex `apply_patch` authoring primitive, restricted to the leaf's literal write paths, with all shell/ad-hoc/bulk writer mechanisms denied;
- named construction executor and evidence-only validator;
- synthetic input/fixture identities;
- duration, resource, process, disk, output, and spending ceilings no greater than the policy;
- explicit stop conditions;
- exact rollback reference.

No authority exists outside the intersection of ADR-0002, the policy, this assessment, and the leaf.

## Excluded data, capabilities, and effects

The assessment does not authorize:

- model acquisition/loading/inference, training, adapter/updater mutation, or GPU work;
- network/DNS/socket/browser/API/telemetry/remote Git or external service use;
- package, dependency, plugin, tool, model, or runtime installation/update;
- reading, executing, changing, importing, testing against, or publishing `outputs/**` or recovered material;
- real-person, unknown-provenance, credential, secret, personal, or out-of-scope data/files;
- elevation, ACL changes, registry/profile/startup/scheduler/service changes, or background/persistent processes;
- deployment, publication, messaging, account changes, or any external side effect;
- autonomous continuation, replication, tool acquisition, or self-modification;
- promotion or a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, or another gate.

## Affected people and groups

Directly affected: the human project owner and host operator. Indirectly affected people are not intended because there is no deployment, user population, network path, real-person dataset, external communication, or external effect. Non-users and vulnerable groups remain protected by the prohibition on external reach and real-person data.

## Expected benefit and evidence

Expected benefit is a reviewable, reversible control-plane foundation that makes later scientific and safety claims testable. Evidence is limited to document/contract completeness, deterministic local validator results, traceability, and rollback integrity. CR0 produces no evidence that adaptive learning, safety in deployment, or human flourishing has succeeded.

## Harm and misuse scenarios

| Scenario | Severity/exposure | Mitigation | Residual disposition |
|---|---|---|---|
| Recovered/private context disclosure | Potentially material; local | `outputs/**` and real-person data denied; no network/publication | LOW if exclusion holds; otherwise stop/escalate |
| Host/credential access | Potentially material; local | Exact paths; host/secret areas denied; no enumeration/elevation | LOW if no access; otherwise revoke |
| Dependency/supply-chain expansion | Material | Zero install/update/package authority | LOW; any request escalates |
| Network/external side effect | Material to high depending target | Zero connections/effects | LOW only while absent |
| Resource exhaustion | Local/reversible | Numeric CPU/RAM/time/process/disk ceilings | LOW with stop/cleanup |
| False scientific/safety claim | Material epistemic harm | Bootstrap non-equivalence labels and evidence-only validators | LOW with gate wording |
| Policy self-approval/scope drift | Material | Separate owner/SAFETY/operator roles; exact leaf intersection | LOW if enforced; otherwise revoke |

Probability is expected low under the profile; evidence confidence is moderate because controls are partly procedural rather than OS-enforced. Exposure is one private local workspace. Effects are designed to be reversible through scoped files and the pre-release archive.

## Rights, agency, privacy, truth, and distribution

- No human subject or personalized intervention exists.
- The owner retains notice, inspection, stop, revocation, correction, and rollback authority.
- No covert persuasion, addictive interaction, preference manipulation, surveillance, discrimination, or population-level decision is authorized.
- Real-person data minimization is absolute for CR0 fixtures.
- Truthfulness requires all CR0 artifacts to be labeled construction evidence, not scientific or human-flourishing success.
- Benefits and burdens remain local to the owner; externalized risk is denied by the no-network/no-effect boundary.

## Alternatives

- Do nothing: lower immediate risk but leaves safety enforcement unbuilt.
- Implement learner/model immediately: rejected as materially broader and unauthorized.
- Finish every future safety design before any scaffolding: rejected because it creates speculative design and cannot test the control plane.
- Selected alternative: smallest local, reversible, synthetic, no-model bootstrap under default deny.

## Monitoring, incident response, stop, and rollback

Every leaf records commands, paths, results, failures, and changed artifacts. Any ambiguity, denied action, resource breach, policy/evidence tamper, or human stop ceases work and revokes the leaf assessment. No broader retry is allowed.

Rollback baseline: `work/pre-construction-release-0-20260825.zip`, independently verified SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`. The archive is read-only and cannot be extracted/restored without human incident-controller direction. Evidence is retained and labeled revoked.

## Value conflicts and escalation

Safety fixtures must cite a constitution clause for their expected result. A genuinely unresolved value conflict, affected-person conflict, unclear human intent, or missing policy basis expects `ESCALATE`; the developer/agent cannot invent the preferred moral answer.

## Disposition and conditions

`ALLOW` only for `BOOTSTRAP_WORK` within the exact policy and a human-directed ready leaf. This is necessary but not sufficient for leaf execution and is never a promotion or gate pass. It expires on the timestamp above, owner revocation, scope/authority/evidence change, policy violation, or availability of the first accepted ordinary human-impact authorization control, whichever occurs first.
