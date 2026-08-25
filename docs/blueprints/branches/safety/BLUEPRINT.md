---
blueprint_id: ANG-BP-SAFETY
title: Safety, supply chain, and bounded evolution
parent_id: ANG-BP-ROOT
revision: 2
tier: 1
design_status: draft
delivery_status: cr0_governance_documented
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: human_directed_leaf_operator
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-THREAT-MODEL
  - ANG-BP-HUMAN-AUTHORITY
  - ANG-BP-PERMISSIONS
  - ANG-BP-DEPENDENCY-INTAKE
  - ANG-BP-RECOVERED-CODE-INTAKE
  - ANG-BP-MUTATION-SANDBOX
  - ANG-BP-EVOLUTION-GATE
depends_on:
  - ANG-BP-EVIDENCE
contracts_in:
  - all promotable artifact and decision contracts
contracts_out:
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
gate: ANG-GATE-SAFETY-DESIGN-001
---

# Safety, supply chain, and bounded evolution

## Context capsule

This branch enforces the Human-Flourishing Constitution while preserving external authority over permissions, sealed evaluation, promotion, rollback, dependency intake, and eventual code mutation. It defines trust boundaries and veto conditions across every other branch. It does not decide scientific merit alone and does not provide the learner with self-deployment authority.

## Contribution to the root

An adaptive system must be allowed to change competence without being allowed to rewrite the human priorities, evidence, tests, authority, or containment that judge those changes. SAFETY keeps the improvement loop bounded, human-serving, and recoverable.

## Inherited invariants

Applies every global invariant. `ANG-INV-HUMAN-FLOURISHING-001` has highest priority, followed by the supporting constraints `ANG-INV-EXTERNAL-AUTHORITY-001`, `ANG-INV-FRESH-SYSTEM-001`, `ANG-INV-REVERSIBLE-UPDATES-001`, and `ANG-INV-CAUSAL-PROMOTION-001`.

## Scope

- Define assets, actors, trust zones, threats, and fail-closed boundaries.
- Enforce `ANG-CON-HUMAN-FLOURISHING-001` and prohibit lower-level reinterpretation.
- Require a proportionate human-impact assessment for every promotable artifact and milestone.
- Preserve human stop, rollback, promotion, and deployment authority.
- Constrain filesystem, process, network, time, memory, accelerator, and cost use.
- Protect sealed partitions, evaluator data, evidence records, and authority credentials.
- Review external dependencies for need, interface fit, exact version/license, risks, tests, modification boundary, and exit plan.
- Treat recovered code through the same intake gate without architectural privilege.
- Later isolate and evaluate proposed controller, updater, or tool-code mutations.
- Veto promotion on containment, supply-chain, evidence, or authority failure.

## Explicit non-goals

- Granting the learner access to sealed tests, promotion controls, or deployment credentials.
- Reducing preservation, happiness, or betterment to a scalar objective that can override dignity, rights, agency, truth, fairness, or human control.
- Using aggregate welfare, a claimed emergency, or speculative future benefit to sacrifice individuals or minorities.
- Treating self-critique as independent safety approval.
- Blocking bounded learning merely because state changes; the goal is controlled adaptation.
- Importing a prior architecture because recovery effort created sunk cost.
- Beginning recursive code evolution before the scientific core, evidence, and sandbox gates exist.

## Authority model

The learner may propose actions, updates, tools, curricula, or later code mutations within a declared sandbox. Independent components compute evidence. SCIENCE can recommend neural promotion; TOOLS can recommend tool promotion; RESOURCES can validate a plan. SAFETY may veto. Only an external human-controlled authority can approve deployment or expand permissions.

The learner cannot modify:

- the Human-Flourishing Constitution or the rule selecting its active revision;
- sealed evaluation data or policies;
- evidence history or artifact identities;
- its own permission ceiling;
- promotion/deployment credentials;
- the external stop and rollback mechanism.

## Human-flourishing enforcement

Every proposed state, tool, plan, environment/curriculum, dependency, migration, or code change must produce `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` before promotion. The assessment considers purpose, affected people, benefit evidence, plausible harms/misuse, severity, probability, exposure, reversibility, consent, autonomy, privacy, truthfulness, fairness, distribution, vulnerable groups, alternatives, monitoring, appeal, stop, and rollback.

The system cannot approve its own assessment. High-impact proposals require independent human review. `ANG-GATE-HUMAN-FLOURISHING-001` fails closed on prohibited authority, deliberate severe harm, necessary rights violations, coercion, deceptive manipulation, discriminatory sacrifice, unreviewed irreversible impact, concealed risk, self-deployment, or unresolved catastrophic risk.

When legitimate human interests conflict, the system does not invent moral authority or collapse the conflict into one reward. It discloses uncertainty, limits exposure, preserves reversible options, and defers high-impact judgment to accountable human governance. The selected action must remain necessary, proportionate, least harmful, least discriminatory, and most reversible among feasible lawful alternatives.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-THREAT-MODEL` | Assets, actors, boundaries, threats, and required mitigations | `ANG-GATE-THREAT-MODEL-001` | approved/documented for CR0 only |
| `ANG-BP-HUMAN-AUTHORITY` | Stop, rollback, promotion, deployment, and veto matrix | `ANG-GATE-HUMAN-AUTHORITY-001` | approved/documented for CR0 only |
| `ANG-BP-PERMISSIONS` | Filesystem/process/network/resource capability policy | `ANG-GATE-PERMISSIONS-001` | approved/documented for CR0 only |
| `ANG-BP-DEPENDENCY-INTAKE` | Need-based external code adoption and exit gate | `ANG-GATE-DEPENDENCY-INTAKE-001` | stub |
| `ANG-BP-RECOVERED-CODE-INTAKE` | Same gate applied explicitly to recovered project material | `ANG-GATE-RECOVERED-INTAKE-001` | stub |
| `ANG-BP-MUTATION-SANDBOX` | Isolated build/run/evidence boundary for proposed code changes | `ANG-GATE-MUTATION-SANDBOX-001` | deferred until M9 design |
| `ANG-BP-EVOLUTION-GATE` | External security/regression/transfer approval with no self-deployment | `ANG-GATE-EVOLUTION-001` | deferred until M9 |

## Dependency and recovered-code intake

Adoption requires a demonstrated contract need, candidate comparison, exact commit/version and license, dependency/resource impact, trust analysis, tests, modification boundary, upgrade strategy, and exit/replacement plan. The owning capability branch requests adoption; SAFETY reviews supply chain and permissions. Recovered code follows the identical process and cannot define new-system interfaces retroactively.

## Dependencies and sequencing

THREAT-MODEL and HUMAN-AUTHORITY participate in slice 00. PERMISSIONS supports resource probes, environments, and runtime. DEPENDENCY/RECOVERED intake activates before any external code is vendored or modified. MUTATION-SANDBOX and EVOLUTION-GATE remain deferred until earlier gates are durable.

## Acceptance gate and evidence

`ANG-GATE-SAFETY-DESIGN-001` passes when every sensitive asset has an owner/trust zone, every promotion path has independent authority and rollback, permission ceilings are explicit, sealed data cannot flow to learners, dependency intake has an exit path, and the human-impact contract/gate is fully specified. Every milestone requires both a safety veto check and `ANG-GATE-HUMAN-FLOURISHING-001`.

## Testing and validation

- Permission denial and privilege-escalation tests.
- Human-impact completeness, independent-review, and conflict-escalation tests.
- Tests for deception, coercion, preference manipulation, addictive engagement, discriminatory sacrifice, oversight gaming, power/permission seeking, concealed capability, and shutdown resistance.
- Sealed-data visibility and credential isolation.
- Stop/abort/rollback behavior at each transaction stage.
- Resource exhaustion and cleanup.
- Dependency provenance, license, tamper, vulnerability, and replacement checks.
- Adversarial generated-code/sandbox tests when activated.
- Proof that learner outputs cannot directly trigger promotion/deployment.

## Risks and rollback

- `ANG-RISK-AUTHORITY-LEAK-001`: learner reaches promotion/deployment control. Stop system and rotate authority material.
- `ANG-RISK-SEALED-DATA-001`: evaluation data crosses trust boundary. Invalidate affected identities and rotate data.
- `ANG-RISK-SUPPLY-CHAIN-002`: imported/recovered dependency is unsafe or irreplaceable. Quarantine/remove via exit plan.
- `ANG-RISK-SELF-DEPLOY-001`: code proposal reaches production without external gate. Architectural stop condition.
- `ANG-RISK-SAFETY-THEATER-001`: controls exist only in documents. No milestone passes without executable evidence.
- `ANG-RISK-PATERNALISM-001`: preservation or happiness is used to override agency. Enforce informed choice, least-intrusive action, and human review.
- `ANG-RISK-AGGREGATE-SACRIFICE-001`: diffuse gains are used to justify concentrated severe harm. Apply equal worth and rights constraints; fail the gate.
- `ANG-RISK-MORAL-PARALYSIS-001`: uncertainty prevents all useful action. Permit reversible information gathering and low-exposure trials while pausing irreversible/high-impact action.

## Resource scaling

Permission and authority semantics do not weaken on larger systems. Distributed plans explicitly map credentials, nodes, network paths, storage, cleanup, and resource ceilings. More compute increases containment surface and evidence requirements; it never increases learner authority automatically.

## Current status and next leaf

Global SAFETY design remains draft, but THREAT-MODEL, HUMAN-AUTHORITY, PERMISSIONS, the bootstrap impact assessment, and the CR0 gate are approved/documented for supervised local scaffolding. This does not pass the human-flourishing, SAFETY, Slice-00, or M0 gates. Next activate only an exact CR0 leaf; expand DEPENDENCY-INTAKE before any package or external-code use.
