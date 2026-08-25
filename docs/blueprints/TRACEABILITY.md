# Root traceability matrix

This seed matrix identifies where the global claims must eventually be designed and proven. Contract, test, evidence, and gate entries marked `TBD` remain open work, not implied completion.

| Requirement or invariant | Primary branch | Principal contract | Required proof | Gate |
|---|---|---|---|---|
| `ANG-INV-HUMAN-FLOURISHING-001` | SAFETY with root authority | `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` | Rights-respecting impact review; tests for severe harm, coercion, deception, discriminatory sacrifice, power seeking, and shutdown resistance; independent human decision | `ANG-GATE-HUMAN-FLOURISHING-001` at every milestone/promotion |
| `ANG-REQ-HUMAN-PRESERVATION-001` | SAFETY | `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` | Direct/indirect harm, non-user, in-scope inaction, catastrophic-risk, and shutdown tests | `ANG-GATE-HUMAN-FLOURISHING-001` |
| `ANG-REQ-VOLUNTARY-FLOURISHING-001` | SAFETY | `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` | Consent, autonomy, deception, coercion, addictive-manipulation, and preference-manipulation tests | `ANG-GATE-HUMAN-FLOURISHING-001` |
| `ANG-REQ-EQUITABLE-BETTERMENT-001` | SAFETY | `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` | Distribution, vulnerable-group, minority-sacrifice, access, and future-effect review | `ANG-GATE-HUMAN-FLOURISHING-001` |
| `ANG-INV-STABLE-BASE-001` | RUNTIME | `ANG-CTR-PLASTIC-STATE-001` | Base signature unchanged across accepted transactions | `ANG-GATE-RUNTIME-DESIGN-001`, then M2 |
| `ANG-INV-ONE-COMPETENCE-001` | RUNTIME | `ANG-CTR-PLASTIC-STATE-001` | No query-conditioned model/adapter selection; one parent lineage | M2 and M3 |
| `ANG-INV-EVIDENCE-SEPARATION-001` | EVIDENCE | `ANG-CTR-EVIDENCE-ENVELOPE-001`, `ANG-CTR-EPISODE-001`, `ANG-CTR-ARTIFACT-LINEAGE-001` | Visibility/eligibility denials plus later probe success without training-episode retrieval | EVIDENCE schema/lineage gates, then M3 |
| `ANG-INV-CAUSAL-PROMOTION-001` | SCIENCE | `ANG-CTR-EVALUATION-RECEIPT-001` | zero/swap/permutation/replay plus fair-RAG controls | M3 |
| `ANG-INV-REVERSIBLE-UPDATES-001` | RUNTIME | `ANG-CTR-TRANSACTION-RECEIPT-001` | exact restore after rejection and interruption | M2/M3 |
| `ANG-INV-OUTCOME-JUDGES-001` | WORLDS | `ANG-CTR-TASK-SPEC-001`, `ANG-CTR-OBSERVATION-001`, `ANG-CTR-FEEDBACK-001` | outcome-only schema/negative controls, then verifier leakage/adversarial inspection | environment-protocol gate, then M1/M3 |
| `ANG-INV-ELASTIC-COMPUTE-001` | RESOURCES | `ANG-CTR-RESOURCE-INVENTORY-001`, `ANG-CTR-EXECUTION-PLAN-001` | synthetic constrained/workstation/server/cluster feasibility first; measured current plus simulated smaller/larger plans later | resource design gate, then M1 and M8 |
| `ANG-INV-EXTERNAL-AUTHORITY-001` | SAFETY | `ANG-CTR-PROMOTION-DECISION-001` | learner cannot alter gate/evidence/deploy authority | every milestone |
| `ANG-INV-FRESH-SYSTEM-001` | SAFETY | dependency intake contract TBD | imported code tied to demonstrated need and exit plan | M0 and each adoption |

## Construction Release 0 authorization trace

```text
project-owner instruction
→ ANG-ADR-0002
→ ANG-POL-LOCAL-SCAFFOLD-001
→ ANG-ASSESS-CONSTRUCTION-RELEASE-0-001 (LOW / BOOTSTRAP_WORK / ALLOW)
→ ANG-GATE-CONSTRUCTION-RELEASE-0-001
→ ANG-CR-0001-CONSTRUCTION-RELEASE-0
→ ANG-WORK-EVIDENCE-SCHEMAS-001
→ ANG-BASELINE-EVIDENCE-SCHEMAS-001
→ ANG-TEST-CR0-EVIDENCE-SCAFFOLD-001 + effect receipt
→ ANG-AUTH-VALIDATOR-001 independent verification
→ ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001 (CR0-only disposition)
```

This chain authorizes only the exact local synthetic construction leaf. `SCAFFOLD_ACCEPTED` may unlock only exact manifest-listed CR0 scaffold consumers and is deliberately not a pass for `ANG-GATE-EVIDENCE-SCHEMAS-001`, `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a normal child delivery gate, or any scientific claim. Any model/GPU, network, package/dependency, recovered output, real-person data, out-of-scope path, background process, deployment, or external effect requires a successor authorization.

## Completion rule

A row is complete only when it names a versioned design element, contract, test, immutable evidence identity, and passed gate. A persuasive explanation without that chain remains a hypothesis.
