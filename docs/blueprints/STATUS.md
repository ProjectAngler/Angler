# Root blueprint roll-up

Freshness: 2026-08-25  
Root blueprint: `ANG-BP-ROOT` revision 2  
Current milestone: `ANG-M0-BLUEPRINT`
Active construction release: `ANG-CR-0001-CONSTRUCTION-RELEASE-0` (`BOOTSTRAP_WORK`, expires 2026-09-24)

## Program state

The Tier-0 project draft and Human-Flourishing Constitution are established. The responsibility tree, recursive protocol, interface/dependency spine, and initial CR0 child designs are registered. A single EVIDENCE-SCHEMAS Tier-4 leaf is authorized for local, synthetic, standard-library-only control-plane construction under ADR-0002, the CR0 policy/assessment/gate, and its exact absent-state baseline. No model/GPU work, training, network/dependency use, recovered/real-person data, deployment, or promoted runtime mutation is authorized.

CR0 bootstrap authorization is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a branch delivery gate, or a scientific claim. It exists only to build the controls needed to evaluate those gates.

The project owner's nonbinding intent to preserve the option for a future direct broad license to OpenAI is recorded in `PROJECT_STEWARDSHIP_INTENT.md`. It grants no present rights and requires a later IP/stewardship blueprint and formal legal instrument.

## Branch roll-up

| Branch | Design | Delivery | Blocking condition | Next gate |
|---|---|---|---|---|
| SCIENCE | approved for CR0 design | blocked by predecessors | CR0 schema work waits for SAFETY, pinned EVIDENCE scaffold decision, RESOURCES, and WORLDS scaffold receipts | `ANG-GATE-CR0-SCIENCE-001` after predecessors; normal design gate remains separate |
| RUNTIME | draft revision 2 | not_started | Model/state and human-impact authorization enforcement details not yet approved | `ANG-GATE-RUNTIME-DESIGN-001` |
| LEARNING | draft | not_started | Episode/state/update contracts not yet approved | `ANG-GATE-LEARNING-DESIGN-001` |
| EVIDENCE | approved for CR0 design revision 3 | first scaffold leaf ready | CR0 consumers wait for scaffold acceptance; ARTIFACT-LINEAGE waits for the normal technical gate | `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`; normal gate remains not run |
| RESOURCES | approved for CR0 design revision 2 | blocked by EVIDENCE scaffold decision | Requires exact independent `SCAFFOLD_ACCEPTED` receipt; real probe delivery is successor-only | `ANG-GATE-CR0-RESOURCES-001` after scaffold receipt |
| WORLDS | approved for CR0 design revision 2 | blocked by SAFETY/EVIDENCE | Requires CR0 SAFETY receipt and pinned independent EVIDENCE scaffold decision; generators/verifiers remain unbuilt | `ANG-GATE-CR0-WORLDS-001` after predecessors |
| TOOLS | draft | deferred | Begins after causal adaptive core; design remains bounded | `ANG-GATE-TOOLS-DESIGN-001` |
| SAFETY | global draft; CR0 governance approved revision 2 | documented for CR0 | Ordinary authorization enforcement, dependency intake, OS sandbox, and high-impact human governance remain unbuilt | CR0 bootstrap gate passed; full SAFETY/human-flourishing gates open |

## Next program action

Execute only `ANG-WORK-EVIDENCE-SCHEMAS-001` under the exact CR0 manifest. Preserve its receipts, run the independent release/tree validators, and keep every dependent leaf blocked until `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` records `SCAFFOLD_ACCEPTED`. Then activate RESOURCES and recheck/bind the SAFETY leaf before WORLDS; continue WORLDS → SCIENCE → integration only in manifest order. Keep the normal EVIDENCE technical gate, ARTIFACT-LINEAGE, runtime, and learning blocked pending ordinary or successor-release authority.

## Rollback point

Release-wide baseline: `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`. First-leaf absent-state baseline: `ANG-BASELINE-EVIDENCE-SCHEMAS-001`, SHA-256 `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F`. Recovery artifacts under `outputs/` remain untouched and excluded from the new repository lineage.
