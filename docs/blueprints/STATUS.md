# Root blueprint roll-up

Freshness: 2026-08-25  
Root blueprint: `ANG-BP-ROOT` revision 2  
Current milestone: `ANG-M0-BLUEPRINT`
Active construction release: `ANG-CR-0001-CONSTRUCTION-RELEASE-0` (`BOOTSTRAP_WORK`, expires 2026-09-24)

## Program state

The EVIDENCE-SCHEMAS bootstrap leaf was executed and its independently owned decision records `SCAFFOLD_ACCEPTED`, SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`. This accepts only the exact local scaffold; the normal Evidence gate remains `NOT_RUN`. Resources activation `ANG-CR0-REVALIDATION-20260825-003` was `REJECTED` because mandatory continuation caches cited a stale baseline; its spec and decision remain immutable failure evidence. Active successor `ANG-CR0-REVALIDATION-20260825-004` is PENDING/NON-AUTHORIZING. Gate `ANG-GATE-CR0-RESOURCES-001@2`, leaf `ANG-WORK-CR0-RESOURCES-001@3`, and baseline `ANG-BASELINE-CR0-RESOURCES-002` SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE` are frozen for review. The leaf remains ready but unusable, and none of its nine outputs or tests may begin until independent approval and a separate manifest PASS/authorization transition. No model/GPU work, real probe, training, network/dependency use, recovered/real-person data, deployment, or promoted runtime mutation is authorized.

CR0 bootstrap authorization is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, a branch delivery gate, or a scientific claim. It exists only to build the controls needed to evaluate those gates.

The project owner's nonbinding intent to preserve the option for a future direct broad license to OpenAI is recorded in `PROJECT_STEWARDSHIP_INTENT.md`. It grants no present rights and requires a later IP/stewardship blueprint and formal legal instrument.

## Branch roll-up

| Branch | Design | Delivery | Blocking condition | Next gate |
|---|---|---|---|---|
| SCIENCE | approved for CR0 design | blocked by predecessors | CR0 schema work waits for SAFETY, pinned EVIDENCE scaffold decision, RESOURCES, and WORLDS scaffold receipts | `ANG-GATE-CR0-SCIENCE-001` after predecessors; normal design gate remains separate |
| RUNTIME | draft revision 2 | not_started | Model/state and human-impact authorization enforcement details not yet approved | `ANG-GATE-RUNTIME-DESIGN-001` |
| LEARNING | draft | not_started | Episode/state/update contracts not yet approved | `ANG-GATE-LEARNING-DESIGN-001` |
| EVIDENCE | approved for CR0 design revision 3 | CR0 scaffold accepted; normal delivery unrun | ARTIFACT-LINEAGE still waits for the normal technical gate | Scaffold decision is accepted; normal gate remains `NOT_RUN` |
| RESOURCES | approved for CR0 design revision 2 | leaf revision 3 ready, unusable while revalidation 004 is PENDING | `ANG-CR0-REVALIDATION-20260825-003` was `REJECTED`; successor 004 requires independent `APPROVED` plus Manifest v2 PASS/authorization; real probe delivery is successor-only | `ANG-GATE-CR0-RESOURCES-001@2` only after activation approval |
| WORLDS | approved for CR0 design revision 2 | blocked by SAFETY/EVIDENCE | Requires CR0 SAFETY receipt and pinned independent EVIDENCE scaffold decision; generators/verifiers remain unbuilt | `ANG-GATE-CR0-WORLDS-001` after predecessors |
| TOOLS | draft | deferred | Begins after causal adaptive core; design remains bounded | `ANG-GATE-TOOLS-DESIGN-001` |
| SAFETY | global draft; CR0 governance approved revision 2 | documented for CR0 | Ordinary authorization enforcement, dependency intake, OS sandbox, and high-impact human governance remain unbuilt | CR0 bootstrap gate passed; full SAFETY/human-flourishing gates open |

## Next program action

Independent reviewer `ANG-REVIEW-CODEX-SAFETY-CR0-REVALIDATION-004` inspects the exact PENDING successor packet and alone writes the reserved revalidation-004 decision. Do not execute the Resources leaf or test during review. If the decision is `APPROVED`, root may perform the separate final PASS/authorization transition, revalidate the frozen hashes, and only then execute `ANG-WORK-CR0-RESOURCES-001@3`. Keep the normal Evidence/Resource/Human-Flourishing gates, ARTIFACT-LINEAGE, Slice 00, M0, runtime, and learning unpassed or blocked.

## Rollback point

Release-wide baseline: `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`. Evidence baseline: `F05FA1F18183D089B53CFC1EA27139775C447709D05B6C8E303DC477E3329F8F`. Authoritative pending Resources absent-state baseline: `ANG-BASELINE-CR0-RESOURCES-002`, SHA-256 `A27DE8AA7D61F0915D2D925E5D384274EC4DD1F5DBF73A09F57C46AF5F9113DE`. Recovery artifacts under `outputs/` remain untouched.
