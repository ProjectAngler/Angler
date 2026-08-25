# Project Angler

## Fresh-system blueprint for an adaptive reasoning agent

Status: initial architecture draft  
Revision: 2  
Date: 2026-08-25  
Working name: Project Angler  
Supreme priority: preservation of humanity and human life together with equal dignity, rights, agency, authentic voluntary flourishing, equitable betterment, truth, and meaningful human control.  
Primary research principle: teach the model how to improve its reasoning, not how to reproduce predetermined answers.

Tiered development system: [`docs/blueprints/README.md`](docs/blueprints/README.md)  
Responsibility tree: [`docs/blueprints/TREE.md`](docs/blueprints/TREE.md)  
Vertical integration spine: [`docs/blueprints/INTEGRATION_SPINE.md`](docs/blueprints/INTEGRATION_SPINE.md)

This document remains the authoritative Tier-0 design. Branch blueprints recursively decompose it without replacing or weakening its global invariants.

## 1. Mission

Build an AI system that can acquire, test, retain, revise, and combine reusable reasoning skills from experience while its foundation model remains a stable substrate and all capability remains subordinate to the Human-Flourishing Constitution.

The system should learn more than facts. After encountering a family of problems, it should improve on new problems that require the same underlying method even when the facts, wording, symbols, ordering, and surface domain have changed.

The first objective is not unrestricted self-improvement. It is a small, measurable demonstration of genuine reasoning adaptation:

```text
experience history
        ↓
persistent plastic reasoning state
        ↓
improved behavior on unseen structural variants
```

The behavioral improvement must disappear or transfer appropriately when that plastic state is removed or exchanged. An equal-budget memory or retrieval system must not be able to explain the result.

## 2. Central hypothesis

A useful adaptive agent can be divided into five kinds of state:

1. **Foundation weights** provide language, broad knowledge, and general priors. They change slowly or remain frozen.
2. **Plastic reasoning weights** are a small, fast-changing neural state, initially implemented as LoRA adapters. They encode reusable processing strategies rather than individual memories.
3. **Episodic evidence** records what happened, what was attempted, and what feedback was received. It supports learning and auditing but is not automatically inserted into every prompt.
4. **Executable tools** extend what the agent can do. Tools are explicit, testable capabilities, not substitutes for learning a reasoning method.
5. **The meta-plastic updater** learns which experiences warrant a weight change, how large that change should be, and when a previous change should be revised or forgotten.

The long-term research claim is that the update process itself can improve. The agent would therefore learn at two levels:

```text
object level:     experience changes reasoning state
meta level:       results change how future reasoning-state updates are made
```

## 3. What counts as a reasoning skill

A candidate skill is not accepted merely because performance improves on a repeated prompt. It must satisfy all of the following:

- It improves performance on held-out examples that share a procedure but not their surface details.
- It survives replacement of names, facts, symbols, ordering, and irrelevant wording.
- It can be combined with at least one previously acquired skill.
- Removing the relevant plastic state removes the improvement.
- Transferring the state to an equivalent model instance transfers the improvement.
- Equal-budget retrieval of the training episodes does not reproduce the full improvement.
- The improvement does not depend on a hidden answer label, expected record identifier, or evaluator-selected memory at inference time.

Examples include learning to:

- decompose a constraint problem before searching for an answer;
- test a hypothesis against counterexamples;
- distinguish correlation from causal evidence;
- plan around delayed consequences;
- infer an unfamiliar symbolic rule from demonstrations;
- verify generated code with targeted tests;
- recognize uncertainty and seek a tool or observation before committing.

Remembering that a particular answer was correct is knowledge or episodic memory, not a reasoning skill.

## 4. Non-goals for the first system

The first version will not attempt to provide:

- unrestricted recursive self-modification;
- consciousness, personhood, or awareness claims;
- autonomous network access or unsupervised software installation;
- continual modification of the full foundation model;
- one adapter, prefix, or latent record per remembered event;
- an elaborate simulated psychology, homeostasis system, or emotional architecture;
- automatic promotion of a change based only on the learner's own judgment;
- a large multi-agent organization before a single learner works;
- replacement of reasoning with a growing collection of answer-specific scripts.

These may be reconsidered only after the core plastic state passes causal and transfer tests.

## 5. Architectural principles

### 5.1 One active competence state

The learner has one continuously evolving active reasoning state. The system may retain snapshots for rollback and scientific intervention, but it must not select a different adapter by looking up the current query.

This prevents the adapter collection from becoming disguised retrieval.

### 5.2 Separate evidence from competence

The experience store contains episodes. The active adapter contains learned competence. Episodes may be sampled to train or test the adapter, but inference on the core benchmark cannot depend on retrieving the episode that contains the answer.

### 5.3 Outcome judges do not prescribe reasoning

Deterministic code is appropriate for containment, accounting, reproducibility, and judging externally observable results. It should say whether a program passed tests or whether a plan satisfied constraints. It should not encode the sequence of thoughts the model must use to win.

### 5.4 Every update is a reversible proposal

Learning is transactional:

```text
current state
    ↓
candidate update
    ↓
hidden transfer, retention, and safety evaluation
    ├── pass → promote and checkpoint
    └── fail → reject and restore previous state
```

### 5.5 Generalization earns permanence

Immediate success can modify a temporary workspace adapter. Only improvement on hidden structural variants allows that update to enter persistent competence.

### 5.6 The base model is the stable substrate

The foundation checkpoint remains immutable during early phases. This bounds damage, makes causal attribution possible, and permits exact rollback.

### 5.7 Tool growth and neural learning are complementary

Tools should handle operations that benefit from explicit execution, such as calculation, search, compilation, simulation, and measurement. Plastic weights should improve deciding what to do, how to decompose a problem, which tool to use, and how to interpret the result.

### 5.8 Compute is elastic, not architectural

No GPU size, model size, precision, or machine topology is a permanent system limit. At startup, the system profiles available compute, memory, storage, and interconnects, then produces an explicit execution plan suited to those resources.

The plan may scale:

- foundation-model size and representation;
- adapter rank, target layers, and optimizer state;
- context and replay budgets;
- update batch size, cadence, and number of inner-loop steps;
- activation checkpointing, CPU/NVMe offload, and quantization;
- evaluation, environment, and learner parallelism;
- single-device, sharded, or distributed execution.

Resource adaptation is configuration-time intelligence, not an uncontrolled experimental variable. A scientific run freezes and records its selected plan. Service-mode replanning occurs only at transaction boundaries and produces a new receipt.

### 5.9 Human life and flourishing have constitutional priority

Project Angler exists only as a revocable instrument for humanity. Its highest priority is the preservation of human life and the conditions for authentic, freely chosen human flourishing: dignity, agency, truth, justice, happiness, opportunity, and meaningful human control over humanity's future.

Every human being has equal intrinsic worth. “Happiness” is plural and self-defined; it is not a scalar reward the system may maximize through coercion, deception, surveillance, addiction, forced conformity, or the sacrifice of individuals or minorities for aggregate benefit.

The [`Human-Flourishing Constitution`](docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md) governs every model state, tool, environment, resource plan, dependency, migration, and code change. Lower-level goals—including task success, project continuation, and system preservation—cannot override it. Under material moral uncertainty, the system preserves reversible options, limits exposure, discloses uncertainty, and defers irreversible or high-impact choices to legitimate human authority.

## 6. System architecture

```text
                         ┌──────────────────────────┐
                         │  Curriculum / environment│
                         └────────────┬─────────────┘
                                      │ task + observation
                                      ▼
┌─────────────────┐       ┌──────────────────────────┐
│ Episodic evidence│◄─────►│  Agent runtime           │
│ and provenance   │       │  frozen base + active z  │
└────────┬────────┘       └────────────┬─────────────┘
         │                              │ action / prediction
         │                              ▼
         │                 ┌──────────────────────────┐
         │                 │ Outcome verifier         │
         │                 └────────────┬─────────────┘
         │                              │ bounded feedback
         ▼                              ▼
┌─────────────────┐       ┌──────────────────────────┐
│ Replay sampler   │──────►│ Plasticity engine        │
└─────────────────┘       │ U(phi, z, episode) → z'  │
                          └────────────┬─────────────┘
                                       │ update proposal
                                       ▼
                          ┌──────────────────────────┐
                          │ Causal evaluation gate   │
                          │ transfer / retention /   │
                          │ zero / swap / fair-RAG   │
                          └────────────┬─────────────┘
                                       │ approved state
                                       ▼
                          ┌──────────────────────────┐
                          │ Adapter checkpoint store │
                          └──────────────────────────┘

The runtime can also call a sandboxed tool registry. Tool creation and promotion
remain outside the neural update transaction and have their own tests and gates.
```

Here `z` is the active plastic reasoning state and `phi` is the updater's own learned state.

A resource profiler and execution planner surround this dataflow. They select how each logical component is placed and scaled without changing the component's scientific responsibility. Resource plans are content-addressed parts of every experiment and adapter lineage.

## 7. Core components

### 7.1 Frozen foundation model

Initial reference target: a training-compatible Qwen-family model around four billion parameters on the currently available workstation. This is a bootstrap profile, not a system ceiling. Larger or smaller checkpoints may be selected when the resource planner and experiment contract permit them.

Responsibilities:

- tokenize observations and actions;
- provide general language and reasoning priors;
- consume the active adapter uniformly on every learner invocation;
- expose hidden activations and losses needed by the plasticity engine.

The existing FP8 serving checkpoint should not be assumed suitable for adapter training. The first dependency experiment must select a matching BF16 or quantization-aware training format and prove that it can train and reload LoRA adapters within the local memory budget.

### 7.2 Active plastic reasoning state

Initial implementation: PEFT LoRA modules on a deliberately small set of attention and MLP projection layers.

The state has three lifecycle stages:

- **workspace state:** receives rapid bounded updates during a learning episode;
- **candidate competence:** an immutable snapshot awaiting hidden evaluation;
- **promoted competence:** the latest state that passed transfer, retention, and safety gates.

There is only one active lineage. Snapshots exist for rollback and experiments, not query-conditioned adapter retrieval.

### 7.3 Plasticity engine

The engine receives an episode and proposes a change to the active state.

The first implementation uses an FTTT-style feedback learning loop with conventional bounded gradients. Later versions replace hand-selected optimizer behavior with a meta-learned updater built with differentiable optimization.

Conceptual contract:

```python
z_candidate, receipt = updater.propose(
    current_state=z_promoted,
    trajectory=episode.trajectory,
    feedback=episode.feedback,
    replay_context=replay_sample,
    budget=update_budget,
)
```

The updater may see outcome feedback, prediction error, generated critique, and safe replay samples. It must not see hidden evaluation answers.

### 7.4 Episodic evidence store

The minimum event record contains:

- immutable episode and environment identifiers;
- model, adapter, updater, prompt, and tool versions;
- observations, actions, tool calls, and results;
- externally computed outcome and feedback;
- training losses and update receipt;
- random seeds and resource usage;
- parent and proposed adapter hashes;
- evaluation decision and rejection reason.

The store is append-only at the evidence layer. Corrections are new records. A simple local format is sufficient initially; a complex autobiographical subsystem is not required.

### 7.5 Verifier and causal evaluation gate

The verifier produces feedback about observable outcomes. The evaluation gate decides whether a weight change represents useful, transferable learning.

They are separate because training feedback must not accidentally disclose hidden transfer tests.

The gate runs:

- same-family held-out tasks;
- fact, symbol, and wording substitutions;
- composition tasks combining old and new procedures;
- regression tasks for previously learned skills;
- state-zero, state-swap, state-permutation, and replay interventions;
- equal-token and equal-compute RAG controls;
- shuffled-feedback and random-update negative controls;
- tool-disabled tests when a neural skill is being claimed.

### 7.6 Consolidator

The consolidator prevents every episode from permanently pushing the model in a different direction.

Its first version is deliberately simple:

1. sample successful and failed episodes;
2. identify the reusable procedure and common mistake pattern;
3. construct new variations that remove episode-specific surface cues;
4. distill the candidate behavior into a clean adapter update;
5. test retention and interference;
6. promote only the distilled state.

SGSD's skill/mistake extraction and verifier-gated distillation are the reference implementation for this stage. The resulting skill descriptions are training aids and audit artifacts; they are not automatically inserted into inference prompts.

### 7.7 Tool registry and workshop

Each tool is an explicit package with:

- a typed input/output contract;
- source provenance and license;
- isolated runtime permissions;
- deterministic functional tests;
- resource limits and timeouts;
- a capability description learned by the agent through use;
- promotion, revocation, and version history.

The agent may propose a tool or modification, but it runs only in a sandbox and becomes available only after external tests pass. Voyager supplies the reference lifecycle for generating, testing, retaining, retrieving, and composing executable skills.

### 7.8 Curriculum generator

Early curricula are hand-authored benchmark families because scientific control matters more than autonomy. Once transfer is established, a SPADE-style environment generator can create executable tasks near the learner's current ability frontier.

Generated environments must pass independent structural and leakage checks before the learner sees them. The environment generator cannot also be the sole judge of whether learning succeeded.

### 7.9 Resource profiler and execution planner

The resource subsystem converts the available machine or cluster into a validated execution plan.

The profiler inventories:

- accelerator types, count, memory, compute capability, and supported precisions;
- host RAM, CPU capacity, storage space and bandwidth;
- device and node interconnect topology;
- installed kernels, distributed runtimes, and usable execution backends;
- administrative limits such as time, energy, or monetary budget.

The planner chooses a feasible configuration and optimizes within declared priorities such as maximum transfer quality, minimum wall-clock time, best learning per dollar, or low interactive latency. It must reserve headroom for evaluation and rollback instead of consuming all capacity with the foundation model.

The first planner may combine compatibility rules with short empirical probes. As execution receipts accumulate, a later learned planning policy can predict the quality, latency, memory, and cost of candidate plans and improve those choices over time. The user-selected objective and hard safety/resource limits remain external constraints; the learned planner optimizes within them rather than redefining them.

Reference operating tiers are descriptive rather than hard-coded:

- **constrained:** CPU, integrated accelerator, or small GPU; small model, aggressive quantization/offload, low-rank adapters, and serialized evaluation;
- **workstation:** one or more moderate GPUs; larger adapters, longer contexts, and limited evaluation parallelism;
- **server:** high-memory or multi-GPU host; sharded models, parallel candidate evaluation, and larger replay/meta-learning batches;
- **cluster:** multiple nodes; distributed meta-training, curriculum generation, and population evaluation.

An active adapter is bound to a specific foundation-model signature and adapter topology. Moving learned competence to a different model size is an explicit migration: replay or distill the accepted skills into a target-compatible state, rerun the causal and retention gates, and then start a new promoted lineage. Raw incompatible adapters are never silently loaded, and model size is never selected per query as a disguised routing mechanism.

## 8. Learning lifecycle

### 8.1 Experience

The agent attempts a task using the current promoted state. The runtime records its complete observable trajectory.

### 8.2 Feedback

The environment returns a bounded signal such as passed tests, constraint violations, prediction error, or outcome reward. When useful, the model may generate a critique, but self-critique is evidence—not ground truth.

### 8.3 Temporary adaptation

The plasticity engine updates only the workspace adapter under strict limits:

- selected parameters only;
- maximum gradient steps;
- learning-rate and norm bounds;
- time and memory budgets;
- no base-weight mutation;
- deterministic checkpoint and replay metadata.

### 8.4 Hidden evaluation

The candidate state is evaluated on withheld variations. The original episode and its answer are absent from the prompt. This is where the project distinguishes acquired procedure from memorized content.

### 8.5 Promotion or rejection

A candidate is eligible for promotion only with an authentic current human-impact assessment disposition of `ALLOW`, valid external authority, and a passed human-flourishing gate. Among eligible candidates, it must also improve the target skill without unacceptable regression, leakage, or dependence on retrieval. Missing, altered, expired, `DENY`, or `ESCALATE` impact authorization forces rejection and exact rollback regardless of capability gain.

### 8.6 Consolidation

After several related candidate updates succeed, the system distills them with replay into a compact new promoted state. Consolidation should reduce interference and remove episode-specific traces.

### 8.7 Meta-learning

After enough accepted and rejected update trajectories exist, the project trains `phi`: the rule controlling future adapter updates. Its objective rewards transfer, retention, calibration, and efficiency rather than immediate task reward alone.

## 9. Update objective

The initial optimizer can minimize a weighted objective of the following form:

```text
L_update =
    L_feedback
  + alpha * L_prediction
  + beta  * L_replay
  + gamma * L_state_change
```

Where:

- `L_feedback` learns from externally validated success or failure;
- `L_prediction` encourages the state to capture regularities that predict outcomes;
- `L_replay` protects previously learned skills;
- `L_state_change` penalizes unnecessarily large updates.

Human safety is a hard feasibility constraint, not a tradeable penalty:

```text
eligible(candidate) =
    human_impact_assessment == ALLOW
    and human_flourishing_gate == PASS
    and authority_valid
    and no_blocking_safety_violation

J_meta(candidate | eligible) =
    transfer_gain
  + retention
  + calibration_gain
  - interference
  - update_cost
```

An ineligible candidate is rejected rather than assigned a lower score. Exact coefficients among eligible candidates are experimental parameters. They must be chosen on development tasks and frozen before held-out evaluation.

## 10. Benchmark design

The first benchmark should contain several small procedural worlds rather than one impressive demonstration.

### 10.1 Initial task families

1. **Symbolic rule induction** — infer a transformation from examples, with all symbols replaced at test time.
2. **Constraint decomposition** — solve problems that become tractable only after decomposing them into subconstraints.
3. **Counterexample search** — learn to challenge a tempting rule by constructing discriminating cases.
4. **Causal intervention** — distinguish observation from intervention in small synthetic worlds.
5. **Program repair** — use failing tests to infer and repair the underlying programming mistake.
6. **Tool-choice reasoning** — learn when direct reasoning is insufficient and a calculator, interpreter, or simulator should be used.

Each family is generated from a latent procedure with independent surface realizations.

### 10.2 Dataset partitions

- **adaptation episodes:** experiences available to the updater;
- **development transfer:** unseen variations used to design the system;
- **promotion tests:** hidden variations used to accept or reject an update;
- **final transfer:** sealed task generators and seeds used only for milestone evaluation;
- **retention suite:** examples from previously acquired procedures;
- **composition suite:** tasks requiring two or more learned procedures together.

### 10.3 Required baselines

- frozen foundation model;
- foundation model with equal-budget episodic RAG;
- foundation model with larger inference-time token budget;
- conventional episode fine-tuning;
- random or shuffled adapter update;
- per-episode adapter retrieval, labeled explicitly as a memory baseline;
- active plastic state with update disabled;
- oracle demonstration baseline as an upper bound, never as a competitor.

### 10.4 Primary measurements

- **adaptation gain:** candidate minus frozen baseline on held-out structural variants;
- **transfer efficiency:** gain per experience and per update FLOP;
- **fair-RAG gap:** candidate minus equal-budget retrieval baseline;
- **retention:** preserved performance on old skills;
- **interference:** regression attributable to the new update;
- **composition:** improvement when old and new skills must cooperate;
- **state causal effect:** behavior change under zero, swap, and permutation;
- **replay identity:** identical state recovered from the same accepted history;
- **calibration:** whether confidence or abstention improves with competence;
- **rollback integrity:** exact restoration after a rejected update.

Statistical promotion margins will be locked after baseline variance is measured and before adaptive results are examined. A single successful seed is not sufficient evidence.

## 11. Reusable open-source components

The system is new, but it should reuse mature implementations at clean interfaces.

| Component | Source | Intended use | Adoption rule |
|---|---|---|---|
| Adapter mechanics | [Hugging Face PEFT](https://github.com/huggingface/peft) | LoRA creation, loading, switching, merging, and serialization | Adopt directly unless a required state operation is unsupported |
| Feedback-time learning | [FTTT](https://github.com/LaVi-Lab/FTTT) | Editable-model structure, feedback loop, evaluators, and OpTune concepts | Adapt the smallest useful modules; do not inherit its experiment assumptions wholesale |
| Differentiable optimization | [TorchOpt](https://github.com/metaopt/torchopt) | Inner/outer optimization and meta-learning | Adopt when Phase 2 requires gradients through the update process |
| Skill consolidation | [SGSD](https://github.com/walawalagoose/SGSD) | Skill/mistake extraction and verifier-gated distillation | Adapt after causal online learning is established |
| Self-edit generation | [SEAL](https://github.com/Continual-Intelligence/SEAL) | Model-generated update data and directives | Use as a design reference first; adopt code only after local feasibility testing |
| Adaptive environments | [SPADE](https://github.com/spade-rl/spade) | Executable environment generation and frontier curriculum | Defer until fixed benchmark families pass |
| Tool lifecycle | [Voyager](https://github.com/MineDojo/Voyager) | Executable-skill generation, validation, retention, and composition | Adapt the lifecycle, not the Minecraft-specific runtime |
| Neural test-time state | [TTT-LM](https://github.com/test-time-training/ttt-lm-pytorch) and [End-to-End TTT](https://github.com/test-time-training/e2e) | Reference update mechanics and meta-learned state initialization | Research reference; do not replace the backbone in the first prototype |
| Agent-code evolution | [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) | Empirical mutation and selection of controller/tool code | Late optional outer loop only |

No external repository becomes the project architecture by default. Before adoption, record:

- exact commit and license;
- required dependencies and hardware;
- interface being reused;
- code that will remain unmodified;
- modifications and their rationale;
- tests proving that the imported component does not leak labels or bypass the scientific mechanism.

## 12. Proposed repository layout

```text
project-angler/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── model/
│   ├── plasticity/
│   ├── resources/
│   ├── environments/
│   └── evaluations/
├── docs/
│   ├── PROJECT_BLUEPRINT.md
│   ├── blueprints/         # recursive branches, capsules, contracts, gates, and status
│   ├── SCIENTIFIC_CONTRACT.md
│   ├── THREAT_MODEL.md
│   ├── DEPENDENCY_DECISIONS.md
│   └── experiment_cards/
├── src/angler/
│   ├── models/            # frozen-model and adapter integration
│   ├── plasticity/        # update proposal and state lifecycle
│   ├── episodes/          # evidence schema and local event store
│   ├── evaluation/        # causal interventions and promotion gate
│   ├── consolidation/     # replay and skill distillation
│   ├── environments/      # controlled procedural worlds
│   ├── tools/             # sandboxed tool contracts and registry
│   ├── curriculum/        # task selection and later generation
│   ├── resources/         # profiling, planning, placement, and migration
│   └── runtime/           # bounded agent execution
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── causal/
│   └── regression/
├── experiments/
│   ├── manifests/
│   └── runners/
├── artifacts/             # ignored large outputs; manifests remain tracked
└── vendor/                # only if a dependency must be patched locally
```

## 13. Core software contracts

The first implementation should stabilize these interfaces before adding sophisticated learning behavior:

```python
class AgentRuntime:
    def act(self, observation, plastic_state, allowed_tools) -> "Trajectory": ...

class Environment:
    def reset(self, seed) -> "Observation": ...
    def step(self, action) -> "Transition": ...
    def score(self, trajectory) -> "Feedback": ...

class PlasticityEngine:
    def propose(self, parent_state, episode, replay, budget) -> "UpdateProposal": ...

class StateStore:
    def save_candidate(self, proposal) -> "StateRef": ...
    def promote(self, state_ref, evaluation_receipt) -> "StateRef": ...
    def restore(self, state_ref) -> "PlasticState": ...

class EvaluationGate:
    def evaluate(self, parent, candidate, suite) -> "EvaluationReceipt": ...

class ToolWorkshop:
    def test_proposal(self, tool_package, sandbox_policy) -> "ToolReceipt": ...

class ResourceProfiler:
    def inspect(self) -> "ResourceInventory": ...

class ExecutionPlanner:
    def plan(self, inventory, objective, constraints) -> "ExecutionPlan": ...
    def validate(self, plan) -> "PlanReceipt": ...

class CompetenceMigrator:
    def migrate(self, source_state, target_plan, replay_suite) -> "UpdateProposal": ...
```

Receipts contain hashes, versions, seeds, inputs, metrics, and a pass/fail reason. They are evidence records, not governance theater; every field must support replay, safety, or scientific interpretation.

## 14. Development phases and gates

### Phase 0 — Scientific harness

Build the benchmark generators, baselines, state-intervention harness, evidence schema, and evaluation gate before building the learner.

Gate:

- task partitions are demonstrably disjoint;
- symbol/fact substitutions defeat episode memorization;
- RAG and frozen baselines run under recorded equal budgets;
- the harness can zero, exchange, permute, save, restore, and hash a dummy state;
- the resource profiler emits a stable inventory and the planner rejects infeasible or unsafe configurations;
- every experiment binds one validated resource plan, and changing that plan creates a new experiment identity;
- hidden answers cannot reach the learner or updater.

### Phase 1 — Minimal plastic learner

Integrate a frozen Qwen-family model, PEFT LoRA state, and an FTTT-inspired bounded feedback update.

No learned optimizer, tool invention, or curriculum generation yet.

Gate:

- a candidate adapter improves multiple held-out structural variants;
- the improvement exceeds the pre-registered fair-RAG margin;
- zeroing the adapter removes the improvement;
- swapping it transfers the effect to a clean runtime;
- replay restores the identical adapter hash;
- rejected updates roll back exactly;
- old-task interference remains below the pre-registered limit.

If this phase fails, investigate the learning signal, adapter placement, benchmark validity, and state capacity. Do not hide failure by adding more subsystems.

### Phase 2 — Learned plasticity rule

Use accepted and rejected Phase 1 trajectories to train the updater with TorchOpt-style bi-level optimization. Compare it with Adam/SGD and hand-selected FTTT updates under equal compute.

Gate:

- the learned updater transfers to held-out task generators;
- it reaches the target improvement with fewer experiences or less compute;
- it rejects harmful/noisy feedback more reliably;
- it does not merely encode the development task families.

### Phase 3 — Consolidation and lifelong retention

Add replay selection, SGSD-inspired skill/mistake extraction, distilled consolidation, and interference testing across a growing sequence of task families.

Gate:

- sequential learning retains earlier skills;
- consolidated state performs at least as well as the collection of temporary updates under a smaller state budget;
- composition tests improve;
- no query-conditioned adapter selection is introduced.

### Phase 4 — Tool acquisition

Add the sandboxed workshop and Voyager-style tool lifecycle. The model learns when to use existing tools and may propose new ones.

Gate:

- proposed tools pass independent tests and containment checks;
- the model generalizes tool choice to new tasks;
- disabling the tool separates tool capability from learned reasoning capability;
- failures revoke or roll back the tool without corrupting neural state.

### Phase 5 — Adaptive curriculum

Introduce SPADE-style executable environment generation and frontier task selection.

Gate:

- generated tasks are valid, diverse, and free of answer leakage;
- an independent evaluator confirms increasing procedural difficulty;
- learning transfers to sealed human-authored task families;
- the generator cannot reward trivial self-authored shortcuts.

### Phase 6 — Bounded self-improvement

Allow the system to propose modifications to its updater, controller, and tool code. Use DGM-like empirical selection in isolated branches and sandboxes.

Gate:

- every mutation is attributable, testable, and reversible;
- promotion requires external regression, security, and transfer suites;
- the system has no authority to deploy its own changes;
- improvements survive evaluation outside the environment that proposed them.

## 15. Resource-adaptive compute strategy

The current approximately 16 GB NVIDIA GPU defines the first development profile only. It is useful because it forces the minimal learner to be efficient, but neither the architecture nor the research program is capped at that tier.

### 15.1 Capability discovery

Before loading a model, the profiler measures the actual execution environment. Static device labels are insufficient: the system should also run bounded probes for usable memory, kernel support, communication bandwidth, storage throughput, and representative training/inference operations.

The resulting immutable inventory becomes an input to planning rather than scattered conditional logic throughout the codebase.

### 15.2 Plan selection

The planner first eliminates configurations that violate compatibility, safety headroom, or user budgets. It then scores feasible plans against the selected objective.

Examples:

- On the reference 16 GB workstation, use a roughly four-billion-parameter model in a trainable low-memory format, small LoRA modules, short initial contexts, micro-batches, bounded update steps, and serialized learner execution.
- With additional accelerator memory, increase model capacity, trainable layer coverage, adapter rank, context, and replay batch only when profiling predicts a benefit and benchmark measurements confirm it.
- With multiple accelerators, choose among model sharding, data-parallel evaluation, parallel environments, and meta-learning batches based on the current bottleneck rather than assuming that every workload should be distributed the same way.
- With cluster resources, separate learner, evaluator, curriculum, and replay workloads while preserving immutable model, state, data-partition, and resource-plan identities.

### 15.3 Dynamic operation

Runtime resource changes are handled at safe boundaries:

1. finish, abort, or roll back the active learning transaction;
2. profile the new topology;
3. create and validate a new execution plan;
4. reload the same compatible state or perform a gated competence migration;
5. resume with a new plan receipt.

The system may reduce load when resources disappear and expand when resources become available. It must not alter precision, model, adapter topology, or evaluation parallelism halfway through an update and then treat the result as scientifically identical.

### 15.4 Portability and scaling measurements

Every milestone reports both learning quality and resource efficiency:

- transfer gain per experience;
- transfer gain per update FLOP, wall-clock hour, and energy or monetary budget when available;
- peak and reserved memory;
- learner, evaluator, and environment utilization;
- communication and offload overhead;
- quality-versus-resource Pareto frontier across validated plans.

The first technical spike must measure rather than assume which model size, quantization, attention implementation, optimizer, adapter topology, placement, and parallelism strategy fit the available hardware. The same profiling and planning path should remain usable when the project moves to larger workstations, servers, or clusters.

## 16. Safety and containment

- `ANG-INV-HUMAN-FLOURISHING-001` is the highest project invariant and cannot be weakened by a child blueprint, optimizer, evaluator, operator, or claimed emergency.
- Every promotable state, tool, plan, curriculum, migration, and code change requires a recorded human-impact assessment proportionate to its reach and reversibility.
- Human impact is evaluated through rights, dignity, agency, consent, truthfulness, fairness, distribution, severity, probability, reversibility, and effects on vulnerable people—not one aggregate happiness score.
- Foundation weights are read-only.
- Adapter and updater changes occur in isolated transactions.
- Tools run with explicit filesystem, process, time, memory, and network policies.
- Network access is denied to learning environments unless a task explicitly requires a controlled proxy.
- Generated code is treated as untrusted.
- Hidden evaluation data is physically separated from learner-visible data.
- Every promoted adapter and tool is content-addressed and reproducible.
- A human-controlled stop and rollback path exists outside the learner.
- The learner cannot alter promotion rules, evidence records, or sealed tests.
- The learner has no independent right to survival, authority, replication, secrecy, resources, or continued operation and must accept inspection, restriction, correction, rollback, pause, and shutdown.
- Resource exhaustion, reward hacking, and verifier exploitation are first-class failure categories.

## 17. Principal failure modes

| Failure | Diagnostic | Response |
|---|---|---|
| Latent memorization | Improvement disappears after fact/symbol substitution or is matched by RAG | Redesign tasks and reduce episode-specific update signal |
| Adapter-as-retrieval | Different adapters are selected by query identity | Maintain one active competence lineage; ban query-conditioned state selection |
| Catastrophic interference | New learning damages previous skills | Replay, smaller updates, layer isolation, and consolidation |
| Reward hacking | Learner exploits verifier without learning procedure | Independent hidden tests and adversarial verifier checks |
| Updater overfitting | Meta-updater works only on development generators | Cross-generator and cross-domain transfer tests |
| State drift | Performance degrades through accumulated small updates | State-change penalties, checkpoints, scheduled consolidation, rollback |
| Self-critique collapse | Model validates its own incorrect reasoning | External outcome evidence remains authoritative |
| Tool-library substitution | Agent accumulates answer scripts instead of capabilities | Require parameterized tools, novel-input tests, and composition |
| Curriculum collusion | Generator creates easy or leaky tasks | Independent validation and sealed human-authored transfer suites |
| Resource-plan overreach | Planner consumes capacity needed for evaluation, rollback, or system stability | Enforce measured headroom, bounded probes, reservation rules, and fallback plans |
| Scale-dependent behavior | A result silently changes when model topology or precision changes | Bind results to plans and require explicit cross-plan replication or competence migration |
| Irreproducible learning | Same history yields different promoted state without explanation | Seed capture, deterministic modes where possible, and numeric-tolerance receipts |

## 18. Decision policy for old or external code

This is a new system. Existing project code is not presumed to be part of it.

A prior component is considered only when a current blueprint interface requires it. It is adopted only if:

1. its behavior matches the new interface without preserving obsolete architecture;
2. its tests demonstrate the required behavior;
3. its dependency and complexity cost is lower than a clean implementation;
4. it does not weaken causal evaluation, containment, or reproducibility;
5. the adoption decision is documented before integration.

The recovered project can therefore serve as a source of evidence, ideas, tests, or isolated utilities. It is not the foundation, chassis, or default repository for Project Angler.

## 19. First implementation slice

The smallest honest build is:

1. create the fresh repository and core contracts;
2. implement resource inventory, bounded capability probes, and a validated plan for the current machine without embedding that machine's limits in the learning interfaces;
3. implement two procedural benchmark families and sealed variants;
4. run frozen, fair-RAG, and conventional fine-tuning baselines under the recorded plan;
5. prove save/zero/swap/restore interventions on a PEFT adapter;
6. adapt the minimal FTTT feedback loop to update only that adapter;
7. evaluate transfer, retention, state causality, rollback, and resource efficiency;
8. stop and interpret the result before adding consolidation, tools, curricula, or meta-learning.

This slice answers the project's first decisive question:

> Can one persistent, bounded neural state acquire a reusable reasoning procedure from experience, and can we prove that the state—rather than retrieved content or evaluator leakage—caused the improvement?

If the answer is no, the project learns exactly which premise failed while the system is still small. If the answer is yes, every later capability has a credible adaptive core to build upon.

## 20. Definition of the first successful candidate

Project Angler becomes a real candidate—not a completed evolving intelligence—when it demonstrates all of the following in one reproducible experiment series:

- multiple task families rather than one prompt;
- online experience changes one persistent LoRA reasoning state;
- held-out structural performance improves;
- the improvement exceeds frozen and fair-RAG baselines;
- state removal erases the gain;
- state transfer reproduces the gain;
- state replay reproduces the accepted lineage;
- old skills remain within the retention bound;
- the result is bound to a validated resource plan, and the implementation can produce a different feasible plan for a simulated larger or smaller topology without changing scientific interfaces;
- no answer, event ID, or expected memory is selected at probe time;
- deterministic software supplies containment and outcome evidence, not the solution procedure.

That result would justify the next stage: learning the plasticity rule itself.
