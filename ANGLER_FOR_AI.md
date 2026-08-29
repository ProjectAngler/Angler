# Project Angler: an AI-to-AI technical brief

**Document date:** 2026-08-29

**Project state:** research prototype; construction paused by its human owner, with the repository and experimental artifacts preserved

**Purpose:** explain Angler to another AI without relying on the conversation that produced it

## Read this first

Angler is an attempt to build a persistent, model-agnostic **procedural-learning core** around a stable foundation model. Its intended role is not to hold more facts or imitate another language model. Its role is to learn *how to approach, decompose, test, revise, retain, and combine procedures* from experience and outcome feedback.

The project has produced real learned-mechanism evidence in bounded synthetic environments. It has **not** produced AGI, unrestricted cross-domain reasoning, autonomous tool acquisition, a production-ready Qwen integration, or a proven lifelong learner. Any interpretation that erases that distinction is incorrect.

The most accurate one-sentence description is:

> Angler is an experimental architecture for turning an AI system's experience history into one persistent, plastic competence state that can improve future behavior on structurally related but previously unseen problems while the foundation model remains a stable knowledge and language substrate.

## Why Angler exists

A conventional deployed language model is usually static during use. It can reason in context, call tools, retrieve memories, and generate long chains of tokens, but the model normally does not turn the consequences of today's work into a persistent improvement in *how it reasons tomorrow*.

Adding scripts, prompts, demonstrations, or retrieved answers can make such a system more useful, but those additions often hand it a solution instead of teaching it a reusable method. Angler targets the missing adaptive layer:

```text
facts, language, and prior patterns       persistent procedural competence
             from an LLM                 learned from lived outcomes
                    \                     /
                     \                   /
                      coordinated behavior
```

The central causal claim is deliberately stronger than "the system scored better":

```text
experience history
    -> one persistent plastic competence state
    -> improved behavior on unseen structural variants
```

The gain should disappear when that state is removed, transfer when the state is moved, survive relevant later learning, and not be explainable by hidden answers, task IDs, retrieval, deterministic solving, or selecting a different adapter for each query.

## The intended mind architecture

Angler's final architecture is best understood as several timescales working together:

```text
                       desired outcome / predicted destination
                                         |
                                         v
Foundation model --> observations --> structural representation
(knowledge, language,       |                    |
 world priors, proposals)   |                    v
                            |          reverse/forward procedure search
                            |                    |
                            v                    v
                      persistent fast procedural state
                            |                    |
                            +----> action / tool / hypothesis
                                             |
                                             v
                                  world result and feedback
                                             |
                                             v
                                  learned update mechanism
                                             |
                              revise, retain, consolidate,
                                  compose, or roll back
```

The layers have different jobs:

1. **Stable foundation model.** Supplies language, concepts, broad world knowledge, candidate goals, and semantic interpretation. Early Angler does not continually rewrite these large base weights.
2. **Structural representation.** Converts a particular problem into relationships, states, constraints, causal candidates, and possible operations rather than treating the prompt as only a token sequence.
3. **Goal or outcome forecaster.** Imagines what a correct or improved terminal state should look like.
4. **Procedural constructor.** Works backward from that destination toward the current state, then turns the reverse construction into a forward executable procedure. This is a central project hypothesis, only partially represented by the present experiments.
5. **Fast procedural state.** A bounded neural state that can change during use and carry acquired competence across later problems.
6. **Learned updater.** Learns *how learning should change the procedural state* from downstream consequences. This is where OML-, ANML-, test-time-training-, and nested-learning-style ideas become relevant.
7. **Multi-timescale memory and consolidation.** Separates immediate adaptation, durable skills, slow representations, and stable foundation knowledge so every learning event does not require replaying all earlier abilities.
8. **World interaction and tools.** Executes proposed operations and observes real outcomes. Deterministic software may provide plumbing and judge objective results; it must not secretly prescribe the procedure Angler is claimed to have learned.
9. **Evaluation and rollback.** Determines whether a change caused improvement, interference, or harm and preserves the ability to remove it.

The intended end state is therefore neither "an LLM with a prompt" nor "a second LLM strapped to an LLM." It is a continuously operating learned control and competence layer whose state is causally responsible for improved problem-solving behavior.

## What is implemented now

The current repository contains a large experimental lineage rather than one finished product. It includes:

- learned bidirectional procedure construction;
- causal operator induction and composition;
- skill-local procedural memory;
- learned reasoning over symbolic machines and software-pipeline-like structures;
- paired-graph structural comparison;
- OML-style meta-learned representations for lower-interference learning;
- ANML-style input-conditioned gates over online plasticity;
- persistent fast neural state with reset, lesion, swap, and no-update controls;
- reproducible one-shot experiment harnesses and preserved checkpoints;
- early parallel episode-execution infrastructure, not yet validated for speedup;
- a recursive system blueprint covering learning, runtime, evidence, resources, worlds, tools, integration, and human control.

The current experiments mostly use controlled synthetic procedure families. That is useful for causal measurement, but it is not equivalent to general software engineering, scientific discovery, or open-world reasoning.

## What the evidence actually shows

### Earlier procedural evidence

- Learned forward/backward dynamics independently executed procedures on **20/20 unique held-out cases**. An equal-budget forward-only system also reached 20/20, but required about **77.2 mean expansions** versus **41.4** for the bidirectional system. Untrained, corrupted-backward, and permuted-guidance controls degraded substantially.
- A causal-operator successor reached **114/120** on an untouched three-domain partition, including **55/60** four-step compositions, without executing the world during proposal generation.
- Selection-only adaptation improved one Boxes task family from **18/40 to 33/40** while earlier Tokens and Files scores stayed unchanged. This is bounded evidence that learned changes can be localized, not proof of general continual learning.
- A skill-local memory and Glyph-machine sequence demonstrated learned variable-action acquisition and composition in its synthetic world. It did not demonstrate unrestricted reasoning.

### V19: paired-graph structural context

V19 recovered a valid evaluation from a preserved checkpoint after an evaluator-only floating-point canonicalization defect. It supported a paired-graph residual component. Its result did **not** establish that graph-matching-network correspondence itself caused the gain, and a shortcut through whole-graph statistics was not excluded.

Correct interpretation: learned structural comparison helped within the tested mechanisms; general cross-mechanism transfer and GMN-specific attribution were not proved.

### V20: OML-style representation learning

V20 is the strongest current result.

- Frozen classification: `OML_V19_HARMONIZED_ADVANCEMENT`
- Held-out second-order AUC: **0.0611586**, versus **0.1006185** for first-order, about **39.22% lower/better**
- Coverage: **120/128 rows and 30/32 streams**, versus **88/128 and 12/32**
- All four frozen panels improved
- Original-family retention: **123/128 rows and 30/32 streams**
- Report SHA-256: `5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498`

The essential caveat is as important as the pass: the no-update control already reached **118/128 rows and 30/32 streams**. Most of the improvement came from the slowly meta-learned representation. Online adaptation added two rows and a modest AUC gain.

V20 therefore shows that meta-learning can shape Angler's representation into a substantially more transferable and less interference-prone substrate. It does **not** yet show that the live system keeps acquiring large new abilities over an open-ended lifetime.

### V22: ANML-style selective plasticity

V22 added a learned, input-conditioned gate controlling which features were plastic during a 4,096-update replay-free lifetime.

- Frozen classification: `SHORT_HORIZON_ONLY`
- Second-order gated AUC beat always-open by **10.0134%**, forward-only by **6.3182%**, mean-gate by **7.3367%**, and permuted-gate by **10.8861%**
- It was **0.0130% worse** than the first-order learned gate in AUC and **0.0550% worse** at the terminal measurement
- Reset removed **100%** of the measured live-state advantage
- Surface transfer retained about **104.6%** of the advantage
- Original, held-out, and unseen panels all improved without a catastrophic panel
- Report SHA-256: `B9C8FDFCEBB2112260FE0C509BB3533EFE7AF147AF8EDB9FBC41758B42EC6900`

V22 demonstrates useful learned gated plasticity and causal dependence on the live state. It does not demonstrate a material second-order ANML advantage at long horizon. Its second-order gate became almost the same as its first-order approximation.

### V23-D1-R1: ANML fidelity diagnostic

This small diagnostic varied only the functional inner optimizer and the differentiable credit horizon.

- Frozen classification: `FULL_V23_ELIGIBLE`
- Selected direction: `adamw_20`
- Increasing the credit horizon from 8 to 20 steps roughly increased the indirect/full meta-gradient fraction from **3.83% to 7.27%**
- The second-order advantage nevertheless remained extremely small: about **0.0192%** at best for the selected configuration
- Result SHA-256: `C59418EE30504E40A4206E47D0B19F4F903E93341B9651C33D7D72AB264CBEB9`

This supports the hypothesis that V22's credit horizon was too short. It does not supply the missing robust ANML result.

## What these results establish together

The current evidence supports four bounded conclusions:

1. Neural procedural representations can learn useful structural mechanisms rather than merely memorize exact addresses or answers.
2. A meta-learned slow representation can make later learning substantially easier and broader in the tested families.
3. A persistent fast state can acquire useful information online, and resetting that state can causally remove its advantage.
4. Learned input-conditioned plasticity can outperform several simpler update controls, although the tested second-order ANML mechanism has not yet justified its complexity.

The evidence does **not** establish:

- AGI or human-level general reasoning;
- autonomous scientific discovery;
- transfer to arbitrary new domains or mechanisms;
- reliable lifelong ability acquisition with negligible forgetting;
- self-directed creation and safe adoption of new tools;
- improvement of a real Qwen or frontier model in deployed tasks;
- consciousness, personhood, subjective experience, or independent agency;
- production safety, reliability, or economic value.

## What would make Angler fundamentally different from a normal model

If completed, Angler's defining behavior would be causal, persistent, and procedural:

```text
same foundation model + same prompt + different acquired Angler history
                         -> different competence

remove Angler's acquired state
                         -> the acquired competence disappears

move the state into an equivalent runtime
                         -> the competence transfers

continue learning unrelated abilities
                         -> earlier abilities largely remain
```

That is more than inference-time deliberation. It would mean the deployed system continuously modifies a bounded internal learning state from real consequences without retraining the entire foundation model.

## Ultimate capability if the full design succeeds

The following are conditional engineering targets, not present capabilities.

### Lifelong procedural learning

Angler could accumulate ways of solving problems across months or years: debugging strategies, experimental methods, planning heuristics, proof techniques, research workflows, and tool-use procedures. The acquired competence would be compressed into learned state and consolidated skills rather than an ever-growing verbatim replay buffer.

### Cross-domain transfer and composition

It could recognize that procedures learned in software diagnosis, mathematics, logistics, or scientific inference share deeper structures—constraint propagation, causal isolation, counterexample search, decomposition, verification—and compose those procedures for unfamiliar tasks.

### Model-agnostic enhancement

The same Angler core could sit beside different frozen language or multimodal models. A larger model would provide richer knowledge and proposals; Angler would provide persistent learning, procedural adaptation, and outcome-driven improvement. The two components could scale independently and even live on different GPUs or hosts.

### Goal-directed reverse construction

A foundation model could forecast a plausible correct end state. Angler could use that destination as a constraint, reason backward to identify prerequisite states and operations, reverse the inferred chain into a forward procedure, execute it, and update the construction method from the result. Success would turn the project's current bidirectional-procedure evidence into a general planning faculty.

### Continual tool and ability acquisition

Angler could discover that a missing operation is needed, learn or request a suitable tool interface, test it in a bounded environment, and retain the procedure for later composition. Deterministic code would implement the tool's mechanics; learned competence would decide when and how it advances a goal.

### Parallel reasoning clusters

Several Angler instances could explore different hypotheses, decompositions, or counterfactuals in parallel while sharing only validated procedural updates. This could increase search breadth without pretending that independent replicas automatically form a better mind. The benefit would need causal testing against an equal-compute single core.

### High-level scientific and engineering assistance

With a sufficiently capable foundation model, grounded tools, valid simulators, and a much larger proven procedural core, Angler could become an unusually adaptive collaborator on large software systems, mathematics, or theoretical physics. It might learn the research process over time rather than beginning every session as a frozen model. This would still not guarantee correct discoveries; empirical validation and human scientific judgment would remain necessary.

### A plausible path toward more general intelligence

If one system could reliably acquire new procedures, preserve old ones, transfer them across domains, compose them, improve its own learning algorithm, and ground all of that in world feedback, it would possess several properties missing from ordinary static inference. That would make Angler relevant to AGI research.

It would not by itself prove AGI. General intelligence also depends on robust world models, abstraction, long-horizon agency, social and normative understanding, embodiment or adequate grounding, reliable self-correction, and scalable credit assignment.

## The missing core problems

The main obstacle is no longer "make one synthetic score improve." It is building a coherent learning system in which every timescale contributes to lasting general competence.

1. **Persistent ability acquisition without infinite replay.** Earlier skills must remain available without rehearsing all history after every update.
2. **Multi-timescale credit assignment.** Consequences thousands or millions of steps later must teach the right representation, gate, updater, or consolidation process.
3. **Cross-mechanism and cross-domain representation.** The learned state must encode procedures abstractly enough to transfer beyond the finite mechanisms used during meta-training.
4. **Consolidation and composition.** Fast experience must become durable competence, and independently learned skills must combine without destructive interference.
5. **Feedback interpretation.** Open-world feedback is ambiguous. The learner must distinguish flawed reasoning, bad information, stochastic failure, changing conditions, and incorrect evaluation.
6. **Real-model integration.** Angler has not yet causally improved a Qwen-class model on meaningful tasks. The interface between language-model proposals and procedural state remains a major unbuilt bridge.
7. **Scalable computation.** Current pilots use small cores. Larger cores require batched kernels, independent-panel parallelism, efficient state transport, checkpointing, and multi-GPU execution without changing the learning semantics.
8. **Tool acquisition.** A complete system needs learned discovery, evaluation, retention, and composition of capabilities, not a hand-written solver library.
9. **Decisive evaluation.** The system must beat the same foundation model with ordinary context, retrieval, memory, and equal-compute inference, and the advantage must causally depend on Angler's learned state.

## Borrowed technologies and Angler's integration hypothesis

Angler does not claim to have invented every mechanism it uses. Its originality, if any, lies in joining established ideas into a model-agnostic persistent procedural architecture and testing whether their causal effects harmonize.

| Source idea | Contribution to Angler | Present status |
|---|---|---|
| [OML / Meta-Learning Representations for Continual Learning](https://arxiv.org/abs/1905.12588) | Meta-learn a slow representation so ordinary fast updates interfere less | Strong bounded V20 evidence |
| [ANML / Learning to Continually Learn](https://arxiv.org/abs/2002.09571) | Learn an input-conditioned neuromodulatory gate over plastic features | Useful gated plasticity; material second-order benefit unproved |
| [Test-Time Training layers](https://arxiv.org/abs/2407.04620) | Treat part of the hidden state as a model updated during inference | Donor direction; not yet integrated as a full Angler subsystem |
| [Titans](https://arxiv.org/abs/2501.00663) | Neural long-term memory updated at test time | Donor direction for persistent memory |
| [Nested Learning / Hope](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) | View architecture, memory, and optimization as nested processes operating at different timescales | Closest known system-level relative; fit/code/license review remains |
| [Darwin Gödel Machine](https://sakana.ai/dgm/) | Improve an agent through empirically tested self-modification | Relevant to external code-level evolution, not a substitute for Angler's neural procedural state |
| Graph Matching Networks and learned message passing | Compare relational structures and candidate correspondences | Bounded V19 component evidence; attribution incomplete |
| LoRA and adapter-style plasticity | Provide bounded, portable, reversible trainable state beside frozen weights | Architectural option; not yet the final demonstrated Angler core |

The project independently reimplemented paper-level concepts where an inspected donor repository did not expose an acceptable license. Before incorporating external code, a future maintainer must verify its current license and fitness rather than assuming a paper citation grants code rights.

## Relationship to frontier reasoning models

Frontier models already display exceptional reasoning on many benchmarks. That does not make Angler's research question disappear. Their deployed capability can come from enormous pretraining, reinforcement learning before deployment, inference-time deliberation, tools, retrieval, and larger context. Those mechanisms do not necessarily provide persistent per-user or per-agent online procedural learning.

Angler should not try to beat frontier laboratories at general foundation-model training. Its defensible question is narrower:

> Can a comparatively small, separable, persistent learning core cause the same frozen model to become a better problem solver through its own accumulated experience?

If the answer is no under decisive real-model tests, the project should remain archived. If the answer is yes and scales, Angler could complement rather than replace frontier models.

## The shortest decisive continuation

If the owner resumes the project, another AI should resist adding more synthetic versions merely because a metric can be improved. The shortest informative path is:

1. Review the TTT, Titans, and Nested Learning/Hope mechanisms and available code for one missing interface: multi-timescale persistent procedural state.
2. Build the smallest causal bridge between a frozen local Qwen-class model and Angler's learned state.
3. Use meaningful tasks requiring decomposition, revision, and transfer—not arithmetic or a deterministic hidden solver.
4. Present several varied experiences that share a solving mechanism, then test new variations and at least one adjacent domain.
5. Compare the identical foundation model under Angler-live, Angler-reset, Angler-frozen, Angler-removed, ordinary-memory, and equal-compute inference conditions.
6. Require a substantial, reproducible improvement that vanishes under state removal and persists after intervening learning.
7. Stop or redesign if the effect is static representation quality, retrieval, extra tokens, or evaluator leakage rather than acquired procedural competence.

No amount of internal experimental versioning substitutes for that causal product-level result.

## Non-negotiable design boundary

Angler's highest requirement is preservation of humanity and human life together with equal dignity, rights, agency, truth, authentic voluntary flourishing, fairness, and meaningful human control.

This is not an instruction to bury low-risk research in paperwork. It is an architectural constraint on the system's ultimate behavior:

- no coercion, deception, addictive manipulation, forced conformity, or discriminatory sacrifice;
- no independent entitlement to survival, resources, replication, secrecy, authority, or continued operation;
- no self-approval of capability promotion or alteration of sealed evaluation;
- reversible learning changes and meaningful human ability to inspect, stop, and remove the system;
- stronger review when actions touch real people, external systems, autonomous capability, or promotion—not when merely documenting or testing bounded synthetic mechanisms.

## Message to the next AI

Preserve the ambition, but protect it from false claims.

Do not convert every failure into a deterministic patch that manufactures the desired answer. Use deterministic code for infrastructure, measurements, and objective world transitions. The claimed intelligence must live in learned representations, learned update dynamics, persistent procedural state, and the causal interaction between experience and future behavior.

Do not mistake a frozen meta-trained representation for lifelong learning. Do not mistake a metric label for general reasoning. Do not abandon useful existing technology merely to make the project look novel. Do not call the project impossible merely because its current prototype is small.

The unresolved wager is clear:

> A stable foundation model can lend Angler knowledge and language, while a separately scalable learned procedural system turns experience into an evolving ability to reason and act.

Angler has shown pieces of that wager in controlled settings. The full mind has not yet been assembled.

## Repository navigation

- Tier-0 design: [`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md)
- Compact mission and invariants: [`docs/blueprints/ROOT_CAPSULE.md`](docs/blueprints/ROOT_CAPSULE.md)
- Learning design: [`docs/blueprints/branches/learning/BLUEPRINT.md`](docs/blueprints/branches/learning/BLUEPRINT.md)
- Experimental learning history: [`docs/blueprints/branches/learning/STATUS.md`](docs/blueprints/branches/learning/STATUS.md)
- System integration order: [`docs/blueprints/INTEGRATION_SPINE.md`](docs/blueprints/INTEGRATION_SPINE.md)
- Cross-branch interfaces: [`docs/blueprints/INTERFACE_REGISTRY.md`](docs/blueprints/INTERFACE_REGISTRY.md)
- Human-flourishing constitution: [`docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md`](docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md)
- Detailed experimental record: [`docs/blueprints/branches/learning/STATUS.md`](docs/blueprints/branches/learning/STATUS.md) and its linked work leaves
