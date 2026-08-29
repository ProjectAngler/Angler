# Project Angler

**A persistent procedural-learning core for adaptive AI.**

Project Angler is an independent open research project. It is not affiliated
with, sponsored by, or endorsed by the operators of the Angler AI commercial
platform or the EU-funded ANGLER maritime project. “Project Angler” is the
research project name; any eventual product may use a different name.

Project Angler investigates whether a stable foundation model can be paired
with a separately scalable neural system that learns *how to solve problems*
from experience. The intended result is not another language model, prompt
filter, retrieval layer, or library of hand-written solvers. It is one
persistent plastic competence state whose acquired procedures causally improve
future behavior.

> **Current status:** preserved research prototype. Angler has bounded learned-
> mechanism results, including strong OML-style representation evidence and
> useful causally active gated plasticity. It has not demonstrated AGI,
> unrestricted cross-domain reasoning, reliable lifelong learning, or a
> product-level improvement to Qwen or another foundation model.

> **AI-assistance disclosure:** OpenAI Codex materially assisted with the
> repository's source, documentation, testing, and audit under sustained human
> direction and review. The human owner supplied the objective, architecture
> direction, corrections, acceptance decisions, and publication decision and
> takes responsibility for the public release. Anthropic Claude supplied
> limited, owner-relayed design consultation. See
> [`PROVENANCE.md`](PROVENANCE.md) for the full boundary.

For the clearest self-contained explanation, begin with
[`ANGLER_FOR_AI.md`](ANGLER_FOR_AI.md).

## The central claim

```text
experience history
    -> one persistent plastic competence state
    -> improved behavior on unseen structural variants
```

A valid gain must not come from answer leakage, task identifiers, retrieval of
the evaluated solution, deterministic solution logic, or query-conditioned
model selection. Removing the learned state should remove the acquired
advantage; moving or replaying accepted state should reproduce it; later
learning should preserve earlier abilities within measured bounds.

## Intended architecture

Angler separates four responsibilities:

1. A stable foundation model supplies language, knowledge, semantic
   interpretation, and candidate goals.
2. A structural reasoner constructs and tests forward and backward procedures.
3. A persistent neural state and learned updater acquire procedural competence
   from outcomes at several timescales.
4. Tools and controlled worlds execute actions and return evidence without
   prescribing the claimed learned solution.

The full design adds consolidation, skill composition, continual tool
acquisition, resource-elastic execution, causal evaluation, and exact rollback.

## Current evidence

| Experiment | Bounded result | Essential limitation |
|---|---|---|
| Learned bidirectional procedure core | 20/20 held-out solutions with about 41.4 mean expansions versus 77.2 forward-only | Equal-budget forward-only also solved 20/20 |
| Causal operator core | 114/120 on an untouched three-domain partition, including 55/60 four-step compositions | Controlled symbolic domains |
| V19 paired-graph context | Learned structural comparison component supported | GMN-specific correspondence and cross-mechanism transfer not established |
| V20 OML | 120/128 held-out rows and 30/32 streams; second-order AUC about 39.22% better than first-order | Most gain came from the slow meta-learned representation; live adaptation added only two rows |
| V22 gated plasticity | Beat several open/forward/mean/permuted controls; resetting live state removed the measured advantage | No material second-order ANML advantage at the 4,096-step horizon |
| V23-D1-R1 | A 20-step credit horizon increased measurable indirect meta-gradient contribution | Absolute second-order advantage remained tiny |

These are causal, synthetic research results—not a claim of general intelligence.
Detailed identities, frozen interpretations, and hashes live in the Learning
status and experimental work leaves.

## Repository map

- [`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md) — authoritative Tier-0 design
- [`ANGLER_FOR_AI.md`](ANGLER_FOR_AI.md) — AI-to-AI project and capability brief
- [`docs/blueprints/ROOT_CAPSULE.md`](docs/blueprints/ROOT_CAPSULE.md) — compact mission and invariants
- [`docs/blueprints/branches/learning/STATUS.md`](docs/blueprints/branches/learning/STATUS.md) — experimental learning history
- [`docs/blueprints/INTEGRATION_SPINE.md`](docs/blueprints/INTEGRATION_SPINE.md) — vertical integration sequence
- [`src/angler/`](src/angler/) — runtime, learning, procedure, world, and evidence modules
- [`experiments/`](experiments/) — evaluators, manifests, and experimental runners
- [`tests/`](tests/) — focused and integration tests
- [`PROVENANCE.md`](PROVENANCE.md) — recovery, authorship, and stewardship record
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — dependencies, donor concepts, licenses, and source-similarity audit

Generated checkpoints, model payloads, recovered conversations, private data,
and local work directories are excluded from version control. Result identities
and hashes are retained in the source records where needed for reproducibility.

## Installation

Project Angler requires Python 3.11 or newer. Install the procedural core in a
virtual environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional foundation-model boundary and development tools when
needed:

```bash
python -m pip install -e ".[qwen,dev]"
python -m unittest discover -s tests -p "test_*.py"
```

CUDA-enabled PyTorch installation varies by operating system, driver, and CUDA
runtime. Select the appropriate official PyTorch build for the target machine
instead of assuming the preserved RTX 5080 environment is portable.

## Development environment

The preserved GPU environment used Ubuntu 24.04 under WSL2, Python 3.12,
PyTorch 2.13.0 with CUDA 13.0, NumPy 2.5.2, SciPy 1.18.1, and pytest 9.1.1 on an
RTX 5080. Many core tests use Python's standard `unittest`; GPU experiments
require their frozen runner-specific environment and preserved checkpoints.

Do not treat a successful unit test as permission to rerun a consumed
scientific identity. Experimental leaves document which results are immutable,
which checkpoints are external, and whether a successor is permitted.

## Human control

All Angler capability is subordinate to preservation of humanity and human life
together with equal dignity, rights, agency, truth, authentic voluntary
flourishing, fairness, and meaningful human control. See the
[`Human-Flourishing Constitution`](docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md).

This requirement forbids coercive optimization, deceptive manipulation,
self-granted authority, and any independent entitlement to survival,
replication, secrecy, resources, or continued operation.

## Provenance and licensing

This is a reconstructed local lineage created after deletion of the original
repository and account. The missing Git history is not represented as
recovered. See [`PROVENANCE.md`](PROVENANCE.md) for the exact boundary.

Project Angler is licensed under the
[`Apache License 2.0`](LICENSE), allowing commercial and non-commercial use,
modification, and distribution under its terms. Contributions are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Paper citations describe intellectual provenance; they do not implicitly
license external source code. External implementations still require an exact
license and provenance review before their code is incorporated. The current
dependency, model, adapted-code, and clean-room boundaries are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`NOTICE`](NOTICE).
