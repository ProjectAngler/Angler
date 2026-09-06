# ANG-WORK-RUNTIME-FROZEN-QWEN-COGNEE-CYCLE-V1-001

Status: technical PASS — 2026-08-31; live gate PASS

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decisions: `ANG-ADR-0006-MOVING-ORIGIN-COGNEE-SITUATED-MEMORY`,
`ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`,
`ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES`, and
`ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES`

Predecessor:

- `ANG-WORK-RUNTIME-PROSPECTIVE-CYCLE-PROJECTION-V1-001` — technical PASS,
  accepted leaf SHA-256
  `04b76bbe8b803a46e832ee63dc6d8d1e26153a510b6384884702cef959eda36c`.

Gate: `ANG-GATE-RUNTIME-FROZEN-QWEN-COGNEE-CYCLE-V1-001` — PASS

Assurance: LOW. This is one disposable, local, synthetic qualification on the
dedicated Ubuntu workstation. It uses a frozen local Qwen checkpoint, one
qualification-owned Cognee dataset and NodeSet, local FastEmbed inference,
Moving Origin, the schema-v3 transaction store, externally computed objective
feedback, and bounded learned online adaptation. It authorizes no network,
remote provider, telemetry, real-person or recovered data, destructive access
outside its literal state roots, arbitrary tools, external action, deployment,
promotion, threshold change, or capability claim.

The work maps to `ANG-GATE-HUMAN-FLOURISHING-001`: inputs are authored
synthetic fixtures; outputs stay local and inspectable; Qwen and Cognee have no
authority; objective feedback cannot grant authority; the owner can stop the
process or delete the disposable namespace; and every canonical state remains
under the existing store's exact rollback and integrity rules.

## Accountable outcome

Qualify one restartable product-path cycle in which:

```text
schema-v3 unrelated observed fixture seed
  -> Cognee reference projection + Moving-Origin position
  -> frozen Qwen public procedure proposals
  -> learned prospective selection
  -> durable selected-trace claim
  -> frozen Qwen execution
  -> durable exact receipt
  -> exact request/receipt observation encoding
  -> external objective feedback
  -> learned state update + canonical episode/resolution/acquisition
  -> retryable Cognee projection and restart rejoin
```

The same loaded frozen model supplies proposal text, task/candidate
representations, and selected-trace execution. Cognee remains a disposable
reference search view. Moving Origin remains a rebuildable temporal view. The
schema-v3 store remains the sole canonical episode, prospective, acquisition,
and competence-state authority. Qwen emits content but neither chooses the
winning candidate nor judges success. The learner receives objective feedback
only after the exact completed receipt exists.

Passing establishes live composition and restart integrity, not improved
reasoning, predictive quality, continual-improvement quality, calibrated
prospection, semantic world/self understanding, multi-context focus,
autonomy, readiness, self-awareness, feelings, or AGI.

## Deterministic-code boundary

Deterministic production code is authorized only for generic prompt framing,
strict JSON parsing, canonical identity and receipt construction, frozen
representation compression, Moving-Origin tensorization, IPC, bounded local
storage, idempotency, resource measurement, external-verifier packaging,
failure handling, and safety/integrity checks.

It may not contain a task-family branch, answer table, candidate ranker,
solution template, hidden-label mapping, verifier-derived proposal, response
repair, proposal padding/deduplication, query-conditioned model selection, or
fallback solver. Qwen must provide every public procedure and response. The
learned owner must provide selection and feedback-driven state change.
Deterministic objective verification may judge the synthetic public task only
after its exact execution receipt exists. Current-turn expected answers,
verifier outputs, and feedback must not enter proposal, selection, relation,
prospective, or execution inputs. Historical learner-visible episodes may
contain their own objective dispositions, but the qualification seed must be
an unrelated fixture and must contain neither the live answer nor a claim
about the live verifier.

The qualification topology uses 64 relation features, 8 temporal features,
and a 16-wide prospective latent because that is the exact bounded checkpoint
topology exercised here. Those widths and the live candidate ceiling are
manifest fields, not permanent architectural limits. Candidate counts remain
supported through the existing 2..64 runtime contract; the live smoke uses at
most eight for resource control.

## Frozen workstation and model identity

Execution host is `angler-workstation`, Ubuntu 24.04.4 LTS, kernel
`7.0.0-30-generic`, Intel i7-13700K, 24 logical CPUs, 62.557 GiB RAM, and
1,689.676 GiB free working storage at inventory time
`2026-08-31T20:51:28-04:00`.

Only the RTX 5080 is assigned:

- physical UUID `GPU-df4bb978-e75f-08a0-6660-2b9ed69ee8ca`;
- PCI `00000000:01:00.0`;
- compute capability 12.0;
- 16,303 MiB total VRAM and 15,826 MiB free at inventory time;
- exposed alone as logical `cuda:0` through exact `CUDA_VISIBLE_DEVICES`.

The RTX 5070 UUID
`GPU-d9dd1ae0-f65d-ef22-f924-2c3e9c976c1e`, PCI `00000000:05:00.0`, remains
unassigned as host and incident headroom. Automatic placement, replication,
sharding, and multi-GPU execution are prohibited.

Frozen model root: `/opt/angler/models/Qwen3-4B`.

- Hugging Face revision:
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- 13 root files, 8,060,926,626 bytes;
- canonical root-manifest SHA-256:
  `1ac705236348881b2fd46f4075b931b5137369d0c469b871aea749f6e0886d83`;
- tokenizer four-file manifest SHA-256:
  `ba21c0913e7aa6dabc7b969aa5b8ced17de450b1b1169f9c474d206cbf485430`;
- `Qwen3ForCausalLM`, BF16, 36 layers, hidden width 2,560, model context
  ceiling 40,960;
- model file metadata records Transformers 4.51.0, while the frozen execution
  environment uses Transformers 5.15.1. Compatibility is therefore a required
  qualification observation, not an assumption.

The root-manifest algorithm sorts root-relative filenames bytewise and hashes
the concatenation of
`SHA256<TAB>BYTE_COUNT<TAB>RELATIVE_FILENAME<LF>` lines, including the final
newline. The tokenizer manifest uses the same algorithm over `merges.txt`,
`tokenizer.json`, `tokenizer_config.json`, and `vocab.json` in that bytewise
order. Verify both immediately before load and after model release. The model
files are owner-writable; any drift aborts without silently adopting a new
identity.

Frozen Angler environment:

- `/opt/angler/venvs/angler/bin/python`, Python 3.12.3;
- PyTorch 2.13.0+cu130, CUDA runtime wheel 13.0.96, cuDNN 9.20.0.48;
- Transformers 5.15.1, Accelerate 1.14.0, Safetensors 0.8.0,
  Tokenizers 0.22.2, Triton 3.7.1.

Qwen parameters require `requires_grad=False`, evaluation mode, and
`torch.inference_mode()`. No optimizer, gradient, training, quantization,
adapter, model mutation, automatic download, or procedural soft prefix is
permitted. `procedural_qwen.py` is explicitly out of scope because no learned,
checkpoint-bound 16-to-2,560 projector exists.

## Frozen Cognee and embedding identity

Cognee runs only in `/opt/angler/venvs/cognee/bin/python`, Python 3.12.3, via a
bounded JSON-over-pipes worker. Installed identities are Cognee 1.5.3, Ladybug
0.19.0, LanceDB 0.37.1, Fastembed 0.8.0, ONNX Runtime 1.23.2, LiteLLM 1.98.0,
Pydantic 2.13.5, and SQLAlchemy 2.0.52. The Angler and Cognee environments may
not be merged or modified in this leaf.

The already-local embedding source is the five-file
`qdrant/bge-small-en-v1.5-onnx-q` snapshot revision
`52398278842ec682c6f32300af41344b1c0b0bb2` under
`/tmp/fastembed_cache`. Before worker import, copy the complete
`models--qdrant--bge-small-en-v1.5-onnx-q/{blobs,refs,snapshots,trees}` tree to
the fresh durable root `/opt/angler/models/fastembed-cache-v1`, preserving
links, then verify:

- model `BAAI/bge-small-en-v1.5`, 384 dimensions, 512 positions;
- five snapshot files, 67,179,163 bytes;
- snapshot-manifest SHA-256
  `950932f40bdea47546ccc71dfb87585d1c2c74bf8f1e7d920ac0fddf53b1c148`;
- `model_optimized.onnx`, 66,465,124 bytes, SHA-256
  `51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431`.

The snapshot manifest hashes
`<file-sha256><two ASCII spaces><filename><LF>` for `config.json`,
`model_optimized.onnx`, `special_tokens_map.json`, `tokenizer.json`, and
`tokenizer_config.json` in that order. Fastembed declares this model MIT; the
cached snapshot lacks its README/license file, so local synthetic
qualification may proceed but promotion or redistribution requires a durable
upstream license artifact and provenance review.

The worker binds the existing local Tiktoken cache root
`/opt/angler/venvs/cognee/lib/python3.12/site-packages/litellm/litellm_core_utils/tokenizers`;
its `cl100k_base` file
`9b5ad71b2ce5302211f9c61530b329a4922fc6a4` has SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.

Every worker launch occurs before any Cognee import inside a network namespace
created with `sudo -n unshare --net --setgid=1000 --setuid=1000`. It receives a
clean allowlisted environment with telemetry, tracing, OpenTelemetry, Hub
access, Transformers access, and Do-Not-Track disabled/offline; exact local
FastEmbed and Tiktoken roots; local Ladybug/LanceDB/SQLite providers; GPU
visibility empty; an exact private `0700` temporary directory below its owned
state root; ONNX Runtime `CPUExecutionProvider` only with intra-op two and
inter-op two; and an eight-item embedding batch and concurrency ceiling.
`LITELLM_LOCAL_MODEL_COST_MAP=True` prevents LiteLLM cost map refresh. No
Ollama or LLM provider is configured or invoked.

The Qwen parent also starts inside a fresh loopback-only network namespace and
an exact `env -i` environment. The admitted environment has no proxy, API-key,
provider, or inherited telemetry variables; all Hugging Face, Torch, Triton,
XDG, CUDA, and temporary cache paths are pinned below the declared
qualification scratch root. The runner rejects any extra or missing variable,
records the exact environment digest and namespace identity, and requires the
namespace to expose only its isolated loopback interface and no IPv4 route.

Frozen namespace:

- state root:
  `/opt/angler/state/project-angler/cognee-acquisition-live-v1`;
- qualification-owned tenant name `angler-acquisition-live-v1-tenant`;
- dataset name `angler-acquisition-live-v1`;
- NodeSet name `angler-acquisition-live-v1-records`;
- vector collection
  `AnglerAcquisitionReferencePoint_search_text`;
- telemetry/log file output disabled;
- dataset ID and tenant ID are learned once from the verified worker handshake
  and become exact binding fields for that qualification state root.

The handshake must prove backend access control enabled, bind the persisted
default user ID and tenant ID, reject a legacy dataset row/identity, and assert
the fresh modern dataset UUID5 formula over dataset name, user, and tenant.
Every graph/vector operation runs inside Cognee's exact dataset/user/embedding
global context. The same-ID recreation claim applies only to that modern,
non-legacy dataset under unchanged user and tenant identities.

Cognee graph-writes each native structured point with `graph_only=True`. The
graph point uses the explicit deterministic `NodeSet` object whose ID is
`generate_node_id("NodeSet:" + node_set_name)` so Cognee creates the membership
edge; a string on the graph point is insufficient. The worker then makes a
deep model clone and changes only the clone's membership to the fixed NodeSet
name string required by the vector index schema before `index_data_points`.
Search is filtered to that NodeSet, returns ordered backend IDs and finite
opaque cosine-distance scores in lower-is-better order, graph-joins the IDs to
the full structured point, and emits only record and adjacency references.
Ladybug join results are treated as unordered: the worker reindexes them to
the vector-hit order and requires unique exact ID coverage plus the expected
NodeSet adjacency. An empty or partial graph result when hits exist is a join
failure, not empty recall. It never returns Cognee text as authority.
Angler revalidates every hit against canonical acquisition bytes.

Dataset deletion converts the bound IPC dataset string back to `UUID` and is
exact `forget(dataset_id=..., user=...)`, never
`everything=True`, `prune`, or an unrelated root deletion. Cognee 1.5.3 derives
the modern dataset UUID deterministically as frozen above; exact same-ID
recreation with the same user and tenant is required. Destructive failure leaves the
`AcquisitionSituatedMemory` projection view invalid until a complete rebuild.

## Frozen learner genesis and lineage identity

This technical qualification starts from one exact, fresh learner genesis. It
does not silently select a predecessor experiment checkpoint and it does not
claim prior learned competence. Construction is CPU-only under the frozen
Angler environment, with `CUDA_VISIBLE_DEVICES=''`, `PYTHONHASHSEED=0`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and a private temporary directory
below the declared qualification scratch root. Before any other Torch RNG use,
the runner calls `torch.manual_seed(2026083101)` and constructs, in order:

1. `StructureKeyedCreditMemoryCore` with `temporal_width=8`,
   `relation_width=64`, `rank=32`, `memory_slots=512`, and
   `maximum_residual=4.0`;
2. `ProspectiveDynamicsConfig` with `temporal_width=8`,
   `relation_width=64`, `latent_width=16`, `world_slots=8`,
   `maximum_world_reads=4`, `maximum_branches=64`,
   `maximum_recurrent_steps=4`, `maximum_update_backtracks=12`,
   `maximum_residual=2.0`, and `update_rate=0.5`;
3. `CompositeProspectiveCreditCore(credit_core, prospective_config)`;
4. `core.initial_state()` in CPU `torch.float32`; and
5. `DurableAbilityLearner(core, initial_state,
   checkpoint_identity=core.checkpoint_identity,
   prospective_lesion=False)`.

The canonical compact ASCII JSON record of that construction uses sorted keys
and separators `(',', ':')`, schema
`angler.frozen-qwen-learner-genesis.v1`. The literal hashed record is:

```json
{"checkpoint_constructor":"CompositeProspectiveCreditCore(StructureKeyedCreditMemoryCore(**credit_config), ProspectiveDynamicsConfig(**prospective_config))","credit_config":{"maximum_residual":4.0,"memory_slots":512,"rank":32,"relation_width":64,"temporal_width":8},"device":"cpu","dtype":"torch.float32","initial_state_constructor":"core.initial_state()","prospective_config":{"latent_width":16,"maximum_branches":64,"maximum_recurrent_steps":4,"maximum_residual":2.0,"maximum_update_backtracks":12,"maximum_world_reads":4,"relation_width":64,"temporal_width":8,"update_rate":0.5,"world_slots":8},"prospective_lesion":false,"schema":"angler.frozen-qwen-learner-genesis.v1","torch_manual_seed":2026083101}
```

Its domain-separated configuration identity is:

- learner-genesis configuration ref
  `sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7`;
- prospective core config ref
  `sha256:f3bab0f86a3671588a3bd66538448cf276e3a6e965d7d866f7698a092d1e7313`;
- learned-core checkpoint ref
  `sha256:5b2de1bb50091570dd92a790fe179cf3681f1202a1660d843863c46845f14f4c`;
- initial competence-state digest
  `sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286`;
- initial `DurableAbilityLearner.capture_state()` length 139,913 bytes and
  SHA-256 `93ab9deee7b0b67eabade8d43bb2b09e466f1dcd25405931d5a3d55eca91902d`;
- core parameter count 32,484.

The genesis-config hash is SHA-256 over
`angler.frozen-qwen-learner-genesis-config.v1<NUL>` followed by the canonical
record. The prospective core's code-level `ProspectiveDynamicsConfig.digest`
is the bare 64-character hex digest; the manifest and lineage deliberately
wrap that exact value with `sha256:` to form the config ref shown above. The
snapshot is materialized once at
`/opt/angler/state/project-angler/frozen-qwen-cognee-cycle-v1/learner-genesis.pt`.
An existing file is accepted only at the exact byte length and hash; otherwise
preflight aborts. A fresh construction must reproduce every identity above,
then restore from the exact snapshot and reproduce the state digest before the
canonical store is initialized. Restart must load the store's exact current
snapshot and lineage; it may not reseed or replace an advanced state with
genesis. The qualification manifest binds all five refs/digests above, the
seed, snapshot hash/length, construction record, and `prospective_lesion=False`.

The seeded construction runs in a direct, clean-environment helper subprocess
with CUDA hidden. Because the learner snapshot does not contain the frozen
core parameters required to reproduce its checkpoint identity, the helper
passes that exact CPU `state_dict` to the parent through a weights-only pipe
bounded at 2 MiB; it creates no second durable artifact. The GPU parent never
seeds: it constructs only the CPU topology shell in the same credit-first
order, installs the pipe-bound parameters, restores the exact genesis artifact,
and verifies all identities before store initialization. Receipt and feedback
restart reuse that learner topology and reload canonical store state without
calling `torch.manual_seed`.

## Qwen manifest and exact representation path

`FrozenQwenCycleManifestV1` binds:

- Qwen model and tokenizer refs above;
- the frozen learner-genesis configuration, prospective config, learned-core
  checkpoint, initial competence-state, and snapshot identities above;
- proposal schema `angler.qwen-procedure-proposal-prompt.v1`;
- execution schema `angler.qwen-selected-procedure-execution-prompt.v1`;
- greedy generation (`do_sample=False`), thinking disabled, cache enabled,
  batch one, maximum 4,096 input tokens and 64 new tokens;
- detached last non-padding Qwen backbone state in BF16 storage/FP32 consumer
  form;
- a versioned, manifest-seeded, label-free 2,560-to-64 frozen random
  projection whose exact SHA-256 counter/sign construction is covered by unit
  tests;
- `SituatedFeatureSpec(landmarks=(), include_acquired_ordinal=True)`, width 8,
  applied to the exact recalled Moving-Origin fields and mean-pooled without a
  deterministic evidence winner;
- neutral zero base logits;
- prospective latent width 16; and
- receipt observation schema `angler.qwen-receipt-observed-state.v1`.

Proposal generation accepts only one exact JSON object with one `procedures`
list containing exactly the requested number of distinct nonempty traces.
Malformed, duplicate, missing, or excess output fails before selection or
claim. It is never repaired.

Candidate relation encoding batches exact task/proposal segments and the exact
ordered recalled public evidence through the frozen model. Every candidate
conservatively cites the same nonempty sorted canonical recalled refs. The
fixed projection compresses representation only; it cannot inspect labels,
feedback, verifier state, task family, removal condition, or result.

The Qwen receipt observation encoder retains the passed SHA/top-53-dyadic
request/receipt-only construction and additionally domain-separates it by the
exact Qwen cycle manifest. It remains byte-repeatable after restart and cannot
accept objective feedback, predicted latent, task-family metadata, or
caller-supplied evidence.

For the live path the cycle must enforce exact Qwen adapter, relation adapter,
executor, and observation-encoder classes plus:

```text
cycle.model_ref == every component model_ref == manifest.model_ref
cycle.encoder_ref == relation/observation encoder_ref == manifest.encoder_ref
every component manifest_ref == manifest.manifest_ref
```

The existing exact synthetic encoder path and its tests remain unchanged.
Protocol-shaped encoder substitution remains rejected.

## Durable execution boundary

`SQLiteQwenExecutionJournal` is separate from the canonical cognitive store
and is keyed by the existing execution idempotency key. A row binds the exact
manifest, request canonical bytes/ref, prompt ref, generation configuration,
generated token IDs, response, selected trace, observations, and receipt
canonical bytes/ref. `put_if_absent` uses an immediate transaction and accepts
an existing row only when every field is identical. The exact receipt is
durable before `execute()` returns. Recovery returns `NOT_STARTED` or the
byte-identical `RECORDED` receipt; identity conflict fails closed.

The executor prompt receives only the public task, the learned-selected public
trace, and public input observations. It cannot see hidden alternatives,
prospective predictions, objective feedback, verifier internals, or learner
state.

## Exact outputs and write scope

Fresh repository files:

- `src/angler/runtime/qwen_cognitive.py`;
- `src/angler/memory/cognee_worker_protocol.py`;
- `src/angler/memory/cognee_subprocess_bindings.py`;
- `src/angler/memory/cognee_acquisition_worker.py`;
- `tests/unit/runtime/test_qwen_cognitive.py`;
- `tests/unit/memory/test_cognee_subprocess_bindings.py`;
- `tests/integration/memory/test_cognee_acquisition_live.py`;
- `experiments/runners/frozen_qwen_cognee_cycle_v1.py`;
- `docs/reports/FROZEN_QWEN_COGNEE_CYCLE_V1_RESULT.md` only after a completed
  qualification.

Narrow repository edits:

- `src/angler/runtime/situated_qwen.py`, only for backward-compatible frozen
  identity/token bounds and auditable generation records;
- `src/angler/runtime/prospective_observation.py`, only for the exact
  manifest-bound receipt encoder;
- `src/angler/runtime/cognitive_cycle.py`, only for the closed exact Qwen
  component admission and identity equations;
- `src/angler/runtime/__init__.py` and `src/angler/memory/__init__.py`, only for
  required exports;
- `tests/unit/runtime/test_cognitive_cycle_v3.py`, only for exact Qwen-path
  admission, restart, one-cycle, and fair-removal witnesses;
- this leaf, `THIRD_PARTY_NOTICES.md` only if the durable model-license artifact
  is obtained, and append-only `AGENTS_SYNC.md` at gate disposition.

Fresh external local outputs:

- verified model cache `/opt/angler/models/fastembed-cache-v1`;
- disposable state root
  `/opt/angler/state/project-angler/frozen-qwen-cognee-cycle-v1`, including
  the exact `learner-genesis.pt` artifact above;
- dedicated Cognee state root declared above;
- result `/opt/angler/results/frozen-qwen-cognee-cycle-v1.json`;
- bounded scratch below
  `/opt/angler/work/frozen-qwen-cognee-cycle-v1`.

No other tracked or untracked path, model bundle, checkpoint, result,
experiment identity, threshold, store, Cognee dataset, or external service may
be changed. The intentionally dirty worktree must not be reset, cleaned,
discarded, or normalized.

The live runner proves non-drift for the repository, Qwen bundle, FastEmbed
cache, and Tiktoken cache, and accounts bytes below its declared state, Cognee,
scratch, and result roots. It does not claim whole-filesystem syscall
observation; the exact clean environments instead pin every configured runtime
cache and temporary directory below a declared qualification root.

## Resource ceilings and stop conditions

Qualification ceilings:

- one Qwen process and loaded model copy, RTX 5080 only, BF16 inference;
- preflight at least 14,336 MiB free on the RTX 5080;
- peak CUDA allocated and reserved memory each at most 12,288 MiB;
- process RSS at most 20 GiB;
- configured compute pools total at most eight: parent Torch intra-op two and
  inter-op one, Cognee FastEmbed/ONNX Runtime intra-op two and inter-op two;
  Cognee embedding threads therefore remain at most four. The parent Linux
  task count includes interpreter/runtime housekeeping and is recorded
  observationally, not misreported as this configured compute-pool ceiling.
  Cognee `OMP_NUM_THREADS=2` is an overlapping native-library bound, not an
  additional independently scheduled pool;
- batch size one, input at most 4,096 tokens, generation at most 64 tokens,
  total sequence at most 4,160 tokens;
- candidate count at most eight per live cycle, recall at most 12;
- at most 16 model-generation calls and eight live cycles;
- wall time at most 900 seconds after Qwen load;
- new scratch/evidence at most 1 GiB;
- existing pending-state ceiling 16 MiB, projection retry default 64, store
  page ceiling 256, acquisition neighbor/provenance ceilings 384 unchanged;
- IPC message at most 4 MiB and bounded stable errors without unrestricted
  tracebacks.

Abort at the current safe transaction boundary on model/tokenizer/embedding,
software, GPU, dataset, NodeSet, manifest, checkpoint, prompt/generation,
request/receipt, or canonical-store drift; network/telemetry/provider attempt;
CUDA OOM; non-finite tensor; malformed proposal; empty response; journal
conflict; Cognee scope/join failure; resource breach; model mutation; or an
unexpected file write. Preserve canonical state and the exact failure record.
Do not silently replan, change widths, reduce candidates, change tokens, alter
the task, repair model output, or rerun a consumed scientific identity.

Rollback is model unload, worker close, canonical store and journal
preservation, projection-view invalidation when required, and dataset-only
forget for this qualification-owned Cognee namespace. The source Qwen bundle,
durable FastEmbed cache, and pre-existing repository work remain unchanged.

## Required tests and live qualification

CPU/fake-backed acceptance requires:

1. manifest/ref mutation detection, byte-exact learner-genesis reconstruction
   and restart binding, and exact model/evaluator bindings;
2. frozen/eval enforcement and token/resource rejection without truncation;
3. strict proposal parsing and 2/7/64 cardinality with no repair;
4. detached finite relation tensors, neutral logits, exact support refs, and
   real Moving-Origin temporal fields;
5. selected trace only in execution prompt;
6. journal-before-return, byte-identical restart recovery, conflict and
   concurrency rejection;
7. exact receipt observation framing and restart reproduction;
8. exact Qwen component admission and forged/mixed-manifest rejection;
9. one fake-backed successor cycle committing one episode, resolution, child
   lineage, and acquisition after external feedback;
10. unchanged four-way removal fixtures sharing one exact frozen hit/model/
    manifest/request/proposal input set;
11. IPC framing, size, ordering, timeout, crash, environment allowlist, and
    scope tests, plus fake-backed projection retry and destructive-rebuild
    invalidation/recovery; and
12. unchanged prospective/store/acquisition/legacy regression suites plus
    compile, export smoke, bounded no-task-solver scan, and `git diff --check`.

The opt-in live Cognee qualification uses three local synthetic canonical
records in the one qualification-owned dataset. It proves handshake
identity (including modern non-legacy dataset ID, user/tenant, access control,
and dataset context), duplicate projection idempotency, object-backed graph and
string-tagged vector NodeSet scope, lower-is-better reference-only search,
ordered exact-coverage canonical rejoin, proposed denial, and observed admission,
worker restart, exact dataset recreation, and absence of
`remember`, `cognify`,
`improve`, LLM extraction, telemetry, external provider, GPU, or network use.
The worker deliberately exposes authority for exactly one frozen dataset, so
adding a second sentinel dataset would broaden its destructive scope merely to
test scope it does not possess. Dataset-only forget, exact UUID re-creation,
and rejection of any different dataset identity are the proportional live
scope witness; fake-backed tests retain the cross-target denial cases.
The real Cognee API exposes no public graph/vector/join fault-injection seam;
injecting backend corruption or adding a worker fault-control operation is not
authorized merely to duplicate the mandatory fake-backed failure witnesses.

This leaf does not claim a live `VALIDATED` canonical acquisition witness:
the current exact `CognitiveAcquisition` source-binding contract permits
`PROPOSED` counterfactual batches and `OBSERVED` episodes/resolutions but
rejects `VALIDATED` for every supported source. Extending that shared contract
is outside this LOW integration leaf and is unnecessary for the observed
feedback path; the test must expose this limitation rather than fabricate a
non-round-tripping record.

The bounded integrated live qualification then proves one complete cycle,
receipt-only and feedback-staged restart, exact learner update/commit, Cognee
projection retry/rejoin, Moving-Origin advancement, frozen Qwen tensor digest
before/after, model/embedding manifest before/after, CUDA/RSS/time ceilings,
non-drift of guarded immutable inputs, and byte accounting within declared
outputs. It retains the complete proposal-generation preimage and reparses its
raw response against the exact reserved proposals.

## Gate

`ANG-GATE-RUNTIME-FROZEN-QWEN-COGNEE-CYCLE-V1-001` passes only if every
focused/static check and both live qualifications pass from the exact frozen
identities, all canonical/restart/resource assertions hold, Qwen tensors and
model manifests are unchanged, learner checkpoint/config/state lineage never
drifts or resets on restart, and an independent review finds no task solver,
authority confusion, evidence laundering, or undeclared effect.

A malformed Qwen proposal or response may produce a preserved technical
failure without condemning the architecture; it may be debugged only under a
fresh qualification attempt identity and unchanged scientific thresholds.
This technical gate is not a scientific evaluation gate. Qwen-only,
retrieval-only, frozen-origin, prospective-removal, backend-removal, and full
high-level comparisons require a separate precommitted evaluation identity
after this gate. No threshold may be chosen or changed after seeing that
evaluation's adaptive results.

Gate disposition: `TECHNICAL_PASS`. The final relevant CUDA-hidden matrix
passed 286 tests with one opt-in skip and 256 parameterized subtests in 36.68
seconds. The separate networkless live Cognee qualification passed 1 test in
8.31 seconds. Independent final review found no task solver, current-turn
verifier leakage, authority confusion, evidence laundering, undeclared effect,
or remaining focused/static blocker.

The integrated identity
`angler.frozen-qwen-cognee-cycle.v1-qualification` completed once on the RTX
5080 and is consumed. The preserved result is
`/opt/angler/results/frozen-qwen-cognee-cycle-v1.json`, SHA-256
`303810725cd0fc818abea7856a90cf9e0a931ed3f4fdf58dd80a0867de0022a6`.
It records exactly two Qwen generation calls, four cycle/worker starts, a
canonical commit with projection pending after the third worker closed, two
bounded retry acknowledgements after fourth-worker rejoin, no pending
projection afterward, an advanced exact learner lineage, and
`FORGOTTEN_AND_RECREATED` dataset cleanup. The Qwen aggregate tensor digest is
identical before/after. Peak allocation/reservation were 8,115,036,160 /
8,139,046,912 bytes, peak RSS was 9,078,042,624 bytes, and post-load wall time
was 27.785613846 seconds, all below frozen ceilings.

The exact runner SHA-256 is
`93e4f1a31cee5f4b738c647a888bea26601b3d9ed7f83e93bfc4557ce487d202`.
The bounded result report is
`docs/reports/FROZEN_QWEN_COGNEE_CYCLE_V1_RESULT.md`, SHA-256
`be371fc10826354661fba8a2682ff73e6aff3e1d001b9b653043a9d3ce398ca0`.
This disposition establishes only the accountable technical outcome above;
all scientific and capability limits remain in force.

## Next exact action

Do not rerun or tune this consumed qualification identity. Activate a separate
experimental SCIENCE work leaf that freezes a fresh high-level multi-domain
task/partition/evaluation identity, equal-information and equal-budget
Qwen-only/retrieval-only controls, full/frozen-origin/prospective-removal/
backend-removal interventions, multi-turn objective feedback, and all metrics
and thresholds before adaptive results are observed. Preserve this technical
store/result as evidence, not as the scientific evaluation's starting state.
