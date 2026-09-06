# ANG-WORK-RUNTIME-COGNEE-NAMESPACE-SCOPE-V2-001

Status: technical PASS; additive namespace-integrity component only; no live
Cognee identity consumed

Tier: 4 runtime integrity work leaf

Active node: `ANG-BP-RUNTIME`

Parent: `ANG-BP-RUNTIME`

Predecessor:

- `ANG-WORK-RUNTIME-FROZEN-QWEN-COGNEE-CYCLE-V1-001` — technical PASS;
  accepted leaf SHA-256
  `8cf7d3bc0b5b9c43040a0c71db8e0cefe6e2c6078d93f8199f96321d60970c39`.

Consumer:

- `ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-001` — its
  evaluator-only construction may proceed, but live runner construction and
  execution remain blocked until this gate passes and successor hashes are
  frozen there.

Gate: `ANG-GATE-RUNTIME-COGNEE-NAMESPACE-SCOPE-V2-001` — PASS

Decisions: `ANG-ADR-0006-MOVING-ORIGIN-COGNEE-SITUATED-MEMORY` and
`ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES`.

## Authorization and impact

Assurance is LOW. The owner authorizes one additive, local, deterministic
runtime-integrity change and CPU/fake-backed tests on the dedicated Ubuntu
workstation. The leaf may make the already-qualified Cognee worker's dataset,
tenant, NodeSet, and state root an explicit immutable launch scope. It may not
run a model or GPU, access a network, install dependencies, ingest real-person
or recovered data, contact a provider, mutate a consumed namespace, change
Cognee/FastEmbed identities, broaden worker operations, or make a scientific,
promotion, readiness, or capability claim.

This is necessary deterministic plumbing. It selects no task answer, plan,
candidate, threshold, or model behavior. The Human-Flourishing risk is limited
to local resource use and accidentally targeting the wrong disposable memory
namespace. Exact launch scoping, validation before subprocess creation,
one-scope-per-worker immutability, dataset-only forget, network isolation,
bounded tests, and owner stop/deletion control keep the work LOW and mapped to
`ANG-GATE-HUMAN-FLOURISHING-001`. This mapping is not a normal protected
assessment or promotion authorization.

## Accountable outcome

Provide a backward-compatible `CogneeWorkerScope` boundary so each worker has
authority over exactly one explicitly admitted disposable namespace for its
lifetime, while the no-argument legacy path retains the predecessor's exact
dataset, tenant, NodeSet, state root, protocol operations, provider settings,
and dataset-only cleanup semantics.

The scoped boundary must allow the downstream experiment to allocate distinct
fresh namespaces below
`/opt/angler/state/project-angler/high-level-multidomain-v1/cognee/**` without
touching `/opt/angler/state/project-angler/cognee-acquisition-live-v1` or any
consumed store/result.

## Frozen inputs

| Input | Predecessor SHA-256 |
|---|---|
| worker protocol | `8e57366797a1e2ef382254175ce1ff6518914f1a9fd8bb038e6b3d22749a44e9` |
| acquisition worker | `28668480bc9ecc7c76cb5aae5a6156bac0a5aacf9a05393bb4255282254bc580` |
| subprocess binding | `ac316646326f0fa36a6f57b91703cb0f143aa8f98fc080e2b9078e6c7b410a58` |
| focused unit tests | `c7877e4d71bdbbbe312f2541a30253a732198559a559cf0c2ca02b66b74beaba` |
| live boundary test | `8a6a282541b9bf441ea8ad88c8512e0e6fdf29e9a0a61d5467be8b4277fca88b` |

Cognee 1.5.3, its Python 3.12.3 environment, FastEmbed/ONNX identities,
Tiktoken/model caches, CPU-only provider, two intra/inter-op threads, exact
clean environment, `unshare --net`, JSON-lines framing, graph/vector schema,
reference-only search, canonical rejoin, and dataset-only forget behavior are
unchanged.

## Exact scope contract

The standard-library-only protocol module owns an immutable exact
`CogneeWorkerScope` value with these fields:

- `dataset_name`;
- `tenant_name`;
- `node_set_name`;
- `state_root`;
- derived `node_set_id` using the existing Cognee-compatible UUID5 formula;
- derived `worker_cwd = state_root / "worker-cwd"`; and
- derived `worker_tmpdir = state_root / "worker-tmp"`; and
- derived scope marker
  `state_root / ".angler-cognee-worker-scope-v1.json"` using marker schema
  `angler.cognee-worker-scope.v1`.

The legacy default is exactly:

```text
dataset_name = angler-acquisition-live-v1
tenant_name = angler-acquisition-live-v1-tenant
node_set_name = angler-acquisition-live-v1-records
state_root = /opt/angler/state/project-angler/cognee-acquisition-live-v1
```

Admission fails before subprocess creation unless:

1. every name is exact NFC ASCII lowercase text of 1 through 128 characters
   matching `[a-z0-9]+(?:-[a-z0-9]+)*`;
2. the tenant and NodeSet names are respectively exact
   `<dataset_name>-tenant` and `<dataset_name>-records`, each still within the
   length bound;
3. `state_root` is an absolute, lexically normalized path strictly below
   `/opt/angler/state/project-angler`, contains no `.` or `..` component, and
   is neither the allowlist root itself, the grandfathered legacy root when
   supplied as a custom scope, nor another scope's path;
4. every existing ancestor from the allowlist root through the scope root is a
   real directory rather than a symlink, and the created scope/cwd/tmp
   directories resolve below that same root with UID/GID `1000:1000` and mode
   `0700` where the predecessor already requires those properties; and
5. the exact scope fields and derived values fit the frozen IPC/environment
   bounds without truncation or normalization.

Before a custom worker launches, the binding create-once writes canonical JSON
containing exactly its dataset, tenant, NodeSet, state-root, and derived NodeSet
identity to the scope marker with UID/GID `1000:1000` and mode `0600`. An
existing marker is accepted only if its canonical bytes and metadata match
exactly; missing-after-creation, symlinked, malformed, mismatched, or
wrong-metadata markers fail closed. The legacy default is grandfathered and
must neither require nor write a marker, preserving consumed state byte-for-byte.

One binding start receives either an explicit `scope=` or the exact legacy
default. The binding passes the admitted scope through the clean worker launch
boundary, sends the same dataset/NodeSet values in `hello`, and validates that
the worker reports the exact scope. The scope becomes immutable binding state;
no request can replace, switch, infer, or select it. All point translation,
modern dataset-ID validation, search-envelope validation, recreation, and
cleanup use that bound scope rather than module-global legacy names.

The worker receives scope data only through a bounded launch configuration
admitted before any Cognee import. It reconstructs and independently validates
the exact same scope before configuring Cognee. Missing, partial, extra,
malformed, mismatched, or inherited scope configuration fails closed. Scope
data are identifiers and local paths, never secrets.

Protocol operations remain exactly `hello`, `project`, `search`,
`forget_namespace`, and `close`. No list-all, arbitrary dataset ID, arbitrary
root deletion, global
forget, pruning, mutation of another scope, multi-scope worker, provider,
generation, answer, or fault-control operation is added. `PROTOCOL_VERSION`
may remain `1` because message shapes and semantics are unchanged; if an IPC
field changes, it must advance and tests must prove mismatched-version denial.

## Explicit scope and non-goals

In scope:

- the immutable scope value, validation, derived identities, and exports;
- scope-bound clean environment/argv/cwd/tmp launch;
- replacing legacy module-global namespace reads in binding and worker paths
  with the one admitted scope;
- preserving exact default behavior and existing API callers; and
- fake-backed regression, custom-scope, cross-target, path, environment,
  request, response, crash, and export tests.

Out of scope:

- Qwen, Moving Origin, learner, task evaluator, runner, prompts, selection,
  feedback, store schema, acquisition semantics, Cognee package configuration,
  model/provider/cache identity, search scoring/order, a namespace pool, worker
  reuse across scopes, per-query scope selection, concurrent multi-tenant
  authority, broad filesystem cleanup, or a live scientific run.

No ADR or interface-registry edit is required: this leaf preserves the existing
Cognee projection semantics and adds an internal runtime launch parameter. Any
new cross-branch contract, operation, authority, or cleanup semantic requires a
separate ADR and is not authorized here.

## Exact outputs and write scope

Repository writes are limited to:

- this leaf;
- `src/angler/memory/cognee_worker_protocol.py`;
- `src/angler/memory/cognee_subprocess_bindings.py`;
- `src/angler/memory/cognee_acquisition_worker.py`;
- `src/angler/memory/__init__.py`, only for the new scope export;
- `tests/unit/memory/test_cognee_subprocess_bindings.py`;
- `tests/integration/memory/test_cognee_acquisition_live.py`, only if a
  parameterized custom-scope regression is required; and
- one append-only disposition in `AGENTS_SYNC.md` after the gate.

CPU/fake tests may write only private temporary directories. This leaf does not
authorize creation or mutation of either the legacy live Cognee root or the
downstream experiment root. All edits use `apply_patch`; formatting may
mechanically rewrite only these files. No unrelated tracked or untracked work
may be changed, and the intentionally dirty worktree must not be reset, cleaned,
discarded, or normalized.

## Acceptance gate and evidence

`ANG-GATE-RUNTIME-COGNEE-NAMESPACE-SCOPE-V2-001` passes only when all are true:

1. exact legacy construction, launch values, hello validation, point NodeSet,
   modern dataset ID, search envelope, and cleanup semantics remain unchanged;
2. an explicit valid custom scope reaches both sides of the fake transport and
   all binding evidence/translation uses only its names and derived IDs;
3. malformed names; derived-name overflow; relative, broad, non-normalized,
   escaping, symlinked, wrong-owner, or wrong-mode paths; partial/extra launch
   configuration; worker/parent mismatch; and cross-target response data all
   fail before any affected operation;
4. a started binding cannot switch its scope, and two bindings with different
   scopes share no mutable scope state;
5. project/search/forget remain limited to the one bound dataset and NodeSet;
   no arbitrary destructive target is representable;
6. existing timeout, cancellation, crash/EOF invalidation, environment,
   provider/thread, framing, canonical payload, proposed-denial, reference-only
   recall, and export tests still pass;
7. the three changed modules compile, `src.angler.memory` export smoke passes,
   the focused CPU-only suite passes with CUDA hidden, a bounded static scan
   finds no new network/provider/global-forget/task-solver path, and
   `git diff --check` passes; and
8. an independent read-only audit finds no namespace confusion, legacy drift,
   authority broadening, unsafe path handling, consumed-state mutation, or
   scientific/evidence overclaim.

No live Cognee run is necessary to pass this additive component gate. The
downstream leaf's separately frozen live qualification must exercise one fresh
custom namespace before its final evaluation can be admitted.

## Gate disposition and evidence

`ANG-GATE-RUNTIME-COGNEE-NAMESPACE-SCOPE-V2-001` passes this additive
component. The exact legacy scope retains its predecessor argv and environment,
requires no marker, and makes no additional write. A custom scope is immutable
for one worker lifetime, is marker-bound below the allowlisted project state
root, and constrains hello, point translation, dataset identity, search,
recreation, and `forget_namespace` to the admitted dataset and NodeSet. This
passes runtime integrity only; it creates no scientific, capability, promotion,
or live-service evidence.

With CUDA hidden, the focused fake-backed suite passed 46 tests plus 24
parameterized subtests, and the complete memory unit suite passed 113 tests plus
44 parameterized subtests. Protocol, binding, and worker compilation; public
export and no-parent-Cognee-import probes; bounded forbidden-pattern scans;
temporary marker/path/scope-mismatch probes; and scoped whitespace checks all
passed. Independent read-only audit found no namespace confusion, legacy drift,
authority broadening, unsafe path handling, consumed-state mutation, or
scientific overclaim.

Final SHA-256 values are:

- worker protocol: `eed327f4d3567356b33cba262b2a96ea9d8dcfc43b0f9b15bb976b2357eb9394`;
- subprocess binding: `1295eca35268073164f12f95f2ed1782a9413da743a86becc4c4b3708c822cc0`;
- acquisition worker: `36925c251fb20726c3f860522c6faaca8c37928997ede30ca0cb29aaa259aa69`;
- memory export: `4bc67a8fc5a1089ef5f3f810e6de77747102d0ecf893fbbc43954dfa4544eddc`;
- focused tests: `e956e9e2d63948dbbc13fa1c226dcc8fe994cb82ebeb8fbfeb78a8d3e513de2c`;
  and
- unchanged live-boundary test:
  `8a6a282541b9bf441ea8ad88c8512e0e6fdf29e9a0a61d5467be8b4277fca88b`.

The independent audit verified that the legacy live-state metadata and content
manifests remained byte-identical, and that the downstream
`high-level-multidomain-v1` state root remained absent. No live Cognee, Qwen,
GPU, network, evaluation seed, experiment identity, result, threshold, or
consumed artifact was invoked or changed.

## Failure, rollback, and next action

Any scope/path/environment mismatch, test regression, or authority broadening
fails the gate. Rollback is omission of the new explicit scope call path; the
legacy predecessor files and consumed evidence remain preserved. No dataset or
state root is deleted by source rollback.

The exact next action is for the downstream science leaf to freeze these
successor hashes and construct its CPU/fake-tested runner. Live Cognee remains
reserved for that leaf's separately admitted qualification; this component
leaf must not start Qwen, a GPU, a science identity, or a final evaluator.
