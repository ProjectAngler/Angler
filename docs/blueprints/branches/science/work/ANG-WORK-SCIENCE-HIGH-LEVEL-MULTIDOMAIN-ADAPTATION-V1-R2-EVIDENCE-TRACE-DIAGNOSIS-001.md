# ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-EVIDENCE-TRACE-DIAGNOSIS-001

Status: complete; create-once result and independent post-run reconstruction
accepted, with the predecessor scientific classification unchanged

Tier: 4 evidence-only diagnostic work leaf

Active node: `ANG-BP-SCIENCE`

Parent: `ANG-BP-SCIENCE`

Gate:
`ANG-GATE-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-EVIDENCE-TRACE-DIAGNOSIS-001`
— PASS

## Accountable outcome

Reconstruct one deterministic, privacy-preserving trace matrix for the three
successful final `QWEN_ONLY` R2 attempts, their exact same-task peers across
the other six arms, and the relevant `FULL` adaptation/recall ancestry. The
diagnosis must distinguish what the consumed evidence establishes, merely
supports, rules out, or cannot identify before any runtime change or fresh
experiment is proposed.

Gate success means exact evidence reconstruction and bounded interpretation.
It does not require finding a root cause and cannot alter the terminal R2
classification.

## Immutable predecessor and scientific boundary

The consumed predecessor is
`ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-LINEAGE-SEQUENCE-RECOVERY-001`
at SHA-256
`a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455`.
Its terminal result, result report, evaluation ledger, arm-factory audit, and
source manifest are frozen inputs:

| Input | SHA-256 |
|---|---|
| R2 terminal result | `c2316da22208cf0a8693f32f4ea28060c529329df1c6d7d67841b56720aca676` |
| R2 result report | `4aae9d581333eebc468ed5c44f52c62b3784cc98ac6896afada70628c95960d1` |
| Evaluation ledger | `914e1695cc4c9187a60f48a035632dbe5e849056e113bab4295ca36f5bbb4eca` |
| Arm-factory audit | `eb6f4e16336f83b4cee95ecc9ec4803af99045dd1ba904373599db8a3f0b4e82` |
| R2 source manifest | `842e39e713c87352a9dbc5e04c8cd73ab74a1b32fceb9aaa72c1d1fd75a4344d` |
| Frozen R2 runner | `fbd10104ee566e327e812f24e6ea44d30e405d4a2dea2650ebd94dad58fee574` |

The terminal classification remains `NOT_SUPPORTED`. R2 is permanently
consumed: no rerun, new generation, seed access, threshold change, hidden
rejudging, or reinterpretation as a capability pass is permitted.

## Inherited invariants and proportional assessment

All SCIENCE and root invariants apply, especially evidence separation,
causal attribution, outcome-only judging, stable foundation, external
authority, and truthful claim limits. Deterministic code is permitted only
for generic parsing, hashing, equality/provenance comparison, redaction,
resource measurement, and integrity enforcement. It may not solve a task,
repair a response, score an unjudged candidate, infer a hidden answer, or
select a procedure.

This leaf is LOW impact under
`ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001`: it reads only preserved local synthetic
evidence, writes one redacted local audit, invokes no model/GPU/network/live
Cognee process, changes no state, and has no user, subject, external action,
deployment, or promotion. Principal risks are disclosing evaluator-private
material and overstating observational evidence. Exact read scope, forbidden
fields, fail-closed redaction, immutable-input checks, independent review,
and create-once output mitigate them. Inaction preserves uncertainty; a
smaller manual inspection would be less reproducible. The work remains
reversible until its create-once result and preserves meaningful human
control throughout. This assessment maps the leaf to
`ANG-GATE-HUMAN-FLOURISHING-001`; no high-impact review trigger is present.

No ADR or interface-registry change is required. The diagnostic schema is a
leaf-local evidence format, not a shared contract, permission, promotion
rule, threshold, visibility expansion, or runtime boundary.

## Literal read scope

Repository reads are limited to:

- `AGENTS.md`;
- `docs/blueprints/ROOT_CAPSULE.md`;
- `docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md`;
- `docs/blueprints/PROTOCOL.md`;
- the SCIENCE capsule, blueprint, and status;
- the active predecessor leaf, R2 result report, frozen R2 runner, and R2
  source manifest; and
- the three source-manifest-frozen schema/record definitions required to
  interpret the authorized cognitive SQLite inputs without guessing:
  - `src/angler/runtime/cognitive_transaction_store.py`, SHA-256
    `4c8add9e1b49649b381b0f4dad439709f7b4691cec514c88b39b386bc8a28c0a`;
  - `src/angler/memory/cognitive_acquisition.py`, SHA-256
    `e93c9f181fe1938a4c033a55a88827ba3d0ca5b8b9f46cb1b5569bc4987d36e7`;
  - `src/angler/cognition/contracts.py`, SHA-256
    `cb2fe5d597c10550425a03951aaecf2e9b104065e6ea32adaba30d1d61b1eb77`;
  these sources may supply only persisted schema, canonical-record, and
  provenance semantics and may not be imported as a runtime; and
- the Human-Impact contract and Human-Flourishing gate referenced above.

Live evidence reads are limited to:

- `/opt/angler/results/high-level-multidomain-v1-r2.json`;
- `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/attempt-evidence.sqlite3`,
  opened only with SQLite `mode=ro`, `immutable=1`, and `query_only=ON`;
- `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-factory-audit.json`; and
- these four evaluation lineage databases, opened under the same immutable
  SQLite restrictions:
  - `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/102e3196e05f0ce031e8585c2eef0f94b48acf1e91e272bc998d5b0cbc30a395/cognitive.sqlite3`,
    SHA-256
    `54a58c587c2c79ec27999ff8f9c48e1607f64e4f7c8d1fcca3e17af6f2d72a57`;
  - `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/75cadc260b52f873a7893ea7c18a3bb027ed25d98fabb300942e429ac08df706/cognitive.sqlite3`,
    SHA-256
    `c27f14fce291334eb785cd02ca77560021a61546c5cabbc0c0db87738d5bd022`;
  - `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/e4acb76aae39f554ee54ad98854eb67f83c7ffc7a0ab2961c7709d31bfb248de/cognitive.sqlite3`,
    SHA-256
    `748a7499667382c754fa389b15c76af57e20f29bacab25a30d51200f7205cfda`;
  - `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/f4f478b15524d0d88cb6363ae2484c05acfbc0c608d8ab1153cdebc01f324312/cognitive.sqlite3`,
    SHA-256
    `534191cc6e404da2880681074906f550a91c811e5d80df45442212eb834dc9fe`.

The sealed seed artifact, raw seeds, evaluator-private solutions/mechanisms/
routes, model files, learner snapshots, journals, qualification databases,
and unrelated V1/R1 or sibling state are excluded. The result and databases
may be parsed in memory, but forbidden raw fields may not be printed, logged,
copied, or persisted.

## Literal write scope and local output schema

Repository writes are limited to:

- this leaf;
- `experiments/runners/high_level_multidomain_v1_r2_trace_diagnostic.py`;
- `tests/unit/experiments/test_high_level_multidomain_v1_r2_trace_diagnostic.py`;
- `docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R2_TRACE_DIAGNOSIS.md`
  only after a terminal diagnostic result; and
- one append-only diagnostic disposition in `AGENTS_SYNC.md` only after
  independent result review.

The sole live write is create-once mode-`0600`:

`/opt/angler/results/high-level-multidomain-v1-r2-trace-diagnosis.json`.

Publication must canonicalize and validate the complete output bytes in
memory, enforce the 1 MiB ceiling, then open that exact final path once with
`O_CREAT | O_EXCL | O_WRONLY` and mode `0600` (`O_NOFOLLOW` where available).
It must reject a pre-existing path, symlink, non-regular file, or link/owner/
mode drift; fully write and `fsync` the exclusive descriptor; and verify the
closed final file's size, digest, mode, ownership, regular-file type, and
single link. No temporary, rename, replace, truncation, overwrite, or
post-publication chmod path is permitted. Any failure after exclusive
creation preserves that file as consumed diagnostic failure evidence.

Its identity is
`angler.high-level-multidomain.v1-r2-posthoc-trace-diagnosis.v1`. Output is
limited to schemas/identities, predecessor hashes, task/receipt/evidence
references, enums, booleans, counts, indices, lengths, equality relations,
provenance classes, before/after input hashes, resource observations, and
question dispositions `ESTABLISHED`, `RULED_OUT`,
`CONSISTENT_WITH_EVIDENCE`, or `INDETERMINATE`.

It must never contain raw tasks, prompts, proposals, selected traces,
responses, recalled text, token IDs, seeds, evaluator answers, solutions,
mechanisms, routes, database rows, or reversible encodings of them. Existing
R2/V1/R1 files, reports, manifests, ledgers, databases, state, and scratch may
not change.

## Diagnostic questions and acceptance tests

The implementation must answer these falsifiable, cohort-scoped questions:

1. Does the immutable evidence contain exactly three successful final
   `QWEN_ONLY` task IDs and exactly seven finalized same-task arm attempts for
   each, for a unique 21-attempt matrix?
2. For each attempt, where is the first evidenced divergence among proposal
   admission, history/recall preimage, candidate class, selection, response
   conformance, objective judgment, and score?
3. Can proposal-parser failure be ruled out as the proximate explanation for
   any admitted attempt, without judging unexecuted alternatives?
4. Which arms share exact recall/proposal preimages, selections, responses,
   and outcomes, and what do the frozen removal interventions rule out only
   for these three tasks?
5. Do target `FULL` recall references rejoin exact retained acquisition and
   adaptation ancestry, and was objective success associated with a selected
   procedure whose declared response-grammar class differs from the executed
   response class?
6. Which proposed explanations remain observational hypotheses because R2
   lacks the isolating intervention?

Acceptance requires:

- all frozen hashes, 468 unique finalized receipts, terminal result, and
  cohort cardinalities rejoin before analysis;
- every emitted reference rejoins source evidence, and every conclusion cites
  only those references and its declared evidence strength;
- task-family response conformance uses only the already-declared public
  output grammar, never a task solution or semantic correctness rule;
- synthetic unit fixtures cover unique/non-unique cohorts, admitted and
  malformed proposals, identical/different preimages, selection/response
  divergence, missing provenance, forbidden-field redaction, hash drift,
  create-once refusal, and byte-identical canonical output;
- static checks reject model/evaluator/runtime/Cognee invocation, network or
  subprocess use, task-solving constants, response repair, raw-field output,
  and access outside the literal paths;
- all input hashes and file metadata are identical before and after, no
  SQLite sidecar or temporary output appears, and no model/GPU/Cognee process
  is started; and
- independent review validates the runner, tests, create-once result,
  redaction, reconstruction, and established-versus-hypothesis boundary.

The gate passes even if every candidate mechanism remains `INDETERMINATE`.
It fails if an observational difference is presented as a causal effect.

## Terminal evidence and disposition

The create-once diagnostic result is
`/opt/angler/results/high-level-multidomain-v1-r2-trace-diagnosis.json`,
SHA-256
`4906c3a77e851012ec2cc12172016ef409ad68c9f1efa4455bb1d570b19eb693`,
100,017 bytes, mode `0600`, owner UID/GID `1000:1000`, and one link. The
frozen terminal runner and test SHA-256 values are respectively
`79068362cb5f30c43e70b6f8d73167eaf2e6c95850de045644f900aea006eeb7`
and
`39fe6f2b67bfa119a64830d6a877c9b48986c810029e3a437b4f4313dee12c41`.
The accepted pre-run leaf hash recorded by the result is
`eedc848394a088e347db68db23f24fabab409315a0775747597edc76abd5f960`.

The final CUDA-hidden synthetic suite passed 31 tests plus 16 parameterized
subtests. Two independent post-run audits accepted canonical publication,
closed-schema validation, the 15 unchanged input witnesses, exact
reconstruction outside intrinsically process-local resource measurements,
and the fail-closed raw-material scan over 1,280 forbidden values. No raw
evaluator or model material leaked.

The result rejoined 468 finalized receipts, 433 admitted plus 35 malformed
attempts, 468 factory dispositions, 60 removal rows, four persistent
lineages, and the exact three-task/21-attempt target matrix. All 36 target
recall occurrences rejoined 24 unique scope-local retained records across two
`FULL` lineages; one repeated raw record reference across scopes confirms the
scope-local provenance key.

For all 18 history-bearing target peers, the first observed divergence from
the successful `QWEN_ONLY` attempt was the history preimage. All were admitted
and all failed. For each task, `FULL`, `RETRIEVAL_ONLY`, `FROZEN_ORIGIN`,
`PROSPECTIVE_REMOVAL`, and `BACKEND_REMOVAL` shared exact recall, proposal
request, and proposal preimages. The three `QWEN_ONLY` attempts selected
grammar-conformant procedures, executed them exactly, and succeeded. This
supports a history-conditioning bottleneck as an observational hypothesis;
it does not causally identify memory content, formatting, relevance, Cognee,
Moving Origin, prospective state, or learned lineage as the mechanism.

Question dispositions are Q1 `ESTABLISHED`, Q2 `ESTABLISHED`, Q3
`RULED_OUT`, Q4 `CONSISTENT_WITH_EVIDENCE`, Q5 `RULED_OUT`, and Q6
`INDETERMINATE`. Full FrozenRecallBatch and prospective-contract
recomputation remain explicitly unperformed. Detailed evidence and bounded
interpretation are in
`docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R2_TRACE_DIAGNOSIS.md`,
SHA-256
`02d478ab96481f5eb97504dfcda3a66211f78da8651f8f5ad11003978f1ef1c3`.

Three earlier attempts stopped before exclusive output creation on exact
metadata, evaluator replicate-label, and removal selection-reference
integrity mismatches. Each was corrected only to match frozen producer/input
identity semantics, with direct regression evidence and fresh review. No
task, score, threshold, arm, seed, learned state, or scientific result was
changed. The fourth attempt created the sole terminal output. This diagnostic
identity is now consumed and may not be rerun.

## Resource, stop, and rollback boundaries

Construction and unit tests run CPU-only with CUDA hidden. The live diagnostic
uses one process, no subprocess, no network, no model, no GPU, no Cognee
worker, at most 120 seconds wall time, 1 GiB RSS, and 1 MiB final output.

Stop before output on any input/hash/permission/link drift, non-unique cohort,
missing receipt or provenance, forbidden-field path, pre-existing result,
SQLite sidecar, model/network/GPU activity, or resource-ceiling risk. A
post-open mutation, forbidden output, partial durable result, or failed
immutability check consumes the diagnostic identity and is preserved as
failure evidence; any retry requires a new versioned diagnostic identity.

Before a durable result, rollback may remove only the new unconsumed runner,
test, and leaf. After a result exists, preserve all diagnostic bytes and add
only the report/disposition after independent audit. R2, R1, and V1 are never
rollback targets.

## Next exact action

Create and independently review one fresh Tier-4 SCIENCE leaf for a
history-conditioning isolation experiment with new identity and seeds. It
must compare no history, ordinary frozen history, an equal-slot/equal-budget
neutral format control, a precommitted shuffled or cross-family history
control, and a same-family/relevance-stratified experimental intervention,
while retaining the necessary integrated and removal controls. Freeze prompt
and token equality, record-selection rules, objective evaluator, resource
ceilings, thresholds, and stop conditions before any model result is seen.

A generic tri-state selected-to-executed congruence witness may be included
in shadow evidence only. This result does not justify making it a corrective
filter or positive-credit gate. The successor may not reuse R2 identities,
tune against the three diagnosed tasks, encode a deterministic task solution,
or claim that the observed history association is already causal.
