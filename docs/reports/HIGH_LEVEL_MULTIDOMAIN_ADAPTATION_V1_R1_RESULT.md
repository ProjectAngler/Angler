# High-level multidomain adaptation V1-R1 result

Date: 2026-09-01

Identity: `angler.high-level-multidomain.v1-r1-harness-qualification`

Disposition: `QUALIFICATION_FAILURE` (technical harness-integrity failure; no
scientific claim)

## Immutable artifacts

- Qualification result:
  `/opt/angler/results/high-level-multidomain-v1-r1-qualification.json`
  (`547dbce00c9175fb97d4abcdf61cc8fb8ad0c2dd91f293c350a7c4700814c16c`)
- Source manifest:
  `experiments/manifests/high-level-multidomain-v1-r1.json`
  (`2b8adadf402d52f1c0cd7a4523c855a0951fc04333fe43350009b1a8bd72c36d`)
- Sealed seed artifact:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r1/sealed/seeds-v1-r1.json`
  (`4f7360f91a82c57a0212bb9fa5e95acd743d84b63a8d08bb32c7d724451f0d4f`)
- Qualification release claim:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r1/sealed/qualification-release-claim-v1-r1.json`
  (`e5c4483e7162244bb5c2101961e253a94c8f628c44599cad4807ed2959683c68`)
- Attempt evidence ledger:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r1/qualification-v1-r1/attempt-evidence.sqlite3`
  (`f167a08029baa46eeff7c028e73c3e16d4f5973c5bbd5234fe6a4da61c0190f5`)
- Arm-factory audit:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r1/qualification-v1-r1/arm-factory-audit.json`
  (`01cd9c91e2407584e4b5a9db5dc9f818029f7c9341cac176121507cabd460183`)
- Frozen runner:
  `experiments/runners/high_level_multidomain_v1_r1.py`
  (`78152a1502ca8317c16c9aa77f913d46ec031229fa51e8aaf2d0de25f2c2398a`)
- Frozen unit suite:
  `tests/unit/experiments/test_high_level_multidomain_runner_v1_r1.py`
  (`9a929950b7f64b0f8220716f39187040ab52f7e3198dd92a39794c72b624e372`)
- Frozen exact-Qwen integration suite:
  `tests/integration/runtime/test_high_level_multidomain_qwen_v1_r1.py`
  (`bb8aa7e931b6ab197741a56d7a54de68f975fcbb2cc1c9c4840424ff11b9cc43`)
- Accepted construction-completion leaf:
  `docs/blueprints/branches/science/work/ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-LIVE-EVALUATION-COMPLETION-001.md`
  (`a54aa78d48fe5435b035e3c057470c915b87c8f4a4de9d001997a1dcd57d39de`)

The terminal result, source manifest, release claim, and sealed seed artifact
are mode `0600`, owned by UID/GID `1000:1000`, and have exactly one link. All
32 manifest sources rehash exactly. The public evaluation-seed commitments are
`sha256:869675ad3b43bdd300f6ed58b8bf0af2b87520318b42e36d721565a7b7e9d9f5`
and
`sha256:44ea52eb9a4085e0e803a965b53ad0fa6ac58cd642fc82ae130a583371b9316f`.
The raw seeds remain sealed and undisclosed. The R1 identity is permanently
consumed and must not be rerun.

## Observed execution

The offline RTX 5080 qualification completed its declared 31-arm schedule in
323.22 seconds wall-clock time. All 31 attempt rows finalized: 31 proposal
attempts, 27 executions, and 58 frozen-model generations. The runner's measured
post-load interval was 286.14990089301136 seconds. Peak measured RSS was
9,101,860,864 bytes; peak CUDA allocation and reservation were respectively
8,347,718,656 and 8,524,922,880 bytes; peak new state-and-scratch bytes were
20,015,401. Every value was within its frozen ceiling.

The qualification did not reach a scientific evaluation admission or any
development/final evaluation payload. The canonical terminal result is
`QUALIFICATION_FAILURE`, category `INVARIANT`, stage `ORCHESTRATION`, with
`scientific_claim=false`, `admission_validated=false`,
`evaluator_cleanup_completed=true`, and no cleanup error code.

## Exact root cause

Each persistent lineage intentionally initializes its transaction store and
then commits one unrelated neutral bootstrap episode. That episode occupies
durable transaction sequence 1 while leaving the qualified learner at its
exact step-0 genesis digest
`sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286`.
Consequently, the first admitted scheduled transition in both `FULL` and
`RANDOM_FEEDBACK` correctly has sequence 2 and the exact genesis digest as its
parent. The observed admitted transitions are continuous through sequence 11;
malformed attempts do not advance either lineage.

Both the live and serialized aggregate adaptation-lineage validators instead
initialize `previous_sequence` to 0. Read-only reconstruction of the 24 durable
adaptation records therefore deterministically raises
`RunnerInvariantError: adaptation learner transition did not advance` at the
first valid sequence-2 transition.

This is a harness-integrity contradiction between the declared bootstrap
baseline and its aggregate validator. It is not evidence for or against Qwen
competence, Cognee recall, Moving-Origin state, prospective learning, causal
adaptation, or the scientific hypothesis. No score, threshold, task, prompt,
model, learner, parser, feedback, or control-arm semantic is implicated.

## Cleanup and retained evidence

Independent read-only audit validated all 31 managed dispositions, both
persistent lineages, the accounting envelope, ledger and factory-audit
references, namespace forget, provider disposal, binding close, and scope
release. Temporary, clone, control, and CUDA-cache directories are empty as
required. No evaluator, Qwen, or Cognee worker remains; both GPUs are idle.
No R1 evaluation runtime, evaluation Cognee scope, evaluation-admission claim,
or evaluation result exists.

The retained R1 state tree has local-tree digest
`136f04653ed9dc32b1d8eff6c09921b12ab7ca6b258a33fde9f58e714a46a30c`
with 53 files, 36 directories, 11,487,875 logical bytes, and 89 records. The
retained scratch tree has digest
`a9191c6653c9b5417c80e1779e821e318768a3dba8a732313d821bbbf979d43f`
with 65 files, 14 directories, 812,194 logical bytes, and 79 records. These
trees use the `angler.local-tree-digest.v1` algorithm inherited from the R1
work leaf. Immutable V1 witnesses and both V1 retained-tree digests also
rejoined exactly.

## Gate and recovery

`ANG-GATE-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-001` does not pass.
R1 evaluation remains blocked and must never be started from this result.

Any justified recovery requires a separately authorized fresh R2 identity,
fresh literal state/scratch/seed/manifest/release/admission/result paths, and
new operating-system-entropy evaluation seeds. The only scientific-semantic
code correction is to bind both aggregate lineage validators to the exact
one-episode bootstrap transaction baseline, so the first admitted transition
must be sequence 2 rather than accepting an arbitrary offset. All tasks,
model/learner/evaluator identities, schedules, arms, budgets, thresholds,
parsing, feedback, controls, deterministic-code restrictions, and
Human-Flourishing boundaries remain frozen.
