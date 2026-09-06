# High-level multidomain adaptation V1 result

Date: 2026-09-01

Identity: `angler.high-level-multidomain.v1-harness-qualification`

Disposition: `QUALIFICATION_FAILURE` (technical harness failure; no scientific
claim)

## Immutable artifacts

- Qualification result:
  `/opt/angler/results/high-level-multidomain-v1-qualification.json`
  (`feae381e64781d458e2cce5661e63393d165680964a2de24739cc9e24a4d34e1`)
- Source manifest:
  `experiments/manifests/high-level-multidomain-v1.json`
  (`91542cc7557495c7793f82247772e71b0f00f6995e3b4b3fbf2185ee38b470f8`)
- Sealed seed artifact:
  `/opt/angler/state/project-angler/high-level-multidomain-v1/sealed/seeds-v1.json`
  (`74e035d78a27c7ddcafe8702b9cb1e35e5b9c5e5816e22f4d65d5b1d46d6d5bc`)
- Frozen runner:
  `experiments/runners/high_level_multidomain_v1.py`
  (`4c9dc1551d5902f4132ac8d7262a025c17d72299de79d872f0b300d0299356d7`)

The canonical result is mode `0600`, has one link, validates under the dedicated
qualification-failure validator, and binds the exact release claim and source
manifest. The identity is permanently consumed and must not be rerun.

## Observed execution

The offline RTX 5080 qualification ran for 125.69 seconds wall time and reached
only the adaptation phase. It retained 12 staged arm attempts, 11 finalized
attempts, 10 learner transitions, six objective judgments, 22 generation
attempts, and 10 executions. No development task, removal comparison, final
payload, evaluation admission, or scientific metric was reached.

The twelfth staged attempt was an adaptation `RANDOM_FEEDBACK` diagnostic whose
proposal parser disposition was `MALFORMED`. A malformed proposal is an
ordinary zero-result attempt: it consumes its one generation opportunity and
must not be repaired, retried, executed, or used for a learner transition.

## Root cause

`EvidenceLedger._validate_finalization` correctly required malformed attempts
to contain neither feedback nor a learner transition, but then incorrectly
required every random-feedback diagnostic to contain both. The contradictory
integrity rules raised `ValueError` before the twelfth row could be finalized.
Terminal factory audit then found the corresponding active in-memory lineage
lease and produced the secondary
`QUALIFICATION_ARM_FACTORY_CLEANUP_FAILED` disposition.

This is a deterministic evidence-validation defect. It is not evidence for or
against the adaptive hypothesis, model competence, Cognee recall, Moving-Origin
state, or prospective learning.

## Cleanup and retained evidence

All 11 finalized arm dispositions validate namespace forget, binding close,
provider disposal, and scope release. No Qwen, evaluator, or Cognee worker
process remains, no retained state file is open, both retained Cognee databases
contain zero data/nodes/edges, and both GPUs returned idle. The terminal
arm-factory audit is absent because the failed row could not be finalized.

The preserved footprint is 6,384,237 bytes under the V1 state root and 728,211
bytes under its scratch root. These artifacts, the release claim, manifest,
ledger, and result are failure evidence and must not be deleted, rewritten, or
reinterpreted as a qualification pass.

## Gate and recovery

`ANG-GATE-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-001` does not pass.
Evaluation is not admitted from this result. Recovery requires a separately
authorized, fresh versioned qualification/evaluation identity with new literal
state, scratch, seed, manifest, release, admission, and result paths. The
scientific suite, model, schedule, arms, budgets, and thresholds remain frozen;
the only semantic code correction is that an admitted random-feedback attempt
requires its learner transition while a malformed random-feedback attempt
continues to require none.
