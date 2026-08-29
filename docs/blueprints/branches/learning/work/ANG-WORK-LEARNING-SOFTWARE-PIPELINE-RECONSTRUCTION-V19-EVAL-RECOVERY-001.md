# ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V19-EVAL-RECOVERY-001

## Identity

- Kind: bounded evaluation-only recovery leaf
- Parent outcome: `ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001`
- Protocol: `phase6.public-v12-champion-paired-graph-context-eval-recovery.v19r1`
- Accountable outcome: complete the already-fitted V19 causal evaluation after correcting one proven duplicate-row numerical false rejection, without retraining or changing the frozen V19 mechanism, thresholds, panels, or evidence.

## Trigger and exact diagnosis

The sealed V19 run completed all 512 optimizer updates and preserved a strict terminal checkpoint, then failed in the first `zero_residual` lesion at panel 0 / stream 0 / heldout 2. Same-contract query context codes, raw predecessor graphs, and masks were byte-identical. The inherited V12 batched comparator nevertheless produced a one-FP32-ULP difference in one of three real logits (`2.9802322387695312e-08`), which propagated to one-ULP differences in two real weights and the null. The context scorer received no relation answer, and positive/negative relation-arm contexts were byte-identical.

The learned, uniform-attention, and mismatch-zero paths already canonicalize exact duplicate query rows. Only the `zero_residual` early return bypasses that projection. Production `score_actions` under the primary lesion returns raw V12 before invoking the paired matcher and must remain byte-exact V12.

## Immutable inputs

- Original active leaf SHA-256: `B819DA5F6D10151E7613ADECBBA076DF7642559D35BEA2EA74551FD791C6668D`
- Frozen V19 runner SHA-256: `54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE`
- Frozen V19 unit test SHA-256: `C0D1DBBDE81B628D8D9CCFA751DCB9CFE951B3809860BE5298494C103D1E12BD`
- Original one-shot harness SHA-256: `099381C7AE58F1FBEEFCEC31B0FE1D53DA591D9D51D4E53549FA534F8D5D3123`
- V19 plan digest: `sha256:e66d9e4e90e4c3b2ccb704144c7a591009cde57b6367c3e1cc0b9dd64b8d40d5`
- Terminal V12 source checkpoint SHA-256: `B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C`
- Original V19 claim SHA-256: `E209F2075C59F2AD1087B2F11FFFABEAF31FC598E8B72B10D08F5F6F5E093C57`
- Original V19 failure SHA-256: `C297B861A26FF53EA489E70E537F6EECA7C20B54769394C85C042147838116EE`
- Preserved terminal V19 checkpoint SHA-256: `10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F`
- Terminal system digest: `sha256:99712cfbc24140703203561f3ca42d904752aae92c8ec8d637128f7fe93bebc6`
- Terminal mutable digest: `sha256:9cb6c11f5ff05fe75737227599094378cdacc9d914a3d558548780b26f7735ed`
- Terminal optimizer digest: `sha256:662fd334ecf56f0120e1b3023598099d7929289821df4a72b54a2ab74e83a388`

Every immutable input must match before evaluation and the checkpoint hash and all three learned-state digests must remain exact afterward.

## Literal implementation outputs

- `experiments/evaluators/phase6_v19_paired_graph_context_recovery.py`
- `tests/unit/experiments/test_phase6_v19_paired_graph_context_recovery.py`
- `.angler_v19_eval_recovery_r1_once.py`
- WSL claim: `/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context-eval-recovery-r1.claim.json`
- WSL report: `/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context-eval-recovery-r1.json`
- WSL failure: `/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context-eval-recovery-r1.failure.json`

No new model checkpoint is permitted.

## Mechanism boundary

The recovery evaluator may install one instance-scoped temporary wrapper around the frozen controller's `_paired_graph_context_logits` only while the frozen causal evaluator runs. The wrapper must:

1. call the frozen method first;
2. return its exact output unchanged unless the active lesion is exactly `zero_residual`;
3. group rows only when query context code, raw graph adjacency, and node mask are each `torch.equal`;
4. before projection, require each raw duplicate-row maximum absolute logit difference to be at most inherited V12's already-frozen covariance tolerance `1e-6`; a larger discrepancy is a hard error rather than something to hide;
5. keep the first row in each group as the representative and copy that exact logit row to later duplicates before softmax;
6. leave every representative and unique row byte-exact;
7. restore the original instance method in `finally`, on success or failure.

It may not use a tolerance, relation tensor, validity mask, label, commitment identity, motif/signature, task answer, threshold, or observed evaluation result. It may not modify production `score_actions`, model parameters, optimizer state, competence state, panels, seeds, gates, or classifications.

## Preflight and acceptance

Tests must prove:

- exact duplicates canonicalize only under `zero_residual`;
- exact duplicates above the inherited `1e-6` raw-difference ceiling fail closed;
- a one-bit/code/graph/mask difference prevents grouping;
- representative and unique rows remain byte-exact;
- learned, uniform-attention, and mismatch-zero outputs are byte-exact frozen V19;
- wrapper installation is instance-scoped, non-nested, and restored after success and exception;
- raw production `score_actions` under `zero_residual` remains byte-exact V12 outside the recovery projection;
- the recovery source and harness contain zero fit/optimizer-step calls and exactly one causal-evaluation call;
- frozen hashes, terminal update count, and learned-state digests are verified;
- original V19 claim/failure/checkpoint remain present and exact, original terminal report remains absent, and every recovery output is absent before launch.

After independent source/test/harness review, execute the evaluation-only identity once on CPU with one thread and deterministic algorithms. Accept the first recovered classification as recorded by the unchanged frozen V19 thresholds. The outer report must label the result as recovered evaluation evidence and must not rewrite the original run as successful.

## Stop and rollback

Stop without evaluation for any immutable-byte mismatch, occupied recovery identity, original terminal report appearance, checkpoint-load/digest mismatch, source/test failure, wrapper effect outside exact duplicate zero-lesion rows, production-score drift, GPU initialization, or authority ambiguity. If evaluation starts and fails, preserve the recovery claim and failure record. Rollback removes only unlaunched local recovery implementation files; never delete or replace original V19 evidence or the terminal learned checkpoint.

## Terminal result

Status: `COMPLETE` as an evaluation-only recovery.  The first and only recovered
evaluation completed on CPU without fitting, optimizer steps, GPU initialization,
new checkpoint creation, panel/threshold changes, or learned-state mutation.

- Frozen pre-execution leaf SHA-256: `4A068443C2FA8A7481576154575FDED9D08CD5ED4064FDE0DAA003A73F2B4A57`
- Recovery claim SHA-256: `056C673787110394C24353C15C4FEB007629F60FE900E606BF1DF4CEA1542584`
- Recovery report SHA-256: `55592E9861EC16301603D0CD7BB2A104E596BAAA97BDC65D50DCC517951A0800`
- Original outcome retained: `HARNESS_ERROR_PRESERVED`, failure SHA-256 `C297B861A26FF53EA489E70E537F6EECA7C20B54769394C85C042147838116EE`
- Recovered classification: `PAIRED_GRAPH_COMPONENT_SUPPORTED`
- Full V12 replacement: `false`
- GMN-specific attribution: `false`

The component gate passed independently recomputed frozen arithmetic: four of
four panels had positive recurrence and nonregression; aggregate unique-valid
top-one improved `63 - 22 = +41` against a `+12` gate; real-normalized valid
mass improved `0.6674962457105953 - 0.35896958928158945 =
+0.30852665642900584` against `+0.05`; informative margin improved
`+1.7923373847435684`; and relation signatures remained exact under the primary
lesion.  This is fresh-instance generalization within the same eight public
train-partition mechanisms, not cross-mechanism transfer.

Attribution did not pass.  Uniform cross-graph attention removed `0` top-one,
`-0.0000042813` mass, and `-0.0000578631` margin, making it causally inert at
this resolution.  `mismatch_zero` removed only one top-one decision but removed
`0.1574633956` mass and `1.1876173501` margin, so mismatch confidence mattered.
The supported component is therefore recorded as a **learned paired whole-graph
context comparison**.  GMN-specific node correspondence is not established,
and a whole-graph-statistic shortcut is not excluded.

Full replacement additionally failed frozen relation-coverage gates:
`92/128 < 96/128` supported rows and `18/32 < 24/32` qualifying streams.  The
learned context-quality measures themselves passed (`0.88043 >= 0.80` supported
full top-one and `0.67442 >= 0.60` supported full valid-set mass).  The remaining
measured blocker has therefore moved back to the inherited V12 relation path.

Recovery integrity: 256 exact duplicate rows were projected; the maximum raw
duplicate difference was `5.960464477539063e-08` under the frozen `1e-6`
ceiling; the temporary wrapper was restored; checkpoint/system/mutable/optimizer
identities remained exact.  Focused tests passed `10/10`, and independent CODEX
and Claude source audits plus independent metric recomputations found no
discrepancy.

Next action: preserve V19 as a supported whole-graph comparison component and
run the separately bounded, no-update representation-overlap diagnostic on the
recurring hard streams before selecting a relation successor.  C25/J35, a
GMN-correspondence claim, additional propagation rounds, or any tuned V19 rerun
are not authorized by this result.
