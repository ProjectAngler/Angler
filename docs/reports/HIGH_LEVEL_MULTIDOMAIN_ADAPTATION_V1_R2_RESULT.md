# High-level multidomain adaptation V1-R2 result

Date: 2026-09-01

Qualification identity:
`angler.high-level-multidomain.v1-r2-harness-qualification`

Evaluation identity: `angler.high-level-multidomain.v1-r2-evaluation`

Disposition: integrity `PASS`; scientific classification `NOT_SUPPORTED`

## Immutable artifacts

- Evaluation result:
  `/opt/angler/results/high-level-multidomain-v1-r2.json`
  (`c2316da22208cf0a8693f32f4ea28060c529329df1c6d7d67841b56720aca676`)
- Evaluation-admission claim:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/sealed/evaluation-admission-claim-v1-r2.json`
  (`09b5bac6ad20ab81e39beeba5efbab80cdb60d4cd187c9b9964c52669c281d30`)
- Qualification result:
  `/opt/angler/results/high-level-multidomain-v1-r2-qualification.json`
  (`83833938c38021d8546db0a721564e1d5ee0c4a14b2d6a6efbd28c976bf2554c`)
- Source manifest:
  `experiments/manifests/high-level-multidomain-v1-r2.json`
  (`842e39e713c87352a9dbc5e04c8cd73ab74a1b32fceb9aaa72c1d1fd75a4344d`)
- Sealed seed artifact:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/sealed/seeds-v1-r2.json`
  (`efc7a50e2cdde18f8fc293a5b29368b28a92f7443129be2ee1202e4f986abea7`)
- Qualification-release claim:
  `/opt/angler/state/project-angler/high-level-multidomain-v1-r2/sealed/qualification-release-claim-v1-r2.json`
  (`95c86153d23e981b90467eacf534426154aab0aade7fdb11646fc6bba814aa0d`)
- Qualification ledger and arm-factory audit:
  `c0f8e4921a114ecb7e282c3487c704ec790dac666dc7c07dc3649fc199ca89aa`
  and
  `a6b1ad1962446c6f076a3245999d1a5599961e243af45361e1b33447df7a6598`
- Evaluation ledger and arm-factory audit:
  `914e1695cc4c9187a60f48a035632dbe5e849056e113bab4295ca36f5bbb4eca`
  and
  `eb6f4e16336f83b4cee95ecc9ec4803af99045dd1ba904373599db8a3f0b4e82`
- Frozen runner, unit suite, exact-Qwen integration suite, and accepted R2 leaf:
  `fbd10104ee566e327e812f24e6ea44d30e405d4a2dea2650ebd94dad58fee574`,
  `a76d28e49c1bba1ffdcf9dbe82ec2fccc621de1b2642ac9d0561f5762caae462`,
  `8e76a56482d04856af5046e30370bf0b708dfdca5e019ebdc82b40785c2b4e1d`,
  and
  `a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455`.

The manifest's 38 sources rehash exactly. The live terminal artifacts, sealed
claims, and seed artifact are mode `0600`, owned by UID/GID `1000:1000`, and
have exactly one link. The public evaluation-seed commitments are
`sha256:a6b8f252b7b8482d0e6648204899dff9d35f4a7c20160d948854bc271d293c8f`
and
`sha256:a7327638bef71c33166f1cce2b6285fefcf1c496d3d4b2708346ee534d2965ae`.
The raw seeds remain sealed and undisclosed. This R2 identity is permanently
consumed and must not be rerun or used for post-result threshold tuning.

The qualification-release and evaluation-admission references are
`sha256:24677d79bc43c14eb3d59fa3aef689a77b695e6f1989b11c683e9c4b0990696a`
and
`sha256:6eb2f0c193d97cc3ca12215148fb00420f42de769c6032fea19bbb840e5ca803`.
The admission digest is
`sha256:d51d669c229f63b4eed48c3c1c1d8607a7ab63cddcbb29fc67d7f425b2ab575a`.

## Observed execution

The one-use offline RTX 5080 qualification completed with
`QUALIFICATION_PASS`: all 31 attempts finalized, with 31 proposals, 27
executions, and 58 frozen-model generations. Its observed process wall time
was 5 minutes 23.24 seconds and maximum resident set size was 8,889,012 KiB.
Independent audit accepted the result before evaluation admission.

The one-use offline RTX 5080 evaluation then completed with exit status zero
and terminal classification `NOT_SUPPORTED`. It finalized all 468 declared
arm-task attempts: 468 proposals, 433 executions, and 901 frozen-model
generations. Phase attempt counts were 48 adaptation, 140 development, and
280 final. The runner measured 4,982.100995 seconds; observed process wall
time was 1 hour 23 minutes 19 seconds. Peak measured RSS was 9,103,310,848
bytes, peak CUDA allocation was 8,382,814,208 bytes, peak CUDA reservation
was 8,610,906,112 bytes, and peak state-plus-scratch use was 101,444,016
bytes. Every value remained within its frozen ceiling.

One Python 3.12 `PidfdChildWatcher` warning reported that a child exit status
had already been read and synthesized return code 255. The run continued to
468/468 durable finalizations and exited zero. Independent reconstruction
found complete worker cleanup and no missing receipt or disposition, so the
line is retained as a non-blocking observer/double-reap warning rather than a
scientific or artifact-integrity failure.

## Frozen scientific result

Final binary successes were:

| Arm | Successes / 40 |
|---|---:|
| `FULL` | 0 / 40 |
| `QWEN_ONLY` | 3 / 40 |
| `RETRIEVAL_ONLY` | 0 / 40 |
| `FROZEN_ORIGIN` | 0 / 40 |
| `PROSPECTIVE_REMOVAL` | 0 / 40 |
| `BACKEND_REMOVAL` | 0 / 40 |
| `RANDOM_FEEDBACK` | 1 / 40 |

The three `QWEN_ONLY` final successes occurred in the glyph-machine family;
its family total was 3/8. `RANDOM_FEEDBACK` had one glyph-machine success.
All other final family-by-arm totals were zero. During adaptation, `FULL` and
`RANDOM_FEEDBACK` each scored 4/24: zero symbolic-demonstration-transfer, two
glyph-machine, and two causal-operator successes. In development,
`QWEN_ONLY` scored 2/20, both glyph-machine; every other arm scored 0/20.

The preregistered integer classifier recomputed exactly:

- P1, `FULL >= 14`: false (`0`);
- P2, `FULL - QWEN_ONLY >= 3`: false (`-3`);
- P3, `FULL - RETRIEVAL_ONLY >= 3`: false (`0`);
- P4, `FULL - RANDOM_FEEDBACK >= 2`: false (`-1`);
- P5, `FULL - PROSPECTIVE_REMOVAL >= 2`: false (`0`);
- P6, `FULL - FROZEN_ORIGIN >= 1`: false (`0`);
- P7, required family control margin: false;
- P8, removal integrity: true; and
- P9, two advanced `FULL` lineages: true.

Metrics, classifier, and run-integrity references are respectively
`sha256:0f3182116181d6f0b50820368c83d9717f43099259c4b9596d2416bd71f8e12e`,
`sha256:c75b35dc0ffacd450aebfa6ab41dac0daaeefe0d1a9db36a9a0af9533e75a952`,
and
`sha256:62c78e8a34a7097271b2b9e729277cf3074aef2c7f4fcd3c8d9d1ce098a06b38`.

Thus the frozen tested hypothesis is `NOT_SUPPORTED`. State adaptation did
occur, but this experiment demonstrated no beneficial final-task transfer
from the integrated system. This bounded synthetic result does not establish
general incapability and does not authorize broader claims about reasoning,
agency, selfhood, readiness, or AGI.

## Integrity, removals, and retained evidence

Two independent audits plus a root reconstruction accepted the terminal
evidence. The evaluation ledger contains 468 unique staged and finalized
receipts: 433 admitted and 35 malformed. All receipt references, objective
judgments, probe evidence, 48 evaluator aggregates, 60 removal comparisons,
468 cleanup dispositions, and 420 disposable-root absences rejoin exactly.

The five integrated/removal arms had exact pre-intervention identity for each
of the 60 development/final comparison groups. Exact search-call counts for
`FULL`, `FROZEN_ORIGIN`, `PROSPECTIVE_REMOVAL`, `BACKEND_REMOVAL`, and
`RETRIEVAL_ONLY` were respectively 1, 1, 1, 0, and 0. `BACKEND_REMOVAL`
matched `FULL` on every required post-intervention field in all 60 groups.
All five arms scored zero in development and final, so these coarse removals
do not identify a beneficial integrated component.

All four persistent lineages reconstruct from the neutral sequence-1
bootstrap and first learned sequence-2 transition. The first-replicate heads
are sequence 13 and the second-replicate heads are sequence 12; all outboxes
are empty and active-turn counts are zero. Both `FULL` lineages advanced.
The frozen foundation tensor digest remained
`sha256:f228ca26e33596461f72195fcbccfa7b873fd7a4dc7c87d19d2854484bcccd3a`
before and after the run, and the model-root digest remained
`1ac705236348881b2fd46f4075b931b5137369d0c469b871aea749f6e0886d83`.

The retained R2 state tree has local-tree digest
`9ed1ee56f53b7250484a7e3c46ae8007a64173fd492bda00963021e6aabb177a`
with 788 files, 792 directories, 104,895,586 logical bytes, and 1,580
records. The retained scratch tree has digest
`93a08289da6afdb7a480f3a328cb6dbb87ea32102713b2a90e8ef1464d84fbe2`
with 65 files, 14 directories, 812,194 logical bytes, and 79 records. The
evaluation runtime and Cognee tree digests are
`d54a348e5e934380d21aac38bb6148e1960250600ca7f4844f2066a7ea6cef4c`
and
`8dee88f74121ec0b84eb84418a235ffab62810efdcd75af843f6af33f2b07b8a`.
Qualification-only state projects back exactly to
`f65d492d39d9c176ddcbc973a57246e3def772189959624badfd9475771fd933`.

Every V1, R1, and R2 immutable witness remains exact. Temporary and disposable
roots are clean, worker directories are empty, SQLite sidecars are absent, no
matching model/evaluator/Cognee process remains, and both GPUs have zero
compute processes.

## Gate and next exact action

`ANG-GATE-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-LINEAGE-SEQUENCE-RECOVERY-001`
passes for experiment integrity and closes with the hypothesis not supported.
The consumed R2 identity, immutable artifacts, lineages, and failure evidence
must be preserved. No threshold may be tuned against this result, and R2 must
not be rerun.

The smallest next action is a fresh, narrow, evidence-only diagnostic successor
leaf. It should compare same-task causal traces around the three
`QWEN_ONLY` glyph successes, integrated-arm failures, learned selected
procedures, and recalled records to distinguish parser effects, retrieval
effects, selection effects, and non-generalizing learned guidance. That work
must preserve private evaluator payloads, add no deterministic task solver,
and make no causal claim beyond the evidence. Only a completed diagnosis may
justify a separately preregistered fresh experimental identity.
