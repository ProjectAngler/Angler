# ANG-WORK-LEARNING-COMPOSITIONAL-PROCEDURE-V4-R1-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local synthetic procedural training, no external effects

Human-impact mapping: local synthetic mechanisms only; no real-person data,
external action, deployment, persuasion, surveillance, or learner authority.
The learner cannot judge or waive `ANG-GATE-HUMAN-FLOURISHING-001`.

## Accountable outcome

Recover the unexecuted V4 experiment under a fresh identity after preflight
proved that V1's zero-initialized plastic slots receive exactly identical
writes. Demonstrate actual slot-diverse, end-to-end sequence credit without
changing V4's curriculum, optimizer budget, scientific thresholds, or hidden
evaluation partition.

## Preserved failed preflight

`angler.compositional-procedure.v4` remains unexecuted and immutable at runner
SHA-256
`511D845BBBB0D6D8C9521E4C65BB101C3E37C6164FC5C64459E60D4EBEBF2DFF`.
No V4 result, checkpoint, or embedding-cache artifact was created.

The preflight reproduced the blocker on CPU across four distinct public
support writes: maximum cross-slot spread was exactly `0.0` for keys, values,
and strengths after every write. Writer gradients were nonzero, proving that
gradient reachability alone does not prevent 64 identical plastic copies.

## Frozen recovery

- Fresh identity: `angler.compositional-procedure.v4-r1`; fresh seed:
  `20260847`.
- The immutable V1 checkpoint, frozen Qwen, 64/16/16 mechanism partitions,
  four two-action supports, two four-action train compositions, eight passes,
  512 optimizer updates, learning rates, retrieval weight, gradient ceiling,
  and original V4 gate thresholds remain unchanged.
- Initial plastic keys are a content-neutral seeded Rademacher address matrix.
  Initial values and strengths remain exactly zero, so all anchored slots are
  masked and reset inference is tensor-exact with the V1 zero-key reset.
- The anchor algorithm, seed, scale, shape, SHA-256, and actual key tensor are
  checkpointed. The JSON report records the same provenance except the tensor.
- Train-only four-action labels are materialized once and their hidden pair
  objects are immediately discarded. Development and final targets remain
  absent and are available only to the existing deterministic judge.
- Final evaluation adds matched `local_only` and `persistent_only` diagnostic
  arms. They do not enter classification or alter the four original causal
  margins.
- Accept the first terminal V4-R1 result without post-result tuning.

## Required preflight evidence

- Anchored reset and original V1 reset emit identical slots, candidate weights,
  and Qwen prefix before any write.
- Distinct supports produce nonzero cross-slot strength and value diversity.
- A downstream four-step decoder loss produces finite nonzero gradients in
  `write_key`, `write_value`, `write_gate`, and the decoder projection.
- Anchor generation and digest are replayable.
- Every one of 128 train-only four-action labels executes correctly, has four
  distinct public candidates, and retains no hidden pair after materialization.
- Development and final queries retain `target_text=None` and are rejected by
  the train-label materializer.

Final preflight passed `20/20` focused tests plus four device-placement
subtests. Runner SHA-256 is
`D27E9C550B4422917770A1230F9EF0E9F9BE6102D5E21CDAF5DEDCE1A6AC5B5B`;
test SHA-256 is
`F027814AB9A5DCAE9D8A81E9B6C4246E3D5CF649EC61F768E98D5E6E2DFCD35F`.
The final runner places frozen Qwen on explicit `cuda:0` and the trainable
Angler core, decoder, and plastic state on explicit `cuda:1`; it rejects
missing, implicit, identical, or unavailable device selections. A zero-update
CUDA mechanics check on the RTX 5080/5070 pair encoded 27 real Qwen rows,
placed Angler on `cuda:1`, and produced finite nonzero gradients through all
three writer stages and the decoder. It created none of the three output
identities. No GPU optimizer step or result creation occurred.

Two attempted process invocations stopped before model loading or output
creation: the first omitted the repository root from `PYTHONPATH`; the second
found that this PyTorch build requires each CUDA context to be initialized
before per-device peak-memory reset. The final runner performs that explicit
initialization. Both failures were pre-semantic and all output identities
remained absent.

## Literal outputs

- `/opt/angler/results/compositional-procedure-v4-r1.json`;
- `/opt/angler/results/compositional-procedure-v4-r1.pt`;
- `/opt/angler/results/compositional-procedure-v4-r1-embeddings.pt`.

All three paths must be absent before execution. V4 paths and all prior
artifacts remain immutable.

## Gate

On the same untouched 32 final queries, full hidden execution must be at least
`0.60` and exceed state reset, coordinates removed, unrelated evidence, and
procedure slots removed by at least `0.10` each. Development composed execution
must be at least `0.60`, and target-evidence attribution at least `0.75`.

`local_only` and `persistent_only` are preserved diagnostics only. They cannot
rescue a failed gate or change its classification.

A pass supports this bounded compositional planning interface only. It is not
a normal LEARNING gate, milestone, general-conversation claim, or permission
for deployment. Rollback is exact deletion of the three absent-or-new V4-R1
outputs; the V1 parent checkpoint and preserved V4 sources remain unchanged.

## Preserved first result

The single frozen run completed `NOT_SUPPORTED` in `311.6878` seconds. Result
SHA-256 is
`86FDA824E11FD459BDF17B9D0DBA5BAF4B6ADC1E62335BDAB389F662106465AA`;
checkpoint SHA-256 is
`029F18E8053DC8B5BDD96317AE571C56B5A2E6AA2548A7F59FC92EAAE239370E`;
embedding-cache SHA-256 is
`5031D348376622732FCB99A2AF44E9C4E4ED2E734D89758C80B92469B9028CE9`.
Qwen used `cuda:0` and Angler used `cuda:1` as declared. Development and all
final execution arms scored `0.0`; target-evidence attribution was
`0.9999861`. Training support exactness rose to `0.5546875`, while four-action
composition exactness remained at or below `0.0078125` and ended at `0.0`.

The learned state was active and diverse, and reset/slot-removal controls
changed plans, but full and persistent-only plans were identical on all 32
final queries. A preserved-checkpoint, no-update, without-replacement decode
diagnostic reached only `0.03125` development and `0.0` final execution. The
failure is therefore not repaired by forbidding repeated choices. The writer
receives outcome but not the executed action trace, so it cannot persist the
successful procedure itself. Any successor must make observed procedure
content an explicit learned feedback input rather than add a deterministic
answer repair.
