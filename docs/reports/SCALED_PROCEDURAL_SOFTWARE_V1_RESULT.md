# Scaled procedural software V1 result

Classification: `NOT_SUPPORTED`

The first workstation-tier semantic run trained Project Angler's 47.8M
parameter procedural core across all 64 training mechanisms (256 visible
support procedures) plus 32 differentiable meta episodes. Frozen Qwen3-4B had
no parameter gradients. The run completed in 802.75 seconds on the RTX 5080.

## What worked

- mean training loss fell from `0.67764` to `0.16722` (`75.32%`);
- development latent loss reached `0.04866`;
- development relevance mass reached `0.99860`;
- untouched final target-evidence attribution reached `0.99981`;
- the preserved checkpoint is 191,716,655 bytes.

## What failed

All seven hidden-execution arms scored zero, including Qwen alone, fair
retrieval, full Angler, reset state, coordinates removed, prefix removed, and
unrelated evidence. This does not isolate a Moving Origin or plastic-memory
failure because the publication contract failed before a valid procedure was
expressed. Evidence-bearing prompts ended after `experience 4`, and Qwen
continued the list with `experience 5`. Query-only prompts ended after the
last component, and Qwen continued with a nonexistent next component `G`.

The V1 result and thresholds remain immutable. The narrow successor is
evaluation-only against the same checkpoint: append `answer=` after the task
and evidence, then recompute every arm without retraining.

## Preserved identities

- result SHA-256: `ae0523155e5b8c1428887767fd0ebbe8acb2b0acbdd3a3456a8469a531ef48e7`
- checkpoint SHA-256: `08cb21c1485cee41565a79b28afdabf1b14a3a379f0d8044cb06870a3d7a1df2`
- runner SHA-256: `a537ad7e43fbfdf6f658bcdde60e06f04d0d751e756c6a9a713c11ed7a33a25d`
- core SHA-256: `0a34084d6b09f00a35fd3d27057721d043b6f33866addae56af9b4f8f28a63e6`
- Qwen bridge SHA-256: `fe05b836509111fbd94ef8b85977861f17725389cf6f533bf70c6d7983b13f1c`
