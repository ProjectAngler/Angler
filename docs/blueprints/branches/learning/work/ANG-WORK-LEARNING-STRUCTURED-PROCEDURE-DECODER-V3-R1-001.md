# ANG-WORK-LEARNING-STRUCTURED-PROCEDURE-DECODER-V3-R1-001

Status: ready

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local synthetic harness recovery, no external effects

## Accountable outcome

Recover the consumed V3 execution after its pre-update inference-tensor harness
failure, without changing any scientific input or decision rule.

## Frozen recovery

- The failed identity and exact hashes are preserved in
  `docs/reports/STRUCTURED_PROCEDURE_DECODER_V3_FAILURE.md`.
- Corpus, Qwen, parent checkpoint, decoder architecture, optimizer, seed, four
  passes/1,024 updates, controls, thresholds, and interpretation are identical
  to V3.
- The only functional change clones an immutable `inference_mode` procedure
  tensor into an ordinary detached tensor before decoder autograd. A regression
  test must prove decoder-weight backward succeeds from that input.
- The original runner remains byte-exact and is invoked by a small recovery
  wrapper under identity `angler.structured-procedure-decoder.v3-r1`.
- Accept the first terminal recovery result without tuning.

## Literal outputs

- `/opt/angler/results/structured-procedure-decoder-v3-r1.json`;
- `/opt/angler/results/structured-procedure-decoder-v3-r1.pt`.

Both must be absent before execution. The failed V3 output paths must remain
absent. No parent or prior result may be overwritten.
