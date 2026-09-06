# ANG-WORK-LEARNING-ACTION-TRACE-MATCHING-V6-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic procedural learning, no external effects

## Outcome

Test whether action-level ordered plastic memory plus learned cross-context
mismatch matching transfers observed procedures to unfamiliar task-local
candidates. This is the narrow successor to V5's successful fitting but failed
held-out transfer.

## Frozen scope

- Parent V5 checkpoint SHA-256:
  `F9F21BD778DD553468E70A000FE9D8033E02AB18298422237940D8126319332B`.
- Qwen embedding cache SHA-256:
  `5031D348376622732FCB99A2AF44E9C4E4ED2E734D89758C80B92469B9028CE9`.
- Same corpus partitions, eight passes, 512 updates, thresholds, and causal
  arms as V4-R1/V5.
- The inherited core and decoder are frozen. Only fresh generic action-trace
  writer and paired mismatch-matcher parameters train.
- Visible support action embeddings are written individually with order
  position. Hidden targets never enter memory or inference.
- Correspondence-removed is diagnostic; it cannot rescue classification.
- First result is accepted without tuning or rerun.

## Outputs

- `/opt/angler/results/compositional-procedure-v6-action-match.json`
- `/opt/angler/results/compositional-procedure-v6-action-match.pt`

## Gate

Full final execution at least `0.60`, at least `0.10` above reset,
coordinate-removed, unrelated-evidence, and procedure-slot-removed arms;
development at least `0.60`; final evidence attribution at least `0.75`.

## Final preflight

Focused component/runner tests pass `7/7`. Final SHA-256 identities are:

- implementation `27E1A752DE3E8F6E4268320D452BBACCA103EA50FBFE10AFBD702544B1553763`;
- runner `1804971BCDAB5B72E61FC8F5C4EB271328F25E072483FB0849D60CC12CCD022E`;
- component test `A2785B64637B465B44EF08AB0A94E282897A469272E3EF83EB3887CC95664230`;
- runner test `633EA1B58926B0A010879AA8373682F667F6C5C90C3C87E48800321ECC541392`.

A zero-output CUDA pass loaded the exact V5/cache identities, wrote four
visible support traces, and backpropagated one complete stream. All action
writer gradients and the initially zero correspondence residual's terminal
layer were finite and nonzero. Exactly `2,894,850` core and `1,051,136`
decoder parameters were trainable; both outputs remained absent.

## Frozen result

The first and only V6 execution completed in `45.51` seconds on the RTX 5070
and is preserved `NOT_SUPPORTED`:

- result SHA-256
  `12DAE1B3A9DA6B6986F11A046CCD3CB265BDE8E0E750A29E7ECDA114865868879`;
- checkpoint SHA-256
  `08A966424CA678E9742A8D80B985BA48FC0C0FF8E3DF4BC86C0262B9C53FAD7A`;
- peak CUDA allocation `384,663,040` bytes;
- terminal support exactness `0.98828125`;
- terminal training-composition exactness `0.671875`;
- development and every final/control execution arm `0.0`;
- development target-evidence attribution `0.9999989438802004` and final
  attribution `0.9999980349093676`.

V6 therefore strengthened acquisition and training-mechanism fitting again,
but did not transfer to a held-out mechanism. A read-only diagnostic over the
frozen Qwen embedding cache then found near-collapse among task-local action
representations: mean pairwise cosine similarity was `0.978418` on train,
`0.973215` on development, and `0.971783` on final, while mean query-to-target
margin remained approximately zero. The next successor must preserve the
public typed relations and graph connectivity through a learned relational
encoder. Deterministic parsing may produce tensors, but it may not search,
select, repair, or otherwise prescribe a procedure.
