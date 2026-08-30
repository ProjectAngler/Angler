# ANG-WORK-LEARNING-SITUATED-MEMORY-V1-001

Status: consumed; `NOT_SUPPORTED`

Assurance: bounded local synthetic research; no promotion authority

## Question

Does a learned Angler reader gain a reusable temporal procedure-selection
ability when Cognee semantic candidates are augmented by Moving Origin
coordinates, with the gain requiring all three roles rather than a hidden
answer rule?

## Frozen construction

- Cognee proposes evidence candidates and never generates an answer.
- Moving Origin supplies monotonic age and landmark coordinates; it never
  ranks a candidate.
- A new GPU-resident `LearnedSituatedMemoryReader` combines frozen content
  representations with those coordinates and is trained only from outcome
  labels.
- Candidate evidence contains prior action/outcome observations but never the
  current query answer or a task/query identity.
- Eight semantic mechanism families, four action symbols, shuffled candidate
  order, changing target actions, repeated surface variations, and fresh
  evaluation instances prevent memorizing one fixed problem.
- Six families may train the generic read/apply behavior; two families remain
  held out to measure same-mechanism structural transfer. This is not evidence
  of unrestricted cross-mechanism reasoning.

## Frozen causal arms

The same trained weights are evaluated as: full live origin; frozen origin;
coordinates removed; content removed; retrieved target removed; and fresh
competence reset. A fair-naive temporal scan must reproduce the live
coordinates/answers while inspecting materially more history.

## First-result interpretation

The combination is `SUPPORTED_FOR_PROTOTYPE` only if all of the following hold
on the first accepted result:

- full fresh-instance accuracy is at least 0.80;
- full held-family accuracy is at least 0.70;
- full accuracy exceeds each of frozen-origin, coordinate-removed,
  content-removed, target-removed, and competence-reset by at least 0.15;
- fair-naive accuracy matches full accuracy within 1e-6 while its mean
  inspected history is at least twice the maintained-index count;
- the neural reader, representations, loss, and optimizer execute on CUDA and
  the result records device, versions, parameters, peak allocation, seeds,
  counts, and wall time;
- a live self-hosted Cognee subset returns the target evidence in at least
  90% of queries and the complete live path reaches at least 0.65 accuracy.

Failure of the live Cognee clause may be classified separately as retrieval or
operations failure if the causal synthetic result passes. Thresholds will not
be changed after a result. No result establishes AGI, consciousness, general
cross-domain transfer, or superiority to a frontier model.

## Scope and stop conditions

Fresh reader/runner/test/result files, this leaf, and narrow exports are in
scope. Existing V19–V23 evidence is immutable. Stop on external API use,
personal/recovered data, model-weight mutation, answer leakage, non-CUDA
neural execution, non-finite state, or an unclean temporary Cognee dataset.
Cognee data are disposable projections; canonical evidence and Moving Origin
remain outside it.

## Preserved result

The first result failed as specified: full accuracy was 0.47802734375,
frozen-origin accuracy was 0.46923828125, and removing the nominated target
did not reduce accuracy. The reader had learned an unintended exclusion
shortcut, not the intended current-success selection procedure. Result
SHA-256: `4df3c604bea1116e8398312435c2b52481942894d75e55d8cdafa551e4dbd71d`.
The identity remains consumed and was not tuned or rerun.
