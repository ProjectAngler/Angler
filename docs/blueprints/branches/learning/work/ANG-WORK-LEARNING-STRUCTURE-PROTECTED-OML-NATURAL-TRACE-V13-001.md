# ANG-WORK-LEARNING-STRUCTURE-PROTECTED-OML-NATURAL-TRACE-V13-001

Active node: `ANG-BP-LEARNING` / reusable procedural representation

Assurance: LOW; generated software-workflow text, frozen local Qwen,
reversible fresh checkpoint/results, no deployment or external effects

## Accountable outcome

V12 is consumed as `DEVELOPMENT_NOT_SUPPORTED` at result SHA-256
`2A448C957FE864A81811E5172F9DDA99640ADEE24303F29139B68FB3B93BC167`
and checkpoint SHA-256
`7D7FDB80B1737A2142C71EFE89CC5E0EB39C516E8F261538F774B1AADB4AABA4`.
Its train objective fell about 9.2%, every functional/meta-gradient invariant
passed, and no gradient clipped, but unseen-development AUC worsened from the
untouched source's `0.69595` to `0.74716`. True and shuffled feedback differed
by only `0.000215` AUC. The 32-D codes escaped global collinearity yet had
effective rank `5.992`, and same-multiset reorder distance was `0.02598` below
same-procedure paraphrase distance. Removing step semantics improved rather
than harmed development behavior. Outcome-only OML therefore optimized an
underdetermined, surface-sensitive geometry instead of transferable procedure
concepts.

V12's evaluation was also partly underdetermined. Canonical procedures were
arbitrary action permutations, heldout families shared no ordered bigram or
trigram with training, and the learner saw only an attempt trace. A support
family could not reveal an unrelated probe family's arbitrary intended order.
Trace length was confounded with outcome, and development/paraphrase renderers
were unseen atomic styles. V13 therefore makes the intended procedure public
as a natural-language reference trace and evaluates a learned reference-to-
attempt relation. This is the minimum observable problem that can support a
cross-family procedural claim without exposing an answer or coding a solver.

## Borrowed mechanism and protected ownership

Each learner row contains exactly two public `Step N:` traces: a reference
procedure and an attempted procedure. A shared directed two-round
`NaturalLanguageTraceGraphEncoder` maps each trace to a 32-D code. The learned
pair trunk receives `[reference, attempt, abs(reference-attempt),
reference*attempt]` and applies `LayerNorm(128) -> Linear(128,64) -> SiLU`.
The fast PLN remains one fixed bias-free `[1,64]` weight with V16 functional
AdamW and an eight-step OML unroll. No family, corruption, renderer, answer, or
distance feature enters these functions.

Borrow normalized-code supervised contrastive / InfoNCE from SimCLR, SupCon,
and GraphCL-style representation learning. For each reference, an independently
paraphrased same-order attempt is its sole positive. Same-sentence-multiset
reorder and length-matched replacement are primary hard negatives; omission
and insertion are separately reported hard negatives. Other procedures in the
batch are ordinary negatives. Pair roles exist only in loss orchestration. Use
symmetric reference-positive InfoNCE at temperature exactly `0.10`, directly
on the 32-D backbone codes with no projector.

The structural objective updates only the shared trace encoder. The robust OML
outer objective updates only the learned pair trunk. The fixed fast head stays
inner-only. Both gradients are computed from the same pre-update snapshot.
This is structure-protected OML: outcome credit cannot overwrite procedure
geometry, while second-order feedback can still teach the pair trunk how to
support rapid adaptation.

Three learned arms begin byte-exact and see identical rows/order/exposure:

1. structure-protected second-order OML;
2. structure-protected detached-first-order OML;
3. paired outcome-only second-order control, whose OML objective owns both
   encoder and pair trunk and receives no structural objective.

The protected arms must retain byte-exact encoder weights and encoder optimizer
states throughout; only their trunk meta-gradient connectivity differs. An
untouched paired source is the fourth control. The consumed V12 artifacts are
diagnostic provenance only and are never rerun or treated as a like-for-like
paired-input baseline.

## Identifiable counterbalanced paired corpus

Create a fresh V13 corpus adapter with the same 48/12/12 topology counts,
length balance, heldout complete signatures/ordered compositions, 384 inner
mechanisms, 1,536 fresh outer mechanisms, 48 development mechanisms, exact
192x8 schedule, and sealed final. Do not edit V12 corpus or runner.

- every episode publishes a reference trace, an independently rendered attempt
  trace, and chronological success/failure feedback;
- success means the attempt has the same ordered procedure under different
  public wording; failures balance reorder, replacement, omission, and
  insertion across mechanisms;
- reorder uses the reference's exact rendered sentence multiset in a different
  order; replacement is length matched; the same-length reorder+replacement
  subset is the primary behavioral claim;
- outcome/length tables are exactly balanced so a length-only rule has balanced
  accuracy `0.50`; insertion/omission cannot establish the main result;
- renderer atoms are shared across train/development/final while complete
  renderer/context combinations and exact texts are partition-disjoint;
- reference/attempt renderer roles and serialized positions are Latin-square
  counterbalanced; no role owns a unique renderer plan;
- learner tensors contain only the two public traces and outcomes used for
  training. Keys, IDs, signatures, seeds, renderer plans, corruption labels,
  edit distance, target indices, diagnostics, and evaluator decisions stay in
  generator/evaluator metadata.

Deterministic code may generate and label synthetic pairs, maintain partitions,
and assign loss relations. It may not compare traces, score candidates, expose
an edit operation, or execute a solution rule at inference.

## Frozen training protocol

Initialize all four arms from one fresh deterministic paired model and record
its digest. Keep exactly 192 owner updates, eight inner mechanisms, eight fresh
outer mechanisms, inner learning rate `1e-3`, structural encoder and OML trunk
AdamW learning rates `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight
decay, and separate gradient clips `5.0`. The OML robust objective remains
`0.5*mean + 0.5*0.05*logmeanexp(loss/0.05)`. No memory, replay, VICReg, ANML,
projector, more updates, new encoder, coefficient search, seed retry, early
stop, autocast, TF32, Qwen update, or final access is permitted.

Before update one, prove exact starts/data; connected second-order and absent
detached support-Hessian paths into the pair trunk; nonzero direct outer trunk
gradients; nonzero structural encoder gradients; exact zero cross-owner
mutation; exact protected encoder equality; full-vs-split OML gradients; and
InfoNCE invariance to balanced view serialization. Every owner partition and
optimizer state is recorded and mechanically checked.

## Development evaluation and frozen gate

Evaluate four renderer- and family-disjoint panels with both same-transition
and cross-transition reference/attempt pairs. Cross-family comparison is valid
only because each row supplies its own public reference. Report protected
second/first online and no-update, paired source online/no-update, paired
outcome-only control, shuffled support feedback, direction removed, and step-
semantics removed. Probe labels are read only after logits exist.

The first complete development result advances only if every condition holds:

- reference-to-attempt retrieval selects the same-order paraphrase over four
  corruptions in at least `40/48` rows, at least `9/12` rows per corruption
  stratum, and at least 75% of every renderer-position stratum;
- paired-order win rate over the same-sentence-multiset reorder is at least
  `0.75`, and mean cosine-score margin is positive; every corruption-minus-
  paraphrase distance margin is at least `0.05`;
- same-length reorder+replacement balanced accuracy is at least `0.65`, at
  least `0.10` above the recorded length-only baseline, and succeeds in at
  least 9/12 families and 2/3 families per transition grouping;
- the protected encoder exceeds the paired outcome-only control and untouched
  source by at least `0.20` retrieval accuracy and `0.05` mean hard-negative
  margin;
- centered code effective rank is at least 8, all 32 variances are finite and
  nonzero, raw mean off-diagonal cosine is at most `0.95`, and at most 10% of
  distinct raw pairs exceed `0.999` cosine;
- protected second-order no-update balanced accuracy is at least `0.75` and at
  least `0.10` above paired source no-update;
- protected second-order online AUC is below protected first-order, at most
  `0.95` times it, below paired source-online and protected no-update, improves
  at least 3/4 panels, and regresses on none;
- shuffled-feedback AUC minus true-feedback AUC is at least `0.005` or true
  feedback accuracy exceeds shuffled by at least `0.05`;
- full semantics beats semantics-removed on the same-length subset by at least
  `0.05` accuracy or inherited equivalent loss margin; direction removal meets
  the same threshold specifically on reorder rows;
- at least 9/12 unseen families improve over source and every transition class
  improves in at least 2/3 families;
- every ownership, pairing, length balance, renderer counterbalance, overlap,
  hash, frozen-Qwen, repeat/padding, gradient, and final-seal invariant is exact.

Passing representation/retrieval but failing feedback/OML transfer is
`STRUCTURAL_REPRESENTATION_SUPPORTED_NOT_INTEGRATED`; it cannot open final or
memory. OML improvement with failed geometry is
`DEVELOPMENT_SHORTCUT_CODE_COLLAPSE`. Otherwise use
`DEVELOPMENT_NOT_SUPPORTED`. Only the full conjunction opens sealed final.
Accept the first complete result without tuning.

## Literal scope, safety, and rollback

Fresh outputs: this leaf; one paired V13 corpus adapter and focused test; one
paired structure-protected OML core and focused test; one V13 runner and
focused test; reasoning export; append-only `AGENTS_SYNC.md`; fresh checkpoint
and development result, plus final only after authorization. V12 artifacts,
Qwen, memory, Cognee, Moving Origin, public remote, recovered/personal data,
and unrelated owner work remain immutable.

Only generated software-workflow text is processed. No network service,
package install, model mutation, deployment, self-promotion, external effect,
or human-affecting decision is authorized. Stop on source/control drift,
length/renderer leakage, split collision, owner divergence, cross-owner
mutation, non-finite gradient, identity reuse, or premature final access.
Rollback removes only fresh V13 files/artifacts while preserving evidence.

## Frozen first result

The one development run completed all 192 owner updates in 647.383 seconds.
The result is preserved at SHA-256
`A25B4ABFD119F71D835A7E73A620FDC4CA2686A0301B66DDD047D3DC29DCA6A4`;
the checkpoint SHA-256 is
`A0F4B6B859274B315B311D3DDE2D4EFDD9C6D494C33C006D313B450403AB9E8B`.
Its frozen classification is `DEVELOPMENT_NOT_SUPPORTED`, and final remains
sealed.

The headline understates a strong but incomplete component result. The
protected representation retrieved the correct held-out procedure in `44/48`
rows, selected correct order in `97.92%`, reached effective rank `14.247`, and
achieved same-length balanced accuracy `0.85797` against a length-only `0.50`.
All twelve held-out families improved over source, with at least two successes
in every transition group. Outcome-only retrieval was `0/48` and untouched
source retrieval `5/48`, while protected retrieval margins were positive for
every corruption.

The structural conjunction missed only its frozen every-renderer-position
condition: four sparse strata containing one or two observations fell below
`0.75`. This threshold is not changed after result inspection. The OML
conjunction missed two substantive conditions. Second-order AUC was
`0.481465` versus first-order `0.482451`, a ratio of `0.99796` rather than the
required `<=0.95`; true feedback beat shuffled feedback by only `0.001598` AUC
and `0.01042` balanced accuracy. Thus V13 establishes a noncollapsed,
order-sensitive learned procedure representation but not useful online credit
assignment or durable adaptation.

Do not tune or rerun V13. The evidence-supported successor must preserve the
V13 encoder exactly and test a persistent learned, content-addressed
eligibility/ability memory driven by objective outcomes. Cognee remains the
durable episodic evidence layer and Moving Origin supplies situated temporal
coordinates; neither may substitute a deterministic procedure answer. Zero,
shuffled-feedback, state-swap, replay reconstruction, unrelated-state, and
sequential-retention controls must establish that later behavior changes due
to learned procedural credit rather than retrieval or global bias.
