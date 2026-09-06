# ANG-WORK-LEARNING-TRIFECTA-ROLE-GRAPH-V10-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic learning, no external effects

## Outcome

Test a freshly initialized neural graph planner that must learn procedure
selection and ordering from public candidate roles and edges, V20/V19
representation features, Cognee-compatible recalled episodes, Moving Origin
coordinates, and Angler plastic state.

## Frozen scope and gate

Reuse V8's exact frozen core, fusion, plastic state, donor checkpoints, Qwen
cache, corpus, partitions, and eight passes. Unlike V9, inherit no trained
decoder tensor: all decoder parameters start fresh so perfect train behavior
cannot starve graph-role credit. Public role plumbing may expose only
origin/goal/forbidden membership, typed producer/consumer incidence, state
read/write incidence, and topology sizes. It may not construct a path, search,
use hidden fields, or repair a plan. Accept the first result without tuning.

Require development/final execution at least `0.60`; final must exceed donor,
public-role, learned-edge, Moving-Origin, unrelated-Cognee, procedure-slot,
and reset removals by `0.10`, with target-evidence attribution at least `0.75`.

Outputs are only `/opt/angler/results/trifecta-role-graph-v10.json` and
`/opt/angler/results/trifecta-role-graph-v10.pt`. Stop on identity mismatch,
non-finite state, output collision, hidden-field need, or deterministic
procedure logic. No normal gate or conversation-readiness claim follows from
this experiment alone.

## Preflight

The focused role/edge/V9 regression suite passes `7/7`. An exact-checkpoint
RTX 5070 preflight processed a public support then query, produced finite
gradients in all `60` fresh decoder tensors, and produced zero gradient tensors
in the frozen core and V8 fusion. Both output paths remained absent. The first
frozen result may proceed without tuning.

## Preserved result

The first result is accepted without tuning and classified `NOT_SUPPORTED`:

- result SHA-256
  `9CAA43C30CAE3192C54378C162B1E39EE642E2BAA1F8E4C0F69BE325ACAE8C0A`;
- checkpoint SHA-256
  `2EDDED9DC7058490522D6F73AE5035E56AB87662FFF93B03E9514949C0478266`;
- runtime `541.739` seconds, peak CUDA allocation `610,596,352` bytes;
- training composition exactness rose `0.25` to `0.8046875`;
- development/full `0.21875`; final/full `0.25`;
- final edge removal `0.15625`, coverage removal `0.0`, donor removal
  `0.1875`, unrelated Cognee retrieval `0.21875`, reset `0.1875`;
- public-role removal, Moving-Origin removal, and procedure-slot removal each
  equaled full `0.25`.

Fresh decoding removed V9's inherited training shortcut and established
learned edge and coverage effects, but the unary-role, Moving-Origin, and core
slot paths were ignored. More epochs under this identity are not authorized.
