# ANG-WORK-LEARNING-SCALABLE-PROCEDURAL-CORE-V1-001

Status: complete — `SCALED_CORE_INTERFACE_READY`

Assurance: local experimental component; no promotion authority

## Outcome

Create the first resource-scalable Angler procedural core that can grow beyond
the tiny task-specific readers while preserving the established trifecta:

- Cognee supplies replaceable semantic candidates;
- Moving Origin supplies trusted autobiographical and validity coordinates;
- Angler performs multi-round learned procedure construction and owns bounded
  persistent plastic state;
- a frozen Qwen supplies language and knowledge, and receives learned Angler
  procedure tokens plus explicitly selected evidence.

## Stable interface

Inputs are detached query embeddings, candidate embeddings, temporal features,
candidate masks, and an optional fixed-capacity plastic state. Outputs are
candidate scores/weights, latent procedure slots, Qwen-width procedure-prefix
tokens, and a proposed next state. Outcomes enter only through the explicit
feedback update. No answer labels, task IDs, hand-written solution procedures,
or token-prediction objective live inside the core interface.

## Resource tiers

| Tier | Width | Depth | Heads | Procedure tokens | Plastic slots | Intended use |
|---|---:|---:|---:|---:|---:|---|
| compact | 256 | 4 | 8 | 8 | 16 | CPU/unit diagnostics |
| workstation | 512 | 8 | 8 | 16 | 64 | Qwen3-4B + Angler on a 16-GB GPU |
| dedicated | 768 | 12 | 12 | 32 | 128 | separate Angler GPU / larger host |

Selection is based on declared available bytes and training/inference headroom,
not a hard-coded 16-GB product limit. Width, depth, slots, and reasoning rounds
scale together. The first implementation must prove monotonically increasing
capacity, fixed-shape state, masking, finite forward/backward behavior,
state-reset identity, outcome-causal state change, and Qwen-width prefix output.

## First validation boundary

Benchmark all tiers with synthetic tensors, then run the largest empirically
safe tier on the RTX 5080 with real Qwen embeddings and the preserved situated
recall. Record parameters, state bytes, peak allocation, latency, candidate
attribution, prefix shapes, and removal/reset effects. This is a component
readiness test, not evidence of complex reasoning. Only after it passes may a
fresh multi-step software task train and evaluate the scaled core.

## First benchmark

The first frozen benchmark passed. Result SHA-256:
`82352E90F4181006449D1A3BDC56676A6102187ED9DCEA3D814D3F1D91345E79`.

| Tier | Parameters | FP32 parameters | Plastic state | Batch-8 forward | Peak with Qwen |
|---|---:|---:|---:|---:|---:|
| compact | 7,714,585 | 30.9 MB | 32,832 B | 3.81 ms | 8.18 GB |
| workstation | 47,846,425 | 191.4 MB | 262,400 B | 6.07 ms | 8.51 GB |
| dedicated | 151,871,769 | 607.5 MB | 786,944 B | 11.66 ms | 9.36 GB |

After frozen Qwen and embeddings were resident, 6.53 GB remained free and the
planner selected the dedicated tier while retaining its 25% declared reserve.
All tiers passed finite backward flow, fixed-state, feedback-causality, and
reset checks. The dedicated tier supplied 32 latent procedure tokens to frozen
Qwen, which completed the soft-prefix smoke request with `READY`.

This is capacity and interface evidence only. The core is untrained, so its
near-uniform candidate entropy and random prefixes have no reasoning claim.
Next is one fresh multi-step software-task training/evaluation identity with
live/reset/memory-removed/prefix-removed controls.
