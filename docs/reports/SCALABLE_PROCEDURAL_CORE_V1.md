# Scalable procedural core V1

Project Angler now has a resource-scalable latent procedural-core interface
that preserves the Angler + Cognee + Moving Origin + frozen-Qwen composition.

## Architecture

The core reads detached Qwen query/candidate representations and trusted
Moving Origin coordinates. Multi-round self-, evidence-, and plastic-memory
attention constructs latent procedure slots. The same state produces candidate
attribution and Qwen-width soft procedure tokens. An explicit outcome update
writes into fixed-capacity learned key/value memory without replay or growth.

The core does not predict language tokens, contain a solution table, or update
Qwen. Cognee remains replaceable candidate storage; Moving Origin remains the
trusted temporal coordinate system.

## RTX 5080 benchmark

Result SHA-256:
`82352E90F4181006449D1A3BDC56676A6102187ED9DCEA3D814D3F1D91345E79`.

| Tier | Width/depth | Parameters | Procedure tokens | Plastic slots/state | Batch-8 latency | Peak allocation with Qwen |
|---|---|---:|---:|---:|---:|---:|
| compact | 256/4 | 7.71M | 8 | 16 / 32.8 KB | 3.81 ms | 8.18 GB |
| workstation | 512/8 | 47.85M | 16 | 64 / 262 KB | 6.07 ms | 8.51 GB |
| dedicated | 768/12 | 151.87M | 32 | 128 / 787 KB | 11.66 ms | 9.36 GB |

All tiers produced finite forward/backward paths. Outcome feedback changed the
fixed-size state, and removing that update restored the initial identity and
changed both latent procedure slots and Qwen prefix tokens. With Qwen3-4B and
the preserved embeddings resident, 6.53 GB remained free; the declared
training planner selected the dedicated tier while reserving 25% of that
reported headroom. Frozen Qwen accepted the dedicated core's 32 latent tokens
and completed the smoke prompt with `READY`.

## Interpretation

This is the scale breakthrough requested by the owner: the operative core can
now grow from 7.7M to 151.9M parameters on the current machine while retaining
one interface, rather than stretching a 33-value or 0.17M task adapter beyond
its purpose. It does **not** yet show improved complex reasoning because these
weights are untrained. The next evidence-bearing step is training on fresh
multi-step software tasks and comparing live state against reset,
plastic-memory removal, procedure-prefix removal, and same-Qwen controls.
