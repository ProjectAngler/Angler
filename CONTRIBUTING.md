# Contributing to Project Angler

Thank you for helping investigate persistent procedural learning. Contributions
from individual researchers, open-source teams, and industry laboratories are
welcome.

## Start with the claim

Read [`ANGLER_FOR_AI.md`](ANGLER_FOR_AI.md) and
[`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md) before proposing an architectural
change. Angler's central question is whether experience causes improvement
through one persistent learned competence state. More tokens, retrieval,
task-specific routing, or a deterministic solver may be useful controls, but
they are not evidence for that claim.

## Useful contributions

- reproduce a preserved result on a new environment;
- identify leakage, attribution gaps, or misleading metrics;
- integrate established continual-learning or test-time-learning mechanisms;
- improve batching, profiling, portability, and multi-GPU execution without
  changing experimental semantics;
- build a causal foundation-model integration with live/reset/frozen/removed
  Angler controls;
- design cross-mechanism or cross-domain procedural-transfer evaluations;
- improve documentation, installation, and independent reproducibility.

## Development setup

The reference environment is Ubuntu 24.04, Python 3.12, PyTorch
`2.13.0+cu130`, NumPy `2.5.2`, SciPy `1.18.1`, Transformers `5.15.1`, PEFT
`0.20.0`, Accelerate `1.14.0`, Safetensors `0.8.0`, and pytest `9.1.1`.

Install a PyTorch build appropriate for your accelerator from the official
PyTorch instructions, then install the remaining libraries. Add `src` to the
Python import path when running directly from a checkout. Exact historical GPU
runs may require their preserved external checkpoints and runner-specific
environment.

Examples that do not consume a frozen scientific identity:

```powershell
python -B -m unittest tests.unit.evidence.test_evidence_schemas
pwsh -NoProfile -File tests/synthetic/slice00/resources/Test-Cr0ResourceScaffold.ps1
pwsh -NoProfile -File docs/blueprints/branches/safety/tests/Test-Cr0SafetyDesign.ps1
```

## Scientific discipline

- State the hypothesis, evaluation identity, thresholds, controls, and stop
  conditions before viewing adaptive results.
- Never tune or silently rerun a consumed one-shot identity.
- Preserve failed and null results alongside successful results.
- Separate mechanism support from unrestricted capability claims.
- Keep deterministic code to plumbing, integrity, orchestration, controlled
  world transitions, and measurement—not the procedure claimed as learned.
- Disclose external code, paper, model, dataset, and AI-assistance provenance.

## Pull requests

Keep each pull request focused. Explain the interface need, changed files,
tests, observed limitations, and how the change can be removed or replaced.
Do not commit credentials, private conversations, personal data, model weights,
or generated checkpoints.

Do not submit code, data, model material, documentation, or generated output
unless you have the right to contribute it. Preserve every applicable
third-party notice and identify the exact source, version, and license in the
pull request. A paper citation alone is not a source-code license.

Every contributed commit must include a `Signed-off-by` line certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Use `git commit -s` to add it. This is a lightweight origin certification, not
a copyright assignment or contributor license agreement.

By intentionally submitting a contribution for inclusion in Project Angler,
you license that contribution under the repository's Apache License 2.0. If a
separate written agreement applies, disclose it before the contribution is
reviewed or merged.
