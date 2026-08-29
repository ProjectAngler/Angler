# Third-party technology, licenses, and research provenance

This record distinguishes software actually used by Project Angler from ideas
described in research papers and from technologies that are only candidates for
future work. A citation to a paper or repository is not permission to copy its
source code. Unless this file says otherwise, no third-party source, model
weights, dataset, checkpoint, or generated artifact is distributed in this
repository.

This is a technical provenance record, not legal advice. It was last checked on
2026-08-29.

## Code incorporated or adapted

### IDSIA `automated-cl` self-referential weight matrix

- Project file: `src/angler/reasoning/self_referential_memory.py`
- Upstream: <https://github.com/IDSIA/automated-cl>
- Inspected commit: `3d7b53adb4b6b43acd82b9a381a2c631d0e59a5d`
- Upstream license: MIT
- Upstream copyright: Copyright (c) 2023 Kazuki Irie
- Local boundary: an Angler-native pure-PyTorch adaptation of the published
  self-referential weight-matrix state representation, initialization, and
  four-block delta equations. Upstream custom CUDA code is not included.
- Preserved license: `third_party/licenses/IDSIA-automated-cl-MIT.txt`

The MIT terms apply to material derived from this upstream work. Project
Angler's surrounding original code remains under the repository's Apache-2.0
license.

## Runtime and development dependencies

These packages are imported or are part of the recorded reference environment.
They are installed separately and are not vendored in this repository. Their
own distributions may contain additional third-party notices and license
expressions; redistributors of an environment or binary bundle must preserve
those notices rather than relying only on this summary.

| Technology | Recorded version | Project use | Upstream license | Official source |
|---|---:|---|---|---|
| PyTorch | `2.13.0+cu130` | Tensor operations, neural modules, optimization, CUDA execution | Installed metadata reports `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`; the main project license is BSD-3-Clause | <https://github.com/pytorch/pytorch> |
| Transformers | `5.15.1` | Loading and running the frozen Qwen-family backbone | Apache-2.0 | <https://github.com/huggingface/transformers> |
| PEFT | `0.20.0` | LoRA adapter creation and state handling | Apache-2.0 | <https://github.com/huggingface/peft> |
| Safetensors | `0.8.0` | Tensor serialization | Apache-2.0 | <https://github.com/huggingface/safetensors> |
| Accelerate | `1.14.0` | Installed support dependency; not directly imported by current Angler source | Apache-2.0 | <https://github.com/huggingface/accelerate> |
| NumPy | `2.5.2` | Recorded scientific environment; not directly imported by current Angler Python source | Installed metadata reports `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | <https://github.com/numpy/numpy> |
| SciPy | `1.18.1` | Recorded scientific environment; not directly imported by current Angler Python source | BSD-3-Clause with bundled-library notices in its distribution | <https://github.com/scipy/scipy> |
| pytest | `9.1.1` | Development and test environment | MIT | <https://github.com/pytest-dev/pytest> |

Python's standard library is used extensively and is not itemized here.

## External model used in local experiments

The local experiments used `Qwen/Qwen3-4B` as a frozen semantic backbone.

- Model page: <https://huggingface.co/Qwen/Qwen3-4B>
- Locally recorded Hub revision:
  `1cfa9a7208912126459214e8b04321603b3df60c`
- License reported by the model card and included local model license:
  Apache-2.0
- Local license-file SHA-256:
  `832DD9E00A68DD83B3C3FB9F5588DAD7DCF337A0DB50F7D9483F310CD292E92E`
- Distribution boundary: no Qwen weights, tokenizer payloads, Hub cache, or
  generated model output is included in this repository.

The separate `QwenLM/Qwen3` GitHub repository was inspected only as a project
reference. Its README states that open-weight models use Apache-2.0 and points
to the individual Hugging Face repositories for their license files. No source
from that GitHub repository is incorporated here.

## Research concepts independently implemented

The following works informed Angler mechanisms. Except for the expressly
identified `automated-cl` adaptation above, the implementation was written for
Angler without copying donor source, configurations, data samplers,
checkpoints, or hyperparameter files.

| Concept | Angler use | Paper or official project | Inspected implementation and license |
|---|---|---|---|
| FTTT | Bounded feedback-time LoRA update structure | <https://github.com/LaVi-Lab/FTTT> | Commit `db2f244a0b5f644e6f48b4069a420fea8b7b3cc6`, MIT; concept-level reuse only |
| Backward Learning | Reversed transition learning and backward rollout | <https://arxiv.org/abs/2312.05044> | <https://github.com/Hauf3n/Backward-Learning-for-Goal-Conditioned-Policies> commit `78b8cb266fe460991667a69bd1befeb14aa58279`, MIT; no donor source copied |
| Hindsight Experience Replay | Future-state relabeling of observed trajectories | <https://arxiv.org/abs/1707.01495> | <https://github.com/openai/baselines> commit `ea25b9e8b234e6ee1bca43083f8f3cf974143998`, MIT; no donor source copied |
| Sub-Goal Trees / SGT-PG | Shared-midpoint auxiliary target | <https://proceedings.mlr.press/v119/jurgenson20a.html> | <https://github.com/tomjur/SGT-PG> commit `89f027c1ba90c0e32bb1601e4890af916e9a9f2d`, MIT; no donor source copied |
| Graph Matching Networks | Cross-graph attention and mismatch messages | <https://arxiv.org/abs/1904.12787> | <https://github.com/google-deepmind/deepmind-research> repository commit `f5de0ede8430809180254ee957abf36ed62579ef`, GMN subtree commit `451d2964904a4e71d8d28ac45cdc5f33c1db1b19`, Apache-2.0, notebook copyright 2019 Google LLC; independently reimplemented in PyTorch |
| OML | Slow representation / fast prediction-head meta-learning | <https://arxiv.org/abs/1905.12588> | <https://github.com/kjaved0/mrcl> commit `2855a6b7e820f171432981b58c49664fcdbf00ed`; no LICENSE, COPYING, or NOTICE found, so no repository source was used |
| ANML | Learned neuromodulation of online plasticity | <https://arxiv.org/abs/2002.09571> | <https://github.com/uvm-neurobotics-lab/ANML> commit `cabaaf7f2336496cd51065847c524a705408d562` and <https://github.com/uvm-neurobotics-lab/higherANML> commit `ce088d9efc7d9298bced02352e81ece81c3530b4`; neither inspected tree contained a LICENSE, COPYING, or NOTICE file, so no repository source was used |

The LoRA technique is described by Hu et al., *LoRA: Low-Rank Adaptation of
Large Language Models*, <https://arxiv.org/abs/2106.09685>. Angler uses the
Apache-2.0 PEFT implementation rather than copying Microsoft's MIT-licensed
reference source.

## Reference-only and future candidates

The Tier-0 blueprint names the projects below as possible donors or design
references. No code from them is currently incorporated unless a different
section of this file says so. A future contribution must repeat license and
provenance review at the exact commit it proposes to adopt.

| Candidate | Intended design role | License observed on 2026-08-29 |
|---|---|---|
| [FTTT](https://github.com/LaVi-Lab/FTTT) | Feedback-time learning | MIT |
| [TorchOpt](https://github.com/metaopt/torchopt) | Differentiable optimization | Apache-2.0 |
| [SGSD](https://github.com/walawalagoose/SGSD) | Skill and mistake consolidation | MIT |
| [SEAL](https://github.com/Continual-Intelligence/SEAL) | Self-edit generation | MIT |
| [SPADE](https://github.com/spade-rl/spade) | Adaptive environment generation | MIT |
| [Voyager](https://github.com/MineDojo/Voyager) | Executable tool lifecycle | MIT |
| [TTT-LM PyTorch](https://github.com/test-time-training/ttt-lm-pytorch) | Neural test-time state | MIT |
| [End-to-End TTT](https://github.com/test-time-training/e2e) | Meta-learned test-time state | No LICENSE, COPYING, or NOTICE found at inspected commit `a4fc4788ace38e29b5067916d4f4be33da894085`; reference only |
| [Darwin Godel Machine](https://github.com/jennyzzt/dgm) | Optional late controller/tool evolution | Apache-2.0 |
| [Titans](https://arxiv.org/abs/2501.00663) | Test-time neural long-term memory | Paper reference only; no source incorporated |
| [Nested Learning / Hope](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) | Multi-timescale architecture, memory, and optimization | Research reference only; no source incorporated |

## Source-similarity audit

The current Python implementation was compared with the inspected donor trees
using tokenized source. Comments and formatting were removed. Two checks were
run:

1. exact contiguous source matches of at least 16 Python tokens;
2. normalized structural matches of at least 40 tokens, replacing identifiers
   and literals with type placeholders to detect simple renaming.

| Angler area | Donor trees checked | Result |
|---|---|---|
| FTTT update | `LaVi-Lab/FTTT` | No exact 16-token or normalized 40-token match |
| OML | `kjaved0/mrcl` | No exact 16-token match; only generic normalized boilerplate matches |
| ANML | `ANML`, `higherANML` | No exact 16-token match; one generic normalized boilerplate match |
| Bidirectional/HER/SGT | Backward Learning, SGT-PG, OpenAI Baselines | No exact 16-token match; only generic normalized training/test boilerplate matches |
| Graph matching | DeepMind GMN notebook | No exact 16-token or normalized 40-token match |
| SRWM | `IDSIA/automated-cl` | No exact 16-token match; an 80-token normalized structural match in the donor layer confirms the declared equation-level adaptation and is covered by the preserved MIT notice |

This is evidence against direct textual copying at the tested thresholds, not
a legal conclusion and not proof that shorter or non-Python fragments can
never coincide. A separate prose/source-text scan found no exact contiguous
18-word match between the candidate public snapshot and the non-license text
of the same donor trees. The inspected donor repositories remain outside this
Git repository.

## AI assistance

OpenAI Codex materially assisted with source construction, documentation,
testing, and this audit. The human owner supplied the project objective,
architecture direction, acceptance decisions, corrections, and publication
authorization. Anthropic Claude supplied limited consultation relayed by the
owner; Claude-authored consultation text and donor implementation code are not
included in the public source.

OpenAI's current terms state that, as between the user and OpenAI and to the
extent permitted by law, the user owns output and OpenAI assigns any interest
it has in that output. The same terms warn that Codex output may be subject to
third-party licenses and that outputs may not be unique. Copyrightability and
authorship of AI-assisted work remain jurisdiction- and fact-specific.

## Scope and limits of this audit

This inventory documents the sources, versions, licenses, implementation
boundaries, and technical similarity checks that were available before the
candidate public release. It is not a legal opinion or a trademark, patent,
employment-rights, export-control, or jurisdiction-specific clearance search.
The Apache-2.0 grant applies only to rights that contributors own or are
authorized to grant; it does not imply ownership of third-party names, models,
datasets, papers, or independently licensed components.
