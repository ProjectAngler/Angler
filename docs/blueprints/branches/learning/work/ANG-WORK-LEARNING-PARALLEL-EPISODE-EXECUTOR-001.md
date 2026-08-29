# ANG-WORK-LEARNING-PARALLEL-EPISODE-EXECUTOR-001

## Identity

- Kind: reusable testing/training execution accelerator
- Parent: `ANG-BP-LEARNING` / `ANG-BP-RETENTION`
- Protocol: `phase6.parallel-episode-executor.v1`
- Accountable outcome: reduce wall time for independent Angler training/evaluation episodes without changing any episode's sequential learner mathematics.

## Boundary

This leaf accelerates only independent experiment episodes, panels, worlds, or hypothesis arms. It does not parallelize causally dependent updates inside one online learner, change Angler reasoning, alter thresholds, or make a capability claim. Each worker loads the same frozen learned checkpoint, retains the exact per-panel stream and arm order, and has no communication with other workers until terminal deterministic merge.

The first bounded benchmark uses the sealed V22 checkpoint and panels `0..3`. For each panel it runs the first 64 lifetime updates with the exact V22 feature capture, six primary arms, functional AdamW transition, seeds, and FP32 numerical mode. Run the same four panel jobs sequentially and concurrently. The semantic worker record excludes elapsed time and process identity and contains exact online accumulators, final fast-state digests, source/gate/controller digests, update count, and cleanup/resource checks.

## Literal outputs

- `experiments/runners/parallel_episode_executor.py`
- `tests/unit/experiments/test_parallel_episode_executor.py`
- `.angler_parallel_episode_benchmark_once.py`
- claim: `/opt/angler/results/parallel-episode-executor-v1.claim.json`
- sequential worker receipts: `/opt/angler/results/parallel-episode-executor-v1.sequential-panel-{0..3}.json`
- parallel worker receipts: `/opt/angler/results/parallel-episode-executor-v1.parallel-panel-{0..3}.json`
- result: `/opt/angler/results/parallel-episode-executor-v1.json`
- failure: `/opt/angler/results/parallel-episode-executor-v1.failure.json`

## Acceptance

1. focused tests cover panel/step bounds, deterministic command construction, exact semantic merge ordering, worker failure propagation, occupied-output rejection, and fallback eligibility;
2. no-write CUDA preflight loads the sealed V22 checkpoint and executes one exact update with controller/gate immutability and hook cleanup;
3. sequential and concurrent runs produce byte-equivalent canonical semantic records for every corresponding panel;
4. all four panels complete exactly 64 updates and all six arm states end at Adam step 64;
5. concurrent wall time is lower than sequential wall time; record speedup without tuning a pass threshold;
6. peak aggregate CUDA allocation remains below 2 GiB and no worker changes the checkpoint or repository.

If semantic equality fails, parallel execution is rejected regardless of speed. If equality passes but speed does not improve, the infrastructure remains mechanically valid but is not selected for the full successor. Runtime integration must retain automatic sequential fallback for resource/process failure; the benchmark itself records failures rather than concealing them.

## Human-Flourishing and effects

LOW local synthetic testing/training infrastructure mapped to `ANG-GATE-HUMAN-FLOURISHING-001`. No model/LLM execution, network, packages, real-person/recovered data, service, deployment, external action, promoted-state mutation, or learner authority. Deterministic code is limited to orchestration, integrity, measurement, and merge.

## Resources and stop

- one RTX 5080; four isolated foreground worker processes
- 64 updates per worker; six arms; four panels
- aggregate CUDA allocation ceiling: 2 GiB
- wall ceiling: 30 minutes
- stop for source/checkpoint mismatch, output occupation, non-finite value, hook leak, worker nonzero exit, semantic mismatch, memory breach, or timeout

## Rollback and next action

Before claim, remove only this leaf's three fresh implementation files. After claim preserve all receipts and terminal evidence. A successful benchmark authorizes reuse of the worker/merge layer in the fresh 20-step full ANML successor; it does not authorize rerunning V22 or changing V23-D1/R1.

## Pre-execution recovery

The first benchmark identity failed before any worker imported V22 or produced a receipt. Workers were invoked by the runner's filesystem path, which made Python place `experiments/runners` rather than the repository root on `sys.path`; all four sequential workers returned `ModuleNotFoundError: experiments`. The claim and failure are preserved under `/opt/angler/results/parallel-episode-executor-v1*`; no timing, semantic metric, or scientific value was exposed.

Recovery identity `phase6.parallel-episode-executor.v1-r1` changes only worker invocation to `python -B -m experiments.runners.parallel_episode_executor` and uses fresh `v1-r1` output paths. Panel computations, checkpoint, steps, arms, ordering, comparison, and acceptance remain unchanged. Accept its first result without tuning.
