# ANG-WORK-LEARNING-ANML-FIDELITY-V23-D1-R1-001

## Identity

- Kind: bounded failed-publication recovery leaf
- Parent: `ANG-WORK-LEARNING-ANML-FIDELITY-V23-D1-001`
- Protocol: `phase6.public-anml-fidelity.v23-d1-r1`
- Accountable outcome: recover the first interpretable V23-D1 optimizer/horizon diagnostic after preserving its terminal JSON-key failure, without changing any scientific input, mechanism, stream, seed, schedule, threshold, or classification.

## Trigger and diagnosis

The first V23-D1 identity completed its frozen CUDA computation and then failed before returning or publishing metrics. `evaluate_pilot` stored probe milestones as integer mapping keys; the inherited V22 JSON-safety validator correctly rejected those keys with `V22 JSON output keys must be strings`. The harness preserved the claim and failure and published no result.

Immutable failed evidence:

- source protocol: `phase6.public-anml-fidelity.v23-d1`
- source leaf SHA-256: `474A7A75EC6D5C1864C573DBF3D8CD61AB568121AE61F97A857F30B7D8AA4901`
- source runner SHA-256: `0DA7D4FFCBF7111ACB2E6F06CE5A33BF0ADFA1DB35E5C42748FB8DC2B0F99BE9`
- source test SHA-256: `48FA2665B324E937046F3C010A6307ABBD16AC985482ECBF3416C8CEB2D18D1B`
- source harness SHA-256: `A4B0F59C71D054DB84C0C344385BCF6107E711DD40DA34FC510DF3446F1AA5C8`
- failed claim SHA-256: `51DC9DB4C72DCBC076A78B8ED9392FAD6B348D52E350C01F3136B697C8052CAE`
- failed record SHA-256: `B1719D5580A40DA82520042EACC69DB8CE8D901A549D7DDF29EF7060EDAF7852`
- source result: absent

The failure disclosed no metric, condition ranking, classification, or adaptive value. The source identity is consumed and must remain failed.

## Exact recovery mechanism

The recovery runner imports the frozen source runner unchanged. While invoking exactly its `run("cuda")` function, it temporarily replaces only the terminal V22 JSON validator with a no-op, restores the validator in `finally`, recursively converts every mapping key to its exact string spelling, rejects any stringification collision or non-finite/non-JSON value, changes only the outer artifact/protocol identity to this recovery identity, and then invokes the original strict V22 validator on the recovered object.

The wrapper may not modify the source runner, V22 checkpoint/evidence, optimizer, horizon, architecture, gate range, random seeds, generated streams, panels, probes, thresholds, classification code, device, numeric mode, resource ceilings, or first-result semantics. It may not inspect a metric before recovery execution.

## Literal outputs

- `experiments/runners/phase6_anml_fidelity_v23_d1_r1.py`
- `tests/unit/experiments/test_phase6_anml_fidelity_v23_d1_r1.py`
- `.angler_v23_d1_r1_once.py`
- claim: `/opt/angler/results/phase6-anml-fidelity-v23-d1-r1.claim.json`
- result: `/opt/angler/results/phase6-anml-fidelity-v23-d1-r1.json`
- failure: `/opt/angler/results/phase6-anml-fidelity-v23-d1-r1.failure.json`

No checkpoint, progress file, model, dataset, service, network action, package installation, Qwen execution, external effect, or promoted-state mutation is permitted.

## Human-Flourishing and impact

LOW local synthetic development work mapped to `ANG-GATE-HUMAN-FLOURISHING-001`. It uses no real-person or recovered data and grants no capability, deployment, promotion, autonomy, or safety-gate claim. Human authority, truthful labeling, bounded resources, preserved failure evidence, and stop control remain intact.

## Preflight and acceptance

Before execution:

1. all immutable source and failed-evidence hashes match and the source result remains absent;
2. every recovery output is absent;
3. focused tests prove nested integer milestone keys become strings, collisions fail closed, non-finite values fail closed, and the temporary validator is restored after both success and injected error;
4. tests prove the recovery delegates once to the frozen source `run`, and configuration/schedule/classification objects are inherited unchanged;
5. synthetic CUDA preflight passes with no semantic update;
6. local and WSL source hashes match.

Then execute this recovery identity exactly once. Accept its first terminal classification without tuning. Report both `HARNESS_ERROR_PRESERVED` for the original identity and the recovered classification. A recovery failure is preserved and consumes this identity.

## Resources and stop conditions

- device: one CUDA device
- allocation ceiling: 2 GiB, inherited from V23-D1
- wall ceiling: 90 minutes, inherited from V23-D1
- no network, package, model, real-data, background-service, or cross-machine effects

Stop for any hash mismatch, occupied output, source-result appearance, serialization collision, non-finite value, validator-restoration failure, source drift, resource breach, or authority ambiguity.

## Rollback

Before launch, recovery implementation files may be removed individually. After claim creation preserve claim/result/failure. Never remove or rewrite the original V23-D1 claim, failure, leaf, runner, test, or harness.

## Terminal result

Status: `COMPLETE`. Frozen pre-execution leaf SHA-256: `246C987BDC5D051C0B992F312211EAFCC48005BF0B635F4C86750D68A2095E31`. The original V23-D1 identity remains `HARNESS_ERROR_PRESERVED`; original claim/failure hashes remain exact and its result remains absent.

R1 claim SHA-256 is `5817B6F494BEDA06B6178F79B0850799EE447AECC9F2EA0CFF82C666E0BEEC31`. R1 report `/opt/angler/results/phase6-anml-fidelity-v23-d1-r1.json` has SHA-256 `C59418EE30504E40A4206E47D0B19F4F903E93341B9651C33D7D72AB264CBEB9`; no R1 failure exists. Runtime was 1,145.751 seconds and peak CUDA allocation was 75,690,496 bytes. Controller digests match before/after.

Independent recomputation agrees with `FULL_V23_ELIGIBLE` and selector `adamw_20`. The development label means only that all four tested configurations had positive second-order-over-first-order AUC and terminal direction in both 512-update panels while not losing to always-open. AdamW-20 had the largest minimum AUC advantage, `0.0192216%`, versus AdamW-8 `0.0061738%`; its minimum terminal advantage was `0.0192994%`. This selects the longer credit horizon as the next direction but is not a material full-scale ANML, Qwen, promotion, or AGI result.

Next exact action: if ANML confirmation remains the priority, preregister one fresh full-scale V22-equivalent test that changes only the inner credit horizon from 8 to 20, retains AdamW, retains the original 4,096-update deployment and causal controls, and requires a meaningful effect floor rather than mere positive floating-point direction. Do not tune or rerun D1/R1.
