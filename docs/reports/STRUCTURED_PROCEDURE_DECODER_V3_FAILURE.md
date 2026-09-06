# Structured Procedure Decoder V3 — Harness Failure

Date: 2026-08-30

Classification: `HARNESS_ERROR_NO_SCIENTIFIC_RESULT`

The first frozen V3 execution completed its Qwen embedding phase and failed on
the first decoder update. `ScalableProceduralCore` output had been produced
under `torch.inference_mode()`. PyTorch correctly rejected using that inference
tensor as input to a trainable linear layer because the layer must retain its
input to compute decoder-weight gradients.

Terminal exception:

```text
RuntimeError: Inference tensors cannot be saved for backward. Please do not use
Tensors created in inference mode in computation tracked by autograd.
```

No result or checkpoint was created at either declared V3 output path. No
training update, evaluation, threshold inspection, or scientific
classification occurred.

Frozen failed identities:

- leaf SHA-256: `488DB40E4B826EDE6FC0923818B5E2661049A52521E0A7EB7A1BEBE24BE5828D`;
- runner SHA-256: `28880E1E44C2BF3D4D3AE298678702755E085E2370FBF5C7A97EA03B393DBE41`;
- decoder SHA-256: `11900A023506605FBDDCE437D374039F98B752AE4586479EAF1963E183A62BC0`;
- parent checkpoint SHA-256:
  `08CB21C1485CEE41565A79B28AFDABF1B14A3A379F0D8044CB06870A3D7A1DF2`;
- parent V2 result SHA-256:
  `902A7EB87275EB7B125484F7393BFF1FB891AC6D56EE20F1DFB7011B8A3C6F4D`.

The original V3 identity is consumed. A recovery may change only the tensor
plumbing needed to clone an immutable inference tensor into an ordinary
detached tensor before decoder autograd. It may not change the corpus, model,
checkpoint, decoder architecture, optimizer, seed, updates, controls,
thresholds, or interpretation.
