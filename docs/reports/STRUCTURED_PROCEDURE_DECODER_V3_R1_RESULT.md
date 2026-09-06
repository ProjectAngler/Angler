# Structured Procedure Decoder V3-R1 Result

Date: 2026-08-30

Classification: `NOT_SUPPORTED`

V3-R1 recovered only the consumed V3 inference-tensor harness error. The
scientific settings remained frozen and the first recovered result was
accepted without tuning.

## Result

- result SHA-256:
  `EF2C56461CC8CC72200BF2E03B94E07D6C12EA41BA778A9C8D79E5A836174698`;
- checkpoint SHA-256:
  `9CD6D34F0E9BF651362899F541E9FE5A51045D2FA996032732EFF4E7C3F5A9D7`;
- development exact sequence: `0.15625`;
- full hidden execution: `0.0`;
- reset, coordinates removed, unrelated evidence, and slots removed: `0.0`;
- target-evidence attribution: `0.9998148698359728`;
- runtime: `1144.8567` seconds;
- peak CUDA allocation: `12,306,228,736` bytes.

Training loss fell from `0.77411` to `0.64385`, and teacher-forced training
exactness rose from `0.06641` to `0.48047`. This was insufficient for held-out
composition.

## Diagnostic interpretation

Full, reset-state, coordinates-removed, and unrelated-evidence arms emitted
the same sequences. Most sequences contained two actions followed by STOP.
The frozen curriculum had exposed only two-action single-motif support targets,
while final queries require four-action composition. The decoder therefore
learned the demonstrated stopping horizon and relied mainly on local candidate
features; it did not establish causal use of Angler's latent procedure state.

The appropriate successor is not post-result epoch tuning. It must expose
several varied, explicitly public four-action compositions in the training
partition, train the procedural core and decoder jointly so sequence credit
reaches the slots and plastic write path, and retain wholly separate
development/final mechanisms and the existing removal controls.
