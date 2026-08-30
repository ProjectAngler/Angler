# Scaled procedural software V2 result

Classification: `NOT_SUPPORTED`

V2 reused the exact V1 checkpoint and all 32 final queries, seven arms,
thresholds, parser, and generation budget. It changed only the publication
boundary by appending `answer=` after all task/evidence text.

The run completed in 162.63 seconds and preserved target-evidence attribution
at `0.999815`. All seven hidden execution arms still scored zero. The cue fixed
the demonstrated continuation defect, but it exposed the deeper interface
failure: the learned latent prefix was not a reliable decoder for executable
local action sequences. Full Angler occasionally emitted sequences such as
`A C`, `A B C`, or `A C STOP`, but none solved a hidden composed query; the
dominant full response was `1`. Fair retrieval and prefix-removed responses
were almost always `?`.

No further prompt repair is warranted. The next mechanism is a trainable,
candidate-conditioned procedure decoder over Angler's latent slots. That
decoder is part of the reasoning core, not a deterministic solver: it must
learn sequence decisions from visible support traces and generalize to
untouched composed queries. Qwen can then receive the explicit Angler plan for
language/tool publication.

Result SHA-256:
`902a7eb87275eb7b125484f7393bff1fb891ac6d56ee20f1dfb7011b8a3c6f4d`.
