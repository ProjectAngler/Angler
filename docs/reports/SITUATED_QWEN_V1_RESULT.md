# Situated Qwen V1 result

Project Angler's first same-Qwen causal generation comparison is preserved as
`NOT_SUPPORTED`.

## Result

| Arm | Correct | Accuracy |
|---|---:|---:|
| Frozen Qwen3-4B alone | 2/8 | 0.25 |
| Frozen Qwen3-4B + unordered fair Cognee retrieval | 0/8 | 0.00 |
| Frozen Qwen3-4B + Angler-selected experience | 4/8 | 0.50 |

Result SHA-256:
`4A795447A37F9CF528094FBE0C4FBFDA266C3F423F8C6E881BD7E5B58002340F`.
The accepted live-Cognee recall and sealed reader hashes remained
`C59083344F1ACCD8D44BD002922138D26628F4E42895C806D61F360185E8DD45`
and `B94E27AD0A42E3F499B7DE5FF1B5217463DADD38E60934A7AF7F3EBD2DA4E3D4`.
No training occurred. Runtime was 8.46 seconds on the RTX 5080 and peak CUDA
allocation was 8,831,196,672 bytes.

## Diagnosis

The learned selector supplied the target-bearing experience in seven of eight
rows. Qwen converted four of those selections into the required answer. In
two additional target-bearing rows, Qwen began a prose explanation and reached
the frozen 12-token ceiling before emitting an action. It returned a wrong
action in one target-bearing row. The remaining row inherited the selector's
known error.

This is evidence of a useful signal, not a passed integration: the attached
arm doubled same-Qwen accuracy, but missed the preregistered 0.75 floor. A
separate interface-hardening successor may test a response-first contract with
enough terminal-publication budget. Such a successor must retain this result,
must not alter the reader, recalls, or Qwen weights, and cannot be presented as
fresh learner-generalization evidence because the same eight rows were already
observed here.

## V2 interface-hardening result

The predeclared response-first, 64-token successor completed once and passed:

| Arm | Correct | Accuracy |
|---|---:|---:|
| Frozen Qwen3-4B alone | 1/8 | 0.125 |
| Frozen Qwen3-4B + unordered fair Cognee retrieval | 3/8 | 0.375 |
| Frozen Qwen3-4B + Angler-selected experience | 7/8 | 0.875 |

V2 result SHA-256:
`A3E9E04C7A709C913E05682055C0317A2AD8C28EAAE30D370433F4BA686A1A42`.
It passed the frozen 0.75 attached-accuracy floor and exceeded each control by
at least 0.25. No model or reader training occurred; runtime was 7.16 seconds
and peak CUDA allocation was 8,831,196,672 bytes.

V2 establishes that the integration can transmit Angler's selected experience
through frozen Qwen into the correct bounded output. It is an engineering
interface result on already observed V1 rows, not fresh evidence of broader
reasoning or learner generalization. The next scientific step requires new,
meaningfully complex tasks and online feedback, not more tuning of these rows.
