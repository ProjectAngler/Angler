---
handoff_id: JENNY2-CLAUDE-TAKEOVER-2026-09-04
snapshot_time: 2026-09-04T18:42:06-04:00
status: ready_for_peer_takeover
repository: /opt/angler/src/angler
live_host: angler-workstation
---

# Jenny 2.0: detailed Claude Code takeover handoff

This document is a compact operational snapshot, not a replacement for the
Project Angler blueprint hierarchy or immutable evidence. Read `AGENTS.md`
before editing. Then read the latest entries in `AGENTS_SYNC.md`, the active
leaf named below, and only the directly relevant runtime files/tests.

## 1. Owner objective and engineering rule

The owner wants Jenny 2.0 to become one coherent, continually improving local
agent around a frozen foundation model—not an LLM with disconnected sidecars.
Cognee memory, Moving Origin, learned procedure/capability state, grounded
appraisal, autonomy, and model cognition should operate as one transactionally
consistent system.

The central constraint is important: deterministic code is allowed for
identity, provenance, schemas, bounds, permissions, transactions, crash
recovery, orchestration, and measurement. It must not encode task answers,
topic-specific behavior, synthetic emotions, or hidden decision policy. New
behavior should arise from model-authored choices conditioned on observed
evidence and persistent learned state.

No present evidence establishes consciousness, feelings, personhood, AGI, or
broad self-improvement. Do not make those claims. The Human-Flourishing
Constitution remains supreme and external effects remain bounded by explicit
permission.

## 2. Exact live snapshot

- Host: `angler-workstation`
- Repository: `/opt/angler/src/angler`
- Branch: `main`
- HEAD: `8e3b51e` (`Preserve scaled procedure publication evaluation`)
- Recent commits:
  - `8e3b51e Preserve scaled procedure publication evaluation`
  - `da2e321 Train scaled Angler core on software procedures`
  - `3ed3a95 Scale Angler procedural core for Qwen`
  - `67c604c Integrate situated Angler runtime with Qwen`
  - `4543e0e Validate learned situated memory integration`
- The working tree intentionally contains a large body of valuable tracked and
  untracked construction beyond HEAD. Do **not** run `git clean`, destructive
  reset, broad checkout/restore, or assume that untracked means disposable.
- Jenny API service: `jenny2-api.service`, active and enabled.
- Owner UI/API: `http://192.168.137.5:8088/ui`
- Raw model API: `http://127.0.0.1:30000/v1`, loopback only. Never expose or
  give the browser direct access to this port.
- Canonical live state:
  - Moving-Origin ordinal: `106`
  - state ref:
    `sha256:c92dfb4fdc54389afe88dad158c0b3231fbb4ebf12c1b688d921fc71a17c73a5`
  - pending operation: false
  - pending projections: `0`
  - projection error: null
  - life error: null
  - scheduler enabled: true
  - life loop running: true
- Cortex: dense text-only Qwen3.8-27B NVFP4, tensor parallel across two GPUs.
- Served base identity: `jenny-qwen3.8-27b`
- Active adapter identity:
  `jenny-qwen3.8-27b--lora-sha256-d1898f1ee287e218ac89ba723992015bb23f7f2309a27392126a84f80431a36d`
- Active binding ref:
  `sha256:af7cfa72907fbcf4a8dd8db0c9510a1d3e395cd0231a5dd3281983b93f1b1bcc`
- Hardware:
  - Intel Core i7-13700K, one NUMA node
  - 64 GiB RAM
  - two RTX 5080 16 GiB GPUs, each capped at 295 W
  - observed resident model use at handoff: about 14.18 GiB per GPU
  - 2 TB NVMe root, about 1.4 TB free at handoff
- Mutating external effects are disabled in the live service. Read-only web
  affordances are bounded separately. The dormant OpenClaw bridge is not part
  of the normal autonomous route.

Revalidate this snapshot before relying on it. State can legitimately advance
after a human turn or a newly actionable life-loop event.

## 3. Current composition and data flow

One foreground or background turn follows this conceptual path:

1. `scripts/jenny2_chat.py` accepts authenticated human ingress or a bounded
   scheduler wake and serializes it through the shared operation lock.
2. `PersistentAutonomySupervisor` reads one canonical SQLite state head,
   validates pending/restart state, and samples Moving Origin trusted time.
3. Canonical memory plus Cognee candidate retrieval supplies bounded episodic
   and procedural context. Cognee returns references; every hit is rejoined to
   exact canonical bytes before model use.
4. The frozen model authors a situated target/actionability judgment. An active
   commitment may remain supported while returning `WAIT_FOR_CHANGE`, which
   creates no operation, episode, learning update, or ordinal advance.
5. If action is justified now, the frozen model authors the bounded operation
   selection. The selected private conclusion can be reused as the public
   response when it is the qualified final authority; the harness must not
   throw away responses and generate redundant replacements.
6. Trusted code validates references, operation membership, permissions,
   idempotency, resource bounds, and transaction identity. It does not choose
   the substantive procedure.
7. The executor yields a receipt. Where an actual evaluator or owner feedback
   exists, consequence remains multidimensional rather than collapsed to one
   arbitrary reward.
8. Reflection/consolidation, learned state, episode, event chain, Moving Origin,
   and projection outbox commit atomically.
9. Cognee projections occur after commit and are disposable/rebuildable from
   canonical history.

Important implementation boundaries:

- API/UI: `scripts/jenny2_chat.py`
- Live assembly: `src/angler/runtime/jenny2_runtime.py`
- Persistent transactions/scheduler: `src/angler/runtime/persistent_autonomy.py`
- Experience, controller, capability, recall/rejoin:
  `src/angler/runtime/higher_level_autonomy_adapter.py`
- Cognee projection/search: `src/angler/runtime/cognee_jenny2.py`
- Procedural case formation: `src/angler/runtime/procedural_experience.py`
- Moving Origin: `src/angler/runtime/temporal_v2.py` and cognition temporal
  modules
- Grounded appraisal: `src/angler/runtime/grounded_appraisal.py`
- Self-observation diary: `src/angler/runtime/self_observation_diary.py`
- Library: `src/angler/runtime/jenny_library.py`
- Latency instrumentation: `src/angler/runtime/latency_trace.py`

## 4. Latest completed correction: active procedural plasticity

Active leaf:

`docs/blueprints/branches/runtime/work/ANG-WORK-RUNTIME-PROCEDURAL-PLASTICITY-CLOSED-LOOP-V1-001.md`

The root defect was at the projection boundary: Cognee had flattened every
canonical record to `EPISODIC/PROPOSED`. Evaluated procedures, failures, and
owner corrections were searchable text, but their procedural and causal graph
meaning was lost.

The completed correction:

- classifies exact evaluated procedural records as `PROCEDURAL`;
- preserves `OBSERVED_PROCEDURAL_OUTCOME` versus
  `OBSERVED_PROCEDURAL_FAILURE`;
- preserves exact feedback-target lineage with a `FEEDBACK_ON` edge;
- rejects malformed/tampered procedural claims;
- continues to classify ordinary/unevaluated records as episodic context;
- reprojects existing evaluated canonical cases at startup so historic valid
  cases receive the same typed treatment;
- adds no model invocation, adapter swap, gradient pass, or resident VRAM.

The end-to-end regression proves:

- an evaluated first case is projected;
- Cognee returns its reference;
- the reference is distrustfully rejoined to immutable canonical bytes;
- the exact case reaches both learned controller and public cortex on a fresh
  related turn; and
- the fixed-input removal arm contains no such procedural evidence.

Validation:

- focused procedural/Cognee/adapter/runtime: 91/91 PASS in 5.141 seconds;
- complete runtime: 480/480 PASS in 53.004 seconds;
- `git diff --check`: PASS;
- live restart: ready, with no pending operation/projection or retained error;
- ordinal/state remained exactly 101 / `6ff2fed4...ba1e6f9`.

Files touched by this leaf:

- `src/angler/runtime/procedural_experience.py`
- `src/angler/runtime/cognee_jenny2.py`
- `src/angler/runtime/higher_level_autonomy_adapter.py`
- `src/angler/runtime/jenny2_runtime.py`
- `tests/unit/runtime/test_procedural_experience.py`
- `tests/unit/runtime/test_cognee_jenny2.py`
- `tests/unit/runtime/test_higher_level_autonomy_adapter.py`
- the active leaf and append-only `AGENTS_SYNC.md`

## 5. Recent latency correction and measured behavior

Do not reintroduce the old serial model cascade.

The bounded structured-controller work removed redundant production
draft/deliberation/self-review behavior. A live two-stage witness used 25,141
prompt tokens instead of 39,362 (36.13% lower), but variable decode length kept
wall time at 81.20 seconds for that complex autonomous witness.

The next correction separated commitment persistence from present
actionability:

- first unchanged-head judgment: 19.652 seconds, one model call, 7,783 prompt
  and 452 generated tokens, no state/ordinal change;
- immediate repeat on the same head: 0.006383 seconds, zero model calls/tokens,
  no state/ordinal change.

Foreground human turns preempt/defer the scheduler. If an owner-visible reply
is slow, inspect the recorded cognitive trace and model request counts before
editing. The failure pattern to prevent is multiple discarded Qwen responses,
serial self-review, or library/retrieval context being injected in full for a
simple question.

No current evidence promises a fixed latency for every human turn. Decode
length, prompt size, operation type, retrieval, and whether genuine
deliberation is selected still matter.

## 6. What “plasticity” honestly means today

Online state that can change without model retraining:

- canonical episodes and event-linked WORLD/SELF/FOCUS hypotheses;
- multidimensional observed consequences and owner corrections;
- content-addressed procedural cases, including failures/counterevidence;
- revision-aware capability modules when an evaluated consequence supports a
  model-authored retention/revision proposal;
- sparse memory/capability selection and contextual utility evidence;
- grounded appraisal moments/covariance derived from observable event signals;
- Moving Origin temporal relations and event chronology.

What remains frozen during ordinary operation:

- Qwen base weights;
- the currently loaded rank-16 LoRA bytes;
- code and permission policy.

Therefore Jenny now has persistent inference-time state plasticity that can
condition later decisions and responses. She does **not** yet continuously
update foundation/LoRA weights, and ordinary unevaluated conversation cannot
honestly manufacture credited competence. Do not describe retrieval alone as
learning, and do not describe this as proof of general self-improvement.

## 7. Grounded appraisal and the next functional frontier

Grounded appraisal is already loaded. It updates exact online moments and
pairwise covariance over observed numeric event signals and presents bounded
innovations/relations to target formation, human-turn routing, experience, and
choice. It contains no hard-coded emotion labels, drive table, valence rule,
or extra model call.

What is not yet established is that this substrate produces a repeatable,
causally attributable improvement in attention/choice across real relevant
interactions. Nor is there evidence that an inferred functional state is a
felt emotion.

Recommended next leaf—keep it narrow:

1. Use the existing appraisal and procedural/correction path; do not bolt on a
   second procedural model or dynamic adapter-swap daemon.
2. Select a bounded sequence with genuinely observed, differing outcomes and
   repeated opportunity for adaptation.
3. Test whether the model uses the learned appraisal/procedural evidence to
   revise attention or method on a fresh related situation.
4. Include fixed-input removal controls for appraisal history and procedural
   memory separately.
5. Preserve raw dimensions and uncertainty; do not assign arbitrary “anger,”
   “love,” “boredom,” or global reward constants.
6. Keep foreground inference to the existing necessary learned stages. Reject
   any design that restores discarded-response cascades.
7. Report functional behavior and evidence only. Emotional labels, if the
   model proposes them, remain revisable hypotheses with alternative
   explanations—not trusted measurements.

The shortest real win is evidence that Jenny learns from an evaluated outcome
or explicit owner correction and applies the change later with no task-coded
solution and no additional model pass. The procedural closed-loop test now
provides the mechanical foundation for that experiment.

## 8. Known limitations and unresolved risks

- Ordinary conversation is generally `COMPLETED_UNEVALUATED`; it supplies
  autobiographical context but no automatic skill credit.
- Only a small amount of live evaluated capability evidence exists. Historic
  synthetic tests show multi-skill transfer/restart/removal, not broad-world
  competence.
- Cognee relevance does not establish truth. Canonical rejoin establishes
  identity/provenance, not factual correctness.
- The complete live-composition manifest described in
  `docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md` is still incomplete/stale as a live
  authority. Do not inject that entire report into every turn.
- `docs/blueprints/branches/runtime/STATUS.md` has a historically stale header;
  use blueprint-owned contracts plus the latest `AGENTS_SYNC.md` entries and
  live inspection rather than trusting its first status block.
- Dynamic adapter libraries/routing were explored but are not the active
  solution. Adapter swapping plus continual backprop would add latency,
  interference risk, and VRAM pressure; do not reactivate without evidence.
- There is no evidence of subjective emotion or consciousness. Functional
  self-observation must retain that uncertainty.
- The repository has substantial uncheckpointed construction. A careless Git
  command is currently a greater data risk than a normal source edit.

## 9. Service, health, and test commands

Run commands from `/opt/angler/src/angler`.

Read-only health:

```bash
systemctl is-active jenny2-api.service
curl -fsS --max-time 10 http://192.168.137.5:8088/health | jq
```

Authenticated status without exposing the credential:

```bash
jenny_api_token=$(< /opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/api-token.txt)
curl -fsS --max-time 10 \
  -H "Authorization: Bearer ${jenny_api_token}" \
  http://192.168.137.5:8088/v1/status | jq
unset jenny_api_token
```

Logs:

```bash
journalctl -u jenny2-api.service -n 100 --no-pager -o short-iso
```

Focused plasticity validation:

```bash
PYTHONPATH=src:scripts /opt/angler/venvs/angler/bin/python -B -m unittest \
  tests.unit.runtime.test_procedural_experience \
  tests.unit.runtime.test_cognee_jenny2 \
  tests.unit.runtime.test_higher_level_autonomy_adapter \
  tests.unit.runtime.test_jenny2_runtime
```

Complete runtime validation:

```bash
PYTHONPATH=src:scripts /opt/angler/venvs/angler/bin/python -B -m unittest \
  discover -s tests/unit/runtime -p 'test_*.py'
```

Static whitespace check:

```bash
git diff --check
```

Restart only after checking `pending_operation: false`, and record state ref
and ordinal before and after:

```bash
sudo -n systemctl restart jenny2-api.service
```

The systemd unit is `/etc/systemd/system/jenny2-api.service`; the active model
and 300-second life-loop configuration is in
`/etc/systemd/system/jenny2-api.service.d/override.conf`.

## 10. Credential-safe access

If Claude Code runs locally on `angler-workstation`, use the repository path
directly and no SSH hop is needed.

If Claude Code runs on the Windows/Echobox coordinator:

```bash
ssh -i <coordinator-held-private-key> -p 22 angler@192.168.137.5
```

The workstation accepts the existing ED25519 public key with fingerprint:

`SHA256:Eb/E5jMVlCiS9Of2Nwk9Ij46fGs7mnFP6l7KqiRQHjc`

Comment: `project-angler-workstation-2026-08-30`.

The matching private key deliberately remains on the coordinator and is not
present in this repository or workstation home. Do not copy a private key,
API token, or bearer credential into source, `AGENTS_SYNC.md`, logs, prompts,
or browser storage. If the coordinator-held key cannot be located, create and
authorize a new dedicated key through the owner rather than extracting or
sharing another credential.

Jenny's API token is local at:

`/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/api-token.txt`

Read it only at execution time as shown above. Do not print it. The separate
OpenClaw bridge credential is unnecessary for normal UI/runtime work.

## 11. State, model, and evidence locations

- Canonical state root:
  `/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1`
- Canonical supervisor database and runtime artifacts live beneath that root;
  inspect before editing and never mutate them with ad hoc SQL.
- Cognee episodic/procedural projection:
  `/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/cognee`
- Cognee capability projection:
  `/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1/cognee-capabilities`
- Active adapter binding:
  `/opt/angler/results/jenny2/skill-adapter-candidate-only-activation-r1/candidate-binding.json`
- System guide (descriptive snapshot, not live authority):
  `docs/reports/JENNY_2_SYSTEM_GUIDE_V1.md`
- Chronological coordination/evidence log: `AGENTS_SYNC.md`

Cognee indexes are rebuildable projections. Canonical state/episodes are not.
Never “fix” a projection by editing canonical history.

## 12. Takeover protocol

Before any code change:

1. Read `AGENTS.md` completely.
2. Check the latest `AGENTS_SYNC.md` entries for another active claim.
3. Verify service health, current ordinal/state ref, pending operation, and
   projection/life errors.
4. Inspect `git status --short --untracked-files=all`; preserve every unrelated
   change.
5. Append a bounded task claim to `AGENTS_SYNC.md`, naming exact files.
6. Diagnose with existing traces/tests before changing architecture.
7. Use `apply_patch` for source/document edits.
8. Run focused tests, then the complete runtime suite for cross-boundary
   changes.
9. Restart only when no operation is pending; verify canonical state identity
   if the change should be projection-only.
10. Append exact result, limitations, and next action to `AGENTS_SYNC.md`.

Do not rerun consumed experimental identities, tune frozen thresholds after
seeing results, substitute conversation claims for durable evidence, or add
governance/process work unless a real integrity boundary requires it.

## 13. Suggested first Claude action

Perform a read-only audit of the latest procedural closed loop and current
latency trace. If it agrees with the recorded evidence, claim a small
grounded-appraisal-to-adaptation evaluation leaf. The goal is not to create an
emotion simulator. The goal is to establish whether observed consequences and
owner corrections alter later model-authored attention/procedure in the right
direction, under fair removal controls, without extra model calls or
task-specific code.

If the audit finds a contradiction, record the exact file/line/state evidence
in `AGENTS_SYNC.md` before modifying anything. Preserve the currently healthy
service and canonical state while resolving it.
