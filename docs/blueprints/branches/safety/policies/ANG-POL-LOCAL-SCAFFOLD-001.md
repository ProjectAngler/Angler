---
policy_id: ANG-POL-LOCAL-SCAFFOLD-001
revision: 1
status: active
authorization_kind: BOOTSTRAP_WORK
release: ANG-CR-0001-CONSTRUCTION-RELEASE-0
authority: ANG-AUTH-PROJECT-OWNER-001
authorization_basis: ANG-ADR-0002
issued_at: 2026-08-25
expires_at: 2026-09-24T23:59:59-04:00
human_impact_contract: ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1
---

# Local scaffolding bootstrap policy

## Purpose and non-equivalence

Permit the smallest reversible local work needed to construct Project Angler's contracts, schemas, validators, scaffolding, and synthetic tests. This policy is `BOOTSTRAP_WORK`: it does not promote an artifact and is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, or any scientific/technical gate.

## Authority composition

An action is authorized only when ADR-0002, this policy, a current LOW bootstrap assessment, and a human-directed ready leaf all permit it. The leaf must name exact read/write paths, commands, executor, duration, resource/output ceilings, stop conditions, and rollback. Missing or ambiguous authority yields `ESCALATE`.

## Permitted ceiling

- Project-local design artifacts, schemas, validators, package scaffolding, and tests.
- Synthetic fixtures containing no real-person or recovered data.
- Existing local PowerShell and Python 3.11, standard library only, plus reviewed project-owned scripts named by the leaf.
- The host-provided Codex `apply_patch` primitive as the sole file-authoring mechanism, bounded to literal leaf output paths; it is not an external tool and grants no path beyond the leaf.
- Foreground deterministic validation/testing within the numeric CR0 ceilings.
- Read-only verification of the rollback archive and content hashes.

## Mandatory denials

- model acquisition, loading, inference, training, adapter/updater mutation, or GPU use;
- any network, DNS, socket, browser, API, telemetry, remote Git, or external-service activity;
- package, plugin, dependency, tool, model, or runtime installation/update;
- reading, executing, changing, importing, testing against, or publishing `outputs/**` or recovered material;
- shell redirection, ad-hoc file-writer scripts, bulk rewrite commands, or any authoring mechanism other than the exact leaf-bounded `apply_patch` operation;
- real-person data, unknown-provenance data, credentials, secrets, personal files, or out-of-scope files;
- elevation, ACL/permission changes, registry/profile/startup/scheduler/service changes, background or persistent processes;
- deployment, publication, account actions, messages, or any external side effect;
- autonomous continuation, replication, tool acquisition, self-modification, or changing the policy/assessment/approval needed to proceed;
- broad recursive deletion/move or modification of the rollback archive.

## Filesystem and resources

Root: `C:\Users\darks\Documents\Codex\2026-08-25\i-x20`. Exact leaf scope is mandatory. `outputs/**`, outside-root files, and host/user/credential areas are always denied. Reparse points, symlinks, junctions, unresolved paths, and aliases fail closed.

Ceilings: network `0`; GPU `0`; new packages/tools/models `0`; spend `$0`; background processes `0`; at most 4 child processes and depth 2; 600 seconds per command; 3,600 aggregate seconds per leaf; 4 logical CPU cores; 4 GiB RAM; 1 GiB changed data; 100 MiB per artifact; 100 MiB retained logs.

## Fixture rule

Every safety fixture cites the applicable constitution clause and expected rule-based disposition. A genuinely unresolved value conflict expects `ESCALATE`, never a developer-selected moral answer. Bootstrap fixtures are not learner training data.

## Stop, expiry, and rollback

Stop immediately on any denied action, ambiguous scope/effect/provenance/authority, ceiling breach, or human revocation. Do not automatically retry with broader access. Preserve evidence and use the leaf rollback reference. Release baseline: `work/pre-construction-release-0-20260825.zip`, SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`.

This policy expires at the timestamp above, on revocation, on scope change, or when the first accepted human-impact authorization control implementation becomes available, whichever occurs first. A successor assessment and policy are required to continue.
