# ANG-WORK-RUNTIME-COGNITIVE-STORE-V3-001

Status: technical PASS — 2026-08-31

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decisions: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`,
`ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES`, and
`ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES`

Predecessor: `ANG-WORK-RUNTIME-PROSPECTIVE-ORIGIN-CONTRACTS-V1-001`

Gate: `ANG-GATE-RUNTIME-COGNITIVE-STORE-V3-001`

Assurance: LOW. This leaf changes one local SQLite aggregate and runs synthetic
CPU tests. It authorizes no cycle activation, executor invocation, model/GPU,
Cognee or Moving-Origin service mutation, network, person data, external
effect, experiment, promotion, or capability claim.

## Accountable outcome

Add one byte-preserving schema-v3 successor lane to the existing cognitive
transaction store. It must atomically persist an immutable prospective batch
and proposed acquisition before claim, advance no acquisition time while
claiming/recording/staging feedback, and append exactly one later resolution
acquisition when a successor turn resolves. An objectively observed turn must
commit its legacy episode/state/outbox, resolution, Episode `002`, and generic
acquisition in the same transaction.

This leaf makes the canonical successor records durable. It does not activate
the cycle, projection backend, Moving Origin, learned prospective component, or
frozen model, and it establishes no learned predictive quality.

## Exact outputs and write scope

Fresh files:

- `tests/unit/runtime/test_cognitive_transaction_store_v3.py`.

Narrow edits:

- `src/angler/runtime/cognitive_transaction_store.py`, only additive schema-v3
  migration, successor lifecycle persistence, generic acquisition/outbox APIs,
  legacy-episode acquisition mirroring, and integrity validation;
- `src/angler/runtime/__init__.py`, only exports required by the new APIs;
- `tests/unit/runtime/test_cognitive_transaction_store.py`, only its existing
  migration fixture/expected terminal version so it constructs literal v1
  rather than leaving new v3 tables behind and verifies v1-to-v3 preservation;
- this leaf and append-only `AGENTS_SYNC.md`.

Read-only and out of scope:

- every cognition/memory contract and existing `001@0.1.0` canonical byte;
- `src/angler/runtime/cognitive_cycle.py` and executor behavior;
- cognitive graph, Cognee adapters, Moving Origin, recall policy, and retry
  orchestration;
- prospective dynamics/credit code, model/Qwen, checkpoints, experiments,
  results, thresholds, and consumed identities.

Stop and split if an existing table must be rebuilt, an existing canonical row
must be rewritten, a missing v2 prospective batch would need to be invented,
or passing requires cycle/backend/model behavior.

## Frozen schema-v3 addition

Keep the v1 and v2 DDL literal and preserve the current v2 fingerprint as the
only accepted v2 migration source. Add exactly four tables.

`cognitive_acquisitions` stores ordinal, acquisition ref, predecessor ref,
closed source contract/ref, record ref, payload digest, and exact canonical
acquisition bytes. Ordinal is primary, acquisition/source/record/payload are
unique, zero has no predecessor, nonzero has one, and locked code/full audit
prove the predecessor is exactly ordinal minus one.

`acquisition_clock` has singleton `1`, `next_ordinal`, last acquisition ref,
and last record ref. Zero has null refs; positive size has both. It is the
canonical store-side acquisition head, not a second temporal model.

`acquisition_projection_outbox` stores ordinal, acquisition ref, record ref,
projection ref, payload digest, exact canonical projection-`002` bytes, and a
boolean acknowledgement. It is disposable delivery state; acknowledgement
never changes canonical acquisition bytes or time.

`prospective_turns_v3` stores one exact `ProspectiveTurnReservationV2` aggregate
with outer and legacy refs, batch/lineage/selection and parent bindings,
pending learner bytes, exact request/receipt/feedback lifecycle bytes, exact
resolution and Episode `002` bytes when applicable, the pre-claim batch
acquisition ref, and the later resolution acquisition ref. Its closed status is
`RESERVED`, `CLAIMED`, `EXECUTION_RECORDED`, or `RESOLVED`; resolution
dispositions are `OBSERVED`, `COMPLETED_UNEVALUATED`, `CANCELLED`,
`CLARIFICATION_REQUIRED`, or `ERROR`. SQL checks mirror the contract truth
boundary, and a partial index permits only one active successor row.

The existing v2 `turn_reservations` table is not rebuilt or semantically
extended. Under `BEGIN IMMEDIATE`, every new reservation path checks that
neither the v2 lane nor the successor lane already has an active turn.
The embedded legacy `reservation_ref` is also a permanent cross-lane effect
identity: its presence in either table, including resolved history, forbids
insertion or replay through the other table. Full audit rejects any
cross-table identity intersection. A consumed executor idempotency key can
never be reintroduced through the successor/legacy compatibility boundary.

The schema-v3 addition is exactly the following SQL. `_SCHEMA_VERSION` becomes
`3`, the current v2 fingerprint is retained as `_V2_SCHEMA_FINGERPRINT =
e8f33be7d9f8dd03d5e1b6242b71a81ca9599507159e6b97029635660d54b089`,
and the fingerprint of `_SCHEMA_V1 + _SCHEMA_V2_ADDITION +
_SCHEMA_V3_ADDITION` is frozen as `_SCHEMA_FINGERPRINT =
7ee40c612af03f4619ba9a017c654ceefb45e513229fe811a3965b42b6501f5f`.

```sql
CREATE TABLE IF NOT EXISTS cognitive_acquisitions (
    ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
    acquisition_ref TEXT NOT NULL UNIQUE,
    predecessor_acquisition_ref TEXT UNIQUE,
    source_contract TEXT NOT NULL CHECK (source_contract IN (
        'ANG-CTR-COGNITIVE-EPISODE-001@0.1.0',
        'ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0',
        'ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0'
    )),
    source_ref TEXT NOT NULL UNIQUE,
    record_ref TEXT NOT NULL UNIQUE,
    payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_payload BLOB NOT NULL UNIQUE,
    UNIQUE(ordinal, acquisition_ref, record_ref),
    UNIQUE(acquisition_ref, record_ref),
    FOREIGN KEY(predecessor_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    CHECK (
        (ordinal = 0 AND predecessor_acquisition_ref IS NULL)
        OR (ordinal > 0 AND predecessor_acquisition_ref IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS acquisition_clock (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_ordinal INTEGER NOT NULL CHECK (next_ordinal >= 0),
    acquisition_ref TEXT,
    record_ref TEXT,
    FOREIGN KEY(acquisition_ref, record_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref, record_ref),
    CHECK (
        (next_ordinal = 0 AND acquisition_ref IS NULL AND record_ref IS NULL)
        OR (next_ordinal > 0 AND acquisition_ref IS NOT NULL AND record_ref IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS acquisition_projection_outbox (
    ordinal INTEGER PRIMARY KEY,
    acquisition_ref TEXT NOT NULL UNIQUE,
    record_ref TEXT NOT NULL UNIQUE,
    projection_ref TEXT NOT NULL UNIQUE,
    payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_payload BLOB NOT NULL UNIQUE,
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    FOREIGN KEY(ordinal, acquisition_ref, record_ref)
        REFERENCES cognitive_acquisitions(ordinal, acquisition_ref, record_ref)
);
CREATE TABLE IF NOT EXISTS prospective_turns_v3 (
    reservation_ref TEXT PRIMARY KEY,
    legacy_reservation_ref TEXT NOT NULL UNIQUE,
    batch_ref TEXT NOT NULL UNIQUE,
    parent_lineage_ref TEXT NOT NULL,
    selected_branch_ref TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('RESERVED', 'CLAIMED', 'EXECUTION_RECORDED', 'RESOLVED')
    ),
    parent_sequence INTEGER NOT NULL CHECK (parent_sequence >= 0),
    parent_event_ref TEXT,
    parent_state_digest TEXT NOT NULL,
    parent_snapshot_sha256 TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    encoder_ref TEXT NOT NULL,
    reservation_payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_reservation BLOB NOT NULL UNIQUE,
    canonical_batch BLOB NOT NULL UNIQUE,
    pending_blob_sha256 TEXT NOT NULL,
    pending_blob_size INTEGER NOT NULL CHECK (
        pending_blob_size >= 0 AND pending_blob_size <= 16777216
    ),
    pending_blob BLOB,
    execution_request_ref TEXT UNIQUE,
    canonical_execution_request BLOB UNIQUE,
    execution_receipt_ref TEXT UNIQUE,
    canonical_execution_receipt BLOB UNIQUE,
    feedback_ref TEXT UNIQUE,
    canonical_feedback BLOB UNIQUE,
    resolution_ref TEXT UNIQUE,
    resolution_disposition TEXT CHECK (
        resolution_disposition IS NULL OR resolution_disposition IN (
            'OBSERVED', 'COMPLETED_UNEVALUATED', 'CANCELLED',
            'CLARIFICATION_REQUIRED', 'ERROR'
        )
    ),
    canonical_resolution BLOB UNIQUE,
    legacy_episode_ref TEXT UNIQUE,
    episode_v2_ref TEXT UNIQUE,
    canonical_episode_v2 BLOB UNIQUE,
    batch_acquisition_ref TEXT NOT NULL UNIQUE,
    resolution_acquisition_ref TEXT UNIQUE,
    FOREIGN KEY(parent_sequence) REFERENCES state_history(sequence),
    FOREIGN KEY(model_ref, encoder_ref)
        REFERENCES store_identity(model_ref, encoder_ref),
    FOREIGN KEY(legacy_episode_ref) REFERENCES episodes(episode_ref),
    FOREIGN KEY(batch_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    FOREIGN KEY(resolution_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    CHECK ((execution_request_ref IS NULL) = (canonical_execution_request IS NULL)),
    CHECK ((execution_receipt_ref IS NULL) = (canonical_execution_receipt IS NULL)),
    CHECK ((feedback_ref IS NULL) = (canonical_feedback IS NULL)),
    CHECK ((resolution_ref IS NULL) = (canonical_resolution IS NULL)),
    CHECK ((episode_v2_ref IS NULL) = (canonical_episode_v2 IS NULL)),
    CHECK (
        (status = 'RESERVED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NULL
            AND execution_receipt_ref IS NULL
            AND feedback_ref IS NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'CLAIMED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND execution_receipt_ref IS NULL
            AND feedback_ref IS NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'EXECUTION_RECORDED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND execution_receipt_ref IS NOT NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'RESOLVED'
            AND pending_blob IS NULL
            AND resolution_ref IS NOT NULL
            AND resolution_disposition IS NOT NULL
            AND resolution_acquisition_ref IS NOT NULL
            AND (
                (resolution_disposition = 'OBSERVED'
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NOT NULL
                    AND legacy_episode_ref IS NOT NULL
                    AND episode_v2_ref IS NOT NULL)
                OR
                (resolution_disposition = 'COMPLETED_UNEVALUATED'
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
                OR
                (resolution_disposition IN ('CLARIFICATION_REQUIRED', 'ERROR')
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
                OR
                (resolution_disposition = 'CANCELLED'
                    AND execution_request_ref IS NULL
                    AND execution_receipt_ref IS NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
            ))
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_prospective_turn_v3
    ON prospective_turns_v3 ((1)) WHERE status != 'RESOLVED';
CREATE INDEX IF NOT EXISTS prospective_turn_v3_legacy_ref
    ON prospective_turns_v3 (legacy_reservation_ref);
```

## Migration and compatibility

- Schema 0 creates the complete v3 schema. Schema 1 uses the existing exact
  v1-to-v2 transaction with literal target version `2` and the frozen v2
  fingerprint, then v2-to-v3 in a separate transaction. Changing the global
  version constant must not make the v1 migration skip or target version `3`.
  Schema 2 must match the frozen known v2 fingerprint before any v3 DDL.
  Schema 3 must match the final v3 fingerprint.
- Any v2 row whose status is not `RESOLVED` makes migration fail before DDL or
  version change. Batch, lineage, and proposed acquisition bytes do not exist
  and must never be guessed.
- In one v2-to-v3 transaction, derive one exact historical acquisition per
  legacy episode in sequence order. Ordinal is `sequence - 1`; the existing
  `episode_record` bytes/ref remain identical; predecessor is exact; projection
  `002` is derived; the legacy projection acknowledgement bit is copied. All
  v1/v2 tables and values remain byte-for-byte unchanged.
- Resolved v2 reservations remain preserved history and generate no fabricated
  prospective records. Migration interruption rolls back all v3 DDL/backfill.
- After v3, a legacy episode commit mirrors that exact episode into the generic
  acquisition clock at its then-current ordinal. The existing sequence-based
  `episode_record` remains the exact source for the legacy projection outbox;
  the generic mirror is
  `dataclasses.replace(episode_record(CognitiveEpisodeItem(sequence,
  episode_ref, episode)), acquired_ordinal=acquisition_clock.next_ordinal)`
  before acquisition and projection `002` are derived. Thus an interleaved
  successor acquisition cannot collide with competence sequence. During
  historical migration the ordinal is still `sequence - 1`, so record refs
  remain identical. The legacy and generic rows/outboxes commit once in the
  same transaction; no second generic episode acquisition is permitted. A
  legacy reservation still has no prospective-batch acquisition because no
  such canonical batch exists; the successor product cycle must use the v3
  lane.

The exact v2-to-v3 transaction order is: `BEGIN IMMEDIATE`; re-read schema
version and known v2 fingerprint; reject every active v2 reservation; execute
the immutable backup protocol below; execute the literal v3 DDL; if and only if
the v2 store has a verified canonical head, insert the acquisition clock row
`(1, 0, NULL, NULL)` and backfill episodes in sequence order while updating
that row after each append; verify the complete acquisition chain/outbox; set
user version `3`; verify the frozen v3 fingerprint; commit. A headless v1/v2
store has no identity, canonical head, episode, or acquisition-clock row after
migration, exactly like a fresh headless v3 schema. `initialize()` creates the
same zero clock row atomically with the competence head. An initialized v2
store always receives exactly one clock row during migration.

Every schema-v2-to-v3 migration creates or verifies an immutable rollback copy
before the first v3 DDL statement. For store path `P`, its exact paths are
`P.pre-v3.sqlite` and `P.pre-v3.sqlite.sha256`; the sidecar is lowercase
64-hex SHA-256 plus one newline. While the migration connection holds its
`BEGIN IMMEDIATE` lock and still has the exact v2 fingerprint, a separate
read-only SQLite source connection uses SQLite's online-backup API to a fresh
temporary file in `P.parent`. Before installation, the temporary copy must
have `PRAGMA user_version == 2`, the exact v2 fingerprint, and
`tuple(connection.iterdump())` equality with the locked version-2 source, be
flushed with `fsync`, and have no group/other/owner write bit. Its digest
sidecar is likewise flushed and read-only. Both final names are installed
without overwriting an existing path, and the directory is synced, all before
v3 DDL.

A collision is accepted only as an exact idempotent retry: both final files
exist, neither has any write bit, the sidecar is canonical, its digest matches
the backup bytes, a read-only open has `PRAGMA user_version == 2` and the exact
v2 fingerprint, and its full `iterdump()` equals the still-locked current
version-2 source. Any missing half, writable file, malformed/mismatched digest,
wrong user version, unknown schema, or logical drift fails before DDL. Schema
zero creates no backup. Schema one first completes the literal v1-to-v2
transaction, then makes this v2 rollback copy before the separate v2-to-v3
transaction.

Fault injection names and positions are frozen as follows:

- `after_v3_backup`: after the verified immutable copy and before v3 DDL;
- `after_v3_schema`: after all four tables/indexes and before any clock row;
- `after_v3_clock_initialization`: after conditionally inserting the zero
  clock and before historical backfill (also reached headless with no row);
- `during_v3_backfill`: after each acquisition/outbox/clock append;
- `before_v3_version`: after full-chain verification and before user-version
  mutation;
- `before_v3_migration_commit`: after the final fingerprint verification and
  immediately before commit;
- `before_prospective_reservation_commit`, `before_prospective_claim_commit`,
  `before_prospective_execution_commit`, `before_prospective_feedback_commit`,
  `before_prospective_resolution_commit`, and
  `before_observed_prospective_commit`: immediately before the named successor
  transaction commits.

Every injected migration failure must leave an exact known-v2 logical schema
and content, including no v3 table, clock, or row; retry must produce the same
v3 bytes. Existing `before_commit` and `before_reserved_commit` remain the
precommit boundaries for legacy episode and legacy-reservation paths.

## Atomic successor APIs

Every canonical write input must have its exact registered concrete class;
subclasses, proxy equality, and overridden canonicalizers are rejected before
any transaction mutation. The store reparses canonical bytes into the exact
registered base class and compares both value and content reference.

`reserve_prospective_turn(reservation_v2, batch_acquisition, pending_blob)`
revalidates both heads, exact wrapper/batch/parent/store identity, pending
bytes, and acquisition source/ordinal/predecessor. It inserts the active
aggregate plus batch acquisition/projection and advances the acquisition clock
in one transaction before any claim.

`claim_prospective_turn(request)`, `record_prospective_execution(receipt)`, and
`stage_prospective_feedback(feedback)` rejoin through the exact legacy
reservation ref embedded in the wrapper. They are idempotent only for exact
bytes and never advance state or acquisition time.

`resolve_prospective_turn(resolution, resolution_acquisition)` accepts only
non-observed dispositions. It validates exact stored lifecycle context,
appends one resolution acquisition/projection, clears pending state, marks the
aggregate resolved, and advances only acquisition time. Cancellation has no
request/receipt; clarification/error require their exact receipt;
completed-unevaluated requires a completed receipt and no feedback/update.

`commit_observed_prospective_turn(resolution, episode_v2,
resolution_acquisition, child_snapshot, expected_parent_digest)` requires
staged exact objective feedback and `OBSERVED`. It atomically appends the
legacy episode/state/current legacy outbox, stores resolution/Episode `002`,
appends the generic resolution acquisition/projection, clears pending bytes,
resolves the aggregate, and advances both heads exactly once. It suppresses an
extra episode-source generic acquisition; the resolution acquisition is the
single post-action clock event. Before any write it additionally requires
`sha256(child_snapshot) == resolution.child_lineage.snapshot_digest`, as well
as the contract's parent/child competence and episode-lineage joins; canonical
child snapshot bytes may never be accepted under a different situated digest.

Generic reads expose verified acquisition head, exact item lookup by record
ref, bounded ordered paging, bounded oldest-first pending projection paging,
and idempotent acknowledgement. Every limit is validated `1..256` and always
reaches SQL as `LIMIT ?`. Every multi-SELECT read/rejoin and full audit uses
one explicit SQLite read transaction so a concurrent valid commit cannot
produce a mixed-snapshot false corruption result.

The public signatures are frozen to:

```python
acquisition_head() -> AcquisitionHead
get_acquisition_item(record_ref: str) -> CognitiveAcquisitionItem
acquisition_items(
    after_ordinal: int = -1,
    limit: int = 64,
) -> tuple[CognitiveAcquisitionItem, ...]
pending_acquisition_projections(
    limit: int = 64,
) -> tuple[AcquisitionProjectionOutboxItem, ...]
ack_acquisition_projection(projection_ref: str) -> None
```

`after_ordinal` is an integer at least `-1`, is exclusive, and therefore `-1`
includes ordinal zero. `AcquisitionHead` contains `next_ordinal`, the nullable
last `acquisition_ref`, and nullable last `record_ref`; each item contains its
stored ordinal and exact parsed canonical contract. The acknowledgement key is
the exact `CognitiveGraphProjectionV2.projection_ref`, not a source or record
alias. Same-key replay is a no-op; an unknown key fails.

## Acceptance

1. New, v1, and known-v2 stores reach the exact v3 fingerprint; unknown or
   altered schemas fail closed.
2. Migration with 0/1/many episodes preserves every old row/byte, produces
   ordinal `0..N-1`, exact predecessor and existing record refs, copies ack
   bits, and is all-or-nothing under injected failure. Headless stores remain
   clock-row-free. Backup creation, exact collision replay, tampered/partial/
   writable/wrong-user-version collision refusal, and offline restore to user
   version `2`, exact v2 fingerprint, and exact logical dump all pass.
3. Every active v2 lifecycle status blocks migration before mutation; resolved
   v2 history migrates without invented batch/resolution records.
4. Successor reserve is one atomic pre-claim boundary. Claim, receipt,
   feedback, and acknowledgement advance neither state nor acquisition clock.
5. All five resolution dispositions restart/replay byte-exactly. Only
   `OBSERVED` advances competence state or produces Episode `002`; all produce
   exactly one resolution acquisition.
6. Observed commit rolls back episode, state, both outboxes, resolution,
   Episode `002`, acquisition, clock, and active-row transition together on
   every injected precommit failure, then exact retry succeeds.
7. Two connections cannot create simultaneous v2/v3 active turns. Same
   identity/different bytes and lineage/source/ordinal/predecessor drift fail
   without mutation. A legacy reservation ref present in either lane, even
   resolved, is permanently rejected by the other lane in both directions.
8. Generic item/outbox paging is ordered and bounded; limit 64 is forwarded;
   corruption of acquisition, record, source, predecessor, projection, or
   clock fails local rejoin/full audit as applicable.
9. Existing transaction-store and contract suites remain green. The bounded
   legacy projection retry defect remains fixed and unchanged.
10. No cycle, backend, Moving-Origin, model, GPU, experiment, authorization,
    external effect, scientific claim, or Human-Flourishing gate state changes.

Passing establishes durable local successor transactions only. It does not
establish cycle use, live projection, trained prospective dynamics, temporal
agency, improved reasoning, selfhood, autonomy, readiness, or AGI.

## Implementation evidence

The additive store implementation preserves schema v2 fingerprint
`e8f33be7d9f8dd03d5e1b6242b71a81ca9599507159e6b97029635660d54b089`
and produces the frozen schema v3 fingerprint
`7ee40c612af03f4619ba9a017c654ceefb45e513229fe811a3965b42b6501f5f`.
The CUDA-hidden fresh v3 suite passes 26/26 and the combined legacy/v3 store
suite passes 55/55 in 13.229 seconds. The directly dependent cognition and
acquisition suites pass 54/54. Compile, schema-fingerprint, scoped whitespace,
and `git diff --check` validation pass.

Independent final review returned PASS after separately exercising the three
last formal witnesses: exact clock-free headless offline restore, a concurrent
legacy/successor reservation race with exactly one winner, and zero-mutation
lineage/source/ordinal/predecessor drift rejection. The store implementation
SHA-256 is
`09beb162fe1ab722993340f5ac475982b69bfb0cb8b037fb0e705477da8a076f`;
the focused v3 test SHA-256 is
`447a956207c82f91877ccccc1d36658b6d6fa5404ad431381b30a6ca7db292b8`.
No model, GPU, live service, network, external effect, experiment, threshold,
or scientific claim changed.

## Proportionate human-impact assessment and rollback

Impact class `LOW`, disposition `ALLOW` for this exact local synthetic SQLite
scope, mapped to `ANG-GATE-HUMAN-FLOURISHING-001`. Principal risks are
fabricating history during migration, upgrading receipts into outcome truth,
partial effect/state commits, unbounded retry pages, and deterministic solution
logic. Known-fingerprint migration, pre-DDL active-turn refusal, exact
objective-feedback binding, atomic transactions, fixed limits, generic
contract validation, and no task solver mitigate them. This does not pass or
waive the gate for cycle activation or promotion; any scope expansion requires
a successor assessment.

Every uncommitted migration failure uses the SQLite transaction rollback and
leaves exact v2 logical content; the already-created immutable copy remains for
exact retry. Rollback after a committed v3 migration is offline copy restore,
never reverse mutation: close all store connections, verify the backup's
read-only mode, canonical sidecar hash, exact v2 fingerprint and integrity;
verify target `P` is the exact known version-3 schema and passes SQLite
integrity (or is already exact dump-equal version 2, making restore a no-op);
copy it to a fresh same-directory temporary file, flush it, atomically replace
only `P`, sync the directory, and reopen to verify exact v2 fingerprint and
`PRAGMA user_version == 2` plus `iterdump()` equality with the backup. Focused
tests perform this procedure on disposable synthetic stores for headless and
populated migration. A newly created schema-v3 store has no v2 history and may
instead be removed and recreated. Existing checkpoints, Cognee data,
Moving-Origin history, and evidence remain untouched.

## Next

After this gate passes, expand a separate cycle/projection leaf that makes the
composite prospective core produce these exact records, rebuilds generic
Cognee/Moving-Origin views from the acquisition outbox, and proves restart and
fair removal controls before frozen-model evaluation.
