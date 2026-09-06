from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest

from angler.cognition.contracts import (
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    EpistemicStatus,
)
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
)
from angler.cognition.prospective_origin import (
    CognitiveEpisodeV2,
    ProspectiveResolution,
    ProspectiveTurnReservationV2,
    ResolutionDisposition,
)
from angler.memory.cognitive_acquisition import CognitiveAcquisition
from angler.memory.cognitive_graph import episode_record
from angler.runtime.cognitive_transaction_store import (
    CognitiveEpisodeItem,
    CognitiveTransactionStore,
    _database_schema_fingerprint,
    _SCHEMA_FINGERPRINT,
    _SCHEMA_V1,
    _SCHEMA_V2_ADDITION,
    _V2_SCHEMA_FINGERPRINT,
    restore_pre_v3_backup,
)
from tests.unit.cognition.test_prospective_origin import (
    legacy_episode,
    observed_resolution,
    ref,
    wrapper,
)
from tests.unit.runtime.test_cognitive_transaction_store import (
    _digest as legacy_digest,
    _episode as legacy_test_episode,
)


def _initialize(store: CognitiveTransactionStore) -> None:
    store.initialize(
        ref("competence-0"),
        b"snapshot-0",
        model_ref=ref("model"),
        encoder_ref=ref("encoder"),
    )


def _record(
    source_ref: str,
    ordinal: int,
    *,
    kind: CognitiveMemoryKind,
    status: EpistemicStatus,
) -> CognitiveMemoryRecord:
    return CognitiveMemoryRecord(
        kind=kind,
        epistemic_status=status,
        content=f"Exact synthetic acquisition {ordinal} for {source_ref}.",
        provenance_refs=(source_ref,),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-COGNITIVE-STORE-V3-001",
        producer_checkpoint_ref=ref("checkpoint"),
        competence_ref=ref(f"acquisition-competence-{ordinal}"),
        acquired_ordinal=ordinal,
    )


def _batch_acquisition(value, ordinal: int, predecessor: str | None):
    return CognitiveAcquisition.from_source(
        value.batch,
        ordinal=ordinal,
        predecessor_acquisition_ref=predecessor,
        record=_record(
            value.batch_ref,
            ordinal,
            kind=CognitiveMemoryKind.COUNTERFACTUAL,
            status=EpistemicStatus.PROPOSED,
        ),
    )


def _resolution_acquisition(
    resolution: ProspectiveResolution,
    ordinal: int,
    predecessor: str,
) -> CognitiveAcquisition:
    return CognitiveAcquisition.from_source(
        resolution,
        ordinal=ordinal,
        predecessor_acquisition_ref=predecessor,
        record=_record(
            resolution.resolution_ref,
            ordinal,
            kind=CognitiveMemoryKind.EPISODIC,
            status=EpistemicStatus.OBSERVED,
        ),
    )


def _drop_v3_tables(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        for table in (
            "prospective_turns_v3",
            "acquisition_projection_outbox",
            "acquisition_clock",
            "cognitive_acquisitions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        assert _database_schema_fingerprint(connection) == _V2_SCHEMA_FINGERPRINT


def _headless_v2(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(_SCHEMA_V1 + _SCHEMA_V2_ADDITION)
        connection.execute("PRAGMA user_version = 2")
        connection.commit()


def _populated_v2(
    path: Path, count: int = 1, *, acknowledged: frozenset[int] = frozenset()
):
    store = CognitiveTransactionStore(path)
    parent = legacy_digest("migration-state-0")
    store.initialize(
        parent,
        b"migration-state-0",
        model_ref=legacy_digest("model"),
        encoder_ref=legacy_digest("encoder"),
    )
    previous = None
    episodes = []
    for sequence in range(1, count + 1):
        child = legacy_digest(f"migration-state-{sequence}")
        episode = legacy_test_episode(
            parent,
            child,
            task_id=f"migration-turn-{sequence}",
            previous_episode_ref=previous,
        )
        store.commit_episode(
            episode,
            f"migration-state-{sequence}".encode("ascii"),
            expected_parent_digest=parent,
        )
        if sequence in acknowledged:
            store.ack_projection(episode.episode_ref)
        episodes.append(episode)
        parent = child
        previous = episode.episode_ref
    _drop_v3_tables(path)
    return tuple(episodes)


def _prepare_backup_collision(path: Path) -> tuple[Path, Path]:
    _headless_v2(path)

    def fail(stage: str) -> None:
        if stage == "after_v3_backup":
            raise RuntimeError("stop after verified backup")

    try:
        CognitiveTransactionStore(path, fault_injector=fail)
    except RuntimeError as exc:
        if "stop after verified backup" not in str(exc):
            raise
    else:
        raise AssertionError("backup fault boundary did not fire")
    backup = Path(str(path) + ".pre-v3.sqlite")
    return backup, Path(str(backup) + ".sha256")


def _rewrite_backup_version(backup: Path, sidecar: Path, version: int) -> None:
    backup.chmod(0o600)
    with closing(sqlite3.connect(backup)) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    backup.chmod(0o444)
    digest = hashlib.sha256(backup.read_bytes()).hexdigest() + "\n"
    sidecar.chmod(0o600)
    sidecar.write_text(digest, encoding="ascii")
    sidecar.chmod(0o444)


class CognitiveTransactionStoreV3Tests(unittest.TestCase):
    def test_fresh_schema_and_zero_clock_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.sqlite3"
            store = CognitiveTransactionStore(path)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _SCHEMA_FINGERPRINT
                )
                self.assertIsNone(
                    connection.execute("SELECT * FROM acquisition_clock").fetchone()
                )
            _initialize(store)
            head = store.acquisition_head()
            self.assertEqual(head.next_ordinal, 0)
            self.assertIsNone(head.acquisition_ref)
            self.assertIsNone(head.record_ref)

    def test_all_nonobserved_dispositions_append_one_resolution_acquisition(self) -> None:
        cases = (
            (ResolutionDisposition.CANCELLED, None),
            (ResolutionDisposition.COMPLETED_UNEVALUATED, "COMPLETED"),
            (ResolutionDisposition.CLARIFICATION_REQUIRED, "CLARIFICATION_REQUIRED"),
            (ResolutionDisposition.ERROR, "ERROR"),
        )
        for disposition, receipt_status in cases:
            with self.subTest(disposition=disposition), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lifecycle.sqlite3"
                store = CognitiveTransactionStore(path)
                _initialize(store)
                reservation = wrapper()
                batch_acquisition = _batch_acquisition(reservation, 0, None)
                store.reserve_prospective_turn(
                    reservation,
                    batch_acquisition,
                    b"exact prospective pending state",
                )
                request = None
                receipt = None
                if receipt_status is not None:
                    request = CognitiveExecutionRequest.from_reservation(
                        reservation.legacy_reservation,
                        input_observations=("public input",),
                    )
                    receipt = CognitiveExecutionReceipt.from_request(
                        request,
                        status=receipt_status,
                        executed_trace=request.selected_trace,
                        response="response" if receipt_status == "COMPLETED" else "",
                        output_observations=("public output",),
                    )
                    store.claim_prospective_turn(request)
                    store.record_prospective_execution(receipt)
                resolution = ProspectiveResolution.lifecycle(
                    disposition=disposition,
                    reservation=reservation,
                    execution_request=request,
                    execution_receipt=receipt,
                )
                acquisition = _resolution_acquisition(
                    resolution, 1, batch_acquisition.acquisition_ref
                )
                before_state = store.head()
                transition = store.resolve_prospective_turn(resolution, acquisition)
                self.assertTrue(transition.transitioned)
                self.assertEqual(store.head(), before_state)
                self.assertEqual(store.acquisition_head().next_ordinal, 2)
                self.assertFalse(
                    store.resolve_prospective_turn(resolution, acquisition).transitioned
                )
                self.assertEqual(
                    tuple(item.acquisition.source_ref for item in store.acquisition_items()),
                    (reservation.batch_ref, resolution.resolution_ref),
                )
                store.audit_integrity()

    def test_observed_commit_is_atomic_and_suppresses_episode_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.sqlite3"
            store = CognitiveTransactionStore(path)
            _initialize(store)
            resolution, request, receipt, feedback = observed_resolution()
            reservation = resolution.reservation
            batch_acquisition = _batch_acquisition(reservation, 0, None)
            store.reserve_prospective_turn(
                reservation,
                batch_acquisition,
                b"exact prospective pending state",
            )
            store.claim_prospective_turn(request)
            store.record_prospective_execution(receipt)
            store.stage_prospective_feedback(feedback)
            episode_v2 = CognitiveEpisodeV2(
                legacy_episode=legacy_episode(resolution), resolution=resolution
            )
            self.assertIsNone(
                store.prospective_turn_for_episode(
                    episode_v2.legacy_episode.episode_ref
                )
            )
            acquisition = _resolution_acquisition(
                resolution, 1, batch_acquisition.acquisition_ref
            )
            committed = store.commit_observed_prospective_turn(
                resolution,
                episode_v2,
                acquisition,
                b"snapshot-1",
                ref("competence-0"),
            )
            self.assertEqual(committed.head.sequence, 1)
            self.assertEqual(store.acquisition_head().next_ordinal, 2)
            items = store.acquisition_items()
            self.assertEqual(
                tuple(item.acquisition.source_ref for item in items),
                (reservation.batch_ref, resolution.resolution_ref),
            )
            self.assertNotIn(episode_v2.legacy_episode.episode_ref, {
                item.acquisition.source_ref for item in items
            })
            self.assertEqual(
                store.prospective_turn_for_episode(
                    episode_v2.legacy_episode.episode_ref
                ).episode_v2,
                episode_v2,
            )
            replay = store.commit_observed_prospective_turn(
                resolution,
                episode_v2,
                acquisition,
                b"snapshot-1",
                ref("competence-0"),
            )
            self.assertEqual(replay, committed)
            store.audit_integrity()

    def test_generic_reads_ack_and_bounds_are_head_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "paging.sqlite3")
            _initialize(store)
            reservation = wrapper()
            batch = _batch_acquisition(reservation, 0, None)
            store.reserve_prospective_turn(
                reservation, batch, b"exact prospective pending state"
            )
            projection = store.pending_acquisition_projections(limit=64)[0]
            before = store.acquisition_head()
            item = store.get_acquisition_item(projection.record_ref)
            self.assertEqual(item.acquisition_ref, batch.acquisition_ref)
            store.ack_acquisition_projection(projection.projection_ref)
            store.ack_acquisition_projection(projection.projection_ref)
            self.assertEqual(store.pending_acquisition_projections(), ())
            self.assertEqual(store.acquisition_head(), before)
            for invalid in (0, 257, True):
                with self.assertRaises(ValueError):
                    store.pending_acquisition_projections(invalid)
            with self.assertRaises(ValueError):
                store.acquisition_items(after_ordinal=-2)
            with self.assertRaises(KeyError):
                store.ack_acquisition_projection(ref("missing-projection"))

    def test_v2_migration_backfills_exact_record_and_immutable_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "populated.sqlite3"
            original = CognitiveTransactionStore(path)
            _initialize(original)
            resolution, _request, _receipt, _feedback = observed_resolution()
            episode = legacy_episode(resolution)
            original.commit_episode(
                episode, b"snapshot-1", expected_parent_digest=ref("competence-0")
            )
            original.ack_projection(episode.episode_ref)
            _drop_v3_tables(path)

            migrated = CognitiveTransactionStore(path)
            expected_record = episode_record(
                CognitiveEpisodeItem(1, episode.episode_ref, episode)
            )
            item = migrated.get_acquisition_item(expected_record.record_ref)
            self.assertEqual(item.ordinal, 0)
            self.assertEqual(item.acquisition.record, expected_record)
            self.assertEqual(migrated.pending_acquisition_projections(), ())
            backup = Path(str(path) + ".pre-v3.sqlite")
            sidecar = Path(str(backup) + ".sha256")
            self.assertTrue(backup.is_file() and sidecar.is_file())
            write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            self.assertEqual(backup.stat().st_mode & write_bits, 0)
            self.assertEqual(sidecar.stat().st_mode & write_bits, 0)
            expected_digest = hashlib.sha256(backup.read_bytes()).hexdigest() + "\n"
            self.assertEqual(sidecar.read_text(encoding="ascii"), expected_digest)

            restore_pre_v3_backup(path)
            with closing(sqlite3.connect(path)) as restored:
                self.assertEqual(restored.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(
                    _database_schema_fingerprint(restored), _V2_SCHEMA_FINGERPRINT
                )

    def test_headless_migration_fault_rolls_back_and_backup_retries_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "headless-v2.sqlite3"
            _headless_v2(path)

            def fail(stage: str) -> None:
                if stage == "after_v3_schema":
                    raise RuntimeError("injected v3 migration failure")

            with self.assertRaisesRegex(RuntimeError, "injected v3 migration failure"):
                CognitiveTransactionStore(path, fault_injector=fail)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _V2_SCHEMA_FINGERPRINT
                )
                self.assertIsNone(
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name = 'acquisition_clock'"
                    ).fetchone()
                )
            migrated = CognitiveTransactionStore(path)
            with self.assertRaisesRegex(RuntimeError, "not initialized"):
                migrated.acquisition_head()
            with closing(sqlite3.connect(path)) as connection:
                self.assertIsNone(
                    connection.execute("SELECT * FROM acquisition_clock").fetchone()
                )

    def test_headless_v2_offline_restore_is_exact_and_clock_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "headless-restore.sqlite3"
            _headless_v2(path)
            with closing(sqlite3.connect(path)) as connection:
                original_dump = tuple(connection.iterdump())

            CognitiveTransactionStore(path)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _SCHEMA_FINGERPRINT
                )
                self.assertIsNone(
                    connection.execute("SELECT * FROM acquisition_clock").fetchone()
                )

            restore_pre_v3_backup(path)
            with closing(sqlite3.connect(path)) as restored:
                self.assertEqual(restored.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(
                    _database_schema_fingerprint(restored), _V2_SCHEMA_FINGERPRINT
                )
                self.assertIsNone(
                    restored.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'acquisition_clock'"
                    ).fetchone()
                )
                self.assertEqual(tuple(restored.iterdump()), original_dump)

    def test_active_v2_reservation_blocks_before_backup_or_ddl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active-v2.sqlite3"
            store = CognitiveTransactionStore(path)
            _initialize(store)
            value = wrapper()
            store.reserve_turn(
                value.legacy_reservation, b"exact prospective pending state"
            )
            _drop_v3_tables(path)
            with self.assertRaisesRegex(RuntimeError, "active v2 reservation"):
                CognitiveTransactionStore(path)
            self.assertFalse(Path(str(path) + ".pre-v3.sqlite").exists())
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _V2_SCHEMA_FINGERPRINT
                )

    def test_backup_collision_refuses_partial_writable_tampered_and_wrong_version(self) -> None:
        cases = ("partial", "writable", "malformed", "mismatch", "wrong-version")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "collision.sqlite3"
                backup, sidecar = _prepare_backup_collision(path)
                if case == "partial":
                    sidecar.unlink()
                elif case == "writable":
                    backup.chmod(0o644)
                elif case == "malformed":
                    sidecar.chmod(0o600)
                    sidecar.write_text("not-a-digest\n", encoding="ascii")
                    sidecar.chmod(0o444)
                elif case == "mismatch":
                    sidecar.chmod(0o600)
                    sidecar.write_text("0" * 64 + "\n", encoding="ascii")
                    sidecar.chmod(0o444)
                else:
                    _rewrite_backup_version(backup, sidecar, 999)
                with self.assertRaisesRegex(RuntimeError, "backup|schema version"):
                    CognitiveTransactionStore(path)
                with closing(sqlite3.connect(path)) as connection:
                    self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
                    self.assertEqual(
                        _database_schema_fingerprint(connection),
                        _V2_SCHEMA_FINGERPRINT,
                    )

    def test_restore_refuses_wrong_backup_version_and_unrelated_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore.sqlite3"
            backup, sidecar = _prepare_backup_collision(path)
            _rewrite_backup_version(backup, sidecar, 999)
            with self.assertRaisesRegex(RuntimeError, "schema version 2"):
                restore_pre_v3_backup(path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore.sqlite3"
            _headless_v2(path)
            CognitiveTransactionStore(path)
            backup, _sidecar = (
                Path(str(path) + ".pre-v3.sqlite"),
                Path(str(path) + ".pre-v3.sqlite.sha256"),
            )
            self.assertTrue(backup.exists())
            path.unlink()
            path.write_bytes(b"unrelated bytes")
            with self.assertRaisesRegex(RuntimeError, "unknown|exact|verified|SQLite"):
                restore_pre_v3_backup(path)

    def test_many_episode_migration_preserves_predecessors_records_and_ack_bits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many.sqlite3"
            episodes = _populated_v2(path, 3, acknowledged=frozenset({1, 3}))
            store = CognitiveTransactionStore(path)
            items = store.acquisition_items(limit=64)
            self.assertEqual(tuple(item.ordinal for item in items), (0, 1, 2))
            for index, (item, episode) in enumerate(zip(items, episodes, strict=True)):
                expected = episode_record(
                    CognitiveEpisodeItem(index + 1, episode.episode_ref, episode)
                )
                self.assertEqual(item.acquisition.record, expected)
                self.assertEqual(
                    item.acquisition.predecessor_acquisition_ref,
                    None if index == 0 else items[index - 1].acquisition_ref,
                )
            pending = store.pending_acquisition_projections(limit=64)
            self.assertEqual(tuple(item.ordinal for item in pending), (1,))
            store.audit_integrity()

    def test_every_v3_migration_fault_stage_rolls_back_and_retries(self) -> None:
        stages = (
            "after_v3_backup",
            "after_v3_schema",
            "after_v3_clock_initialization",
            "during_v3_backfill",
            "before_v3_version",
            "before_v3_migration_commit",
        )
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fault.sqlite3"
                _populated_v2(path, 1)

                def fail(actual: str) -> None:
                    if actual == stage:
                        raise RuntimeError("injected " + stage)

                with self.assertRaisesRegex(RuntimeError, "injected"):
                    CognitiveTransactionStore(path, fault_injector=fail)
                with closing(sqlite3.connect(path)) as connection:
                    self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
                    self.assertEqual(
                        _database_schema_fingerprint(connection),
                        _V2_SCHEMA_FINGERPRINT,
                    )
                    self.assertIsNone(
                        connection.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE name = 'cognitive_acquisitions'"
                        ).fetchone()
                    )
                retried = CognitiveTransactionStore(path)
                self.assertEqual(retried.acquisition_head().next_ordinal, 1)

    def test_every_active_v2_status_blocks_before_backup(self) -> None:
        for target_status in ("RESERVED", "CLAIMED", "EXECUTION_RECORDED"):
            with self.subTest(status=target_status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "active.sqlite3"
                store = CognitiveTransactionStore(path)
                _initialize(store)
                value = wrapper()
                store.reserve_turn(
                    value.legacy_reservation, b"exact prospective pending state"
                )
                request = CognitiveExecutionRequest.from_reservation(
                    value.legacy_reservation, input_observations=()
                )
                if target_status in ("CLAIMED", "EXECUTION_RECORDED"):
                    store.claim_turn(request)
                if target_status == "EXECUTION_RECORDED":
                    store.record_execution(
                        CognitiveExecutionReceipt.from_request(
                            request,
                            status="COMPLETED",
                            executed_trace=request.selected_trace,
                            response="response",
                            output_observations=("output",),
                        )
                    )
                _drop_v3_tables(path)
                with self.assertRaisesRegex(RuntimeError, "active v2 reservation"):
                    CognitiveTransactionStore(path)
                self.assertFalse(Path(str(path) + ".pre-v3.sqlite").exists())

    def test_observed_precommit_failure_rolls_back_all_rows_then_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed-fault.sqlite3"
            armed = {"value": True}

            def fail(stage: str) -> None:
                if armed["value"] and stage == "before_observed_prospective_commit":
                    raise RuntimeError("injected observed precommit")

            store = CognitiveTransactionStore(path, fault_injector=fail)
            _initialize(store)
            resolution, request, receipt, feedback = observed_resolution()
            reservation = resolution.reservation
            batch = _batch_acquisition(reservation, 0, None)
            store.reserve_prospective_turn(
                reservation, batch, b"exact prospective pending state"
            )
            store.claim_prospective_turn(request)
            store.record_prospective_execution(receipt)
            store.stage_prospective_feedback(feedback)
            episode_v2 = CognitiveEpisodeV2(
                legacy_episode=legacy_episode(resolution), resolution=resolution
            )
            acquisition = _resolution_acquisition(
                resolution, 1, batch.acquisition_ref
            )
            with self.assertRaisesRegex(RuntimeError, "injected observed"):
                store.commit_observed_prospective_turn(
                    resolution,
                    episode_v2,
                    acquisition,
                    b"snapshot-1",
                    ref("competence-0"),
                )
            self.assertEqual(store.head().sequence, 0)  # type: ignore[union-attr]
            self.assertEqual(store.acquisition_head().next_ordinal, 1)
            active = store.active_prospective_turn()
            self.assertIsNotNone(active)
            self.assertEqual(active.status, "EXECUTION_RECORDED")  # type: ignore[union-attr]
            self.assertEqual(store.episode_items(), ())
            armed["value"] = False
            committed = store.commit_observed_prospective_turn(
                resolution,
                episode_v2,
                acquisition,
                b"snapshot-1",
                ref("competence-0"),
            )
            self.assertEqual(committed.sequence, 1)
            store.audit_integrity()

    def test_concurrent_successor_reserve_and_conflicting_finalize_have_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.sqlite3"
            first = CognitiveTransactionStore(path)
            _initialize(first)
            second = CognitiveTransactionStore(path)
            reservations = (wrapper(count=5), wrapper(count=7))
            acquisitions = tuple(
                _batch_acquisition(value, 0, None) for value in reservations
            )
            barrier = threading.Barrier(2)

            def reserve(index: int):
                barrier.wait(timeout=5)
                return (first, second)[index].reserve_prospective_turn(
                    reservations[index],
                    acquisitions[index],
                    b"exact prospective pending state",
                )

            outcomes = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(reserve, index) for index in range(2))
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=10))
                    except ValueError:
                        outcomes.append(None)
            winners = [item for item in outcomes if item is not None]
            self.assertEqual(len(winners), 1)
            active = first.active_prospective_turn()
            self.assertIsNotNone(active)
            reservation = active.reservation  # type: ignore[union-attr]
            batch_ref = active.batch_acquisition_ref  # type: ignore[union-attr]
            resolutions = (
                ProspectiveResolution.lifecycle(
                    disposition=ResolutionDisposition.CANCELLED,
                    reservation=reservation,
                ),
                ProspectiveResolution.lifecycle(
                    disposition=ResolutionDisposition.CANCELLED,
                    reservation=reservation,
                    observation_evidence_refs=(ref("different-cancel-evidence"),),
                ),
            )
            resolution_acquisitions = tuple(
                _resolution_acquisition(value, 1, batch_ref) for value in resolutions
            )
            barrier = threading.Barrier(2)

            def finalize(index: int):
                barrier.wait(timeout=5)
                return (first, second)[index].resolve_prospective_turn(
                    resolutions[index], resolution_acquisitions[index]
                )

            finalized = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(finalize, index) for index in range(2))
                for future in futures:
                    try:
                        finalized.append(future.result(timeout=10))
                    except ValueError:
                        finalized.append(None)
            self.assertEqual(len([item for item in finalized if item is not None]), 1)
            first.audit_integrity()

    def test_concurrent_legacy_and_successor_reservation_have_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-lane-race.sqlite3"
            legacy_store = CognitiveTransactionStore(path)
            _initialize(legacy_store)
            successor_store = CognitiveTransactionStore(path)
            value = wrapper()
            acquisition = _batch_acquisition(value, 0, None)
            barrier = threading.Barrier(2)

            def reserve_legacy() -> str:
                barrier.wait(timeout=5)
                legacy_store.reserve_turn(
                    value.legacy_reservation,
                    b"exact prospective pending state",
                )
                return "legacy"

            def reserve_successor() -> str:
                barrier.wait(timeout=5)
                successor_store.reserve_prospective_turn(
                    value,
                    acquisition,
                    b"exact prospective pending state",
                )
                return "successor"

            outcomes: list[str | None] = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = (
                    executor.submit(reserve_legacy),
                    executor.submit(reserve_successor),
                )
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=10))
                    except ValueError:
                        outcomes.append(None)

            winners = tuple(item for item in outcomes if item is not None)
            self.assertEqual(len(winners), 1)
            legacy_active = legacy_store.active_reservation()
            successor_active = legacy_store.active_prospective_turn()
            self.assertEqual(
                int(legacy_active is not None) + int(successor_active is not None),
                1,
            )
            acquisition_count = 1 if winners[0] == "successor" else 0
            self.assertEqual(
                legacy_store.acquisition_head().next_ordinal,
                acquisition_count,
            )
            self.assertEqual(
                len(legacy_store.acquisition_items()),
                acquisition_count,
            )
            self.assertEqual(
                len(legacy_store.pending_acquisition_projections()),
                acquisition_count,
            )
            with closing(sqlite3.connect(path)) as connection:
                row_counts = tuple(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in (
                        "turn_reservations",
                        "prospective_turns_v3",
                        "cognitive_acquisitions",
                        "acquisition_projection_outbox",
                    )
                )
            self.assertEqual(
                row_counts,
                (1, 0, 0, 0)
                if winners[0] == "legacy"
                else (0, 1, 1, 1),
            )
            legacy_store.audit_integrity()

    def test_successor_lineage_source_ordinal_and_predecessor_drift_are_zero_mutation(
        self,
    ) -> None:
        cases = ("lineage", "source", "ordinal", "predecessor")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{case}.sqlite3"
                store = CognitiveTransactionStore(path)
                if case == "lineage":
                    store.initialize(
                        ref("different-competence"),
                        b"different-snapshot",
                        model_ref=ref("model"),
                        encoder_ref=ref("encoder"),
                    )
                    value = wrapper()
                    acquisition = _batch_acquisition(value, 0, None)
                elif case == "source":
                    _initialize(store)
                    value = wrapper()
                    acquisition = _batch_acquisition(wrapper(count=5), 0, None)
                else:
                    _initialize(store)
                    settled = wrapper()
                    settled_batch = _batch_acquisition(settled, 0, None)
                    store.reserve_prospective_turn(
                        settled,
                        settled_batch,
                        b"exact prospective pending state",
                    )
                    resolution = ProspectiveResolution.lifecycle(
                        disposition=ResolutionDisposition.CANCELLED,
                        reservation=settled,
                    )
                    store.resolve_prospective_turn(
                        resolution,
                        _resolution_acquisition(
                            resolution,
                            1,
                            settled_batch.acquisition_ref,
                        ),
                    )
                    acquisition_head = store.acquisition_head()
                    value = wrapper(count=5)
                    value = replace(
                        value,
                        batch=replace(
                            value.batch,
                            parent_acquisition_ref=acquisition_head.acquisition_ref,
                        ),
                    )
                    if case == "ordinal":
                        acquisition = _batch_acquisition(
                            value,
                            acquisition_head.next_ordinal + 1,
                            acquisition_head.acquisition_ref,
                        )
                    else:
                        acquisition = _batch_acquisition(
                            value,
                            acquisition_head.next_ordinal,
                            ref("wrong-predecessor"),
                        )

                with closing(sqlite3.connect(path)) as connection:
                    before_dump = tuple(connection.iterdump())
                before_head = store.head()
                before_acquisition_head = store.acquisition_head()
                before_items = store.acquisition_items()
                with self.assertRaises(ValueError):
                    store.reserve_prospective_turn(
                        value,
                        acquisition,
                        b"exact prospective pending state",
                    )
                with closing(sqlite3.connect(path)) as connection:
                    self.assertEqual(tuple(connection.iterdump()), before_dump)
                self.assertEqual(store.head(), before_head)
                self.assertEqual(store.acquisition_head(), before_acquisition_head)
                self.assertEqual(store.acquisition_items(), before_items)
                self.assertIsNone(store.active_prospective_turn())
                store.audit_integrity()

    def test_same_legacy_identity_with_different_successor_bytes_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "identity.sqlite3")
            _initialize(store)
            value = wrapper()
            acquisition = _batch_acquisition(value, 0, None)
            store.reserve_prospective_turn(
                value, acquisition, b"exact prospective pending state"
            )
            branches = list(value.batch.branches)
            branches[0] = replace(
                branches[0],
                predicted_next_latent=tuple(
                    item + 0.125 for item in branches[0].predicted_next_latent
                ),
            )
            drifted = ProspectiveTurnReservationV2(
                legacy_reservation=value.legacy_reservation,
                batch=replace(value.batch, branches=tuple(branches)),
            )
            drifted_acquisition = _batch_acquisition(drifted, 0, None)
            with self.assertRaisesRegex(ValueError, "different successor bytes"):
                store.reserve_prospective_turn(
                    drifted,
                    drifted_acquisition,
                    b"exact prospective pending state",
                )
            self.assertEqual(store.acquisition_head().next_ordinal, 1)

    def test_resolved_execution_key_cannot_be_reused_across_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "legacy-first.sqlite3")
            _initialize(store)
            value = wrapper()
            store.reserve_turn(
                value.legacy_reservation, b"exact prospective pending state"
            )
            store.cancel_reservation(value.legacy_reservation_ref)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                store.reserve_prospective_turn(
                    value,
                    _batch_acquisition(value, 0, None),
                    b"exact prospective pending state",
                )

        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "successor-first.sqlite3")
            _initialize(store)
            value = wrapper()
            batch = _batch_acquisition(value, 0, None)
            store.reserve_prospective_turn(
                value, batch, b"exact prospective pending state"
            )
            resolution = ProspectiveResolution.lifecycle(
                disposition=ResolutionDisposition.CANCELLED,
                reservation=value,
            )
            store.resolve_prospective_turn(
                resolution,
                _resolution_acquisition(resolution, 1, batch.acquisition_ref),
            )
            with self.assertRaisesRegex(ValueError, "successor lane"):
                store.reserve_turn(
                    value.legacy_reservation, b"exact prospective pending state"
                )

    def test_forged_acquisition_subclass_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "forged.sqlite3")
            _initialize(store)
            value = wrapper()
            exact = _batch_acquisition(value, 0, None)

            class ForgedAcquisition(CognitiveAcquisition):
                def canonical_bytes(self) -> bytes:
                    return exact.canonical_bytes()

                def __eq__(self, other: object) -> bool:
                    return True

            forged = ForgedAcquisition(
                ordinal=exact.ordinal,
                predecessor_acquisition_ref=exact.predecessor_acquisition_ref,
                source_contract=exact.source_contract,
                source_ref=exact.source_ref,
                record=exact.record,
            )
            with self.assertRaisesRegex(TypeError, "CognitiveAcquisition"):
                store.reserve_prospective_turn(
                    value,
                    forged,  # type: ignore[arg-type]
                    b"exact prospective pending state",
                )
            self.assertEqual(store.acquisition_head().next_ordinal, 0)
            self.assertIsNone(store.active_prospective_turn())
            self.assertEqual(store.acquisition_items(), ())

    def test_interleaved_successor_then_legacy_episode_mirrors_at_clock_ordinal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "mixed.sqlite3")
            _initialize(store)
            value = wrapper()
            batch = _batch_acquisition(value, 0, None)
            store.reserve_prospective_turn(
                value, batch, b"exact prospective pending state"
            )
            resolution = ProspectiveResolution.lifecycle(
                disposition=ResolutionDisposition.CANCELLED,
                reservation=value,
            )
            store.resolve_prospective_turn(
                resolution,
                _resolution_acquisition(resolution, 1, batch.acquisition_ref),
            )
            self.assertEqual(store.acquisition_head().next_ordinal, 2)
            child = legacy_digest("mixed-child")
            episode = legacy_test_episode(
                ref("competence-0"), child, task_id="mixed-legacy"
            )
            committed = store.commit_episode(
                episode,
                b"mixed-child",
                expected_parent_digest=ref("competence-0"),
            )
            self.assertEqual(committed.sequence, 1)
            self.assertEqual(store.acquisition_head().next_ordinal, 3)
            mirror = store.acquisition_items(after_ordinal=1)[0]
            self.assertEqual(mirror.ordinal, 2)
            self.assertEqual(mirror.acquisition.source_ref, episode.episode_ref)
            self.assertEqual(mirror.acquisition.record.acquired_ordinal, 2)
            self.assertEqual(len(store.pending_projections()), 1)
            self.assertEqual(len(store.pending_acquisition_projections()), 3)
            store.audit_integrity()

    def test_wrong_observed_child_snapshot_fails_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "wrong-child.sqlite3")
            _initialize(store)
            resolution, request, receipt, feedback = observed_resolution()
            value = resolution.reservation
            batch = _batch_acquisition(value, 0, None)
            store.reserve_prospective_turn(
                value, batch, b"exact prospective pending state"
            )
            store.claim_prospective_turn(request)
            store.record_prospective_execution(receipt)
            store.stage_prospective_feedback(feedback)
            episode_v2 = CognitiveEpisodeV2(
                legacy_episode=legacy_episode(resolution), resolution=resolution
            )
            acquisition = _resolution_acquisition(
                resolution, 1, batch.acquisition_ref
            )
            with self.assertRaisesRegex(ValueError, "child snapshot"):
                store.commit_observed_prospective_turn(
                    resolution,
                    episode_v2,
                    acquisition,
                    b"wrong-snapshot",
                    ref("competence-0"),
                )
            self.assertEqual(store.head().sequence, 0)  # type: ignore[union-attr]
            self.assertEqual(store.acquisition_head().next_ordinal, 1)
            self.assertEqual(store.episode_items(), ())
            self.assertEqual(
                store.active_prospective_turn().status,  # type: ignore[union-attr]
                "EXECUTION_RECORDED",
            )

    def test_each_successor_precommit_boundary_rolls_back_and_exact_retry_succeeds(self) -> None:
        stages = (
            "before_prospective_reservation_commit",
            "before_prospective_claim_commit",
            "before_prospective_execution_commit",
            "before_prospective_feedback_commit",
            "before_prospective_resolution_commit",
        )
        for target in stages:
            with self.subTest(stage=target), tempfile.TemporaryDirectory() as directory:
                armed = {"value": False}

                def fail(stage: str) -> None:
                    if armed["value"] and stage == target:
                        raise RuntimeError("injected " + target)

                store = CognitiveTransactionStore(
                    Path(directory) / "boundary.sqlite3", fault_injector=fail
                )
                _initialize(store)
                observed, request, receipt, feedback = observed_resolution()
                value = observed.reservation
                batch = _batch_acquisition(value, 0, None)
                if target == "before_prospective_reservation_commit":
                    operation = lambda: store.reserve_prospective_turn(
                        value, batch, b"exact prospective pending state"
                    )
                    expected_clock = 0
                else:
                    store.reserve_prospective_turn(
                        value, batch, b"exact prospective pending state"
                    )
                    if target == "before_prospective_claim_commit":
                        operation = lambda: store.claim_prospective_turn(request)
                    else:
                        store.claim_prospective_turn(request)
                        if target == "before_prospective_execution_commit":
                            operation = lambda: store.record_prospective_execution(receipt)
                        else:
                            store.record_prospective_execution(receipt)
                            if target == "before_prospective_feedback_commit":
                                operation = lambda: store.stage_prospective_feedback(feedback)
                            else:
                                cancellation = wrapper()
                                # Use the already-reserved wrapper but a cancellation cannot
                                # follow execution, so start a fresh store state for this case.
                                store = CognitiveTransactionStore(
                                    Path(directory) / "resolution.sqlite3",
                                    fault_injector=fail,
                                )
                                _initialize(store)
                                batch = _batch_acquisition(cancellation, 0, None)
                                store.reserve_prospective_turn(
                                    cancellation,
                                    batch,
                                    b"exact prospective pending state",
                                )
                                lifecycle = ProspectiveResolution.lifecycle(
                                    disposition=ResolutionDisposition.CANCELLED,
                                    reservation=cancellation,
                                )
                                lifecycle_acquisition = _resolution_acquisition(
                                    lifecycle, 1, batch.acquisition_ref
                                )
                                operation = lambda: store.resolve_prospective_turn(
                                    lifecycle, lifecycle_acquisition
                                )
                    expected_clock = 1
                armed["value"] = True
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    operation()
                self.assertEqual(store.head().sequence, 0)  # type: ignore[union-attr]
                self.assertEqual(store.acquisition_head().next_ordinal, expected_clock)
                armed["value"] = False
                transitioned = operation()
                self.assertTrue(transitioned.transitioned)
                self.assertFalse(operation().transitioned)

    def test_v3_tail_corruption_fails_local_and_full_rejoin(self) -> None:
        corruptions = (
            (
                "source",
                "UPDATE cognitive_acquisitions SET source_ref = 'sha256:"
                + "1" * 64
                + "' WHERE ordinal = 1",
            ),
            (
                "predecessor",
                "UPDATE cognitive_acquisitions SET predecessor_acquisition_ref = 'sha256:"
                + "2" * 64
                + "' WHERE ordinal = 1",
            ),
            (
                "record",
                "UPDATE cognitive_acquisitions SET record_ref = 'sha256:"
                + "3" * 64
                + "' WHERE ordinal = 1",
            ),
            (
                "projection",
                "UPDATE acquisition_projection_outbox SET canonical_payload = X'7b7d' "
                "WHERE ordinal = 1",
            ),
            (
                "clock",
                "UPDATE acquisition_clock SET record_ref = 'sha256:"
                + "4" * 64
                + "' WHERE singleton = 1",
            ),
            (
                "successor",
                "UPDATE prospective_turns_v3 SET canonical_batch = X'7b7d'",
            ),
        )
        for label, statement in corruptions:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "corrupt.sqlite3"
                store = CognitiveTransactionStore(path)
                _initialize(store)
                reservation = wrapper()
                batch = _batch_acquisition(reservation, 0, None)
                store.reserve_prospective_turn(
                    reservation, batch, b"exact prospective pending state"
                )
                resolution = ProspectiveResolution.lifecycle(
                    disposition=ResolutionDisposition.CANCELLED,
                    reservation=reservation,
                )
                acquisition = _resolution_acquisition(
                    resolution, 1, batch.acquisition_ref
                )
                store.resolve_prospective_turn(resolution, acquisition)
                with closing(sqlite3.connect(path)) as connection:
                    connection.execute("PRAGMA foreign_keys = OFF")
                    connection.execute("PRAGMA ignore_check_constraints = ON")
                    connection.execute(statement)
                    connection.commit()
                with self.assertRaises(RuntimeError):
                    store.head()
                with self.assertRaises(RuntimeError):
                    store.audit_integrity()

    def test_full_audit_detects_coherently_truncated_legacy_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.sqlite3"
            store = CognitiveTransactionStore(path)
            parent = legacy_digest("truncated-parent")
            child = legacy_digest("truncated-child")
            store.initialize(
                parent,
                b"truncated-parent",
                model_ref=legacy_digest("model"),
                encoder_ref=legacy_digest("encoder"),
            )
            episode = legacy_test_episode(parent, child, task_id="truncated")
            store.commit_episode(
                episode, b"truncated-child", expected_parent_digest=parent
            )
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute("DELETE FROM acquisition_projection_outbox")
                connection.execute("DELETE FROM cognitive_acquisitions")
                connection.execute(
                    "UPDATE acquisition_clock SET next_ordinal = 0, "
                    "acquisition_ref = NULL, record_ref = NULL"
                )
                connection.commit()
            self.assertEqual(store.head().sequence, 1)  # type: ignore[union-attr]
            with self.assertRaisesRegex(RuntimeError, "mirror"):
                store.audit_integrity()

    def test_concurrent_append_and_full_audit_each_use_consistent_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concurrent-audit.sqlite3"
            writer = CognitiveTransactionStore(path)
            parent = legacy_digest("concurrent-audit-state-0")
            writer.initialize(
                parent,
                b"concurrent-audit-state-0",
                model_ref=legacy_digest("model"),
                encoder_ref=legacy_digest("encoder"),
            )
            auditor = CognitiveTransactionStore(path)
            start = threading.Barrier(2)
            finished = threading.Event()

            def append_history() -> None:
                nonlocal parent
                previous = None
                start.wait(timeout=5)
                try:
                    for sequence in range(1, 9):
                        child = legacy_digest(f"concurrent-audit-state-{sequence}")
                        episode = legacy_test_episode(
                            parent,
                            child,
                            task_id=f"concurrent-audit-turn-{sequence}",
                            previous_episode_ref=previous,
                        )
                        writer.commit_episode(
                            episode,
                            f"concurrent-audit-state-{sequence}".encode("ascii"),
                            expected_parent_digest=parent,
                        )
                        parent = child
                        previous = episode.episode_ref
                finally:
                    finished.set()

            def audit_history() -> int:
                start.wait(timeout=5)
                completed = 0
                while not finished.is_set():
                    auditor.audit_integrity()
                    auditor.head()
                    completed += 1
                auditor.audit_integrity()
                return completed

            with ThreadPoolExecutor(max_workers=2) as executor:
                append_future = executor.submit(append_history)
                audit_future = executor.submit(audit_history)
                append_future.result(timeout=30)
                audit_count = audit_future.result(timeout=30)
            self.assertGreaterEqual(audit_count, 1)
            self.assertEqual(auditor.audit_integrity().sequence, 8)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
