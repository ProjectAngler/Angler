from __future__ import annotations

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from angler.cognition.contracts import CognitiveEpisode, ProspectiveCommitment
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveTurnReservation,
)
from angler.runtime.cognitive_transaction_store import (
    CognitiveEpisodeItem,
    CognitiveTransactionStore,
    ReservationTransition,
    TransactionCommit,
    _database_schema_fingerprint,
    _SCHEMA_FINGERPRINT,
    _SCHEMA_V1,
    _V1_SCHEMA_FINGERPRINT,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _initialize(
    store: CognitiveTransactionStore, state_digest: str, state_snapshot: bytes
):
    return store.initialize(
        state_digest,
        state_snapshot,
        model_ref=_digest("model"),
        encoder_ref=_digest("encoder"),
    )


def _episode(
    parent: str,
    child: str,
    *,
    task_id: str = "turn-1",
    previous_episode_ref: str | None = None,
    model_ref: str | None = None,
    encoder_ref: str | None = None,
) -> CognitiveEpisode:
    proposals = (
        "Inspect the supplied evidence before acting.",
        "Ask for one missing constraint.",
        "Apply the bounded public procedure.",
        "Decline because the evidence is insufficient.",
    )
    return CognitiveEpisode(
        task_id=task_id,
        request="Use the public evidence to select a bounded procedure.",
        recalled_refs=(_digest("evidence"),),
        proposals=proposals,
        selected_index=2,
        commitment=ProspectiveCommitment(
            parent_event_ref=previous_episode_ref,
            task_id=task_id,
            candidate_index=2,
            candidate_trace=proposals[2],
            predicted_score=0.75,
            uncertainty=0.25,
            horizon=1,
            competence_state_digest=parent,
        ),
        response="The bounded procedure was applied.",
        observations=("The objective check returned success.",),
        outcome="success",
        feedback_text="Objective feedback: success.",
        feedback_source_ref=_digest("feedback" + task_id),
        parent_state_digest=parent,
        child_state_digest=child,
        model_ref=model_ref or _digest("model"),
        encoder_ref=encoder_ref or _digest("encoder"),
        supporting_evidence_refs=(_digest("evidence"),),
    )


def _populate(
    store: CognitiveTransactionStore, count: int
) -> tuple[CognitiveEpisode, ...]:
    parent = _digest("page-state-0")
    _initialize(store, parent, b"page-state-0")
    previous_episode_ref = None
    episodes = []
    for index in range(1, count + 1):
        child = _digest(f"page-state-{index}")
        value = _episode(
            parent,
            child,
            task_id=f"page-turn-{index}",
            previous_episode_ref=previous_episode_ref,
        )
        store.commit_episode(
            value,
            f"page-state-{index}".encode("ascii"),
            expected_parent_digest=parent,
        )
        episodes.append(value)
        parent = child
        previous_episode_ref = value.episode_ref
    return tuple(episodes)


def _reservation_bundle(
    store: CognitiveTransactionStore,
    parent: str,
    child: str,
    *,
    task_id: str = "reserved-turn",
    receipt_status: str = "COMPLETED",
):
    head = store.head()
    assert head is not None
    episode = _episode(
        parent,
        child,
        task_id=task_id,
        previous_episode_ref=head.episode_ref,
    )
    pending_blob = ("pending-state:" + task_id).encode("utf-8")
    reservation = ProspectiveTurnReservation.create(
        pending_blob=pending_blob,
        parent_sequence=head.sequence,
        parent_event_ref=head.episode_ref,
        parent_competence_digest=head.state_digest,
        parent_snapshot_digest=head.snapshot_sha256,
        model_ref=episode.model_ref,
        encoder_ref=episode.encoder_ref,
        agent_ref=_digest("agent"),
        world_ref=_digest("world"),
        task_id=episode.task_id,
        request=episode.request,
        recalled_refs=episode.recalled_refs,
        proposals=episode.proposals,
        selected_index=episode.selected_index,
        selected_trace=episode.proposals[episode.selected_index],
        logits=(0.0, 0.25, 0.75, -0.5),
        decision_evidence_ref=_digest("decision:" + task_id),
        supporting_evidence_refs=episode.supporting_evidence_refs,
        commitment=episode.commitment,
    )
    request = CognitiveExecutionRequest.from_reservation(
        reservation,
        input_observations=(),
    )
    receipt = CognitiveExecutionReceipt.from_request(
        request,
        status=receipt_status,
        executed_trace=request.selected_trace,
        response=episode.response if receipt_status == "COMPLETED" else "",
        output_observations=(
            episode.observations
            if receipt_status == "COMPLETED"
            else ("Execution did not complete.",)
        ),
    )
    feedback = None
    if receipt_status == "COMPLETED":
        feedback = ObjectiveFeedbackRecord.from_execution(
            reservation,
            request,
            receipt,
            outcome=episode.outcome,
            feedback_text=episode.feedback_text,
            feedback_source_ref=episode.feedback_source_ref,
        )
    return episode, pending_blob, reservation, request, receipt, feedback


class CognitiveTransactionStoreTests(unittest.TestCase):
    def test_commit_restart_exact_state_episode_and_pending_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            parent = _digest("state-zero")
            child = _digest("state-one")
            initial = b"exact initial competence bytes\x00"
            terminal = b"exact terminal competence bytes\xff"
            episode = _episode(parent, child)
            store = CognitiveTransactionStore(path)

            initial_head = _initialize(store, parent, initial)
            committed = store.commit_episode(
                episode,
                terminal,
                expected_parent_digest=parent,
            )

            self.assertEqual(initial_head.sequence, 0)
            self.assertEqual(committed.episode_ref, episode.episode_ref)
            self.assertEqual(committed.sequence, 1)
            self.assertTrue(committed.projection_pending)
            self.assertEqual(committed.head.state_digest, child)
            restarted = CognitiveTransactionStore(path)
            self.assertEqual(restarted.head(), committed.head)
            self.assertEqual(restarted.load_head_state(), terminal)
            self.assertEqual(restarted.get_episode(episode.episode_ref), episode)
            pending = restarted.pending_projections()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].sequence, 1)
            self.assertEqual(pending[0].episode_ref, episode.episode_ref)
            self.assertEqual(pending[0].episode, episode)
            # Re-initialization verifies sequence zero without rewinding head.
            self.assertEqual(_initialize(restarted, parent, initial), committed.head)

    def test_projection_acknowledgement_is_idempotent_and_head_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            parent, child = _digest("p"), _digest("c")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"parent")
            episode = _episode(parent, child)
            commit = store.commit_episode(episode, b"child", expected_parent_digest=parent)

            store.ack_projection(episode.episode_ref)
            store.ack_projection(episode.episode_ref)

            self.assertEqual(store.pending_projections(), ())
            self.assertEqual(store.head(), commit.head)
            self.assertEqual(store.load_head_state(), b"child")
            self.assertEqual(store.get_episode(episode.episode_ref), episode)
            with self.assertRaises(KeyError):
                store.ack_projection(_digest("unknown-event"))

    def test_parent_mismatch_duplicate_and_invalid_snapshot_do_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            parent, child = _digest("p"), _digest("c")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"parent")
            first = _episode(parent, child)
            store.commit_episode(first, b"child", expected_parent_digest=parent)
            expected_head = store.head()
            expected_state = store.load_head_state()
            expected_pending = store.pending_projections()
            expected_bytes = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "optimistic parent"):
                store.commit_episode(
                    _episode(parent, _digest("other"), task_id="wrong-parent"),
                    b"other",
                    expected_parent_digest=parent,
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                store.commit_episode(first, b"duplicate", expected_parent_digest=child)
            with self.assertRaisesRegex(ValueError, "non-empty bytes"):
                store.commit_episode(
                    _episode(child, _digest("empty"), task_id="empty", previous_episode_ref=first.episode_ref),
                    b"",
                    expected_parent_digest=child,
                )

            self.assertEqual(store.head(), expected_head)
            self.assertEqual(store.load_head_state(), expected_state)
            self.assertEqual(store.pending_projections(), expected_pending)
            self.assertEqual(path.read_bytes(), expected_bytes)

    def test_injected_precommit_failure_rolls_back_every_logical_and_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            parent, child = _digest("p"), _digest("c")
            base = CognitiveTransactionStore(path)
            head = _initialize(base, parent, b"parent")
            before = path.read_bytes()

            def fail(stage: str) -> None:
                self.assertEqual(stage, "before_commit")
                raise RuntimeError("injected precommit failure")

            failing = CognitiveTransactionStore(path, fault_injector=fail)
            with self.assertRaisesRegex(RuntimeError, "injected precommit"):
                failing.commit_episode(
                    _episode(parent, child),
                    b"child",
                    expected_parent_digest=parent,
                )

            restarted = CognitiveTransactionStore(path)
            self.assertEqual(restarted.head(), head)
            self.assertEqual(restarted.load_head_state(), b"parent")
            self.assertEqual(restarted.pending_projections(), ())
            with self.assertRaises(KeyError):
                restarted.get_episode(_episode(parent, child).episode_ref)
            self.assertEqual(path.read_bytes(), before)

    def test_initialize_conflict_and_argument_validation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            store = CognitiveTransactionStore(path)
            parent = _digest("parent")
            expected = _initialize(store, parent, b"parent")
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "different state"):
                _initialize(store, _digest("other"), b"other")
            with self.assertRaisesRegex(ValueError, "limit"):
                store.pending_projections(limit=0)

            self.assertEqual(store.head(), expected)
            self.assertEqual(path.read_bytes(), before)

    def test_no_episode_restart_rejects_model_and_encoder_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compatibility.sqlite3"
            parent = _digest("parent")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"parent")
            before = path.read_bytes()

            restarted = CognitiveTransactionStore(path)
            restarted.assert_compatibility(_digest("model"), _digest("encoder"))
            with self.assertRaisesRegex(ValueError, "another model or encoder"):
                restarted.assert_compatibility(_digest("different-model"), _digest("encoder"))
            with self.assertRaisesRegex(ValueError, "another model or encoder"):
                restarted.assert_compatibility(_digest("model"), _digest("different-encoder"))
            with self.assertRaisesRegex(ValueError, "lowercase sha256"):
                restarted.assert_compatibility("sha256:" + "A" * 64, _digest("encoder"))

            self.assertEqual(restarted.head().sequence, 0)
            self.assertEqual(path.read_bytes(), before)

    def test_mixed_model_or_encoder_episode_fails_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-identity.sqlite3"
            parent, child = _digest("parent"), _digest("child")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"parent")
            before = path.read_bytes()
            for label, episode in (
                (
                    "model",
                    _episode(parent, child, task_id="mixed-model", model_ref=_digest("other-model")),
                ),
                (
                    "encoder",
                    _episode(
                        parent,
                        child,
                        task_id="mixed-encoder",
                        encoder_ref=_digest("other-encoder"),
                    ),
                ),
            ):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "another model or encoder"):
                        store.commit_episode(episode, b"child", expected_parent_digest=parent)
                    self.assertEqual(store.head().sequence, 0)
                    self.assertEqual(store.load_head_state(), b"parent")
                    self.assertEqual(path.read_bytes(), before)

    def test_forged_frozen_episode_is_revalidated_at_store_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged.sqlite3"
            parent, child = _digest("parent"), _digest("child")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"parent")
            forged = _episode(parent, child)
            object.__setattr__(
                forged.commitment,
                "competence_state_digest",
                _digest("forged-competence"),
            )
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "canonical contract|competence state"):
                store.commit_episode(forged, b"child", expected_parent_digest=parent)

            self.assertEqual(store.head().sequence, 0)
            self.assertEqual(store.load_head_state(), b"parent")
            self.assertEqual(path.read_bytes(), before)

    def test_first_and_subsequent_forged_event_parents_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            zero, one, two = _digest("zero"), _digest("one"), _digest("two")
            store = CognitiveTransactionStore(path)
            _initialize(store, zero, b"zero")
            before_first = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "current event head"):
                store.commit_episode(
                    _episode(
                        zero,
                        one,
                        task_id="forged-first",
                        previous_episode_ref=_digest("not-the-initial-head"),
                    ),
                    b"one",
                    expected_parent_digest=zero,
                )
            self.assertEqual(path.read_bytes(), before_first)

            first = _episode(zero, one, task_id="valid-first")
            store.commit_episode(first, b"one", expected_parent_digest=zero)
            expected_head = store.head()
            before_second = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "current event head"):
                store.commit_episode(
                    _episode(
                        one,
                        two,
                        task_id="forged-second",
                        previous_episode_ref=_digest("not-first"),
                    ),
                    b"two",
                    expected_parent_digest=one,
                )
            self.assertEqual(store.head(), expected_head)
            self.assertEqual(store.load_head_state(), b"one")
            self.assertEqual(path.read_bytes(), before_second)

    def test_repeated_state_digest_cannot_create_an_aba_stale_parent_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cognitive.sqlite3"
            zero, one = _digest("zero"), _digest("one")
            store = CognitiveTransactionStore(path)
            _initialize(store, zero, b"zero-v0")
            first = _episode(zero, one, task_id="to-one")
            store.commit_episode(first, b"one", expected_parent_digest=zero)
            second = _episode(
                one,
                zero,
                task_id="back-to-zero",
                previous_episode_ref=first.episode_ref,
            )
            store.commit_episode(second, b"zero-v2", expected_parent_digest=one)
            expected_head = store.head()
            before = path.read_bytes()

            stale = _episode(zero, _digest("stale-child"), task_id="stale-aba")
            with self.assertRaisesRegex(ValueError, "current event head"):
                store.commit_episode(stale, b"stale", expected_parent_digest=zero)

            self.assertEqual(store.head(), expected_head)
            self.assertEqual(store.load_head_state(), b"zero-v2")
            self.assertEqual(path.read_bytes(), before)

    def test_restart_rejects_snapshot_outbox_and_schema_corruption(self) -> None:
        corruptions = (
            (
                "snapshot",
                "UPDATE state_history SET snapshot = X'626164' WHERE sequence = 1",
                "snapshot digest|canonical cognitive head",
            ),
            (
                "outbox",
                "UPDATE projection_outbox SET canonical_payload = X'7b7d' WHERE sequence = 1",
                "foreign-key|outbox|local binding",
            ),
            ("head", "UPDATE canonical_head SET state_digest = 'sha256:" + "0" * 64 + "'", "foreign-key|head"),
        )
        for label, statement, expected_error in corruptions:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cognitive.sqlite3"
                parent, child = _digest("parent"), _digest("child")
                store = CognitiveTransactionStore(path)
                _initialize(store, parent, b"parent")
                store.commit_episode(
                    _episode(parent, child), b"child", expected_parent_digest=parent
                )
                with closing(sqlite3.connect(path)) as connection:
                    connection.execute("PRAGMA foreign_keys = OFF")
                    connection.execute(statement)
                    connection.commit()
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    CognitiveTransactionStore(path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-version.sqlite3"
            CognitiveTransactionStore(path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA user_version = 999")
                connection.commit()
            with self.assertRaisesRegex(RuntimeError, "schema version 999"):
                CognitiveTransactionStore(path)

    def test_same_instance_tail_reads_reject_current_snapshot_and_payload_corruption(self) -> None:
        corruptions = (
            (
                "snapshot",
                "UPDATE state_history SET snapshot = X'626164' WHERE sequence = 1",
                "head|snapshot|binding",
            ),
            (
                "episode-payload",
                "UPDATE episodes SET canonical_payload = X'7b7d' WHERE sequence = 1",
                "local binding|canonical",
            ),
            (
                "outbox-payload",
                "UPDATE projection_outbox SET canonical_payload = X'7b7d' WHERE sequence = 1",
                "local binding",
            ),
        )
        for label, statement, expected_error in corruptions:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "same-instance.sqlite3"
                parent, child = _digest("parent"), _digest("child")
                store = CognitiveTransactionStore(path)
                _initialize(store, parent, b"parent")
                store.commit_episode(
                    _episode(parent, child), b"child", expected_parent_digest=parent
                )
                with closing(sqlite3.connect(path)) as connection:
                    connection.execute("PRAGMA foreign_keys = OFF")
                    connection.execute(statement)
                    connection.commit()

                with self.assertRaisesRegex(RuntimeError, expected_error):
                    store.head()

    def test_concurrent_same_parent_two_store_race_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.sqlite3"
            parent = _digest("parent")
            first_store = CognitiveTransactionStore(path)
            _initialize(first_store, parent, b"parent")
            second_store = CognitiveTransactionStore(path)
            barrier = threading.Barrier(2)
            candidates = (
                (first_store, _episode(parent, _digest("child-a"), task_id="race-a"), b"a"),
                (second_store, _episode(parent, _digest("child-b"), task_id="race-b"), b"b"),
            )

            def attempt(item):
                store, episode, snapshot = item
                barrier.wait(timeout=5.0)
                return store.commit_episode(
                    episode, snapshot, expected_parent_digest=parent
                )

            results: list[TransactionCommit] = []
            failures: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(attempt, item) for item in candidates)
                for future in futures:
                    try:
                        results.append(future.result(timeout=10.0))
                    except BaseException as exc:
                        failures.append(exc)

            self.assertEqual(len(results), 1)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], ValueError)
            self.assertRegex(str(failures[0]), "optimistic parent state digest mismatch")
            winner = results[0]
            reopened = CognitiveTransactionStore(path)
            self.assertEqual(reopened.head(), winner.head)
            self.assertEqual(reopened.head().sequence, 1)
            self.assertEqual(len(reopened.pending_projections()), 1)
            self.assertEqual(reopened.audit_integrity(), winner.head)

    def test_operational_paths_never_run_the_full_history_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.sqlite3"
            store = CognitiveTransactionStore(path)
            parent, child = _digest("parent"), _digest("child")
            episode = _episode(parent, child)
            with patch.object(
                CognitiveTransactionStore,
                "_verify_database",
                side_effect=AssertionError("ordinary path invoked full-history audit"),
            ):
                _initialize(store, parent, b"parent")
                self.assertEqual(store.head().sequence, 0)
                self.assertEqual(store.load_head_state(), b"parent")
                store.commit_episode(episode, b"child", expected_parent_digest=parent)
                self.assertEqual(store.get_episode(episode.episode_ref), episode)
                self.assertEqual(store.get_episode_item(episode.episode_ref).episode, episode)
                self.assertEqual(len(store.episode_items(limit=1)), 1)
                self.assertEqual(len(store.pending_projections()), 1)
                store.ack_projection(episode.episode_ref)
                self.assertEqual(store.pending_projections(), ())

    def test_counterfeit_or_unversioned_schemas_fail_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "blank-v1.sqlite3"
            with closing(sqlite3.connect(blank)) as connection:
                connection.execute("PRAGMA user_version = 1")
                connection.commit()
            with self.assertRaisesRegex(RuntimeError, "schema fingerprint"):
                CognitiveTransactionStore(blank)
            with closing(sqlite3.connect(blank)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
                    ).fetchone()[0],
                    0,
                )

            counterfeit = Path(directory) / "constraint-free-v1.sqlite3"
            with closing(sqlite3.connect(counterfeit)) as connection:
                connection.executescript(
                    "CREATE TABLE state_history(sequence INTEGER);"
                    "CREATE TABLE episodes(episode_ref TEXT);"
                    "CREATE TABLE canonical_head(singleton INTEGER);"
                    "CREATE TABLE projection_outbox(sequence INTEGER);"
                    "CREATE TABLE store_identity(singleton INTEGER);"
                    "PRAGMA user_version = 1;"
                )
                before = tuple(
                    connection.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "schema fingerprint"):
                CognitiveTransactionStore(counterfeit)
            with closing(sqlite3.connect(counterfeit)) as connection:
                after = tuple(
                    connection.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
                    )
                )
            self.assertEqual(after, before)

            unversioned = Path(directory) / "unversioned.sqlite3"
            with closing(sqlite3.connect(unversioned)) as connection:
                connection.execute("CREATE TABLE unrelated(value TEXT)")
                connection.commit()
            with self.assertRaisesRegex(RuntimeError, "unversioned"):
                CognitiveTransactionStore(unversioned)
            with closing(sqlite3.connect(unversioned)) as connection:
                names = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                    )
                )
            self.assertEqual(names, ("unrelated",))

    def test_episode_item_exact_lookup_and_ordered_bounded_paging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "paging.sqlite3")
            episodes = _populate(store, 5)

            exact = store.get_episode_item(episodes[2].episode_ref)
            self.assertIsInstance(exact, CognitiveEpisodeItem)
            self.assertEqual(exact.sequence, 3)
            self.assertEqual(exact.episode_ref, episodes[2].episode_ref)
            self.assertEqual(exact.episode, episodes[2])
            with self.assertRaises(FrozenInstanceError):
                exact.sequence = 99  # type: ignore[misc]

            first = store.episode_items(limit=2)
            second = store.episode_items(after_sequence=first[-1].sequence, limit=2)
            third = store.episode_items(after_sequence=second[-1].sequence, limit=2)
            self.assertEqual([item.sequence for item in first], [1, 2])
            self.assertEqual([item.sequence for item in second], [3, 4])
            self.assertEqual([item.sequence for item in third], [5])
            self.assertEqual(
                tuple(item.episode for item in first + second + third), episodes
            )
            self.assertEqual(store.episode_items(after_sequence=5, limit=256), ())

            projection_page = store.pending_projections(limit=2)
            self.assertEqual([item.sequence for item in projection_page], [1, 2])
            self.assertEqual(
                tuple(item.episode for item in projection_page), episodes[:2]
            )

    def test_episode_item_unknown_and_page_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "bounds.sqlite3")
            _populate(store, 1)
            with self.assertRaises(KeyError):
                store.get_episode_item(_digest("unknown-episode"))
            with self.assertRaisesRegex(ValueError, "lowercase sha256"):
                store.get_episode_item("not-a-reference")
            for value in (-1, True, 1.0, "0"):
                with self.subTest(after_sequence=value):
                    with self.assertRaisesRegex(ValueError, "after_sequence"):
                        store.episode_items(after_sequence=value)  # type: ignore[arg-type]
            for value in (0, 257, True, 1.0, "1"):
                with self.subTest(limit=value):
                    with self.assertRaisesRegex(ValueError, "limit"):
                        store.episode_items(limit=value)  # type: ignore[arg-type]
                    with self.assertRaisesRegex(ValueError, "limit"):
                        store.pending_projections(limit=value)  # type: ignore[arg-type]

    def test_episode_paging_revalidates_corrupted_non_tail_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt-page.sqlite3"
            store = CognitiveTransactionStore(path)
            episodes = _populate(store, 3)
            # Corrupt only sequence two.  The current tail (three) remains
            # valid, so rejection proves paging checks each requested row.
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute(
                    "UPDATE projection_outbox SET canonical_payload = X'7b7d' "
                    "WHERE sequence = 2"
                )
                connection.commit()

            self.assertEqual(store.head().sequence, 3)
            with self.assertRaisesRegex(RuntimeError, "local binding"):
                store.episode_items(after_sequence=1, limit=2)
            with self.assertRaisesRegex(RuntimeError, "local binding"):
                store.get_episode_item(episodes[1].episode_ref)

    def test_reservation_lifecycle_restarts_and_commits_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservation.sqlite3"
            parent, child = _digest("reserved-parent"), _digest("reserved-child")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"reserved-parent")
            episode, pending, reservation, request, receipt, feedback = (
                _reservation_bundle(store, parent, child)
            )
            assert feedback is not None

            reserved = store.reserve_turn(reservation, pending)
            self.assertIsInstance(reserved, ReservationTransition)
            self.assertTrue(reserved.transitioned)
            self.assertEqual(reserved.record.status, "RESERVED")
            self.assertEqual(reserved.record.pending_blob, pending)
            self.assertFalse(store.reserve_turn(reservation, pending).transitioned)

            restarted = CognitiveTransactionStore(path)
            self.assertEqual(restarted.active_reservation(), reserved.record)
            claimed = restarted.claim_turn(request)
            self.assertTrue(claimed.transitioned)
            self.assertEqual(claimed.record.status, "CLAIMED")
            self.assertEqual(
                CognitiveTransactionStore(path).active_reservation(), claimed.record
            )
            self.assertFalse(restarted.claim_turn(request).transitioned)

            recorded = restarted.record_execution(receipt)
            self.assertTrue(recorded.transitioned)
            self.assertEqual(recorded.record.status, "EXECUTION_RECORDED")
            self.assertEqual(
                CognitiveTransactionStore(path).active_reservation(), recorded.record
            )
            self.assertFalse(restarted.record_execution(receipt).transitioned)
            conflicting_receipt = CognitiveExecutionReceipt.from_request(
                request,
                status="COMPLETED",
                executed_trace=request.selected_trace,
                response="A conflicting public response.",
                output_observations=episode.observations,
            )
            with self.assertRaisesRegex(ValueError, "different receipt bytes"):
                restarted.record_execution(conflicting_receipt)
            staged = restarted.stage_feedback(feedback)
            self.assertTrue(staged.transitioned)
            self.assertEqual(staged.record.feedback, feedback)
            self.assertEqual(
                CognitiveTransactionStore(path).active_reservation(), staged.record
            )
            self.assertFalse(restarted.stage_feedback(feedback).transitioned)
            conflicting_feedback = ObjectiveFeedbackRecord.from_execution(
                reservation,
                request,
                receipt,
                outcome=episode.outcome,
                feedback_text="Conflicting objective feedback.",
                feedback_source_ref=episode.feedback_source_ref,
            )
            with self.assertRaisesRegex(ValueError, "different feedback bytes"):
                restarted.stage_feedback(conflicting_feedback)

            committed = restarted.commit_reserved_episode(
                reservation.reservation_ref,
                episode,
                b"reserved-child",
                expected_parent_digest=parent,
            )
            self.assertEqual(committed.episode_ref, episode.episode_ref)
            self.assertEqual(committed.head.state_digest, child)
            self.assertIsNone(restarted.active_reservation())
            resolved = restarted.get_reservation(reservation.reservation_ref)
            self.assertEqual(resolved.status, "RESOLVED")
            self.assertEqual(resolved.resolution_disposition, "EPISODE_COMMITTED")
            self.assertEqual(resolved.resolved_episode_ref, episode.episode_ref)
            self.assertIsNone(resolved.pending_blob)
            self.assertEqual(resolved.feedback, feedback)
            replay = restarted.commit_reserved_episode(
                reservation.reservation_ref,
                episode,
                b"reserved-child",
                expected_parent_digest=parent,
            )
            self.assertEqual(replay, committed)
            self.assertEqual(restarted.audit_integrity(), committed.head)

    def test_reservation_identity_allows_reused_commitment_after_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservation-identity.sqlite3"
            parent = _digest("shared-commitment-parent")
            store = CognitiveTransactionStore(path)
            initial = _initialize(store, parent, b"shared-commitment-parent")
            _, pending, first, _, _, _ = _reservation_bundle(
                store,
                parent,
                _digest("unused-first-child"),
                task_id="shared-commitment",
            )
            store.reserve_turn(first, pending)
            first_result = store.cancel_reservation(first.reservation_ref)
            self.assertEqual(first_result.record.resolution_disposition, "CANCELLED")

            second = replace(
                first,
                request="A second situated request using the same procedure commitment.",
                agent_ref=_digest("second-agent"),
                world_ref=_digest("second-world"),
            )
            self.assertEqual(second.commitment_ref, first.commitment_ref)
            self.assertNotEqual(second.reservation_ref, first.reservation_ref)
            self.assertEqual(second.idempotency_key, second.reservation_ref)
            store.reserve_turn(second, pending)
            request = CognitiveExecutionRequest.from_reservation(second)
            self.assertEqual(request.idempotency_key, second.reservation_ref)
            store.claim_turn(request)
            receipt = CognitiveExecutionReceipt.from_request(
                request,
                status="ERROR",
                executed_trace=request.selected_trace,
                response="",
                output_observations=("Known non-completion.",),
            )
            store.record_execution(receipt)
            second_result = store.resolve_noncompletion(second.reservation_ref)
            self.assertEqual(second_result.record.resolution_disposition, "ERROR")
            self.assertEqual(store.head(), initial)

            first_record = store.get_reservation(first.reservation_ref)
            second_record = store.get_reservation(second.reservation_ref)
            self.assertEqual(first_record.commitment_ref, second_record.commitment_ref)
            self.assertNotEqual(first_record.reservation_ref, second_record.reservation_ref)
            with closing(sqlite3.connect(path)) as connection:
                rows = connection.execute(
                    "SELECT reservation_ref FROM turn_reservations "
                    "WHERE commitment_ref = ? ORDER BY reservation_ref",
                    (first.commitment_ref,),
                ).fetchall()
                table_info = {
                    str(row[1]): int(row[5])
                    for row in connection.execute("PRAGMA table_info(turn_reservations)")
                }
            self.assertEqual(len(rows), 2)
            self.assertEqual(table_info["reservation_ref"], 1)
            self.assertEqual(table_info["commitment_ref"], 0)
            self.assertEqual(store.audit_integrity(), initial)

    def test_one_active_conflicts_and_claim_race_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race-reservation.sqlite3"
            parent = _digest("reservation-race-parent")
            first = CognitiveTransactionStore(path)
            _initialize(first, parent, b"reservation-race-parent")
            _, pending, reservation, request, _, _ = _reservation_bundle(
                first, parent, _digest("unused-child"), task_id="race-reservation"
            )
            first.reserve_turn(reservation, pending)

            distinct_same_commitment = replace(
                reservation,
                request="A different request under the same commitment identity.",
            )
            self.assertEqual(
                distinct_same_commitment.commitment_ref, reservation.commitment_ref
            )
            self.assertNotEqual(
                distinct_same_commitment.reservation_ref, reservation.reservation_ref
            )
            with self.assertRaisesRegex(ValueError, "already active"):
                first.reserve_turn(distinct_same_commitment, pending)

            _, second_pending, second_reservation, _, _, _ = _reservation_bundle(
                first, parent, _digest("second-child"), task_id="second-reservation"
            )
            with self.assertRaisesRegex(ValueError, "already active"):
                first.reserve_turn(second_reservation, second_pending)

            second = CognitiveTransactionStore(path)
            barrier = threading.Barrier(2)

            def claim(store: CognitiveTransactionStore) -> ReservationTransition:
                barrier.wait(timeout=5.0)
                return store.claim_turn(request)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(
                    future.result(timeout=10.0)
                    for future in (
                        executor.submit(claim, first),
                        executor.submit(claim, second),
                    )
                )
            self.assertEqual(sorted(result.transitioned for result in results), [False, True])
            self.assertTrue(all(result.record.status == "CLAIMED" for result in results))

            conflicting_request = CognitiveExecutionRequest.from_reservation(
                reservation,
                input_observations=("Different exact request bytes.",),
            )
            with self.assertRaisesRegex(ValueError, "different execution request"):
                first.claim_turn(conflicting_request)
            with self.assertRaisesRegex(ValueError, "cannot be cancelled"):
                first.cancel_reservation(reservation.reservation_ref)

    def test_two_store_reservation_race_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reserve-race.sqlite3"
            parent = _digest("reserve-race-parent")
            first_store = CognitiveTransactionStore(path)
            _initialize(first_store, parent, b"reserve-race-parent")
            second_store = CognitiveTransactionStore(path)
            bundles = (
                _reservation_bundle(
                    first_store,
                    parent,
                    _digest("reserve-race-child-a"),
                    task_id="reserve-race-a",
                ),
                _reservation_bundle(
                    second_store,
                    parent,
                    _digest("reserve-race-child-b"),
                    task_id="reserve-race-b",
                ),
            )
            barrier = threading.Barrier(2)

            def attempt(index: int) -> ReservationTransition:
                store = first_store if index == 0 else second_store
                pending = bundles[index][1]
                reservation = bundles[index][2]
                barrier.wait(timeout=5.0)
                return store.reserve_turn(reservation, pending)

            results: list[ReservationTransition] = []
            failures: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(attempt, index) for index in range(2))
                for future in futures:
                    try:
                        results.append(future.result(timeout=10.0))
                    except BaseException as exc:
                        failures.append(exc)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].transitioned)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], ValueError)
            self.assertRegex(str(failures[0]), "already active")
            active = CognitiveTransactionStore(path).active_reservation()
            assert active is not None
            self.assertEqual(active.commitment_ref, results[0].record.commitment_ref)

    def test_cancellation_and_known_noncompletion_free_the_active_slot(self) -> None:
        for receipt_status in ("CLARIFICATION_REQUIRED", "ERROR"):
            with self.subTest(receipt_status=receipt_status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "noncompletion.sqlite3"
                parent = _digest("noncompletion-parent")
                store = CognitiveTransactionStore(path)
                initial = _initialize(store, parent, b"noncompletion-parent")
                episode, pending, reservation, request, receipt, feedback = _reservation_bundle(
                    store,
                    parent,
                    _digest("never-committed"),
                    task_id="noncompletion-" + receipt_status.lower(),
                    receipt_status=receipt_status,
                )
                self.assertIsNone(feedback)
                store.reserve_turn(reservation, pending)
                store.claim_turn(request)
                store.record_execution(receipt)
                with self.assertRaisesRegex(
                    ValueError, "completed execution|request, receipt, and feedback"
                ):
                    store.commit_reserved_episode(
                        reservation.reservation_ref,
                        _episode(parent, _digest("never")),
                        b"never",
                        expected_parent_digest=parent,
                    )
                resolved = store.resolve_noncompletion(reservation.reservation_ref)
                self.assertTrue(resolved.transitioned)
                self.assertEqual(resolved.record.resolution_disposition, receipt_status)
                self.assertIsNone(resolved.record.pending_blob)
                self.assertEqual(store.head(), initial)
                self.assertFalse(
                    store.resolve_noncompletion(reservation.reservation_ref).transitioned
                )
                with self.assertRaisesRegex(ValueError, "exact reserved context"):
                    store.commit_episode(
                        episode,
                        b"must-not-commit",
                        expected_parent_digest=parent,
                    )

                _, next_pending, next_reservation, _, _, _ = _reservation_bundle(
                    store,
                    parent,
                    _digest("next-child"),
                    task_id="next-" + receipt_status.lower(),
                )
                store.reserve_turn(next_reservation, next_pending)
                cancelled = store.cancel_reservation(next_reservation.reservation_ref)
                self.assertTrue(cancelled.transitioned)
                self.assertEqual(cancelled.record.resolution_disposition, "CANCELLED")
                self.assertFalse(
                    store.cancel_reservation(next_reservation.reservation_ref).transitioned
                )
                self.assertEqual(store.head(), initial)

    def test_reserved_commit_fault_preserves_recorded_execution_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reserved-fault.sqlite3"

            def inject(stage: str) -> None:
                if stage == "before_reserved_commit":
                    raise RuntimeError("injected final transaction failure")

            store = CognitiveTransactionStore(path, fault_injector=inject)
            parent, child = _digest("fault-parent"), _digest("fault-child")
            initial = _initialize(store, parent, b"fault-parent")
            episode, pending, reservation, request, receipt, feedback = (
                _reservation_bundle(store, parent, child, task_id="fault-reserved")
            )
            assert feedback is not None
            store.reserve_turn(reservation, pending)
            store.claim_turn(request)
            store.record_execution(receipt)
            store.stage_feedback(feedback)

            with self.assertRaisesRegex(RuntimeError, "injected final"):
                store.commit_reserved_episode(
                    reservation.reservation_ref,
                    episode,
                    b"fault-child",
                    expected_parent_digest=parent,
                )
            self.assertEqual(store.head(), initial)
            retryable = store.active_reservation()
            assert retryable is not None
            self.assertEqual(retryable.status, "EXECUTION_RECORDED")
            self.assertEqual(retryable.pending_blob, pending)
            self.assertEqual(retryable.feedback, feedback)
            self.assertEqual(store.pending_projections(), ())

            restarted = CognitiveTransactionStore(path)
            result = restarted.commit_reserved_episode(
                reservation.reservation_ref,
                episode,
                b"fault-child",
                expected_parent_digest=parent,
            )
            self.assertEqual(result.head.state_digest, child)
            self.assertIsNone(restarted.active_reservation())

    def test_each_precommit_lifecycle_fault_rolls_back_to_prior_boundary(self) -> None:
        cases = (
            ("before_reservation_commit", None),
            ("before_claim_commit", "RESERVED"),
            ("before_execution_record_commit", "CLAIMED"),
            ("before_feedback_commit", "EXECUTION_RECORDED"),
        )
        for target, expected_status in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "boundary-fault.sqlite3"

                def inject(stage: str) -> None:
                    if stage == target:
                        raise RuntimeError("injected boundary failure")

                store = CognitiveTransactionStore(path, fault_injector=inject)
                parent = _digest("boundary-parent")
                _initialize(store, parent, b"boundary-parent")
                _, pending, reservation, request, receipt, feedback = _reservation_bundle(
                    store,
                    parent,
                    _digest("boundary-child"),
                    task_id="boundary-" + target,
                )
                assert feedback is not None
                operations = (
                    lambda: store.reserve_turn(reservation, pending),
                    lambda: store.claim_turn(request),
                    lambda: store.record_execution(receipt),
                    lambda: store.stage_feedback(feedback),
                )
                target_index = tuple(item[0] for item in cases).index(target)
                for operation in operations[:target_index]:
                    operation()
                with self.assertRaisesRegex(RuntimeError, "injected boundary failure"):
                    operations[target_index]()
                reopened = CognitiveTransactionStore(path)
                active = reopened.active_reservation()
                if expected_status is None:
                    self.assertIsNone(active)
                else:
                    assert active is not None
                    self.assertEqual(active.status, expected_status)

    def test_exact_v1_migration_preserves_aggregate_and_interruption_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known-v1.sqlite3"
            parent, child = _digest("migration-parent"), _digest("migration-child")
            current = CognitiveTransactionStore(path)
            _initialize(current, parent, b"migration-parent")
            episode = _episode(parent, child, task_id="migration")
            committed = current.commit_episode(
                episode, b"migration-child", expected_parent_digest=parent
            )
            with closing(sqlite3.connect(path)) as connection:
                before = {
                    table: tuple(connection.execute(f"SELECT * FROM {table}"))
                    for table in (
                        "store_identity",
                        "state_history",
                        "episodes",
                        "canonical_head",
                        "projection_outbox",
                    )
                }
                connection.execute("DROP TABLE prospective_turns_v3")
                connection.execute("DROP TABLE acquisition_projection_outbox")
                connection.execute("DROP TABLE acquisition_clock")
                connection.execute("DROP TABLE cognitive_acquisitions")
                connection.execute("DROP INDEX one_active_turn_reservation")
                connection.execute("DROP TABLE turn_reservations")
                connection.execute("PRAGMA user_version = 1")
                connection.commit()
                self.assertEqual(
                    _database_schema_fingerprint(connection), _V1_SCHEMA_FINGERPRINT
                )

            migrated = CognitiveTransactionStore(path)
            self.assertEqual(migrated.head(), committed.head)
            self.assertEqual(migrated.get_episode(episode.episode_ref), episode)
            self.assertIsNone(migrated.active_reservation())
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _SCHEMA_FINGERPRINT
                )
                after = {
                    table: tuple(connection.execute(f"SELECT * FROM {table}"))
                    for table in before
                }
            self.assertEqual(after, before)

            interrupted = Path(directory) / "interrupted-v1.sqlite3"
            with closing(sqlite3.connect(interrupted)) as connection:
                connection.executescript(_SCHEMA_V1)
                connection.execute("PRAGMA user_version = 1")
                connection.commit()

            def fail_migration(stage: str) -> None:
                if stage == "before_migration_commit":
                    raise RuntimeError("injected migration failure")

            with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                CognitiveTransactionStore(interrupted, fault_injector=fail_migration)
            with closing(sqlite3.connect(interrupted)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _V1_SCHEMA_FINGERPRINT
                )
                self.assertIsNone(
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name = 'turn_reservations'"
                    ).fetchone()
                )
            CognitiveTransactionStore(interrupted)

            concurrent = Path(directory) / "concurrent-v1.sqlite3"
            with closing(sqlite3.connect(concurrent)) as connection:
                connection.executescript(_SCHEMA_V1)
                connection.execute("PRAGMA user_version = 1")
                connection.commit()
            barrier = threading.Barrier(2)

            def migrate_concurrently() -> CognitiveTransactionStore:
                barrier.wait(timeout=5.0)
                return CognitiveTransactionStore(concurrent)

            with ThreadPoolExecutor(max_workers=2) as executor:
                migrated_stores = tuple(
                    future.result(timeout=10.0)
                    for future in (
                        executor.submit(migrate_concurrently),
                        executor.submit(migrate_concurrently),
                    )
                )
            self.assertEqual(len(migrated_stores), 2)
            with closing(sqlite3.connect(concurrent)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(
                    _database_schema_fingerprint(connection), _SCHEMA_FINGERPRINT
                )

    def test_reservation_corruption_fails_restart_and_full_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservation-corrupt.sqlite3"
            parent = _digest("corrupt-reservation-parent")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"corrupt-reservation-parent")
            _, pending, reservation, _, _, _ = _reservation_bundle(
                store, parent, _digest("corrupt-child")
            )
            store.reserve_turn(reservation, pending)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA ignore_check_constraints = ON")
                connection.execute(
                    "UPDATE turn_reservations SET pending_blob = X'626164'"
                )
                connection.commit()
            with self.assertRaisesRegex(
                RuntimeError, "integrity|prospective reservation|canonical"
            ):
                CognitiveTransactionStore(path)
            with self.assertRaisesRegex(
                RuntimeError, "integrity|prospective reservation|canonical"
            ):
                store.audit_integrity()

    def test_reservation_operational_paths_do_not_run_full_history_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservation-bounded.sqlite3"
            parent, child = _digest("bounded-parent"), _digest("bounded-child")
            store = CognitiveTransactionStore(path)
            _initialize(store, parent, b"bounded-parent")
            episode, pending, reservation, request, receipt, feedback = (
                _reservation_bundle(store, parent, child, task_id="bounded-reservation")
            )
            assert feedback is not None
            with patch.object(
                CognitiveTransactionStore,
                "_verify_database",
                side_effect=AssertionError("ordinary path invoked full-history audit"),
            ):
                store.reserve_turn(reservation, pending)
                store.get_reservation(reservation.reservation_ref)
                store.active_reservation()
                store.claim_turn(request)
                store.record_execution(receipt)
                store.stage_feedback(feedback)
                store.commit_reserved_episode(
                    reservation.reservation_ref,
                    episode,
                    b"bounded-child",
                    expected_parent_digest=parent,
                )
                self.assertIsNone(store.active_reservation())

    def test_constructor_and_restart_never_run_full_history_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded-startup.sqlite3"
            parent, child = _digest("startup-parent"), _digest("startup-child")
            with patch.object(
                CognitiveTransactionStore,
                "_verify_database",
                side_effect=AssertionError("constructor invoked full-history audit"),
            ):
                store = CognitiveTransactionStore(path)
                _initialize(store, parent, b"startup-parent")
                episode = _episode(parent, child, task_id="bounded-startup")
                store.commit_episode(
                    episode,
                    b"startup-child",
                    expected_parent_digest=parent,
                )
                restarted = CognitiveTransactionStore(path)
                self.assertEqual(restarted.head().sequence, 1)
                self.assertEqual(restarted.get_episode(episode.episode_ref), episode)


if __name__ == "__main__":
    unittest.main()
