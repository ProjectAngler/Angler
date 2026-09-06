from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import patch

from angler.runtime.tool_operations import (
    ToolCallReservation,
    ToolDefinition,
    ToolEvent,
    ToolManifest,
    ToolOperationStore,
)


REQUESTER_REF = "sha256:" + "a" * 64


def _tool(*, external_effect: bool = False, description: str = "Read a fact."):
    return ToolDefinition(
        tool_id="research.lookup",
        tool_version="1.0.0",
        description=description,
        permission_scope="internal.research",
        input_schema_json=(
            '{"additionalProperties":false,"properties":{"query":{"type":"string"}},'
            '"required":["query"],"type":"object"}'
        ),
        output_schema_json=(
            '{"additionalProperties":false,"properties":{"answer":{"type":"string"}},'
            '"required":["answer"],"type":"object"}'
        ),
        external_effect=external_effect,
    )


def _manifest(tool=None):
    return ToolManifest("jenny.tools", "1.0.0", (tool or _tool(),))


def _reservation(manifest, *, call_id="tool-call:one", arguments=None):
    return ToolCallReservation(
        call_id=call_id,
        manifest_ref=manifest.manifest_ref,
        tool_ref=manifest.tools[0].definition_ref,
        requester_ref=REQUESTER_REF,
        arguments_json=arguments or '{"query":"moving origin"}',
    )


def _event(
    operation_ref,
    sequence,
    status,
    result_json,
    recorded_at_utc,
):
    return ToolEvent(
        operation_ref=operation_ref,
        sequence=sequence,
        status=status,
        result_json=result_json,
        recorded_at_utc=recorded_at_utc,
    )


class ToolOperationContractTests(unittest.TestCase):
    def test_store_uses_private_regular_database_and_wal_files(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "operations.sqlite3"
            store = ToolOperationStore(database)
            self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)

            connection = store._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                sidecars = tuple(
                    Path(f"{database}{suffix}") for suffix in ("-wal", "-shm")
                )
                self.assertTrue(all(path.is_file() for path in sidecars))
                self.assertTrue(
                    all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in sidecars)
                )

                os.chmod(database, 0o666)
                for path in sidecars:
                    os.chmod(path, 0o666)
                check = store._connect()
                check.close()
                self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
                self.assertTrue(
                    all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in sidecars)
                )
            finally:
                connection.rollback()
                connection.close()

    def test_store_rejects_symlink_nonregular_and_foreign_owner_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "unrelated.txt"
            target.write_text("preserve", encoding="utf-8")
            symlink = root / "linked.sqlite3"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "cannot be opened safely"):
                ToolOperationStore(symlink)
            self.assertEqual(target.read_text(encoding="utf-8"), "preserve")

            nonregular = root / "directory.sqlite3"
            nonregular.mkdir()
            with self.assertRaisesRegex(RuntimeError, "cannot be opened safely"):
                ToolOperationStore(nonregular)

            foreign = root / "foreign.sqlite3"
            foreign.touch(mode=0o600)
            current_uid = os.getuid()
            with patch(
                "angler.runtime.tool_operations.os.getuid",
                return_value=current_uid + 1,
            ):
                with self.assertRaisesRegex(RuntimeError, "ownership differs"):
                    ToolOperationStore(foreign)

    def test_records_are_content_addressed_and_json_must_be_bounded_canonical(self):
        first = _tool()
        second = _tool()
        self.assertEqual(first.definition_ref, second.definition_ref)
        self.assertTrue(first.definition_ref.startswith("sha256:"))
        manifest = _manifest(first)
        self.assertEqual(manifest.manifest_ref, _manifest(second).manifest_ref)
        reservation = _reservation(manifest)
        self.assertEqual(reservation.operation_ref, _reservation(manifest).operation_ref)

        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            replace(first, input_schema_json='{ "type": "object" }')
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            replace(first, input_schema_json='{"type":"object","type":"object"}')
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            replace(reservation, arguments_json='{"query":"x", "unused":1}')
        with self.assertRaisesRegex(ValueError, "bounded text"):
            replace(reservation, arguments_json='{"x":"' + "z" * 65_536 + '"}')

    def test_manifest_versions_are_immutable_and_external_effects_stay_off(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ToolOperationStore(Path(directory) / "operations.sqlite3")
            manifest = _manifest()
            self.assertEqual(store.register_manifest(manifest), manifest.manifest_ref)
            self.assertEqual(store.register_manifest(manifest), manifest.manifest_ref)

            changed = _manifest(_tool(description="Changed under the same version."))
            with self.assertRaisesRegex(ValueError, "definition version conflicts"):
                store.register_manifest(changed)

            external = _manifest(
                replace(_tool(external_effect=True), tool_version="2.0.0")
            )
            external = replace(external, manifest_version="2.0.0")
            store.register_manifest(external)
            with self.assertRaisesRegex(PermissionError, "external-effect"):
                store.reserve(_reservation(external, call_id="tool-call:external"))

    def test_reservation_is_crash_safe_idempotent_and_rejects_call_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "operations.sqlite3"
            first_store = ToolOperationStore(database)
            manifest = _manifest()
            first_store.register_manifest(manifest)
            reservation = _reservation(manifest)

            self.assertEqual(first_store.reserve(reservation), reservation)
            self.assertEqual(first_store.reserve(reservation), reservation)

            reopened = ToolOperationStore(database)
            self.assertEqual(reopened.reservation_for_call(reservation.call_id), reservation)
            self.assertEqual(
                reopened.reservation_for_operation(reservation.operation_ref),
                reservation,
            )
            self.assertEqual(reopened.reserve(reservation), reservation)
            with self.assertRaisesRegex(ValueError, "call_id conflicts"):
                reopened.reserve(
                    replace(reservation, arguments_json='{"query":"different"}')
                )

            unknown_tool = replace(reservation, call_id="tool-call:two", tool_ref="sha256:" + "b" * 64)
            with self.assertRaisesRegex(ValueError, "not a member"):
                reopened.reserve(unknown_tool)

    def test_events_are_monotonic_and_one_terminal_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "operations.sqlite3"
            store = ToolOperationStore(database)
            manifest = _manifest()
            store.register_manifest(manifest)
            reservation = store.reserve(_reservation(manifest))
            running = _event(
                reservation.operation_ref,
                1,
                "RUNNING",
                "{}",
                "2026-09-02T12:00:00.000000Z",
            )
            terminal = _event(
                reservation.operation_ref,
                2,
                "COMPLETED",
                '{"answer":"bounded observation"}',
                "2026-09-02T12:00:01.000000Z",
            )

            with self.assertRaisesRegex(ValueError, "first event must be RUNNING"):
                store.append_event(terminal)
            self.assertEqual(store.append_event(running), running)
            self.assertEqual(store.append_event(running), running)
            self.assertEqual(store.append_event(terminal), terminal)
            self.assertEqual(store.append_event(terminal), terminal)

            with self.assertRaisesRegex(ValueError, "event sequence conflicts"):
                store.append_event(replace(terminal, status="ERROR"))
            with self.assertRaisesRegex(ValueError, "event sequence conflicts"):
                store.append_event(
                    replace(running, recorded_at_utc="2026-09-02T12:00:00.500000Z")
                )

            reopened = ToolOperationStore(database)
            self.assertEqual(
                reopened.events_for_operation(reservation.operation_ref),
                (running, terminal),
            )

    def test_terminal_time_cannot_precede_running_and_failed_append_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "operations.sqlite3"
            store = ToolOperationStore(database)
            manifest = _manifest()
            store.register_manifest(manifest)
            reservation = store.reserve(_reservation(manifest))
            running = _event(
                reservation.operation_ref,
                1,
                "RUNNING",
                "{}",
                "2026-09-02T12:00:01.000000Z",
            )
            store.append_event(running)
            invalid_terminal = _event(
                reservation.operation_ref,
                2,
                "DENIED",
                '{"reason":"permission"}',
                "2026-09-02T12:00:00.000000Z",
            )
            with self.assertRaisesRegex(ValueError, "cannot precede"):
                store.append_event(invalid_terminal)
            self.assertEqual(store.events_for_operation(reservation.operation_ref), (running,))

            valid_terminal = replace(
                invalid_terminal,
                recorded_at_utc="2026-09-02T12:00:02.000000Z",
            )
            store.append_event(valid_terminal)
            self.assertEqual(
                store.events_for_operation(reservation.operation_ref),
                (running, valid_terminal),
            )

            with sqlite3.connect(database) as connection:
                statuses = connection.execute(
                    "SELECT status FROM tool_events ORDER BY sequence"
                ).fetchall()
            self.assertEqual(statuses, [("RUNNING",), ("DENIED",)])


if __name__ == "__main__":
    unittest.main()
