from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from uuid import UUID

from angler.memory.cognee_state_clone import (
    CogneeCloneRebaseError,
    rebase_cloned_cognee_state_roots,
)


_SCHEMA = """
CREATE TABLE dataset_database (
    owner_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    vector_database_name TEXT NOT NULL,
    graph_database_name TEXT NOT NULL,
    vector_database_url TEXT,
    graph_database_url TEXT,
    PRIMARY KEY (dataset_id)
)
"""


def _seed_cognee_root(
    application_root: Path,
    name: str,
    *,
    owner_hex: str,
    dataset_hex: str,
    graph_url_present: bool,
) -> tuple[Path, tuple[object, ...]]:
    state = application_root / name
    database_root = state / "system" / "databases"
    database_root.mkdir(parents=True)
    owner = str(UUID(owner_hex))
    dataset = str(UUID(dataset_hex))
    owner_root = database_root / owner
    owner_root.mkdir()
    vector_name = f"{dataset}.lance.db"
    graph_name = f"{dataset}.lbug"
    (owner_root / vector_name).mkdir()
    (owner_root / vector_name / "segment.bin").write_bytes(b"graph")
    (owner_root / graph_name).write_bytes(b"graph")
    vector_url = str(owner_root / vector_name)
    graph_url = str(owner_root / graph_name) if graph_url_present else ""
    metadata = database_root / "angler_cognee.sqlite"
    connection = sqlite3.connect(metadata)
    try:
        connection.execute(_SCHEMA)
        connection.execute(
            "INSERT INTO dataset_database VALUES (?,?,?,?,?,?)",
            (owner_hex, dataset_hex, vector_name, graph_name, vector_url, graph_url),
        )
        connection.commit()
    finally:
        connection.close()
    return metadata, (
        owner_hex,
        dataset_hex,
        vector_name,
        graph_name,
        vector_url,
        graph_url,
    )


def _row(metadata: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(
        f"{metadata.as_uri()}?mode=ro", uri=True
    )
    try:
        return connection.execute(
            "SELECT owner_id,dataset_id,vector_database_name,graph_database_name,"
            "vector_database_url,graph_database_url FROM dataset_database"
        ).fetchone()
    finally:
        connection.close()


class CogneeStateCloneTests(unittest.TestCase):
    def test_wal_source_inspection_creates_no_sqlite_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            metadata, _ = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="0" * 32,
                dataset_hex="1" * 32,
                graph_url_present=False,
            )
            connection = sqlite3.connect(metadata)
            try:
                self.assertEqual(
                    connection.execute("PRAGMA journal_mode=WAL").fetchone(),
                    ("wal",),
                )
            finally:
                connection.close()
            wal = Path(f"{metadata}-wal")
            shm = Path(f"{metadata}-shm")
            self.assertFalse(wal.exists())
            self.assertFalse(shm.exists())

            clone = base / "clone"
            shutil.copytree(source, clone)

            self.assertEqual(
                rebase_cloned_cognee_state_roots(source, clone),
                ("cognee",),
            )

            self.assertFalse(wal.exists())
            self.assertFalse(shm.exists())

    def test_rejects_nonempty_source_wal_before_immutable_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            metadata, _ = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="2" * 32,
                dataset_hex="3" * 32,
                graph_url_present=False,
            )
            Path(f"{metadata}-wal").write_bytes(b"uncheckpointed")
            clone = base / "clone"
            shutil.copytree(source, clone)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "uncheckpointed WAL"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_rebases_all_direct_cognee_roots_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            first_db, first_source_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="1" * 32,
                dataset_hex="2" * 32,
                graph_url_present=True,
            )
            second_db, second_source_row = _seed_cognee_root(
                source,
                "cognee-capabilities",
                owner_hex="3" * 32,
                dataset_hex="4" * 32,
                graph_url_present=False,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)

            rebound = rebase_cloned_cognee_state_roots(source, clone)

            self.assertEqual(rebound, ("cognee", "cognee-capabilities"))
            self.assertEqual(_row(first_db), first_source_row)
            self.assertEqual(_row(second_db), second_source_row)
            first_clone_row = _row(
                clone / "cognee/system/databases/angler_cognee.sqlite"
            )
            second_clone_row = _row(
                clone
                / "cognee-capabilities/system/databases/angler_cognee.sqlite"
            )
            first_owner = str(UUID(str(first_clone_row[0])))
            second_owner = str(UUID(str(second_clone_row[0])))
            self.assertEqual(
                first_clone_row[4],
                str(
                    clone
                    / "cognee/system/databases"
                    / first_owner
                    / str(first_clone_row[2])
                ),
            )
            self.assertEqual(
                first_clone_row[5],
                str(
                    clone
                    / "cognee/system/databases"
                    / first_owner
                    / str(first_clone_row[3])
                ),
            )
            self.assertEqual(
                second_clone_row[4],
                str(
                    clone
                    / "cognee-capabilities/system/databases"
                    / second_owner
                    / str(second_clone_row[2])
                ),
            )
            self.assertEqual(second_clone_row[5], "")

    def test_rejects_nonlocal_source_url_without_modifying_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            metadata, _ = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="5" * 32,
                dataset_hex="6" * 32,
                graph_url_present=True,
            )
            connection = sqlite3.connect(metadata)
            try:
                connection.execute(
                    "UPDATE dataset_database SET vector_database_url=?",
                    (str(base / "outside.lance.db"),),
                )
                connection.commit()
            finally:
                connection.close()
            clone = base / "clone"
            shutil.copytree(source, clone)
            clone_metadata = (
                clone / "cognee/system/databases/angler_cognee.sqlite"
            )
            before = _row(clone_metadata)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "escapes its source state root"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

            self.assertEqual(_row(clone_metadata), before)

    def test_rejects_divergent_clone_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            _seed_cognee_root(
                source,
                "cognee",
                owner_hex="7" * 32,
                dataset_hex="8" * 32,
                graph_url_present=False,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            clone_metadata = (
                clone / "cognee/system/databases/angler_cognee.sqlite"
            )
            connection = sqlite3.connect(clone_metadata)
            try:
                connection.execute(
                    "UPDATE dataset_database SET graph_database_url='changed'"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "differs from its source"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_prevalidates_every_root_before_rebinding_any_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            first_metadata, first_source_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="9" * 32,
                dataset_hex="a" * 32,
                graph_url_present=False,
            )
            _seed_cognee_root(
                source,
                "cognee-capabilities",
                owner_hex="b" * 32,
                dataset_hex="c" * 32,
                graph_url_present=False,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            second_clone_metadata = (
                clone
                / "cognee-capabilities/system/databases/angler_cognee.sqlite"
            )
            connection = sqlite3.connect(second_clone_metadata)
            try:
                connection.execute(
                    "UPDATE dataset_database SET vector_database_url='changed'"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "differs from its source"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

            self.assertEqual(
                _row(clone / first_metadata.relative_to(source)),
                first_source_row,
            )

    def test_rejects_symlinked_vector_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            metadata, source_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="d" * 32,
                dataset_hex="e" * 32,
                graph_url_present=False,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            source_vector = Path(str(source_row[4]))
            clone_vector = clone / source_vector.relative_to(source)
            shutil.rmtree(clone_vector)
            clone_vector.symlink_to(source_vector, target_is_directory=True)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "symbolic link"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

            self.assertEqual(_row(metadata), source_row)

    def test_rejects_hardlinked_graph_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            _, source_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="f" * 32,
                dataset_hex="0" * 32,
                graph_url_present=True,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            source_graph = Path(str(source_row[5]))
            clone_graph = clone / source_graph.relative_to(source)
            clone_graph.unlink()
            os.link(source_graph, clone_graph)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "aliases its source"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_rejects_cross_root_hardlink_in_application_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            _, primary_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="1" * 32,
                dataset_hex="3" * 32,
                graph_url_present=True,
            )
            _, secondary_row = _seed_cognee_root(
                source,
                "cognee-capabilities",
                owner_hex="5" * 32,
                dataset_hex="7" * 32,
                graph_url_present=True,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            source_graph = Path(str(secondary_row[5]))
            source_vector = Path(str(primary_row[4]))
            clone_segment = (
                clone / source_vector.relative_to(source) / "segment.bin"
            )
            clone_segment.unlink()
            os.link(source_graph, clone_segment)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "aliases its source"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_rejects_equal_length_storage_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            _, source_row = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="2" * 32,
                dataset_hex="4" * 32,
                graph_url_present=False,
            )
            clone = base / "clone"
            shutil.copytree(source, clone)
            source_vector = Path(str(source_row[4]))
            clone_segment = (
                clone / source_vector.relative_to(source) / "segment.bin"
            )
            clone_segment.write_bytes(b"other")

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "differs from its source"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_requires_runtime_cognee_dataset_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            database_root = source / "cognee/system/databases"
            database_root.mkdir(parents=True)
            connection = sqlite3.connect(database_root / "angler_cognee.sqlite")
            try:
                connection.execute(_SCHEMA)
                connection.commit()
            finally:
                connection.close()
            clone = base / "clone"
            shutil.copytree(source, clone)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "expected single Cognee dataset row differs"
            ):
                rebase_cloned_cognee_state_roots(source, clone)

    def test_rejects_multiple_runtime_cognee_dataset_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            source.mkdir()
            metadata, _ = _seed_cognee_root(
                source,
                "cognee",
                owner_hex="5" * 32,
                dataset_hex="6" * 32,
                graph_url_present=False,
            )
            database_root = metadata.parent
            owner_hex = "7" * 32
            dataset_hex = "8" * 32
            owner = str(UUID(owner_hex))
            dataset = str(UUID(dataset_hex))
            owner_root = database_root / owner
            owner_root.mkdir()
            vector_name = f"{dataset}.lance.db"
            graph_name = f"{dataset}.lbug"
            (owner_root / vector_name).mkdir()
            (owner_root / vector_name / "segment.bin").write_bytes(b"graph")
            (owner_root / graph_name).write_bytes(b"graph")
            connection = sqlite3.connect(metadata)
            try:
                connection.execute(
                    "INSERT INTO dataset_database VALUES (?,?,?,?,?,?)",
                    (
                        owner_hex,
                        dataset_hex,
                        vector_name,
                        graph_name,
                        str(owner_root / vector_name),
                        "",
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            clone = base / "clone"
            shutil.copytree(source, clone)

            with self.assertRaisesRegex(
                CogneeCloneRebaseError, "expected single Cognee dataset row differs"
            ):
                rebase_cloned_cognee_state_roots(source, clone)


if __name__ == "__main__":
    unittest.main()
