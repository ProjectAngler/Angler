"""Safe path rebasing for controlled copies of Cognee state roots."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import stat
from typing import NamedTuple
from uuid import UUID


_METADATA_RELATIVE_PATH = Path("system/databases/angler_cognee.sqlite")
_DATASET_COLUMNS = (
    "owner_id,dataset_id,vector_database_name,graph_database_name,"
    "vector_database_url,graph_database_url"
)


class CogneeCloneRebaseError(RuntimeError):
    """A controlled Cognee clone cannot be rebased without ambiguity."""


class _RebasePlan(NamedTuple):
    metadata: Path
    updates: tuple[tuple[str, str | None, str, str, str], ...]


def _real_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CogneeCloneRebaseError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CogneeCloneRebaseError(f"{label} is not a real directory")
    if resolved != path:
        raise CogneeCloneRebaseError(f"{label} is not canonical")
    return resolved


def _regular_metadata(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CogneeCloneRebaseError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CogneeCloneRebaseError(f"{label} is not a regular file")
    return path


def _sqlite_uri(path: Path, mode: str, *, immutable: bool = False) -> str:
    suffix = "&immutable=1" if immutable else ""
    return f"{path.as_uri()}?mode={mode}{suffix}"


def _read_rows(
    path: Path, *, immutable: bool = False
) -> tuple[tuple[object, ...], ...]:
    wal = Path(f"{path}-wal")
    if immutable and wal.exists() and wal.stat().st_size:
        raise CogneeCloneRebaseError(
            "source Cognee metadata has an uncheckpointed WAL"
        )
    try:
        connection = sqlite3.connect(
            _sqlite_uri(path, "ro", immutable=immutable), uri=True, timeout=5.0
        )
        try:
            check = connection.execute("PRAGMA quick_check").fetchall()
            if check != [("ok",)]:
                raise CogneeCloneRebaseError(
                    "Cognee clone metadata integrity check failed"
                )
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='dataset_database'"
            ).fetchone()
            if exists is None:
                return ()
            return tuple(
                connection.execute(
                    f"SELECT {_DATASET_COLUMNS} FROM dataset_database "
                    "ORDER BY owner_id,dataset_id"
                ).fetchall()
            )
        finally:
            connection.close()
    except CogneeCloneRebaseError:
        raise
    except sqlite3.Error as exc:
        raise CogneeCloneRebaseError(
            "Cognee clone metadata cannot be inspected"
        ) from exc


def _canonical_uuid(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise CogneeCloneRebaseError(f"{label} differs")
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise CogneeCloneRebaseError(f"{label} differs") from exc


def _local_url(
    value: object,
    *,
    expected: Path,
    label: str,
    optional: bool,
) -> bool:
    if optional and value in (None, ""):
        return False
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise CogneeCloneRebaseError(f"{label} differs")
    if Path(value).resolve(strict=False) != expected.resolve(strict=False):
        raise CogneeCloneRebaseError(f"{label} escapes its source state root")
    return True


def _storage_inventory(
    path: Path,
    *,
    label: str,
    directory: bool,
) -> dict[str, tuple[str, str, int, int]]:
    def fingerprint(metadata: os.stat_result, candidate: Path) -> tuple[str, str, int, int]:
        if stat.S_ISDIR(metadata.st_mode):
            return ("directory", "", metadata.st_dev, metadata.st_ino)
        digest = sha256()
        try:
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise CogneeCloneRebaseError(f"{label} cannot be read") from exc
        return ("file", digest.hexdigest(), metadata.st_dev, metadata.st_ino)

    try:
        root = path.lstat()
    except OSError as exc:
        raise CogneeCloneRebaseError(f"{label} is unavailable") from exc
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(root.st_mode) or not expected_type(root.st_mode):
        raise CogneeCloneRebaseError(f"{label} has the wrong storage type")
    entries = {".": fingerprint(root, path)}
    if not directory:
        return entries
    try:
        descendants = tuple(path.rglob("*"))
    except OSError as exc:
        raise CogneeCloneRebaseError(f"{label} cannot be inspected") from exc
    for item in descendants:
        try:
            metadata = item.lstat()
        except OSError as exc:
            raise CogneeCloneRebaseError(f"{label} cannot be inspected") from exc
        relative = item.relative_to(path).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise CogneeCloneRebaseError(f"{label} contains a symbolic link")
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise CogneeCloneRebaseError(f"{label} contains a special file")
        entries[relative] = fingerprint(metadata, item)
    return entries


def _require_independent_storage(
    source: Path,
    clone: Path,
    *,
    label: str,
    directory: bool,
) -> None:
    source_inventory = _storage_inventory(
        source, label=f"source {label}", directory=directory
    )
    clone_inventory = _storage_inventory(
        clone, label=f"cloned {label}", directory=directory
    )
    source_shape = {
        name: values[:2] for name, values in source_inventory.items()
    }
    clone_shape = {name: values[:2] for name, values in clone_inventory.items()}
    if source_shape != clone_shape:
        raise CogneeCloneRebaseError(f"cloned {label} differs from its source")
    source_inodes = {values[2:] for values in source_inventory.values()}
    clone_inodes = {values[2:] for values in clone_inventory.values()}
    if source_inodes & clone_inodes:
        raise CogneeCloneRebaseError(f"cloned {label} aliases its source")


def _plan_one_cognee_root(source: Path, clone: Path) -> _RebasePlan:
    source = _real_directory(source, "source Cognee state root")
    clone = _real_directory(clone, "cloned Cognee state root")
    source_database_root = _real_directory(
        source / "system" / "databases", "source Cognee database root"
    )
    clone_database_root = _real_directory(
        clone / "system" / "databases", "cloned Cognee database root"
    )
    _require_independent_storage(
        source_database_root,
        clone_database_root,
        label="Cognee database root",
        directory=True,
    )
    source_metadata = _regular_metadata(
        source / _METADATA_RELATIVE_PATH, "source Cognee metadata"
    )
    clone_metadata = _regular_metadata(
        clone / _METADATA_RELATIVE_PATH, "cloned Cognee metadata"
    )
    try:
        if os.path.samefile(source_metadata, clone_metadata):
            raise CogneeCloneRebaseError("cloned Cognee metadata aliases its source")
    except OSError as exc:
        raise CogneeCloneRebaseError(
            "Cognee metadata identity cannot be inspected"
        ) from exc

    # The source is an offline immutable input.  SQLite's ordinary read-only
    # WAL handling can still create ``-wal``/``-shm`` sidecars, invalidating
    # the source manifest.  Immutable mode prevents that write; a non-empty
    # WAL is rejected above so no committed source row can be silently missed.
    source_rows = _read_rows(source_metadata, immutable=True)
    clone_rows = _read_rows(clone_metadata)
    if clone_rows != source_rows:
        raise CogneeCloneRebaseError(
            "cloned dataset storage metadata differs from its source"
        )
    if not source_rows:
        return _RebasePlan(clone_metadata, ())

    updates: list[tuple[str, str | None, str, str, str]] = []
    for row in source_rows:
        owner_id = _canonical_uuid(row[0], "persisted storage owner id")
        dataset_id = _canonical_uuid(row[1], "persisted storage dataset id")
        vector_name = f"{dataset_id}.lance.db"
        graph_name = f"{dataset_id}.lbug"
        if row[2] != vector_name or row[3] != graph_name:
            raise CogneeCloneRebaseError("persisted dataset storage names differ")
        source_owner = _real_directory(
            source_database_root / owner_id,
            "source Cognee dataset owner root",
        )
        clone_owner = _real_directory(
            clone_database_root / owner_id,
            "cloned Cognee dataset owner root",
        )
        _local_url(
            row[4],
            expected=source_owner / vector_name,
            label="persisted vector database URL",
            optional=False,
        )
        graph_is_present = _local_url(
            row[5],
            expected=source_owner / graph_name,
            label="persisted graph database URL",
            optional=True,
        )
        _require_independent_storage(
            source_owner / vector_name,
            clone_owner / vector_name,
            label="vector database",
            directory=True,
        )
        _require_independent_storage(
            source_owner / graph_name,
            clone_owner / graph_name,
            label="graph database",
            directory=False,
        )
        graph_url = (
            str(clone_owner / graph_name)
            if graph_is_present
            else None if row[5] is None else ""
        )
        updates.append(
            (
                str(clone_owner / vector_name),
                graph_url,
                str(row[0]),
                str(row[1]),
                dataset_id,
            )
        )
    return _RebasePlan(clone_metadata, tuple(updates))


def _apply_plan(plan: _RebasePlan) -> None:
    clone_metadata = plan.metadata
    updates = plan.updates
    if not updates:
        return

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            _sqlite_uri(clone_metadata, "rw"),
            uri=True,
            isolation_level=None,
            timeout=5.0,
        )
        connection.execute("BEGIN IMMEDIATE")
        for vector_url, graph_url, owner_value, dataset_value, _ in updates:
            cursor = connection.execute(
                "UPDATE dataset_database SET vector_database_url=?,"
                "graph_database_url=? WHERE owner_id=? AND dataset_id=?",
                (vector_url, graph_url, owner_value, dataset_value),
            )
            if cursor.rowcount != 1:
                raise CogneeCloneRebaseError(
                    "cloned dataset storage row cannot be rebound exactly once"
                )
        connection.execute("COMMIT")
    except BaseException:
        if connection is not None:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()

    rebound_rows = _read_rows(clone_metadata)
    expected_urls = {
        dataset_id: (vector_url, graph_url)
        for vector_url, graph_url, _, _, dataset_id in updates
    }
    if len(rebound_rows) != len(expected_urls):
        raise CogneeCloneRebaseError("rebased dataset storage row count differs")
    for row in rebound_rows:
        dataset_id = _canonical_uuid(row[1], "rebased storage dataset id")
        if (row[4], row[5]) != expected_urls.get(dataset_id):
            raise CogneeCloneRebaseError("rebased dataset storage URL differs")


def rebase_cloned_cognee_state_roots(
    source_state_root: str | Path,
    cloned_state_root: str | Path,
) -> tuple[str, ...]:
    """Rebind every direct Cognee state copy before a worker can open it.

    Only metadata that is an exact byte-copy of source-local routes is changed.
    Unexpected, aliased, symlinked, or already-divergent metadata fails closed.
    """

    source = _real_directory(Path(source_state_root), "source application state root")
    clone = _real_directory(Path(cloned_state_root), "cloned application state root")
    if source == clone or source in clone.parents or clone in source.parents:
        raise CogneeCloneRebaseError("source and cloned state roots are not disjoint")
    _require_independent_storage(
        source,
        clone,
        label="application state root",
        directory=True,
    )

    def discover(root: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        try:
            children = tuple(root.iterdir())
        except OSError as exc:
            raise CogneeCloneRebaseError("application state root cannot be inspected") from exc
        for child in children:
            metadata = child / _METADATA_RELATIVE_PATH
            if not metadata.exists() and not metadata.is_symlink():
                continue
            if child.name in found:
                raise CogneeCloneRebaseError("Cognee state root name is duplicated")
            found[child.name] = child
        return found

    source_roots = discover(source)
    clone_roots = discover(clone)
    if source_roots.keys() != clone_roots.keys():
        raise CogneeCloneRebaseError("cloned Cognee state-root inventory differs")
    if "cognee" not in source_roots:
        raise CogneeCloneRebaseError("expected Cognee state root is absent")
    plans: list[tuple[str, _RebasePlan]] = []
    for name in sorted(source_roots):
        plans.append(
            (
                name,
                _plan_one_cognee_root(source_roots[name], clone_roots[name]),
            )
        )
    if not plans or len(dict(plans)["cognee"].updates) != 1:
        raise CogneeCloneRebaseError("expected single Cognee dataset row differs")
    for _, plan in plans:
        _apply_plan(plan)
    return tuple(name for name, plan in plans if plan.updates)


__all__ = ["CogneeCloneRebaseError", "rebase_cloned_cognee_state_roots"]
