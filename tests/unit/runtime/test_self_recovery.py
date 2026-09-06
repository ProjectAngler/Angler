import json
import pathlib
import sqlite3
import tempfile
import unittest

from angler.runtime import self_recovery as sr


def _make_db(path, rows=3):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t(x)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    con.commit(); con.close()


class CanonicalGuardTest(unittest.TestCase):
    def test_good_record_is_left_alone_and_damaged_record_is_restored_from_newest_good_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); db = root / "jenny2.sqlite3"; snaps = root / "snaps"
            _make_db(db)
            self.assertEqual(sr.guard_canonical(db, snapshot_root=snaps), "ok")
            (snaps / "20260906T100000Z").mkdir(parents=True); _make_db(snaps / "20260906T100000Z" / "jenny2.sqlite3", rows=5)
            (snaps / "20260906T110000Z").mkdir(); (snaps / "20260906T110000Z" / "jenny2.sqlite3").write_bytes(b"garbage")
            db.write_bytes(b"not a database at all")
            result = sr.guard_canonical(db, snapshot_root=snaps)
            self.assertTrue(result.startswith("restored:") and result.endswith("20260906T100000Z/jenny2.sqlite3"))
            self.assertTrue(sr.canonical_ok(db))
            self.assertEqual(sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM t").fetchone()[0], 5)
            self.assertEqual(len(list(root.glob("jenny2.sqlite3.damaged-*"))), 1)
            db.write_bytes(b"broken again")
            import shutil; shutil.rmtree(snaps)
            self.assertEqual(sr.guard_canonical(db, snapshot_root=snaps), "damaged-unrecoverable")

    def test_snapshot_is_online_verified_and_rotated(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); db = root / "jenny2.sqlite3"; snaps = root / "snaps"; diary = root / "diary"
            _make_db(db); diary.mkdir(); (diary / "sealed.txt").write_text("private")
            for _ in range(3):
                folder = sr.snapshot_canonical(db, snapshot_root=snaps, keep=2, extra_dirs=(diary,))
                self.assertTrue((folder / "jenny2.sqlite3").exists()); self.assertTrue((folder / "diary" / "sealed.txt").exists())
                import time; time.sleep(1.05)
            self.assertEqual(len([p for p in snaps.iterdir() if p.is_dir()]), 2)


class MemoryRebuildTest(unittest.TestCase):
    def test_request_consume_and_finish_with_a_bounded_attempt_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "cognee").mkdir(); (root / "cognee" / "system").mkdir(); (root / "cognee-capabilities").mkdir()
            self.assertFalse(sr.consume_memory_rebuild_request(root))
            self.assertEqual(sr.request_memory_rebuild(root, "COGNEE_OPERATION_FAILED"), 1)
            self.assertEqual(sr.request_memory_rebuild(root, "again"), 2)
            self.assertTrue(sr.consume_memory_rebuild_request(root))
            self.assertFalse((root / "cognee").exists())
            self.assertEqual(len(list(root.glob("cognee.damaged-*"))), 1)
            self.assertEqual(len(list(root.glob("cognee-capabilities.damaged-*"))), 1)
            sr.finish_memory_rebuild(root, cleared=12)
            self.assertFalse((root / sr.REBUILD_REQUEST_NAME).exists())
            for _ in range(sr.MAX_REBUILD_ATTEMPTS + 1):
                sr.request_memory_rebuild(root, "x")
            self.assertFalse(sr.consume_memory_rebuild_request(root))
            self.assertTrue((root / sr.REBUILD_REQUEST_NAME).exists())


class RebuildProjectionsTest(unittest.TestCase):
    def test_supervisor_clears_projection_marks(self):
        from angler.runtime.persistent_autonomy import PersistentAutonomySupervisor
        method = getattr(PersistentAutonomySupervisor, "rebuild_projections", None)
        self.assertTrue(callable(method))
        with tempfile.TemporaryDirectory() as d:
            db = pathlib.Path(d) / "s.sqlite3"
            con = sqlite3.connect(str(db))
            con.execute("CREATE TABLE projection_outbox(episode_ref TEXT PRIMARY KEY, event_ref TEXT, ordinal INTEGER, payload_json BLOB, backend_ref TEXT)")
            con.executemany("INSERT INTO projection_outbox VALUES (?,?,?,?,?)", [("e1","v1",1,b"{}","b"),("e2","v2",2,b"{}",None),("e3","v3",3,b"{}","c")])
            con.commit(); con.close()
            class _Shell:
                def _connect(self_inner):
                    return sqlite3.connect(str(db))
            self.assertEqual(PersistentAutonomySupervisor.rebuild_projections(_Shell()), 2)
            self.assertEqual(sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM projection_outbox WHERE backend_ref IS NULL").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
