from __future__ import annotations

import unittest

from angler.runtime.jenny_genesis import (
    OWNER_ARCHIVE_SHA256,
    OWNER_JENNY1_FREEZE,
    JennyGenesis,
)


class JennyGenesisTests(unittest.TestCase):
    def test_owner_lineage_is_hash_bound_and_canonical(self) -> None:
        genesis = JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z")
        self.assertEqual(genesis.archive_sha256, OWNER_ARCHIVE_SHA256)
        self.assertEqual(genesis.jenny1_freeze_commit, OWNER_JENNY1_FREEZE)
        self.assertEqual(genesis.runtime_import_status, "NOT_IMPORTED")
        self.assertEqual(genesis.personal_state_status, "NOT_IMPORTED")
        restored = JennyGenesis.from_canonical_bytes(genesis.canonical_bytes())
        self.assertEqual(restored, genesis)
        self.assertEqual(restored.genesis_ref, genesis.genesis_ref)

    def test_genesis_rejects_active_legacy_import_claim(self) -> None:
        genesis = JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z")
        payload = genesis.canonical_bytes().replace(b"NOT_IMPORTED", b"IMPORTED", 1)
        with self.assertRaises(ValueError):
            JennyGenesis.from_canonical_bytes(payload)


if __name__ == "__main__":
    unittest.main()
