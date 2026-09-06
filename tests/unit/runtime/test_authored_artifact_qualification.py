import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from run_authored_artifact_qualification import (
    ArmCapture,
    _artifact_gates,
    _quiescent_gates,
    _rearm_autonomy_wake,
)


class AuthoredArtifactQualificationTests(unittest.TestCase):
    def test_rearm_deletes_only_mechanical_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE supervisor_state(
                      singleton INTEGER PRIMARY KEY, state_ref TEXT, state_blob BLOB,
                      scheduler_enabled INTEGER, moving_origin_ordinal INTEGER,
                      last_event_ref TEXT, pending_json BLOB, shadow_json BLOB,
                      revision INTEGER);
                    CREATE TABLE autonomy_wake_checkpoint(
                      singleton INTEGER PRIMARY KEY, state_ref TEXT,
                      last_event_ref TEXT, checkpoint_ref TEXT);
                    INSERT INTO supervisor_state VALUES
                      (1, 'sha256:state', X'7b7d', 1, 7, 'sha256:event', NULL, NULL, 11);
                    INSERT INTO autonomy_wake_checkpoint VALUES
                      (1, 'sha256:state', 'sha256:event', 'sha256:checkpoint');
                    """
                )
                connection.commit()
            before = database.read_bytes()
            _rearm_autonomy_wake(database)
            after = database.read_bytes()
            self.assertNotEqual(before, after)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM autonomy_wake_checkpoint").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute("SELECT state_blob,moving_origin_ordinal,revision FROM supervisor_state").fetchone(),
                    (b"{}", 7, 11),
                )

    def test_quiescent_requires_zero_semantic_mutation(self):
        arm = self._arm(status="QUIESCENT")
        self.assertTrue(all(_quiescent_gates(arm).values()))
        changed = self._arm(status="QUIESCENT", after_state=b'{"changed":true}')
        self.assertFalse(_quiescent_gates(changed)["canonical_state_unchanged"])

    def test_artifact_pass_is_conjunctive_and_content_addressed(self):
        action = {
            "version": 1,
            "kind": "open reflection",
            "title": "A model-selected subject",
            "body": "A bounded work selected and written by the model without a harness topic.",
            "purpose": "Keep a revisitable private artifact.",
            "evidence_refs": ["sha256:" + "1" * 64],
            "source_episode_refs": ["sha256:" + "2" * 64],
            "supersedes_ref": None,
        }
        from angler.runtime.authored_artifact import AuthoredArtifactAction
        parsed = AuthoredArtifactAction.from_json(
            json.dumps(action, separators=(",", ":"), sort_keys=True)
        )
        entry = {"artifact": action, "artifact_ref": parsed.artifact_ref, "entry_ref": "sha256:" + "3" * 64}
        arm = self._arm(
            status="COMMITTED",
            affordance_id="internal.authored-artifact",
            episode_ref="sha256:" + "4" * 64,
            after_state=json.dumps({"authored_artifacts": [entry]}).encode(),
            after_artifacts=(entry,),
            after_ordinal=8,
            episodes_after=(("sha256:" + "4" * 64, "q:wake", 8, "sha256:" + "5" * 64),),
            projections_after=(("sha256:" + "4" * 64, "sha256:" + "5" * 64, 8, "sha256:" + "6" * 64),),
            episode={
                "receipt": {
                    "consequence": [],
                    "observable_consequence": {
                        "source_kind": "WORLD",
                        "source_ref": __import__("angler.runtime.authored_artifact", fromlist=["AUTHORED_ARTIFACT_SOURCE_REF"]).AUTHORED_ARTIFACT_SOURCE_REF,
                        "artifact_refs": [parsed.artifact_ref],
                    },
                }
            },
        )
        gates, artifact = _artifact_gates(arm)
        self.assertTrue(all(gates.values()), gates)
        self.assertEqual(artifact["artifact_ref"], parsed.artifact_ref)

    @staticmethod
    def _arm(
        *,
        status="QUIESCENT",
        affordance_id=None,
        episode_ref=None,
        after_state=b"{}",
        after_artifacts=(),
        after_ordinal=7,
        episodes_after=(),
        projections_after=(),
        episode=None,
    ):
        base_status = {
            "external_effects_enabled": False,
            "last_event_ref": None,
            "moving_origin_ordinal": 7,
            "pending_operation": False,
            "pending_projections": 0,
            "scheduler_enabled": True,
            "state_ref": "sha256:" + "0" * 64,
        }
        after_status = dict(base_status, moving_origin_ordinal=after_ordinal)
        return ArmCapture(
            name="test",
            result={
                "status": status,
                "selected_affordance_id": affordance_id,
                "episode_ref": episode_ref,
            },
            before_status=base_status,
            after_status=after_status,
            before_state=b"{}",
            after_state=after_state,
            before_db={"episodes": (), "projections": ()},
            after_db={"episodes": episodes_after, "projections": projections_after},
            before_credit={name: None for name in ("affordance_utility", "memory_utility", "affordance_outcome_profiles", "capability_evidence", "capability_use_evidence")},
            after_credit={name: None for name in ("affordance_utility", "memory_utility", "affordance_outcome_profiles", "capability_evidence", "capability_use_evidence")},
            before_artifacts=(),
            after_artifacts=after_artifacts,
            episode=episode,
            catalog=(),
        )


if __name__ == "__main__":
    unittest.main()
