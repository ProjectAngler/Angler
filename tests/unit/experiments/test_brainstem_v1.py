"""Brainstem v1: the senses are code, the judgment is from her criteria, and it
never wakes her while a person is with her or her cycle is paused."""
import importlib.util
import json
import os
import pathlib
import tempfile
import time
import unittest

SPEC = importlib.util.spec_from_file_location(
    "brainstem_v1",
    pathlib.Path(__file__).resolve().parents[3] / "experiments" / "brainstem" / "brainstem_v1.py",
)
brainstem = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brainstem)


class CriteriaTest(unittest.TestCase):
    def test_her_newest_public_criteria_work_wins_and_private_works_are_never_read(self):
        payload = {"authored_artifacts": [
            {"artifact": {"title": "What should wake me", "kind": "note", "body": "old"}, "moving_origin_ordinal": 10, "artifact_ref": "sha256:a"},
            {"artifact": {"title": "what should wake me", "kind": "standard", "body": "Wake me for X.\nWatch: /tmp/somewhere"}, "moving_origin_ordinal": 20, "artifact_ref": "sha256:b"},
            {"artifact": {"title": "What should wake me", "kind": "private note", "body": "SECRET"}, "moving_origin_ordinal": 30, "artifact_ref": "sha256:c"},
        ]}
        body, ref = brainstem.her_criteria(payload)
        self.assertEqual(ref, "sha256:b")
        self.assertNotIn("SECRET", body)
        self.assertEqual(brainstem.watched_paths(body), [pathlib.Path("/tmp/somewhere")])
        body, ref = brainstem.her_criteria({"authored_artifacts": []})
        self.assertEqual(ref, "default")
        self.assertIn("Default criteria", body)


class SensesTest(unittest.TestCase):
    def test_a_stranded_undertaking_is_sensed_once_after_quiet_time(self):
        senses = brainstem.Senses()
        payload = {"follow_through": {"turn_complete": False, "moving_origin_ordinal": 5, "undertaking": "Retire the work.", "why": "I said so."}}
        brainstem.STRANDED_AFTER_SECONDS = 0.05
        self.assertEqual(senses.sense(payload, 5, ""), [])
        time.sleep(0.06)
        signals = senses.sense(payload, 5, "")
        self.assertEqual([s["sense"] for s in signals], ["stranded_undertaking"])
        self.assertEqual(signals[0]["undertaking"], "Retire the work.")
        senses.stranded_reported_for = 5
        self.assertEqual(senses.sense(payload, 5, ""), [])
        self.assertEqual(senses.sense({"follow_through": {"turn_complete": True, "moving_origin_ordinal": 6}}, 6, ""), [])

    def test_watched_paths_and_quarantine_are_sensed_as_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            watched = pathlib.Path(directory) / "watched"
            watched.mkdir()
            (watched / "a.txt").write_text("1")
            senses = brainstem.Senses()
            criteria = f"Watch: {watched}"
            self.assertEqual(senses.sense({}, 1, criteria), [])
            time.sleep(0.02)
            (watched / "b.txt").write_text("2")
            os.utime(watched / "b.txt", (time.time() + 5, time.time() + 5))
            signals = senses.sense({}, 1, criteria)
            self.assertEqual([s["sense"] for s in signals], ["watched_path_change"])
            quarantine = pathlib.Path(directory) / "q"
            quarantine.mkdir()
            (quarantine / "1-abc.json").write_text("{}")
            (quarantine / "1-abc.json.reason.txt").write_text("ValueError: shape")
            brainstem.QUARANTINE_DIR = quarantine
            signals = senses.sense({}, 1, "")
            self.assertEqual(signals[0]["sense"], "quarantined_effect")
            self.assertEqual(signals[0]["reason"], "ValueError: shape")
            self.assertEqual(senses.sense({}, 1, ""), [])


class MayWakeTest(unittest.TestCase):
    def test_never_while_a_person_is_with_her_paused_or_in_flight(self):
        with tempfile.TemporaryDirectory() as directory:
            pause = pathlib.Path(directory) / "paused.json"
            brainstem.PAUSE_PATH = pause
            self.assertEqual(brainstem.may_wake({}), (False, "her runtime is not answering"))
            self.assertEqual(brainstem.may_wake({"door": {"busy": True, "label": "Becca is with her"}, "activity": {}})[0], False)
            self.assertEqual(brainstem.may_wake({"door": {"privacy": "HER_TIME"}, "activity": {}})[0], False)
            self.assertEqual(brainstem.may_wake({"door": {"privacy": "OPEN"}, "activity": {"phase": "EXECUTE"}})[0], False)
            self.assertEqual(brainstem.may_wake({"door": {"privacy": "OPEN"}, "activity": {"phase": "ERROR"}}), (True, "clear"))
            pause.write_text(json.dumps({"paused": True}))
            self.assertEqual(brainstem.may_wake({"door": {"privacy": "OPEN"}, "activity": {}}), (False, "Becca paused her cycle"))


if __name__ == "__main__":
    unittest.main()
