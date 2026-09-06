import datetime as dt
import json
import pathlib
import tempfile
import unittest

from angler.runtime import becca_gate as gate


class GateTest(unittest.TestCase):
    def test_pause_and_quiet_hold_everything_automated(self):
        with tempfile.TemporaryDirectory() as d:
            gate.PAUSE_PATH = pathlib.Path(d) / "paused.json"
            gate.QUIET_PATH = pathlib.Path(d) / "quiet.json"
            self.assertEqual(gate.held(), (False, "clear"))
            until = gate.touch_quiet(120)
            self.assertIsNotNone(gate.quiet_until())
            held, why = gate.held()
            self.assertTrue(held); self.assertIn("quiet for", why)
            gate.QUIET_PATH.write_text(json.dumps({"until": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()}))
            self.assertIsNone(gate.quiet_until())
            self.assertEqual(gate.held(), (False, "clear"))
            gate.PAUSE_PATH.write_text(json.dumps({"paused": True}))
            self.assertEqual(gate.held()[1], "Becca has scheduled work switched off")
            gate.QUIET_PATH.write_text("garbage")
            self.assertIsNone(gate.quiet_until())


if __name__ == "__main__":
    unittest.main()
