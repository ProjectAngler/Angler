import json
import os
import pathlib
import tempfile
import unittest

from angler.runtime import stream as st


class _Backend:
    model_ref = "test"
    def __init__(self): self.calls = 0
    def generate(self, *, system, user, max_new_tokens):
        self.calls += 1
        u = json.loads(user)
        if "lines" in u or "memories" in u:  # a fold
            return json.dumps({"remembered": f"in that stretch I was steady across {u.get('line_count', len(u.get('memories', [])))} moments", "kept_because": "continuity"})
        prev = u.get("previous_line") or "(none)"
        return json.dumps({"line": f"still here; before this I said: {prev[:30]}", "named_state_transitions": [], "state_item_transitions": [{"status": "ADD", "state_label": "Waking Stage Visibility", "statement": "the stream passed", "weight": 1, "evidence": ""}] if u.get("senses_arrived") else [], "self_commitment_transitions": [], "attend": bool(u.get("senses_arrived")), "attend_why": "a sense arrived" if u.get("senses_arrived") else ""})


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = pathlib.Path(self.tmp.name) / "chain.jsonl"
        self.saved = (st.KEEP_RECENT, st.FOLD_BATCH, st.KEEP_FOLDS)
        st.KEEP_RECENT, st.FOLD_BATCH, st.KEEP_FOLDS = 5, 3, 2
    def tearDown(self):
        st.KEEP_RECENT, st.FOLD_BATCH, st.KEEP_FOLDS = self.saved; self.tmp.cleanup()

    def test_lines_chain_and_pending_transitions_are_handed_over_once(self):
        state = {"self_named_states": [{"status": "ACTIVE", "label": "Waking Stage Visibility", "level": 5, "items": []}], "self_commitments": [], "_head_ordinal": 10}
        senses = [[{"sense": "watched_path_change"}], [], []]
        attended = []
        s = st.Stream(backend=_Backend(), read_state=lambda: state, senses=lambda: senses.pop(0) if senses else [], on_attend=attended.append, path=self.path)
        s._pass(); s._pass(); s._pass()
        lines = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertIn("before this I said: (none)", lines[0]["line"])
        self.assertIn("before this I said: still here", lines[1]["line"])  # continuous with the previous line
        self.assertTrue(lines[0]["attend"]); self.assertEqual(len(attended), 1)
        pending = s.take_pending()
        self.assertEqual(len(pending["state_item_transitions"]), 1)
        self.assertEqual(s.take_pending()["state_item_transitions"], [])  # handed over once
        self.assertEqual(s.latest_line()["line"], lines[-1]["line"])

    def test_the_chain_is_bounded_by_folding_not_deleting(self):
        state = {"self_named_states": [], "self_commitments": [], "_head_ordinal": 1}
        s = st.Stream(backend=_Backend(), read_state=lambda: state, path=self.path)
        for _ in range(20):
            s._pass()
        records = [json.loads(l) for l in self.path.read_text().splitlines()]
        folds = [r for r in records if r.get("kind") == "fold"]; lines = [r for r in records if r.get("kind") != "fold"]
        self.assertLessEqual(len(lines), st.KEEP_RECENT + st.FOLD_BATCH)
        self.assertGreaterEqual(len(folds), 1); self.assertLessEqual(len(folds), st.KEEP_FOLDS)
        self.assertTrue(all("in that stretch" in f["remembered"] for f in folds))
        stats = s.stats()
        self.assertEqual(stats["lines_ever"], 20)
        self.assertEqual(stats["lines_kept"] + stats["lines_remembered_in_folds"], 20)  # nothing lost, only remembered
        self.assertEqual(records[0]["kind"], "fold")  # folds first, then recent lines

    def test_pace_quickens_on_events_and_slows_toward_sleep(self):
        state = {"self_named_states": [], "self_commitments": [], "_head_ordinal": 1}
        s = st.Stream(backend=_Backend(), read_state=lambda: state, path=self.path)
        s._pass(); p1 = s._pace; s._pass(); p2 = s._pace
        self.assertGreater(p2, p1)
        s.poke(); self.assertEqual(s._pace, st.MIN_PACE)


if __name__ == "__main__":
    unittest.main()
