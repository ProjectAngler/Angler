import os
import threading
import time
import unittest
from contextvars import ContextVar

from angler.runtime import lanes

_cv: ContextVar[str] = ContextVar("lane_test_cv", default="unset")


class LanesTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("JENNY2_LANES", None)

    def test_independent_tasks_run_at_once_and_return_by_name(self):
        os.environ["JENNY2_LANES"] = "3"
        seen = []
        def slow(tag):
            time.sleep(0.3); seen.append((tag, threading.current_thread().name)); return tag.upper()
        t0 = time.perf_counter()
        out = lanes.run_lanes({"a": lambda: slow("a"), "b": lambda: slow("b"), "c": lambda: slow("c")})
        self.assertEqual(out, {"a": "A", "b": "B", "c": "C"})
        self.assertLess(time.perf_counter() - t0, 0.75)  # ran together, not 0.9s in series
        self.assertEqual(len({name for _, name in seen}), 3)

    def test_a_failing_lane_returns_its_exception_and_others_complete(self):
        os.environ["JENNY2_LANES"] = "3"
        def boom(): raise ValueError("lane failed")
        out = lanes.run_lanes({"ok": lambda: 1, "bad": boom})
        self.assertEqual(out["ok"], 1)
        self.assertIsInstance(out["bad"], ValueError)

    def test_one_lane_is_plain_sequential_and_context_is_copied_into_lanes(self):
        os.environ["JENNY2_LANES"] = "1"
        order = []
        out = lanes.run_lanes({"x": lambda: order.append("x") or 1, "y": lambda: order.append("y") or 2})
        self.assertEqual((out, order), ({"x": 1, "y": 2}, ["x", "y"]))
        os.environ["JENNY2_LANES"] = "2"
        _cv.set("turn-42")
        out = lanes.run_lanes({"p": lambda: _cv.get(), "q": lambda: _cv.get()})
        self.assertEqual(out, {"p": "turn-42", "q": "turn-42"})


if __name__ == "__main__":
    unittest.main()
