from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.runners import parallel_episode_executor as executor


def receipt(panel: int, value: float = 1.0) -> dict[str, object]:
    return {
        "artifact_schema": "angler.parallel-episode-worker.v1",
        "semantic": {
            "panel": panel,
            "updates": 64,
            "value": value,
        },
        "execution": {"elapsed_seconds": 1.0},
    }


class ParallelEpisodeExecutorTests(unittest.TestCase):
    def test_panel_and_step_bounds(self) -> None:
        for panel in executor.PANELS:
            executor.validate_panel_steps(panel, 1)
            executor.validate_panel_steps(panel, 4_096)
        for panel in (-1, 4, True):
            with self.assertRaises(ValueError):
                executor.validate_panel_steps(panel, 1)
        for steps in (0, 4_097, True):
            with self.assertRaises(ValueError):
                executor.validate_panel_steps(0, steps)

    def test_worker_command_is_deterministic(self) -> None:
        first = executor.worker_command(2, 64, "/tmp/panel-2.json")
        second = executor.worker_command(2, 64, "/tmp/panel-2.json")
        self.assertEqual(first, second)
        self.assertEqual(first[2:4], ("-m", "experiments.runners.parallel_episode_executor"))
        self.assertEqual(first[-6:], ("--panel", "2", "--steps", "64", "--output", "/tmp/panel-2.json"))

    def test_exact_semantics_accepts_ordered_equal_receipts(self) -> None:
        sequential = tuple(receipt(panel) for panel in executor.PANELS)
        parallel = tuple(receipt(panel) for panel in executor.PANELS)
        result = executor.compare_semantics(sequential, parallel)
        self.assertTrue(result["exact_match"])
        self.assertEqual([item["panel"] for item in result["panels"]], list(executor.PANELS))

    def test_semantic_difference_fails_closed(self) -> None:
        sequential = tuple(receipt(panel) for panel in executor.PANELS)
        parallel = tuple(receipt(panel, 2.0 if panel == 3 else 1.0) for panel in executor.PANELS)
        with self.assertRaises(executor.SemanticMismatchError):
            executor.compare_semantics(sequential, parallel)

    def test_receipts_are_loaded_in_panel_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for panel in reversed(executor.PANELS):
                path = Path(directory) / f"panel-{panel}.json"
                path.write_text(json.dumps(receipt(panel)), encoding="utf-8")
                paths[panel] = path
            loaded = executor.load_worker_receipts(paths)
        self.assertEqual([item["semantic"]["panel"] for item in loaded], list(executor.PANELS))

    def test_invalid_or_misaligned_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for panel in executor.PANELS:
                path = Path(directory) / f"panel-{panel}.json"
                path.write_text(json.dumps(receipt(1 if panel == 0 else panel)), encoding="utf-8")
                paths[panel] = path
            with self.assertRaisesRegex(RuntimeError, "panel identity"):
                executor.load_worker_receipts(paths)

    def test_fallback_excludes_semantic_mismatch(self) -> None:
        self.assertTrue(executor.fallback_eligible(executor.WorkerExecutionError("worker")))
        self.assertTrue(executor.fallback_eligible(subprocess.TimeoutExpired(("x",), 1.0)))
        self.assertTrue(executor.fallback_eligible(OSError("spawn")))
        self.assertFalse(executor.fallback_eligible(executor.SemanticMismatchError("science")))
        self.assertFalse(executor.fallback_eligible(ValueError("configuration")))

    def test_output_map_requires_every_panel(self) -> None:
        with self.assertRaises(ValueError):
            executor.launch_panel_workers({0: "a"}, parallel=False)


if __name__ == "__main__":
    unittest.main()
