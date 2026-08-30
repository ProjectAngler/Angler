from __future__ import annotations

import unittest

from experiments.runners.scalable_procedural_core_benchmark_v1 import IDENTITY, SEED


class ScalableProceduralCoreBenchmarkV1Tests(unittest.TestCase):
    def test_identity_and_seed_are_frozen(self) -> None:
        self.assertEqual(IDENTITY, "angler.scalable-procedural-core-benchmark.v1")
        self.assertEqual(SEED, 20260843)


if __name__ == "__main__":
    unittest.main()
