from __future__ import annotations

import unittest

from experiments.runners.situated_qwen_generation_v2 import (
    RESPONSE_CONSTRAINT,
    V1_RESULT_SHA256,
)


class SituatedQwenGenerationV2Tests(unittest.TestCase):
    def test_interface_change_is_narrow_and_v1_is_pinned(self) -> None:
        self.assertIn("first word", RESPONSE_CONSTRAINT)
        self.assertIn("only one of", RESPONSE_CONSTRAINT)
        self.assertEqual(len(V1_RESULT_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
