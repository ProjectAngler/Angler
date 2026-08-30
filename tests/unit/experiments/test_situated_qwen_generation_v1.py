from __future__ import annotations

import unittest

from experiments.runners.situated_qwen_generation_v1 import parse_action


class SituatedQwenGenerationTests(unittest.TestCase):
    def test_declared_action_parser_accepts_one_unambiguous_name(self) -> None:
        self.assertEqual(parse_action("violet"), "violet")
        self.assertEqual(parse_action("The answer is Cobalt."), "cobalt")
        self.assertIsNone(parse_action("amber or jade"))
        self.assertIsNone(parse_action("unknown"))


if __name__ == "__main__":
    unittest.main()
