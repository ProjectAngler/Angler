from __future__ import annotations

import unittest

from experiments.runners.scaled_procedural_software_v2 import (
    IDENTITY,
    V1_CHECKPOINT_SHA256,
    V1_RESULT_SHA256,
    build_prompt_v2,
)


class ScaledProceduralSoftwareV2Tests(unittest.TestCase):
    def test_identity_and_parent_hashes_are_frozen(self) -> None:
        self.assertEqual(IDENTITY, "angler.scaled-procedural-software-reconstruction.v2-publication")
        self.assertEqual(len(V1_RESULT_SHA256), 64)
        self.assertEqual(len(V1_CHECKPOINT_SHA256), 64)

    def test_answer_boundary_is_literal_final_line(self) -> None:
        prompt = build_prompt_v2("task", ("evidence",))
        self.assertTrue(prompt.endswith("\nanswer="))
        self.assertLess(prompt.index("evidence"), prompt.index("\nanswer="))


if __name__ == "__main__":
    unittest.main()
