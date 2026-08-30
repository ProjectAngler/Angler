from __future__ import annotations

import unittest

import torch

from experiments.runners.situated_memory_value_v1 import ACTIONS
from experiments.runners.situated_memory_value_v2 import (
    SPEC_V2,
    _mismatched_indices,
    build_samples_v2,
    tensorize_v2,
)


class SituatedMemoryValueV2Tests(unittest.TestCase):
    def test_balanced_evidence_removes_v1_exclusion_shortcut(self) -> None:
        samples = build_samples_v2(
            128,
            seed=111,
            family_indices=(0, 1, 2),
            evaluation_surfaces=False,
        )
        for sample in samples:
            self.assertEqual(sum(" status success." in text for text in sample.candidate_texts), 2)
            self.assertEqual(sum(" status failure." in text for text in sample.candidate_texts), 4)
            for action in ACTIONS:
                self.assertGreaterEqual(
                    sum(f"procedure {action}" in text for text in sample.candidate_texts),
                    1,
                )

    def test_absolute_ordinal_is_not_learner_visible(self) -> None:
        samples = build_samples_v2(
            8,
            seed=112,
            family_indices=(0,),
            evaluation_surfaces=True,
        )
        texts = {
            text for sample in samples for text in (sample.query_text, *sample.candidate_texts)
        }
        embeddings = {
            text: torch.randn(16, generator=torch.Generator().manual_seed(index))
            for index, text in enumerate(sorted(texts))
        }
        dataset = tensorize_v2(samples, embeddings)
        self.assertEqual(dataset.live.shape[-1], SPEC_V2.width)
        self.assertEqual(SPEC_V2.width, 11)
        self.assertTrue(torch.equal(dataset.frozen[:, :, 0], torch.zeros(8, 6)))

    def test_retrieval_removal_uses_different_target_candidate_sets(self) -> None:
        targets = torch.tensor([0, 0, 1, 2, 2, 3])
        mismatched = _mismatched_indices(targets)
        self.assertTrue((targets[mismatched] != targets).all())


if __name__ == "__main__":
    unittest.main()
