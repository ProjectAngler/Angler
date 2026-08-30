from __future__ import annotations

import unittest

import torch

from experiments.runners.situated_memory_value_v1 import (
    ACTIONS,
    SPEC,
    build_samples,
    tensorize,
)


class SituatedMemoryValueDatasetTests(unittest.TestCase):
    def test_target_varies_and_is_not_exposed_by_candidate_order(self) -> None:
        samples = build_samples(
            128,
            seed=91,
            family_indices=(0, 1, 2),
            evaluation_surfaces=False,
        )
        self.assertEqual({item.target_action for item in samples}, set(range(len(ACTIONS))))
        self.assertGreater(len({item.target_candidate for item in samples}), 3)
        for item in samples:
            self.assertNotIn(ACTIONS[item.target_action], item.query_text)
            self.assertEqual(sum(" succeeded " in text for text in item.candidate_texts), 4)
            self.assertEqual(sum(" failed " in text for text in item.candidate_texts), 2)

    def test_live_origin_and_frozen_origin_differ_but_naive_matches_live(self) -> None:
        samples = build_samples(
            16,
            seed=92,
            family_indices=(0,),
            evaluation_surfaces=True,
        )
        texts = {
            text
            for item in samples
            for text in (item.query_text, *item.candidate_texts)
        }
        embeddings = {
            text: torch.randn(32, generator=torch.Generator().manual_seed(index))
            for index, text in enumerate(sorted(texts))
        }
        dataset = tensorize(samples, embeddings)
        self.assertEqual(dataset.live.shape, (16, 6, SPEC.width))
        self.assertFalse(torch.equal(dataset.live, dataset.frozen))
        self.assertTrue(dataset.mask.all())
        self.assertTrue((dataset.target_mask.sum(dim=1) == 5).all())
        self.assertGreaterEqual(
            dataset.naive_inspected,
            2.0 * dataset.maintained_inspected,
        )


if __name__ == "__main__":
    unittest.main()
