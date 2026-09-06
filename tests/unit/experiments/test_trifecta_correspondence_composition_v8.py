import unittest

import torch

from experiments.runners import phase6_software_pipeline_reconstruction as legacy
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import trifecta_correspondence_composition_v8 as v8


class TrifectaCorrespondenceCompositionV8Tests(unittest.TestCase):
    def test_public_donor_features_have_declared_shape(self) -> None:
        controller = v19.V12ChampionPairedGraphContextController(
            legacy.SOFTWARE_PIPELINE_PROFILES["smoke"]
        )
        stream = v8._raw_partition("train", 1, 1_000_000)[0]
        state = controller.initial_state()
        for pair in stream.supports:
            state = v19.acquire_v19_public_pipeline_traces(
                controller, pair.learner, state
            ).state
        features = v8.public_donor_features(
            controller, stream.queries[0].learner, state
        )
        self.assertEqual(features.shape, (6, v8.DONOR_WIDTH))
        self.assertTrue(bool(torch.isfinite(features).all().item()))

    def test_raw_corpus_matches_declared_partition_sizes(self) -> None:
        train, development, final = v8._raw_corpus()
        self.assertEqual(len(train), 64)
        self.assertEqual(len(development), 16)
        self.assertEqual(len(final), 16)
        self.assertTrue(all(len(stream.supports) == 4 for stream in train))
        self.assertTrue(all(len(stream.queries) == 2 for stream in final))


if __name__ == "__main__":
    unittest.main()

