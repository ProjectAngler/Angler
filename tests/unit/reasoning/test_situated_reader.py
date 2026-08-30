from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from angler.memory import RecallBatch, SituatedRecall
from angler.reasoning import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    encode_situated_features,
)


def _recall(*, age: int, acquired: int, relation: str, score: float | None):
    return SituatedRecall(
        artifact_ref=f"sha256:{acquired + 1:064x}",
        text="Prior outcome-bearing evidence, not a query answer.",
        source_ref=f"sha256:{acquired + 101:064x}",
        context=(),
        visibility="LEARNER_VISIBLE",
        acquired_ordinal=acquired,
        age=age,
        landmark_relations=(("regime", relation),),
        world_valid_from=None,
        world_valid_until=None,
        world_valid_at_query=None,
        backend_score=score,
        backend_ref=None,
    )


class SituatedFeatureTests(unittest.TestCase):
    def test_coordinates_are_tensorized_without_ranking_candidates(self) -> None:
        spec = SituatedFeatureSpec(("regime",))
        batch = RecallBatch(
            items=(
                _recall(age=4, acquired=1, relation="BEFORE", score=0.8),
                _recall(age=0, acquired=5, relation="AFTER", score=None),
            )
        )
        features, mask = encode_situated_features(
            [batch], now=[5], spec=spec
        )

        self.assertEqual(features.shape, (1, 2, 12))
        self.assertTrue(mask.tolist() == [[True, True]])
        self.assertGreater(features[0, 0, 0].item(), features[0, 1, 0].item())
        self.assertEqual(features[0, 0, 8:12].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(features[0, 1, 8:12].tolist(), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(features[0, 1, 4].item(), 1.0)

    def test_padding_is_explicit_and_empty_collection_is_rejected(self) -> None:
        spec = SituatedFeatureSpec()
        first = RecallBatch(items=(_recall(age=0, acquired=0, relation="AT", score=1.0),))
        second = RecallBatch(
            items=(
                _recall(age=1, acquired=0, relation="AT", score=1.0),
                _recall(age=0, acquired=1, relation="AT", score=1.0),
            )
        )
        _, mask = encode_situated_features([first, second], now=[0, 1], spec=spec)
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        with self.assertRaisesRegex(ValueError, "candidate"):
            encode_situated_features([RecallBatch(items=())], now=[0], spec=spec)

    def test_absolute_acquisition_position_can_be_withheld_from_the_learner(self) -> None:
        batch = RecallBatch(
            items=(_recall(age=3, acquired=11, relation="BEFORE", score=0.5),)
        )
        full, _ = encode_situated_features(
            [batch], now=[14], spec=SituatedFeatureSpec(("regime",))
        )
        relative, _ = encode_situated_features(
            [batch],
            now=[14],
            spec=SituatedFeatureSpec(
                ("regime",),
                include_acquired_ordinal=False,
            ),
        )
        self.assertEqual(full.shape[-1], relative.shape[-1] + 1)
        self.assertEqual(relative[0, 0, 0].item(), full[0, 0, 0].item())
        self.assertEqual(relative[0, 0, 1].item(), full[0, 0, 2].item())


class LearnedSituatedReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.reader = LearnedSituatedMemoryReader(
            content_width=16,
            temporal_width=8,
            hidden_width=24,
            action_count=5,
        )

    def test_reader_masks_padding_and_returns_inspectable_attribution(self) -> None:
        query = torch.randn(2, 16)
        candidates = torch.randn(2, 3, 16)
        temporal = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        output = self.reader(query, candidates, temporal, mask)

        self.assertEqual(output.logits.shape, (2, 5))
        self.assertEqual(output.context.shape, (2, 24))
        self.assertTrue(torch.allclose(output.weights.sum(dim=1), torch.ones(2)))
        self.assertEqual(output.weights[0, 2].item(), 0.0)
        self.assertTrue(torch.isneginf(output.scores[0, 2]).item())

    def test_outcome_loss_reaches_temporal_and_content_paths(self) -> None:
        query = torch.randn(4, 16)
        candidates = torch.randn(4, 3, 16)
        temporal = torch.randn(4, 3, 8)
        mask = torch.ones(4, 3, dtype=torch.bool)
        target = torch.tensor([0, 1, 2, 3])

        loss = F.cross_entropy(
            self.reader(query, candidates, temporal, mask).logits,
            target,
        )
        loss.backward()
        temporal_grad = self.reader.temporal_projection[1].weight.grad
        content_grad = self.reader.content_projection[1].weight.grad
        self.assertIsNotNone(temporal_grad)
        self.assertIsNotNone(content_grad)
        self.assertGreater(temporal_grad.norm().item(), 0.0)
        self.assertGreater(content_grad.norm().item(), 0.0)

    def test_removed_temporal_coordinates_are_a_weight_preserving_lesion(self) -> None:
        query = torch.randn(2, 16)
        candidates = torch.randn(2, 3, 16)
        temporal_a = torch.randn(2, 3, 8)
        temporal_b = torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)

        removed_a = self.reader(query, candidates, torch.zeros_like(temporal_a), mask)
        removed_b = self.reader(query, candidates, torch.zeros_like(temporal_b), mask)
        self.assertTrue(torch.equal(removed_a.logits, removed_b.logits))

    def test_invalid_device_shape_or_empty_row_fails_closed(self) -> None:
        query = torch.randn(1, 16)
        candidates = torch.randn(1, 2, 16)
        temporal = torch.randn(1, 2, 8)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.reader(query, candidates, temporal, torch.zeros(1, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "temporal_features"):
            self.reader(query, candidates, torch.randn(1, 2, 7), torch.ones(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
