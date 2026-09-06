from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from angler.reasoning.grouped_public_relation_geometry import (
    GroupedPublicRelationGeometry,
    cosine_similarity_logits,
    supervised_contrastive_loss,
)


def _features(rows: int = 10) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20_260_831_20)
    return torch.randn(rows, 64, generator=generator, dtype=torch.float32)


def _positive_mask(rows: int = 10) -> torch.Tensor:
    groups = torch.arange(rows) // 2
    mask = groups[:, None].eq(groups[None, :])
    mask.fill_diagonal_(False)
    return mask


class GroupedPublicRelationGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20_260_831_21)
        self.model = GroupedPublicRelationGeometry()

    def test_exact_architecture_shape_and_normalization(self) -> None:
        self.assertIsInstance(self.model.input_normalization, nn.LayerNorm)
        self.assertEqual(self.model.input_normalization.normalized_shape, (64,))
        self.assertIsInstance(self.model.hidden_projection, nn.Linear)
        self.assertEqual(self.model.hidden_projection.in_features, 64)
        self.assertEqual(self.model.hidden_projection.out_features, 128)
        self.assertIsNone(self.model.hidden_projection.bias)
        self.assertIsInstance(self.model.activation, nn.SiLU)
        self.assertIsInstance(self.model.output_projection, nn.Linear)
        self.assertEqual(self.model.output_projection.in_features, 128)
        self.assertEqual(self.model.output_projection.out_features, 64)
        self.assertIsNone(self.model.output_projection.bias)

        codes = self.model(_features())
        self.assertEqual(codes.shape, (10, 64))
        self.assertEqual(codes.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(codes).all().item()))
        torch.testing.assert_close(
            torch.linalg.vector_norm(codes, dim=-1),
            torch.ones(10),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_strict_public_tensor_boundary(self) -> None:
        with self.assertRaises(TypeError):
            self.model([[0.0] * 64])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.model(_features().to(torch.float64))
        with self.assertRaises(ValueError):
            self.model(torch.randn(64, dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 63, dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.model(torch.empty(0, 64, dtype=torch.float32))
        nonfinite = _features()
        nonfinite[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(nonfinite)
        noncontiguous = torch.randn(64, 10, dtype=torch.float32).T
        self.assertEqual(noncontiguous.shape, (10, 64))
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaises(ValueError):
            self.model(noncontiguous)

    def test_row_permutation_equivariance_and_loss_invariance(self) -> None:
        features = _features()
        mask = _positive_mask()
        permutation = torch.tensor((7, 2, 9, 0, 5, 4, 1, 8, 3, 6))
        original_codes = self.model(features)
        original_loss = supervised_contrastive_loss(original_codes, mask)
        permuted_features = features[permutation].contiguous()
        permuted_mask = mask[permutation][:, permutation].contiguous()
        permuted_codes = self.model(permuted_features)
        permuted_loss = supervised_contrastive_loss(permuted_codes, permuted_mask)
        torch.testing.assert_close(
            permuted_codes,
            original_codes[permutation],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(original_loss, permuted_loss, rtol=0.0, atol=1.0e-7)

    def test_supcon_validation_and_gradient_reachability(self) -> None:
        codes = self.model(_features())
        mask = _positive_mask()
        loss = supervised_contrastive_loss(codes, mask)
        self.assertEqual(loss.shape, ())
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()))
        self.assertEqual(len(gradients), 4)
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().sum().item()), 0.0)

        diagonal = mask.clone()
        diagonal.fill_diagonal_(True)
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(codes.detach(), diagonal)
        asymmetric = mask.clone()
        asymmetric[0, 1] = False
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(codes.detach(), asymmetric)
        empty = torch.zeros_like(mask)
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(codes.detach(), empty)
        with self.assertRaises(TypeError):
            supervised_contrastive_loss(codes.detach(), mask.to(torch.int64))

    def test_metadata_blind_api_and_cosine_logits(self) -> None:
        signature = inspect.signature(self.model.forward)
        self.assertEqual(tuple(signature.parameters), ("relation_features",))
        features = _features(6)
        codes = self.model(features)
        logits = cosine_similarity_logits(codes[:2].contiguous(), codes[2:].contiguous())
        self.assertEqual(logits.shape, (2, 4))
        torch.testing.assert_close(
            logits,
            codes[:2] @ codes[2:].T / 0.10,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
