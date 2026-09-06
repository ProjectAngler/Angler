import unittest

import torch

from angler.reasoning import DonorCandidateFusion


class DonorCandidateFusionTests(unittest.TestCase):
    def test_shape_gradient_and_removal_identity(self) -> None:
        module = DonorCandidateFusion(
            semantic_width=32,
            donor_width=13,
            hidden_width=24,
        )
        semantic = torch.randn(2, 6, 32)
        donor = torch.randn(2, 6, 13)
        removed = module(semantic, donor, include_donor=False)
        self.assertTrue(torch.equal(removed, semantic))
        output = module(semantic, donor)
        self.assertEqual(output.shape, semantic.shape)
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in module.parameters()
            )
        )

    def test_rejects_misaligned_candidates(self) -> None:
        module = DonorCandidateFusion(semantic_width=8, donor_width=5)
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 6, 8), torch.zeros(1, 5, 5))


if __name__ == "__main__":
    unittest.main()
