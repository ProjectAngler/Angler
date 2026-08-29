from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import ModuleType
import unittest

import torch
from torch import nn

from experiments.evaluators import phase6_v11_representation_overlap as overlap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class _Contract:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.procedure_start = nn.Parameter(torch.zeros(32))
        self.relation_comparator = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 1, bias=False),
        )
        self._calls = 0

    def _factorized_relation_embeddings(self, components, reference):
        del reference
        self._calls += 1
        base = torch.arange(len(components) * 32, dtype=torch.float32).reshape(
            len(components), 32
        )
        relation = base / 100.0 + self._calls
        context = relation + 0.25
        return context, relation

    def _relation_evidence_read(
        self,
        query_context_codes,
        query_relation_codes,
        stored_contexts,
        stored_relations,
    ):
        del query_context_codes, stored_contexts
        query = query_relation_codes[:, None, :]
        stored = stored_relations[None, :, :]
        features = torch.cat(
            (query * stored, (query - stored).abs(), (query + stored) * 0.5),
            dim=-1,
        )
        logits = self.relation_comparator(features).squeeze(-1)
        return logits.mean(dim=1), logits.softmax(dim=1), logits.new_zeros(3), logits


def _fake_source(*, fail: bool = False) -> ModuleType:
    source = ModuleType("frozen_v11_for_test")
    source.SoftwarePipelineController = _FakeController
    source._public_static_contract_fields = lambda component: (component.key,)

    def public_rows(controller, stream):
        del stream
        encoded = []
        for _ in range(4):
            components = (_Contract("root"), _Contract("twin"), _Contract("twin"))
            encoded.append(
                controller._factorized_relation_embeddings(
                    components,
                    controller.procedure_start,
                )
            )
        for index, (contexts, relations) in enumerate(encoded):
            stored = torch.roll(relations, shifts=index, dims=0)
            controller._relation_evidence_read(contexts, relations, contexts, stored)
            if fail and index == 0:
                raise InjectedCaptureError("injected public-row error")
            controller._relation_evidence_read(contexts, relations, contexts, stored + 0.1)
        return (object(), object(), object(), object())

    source.public_relation_credit_rows = public_rows
    return source


class InjectedCaptureError(RuntimeError):
    pass


def _structured_matrix() -> torch.Tensor:
    matrix = torch.eye(8, dtype=torch.float64)
    for left in range(8):
        for right in range(left + 1, 8):
            value = 0.05 + 0.01 * (left * 8 + right)
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


class Phase6V11RepresentationOverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_protocol_and_frozen_leaf_identity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        leaf = root / (
            "docs/blueprints/branches/learning/work/"
            "ANG-WORK-LEARNING-REPRESENTATION-OVERLAP-V11-D2-001.md"
        )
        self.assertEqual(
            _sha256(leaf),
            "0C6275639A03E1FEAFBBB1C96A2E93A8B388107CAD975121BF14FF8D05CD674D",
        )
        self.assertEqual(
            overlap.PROTOCOL_ID,
            "phase6.public-representation-overlap.v11-d2",
        )
        self.assertEqual(overlap.PERMUTATION_COUNT, 40_320)

    def test_covariance_frobenius_overlap_and_sample_order_invariance(self) -> None:
        values = torch.tensor(
            ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (2.0, 1.0, 0.0), (3.0, 2.0, 1.0))
        )
        covariance = overlap.covariance_matrix(values)
        reversed_covariance = overlap.covariance_matrix(values.flip(0))
        self.assertTrue(torch.allclose(covariance, reversed_covariance, atol=1e-12, rtol=0))
        self.assertAlmostEqual(overlap.frobenius_overlap(covariance, covariance), 1.0)
        self.assertAlmostEqual(
            overlap.frobenius_overlap(covariance, reversed_covariance),
            1.0,
        )

    def test_zero_variance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero variance"):
            overlap.covariance_matrix(torch.ones((8, 4)))
        with self.assertRaisesRegex(ValueError, "zero"):
            overlap.frobenius_overlap(torch.zeros((3, 3)), torch.eye(3))

    def test_hoyer_near_zero_and_effective_rank_are_bounded(self) -> None:
        values = torch.eye(6, dtype=torch.float64).repeat(2, 1)
        description = overlap.activation_description(values)
        self.assertGreaterEqual(description["mean_row_hoyer_sparsity"], 0.0)
        self.assertLessEqual(description["mean_row_hoyer_sparsity"], 1.0)
        self.assertGreaterEqual(description["near_zero_fraction"], 0.0)
        self.assertLessEqual(description["near_zero_fraction"], 1.0)
        self.assertGreaterEqual(description["covariance_effective_rank"], 1.0)
        self.assertLessEqual(description["covariance_effective_rank"], 6.0)

    def test_relative_mantel_permutation_is_exact_and_joint_noop_is_rejected(self) -> None:
        matrix = _structured_matrix()
        gradient = torch.eye(8, dtype=torch.float64)
        gradient[0, 7] = gradient[7, 0] = -1.0
        gradient[1, 6] = gradient[6, 1] = -0.8
        result = overlap.exact_relative_permutation_test(matrix, gradient)
        self.assertEqual(result["permutations"], math.factorial(8))
        self.assertGreater(result["exceedances"], 0)
        self.assertLessEqual(result["exceedances"], 40_320)
        self.assertEqual(result["permuted_matrix"], "representation_overlap_only")
        permutation = torch.tensor((7, 6, 5, 4, 3, 2, 1, 0))
        jointly_permuted = matrix[permutation][:, permutation]
        jointly_permuted_gradient = gradient[permutation][:, permutation]
        self.assertAlmostEqual(
            overlap.burden_statistic(matrix, gradient),
            overlap.burden_statistic(jointly_permuted, jointly_permuted_gradient),
        )
        with self.assertRaisesRegex(ValueError, "invalid no-op"):
            overlap.exact_relative_permutation_test(
                matrix,
                gradient,
                jointly_permute_both=True,
            )

    def test_relative_permutation_identity_uses_the_null_reduction_path(self) -> None:
        matrix = torch.eye(8, dtype=torch.float64)
        gradient = torch.eye(8, dtype=torch.float64)
        for left in range(8):
            for right in range(left + 1, 8):
                value = (((left + 1) * 17 + (right + 1) * 29) % 97) / 100.0
                gradient_value = (
                    -(((((left + 1) * 11 + (right + 1) * 7) % 19) + 1) / 20.0)
                    if (left + right) % 3 != 0
                    else 0.2
                )
                matrix[left, right] = matrix[right, left] = value
                gradient[left, right] = gradient[right, left] = gradient_value

        result = overlap.exact_relative_permutation_test(matrix, gradient)
        self.assertEqual(result["permutations"], 40_320)
        self.assertEqual(result["exceedances"], 17_324)
        self.assertGreaterEqual(result["exceedances"], 1)

    def test_group_overlap_summary_uses_nine_and_three_pair_groups(self) -> None:
        matrix = torch.eye(8, dtype=torch.float64)
        for easy in overlap.EASY_STREAMS:
            for hard in overlap.HARD_STREAMS:
                matrix[easy, hard] = matrix[hard, easy] = 0.9
        summary = overlap.group_overlap_summary(matrix)
        self.assertEqual(summary["easy_hard_pair_count"], 9)
        self.assertEqual(summary["easy_easy_pair_count"], 3)
        self.assertEqual(summary["hard_hard_pair_count"], 3)
        self.assertAlmostEqual(summary["easy_hard_mean"], 0.9)
        self.assertTrue(summary["easy_hard_exceeds_both_within_groups"])

    def test_classification_boundary_requires_both_t0_replicates(self) -> None:
        significant = {
            group: {
                "observed_mean_off_diagonal_burden": 0.1,
                "p_value_one_sided": 0.05 if index < 3 else 0.2,
            }
            for index, group in enumerate(overlap.GRADIENT_GROUPS)
        }
        cell = {
            "group_overlap_summary": {
                "easy_hard_exceeds_both_within_groups": True,
            },
            "gradient_alignment": significant,
        }
        cells = {"t0_s0": cell, "t0_s1": cell}
        self.assertEqual(overlap.classify_interference(cells), overlap.SUPPORTED)
        failed = dict(cell)
        failed["group_overlap_summary"] = {
            "easy_hard_exceeds_both_within_groups": False,
        }
        self.assertEqual(
            overlap.classify_interference({"t0_s0": cell, "t0_s1": failed}),
            overlap.NOT_SUPPORTED,
        )

    def assertCaptureAbsent(self, controller: _FakeController) -> None:
        self.assertNotIn("_factorized_relation_embeddings", controller.__dict__)
        self.assertNotIn("_relation_evidence_read", controller.__dict__)
        self.assertNotIn("_v11_d2_representation_capture_active", controller.__dict__)
        self.assertEqual(len(controller.relation_comparator[1]._forward_hooks), 0)

    def test_capture_has_exact_8_and_48_rows_and_cleans_up_after_success(self) -> None:
        source = _fake_source()
        controller = _FakeController()
        matrices, rows = overlap.capture_stream_representations(
            source,
            controller,
            object(),
        )
        self.assertEqual(rows, 4)
        self.assertEqual(matrices["fused_relation_code"].shape, (8, 32))
        self.assertEqual(matrices["relation_comparator_hidden"].shape, (48, 64))
        self.assertCaptureAbsent(controller)
        self.assertTrue(all(parameter.grad is None for parameter in controller.parameters()))

    def test_capture_cleans_up_after_injected_failure(self) -> None:
        source = _fake_source(fail=True)
        controller = _FakeController()
        with self.assertRaisesRegex(InjectedCaptureError, "injected"):
            overlap.capture_stream_representations(source, controller, object())
        self.assertCaptureAbsent(controller)
        self.assertTrue(all(parameter.grad is None for parameter in controller.parameters()))

    def test_serialization_rejects_raw_activations_and_enforces_ceiling(self) -> None:
        payload = overlap.serialize_bounded_report({"classification": overlap.NOT_SUPPORTED})
        self.assertNotIn(b"activation", payload)
        with self.assertRaisesRegex(ValueError, "forbidden raw"):
            overlap.serialize_bounded_report({"raw_activations": [[1.0]]})
        with self.assertRaisesRegex(ValueError, "byte ceiling"):
            overlap.serialize_bounded_report({"safe": "x" * 100}, maximum_bytes=10)

    def test_altered_frozen_identity_is_rejected(self) -> None:
        identity = dict(overlap.EXPECTED_FROZEN_IDENTITY)
        overlap.validate_frozen_identity(identity)
        identity["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            overlap.validate_frozen_identity(identity)


if __name__ == "__main__":
    unittest.main()
