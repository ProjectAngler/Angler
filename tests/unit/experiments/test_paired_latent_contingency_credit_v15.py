from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditState,
)
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    ACQUISITION_INDICES,
    PROBE_INDICES,
    build_paired_latent_contingency_credit_v15,
)
from experiments.runners import paired_latent_contingency_credit_v15 as v15


def _fake_encoded(pair: object, index: int) -> v15.EncodedTwinPair:
    generator = torch.Generator().manual_seed(10_000 + index)
    relation = torch.randn(6, 64, generator=generator)
    base = torch.randn(6, generator=generator) * 0.1
    temporal = torch.tensor(
        [v15._temporal(event, position) for position, event in enumerate(pair.first.public.events)],
        dtype=torch.float32,
    )
    input_hash = f"INPUT-{pair.pair_ref}"
    payload_hash = f"PAYLOAD-{pair.pair_ref}"

    def episode(value: object) -> v15.EncodedContingencyEpisode:
        return v15.EncodedContingencyEpisode(
            episode_ref=value.metadata.episode_ref,
            pair_ref=value.metadata.pair_ref,
            twin_ref=value.metadata.twin_ref,
            generator_family=value.metadata.generator_family,
            transition_group=value.metadata.transition_group,
            relation_features=relation.clone(),
            base_logits=base.clone(),
            temporal_features=temporal.clone(),
            outcomes=torch.tensor(value.supervision.outcome_values, dtype=torch.float32),
            relation_classes=tuple(value.supervision.relation_classes),
            public_payload_sha256=payload_hash,
            input_tensor_sha256=input_hash,
        )

    return v15.EncodedTwinPair(pair.pair_ref, episode(pair.first), episode(pair.second))


def _arm(balanced: float, nll: float) -> dict[str, object]:
    return {
        "balanced_accuracy": balanced,
        "mean_probe_nll": nll,
        "finite": True,
        "read_weight_min": 0.0,
        "read_weight_max": 1.0,
        "write_strength_min": 0.0,
        "write_strength_max": 1.0,
    }


def _passing_metrics() -> dict[str, object]:
    return {
        "pair_count": 48,
        "arms": {
            "true": _arm(0.71, 0.20),
            "deranged": _arm(0.60, 0.23),
            "zero_no_read": _arm(0.60, 0.23),
            "post_acquisition_reset": _arm(0.60, 0.30),
            "outcome_blind_writer": _arm(0.60, 0.30),
        },
        "affine_calibrator": {"balanced_accuracy": 0.60},
        "outcome_blind_integrity": {
            "write_count_exact": True,
            "allocation_exact": True,
            "write_gate_exact": True,
            "erase_gate_exact": True,
            "write_keys_exact": True,
            "usage_exact": True,
            "acquisition_exact": True,
            "stored_values_differ": True,
            "shared_value_token_outcome_invariant": True,
            "only_stored_outcome_value_content_changed": True,
        },
        "paired_twin": {
            "directional_accuracy": 0.75,
            "mean_outcome_directed_logit_margin": 0.20,
        },
        "state_controls": {
            "matched_balanced_accuracy": 0.71,
            "zero_balanced_accuracy": 0.60,
            "unrelated_balanced_accuracy": 0.60,
            "key_value_mismatch_balanced_accuracy": 0.60,
            "full_causal_gain": 0.11,
            "opposite_twin_swap_reversal_fraction": 0.70,
            "opposite_twin_swap_gain_retention": 0.80,
            "mean_nll": {
                "matched": 0.20,
                "zero": 0.23,
                "unrelated": 0.23,
                "key_value_mismatch": 0.23,
            },
        },
        "improved_family_count": 9,
        "transition_improved_family_counts": {
            "insertion": 3,
            "removal": 2,
            "reorder": 2,
            "replacement": 2,
        },
        "ordering": {"changed_fraction": 0.25},
        "replay": {"exact": True, "maximum_logit_error": 1.0e-6},
        "retention": {"maximum_drop": 0.05},
        "state_bounds": {
            "finite": True,
            "keys_max_abs": 1.0,
            "values_max_abs": 1.0,
            "usage_min": 0.0,
            "usage_max": 1.0,
            "acquisition_min": 0.0,
            "acquisition_max": 1.0,
        },
        "protocol_invariants": {
            "exact_training_schedule": True,
            "all_training_gradients_finite": True,
            "true_deranged_paths_differentiable": True,
            "zero_baseline_detached": True,
            "eight_pair_state_horizon": True,
        },
    }


class FrozenMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()

    def test_schedule_is_exact_and_every_pair_occurs_twice(self) -> None:
        schedule = v15._training_schedule(self.corpus)
        self.assertEqual(len(schedule), 96)
        self.assertTrue(all(len(row) == 8 and len(set(row)) == 8 for row in schedule))
        counts = {index: 0 for index in range(384)}
        for row in schedule:
            for index in row:
                counts[index] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_identifiability_preflight_accepts_exact_twins(self) -> None:
        train = tuple(_fake_encoded(pair, index) for index, pair in enumerate(self.corpus.train))
        development = tuple(
            _fake_encoded(pair, 10_000 + index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = StructureKeyedCreditMemoryCore(temporal_width=4)
        report = v15._identifiability_preflight(
            core, train, development, v15._training_schedule(self.corpus)
        )
        self.assertTrue(report["exact"])
        self.assertTrue(report["twin_input_tensor_hashes_exact"])
        self.assertTrue(report["twin_acquisition_histories_fixed_point_free"])
        self.assertTrue(report["relation_class_by_polarity_balanced_at_every_position"])
        self.assertEqual(report["matched_probe_label_entropy_bits"], 1.0)

    def test_pair_public_input_is_encoded_once_then_cloned_to_twins(self) -> None:
        class Frozen(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

            def relation_features(
                self, reference: torch.Tensor, reference_mask: torch.Tensor,
                attempt: torch.Tensor, attempt_mask: torch.Tensor,
            ) -> torch.Tensor:
                del reference_mask, attempt_mask
                batch, events = reference.shape[:2]
                seed = reference[..., 0, 0] + attempt[..., 0, 0]
                return seed.unsqueeze(-1).expand(batch, events, 64).clone()

            def functional_logits(
                self, reference: torch.Tensor, reference_mask: torch.Tensor,
                attempt: torch.Tensor, attempt_mask: torch.Tensor,
            ) -> torch.Tensor:
                del reference_mask, attempt_mask
                return reference[..., 0, 0] - attempt[..., 0, 0]

        calls: list[int] = []

        def embed_groups(_qwen: object, groups: object) -> tuple[torch.Tensor, torch.Tensor]:
            count = len(groups)
            calls.append(count)
            features = torch.arange(count * 6, dtype=torch.float32).reshape(count, 6, 1, 1)
            mask = torch.ones(count, 6, 1, dtype=torch.bool)
            return features, mask

        source = self.corpus.development[:2]
        with patch.object(v15.v13, "_embed_groups", side_effect=embed_groups):
            encoded = v15._encode_pairs(object(), Frozen(), source)
        self.assertEqual(calls, [len(source), len(source)])
        for pair in encoded:
            self.assertTrue(torch.equal(pair.first.relation_features, pair.second.relation_features))
            self.assertTrue(torch.equal(pair.first.base_logits, pair.second.base_logits))
            self.assertTrue(torch.equal(pair.first.temporal_features, pair.second.temporal_features))
            self.assertEqual(pair.first.input_tensor_sha256, pair.second.input_tensor_sha256)
            self.assertEqual(pair.first.public_payload_sha256, pair.second.public_payload_sha256)
            self.assertNotEqual(pair.first.outcomes.tolist(), pair.second.outcomes.tolist())

    def test_causal_losses_are_per_probe_relu_and_zero_is_detached(self) -> None:
        true = torch.tensor([0.20, 0.40], requires_grad=True)
        deranged = torch.tensor([0.30, 0.35], requires_grad=True)
        zero = torch.tensor([0.25, 0.60], requires_grad=True)
        margins = torch.tensor([0.10, 0.30], requires_grad=True)
        objective, _, fixed_zero, separation, ranking, total = v15._causal_training_losses(
            true, deranged, zero, margins
        )
        expected_separation = 0.5 * (
            torch.relu(0.05 + true - deranged).mean()
            + torch.relu(0.05 + true - zero.detach()).mean()
        )
        self.assertTrue(torch.allclose(separation, expected_separation))
        self.assertTrue(torch.allclose(ranking, torch.relu(0.20 - margins).mean()))
        self.assertIsNone(fixed_zero.grad_fn)
        total.backward()
        self.assertIsNotNone(true.grad)
        self.assertIsNotNone(deranged.grad)
        self.assertIsNone(zero.grad)
        self.assertIsNotNone(margins.grad)
        self.assertAlmostEqual(float(objective.detach()), 0.30, places=6)

    def test_real_fit_update_uses_aligned_scalar_rows_and_steps_optimizer(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, index) for index, pair in enumerate(self.corpus.train[:8])
        )
        core = StructureKeyedCreditMemoryCore(temporal_width=4)
        before = {
            name: value.detach().clone() for name, value in core.named_parameters()
        }
        with (
            patch.object(v15, "TRAIN_PAIRS", 8),
            patch.object(v15, "TRAIN_UPDATES", 1),
        ):
            report, _ = v15._fit(core, pairs, (tuple(range(8)),))
        self.assertEqual(report["completed_updates"], 1)
        update = report["updates"][0]
        self.assertTrue(update["true_deranged_paths_differentiable"])
        self.assertTrue(update["zero_baseline_detached"])
        self.assertTrue(update["predict_before_feedback"])
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in core.named_parameters()
            )
        )

    def test_development_uses_six_independent_eight_pair_blocks(self) -> None:
        pairs = tuple(_fake_encoded(pair, index) for index, pair in enumerate(self.corpus.development))
        core = StructureKeyedCreditMemoryCore(temporal_width=4)
        arm = v15._arm_stream(core, pairs, arm="true")
        self.assertEqual(arm["maximum_state_step"], 16)
        self.assertEqual(len(arm["event_blocks"]), 6)
        self.assertTrue(all(len(block[side]) == 16 for block in arm["event_blocks"] for side in ("first", "second")))
        expected = [2, 4, 6, 8, 10, 12, 14, 16] * 6
        self.assertEqual([row["first"].step for row in arm["states_after_pair"]], expected)
        self.assertEqual(len(arm["logits"]), 48 * 2 * 4)

    def test_mismatch_permutes_only_occupied_values(self) -> None:
        core = StructureKeyedCreditMemoryCore(temporal_width=4)
        state = core.initial_state()
        values = state.values.clone()
        values[:4] = torch.arange(4, dtype=values.dtype).unsqueeze(1)
        populated = StructureKeyedCreditState(state.keys, values, state.usage, state.acquisition, 4)
        mismatch = v15._mismatched_state(populated)
        self.assertTrue(torch.equal(mismatch.values[:4], values[:4].flip(0)))
        self.assertTrue(torch.equal(mismatch.values[4:], values[4:]))
        self.assertTrue(torch.equal(mismatch.keys, populated.keys))

    def test_unrelated_donor_changes_family_and_one_history_bit(self) -> None:
        pairs = tuple(_fake_encoded(pair, index) for index, pair in enumerate(self.corpus.development))
        for index, target in enumerate(pairs):
            donor = pairs[v15._unrelated_donor_index(pairs, index)]
            self.assertNotEqual(donor.first.generator_family, target.first.generator_family)
            target_history = [float(target.first.outcomes[i]) for i in ACQUISITION_INDICES]
            donor_history = [float(donor.first.outcomes[i]) for i in ACQUISITION_INDICES]
            self.assertEqual(sum(left != right for left, right in zip(target_history, donor_history, strict=True)), 1)

    def test_outcome_blind_intervention_changes_only_shared_token(self) -> None:
        core = StructureKeyedCreditMemoryCore(temporal_width=4)
        pair = _fake_encoded(self.corpus.development[0], 0)
        integrity = v15._blind_addressing_integrity(core, pair)
        self.assertTrue(all(integrity.values()), integrity)
        episode, twin = pair.first, pair.second
        normal_state, normal_events, _, _ = v15._write_anchors(
            core, episode, twin, core.initial_state(), arm="true", detach_state=True,
        )
        blind_state, _, blind_events, _, _ = v15._write_blind_anchors(
            core, episode, core.initial_state(), core.initial_state(),
        )
        for normal, blind in zip(normal_events, blind_events, strict=True):
            self.assertEqual(normal.allocation_index, blind.allocation_index)
            self.assertEqual(normal.write_strength, blind.write_strength)
            self.assertEqual(normal.erase_gate, blind.erase_gate)
            self.assertTrue(torch.equal(normal.write_key, blind.write_key))
        self.assertEqual(normal_state.step, blind_state.step)
        self.assertTrue(torch.equal(normal_state.keys, blind_state.keys))
        self.assertTrue(torch.equal(normal_state.usage, blind_state.usage))
        self.assertTrue(torch.equal(normal_state.acquisition, blind_state.acquisition))
        self.assertFalse(torch.equal(normal_state.values, blind_state.values))


class GateTests(unittest.TestCase):
    def test_gate_accepts_exact_boundary_packet(self) -> None:
        self.assertTrue(v15._gate(_passing_metrics(), identifiability_exact=True, identity_exact=True))

    def test_gate_rejects_every_causal_operand(self) -> None:
        passing = _passing_metrics()
        mutations = []
        value = copy.deepcopy(passing); value["arms"]["true"]["balanced_accuracy"] = 0.69; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["deranged"]["balanced_accuracy"] = 0.62; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["deranged"]["mean_probe_nll"] = 0.219; mutations.append(value)
        value = copy.deepcopy(passing); value["paired_twin"]["directional_accuracy"] = 0.749; mutations.append(value)
        value = copy.deepcopy(passing); value["paired_twin"]["mean_outcome_directed_logit_margin"] = 0.199; mutations.append(value)
        value = copy.deepcopy(passing); value["state_controls"]["full_causal_gain"] = 0.0; mutations.append(value)
        value = copy.deepcopy(passing); value["state_controls"]["opposite_twin_swap_reversal_fraction"] = 0.699; mutations.append(value)
        value = copy.deepcopy(passing); value["state_controls"]["opposite_twin_swap_gain_retention"] = 0.799; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["post_acquisition_reset"]["balanced_accuracy"] = 0.62; mutations.append(value)
        value = copy.deepcopy(passing); value["affine_calibrator"]["balanced_accuracy"] = 0.62; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["outcome_blind_writer"]["balanced_accuracy"] = 0.62; mutations.append(value)
        value = copy.deepcopy(passing); value["improved_family_count"] = 8; mutations.append(value)
        value = copy.deepcopy(passing); value["transition_improved_family_counts"]["removal"] = 1; mutations.append(value)
        value = copy.deepcopy(passing); value["ordering"]["changed_fraction"] = 0.249; mutations.append(value)
        value = copy.deepcopy(passing); value["state_controls"]["mean_nll"]["unrelated"] = 0.219; value["state_controls"]["unrelated_balanced_accuracy"] = 0.71; mutations.append(value)
        value = copy.deepcopy(passing); value["replay"]["exact"] = False; mutations.append(value)
        value = copy.deepcopy(passing); value["retention"]["maximum_drop"] = 0.051; mutations.append(value)
        value = copy.deepcopy(passing); value["state_bounds"]["values_max_abs"] = 1.001; mutations.append(value)
        value = copy.deepcopy(passing); value["protocol_invariants"]["eight_pair_state_horizon"] = False; mutations.append(value)
        for index, metrics in enumerate(mutations):
            with self.subTest(index=index):
                self.assertFalse(v15._gate(metrics, identifiability_exact=True, identity_exact=True))
        self.assertFalse(v15._gate(passing, identifiability_exact=False, identity_exact=True))
        self.assertFalse(v15._gate(passing, identifiability_exact=True, identity_exact=False))

    def test_final_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-authorized")
            with self.assertRaises(RuntimeError):
                v15._validate_final_authorization({}, checkpoint)


if __name__ == "__main__":
    unittest.main()
