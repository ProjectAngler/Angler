from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest import mock

import torch

from angler.reasoning import NaturalTraceGraphCausalMemoryCore, parse_step_trace
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import natural_trace_graph_causal_memory_v11 as v11


class _DeterministicQwen:
    def __init__(self, width: int = 6) -> None:
        self.width = width
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts) -> torch.Tensor:
        texts = tuple(texts)
        self.calls.append(texts)
        rows = []
        for text in texts:
            codepoints = [ord(character) for character in text]
            rows.append(
                [
                    float((sum(codepoints[index:: self.width]) + 17 * index) % 997)
                    / 997.0
                    for index in range(self.width)
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)


def _fake_rows() -> tuple[v11.EncodedGraphMechanism, ...]:
    rows = []
    outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
    for index in range(8):
        generator = torch.Generator().manual_seed(11_000 + index)
        target = index % 4
        episode_mask = torch.zeros(6, 8, dtype=torch.bool)
        challenge_mask = torch.zeros(4, 8, dtype=torch.bool)
        episode_mask[:, :3] = True
        challenge_mask[:, :3] = True
        rows.append(
            v11.EncodedGraphMechanism(
                mechanism_ref=f"synthetic-v11-{index}",
                generator_family=f"family-{index // 2}",
                episode_queries=torch.randn(6, 6, generator=generator),
                episode_candidates=torch.randn(6, 6, generator=generator),
                episode_temporal=torch.randn(6, 8, generator=generator),
                episode_outcomes=outcomes.clone(),
                challenge_query=torch.randn(6, generator=generator),
                challenge_candidates=torch.randn(4, 6, generator=generator),
                challenge_temporal=torch.randn(4, 8, generator=generator),
                target_index=target,
                failed_indices=tuple(slot for slot in range(4) if slot != target),
                episode_step_features=torch.randn(6, 8, 6, generator=generator),
                episode_step_mask=episode_mask,
                challenge_step_features=torch.randn(4, 8, 6, generator=generator),
                challenge_step_mask=challenge_mask,
            )
        )
    return tuple(rows)


def _passing_v11_metrics() -> dict[str, object]:
    return {
        "full_failed_to_pass_repairs": 2,
        "graph_component": {
            "full_minus_graph_removed_successes": 2,
            "full_minus_graph_removed_target_probability": 0.10,
            "full_minus_direction_removed_successes": 1,
            "full_minus_direction_removed_target_probability": 0.0,
        },
        "graph_invariance": {
            "mechanism_count": 8,
            "maximum_absolute_graph_logit": 1.0e-12,
            "maximum_graph_logit_difference_under_shuffle": 1.0e-6,
            "maximum_graph_sidecar_difference_under_shuffle": 1.0e-6,
            "maximum_graph_sidecar_mask_difference_under_shuffle": 1.0e-6,
        },
        "allocation": {
            "sidecar_occupied_slots": 48,
            "sidecar_usage_alignment": True,
            "stress_sidecar_usage_alignment": True,
        },
    }


class NaturalTraceGraphCausalMemoryV11RunnerTests(unittest.TestCase):
    def tearDown(self) -> None:
        v11.v7._apply_stream = v11._ORIGINAL_APPLY_STREAM
        v11.v7._challenge = v11._ORIGINAL_CHALLENGE
        v11.v7.MEMORY_SLOTS = 16

    def test_encoding_preserves_pooled_v6_rows_and_embeds_only_parsed_steps(self) -> None:
        corpus = build_causal_neuromodulated_apprenticeship_v6()
        mechanisms = corpus.development[:2]
        base_qwen = _DeterministicQwen()
        graph_qwen = _DeterministicQwen()

        expected = v11.v7.v6._encode_partition(base_qwen, mechanisms)
        actual = v11._encode_partition(graph_qwen, mechanisms)

        self.assertEqual(len(graph_qwen.calls), 2)
        expected_steps = []
        for mechanism in mechanisms:
            traces = [episode.action_trace_text for episode in mechanism.public.episodes]
            traces.extend(mechanism.public.challenge.candidate_action_trace_texts)
            for trace in traces:
                expected_steps.extend(parse_step_trace(trace))
        self.assertEqual(graph_qwen.calls[1], tuple(expected_steps))
        self.assertEqual(graph_qwen.calls[0], base_qwen.calls[0])
        for base, row, mechanism in zip(expected, actual, mechanisms):
            for name in (
                "episode_queries",
                "episode_candidates",
                "episode_temporal",
                "episode_outcomes",
                "challenge_query",
                "challenge_candidates",
                "challenge_temporal",
            ):
                self.assertTrue(torch.equal(getattr(row, name), getattr(base, name)))
            self.assertEqual(row.episode_step_features.shape, (6, 8, 6))
            self.assertEqual(row.challenge_step_features.shape, (4, 8, 6))
            traces = [episode.action_trace_text for episode in mechanism.public.episodes]
            traces.extend(mechanism.public.challenge.candidate_action_trace_texts)
            lengths = [len(parse_step_trace(trace)) for trace in traces]
            self.assertEqual(row.episode_step_mask.sum(dim=-1).tolist(), lengths[:6])
            self.assertEqual(row.challenge_step_mask.sum(dim=-1).tolist(), lengths[6:])

        forbidden = {
            mechanism.metadata.mechanism_ref for mechanism in mechanisms
        } | {
            mechanism.metadata.generator_family for mechanism in mechanisms
        } | {
            episode.objective_diagnostic_text
            for mechanism in mechanisms
            for episode in mechanism.public.episodes
        }
        self.assertTrue(forbidden.isdisjoint(graph_qwen.calls[1]))

    def test_inherited_runner_patch_restores_on_success_and_error(self) -> None:
        original_apply = v11.v7._apply_stream
        original_challenge = v11.v7._challenge
        with v11._v11_runner_patch():
            self.assertIs(v11.v7._apply_stream, v11._apply_stream)
            self.assertIs(v11.v7._challenge, v11._challenge)
        self.assertIs(v11.v7._apply_stream, original_apply)
        self.assertIs(v11.v7._challenge, original_challenge)

        original_flags = (v11._INCLUDE_DIRECTION, v11._INCLUDE_GRAPH_ADDRESS)
        with v11._graph_lesion_scope(
            include_direction=False, include_graph_address=False
        ):
            self.assertEqual(
                (v11._INCLUDE_DIRECTION, v11._INCLUDE_GRAPH_ADDRESS),
                (False, False),
            )
        self.assertEqual(
            (v11._INCLUDE_DIRECTION, v11._INCLUDE_GRAPH_ADDRESS), original_flags
        )
        with self.assertRaisesRegex(RuntimeError, "lesion failure"):
            with v11._graph_lesion_scope(
                include_direction=False, include_graph_address=False
            ):
                raise RuntimeError("lesion failure")
        self.assertEqual(
            (v11._INCLUDE_DIRECTION, v11._INCLUDE_GRAPH_ADDRESS), original_flags
        )

        with self.assertRaisesRegex(RuntimeError, "injected"):
            with v11._v11_runner_patch():
                raise RuntimeError("injected")
        self.assertIs(v11.v7._apply_stream, original_apply)
        self.assertIs(v11.v7._challenge, original_challenge)

    def test_step_sidecars_align_through_stream_challenge_and_permutation(self) -> None:
        torch.manual_seed(11_101)
        core = NaturalTraceGraphCausalMemoryCore(
            content_width=6,
            temporal_width=8,
            step_width=6,
            memory_slots=64,
        )
        rows = _fake_rows()

        local, _ = v11._apply_stream(
            core,
            rows[0],
            core.initial_memory_state(),
            detach_state=True,
            collect_outputs=True,
        )
        self.assertEqual(local.step, 6)
        self.assertEqual(int(local.graph_step_mask.any(dim=-1).sum().item()), 6)
        challenge = v11._challenge(core, rows[0], local)
        self.assertEqual(challenge.candidate_step_features.shape, (1, 4, 8, 6))
        self.assertTrue(torch.equal(challenge.candidate_step_mask[0], rows[0].challenge_step_mask))

        metrics = v11._allocation_metrics(core, rows)
        self.assertEqual(metrics["event_count"], 48)
        self.assertEqual(metrics["persistent_unique_winning_slots"], 48)
        self.assertEqual(metrics["sidecar_occupied_slots"], 48)
        self.assertIs(metrics["sidecar_usage_alignment"], True)
        self.assertLessEqual(metrics["slot_permutation_read_value_difference"], 1.0e-6)
        self.assertLessEqual(metrics["slot_permutation_residual_difference"], 1.0e-6)
        self.assertLessEqual(metrics["slot_permutation_score_difference"], 1.0e-6)
        self.assertEqual(metrics["stress_step"], 72)
        self.assertIs(metrics["stress_bounds_valid"], True)
        self.assertIs(metrics["stress_sidecar_usage_alignment"], True)

    def test_family_report_includes_behavioral_and_state_swap_arms(self) -> None:
        row = {
            "mechanism_ref": "mechanism",
            "generator_family": "family",
            "success": True,
            "target_probability_after": 0.75,
        }
        report = v11._per_family_report(
            {
                "full": {"per_mechanism": [row]},
                "graph_address_removed": {"per_mechanism": [row]},
                "equivalent_state_swap": {"per_mechanism": [row]},
                "unrelated_state_swap": {"per_mechanism": [row]},
            }
        )
        self.assertEqual(
            set(report["family"]),
            {
                "full",
                "graph_address_removed",
                "equivalent_state_swap",
                "unrelated_state_swap",
            },
        )
        self.assertEqual(report["family"]["full"]["successes"], 1)
        self.assertEqual(
            report["family"]["unrelated_state_swap"]["mean_target_probability"],
            0.75,
        )

    def test_development_gate_accepts_boundaries_and_rejects_each_new_operand(self) -> None:
        passing = _passing_v11_metrics()
        cases = {}
        replacements = {
            ("full_failed_to_pass_repairs",): 1,
            ("graph_component", "full_minus_graph_removed_successes"): 1,
            ("graph_component", "full_minus_graph_removed_target_probability"): 0.099999,
            ("graph_invariance", "mechanism_count"): 7,
            ("graph_invariance", "maximum_absolute_graph_logit"): 0.0,
            ("allocation", "sidecar_occupied_slots"): 47,
            ("allocation", "sidecar_usage_alignment"): False,
            ("allocation", "stress_sidecar_usage_alignment"): False,
        }
        for path, replacement in replacements.items():
            candidate = copy.deepcopy(passing)
            cursor = candidate
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = replacement
            cases[".".join(path)] = candidate
        direction = copy.deepcopy(passing)
        direction["graph_component"]["full_minus_direction_removed_successes"] = 0
        direction["graph_component"]["full_minus_direction_removed_target_probability"] = 0.049999
        cases["direction_disjunction"] = direction
        for name in (
            "maximum_graph_logit_difference_under_shuffle",
            "maximum_graph_sidecar_difference_under_shuffle",
            "maximum_graph_sidecar_mask_difference_under_shuffle",
        ):
            candidate = copy.deepcopy(passing)
            candidate["graph_invariance"][name] = 1.000001e-6
            cases[name] = candidate

        with mock.patch.object(v11.v10, "_development_gate", return_value=True):
            self.assertTrue(v11._development_gate(passing, True))
            probability_direction = copy.deepcopy(passing)
            probability_direction["graph_component"]["full_minus_direction_removed_successes"] = 0
            probability_direction["graph_component"]["full_minus_direction_removed_target_probability"] = 0.05
            self.assertTrue(v11._development_gate(probability_direction, True))
            for label, candidate in cases.items():
                with self.subTest(label=label):
                    self.assertFalse(v11._development_gate(candidate, True))

    def test_identity_schedule_source_chain_and_final_seal_are_fresh(self) -> None:
        self.assertNotEqual(v11.IDENTITY, v11.v10.IDENTITY)
        self.assertEqual(v11.SEED, 2026083111)
        self.assertEqual(v11.MEMORY_SLOTS, 64)
        schedule = v11.v7.v6._pair_schedule()
        self.assertEqual(len(schedule), 256)
        self.assertTrue(all(anchor != current for anchor, current in schedule))
        self.assertTrue(all(sum(anchor == index for anchor, _ in schedule) == 4 for index in range(64)))
        self.assertTrue(all(sum(current == index for _, current in schedule) == 4 for index in range(64)))
        hashes = v11._source_hashes()
        self.assertEqual({"runner", "core"}, set(hashes) & {"runner", "core"})
        self.assertTrue(any(name.startswith("inherited_") for name in hashes))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

        corpus = build_causal_neuromodulated_apprenticeship_v6()
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final

    def test_final_authorization_rejects_any_unbound_record(self) -> None:
        sources = {"runner": "source-digest"}
        valid = {
            "identity": v11.IDENTITY,
            "phase": "train-development",
            "classification": "DEVELOPMENT_GATE_PASSED",
            "development_authorized": True,
            "source_hashes": sources,
            "checkpoint_sha256": "checkpoint-digest",
            "development_metrics": {"synthetic": "passing"},
            "identity_checks": {
                "exact": True,
                "foundation_before": "foundation",
                "foundation_after": "foundation",
                "core_before_evaluation": "core",
                "core_after_evaluation": "core",
            },
        }
        checkpoint = Path("synthetic-v11-checkpoint.pt")
        with (
            mock.patch.object(v11, "_source_hashes", return_value=sources),
            mock.patch.object(v11.v7.v6.v5, "_sha256", return_value="checkpoint-digest"),
            mock.patch.object(v11, "_development_gate", return_value=True),
        ):
            v11._validate_development_authorization(valid, checkpoint)
            for field, value in (
                ("identity", v11.v10.IDENTITY),
                ("phase", "final"),
                ("classification", "DEVELOPMENT_NOT_SUPPORTED"),
                ("development_authorized", False),
                ("source_hashes", {}),
                ("checkpoint_sha256", "wrong"),
            ):
                candidate = copy.deepcopy(valid)
                candidate[field] = value
                with self.subTest(field=field), self.assertRaises(RuntimeError):
                    v11._validate_development_authorization(candidate, checkpoint)

            for field, value in (
                ("exact", False),
                ("foundation_after", "changed"),
                ("core_after_evaluation", "changed"),
            ):
                candidate = copy.deepcopy(valid)
                candidate["identity_checks"][field] = value
                with self.subTest(identity_field=field), self.assertRaises(RuntimeError):
                    v11._validate_development_authorization(candidate, checkpoint)

        fabricated = copy.deepcopy(valid)
        fabricated["development_metrics"] = {"claimed": "passing"}
        with (
            mock.patch.object(v11, "_source_hashes", return_value=sources),
            mock.patch.object(v11.v7.v6.v5, "_sha256", return_value="checkpoint-digest"),
            self.assertRaises(RuntimeError),
        ):
            v11._validate_development_authorization(fabricated, checkpoint)


if __name__ == "__main__":
    unittest.main()
