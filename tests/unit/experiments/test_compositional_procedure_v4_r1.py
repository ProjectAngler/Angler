from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    CandidateProcedureDecoder,
    ScalableProceduralCore,
    candidate_procedure_loss,
    procedural_core_config,
)
from experiments.runners.compositional_procedure_v4_r1 import (
    CAUSAL_MARGIN_ARMS,
    DIAGNOSTIC_ARMS,
    EPOCHS,
    IDENTITY,
    _anchored_initial_state,
    _public_training_streams,
    _resolve_device_split,
    _slot_anchor_digest,
    _slot_anchor_keys,
    _slot_anchor_record,
    supervised_composed_target,
)
from experiments.runners.scaled_procedural_software_v1 import (
    commit_software_pipeline,
    judge_software_pipeline_attempt,
    prepare_corpus,
)


class CompositionalProcedureV4R1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.raw_train, cls.development, cls.final = prepare_corpus()

    @staticmethod
    def _core() -> ScalableProceduralCore:
        return ScalableProceduralCore(
            procedural_core_config("compact", content_width=32, temporal_width=7)
        )

    @staticmethod
    def _inputs(seed: int):
        generator = torch.Generator().manual_seed(seed)
        return (
            torch.randn(1, 32, generator=generator),
            torch.randn(1, 5, 32, generator=generator),
            torch.randn(1, 5, 7, generator=generator),
            torch.ones(1, 5, dtype=torch.bool),
        )

    def test_fresh_identity_keeps_v4_budget_and_gate_controls(self) -> None:
        self.assertEqual(IDENTITY, "angler.compositional-procedure.v4-r1")
        self.assertEqual(EPOCHS, 8)
        self.assertEqual(
            CAUSAL_MARGIN_ARMS,
            (
                "reset_state",
                "coordinates_removed",
                "unrelated_evidence",
                "procedure_slots_removed",
            ),
        )
        self.assertEqual(DIAGNOSTIC_ARMS, ("local_only", "persistent_only"))
        self.assertTrue(set(CAUSAL_MARGIN_ARMS).isdisjoint(DIAGNOSTIC_ARMS))

    def test_dual_gpu_placement_requires_two_explicit_distinct_devices(self) -> None:
        qwen, angler = _resolve_device_split("cuda:0", "cuda:1", device_count=2)
        self.assertEqual((qwen, angler), (torch.device("cuda:0"), torch.device("cuda:1")))
        for qwen_name, angler_name in (
            ("cuda:0", "cuda:0"),
            ("cpu", "cuda:1"),
            ("cuda", "cuda:1"),
            ("cuda:0", "cuda:2"),
        ):
            with self.subTest(qwen=qwen_name, angler=angler_name):
                with self.assertRaises(ValueError):
                    _resolve_device_split(qwen_name, angler_name, device_count=2)

    def test_anchor_identity_is_replayable_and_content_neutral(self) -> None:
        core = self._core()
        first = _slot_anchor_keys(core.config)
        second = _slot_anchor_keys(core.config)
        record = _slot_anchor_record(core.config)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(record["sha256"], _slot_anchor_digest(core.config))
        self.assertEqual(record["shape"], list(first.shape))
        self.assertEqual(set(first.unique().tolist()), {-1.0, 1.0})

    def test_zero_strength_anchor_reset_is_read_equivalent_to_v1_reset(self) -> None:
        torch.manual_seed(11)
        core = self._core().eval()
        inputs = self._inputs(12)
        with torch.inference_mode():
            original = core(*inputs, plastic_state=core.initial_plastic_state())
            anchored = core(*inputs, plastic_state=_anchored_initial_state(core))
        self.assertTrue(torch.equal(original.procedure_slots, anchored.procedure_slots))
        self.assertTrue(torch.equal(original.candidate_weights, anchored.candidate_weights))
        self.assertTrue(torch.equal(original.qwen_prefix, anchored.qwen_prefix))

    def test_distinct_writes_and_query_loss_reach_every_writer_stage(self) -> None:
        torch.manual_seed(13)
        core = self._core()
        state = _anchored_initial_state(core)
        for seed in (14, 15):
            output = core(*self._inputs(seed), plastic_state=state)
            state = core.apply_feedback(
                state,
                output,
                torch.ones(1, dtype=output.query_state.dtype),
                detach_state=False,
            )
        strength_spread = (state.strengths.max() - state.strengths.min()).detach()
        value_spread = (state.values - state.values[:1]).abs().max().detach()
        self.assertGreater(float(strength_spread), 1.0e-6)
        self.assertGreater(float(value_spread), 1.0e-6)

        query_output = core(*self._inputs(16), plastic_state=state)
        decoder = CandidateProcedureDecoder(
            content_width=32,
            procedure_width=core.config.model_width,
            hidden_width=32,
            heads=4,
            maximum_actions=6,
            maximum_steps=4,
        )
        decoded = decoder(
            query_output.procedure_slots,
            torch.randn(1, 6, 32),
            torch.ones(1, 6, dtype=torch.bool),
            teacher_actions=torch.tensor([[0, 1, 2, 3]]),
        )
        candidate_procedure_loss(decoded, torch.tensor([[0, 1, 2, 3]])).backward()
        for stage in (core.write_key, core.write_value, core.write_gate):
            gradient = stage[0].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertIsNotNone(decoder.procedure_projection.weight.grad)

    def test_train_labels_execute_but_hidden_pairs_are_discarded(self) -> None:
        train = _public_training_streams(self.raw_train)
        self.assertEqual(len(train), 64)
        for raw_stream, public_stream in zip(self.raw_train, train, strict=True):
            self.assertEqual((len(public_stream.supports), len(public_stream.queries)), (4, 2))
            for raw, public in zip(raw_stream.queries, public_stream.queries, strict=True):
                self.assertIsNone(public.pair)
                labels = public.target_text.split()
                indices = [ord(value) - ord("A") for value in labels]
                self.assertEqual(len(indices), 4)
                self.assertEqual(len(set(indices)), 4)
                actions = [raw.pair.learner.grounded_candidates[index] for index in indices]
                attempt = commit_software_pipeline(raw.pair.learner, actions, stopped=False)
                self.assertEqual(judge_software_pipeline_attempt(raw.pair, attempt), 1.0)

    def test_evaluation_partitions_never_expose_targets(self) -> None:
        for partition in (self.development, self.final):
            for stream in partition:
                for example in (*stream.supports, *stream.queries):
                    if example.pair is not None:
                        self.assertIsNone(example.target_text)
            with self.assertRaisesRegex(ValueError, "train partition"):
                supervised_composed_target(partition[0].queries[0])


if __name__ == "__main__":
    unittest.main()
