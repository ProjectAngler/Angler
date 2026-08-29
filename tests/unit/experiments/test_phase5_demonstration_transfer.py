from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.evaluators.relational_procedure_transfer_suite import (
    make_relational_procedure_transfer_stream,
)
from experiments.evaluators.skill_memory_suite import make_skill_memory_meta_partition
from experiments.evaluators.symbolic_procedure_transfer_suite import (
    make_demonstration_procedure_transfer_stream,
)
from experiments.runners import phase5_skill_memory_stream as phase5
from experiments.runners.phase5_cross_family_transfer import SharedPublicFactAdapter
from experiments.runners.phase5_demonstration_transfer import (
    SymbolicDemonstrationAdapter,
    TypedPublicFactPorts,
    _causal_pass_bounds,
    _attach_public_evidence_reader,
    _attach_public_evidence_writer,
    _load_demonstration_adapter,
    _public_delta_preference_alignment,
    _root_public_transition_gate_mean_absolute,
    _save_train_only_adapter,
)


class SymbolicDemonstrationAdapterTests(unittest.TestCase):
    def test_pass_bounds_include_the_final_partial_mechanism_pass(self) -> None:
        self.assertEqual(_causal_pass_bounds(64, 80), ((1, 64),))
        self.assertEqual(
            _causal_pass_bounds(256, 80),
            ((1, 80), (81, 160), (161, 240), (241, 256)),
        )

    def test_public_delta_alignment_counts_only_observed_reward_edges(self) -> None:
        delta = torch.zeros(120)
        delta[3] = 2.0
        delta[7] = 1.0
        delta[11] = -1.0
        delta[13] = 0.25

        aligned, edges = _public_delta_preference_alignment(
            delta,
            (3, 7, 11, 13),
            (1.0, 0.5, 0.5, 0.0),
        )

        self.assertEqual(edges, 5)
        self.assertEqual(aligned, 4)

    def test_root_gate_diagnostic_is_detached_and_candidate_blind(self) -> None:
        gate = torch.tensor([[1.0, -3.0]], requires_grad=True)
        scores = SimpleNamespace(
            nodes=(
                SimpleNamespace(
                    path=(),
                    memory_read=SimpleNamespace(public_transition_gate=gate),
                ),
            )
        )

        self.assertEqual(_root_public_transition_gate_mean_absolute(scores), 2.0)
        self.assertIsNone(
            _root_public_transition_gate_mean_absolute(
                SimpleNamespace(
                    nodes=(
                        SimpleNamespace(
                            path=(),
                            memory_read=SimpleNamespace(
                                public_transition_gate=None
                            ),
                        ),
                    )
                )
            )
        )

    def test_initial_interface_cannot_change_current_support_logits(self) -> None:
        torch.manual_seed(96_001)
        adapter = SymbolicDemonstrationAdapter()
        pair = make_demonstration_procedure_transfer_stream(
            96_001,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[1]
        public = torch.randn(5, 14)

        self.assertTrue(torch.equal(adapter.encode_public_task(pair.learner, public), public))
        reference = torch.randn(1, 64)
        evidence = adapter.feedback_evidence(pair.learner, reference)
        self.assertAlmostEqual(float(evidence.norm().item()), 1.0, places=5)
        self.assertAlmostEqual(
            float(evidence.mean().item()),
            0.0,
            places=5,
        )

    def test_demonstration_free_query_is_always_bit_exact_identity(self) -> None:
        torch.manual_seed(96_003)
        adapter = SymbolicDemonstrationAdapter()
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.normal_(mean=0.0, std=0.2)
        pair = make_demonstration_procedure_transfer_stream(
            96_003,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).queries[0]
        public = torch.randn(5, 14)

        self.assertTrue(torch.equal(adapter.encode_public_task(pair.learner, public), public))
        reference = torch.randn(1, 64)
        self.assertTrue(
            torch.equal(
                adapter.feedback_evidence(pair.learner, reference),
                torch.zeros_like(reference),
            )
        )

    def test_feedback_evidence_accepts_only_the_matched_query_batch(self) -> None:
        adapter = SymbolicDemonstrationAdapter()
        stream = make_demonstration_procedure_transfer_stream(
            96_005,
            supports_per_procedure=1,
            queries_per_procedure=1,
        )
        batched_reference = torch.randn(3, 64)

        query_evidence = adapter.feedback_evidence(
            stream.queries[0].learner,
            batched_reference,
        )

        self.assertTrue(torch.equal(query_evidence, torch.zeros_like(batched_reference)))
        with self.assertRaisesRegex(ValueError, "supported only for queries"):
            adapter.feedback_evidence(stream.supports[1].learner, batched_reference)
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            adapter.feedback_evidence(stream.queries[0].learner, torch.randn(2, 64))

    def test_tokenizer_preserves_entities_without_canonicalizing_mechanism(self) -> None:
        torch.manual_seed(96_007)
        adapter = SymbolicDemonstrationAdapter().eval()
        with torch.no_grad():
            adapter.evidence_projection.weight.normal_(mean=0.0, std=0.1)
        original_stream = make_demonstration_procedure_transfer_stream(
            96_007,
            supports_per_procedure=1,
            queries_per_procedure=1,
        )
        original_pair = original_stream.supports[1]
        task = original_pair.learner
        public = torch.randn(5, 14)

        reversed_task = replace(task, demonstrations=tuple(reversed(task.demonstrations)))
        raw = adapter._raw_public_entities(
            task,
            device=public.device,
            dtype=public.dtype,
        )
        for demo_index, demonstration in enumerate(task.demonstrations):
            for input_position, symbol in enumerate(demonstration.input_symbols):
                output_position = demonstration.output_symbols.index(symbol)
                self.assertTrue(
                    torch.equal(
                        raw[demo_index, 0, input_position],
                        raw[demo_index, 1, output_position],
                    )
                )

        same_mechanism_fresh_symbols = make_demonstration_procedure_transfer_stream(
            96_008,
            supports_per_procedure=1,
            queries_per_procedure=1,
            position_permutation=(
                original_pair.hidden.source_solution.position_permutation
            ),
        ).supports[1].learner
        fresh_raw = adapter._raw_public_entities(
            same_mechanism_fresh_symbols,
            device=public.device,
            dtype=public.dtype,
        )
        self.assertFalse(torch.equal(raw, fresh_raw))

        reference = torch.randn(1, 64)
        expected = adapter.feedback_evidence(task, reference)
        self.assertTrue(
            torch.allclose(
                adapter.feedback_evidence(reversed_task, reference),
                expected,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_demonstration_port_cannot_change_native_or_precedence_logits(self) -> None:
        torch.manual_seed(96_011)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        precedence = SharedPublicFactAdapter()
        ports = TypedPublicFactPorts(precedence, SymbolicDemonstrationAdapter())
        policy.public_fact_adapter = ports
        state = policy.initial_state(1)
        native = make_skill_memory_meta_partition(
            96_011,
            instances_per_program=8,
        ).tasks[0]
        relational = make_relational_procedure_transfer_stream(
            96_011,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[0]
        native_before = policy.score_task(native.learner, state).logits.detach().clone()
        relational_before = policy.score_task(
            relational.learner,
            state,
        ).logits.detach().clone()

        with torch.no_grad():
            for parameter in ports.demonstration_adapter.parameters():
                parameter.normal_(mean=0.0, std=0.5)

        native_after = policy.score_task(native.learner, state).logits.detach().clone()
        relational_after = policy.score_task(
            relational.learner,
            state,
        ).logits.detach().clone()
        self.assertTrue(torch.equal(native_after, native_before))
        self.assertTrue(torch.equal(relational_after, relational_before))

    def test_public_evidence_changes_only_a_nonzero_residual_write(self) -> None:
        torch.manual_seed(96_012)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        ports = TypedPublicFactPorts(
            SharedPublicFactAdapter(),
            SymbolicDemonstrationAdapter(),
        )
        policy.public_fact_adapter = ports
        writer = _attach_public_evidence_writer(policy)
        pair = make_demonstration_procedure_transfer_stream(
            96_012,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[1]
        wrong_pair = make_demonstration_procedure_transfer_stream(
            96_012,
            supports_per_procedure=1,
            queries_per_procedure=1,
            rotate_demonstration_outputs=1,
        ).supports[1]
        state = policy.initial_state(1)
        before = policy.score_task(pair.learner, state)

        after = policy.score_task(pair.learner, state)

        self.assertTrue(torch.equal(after.logits, before.logits))
        self.assertGreater(float(after.public_feedback_evidence.norm().item()), 0.0)

        proposal = phase5._proposal_for_candidate(
            policy,
            pair.learner,
            state,
            0,
        )
        zero_writer = phase5.propose_differentiable_feedback(
            policy,
            proposal,
            0.5,
            state,
        )
        no_evidence_task = replace(pair.learner, demonstrations=())
        no_evidence_proposal = phase5._proposal_for_candidate(
            policy,
            no_evidence_task,
            state,
            0,
        )
        no_evidence_write = phase5.propose_differentiable_feedback(
            policy,
            no_evidence_proposal,
            0.5,
            state,
        )
        self.assertTrue(
            torch.equal(
                zero_writer.candidate_state.slot_latents,
                no_evidence_write.candidate_state.slot_latents,
            )
        )
        hidden_pair = next(writer.hidden[0].parameters())
        with torch.no_grad():
            hidden_pair.normal_(mean=0.0, std=0.1)
            writer.content_head.weight.normal_(mean=0.0, std=0.1)
            writer.direction_head.weight.normal_(mean=0.0, std=0.1)
        changed_proposal = phase5._proposal_for_candidate(
            policy,
            pair.learner,
            state,
            0,
        )
        changed = phase5.propose_differentiable_feedback(
            policy,
            changed_proposal,
            0.5,
            state,
        )
        wrong_proposal = phase5._proposal_for_candidate(
            policy,
            wrong_pair.learner,
            state,
            0,
        )
        wrong = phase5.propose_differentiable_feedback(
            policy,
            wrong_proposal,
            0.5,
            state,
        )
        self.assertFalse(
            torch.equal(
                changed.candidate_state.slot_latents,
                zero_writer.candidate_state.slot_latents,
            )
        )
        self.assertTrue(torch.equal(changed_proposal.scores.logits, proposal.scores.logits))
        self.assertTrue(
            torch.equal(changed_proposal.scores.logits, wrong_proposal.scores.logits)
        )
        self.assertEqual(changed.write_slot, wrong.write_slot)
        self.assertFalse(
            torch.equal(
                changed.candidate_state.slot_latents,
                wrong.candidate_state.slot_latents,
            )
        )

    def test_train_only_checkpoint_round_trip_precedes_evaluation(self) -> None:
        torch.manual_seed(96_013)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        policy.public_fact_adapter = TypedPublicFactPorts(
            SharedPublicFactAdapter(),
            SymbolicDemonstrationAdapter(),
        )
        _attach_public_evidence_reader(policy)
        adapter_fingerprint = phase5._named_state_fingerprint(
            policy,
            include=lambda name: name.startswith(
                "public_fact_adapter.demonstration_adapter."
            ),
            domain=b"project-angler.demonstration-adapter.v1",
        )
        training = {
            "adapter_fingerprint_after": adapter_fingerprint,
            "mechanism_partition": "train",
            "target_order_used": False,
            "deterministic_solver_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "train-only.pt"
            saved = _save_train_only_adapter(
                destination,
                policy=policy,
                seed=96_013,
                base_checkpoint_sha256="base-sha",
                precedence_adapter_sha256="precedence-sha",
                training=training,
            )
            loaded, reader_state, loaded_training, record = _load_demonstration_adapter(
                destination,
                expected_base_sha256="base-sha",
                expected_precedence_sha256="precedence-sha",
            )

        self.assertEqual(saved["stage"], "train_only")
        self.assertEqual(record["stage"], "train_only")
        self.assertEqual(saved["sha256"], record["sha256"])
        self.assertEqual(loaded_training, training)
        self.assertTrue(reader_state)
        for name, value in policy.public_fact_adapter.demonstration_adapter.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], value))


if __name__ == "__main__":
    unittest.main()
