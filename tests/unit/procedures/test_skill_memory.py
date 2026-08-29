from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.skill_memory import (  # noqa: E402
    PublicEvidenceLatentReader,
    ProceduralSkillState,
    RoutedProceduralMemory,
    differentiable_zero_public_evidence_skill_content,
    permute_procedural_skill_slots,
    public_evidence_skill_content,
    procedural_skill_state_digest,
    restore_procedural_skill_state,
    snapshot_procedural_skill_state,
    zero_public_evidence_skill_content,
    zero_procedural_skill_content,
)
from angler.reasoning.self_referential_memory import (  # noqa: E402
    SelfReferentialState,
)


class RoutedProceduralMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2701)
        torch.set_num_threads(1)
        self.model = RoutedProceduralMemory(
            8,
            slots=4,
            heads=2,
            read_top_k=2,
            hidden_width=16,
        )
        self.states = torch.randn(2, 8)
        self.goals = torch.randn(2, 8)
        self.candidates = torch.randn(2, 4, 8)
        self.mask = torch.tensor(
            (
                (True, True, True, True),
                (True, True, True, False),
            )
        )
        self.attempted = torch.tensor((0, 1), dtype=torch.long)
        self.rewards = torch.tensor((1.0, 0.0))

    @staticmethod
    def _fast_slot(
        state: ProceduralSkillState,
        name: str,
        batch_index: int,
        slot_index: int,
    ) -> torch.Tensor:
        value = getattr(state.fast_weights, name)
        shaped = value.reshape(
            state.batch_size,
            state.slot_count,
            *value.shape[1:],
        )
        return shaped[batch_index, slot_index]

    def test_empty_read_is_exact_zero_read_only_and_capacity_is_fixed(self) -> None:
        state = self.model.initial_state(2)
        before = procedural_skill_state_digest(state)

        read = self.model.read(
            self.states,
            self.goals,
            self.candidates,
            state=state,
            candidate_mask=self.mask,
        )

        self.assertTrue(torch.equal(read.plastic_context, torch.zeros_like(read.plastic_context)))
        self.assertTrue(torch.equal(read.score_bias, torch.zeros_like(read.score_bias)))
        self.assertTrue(torch.equal(read.read_weights, torch.zeros_like(read.read_weights)))
        self.assertEqual(procedural_skill_state_digest(state), before)
        self.assertEqual(state.numel(), self.model.state_numel(2))
        self.assertTrue(
            torch.equal(
                read.write_weights,
                torch.nn.functional.one_hot(read.write_slots, 4).to(torch.float32),
            )
        )

        with torch.no_grad():
            evolved = state
            for index in range(12):
                proposal = self.model.propose_feedback(
                    self.states,
                    self.goals,
                    self.candidates,
                    self.attempted,
                    torch.tensor((float(index % 2), float((index + 1) % 2))),
                    torch.zeros(2, 4),
                    state=evolved,
                    candidate_mask=self.mask,
                )
                evolved = proposal.candidate_state
                self.assertEqual(evolved.numel(), state.numel())
        self.assertEqual(int(evolved.write_counts.sum().item()), 24)

        with self.assertRaisesRegex(ValueError, "must match.*device and dtype"):
            self.model.initial_state(1, dtype=torch.float64)

    def test_opaque_route_address_does_not_contaminate_written_content(self) -> None:
        state = self.model.initial_state(1)
        common = {
            "state_embeddings": self.states[:1],
            "candidate_embeddings": self.candidates[:1],
            "attempted_indices": self.attempted[:1],
            "reward": self.rewards[:1],
            "base_logits": torch.zeros(1, 4),
            "state": state,
            "candidate_mask": self.mask[:1],
        }
        first = self.model.propose_feedback(
            goal_embeddings=self.goals[:1],
            **common,
        )
        second = self.model.propose_feedback(
            goal_embeddings=-self.goals[:1],
            **common,
        )
        first_slot = int(first.write_slots.item())
        second_slot = int(second.write_slots.item())

        self.assertFalse(torch.allclose(first.read.route_key, second.read.route_key))
        self.assertTrue(
            torch.equal(
                first.candidate_state.slot_latents[0, first_slot],
                second.candidate_state.slot_latents[0, second_slot],
            )
        )

    def test_scalar_outcome_has_a_dedicated_learned_content_path(self) -> None:
        state = self.model.initial_state(1)
        common = {
            "state_embeddings": self.states[:1],
            "goal_embeddings": self.goals[:1],
            "candidate_embeddings": self.candidates[:1],
            "attempted_indices": self.attempted[:1],
            "base_logits": torch.zeros(1, 4),
            "state": state,
            "candidate_mask": self.mask[:1],
        }
        low = self.model.propose_feedback(
            reward=torch.tensor((0.0,)),
            **common,
        )
        high = self.model.propose_feedback(
            reward=torch.tensor((1.0,)),
            **common,
        )
        neutral = self.model.propose_feedback(
            reward=torch.tensor((0.5,)),
            **common,
        )
        split = self.model.evidence_content_width
        self.assertEqual(int(low.write_slots.item()), int(high.write_slots.item()))
        self.assertTrue(
            torch.equal(
                low.feedback_event[:, :split],
                high.feedback_event[:, :split],
            )
        )
        self.assertTrue(
            torch.equal(
                low.feedback_event[:, split:],
                -high.feedback_event[:, split:],
            )
        )
        self.assertTrue(
            torch.equal(
                neutral.feedback_event[:, :split],
                low.feedback_event[:, :split],
            )
        )
        self.assertTrue(
            torch.equal(
                neutral.feedback_event[:, split:],
                torch.zeros_like(neutral.feedback_event[:, split:]),
            )
        )
        self.assertGreater(
            float((low.feedback_event - high.feedback_event).norm().item()),
            1e-4,
        )
        self.assertFalse(
            torch.allclose(
                low.candidate_state.slot_latents,
                high.candidate_state.slot_latents,
            )
        )

    def test_canonical_outcome_basis_bypasses_context_direction_only(self) -> None:
        state = self.model.initial_state(1)
        basis = torch.linspace(
            -0.75,
            0.75,
            self.model.evidence_outcome_width,
        ).unsqueeze(0)
        common = {
            "goal_embeddings": self.goals[:1],
            "attempted_indices": self.attempted[:1],
            "base_logits": torch.zeros(1, 4),
            "state": state,
            "candidate_mask": self.mask[:1],
            "outcome_direction_basis": basis,
        }
        high = self.model.propose_feedback(
            self.states[:1],
            candidate_embeddings=self.candidates[:1],
            reward=torch.tensor((1.0,)),
            **common,
        )
        low = self.model.propose_feedback(
            self.states[:1] + 7.0,
            candidate_embeddings=self.candidates[:1] - 5.0,
            reward=torch.tensor((0.0,)),
            **common,
        )
        neutral = self.model.propose_feedback(
            self.states[:1],
            candidate_embeddings=self.candidates[:1],
            reward=torch.tensor((0.5,)),
            **common,
        )
        split = self.model.evidence_content_width
        self.assertTrue(torch.equal(high.feedback_event[:, split:], basis))
        self.assertTrue(torch.equal(low.feedback_event[:, split:], -basis))
        self.assertTrue(
            torch.equal(
                neutral.feedback_event[:, split:],
                torch.zeros_like(basis),
            )
        )
        self.assertFalse(
            torch.equal(
                high.feedback_event[:, :split],
                low.feedback_event[:, :split],
            )
        )
        differentiable_basis = basis.clone().requires_grad_(True)
        detached = self.model.propose_feedback(
            self.states[:1],
            candidate_embeddings=self.candidates[:1],
            reward=torch.tensor((1.0,)),
            **{
                **common,
                "outcome_direction_basis": differentiable_basis,
            },
        )
        (basis_gradient,) = torch.autograd.grad(
            detached.feedback_event[:, split:].sum(),
            differentiable_basis,
            allow_unused=True,
        )
        self.assertIsNone(basis_gradient)
        with self.assertRaisesRegex(ValueError, "outcome_direction_basis"):
            self.model.propose_feedback(
                self.states[:1],
                self.goals[:1],
                self.candidates[:1],
                self.attempted[:1],
                torch.tensor((1.0,)),
                torch.zeros(1, 4),
                state=state,
                candidate_mask=self.mask[:1],
                outcome_direction_basis=torch.full_like(basis, 1.01),
            )

    def test_feedback_set_mean_and_query_are_order_invariant(self) -> None:
        records = tuple(
            (
                self.states[:1] + 0.07 * index,
                self.candidates[:1].roll(index, dims=1),
                torch.tensor(((index + 1) % 4,), dtype=torch.long),
                torch.tensor((reward,)),
            )
            for index, reward in enumerate((0.0, 0.5, 1.0, 0.4))
        )

        def accumulate(order: tuple[int, ...]):
            state = self.model.initial_state(1)
            events = []
            slots = []
            for index in order:
                states, candidates, attempted, reward = records[index]
                proposal = self.model.propose_feedback(
                    states,
                    self.goals[:1],
                    candidates,
                    attempted,
                    reward,
                    torch.zeros(1, 4),
                    state=state,
                    candidate_mask=self.mask[:1],
                )
                events.append(proposal.feedback_event)
                slots.append(int(proposal.write_slots.item()))
                state = proposal.candidate_state
            read = self.model.read(
                self.states[:1] + 0.33,
                self.goals[:1],
                self.candidates[:1],
                state=state,
                candidate_mask=self.mask[:1],
            )
            return state, read, events, slots

        forward, forward_read, events, slots = accumulate((0, 1, 2, 3))
        reverse, reverse_read, _, reverse_slots = accumulate((3, 2, 1, 0))
        slot = slots[0]

        self.assertEqual(len(set(slots + reverse_slots)), 1)
        self.assertEqual(int(forward.write_counts[0, slot].item()), 4)
        self.assertEqual(int(reverse.write_counts[0, slot].item()), 4)
        self.assertTrue(
            torch.allclose(
                forward.slot_latents[0, slot],
                torch.cat(events, dim=0).mean(dim=0),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                forward.slot_latents,
                reverse.slot_latents,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                forward_read.score_bias,
                reverse_read.score_bias,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            self.assertTrue(
                torch.equal(
                    getattr(forward.fast_weights, name),
                    getattr(self.model.initial_state(1).fast_weights, name),
                )
            )

    def test_snapshot_restore_and_digest_preserve_exact_independent_state(self) -> None:
        state = self.model.initial_state(2)
        proposal = self.model.propose_feedback(
            self.states,
            self.goals,
            self.candidates,
            self.attempted,
            self.rewards,
            torch.zeros(2, 4),
            state=state,
            candidate_mask=self.mask,
        )
        snapshot = snapshot_procedural_skill_state(proposal.candidate_state)
        restored = restore_procedural_skill_state(snapshot)

        self.assertEqual(
            procedural_skill_state_digest(restored),
            procedural_skill_state_digest(proposal.candidate_state),
        )
        for name, tensor in snapshot.items():
            restored_tensor = (
                getattr(restored.fast_weights, name.removeprefix("fast."))
                if name.startswith("fast.")
                else getattr(restored, name)
            )
            self.assertTrue(torch.equal(tensor, restored_tensor), name)
            self.assertNotEqual(tensor.data_ptr(), restored_tensor.data_ptr(), name)

    def test_zero_content_preserves_routing_metadata_but_erases_read_effect(self) -> None:
        state = self.model.initial_state(1)
        proposal = self.model.propose_feedback(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            torch.tensor((0,), dtype=torch.long),
            torch.tensor((1.0,)),
            torch.zeros(1, 4),
            state=state,
            candidate_mask=self.mask[:1],
        )
        populated = proposal.candidate_state
        erased = zero_procedural_skill_content(populated)

        self.assertTrue(torch.equal(erased.key_offsets, populated.key_offsets))
        self.assertTrue(torch.equal(erased.occupied, populated.occupied))
        self.assertTrue(torch.equal(erased.write_counts, populated.write_counts))
        self.assertTrue(
            torch.equal(erased.slot_latents, torch.zeros_like(erased.slot_latents))
        )
        for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
            value = getattr(erased.fast_weights, name)
            self.assertTrue(torch.equal(value, torch.zeros_like(value)), name)
        read = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=erased,
            candidate_mask=self.mask[:1],
        )
        self.assertTrue(torch.equal(read.plastic_context, torch.zeros_like(read.plastic_context)))
        self.assertTrue(torch.equal(read.score_bias, torch.zeros_like(read.score_bias)))

    def test_feedback_changes_exactly_one_slot_and_reset_swap_are_causal(self) -> None:
        state = self.model.initial_state(1)
        states = self.states[:1]
        goals = self.goals[:1]
        candidates = self.candidates[:1]
        mask = self.mask[:1]
        proposal = self.model.propose_feedback(
            states,
            goals,
            candidates,
            torch.tensor((0,), dtype=torch.long),
            torch.tensor((1.0,)),
            torch.zeros(1, 4),
            state=state,
            candidate_mask=mask,
        )
        selected = int(proposal.write_slots.item())

        self.assertGreater(float(proposal.delta_norm.item()), 0.0)
        self.assertTrue(proposal.candidate_state.occupied[0, selected])
        self.assertEqual(int(proposal.candidate_state.write_counts[0, selected]), 1)
        for slot in range(state.slot_count):
            changed = slot == selected
            self.assertEqual(
                not torch.equal(
                    state.slot_latents[0, slot],
                    proposal.candidate_state.slot_latents[0, slot],
                ),
                changed,
            )
            self.assertEqual(
                not torch.equal(
                    state.key_offsets[0, slot],
                    proposal.candidate_state.key_offsets[0, slot],
                ),
                changed,
            )
            for name in ("delta_y", "delta_q", "delta_k", "delta_beta"):
                self.assertTrue(
                    torch.equal(
                        self._fast_slot(state, name, 0, slot),
                        self._fast_slot(proposal.candidate_state, name, 0, slot),
                    ),
                    f"reserved recurrent tensor changed in slot {slot}: {name}",
                )

        fresh_states = states + 0.31
        fresh_goals = goals
        reset = self.model.read(
            fresh_states,
            fresh_goals,
            candidates,
            state=self.model.initial_state(1),
            candidate_mask=mask,
        )
        adapted = self.model.read(
            fresh_states,
            fresh_goals,
            candidates,
            state=proposal.candidate_state,
            candidate_mask=mask,
        )
        self.assertTrue(torch.equal(reset.score_bias, torch.zeros_like(reset.score_bias)))
        self.assertFalse(torch.equal(adapted.plastic_context, reset.plastic_context))
        self.assertFalse(torch.equal(adapted.score_bias, reset.score_bias))

        receiver = copy.deepcopy(self.model)
        swapped = receiver.read(
            fresh_states,
            fresh_goals,
            candidates,
            state=proposal.candidate_state,
            candidate_mask=mask,
        )
        self.assertTrue(torch.equal(swapped.plastic_context, adapted.plastic_context))
        self.assertTrue(torch.equal(swapped.score_bias, adapted.score_bias))

    def test_public_evidence_uses_an_exchangeable_isolated_state_channel(self) -> None:
        reader = PublicEvidenceLatentReader(8, hidden_width=4)
        baseline_state = self.model.initial_state(1)
        baseline_read = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=baseline_state,
            candidate_mask=self.mask[:1],
        )
        self.model.public_evidence_reader = reader
        attached_read = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=baseline_state,
            candidate_mask=self.mask[:1],
            include_public_evidence=True,
        )
        self.assertTrue(torch.equal(attached_read.plastic_context, baseline_read.plastic_context))
        self.assertTrue(torch.equal(attached_read.score_bias, baseline_read.score_bias))

        evidence_a = torch.nn.functional.normalize(
            torch.arange(1, 9, dtype=torch.float32).unsqueeze(0),
            dim=-1,
        )
        evidence_b = torch.nn.functional.normalize(
            torch.arange(8, 0, -1, dtype=torch.float32).unsqueeze(0),
            dim=-1,
        )

        def write_public(
            state: ProceduralSkillState,
            evidence: torch.Tensor,
        ) -> ProceduralSkillState:
            return self.model.propose_feedback(
                self.states[:1],
                self.goals[:1],
                self.candidates[:1],
                self.attempted[:1],
                self.rewards[:1],
                torch.zeros(1, 4),
                state=state,
                candidate_mask=self.mask[:1],
                public_evidence=evidence,
                include_public_evidence=True,
            ).candidate_state

        forward_first = write_public(baseline_state, evidence_a)
        forward = write_public(forward_first, evidence_b)
        reverse_first = write_public(baseline_state, evidence_b)
        reverse = write_public(reverse_first, evidence_a)
        forward_codes, forward_counts = public_evidence_skill_content(forward)
        reverse_codes, reverse_counts = public_evidence_skill_content(reverse)
        selected = int(
            self.model.read(
                self.states[:1],
                self.goals[:1],
                self.candidates[:1],
                state=baseline_state,
                candidate_mask=self.mask[:1],
                include_public_evidence=True,
            ).write_slots.item()
        )
        self.assertEqual(float(forward_counts[0, selected].item()), 2.0)
        self.assertTrue(torch.equal(forward_counts, reverse_counts))
        self.assertTrue(
            torch.allclose(
                forward_codes[0, selected],
                torch.cat((evidence_a, evidence_b), dim=0).mean(dim=0),
                atol=1e-7,
                rtol=1e-7,
            )
        )
        self.assertTrue(torch.allclose(forward_codes, reverse_codes, atol=1e-7, rtol=1e-7))
        for slot in range(baseline_state.slot_count):
            changed = slot == selected
            self.assertEqual(
                not torch.equal(
                    self._fast_slot(
                        baseline_state,
                        "delta_beta",
                        0,
                        slot,
                    ),
                    self._fast_slot(
                        forward_first,
                        "delta_beta",
                        0,
                        slot,
                    ),
                ),
                changed,
            )
            for name in ("delta_y", "delta_q", "delta_k"):
                self.assertTrue(
                    torch.equal(
                        self._fast_slot(baseline_state, name, 0, slot),
                        self._fast_slot(forward_first, name, 0, slot),
                    )
                )

        zero_after_public = write_public(forward, torch.zeros_like(evidence_a))
        zero_codes, zero_counts = public_evidence_skill_content(zero_after_public)
        self.assertTrue(torch.equal(zero_codes, forward_codes))
        self.assertTrue(torch.equal(zero_counts, forward_counts))

        with torch.no_grad():
            torch.nn.init.normal_(reader.output.weight, mean=0.0, std=0.2)
        live_read = self.model.read(
            self.states[:1] + 0.13,
            self.goals[:1],
            self.candidates[:1],
            state=forward,
            candidate_mask=self.mask[:1],
            include_public_evidence=True,
        )
        erased = zero_public_evidence_skill_content(forward)
        erased_codes, erased_counts = public_evidence_skill_content(erased)
        erased_read = self.model.read(
            self.states[:1] + 0.13,
            self.goals[:1],
            self.candidates[:1],
            state=erased,
            candidate_mask=self.mask[:1],
            include_public_evidence=True,
        )
        self.assertTrue(torch.equal(erased_codes, torch.zeros_like(erased_codes)))
        self.assertTrue(torch.equal(erased_counts, forward_counts))
        self.assertFalse(torch.equal(live_read.plastic_context, erased_read.plastic_context))

    def test_public_reader_cannot_use_base_context_as_a_presence_shortcut(self) -> None:
        torch.manual_seed(91_211)
        reader = PublicEvidenceLatentReader(8, hidden_width=4)
        with torch.no_grad():
            reader.hidden[0].weight.zero_()
            reader.hidden[0].weight[:, 8:].normal_(mean=0.0, std=0.2)
            reader.output.weight.normal_(mean=0.0, std=0.2)
            reader.transition_output.weight.normal_(mean=0.0, std=0.2)
        public = torch.randn(3, 8)
        base = torch.randn(3, 8)

        base_only, base_only_gate = reader.read_effects(public, base)

        self.assertTrue(torch.equal(base_only, torch.zeros_like(base_only)))
        self.assertTrue(
            torch.equal(base_only_gate, torch.zeros_like(base_only_gate))
        )

        with torch.no_grad():
            reader.hidden[0].weight[:, :8].normal_(mean=0.0, std=0.2)
        content_dependent = reader(public, base)
        zero_public = reader(torch.zeros_like(public), base)
        _, transition_gate = reader.read_effects(public, base)
        _, zero_transition_gate = reader.read_effects(torch.zeros_like(public), base)
        low_confidence = reader(
            public,
            base,
            public_confidence=torch.full((3, 1), 0.25),
        )
        high_confidence = reader(
            public,
            base,
            public_confidence=torch.full((3, 1), 0.75),
        )
        self.assertFalse(torch.equal(content_dependent, torch.zeros_like(content_dependent)))
        self.assertTrue(torch.equal(zero_public, torch.zeros_like(zero_public)))
        self.assertFalse(torch.equal(transition_gate, torch.zeros_like(transition_gate)))
        self.assertTrue(
            torch.equal(zero_transition_gate, torch.zeros_like(zero_transition_gate))
        )
        self.assertFalse(torch.equal(low_confidence, high_confidence))

    def test_differentiable_public_ablation_preserves_only_the_base_graph(self) -> None:
        state = self.model.initial_state(1)
        base_marker = torch.tensor(0.25, requires_grad=True)
        public_marker = torch.tensor(0.5, requires_grad=True)
        beta = state.fast_weights.delta_beta.clone()
        beta[..., 0] = beta[..., 0] + public_marker
        marked = ProceduralSkillState(
            fast_weights=SelfReferentialState(
                delta_y=state.fast_weights.delta_y,
                delta_q=state.fast_weights.delta_q,
                delta_k=state.fast_weights.delta_k,
                delta_beta=beta,
            ),
            slot_latents=state.slot_latents + base_marker,
            key_offsets=state.key_offsets,
            occupied=state.occupied,
            write_counts=state.write_counts,
        )

        ablated = differentiable_zero_public_evidence_skill_content(marked)
        codes, _ = public_evidence_skill_content(ablated)
        loss = ablated.slot_latents.sum() + codes.sum()
        base_gradient, public_gradient = torch.autograd.grad(
            loss,
            (base_marker, public_marker),
            allow_unused=True,
        )
        self.assertGreater(float(base_gradient.abs().item()), 0.0)
        self.assertTrue(
            public_gradient is None
            or torch.equal(public_gradient, torch.zeros_like(public_gradient))
        )

    def test_top_k_read_combines_occupied_slots(self) -> None:
        state = self.model.initial_state(1)
        fast = state.fast_weights
        latents = state.slot_latents.clone()
        latents[:, 0] += 0.15
        latents[:, 1] -= 0.08
        occupied = state.occupied.clone()
        occupied[:, :2] = True
        counts = state.write_counts.clone()
        counts[:, :2] = 1
        route = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=state,
            candidate_mask=self.mask[:1],
        )
        anchors = torch.nn.functional.normalize(
            self.model.slot_anchors.detach(), dim=-1
        )
        key_offsets = state.key_offsets.clone()
        key_offsets[:, :2] = route.route_key.unsqueeze(1) - anchors[:2].unsqueeze(0)
        populated = ProceduralSkillState(
            fast_weights=fast,
            slot_latents=latents,
            key_offsets=key_offsets,
            occupied=occupied,
            write_counts=counts,
        )

        read = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=populated,
            candidate_mask=self.mask[:1],
        )

        self.assertEqual(int((read.read_weights > 0).sum().item()), 2)
        self.assertTrue(torch.allclose(read.read_weights.sum(dim=-1), torch.ones(1)))
        self.assertFalse(torch.equal(read.plastic_context, torch.zeros_like(read.plastic_context)))

    def test_no_effect_transaction_rejects_with_exact_rollback(self) -> None:
        state = self.model.initial_state(2)
        before = procedural_skill_state_digest(state)
        slow_before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }

        proposal = self.model.propose_feedback(
            self.states,
            self.goals,
            self.candidates,
            self.attempted,
            self.rewards,
            torch.randn(2, 4),
            state=state,
            candidate_mask=self.mask,
        )
        write = self.model.commit_bounded_feedback(
            proposal,
            minimum_effect=float(proposal.delta_norm.max().item()) + 1.0,
        )

        self.assertFalse(bool(write.accepted.any().item()))
        self.assertIs(write.state, state)
        self.assertEqual(procedural_skill_state_digest(write.state), before)
        self.assertTrue(torch.equal(write.delta_norm, torch.zeros_like(write.delta_norm)))
        self.assertTrue(bool(torch.isfinite(write.before_loss).all().item()))
        self.assertTrue(bool(torch.isfinite(write.after_loss).all().item()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, slow_before[name]), name)

    def test_slot_and_candidate_permutations_are_equivariant(self) -> None:
        state = self.model.initial_state(1)
        proposal = self.model.propose_feedback(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            torch.tensor((0,), dtype=torch.long),
            torch.tensor((1.0,)),
            torch.zeros(1, 4),
            state=state,
            candidate_mask=self.mask[:1],
        )
        state = proposal.candidate_state
        original = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=state,
            candidate_mask=self.mask[:1],
        )

        slot_order = (2, 0, 3, 1)
        permuted_model = copy.deepcopy(self.model)
        with torch.no_grad():
            permuted_model.slot_anchors.copy_(
                self.model.slot_anchors.detach()[list(slot_order)]
            )
        permuted_state = permute_procedural_skill_slots(state, slot_order)
        permuted = permuted_model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1],
            state=permuted_state,
            candidate_mask=self.mask[:1],
        )

        self.assertTrue(torch.allclose(permuted.plastic_context, original.plastic_context))
        self.assertTrue(torch.allclose(permuted.score_bias, original.score_bias))
        self.assertTrue(
            torch.allclose(
                permuted.read_weights,
                original.read_weights[:, list(slot_order)],
            )
        )
        self.assertTrue(
            torch.allclose(
                permuted.write_weights,
                original.write_weights[:, list(slot_order)],
            )
        )

        candidate_order = (2, 0, 3, 1)
        candidate_permuted = self.model.read(
            self.states[:1],
            self.goals[:1],
            self.candidates[:1, list(candidate_order)],
            state=state,
            candidate_mask=self.mask[:1, list(candidate_order)],
        )
        self.assertTrue(torch.allclose(candidate_permuted.route_query, original.route_query))
        self.assertTrue(torch.allclose(candidate_permuted.read_weights, original.read_weights))
        self.assertTrue(
            torch.allclose(
                candidate_permuted.score_bias,
                original.score_bias[:, list(candidate_order)],
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_later_query_meta_gradient_crosses_earlier_feedback_write(self) -> None:
        torch.manual_seed(2702)
        model = RoutedProceduralMemory(
            8,
            slots=4,
            heads=2,
            read_top_k=2,
            hidden_width=16,
        ).double()
        states = torch.randn(1, 8, dtype=torch.double)
        goals = torch.randn(1, 8, dtype=torch.double)
        candidates = torch.randn(1, 4, 8, dtype=torch.double)
        mask = torch.ones(1, 4, dtype=torch.bool)
        write = None
        for attempted in range(4):
            for reward in (0.0, 1.0):
                proposal = model.propose_feedback(
                    states,
                    goals,
                    candidates,
                    torch.tensor((attempted,), dtype=torch.long),
                    torch.tensor((reward,), dtype=torch.double),
                    torch.zeros(1, 4, dtype=torch.double),
                    state=model.initial_state(1),
                    candidate_mask=mask,
                )
                candidate = model.admit_feedback(proposal)
                if bool(candidate.accepted.item()):
                    write = candidate
                    break
            if write is not None:
                break
        self.assertIsNotNone(write)
        assert write is not None
        self.assertTrue(bool(write.accepted.item()))
        later = model.read(
            states + 0.17,
            goals,
            candidates.roll(1, dims=1),
            state=write.state,
            candidate_mask=mask,
        )
        loss = torch.nn.functional.cross_entropy(
            later.score_bias,
            torch.tensor((2,), dtype=torch.long),
        )
        parameters = (
            model.feedback_encoder[-1].weight,
            model.utility_decoder[-1].weight,
        )
        gradients = torch.autograd.grad(loss, parameters)

        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().sum().item()), 1e-12)

    def test_rejected_transaction_restores_state_and_has_no_feedback_gradient(self) -> None:
        model = RoutedProceduralMemory(
            8,
            slots=4,
            heads=2,
            read_top_k=2,
            hidden_width=16,
        ).double()
        states = torch.randn(1, 8, dtype=torch.double)
        goals = torch.randn(1, 8, dtype=torch.double)
        candidates = torch.randn(1, 4, 8, dtype=torch.double)
        mask = torch.ones(1, 4, dtype=torch.bool)
        incoming = model.initial_state(1)
        proposal = model.propose_feedback(
            states,
            goals,
            candidates,
            torch.tensor((0,), dtype=torch.long),
            torch.tensor((1.0,), dtype=torch.double),
            torch.zeros(1, 4, dtype=torch.double),
            state=incoming,
            candidate_mask=mask,
        )
        write = model.admit_feedback(proposal, minimum_improvement=1.0e6)
        self.assertFalse(bool(write.accepted.item()))
        self.assertIs(write.state, incoming)

        later = model.read(
            states + 0.1,
            goals - 0.1,
            candidates,
            state=write.state,
            candidate_mask=mask,
        )
        loss = torch.nn.functional.cross_entropy(
            later.score_bias,
            torch.tensor((1,), dtype=torch.long),
        )
        (gradient,) = torch.autograd.grad(
            loss,
            (model.feedback_encoder[-1].weight,),
            allow_unused=True,
        )
        self.assertTrue(
            gradient is None or torch.equal(gradient, torch.zeros_like(gradient))
        )

    def test_proposal_binds_base_logits_and_admission_accepts_no_replacement(self) -> None:
        state = self.model.initial_state(2)
        base = torch.randn(2, 4)
        proposal = self.model.propose_feedback(
            self.states,
            self.goals,
            self.candidates,
            self.attempted,
            self.rewards,
            base,
            state=state,
            candidate_mask=self.mask,
        )
        bound = proposal.base_logits.detach().clone()
        base.add_(1000.0)
        self.assertTrue(torch.equal(proposal.base_logits, bound))
        self.assertNotIn("base_logits", inspect.signature(self.model.admit_feedback).parameters)

        with self.assertRaisesRegex(ValueError, "base_logits"):
            self.model.propose_feedback(
                self.states,
                self.goals,
                self.candidates,
                self.attempted,
                self.rewards,
                torch.zeros(2, 3),
                state=state,
                candidate_mask=self.mask,
            )

    def test_public_update_api_has_no_external_routing_identity(self) -> None:
        forbidden = {
            "task",
            "domain",
            "family",
            "episode",
            "adapter",
            "namespace",
            "split",
            "mechanism",
            "slot",
        }
        for method in (
            self.model.read,
            self.model.propose_feedback,
            self.model.incorporate_feedback,
        ):
            names = set(inspect.signature(method).parameters)
            self.assertFalse(names & forbidden, method.__name__)


if __name__ == "__main__":
    unittest.main()
