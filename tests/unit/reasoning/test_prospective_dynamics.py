from __future__ import annotations

from dataclasses import replace
import json
import unittest

import torch

from angler.reasoning.prospective_dynamics import (
    CompositeProspectiveCreditCore,
    CompositeProspectiveCreditEvent,
    ProspectiveDynamicsConfig,
    ProspectiveResourceBudget,
)
from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditMemoryCore,
)


def _core(*, temporal_width: int = 5, latent_width: int = 8):
    torch.manual_seed(20260831 + temporal_width + latent_width)
    credit = StructureKeyedCreditMemoryCore(temporal_width=temporal_width)
    return CompositeProspectiveCreditCore(
        credit,
        ProspectiveDynamicsConfig(
            temporal_width=temporal_width,
            latent_width=latent_width,
            world_slots=6,
            maximum_world_reads=3,
            maximum_branches=64,
            maximum_recurrent_steps=4,
        ),
    )


def _observations(batch: int, temporal_width: int = 5):
    relation = torch.stack(
        tuple(
            torch.linspace(-1.0, 1.0, 64) + float(index) / 13.0
            for index in range(batch)
        )
    )
    temporal = torch.stack(
        tuple(
            torch.tensor(
                [
                    float((index + 1) * (offset + 2)) / 17.0
                    for offset in range(temporal_width)
                ]
            )
            for index in range(batch)
        )
    )
    base = torch.linspace(-0.5, 0.75, batch)
    return relation, temporal, base


class ProspectiveDynamicsTests(unittest.TestCase):
    def test_initial_composite_is_exact_credit_control_for_variable_batches(self) -> None:
        for temporal_width, latent_width in ((3, 4), (5, 8), (11, 17)):
            for count in (2, 7, 64):
                with self.subTest(
                    temporal_width=temporal_width,
                    latent_width=latent_width,
                    count=count,
                ):
                    core = _core(
                        temporal_width=temporal_width,
                        latent_width=latent_width,
                    )
                    state = core.initial_state()
                    relation, temporal, base = _observations(count, temporal_width)
                    composite = core.predict(
                        relation,
                        temporal,
                        base,
                        state=state,
                    )
                    control = core.credit_core.predict(
                        relation,
                        temporal,
                        base,
                        state=state.credit_state,
                    )
                    self.assertTrue(torch.equal(composite.logits, control.logits))
                    self.assertTrue(torch.equal(composite.residuals, control.residuals))
                    self.assertTrue(torch.equal(composite.logits, base))
                    self.assertTrue(
                        torch.equal(
                            core._one_world_prospect(relation, temporal, state)
                            .selection_residuals,
                            torch.zeros_like(base),
                        )
                    )

                    permutation = torch.arange(count - 1, -1, -1)
                    permuted = core.predict(
                        relation[permutation],
                        temporal[permutation],
                        base[permutation],
                        state=state,
                    )
                    self.assertTrue(
                        torch.equal(permuted.logits, composite.logits[permutation])
                    )
                    chosen = count // 2
                    single = core.predict(
                        relation[chosen : chosen + 1],
                        temporal[chosen : chosen + 1],
                        base[chosen : chosen + 1],
                        state=state,
                    )
                    self.assertTrue(
                        torch.equal(single.logits, composite.logits[chosen : chosen + 1])
                    )

    def test_same_call_full_output_integrity_and_declared_residual_lesion(self) -> None:
        core = _core()
        state = core.initial_state()
        relation, temporal, base = _observations(1)
        parent_output = core.predict(relation, temporal, base, state=state)
        state, _event = core.apply_feedback(
            state,
            parent_output,
            torch.tensor((1.0,)),
            evidence_refs="evidence://same-call-parent",
        )

        relation, temporal, base = _observations(7)
        combined, prospective = core.predict_with_prospective(
            relation,
            temporal,
            base,
            state=state,
        )
        self.assertTrue(
            torch.equal(combined.logits, core.predict(relation, temporal, base, state=state).logits)
        )
        credit = core.credit_core.predict(
            relation,
            temporal,
            base,
            state=state.credit_state,
        )
        lesioned, retained = core.predict_with_prospective(
            relation,
            temporal,
            base,
            state=state,
            prospective_read_enabled=False,
        )
        self.assertTrue(torch.equal(lesioned.residuals, credit.residuals))
        self.assertTrue(torch.equal(lesioned.logits, credit.logits))
        for name in (
            "future_latents",
            "outcome_logits",
            "uncertainties",
            "selection_residuals",
            "focus_weights",
            "focused_mask",
        ):
            self.assertTrue(torch.equal(getattr(retained, name), getattr(prospective, name)))
        self.assertEqual(retained.recurrent_steps, prospective.recurrent_steps)
        self.assertEqual(retained.state_step, prospective.state_step)

        integrity = core.component_state_integrity(state)
        self.assertEqual(integrity.step, state.step)
        self.assertEqual(integrity.checkpoint_ref, core.checkpoint_identity)
        self.assertEqual(integrity.config_ref, "sha256:" + core.config.digest)
        for value in (
            integrity.world_state_digest,
            integrity.self_state_digest,
            integrity.focus_state_digest,
            integrity.outcome_state_digest,
        ):
            self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

        world = state.world_hypotheses.clone()
        usage = state.world_usage.clone()
        world[1, 0] = 0.321
        usage[1] = 0.5
        world_integrity = core.component_state_integrity(
            replace(state, world_hypotheses=world, world_usage=usage)
        )
        self.assertNotEqual(world_integrity.world_state_digest, integrity.world_state_digest)
        self.assertEqual(world_integrity.self_state_digest, integrity.self_state_digest)
        self.assertEqual(world_integrity.focus_state_digest, integrity.focus_state_digest)
        self.assertEqual(world_integrity.outcome_state_digest, integrity.outcome_state_digest)

    def test_component_focus_respects_eligibility_budgets_and_dormant_state(self) -> None:
        core = _core()
        initial = core.initial_state()
        world_state = initial.world_hypotheses.clone()
        self_state = initial.self_hypotheses.clone()
        usage = initial.world_usage.clone()
        world_state[4] = 0.75
        self_state[4] = -0.25
        usage[4] = 1.0
        state = replace(
            initial,
            world_hypotheses=world_state,
            self_hypotheses=self_state,
            world_usage=usage,
        )
        relation, _, _ = _observations(2)
        worlds = torch.stack(
            (
                torch.stack(
                    tuple(torch.full((5,), float(index + 1) / 10.0) for index in range(4))
                ),
                torch.stack(
                    tuple(torch.full((5,), -float(index + 1) / 11.0) for index in range(4))
                ),
            )
        )
        eligible = torch.tensor(
            ((True, False, True, False), (False, True, False, True))
        )
        output = core.prospect(
            relation,
            worlds,
            eligible,
            state=state,
            budget=ProspectiveResourceBudget(1, 2, 3),
        )
        self.assertEqual(output.focus_weights.shape, (2, 4))
        self.assertTrue(torch.equal(output.focused_mask & ~eligible, torch.zeros_like(eligible)))
        self.assertTrue(
            torch.equal(
                output.focus_weights.masked_select(~eligible),
                torch.zeros_like(output.focus_weights.masked_select(~eligible)),
            )
        )
        torch.testing.assert_close(
            output.focus_weights.sum(dim=-1),
            torch.ones(2),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertTrue(torch.equal(state.world_hypotheses[4], torch.full((8,), 0.75)))
        restored = core.restore_state(core.capture_state(state))
        self.assertTrue(torch.equal(restored.world_hypotheses[4], state.world_hypotheses[4]))
        self.assertTrue(torch.equal(restored.self_hypotheses[4], state.self_hypotheses[4]))
        self.assertEqual(restored.world_usage[4].item(), 1.0)
        report = core.parameter_report()
        self.assertTrue(report["component_only_multiworld"])
        self.assertFalse(report["runtime_multiworld_context"])
        self.assertFalse(report["authorization_inputs"])

        with self.assertRaisesRegex(ValueError, "world-read budget"):
            core.prospect(
                relation,
                worlds,
                eligible,
                state=state,
                budget=ProspectiveResourceBudget(3, 2, 1),
            )
        bad_worlds = worlds.clone()
        bad_worlds[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            core.prospect(
                relation,
                bad_worlds,
                eligible,
                state=state,
                budget=ProspectiveResourceBudget(1, 2, 1),
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            core.prospect(
                relation,
                worlds,
                eligible.to(torch.float32),
                state=state,
                budget=ProspectiveResourceBudget(1, 2, 1),
            )

    def test_outcomes_diverge_composite_state_and_action_condition_predictions(self) -> None:
        core = _core()
        initial = core.initial_state()
        relation, temporal, base = _observations(1)
        prediction = core.predict(relation, temporal, base, state=initial)
        positive, positive_event = core.apply_feedback(
            initial,
            prediction,
            torch.ones(1),
            evidence_refs="evidence://positive",
        )
        negative, negative_event = core.apply_feedback(
            initial,
            prediction,
            -torch.ones(1),
            evidence_refs="evidence://negative",
        )
        self.assertEqual(positive.step, 1)
        self.assertEqual(negative.step, 1)
        self.assertEqual(positive.credit_state.step, 1)
        self.assertNotEqual(core.state_digest(positive), core.state_digest(negative))
        self.assertFalse(torch.equal(positive.world_hypotheses, negative.world_hypotheses))
        self.assertFalse(torch.equal(positive.self_hypotheses, negative.self_hypotheses))
        self.assertFalse(torch.equal(positive.focus_hypothesis, negative.focus_hypothesis))
        self.assertFalse(torch.equal(positive.outcome_hypothesis, negative.outcome_hypothesis))
        self.assertLess(
            positive_event.outcome_loss_after,
            positive_event.outcome_loss_before,
        )
        self.assertLess(
            negative_event.outcome_loss_after,
            negative_event.outcome_loss_before,
        )
        self.assertEqual(positive_event.evidence_refs, ("evidence://positive",))
        self.assertEqual(negative_event.evidence_refs, ("evidence://negative",))

        later_relation, later_temporal, later_base = _observations(2)
        positive_later = core.predict(
            later_relation,
            later_temporal,
            later_base,
            state=positive,
        )
        negative_later = core.predict(
            later_relation,
            later_temporal,
            later_base,
            state=negative,
        )
        self.assertFalse(
            torch.equal(positive_later.residuals, negative_later.residuals)
        )
        component = core.prospect(
            later_relation,
            later_temporal.unsqueeze(1),
            torch.ones(2, 1, dtype=torch.bool),
            state=positive,
            budget=ProspectiveResourceBudget(1, 2, 2),
        )
        self.assertFalse(torch.equal(component.future_latents[0], component.future_latents[1]))
        self.assertFalse(
            torch.equal(component.selection_residuals[0], component.selection_residuals[1])
        )

        probe_relation, probe_temporal, probe_base = _observations(7)
        learned_batch = core.predict(
            probe_relation,
            probe_temporal,
            probe_base,
            state=positive,
        )
        permutation = torch.arange(6, -1, -1)
        learned_permuted = core.predict(
            probe_relation[permutation],
            probe_temporal[permutation],
            probe_base[permutation],
            state=positive,
        )
        torch.testing.assert_close(
            learned_permuted.logits,
            learned_batch.logits[permutation],
            rtol=0.0,
            atol=1.0e-6,
        )
        singles = torch.cat(
            tuple(
                core.predict(
                    probe_relation[index : index + 1],
                    probe_temporal[index : index + 1],
                    probe_base[index : index + 1],
                    state=positive,
                ).logits
                for index in range(7)
            )
        )
        torch.testing.assert_close(
            singles,
            learned_batch.logits,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(
            int(torch.argmax(singles).item()),
            int(torch.argmax(learned_batch.logits).item()),
        )

    def test_declared_training_objective_reaches_every_prospective_head(self) -> None:
        core = _core()
        initial = core.initial_state()
        world = initial.world_hypotheses.clone()
        self_state = initial.self_hypotheses.clone()
        usage = initial.world_usage.clone()
        for index in range(3):
            world[index] = torch.linspace(-0.2, 0.3, 8) + float(index) * 0.07
            self_state[index] = torch.linspace(0.1, -0.25, 8) + float(index) * 0.03
            usage[index] = 1.0
        state = replace(
            initial,
            world_hypotheses=world,
            self_hypotheses=self_state,
            focus_hypothesis=torch.linspace(-0.1, 0.2, 8),
            outcome_hypothesis=torch.linspace(0.05, 0.25, 8),
            world_usage=usage,
        )
        relation, _, _ = _observations(2)
        worlds = torch.empty(2, 3, 5)
        offsets = torch.tensor((0.0, 0.03, -0.02, 0.04, -0.01))
        for branch in range(2):
            for world_index in range(3):
                worlds[branch, world_index] = torch.tensor(
                    tuple(
                        float((branch + 1) * (world_index + 1) * (index + 1)) / 17.0
                        for index in range(5)
                    )
                ) + offsets
        output = core.prospect(
            relation,
            worlds,
            torch.ones(2, 3, dtype=torch.bool),
            state=state,
            budget=ProspectiveResourceBudget(2, 2, 2),
        )
        future_targets = (
            torch.flip(output.future_latents.detach(), dims=(0,))
            + torch.linspace(-0.2, 0.2, 8)
        )
        loss = core.prospective_training_loss(
            output,
            future_targets=future_targets,
            outcomes=torch.tensor((1.0, -1.0)),
        )
        loss.backward()
        modules = {
            "action": core.action_network,
            "world": core.world_network,
            "self": core.self_network,
            "focus_query": core.focus_query,
            "focus_key": core.focus_key,
            "future": core.future_cell,
            "outcome": core.outcome_network,
            "uncertainty": core.uncertainty_network,
            "residual_gate": core.residual_gate,
            "residual": core.residual_network,
        }
        for name, module in modules.items():
            with self.subTest(module=name):
                magnitude = sum(
                    float(parameter.grad.abs().sum().item())
                    for parameter in module.parameters()
                    if parameter.grad is not None
                )
                self.assertGreater(magnitude, 0.0)
        report = core.parameter_report()
        self.assertTrue(report["declared_training_objective"])
        self.assertTrue(report["runtime_loss_directed_fast_state"])
        self.assertFalse(report["trained_checkpoint_supplied"])

    def test_snapshot_lesion_event_roundtrip_and_exact_replay(self) -> None:
        core = _core()
        state = core.initial_state()
        events = []
        for index, outcome in enumerate((1.0, -1.0, 1.0)):
            relation, temporal, base = _observations(1)
            relation = relation + float(index) / 19.0
            temporal = temporal + float(index) / 23.0
            output = core.predict(
                relation,
                temporal,
                base,
                state=state,
                read_enabled=index != 1,
            )
            state, event = core.apply_feedback(
                state,
                output,
                torch.tensor((outcome,)),
                evidence_refs=f"evidence://{index}",
            )
            encoded = json.loads(json.dumps(event.to_record()))
            events.append(core.event_from_record(encoded))
        self.assertFalse(events[1].read_enabled)

        snapshot = core.capture_state(state)
        restored = core.restore_state(snapshot)
        self.assertEqual(core.state_digest(restored), core.state_digest(state))
        replayed, replay_outputs = core.replay(events)
        self.assertEqual(core.state_digest(replayed), core.state_digest(state))
        self.assertEqual(len(replay_outputs), 3)

        relation, temporal, base = _observations(2)
        lesion = core.zero_prospective_state_like(state)
        lesion_output = core.predict(relation, temporal, base, state=lesion)
        credit_output = core.credit_core.predict(
            relation,
            temporal,
            base,
            state=state.credit_state,
        )
        self.assertTrue(torch.equal(lesion_output.logits, credit_output.logits))
        self.assertTrue(torch.equal(lesion.credit_state.values, state.credit_state.values))
        self.assertEqual(lesion.step, state.step)
        zero = core.zero_state_like(state)
        zero_output = core.predict(relation, temporal, base, state=zero)
        self.assertTrue(torch.equal(zero_output.logits, base))
        self.assertEqual(zero.step, 0)

        wrong_snapshot = replace(snapshot, config_digest="0" * 64)
        with self.assertRaisesRegex(ValueError, "configuration"):
            core.restore_state(wrong_snapshot)
        record = events[0].to_record()
        record["prospective_state_digest"] = "f" * 64
        tampered = CompositeProspectiveCreditEvent.from_record(record)
        with self.assertRaisesRegex(RuntimeError, "replay diverged"):
            core.replay((tampered,))

        another = _core()
        with torch.no_grad():
            another.uncertainty_network.bias.add_(0.125)
        self.assertNotEqual(another.checkpoint_identity, core.checkpoint_identity)
        with self.assertRaisesRegex(ValueError, "another prospective checkpoint"):
            another.event_from_record(events[0].to_record())
        with self.assertRaisesRegex(ValueError, "another prospective checkpoint"):
            another.replay((events[0],))

    def test_validation_rejects_state_event_and_capacity_tampering(self) -> None:
        core = _core()
        state = core.initial_state()
        with self.assertRaisesRegex(ValueError, "unsynchronized"):
            core.validate_state(replace(state, step=1))
        bad_world = state.world_hypotheses.clone()
        bad_world[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            core.validate_state(replace(state, world_hypotheses=bad_world))
        with self.assertRaisesRegex(ValueError, "maximum_branches"):
            ProspectiveDynamicsConfig(temporal_width=5, maximum_branches=0)

        focus_relation, _, _ = _observations(1)
        focus_worlds = torch.stack(
            (
                torch.linspace(-0.2, 0.4, 5),
                torch.linspace(0.3, -0.1, 5),
            )
        ).unsqueeze(0)
        eligible = torch.ones(1, 2, dtype=torch.bool)
        budget = ProspectiveResourceBudget(2, 1, 2)
        focus_output = core.prospect(
            focus_relation,
            focus_worlds,
            eligible,
            state=state,
            budget=budget,
        )
        validation = dict(
            batch=1,
            worlds=2,
            eligible_mask=eligible,
            state=state,
            budget=budget,
        )
        with self.assertRaisesRegex(ValueError, "shape or type"):
            core._validate_prospective_output(
                replace(focus_output, state_step=999),
                **validation,
            )
        with self.assertRaisesRegex(ValueError, "shape or type"):
            core._validate_prospective_output(
                replace(focus_output, recurrent_steps=999),
                **validation,
            )
        with self.assertRaisesRegex(ValueError, "branches"):
            core._validate_prospective_output(
                replace(focus_output, recurrent_steps=999),
                batch=1,
                worlds=2,
                eligible_mask=eligible,
                state=state,
                budget=ProspectiveResourceBudget(2, 999, 999),
            )
        with self.assertRaisesRegex(ValueError, "must be probabilities"):
            core._validate_prospective_output(
                replace(focus_output, focus_weights=torch.tensor(((-1.0, 2.0),))),
                **validation,
            )
        with self.assertRaisesRegex(ValueError, "focus count"):
            core._validate_prospective_output(
                replace(focus_output, focused_mask=torch.tensor(((True, False),))),
                **validation,
            )
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            core._validate_prospective_output(
                replace(focus_output, outcome_logits=focus_output.outcome_logits.double()),
                **validation,
            )

        relation, temporal, base = _observations(1)
        output = core.predict(relation, temporal, base, state=state)
        _, event = core.apply_feedback(
            state,
            output,
            torch.ones(1),
            evidence_refs="evidence://event",
        )
        unknown = event.to_record()
        unknown["unknown"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            CompositeProspectiveCreditEvent.from_record(unknown)
        wrong_config = event.to_record()
        wrong_config["config_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "another prospective configuration"):
            core.event_from_record(wrong_config)


if __name__ == "__main__":
    unittest.main()
