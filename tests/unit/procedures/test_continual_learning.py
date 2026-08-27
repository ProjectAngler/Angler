from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.procedures.continual import (  # noqa: E402
    GuardedTrunkConsolidator,
    ModuleRetirementState,
    ReservoirSampler,
    RetentionProbe,
    ThresholdArm,
    ThresholdBandit,
    mix_replay,
)


class ReservoirAndReplayTests(unittest.TestCase):
    def test_reservoir_has_fixed_capacity_and_replays_from_snapshot(self) -> None:
        first = ReservoirSampler[int](4, seed=2201)
        replay = ReservoirSampler[int](4, seed=2201)
        first.extend(tuple(range(100)))
        replay.extend(tuple(range(100)))

        self.assertEqual(first.seen_count, 100)
        self.assertEqual(len(first.items), 4)
        self.assertEqual(first.items, replay.items)
        snapshot = first.state_dict()
        first.add(100)
        first.load_state_dict(snapshot)
        self.assertEqual(first.items, replay.items)
        self.assertEqual(first.seen_count, replay.seen_count)

    def test_replay_mix_preserves_the_explicit_generated_ratio(self) -> None:
        batch = mix_replay(
            tuple(range(20)),
            tuple(range(100, 110)),
            batch_size=10,
            generated_ratio=0.3,
            seed=2202,
        )

        self.assertEqual(len(batch.items), 10)
        self.assertEqual(sum(item.source == "generated" for item in batch.items), 3)
        self.assertEqual(batch.realized_generated_ratio, 0.3)
        self.assertEqual(batch.requested_generated_ratio, 0.3)
        with self.assertRaisesRegex(ValueError, "empty pool"):
            mix_replay(
                (1, 2),
                (),
                batch_size=4,
                generated_ratio=0.5,
            )


class _CounterState:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class GuardedConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2203)
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # Materialize optimizer state so rollback covers more than param groups.
        self.optimizer.zero_grad(set_to_none=True)
        self.model(torch.ones(1, 1)).sum().backward()
        self.optimizer.step()
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.counter = _CounterState(3)
        self.probes = (
            RetentionProbe("op-a", "example-1", 1.0),
            RetentionProbe("op-a", "example-2", 2.0),
            RetentionProbe("op-b", "example-3", 3.0),
        )

    @staticmethod
    def _score(model: nn.Module, value: float) -> float:
        parameter = next(model.parameters())
        return float(parameter.item() * value)

    def test_regressing_update_restores_model_optimizer_and_stateful_objects(self) -> None:
        guard = GuardedTrunkConsolidator(maximum_regression_fraction=0.25)
        original_weight = self.model.weight.detach().clone()
        original_optimizer = self.optimizer.state_dict()

        def harmful(model: nn.Module) -> None:
            with torch.no_grad():
                next(model.parameters()).fill_(0.5)
            self.optimizer.param_groups[0]["lr"] = 0.9
            self.counter.value = 99

        receipt = guard.consolidate(
            self.model,
            self.probes,
            scorer=self._score,
            update=harmful,
            optimizer=self.optimizer,
            stateful_objects=(self.counter,),
        )

        self.assertFalse(receipt.accepted)
        self.assertTrue(receipt.rolled_back)
        self.assertEqual(receipt.regression_count, 3)
        self.assertEqual(receipt.regression_fraction, 1.0)
        self.assertTrue(torch.equal(self.model.weight.detach(), original_weight))
        self.assertEqual(
            self.optimizer.param_groups[0]["lr"],
            original_optimizer["param_groups"][0]["lr"],
        )
        self.assertEqual(self.counter.value, 3)
        self.assertEqual(len(receipt.scores), len(self.probes))

    def test_nonregressing_update_is_retained_after_all_probes_are_scored(self) -> None:
        guard = GuardedTrunkConsolidator(maximum_regression_fraction=0.0)
        calls: list[float] = []

        def scorer(model: nn.Module, value: float) -> float:
            calls.append(value)
            return self._score(model, value)

        def beneficial(model: nn.Module) -> None:
            with torch.no_grad():
                next(model.parameters()).fill_(1.25)

        receipt = guard.consolidate(
            self.model,
            self.probes,
            scorer=scorer,
            update=beneficial,
        )

        self.assertTrue(receipt.accepted)
        self.assertFalse(receipt.rolled_back)
        self.assertEqual(receipt.regression_count, 0)
        self.assertEqual(len(calls), len(self.probes) * 2)
        self.assertAlmostEqual(float(self.model.weight.item()), 1.25)


class BanditAndRetirementTests(unittest.TestCase):
    def test_bandit_selects_complete_threshold_arms_and_learns_reward(self) -> None:
        arms = (
            ThresholdArm("conservative", 2.0, 0.9, 1e-4, 0.1, 0.05),
            ThresholdArm("plastic", 0.5, 0.6, 1e-3, 0.4, 0.2),
        )
        bandit = ThresholdBandit(arms, ucb_strength=0.0, seed=17)
        first = bandit.select()
        bandit.record(first.arm_id, 0.1)
        second = bandit.select()
        self.assertNotEqual(first.arm_id, second.arm_id)
        bandit.record(second.arm_id, 0.8)

        self.assertEqual(bandit.select().arm_id, second.arm_id)
        snapshot = bandit.state_dict()
        restored = ThresholdBandit(arms, ucb_strength=0.0, seed=999)
        restored.load_state_dict(snapshot)
        self.assertEqual(restored.select().arm_id, second.arm_id)

    def test_retirement_drops_only_active_module_and_preserves_evidence(self) -> None:
        mirror = "sha256:" + "a" * 64
        exemplar = "sha256:" + "b" * 64
        state = ModuleRetirementState("module-a", mirror, exemplar)

        state = state.advance(used=False, retire_after=3)
        state = state.advance(used=False, retire_after=3)
        self.assertTrue(state.module_active)
        state = state.advance(used=False, retire_after=3)

        self.assertFalse(state.module_active)
        self.assertEqual(state.unused_cycles, 3)
        self.assertEqual(state.mirror_digest, mirror)
        self.assertEqual(state.exemplar_digest, exemplar)
        self.assertIs(state.advance(used=True, retire_after=3), state)


if __name__ == "__main__":
    unittest.main()
