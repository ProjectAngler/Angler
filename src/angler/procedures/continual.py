"""Bounded replay, guarded consolidation, and module lifecycle primitives.

All operations are synchronous and domain-neutral.  Consolidation evaluates
every supplied retention probe and restores every snapshotted object on a
rejected or exceptional update; no component in this module runs work in the
background or calls an environment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import random
import re
from typing import Any, Callable, Generic, Literal, Mapping, Protocol, Sequence, TypeVar

import torch
from torch import nn


T = TypeVar("T")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReservoirSampler(Generic[T]):
    """Classic fixed-capacity reservoir over an arbitrarily long stream."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.capacity = capacity
        self.seen_count = 0
        self._items: list[T] = []
        self._rng = random.Random(seed)

    @property
    def items(self) -> tuple[T, ...]:
        return tuple(self._items)

    def add(self, item: T) -> None:
        self.seen_count += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        index = self._rng.randrange(self.seen_count)
        if index < self.capacity:
            self._items[index] = item

    def extend(self, items: Sequence[T]) -> None:
        for item in items:
            self.add(item)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seen_count": self.seen_count,
            "items": copy.deepcopy(self._items),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("capacity") != self.capacity:
            raise ValueError("reservoir snapshot capacity differs")
        items = copy.deepcopy(state.get("items"))
        seen = state.get("seen_count")
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError("reservoir snapshot items are invalid")
        if isinstance(seen, bool) or not isinstance(seen, int) or seen < len(items):
            raise ValueError("reservoir snapshot seen_count is invalid")
        self._items = items
        self.seen_count = seen
        self._rng.setstate(state["rng_state"])


@dataclass(frozen=True, slots=True)
class ReplayItem(Generic[T]):
    value: T
    source: Literal["actual", "generated"]


@dataclass(frozen=True, slots=True)
class ReplayBatch(Generic[T]):
    items: tuple[ReplayItem[T], ...]
    requested_generated_ratio: float

    @property
    def realized_generated_ratio(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.source == "generated" for item in self.items) / len(self.items)


def mix_replay(
    actual: Sequence[T],
    generated: Sequence[T],
    *,
    batch_size: int,
    generated_ratio: float,
    seed: int = 0,
) -> ReplayBatch[T]:
    """Sample a replay batch with an explicit, auditable source ratio."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(generated_ratio) or not 0.0 <= generated_ratio <= 1.0:
        raise ValueError("generated_ratio must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    generated_count = int(math.floor(batch_size * generated_ratio + 0.5))
    actual_count = batch_size - generated_count
    if actual_count and not actual:
        raise ValueError("actual replay was requested from an empty pool")
    if generated_count and not generated:
        raise ValueError("generated replay was requested from an empty pool")
    rng = random.Random(seed)

    def draw(values: Sequence[T], count: int) -> list[T]:
        if count <= len(values):
            return rng.sample(list(values), count)
        return [rng.choice(values) for _ in range(count)]

    items = [ReplayItem(value, "actual") for value in draw(actual, actual_count)]
    items.extend(
        ReplayItem(value, "generated")
        for value in draw(generated, generated_count)
    )
    rng.shuffle(items)
    return ReplayBatch(tuple(items), generated_ratio)


class StatefulObject(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class RetentionProbe(Generic[T]):
    operator_identity: str
    example_identity: str
    payload: T

    def __post_init__(self) -> None:
        if not self.operator_identity or not self.example_identity:
            raise ValueError("probe identities must be non-empty")


@dataclass(frozen=True, slots=True)
class ProbeScore:
    operator_identity: str
    example_identity: str
    before: float
    after: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class ConsolidationReceipt:
    accepted: bool
    rolled_back: bool
    regression_count: int
    regression_fraction: float
    maximum_regression_fraction: float
    scores: tuple[ProbeScore, ...]


class GuardedTrunkConsolidator:
    """Apply one update transaction and atomically reject excess regression."""

    def __init__(
        self,
        *,
        maximum_regression_fraction: float,
        score_tolerance: float = 0.0,
    ) -> None:
        if (
            not math.isfinite(maximum_regression_fraction)
            or not 0.0 <= maximum_regression_fraction <= 1.0
        ):
            raise ValueError("maximum_regression_fraction must be in [0, 1]")
        if not math.isfinite(score_tolerance) or score_tolerance < 0.0:
            raise ValueError("score_tolerance must be finite and nonnegative")
        self.maximum_regression_fraction = maximum_regression_fraction
        self.score_tolerance = score_tolerance

    def consolidate(
        self,
        model: nn.Module,
        probes: Sequence[RetentionProbe[T]],
        *,
        scorer: Callable[[nn.Module, T], float],
        update: Callable[[nn.Module], None],
        optimizer: torch.optim.Optimizer | None = None,
        stateful_objects: Sequence[StatefulObject] = (),
    ) -> ConsolidationReceipt:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not probes:
            raise ValueError("at least one retained probe is required")
        identities = [
            (probe.operator_identity, probe.example_identity) for probe in probes
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("retention probe identities must be unique")
        if any(not isinstance(probe, RetentionProbe) for probe in probes):
            raise TypeError("probes must contain only RetentionProbe values")
        for item in stateful_objects:
            if not hasattr(item, "state_dict") or not hasattr(item, "load_state_dict"):
                raise TypeError("stateful objects must support state_dict/load_state_dict")

        model_snapshot = copy.deepcopy(model.state_dict())
        optimizer_snapshot = (
            None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        )
        object_snapshots = [
            copy.deepcopy(item.state_dict()) for item in stateful_objects
        ]
        training = model.training
        before = self._score_all(model, probes, scorer)

        def restore() -> None:
            model.load_state_dict(model_snapshot)
            model.train(training)
            if optimizer is not None and optimizer_snapshot is not None:
                optimizer.load_state_dict(optimizer_snapshot)
            for item, snapshot in zip(
                stateful_objects,
                object_snapshots,
                strict=True,
            ):
                item.load_state_dict(snapshot)

        try:
            update(model)
            after = self._score_all(model, probes, scorer)
        except BaseException:
            restore()
            raise
        finally:
            model.train(training)

        scores = tuple(
            ProbeScore(
                operator_identity=probe.operator_identity,
                example_identity=probe.example_identity,
                before=old,
                after=new,
                regressed=new < old - self.score_tolerance,
            )
            for probe, old, new in zip(probes, before, after, strict=True)
        )
        regression_count = sum(score.regressed for score in scores)
        regression_fraction = regression_count / len(scores)
        accepted = regression_fraction <= self.maximum_regression_fraction
        if not accepted:
            restore()
        return ConsolidationReceipt(
            accepted=accepted,
            rolled_back=not accepted,
            regression_count=regression_count,
            regression_fraction=regression_fraction,
            maximum_regression_fraction=self.maximum_regression_fraction,
            scores=scores,
        )

    @staticmethod
    def _score_all(
        model: nn.Module,
        probes: Sequence[RetentionProbe[T]],
        scorer: Callable[[nn.Module, T], float],
    ) -> tuple[float, ...]:
        values: list[float] = []
        with torch.no_grad():
            for probe in probes:
                value = float(scorer(model, probe.payload))
                if not math.isfinite(value):
                    raise RuntimeError("retention scorer returned a non-finite value")
                values.append(value)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ThresholdArm:
    """One jointly selected consolidation/induction threshold setting."""

    arm_id: str
    mdl_penalty: float
    merge_threshold: float
    trunk_learning_rate: float
    generated_replay_ratio: float
    exploration_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("arm_id must be a non-empty string")
        if not math.isfinite(self.mdl_penalty) or self.mdl_penalty < 0.0:
            raise ValueError("mdl_penalty must be finite and nonnegative")
        if not math.isfinite(self.trunk_learning_rate) or self.trunk_learning_rate <= 0.0:
            raise ValueError("trunk_learning_rate must be finite and positive")
        for name in (
            "merge_threshold",
            "generated_replay_ratio",
            "exploration_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


class ThresholdBandit:
    """Small UCB bandit over complete predeclared threshold arms."""

    def __init__(
        self,
        arms: Sequence[ThresholdArm],
        *,
        ucb_strength: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not arms or any(not isinstance(arm, ThresholdArm) for arm in arms):
            raise ValueError("arms must be a non-empty ThresholdArm sequence")
        if len({arm.arm_id for arm in arms}) != len(arms):
            raise ValueError("bandit arm IDs must be unique")
        if not math.isfinite(ucb_strength) or ucb_strength < 0.0:
            raise ValueError("ucb_strength must be finite and nonnegative")
        self.arms = tuple(arms)
        self.ucb_strength = ucb_strength
        self._counts = {arm.arm_id: 0 for arm in arms}
        self._means = {arm.arm_id: 0.0 for arm in arms}
        self._initial_order = list(range(len(arms)))
        random.Random(seed).shuffle(self._initial_order)

    def select(self) -> ThresholdArm:
        for index in self._initial_order:
            arm = self.arms[index]
            if self._counts[arm.arm_id] == 0:
                return arm
        total = sum(self._counts.values())
        return max(
            self.arms,
            key=lambda arm: (
                self._means[arm.arm_id]
                + self.ucb_strength
                * math.sqrt(math.log(total + 1) / self._counts[arm.arm_id]),
                arm.arm_id,
            ),
        )

    def record(self, arm_id: str, reward: float) -> None:
        if arm_id not in self._counts:
            raise KeyError(arm_id)
        if not math.isfinite(reward):
            raise ValueError("bandit reward must be finite")
        count = self._counts[arm_id] + 1
        mean = self._means[arm_id]
        self._means[arm_id] = mean + (reward - mean) / count
        self._counts[arm_id] = count

    def state_dict(self) -> dict[str, Any]:
        return {
            "arm_ids": tuple(arm.arm_id for arm in self.arms),
            "counts": dict(self._counts),
            "means": dict(self._means),
            "initial_order": tuple(self._initial_order),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = tuple(arm.arm_id for arm in self.arms)
        if tuple(state.get("arm_ids", ())) != expected:
            raise ValueError("bandit snapshot arm identities differ")
        counts = dict(state["counts"])
        means = dict(state["means"])
        if set(counts) != set(expected) or set(means) != set(expected):
            raise ValueError("bandit snapshot keys differ")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ) or any(not math.isfinite(float(value)) for value in means.values()):
            raise ValueError("bandit snapshot statistics are invalid")
        self._counts = counts
        self._means = {key: float(value) for key, value in means.items()}
        self._initial_order = list(state["initial_order"])


@dataclass(frozen=True, slots=True)
class ModuleRetirementState:
    """Active neural-module status with durable mirror/evidence identities."""

    module_identity: str
    mirror_digest: str
    exemplar_digest: str
    unused_cycles: int = 0
    module_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.module_identity, str) or not self.module_identity:
            raise ValueError("module_identity must be non-empty")
        for name in ("mirror_digest", "exemplar_digest"):
            if not isinstance(getattr(self, name), str) or not _DIGEST.fullmatch(
                getattr(self, name)
            ):
                raise ValueError(f"{name} must be a canonical sha256 digest")
        if (
            isinstance(self.unused_cycles, bool)
            or not isinstance(self.unused_cycles, int)
            or self.unused_cycles < 0
        ):
            raise ValueError("unused_cycles must be a nonnegative integer")
        if not isinstance(self.module_active, bool):
            raise TypeError("module_active must be bool")

    def advance(self, *, used: bool, retire_after: int) -> "ModuleRetirementState":
        if not isinstance(used, bool):
            raise TypeError("used must be bool")
        if (
            isinstance(retire_after, bool)
            or not isinstance(retire_after, int)
            or retire_after <= 0
        ):
            raise ValueError("retire_after must be a positive integer")
        if not self.module_active:
            return self
        unused = 0 if used else self.unused_cycles + 1
        return replace(
            self,
            unused_cycles=unused,
            module_active=unused < retire_after,
        )


__all__ = [
    "ConsolidationReceipt",
    "GuardedTrunkConsolidator",
    "ModuleRetirementState",
    "ProbeScore",
    "ReplayBatch",
    "ReplayItem",
    "ReservoirSampler",
    "RetentionProbe",
    "StatefulObject",
    "ThresholdArm",
    "ThresholdBandit",
    "mix_replay",
]
