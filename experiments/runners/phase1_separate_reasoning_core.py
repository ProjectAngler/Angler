"""Train Project Angler's reasoning core across two task families.

The Transformers donor is a frozen, model-agnostic hidden-state backbone.  It
encodes each public goal/fact, action symbol, and focused action mention
separately.  It never receives an assembled problem or produces an answer.
Only the standalone recurrent reasoning core is optimized, using leave-one-out
REINFORCE over outcome-only scalar satisfaction.  Every task is encountered
once; independently sampled actions provide immediate credit without replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from angler.learning.policy_credit import leave_one_out_advantages  # noqa: E402
from angler.reasoning import (  # noqa: E402
    ReasoningCoreConfig,
    RecurrentReasoningCore,
    reasoning_state_digest,
    restore_reasoning_state,
    snapshot_reasoning_state,
)
from angler.runtime import (  # noqa: E402
    encode_detached_segments,
    foundation_tensor_digest,
    freeze_knowledge_model,
)
from angler.worlds import (  # noqa: E402
    DEFAULT_DEMONSTRATION_SURFACE_FORMS,
    DEFAULT_GOAL_SURFACE_FORMS,
    DEFAULT_QUERY_SURFACE_FORMS,
    DEFAULT_RELATION_SURFACE_FORMS,
    FAMILY_ID as RELATIONAL_FAMILY_ID,
    HiddenSymbolicRuleSolution,
    LearnerTask,
    SYMBOLIC_RULE_FAMILY_ID,
    SymbolicRuleLearnerTask,
    generate_relational_task,
    generate_symbolic_rule_task,
    score_constraint_satisfaction,
    verify_symbolic_rule_answer,
)


_STATE_FILENAME = "reasoning_state.safetensors"
_METADATA_FILENAME = "reasoning_state.json"
_STATE_FORMAT = "angler.recurrent-reasoning-state@2"
_KNOWLEDGE_INTERFACE = "transformers.AutoModel.detached-independent-segments@1"
_FULL_MODEL_PATH = Path("/opt/angler/models/Qwen3-4B")
_FULL_FOUNDATION_DIGEST = (
    "sha256:f228ca26e33596461f72195fcbccfa7b873fd7a4dc7c87d19d2854484bcccd3a"
)
_FAMILY_RELATIONAL = RELATIONAL_FAMILY_ID
_FAMILY_SYMBOLIC = SYMBOLIC_RULE_FAMILY_ID
_FAMILY_IDS = (_FAMILY_RELATIONAL, _FAMILY_SYMBOLIC)
_RELATIONAL_GOAL = (
    "Produce one complete ordering of the public action symbols that satisfies "
    "all visible relation facts."
)
_TRAINING_SURFACE_FORMS = (
    *DEFAULT_RELATION_SURFACE_FORMS,
    "{earlier} comes earlier than {later}.",
    "{later} comes later than {earlier}.",
    "{earlier} occurs before {later}.",
    "{later} occurs after {earlier}.",
)
_UNSEEN_SURFACE_FORMS = (
    "{earlier} is prior to {later}.",
    "{later} is subsequent to {earlier}.",
    "{earlier} is ahead of {later} in the sequence.",
    "{later} is behind {earlier} in the sequence.",
)
_SYMBOLIC_TRAINING_GOAL_FORMS = DEFAULT_GOAL_SURFACE_FORMS
_SYMBOLIC_TRAINING_DEMO_FORMS = DEFAULT_DEMONSTRATION_SURFACE_FORMS
_SYMBOLIC_TRAINING_QUERY_FORMS = DEFAULT_QUERY_SURFACE_FORMS
_SYMBOLIC_UNSEEN_GOAL_FORMS = (
    (
        "Discover the positional transformation shared by {demo_count} worked "
        "mappings and use it for the new {item_count}-item sequence."
    ),
    (
        "Derive one rearrangement from {demo_count} cases, then transfer it to "
        "the fresh sequence containing {item_count} items."
    ),
)
_SYMBOLIC_UNSEEN_DEMO_FORMS = (
    "Worked mapping {demo_number}: source <{inputs}> produces target <{outputs}>.",
    "Observation {demo_number}: sequence <{inputs}> is rearranged as <{outputs}>.",
)
_SYMBOLIC_UNSEEN_QUERY_FORMS = (
    "Fresh source sequence: <{inputs}>",
    "Transfer the inferred rule to: <{inputs}>",
)

_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "smoke": {
        "training_tasks": 8,
        "heldout_tasks": 8,
        "presentations_per_task": 1,
        "samples_per_presentation": 4,
        "entity_sizes": (4, 5, 6, 7),
        "core_width": 128,
        "workspace_slots": 4,
        "attention_heads": 4,
        "feedforward_width": 512,
        "reasoning_steps": 2,
        "maximum_reasoning_steps": 8,
        "learning_rate": 3e-4,
        "cohort_size": 8,
        "sampling_temperature": 1.5,
    },
    "full": {
        "training_tasks": 4096,
        "heldout_tasks": 256,
        "presentations_per_task": 1,
        "samples_per_presentation": 8,
        "entity_sizes": (4, 5, 6, 7),
        "core_width": 512,
        "workspace_slots": 16,
        "attention_heads": 8,
        "feedforward_width": 2048,
        "reasoning_steps": 8,
        "maximum_reasoning_steps": 12,
        "learning_rate": 2e-5,
        "cohort_size": 8,
        "sampling_temperature": 1.5,
    },
}


@dataclass(frozen=True, slots=True)
class RunSettings:
    training_tasks: int
    heldout_tasks: int
    presentations_per_task: int
    samples_per_presentation: int
    entity_sizes: tuple[int, ...]
    core_width: int
    workspace_slots: int
    attention_heads: int
    feedforward_width: int
    reasoning_steps: int
    maximum_reasoning_steps: int
    learning_rate: float
    cohort_size: int
    sampling_temperature: float


@dataclass(frozen=True, slots=True)
class RetentionThresholds:
    maximum_state_delta_norm: float
    exact_reward_bonus: float


PublicTask = LearnerTask | SymbolicRuleLearnerTask


@dataclass(frozen=True, slots=True)
class _OutcomeScore:
    valid: bool
    exact: bool
    satisfaction: float


class _SealedOutcomeOracle:
    """Evaluator-only symbolic targets, never attached to learner material."""

    __slots__ = ("__symbolic",)

    def __init__(
        self,
        symbolic: Mapping[str, HiddenSymbolicRuleSolution],
    ) -> None:
        if any(
            instance_id != hidden.instance_id
            for instance_id, hidden in symbolic.items()
        ):
            raise ValueError("sealed symbolic identity binding is inconsistent")
        self.__symbolic = dict(symbolic)

    def score(
        self,
        task: PublicTask,
        answer: Sequence[str],
    ) -> _OutcomeScore:
        if isinstance(task, LearnerTask):
            outcome = score_constraint_satisfaction(task, answer)
            return _OutcomeScore(
                valid=outcome.valid,
                exact=outcome.exact,
                satisfaction=outcome.constraint_satisfaction,
            )
        if isinstance(task, SymbolicRuleLearnerTask):
            try:
                hidden = self.__symbolic[task.instance_id]
            except KeyError as error:
                raise RuntimeError(
                    "symbolic task has no evaluator-only sealed solution"
                ) from error
            outcome = verify_symbolic_rule_answer(task, hidden, answer)
            return _OutcomeScore(
                valid=outcome.valid,
                exact=outcome.exact,
                satisfaction=outcome.pairwise_order_agreement,
            )
        raise TypeError(f"unsupported public task type: {type(task).__name__}")


@dataclass(frozen=True, slots=True)
class _TaskSegments:
    task: PublicTask
    facts: tuple[str, ...]
    entities: tuple[str, ...]
    focused_mentions: tuple[str, ...]
    mention_entity_indices: tuple[int, ...]
    mention_fact_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TaskKnowledge:
    task: PublicTask
    surface_partition: str
    fact_features: torch.Tensor
    entity_features: torch.Tensor
    mention_features: torch.Tensor
    mention_entity_indices: torch.Tensor
    mention_fact_indices: torch.Tensor

    def model_inputs(
        self,
        device: torch.device,
        *,
        remove_fact_evidence: bool = False,
        derange_incidence: bool = False,
    ) -> dict[str, torch.Tensor]:
        facts = self.fact_features.to(device=device)
        entities = self.entity_features.to(device=device)
        mentions = self.mention_features.to(device=device)
        fact_mask = torch.ones(
            (1, facts.shape[1]), dtype=torch.bool, device=device
        )
        entity_mask = torch.ones(
            (1, entities.shape[1]), dtype=torch.bool, device=device
        )
        mention_mask = torch.ones(
            (1, mentions.shape[1]), dtype=torch.bool, device=device
        )
        if remove_fact_evidence:
            facts = torch.zeros_like(facts)
            mentions = torch.zeros_like(mentions)
        entity_indices = self.mention_entity_indices.to(device=device)
        if derange_incidence:
            entity_indices = (entity_indices + 1) % entities.shape[1]
        return {
            "fact_features": facts,
            "fact_mask": fact_mask,
            "entity_features": entities,
            "entity_mask": entity_mask,
            "mention_features": mentions,
            "mention_mask": mention_mask,
            "mention_fact_indices": self.mention_fact_indices.to(device=device),
            "mention_entity_indices": entity_indices,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    exact: int
    total: int
    mean_satisfaction: float
    by_family: Mapping[str, Mapping[str, float | int]]
    by_entity_size: Mapping[int, Mapping[str, float | int]]
    by_surface_partition: Mapping[str, Mapping[str, float | int]]
    orders: tuple[tuple[str, ...], ...]
    records: tuple[dict[str, Any], ...]

    def public(self, *, include_records: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "exact": self.exact,
            "total": self.total,
            "exact_accuracy": self.exact / self.total if self.total else 0.0,
            "mean_satisfaction": self.mean_satisfaction,
            "by_family": {
                name: dict(metrics)
                for name, metrics in sorted(self.by_family.items())
            },
            "by_entity_size": {
                str(size): dict(metrics)
                for size, metrics in sorted(self.by_entity_size.items())
            },
            "by_surface_partition": {
                name: dict(metrics)
                for name, metrics in sorted(self.by_surface_partition.items())
            },
        }
        if include_records:
            result["records"] = list(self.records)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(_PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--parent-state")
    parser.add_argument(
        "--candidate-dir",
        default="/opt/angler/project/work/cross-family-rloo-candidate",
    )
    parser.add_argument("--result-json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--hidden-state-index", type=int, default=-1)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--training-tasks", type=int)
    parser.add_argument("--heldout-tasks", type=int)
    parser.add_argument("--presentations-per-task", type=int)
    parser.add_argument("--samples-per-presentation", type=int)
    parser.add_argument("--entity-sizes", type=int, nargs="+")
    parser.add_argument("--core-width", type=int)
    parser.add_argument("--workspace-slots", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--feedforward-width", type=int)
    parser.add_argument("--reasoning-steps", type=int)
    parser.add_argument("--maximum-reasoning-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cohort-size", type=int)
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--max-state-delta-norm", type=float)
    parser.add_argument("--exact-reward-bonus", type=float)
    parser.add_argument("--include-records", action="store_true")
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> RunSettings:
    defaults = _PROFILE_DEFAULTS[args.profile]
    values: dict[str, object] = {}
    for name, default in defaults.items():
        supplied = getattr(args, name)
        if name == "entity_sizes" and supplied is not None:
            supplied = tuple(supplied)
        values[name] = default if supplied is None else supplied
    settings = RunSettings(**values)  # type: ignore[arg-type]
    numeric = {
        name: value
        for name, value in asdict(settings).items()
        if name != "entity_sizes"
    }
    invalid = [name for name, value in numeric.items() if value <= 0]
    if invalid:
        raise ValueError("run settings must be positive: " + ", ".join(invalid))
    if (
        not settings.entity_sizes
        or len(set(settings.entity_sizes)) != len(settings.entity_sizes)
        or any(size < 4 or size > 7 for size in settings.entity_sizes)
    ):
        raise ValueError("entity_sizes must be unique values between four and seven")
    family_size_cells = len(_FAMILY_IDS) * len(settings.entity_sizes)
    if settings.training_tasks % family_size_cells:
        raise ValueError(
            "training_tasks must divide evenly across families and entity sizes"
        )
    if settings.heldout_tasks % family_size_cells:
        raise ValueError(
            "heldout_tasks must divide evenly across families and entity sizes"
        )
    if settings.training_tasks % settings.cohort_size:
        raise ValueError("training_tasks must divide evenly across cohort_size")
    if settings.cohort_size != family_size_cells:
        raise ValueError(
            "cohort_size must contain one balanced family-by-size block"
        )
    if settings.presentations_per_task != 1:
        raise ValueError("every task must be encountered exactly once")
    if settings.samples_per_presentation < 2:
        raise ValueError("leave-one-out credit requires at least two samples")
    if settings.reasoning_steps > settings.maximum_reasoning_steps:
        raise ValueError("reasoning_steps exceeds maximum_reasoning_steps")
    if settings.core_width % settings.attention_heads:
        raise ValueError("core_width must be divisible by attention_heads")
    return settings


def resolve_thresholds(args: argparse.Namespace) -> RetentionThresholds:
    values = {
        "maximum_state_delta_norm": (
            5.0 if args.max_state_delta_norm is None else args.max_state_delta_norm
        ),
        "exact_reward_bonus": (
            1.0 if args.exact_reward_bonus is None else args.exact_reward_bonus
        ),
    }
    thresholds = RetentionThresholds(**values)
    if thresholds.maximum_state_delta_norm <= 0.0:
        raise ValueError("maximum_state_delta_norm must be positive")
    if thresholds.exact_reward_bonus < 0.0:
        raise ValueError("exact_reward_bonus must be nonnegative")
    return thresholds


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_frozen_knowledge_backbone(
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention_implementation: str,
) -> tuple[torch.nn.Module, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("the tokenizer exposes neither padding nor EOS")
        tokenizer.pad_token = tokenizer.eos_token

    load_options: dict[str, Any] = {
        "local_files_only": True,
        "dtype": dtype,
    }
    if attention_implementation:
        load_options["attn_implementation"] = attention_implementation
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("the requested CUDA device is unavailable")
        load_options["device_map"] = {"": device.index or 0}
    model = AutoModel.from_pretrained(str(model_path), **load_options)
    if device.type != "cuda":
        model.to(device)
    freeze_knowledge_model(model)
    return model, tokenizer


def _whole_label_matches(text: str, label: str) -> tuple[re.Match[str], ...]:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])"
    return tuple(re.finditer(pattern, text))


def _action_symbols(task: PublicTask) -> tuple[str, ...]:
    if isinstance(task, LearnerTask):
        return task.symbols
    if isinstance(task, SymbolicRuleLearnerTask):
        return task.query_symbols
    raise TypeError(f"unsupported public task type: {type(task).__name__}")


def _independent_public_facts(task: PublicTask) -> tuple[str, ...]:
    if isinstance(task, LearnerTask):
        return (
            f"Public goal: {_RELATIONAL_GOAL}",
            *(f"Public relation fact: {fact}" for fact in task.fact_statements),
        )
    if isinstance(task, SymbolicRuleLearnerTask):
        return (
            f"Public goal: {task.goal_text}",
            *(
                f"Public demonstration: {demonstration.statement}"
                for demonstration in task.demonstrations
            ),
            f"Public query: {task.query_statement}",
        )
    raise TypeError(f"unsupported public task type: {type(task).__name__}")


def task_segments(task: PublicTask) -> _TaskSegments:
    """Create independent public segments using literal action identity only.

    Goals, demonstrations, relation facts, and queries remain separate donor
    inputs.  A fact may mention no action symbol (all symbolic demonstrations
    intentionally do); incidence is created only for literal public action
    occurrences.  This adapter never aligns demonstration positions, parses a
    relation, or observes an evaluator target.
    """

    facts = _independent_public_facts(task)
    actions = _action_symbols(task)
    mentions: list[tuple[int, int, str, int]] = []
    action_mentions = [0] * len(actions)
    for fact_index, fact in enumerate(facts):
        for entity_index, entity in enumerate(actions):
            matches = _whole_label_matches(fact, entity)
            if len(matches) > 1:
                raise RuntimeError("a public action appears repeatedly in one fact")
            if matches:
                match = matches[0]
                focused = (
                    f"{fact}\n"
                    f"Focus public action symbol: {entity}"
                )
                mentions.append(
                    (fact_index, match.start(), focused, entity_index)
                )
                action_mentions[entity_index] += 1
    mentions.sort(key=lambda item: (item[0], item[1]))
    if any(count == 0 for count in action_mentions):
        raise RuntimeError("every public action symbol must be mentioned at least once")
    return _TaskSegments(
        task=task,
        facts=facts,
        entities=tuple(f"Public action symbol: {entity}" for entity in actions),
        focused_mentions=tuple(item[2] for item in mentions),
        mention_entity_indices=tuple(item[3] for item in mentions),
        mention_fact_indices=tuple(item[0] for item in mentions),
    )


def _generate_public_task(
    family_id: str,
    *,
    structure_seed: int,
    surface_seed: int,
    entity_size: int,
    demonstration_count: int,
    unseen_wording: bool,
) -> tuple[PublicTask, HiddenSymbolicRuleSolution | None]:
    if family_id == _FAMILY_RELATIONAL:
        generated = generate_relational_task(
            structure_seed,
            item_count=entity_size,
            surface_forms=(
                _UNSEEN_SURFACE_FORMS
                if unseen_wording
                else _TRAINING_SURFACE_FORMS
            ),
        )
        return generated.learner, None
    if family_id == _FAMILY_SYMBOLIC:
        generated_symbolic = generate_symbolic_rule_task(
            structure_seed,
            item_count=entity_size,
            demonstration_count=demonstration_count,
            surface_seed=surface_seed,
            goal_surface_forms=(
                _SYMBOLIC_UNSEEN_GOAL_FORMS
                if unseen_wording
                else _SYMBOLIC_TRAINING_GOAL_FORMS
            ),
            demonstration_surface_forms=(
                _SYMBOLIC_UNSEEN_DEMO_FORMS
                if unseen_wording
                else _SYMBOLIC_TRAINING_DEMO_FORMS
            ),
            query_surface_forms=(
                _SYMBOLIC_UNSEEN_QUERY_FORMS
                if unseen_wording
                else _SYMBOLIC_TRAINING_QUERY_FORMS
            ),
        )
        return generated_symbolic.learner, generated_symbolic.hidden
    raise ValueError(f"unsupported task family: {family_id}")


def generate_balanced_partitions(
    settings: RunSettings,
    *,
    seed: int,
) -> tuple[
    tuple[PublicTask, ...],
    tuple[PublicTask, ...],
    dict[str, str],
    _SealedOutcomeOracle,
]:
    """Generate an exactly family/size-balanced, single-encounter stream."""

    cell_count = len(_FAMILY_IDS) * len(settings.entity_sizes)
    training_per_cell = settings.training_tasks // cell_count
    heldout_per_cell = settings.heldout_tasks // cell_count
    training_cells: dict[tuple[str, int], list[PublicTask]] = {}
    heldout_cells: dict[tuple[str, int], list[PublicTask]] = {}
    partitions: dict[str, str] = {}
    sealed_symbolic: dict[str, HiddenSymbolicRuleSolution] = {}
    for size_index, entity_size in enumerate(settings.entity_sizes):
        for family_index, family_id in enumerate(_FAMILY_IDS):
            cell = (family_id, entity_size)
            training_cells[cell] = []
            heldout_cells[cell] = []
            domain = family_index * 10_000_019 + size_index * 1_000_003
            for index in range(training_per_cell):
                structure_seed = seed + domain + index * 101
                task, hidden = _generate_public_task(
                    family_id,
                    structure_seed=structure_seed,
                    surface_seed=structure_seed + 97_409,
                    entity_size=entity_size,
                    demonstration_count=2 + ((index + size_index) % 3),
                    unseen_wording=False,
                )
                training_cells[cell].append(task)
                partitions[task.instance_id] = "training_seen_wording"
                if hidden is not None:
                    sealed_symbolic[task.instance_id] = hidden
            for index in range(heldout_per_cell):
                unseen = (index + family_index + size_index) % 2 == 1
                structure_seed = seed + 100_000_007 + domain + index * 103
                task, hidden = _generate_public_task(
                    family_id,
                    structure_seed=structure_seed,
                    surface_seed=structure_seed + 193_939,
                    entity_size=entity_size,
                    demonstration_count=2 + ((index + family_index) % 3),
                    unseen_wording=unseen,
                )
                heldout_cells[cell].append(task)
                partitions[task.instance_id] = (
                    "heldout_unseen_wording"
                    if unseen
                    else "heldout_seen_wording"
                )
                if hidden is not None:
                    sealed_symbolic[task.instance_id] = hidden

    # Each contiguous full-profile cohort has one task from every family/size
    # cell.  Cohort blocks may be shuffled later without destroying balance.
    training = tuple(
        training_cells[(family_id, entity_size)][index]
        for index in range(training_per_cell)
        for entity_size in settings.entity_sizes
        for family_id in _FAMILY_IDS
    )
    heldout = tuple(
        heldout_cells[(family_id, entity_size)][index]
        for index in range(heldout_per_cell)
        for entity_size in settings.entity_sizes
        for family_id in _FAMILY_IDS
    )
    if len(partitions) != len(training) + len(heldout):
        raise RuntimeError("training and heldout task identities must be unique")
    if sum(task.family_id == _FAMILY_RELATIONAL for task in training) * 2 != len(training):
        raise RuntimeError("training task families are not exactly balanced")
    if sum(task.family_id == _FAMILY_RELATIONAL for task in heldout) * 2 != len(heldout):
        raise RuntimeError("heldout task families are not exactly balanced")
    return training, heldout, partitions, _SealedOutcomeOracle(sealed_symbolic)


def encode_task_knowledge(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: Sequence[PublicTask],
    *,
    surface_partitions: Mapping[str, str],
    feature_batch_size: int,
    hidden_state_index: int,
) -> tuple[TaskKnowledge, ...]:
    segmented = tuple(task_segments(task) for task in tasks)
    unique_texts = tuple(
        dict.fromkeys(
            text
            for item in segmented
            for text in (*item.facts, *item.entities, *item.focused_mentions)
        )
    )
    encoded = encode_detached_segments(
        model,
        tokenizer,
        unique_texts,
        batch_size=feature_batch_size,
        hidden_state_index=hidden_state_index,
    )
    feature_by_text = {
        text: encoded[index] for index, text in enumerate(unique_texts)
    }
    result: list[TaskKnowledge] = []
    for item in segmented:
        result.append(
            TaskKnowledge(
                task=item.task,
                surface_partition=surface_partitions[item.task.instance_id],
                fact_features=torch.stack(
                    [feature_by_text[text] for text in item.facts]
                ).unsqueeze(0),
                entity_features=torch.stack(
                    [feature_by_text[text] for text in item.entities]
                ).unsqueeze(0),
                mention_features=torch.stack(
                    [feature_by_text[text] for text in item.focused_mentions]
                ).unsqueeze(0),
                mention_entity_indices=torch.tensor(
                    [item.mention_entity_indices], dtype=torch.long
                ),
                mention_fact_indices=torch.tensor(
                    [item.mention_fact_indices], dtype=torch.long
                ),
            )
        )
    return tuple(result)


def _symbols_from_indices(
    task: PublicTask,
    order_indices: Iterable[int],
) -> tuple[str, ...]:
    symbols = _action_symbols(task)
    indices = tuple(int(index) for index in order_indices if int(index) >= 0)
    if len(indices) != len(symbols) or set(indices) != set(range(len(symbols))):
        raise RuntimeError("the neural policy emitted a non-permutation")
    return tuple(symbols[index] for index in indices)


def _score_sampled_orders(
    item: TaskKnowledge,
    order_indices: torch.Tensor,
    oracle: _SealedOutcomeOracle,
    *,
    exact_reward_bonus: float,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...]]:
    rewards: list[float] = []
    records: list[dict[str, Any]] = []
    for sample in order_indices.detach().cpu():
        order = _symbols_from_indices(item.task, sample.tolist())
        outcome = oracle.score(item.task, order)
        rewards.append(
            outcome.satisfaction
            + exact_reward_bonus * int(outcome.exact)
        )
        records.append(
            {
                "valid": outcome.valid,
                "exact": outcome.exact,
                "satisfaction": outcome.satisfaction,
            }
        )
    return (
        torch.tensor(rewards, dtype=torch.float32, device=order_indices.device),
        tuple(records),
    )


@torch.no_grad()
def evaluate(
    core: RecurrentReasoningCore,
    tasks: Sequence[TaskKnowledge],
    oracle: _SealedOutcomeOracle,
    *,
    device: torch.device,
    reasoning_steps: int,
    remove_fact_evidence: bool = False,
    derange_incidence: bool = False,
) -> EvaluationSummary:
    prior_training = core.training
    core.eval()
    exact = 0
    satisfaction = 0.0
    strata: dict[int, dict[str, float | int]] = {}
    family_strata: dict[str, dict[str, float | int]] = {}
    surface_strata: dict[str, dict[str, float | int]] = {}
    orders: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    for item in tasks:
        trajectory = core.act(
            samples_per_task=1,
            greedy=True,
            reasoning_steps=reasoning_steps,
            **item.model_inputs(
                device,
                remove_fact_evidence=remove_fact_evidence,
                derange_incidence=derange_incidence,
            ),
        )
        order = _symbols_from_indices(
            item.task,
            trajectory.order_indices[0, 0].cpu().tolist(),
        )
        outcome = oracle.score(item.task, order)
        exact += int(outcome.exact)
        satisfaction += outcome.satisfaction
        entity_size = len(_action_symbols(item.task))
        family_id = item.task.family_id
        stratum = strata.setdefault(
            entity_size,
            {"exact": 0, "total": 0, "satisfaction_sum": 0.0},
        )
        stratum["exact"] = int(stratum["exact"]) + int(outcome.exact)
        stratum["total"] = int(stratum["total"]) + 1
        stratum["satisfaction_sum"] = (
            float(stratum["satisfaction_sum"])
            + outcome.satisfaction
        )
        family_stratum = family_strata.setdefault(
            family_id,
            {"exact": 0, "total": 0, "satisfaction_sum": 0.0},
        )
        family_stratum["exact"] = (
            int(family_stratum["exact"]) + int(outcome.exact)
        )
        family_stratum["total"] = int(family_stratum["total"]) + 1
        family_stratum["satisfaction_sum"] = (
            float(family_stratum["satisfaction_sum"]) + outcome.satisfaction
        )
        surface_stratum = surface_strata.setdefault(
            item.surface_partition,
            {"exact": 0, "total": 0, "satisfaction_sum": 0.0},
        )
        surface_stratum["exact"] = (
            int(surface_stratum["exact"]) + int(outcome.exact)
        )
        surface_stratum["total"] = int(surface_stratum["total"]) + 1
        surface_stratum["satisfaction_sum"] = (
            float(surface_stratum["satisfaction_sum"])
            + outcome.satisfaction
        )
        orders.append(order)
        records.append(
            {
                "task_id": item.task.instance_id,
                "valid": outcome.valid,
                "exact": outcome.exact,
                "satisfaction": outcome.satisfaction,
                "family_id": family_id,
                "entity_size": entity_size,
                "surface_partition": item.surface_partition,
            }
        )
    core.train(prior_training)
    by_entity_size = {
        size: {
            "exact": int(values["exact"]),
            "total": int(values["total"]),
            "exact_accuracy": int(values["exact"]) / int(values["total"]),
            "mean_satisfaction": (
                float(values["satisfaction_sum"]) / int(values["total"])
            ),
        }
        for size, values in strata.items()
    }
    by_family = {
        name: {
            "exact": int(values["exact"]),
            "total": int(values["total"]),
            "exact_accuracy": int(values["exact"]) / int(values["total"]),
            "mean_satisfaction": (
                float(values["satisfaction_sum"]) / int(values["total"])
            ),
        }
        for name, values in family_strata.items()
    }
    by_surface_partition = {
        name: {
            "exact": int(values["exact"]),
            "total": int(values["total"]),
            "exact_accuracy": int(values["exact"]) / int(values["total"]),
            "mean_satisfaction": (
                float(values["satisfaction_sum"]) / int(values["total"])
            ),
        }
        for name, values in surface_strata.items()
    }
    return EvaluationSummary(
        exact=exact,
        total=len(tasks),
        mean_satisfaction=(satisfaction / len(tasks) if tasks else 0.0),
        by_family=by_family,
        by_entity_size=by_entity_size,
        by_surface_partition=by_surface_partition,
        orders=tuple(orders),
        records=tuple(records),
    )


def _state_delta_norm(
    core: RecurrentReasoningCore,
    parent: Mapping[str, torch.Tensor],
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name, value in core.state_dict().items():
        difference = value.detach().cpu().double() - parent[name].double()
        total += difference.square().sum()
    return math.sqrt(float(total.item()))


def train_outcome_policy(
    core: RecurrentReasoningCore,
    tasks: Sequence[TaskKnowledge],
    oracle: _SealedOutcomeOracle,
    parent_snapshot: Mapping[str, torch.Tensor],
    settings: RunSettings,
    *,
    device: torch.device,
    entropy_coefficient: float,
    max_gradient_norm: float,
    max_state_delta_norm: float,
    exact_reward_bonus: float,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    parameters = tuple(core.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("every reasoning-core parameter must be trainable")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=0.0,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError("optimizer scope is not exactly the reasoning core")
    task_ids = [item.task.instance_id for item in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("the experience stream contains a repeated task identity")

    core.train(True)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    optimizer_steps = 0
    constant_reward_groups = 0
    total_unique_sampled_orders = 0
    cohorts = [
        list(range(start, start + settings.cohort_size))
        for start in range(0, len(tasks), settings.cohort_size)
    ]
    rng.shuffle(cohorts)
    for cohort in cohorts:
        optimizer.zero_grad(set_to_none=True)
        for task_index in cohort:
            item = tasks[task_index]
            trajectory = core.act(
                samples_per_task=settings.samples_per_presentation,
                greedy=False,
                reasoning_steps=settings.reasoning_steps,
                temperature=settings.sampling_temperature,
                **item.model_inputs(device),
            )
            rewards, outcomes = _score_sampled_orders(
                item,
                trajectory.order_indices[0],
                oracle,
                exact_reward_bonus=exact_reward_bonus,
            )
            reward_std = rewards.std(unbiased=False)
            constant_reward_groups += int(float(reward_std.item()) <= 1e-8)
            sampled_orders = {
                tuple(int(index) for index in sample.tolist())
                for sample in trajectory.order_indices[0].detach().cpu()
            }
            total_unique_sampled_orders += len(sampled_orders)

            advantages = leave_one_out_advantages(rewards)
            entity_count = len(_action_symbols(item.task))
            log_probability = trajectory.log_probability[0] / entity_count
            entropy = trajectory.entropy[0] / entity_count
            policy_loss = -(advantages * log_probability).mean()
            loss = policy_loss - entropy_coefficient * entropy.mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("the outcome-policy loss is not finite")
            (loss / len(cohort)).backward()
            records.append(
                {
                    "presentation": 1,
                    "task_id": item.task.instance_id,
                    "family_id": item.task.family_id,
                    "reward_mean": float(rewards.mean().item()),
                    "reward_min": float(rewards.min().item()),
                    "reward_max": float(rewards.max().item()),
                    "exact_samples": sum(int(record["exact"]) for record in outcomes),
                    "constant_sample_reward": float(reward_std.item()) <= 1e-8,
                    "unique_sampled_orders": len(sampled_orders),
                    "loss": float(loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "advantage_mean": float(advantages.mean().item()),
                    "mean_entropy": float(entropy.detach().mean().cpu().item()),
                }
            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_gradient_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer_steps += 1
        if not all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in parameters
        ):
            raise RuntimeError("the reasoning state contains a non-finite tensor")
        records[-1]["cohort_gradient_norm_before_clip"] = float(
            gradient_norm.detach().cpu().item()
        )

    optimizer.zero_grad(set_to_none=True)
    delta_norm = _state_delta_norm(core, parent_snapshot)
    if delta_norm > max_state_delta_norm:
        raise RuntimeError(
            f"reasoning-state delta {delta_norm:.8g} exceeds {max_state_delta_norm:.8g}"
        )
    records.append(
        {
            "summary": True,
            "optimizer_steps": optimizer_steps,
            "experience_count": len(tasks),
            "unique_task_ids": len(set(task_ids)),
            "replayed_task_ids": 0,
            "constant_reward_groups": constant_reward_groups,
            "mean_unique_orders_per_experience": (
                total_unique_sampled_orders
                / len(tasks)
            ),
            "state_delta_norm": delta_norm,
            "trust_region_projection": None,
        }
    )
    return tuple(records)


def _load_parent_state(
    core: RecurrentReasoningCore,
    directory: Path,
) -> dict[str, Any]:
    metadata_path = directory / _METADATA_FILENAME
    tensor_path = directory / _STATE_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != _STATE_FORMAT:
        raise RuntimeError("parent reasoning-state format is unsupported")
    if metadata.get("config") != asdict(core.config):
        raise RuntimeError("parent reasoning-state topology is incompatible")
    restore_reasoning_state(core, load_file(str(tensor_path), device="cpu"))
    digest = reasoning_state_digest(core)
    if digest != metadata.get("state_digest"):
        raise RuntimeError("parent reasoning-state digest does not match metadata")
    return metadata


def _stage_candidate_state(
    core: RecurrentReasoningCore,
    candidate_dir: Path,
    *,
    foundation_digest: str,
    parent_digest: str,
) -> Path:
    staging = candidate_dir.with_name(
        f".{candidate_dir.name}.staging-{os.getpid()}"
    )
    if candidate_dir.exists() or staging.exists():
        raise FileExistsError(candidate_dir if candidate_dir.exists() else staging)
    staging.mkdir(parents=True)
    snapshot = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in core.state_dict().items()
    }
    digest = reasoning_state_digest(core)
    save_file(snapshot, str(staging / _STATE_FILENAME))
    metadata = {
        "format": _STATE_FORMAT,
        "config": asdict(core.config),
        "state_digest": digest,
        "parent_state_digest": parent_digest,
        "foundation_digest": foundation_digest,
        "knowledge_interface": _KNOWLEDGE_INTERFACE,
    }
    (staging / _METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staging


def _verify_staged_state(
    staging: Path,
    config: ReasoningCoreConfig,
    expected_digest: str,
    *,
    device: torch.device,
) -> RecurrentReasoningCore:
    reloaded = RecurrentReasoningCore(config).to(device)
    metadata = _load_parent_state(reloaded, staging)
    if metadata["state_digest"] != expected_digest:
        raise RuntimeError("reloaded candidate has the wrong state identity")
    return reloaded


def _assert_foundation_inert(model: torch.nn.Module) -> None:
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("the knowledge backbone became trainable")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("the knowledge backbone received a gradient")


def _stratum_gains(
    parent: EvaluationSummary,
    candidate: EvaluationSummary,
) -> tuple[dict[str, dict[str, float | int]], int]:
    gains: dict[str, dict[str, float | int]] = {}
    improved = 0
    for size in sorted(parent.by_entity_size):
        parent_metrics = parent.by_entity_size[size]
        candidate_metrics = candidate.by_entity_size[size]
        satisfaction_gain = (
            float(candidate_metrics["mean_satisfaction"])
            - float(parent_metrics["mean_satisfaction"])
        )
        exact_gain = int(candidate_metrics["exact"]) - int(parent_metrics["exact"])
        if satisfaction_gain > 0.0:
            improved += 1
        gains[str(size)] = {
            "exact_gain": exact_gain,
            "satisfaction_gain": satisfaction_gain,
            "improved": satisfaction_gain > 0.0,
        }
    return gains, improved


def _surface_partition_gains(
    parent: EvaluationSummary,
    candidate: EvaluationSummary,
) -> dict[str, dict[str, float | int]]:
    gains: dict[str, dict[str, float | int]] = {}
    for name in sorted(parent.by_surface_partition):
        parent_metrics = parent.by_surface_partition[name]
        candidate_metrics = candidate.by_surface_partition[name]
        gains[name] = {
            "exact_gain": (
                int(candidate_metrics["exact"]) - int(parent_metrics["exact"])
            ),
            "satisfaction_gain": (
                float(candidate_metrics["mean_satisfaction"])
                - float(parent_metrics["mean_satisfaction"])
            ),
        }
    return gains


def _family_gains(
    parent: EvaluationSummary,
    candidate: EvaluationSummary,
) -> dict[str, dict[str, float | int]]:
    gains: dict[str, dict[str, float | int]] = {}
    if set(parent.by_family) != set(candidate.by_family):
        raise RuntimeError("parent and candidate family strata differ")
    for family_id in sorted(parent.by_family):
        parent_metrics = parent.by_family[family_id]
        candidate_metrics = candidate.by_family[family_id]
        gains[family_id] = {
            "exact_gain": (
                int(candidate_metrics["exact"]) - int(parent_metrics["exact"])
            ),
            "satisfaction_gain": (
                float(candidate_metrics["mean_satisfaction"])
                - float(parent_metrics["mean_satisfaction"])
            ),
        }
    return gains


def _full_protocol_eligible(
    settings: RunSettings,
    thresholds: RetentionThresholds,
    *,
    args: argparse.Namespace,
    model_path: Path,
    foundation_digest: str,
) -> bool:
    expected_settings = RunSettings(**_PROFILE_DEFAULTS["full"])  # type: ignore[arg-type]
    return (
        args.profile == "full"
        and settings == expected_settings
        and thresholds.maximum_state_delta_norm == 5.0
        and thresholds.exact_reward_bonus == 1.0
        and args.seed == 4701
        and args.dtype == "bfloat16"
        and args.attention_implementation == "sdpa"
        and args.hidden_state_index == -1
        and args.feature_batch_size == 64
        and args.entropy_coefficient == 0.01
        and args.max_gradient_norm == 1.0
        and args.parent_state is None
        and args.device == "cuda:0"
        and model_path == _FULL_MODEL_PATH
        and foundation_digest == _FULL_FOUNDATION_DIGEST
    )


def _write_result_json(result: Mapping[str, Any], destination: Path) -> None:
    if not destination.is_absolute():
        raise ValueError("result-json must be an absolute local path")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    thresholds = resolve_thresholds(args)
    if args.feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    if args.entropy_coefficient < 0.0:
        raise ValueError("entropy_coefficient must be nonnegative")
    if args.max_gradient_norm <= 0.0:
        raise ValueError("the gradient bound must be positive")

    device = torch.device(args.device)
    model_path = Path(args.model).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser()
    if not candidate_dir.is_absolute():
        raise ValueError("candidate-dir must be an absolute local path")
    candidate_dir = candidate_dir.resolve()
    parent_dir = (
        Path(args.parent_state).expanduser().resolve()
        if args.parent_state
        else None
    )
    if parent_dir == candidate_dir:
        raise ValueError("parent-state and candidate-dir must differ")
    if candidate_dir.exists():
        raise FileExistsError(candidate_dir)
    result_json = (
        Path(args.result_json).expanduser()
        if args.result_json is not None
        else None
    )
    if result_json is not None:
        if not result_json.is_absolute():
            raise ValueError("result-json must be an absolute local path")
        result_json = result_json.resolve()
        if result_json.exists():
            raise FileExistsError(result_json)
        if candidate_dir == result_json or candidate_dir in result_json.parents:
            raise ValueError("result-json must be outside candidate-dir")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cuda_device_index: int | None = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        cuda_device_index = torch.cuda.current_device()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
    started = time.perf_counter()

    print("loading frozen hidden-state donor", file=sys.stderr, flush=True)
    foundation, tokenizer = load_frozen_knowledge_backbone(
        model_path,
        device=device,
        dtype=_torch_dtype(args.dtype),
        attention_implementation=args.attention_implementation,
    )
    foundation_before = foundation_tensor_digest(foundation)
    if (
        args.profile == "full"
        and model_path == _FULL_MODEL_PATH
        and foundation_before != _FULL_FOUNDATION_DIGEST
    ):
        raise RuntimeError("the full protocol donor fingerprint is not the pinned Qwen3-4B")

    training_tasks, heldout_tasks, surface_partitions, outcome_oracle = (
        generate_balanced_partitions(settings, seed=args.seed)
    )
    print("encoding independent public segments", file=sys.stderr, flush=True)
    encoded = encode_task_knowledge(
        foundation,
        tokenizer,
        (*training_tasks, *heldout_tasks),
        surface_partitions=surface_partitions,
        feature_batch_size=args.feature_batch_size,
        hidden_state_index=args.hidden_state_index,
    )
    training_knowledge = encoded[: settings.training_tasks]
    heldout_knowledge = encoded[settings.training_tasks :]
    knowledge_width = int(encoded[0].fact_features.shape[-1])
    config = ReasoningCoreConfig(
        knowledge_width=knowledge_width,
        core_width=settings.core_width,
        workspace_slots=settings.workspace_slots,
        attention_heads=settings.attention_heads,
        feedforward_width=settings.feedforward_width,
        reasoning_steps=settings.reasoning_steps,
        maximum_reasoning_steps=settings.maximum_reasoning_steps,
        maximum_entities=max(settings.entity_sizes),
    )
    core = RecurrentReasoningCore(config).to(device)
    parent_metadata: dict[str, Any] | None = None
    if parent_dir is not None:
        parent_metadata = _load_parent_state(core, parent_dir)

    parent_digest = reasoning_state_digest(core)
    parent_snapshot = snapshot_reasoning_state(core)
    parent_heldout = evaluate(
        core,
        heldout_knowledge,
        outcome_oracle,
        device=device,
        reasoning_steps=settings.reasoning_steps,
    )
    parent_no_recurrence = evaluate(
        core,
        heldout_knowledge,
        outcome_oracle,
        device=device,
        reasoning_steps=0,
    )
    parent_one_recurrent_step = evaluate(
        core,
        heldout_knowledge,
        outcome_oracle,
        device=device,
        reasoning_steps=1,
    )
    parent_no_fact_evidence = evaluate(
        core,
        heldout_knowledge,
        outcome_oracle,
        device=device,
        reasoning_steps=settings.reasoning_steps,
        remove_fact_evidence=True,
    )
    parent_deranged_incidence = evaluate(
        core,
        heldout_knowledge,
        outcome_oracle,
        device=device,
        reasoning_steps=settings.reasoning_steps,
        derange_incidence=True,
    )

    try:
        print("learning from the unique scalar-outcome stream", file=sys.stderr, flush=True)
        training_receipts = train_outcome_policy(
            core,
            training_knowledge,
            outcome_oracle,
            parent_snapshot,
            settings,
            device=device,
            entropy_coefficient=args.entropy_coefficient,
            max_gradient_norm=args.max_gradient_norm,
            max_state_delta_norm=thresholds.maximum_state_delta_norm,
            exact_reward_bonus=thresholds.exact_reward_bonus,
            seed=args.seed + 7_919,
        )
        candidate_digest = reasoning_state_digest(core)
        state_changed = candidate_digest != parent_digest

        candidate_heldout = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=settings.reasoning_steps,
        )
        no_recurrence = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=0,
        )
        one_recurrent_step = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=1,
        )
        no_fact_evidence = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=settings.reasoning_steps,
            remove_fact_evidence=True,
        )
        deranged_incidence = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=settings.reasoning_steps,
            derange_incidence=True,
        )
        _assert_foundation_inert(foundation)
        foundation_after = foundation_tensor_digest(foundation)
        if foundation_after != foundation_before:
            raise RuntimeError("the frozen knowledge-backbone fingerprint changed")

        satisfaction_gain = (
            candidate_heldout.mean_satisfaction
            - parent_heldout.mean_satisfaction
        )
        exact_gain = candidate_heldout.exact - parent_heldout.exact
        stratum_gains, improved_strata = _stratum_gains(
            parent_heldout,
            candidate_heldout,
        )
        surface_partition_gains = _surface_partition_gains(
            parent_heldout,
            candidate_heldout,
        )
        family_gains = _family_gains(parent_heldout, candidate_heldout)
        seen_wording_gain = float(
            surface_partition_gains["heldout_seen_wording"]["satisfaction_gain"]
        )
        unseen_wording_gain = float(
            surface_partition_gains["heldout_unseen_wording"][
                "satisfaction_gain"
            ]
        )
        recurrence_ablated_gain = (
            no_recurrence.mean_satisfaction
            - parent_no_recurrence.mean_satisfaction
        )
        one_step_ablated_gain = (
            one_recurrent_step.mean_satisfaction
            - parent_one_recurrent_step.mean_satisfaction
        )
        fact_ablated_gain = (
            no_fact_evidence.mean_satisfaction
            - parent_no_fact_evidence.mean_satisfaction
        )
        deranged_incidence_gain = (
            deranged_incidence.mean_satisfaction
            - parent_deranged_incidence.mean_satisfaction
        )
        recurrence_erasure = satisfaction_gain - recurrence_ablated_gain
        one_step_erasure = satisfaction_gain - one_step_ablated_gain
        fact_erasure = satisfaction_gain - fact_ablated_gain
        incidence_erasure = satisfaction_gain - deranged_incidence_gain

        candidate_snapshot = snapshot_reasoning_state(core)
        restore_reasoning_state(core, parent_snapshot)
        swapped_parent = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=settings.reasoning_steps,
        )
        parent_swap_verified = (
            reasoning_state_digest(core) == parent_digest
            and swapped_parent.orders == parent_heldout.orders
        )
        restore_reasoning_state(core, candidate_snapshot)
        swapped_candidate = evaluate(
            core,
            heldout_knowledge,
            outcome_oracle,
            device=device,
            reasoning_steps=settings.reasoning_steps,
        )
        candidate_swap_verified = (
            reasoning_state_digest(core) == candidate_digest
            and swapped_candidate.orders == candidate_heldout.orders
        )
        state_swap_verified = parent_swap_verified and candidate_swap_verified

        families_improved = all(
            float(metrics["satisfaction_gain"]) > 0.0
            and int(metrics["exact_gain"]) >= 0
            for metrics in family_gains.values()
        )
        preliminary_retention = (
            state_changed
            and satisfaction_gain > 0.0
            and exact_gain >= 0
            and families_improved
            and seen_wording_gain > 0.0
            and unseen_wording_gain > 0.0
            and recurrence_erasure > 0.0
            and one_step_erasure > 0.0
            and fact_erasure > 0.0
            and incidence_erasure > 0.0
            and state_swap_verified
            and foundation_after == foundation_before
        )
        saved_path: str | None = None
        reload_verified = False
        retained = False
        if preliminary_retention:
            staging = _stage_candidate_state(
                core,
                candidate_dir,
                foundation_digest=foundation_after,
                parent_digest=parent_digest,
            )
            reloaded = _verify_staged_state(
                staging,
                config,
                candidate_digest,
                device=device,
            )
            reloaded_heldout = evaluate(
                reloaded,
                heldout_knowledge,
                outcome_oracle,
                device=device,
                reasoning_steps=settings.reasoning_steps,
            )
            if (
                reloaded_heldout.orders != candidate_heldout.orders
                or reloaded_heldout.exact != candidate_heldout.exact
                or reloaded_heldout.mean_satisfaction
                != candidate_heldout.mean_satisfaction
            ):
                raise RuntimeError("reloaded reasoning state changed behavior")
            staging.rename(candidate_dir)
            saved_path = str(candidate_dir)
            reload_verified = True
            retained = True
            final_digest = candidate_digest
            status = "CANDIDATE_RETAINED"
        else:
            restore_reasoning_state(core, parent_snapshot)
            final_digest = reasoning_state_digest(core)
            if final_digest != parent_digest:
                raise RuntimeError("candidate rejection failed exact state rollback")
            status = "CANDIDATE_REJECTED_ROLLED_BACK"

        if cuda_device_index is not None:
            torch.cuda.synchronize(cuda_device_index)
        finished = time.perf_counter()
        summary_receipt = training_receipts[-1]
        reported_receipts: Sequence[dict[str, Any]] = (
            training_receipts if args.include_records else (summary_receipt,)
        )
        result = {
            "experiment": "ANGLER-CROSS-FAMILY-RLOO-V1",
            "status": status,
            "profile": args.profile,
            "claim_level": (
                "predeclared_full_experiment"
                if _full_protocol_eligible(
                    settings,
                    thresholds,
                    args=args,
                    model_path=model_path,
                    foundation_digest=foundation_after,
                )
                else "development_only_not_full_protocol"
            ),
            "model_path": str(model_path),
            "run_settings": {
                "seed": args.seed,
                "profile_settings": asdict(settings),
                "retention_bounds": asdict(thresholds),
                "device": args.device,
                "dtype": args.dtype,
                "attention_implementation": args.attention_implementation,
                "hidden_state_index": args.hidden_state_index,
                "feature_batch_size": args.feature_batch_size,
                "entropy_coefficient": args.entropy_coefficient,
                "maximum_gradient_norm": args.max_gradient_norm,
            },
            "knowledge_interface": {
                "id": _KNOWLEDGE_INTERFACE,
                "transformers_class": type(foundation).__name__,
                "knowledge_width_inferred": knowledge_width,
                "hidden_state_index": args.hidden_state_index,
                "segment_policy": (
                    "independent public goals/facts, action labels, and literal "
                    "identity-focused single-fact mentions only"
                ),
                "assembled_problem_encoded": False,
                "language_generation_used": False,
                "foundation_trainable_parameters": sum(
                    parameter.numel()
                    for parameter in foundation.parameters()
                    if parameter.requires_grad
                ),
                "foundation_digest_before": foundation_before,
                "foundation_digest_after": foundation_after,
            },
            "reasoning_core": {
                "config": asdict(config),
                "parameters": core.parameter_count(),
                "parent_digest": parent_digest,
                "parent_path": str(parent_dir) if parent_dir else None,
                "parent_metadata": parent_metadata,
                "candidate_digest": candidate_digest,
                "candidate_changed": state_changed,
                "final_digest": final_digest,
                "saved_path": saved_path,
                "save_reload_exact": reload_verified,
            },
            "experience": {
                "training_task_variations": settings.training_tasks,
                "heldout_task_variations": settings.heldout_tasks,
                "entity_sizes": list(settings.entity_sizes),
                "training_per_entity_size": (
                    settings.training_tasks // len(settings.entity_sizes)
                ),
                "heldout_per_entity_size": (
                    settings.heldout_tasks // len(settings.entity_sizes)
                ),
                "training_per_family": settings.training_tasks // len(_FAMILY_IDS),
                "heldout_per_family": settings.heldout_tasks // len(_FAMILY_IDS),
                "training_per_family_entity_size": (
                    settings.training_tasks
                    // (len(_FAMILY_IDS) * len(settings.entity_sizes))
                ),
                "heldout_per_family_entity_size": (
                    settings.heldout_tasks
                    // (len(_FAMILY_IDS) * len(settings.entity_sizes))
                ),
                "families": list(_FAMILY_IDS),
                "relational_training_surface_forms": list(_TRAINING_SURFACE_FORMS),
                "relational_unseen_surface_forms": list(_UNSEEN_SURFACE_FORMS),
                "symbolic_training_surface_forms": {
                    "goal": list(_SYMBOLIC_TRAINING_GOAL_FORMS),
                    "demonstration": list(_SYMBOLIC_TRAINING_DEMO_FORMS),
                    "query": list(_SYMBOLIC_TRAINING_QUERY_FORMS),
                },
                "symbolic_unseen_surface_forms": {
                    "goal": list(_SYMBOLIC_UNSEEN_GOAL_FORMS),
                    "demonstration": list(_SYMBOLIC_UNSEEN_DEMO_FORMS),
                    "query": list(_SYMBOLIC_UNSEEN_QUERY_FORMS),
                },
                "heldout_seen_wording_tasks": settings.heldout_tasks // 2,
                "heldout_unseen_wording_tasks": settings.heldout_tasks // 2,
                "presentations_per_task": settings.presentations_per_task,
                "samples_per_presentation": settings.samples_per_presentation,
                "unique_task_stream": settings.presentations_per_task == 1,
                "task_replay_used": settings.presentations_per_task != 1,
                "cohort_size_per_optimizer_step": settings.cohort_size,
                "each_cohort_balanced_by_family_and_size": True,
                "sampling_temperature": settings.sampling_temperature,
                "credit_assignment": "leave_one_out_reinforce",
                "learned_value_used_by_optimizer": False,
                "total_sampled_outcomes": (
                    settings.training_tasks
                    * settings.presentations_per_task
                    * settings.samples_per_presentation
                ),
                "feedback_to_optimizer": (
                    "generic scalar outcome satisfaction plus scalar exact bonus"
                ),
                "exact_reward_bonus": thresholds.exact_reward_bonus,
                "teacher_answers": False,
                "sealed_symbolic_solutions_supplied_to_learner": False,
                "sealed_symbolic_solutions_serialized": False,
                "normalized_constraints_passed_to_core_or_loss": False,
                "sealed_truth_used_only_by_external_outcome_verifier": True,
                "relation_parser": False,
                "training_receipts": list(reported_receipts),
            },
            "heldout_parent": parent_heldout.public(
                include_records=args.include_records
            ),
            "heldout_candidate": candidate_heldout.public(
                include_records=args.include_records
            ),
            "improvement_decision": {
                "required_conditions": {
                    "overall_satisfaction_gain_positive": True,
                    "global_exact_nondecline": True,
                    "each_family_satisfaction_gain_positive": True,
                    "each_family_exact_nondecline": True,
                    "seen_wording_satisfaction_gain_positive": True,
                    "unseen_wording_satisfaction_gain_positive": True,
                    "each_causal_control_erases_positive_gain": True,
                    "state_swap_reload_and_frozen_donor_exact": True,
                },
                "bounds": asdict(thresholds),
                "exact_gain": exact_gain,
                "satisfaction_gain": satisfaction_gain,
                "stratum_gains": stratum_gains,
                "improved_strata": improved_strata,
                "family_gains": family_gains,
                "families_improved_without_exact_decline": families_improved,
                "surface_partition_gains": surface_partition_gains,
                "seen_wording_satisfaction_gain": seen_wording_gain,
                "unseen_wording_satisfaction_gain": unseen_wording_gain,
                "preliminary_retention": preliminary_retention,
                "save_reload_required_and_exact": reload_verified,
                "retained": retained,
            },
            "causal_ablations": {
                "zero_recurrent_steps": {
                    "parent": parent_no_recurrence.public(
                        include_records=args.include_records
                    ),
                    "candidate": no_recurrence.public(
                        include_records=args.include_records
                    ),
                    "ablated_parent_to_candidate_gain": recurrence_ablated_gain,
                },
                "one_recurrent_step": {
                    "parent": parent_one_recurrent_step.public(
                        include_records=args.include_records
                    ),
                    "candidate": one_recurrent_step.public(
                        include_records=args.include_records
                    ),
                    "ablated_parent_to_candidate_gain": one_step_ablated_gain,
                },
                "fact_and_focus_evidence_removed": {
                    "parent": parent_no_fact_evidence.public(
                        include_records=args.include_records
                    ),
                    "candidate": no_fact_evidence.public(
                        include_records=args.include_records
                    ),
                    "ablated_parent_to_candidate_gain": fact_ablated_gain,
                },
                "mention_entity_incidence_deranged": {
                    "parent": parent_deranged_incidence.public(
                        include_records=args.include_records
                    ),
                    "candidate": deranged_incidence.public(
                        include_records=args.include_records
                    ),
                    "ablated_parent_to_candidate_gain": deranged_incidence_gain,
                },
                "full_gain_minus_zero_recurrence_gain": recurrence_erasure,
                "full_gain_minus_one_step_gain": one_step_erasure,
                "full_gain_minus_no_fact_evidence_gain": fact_erasure,
                "full_gain_minus_deranged_incidence_gain": incidence_erasure,
                "zero_recurrence_erases_positive_gain": recurrence_erasure > 0.0,
                "one_step_erases_positive_gain": one_step_erasure > 0.0,
                "fact_removal_erases_positive_gain": fact_erasure > 0.0,
                "incidence_derangement_erases_positive_gain": incidence_erasure > 0.0,
            },
            "state_swap_restore": {
                "parent_actions_reproduced": parent_swap_verified,
                "candidate_actions_reproduced": candidate_swap_verified,
                "exact": state_swap_verified,
            },
            "frozen_donor_exact": foundation_after == foundation_before,
            "rollback": {
                "parent_snapshot_exact": True,
                "performed": not retained,
                "final_matches_parent": final_digest == parent_digest,
            },
            "wall_seconds": round(finished - started, 3),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(cuda_device_index))
                if cuda_device_index is not None
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(cuda_device_index))
                if cuda_device_index is not None
                else 0
            ),
            "local_files_only": True,
        }
        if result_json is not None:
            _write_result_json(result, result_json)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except BaseException:
        restore_reasoning_state(core, parent_snapshot)
        if reasoning_state_digest(core) != parent_digest:
            raise RuntimeError("failure rollback did not restore the parent state")
        _assert_foundation_inert(foundation)
        raise


if __name__ == "__main__":
    main()
