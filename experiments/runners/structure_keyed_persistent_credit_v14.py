"""Frozen V14 structure-keyed persistent procedural-credit experiment.

Deterministic code in this module only embeds public procedure text, preserves
chronology, orchestrates controls, and measures preregistered effects.  The
frozen V13 representation supplies relations; only the shared V14 memory
networks are optimized.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from angler.reasoning.natural_trace_graph_causal_memory import parse_step_trace
from angler.reasoning.oml_natural_trace_representation import representation_metrics
from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditEvent,
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditState,
)
from angler.reasoning.structure_protected_oml import StructureProtectedOMLCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.structure_keyed_persistent_credit_v14 import (
    CORPUS_ID,
    DEVELOPMENT_MECHANISMS,
    EVENTS_PER_MECHANISM,
    FINAL_MECHANISMS,
    SHUFFLE_PERMUTATION,
    STRUCTURAL_REPLICATION_ROWS,
    TRAIN_MECHANISMS,
    build_structure_keyed_persistent_credit_v14,
)
from experiments.corpora.structure_protected_oml_natural_trace_v13 import (
    FinalPartitionSealedError,
)
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners import structure_protected_oml_natural_trace_v13 as v13


IDENTITY = "angler.structure-keyed-persistent-credit.v14-first-result"
SEED = 2026083114
TEMPORAL_WIDTH = 4
TRAIN_STEPS = 96
MECHANISMS_PER_STEP = 8
SCHEDULE_MULTIPLIER = 49
LEARNING_RATE = 3.0e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 2.0
WALL_TIME_CEILING_SECONDS = 60 * 60
GPU_MEMORY_CEILING_BYTES = 12 * 1024**3
CORRUPTIONS = ("reorder", "replacement", "omission", "insertion")

V13_RESULT_SHA256 = "A25B4ABFD119F71D835A7E73A620FDC4CA2686A0301B66DDD047D3DC29DCA6A4"
V13_CHECKPOINT_SHA256 = "A0F4B6B859274B315B311D3DDE2D4EFDD9C6D494C33C006D313B450403AB9E8B"


@dataclass(frozen=True, slots=True)
class EncodedCreditMechanism:
    mechanism_ref: str
    generator_family: str
    transition_group: str
    relation_features: torch.Tensor  # [6,64]
    base_logits: torch.Tensor  # [6]
    outcomes: torch.Tensor  # [6], +/-1
    temporal_features: torch.Tensor  # [6,4]

    def to(self, device: torch.device) -> "EncodedCreditMechanism":
        return EncodedCreditMechanism(
            self.mechanism_ref,
            self.generator_family,
            self.transition_group,
            self.relation_features.to(device=device, dtype=torch.float32),
            self.base_logits.to(device=device, dtype=torch.float32),
            self.outcomes.to(device=device, dtype=torch.float32),
            self.temporal_features.to(device=device, dtype=torch.float32),
        )


@dataclass(frozen=True, slots=True)
class EncodedReplication:
    reference_features: torch.Tensor  # [240,6,S,W], repeated public reference
    reference_mask: torch.Tensor
    candidate_features: torch.Tensor  # [240,6,S,W], first five are candidates
    candidate_mask: torch.Tensor
    target_indices: torch.Tensor
    candidate_types: tuple[tuple[str, ...], ...]
    renderer_pairs: tuple[str, ...]
    target_positions: tuple[int, ...]

    def to(self, device: torch.device) -> "EncodedReplication":
        return EncodedReplication(
            self.reference_features.to(device=device, dtype=torch.float32),
            self.reference_mask.to(device=device),
            self.candidate_features.to(device=device, dtype=torch.float32),
            self.candidate_mask.to(device=device),
            self.target_indices.to(device=device),
            self.candidate_types,
            self.renderer_pairs,
            self.target_positions,
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "runner": Path(__file__).resolve(),
        "v14_core": root / "src/angler/reasoning/structure_keyed_credit_memory.py",
        "v14_corpus": root / "experiments/corpora/structure_keyed_persistent_credit_v14.py",
    }
    own = {name: _sha256(path) for name, path in paths.items()}
    inherited = {
        f"v13_{name}": value.upper()
        for name, value in v13._source_hashes().items()
    }
    overlap = set(own).intersection(inherited)
    if overlap:
        raise RuntimeError(f"V14 source-map key collision: {sorted(overlap)}")
    return {**own, **inherited}


def _tensor_digest(value: torch.Tensor) -> str:
    return v13._tensor_digest(value).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v13._model_digest(model).upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V14 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V14 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _enforce_resource_ceiling(started: float, device: torch.device) -> None:
    if time.perf_counter() - started > WALL_TIME_CEILING_SECONDS:
        raise RuntimeError("V14 exceeded its 60-minute wall-time ceiling")
    if torch.cuda.max_memory_allocated(device) > GPU_MEMORY_CEILING_BYTES:
        raise RuntimeError("V14 exceeded its 12-GiB Angler-device memory ceiling")


def _training_schedule() -> tuple[tuple[int, ...], ...]:
    flat = tuple((position * SCHEDULE_MULTIPLIER) % TRAIN_MECHANISMS for position in range(768))
    if math.gcd(SCHEDULE_MULTIPLIER, TRAIN_MECHANISMS) != 1:
        raise RuntimeError("V14 schedule multiplier is not coprime")
    if Counter(flat) != Counter({index: 2 for index in range(TRAIN_MECHANISMS)}):
        raise RuntimeError("V14 training schedule does not expose every mechanism twice")
    return tuple(
        flat[start : start + MECHANISMS_PER_STEP]
        for start in range(0, len(flat), MECHANISMS_PER_STEP)
    )


def _episode_temporal(episode: Any, event_index: int) -> tuple[float, ...]:
    """Moving-Origin coordinates only; deliberately excludes outcome/metadata."""
    coordinate = episode.temporal
    ordinal = float(coordinate.acquired_ordinal)
    age = float(coordinate.age)
    return (
        ordinal / max(1.0, ordinal + age),
        age / max(1.0, ordinal + age),
        math.log1p(age) / math.log(7.0),
        event_index / float(EVENTS_PER_MECHANISM - 1),
    )


def _encode_credit_mechanisms(
    frozen: StructureProtectedOMLCore,
    encoded: Sequence[v13.EncodedPairMechanism],
    mechanisms: Sequence[Any],
) -> tuple[EncodedCreditMechanism, ...]:
    if len(encoded) != len(mechanisms):
        raise ValueError("V14 encoded/public rows do not align")
    device = next(frozen.parameters()).device
    result = []
    frozen.eval()
    with torch.no_grad():
        for row, mechanism in zip(encoded, mechanisms, strict=True):
            learner = mechanism.to_learner_payload()
            if set(learner) != {"episodes"} or len(learner["episodes"]) != EVENTS_PER_MECHANISM:
                raise RuntimeError("V14 learner boundary changed")
            values = row.to(device)
            relation = frozen.relation_features(
                values.reference_features.unsqueeze(0),
                values.reference_mask.unsqueeze(0),
                values.attempt_features.unsqueeze(0),
                values.attempt_mask.unsqueeze(0),
            )[0]
            base = frozen.functional_logits(
                values.reference_features.unsqueeze(0),
                values.reference_mask.unsqueeze(0),
                values.attempt_features.unsqueeze(0),
                values.attempt_mask.unsqueeze(0),
            )[0]
            temporal = torch.tensor(
                tuple(_episode_temporal(event, index) for index, event in enumerate(mechanism.public.episodes)),
                dtype=torch.float32,
            )
            result.append(
                EncodedCreditMechanism(
                    row.mechanism_ref,
                    row.generator_family,
                    row.transition_group,
                    relation.detach().cpu(),
                    base.detach().cpu(),
                    row.outcomes.detach().cpu(),
                    temporal,
                )
            )
    return tuple(result)


def _encode_replication(qwen: Any, rows: Sequence[Any]) -> EncodedReplication:
    references: list[tuple[tuple[str, ...], ...]] = []
    candidates: list[tuple[tuple[str, ...], ...]] = []
    targets = []
    types = []
    renderers = []
    positions = []
    for row in rows:
        payload = row.to_encoder_payload()
        if set(payload) != {"reference_trace_text", "candidate_attempt_trace_texts"}:
            raise RuntimeError("V14 structural encoder boundary changed")
        reference = tuple(parse_step_trace(row.public.reference_trace_text))
        attempts = tuple(tuple(parse_step_trace(text)) for text in row.public.candidate_attempt_trace_texts)
        references.append((reference,) * 6)
        candidates.append((*attempts, attempts[0]))
        targets.append(row.supervision.target_candidate_index)
        types.append(tuple(row.supervision.candidate_corruption_types))
        renderers.append(f"{row.metadata.renderer_pair[0]}:{row.metadata.renderer_pair[1]}")
        positions.append(row.metadata.target_position)
    ref_features, ref_mask = v13._embed_groups(qwen, references)
    cand_features, cand_mask = v13._embed_groups(qwen, candidates)
    return EncodedReplication(
        ref_features,
        ref_mask,
        cand_features,
        cand_mask,
        torch.tensor(targets, dtype=torch.long),
        tuple(types),
        tuple(renderers),
        tuple(positions),
    )


def _replication_report(
    frozen: StructureProtectedOMLCore,
    rows: EncodedReplication,
    *,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> dict[str, Any]:
    device = next(frozen.parameters()).device
    data = rows.to(device)
    with torch.no_grad():
        reference_codes, candidate_codes = frozen.encode_views(
            data.reference_features,
            data.reference_mask,
            data.candidate_features,
            data.candidate_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
    reference = F.normalize(reference_codes[:, 0], dim=-1)
    candidates = F.normalize(candidate_codes[:, :5], dim=-1)
    scores = torch.einsum("md,mcd->mc", reference, candidates)
    targets = data.target_indices
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    correct = scores.argmax(-1).eq(targets)

    def marginal(labels: Sequence[Any]) -> dict[str, Any]:
        report = {}
        for label in sorted(set(labels), key=str):
            indices = [i for i, value in enumerate(labels) if value == label]
            count = int(correct[indices].sum().item())
            report[str(label)] = {"correct": count, "count": len(indices), "accuracy": count / len(indices)}
        return report

    corruption = {}
    for name in CORRUPTIONS:
        negatives = torch.stack(
            [scores[i, data.candidate_types[i].index(name)] for i in range(len(data.candidate_types))]
        )
        margin = target_scores - negatives
        corruption[name] = {
            "wins": int(margin.gt(0).sum().item()),
            "count": len(margin),
            "win_rate": float(margin.gt(0).float().mean().item()),
            "mean_distance_margin": float(margin.mean().item()),
        }
    raw_codes = torch.cat((reference_codes[:, :1], candidate_codes[:, :5]), dim=1).detach().cpu()
    geometry = representation_metrics(raw_codes)
    geometry.update({"raw_code_sha256": _tensor_digest(raw_codes), "raw_code_shape": list(raw_codes.shape)})
    return {
        "correct": int(correct.sum().item()),
        "count": len(correct),
        "accuracy": float(correct.float().mean().item()),
        "renderer_pair_marginals": marginal(data.renderer_pairs),
        "target_position_marginals": marginal(data.target_positions),
        "corruptions": corruption,
        "paired_order_win_rate": corruption["reorder"]["win_rate"],
        "paired_order_mean_margin": corruption["reorder"]["mean_distance_margin"],
        "geometry": geometry,
        "scores_sha256": _tensor_digest(scores),
    }


def _structure_gate(metrics: Mapping[str, Any], frozen_exact: bool) -> bool:
    try:
        full = metrics["full"]
        direction = metrics["direction_removed"]
        semantics = metrics["step_semantics_removed"]
        geometry = full["geometry"]
        return bool(
            frozen_exact is True
            and full["count"] == STRUCTURAL_REPLICATION_ROWS
            and full["correct"] >= 200
            and all(row["accuracy"] >= 0.75 for row in full["renderer_pair_marginals"].values())
            and all(row["accuracy"] >= 0.75 for row in full["target_position_marginals"].values())
            and full["paired_order_win_rate"] >= 0.75
            and full["paired_order_mean_margin"] > 0.0
            and all(
                full["corruptions"][name]["win_rate"] >= 0.75
                and full["corruptions"][name]["mean_distance_margin"] >= 0.05
                for name in CORRUPTIONS
            )
            and geometry["effective_rank"] >= 8.0
            and geometry["all_finite"] is True
            and geometry["finite_nonzero_variance_dimensions"] == 32
            and geometry["mean_off_diagonal_cosine"] <= 0.95
            and geometry["fraction_distinct_pairs_above_0_999"] <= 0.10
            and full["paired_order_win_rate"] - direction["paired_order_win_rate"] >= 0.10
            and full["accuracy"] - semantics["accuracy"] >= 0.20
        )
    except (KeyError, TypeError, ValueError):
        return False


def _fit(
    core: StructureKeyedCreditMemoryCore,
    rows: Sequence[EncodedCreditMechanism],
    schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(rows) != TRAIN_MECHANISMS or tuple(map(tuple, schedule)) != _training_schedule():
        raise ValueError("V14 fit requires the exact frozen rows and schedule")
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
        eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY,
    )
    device = next(core.parameters()).device
    updates = []
    core.train()
    for update_index, indices in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        state = core.initial_state()
        losses = []
        predicted_steps = []
        for index in indices:
            row = rows[index].to(device)
            for event_index in range(EVENTS_PER_MECHANISM):
                predicted_steps.append(state.step)
                output = core.predict(
                    row.relation_features[event_index : event_index + 1],
                    row.temporal_features[event_index : event_index + 1],
                    row.base_logits[event_index : event_index + 1],
                    state=state,
                )
                outcome = row.outcomes[event_index : event_index + 1]
                losses.append(core.outcome_loss(output, outcome))
                state, _ = core.apply_feedback(
                    state, output, outcome,
                    evidence_refs=f"train:{index}:{event_index}", detach_state=False,
                )
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("V14 training loss is non-finite")
        loss.backward()
        gradients = [p.grad for p in core.parameters() if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all().item()) for g in gradients):
            raise RuntimeError("V14 training gradients are absent or non-finite")
        norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_CLIP)
        optimizer.step()
        updates.append({
            "update": update_index,
            "loss": float(loss.detach().item()),
            "gradient_norm": float(norm.detach().item()),
            "mechanism_indices": list(indices),
            "predict_before_feedback": predicted_steps == list(range(MECHANISMS_PER_STEP * EVENTS_PER_MECHANISM)),
        })
    return {
        "updates": updates,
        "completed_steps": len(updates),
        "exposure_counts": dict(sorted(Counter(i for row in schedule for i in row).items())),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
    }


def _balanced_accuracy(logits: Sequence[float], outcomes: Sequence[float]) -> float:
    positive = [float(logit) >= 0 for logit, y in zip(logits, outcomes, strict=True) if y > 0]
    negative = [float(logit) < 0 for logit, y in zip(logits, outcomes, strict=True) if y < 0]
    if not positive or not negative:
        raise ValueError("balanced accuracy requires both classes")
    return 0.5 * (sum(positive) / len(positive) + sum(negative) / len(negative))


def _state_metrics(state: StructureKeyedCreditState) -> dict[str, Any]:
    tensors = (state.keys, state.values, state.usage, state.acquisition)
    usage = state.usage.detach()
    return {
        "step": state.step,
        "finite": all(bool(torch.isfinite(value).all().item()) for value in tensors),
        "keys_max_abs": float(state.keys.detach().abs().max().item()),
        "values_max_abs": float(state.values.detach().abs().max().item()),
        "usage_min": float(usage.min().item()),
        "usage_max": float(usage.max().item()),
        "acquisition_min": float(state.acquisition.detach().min().item()),
        "acquisition_max": float(state.acquisition.detach().max().item()),
        "occupied_slots": int(usage.gt(0).sum().item()),
    }


def _run_stream(
    core: StructureKeyedCreditMemoryCore,
    rows: Sequence[EncodedCreditMechanism],
    *,
    feedback: str,
    reset_each_mechanism: bool = False,
    read_enabled: bool = True,
    initial_state: StructureKeyedCreditState | None = None,
    write_enabled: bool = True,
) -> dict[str, Any]:
    if feedback not in {"true", "shuffled"}:
        raise ValueError("unknown V14 feedback arm")
    device = next(core.parameters()).device
    state = core.initial_state() if initial_state is None else initial_state
    logits: list[float] = []
    labels: list[float] = []
    losses: list[float] = []
    families: list[str] = []
    transitions: list[str] = []
    events: list[StructureKeyedCreditEvent] = []
    read_maxima = []
    write_strengths = []
    allocations = []
    mechanism_slices = []
    states_after_mechanism: list[StructureKeyedCreditState] = []
    with torch.no_grad():
        for mechanism_index, cpu_row in enumerate(rows):
            if reset_each_mechanism:
                state = core.initial_state()
            row = cpu_row.to(device)
            start = len(logits)
            feedback_outcomes = row.outcomes
            if feedback == "shuffled":
                feedback_outcomes = row.outcomes[list(SHUFFLE_PERMUTATION)]
            for event_index in range(EVENTS_PER_MECHANISM):
                output = core.predict(
                    row.relation_features[event_index : event_index + 1],
                    row.temporal_features[event_index : event_index + 1],
                    row.base_logits[event_index : event_index + 1],
                    state=state, read_enabled=read_enabled,
                )
                target = row.outcomes[event_index : event_index + 1]
                logits.append(float(output.logits.item()))
                labels.append(float(target.item()))
                losses.append(float(core.outcome_loss(output, target).item()))
                families.append(row.generator_family)
                transitions.append(row.transition_group)
                read_maxima.append(float(output.read_weights.max().item()))
                write_strengths.append(float(output.write_strengths.item()))
                if write_enabled:
                    state, event = core.apply_feedback(
                        state, output, feedback_outcomes[event_index : event_index + 1],
                        evidence_refs=f"development:{mechanism_index}:{event_index}",
                    )
                    events.append(event)
                    allocations.append(event.allocation_index)
            mechanism_slices.append((start, len(logits)))
            states_after_mechanism.append(state.detached_clone())
    finite_scores = all(math.isfinite(value) for value in (*logits, *losses))
    return {
        "online_loss_auc": sum(losses) / len(losses),
        "balanced_accuracy": _balanced_accuracy(logits, labels),
        "logits": logits,
        "outcomes": labels,
        "losses": losses,
        "families": families,
        "transitions": transitions,
        "mechanism_slices": mechanism_slices,
        "events": tuple(events),
        "state": state,
        "states_after_mechanism": tuple(states_after_mechanism),
        "read_weight_max_mean": sum(read_maxima) / len(read_maxima),
        "read_weight_min": min(read_maxima),
        "read_weight_max": max(read_maxima),
        "write_strength_mean": sum(write_strengths) / len(write_strengths),
        "write_strength_min": min(write_strengths),
        "write_strength_max": max(write_strengths),
        "allocation_unique_slots": len(set(allocations)),
        "state_metrics": _state_metrics(state),
        "finite_scores": finite_scores,
    }


def _group_report(arm: Mapping[str, Any], key: str) -> dict[str, Any]:
    labels = arm[key]
    report = {}
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        logits = [arm["logits"][i] for i in indices]
        outcomes = [arm["outcomes"][i] for i in indices]
        report[label] = {
            "count": len(indices),
            "online_loss_auc": sum(arm["losses"][i] for i in indices) / len(indices),
            "balanced_accuracy": _balanced_accuracy(logits, outcomes),
        }
    return report


def _score_state(
    core: StructureKeyedCreditMemoryCore,
    rows: Sequence[EncodedCreditMechanism],
    state: StructureKeyedCreditState,
) -> dict[str, Any]:
    return _run_stream(
        core, rows, feedback="true", initial_state=state,
        write_enabled=False, read_enabled=True,
    )


def _key_shuffled_state(state: StructureKeyedCreditState) -> StructureKeyedCreditState:
    permutation = torch.arange(state.keys.shape[0] - 1, -1, -1, device=state.keys.device)
    return StructureKeyedCreditState(
        keys=state.keys[permutation],
        values=state.values,
        usage=state.usage,
        acquisition=state.acquisition,
        step=state.step,
    )


def _evaluate_credit(
    core: StructureKeyedCreditMemoryCore,
    rows: Sequence[EncodedCreditMechanism],
) -> dict[str, Any]:
    if len(rows) != DEVELOPMENT_MECHANISMS:
        raise ValueError("V14 evaluation requires exactly 48 mechanisms")
    core.eval()
    true = _run_stream(core, rows, feedback="true")
    shuffled = _run_stream(core, rows, feedback="shuffled")
    zero = _run_stream(core, rows, feedback="true", read_enabled=False, write_enabled=False)
    reset = _run_stream(core, rows, feedback="true", reset_each_mechanism=True)

    terminal_true = _score_state(core, rows, true["state"])
    matched_state = core.restore_state(core.capture_state(true["state"]))
    terminal_matched = _score_state(core, rows, matched_state)
    terminal_zero = _score_state(core, rows, core.zero_state_like(true["state"]))
    terminal_unrelated = _score_state(core, rows, shuffled["state"])
    terminal_key_shuffled = _score_state(core, rows, _key_shuffled_state(true["state"]))
    replay_state, replay_outputs = core.replay(true["events"])
    replay_logits = [float(output.logits.item()) for output in replay_outputs]
    replay_exact = bool(
        core.state_digest(replay_state) == core.state_digest(true["state"])
        and len(replay_logits) == len(true["logits"])
        and max(abs(a - b) for a, b in zip(replay_logits, true["logits"], strict=True)) <= 1.0e-6
    )

    family_true = _group_report(true, "families")
    family_zero = _group_report(zero, "families")
    transition_counts: dict[str, int] = defaultdict(int)
    improved_families = {}
    for family, full in family_true.items():
        baseline = family_zero[family]
        improved = (
            baseline["online_loss_auc"] - full["online_loss_auc"] >= 0.005
            or full["balanced_accuracy"] - baseline["balanced_accuracy"] >= 0.05
        )
        transition = next(row.transition_group for row in rows if row.generator_family == family)
        improved_families[family] = {"improved": improved, "transition": transition}
        transition_counts[transition] += int(improved)

    informative = 0
    changed = 0
    for start, end in true["mechanism_slices"]:
        left = true["logits"][start:end]
        right = shuffled["logits"][start:end]
        if max(left) - min(left) > 1.0e-8 or max(right) - min(right) > 1.0e-8:
            informative += 1
            changed += int(sorted(range(6), key=lambda i: left[i]) != sorted(range(6), key=lambda i: right[i]))

    # Freeze the first three sorted families as the early cohort.  Their
    # post-acquisition snapshot is replayed only through the last early event;
    # terminal scoring performs no write and therefore no replay of early data.
    early_names = tuple(sorted(family_true)[:3])
    last_early_mechanism = max(
        index for index, row in enumerate(rows) if row.generator_family in early_names
    )
    early_state = true["states_after_mechanism"][last_early_mechanism]
    early_rows = tuple(row for row in rows if row.generator_family in early_names)
    post_early = _score_state(core, early_rows, early_state)["balanced_accuracy"]
    after_later = _score_state(core, early_rows, true["state"])["balanced_accuracy"]

    arms = {}
    for name, value in (("true", true), ("shuffled", shuffled), ("zero_no_read", zero), ("per_mechanism_reset", reset)):
        arms[name] = {
            "online_loss_auc": value["online_loss_auc"],
            "balanced_accuracy": value["balanced_accuracy"],
            "early_stage_balanced_accuracy": _balanced_accuracy(value["logits"][:144], value["outcomes"][:144]),
            "late_stage_balanced_accuracy": _balanced_accuracy(value["logits"][144:], value["outcomes"][144:]),
            "by_family": _group_report(value, "families"),
            "by_transition": _group_report(value, "transitions"),
            "read_weight_max_mean": value["read_weight_max_mean"],
            "read_weight_min": value["read_weight_min"],
            "read_weight_max": value["read_weight_max"],
            "write_strength_mean": value["write_strength_mean"],
            "write_strength_min": value["write_strength_min"],
            "write_strength_max": value["write_strength_max"],
            "allocation_unique_slots": value["allocation_unique_slots"],
            "finite_scores": value["finite_scores"],
        }
    true_terminal_ba = terminal_true["balanced_accuracy"]
    zero_terminal_ba = terminal_zero["balanced_accuracy"]
    gain = true_terminal_ba - zero_terminal_ba
    return {
        "mechanism_count": len(rows),
        "event_count": len(true["logits"]),
        "arms": arms,
        "family_improvements": improved_families,
        "improved_family_count": sum(int(row["improved"]) for row in improved_families.values()),
        "transition_improved_family_counts": dict(sorted(transition_counts.items())),
        "candidate_order": {
            "informative_rows": informative,
            "changed_rows": changed,
            "changed_fraction": changed / informative if informative else 0.0,
        },
        "terminal_removal_and_swaps": {
            "true_balanced_accuracy": true_terminal_ba,
            "zero_balanced_accuracy": zero_terminal_ba,
            "gain": gain,
            "matched_balanced_accuracy": terminal_matched["balanced_accuracy"],
            "matched_state_digest_exact": core.state_digest(matched_state) == core.state_digest(true["state"]),
            "matched_gain_retention": (
                (terminal_matched["balanced_accuracy"] - zero_terminal_ba) / gain
                if gain > 0 else 0.0
            ),
            "unrelated_balanced_accuracy": terminal_unrelated["balanced_accuracy"],
            "key_shuffled_balanced_accuracy": terminal_key_shuffled["balanced_accuracy"],
        },
        "replay": {
            "exact": replay_exact,
            "original_digest": core.state_digest(true["state"]),
            "replay_digest": core.state_digest(replay_state),
            "maximum_logit_error": max(abs(a - b) for a, b in zip(replay_logits, true["logits"], strict=True)),
        },
        "retention": {
            "early_families": list(early_names),
            "post_acquisition_balanced_accuracy": post_early,
            "after_later_writes_balanced_accuracy": after_later,
            "drop": post_early - after_later,
        },
        "state_metrics": true["state_metrics"],
        "feedback_integrity": {
            "permutation": list(SHUFFLE_PERMUTATION),
            "same_visible_inputs": True,
            "same_true_evaluation_labels": true["outcomes"] == shuffled["outcomes"],
            "balanced_shuffled_feedback": all(
                Counter(row.outcomes.tolist()) == Counter(row.outcomes[list(SHUFFLE_PERMUTATION)].tolist())
                for row in rows
            ),
        },
    }


def _persistent_gate(metrics: Mapping[str, Any], identity_exact: bool) -> bool:
    try:
        arms = metrics["arms"]
        true = arms["true"]
        shuffled = arms["shuffled"]
        zero = arms["zero_no_read"]
        reset = arms["per_mechanism_reset"]
        swaps = metrics["terminal_removal_and_swaps"]
        state = metrics["state_metrics"]
        causal_shuffled = (
            shuffled["online_loss_auc"] - true["online_loss_auc"] >= 0.005
            or true["balanced_accuracy"] - shuffled["balanced_accuracy"] >= 0.05
        )
        causal_zero = (
            zero["online_loss_auc"] - true["online_loss_auc"] >= 0.005
            or true["balanced_accuracy"] - zero["balanced_accuracy"] >= 0.05
        )
        unrelated_loss = max(
            swaps["true_balanced_accuracy"] - swaps["unrelated_balanced_accuracy"],
            swaps["true_balanced_accuracy"] - swaps["key_shuffled_balanced_accuracy"],
        )
        return bool(
            identity_exact is True
            and metrics["mechanism_count"] == 48 and metrics["event_count"] == 288
            and true["balanced_accuracy"] >= 0.70
            and causal_shuffled and causal_zero
            and metrics["improved_family_count"] >= 9
            and all(value >= 2 for value in metrics["transition_improved_family_counts"].values())
            and true["balanced_accuracy"] - reset["balanced_accuracy"] >= 0.05
            and swaps["gain"] >= 0.05
            and swaps["matched_state_digest_exact"] is True
            and metrics["replay"]["exact"] is True
            and metrics["replay"]["maximum_logit_error"] <= 1.0e-6
            and swaps["matched_gain_retention"] >= 0.90
            and unrelated_loss >= 0.05
            and metrics["retention"]["drop"] <= 0.05
            and metrics["candidate_order"]["changed_fraction"] >= 0.10
            and all(
                arm["finite_scores"] is True
                and 0.0 <= arm["read_weight_min"] <= arm["read_weight_max"] <= 1.0
                and 0.0 <= arm["write_strength_min"] <= arm["write_strength_max"] <= 1.0
                for arm in arms.values()
            )
            and state["finite"] is True and state["step"] == 288
            and state["keys_max_abs"] <= 1.0 and state["values_max_abs"] <= 1.0
            and 0.0 <= state["usage_min"] <= state["usage_max"] <= 1.0
            and 0.0 <= state["acquisition_min"] <= state["acquisition_max"] <= 1.0
            and metrics["feedback_integrity"]["same_visible_inputs"] is True
            and metrics["feedback_integrity"]["same_true_evaluation_labels"] is True
            and metrics["feedback_integrity"]["balanced_shuffled_feedback"] is True
            and metrics["protocol_invariants"]["exact_schedule"] is True
            and metrics["protocol_invariants"]["all_training_gradients_finite"] is True
            and metrics["protocol_invariants"]["train_dev_refs_disjoint"] is True
            and metrics["protocol_invariants"]["final_sealed"] is True
            and metrics["protocol_invariants"]["only_v14_core_trainable"] is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _classification(structure: bool, persistent: bool) -> str:
    if structure and persistent:
        return "DEVELOPMENT_GATE_PASSED"
    if structure:
        return "STRUCTURE_REPLICATED_NOT_ADAPTIVE"
    if persistent:
        return "PERSISTENT_CREDIT_WITHOUT_STRUCTURE"
    return "DEVELOPMENT_NOT_SUPPORTED"


def _validate_final_authorization(
    development: Mapping[str, Any], checkpoint_path: Path,
) -> None:
    structure = development.get("structure_replication")
    credit = development.get("development_metrics")
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "DEVELOPMENT_GATE_PASSED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and development.get("identity_checks", {}).get("exact") is True
        and isinstance(structure, Mapping) and _structure_gate(structure, True)
        and isinstance(credit, Mapping) and _persistent_gate(credit, True)
    )
    v5._validate_final_authorization(valid)


def _load_frozen_v13(
    checkpoint_path: Path,
    result_path: Path,
    device: torch.device,
) -> tuple[StructureProtectedOMLCore, Mapping[str, Any], Mapping[str, Any]]:
    if _sha256(checkpoint_path) != V13_CHECKPOINT_SHA256 or _sha256(result_path) != V13_RESULT_SHA256:
        raise RuntimeError("frozen V13 result/checkpoint identity changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if result.get("checkpoint_sha256", "").upper() != V13_CHECKPOINT_SHA256:
        raise RuntimeError("V13 result does not bind the consumed checkpoint")
    frozen = StructureProtectedOMLCore(step_width=int(checkpoint["step_width"])).to(device)
    frozen.load_state_dict(checkpoint["protected_second_state"], strict=True)
    if _model_digest(frozen) != str(checkpoint["model_digests"]["protected_second"]).upper():
        raise RuntimeError("V13 protected-second state digest mismatch")
    frozen.requires_grad_(False).eval()
    return frozen, checkpoint, result


def _identity_report(
    *, qwen_before: str, qwen_after: str, frozen_before: str, frozen_after: str,
    sources_before: Mapping[str, str], sources_after: Mapping[str, str],
) -> dict[str, Any]:
    exact = bool(
        qwen_before == qwen_after and frozen_before == frozen_after
        and dict(sources_before) == dict(sources_after)
    )
    return {
        "exact": exact,
        "qwen_before": qwen_before,
        "qwen_after": qwen_after,
        "v13_protected_second_before": frozen_before,
        "v13_protected_second_after": frozen_after,
        "sources_before": dict(sources_before),
        "sources_after": dict(sources_after),
    }


def _run_train_development(args: argparse.Namespace) -> None:
    output_checkpoint = Path(args.checkpoint)
    output_result = Path(args.development_result)
    if output_checkpoint.exists() or output_result.exists():
        raise FileExistsError("V14 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device != torch.device("cuda:1"):
        raise RuntimeError("V14 frozen identity requires V13/V14 on cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    frozen, v13_checkpoint, _ = _load_frozen_v13(
        Path(args.v13_checkpoint), Path(args.v13_result), device,
    )
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    corpus = build_structure_keyed_persistent_credit_v14()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V14 final partition opened before development")

    train_v13 = v13._encode_mechanisms(qwen, corpus.train)
    development_v13 = v13._encode_mechanisms(qwen, corpus.development)
    train = _encode_credit_mechanisms(frozen, train_v13, corpus.train)
    development = _encode_credit_mechanisms(frozen, development_v13, corpus.development)
    replication_rows = _encode_replication(qwen, corpus.structural_replication)
    _enforce_resource_ceiling(started, device)
    structure = {
        "full": _replication_report(frozen, replication_rows),
        "direction_removed": _replication_report(frozen, replication_rows, include_direction=False),
        "step_semantics_removed": _replication_report(frozen, replication_rows, include_step_semantics=False),
    }
    _enforce_resource_ceiling(started, device)
    core = StructureKeyedCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(device=device, dtype=torch.float32)
    initial_core_digest = _model_digest(core)
    training = _fit(core, train, _training_schedule())
    _enforce_resource_ceiling(started, device)
    trained_core_digest = _model_digest(core)
    credit = _evaluate_credit(core, development)
    _enforce_resource_ceiling(started, device)
    train_refs = {row.mechanism_ref for row in train}
    development_refs = {row.mechanism_ref for row in development}
    credit["protocol_invariants"] = {
        "exact_schedule": training["completed_steps"] == TRAIN_STEPS
        and set(training["exposure_counts"].values()) == {2},
        "all_training_gradients_finite": all(
            math.isfinite(float(row["gradient_norm"]))
            and row["predict_before_feedback"] is True
            for row in training["updates"]
        ),
        "train_dev_refs_disjoint": train_refs.isdisjoint(development_refs),
        "final_sealed": True,
        "only_v14_core_trainable": all(not parameter.requires_grad for parameter in frozen.parameters())
        and all(parameter.requires_grad for parameter in core.parameters()),
    }

    qwen_after = foundation_tensor_digest(qwen.model)
    frozen_after = _model_digest(frozen)
    sources_after = _source_hashes()
    identity = _identity_report(
        qwen_before=qwen_before, qwen_after=qwen_after,
        frozen_before=frozen_before, frozen_after=frozen_after,
        sources_before=sources_before, sources_after=sources_after,
    )
    structure_supported = _structure_gate(structure, bool(identity["exact"]))
    credit_supported = _persistent_gate(credit, bool(identity["exact"]))
    classification = _classification(structure_supported, credit_supported)
    checkpoint = {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "completed_steps": TRAIN_STEPS,
        "temporal_width": TEMPORAL_WIDTH,
        "core_state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "core_digest": trained_core_digest,
        "initial_core_digest": initial_core_digest,
        "v13_checkpoint_sha256": V13_CHECKPOINT_SHA256,
        "v13_result_sha256": V13_RESULT_SHA256,
        "v13_protected_second_digest": frozen_before,
        "qwen_digest": qwen_before,
        "source_hashes": sources_before,
        "structure_replication": structure,
        "development_metrics": credit,
    }
    result: dict[str, Any] = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": classification,
        "development_authorized": structure_supported and credit_supported,
        "structure_replicated": structure_supported,
        "persistent_procedural_credit_supported": credit_supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "steps": TRAIN_STEPS,
            "mechanisms_per_step": MECHANISMS_PER_STEP,
            "events_per_step": MECHANISMS_PER_STEP * EVENTS_PER_MECHANISM,
            "train_mechanisms": TRAIN_MECHANISMS,
            "exposures_per_mechanism": 2,
            "development_mechanisms": DEVELOPMENT_MECHANISMS,
            "structural_replication_rows": STRUCTURAL_REPLICATION_ROWS,
            "dtype": "float32",
            "autocast": False,
            "tf32": False,
            "qwen_device": "cuda:0",
            "v13_v14_device": "cuda:1",
        },
        "training": training,
        "structure_replication": structure,
        "development_metrics": credit,
        "identity_checks": identity,
        "source_hashes": sources_before,
        "v13_evidence": {
            "checkpoint_sha256": V13_CHECKPOINT_SHA256,
            "result_sha256": V13_RESULT_SHA256,
            "protected_second_digest": frozen_before,
            "checkpoint_qwen_digest": v13_checkpoint["qwen_digest"],
        },
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Synthetic software-procedure replication and persistent-credit evidence only; "
            "not arbitrary cross-domain reasoning, deployment, AGI, or human authority."
        ),
    }
    if not identity["exact"]:
        raise RuntimeError("V14 frozen source/foundation/representation changed during execution")
    _atomic_torch(output_checkpoint, checkpoint)
    result["checkpoint_sha256"] = _sha256(output_checkpoint)
    _atomic_json(output_result, result)
    print(json.dumps({"classification": classification, "result": str(output_result)}, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    output = Path(args.final_result)
    if output.exists():
        raise FileExistsError("V14 final identity is already consumed")
    development_path = Path(args.development_result)
    checkpoint_path = Path(args.checkpoint)
    development_record = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development_record, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sources_before = _source_hashes()
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != sources_before
        or checkpoint.get("completed_steps") != TRAIN_STEPS
        or checkpoint.get("core_digest", "").upper() != _model_digest_from_state(checkpoint["core_state"])
    ):
        raise RuntimeError("V14 checkpoint authorization binding failed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device != torch.device("cuda:1"):
        raise RuntimeError("V14 final requires cuda:1")
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    frozen, _, _ = _load_frozen_v13(Path(args.v13_checkpoint), Path(args.v13_result), device)
    if qwen_before != checkpoint["qwen_digest"] or _model_digest(frozen) != checkpoint["v13_protected_second_digest"]:
        raise RuntimeError("V14 final frozen dependencies changed")
    corpus = build_structure_keyed_persistent_credit_v14(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_MECHANISMS:
        raise RuntimeError("V14 authorized final is incomplete")
    rows = _encode_credit_mechanisms(frozen, v13._encode_mechanisms(qwen, corpus.final), corpus.final)
    core = StructureKeyedCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(device=device, dtype=torch.float32)
    core.load_state_dict(checkpoint["core_state"], strict=True)
    metrics = _evaluate_credit(core, rows)
    identity_exact = bool(
        foundation_tensor_digest(qwen.model) == qwen_before
        and _model_digest(frozen) == checkpoint["v13_protected_second_digest"]
        and _source_hashes() == sources_before
    )
    supported = _persistent_gate(metrics, identity_exact)
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": metrics,
        "source_hashes": sources_before,
    }
    if not identity_exact:
        raise RuntimeError("V14 source/foundation drift during final")
    _atomic_json(output, result)


def _model_digest_from_state(state: Mapping[str, torch.Tensor]) -> str:
    # Constructing a module is the canonical digest path and avoids depending
    # on checkpoint serialization order.
    core = StructureKeyedCreditMemoryCore(temporal_width=TEMPORAL_WIDTH)
    core.load_state_dict(state, strict=True)
    return _model_digest(core)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v13-checkpoint", default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt")
    parser.add_argument("--v13-result", default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json")
    parser.add_argument("--checkpoint", default="/opt/angler/results/structure-keyed-persistent-credit-v14.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
