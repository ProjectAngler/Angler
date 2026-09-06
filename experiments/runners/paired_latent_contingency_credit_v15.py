"""Frozen V15 paired latent-contingency persistent-credit experiment.

V15 leaves Qwen and V13 immutable and instantiates the unchanged V14 neural
memory architecture from a fresh seed.  Counterfactual twins have identical
current inputs and opposite probe targets; deterministic code only preserves
that causal experiment and never decodes a regime or procedure rule.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from angler.reasoning.natural_trace_graph_causal_memory import parse_step_trace
from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditEvent,
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditState,
)
from angler.reasoning.structure_protected_oml import StructureProtectedOMLCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    ACQUISITION_INDICES,
    CORPUS_ID,
    DEVELOPMENT_PAIRS,
    EVENTS_PER_EPISODE,
    FINAL_PAIRS,
    PAIRS_PER_UPDATE,
    PROBE_INDICES,
    TRAIN_PAIRS,
    TRAIN_UPDATES,
    build_paired_latent_contingency_credit_v15,
)
from experiments.corpora.structure_protected_oml_natural_trace_v13 import FinalPartitionSealedError
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners import structure_keyed_persistent_credit_v14 as v14
from experiments.runners import structure_protected_oml_natural_trace_v13 as v13


IDENTITY = "angler.paired-latent-contingency-credit.v15-first-result"
SEED = 2026083115
TEMPORAL_WIDTH = 4
LEARNING_RATE = 3.0e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 2.0
SEPARATION_MARGIN = 0.05
TWIN_MARGIN = 0.20
LOSS_WEIGHTS = (1.0, 1.0, 1.0)
SCHEDULE_MULTIPLIER = 49
WALL_TIME_CEILING_SECONDS = 60 * 60
GPU_MEMORY_CEILING_BYTES = 12 * 1024**3

V13_RESULT_SHA256 = v14.V13_RESULT_SHA256
V13_CHECKPOINT_SHA256 = v14.V13_CHECKPOINT_SHA256
V14_RESULT_SHA256 = "768DDEB46599728CE3BC917E8C3BA6F67BEB85376C092F1CECC39F2F781DB2C6"


@dataclass(frozen=True, slots=True)
class EncodedContingencyEpisode:
    episode_ref: str
    pair_ref: str
    twin_ref: str
    generator_family: str
    transition_group: str
    relation_features: torch.Tensor
    base_logits: torch.Tensor
    temporal_features: torch.Tensor
    outcomes: torch.Tensor
    relation_classes: tuple[str, ...]
    public_payload_sha256: str
    input_tensor_sha256: str

    def to(self, device: torch.device) -> "EncodedContingencyEpisode":
        return EncodedContingencyEpisode(
            self.episode_ref, self.pair_ref, self.twin_ref,
            self.generator_family, self.transition_group,
            self.relation_features.to(device=device, dtype=torch.float32),
            self.base_logits.to(device=device, dtype=torch.float32),
            self.temporal_features.to(device=device, dtype=torch.float32),
            self.outcomes.to(device=device, dtype=torch.float32),
            self.relation_classes,
            self.public_payload_sha256, self.input_tensor_sha256,
        )


@dataclass(frozen=True, slots=True)
class EncodedTwinPair:
    pair_ref: str
    first: EncodedContingencyEpisode
    second: EncodedContingencyEpisode


def _sha256(path: str | Path) -> str:
    return v14._sha256(path).upper()


def _tensor_digest(value: torch.Tensor) -> str:
    return v14._tensor_digest(value).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v14._model_digest(model).upper()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    own = {
        "runner": _sha256(Path(__file__).resolve()),
        "corpus": _sha256(root / "experiments/corpora/paired_latent_contingency_credit_v15.py"),
    }
    inherited = {f"v14_{key}": value.upper() for key, value in v14._source_hashes().items()}
    return {**own, **inherited}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V15 output already exists: {path}")
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
        raise FileExistsError(f"V15 output already exists: {path}")
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


def _enforce_resources(started: float, device: torch.device) -> None:
    if time.perf_counter() - started > WALL_TIME_CEILING_SECONDS:
        raise RuntimeError("V15 exceeded its 60-minute wall-time ceiling")
    if torch.cuda.max_memory_allocated(device) > GPU_MEMORY_CEILING_BYTES:
        raise RuntimeError("V15 exceeded its 12-GiB Angler-device memory ceiling")


def _pair_ref(pair: Any) -> str:
    first = str(pair.first.metadata.pair_ref)
    second = str(pair.second.metadata.pair_ref)
    if first != second:
        raise RuntimeError("V15 twin pair references disagree")
    return first


def _training_schedule(corpus: Any) -> tuple[tuple[int, ...], ...]:
    index = {_pair_ref(pair): position for position, pair in enumerate(corpus.train)}
    declared = corpus.training_schedule()
    schedule = []
    for update, row in enumerate(declared):
        if len(row) != PAIRS_PER_UPDATE:
            raise RuntimeError("V15 declared schedule width changed")
        indices = tuple(
            int(item) if type(item) is int else index[_pair_ref(item)]
            for item in row
        )
        if len(set(indices)) != PAIRS_PER_UPDATE:
            raise RuntimeError("V15 schedule repeats a pair inside one update")
        expected = tuple(index[_pair_ref(item)] for item in corpus.train_pairs_for_update(update))
        if indices != expected:
            raise RuntimeError("V15 schedule helpers disagree")
        schedule.append(indices)
    result = tuple(schedule)
    if len(result) != TRAIN_UPDATES or Counter(i for row in result for i in row) != Counter(
        {i: 2 for i in range(TRAIN_PAIRS)}
    ):
        raise RuntimeError("V15 schedule must expose every pair exactly twice")
    return result


def _recursive_keys(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.append(str(key))
            result.extend(_recursive_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_recursive_keys(child))
    return tuple(result)


def _public_episode_payload(episode: Any) -> Mapping[str, Any]:
    payload = episode.to_learner_payload()
    if set(payload) != {"events"} or len(payload["events"]) != EVENTS_PER_EPISODE:
        raise RuntimeError("V15 learner payload shape changed")
    forbidden = ("regime", "class", "outcome", "answer", "target", "family", "transition")
    keys = tuple(key.lower() for key in _recursive_keys(payload))
    if any(any(token in key for token in forbidden) or key in {"id", "ref"} or key.endswith("_id") for key in keys):
        raise RuntimeError("V15 public learner payload exposes evaluator identity/outcome")
    return payload


def _temporal(event: Any, index: int) -> tuple[float, ...]:
    coordinate = event.temporal
    ordinal = float(coordinate.acquired_ordinal)
    age = float(coordinate.age)
    return (
        ordinal / max(1.0, ordinal + age),
        age / max(1.0, ordinal + age),
        math.log1p(age) / math.log(7.0),
        index / float(EVENTS_PER_EPISODE - 1),
    )


def _encode_pairs(
    qwen: Any,
    frozen: StructureProtectedOMLCore,
    pairs: Sequence[Any],
) -> tuple[EncodedTwinPair, ...]:
    for pair in pairs:
        if (
            tuple(pair.deranged_acquisition_outcomes(pair.first))
            != tuple(pair.second.supervision.acquisition_outcomes)
            or tuple(pair.deranged_acquisition_outcomes(pair.second))
            != tuple(pair.first.supervision.acquisition_outcomes)
            or any(
                left == right
                for left, right in zip(
                    pair.first.supervision.acquisition_outcomes,
                    pair.deranged_acquisition_outcomes(pair.first),
                    strict=True,
                )
            )
        ):
            raise RuntimeError("V15 derangement is not fixed-point-free across complete histories")
    # Counterfactual twins have byte-identical public observations.  Encode the
    # public episode once per pair so device-level duplicate-kernel drift can
    # never masquerade as a causal input difference.
    episodes = tuple(pair.first for pair in pairs)
    references = []
    attempts = []
    payload_hashes = []
    for pair, episode in zip(pairs, episodes, strict=True):
        payload = _public_episode_payload(episode)
        twin_payload = _public_episode_payload(pair.second)
        encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        encoded_twin_payload = json.dumps(
            twin_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        if encoded_payload != encoded_twin_payload:
            raise RuntimeError("V15 twin public payloads differ before shared encoding")
        payload_hashes.append(hashlib.sha256(encoded_payload).hexdigest().upper())
        references.append(tuple(tuple(parse_step_trace(event.reference_trace_text)) for event in episode.public.events))
        attempts.append(tuple(tuple(parse_step_trace(event.attempt_trace_text)) for event in episode.public.events))
    reference_features, reference_mask = v13._embed_groups(qwen, references)
    attempt_features, attempt_mask = v13._embed_groups(qwen, attempts)
    device = next(frozen.parameters()).device
    relation_rows = []
    base_rows = []
    frozen.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), 32):
            stop = min(start + 32, len(episodes))
            ref = reference_features[start:stop].to(device=device, dtype=torch.float32)
            ref_mask = reference_mask[start:stop].to(device)
            att = attempt_features[start:stop].to(device=device, dtype=torch.float32)
            att_mask = attempt_mask[start:stop].to(device)
            relation_rows.extend(frozen.relation_features(ref, ref_mask, att, att_mask).detach().cpu())
            base_rows.extend(frozen.functional_logits(ref, ref_mask, att, att_mask).detach().cpu())
    result = []
    for position, pair in enumerate(pairs):
        representative = pair.first
        temporal = torch.tensor(
            tuple(
                _temporal(event, index)
                for index, event in enumerate(representative.public.events)
            ),
            dtype=torch.float32,
        )
        relation = relation_rows[position]
        base = base_rows[position]
        input_hash = hashlib.sha256(
            (_tensor_digest(relation) + _tensor_digest(base) + _tensor_digest(temporal)).encode()
        ).hexdigest().upper()
        def encoded_episode(episode: Any) -> EncodedContingencyEpisode:
            metadata = episode.metadata
            return EncodedContingencyEpisode(
                str(metadata.episode_ref), str(metadata.pair_ref), str(metadata.twin_ref),
                str(metadata.generator_family), str(metadata.transition_group),
                relation.clone(), base.clone(), temporal.clone(),
                torch.tensor(episode.supervision.outcome_values, dtype=torch.float32),
                tuple(str(value) for value in episode.supervision.relation_classes),
                payload_hashes[position], input_hash,
            )
        result.append(
            EncodedTwinPair(
                _pair_ref(pair), encoded_episode(pair.first), encoded_episode(pair.second)
            )
        )
    return tuple(result)


def _identifiability_preflight(
    core: StructureKeyedCreditMemoryCore,
    train: Sequence[EncodedTwinPair],
    development: Sequence[EncodedTwinPair],
    schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    all_pairs = (*train, *development)
    tensor_exact = all(pair.first.input_tensor_sha256 == pair.second.input_tensor_sha256 for pair in all_pairs)
    payload_exact = all(pair.first.public_payload_sha256 == pair.second.public_payload_sha256 for pair in all_pairs)
    opposite_probes = all(
        all(float(pair.first.outcomes[i]) == -float(pair.second.outcomes[i]) for i in PROBE_INDICES)
        for pair in all_pairs
    )
    opposite_acquisition = all(
        all(float(pair.first.outcomes[i]) == -float(pair.second.outcomes[i]) for i in ACQUISITION_INDICES)
        for pair in all_pairs
    )
    position_balance = {}
    for name, rows in (("train", train), ("development", development)):
        for index in range(EVENTS_PER_EPISODE):
            labels = [int(value.outcomes[index].item()) for pair in rows for value in (pair.first, pair.second)]
            position_balance[f"{name}:{index}"] = Counter(labels) == Counter({-1: len(rows), 1: len(rows)})
    grouped_balance = {}
    for attribute in ("generator_family", "transition_group"):
        groups: dict[str, list[int]] = defaultdict(list)
        for pair in development:
            for episode in (pair.first, pair.second):
                groups[str(getattr(episode, attribute))].extend(int(episode.outcomes[i].item()) for i in PROBE_INDICES)
        grouped_balance[attribute] = all(values.count(-1) == values.count(1) for values in groups.values())
    relation_position_balance = {}
    relation_polarity_balance = {}
    for name, rows in (("train", train), ("development", development)):
        for index in range(EVENTS_PER_EPISODE):
            values = [episode.relation_classes[index] for pair in rows for episode in (pair.first, pair.second)]
            relation_position_balance[f"{name}:{index}"] = Counter(values) == Counter(
                {"alpha": len(rows), "beta": len(rows)}
            )
            joint = Counter(
                (episode.relation_classes[index], int(episode.outcomes[index].item()))
                for pair in rows for episode in (pair.first, pair.second)
            )
            relation_polarity_balance[f"{name}:{index}"] = joint == Counter(
                {(relation, outcome): len(rows) // 2 for relation in ("alpha", "beta") for outcome in (-1, 1)}
            )
    grouped_relation_polarity = {}
    for attribute in ("generator_family", "transition_group"):
        groups: dict[str, list[EncodedContingencyEpisode]] = defaultdict(list)
        for pair in development:
            groups[str(getattr(pair.first, attribute))].extend((pair.first, pair.second))
        grouped_relation_polarity[attribute] = all(
            all(
                len(set(Counter(
                    (episode.relation_classes[index], int(episode.outcomes[index].item()))
                    for episode in episodes
                ).values())) == 1
                and len(Counter(
                    (episode.relation_classes[index], int(episode.outcomes[index].item()))
                    for episode in episodes
                )) == 4
                for index in range(EVENTS_PER_EPISODE)
            )
            for episodes in groups.values()
        )
    train_families = {pair.first.generator_family for pair in train}
    dev_families = {pair.first.generator_family for pair in development}
    train_inputs = {pair.first.public_payload_sha256 for pair in train}
    dev_inputs = {pair.first.public_payload_sha256 for pair in development}
    device = next(core.parameters()).device
    prefeedback_equal = True
    with torch.no_grad():
        initial = core.initial_state()
        for pair in all_pairs:
            first = pair.first.to(device)
            second = pair.second.to(device)
            for index in PROBE_INDICES:
                left = core.predict(first.relation_features[index:index+1], first.temporal_features[index:index+1], first.base_logits[index:index+1], state=initial)
                right = core.predict(second.relation_features[index:index+1], second.temporal_features[index:index+1], second.base_logits[index:index+1], state=initial)
                if not torch.equal(left.logits, right.logits):
                    prefeedback_equal = False
                    break
    report = {
        "train_pair_count": len(train),
        "development_pair_count": len(development),
        "twin_public_payload_hashes_exact": payload_exact,
        "twin_input_tensor_hashes_exact": tensor_exact,
        "prefeedback_logits_exact": prefeedback_equal,
        "matched_probe_label_entropy_bits": 1.0 if opposite_probes else 0.0,
        "twin_acquisition_histories_fixed_point_free": opposite_acquisition,
        "position_outcomes_balanced": all(position_balance.values()),
        "relation_classes_balanced_at_every_position": all(relation_position_balance.values()),
        "relation_class_by_polarity_balanced_at_every_position": all(relation_polarity_balance.values()),
        "position_balance_rows": position_balance,
        "relation_position_balance_rows": relation_position_balance,
        "development_family_outcomes_balanced": grouped_balance["generator_family"],
        "development_transition_outcomes_balanced": grouped_balance["transition_group"],
        "development_family_relation_polarity_balanced": grouped_relation_polarity["generator_family"],
        "development_transition_relation_polarity_balanced": grouped_relation_polarity["transition_group"],
        "train_development_families_disjoint": train_families.isdisjoint(dev_families),
        "train_development_surfaces_disjoint": train_inputs.isdisjoint(dev_inputs),
        "schedule_exact": len(schedule) == TRAIN_UPDATES
        and Counter(i for row in schedule for i in row) == Counter({i: 2 for i in range(TRAIN_PAIRS)}),
        "public_regime_or_label_keys_absent": True,
        "forbidden_key_guard": "_public_episode_payload raises before tensorization",
        "unique_learner_payload_hashes": len({pair.first.public_payload_sha256 for pair in all_pairs}),
    }
    report["exact"] = all(
        value is True for key, value in report.items()
        if key not in {"train_pair_count", "development_pair_count", "matched_probe_label_entropy_bits", "position_balance_rows", "relation_position_balance_rows", "forbidden_key_guard", "unique_learner_payload_hashes"}
    ) and report["matched_probe_label_entropy_bits"] == 1.0
    return report


def _feedback_value(episode: EncodedContingencyEpisode, twin: EncodedContingencyEpisode, index: int, arm: str) -> torch.Tensor:
    if arm == "true":
        return episode.outcomes[index:index+1]
    if arm == "deranged":
        return twin.outcomes[index:index+1]
    if arm == "blind":
        return episode.outcomes[index:index+1]
    raise ValueError("unknown V15 feedback arm")


def _probe_losses(core: StructureKeyedCreditMemoryCore, episode: EncodedContingencyEpisode, state: StructureKeyedCreditState, *, read_enabled: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logits, losses = [], []
    for index in PROBE_INDICES:
        output = core.predict(
            episode.relation_features[index:index+1], episode.temporal_features[index:index+1],
            episode.base_logits[index:index+1], state=state, read_enabled=read_enabled,
        )
        logits.append(output.logits)
        losses.append(core.outcome_loss(output, episode.outcomes[index:index+1]))
    return logits, losses


def _write_anchors(
    core: StructureKeyedCreditMemoryCore,
    episode: EncodedContingencyEpisode,
    twin: EncodedContingencyEpisode,
    state: StructureKeyedCreditState,
    *, arm: str, detach_state: bool,
) -> tuple[StructureKeyedCreditState, list[StructureKeyedCreditEvent], list[int], list[float]]:
    events, predicted_steps, logits = [], [], []
    for index in ACQUISITION_INDICES:
        predicted_steps.append(state.step)
        output = core.predict(
            episode.relation_features[index:index+1], episode.temporal_features[index:index+1],
            episode.base_logits[index:index+1], state=state,
        )
        logits.append(float(output.logits.detach().item()))
        feedback = _feedback_value(episode, twin, index, arm)
        evidence_ref = f"{episode.pair_ref}:{episode.twin_ref}:anchor:{index}"
        if arm == "blind":
            raise ValueError("use _write_blind_anchors with an explicit routing state")
        state, event = core.apply_feedback(
            state, output, feedback,
            evidence_refs=evidence_ref,
            detach_state=detach_state,
        )
        events.append(event)
    return state, events, predicted_steps, logits


def _write_blind_anchors(
    core: StructureKeyedCreditMemoryCore,
    episode: EncodedContingencyEpisode,
    state: StructureKeyedCreditState,
    routing_state: StructureKeyedCreditState,
) -> tuple[
    StructureKeyedCreditState,
    StructureKeyedCreditState,
    list[StructureKeyedCreditEvent],
    list[int],
    list[float],
]:
    """Write shared values while a normal shadow preserves routing exactly."""
    events, predicted_steps, logits = [], [], []
    for index in ACQUISITION_INDICES:
        if state.step != routing_state.step:
            raise RuntimeError("V15 blind/routing state chronology diverged")
        predicted_steps.append(state.step)
        output = core.predict(
            episode.relation_features[index:index+1], episode.temporal_features[index:index+1],
            episode.base_logits[index:index+1], state=state,
        )
        routing_output = core.predict(
            episode.relation_features[index:index+1], episode.temporal_features[index:index+1],
            episode.base_logits[index:index+1], state=routing_state,
        )
        logits.append(float(output.logits.detach().item()))
        feedback = episode.outcomes[index:index+1]
        evidence_ref = f"{episode.pair_ref}:{episode.twin_ref}:anchor:{index}"
        state, event = _apply_outcome_blind_feedback(
            core, state, output, feedback, evidence_ref=evidence_ref,
            routing_output=routing_output,
        )
        routing_state, _ = core.apply_feedback(
            routing_state, routing_output, feedback,
            evidence_refs=f"{evidence_ref}:routing-reference", detach_state=True,
        )
        events.append(event)
    return state, routing_state, events, predicted_steps, logits


def _apply_outcome_blind_feedback(
    core: StructureKeyedCreditMemoryCore,
    state: StructureKeyedCreditState,
    output: Any,
    outcomes: torch.Tensor,
    *,
    evidence_ref: str,
    routing_output: Any | None = None,
) -> tuple[StructureKeyedCreditState, StructureKeyedCreditEvent]:
    """Write one shared value token while preserving all normal routing/gates."""
    core.validate_state(state)
    core._validate_output(output)
    core._validate_outcomes(outcomes, output.logits)
    if output.logits.shape != (1,) or output.state_step != state.step:
        raise ValueError("V15 blind output/state chronology is invalid")
    if state.step >= core.memory_slots:
        raise OverflowError("persistent-credit memory capacity is exhausted")
    route = output if routing_output is None else routing_output
    core._validate_output(route)
    if route.state_step != state.step:
        raise ValueError("V15 blind routing output/state chronology is invalid")
    if not torch.equal(route.write_keys, output.write_keys) or not torch.equal(
        route.write_strengths, output.write_strengths
    ):
        raise RuntimeError("V15 blind routing changed outcome-free address/write gate")
    token_index = (outcomes.detach() > 0).to(dtype=torch.long)
    original_outcome = core.outcome_embedding(token_index)
    shared_outcome = core.outcome_embedding.weight.mean(dim=0, keepdim=True)

    def writer_input(outcome_hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                output.write_keys,
                outcome_hidden,
                output.write_keys * outcome_hidden,
                output.read_values,
                output.temporal_hidden,
            ),
            dim=-1,
        )

    # Erase is a routing gate and therefore uses the untouched objective token;
    # only the stored value content receives the shared outcome-independent token.
    erase_input = torch.cat(
        (
            route.write_keys,
            original_outcome,
            route.write_keys * original_outcome,
            route.read_values,
            route.temporal_hidden,
        ),
        dim=-1,
    )
    erase_gate = torch.sigmoid(core.erase_network(erase_input).squeeze(-1))
    write_value = core.value_network(writer_input(shared_outcome))
    allocation = core._least_unused_allocation(state.usage)
    slot_weights = output.write_strengths[0] * allocation
    slot = slot_weights.unsqueeze(-1)
    write_key = output.write_keys[0].unsqueeze(0)
    value = write_value[0].unsqueeze(0)
    keys = (1.0 - slot) * state.keys + slot * write_key
    erased = state.values * (1.0 - slot * erase_gate[0])
    values = (1.0 - slot) * erased + slot * value
    usage = state.usage + slot_weights * (1.0 - state.usage)
    coordinate = state.usage.new_tensor((state.step + 1) / core.memory_slots)
    acquisition = (1.0 - slot_weights) * state.acquisition + slot_weights * coordinate
    updated = StructureKeyedCreditState(keys, values, usage, acquisition, state.step + 1)
    core.validate_state(updated)
    event = StructureKeyedCreditEvent(
        relation_features=output.observed_relations.detach().cpu().clone(),
        temporal_features=output.observed_temporal.detach().cpu().clone(),
        base_logits=output.base_logits.detach().cpu().clone(),
        outcomes=outcomes.detach().cpu().clone(),
        evidence_refs=(evidence_ref,),
        allocation_index=int(allocation.argmax().item()),
        write_strength=float(output.write_strengths[0].detach().item()),
        erase_gate=float(erase_gate[0].detach().item()),
        write_key=write_key.detach().cpu().clone(),
        write_value=value.detach().cpu().clone(),
    )
    event._validate_record_fields()
    return updated.detached_clone(), event


def _causal_training_losses(
    true_losses: torch.Tensor,
    deranged_losses: torch.Tensor,
    zero_losses: torch.Tensor,
    twin_margins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (
        true_losses.shape == deranged_losses.shape == zero_losses.shape == twin_margins.shape
        and true_losses.ndim == 1 and true_losses.numel() > 0
    ):
        raise ValueError(
            "V15 causal loss vectors must be aligned non-empty rows: "
            f"true={tuple(true_losses.shape)} deranged={tuple(deranged_losses.shape)} "
            f"zero={tuple(zero_losses.shape)} twin={tuple(twin_margins.shape)}"
        )
    fixed_zero = zero_losses.detach()
    objective = true_losses.mean()
    separation = 0.5 * (
        F.relu(SEPARATION_MARGIN + true_losses - deranged_losses).mean()
        + F.relu(SEPARATION_MARGIN + true_losses - fixed_zero).mean()
    )
    ranking = F.relu(TWIN_MARGIN - twin_margins).mean()
    total = objective + separation + ranking
    return objective, deranged_losses.mean(), fixed_zero.mean(), separation, ranking, total


def _fit(
    core: StructureKeyedCreditMemoryCore,
    pairs: Sequence[EncodedTwinPair],
    schedule: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], tuple[float, float]]:
    if len(pairs) != TRAIN_PAIRS or len(schedule) != TRAIN_UPDATES:
        raise ValueError("V15 fit requires frozen corpus/schedule")
    device = next(core.parameters()).device
    optimizer = torch.optim.AdamW(core.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
    affine_scale = torch.nn.Parameter(torch.ones((), device=device))
    affine_bias = torch.nn.Parameter(torch.zeros((), device=device))
    affine_optimizer = torch.optim.AdamW((affine_scale, affine_bias), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
    updates = []
    core.train()
    for update_index, indices in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        states = {
            (arm, side): core.initial_state()
            for arm in ("true", "deranged") for side in ("first", "second")
        }
        true_losses: list[torch.Tensor] = []
        deranged_losses: list[torch.Tensor] = []
        zero_losses: list[torch.Tensor] = []
        twin_margins: list[torch.Tensor] = []
        affine_losses: list[torch.Tensor] = []
        chronology = {
            (arm, side): []
            for arm in ("true", "deranged") for side in ("first", "second")
        }
        for pair_index in indices:
            pair = pairs[pair_index]
            first, second = pair.first.to(device), pair.second.to(device)
            for arm in ("true", "deranged"):
                states[(arm, "first")], _, steps_a, _ = _write_anchors(core, first, second, states[(arm, "first")], arm=arm, detach_state=False)
                states[(arm, "second")], _, steps_b, _ = _write_anchors(core, second, first, states[(arm, "second")], arm=arm, detach_state=False)
                chronology[(arm, "first")].extend(steps_a)
                chronology[(arm, "second")].extend(steps_b)
            true_first_logits, true_first_loss = _probe_losses(core, first, states[("true", "first")])
            true_second_logits, true_second_loss = _probe_losses(core, second, states[("true", "second")])
            _, deranged_first_loss = _probe_losses(core, first, states[("deranged", "first")])
            _, deranged_second_loss = _probe_losses(core, second, states[("deranged", "second")])
            zero = core.initial_state()
            _, zero_first_loss = _probe_losses(core, first, zero, read_enabled=False)
            _, zero_second_loss = _probe_losses(core, second, zero, read_enabled=False)
            for local, probe_index in enumerate(PROBE_INDICES):
                true_losses.extend((true_first_loss[local], true_second_loss[local]))
                deranged_losses.extend((deranged_first_loss[local], deranged_second_loss[local]))
                zero_losses.extend((zero_first_loss[local], zero_second_loss[local]))
                first_y = first.outcomes[probe_index].reshape(())
                second_y = second.outcomes[probe_index].reshape(())
                first_logit = true_first_logits[local].reshape(-1)
                second_logit = true_second_logits[local].reshape(-1)
                if first_logit.numel() != 1 or second_logit.numel() != 1:
                    raise RuntimeError("V15 twin probe logits must each contain one scalar")
                twin_margins.extend((
                    (first_y * (first_logit[0] - second_logit[0])).reshape(()),
                    (second_y * (second_logit[0] - first_logit[0])).reshape(()),
                ))
                affine_losses.extend((
                    F.softplus(-first.outcomes[probe_index] * (affine_scale * first.base_logits[probe_index] + affine_bias)),
                    F.softplus(-second.outcomes[probe_index] * (affine_scale * second.base_logits[probe_index] + affine_bias)),
                ))
        true_loss, deranged_loss, zero_loss, separation, ranking, total = _causal_training_losses(
            torch.stack(true_losses), torch.stack(deranged_losses),
            torch.stack(zero_losses), torch.stack(twin_margins),
        )
        if not all(bool(torch.isfinite(value).item()) for value in (true_loss, deranged_loss, zero_loss, separation, ranking, total)):
            raise RuntimeError("V15 training loss is non-finite")
        total.backward()
        gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(value).all().item()) for value in gradients):
            raise RuntimeError("V15 core gradients are absent/non-finite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), GRADIENT_CLIP)
        optimizer.step()
        affine_loss = torch.stack(affine_losses).mean()
        affine_loss.backward()
        affine_optimizer.step()
        updates.append({
            "update": update_index,
            "pair_indices": list(indices),
            "true_probe_loss": float(true_loss.detach()),
            "deranged_probe_loss": float(deranged_loss.detach()),
            "zero_probe_loss": float(zero_loss.detach()),
            "separation_loss": float(separation.detach()),
            "twin_ranking_loss": float(ranking.detach()),
            "total_loss": float(total.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "true_deranged_paths_differentiable": all(value.grad_fn is not None for value in (true_loss, deranged_loss, separation, ranking)),
            "zero_baseline_detached": zero_loss.grad_fn is None,
            "predict_before_feedback": all(
                steps == list(range(PAIRS_PER_UPDATE * len(ACQUISITION_INDICES)))
                for steps in chronology.values()
            ),
        })
    return ({
        "completed_updates": len(updates),
        "updates": updates,
        "exposure_counts": dict(sorted(Counter(i for row in schedule for i in row).items())),
        "loss_weights": list(LOSS_WEIGHTS),
        "separation_margin": SEPARATION_MARGIN,
        "twin_margin": TWIN_MARGIN,
    }, (float(affine_scale.detach()), float(affine_bias.detach())))


def _balanced_accuracy(logits: Sequence[float], outcomes: Sequence[float]) -> float:
    positives = [value >= 0 for value, target in zip(logits, outcomes, strict=True) if target > 0]
    negatives = [value < 0 for value, target in zip(logits, outcomes, strict=True) if target < 0]
    return 0.5 * (sum(positives) / len(positives) + sum(negatives) / len(negatives))


def _arm_stream(
    core: StructureKeyedCreditMemoryCore,
    pairs: Sequence[EncodedTwinPair],
    *, arm: str,
    reset_after_acquisition: bool = False,
) -> dict[str, Any]:
    device = next(core.parameters()).device
    states = {side: core.initial_state() for side in ("first", "second")}
    routing_states = {side: core.initial_state() for side in ("first", "second")}
    logits, outcomes, losses, families, transitions = [], [], [], [], []
    pair_slices = []
    event_blocks: list[dict[str, list[StructureKeyedCreditEvent]]] = []
    anchor_logit_blocks: list[dict[str, list[float]]] = []
    states_after_pair: list[dict[str, StructureKeyedCreditState]] = []
    read_values, write_values = [], []
    with torch.no_grad():
        for pair_index, pair in enumerate(pairs):
            if pair_index % PAIRS_PER_UPDATE == 0:
                states = {side: core.initial_state() for side in ("first", "second")}
                routing_states = {side: core.initial_state() for side in ("first", "second")}
                event_blocks.append({"first": [], "second": []})
                anchor_logit_blocks.append({"first": [], "second": []})
            first, second = pair.first.to(device), pair.second.to(device)
            start = len(logits)
            for side, episode, twin in (("first", first, second), ("second", second, first)):
                if arm == "zero":
                    probe_state = core.initial_state()
                elif arm == "blind":
                    states[side], routing_states[side], acquired, _, anchor_logits = _write_blind_anchors(
                        core, episode, states[side], routing_states[side],
                    )
                    event_blocks[-1][side].extend(acquired)
                    anchor_logit_blocks[-1][side].extend(anchor_logits)
                    probe_state = states[side]
                else:
                    states[side], acquired, _, anchor_logits = _write_anchors(core, episode, twin, states[side], arm=arm, detach_state=True)
                    event_blocks[-1][side].extend(acquired)
                    anchor_logit_blocks[-1][side].extend(anchor_logits)
                    probe_state = core.initial_state() if reset_after_acquisition else states[side]
                probe_logits, probe_losses = _probe_losses(core, episode, probe_state, read_enabled=arm != "zero")
                for local, index in enumerate(PROBE_INDICES):
                    logits.append(float(probe_logits[local].item()))
                    outcomes.append(float(episode.outcomes[index].item()))
                    losses.append(float(probe_losses[local].item()))
                    families.append(episode.generator_family)
                    transitions.append(episode.transition_group)
                # Bounds come from an identical first probe observation.
                output = core.predict(episode.relation_features[2:3], episode.temporal_features[2:3], episode.base_logits[2:3], state=probe_state, read_enabled=arm != "zero")
                read_values.extend((float(output.read_weights.min()), float(output.read_weights.max())))
                write_values.append(float(output.write_strengths.item()))
            pair_slices.append((start, len(logits)))
            states_after_pair.append({key: value.detached_clone() for key, value in states.items()})
    return {
        "balanced_accuracy": _balanced_accuracy(logits, outcomes),
        "mean_probe_nll": sum(losses) / len(losses),
        "logits": logits,
        "outcomes": outcomes,
        "losses": losses,
        "families": families,
        "transitions": transitions,
        "pair_slices": pair_slices,
        "states": states,
        "states_after_pair": states_after_pair,
        "event_blocks": tuple(
            {key: tuple(value) for key, value in block.items()} for block in event_blocks
        ),
        "anchor_logit_blocks": tuple(anchor_logit_blocks),
        "finite": all(math.isfinite(value) for value in (*logits, *losses, *read_values, *write_values)),
        "read_weight_min": min(read_values),
        "read_weight_max": max(read_values),
        "write_strength_min": min(write_values),
        "write_strength_max": max(write_values),
        "maximum_state_step": max(
            snapshot[side].step for snapshot in states_after_pair for side in ("first", "second")
        ),
    }


def _group_report(arm: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = {}
    for name in sorted(set(arm[key])):
        indices = [index for index, value in enumerate(arm[key]) if value == name]
        result[name] = {
            "count": len(indices),
            "balanced_accuracy": _balanced_accuracy([arm["logits"][i] for i in indices], [arm["outcomes"][i] for i in indices]),
            "mean_probe_nll": sum(arm["losses"][i] for i in indices) / len(indices),
        }
    return result


def _mismatched_state(state: StructureKeyedCreditState) -> StructureKeyedCreditState:
    values = state.values.clone()
    if state.step > 1:
        values[: state.step] = values[: state.step].flip(0)
    return StructureKeyedCreditState(state.keys, values, state.usage, state.acquisition, state.step)


def _score_episode(core: StructureKeyedCreditMemoryCore, episode: EncodedContingencyEpisode, state: StructureKeyedCreditState) -> tuple[list[float], list[float]]:
    values = episode.to(next(core.parameters()).device)
    with torch.no_grad():
        logits, losses = _probe_losses(core, values, state)
    return [float(value.item()) for value in logits], [float(value.item()) for value in losses]


def _blind_addressing_integrity(
    core: StructureKeyedCreditMemoryCore,
    pair: EncodedTwinPair,
) -> dict[str, bool]:
    device = next(core.parameters()).device
    episode = pair.first.to(device)
    twin = pair.second.to(device)
    normal_state, normal_events, _, _ = _write_anchors(core, episode, twin, core.initial_state(), arm="true", detach_state=True)
    blind_state, _, blind_events, _, _ = _write_blind_anchors(
        core, episode, core.initial_state(), core.initial_state(),
    )
    event_count_exact = len(normal_events) == len(blind_events) == len(ACQUISITION_INDICES)
    allocation_exact = event_count_exact and all(
        left.allocation_index == right.allocation_index
        for left, right in zip(normal_events, blind_events, strict=True)
    )
    write_gate_exact = event_count_exact and all(
        left.write_strength == right.write_strength
        for left, right in zip(normal_events, blind_events, strict=True)
    )
    erase_gate_exact = event_count_exact and all(
        left.erase_gate == right.erase_gate
        for left, right in zip(normal_events, blind_events, strict=True)
    )
    write_keys_exact = event_count_exact and all(
        torch.equal(left.write_key, right.write_key)
        for left, right in zip(normal_events, blind_events, strict=True)
    )

    # Mechanical evidence that the blind value token itself cannot encode the
    # sign of objective feedback.  The normal erase gate intentionally still
    # receives that sign and is checked against the normal event above.
    initial = core.initial_state()
    output = core.predict(
        episode.relation_features[0:1], episode.temporal_features[0:1],
        episode.base_logits[0:1], state=initial,
    )
    _, positive = _apply_outcome_blind_feedback(
        core, initial, output, output.logits.new_tensor((1.0,)), evidence_ref="blind-integrity:+1",
    )
    _, negative = _apply_outcome_blind_feedback(
        core, initial, output, output.logits.new_tensor((-1.0,)), evidence_ref="blind-integrity:-1",
    )
    shared_value_token_outcome_invariant = torch.equal(positive.write_value, negative.write_value)
    report = {
        "write_count_exact": event_count_exact and normal_state.step == blind_state.step == len(ACQUISITION_INDICES),
        "allocation_exact": allocation_exact,
        "write_gate_exact": write_gate_exact,
        "erase_gate_exact": erase_gate_exact,
        "write_keys_exact": write_keys_exact and torch.equal(normal_state.keys, blind_state.keys),
        "usage_exact": torch.equal(normal_state.usage, blind_state.usage),
        "acquisition_exact": torch.equal(normal_state.acquisition, blind_state.acquisition),
        "stored_values_differ": not torch.equal(normal_state.values, blind_state.values),
        "shared_value_token_outcome_invariant": shared_value_token_outcome_invariant,
    }
    report["only_stored_outcome_value_content_changed"] = all(report.values())
    return report


def _unrelated_donor_index(pairs: Sequence[EncodedTwinPair], target_index: int) -> int:
    target = pairs[target_index]
    target_history = tuple(float(target.first.outcomes[index]) for index in ACQUISITION_INDICES)
    for offset in range(1, len(pairs)):
        candidate_index = (target_index + offset) % len(pairs)
        candidate = pairs[candidate_index]
        candidate_history = tuple(float(candidate.first.outcomes[index]) for index in ACQUISITION_INDICES)
        if (
            candidate.first.generator_family != target.first.generator_family
            and sum(left != right for left, right in zip(candidate_history, target_history, strict=True)) == 1
        ):
            return candidate_index
    raise RuntimeError("V15 has no genuinely unrelated-family/history donor")


def _paired_state_controls(core: StructureKeyedCreditMemoryCore, pairs: Sequence[EncodedTwinPair]) -> dict[str, Any]:
    device = next(core.parameters()).device
    normal_logits, swapped_logits, labels = [], [], []
    zero_logits, unrelated_logits, mismatch_logits = [], [], []
    for index, pair in enumerate(pairs):
        first, second = pair.first.to(device), pair.second.to(device)
        state_a, _, _, _ = _write_anchors(core, first, second, core.initial_state(), arm="true", detach_state=True)
        state_b, _, _, _ = _write_anchors(core, second, first, core.initial_state(), arm="true", detach_state=True)
        donor = pairs[_unrelated_donor_index(pairs, index)]
        donor_first, donor_second = donor.first.to(device), donor.second.to(device)
        unrelated_a, _, _, _ = _write_anchors(core, donor_first, donor_second, core.initial_state(), arm="true", detach_state=True)
        unrelated_b, _, _, _ = _write_anchors(core, donor_second, donor_first, core.initial_state(), arm="true", detach_state=True)
        for episode, matched, opposite, unrelated in ((first, state_a, state_b, unrelated_a), (second, state_b, state_a, unrelated_b)):
            current, _ = _score_episode(core, episode, matched)
            swapped, _ = _score_episode(core, episode, opposite)
            zero, _ = _score_episode(core, episode, core.initial_state())
            other, _ = _score_episode(core, episode, unrelated)
            mismatch, _ = _score_episode(core, episode, _mismatched_state(matched))
            normal_logits.extend(current); swapped_logits.extend(swapped); zero_logits.extend(zero)
            unrelated_logits.extend(other); mismatch_logits.extend(mismatch)
            labels.extend(float(episode.outcomes[i].item()) for i in PROBE_INDICES)
    normal_ba = _balanced_accuracy(normal_logits, labels)
    zero_ba = _balanced_accuracy(zero_logits, labels)
    full_gain = normal_ba - zero_ba
    reversal = sum(
        int(label * normal > 0 and label * swapped < 0)
        for label, normal, swapped in zip(labels, normal_logits, swapped_logits, strict=True)
    ) / len(labels)
    opposite_labels = [-value for value in labels]
    swapped_opposite_ba = _balanced_accuracy(swapped_logits, opposite_labels)
    zero_opposite_ba = _balanced_accuracy(zero_logits, opposite_labels)
    return {
        "matched_balanced_accuracy": normal_ba,
        "zero_balanced_accuracy": zero_ba,
        "unrelated_balanced_accuracy": _balanced_accuracy(unrelated_logits, labels),
        "key_value_mismatch_balanced_accuracy": _balanced_accuracy(mismatch_logits, labels),
        "opposite_twin_swap_reversal_fraction": reversal,
        "full_causal_gain": full_gain,
        "swapped_opposite_balanced_accuracy": swapped_opposite_ba,
        "zero_opposite_balanced_accuracy": zero_opposite_ba,
        "opposite_twin_swap_gain_retention": (
            (swapped_opposite_ba - zero_opposite_ba) / full_gain
            if full_gain > 0.0 else 0.0
        ),
        "mean_nll": {
            "matched": sum(float(F.softplus(torch.tensor(-y * z))) for y, z in zip(labels, normal_logits, strict=True)) / len(labels),
            "zero": sum(float(F.softplus(torch.tensor(-y * z))) for y, z in zip(labels, zero_logits, strict=True)) / len(labels),
            "unrelated": sum(float(F.softplus(torch.tensor(-y * z))) for y, z in zip(labels, unrelated_logits, strict=True)) / len(labels),
            "key_value_mismatch": sum(float(F.softplus(torch.tensor(-y * z))) for y, z in zip(labels, mismatch_logits, strict=True)) / len(labels),
        },
    }


def _evaluate(core: StructureKeyedCreditMemoryCore, pairs: Sequence[EncodedTwinPair], affine: tuple[float, float]) -> dict[str, Any]:
    if len(pairs) != DEVELOPMENT_PAIRS:
        raise ValueError("V15 development requires 48 twin pairs")
    core.eval()
    true = _arm_stream(core, pairs, arm="true")
    deranged = _arm_stream(core, pairs, arm="deranged")
    zero = _arm_stream(core, pairs, arm="zero")
    reset = _arm_stream(core, pairs, arm="true", reset_after_acquisition=True)
    blind = _arm_stream(core, pairs, arm="blind")
    blind_integrity = _blind_addressing_integrity(core, pairs[0])
    affine_logits = []
    affine_labels = []
    for pair in pairs:
        for episode in (pair.first, pair.second):
            for index in PROBE_INDICES:
                affine_logits.append(affine[0] * float(episode.base_logits[index]) + affine[1])
                affine_labels.append(float(episode.outcomes[index]))
    affine_ba = _balanced_accuracy(affine_logits, affine_labels)
    paired_correct, paired_margins = 0, []
    for pair_index in range(len(pairs)):
        first_start = pair_index * 8
        second_start = first_start + 4
        for local, probe in enumerate(PROBE_INDICES):
            margin = float(pairs[pair_index].first.outcomes[probe]) * (
                true["logits"][first_start + local] - true["logits"][second_start + local]
            )
            paired_margins.append(margin)
            paired_correct += int(margin > 0)
    family_true = _group_report(true, "families")
    family_deranged = _group_report(deranged, "families")
    improved = {}
    transitions: dict[str, int] = defaultdict(int)
    for family, full in family_true.items():
        control = family_deranged[family]
        succeeded = full["balanced_accuracy"] > control["balanced_accuracy"] and full["mean_probe_nll"] < control["mean_probe_nll"]
        transition = next(pair.first.transition_group for pair in pairs if pair.first.generator_family == family)
        improved[family] = {"improved": succeeded, "transition": transition}
        transitions[transition] += int(succeeded)
    informative, changed = 0, 0
    for (start, end) in true["pair_slices"]:
        left, right = true["logits"][start:end], deranged["logits"][start:end]
        if max(left) - min(left) > 1.0e-8 or max(right) - min(right) > 1.0e-8:
            informative += 1
            changed += int(sorted(range(8), key=lambda i: left[i]) != sorted(range(8), key=lambda i: right[i]))
    replay_exact, replay_error = True, 0.0
    for block_index, block in enumerate(true["event_blocks"]):
        terminal_index = (block_index + 1) * PAIRS_PER_UPDATE - 1
        for side in ("first", "second"):
            replay_state, outputs = core.replay(block[side])
            replay_exact &= core.state_digest(replay_state) == core.state_digest(
                true["states_after_pair"][terminal_index][side]
            )
            expected = true["anchor_logit_blocks"][block_index][side]
            replay_error = max(
                replay_error,
                max(abs(float(output.logits.item()) - expected[index]) for index, output in enumerate(outputs)),
            )
    # Each block reproduces the 16-write training horizon.  Score the first
    # pair immediately after acquisition and after seven later pair writes,
    # without replaying the first pair; report the worst block drop.
    retention_rows = []
    for block_start in range(0, len(pairs), PAIRS_PER_UPDATE):
        block_end = block_start + PAIRS_PER_UPDATE - 1
        pair = pairs[block_start]
        post_logits, later_logits, labels = [], [], []
        for side in ("first", "second"):
            episode = getattr(pair, side)
            values, _ = _score_episode(core, episode, true["states_after_pair"][block_start][side])
            later, _ = _score_episode(core, episode, true["states_after_pair"][block_end][side])
            post_logits.extend(values); later_logits.extend(later)
            labels.extend(float(episode.outcomes[i]) for i in PROBE_INDICES)
        post = _balanced_accuracy(post_logits, labels)
        after = _balanced_accuracy(later_logits, labels)
        retention_rows.append({"block_start": block_start, "post_acquisition": post, "after_seven_later_pairs": after, "drop": post - after})
    state_controls = _paired_state_controls(core, pairs)
    state_rows = [
        v14._state_metrics(snapshot[side])
        for snapshot in true["states_after_pair"] for side in ("first", "second")
    ]
    state_bounds = {
        "finite": all(row["finite"] for row in state_rows),
        "keys_max_abs": max(row["keys_max_abs"] for row in state_rows),
        "values_max_abs": max(row["values_max_abs"] for row in state_rows),
        "usage_min": min(row["usage_min"] for row in state_rows),
        "usage_max": max(row["usage_max"] for row in state_rows),
        "acquisition_min": min(row["acquisition_min"] for row in state_rows),
        "acquisition_max": max(row["acquisition_max"] for row in state_rows),
    }
    arms = {}
    for name, value in (("true", true), ("deranged", deranged), ("zero_no_read", zero), ("post_acquisition_reset", reset), ("outcome_blind_writer", blind)):
        arms[name] = {
            "balanced_accuracy": value["balanced_accuracy"],
            "mean_probe_nll": value["mean_probe_nll"],
            "by_family": _group_report(value, "families"),
            "by_transition": _group_report(value, "transitions"),
            "finite": value["finite"],
            "read_weight_min": value["read_weight_min"],
            "read_weight_max": value["read_weight_max"],
            "write_strength_min": value["write_strength_min"],
            "write_strength_max": value["write_strength_max"],
        }
    return {
        "pair_count": len(pairs),
        "probe_count": len(true["logits"]),
        "maximum_state_step": true["maximum_state_step"],
        "arms": arms,
        "affine_calibrator": {"scale": affine[0], "bias": affine[1], "balanced_accuracy": affine_ba},
        "outcome_blind_integrity": blind_integrity,
        "paired_twin": {
            "directional_accuracy": paired_correct / len(paired_margins),
            "mean_outcome_directed_logit_margin": sum(paired_margins) / len(paired_margins),
        },
        "state_controls": state_controls,
        "state_bounds": state_bounds,
        "improved_families": improved,
        "improved_family_count": sum(int(value["improved"]) for value in improved.values()),
        "transition_improved_family_counts": dict(sorted(transitions.items())),
        "ordering": {"informative_pairs": informative, "changed_pairs": changed, "changed_fraction": changed / informative if informative else 0.0},
        "replay": {"exact": replay_exact, "maximum_logit_error": replay_error},
        "retention": {"blocks": retention_rows, "maximum_drop": max(row["drop"] for row in retention_rows)},
    }


def _gate(metrics: Mapping[str, Any], *, identifiability_exact: bool, identity_exact: bool) -> bool:
    try:
        arms = metrics["arms"]
        true, deranged, zero = arms["true"], arms["deranged"], arms["zero_no_read"]
        controls = metrics["state_controls"]
        state = metrics["state_bounds"]
        def loss(control: str) -> bool:
            ba_delta = controls["matched_balanced_accuracy"] - controls[f"{control}_balanced_accuracy"]
            nll_delta = controls["mean_nll"][control] - controls["mean_nll"]["matched"]
            return ba_delta >= 0.10 or nll_delta >= 0.02
        return bool(
            identifiability_exact is True and identity_exact is True
            and metrics["pair_count"] == DEVELOPMENT_PAIRS
            and true["balanced_accuracy"] >= 0.70
            and true["balanced_accuracy"] - deranged["balanced_accuracy"] >= 0.10
            and deranged["mean_probe_nll"] - true["mean_probe_nll"] >= 0.02
            and true["balanced_accuracy"] - zero["balanced_accuracy"] >= 0.10
            and zero["mean_probe_nll"] - true["mean_probe_nll"] >= 0.02
            and metrics["paired_twin"]["directional_accuracy"] >= 0.75
            and metrics["paired_twin"]["mean_outcome_directed_logit_margin"] >= 0.20
            and controls["full_causal_gain"] > 0.0
            and controls["opposite_twin_swap_reversal_fraction"] >= 0.70
            and controls["opposite_twin_swap_gain_retention"] >= 0.80
            and true["balanced_accuracy"] - arms["post_acquisition_reset"]["balanced_accuracy"] >= 0.10
            and true["balanced_accuracy"] - metrics["affine_calibrator"]["balanced_accuracy"] >= 0.10
            and true["balanced_accuracy"] - arms["outcome_blind_writer"]["balanced_accuracy"] >= 0.10
            and all(value is True for value in metrics["outcome_blind_integrity"].values())
            and metrics["improved_family_count"] >= 9
            and all(value >= 2 for value in metrics["transition_improved_family_counts"].values())
            and metrics["ordering"]["changed_fraction"] >= 0.25
            and loss("zero") and loss("unrelated") and loss("key_value_mismatch")
            and metrics["replay"]["exact"] is True and metrics["replay"]["maximum_logit_error"] <= 1.0e-6
            and metrics["retention"]["maximum_drop"] <= 0.05
            and state["finite"] is True
            and state["keys_max_abs"] <= 1.0 and state["values_max_abs"] <= 1.0
            and 0.0 <= state["usage_min"] <= state["usage_max"] <= 1.0
            and 0.0 <= state["acquisition_min"] <= state["acquisition_max"] <= 1.0
            and all(
                arm["finite"] is True
                and 0.0 <= arm["read_weight_min"] <= arm["read_weight_max"] <= 1.0
                and 0.0 <= arm["write_strength_min"] <= arm["write_strength_max"] <= 1.0
                for arm in arms.values()
            )
            and metrics["protocol_invariants"]["exact_training_schedule"] is True
            and metrics["protocol_invariants"]["all_training_gradients_finite"] is True
            and metrics["protocol_invariants"]["true_deranged_paths_differentiable"] is True
            and metrics["protocol_invariants"]["zero_baseline_detached"] is True
            and metrics["protocol_invariants"]["eight_pair_state_horizon"] is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_final_authorization(development: Mapping[str, Any], checkpoint_path: Path) -> None:
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "PAIRED_LATENT_PROCEDURAL_CREDIT_SUPPORTED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and development.get("identifiability_preflight", {}).get("exact") is True
        and development.get("identity_checks", {}).get("exact") is True
        and isinstance(development.get("development_metrics"), Mapping)
        and _gate(development["development_metrics"], identifiability_exact=True, identity_exact=True)
    )
    v5._validate_final_authorization(valid)


def _load_frozen_v13(checkpoint: Path, result: Path, device: torch.device) -> tuple[StructureProtectedOMLCore, Mapping[str, Any]]:
    frozen, v13_checkpoint, _ = v14._load_frozen_v13(checkpoint, result, device)
    return frozen, v13_checkpoint


def _v14_structure_evidence(path: Path) -> Mapping[str, Any]:
    if _sha256(path) != V14_RESULT_SHA256:
        raise RuntimeError("consumed V14 structural evidence changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("structure_replicated") is not True or not v14._structure_gate(result["structure_replication"], True):
        raise RuntimeError("V14 structural evidence is not accepted")
    return result


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path, result_path = Path(args.checkpoint), Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V15 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device != torch.device("cuda:1"):
        raise RuntimeError("V15 requires V13/V15 on cuda:1")
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    v14_evidence = _v14_structure_evidence(Path(args.v14_result))
    frozen, v13_checkpoint = _load_frozen_v13(Path(args.v13_checkpoint), Path(args.v13_result), device)
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != v13_checkpoint["qwen_digest"]:
        raise RuntimeError("V15 local Qwen does not match the frozen V13 foundation")
    corpus = build_paired_latent_contingency_credit_v15()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V15 final opened before authorization")
    schedule = _training_schedule(corpus)
    train = _encode_pairs(qwen, frozen, corpus.train)
    development = _encode_pairs(qwen, frozen, corpus.development)
    core = StructureKeyedCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(device=device, dtype=torch.float32)
    preflight = _identifiability_preflight(core, train, development, schedule)
    if preflight["exact"] is not True:
        raise RuntimeError("V15 identifiability preflight failed before training")
    initial_core_digest = _model_digest(core)
    training, affine = _fit(core, train, schedule)
    _enforce_resources(started, device)
    metrics = _evaluate(core, development, affine)
    metrics["protocol_invariants"] = {
        "exact_training_schedule": training["completed_updates"] == TRAIN_UPDATES
        and set(training["exposure_counts"].values()) == {2},
        "all_training_gradients_finite": all(
            math.isfinite(float(row["gradient_norm"])) for row in training["updates"]
        ),
        "true_deranged_paths_differentiable": all(
            row["true_deranged_paths_differentiable"] is True for row in training["updates"]
        ),
        "zero_baseline_detached": all(
            row["zero_baseline_detached"] is True for row in training["updates"]
        ),
        "eight_pair_state_horizon": metrics["maximum_state_step"]
        == PAIRS_PER_UPDATE * len(ACQUISITION_INDICES),
    }
    _enforce_resources(started, device)
    sources_after = _source_hashes()
    identity = {
        "qwen_before": qwen_before,
        "qwen_after": foundation_tensor_digest(qwen.model),
        "v13_before": frozen_before,
        "v13_after": _model_digest(frozen),
        "sources_before": sources_before,
        "sources_after": sources_after,
    }
    identity["exact"] = identity["qwen_before"] == identity["qwen_after"] and identity["v13_before"] == identity["v13_after"] and sources_before == sources_after
    supported = _gate(metrics, identifiability_exact=True, identity_exact=bool(identity["exact"]))
    classification = "PAIRED_LATENT_PROCEDURAL_CREDIT_SUPPORTED" if supported else "DEVELOPMENT_NOT_SUPPORTED"
    checkpoint = {
        "identity": IDENTITY, "seed": SEED, "corpus_id": CORPUS_ID,
        "completed_updates": TRAIN_UPDATES, "temporal_width": TEMPORAL_WIDTH,
        "core_state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "core_digest": _model_digest(core), "initial_core_digest": initial_core_digest,
        "affine_calibrator": list(affine), "source_hashes": sources_before,
        "qwen_digest": qwen_before, "v13_digest": frozen_before,
        "v13_checkpoint_sha256": V13_CHECKPOINT_SHA256, "v13_result_sha256": V13_RESULT_SHA256,
        "v14_result_sha256": V14_RESULT_SHA256,
        "identifiability_preflight": preflight, "development_metrics": metrics,
    }
    result: dict[str, Any] = {
        "identity": IDENTITY, "phase": "train-development", "classification": classification,
        "development_authorized": supported, "corpus_id": CORPUS_ID, "seed": SEED,
        "identifiability_preflight": preflight, "training": training,
        "development_metrics": metrics, "identity_checks": identity,
        "source_hashes": sources_before,
        "frozen_evidence": {
            "v13_checkpoint_sha256": V13_CHECKPOINT_SHA256,
            "v13_result_sha256": V13_RESULT_SHA256,
            "v14_result_sha256": V14_RESULT_SHA256,
            "v14_structure_replicated": v14_evidence["structure_replicated"],
            "v13_checkpoint_qwen_digest": v13_checkpoint["qwen_digest"],
        },
        "frozen_compute": {
            "updates": TRAIN_UPDATES, "pairs_per_update": PAIRS_PER_UPDATE,
            "train_pairs": TRAIN_PAIRS, "development_pairs": DEVELOPMENT_PAIRS,
            "loss_weights": list(LOSS_WEIGHTS), "dtype": "float32", "autocast": False,
            "tf32": False, "qwen_device": "cuda:0", "v13_v15_device": "cuda:1",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    if not identity["exact"]:
        raise RuntimeError("V15 source/foundation drift during execution")
    _atomic_torch(checkpoint_path, checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    _atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "result": str(result_path)}, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    output = Path(args.final_result)
    if output.exists():
        raise FileExistsError("V15 final identity is already consumed")
    development_path, checkpoint_path = Path(args.development_result), Path(args.checkpoint)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sources = _source_hashes()
    if checkpoint.get("identity") != IDENTITY or checkpoint.get("source_hashes") != sources or checkpoint.get("completed_updates") != TRAIN_UPDATES:
        raise RuntimeError("V15 checkpoint binding failed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    qwen = v5._load_qwen(args.model_path)
    frozen, _ = _load_frozen_v13(Path(args.v13_checkpoint), Path(args.v13_result), device)
    _v14_structure_evidence(Path(args.v14_result))
    if foundation_tensor_digest(qwen.model) != checkpoint["qwen_digest"] or _model_digest(frozen) != checkpoint["v13_digest"]:
        raise RuntimeError("V15 final frozen dependency changed")
    corpus = build_paired_latent_contingency_credit_v15(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_PAIRS:
        raise RuntimeError("V15 authorized final is incomplete")
    pairs = _encode_pairs(qwen, frozen, corpus.final)
    core = StructureKeyedCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(device=device, dtype=torch.float32)
    core.load_state_dict(checkpoint["core_state"], strict=True)
    if _model_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V15 final core digest does not match checkpoint")
    metrics = _evaluate(core, pairs, tuple(checkpoint["affine_calibrator"]))
    metrics["protocol_invariants"] = copy.deepcopy(
        checkpoint["development_metrics"]["protocol_invariants"]
    )
    exact = foundation_tensor_digest(qwen.model) == checkpoint["qwen_digest"] and _model_digest(frozen) == checkpoint["v13_digest"] and _source_hashes() == sources
    supported = _gate(metrics, identifiability_exact=True, identity_exact=exact)
    result = {
        "identity": IDENTITY, "phase": "final", "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported, "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path), "final_metrics": metrics, "source_hashes": sources,
    }
    if not exact:
        raise RuntimeError("V15 source/foundation drift during final")
    _atomic_json(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v13-checkpoint", default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt")
    parser.add_argument("--v13-result", default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json")
    parser.add_argument("--v14-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-development.json")
    parser.add_argument("--checkpoint", default="/opt/angler/results/paired-latent-contingency-credit-v15.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/paired-latent-contingency-credit-v15-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/paired-latent-contingency-credit-v15-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
