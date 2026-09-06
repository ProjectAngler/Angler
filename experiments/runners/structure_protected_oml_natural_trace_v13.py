"""Frozen structure-protected paired-trace OML experiment V13.

The runner is deliberately generic: deterministic code only parses, batches,
measures, and enforces ownership.  The learned shared encoder receives public
trace relations through InfoNCE; the learned pair trunk receives outcome-only
OML credit; the functional fast head remains the sole inner-loop state.
"""

from __future__ import annotations

import argparse
import copy
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
from angler.reasoning.oml_natural_trace_representation import representation_metrics
from angler.reasoning.structure_protected_oml import StructureProtectedOMLCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.structure_protected_oml_natural_trace_v13 import (
    CORPUS_ID,
    DEVELOPMENT_MECHANISMS,
    FINAL_MECHANISMS,
    TRAIN_INNER_MECHANISMS,
    TRAIN_OUTER_MECHANISMS,
    TRAIN_OUTER_SLOTS,
    TRAIN_OUTER_UPDATES,
    FinalPartitionSealedError,
    build_structure_protected_oml_natural_trace_v13,
)
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners.phase6_cross_variation_plasticity import AdamWSlot
from experiments.runners.phase6_cross_variation_plasticity_v16 import (
    functional_adamw_step,
)


IDENTITY = "angler.structure-protected-oml-natural-trace.v13-first-result"
SEED = 2026083113
MAXIMUM_TRACE_STEPS = 8
INNER_STEPS = 8
OUTER_MECHANISMS = 8
OUTER_UPDATES = 192
INNER_LEARNING_RATE = 1.0e-3
OWNER_LEARNING_RATE = 3.0e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1.0e-8
ADAM_WEIGHT_DECAY = 0.0
OWNER_GRADIENT_CLIP = 5.0
ROBUST_TEMPERATURE = 0.05
INFONCE_TEMPERATURE = 0.10
SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4)

ARM_PROTECTED_SECOND = "protected_second_online"
ARM_PROTECTED_FIRST = "protected_first_online"
ARM_PROTECTED_SECOND_NO = "protected_second_no_update"
ARM_PROTECTED_FIRST_NO = "protected_first_no_update"
ARM_OUTCOME_ONLY = "paired_outcome_only_second"
ARM_SOURCE = "paired_source_online"
ARM_SOURCE_NO = "paired_source_no_update"
ARM_SHUFFLED = "shuffled_support_feedback"
ARM_DIRECTION_REMOVED = "direction_removed"
ARM_SEMANTICS_REMOVED = "step_semantics_removed"

CORRUPTIONS = ("reorder", "replacement", "omission", "insertion")
TRANSITIONS = ("insertion", "removal", "reorder", "replacement")


@dataclass(frozen=True, slots=True)
class EncodedPairMechanism:
    mechanism_ref: str
    generator_family: str
    transition_group: str
    renderer_position_strata: tuple[str, ...]
    corruption_types: tuple[str, ...]
    reference_features: torch.Tensor
    reference_mask: torch.Tensor
    attempt_features: torch.Tensor
    attempt_mask: torch.Tensor
    outcomes: torch.Tensor
    structural_features: torch.Tensor
    structural_mask: torch.Tensor

    def to(self, device: torch.device) -> "EncodedPairMechanism":
        return EncodedPairMechanism(
            mechanism_ref=self.mechanism_ref,
            generator_family=self.generator_family,
            transition_group=self.transition_group,
            renderer_position_strata=self.renderer_position_strata,
            corruption_types=self.corruption_types,
            reference_features=self.reference_features.to(device=device, dtype=torch.float32),
            reference_mask=self.reference_mask.to(device=device),
            attempt_features=self.attempt_features.to(device=device, dtype=torch.float32),
            attempt_mask=self.attempt_mask.to(device=device),
            outcomes=self.outcomes.to(device=device, dtype=torch.float32),
            structural_features=self.structural_features.to(device=device, dtype=torch.float32),
            structural_mask=self.structural_mask.to(device=device),
        )


@dataclass(frozen=True, slots=True)
class EncodedRetrievalSet:
    mechanism_refs: tuple[str, ...]
    generator_families: tuple[str, ...]
    transition_groups: tuple[str, ...]
    renderer_position_strata: tuple[str, ...]
    reference_features: torch.Tensor
    reference_mask: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_indices: torch.Tensor
    candidate_types: tuple[tuple[str, ...], ...]

    def to(self, device: torch.device) -> "EncodedRetrievalSet":
        return EncodedRetrievalSet(
            mechanism_refs=self.mechanism_refs,
            generator_families=self.generator_families,
            transition_groups=self.transition_groups,
            renderer_position_strata=self.renderer_position_strata,
            reference_features=self.reference_features.to(device=device, dtype=torch.float32),
            reference_mask=self.reference_mask.to(device=device),
            candidate_features=self.candidate_features.to(device=device, dtype=torch.float32),
            candidate_mask=self.candidate_mask.to(device=device),
            target_indices=self.target_indices.to(device=device),
            candidate_types=self.candidate_types,
        )


@dataclass(slots=True)
class LearnedArm:
    name: str
    model: StructureProtectedOMLCore
    encoder_optimizer: torch.optim.Optimizer | None
    trunk_optimizer: torch.optim.Optimizer | None
    outcome_optimizer: torch.optim.Optimizer | None
    updates: int = 0


@dataclass(slots=True)
class V13System:
    protected_second: LearnedArm
    protected_first: LearnedArm
    outcome_only: LearnedArm
    source: LearnedArm
    initial_digest: str


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = {
        "runner": Path(__file__).resolve(),
        "core": root / "src/angler/reasoning/structure_protected_oml.py",
        "trace_graph_core": root / "src/angler/reasoning/natural_trace_graph_causal_memory.py",
        "content_memory_core": root / "src/angler/reasoning/content_addressed_causal_memory.py",
        "dnc_memory_core": root / "src/angler/reasoning/dnc_allocated_factorized_causal_memory.py",
        "factorized_memory_core": root / "src/angler/reasoning/factorized_key_value_causal_memory.py",
        "sparse_memory_core": root / "src/angler/reasoning/sparse_memory_causal_routing.py",
        "causal_neuromodulator_core": root / "src/angler/reasoning/causal_neuromodulated_apprenticeship.py",
        "outcome_apprenticeship_core": root / "src/angler/reasoning/outcome_aware_apprenticeship.py",
        "situated_feedback_core": root / "src/angler/reasoning/situated_feedback.py",
        "reasoning_exports": root / "src/angler/reasoning/__init__.py",
        "corpus": root / "experiments/corpora/structure_protected_oml_natural_trace_v13.py",
        "v12_corpus": root / "experiments/corpora/scaled_oml_natural_trace_v12.py",
        "v6_corpus": root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py",
        "v5_corpus": root / "experiments/corpora/outcome_aware_apprenticeship_v5.py",
        "functional_adamw": root / "experiments/runners/phase6_cross_variation_plasticity_v16.py",
        "adamw_slot": root / "experiments/runners/phase6_cross_variation_plasticity.py",
        "qwen_runtime": root / "src/angler/runtime/qwen_knowledge.py",
        "runtime_exports": root / "src/angler/runtime/__init__.py",
        "qwen_loader": root / "experiments/runners/outcome_aware_apprenticeship_v5.py",
    }
    return {name: _sha256(path) for name, path in files.items()}


def _model_digest(model: torch.nn.Module) -> str:
    return v5._module_digest(model)


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(tuple(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_digest(optimizer: torch.optim.Optimizer | None) -> str:
    if optimizer is None:
        return "NONE"
    buffer = []
    state = optimizer.state_dict()
    buffer.append(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for index in sorted(state["state"]):
        for name, value in sorted(state["state"][index].items()):
            buffer.append(f"{index}:{name}".encode())
            if isinstance(value, torch.Tensor):
                buffer.append(_tensor_digest(value).encode())
            else:
                buffer.append(str(value).encode())
    return hashlib.sha256(b"\0".join(buffer)).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V13 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale V13 temporary output exists: {temporary}")
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
        raise FileExistsError(f"V13 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale V13 temporary output exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_sequence(value: Any, *names: str, length: int) -> tuple[str, ...]:
    for name in names:
        found = getattr(value, name, None)
        if found is not None:
            result = tuple(str(item) for item in found)
            if len(result) != length:
                raise ValueError(f"V13 metadata {name} length changed")
            return result
    raise AttributeError(f"V13 metadata lacks required field: {names}")


def _mechanism_metadata(mechanism: Any) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    metadata = mechanism.metadata
    reference = str(metadata.mechanism_ref)
    family = str(metadata.generator_family)
    transition = str(metadata.heldout_variant)
    if not transition:
        raise AttributeError("V13 mechanism metadata lacks transition grouping")
    corruptions = tuple(str(value) for value in mechanism.supervision.episode_attempt_relations)
    if len(corruptions) != 6:
        raise ValueError("V13 episode relation reporting length changed")
    reference_plans = tuple(str(value) for value in metadata.episode_reference_plan_ids)
    attempt_plans = tuple(str(value) for value in metadata.episode_attempt_plan_ids)
    if len(reference_plans) != 6 or len(attempt_plans) != 6:
        raise ValueError("V13 episode renderer provenance length changed")
    renderer = tuple(
        f"{reference}:{attempt}" for reference, attempt in zip(reference_plans, attempt_plans, strict=True)
    )
    return reference, family, transition, corruptions, renderer


def _parse_public_pairs(mechanisms: Sequence[Any]) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    references = []
    attempts = []
    for mechanism in mechanisms:
        public_payload = mechanism.to_learner_payload()
        if set(public_payload) != {"episodes"}:
            raise RuntimeError("V13 learner boundary exposed non-episode data")
        ref_row = []
        attempt_row = []
        if len(mechanism.public.episodes) != 6:
            raise ValueError("V13 mechanism must contain six paired episodes")
        for episode, payload in zip(mechanism.public.episodes, public_payload["episodes"], strict=True):
            if set(payload) != {"reference_trace_text", "attempt_trace_text", "outcome_label"}:
                raise RuntimeError("V13 paired-row learner payload changed")
            reference = tuple(parse_step_trace(episode.reference_trace_text))
            attempt = tuple(parse_step_trace(episode.attempt_trace_text))
            if not 1 <= len(reference) <= MAXIMUM_TRACE_STEPS or not 1 <= len(attempt) <= MAXIMUM_TRACE_STEPS:
                raise ValueError("V13 public trace exceeds frozen step ceiling")
            ref_row.append(reference)
            attempt_row.append(attempt)
        references.append(tuple(ref_row))
        attempts.append(tuple(attempt_row))
    return tuple(references), tuple(attempts)


def _embed_groups(qwen: Any, groups: Sequence[Sequence[Sequence[str]]]) -> tuple[torch.Tensor, torch.Tensor]:
    sentences: list[str] = []
    spans = []
    for group in groups:
        if len(group) != 6:
            raise ValueError("V13 encoded trace group must contain six traces")
        row = []
        for trace in group:
            start = len(sentences)
            sentences.extend(trace)
            row.append((start, len(trace)))
        spans.append(tuple(row))
    encoded = qwen.embed(sentences).to(device="cpu", dtype=torch.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(sentences):
        raise RuntimeError("Qwen embeddings do not align with V13 public steps")
    features = encoded.new_zeros((len(groups), 6, MAXIMUM_TRACE_STEPS, encoded.shape[-1]))
    mask = torch.zeros((len(groups), 6, MAXIMUM_TRACE_STEPS), dtype=torch.bool)
    for mechanism, row in enumerate(spans):
        for episode, (start, length) in enumerate(row):
            features[mechanism, episode, :length] = encoded[start : start + length]
            mask[mechanism, episode, :length] = True
    return features, mask


def _encode_mechanisms(qwen: Any, mechanisms: Sequence[Any]) -> tuple[EncodedPairMechanism, ...]:
    references, attempts = _parse_public_pairs(mechanisms)
    reference_features, reference_mask = _embed_groups(qwen, references)
    attempt_features, attempt_mask = _embed_groups(qwen, attempts)
    structural_groups = []
    for mechanism in mechanisms:
        contrasts = mechanism.structural_contrasts
        texts = (
            contrasts.reference_trace_text,
            contrasts.paraphrase_attempt_trace_text,
            contrasts.reordered_attempt_trace_text,
            contrasts.replacement_attempt_trace_text,
            contrasts.omitted_attempt_trace_text,
            contrasts.inserted_attempt_trace_text,
        )
        structural_groups.append(tuple(tuple(parse_step_trace(text)) for text in texts))
    structural_features, structural_mask = _embed_groups(qwen, structural_groups)
    rows = []
    for index, mechanism in enumerate(mechanisms):
        ref, family, transition, corruptions, renderer = _mechanism_metadata(mechanism)
        rows.append(
            EncodedPairMechanism(
                mechanism_ref=ref,
                generator_family=family,
                transition_group=transition,
                renderer_position_strata=renderer,
                corruption_types=corruptions,
                reference_features=reference_features[index].clone(),
                reference_mask=reference_mask[index].clone(),
                attempt_features=attempt_features[index].clone(),
                attempt_mask=attempt_mask[index].clone(),
                outcomes=torch.tensor(
                    tuple(int(episode.outcome_value) for episode in mechanism.public.episodes),
                    dtype=torch.float32,
                ),
                structural_features=structural_features[index].clone(),
                structural_mask=structural_mask[index].clone(),
            )
        )
    return tuple(rows)


def _encode_retrieval_sets(qwen: Any, mechanisms: Sequence[Any]) -> EncodedRetrievalSet:
    reference_groups = []
    candidate_groups = []
    refs = []
    families = []
    transitions = []
    strata = []
    targets = []
    candidate_types = []
    for mechanism in mechanisms:
        contrast = mechanism.structural_contrasts
        serialized = tuple(contrast.serialized_candidate_attempt_trace_texts)
        relations = tuple(str(value) for value in mechanism.supervision.candidate_attempt_relations)
        if len(serialized) != 5 or len(relations) != 5 or set(relations) != {
            "paraphrase", "reorder", "replacement", "omission", "insertion"
        }:
            raise RuntimeError("V13 serialized retrieval candidates changed")
        reference = tuple(parse_step_trace(contrast.reference_trace_text))
        parsed_candidates = tuple(tuple(parse_step_trace(text)) for text in serialized)
        reference_groups.append((reference,) * 6)
        candidate_groups.append((*parsed_candidates, parsed_candidates[0]))
        refs.append(str(mechanism.metadata.mechanism_ref))
        families.append(str(mechanism.metadata.generator_family))
        transitions.append(str(mechanism.metadata.heldout_variant))
        target = int(mechanism.supervision.same_order_candidate_index)
        if not 0 <= target < 5 or relations[target] != "paraphrase":
            raise RuntimeError("V13 retrieval target binding changed")
        targets.append(target)
        candidate_types.append(relations)
        strata.append(
            f"{mechanism.metadata.structural_reference_plan_id}:"
            f"{mechanism.metadata.structural_attempt_plan_id}:{target}"
        )
    reference_features, reference_mask = _embed_groups(qwen, reference_groups)
    candidate_features, candidate_mask = _embed_groups(qwen, candidate_groups)
    return EncodedRetrievalSet(
        mechanism_refs=tuple(refs),
        generator_families=tuple(families),
        transition_groups=tuple(transitions),
        renderer_position_strata=tuple(strata),
        reference_features=reference_features,
        reference_mask=reference_mask,
        candidate_features=candidate_features[:, :5].clone(),
        candidate_mask=candidate_mask[:, :5].clone(),
        target_indices=torch.tensor(targets, dtype=torch.long),
        candidate_types=tuple(candidate_types),
    )


def _batch(
    rows: Sequence[EncodedPairMechanism], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("V13 pair batch cannot be empty")
    moved = tuple(row.to(device) for row in rows)
    return (
        torch.stack(tuple(row.reference_features for row in moved)),
        torch.stack(tuple(row.reference_mask for row in moved)),
        torch.stack(tuple(row.attempt_features for row in moved)),
        torch.stack(tuple(row.attempt_mask for row in moved)),
        torch.stack(tuple(row.outcomes for row in moved)),
    )


def _structural_batch(
    rows: Sequence[EncodedPairMechanism], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("V13 structural batch cannot be empty")
    moved = tuple(row.to(device) for row in rows)
    return (
        torch.stack(tuple(row.structural_features for row in moved)),
        torch.stack(tuple(row.structural_mask for row in moved)),
    )


def _structural_codes(
    model: StructureProtectedOMLCore,
    rows: Sequence[EncodedPairMechanism],
    *,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> torch.Tensor:
    features, mask = _structural_batch(rows, next(model.parameters()).device)
    references = features[:, :1].expand(-1, 6, -1, -1)
    reference_mask = mask[:, :1].expand(-1, 6, -1)
    reference_codes, candidate_codes = model.encode_views(
        references,
        reference_mask,
        features,
        mask,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    if not torch.equal(reference_codes[:, :1], reference_codes[:, 1:2]):
        raise RuntimeError("V13 repeated structural reference encoding is not exact")
    return torch.cat((reference_codes[:, :1], candidate_codes[:, 1:]), dim=1)


def _symmetric_infonce_from_codes(codes: torch.Tensor) -> torch.Tensor:
    """Direct-code symmetric InfoNCE over ref, positive, and four hard negatives."""

    if codes.ndim != 3 or codes.shape[1:] != (6, 32) or codes.shape[0] == 0:
        raise ValueError("V13 InfoNCE codes must be [batch,6,32]")
    normalized = F.normalize(codes, dim=-1)
    references = normalized[:, 0]
    positives = normalized[:, 1]
    negatives = normalized[:, 2:].reshape(-1, 32)
    forward_candidates = torch.cat((positives, negatives), dim=0)
    reverse_candidates = torch.cat((references, negatives), dim=0)
    targets = torch.arange(codes.shape[0], device=codes.device)
    forward = F.cross_entropy(
        references @ forward_candidates.T / INFONCE_TEMPERATURE, targets
    )
    reverse = F.cross_entropy(
        positives @ reverse_candidates.T / INFONCE_TEMPERATURE, targets
    )
    loss = 0.5 * (forward + reverse)
    if loss.shape != () or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("V13 direct-code InfoNCE is invalid")
    return loss


def _symmetric_infonce(
    model: StructureProtectedOMLCore,
    rows: Sequence[EncodedPairMechanism],
) -> torch.Tensor:
    return _symmetric_infonce_from_codes(_structural_codes(model, rows))


def _serialization_invariance(codes: torch.Tensor) -> bool:
    swapped = codes.clone()
    swapped[:, 0], swapped[:, 1] = codes[:, 1].clone(), codes[:, 0].clone()
    negative_permutation = torch.tensor((0, 1, 4, 5, 2, 3), device=codes.device)
    swapped = swapped[:, negative_permutation]
    return bool(
        torch.allclose(
            _symmetric_infonce_from_codes(codes),
            _symmetric_infonce_from_codes(swapped),
            atol=1.0e-7,
            rtol=0.0,
        )
    )


def _robust_objective(losses: torch.Tensor, expected: int = OUTER_MECHANISMS) -> torch.Tensor:
    if losses.ndim != 1 or losses.numel() != expected or not bool(torch.isfinite(losses).all().item()):
        raise ValueError("V13 robust objective requires its exact finite loss vector")
    temperature = losses.new_tensor(ROBUST_TEMPERATURE)
    return 0.5 * losses.mean() + 0.5 * temperature * (
        torch.logsumexp(losses / temperature, dim=0) - math.log(losses.numel())
    )


def _robust_weights(losses: torch.Tensor) -> torch.Tensor:
    if losses.ndim != 1 or not bool(torch.isfinite(losses).all().item()):
        raise ValueError("V13 robust weights require finite losses")
    return 0.5 / losses.numel() + 0.5 * torch.softmax(
        losses.detach() / ROBUST_TEMPERATURE, dim=0
    )


def _fresh_fast(
    model: StructureProtectedOMLCore,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...]]:
    fast = model.fresh_fast_weight()
    zero = torch.zeros_like(fast)
    return fast, (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _inner_step(
    model: StructureProtectedOMLCore,
    fast: torch.Tensor,
    state: Sequence[AdamWSlot],
    row: EncodedPairMechanism,
    *,
    second_order: bool,
    outcomes: torch.Tensor | None = None,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], float]:
    data = _batch((row,), fast.device)
    observed = data[-1] if outcomes is None else outcomes.reshape(1, 6).to(fast)
    loss = model.functional_loss(
        *data[:-1],
        observed,
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    gradient = torch.autograd.grad(
        loss, (fast,), create_graph=second_order, retain_graph=second_order
    )[0]
    used = gradient if second_order else gradient.detach()
    updated, moments = functional_adamw_step(
        (fast,),
        (used,),
        tuple(state),
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    if not bool(torch.isfinite(updated[0]).all().item()):
        raise RuntimeError("V13 functional fast update is non-finite")
    return updated[0], tuple(moments), float(loss.detach().item())


def _unroll(
    model: StructureProtectedOMLCore,
    rows: Sequence[EncodedPairMechanism],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], tuple[float, ...]]:
    if len(rows) != INNER_STEPS:
        raise ValueError("V13 OML trajectory requires eight mechanisms")
    before = _model_digest(model)
    fast, state = _fresh_fast(model)
    losses = []
    for row in rows:
        fast, state, loss = _inner_step(
            model, fast, state, row, second_order=second_order
        )
        losses.append(loss)
    if state[0].step != INNER_STEPS or _model_digest(model) != before:
        raise RuntimeError("V13 inner trajectory crossed ownership boundary")
    return fast, state, tuple(losses)


def _outer_losses(
    model: StructureProtectedOMLCore,
    fast: torch.Tensor,
    rows: Sequence[EncodedPairMechanism],
    *,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> torch.Tensor:
    if not 1 <= len(rows) <= OUTER_MECHANISMS:
        raise ValueError("V13 outer loss group requires one to eight rows")
    return torch.stack(
        tuple(
            model.functional_loss(
                *_batch((row,), fast.device),
                fast,
                include_direction=include_direction,
                include_step_semantics=include_step_semantics,
            )
            for row in rows
        )
    )


def _oml_gradients(
    model: StructureProtectedOMLCore,
    inner: Sequence[EncodedPairMechanism],
    outer: Sequence[EncodedPairMechanism],
    *,
    second_order: bool,
    outcome_only: bool,
    split: bool = False,
) -> tuple[float, tuple[torch.Tensor, ...], dict[str, Any]]:
    fast, state, inner_losses = _unroll(model, inner, second_order=second_order)
    named = (
        tuple(model.named_parameters())
        if outcome_only
        else model.trunk_named_parameters()
    )
    parameters = tuple(value for _, value in named)
    if not split:
        losses = _outer_losses(model, fast, outer)
        objective = _robust_objective(losses)
        gradients = torch.autograd.grad(objective, parameters, allow_unused=False)
    else:
        with torch.no_grad():
            detached = _outer_losses(model, fast, outer)
            objective = _robust_objective(detached)
            weights = _robust_weights(detached)
        accumulated = [torch.zeros_like(value) for value in parameters]
        for group, start in enumerate((0, 4)):
            losses_group = _outer_losses(model, fast, outer[start : start + 4])
            gradients_group = torch.autograd.grad(
                (weights[start : start + 4].to(losses_group) * losses_group).sum(),
                parameters,
                retain_graph=group == 0,
                allow_unused=False,
            )
            for index, gradient in enumerate(gradients_group):
                accumulated[index] += gradient.detach()
        gradients = tuple(accumulated)
        losses = detached
    if any(not bool(value.isfinite().all().item()) for value in gradients):
        raise RuntimeError("V13 OML owner gradient is non-finite")
    return float(objective.detach().item()), tuple(value.detach() for value in gradients), {
        "inner_losses": inner_losses,
        "outer_losses": tuple(float(value) for value in losses.detach().tolist()),
        "terminal_fast_step": state[0].step,
        "parameter_names": tuple(name for name, _ in named),
    }


def _structural_gradients(
    model: StructureProtectedOMLCore,
    rows: Sequence[EncodedPairMechanism],
) -> tuple[float, tuple[torch.Tensor, ...], torch.Tensor]:
    codes = _structural_codes(model, rows)
    loss = _symmetric_infonce_from_codes(codes)
    parameters = tuple(value for _, value in model.encoder_named_parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    if any(not bool(value.isfinite().all().item()) for value in gradients):
        raise RuntimeError("V13 structural encoder gradient is non-finite")
    return float(loss.detach().item()), tuple(value.detach() for value in gradients), codes.detach()


def _named_digest(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode() + b"\0" + _tensor_digest(value).encode() + b"\0")
    return digest.hexdigest()


def _new_optimizer(parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        tuple(parameters),
        lr=OWNER_LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )


def _build_system(step_width: int, device: torch.device) -> V13System:
    torch.manual_seed(SEED)
    initial = StructureProtectedOMLCore(step_width=step_width).to(
        device=device, dtype=torch.float32
    )
    models = [copy.deepcopy(initial) for _ in range(4)]
    digest = _model_digest(initial)
    if any(_model_digest(model) != digest for model in models):
        raise RuntimeError("V13 learned/source arms did not start byte-exact")
    del initial

    def protected(name: str, model: StructureProtectedOMLCore) -> LearnedArm:
        return LearnedArm(
            name=name,
            model=model,
            encoder_optimizer=_new_optimizer(
                tuple(value for _, value in model.encoder_named_parameters())
            ),
            trunk_optimizer=_new_optimizer(
                tuple(value for _, value in model.trunk_named_parameters())
            ),
            outcome_optimizer=None,
        )

    outcome_model = models[2]
    source_model = models[3]
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    return V13System(
        protected_second=protected(ARM_PROTECTED_SECOND, models[0]),
        protected_first=protected(ARM_PROTECTED_FIRST, models[1]),
        outcome_only=LearnedArm(
            name=ARM_OUTCOME_ONLY,
            model=outcome_model,
            encoder_optimizer=None,
            trunk_optimizer=None,
            outcome_optimizer=_new_optimizer(tuple(outcome_model.parameters())),
        ),
        source=LearnedArm(ARM_SOURCE, source_model, None, None, None),
        initial_digest=digest,
    )


def _apply_owner(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer | None,
    gradients: Sequence[torch.Tensor],
) -> float:
    if optimizer is None or len(named) != len(gradients):
        raise RuntimeError("V13 owner optimizer/gradient partition mismatch")
    optimizer.zero_grad(set_to_none=True)
    for (name, parameter), gradient in zip(named, gradients, strict=True):
        if parameter.shape != gradient.shape:
            raise RuntimeError(f"V13 owner gradient shape changed: {name}")
        parameter.grad = gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(
        tuple(parameter for _, parameter in named), OWNER_GRADIENT_CLIP
    )
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("V13 owner gradient norm is non-finite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.item())


def _validate_schedule(schedule: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if len(schedule) != OUTER_UPDATES or any(len(row) != INNER_STEPS for row in schedule):
        raise ValueError("V13 inner schedule shape changed")
    uses = [0] * TRAIN_INNER_MECHANISMS
    for row in schedule:
        if len(set(row)) != INNER_STEPS:
            raise RuntimeError("V13 update reuses an inner mechanism")
        for index in row:
            if type(index) is not int or not 0 <= index < TRAIN_INNER_MECHANISMS:
                raise ValueError("V13 schedule index is invalid")
            uses[index] += 1
    if any(value != 4 for value in uses):
        raise RuntimeError("V13 inner schedule lost exact four-use balance")
    return tuple(uses)


def _full_split_equivalence(
    model: StructureProtectedOMLCore,
    inner: Sequence[EncodedPairMechanism],
    outer: Sequence[EncodedPairMechanism],
    *,
    second_order: bool,
    outcome_only: bool,
) -> dict[str, float]:
    full_objective, full_gradients, _ = _oml_gradients(
        model,
        inner,
        outer,
        second_order=second_order,
        outcome_only=outcome_only,
    )
    split_objective, split_gradients, _ = _oml_gradients(
        model,
        inner,
        outer,
        second_order=second_order,
        outcome_only=outcome_only,
        split=True,
    )
    maximum = max(
        float((left - right).abs().max().item())
        for left, right in zip(full_gradients, split_gradients, strict=True)
    )
    objective_delta = abs(full_objective - split_objective)
    if maximum > 1.0e-6 or objective_delta > 1.0e-6:
        raise RuntimeError("V13 full/split OML gradients diverged")
    return {
        "objective_absolute_delta": objective_delta,
        "maximum_gradient_absolute_delta": maximum,
        "tolerance": 1.0e-6,
    }


def _connectivity_preflight(
    second: StructureProtectedOMLCore,
    first: StructureProtectedOMLCore,
    row: EncodedPairMechanism,
) -> dict[str, bool]:
    second_fast, second_state = _fresh_fast(second)
    first_fast, first_state = _fresh_fast(first)
    second_data = _batch((row,), second_fast.device)
    first_data = _batch((row,), first_fast.device)
    second_loss = second.functional_loss(*second_data, second_fast)
    first_loss = first.functional_loss(*first_data, first_fast)
    second_gradient = torch.autograd.grad(
        second_loss, (second_fast,), create_graph=True, retain_graph=True
    )[0]
    first_gradient = torch.autograd.grad(first_loss, (first_fast,))[0]
    (second_updated,), second_moments = functional_adamw_step(
        (second_fast,), (second_gradient,), second_state, (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    (first_updated,), first_moments = functional_adamw_step(
        (first_fast,), (first_gradient.detach(),), first_state, (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1, beta2=ADAM_BETA2, epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    second_hessian = torch.autograd.grad(
        second_updated.sum(),
        tuple(value for _, value in second.trunk_named_parameters()),
        allow_unused=True,
    )
    first_hessian = torch.autograd.grad(
        first_updated.sum(),
        tuple(value for _, value in first.trunk_named_parameters()),
        allow_unused=True,
    )
    report = {
        "paired_inputs_exact": all(torch.equal(a, b) for a, b in zip(second_data, first_data)),
        "raw_loss_exact": bool(torch.equal(second_loss.detach(), first_loss.detach())),
        "raw_fast_gradient_exact": bool(torch.equal(second_gradient.detach(), first_gradient.detach())),
        "updated_fast_exact": bool(torch.equal(second_updated.detach(), first_updated.detach())),
        "moments_exact": bool(
            torch.equal(second_moments[0].exp_avg.detach(), first_moments[0].exp_avg.detach())
            and torch.equal(second_moments[0].exp_avg_sq.detach(), first_moments[0].exp_avg_sq.detach())
        ),
        "second_trunk_hessian_nonzero": any(
            value is not None and float(value.abs().sum().item()) > 0.0
            for value in second_hessian
        ),
        "first_trunk_hessian_absent": all(value is None for value in first_hessian),
    }
    if not all(report.values()):
        raise RuntimeError("V13 protected connectivity preflight failed")
    return report


def _owner_preflight(
    system: V13System,
    inner: Sequence[EncodedPairMechanism],
    outer: Sequence[EncodedPairMechanism],
) -> dict[str, Any]:
    second = system.protected_second.model
    first = system.protected_first.model
    connectivity = _connectivity_preflight(second, first, inner[0])
    structural_loss, structural_gradients, codes = _structural_gradients(second, outer)
    second_objective, trunk_gradients, _ = _oml_gradients(
        second, inner, outer, second_order=True, outcome_only=False
    )
    if not any(float(value.abs().sum().item()) > 0.0 for value in structural_gradients):
        raise RuntimeError("V13 structural encoder gradient path is absent")
    if not any(float(value.abs().sum().item()) > 0.0 for value in trunk_gradients):
        raise RuntimeError("V13 direct OML trunk gradient path is absent")

    structure_probe = copy.deepcopy(second)
    structure_optimizer = _new_optimizer(
        tuple(value for _, value in structure_probe.encoder_named_parameters())
    )
    structure_trunk_before = _named_digest(structure_probe.trunk_named_parameters())
    _apply_owner(
        structure_probe.encoder_named_parameters(), structure_optimizer, structural_gradients
    )
    structure_cross_owner_exact = (
        _named_digest(structure_probe.trunk_named_parameters()) == structure_trunk_before
    )

    trunk_probe = copy.deepcopy(second)
    trunk_optimizer = _new_optimizer(
        tuple(value for _, value in trunk_probe.trunk_named_parameters())
    )
    trunk_encoder_before = _named_digest(trunk_probe.encoder_named_parameters())
    _apply_owner(trunk_probe.trunk_named_parameters(), trunk_optimizer, trunk_gradients)
    trunk_cross_owner_exact = (
        _named_digest(trunk_probe.encoder_named_parameters()) == trunk_encoder_before
    )
    report = {
        "connectivity": connectivity,
        "structural_loss": structural_loss,
        "second_objective": second_objective,
        "structural_encoder_gradient_nonzero": True,
        "direct_outer_trunk_gradient_nonzero": True,
        "structure_cross_owner_exact": structure_cross_owner_exact,
        "trunk_cross_owner_exact": trunk_cross_owner_exact,
        "infonce_serialization_invariant": _serialization_invariance(codes),
        "protected_second_full_split": _full_split_equivalence(
            second, inner, outer, second_order=True, outcome_only=False
        ),
        "protected_first_full_split": _full_split_equivalence(
            first, inner, outer, second_order=False, outcome_only=False
        ),
        "outcome_control_full_split": _full_split_equivalence(
            system.outcome_only.model,
            inner,
            outer,
            second_order=True,
            outcome_only=True,
        ),
    }
    if not structure_cross_owner_exact or not trunk_cross_owner_exact or not report[
        "infonce_serialization_invariant"
    ]:
        raise RuntimeError("V13 owner/InfoNCE preflight failed")
    return report


def _fit(
    system: V13System,
    inner_rows: Sequence[EncodedPairMechanism],
    outer_rows: Sequence[EncodedPairMechanism],
    schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(inner_rows) != TRAIN_INNER_MECHANISMS or len(outer_rows) != TRAIN_OUTER_MECHANISMS:
        raise ValueError("V13 encoded train counts changed")
    uses = _validate_schedule(schedule)
    source_before = _model_digest(system.source.model)
    preflight = None
    diagnostics = []
    for update in range(OUTER_UPDATES):
        inner = tuple(inner_rows[index] for index in schedule[update])
        outer = tuple(
            outer_rows[update * OUTER_MECHANISMS + index]
            for index in range(OUTER_MECHANISMS)
        )
        if update == 0:
            preflight = _owner_preflight(system, inner, outer)
        second_objective, second_trunk_gradients, second_details = _oml_gradients(
            system.protected_second.model,
            inner,
            outer,
            second_order=True,
            outcome_only=False,
        )
        first_objective, first_trunk_gradients, first_details = _oml_gradients(
            system.protected_first.model,
            inner,
            outer,
            second_order=False,
            outcome_only=False,
        )
        outcome_objective, outcome_gradients, _ = _oml_gradients(
            system.outcome_only.model,
            inner,
            outer,
            second_order=True,
            outcome_only=True,
        )
        second_structure, second_encoder_gradients, _ = _structural_gradients(
            system.protected_second.model, outer
        )
        first_structure, first_encoder_gradients, _ = _structural_gradients(
            system.protected_first.model, outer
        )
        if update == 0 and (
            second_objective != first_objective
            or second_objective != outcome_objective
            or second_details["outer_losses"] != first_details["outer_losses"]
        ):
            raise RuntimeError("V13 paired numeric starts diverged")
        if (
            second_structure != first_structure
            or any(
                not torch.equal(left, right)
                for left, right in zip(
                    second_encoder_gradients, first_encoder_gradients, strict=True
                )
            )
        ):
            raise RuntimeError("V13 protected structural owner gradients diverged")

        second_encoder_norm = _apply_owner(
            system.protected_second.model.encoder_named_parameters(),
            system.protected_second.encoder_optimizer,
            second_encoder_gradients,
        )
        first_encoder_norm = _apply_owner(
            system.protected_first.model.encoder_named_parameters(),
            system.protected_first.encoder_optimizer,
            first_encoder_gradients,
        )
        second_trunk_norm = _apply_owner(
            system.protected_second.model.trunk_named_parameters(),
            system.protected_second.trunk_optimizer,
            second_trunk_gradients,
        )
        first_trunk_norm = _apply_owner(
            system.protected_first.model.trunk_named_parameters(),
            system.protected_first.trunk_optimizer,
            first_trunk_gradients,
        )
        outcome_norm = _apply_owner(
            tuple(system.outcome_only.model.named_parameters()),
            system.outcome_only.outcome_optimizer,
            outcome_gradients,
        )
        for arm in (
            system.protected_second,
            system.protected_first,
            system.outcome_only,
        ):
            arm.updates += 1
        encoders_exact = _named_digest(
            system.protected_second.model.encoder_named_parameters()
        ) == _named_digest(system.protected_first.model.encoder_named_parameters())
        encoder_optimizers_exact = _optimizer_digest(
            system.protected_second.encoder_optimizer
        ) == _optimizer_digest(system.protected_first.encoder_optimizer)
        if not encoders_exact or not encoder_optimizers_exact:
            raise RuntimeError("V13 protected encoder weights/owner state diverged")
        diagnostics.append(
            {
                "update": update,
                "second_oml_objective": second_objective,
                "first_oml_objective": first_objective,
                "outcome_only_objective": outcome_objective,
                "structural_objective": second_structure,
                "second_encoder_gradient_norm": second_encoder_norm,
                "first_encoder_gradient_norm": first_encoder_norm,
                "second_trunk_gradient_norm": second_trunk_norm,
                "first_trunk_gradient_norm": first_trunk_norm,
                "outcome_only_gradient_norm": outcome_norm,
                "protected_encoders_exact": encoders_exact,
                "protected_encoder_optimizers_exact": encoder_optimizers_exact,
            }
        )
        if (update + 1) % 32 == 0:
            print(json.dumps({"v13_update": update + 1, "total": OUTER_UPDATES}), flush=True)
    if preflight is None or _model_digest(system.source.model) != source_before:
        raise RuntimeError("V13 preflight/source invariant failed")
    return {
        "updates": OUTER_UPDATES,
        "minimum_inner_exposure": min(uses),
        "maximum_inner_exposure": max(uses),
        "preflight": preflight,
        "diagnostics": diagnostics,
        "source_unchanged": True,
        "protected_encoders_exact": True,
        "protected_encoder_optimizers_exact": True,
    }


def _shuffled_outcomes(outcomes: torch.Tensor) -> torch.Tensor:
    if outcomes.shape != (6,) or int((outcomes == 1).sum()) != 3 or int((outcomes == -1).sum()) != 3:
        raise ValueError("V13 shuffled feedback requires one balanced six-label stream")
    shuffled = outcomes[list(SHUFFLE_PERMUTATION)]
    if torch.equal(shuffled, outcomes) or sorted(shuffled.tolist()) != sorted(outcomes.tolist()):
        raise RuntimeError("V13 frozen feedback permutation is invalid")
    return shuffled


def _balanced_accuracy(logits: torch.Tensor, outcomes: torch.Tensor) -> float:
    predicted = torch.where(logits >= 0, 1.0, -1.0)
    positive = outcomes == 1
    negative = outcomes == -1
    return float(
        (0.5 * ((predicted[positive] == 1).float().mean() + (predicted[negative] == -1).float().mean())).item()
    )


def _panel_pairs(
    rows: Sequence[EncodedPairMechanism],
) -> tuple[tuple[tuple[EncodedPairMechanism, EncodedPairMechanism], ...], ...]:
    by_family: dict[str, list[EncodedPairMechanism]] = {}
    for row in rows:
        by_family.setdefault(row.generator_family, []).append(row)
    families = tuple(by_family)
    if len(families) != 12 or any(len(by_family[name]) != 4 for name in families):
        raise ValueError("V13 evaluation partition lost 12x4 family layout")
    panels = (
        tuple((by_family[name][0], by_family[name][1]) for name in families),
        tuple((by_family[name][2], by_family[name][3]) for name in families),
        tuple(
            (by_family[name][0], by_family[families[(index + 3) % 12]][2])
            for index, name in enumerate(families)
        ),
        tuple(
            (by_family[name][1], by_family[families[(index + 6) % 12]][3])
            for index, name in enumerate(families)
        ),
    )
    if any(left.mechanism_ref == right.mechanism_ref for panel in panels for left, right in panel):
        raise RuntimeError("V13 evaluation panel reuses one mechanism as support/probe")
    return panels


def _evaluate_pair(
    model: StructureProtectedOMLCore,
    support: EncodedPairMechanism,
    probe: EncodedPairMechanism,
    *,
    updates_enabled: bool,
    shuffle_feedback: bool = False,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    fast, state = _fresh_fast(model)
    probe_data = _batch((probe,), device)
    # Probe labels are intentionally not read until both logits exist.
    pre_logits = model.functional_logits(
        *probe_data[:-1],
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    support_outcomes = support.outcomes
    adaptation_loss = None
    if updates_enabled:
        if shuffle_feedback:
            support_outcomes = _shuffled_outcomes(support.outcomes)
        fast, state, adaptation_loss = _inner_step(
            model,
            fast,
            state,
            support,
            second_order=False,
            outcomes=support_outcomes,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
    post_logits = model.functional_logits(
        *probe_data[:-1],
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    probe_outcomes = probe_data[-1]
    pre_loss = F.softplus(-probe_outcomes * pre_logits).mean()
    post_loss = F.softplus(-probe_outcomes * post_logits).mean()
    predicted = torch.where(post_logits[0] >= 0, 1.0, -1.0)
    relations = probe.corruption_types
    same_length = torch.tensor(
        tuple(value in ("paraphrase", "reorder", "replacement") for value in relations),
        device=device,
    )
    reorder = torch.tensor(tuple(value == "reorder" for value in relations), device=device)
    same_length_predictions = predicted[same_length]
    same_length_outcomes = probe_outcomes[0, same_length]
    same_length_positive = same_length_outcomes == 1
    same_length_negative = same_length_outcomes == -1
    same_length_correct = same_length_predictions == same_length_outcomes
    reorder_correct = predicted[reorder] == probe_outcomes[0, reorder]
    return {
        "support_family": support.generator_family,
        "probe_family": probe.generator_family,
        "support_transition": support.transition_group,
        "probe_transition": probe.transition_group,
        "pre_loss": float(pre_loss.item()),
        "post_loss": float(post_loss.item()),
        "online_loss_auc": float((0.5 * (pre_loss + post_loss)).item()),
        "balanced_accuracy": _balanced_accuracy(post_logits[0], probe_outcomes[0]),
        "same_length_correct": int(same_length_correct.sum().item()),
        "same_length_count": int(same_length_correct.numel()),
        "same_length_positive_correct": int(
            (same_length_predictions[same_length_positive] == 1).sum().item()
        ),
        "same_length_positive_count": int(same_length_positive.sum().item()),
        "same_length_negative_correct": int(
            (same_length_predictions[same_length_negative] == -1).sum().item()
        ),
        "same_length_negative_count": int(same_length_negative.sum().item()),
        "reorder_correct": int(reorder_correct.sum().item()),
        "reorder_count": int(reorder_correct.numel()),
        "adaptation_loss": adaptation_loss,
        "fast_step": state[0].step,
        "support_reference_sha256": _tensor_digest(support.reference_features),
        "support_attempt_sha256": _tensor_digest(support.attempt_features),
        "support_mask_sha256": hashlib.sha256(
            (_tensor_digest(support.reference_mask) + _tensor_digest(support.attempt_mask)).encode()
        ).hexdigest(),
        "probe_outcome_sha256": _tensor_digest(probe.outcomes),
        "adaptation_outcome_sha256": _tensor_digest(support_outcomes),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if len(records) != 48:
        raise ValueError("V13 aggregate requires 48 panel records")
    positive_correct = sum(int(row["same_length_positive_correct"]) for row in records)
    positive_count = sum(int(row["same_length_positive_count"]) for row in records)
    negative_correct = sum(int(row["same_length_negative_correct"]) for row in records)
    negative_count = sum(int(row["same_length_negative_count"]) for row in records)
    reorder_correct = sum(int(row["reorder_correct"]) for row in records)
    reorder_count = sum(int(row["reorder_count"]) for row in records)
    return {
        "online_loss_auc": sum(float(row["online_loss_auc"]) for row in records) / 48.0,
        "balanced_accuracy": sum(float(row["balanced_accuracy"]) for row in records) / 48.0,
        "same_length_balanced_accuracy": 0.5 * (
            positive_correct / positive_count + negative_correct / negative_count
        ),
        "reorder_accuracy": reorder_correct / reorder_count if reorder_count else 0.0,
        "mean_pre_loss": sum(float(row["pre_loss"]) for row in records) / 48.0,
        "mean_post_loss": sum(float(row["post_loss"]) for row in records) / 48.0,
    }


def _retrieval_report(
    model: StructureProtectedOMLCore,
    retrieval: EncodedRetrievalSet,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    values = retrieval.to(device)
    duplicate = values.candidate_features[:, :1]
    duplicate_mask = values.candidate_mask[:, :1]
    attempts = torch.cat((values.candidate_features, duplicate), dim=1)
    attempt_mask = torch.cat((values.candidate_mask, duplicate_mask), dim=1)
    with torch.no_grad():
        references, candidates = model.encode_views(
            values.reference_features,
            values.reference_mask,
            attempts,
            attempt_mask,
        )
    reference = F.normalize(references[:, 0], dim=-1)
    candidate = F.normalize(candidates[:, :5], dim=-1)
    scores = torch.einsum("md,mcd->mc", reference, candidate)
    # Evaluator labels enter only after all candidate scores exist.
    targets = values.target_indices
    predicted = scores.argmax(dim=-1)
    correct = predicted == targets
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    margins: dict[str, list[float]] = {name: [] for name in CORRUPTIONS}
    reorder_wins = []
    for row, relations in enumerate(values.candidate_types):
        relation_to_index = {relation: index for index, relation in enumerate(relations)}
        for name in CORRUPTIONS:
            negative_score = scores[row, relation_to_index[name]]
            margins[name].append(float((target_scores[row] - negative_score).item()))
        reorder_wins.append(
            bool(target_scores[row] > scores[row, relation_to_index["reorder"]])
        )
    by_transition = {}
    for name in sorted(set(values.transition_groups)):
        indices = [index for index, value in enumerate(values.transition_groups) if value == name]
        by_transition[name] = {
            "correct": int(correct[indices].sum().item()),
            "count": len(indices),
            "accuracy": float(correct[indices].float().mean().item()),
        }
    by_renderer = {}
    for name in sorted(set(values.renderer_position_strata)):
        indices = [index for index, value in enumerate(values.renderer_position_strata) if value == name]
        by_renderer[name] = {
            "correct": int(correct[indices].sum().item()),
            "count": len(indices),
            "accuracy": float(correct[indices].float().mean().item()),
        }
    raw_codes = torch.cat((references[:, :1], candidates[:, :5]), dim=1).detach().cpu()
    geometry = representation_metrics(raw_codes)
    geometry.update(
        {
            "raw_code_shape": list(raw_codes.shape),
            "raw_code_sha256": _tensor_digest(raw_codes),
            "raw_code_tensor": raw_codes.tolist(),
        }
    )
    mean_margins = {name: sum(rows) / len(rows) for name, rows in margins.items()}
    return {
        "correct": int(correct.sum().item()),
        "count": len(correct),
        "accuracy": float(correct.float().mean().item()),
        "paired_order_win_rate": sum(reorder_wins) / len(reorder_wins),
        "mean_cosine_score_margin": mean_margins["reorder"],
        "corruption_distance_margins": mean_margins,
        "by_transition": by_transition,
        "by_renderer_position": by_renderer,
        "geometry": geometry,
        "score_sha256": _tensor_digest(scores),
    }


def _evaluate(
    system: V13System,
    rows: Sequence[EncodedPairMechanism],
    retrieval: EncodedRetrievalSet,
) -> dict[str, Any]:
    panels = _panel_pairs(rows)
    configurations = {
        ARM_PROTECTED_SECOND: (system.protected_second.model, True, False, True, True),
        ARM_PROTECTED_FIRST: (system.protected_first.model, True, False, True, True),
        ARM_PROTECTED_SECOND_NO: (system.protected_second.model, False, False, True, True),
        ARM_PROTECTED_FIRST_NO: (system.protected_first.model, False, False, True, True),
        ARM_OUTCOME_ONLY: (system.outcome_only.model, True, False, True, True),
        ARM_SOURCE: (system.source.model, True, False, True, True),
        ARM_SOURCE_NO: (system.source.model, False, False, True, True),
        ARM_SHUFFLED: (system.protected_second.model, True, True, True, True),
        ARM_DIRECTION_REMOVED: (system.protected_second.model, True, False, False, True),
        ARM_SEMANTICS_REMOVED: (system.protected_second.model, True, False, True, False),
    }
    records_by_arm: dict[str, list[dict[str, Any]]] = {
        name: [] for name in configurations
    }
    panel_reports = []
    with torch.enable_grad():
        for panel_index, pairs in enumerate(panels):
            arms = {}
            for name, (model, updates, shuffled, direction, semantics) in configurations.items():
                model.eval()
                records = [
                    _evaluate_pair(
                        model,
                        support,
                        probe,
                        updates_enabled=updates,
                        shuffle_feedback=shuffled,
                        include_direction=direction,
                        include_step_semantics=semantics,
                    )
                    for support, probe in pairs
                ]
                records_by_arm[name].extend(records)
                arm = {
                    "online_loss_auc": sum(row["online_loss_auc"] for row in records) / 12.0,
                    "balanced_accuracy": sum(row["balanced_accuracy"] for row in records) / 12.0,
                    "records": records,
                }
                arms[name] = arm
            panel_reports.append(
                {
                    "panel": panel_index,
                    "pairing": "same_transition" if panel_index < 2 else "cross_transition",
                    "arms": arms,
                }
            )
    aggregate = {name: _aggregate(records) for name, records in records_by_arm.items()}
    comparisons = []
    for panel in panel_reports:
        second = panel["arms"][ARM_PROTECTED_SECOND]["online_loss_auc"]
        first = panel["arms"][ARM_PROTECTED_FIRST]["online_loss_auc"]
        comparisons.append(
            {
                "panel": panel["panel"],
                "improved": second < first,
                "nonregressed": second <= first,
                "second_auc": second,
                "first_auc": first,
            }
        )

    family_improvements = {}
    behavior_family_success = {}
    transition_improved = {name: 0 for name in TRANSITIONS}
    behavior_transition_success = {name: 0 for name in TRANSITIONS}
    for family in sorted({row.generator_family for row in rows}):
        second = [
            row for row in records_by_arm[ARM_PROTECTED_SECOND]
            if row["probe_family"] == family
        ]
        source = [
            row for row in records_by_arm[ARM_SOURCE]
            if row["probe_family"] == family
        ]
        transition = second[0]["probe_transition"]
        improved = (
            sum(row["online_loss_auc"] for row in second) / len(second)
            < sum(row["online_loss_auc"] for row in source) / len(source)
        )
        positive_correct = sum(row["same_length_positive_correct"] for row in second)
        positive_count = sum(row["same_length_positive_count"] for row in second)
        negative_correct = sum(row["same_length_negative_correct"] for row in second)
        negative_count = sum(row["same_length_negative_count"] for row in second)
        balanced = 0.5 * (
            positive_correct / positive_count + negative_correct / negative_count
        )
        behavior = balanced >= 0.65
        family_improvements[family] = {"improved": improved, "transition": transition}
        behavior_family_success[family] = {
            "succeeded": behavior,
            "transition": transition,
            "same_length_balanced_accuracy": balanced,
        }
        transition_improved[transition] += int(improved)
        behavior_transition_success[transition] += int(behavior)

    full_records = records_by_arm[ARM_PROTECTED_SECOND]
    shuffled_records = records_by_arm[ARM_SHUFFLED]
    retrieval_reports = {
        "protected": _retrieval_report(system.protected_second.model, retrieval),
        "outcome_only": _retrieval_report(system.outcome_only.model, retrieval),
        "source": _retrieval_report(system.source.model, retrieval),
    }
    return {
        "mechanism_count": len(rows),
        "arms": aggregate,
        "panels": panel_reports,
        "panel_comparisons": comparisons,
        "improved_panel_count": sum(int(value["improved"]) for value in comparisons),
        "all_panels_nonregressed": all(value["nonregressed"] for value in comparisons),
        "family_improvements": family_improvements,
        "improved_family_count": sum(int(value["improved"]) for value in family_improvements.values()),
        "transition_improved_family_counts": transition_improved,
        "behavior_family_success": behavior_family_success,
        "behavior_family_success_count": sum(
            int(value["succeeded"]) for value in behavior_family_success.values()
        ),
        "behavior_transition_success_counts": behavior_transition_success,
        "retrieval": retrieval_reports,
        "length_only_baseline_balanced_accuracy": 0.50,
        "feedback_integrity": {
            "permutation": list(SHUFFLE_PERMUTATION),
            "same_support_inputs": all(
                left["support_reference_sha256"] == right["support_reference_sha256"]
                and left["support_attempt_sha256"] == right["support_attempt_sha256"]
                and left["support_mask_sha256"] == right["support_mask_sha256"]
                for left, right in zip(full_records, shuffled_records, strict=True)
            ),
            "same_probe_labels": all(
                left["probe_outcome_sha256"] == right["probe_outcome_sha256"]
                for left, right in zip(full_records, shuffled_records, strict=True)
            ),
            "all_support_associations_changed": all(
                left["adaptation_outcome_sha256"] != right["adaptation_outcome_sha256"]
                for left, right in zip(full_records, shuffled_records, strict=True)
            ),
            "balanced_multiset_preserved": True,
        },
    }


def _corpus_audit(
    inner: Sequence[Any], outer: Sequence[Any], development: Sequence[Any]
) -> dict[str, Any]:
    def texts(mechanisms: Sequence[Any]) -> set[str]:
        values = set()
        for mechanism in mechanisms:
            for episode in mechanism.public.episodes:
                values.add(episode.reference_trace_text)
                values.add(episode.attempt_trace_text)
            contrast = mechanism.structural_contrasts
            values.add(contrast.reference_trace_text)
            values.update(contrast.serialized_candidate_attempt_trace_texts)
        return values

    surfaces = {"inner": texts(inner), "outer": texts(outer), "development": texts(development)}
    length_table: dict[int, dict[str, int]] = {}
    scope_counterbalance: dict[str, dict[str, Any]] = {}
    learner_boundary = True
    for scope, mechanisms in (
        ("inner", inner), ("outer", outer), ("development", development)
    ):
        reference_plans: dict[int, int] = {}
        attempt_plans: dict[int, int] = {}
        structural_reference: dict[int, int] = {}
        structural_attempt: dict[int, int] = {}
        target_positions: dict[int, int] = {}
        for mechanism in mechanisms:
            learner_boundary = learner_boundary and set(mechanism.to_learner_payload()) == {"episodes"}
            for episode in mechanism.public.episodes:
                length = len(parse_step_trace(episode.attempt_trace_text))
                bucket = length_table.setdefault(length, {"SUCCESS": 0, "FAILURE": 0})
                bucket[episode.outcome_label] += 1
            for value in mechanism.metadata.episode_reference_plan_ids:
                reference_plans[value] = reference_plans.get(value, 0) + 1
            for value in mechanism.metadata.episode_attempt_plan_ids:
                attempt_plans[value] = attempt_plans.get(value, 0) + 1
            structural_reference[mechanism.metadata.structural_reference_plan_id] = (
                structural_reference.get(mechanism.metadata.structural_reference_plan_id, 0) + 1
            )
            structural_attempt[mechanism.metadata.structural_attempt_plan_id] = (
                structural_attempt.get(mechanism.metadata.structural_attempt_plan_id, 0) + 1
            )
            target = mechanism.supervision.same_order_candidate_index
            target_positions[target] = target_positions.get(target, 0) + 1
        scope_counterbalance[scope] = {
            "all_reference_plans_shared": set(reference_plans) == set(range(8)),
            "all_attempt_plans_shared": set(attempt_plans) == set(range(8)),
            "reference_plans_balanced": len(set(reference_plans.values())) == 1,
            "structural_reference_balanced": (
                set(structural_reference) == set(range(8))
                and len(set(structural_reference.values())) == 1
            ),
            "structural_attempt_balanced": (
                set(structural_attempt) == set(range(8))
                and len(set(structural_attempt.values())) == 1
            ),
            "candidate_positions_counterbalanced": (
                set(target_positions) == set(range(5))
                and max(target_positions.values()) - min(target_positions.values()) <= 1
            ),
            "target_position_counts": target_positions,
        }
    report = {
        "inner_outer_overlap": len(surfaces["inner"] & surfaces["outer"]),
        "inner_development_overlap": len(surfaces["inner"] & surfaces["development"]),
        "outer_development_overlap": len(surfaces["outer"] & surfaces["development"]),
        "outer_unique_mechanisms": len({value.metadata.mechanism_ref for value in outer}),
        "length_outcome_table": length_table,
        "length_outcome_exact": all(value["SUCCESS"] == value["FAILURE"] for value in length_table.values()),
        "scope_counterbalance": scope_counterbalance,
        "learner_boundary_exact": learner_boundary,
    }
    report["passed"] = bool(
        report["inner_outer_overlap"] == 0
        and report["inner_development_overlap"] == 0
        and report["outer_development_overlap"] == 0
        and report["outer_unique_mechanisms"] == TRAIN_OUTER_MECHANISMS
        and report["length_outcome_exact"] is True
        and all(
            all(value for key, value in scope.items() if key != "target_position_counts")
            for scope in scope_counterbalance.values()
        )
        and report["learner_boundary_exact"] is True
    )
    if not report["passed"]:
        raise RuntimeError("V13 corpus integrity/counterbalance audit failed")
    return report


def _padding_repeat_invariance(
    model: StructureProtectedOMLCore, row: EncodedPairMechanism
) -> dict[str, bool]:
    device = next(model.parameters()).device
    data = _batch((row,), device)
    with torch.no_grad():
        original = model.encode_views(*data[:-1])
        repeated = model.encode_views(*(value.clone() for value in data[:-1]))
        reference_features = torch.cat(
            (data[0], torch.zeros((*data[0].shape[:-2], 2, data[0].shape[-1]), device=device)),
            dim=2,
        )
        reference_mask = torch.cat(
            (data[1], torch.zeros((*data[1].shape[:-1], 2), device=device, dtype=torch.bool)),
            dim=2,
        )
        attempt_features = torch.cat(
            (data[2], torch.zeros((*data[2].shape[:-2], 2, data[2].shape[-1]), device=device)),
            dim=2,
        )
        attempt_mask = torch.cat(
            (data[3], torch.zeros((*data[3].shape[:-1], 2), device=device, dtype=torch.bool)),
            dim=2,
        )
        padded = model.encode_views(
            reference_features, reference_mask, attempt_features, attempt_mask
        )
    return {
        "repeat_exact": all(torch.equal(left, right) for left, right in zip(original, repeated)),
        "padding_exact": all(torch.equal(left, right) for left, right in zip(original, padded)),
    }


def _mean_hard_margin(report: Mapping[str, Any]) -> float:
    margins = report["corruption_distance_margins"]
    return sum(float(margins[name]) for name in CORRUPTIONS) / len(CORRUPTIONS)


def _structural_gate(metrics: Mapping[str, Any]) -> bool:
    try:
        retrieval = metrics["retrieval"]
        protected = retrieval["protected"]
        outcome = retrieval["outcome_only"]
        source = retrieval["source"]
        geometry = protected["geometry"]
        full = metrics["arms"][ARM_PROTECTED_SECOND]
        return bool(
            protected["correct"] >= 40
            and protected["count"] == 48
            and set(protected["by_transition"]) == set(TRANSITIONS)
            and all(value["correct"] >= 9 and value["count"] == 12 for value in protected["by_transition"].values())
            and all(float(value["accuracy"]) >= 0.75 for value in protected["by_renderer_position"].values())
            and float(protected["paired_order_win_rate"]) >= 0.75
            and float(protected["mean_cosine_score_margin"]) > 0.0
            and all(float(protected["corruption_distance_margins"][name]) >= 0.05 for name in CORRUPTIONS)
            and float(full["same_length_balanced_accuracy"]) >= 0.65
            and float(metrics["length_only_baseline_balanced_accuracy"]) == 0.50
            and float(full["same_length_balanced_accuracy"]) >= float(metrics["length_only_baseline_balanced_accuracy"]) + 0.10
            and metrics["behavior_family_success_count"] >= 9
            and all(int(value) >= 2 for value in metrics["behavior_transition_success_counts"].values())
            and float(protected["accuracy"]) >= float(outcome["accuracy"]) + 0.20
            and float(protected["accuracy"]) >= float(source["accuracy"]) + 0.20
            and _mean_hard_margin(protected) >= _mean_hard_margin(outcome) + 0.05
            and _mean_hard_margin(protected) >= _mean_hard_margin(source) + 0.05
            and float(geometry["effective_rank"]) >= 8.0
            and geometry["finite_nonzero_variance_dimensions"] == 32
            and float(geometry["minimum_dimension_variance"]) > 0.0
            and float(geometry["mean_off_diagonal_cosine"]) <= 0.95
            and float(geometry["fraction_distinct_pairs_above_0_999"]) <= 0.10
        )
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _oml_gate(metrics: Mapping[str, Any]) -> bool:
    try:
        arms = metrics["arms"]
        second = arms[ARM_PROTECTED_SECOND]
        first = arms[ARM_PROTECTED_FIRST]
        second_no = arms[ARM_PROTECTED_SECOND_NO]
        source = arms[ARM_SOURCE]
        source_no = arms[ARM_SOURCE_NO]
        shuffled = arms[ARM_SHUFFLED]
        direction = arms[ARM_DIRECTION_REMOVED]
        semantics = arms[ARM_SEMANTICS_REMOVED]
        second_auc = float(second["online_loss_auc"])
        feedback_credit = bool(
            float(shuffled["online_loss_auc"]) - second_auc >= 0.005
            or float(second["balanced_accuracy"]) >= float(shuffled["balanced_accuracy"]) + 0.05
        )
        semantics_credit = bool(
            float(second["same_length_balanced_accuracy"])
            >= float(semantics["same_length_balanced_accuracy"]) + 0.05
            or float(semantics["online_loss_auc"]) - second_auc >= 0.05
        )
        direction_credit = bool(
            float(second["reorder_accuracy"]) >= float(direction["reorder_accuracy"]) + 0.05
            or float(direction["online_loss_auc"]) - second_auc >= 0.05
        )
        return bool(
            float(second_no["balanced_accuracy"]) >= 0.75
            and float(second_no["balanced_accuracy"]) >= float(source_no["balanced_accuracy"]) + 0.10
            and second_auc < float(first["online_loss_auc"])
            and second_auc <= 0.95 * float(first["online_loss_auc"])
            and second_auc < float(source["online_loss_auc"])
            and second_auc < float(second_no["online_loss_auc"])
            and metrics["improved_panel_count"] >= 3
            and metrics["all_panels_nonregressed"] is True
            and feedback_credit
            and semantics_credit
            and direction_credit
            and metrics["improved_family_count"] >= 9
            and set(metrics["transition_improved_family_counts"]) == set(TRANSITIONS)
            and all(int(value) >= 2 for value in metrics["transition_improved_family_counts"].values())
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _development_gate(metrics: Mapping[str, Any], identity_exact: bool) -> bool:
    try:
        protocol = metrics["protocol_invariants"]
        preflight = protocol["preflight"]
        full_split = (
            preflight["protected_second_full_split"],
            preflight["protected_first_full_split"],
            preflight["outcome_control_full_split"],
        )
        return bool(
            identity_exact is True
            and _structural_gate(metrics)
            and _oml_gate(metrics)
            and protocol["corpus_audit"]["passed"] is True
            and protocol["protected_encoders_exact"] is True
            and protocol["protected_encoder_optimizers_exact"] is True
            and protocol["source_unchanged"] is True
            and protocol["final_sealed"] is True
            and protocol["ownership_exact"] is True
            and protocol["padding_repeat"]["padding_exact"] is True
            and protocol["padding_repeat"]["repeat_exact"] is True
            and all(preflight["connectivity"].values())
            and preflight["structural_encoder_gradient_nonzero"] is True
            and preflight["direct_outer_trunk_gradient_nonzero"] is True
            and preflight["structure_cross_owner_exact"] is True
            and preflight["trunk_cross_owner_exact"] is True
            and preflight["infonce_serialization_invariant"] is True
            and all(
                float(item[field]) <= 1.0e-6
                for item in full_split
                for field in ("objective_absolute_delta", "maximum_gradient_absolute_delta")
            )
            and metrics["feedback_integrity"]["same_support_inputs"] is True
            and metrics["feedback_integrity"]["same_probe_labels"] is True
            and metrics["feedback_integrity"]["all_support_associations_changed"] is True
            and metrics["feedback_integrity"]["balanced_multiset_preserved"] is True
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _classification(metrics: Mapping[str, Any], *, authorized: bool) -> str:
    if authorized:
        return "DEVELOPMENT_GATE_PASSED"
    if _structural_gate(metrics):
        return "STRUCTURAL_REPRESENTATION_SUPPORTED_NOT_INTEGRATED"
    if _oml_gate(metrics):
        return "DEVELOPMENT_SHORTCUT_CODE_COLLAPSE"
    return "DEVELOPMENT_NOT_SUPPORTED"


def _identity_report(
    system: V13System,
    *,
    qwen_before: str,
    qwen_after: str,
    sources_before: Mapping[str, str],
    sources_after: Mapping[str, str],
    model_digests_before_evaluation: Mapping[str, str],
) -> dict[str, Any]:
    current = {
        "protected_second": _model_digest(system.protected_second.model),
        "protected_first": _model_digest(system.protected_first.model),
        "outcome_only": _model_digest(system.outcome_only.model),
        "source": _model_digest(system.source.model),
    }
    report = {
        "qwen_before": qwen_before,
        "qwen_after": qwen_after,
        "sources_before": dict(sources_before),
        "sources_after": dict(sources_after),
        "models_before_evaluation": dict(model_digests_before_evaluation),
        "models_after_evaluation": current,
        "protected_encoder_digest_second": _named_digest(
            system.protected_second.model.encoder_named_parameters()
        ),
        "protected_encoder_digest_first": _named_digest(
            system.protected_first.model.encoder_named_parameters()
        ),
    }
    report["exact"] = bool(
        qwen_before == qwen_after
        and sources_before == sources_after
        and dict(model_digests_before_evaluation) == current
        and report["protected_encoder_digest_second"] == report["protected_encoder_digest_first"]
    )
    return report


def _checkpoint_payload(
    system: V13System,
    *,
    step_width: int,
    qwen_digest: str,
    sources: Mapping[str, str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "completed_updates": OUTER_UPDATES,
        "step_width": step_width,
        "qwen_digest": qwen_digest,
        "source_hashes": dict(sources),
        "development_metrics": dict(metrics),
        "initial_digest": system.initial_digest,
        "protected_second_state": {
            name: value.detach().cpu()
            for name, value in system.protected_second.model.state_dict().items()
        },
        "protected_first_state": {
            name: value.detach().cpu()
            for name, value in system.protected_first.model.state_dict().items()
        },
        "outcome_only_state": {
            name: value.detach().cpu()
            for name, value in system.outcome_only.model.state_dict().items()
        },
        "source_state": {
            name: value.detach().cpu() for name, value in system.source.model.state_dict().items()
        },
        "protected_second_encoder_optimizer": system.protected_second.encoder_optimizer.state_dict(),
        "protected_first_encoder_optimizer": system.protected_first.encoder_optimizer.state_dict(),
        "protected_second_trunk_optimizer": system.protected_second.trunk_optimizer.state_dict(),
        "protected_first_trunk_optimizer": system.protected_first.trunk_optimizer.state_dict(),
        "outcome_only_optimizer": system.outcome_only.outcome_optimizer.state_dict(),
        "model_digests": {
            "protected_second": _model_digest(system.protected_second.model),
            "protected_first": _model_digest(system.protected_first.model),
            "outcome_only": _model_digest(system.outcome_only.model),
            "source": _model_digest(system.source.model),
        },
    }


def _validate_final_authorization(
    development: Mapping[str, Any], checkpoint_path: Path
) -> None:
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "DEVELOPMENT_GATE_PASSED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and development.get("identity_checks", {}).get("exact") is True
        and isinstance(development.get("development_metrics"), dict)
        and _development_gate(development["development_metrics"], True)
    )
    v5._validate_final_authorization(valid)


def _load_checkpoint_system(
    checkpoint: Mapping[str, Any], device: torch.device
) -> V13System:
    system = _build_system(int(checkpoint["step_width"]), device)
    system.protected_second.model.load_state_dict(checkpoint["protected_second_state"], strict=True)
    system.protected_first.model.load_state_dict(checkpoint["protected_first_state"], strict=True)
    system.outcome_only.model.load_state_dict(checkpoint["outcome_only_state"], strict=True)
    system.source.model.load_state_dict(checkpoint["source_state"], strict=True)
    for arm in (system.protected_second, system.protected_first, system.outcome_only):
        arm.updates = OUTER_UPDATES
    current = {
        "protected_second": _model_digest(system.protected_second.model),
        "protected_first": _model_digest(system.protected_first.model),
        "outcome_only": _model_digest(system.outcome_only.model),
        "source": _model_digest(system.source.model),
    }
    if current != checkpoint["model_digests"]:
        raise RuntimeError("V13 checkpoint model digest mismatch")
    return system


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V13 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    corpus = build_structure_protected_oml_natural_trace_v13()
    if corpus.final_is_open:
        raise RuntimeError("V13 final opened before development")
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V13 final seal failed open")
    outer_mechanisms = tuple(
        corpus.train_outer(update, slot)
        for update in range(TRAIN_OUTER_UPDATES)
        for slot in range(TRAIN_OUTER_SLOTS)
    )
    corpus_audit = _corpus_audit(corpus.train_inner, outer_mechanisms, corpus.development)
    index = {
        mechanism.metadata.mechanism_ref: position
        for position, mechanism in enumerate(corpus.train_inner)
    }
    schedule = tuple(
        tuple(index[value.metadata.mechanism_ref] for value in corpus.train_inner_for_update(update))
        for update in range(OUTER_UPDATES)
    )
    inner_rows = _encode_mechanisms(qwen, corpus.train_inner)
    outer_rows = _encode_mechanisms(qwen, outer_mechanisms)
    development_rows = _encode_mechanisms(qwen, corpus.development)
    development_retrieval = _encode_retrieval_sets(qwen, corpus.development)
    step_width = int(inner_rows[0].reference_features.shape[-1])
    system = _build_system(step_width, device)
    partition = system.protected_second.model.parameter_partition_report()
    ownership_exact = bool(
        partition["partitions_disjoint"] is True
        and partition["all_trainable_parameters_owned"] is True
        and partition["fast_initial_outer_owned"] is False
        and partition["outcome_encoder_inputs"] is False
        and partition["metadata_encoder_inputs"] is False
    )
    training = _fit(system, inner_rows, outer_rows, schedule)
    before_evaluation = {
        "protected_second": _model_digest(system.protected_second.model),
        "protected_first": _model_digest(system.protected_first.model),
        "outcome_only": _model_digest(system.outcome_only.model),
        "source": _model_digest(system.source.model),
    }
    metrics = _evaluate(system, development_rows, development_retrieval)
    metrics["protocol_invariants"] = {
        "preflight": training["preflight"],
        "corpus_audit": corpus_audit,
        "protected_encoders_exact": training["protected_encoders_exact"],
        "protected_encoder_optimizers_exact": training["protected_encoder_optimizers_exact"],
        "source_unchanged": training["source_unchanged"],
        "final_sealed": True,
        "ownership_exact": ownership_exact,
        "parameter_partition": partition,
        "padding_repeat": _padding_repeat_invariance(
            system.protected_second.model, development_rows[0]
        ),
    }
    qwen_after = foundation_tensor_digest(qwen.model)
    sources_after = _source_hashes()
    identity = _identity_report(
        system,
        qwen_before=qwen_before,
        qwen_after=qwen_after,
        sources_before=sources_before,
        sources_after=sources_after,
        model_digests_before_evaluation=before_evaluation,
    )
    authorized = _development_gate(metrics, bool(identity["exact"]))
    checkpoint = _checkpoint_payload(
        system,
        step_width=step_width,
        qwen_digest=qwen_before,
        sources=sources_before,
        metrics=metrics,
    )
    result = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": _classification(metrics, authorized=authorized),
        "development_authorized": authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "updates": OUTER_UPDATES,
            "inner_steps": INNER_STEPS,
            "outer_mechanisms_per_update": OUTER_MECHANISMS,
            "train_inner_mechanisms": TRAIN_INNER_MECHANISMS,
            "train_outer_mechanisms": TRAIN_OUTER_MECHANISMS,
            "development_mechanisms": DEVELOPMENT_MECHANISMS,
            "final_mechanisms_opened": 0,
        },
        "training": training,
        "development_metrics": metrics,
        "identity_checks": identity,
        "source_hashes": sources_before,
        "checkpoint": str(checkpoint_path),
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Generated public reference-to-attempt software procedures only; this is not "
            "unrestricted reasoning, AGI, deployment, or human-affecting authority."
        ),
    }
    if sources_before != sources_after:
        raise RuntimeError("V13 executable source changed during run")
    _atomic_torch(checkpoint_path, checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "development_result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_final(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V13 final identity is already consumed")
    sources_before = _source_hashes()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != sources_before
        or checkpoint.get("completed_updates") != OUTER_UPDATES
        or checkpoint.get("development_metrics") != development.get("development_metrics")
    ):
        raise RuntimeError("V13 checkpoint/development binding mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != checkpoint["qwen_digest"]:
        raise RuntimeError("V13 frozen Qwen digest changed")
    corpus = build_structure_protected_oml_natural_trace_v13(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_MECHANISMS:
        raise RuntimeError("V13 authorized final partition is incomplete")
    rows = _encode_mechanisms(qwen, corpus.final)
    retrieval = _encode_retrieval_sets(qwen, corpus.final)
    system = _load_checkpoint_system(checkpoint, device)
    before_evaluation = {
        "protected_second": _model_digest(system.protected_second.model),
        "protected_first": _model_digest(system.protected_first.model),
        "outcome_only": _model_digest(system.outcome_only.model),
        "source": _model_digest(system.source.model),
    }
    metrics = _evaluate(system, rows, retrieval)
    metrics["protocol_invariants"] = copy.deepcopy(
        checkpoint["development_metrics"]["protocol_invariants"]
    )
    qwen_after = foundation_tensor_digest(qwen.model)
    sources_after = _source_hashes()
    identity = _identity_report(
        system,
        qwen_before=qwen_before,
        qwen_after=qwen_after,
        sources_before=sources_before,
        sources_after=sources_after,
        model_digests_before_evaluation=before_evaluation,
    )
    supported = _development_gate(metrics, bool(identity["exact"]))
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": metrics,
        "identity_checks": identity,
        "source_hashes": sources_before,
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    if sources_before != sources_after:
        raise RuntimeError("V13 executable source changed during final")
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "final_result": str(result_path),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/structure-protected-oml-natural-trace-v13-final.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
