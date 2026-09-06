"""V16 strict key/value procedural-credit experiment.

The runner deliberately reuses V15's corpus, chronology, losses, controls,
and thresholds.  Its only learned change is the strict memory core: public
structure forms addresses, revealed outcome forms stored value, and retrieved
value alone forms the residual.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

from angler.reasoning.strict_key_value_credit_memory import (
    StrictKeyValueCreditEvent,
    StrictKeyValueCreditMemoryCore,
    StrictKeyValueCreditState,
)
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
from experiments.corpora.structure_protected_oml_natural_trace_v13 import (
    FinalPartitionSealedError,
)
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners import paired_latent_contingency_credit_v15 as v15


IDENTITY = "angler.strict-key-value-credit.v16-first-result"
SEED = 2026083116
TEMPORAL_WIDTH = v15.TEMPORAL_WIDTH
V15_RESULT_SHA256 = "E2DBEE07C91189736382CB2D3EBCBE1162653F19C84336894C9F5F2FD1F6C0C3"
V15_CHECKPOINT_SHA256 = "DE63B238125C307ED745FD3998362CE845A47875A199A70FB2B7146FC9C4F5B1"
QUERY_SWAP = (3, 2, 5, 4)


def _sha256(path: str | Path) -> str:
    return v15._sha256(path).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v15._model_digest(model).upper()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inherited = {f"v15_{key}": value.upper() for key, value in v15._source_hashes().items()}
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "strict_core": _sha256(root / "src/angler/reasoning/strict_key_value_credit_memory.py"),
        **inherited,
    }


def _prepare_angler_device(value: str) -> tuple[torch.device, float]:
    device = torch.device(value)
    if device != torch.device("cuda:1"):
        raise RuntimeError("V16 requires V13/V16 on cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device, time.perf_counter()


def _strict_event(
    core: StrictKeyValueCreditMemoryCore,
    state: StrictKeyValueCreditState,
    output: Any,
    outcomes: torch.Tensor,
    *,
    evidence_ref: str,
    shared_value: bool,
) -> tuple[StrictKeyValueCreditState, StrictKeyValueCreditEvent]:
    core.validate_state(state)
    core._validate_output(output)
    core._validate_outcomes(outcomes, output.logits)
    if output.logits.shape != (1,) or output.state_step != state.step:
        raise ValueError("V16 feedback output/state chronology is invalid")
    if state.step >= core.memory_slots:
        raise OverflowError("V16 strict memory capacity is exhausted")
    if shared_value:
        signs = outcomes.new_tensor((-1.0, 1.0))
        write_value = core.outcome_values(signs).mean(dim=0, keepdim=True)
    else:
        write_value = core.outcome_values(outcomes)
    allocation = core._least_unused_allocation(state.usage)
    slot_weights = output.write_strengths[0] * allocation
    slot = slot_weights.unsqueeze(-1)
    write_key = output.write_keys[0].unsqueeze(0)
    value = write_value[0].unsqueeze(0)
    keys = (1.0 - slot) * state.keys + slot * write_key
    erased = state.values * (1.0 - slot * output.erase_gates[0])
    values = (1.0 - slot) * erased + slot * value
    usage = state.usage + slot_weights * (1.0 - state.usage)
    coordinate = state.usage.new_tensor((state.step + 1) / core.memory_slots)
    acquisition = (1.0 - slot_weights) * state.acquisition + slot_weights * coordinate
    updated = StrictKeyValueCreditState(keys, values, usage, acquisition, state.step + 1)
    core.validate_state(updated)
    event = StrictKeyValueCreditEvent(
        relation_features=output.observed_relations.detach().cpu().clone(),
        temporal_features=output.observed_temporal.detach().cpu().clone(),
        base_logits=output.base_logits.detach().cpu().clone(),
        outcomes=outcomes.detach().cpu().clone(),
        evidence_refs=(evidence_ref,),
        allocation_index=int(allocation.argmax().item()),
        read_strength=float(output.read_strengths[0].detach().item()),
        write_strength=float(output.write_strengths[0].detach().item()),
        erase_gate=float(output.erase_gates[0].detach().item()),
        write_key=write_key.detach().cpu().clone(),
        write_value=value.detach().cpu().clone(),
    )
    event._validate_record_fields()
    return updated.detached_clone(), event


def _apply_outcome_blind_feedback(
    core: StrictKeyValueCreditMemoryCore,
    state: StrictKeyValueCreditState,
    output: Any,
    outcomes: torch.Tensor,
    *,
    evidence_ref: str,
    routing_output: Any | None = None,
) -> tuple[StrictKeyValueCreditState, StrictKeyValueCreditEvent]:
    if routing_output is not None:
        core._validate_output(routing_output)
        if not all(
            torch.equal(getattr(output, name), getattr(routing_output, name))
            for name in ("write_keys", "read_strengths", "write_strengths", "erase_gates")
        ):
            raise RuntimeError("V16 routing reference changed strict address/gates")
    return _strict_event(
        core, state, output, outcomes,
        evidence_ref=evidence_ref, shared_value=True,
    )


def _write_blind_anchors(
    core: StrictKeyValueCreditMemoryCore,
    episode: Any,
    state: StrictKeyValueCreditState,
    routing_state: StrictKeyValueCreditState,
) -> tuple[
    StrictKeyValueCreditState,
    StrictKeyValueCreditState,
    list[StrictKeyValueCreditEvent],
    list[int],
    list[float],
]:
    events: list[StrictKeyValueCreditEvent] = []
    steps: list[int] = []
    logits: list[float] = []
    for index in ACQUISITION_INDICES:
        if state.step != routing_state.step:
            raise RuntimeError("V16 blind/routing state chronology diverged")
        steps.append(state.step)
        output = core.predict(
            episode.relation_features[index:index + 1],
            episode.temporal_features[index:index + 1],
            episode.base_logits[index:index + 1],
            state=state,
        )
        routing_output = core.predict(
            episode.relation_features[index:index + 1],
            episode.temporal_features[index:index + 1],
            episode.base_logits[index:index + 1],
            state=routing_state,
        )
        feedback = episode.outcomes[index:index + 1]
        ref = f"{episode.pair_ref}:{episode.twin_ref}:anchor:{index}"
        state, event = _apply_outcome_blind_feedback(
            core, state, output, feedback, evidence_ref=ref,
            routing_output=routing_output,
        )
        routing_state, _ = core.apply_feedback(
            routing_state, routing_output, feedback,
            evidence_refs=f"{ref}:routing-reference", detach_state=True,
        )
        logits.append(float(output.logits.detach().item()))
        events.append(event)
    return state, routing_state, events, steps, logits


@contextmanager
def _v15_control_scope() -> Iterator[None]:
    """Bind V15's frozen orchestration to V16 state/intervention types."""
    bindings = {
        "StructureKeyedCreditState": StrictKeyValueCreditState,
        "StructureKeyedCreditEvent": StrictKeyValueCreditEvent,
        "_apply_outcome_blind_feedback": _apply_outcome_blind_feedback,
        "_write_blind_anchors": _write_blind_anchors,
    }
    originals = {name: getattr(v15, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(v15, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(v15, name, value)


def _relation_query_swap_control(
    core: StrictKeyValueCreditMemoryCore,
    pairs: Sequence[Any],
) -> dict[str, Any]:
    if len(pairs) != DEVELOPMENT_PAIRS:
        raise ValueError("V16 relation-query lesion requires 48 development pairs")
    device = next(core.parameters()).device
    with _v15_control_scope():
        true = v15._arm_stream(core, pairs, arm="true")
    logits: list[float] = []
    outcomes: list[float] = []
    losses: list[float] = []
    relation_inputs_changed = 0
    changed = 0
    state_exact = True
    nonrelation_exact = True
    relation_source_exact = True
    opposite_relation_class_exact = True
    read_strength_exact = True
    with torch.no_grad():
        for pair_index, pair in enumerate(pairs):
            for side in ("first", "second"):
                episode = getattr(pair, side).to(device)
                state = true["states_after_pair"][pair_index][side]
                before = core.state_digest(state)
                for target, source in zip(PROBE_INDICES, QUERY_SWAP, strict=True):
                    target_temporal = episode.temporal_features[target:target + 1].clone()
                    target_base = episode.base_logits[target:target + 1].clone()
                    target_label = episode.outcomes[target:target + 1].clone()
                    source_relation = episode.relation_features[source:source + 1].clone()
                    relation_inputs_changed += int(not torch.equal(
                        episode.relation_features[target], episode.relation_features[source]
                    ))
                    opposite_relation_class_exact &= (
                        {episode.relation_classes[target], episode.relation_classes[source]}
                        == {"alpha", "beta"}
                    )
                    target_output = core.predict(
                        episode.relation_features[target:target + 1],
                        target_temporal,
                        target_base,
                        state=state,
                    )
                    source_output = core.predict(
                        source_relation,
                        target_temporal,
                        target_base,
                        state=state,
                    )
                    relation_source_exact &= torch.equal(
                        source_output.observed_relations, source_relation
                    )
                    changed += int(not torch.equal(
                        target_output.read_queries, source_output.read_queries
                    ))
                    target_strength = target_output.read_strengths.detach().clone()
                    read_weights = core._content_read_weights(
                        source_output.read_queries, target_strength, state,
                    )
                    read_values = read_weights @ state.values
                    residual = core.decode_retrieved_values(read_values)
                    hybrid_logits = target_base + residual
                    read_strength_exact &= torch.equal(
                        target_strength, target_output.read_strengths
                    )
                    nonrelation_exact &= (
                        torch.equal(target_output.observed_temporal, target_temporal)
                        and torch.equal(source_output.observed_temporal, target_temporal)
                        and torch.equal(target_output.base_logits, target_base)
                        and torch.equal(target_label, episode.outcomes[target:target + 1])
                        and torch.equal(target_temporal, episode.temporal_features[target:target + 1])
                        and torch.equal(target_base, episode.base_logits[target:target + 1])
                    )
                    logits.append(float(hybrid_logits.item()))
                    outcomes.append(float(target_label.item()))
                    losses.append(float(torch.nn.functional.softplus(
                        -target_label * hybrid_logits
                    ).mean().item()))
                state_exact &= before == core.state_digest(state)
    return {
        "balanced_accuracy": v15._balanced_accuracy(logits, outcomes),
        "mean_probe_nll": sum(losses) / len(losses),
        "probe_count": len(logits),
        "relation_inputs_changed": relation_inputs_changed == len(logits),
        "relation_queries_changed": changed == len(logits),
        "read_strength_exact": read_strength_exact,
        "relation_source_exact": relation_source_exact,
        "opposite_relation_class_exact": opposite_relation_class_exact,
        "temporal_base_labels_exact": nonrelation_exact,
        "state_and_stored_memory_exact": state_exact,
        "only_read_query_swapped": bool(
            relation_inputs_changed == len(logits)
            and changed == len(logits)
            and read_strength_exact
            and relation_source_exact
            and nonrelation_exact
            and state_exact
        ),
        "permutation": list(QUERY_SWAP),
    }


def _architecture_evidence(
    core: StrictKeyValueCreditMemoryCore,
    pair: Any,
) -> dict[str, Any]:
    device = next(core.parameters()).device
    first = pair.first.to(device)
    initial = core.initial_state()
    output = core.predict(
        first.relation_features[0:1], first.temporal_features[0:1],
        first.base_logits[0:1], state=initial,
    )
    positive_state, positive = core.apply_feedback(
        initial, output, output.logits.new_tensor((1.0,)),
        evidence_refs="v16-architecture:+1", detach_state=True,
    )
    negative_state, negative = core.apply_feedback(
        initial, output, output.logits.new_tensor((-1.0,)),
        evidence_refs="v16-architecture:-1", detach_state=True,
    )
    invariance = {
        "allocation_exact": positive.allocation_index == negative.allocation_index,
        "read_strength_exact": positive.read_strength == negative.read_strength,
        "write_strength_exact": positive.write_strength == negative.write_strength,
        "erase_gate_exact": positive.erase_gate == negative.erase_gate,
        "write_key_exact": torch.equal(positive.write_key, negative.write_key),
        "keys_exact": torch.equal(positive_state.keys, negative_state.keys),
        "usage_exact": torch.equal(positive_state.usage, negative_state.usage),
        "acquisition_exact": torch.equal(positive_state.acquisition, negative_state.acquisition),
        "stored_values_differ": not torch.equal(positive_state.values, negative_state.values),
    }
    core.zero_grad(set_to_none=True)
    state = core.initial_state()
    for index in ACQUISITION_INDICES:
        acquired = core.predict(
            first.relation_features[index:index + 1],
            first.temporal_features[index:index + 1],
            first.base_logits[index:index + 1], state=state,
        )
        state, _ = core.apply_feedback(
            state, acquired, first.outcomes[index:index + 1],
            evidence_refs=f"v16-gradient:{index}", detach_state=False,
        )
    loss = sum(
        core.outcome_loss(
            core.predict(
                first.relation_features[index:index + 1],
                first.temporal_features[index:index + 1],
                first.base_logits[index:index + 1], state=state,
            ),
            first.outcomes[index:index + 1],
        )
        for index in PROBE_INDICES
    )
    loss.backward()
    gradient_names = tuple(
        name for name, parameter in core.named_parameters()
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and bool((parameter.grad != 0).any().item())
    )
    required_prefixes = (
        "temporal_network.",
        "query_network.",
        "key_network.",
        "read_strength_network.",
        "write_strength_network.",
        "slot_temporal_network.",
        "outcome_embedding.",
        "outcome_value_network.",
        "residual_network.",
    )
    gradient_report = {
        prefix: any(name.startswith(prefix) for name in gradient_names)
        for prefix in required_prefixes
    }
    erase_gradient_names = tuple(
        name for name in gradient_names if name.startswith("erase_network.")
    )
    core.zero_grad(set_to_none=True)
    report = core.architecture_report()
    exact = (
        all(invariance.values())
        and all(gradient_report.values())
        and report["outcome_affects_address_or_gates"] is False
        and report["query_affects_value_content_or_decoder"] is False
        and report["residual_decoder_inputs"] == ("retrieved_value",)
        and report["metadata_inputs"] is False
        and report["erase_gate_active_before_capacity"] is False
        and report["learned_retention_evidence_claimed"] is False
    )
    return {
        "architecture_report": report,
        "outcome_address_invariance": invariance,
        "required_gradient_prefixes_reached": gradient_report,
        "erase_gradient_inactive_by_capacity": not erase_gradient_names,
        "erase_nonzero_gradient_parameter_names": list(erase_gradient_names),
        "nonzero_gradient_parameter_names": list(gradient_names),
        "exact": exact,
    }


def _evaluate(
    core: StrictKeyValueCreditMemoryCore,
    pairs: Sequence[Any],
    affine: tuple[float, float],
) -> dict[str, Any]:
    with _v15_control_scope():
        metrics = v15._evaluate(core, pairs, affine)
    metrics["probe_relation_query_swap"] = _relation_query_swap_control(core, pairs)
    return metrics


def _gate(
    metrics: Mapping[str, Any],
    architecture: Mapping[str, Any],
    *,
    identifiability_exact: bool,
    identity_exact: bool,
) -> bool:
    try:
        lesion = metrics["probe_relation_query_swap"]
        true = metrics["arms"]["true"]
        relation_loss = (
            true["balanced_accuracy"] - lesion["balanced_accuracy"] >= 0.10
            or lesion["mean_probe_nll"] - true["mean_probe_nll"] >= 0.02
        )
        return bool(
            v15._gate(
                metrics,
                identifiability_exact=identifiability_exact,
                identity_exact=identity_exact,
            )
            and architecture["exact"] is True
            and relation_loss
            and lesion["probe_count"] == DEVELOPMENT_PAIRS * 2 * len(PROBE_INDICES)
            and lesion["relation_queries_changed"] is True
            and lesion["relation_inputs_changed"] is True
            and lesion["read_strength_exact"] is True
            and lesion["relation_source_exact"] is True
            and lesion["opposite_relation_class_exact"] is True
            and lesion["temporal_base_labels_exact"] is True
            and lesion["state_and_stored_memory_exact"] is True
            and lesion["only_read_query_swapped"] is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _v15_evidence(result_path: Path, checkpoint_path: Path) -> Mapping[str, Any]:
    if _sha256(result_path) != V15_RESULT_SHA256 or _sha256(checkpoint_path) != V15_CHECKPOINT_SHA256:
        raise RuntimeError("consumed V15 evidence changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current_sources = v15._source_hashes()
    if (
        result.get("identity") != v15.IDENTITY
        or result.get("phase") != "train-development"
        or result.get("classification") != "DEVELOPMENT_NOT_SUPPORTED"
        or result.get("development_authorized") is not False
        or result.get("checkpoint_sha256") != V15_CHECKPOINT_SHA256
        or result.get("source_hashes") != current_sources
        or checkpoint.get("identity") != v15.IDENTITY
        or checkpoint.get("source_hashes") != current_sources
    ):
        raise RuntimeError("consumed V15 evidence identity/classification/source chain changed")
    return result


def _validate_final_authorization(development: Mapping[str, Any], checkpoint_path: Path) -> None:
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "STRICT_KEY_VALUE_PROCEDURAL_CREDIT_SUPPORTED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and development.get("identifiability_preflight", {}).get("exact") is True
        and development.get("identity_checks", {}).get("exact") is True
        and development.get("architecture_evidence", {}).get("exact") is True
        and isinstance(development.get("development_metrics"), Mapping)
        and _gate(
            development["development_metrics"],
            development["architecture_evidence"],
            identifiability_exact=True, identity_exact=True,
        )
    )
    v5._validate_final_authorization(valid)


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V16 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, started = _prepare_angler_device(args.device)
    consumed_v15 = _v15_evidence(Path(args.v15_result), Path(args.v15_checkpoint))
    v14_evidence = v15._v14_structure_evidence(Path(args.v14_result))
    frozen, v13_checkpoint = v15._load_frozen_v13(
        Path(args.v13_checkpoint), Path(args.v13_result), device,
    )
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != v13_checkpoint["qwen_digest"]:
        raise RuntimeError("V16 Qwen does not match frozen V13 foundation")
    corpus = build_paired_latent_contingency_credit_v15()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V16 final opened before authorization")
    schedule = v15._training_schedule(corpus)
    train = v15._encode_pairs(qwen, frozen, corpus.train)
    development = v15._encode_pairs(qwen, frozen, corpus.development)
    core = StrictKeyValueCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(
        device=device, dtype=torch.float32,
    )
    preflight = v15._identifiability_preflight(core, train, development, schedule)
    architecture = _architecture_evidence(core, train[0])
    if preflight["exact"] is not True or architecture["exact"] is not True:
        raise RuntimeError("V16 identifiability/architecture preflight failed before training")
    initial_core_digest = _model_digest(core)
    training, affine = v15._fit(core, train, schedule)
    v15._enforce_resources(started, device)
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
    v15._enforce_resources(started, device)
    sources_after = _source_hashes()
    identity_checks = {
        "qwen_before": qwen_before,
        "qwen_after": foundation_tensor_digest(qwen.model),
        "v13_before": frozen_before,
        "v13_after": _model_digest(frozen),
        "sources_before": sources_before,
        "sources_after": sources_after,
    }
    identity_checks["exact"] = (
        identity_checks["qwen_before"] == identity_checks["qwen_after"]
        and identity_checks["v13_before"] == identity_checks["v13_after"]
        and sources_before == sources_after
    )
    supported = _gate(
        metrics, architecture,
        identifiability_exact=True,
        identity_exact=bool(identity_checks["exact"]),
    )
    classification = (
        "STRICT_KEY_VALUE_PROCEDURAL_CREDIT_SUPPORTED"
        if supported else "DEVELOPMENT_NOT_SUPPORTED"
    )
    checkpoint = {
        "identity": IDENTITY,
        "seed": SEED,
        "corpus_id": CORPUS_ID,
        "completed_updates": TRAIN_UPDATES,
        "temporal_width": TEMPORAL_WIDTH,
        "core_state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "core_digest": _model_digest(core),
        "initial_core_digest": initial_core_digest,
        "affine_calibrator": list(affine),
        "source_hashes": sources_before,
        "qwen_digest": qwen_before,
        "v13_digest": frozen_before,
        "v15_result_sha256": V15_RESULT_SHA256,
        "v15_checkpoint_sha256": V15_CHECKPOINT_SHA256,
        "identifiability_preflight": preflight,
        "architecture_evidence": architecture,
        "development_metrics": metrics,
    }
    result: dict[str, Any] = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": classification,
        "development_authorized": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "identifiability_preflight": preflight,
        "architecture_evidence": architecture,
        "training": training,
        "development_metrics": metrics,
        "identity_checks": identity_checks,
        "source_hashes": sources_before,
        "frozen_evidence": {
            "v15_result_sha256": V15_RESULT_SHA256,
            "v15_checkpoint_sha256": V15_CHECKPOINT_SHA256,
            "v15_classification": consumed_v15["classification"],
            "v14_structure_replicated": v14_evidence["structure_replicated"],
            "v13_checkpoint_qwen_digest": v13_checkpoint["qwen_digest"],
        },
        "frozen_compute": {
            "updates": TRAIN_UPDATES,
            "pairs_per_update": PAIRS_PER_UPDATE,
            "train_pairs": TRAIN_PAIRS,
            "development_pairs": DEVELOPMENT_PAIRS,
            "loss_weights": list(v15.LOSS_WEIGHTS),
            "dtype": "float32",
            "autocast": False,
            "tf32": False,
            "qwen_device": "cuda:0",
            "v13_v16_device": "cuda:1",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    if not identity_checks["exact"]:
        raise RuntimeError("V16 source/foundation drift during execution")
    v15._atomic_torch(checkpoint_path, checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    v15._atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "result": str(result_path)}, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    output = Path(args.final_result)
    if output.exists():
        raise FileExistsError("V16 final identity is already consumed")
    development_path = Path(args.development_result)
    checkpoint_path = Path(args.checkpoint)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sources = _source_hashes()
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != sources
        or checkpoint.get("completed_updates") != TRAIN_UPDATES
    ):
        raise RuntimeError("V16 checkpoint binding failed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, started = _prepare_angler_device(args.device)
    qwen = v5._load_qwen(args.model_path)
    frozen, _ = v15._load_frozen_v13(
        Path(args.v13_checkpoint), Path(args.v13_result), device,
    )
    _v15_evidence(Path(args.v15_result), Path(args.v15_checkpoint))
    v15._v14_structure_evidence(Path(args.v14_result))
    v15._enforce_resources(started, device)
    if (
        foundation_tensor_digest(qwen.model) != checkpoint["qwen_digest"]
        or _model_digest(frozen) != checkpoint["v13_digest"]
    ):
        raise RuntimeError("V16 final frozen dependency changed")
    corpus = build_paired_latent_contingency_credit_v15(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_PAIRS:
        raise RuntimeError("V16 authorized final is incomplete")
    pairs = v15._encode_pairs(qwen, frozen, corpus.final)
    core = StrictKeyValueCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(
        device=device, dtype=torch.float32,
    )
    core.load_state_dict(checkpoint["core_state"], strict=True)
    if _model_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V16 final core digest does not match checkpoint")
    metrics = _evaluate(core, pairs, tuple(checkpoint["affine_calibrator"]))
    v15._enforce_resources(started, device)
    metrics["protocol_invariants"] = copy.deepcopy(
        checkpoint["development_metrics"]["protocol_invariants"]
    )
    exact = (
        foundation_tensor_digest(qwen.model) == checkpoint["qwen_digest"]
        and _model_digest(frozen) == checkpoint["v13_digest"]
        and _source_hashes() == sources
    )
    supported = _gate(
        metrics, checkpoint["architecture_evidence"],
        identifiability_exact=True, identity_exact=exact,
    )
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": metrics,
        "source_hashes": sources,
    }
    if not exact:
        raise RuntimeError("V16 source/foundation drift during final")
    v15._enforce_resources(started, device)
    v15._atomic_json(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v13-checkpoint", default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt")
    parser.add_argument("--v13-result", default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json")
    parser.add_argument("--v14-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-development.json")
    parser.add_argument("--v15-result", default="/opt/angler/results/paired-latent-contingency-credit-v15-development.json")
    parser.add_argument("--v15-checkpoint", default="/opt/angler/results/paired-latent-contingency-credit-v15.pt")
    parser.add_argument("--checkpoint", default="/opt/angler/results/strict-key-value-credit-v16.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/strict-key-value-credit-v16-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/strict-key-value-credit-v16-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
