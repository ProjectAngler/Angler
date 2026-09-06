"""V17 shared-semantic-metric persistent-credit experiment.

V17 changes only V16's address geometry: one normalized relation-only Siamese
encoder is used for both write keys and read queries.  Corpus, objective,
chronology, controls, thresholds, state, and value bottleneck remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from angler.reasoning.shared_semantic_metric_credit_memory import (
    SharedSemanticMetricCreditMemoryCore,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    ACQUISITION_INDICES,
    CORPUS_ID,
    DEVELOPMENT_PAIRS,
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
from experiments.runners import strict_key_value_credit_v16 as v16


IDENTITY = "angler.shared-semantic-metric-credit.v17-first-result"
SEED = 2026083117
TEMPORAL_WIDTH = v15.TEMPORAL_WIDTH
V16_RESULT_SHA256 = "5092DC199123606CE2F987C0DC91F322E271FEB1DEDD2BD36D5B7388F8B21C00"
V16_CHECKPOINT_SHA256 = "503F305D3702F639774BEB8CA777B94B62FAA0ECADED8F131541BF92DB8CF373"


def _sha256(path: str | Path) -> str:
    return v15._sha256(path).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v15._model_digest(model).upper()


def _canonical_json(value: Any) -> str:
    """Canonical evidence form shared by torch checkpoints and JSON packets."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inherited = {f"v16_{key}": value.upper() for key, value in v16._source_hashes().items()}
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "shared_metric_core": _sha256(
            root / "src/angler/reasoning/shared_semantic_metric_credit_memory.py"
        ),
        **inherited,
    }


def _v16_evidence(result_path: Path, checkpoint_path: Path) -> Mapping[str, Any]:
    if (
        _sha256(result_path) != V16_RESULT_SHA256
        or _sha256(checkpoint_path) != V16_CHECKPOINT_SHA256
    ):
        raise RuntimeError("consumed V16 evidence changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current_sources = v16._source_hashes()
    if (
        result.get("identity") != v16.IDENTITY
        or result.get("phase") != "train-development"
        or result.get("classification") != "DEVELOPMENT_NOT_SUPPORTED"
        or result.get("development_authorized") is not False
        or result.get("checkpoint_sha256") != V16_CHECKPOINT_SHA256
        or result.get("source_hashes") != current_sources
        or checkpoint.get("identity") != v16.IDENTITY
        or checkpoint.get("source_hashes") != current_sources
    ):
        raise RuntimeError("consumed V16 evidence identity/source chain changed")
    return result


def _architecture_evidence(
    core: SharedSemanticMetricCreditMemoryCore,
    pair: Any,
) -> dict[str, Any]:
    device = next(core.parameters()).device
    first = pair.first.to(device)
    initial = core.initial_state()
    output = core.predict(
        first.relation_features[0:1], first.temporal_features[0:1],
        first.base_logits[0:1], state=initial,
    )
    alternate_temporal = first.temporal_features[1:2]
    same_relation_other_time = core.predict(
        first.relation_features[0:1], alternate_temporal,
        first.base_logits[0:1], state=initial,
    )
    positive_state, positive = core.apply_feedback(
        initial, output, output.logits.new_tensor((1.0,)),
        evidence_refs="v17-architecture:+1", detach_state=True,
    )
    negative_state, negative = core.apply_feedback(
        initial, output, output.logits.new_tensor((-1.0,)),
        evidence_refs="v17-architecture:-1", detach_state=True,
    )
    invariance = {
        "shared_read_write_code_exact": torch.equal(output.read_queries, output.write_keys),
        "shared_read_write_storage_exact": (
            output.read_queries.data_ptr() == output.write_keys.data_ptr()
        ),
        "same_relation_other_time_exact": torch.equal(
            output.read_queries, same_relation_other_time.read_queries
        ),
        "normalized_code_exact": bool(torch.allclose(
            output.read_queries.norm(dim=-1), torch.ones(1, device=device), atol=1.0e-6,
        )),
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
            evidence_refs=f"v17-gradient:{index}", detach_state=False,
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
        "semantic_metric_network.", "temporal_network.",
        "read_strength_network.", "write_strength_network.",
        "slot_temporal_network.", "outcome_embedding.",
        "outcome_value_network.", "residual_network.",
    )
    reached = {
        prefix: any(name.startswith(prefix) for name in gradient_names)
        for prefix in required_prefixes
    }
    erase_names = tuple(name for name in gradient_names if name.startswith("erase_network."))
    core.zero_grad(set_to_none=True)
    architecture = core.architecture_report()
    exact = bool(
        all(invariance.values())
        and all(reached.values())
        and not erase_names
        and architecture["semantic_metric_shared_for"] == ("read_query", "write_key")
        and architecture["semantic_metric_inputs"] == ("detached_relation",)
        and architecture["independent_query_or_key_networks"] is False
        and architecture["temporal_affects_semantic_code_or_decoder"] is False
        and architecture["outcome_affects_semantic_code_or_gates"] is False
        and architecture["residual_decoder_inputs"] == ("retrieved_value",)
        and architecture["metadata_inputs"] is False
    )
    return {
        "architecture_report": architecture,
        "shared_metric_invariance": invariance,
        "required_gradient_prefixes_reached": reached,
        "erase_gradient_inactive_by_capacity": not erase_names,
        "erase_nonzero_gradient_parameter_names": list(erase_names),
        "nonzero_gradient_parameter_names": list(gradient_names),
        "exact": exact,
    }


def _validated_event_slots(
    events: Sequence[Any], expected_refs: Sequence[str], memory_slots: int,
) -> dict[str, int]:
    """Resolve evidence references to recorded slots without layout assumptions."""

    observed: dict[str, int] = {}
    for event in events:
        if len(event.evidence_refs) != 1:
            raise RuntimeError("V17 anchor event must bind one evidence reference")
        reference = event.evidence_refs[0]
        if reference in observed:
            raise RuntimeError("V17 anchor event evidence reference is duplicated")
        slot = event.allocation_index
        if type(slot) is not int or not 0 <= slot < memory_slots:
            raise RuntimeError("V17 anchor event allocation is outside memory")
        observed[reference] = slot
    if set(observed) != set(expected_refs) or len(observed) != len(expected_refs):
        raise RuntimeError("V17 anchor event lineage is missing or unexpected")
    if len(set(observed.values())) != len(observed):
        raise RuntimeError("V17 anchor event allocation is not one-to-one")
    return observed


def _corresponding_anchor_read_report(
    core: SharedSemanticMetricCreditMemoryCore,
    pairs: Sequence[Any],
) -> dict[str, Any]:
    """Measure evaluator-only structural ownership without feeding labels to the core."""

    if len(pairs) != DEVELOPMENT_PAIRS:
        raise ValueError("V17 correspondence report requires 48 development pairs")
    device = next(core.parameters()).device
    with v16._v15_control_scope():
        true = v15._arm_stream(core, pairs, arm="true")
    rows: list[dict[str, Any]] = []
    state_unchanged = True
    prediction_before_sidecars = True
    lineage_exact = True

    def event_slots(block_index: int, side: str, through_pair: int) -> dict[str, int]:
        block_start = block_index * PAIRS_PER_UPDATE
        expected = []
        for block_pair in pairs[block_start:through_pair + 1]:
            episode = getattr(block_pair, side)
            expected.extend(
                f"{episode.pair_ref}:{episode.twin_ref}:anchor:{anchor}"
                for anchor in ACQUISITION_INDICES
            )
        events = true["event_blocks"][block_index][side][:len(expected)]
        return _validated_event_slots(events, expected, core.memory_slots)

    with torch.no_grad():
        for pair_index, pair in enumerate(pairs):
            local_pair = pair_index % PAIRS_PER_UPDATE
            block_index = pair_index // PAIRS_PER_UPDATE
            expected_step = (local_pair + 1) * len(ACQUISITION_INDICES)
            for side in ("first", "second"):
                episode = getattr(pair, side).to(device)
                state = true["states_after_pair"][pair_index][side]
                if state.step != expected_step:
                    raise RuntimeError("V17 evaluator state/allocation chronology changed")
                for probe in PROBE_INDICES:
                    before = core.state_digest(state)
                    output = core.predict(
                        episode.relation_features[probe:probe + 1],
                        episode.temporal_features[probe:probe + 1],
                        episode.base_logits[probe:probe + 1], state=state,
                    )
                    state_unchanged &= before == core.state_digest(state)
                    # Evaluator-only sidecars are intentionally opened only
                    # after the prediction is complete.
                    slot_by_ref = event_slots(block_index, side, pair_index)
                    prefix_events = true["event_blocks"][block_index][side][:expected_step]
                    replayed, _ = core.replay(prefix_events)
                    if core.state_digest(replayed) != core.state_digest(state):
                        raise RuntimeError("V17 anchor event prefix does not reconstruct state")
                    occupied_slots = tuple(event.allocation_index for event in prefix_events)
                    if (
                        len(occupied_slots) != state.step
                        or len(set(occupied_slots)) != state.step
                        or any(not 0 <= slot < core.memory_slots for slot in occupied_slots)
                    ):
                        raise RuntimeError("V17 occupied event-slot lineage is invalid")
                    occupied_set = set(occupied_slots)
                    anchor_classes = tuple(
                        episode.relation_classes[index] for index in ACQUISITION_INDICES
                    )
                    relation_class = episode.relation_classes[probe]
                    prediction_before_sidecars &= output.state_step == state.step
                    if set(anchor_classes) != {"alpha", "beta"}:
                        raise RuntimeError("V17 evaluator anchor classes are invalid")
                    local_anchor = anchor_classes.index(relation_class)
                    opposite_anchor = 1 - local_anchor
                    matching_ref = (
                        f"{episode.pair_ref}:{episode.twin_ref}:anchor:{local_anchor}"
                    )
                    opposite_ref = (
                        f"{episode.pair_ref}:{episode.twin_ref}:anchor:{opposite_anchor}"
                    )
                    matching_slot = slot_by_ref[matching_ref]
                    opposite_slot = slot_by_ref[opposite_ref]
                    lineage_exact &= (
                        matching_slot in occupied_set and opposite_slot in occupied_set
                    )
                    weights = output.read_weights[0]
                    matching = float(weights[matching_slot].item())
                    older_same_slots = []
                    block_start = pair_index - local_pair
                    for older_local in range(local_pair):
                        older = getattr(pairs[block_start + older_local], side)
                        older_classes = tuple(
                            older.relation_classes[index] for index in ACQUISITION_INDICES
                        )
                        older_anchor = older_classes.index(relation_class)
                        older_ref = (
                            f"{older.pair_ref}:{older.twin_ref}:anchor:{older_anchor}"
                        )
                        older_slot = slot_by_ref[older_ref]
                        lineage_exact &= older_slot in occupied_set
                        older_same_slots.append(older_slot)
                    competitors = [opposite_slot, *older_same_slots]
                    competitor = max(float(weights[index].item()) for index in competitors)
                    other_max = max(
                        float(weights[index].item())
                        for index in occupied_slots if index != matching_slot
                    )
                    rows.append({
                        "probe_position": probe,
                        "transition": episode.transition_group,
                        "family": episode.generator_family,
                        "top1": int(matching > other_max),
                        "matching_mass": matching,
                        "largest_registered_competitor_mass": competitor,
                        "margin": matching - competitor,
                    })

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(selected),
            "top1_accuracy": sum(row["top1"] for row in selected) / len(selected),
            "mean_matching_mass": sum(row["matching_mass"] for row in selected) / len(selected),
            "mean_largest_registered_competitor_mass": sum(
                row["largest_registered_competitor_mass"] for row in selected
            ) / len(selected),
            "mean_matching_minus_competitor_margin": sum(row["margin"] for row in selected) / len(selected),
        }

    return {
        **summarize(rows),
        "by_probe_position": {
            str(position): summarize([row for row in rows if row["probe_position"] == position])
            for position in PROBE_INDICES
        },
        "by_transition": {
            transition: summarize([row for row in rows if row["transition"] == transition])
            for transition in sorted({row["transition"] for row in rows})
        },
        "by_family": {
            family: summarize([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
        "metadata_evaluator_only": True,
        "anchor_slot_lineage_exact": lineage_exact,
        "prediction_before_sidecars_exact": prediction_before_sidecars,
        "learner_state_unchanged": state_unchanged,
    }


def _evaluate(
    core: SharedSemanticMetricCreditMemoryCore,
    pairs: Sequence[Any],
    affine: tuple[float, float],
) -> dict[str, Any]:
    metrics = v16._evaluate(core, pairs, affine)
    metrics["corresponding_anchor_read_mass"] = _corresponding_anchor_read_report(core, pairs)
    return metrics


def _component_gate(
    metrics: Mapping[str, Any], architecture: Mapping[str, Any], *,
    identifiability_exact: bool, identity_exact: bool,
) -> bool:
    try:
        arms = metrics["arms"]
        true, deranged = arms["true"], arms["deranged"]
        controls = metrics["state_controls"]
        lesion = metrics["probe_relation_query_swap"]
        read = metrics["corresponding_anchor_read_mass"]
        query_effect = (
            true["balanced_accuracy"] - lesion["balanced_accuracy"] >= 0.10
            or lesion["mean_probe_nll"] - true["mean_probe_nll"] >= 0.02
        )
        mismatch_effect = (
            controls["matched_balanced_accuracy"]
            - controls["key_value_mismatch_balanced_accuracy"] >= 0.10
            or controls["mean_nll"]["key_value_mismatch"]
            - controls["mean_nll"]["matched"] >= 0.02
        )
        paired_state = all(
            controls["matched_balanced_accuracy"] - controls[f"{name}_balanced_accuracy"] >= 0.10
            and controls["mean_nll"][name] - controls["mean_nll"]["matched"] >= 0.02
            for name in ("zero", "unrelated")
        )
        true_deranged = (
            true["balanced_accuracy"] - deranged["balanced_accuracy"] >= 0.10
            and deranged["mean_probe_nll"] - true["mean_probe_nll"] >= 0.02
        )
        return bool(
            identifiability_exact is True and identity_exact is True
            and architecture["exact"] is True
            and metrics["pair_count"] == DEVELOPMENT_PAIRS
            and lesion["probe_count"] == DEVELOPMENT_PAIRS * 2 * len(PROBE_INDICES)
            and read["count"] == DEVELOPMENT_PAIRS * 2 * len(PROBE_INDICES)
            and read["top1_accuracy"] >= 0.75
            and read["mean_matching_mass"] >= 0.30
            and read["mean_matching_minus_competitor_margin"] >= 0.10
            and read["metadata_evaluator_only"] is True
            and read["anchor_slot_lineage_exact"] is True
            and read["prediction_before_sidecars_exact"] is True
            and read["learner_state_unchanged"] is True
            and all(
                row["count"] == DEVELOPMENT_PAIRS * 2
                and row["top1_accuracy"] >= 0.65
                for row in read["by_probe_position"].values()
            )
            and len(read["by_probe_position"]) == len(PROBE_INDICES)
            and all(
                row["count"] == (DEVELOPMENT_PAIRS // 4) * 2 * len(PROBE_INDICES)
                and row["top1_accuracy"] >= 0.65
                for row in read["by_transition"].values()
            )
            and len(read["by_transition"]) == 4
            and all(
                row["count"] == (DEVELOPMENT_PAIRS // 12) * 2 * len(PROBE_INDICES)
                for row in read["by_family"].values()
            )
            and len(read["by_family"]) == 12
            and query_effect and mismatch_effect and paired_state and true_deranged
            and true["balanced_accuracy"]
            - arms["post_acquisition_reset"]["balanced_accuracy"] >= 0.10
            and true["balanced_accuracy"]
            - arms["outcome_blind_writer"]["balanced_accuracy"] >= 0.10
            and true["balanced_accuracy"]
            - metrics["affine_calibrator"]["balanced_accuracy"] >= 0.10
            and metrics["paired_twin"]["directional_accuracy"] >= 0.75
            and metrics["paired_twin"]["mean_outcome_directed_logit_margin"] >= 0.20
            and controls["opposite_twin_swap_reversal_fraction"] >= 0.70
            and controls["opposite_twin_swap_gain_retention"] >= 0.80
            and all(metrics["outcome_blind_integrity"].values())
            and all(
                arm["finite"] is True
                and 0.0 <= arm["read_weight_min"] <= arm["read_weight_max"] <= 1.0
                and 0.0 <= arm["write_strength_min"] <= arm["write_strength_max"] <= 1.0
                for arm in arms.values()
            )
            and metrics["replay"]["exact"] is True
            and metrics["replay"]["maximum_logit_error"] <= 1.0e-6
            and metrics["state_bounds"]["finite"] is True
            and metrics["state_bounds"]["keys_max_abs"] <= 1.0
            and metrics["state_bounds"]["values_max_abs"] <= 1.0
            and 0.0 <= metrics["state_bounds"]["usage_min"]
            <= metrics["state_bounds"]["usage_max"] <= 1.0
            and 0.0 <= metrics["state_bounds"]["acquisition_min"]
            <= metrics["state_bounds"]["acquisition_max"] <= 1.0
            and all(metrics["protocol_invariants"].values())
            and lesion["only_read_query_swapped"] is True
            and lesion["relation_queries_changed"] is True
            and lesion["relation_inputs_changed"] is True
            and lesion["read_strength_exact"] is True
            and lesion["relation_source_exact"] is True
            and lesion["opposite_relation_class_exact"] is True
            and lesion["temporal_base_labels_exact"] is True
            and lesion["state_and_stored_memory_exact"] is True
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _full_gate(
    metrics: Mapping[str, Any], architecture: Mapping[str, Any], *,
    identifiability_exact: bool, identity_exact: bool,
) -> bool:
    return bool(
        _component_gate(
            metrics, architecture,
            identifiability_exact=identifiability_exact,
            identity_exact=identity_exact,
        )
        and v16._gate(
            metrics, architecture,
            identifiability_exact=identifiability_exact,
            identity_exact=identity_exact,
        )
    )


def _validate_final_authorization(development: Mapping[str, Any], checkpoint_path: Path) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError):
        checkpoint = {}
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification")
        == "FULL_DURABLE_SEMANTIC_CREDIT_SUPPORTED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and development.get("core_digest") == checkpoint.get("core_digest")
        and checkpoint.get("identity") == IDENTITY
        and checkpoint.get("source_hashes") == development.get("source_hashes")
        and _canonical_json(checkpoint.get("identifiability_preflight"))
        == _canonical_json(development.get("identifiability_preflight"))
        and _canonical_json(checkpoint.get("architecture_evidence"))
        == _canonical_json(development.get("architecture_evidence"))
        and _canonical_json(checkpoint.get("development_metrics"))
        == _canonical_json(development.get("development_metrics"))
        and _canonical_json(checkpoint.get("identity_checks"))
        == _canonical_json(development.get("identity_checks"))
        and development.get("identifiability_preflight", {}).get("exact") is True
        and development.get("identity_checks", {}).get("exact") is True
        and isinstance(development.get("development_metrics"), Mapping)
        and _full_gate(
            development["development_metrics"], development["architecture_evidence"],
            identifiability_exact=True, identity_exact=True,
        )
    )
    v5._validate_final_authorization(valid)


def _protocol(training: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
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


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path, result_path = Path(args.checkpoint), Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V17 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device, started = v16._prepare_angler_device(args.device)
    consumed_v16 = _v16_evidence(Path(args.v16_result), Path(args.v16_checkpoint))
    v14_evidence = v15._v14_structure_evidence(Path(args.v14_result))
    frozen, v13_checkpoint = v15._load_frozen_v13(
        Path(args.v13_checkpoint), Path(args.v13_result), device,
    )
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != v13_checkpoint["qwen_digest"]:
        raise RuntimeError("V17 Qwen does not match frozen V13 foundation")
    corpus = build_paired_latent_contingency_credit_v15()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V17 final opened before authorization")
    schedule = v15._training_schedule(corpus)
    train = v15._encode_pairs(qwen, frozen, corpus.train)
    development = v15._encode_pairs(qwen, frozen, corpus.development)
    core = SharedSemanticMetricCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(
        device=device, dtype=torch.float32,
    )
    preflight = v15._identifiability_preflight(core, train, development, schedule)
    architecture = _architecture_evidence(core, train[0])
    if preflight["exact"] is not True or architecture["exact"] is not True:
        raise RuntimeError("V17 identifiability/architecture preflight failed")
    initial_core_digest = _model_digest(core)
    training, affine = v15._fit(core, train, schedule)
    v15._enforce_resources(started, device)
    metrics = _evaluate(core, development, affine)
    metrics["protocol_invariants"] = _protocol(training, metrics)
    v15._enforce_resources(started, device)
    sources_after = _source_hashes()
    identity_checks = {
        "qwen_before": qwen_before, "qwen_after": foundation_tensor_digest(qwen.model),
        "v13_before": frozen_before, "v13_after": _model_digest(frozen),
        "sources_before": sources_before, "sources_after": sources_after,
    }
    identity_checks["exact"] = (
        qwen_before == identity_checks["qwen_after"]
        and frozen_before == identity_checks["v13_after"]
        and sources_before == sources_after
    )
    component = _component_gate(
        metrics, architecture, identifiability_exact=True,
        identity_exact=bool(identity_checks["exact"]),
    )
    full = _full_gate(
        metrics, architecture, identifiability_exact=True,
        identity_exact=bool(identity_checks["exact"]),
    )
    classification = (
        "FULL_DURABLE_SEMANTIC_CREDIT_SUPPORTED" if full else
        "SHARED_SEMANTIC_CORRESPONDENCE_SUPPORTED" if component else
        "DEVELOPMENT_NOT_SUPPORTED"
    )
    checkpoint = {
        "identity": IDENTITY, "seed": SEED, "corpus_id": CORPUS_ID,
        "completed_updates": TRAIN_UPDATES, "temporal_width": TEMPORAL_WIDTH,
        "core_state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "core_digest": _model_digest(core), "initial_core_digest": initial_core_digest,
        "affine_calibrator": list(affine), "source_hashes": sources_before,
        "qwen_digest": qwen_before, "v13_digest": frozen_before,
        "v16_result_sha256": V16_RESULT_SHA256,
        "v16_checkpoint_sha256": V16_CHECKPOINT_SHA256,
        "identifiability_preflight": preflight,
        "architecture_evidence": architecture, "development_metrics": metrics,
        "identity_checks": identity_checks,
    }
    result: dict[str, Any] = {
        "identity": IDENTITY, "phase": "train-development",
        "classification": classification, "development_authorized": full,
        "component_supported": component, "full_durable_supported": full,
        "core_digest": checkpoint["core_digest"],
        "corpus_id": CORPUS_ID, "seed": SEED,
        "identifiability_preflight": preflight, "architecture_evidence": architecture,
        "training": training, "development_metrics": metrics,
        "identity_checks": identity_checks, "source_hashes": sources_before,
        "frozen_evidence": {
            "v16_result_sha256": V16_RESULT_SHA256,
            "v16_checkpoint_sha256": V16_CHECKPOINT_SHA256,
            "v16_classification": consumed_v16["classification"],
            "v14_structure_replicated": v14_evidence["structure_replicated"],
            "v13_checkpoint_qwen_digest": v13_checkpoint["qwen_digest"],
        },
        "frozen_compute": {
            "updates": TRAIN_UPDATES, "pairs_per_update": PAIRS_PER_UPDATE,
            "train_pairs": TRAIN_PAIRS, "development_pairs": DEVELOPMENT_PAIRS,
            "loss_weights": list(v15.LOSS_WEIGHTS), "dtype": "float32",
            "autocast": False, "tf32": False, "qwen_device": "cuda:0",
            "v13_v17_device": "cuda:1", "contrastive_loss": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    if not identity_checks["exact"]:
        raise RuntimeError("V17 source/foundation drift during execution")
    v15._atomic_torch(checkpoint_path, checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    v15._atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "result": str(result_path)}, sort_keys=True))


def _run_final(args: argparse.Namespace) -> None:
    output = Path(args.final_result)
    if output.exists():
        raise FileExistsError("V17 final identity is already consumed")
    development_path, checkpoint_path = Path(args.development_result), Path(args.checkpoint)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sources = _source_hashes()
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != sources
        or checkpoint.get("completed_updates") != TRAIN_UPDATES
    ):
        raise RuntimeError("V17 checkpoint binding failed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device, started = v16._prepare_angler_device(args.device)
    qwen = v5._load_qwen(args.model_path)
    frozen, _ = v15._load_frozen_v13(
        Path(args.v13_checkpoint), Path(args.v13_result), device,
    )
    _v16_evidence(Path(args.v16_result), Path(args.v16_checkpoint))
    v15._v14_structure_evidence(Path(args.v14_result))
    v15._enforce_resources(started, device)
    if (
        foundation_tensor_digest(qwen.model) != checkpoint["qwen_digest"]
        or _model_digest(frozen) != checkpoint["v13_digest"]
    ):
        raise RuntimeError("V17 final frozen dependency changed")
    corpus = build_paired_latent_contingency_credit_v15(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_PAIRS:
        raise RuntimeError("V17 authorized final is incomplete")
    pairs = v15._encode_pairs(qwen, frozen, corpus.final)
    core = SharedSemanticMetricCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(
        device=device, dtype=torch.float32,
    )
    core.load_state_dict(checkpoint["core_state"], strict=True)
    if _model_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V17 final core digest does not match checkpoint")
    metrics = _evaluate(core, pairs, tuple(checkpoint["affine_calibrator"]))
    metrics["protocol_invariants"] = copy.deepcopy(
        checkpoint["development_metrics"]["protocol_invariants"]
    )
    metrics["protocol_invariants"]["eight_pair_state_horizon"] = (
        metrics["maximum_state_step"]
        == PAIRS_PER_UPDATE * len(ACQUISITION_INDICES)
    )
    v15._enforce_resources(started, device)
    exact = (
        foundation_tensor_digest(qwen.model) == checkpoint["qwen_digest"]
        and _model_digest(frozen) == checkpoint["v13_digest"]
        and _source_hashes() == sources
    )
    component = _component_gate(
        metrics, checkpoint["architecture_evidence"],
        identifiability_exact=True, identity_exact=exact,
    )
    full = _full_gate(
        metrics, checkpoint["architecture_evidence"],
        identifiability_exact=True, identity_exact=exact,
    )
    if not exact:
        raise RuntimeError("V17 source/foundation drift during final")
    v15._enforce_resources(started, device)
    v15._atomic_json(output, {
        "identity": IDENTITY, "phase": "final",
        "classification": "FULL_SUPPORTED" if full else "COMPONENT_SUPPORTED" if component else "NOT_SUPPORTED",
        "component_supported": component, "full_durable_supported": full,
        "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": metrics, "source_hashes": sources,
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v13-checkpoint", default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt")
    parser.add_argument("--v13-result", default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json")
    parser.add_argument("--v14-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-development.json")
    parser.add_argument("--v16-result", default="/opt/angler/results/strict-key-value-credit-v16-development.json")
    parser.add_argument("--v16-checkpoint", default="/opt/angler/results/strict-key-value-credit-v16.pt")
    parser.add_argument("--checkpoint", default="/opt/angler/results/shared-semantic-metric-credit-v17.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/shared-semantic-metric-credit-v17-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/shared-semantic-metric-credit-v17-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
