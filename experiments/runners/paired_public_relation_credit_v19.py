"""V19 paired public-relation comparison for strict causal credit memory.

V19 changes one learned operation from V17: a bounded, zero-initialized
comparison of the current detached public relation with each occupied slot's
detached public relation.  Both arms retain the frozen V15 task, chronology,
corpus, evaluator, and optimizer.  This runner owns no final phase.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

from angler.reasoning.paired_public_relation_credit_memory import (
    PairedPublicRelationCreditEvent,
    PairedPublicRelationCreditMemoryCore,
    PairedPublicRelationCreditState,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    ACQUISITION_INDICES,
    CORPUS_ID,
    DEVELOPMENT_PAIRS,
    PAIRS_PER_UPDATE,
    PROBE_INDICES,
    TRAIN_PAIRS,
    TRAIN_UPDATES,
    build_paired_latent_contingency_credit_v15,
)
from experiments.corpora.structure_protected_oml_natural_trace_v13 import (
    FinalPartitionSealedError,
)
from experiments.runners import barlow_semantic_invariance_v18 as v18
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import shared_semantic_metric_credit_v17 as v17
from experiments.runners import strict_key_value_credit_v16 as v16


IDENTITY = "angler.paired-public-relation-credit.v19-first-result"
SEED = 2026083119
TEMPORAL_WIDTH = v15.TEMPORAL_WIDTH
QUERY_SWAP = (3, 2, 5, 4)
PAIR_PREFIX = "paired_relation_scorer."

LEAF_SHA256 = "4F5E84C726BFA15EA1E61173288528AA40DBE1EABBE35EFEC54332FCFC6BC5C0"
V18_RESULT_SHA256 = "8289750FF87394493E0E23681BC52128CC55429F291A84488E700179F9FE87A9"
V18_FULL_CHECKPOINT_SHA256 = "BBCD20BCE682936181BDB5FA89A72674BE92F08A54515A35024EE6776F39AD80"
V18_UNARY_CHECKPOINT_SHA256 = "27212FB2B531CCA13B29D1F8CC69343486BC16998E6242875F23B768B931ED2C"
V18_RUNNER_SHA256 = "77382FA24C76BD9A1B221F3B2F808C0484F2BFCC1DC65BF433193963293B49F2"
V18_TEST_SHA256 = "D26042AB559CD4D0B127B9BE1D80FB78655057AB93DEF696022391D7F2855E35"
V17_CORE_SHA256 = "A25D42E9A4B8697FC441FC4C55543BFABD661AD5BD7756C4BF87EA80EAB14790"
PHASE6_PAIRED_DONOR_SHA256 = "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
PHASE6_OML_DONOR_SHA256 = "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
V11_SIDECAR_DONOR_SHA256 = "99C2452B6F42E6241CC2B4CC4CA86DA8803E4B08924E3D1376A0813460070C7D"
REASONING_EXPORT_SHA256 = "B06E4F87620974FD5C2F2AD75A433DBE55A9CBA799B0F442C5FE9A51785D5798"


def _sha256(path: str | Path) -> str:
    return v15._sha256(path).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v15._model_digest(model).upper()


def _safe_torch_load(path: str | Path) -> Any:
    """Restricted checkpoint load allowing only PyTorch's version metadata type."""

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inherited = {f"v18_{key}": value.upper() for key, value in v18._source_hashes().items()}
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "runner_test": _sha256(root / "tests/unit/experiments/test_paired_public_relation_credit_v19.py"),
        "paired_public_relation_core": _sha256(root / "src/angler/reasoning/paired_public_relation_credit_memory.py"),
        "paired_public_relation_core_test": _sha256(root / "tests/unit/reasoning/test_paired_public_relation_credit_memory.py"),
        "leaf": _sha256(root / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-PAIRED-PUBLIC-RELATION-CREDIT-V19-001.md"),
        "phase6_paired_comparator_donor": _sha256(root / "experiments/runners/phase6_v12_champion_paired_graph_context.py"),
        "phase6_oml_relation_donor": _sha256(root / "experiments/runners/phase6_oml_relation_representation.py"),
        "v11_public_sidecar_donor": _sha256(root / "src/angler/reasoning/natural_trace_graph_causal_memory.py"),
        **inherited,
    }


def _canonical_digest(value: Any) -> str:
    canonical = v17._canonical_json(value).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def _final_checkpoint_seal(
    full_path: Path,
    unary_path: Path,
    *,
    expected_common: Mapping[str, Any],
    full_core_digest: str,
    unary_core_digest: str,
    full_metrics: Mapping[str, Any],
    unary_metrics: Mapping[str, Any],
    full_affine: Sequence[float],
    unary_affine: Sequence[float],
) -> dict[str, Any]:
    """Reload and bind both durable arms before the terminal result is written."""

    expected = {
        "full": (full_path, full_core_digest, full_metrics, full_affine, True),
        "unary": (unary_path, unary_core_digest, unary_metrics, unary_affine, False),
    }
    sealed_arms: dict[str, Any] = {}
    for arm, (path, expected_core_digest, expected_metrics, expected_affine, expected_mode) in expected.items():
        checkpoint = _safe_torch_load(path)
        restored = PairedPublicRelationCreditMemoryCore(
            temporal_width=TEMPORAL_WIDTH,
            pair_residual_enabled_by_default=expected_mode,
        ).to(device="cpu", dtype=torch.float32)
        restored.load_state_dict(checkpoint.get("core_state", {}), strict=True)
        loaded_core_digest = _model_digest(restored)
        checks = {
            "identity_exact": checkpoint.get("identity") == IDENTITY,
            "arm_exact": checkpoint.get("arm") == arm,
            "arm_mode_exact": checkpoint.get("pair_residual_enabled_by_default") is expected_mode,
            "common_evidence_exact": all(
                _canonical_digest(checkpoint.get(name)) == _canonical_digest(value)
                for name, value in expected_common.items()
            ),
            "core_field_exact": checkpoint.get("core_digest") == expected_core_digest,
            "loaded_core_exact": loaded_core_digest == expected_core_digest,
            "metrics_exact": (
                _canonical_digest(checkpoint.get("development_metrics"))
                == _canonical_digest(expected_metrics)
            ),
            "affine_exact": (
                _canonical_digest(checkpoint.get("affine_calibrator"))
                == _canonical_digest(list(expected_affine))
            ),
            "final_partition_sealed": (
                checkpoint.get("identity_checks", {}).get("final_partition_sealed") is True
            ),
        }
        checks["exact"] = all(checks.values())
        sealed_arms[arm] = {
            **checks,
            "core_digest": loaded_core_digest,
            "development_metrics_sha256": _canonical_digest(expected_metrics),
            "common_evidence_sha256": _canonical_digest(expected_common),
            "affine_calibrator_sha256": _canonical_digest(list(expected_affine)),
            "checkpoint_sha256": _sha256(path),
        }
    current_sources_exact = _source_hashes() == expected_common.get("source_hashes")
    exact = bool(current_sources_exact and all(arm["exact"] for arm in sealed_arms.values()))
    seal = {
        "identity": IDENTITY,
        "source_hashes_exact": current_sources_exact,
        "arms": sealed_arms,
        "exact": exact,
    }
    if not exact:
        raise RuntimeError("V19 final dual-checkpoint seal failed")
    return seal


def _v18_evidence(result_path: Path, full_path: Path, unary_path: Path) -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if (
        _sha256(result_path) != V18_RESULT_SHA256
        or _sha256(full_path) != V18_FULL_CHECKPOINT_SHA256
        or _sha256(unary_path) != V18_UNARY_CHECKPOINT_SHA256
        or _sha256(root / "experiments/runners/barlow_semantic_invariance_v18.py") != V18_RUNNER_SHA256
        or _sha256(root / "tests/unit/experiments/test_barlow_semantic_invariance_v18.py") != V18_TEST_SHA256
        or _sha256(root / "src/angler/reasoning/shared_semantic_metric_credit_memory.py") != V17_CORE_SHA256
        or _sha256(root / "src/angler/reasoning/__init__.py") != REASONING_EXPORT_SHA256
        or _sha256(root / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-PAIRED-PUBLIC-RELATION-CREDIT-V19-001.md") != LEAF_SHA256
    ):
        raise RuntimeError("V19 frozen V18/leaf/export evidence changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    full = _safe_torch_load(full_path)
    unary = _safe_torch_load(unary_path)
    restored_full = v17.SharedSemanticMetricCreditMemoryCore(temporal_width=v18.TEMPORAL_WIDTH)
    restored_unary = v17.SharedSemanticMetricCreditMemoryCore(temporal_width=v18.TEMPORAL_WIDTH)
    restored_full.load_state_dict(full.get("core_state", {}), strict=True)
    restored_unary.load_state_dict(unary.get("core_state", {}), strict=True)
    current = v18._source_hashes()
    mask_exact = result.get("preflight", {}).get("mask_schedule_after_preflight", {}).get("exact") is True
    mask_exact &= result.get("training", {}).get("mask_schedule_after_training", {}).get("exact") is True
    if (
        current.get("runner") != V18_RUNNER_SHA256
        or current.get("runner_test") != V18_TEST_SHA256
        or result.get("identity") != v18.IDENTITY
        or result.get("phase") != "train-development"
        or result.get("classification") != "DEVELOPMENT_NOT_SUPPORTED"
        or result.get("barlow_semantic_objective_supported") is not False
        or result.get("development_authorized") is not False
        or result.get("final_partition_sealed") is not True
        or "final_seal" in result
        or result.get("identity_checks", {}).get("exact") is not True
        or result.get("identity_checks", {}).get("final_partition_sealed") is not True
        or result.get("preflight", {}).get("exact") is not True
        or result.get("identity_checks", {}).get("exact") is not True
        or not mask_exact
        or result.get("source_hashes") != current
        or result.get("full_checkpoint_sha256") != V18_FULL_CHECKPOINT_SHA256
        or result.get("objective_off_checkpoint_sha256") != V18_UNARY_CHECKPOINT_SHA256
        or full.get("identity") != v18.IDENTITY
        or full.get("arm") != "full"
        or unary.get("identity") != v18.IDENTITY
        or unary.get("arm") != "objective_off"
        or full.get("source_hashes") != current
        or unary.get("source_hashes") != current
        or full.get("core_digest") != _model_digest(restored_full)
        or unary.get("core_digest") != _model_digest(restored_unary)
        or full.get("identity_checks", {}).get("exact") is not True
        or unary.get("identity_checks", {}).get("exact") is not True
        or full.get("identity_checks", {}).get("final_partition_sealed") is not True
        or unary.get("identity_checks", {}).get("final_partition_sealed") is not True
    ):
        raise RuntimeError("V19 consumed V18 identity/source/checkpoint chain changed")
    return result


class _PairModeView:
    """Non-owning core view that makes the arm mode explicit to inherited code."""

    def __init__(self, core: PairedPublicRelationCreditMemoryCore, enabled: bool) -> None:
        self.core = core
        self.enabled = bool(enabled)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        if "pair_residual_enabled" in kwargs:
            raise TypeError("arm view owns pair_residual_enabled")
        return self.core.predict(*args, **kwargs, pair_residual_enabled=self.enabled)

    forward = predict

    def replay(self, *args: Any, **kwargs: Any) -> Any:
        """Replay with this view's arm mode without changing the launch core."""

        default_before = self.core.pair_residual_enabled_by_default
        model_before = _model_digest(self.core)
        replay_core = copy.copy(self.core)
        replay_core.pair_residual_enabled_by_default = self.enabled
        result = replay_core.replay(*args, **kwargs)
        if (
            self.core.pair_residual_enabled_by_default is not default_before
            or _model_digest(self.core) != model_before
        ):
            raise RuntimeError("V19 mode-aware replay mutated the underlying core")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)


def _optimizer_snapshot(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    payload = {
        "state": {
            str(index): {
                key: v15._tensor_digest(value) if isinstance(value, torch.Tensor) else value
                for key, value in sorted(values.items())
            }
            for index, values in sorted(state["state"].items())
        },
        "param_groups": [
            {key: value for key, value in sorted(group.items())}
            for group in state["param_groups"]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest().upper()


def _optimizer_parameter_order(
    core: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> tuple[tuple[str, ...], ...]:
    names = {id(parameter): name for name, parameter in core.named_parameters()}
    return tuple(
        tuple(names.get(id(parameter), "<unbound>") for parameter in group["params"])
        for group in optimizer.param_groups
    )


def _non_sidecar_state_digest(state: PairedPublicRelationCreditState) -> str:
    payload = {
        "keys": v15._tensor_digest(state.keys),
        "values": v15._tensor_digest(state.values),
        "usage": v15._tensor_digest(state.usage),
        "acquisition": v15._tensor_digest(state.acquisition),
        "step": state.step,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest().upper()


def _mismatched_state(state: PairedPublicRelationCreditState) -> PairedPublicRelationCreditState:
    values = state.values.clone()
    if state.step > 1:
        occupied = torch.nonzero(state.usage > 0, as_tuple=False).flatten()
        values[occupied] = values[occupied].flip(0)
    return replace(state, values=values)


def _apply_outcome_blind_feedback(
    core: Any,
    state: PairedPublicRelationCreditState,
    output: Any,
    outcomes: torch.Tensor,
    *,
    evidence_ref: str,
    routing_output: Any | None = None,
) -> tuple[PairedPublicRelationCreditState, PairedPublicRelationCreditEvent]:
    """Reuse normal routing/sidecar timing while replacing only value content."""

    actual = core.core if isinstance(core, _PairModeView) else core
    if routing_output is not None and not all(
        torch.equal(getattr(output, name), getattr(routing_output, name))
        for name in ("write_keys", "read_strengths", "write_strengths", "erase_gates")
    ):
        raise RuntimeError("V19 blind routing changed outcome-free address/gates")
    normal_state, event = actual.apply_feedback(
        state, output, outcomes, evidence_refs=evidence_ref, detach_state=True,
    )
    shared = actual.outcome_values(outcomes.new_tensor((-1.0, 1.0))).mean(dim=0, keepdim=True)
    allocation = event.allocation_index
    slot_weights = state.usage.new_zeros(actual.memory_slots)
    slot_weights[allocation] = event.write_strength
    slot = slot_weights.unsqueeze(-1)
    erased = state.values * (1.0 - slot * event.erase_gate)
    values = (1.0 - slot) * erased + slot * shared[0].unsqueeze(0)
    updated = replace(normal_state, values=values.detach())
    actual.validate_state(updated)
    blinded = replace(event, write_value=shared.detach().cpu().clone())
    return updated, blinded


def _write_blind_anchors(
    core: Any,
    episode: Any,
    state: PairedPublicRelationCreditState,
    routing_state: PairedPublicRelationCreditState,
) -> tuple[PairedPublicRelationCreditState, PairedPublicRelationCreditState, list[Any], list[int], list[float]]:
    events: list[Any] = []
    steps: list[int] = []
    logits: list[float] = []
    for index in ACQUISITION_INDICES:
        if state.step != routing_state.step:
            raise RuntimeError("V19 blind/routing state chronology diverged")
        steps.append(state.step)
        output = core.predict(
            episode.relation_features[index:index + 1], episode.temporal_features[index:index + 1],
            episode.base_logits[index:index + 1], state=state,
        )
        route = core.predict(
            episode.relation_features[index:index + 1], episode.temporal_features[index:index + 1],
            episode.base_logits[index:index + 1], state=routing_state,
        )
        feedback = episode.outcomes[index:index + 1]
        ref = f"{episode.pair_ref}:{episode.twin_ref}:anchor:{index}"
        state, event = _apply_outcome_blind_feedback(
            core, state, output, feedback, evidence_ref=ref, routing_output=route,
        )
        routing_state, _ = core.apply_feedback(
            routing_state, route, feedback,
            evidence_refs=f"{ref}:routing-reference", detach_state=True,
        )
        logits.append(float(output.logits.detach().item()))
        events.append(event)
    return state, routing_state, events, steps, logits


@contextmanager
def _v19_control_scope() -> Iterator[None]:
    bindings = {
        "StructureKeyedCreditState": PairedPublicRelationCreditState,
        "StructureKeyedCreditEvent": PairedPublicRelationCreditEvent,
        "_apply_outcome_blind_feedback": _apply_outcome_blind_feedback,
        "_write_blind_anchors": _write_blind_anchors,
        "_mismatched_state": _mismatched_state,
    }
    originals = {name: getattr(v15, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(v15, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(v15, name, value)


def _rotate_public_sidecars(state: PairedPublicRelationCreditState) -> PairedPublicRelationCreditState:
    occupied = torch.nonzero(state.usage > 0, as_tuple=False).flatten().sort().values
    if occupied.numel() < 2:
        raise ValueError("V19 sidecar derangement requires at least two occupied slots")
    rows = state.public_relations.clone()
    rows[occupied] = state.public_relations[occupied.roll(-1)]
    changed = torch.any(rows[occupied] != state.public_relations[occupied], dim=-1)
    if not bool(changed.all().item()):
        raise RuntimeError("V19 occupied public-sidecar rotation has a fixed row")
    return replace(state, public_relations=rows)


def _task_parity_preflight(
    full: PairedPublicRelationCreditMemoryCore,
    unary: PairedPublicRelationCreditMemoryCore,
    pairs: Sequence[Any],
    indices: Sequence[int],
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> dict[str, Any]:
    if set(optimizers) != {"full", "unary"}:
        raise ValueError("V19 task parity requires the two launch optimizers")
    device = next(full.parameters()).device
    variables = {
        arm: (torch.ones((), device=device, requires_grad=True), torch.zeros((), device=device, requires_grad=True))
        for arm in ("full", "unary")
    }
    operands = {
        "full": v18._task_operands(_PairModeView(full, True), pairs, indices, *variables["full"]),
        "unary": v18._task_operands(_PairModeView(unary, False), pairs, indices, *variables["unary"]),
    }
    keys = ("objective", "deranged", "zero", "separation", "ranking", "total", "affine_loss")
    losses_exact = all(torch.equal(operands["full"][key].detach(), operands["unary"][key].detach()) for key in keys)
    named = {arm: tuple(core.named_parameters()) for arm, core in (("full", full), ("unary", unary))}
    gradients = {
        arm: torch.autograd.grad(operands[arm]["total"], tuple(value for _, value in named[arm]), allow_unused=True)
        for arm in ("full", "unary")
    }
    names_exact = tuple(name for name, _ in named["full"]) == tuple(name for name, _ in named["unary"])
    non_scorer_exact = names_exact and all(
        name.startswith(PAIR_PREFIX)
        or (left is None and right is None)
        or (left is not None and right is not None and torch.equal(left, right))
        for (name, _), left, right in zip(named["full"], gradients["full"], gradients["unary"], strict=True)
    )
    unary_scorer_none = all(
        gradient is None for (name, _), gradient in zip(named["unary"], gradients["unary"], strict=True)
        if name.startswith(PAIR_PREFIX)
    )
    full_head = {
        name: gradient is not None and bool(torch.isfinite(gradient).all().item()) and bool((gradient != 0).any().item())
        for (name, _), gradient in zip(named["full"], gradients["full"], strict=True)
        if name.startswith(PAIR_PREFIX) and name.endswith("3.weight")
    }
    report = {
        "initial_parameter_digest_exact": _model_digest(full) == _model_digest(unary),
        "task_loss_operands_exact": losses_exact,
        "non_scorer_task_gradients_exact": non_scorer_exact,
        "unary_scorer_gradients_absent": unary_scorer_none,
        "full_zero_head_first_gradient_nonzero": bool(full_head) and all(full_head.values()),
        "chronology_exact": operands["full"]["chronology_exact"] and operands["unary"]["chronology_exact"],
        "optimizer_configuration_exact": (
            _optimizer_snapshot(optimizers["full"])
            == _optimizer_snapshot(optimizers["unary"])
        ),
        "optimizer_parameter_order_exact": (
            _optimizer_parameter_order(full, optimizers["full"])
            == _optimizer_parameter_order(unary, optimizers["unary"])
            == (tuple(name for name, _ in full.named_parameters()),)
        ),
    }
    report["exact"] = all(report.values())
    return report


def _mechanics_smoke(
    full: PairedPublicRelationCreditMemoryCore,
    unary: PairedPublicRelationCreditMemoryCore,
    launch_optimizers: Mapping[str, torch.optim.Optimizer],
    pairs: Sequence[Any],
    schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    model_before = {"full": _model_digest(full), "unary": _model_digest(unary)}
    optimizer_before = {name: _optimizer_snapshot(value) for name, value in launch_optimizers.items()}
    state_before = {"full": full.state_digest(full.initial_state()), "unary": unary.state_digest(unary.initial_state())}
    rng_before = v18._rng_snapshot()
    disposable = copy.deepcopy(full)
    optimizer = torch.optim.AdamW(
        disposable.parameters(), lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS,
        eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY,
    )
    gradients: list[dict[str, bool]] = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        device = next(disposable.parameters()).device
        scale = torch.ones((), device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)
        operands = v18._task_operands(_PairModeView(disposable, True), pairs, schedule[step], scale, bias)
        operands["total"].backward()
        row = {
            name: parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            and bool((parameter.grad != 0).any().item())
            for name, parameter in disposable.named_parameters()
            if name.startswith(PAIR_PREFIX)
        }
        gradients.append(row)
        if not math.isfinite(float(torch.nn.utils.clip_grad_norm_(disposable.parameters(), v15.GRADIENT_CLIP))):
            raise RuntimeError("V19 disposable smoke gradient is non-finite")
        optimizer.step()
    del optimizer, disposable
    rng_after = v18._rng_snapshot()
    report = {
        "first_update_head_nonzero": bool(gradients[0]) and any(
            value for name, value in gradients[0].items() if name.endswith("3.weight")
        ),
        "second_update_all_scorer_nonzero": bool(gradients[1]) and all(gradients[1].values()),
        "gradient_rows": gradients,
        "launch_model_digests_exact": model_before == {"full": _model_digest(full), "unary": _model_digest(unary)},
        "launch_optimizer_states_exact": optimizer_before == {name: _optimizer_snapshot(value) for name, value in launch_optimizers.items()},
        "launch_initial_states_exact": state_before == {"full": full.state_digest(full.initial_state()), "unary": unary.state_digest(unary.initial_state())},
        "global_rng_exact": v18._rng_exact(rng_before, rng_after),
        "disposable_only": True,
    }
    report["exact"] = all(value for key, value in report.items() if key not in {"gradient_rows"})
    return report


def _architecture_evidence(core: PairedPublicRelationCreditMemoryCore, pair: Any) -> dict[str, Any]:
    inherited = v17._architecture_evidence(core, pair)
    device = next(core.parameters()).device
    episode = pair.first.to(device)
    initial = core.initial_state()
    output = core.predict(
        episode.relation_features[0:1], episode.temporal_features[0:1],
        episode.base_logits[0:1], state=initial,
    )
    zero = core.predict(
        episode.relation_features[0:1], episode.temporal_features[0:1],
        episode.base_logits[0:1], state=initial, pair_residual_enabled=False,
    )
    positive, _ = core.apply_feedback(
        initial, output, output.logits.new_tensor((1.0,)), evidence_refs="v19-arch:+1", detach_state=True,
    )
    negative, _ = core.apply_feedback(
        initial, output, output.logits.new_tensor((-1.0,)), evidence_refs="v19-arch:-1", detach_state=True,
    )
    twin = pair.second.to(device)
    occupied_state, occupied_events, _, _ = v15._write_anchors(
        _PairModeView(core, True), episode, twin, initial, arm="true", detach_state=True,
    )
    snapshot = core.capture_state(occupied_state)
    restored = core.restore_state(snapshot)
    replayed, replay_outputs = core.replay(occupied_events)
    probe_index = PROBE_INDICES[0]
    probe_args = (
        episode.relation_features[probe_index:probe_index + 1],
        episode.temporal_features[probe_index:probe_index + 1],
        episode.base_logits[probe_index:probe_index + 1],
    )
    occupied_zero = core.predict(
        *probe_args, state=occupied_state, pair_residual_enabled=False,
    )
    inherited_zero = super(PairedPublicRelationCreditMemoryCore, core).predict(
        *probe_args, state=occupied_state,
    )
    inherited_fields = (
        "logits", "base_logits", "residuals", "read_queries", "read_weights",
        "read_values", "temporal_hidden", "write_keys", "read_strengths",
        "write_strengths", "erase_gates", "observed_relations", "observed_temporal",
    )
    report = core.architecture_report()
    paired = {
        "initial_residual_zero": torch.equal(output.pair_address_residuals, torch.zeros_like(output.pair_address_residuals)),
        "residual_zero_v17_weights_exact": torch.equal(output.read_weights, zero.read_weights),
        "public_sidecar_outcome_invariant": torch.equal(positive.public_relations, negative.public_relations),
        "public_sidecar_exact_observation": torch.equal(positive.public_relations[0], output.observed_relations[0]),
        "stored_values_differ": not torch.equal(positive.values, negative.values),
        "pair_inputs_public_only": report.get("pair_scorer_inputs") == ("detached_query_relation", "detached_stored_relation"),
        "pair_residual_bound": report.get("pair_residual_bound") == 2.0,
        "metadata_inputs_absent": report.get("metadata_inputs") is False,
        "occupied_state_nonempty": occupied_state.step == len(ACQUISITION_INDICES),
        "occupied_residual_zero_direct_v17_exact": all(
            torch.equal(getattr(occupied_zero, name), getattr(inherited_zero, name))
            for name in inherited_fields
        ),
        "occupied_pair_residual_exact_zero": torch.equal(
            occupied_zero.pair_address_residuals,
            torch.zeros_like(occupied_zero.pair_address_residuals),
        ),
        "sidecar_capture_restore_exact": (
            core.state_digest(restored) == core.state_digest(occupied_state)
        ),
        "sidecar_replay_exact": (
            len(replay_outputs) == len(occupied_events)
            and core.state_digest(replayed) == core.state_digest(occupied_state)
        ),
    }
    return {**inherited, "pair_public_relation_invariance": paired, "architecture_report": report, "exact": inherited["exact"] and all(paired.values())}


def _paired_fit(
    full: PairedPublicRelationCreditMemoryCore,
    unary: PairedPublicRelationCreditMemoryCore,
    pairs: Sequence[Any],
    schedule: Sequence[Sequence[int]],
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> tuple[dict[str, Any], tuple[float, float], tuple[float, float]]:
    if len(pairs) != TRAIN_PAIRS or len(schedule) != TRAIN_UPDATES:
        raise ValueError("V19 fit requires the frozen V15 corpus and schedule")
    device = next(full.parameters()).device
    affine = {
        arm: (torch.nn.Parameter(torch.ones((), device=device)), torch.nn.Parameter(torch.zeros((), device=device)))
        for arm in ("full", "unary")
    }
    affine_optimizers = {
        arm: torch.optim.AdamW(affine[arm], lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS, eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY)
        for arm in affine
    }
    views = {"full": _PairModeView(full, True), "unary": _PairModeView(unary, False)}
    updates = []
    for update_index, indices in enumerate(schedule):
        rows = {}
        for arm in ("full", "unary"):
            optimizers[arm].zero_grad(set_to_none=True)
            affine_optimizers[arm].zero_grad(set_to_none=True)
            rows[arm] = v18._task_operands(views[arm], pairs, indices, *affine[arm])
        gradients = {}
        for arm, core in (("full", full), ("unary", unary)):
            loss = rows[arm]["total"]
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("V19 paired task loss is non-finite")
            loss.backward()
            scorer = [parameter.grad for name, parameter in core.named_parameters() if name.startswith(PAIR_PREFIX)]
            if arm == "unary" and any(value is not None for value in scorer):
                raise RuntimeError("V19 unary scorer received a gradient")
            finite = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
            if not finite or any(not bool(value.isfinite().all().item()) for value in finite):
                raise RuntimeError("V19 paired gradients are absent/non-finite")
            gradients[arm] = float(torch.nn.utils.clip_grad_norm_(core.parameters(), v15.GRADIENT_CLIP).detach())
            optimizers[arm].step()
            rows[arm]["affine_loss"].backward()
            affine_optimizers[arm].step()
        updates.append({
            "update": update_index,
            "pair_indices": list(indices),
            **{
                arm: {
                    "true_probe_loss": float(rows[arm]["objective"].detach()),
                    "deranged_probe_loss": float(rows[arm]["deranged"].detach()),
                    "zero_probe_loss": float(rows[arm]["zero"].detach()),
                    "separation_loss": float(rows[arm]["separation"].detach()),
                    "twin_ranking_loss": float(rows[arm]["ranking"].detach()),
                    "task_total_loss": float(rows[arm]["total"].detach()),
                    "gradient_norm": gradients[arm],
                }
                for arm in ("full", "unary")
            },
            "chronology_exact": rows["full"]["chronology_exact"] and rows["unary"]["chronology_exact"],
            "true_deranged_paths_differentiable": all(rows[arm]["true_deranged_paths_differentiable"] for arm in ("full", "unary")),
            "zero_baseline_detached": all(rows[arm]["zero_baseline_detached"] for arm in ("full", "unary")),
        })
    exposure = dict(sorted(Counter(index for row in schedule for index in row).items()))
    return ({
        "completed_updates": len(updates), "updates": updates,
        "exposure_counts": exposure, "loss_weights": list(v15.LOSS_WEIGHTS),
        "auxiliary_objective": False,
    }, tuple(float(value.detach()) for value in affine["full"]), tuple(float(value.detach()) for value in affine["unary"]))


def _branch_metrics(logits: Sequence[float], labels: Sequence[float]) -> dict[str, float]:
    return {
        "balanced_accuracy": v15._balanced_accuracy(logits, labels),
        "mean_probe_nll": sum(float(torch.nn.functional.softplus(torch.tensor(-y * z))) for y, z in zip(labels, logits, strict=True)) / len(labels),
    }


def _paired_controls(
    core: PairedPublicRelationCreditMemoryCore,
    pairs: Sequence[Any],
    matched_expected: Mapping[str, Any],
) -> dict[str, Any]:
    if len(pairs) != DEVELOPMENT_PAIRS:
        raise ValueError("V19 controls require 48 development pairs")
    device = next(core.parameters()).device
    logits = {name: [] for name in ("matched", "residual_zero", "sidecar_deranged", "pair_query_swap")}
    labels: list[float] = []
    unchanged = {name: True for name in logits}
    non_target_exact = True
    query_swap_output_exact = {
        name: True
        for name in (
            "read_queries", "read_strengths", "semantic_content_logits",
            "observed_temporal", "base_logits",
        )
    }
    query_swap_state_exact = True
    query_swap_raw_source_exact = True
    target_labels = {name: [] for name in ("matched", "pair_query_swap")}
    exposure = {
        name: {axis: Counter() for axis in ("family", "transition", "probe_position")}
        for name in ("matched", "pair_query_swap")
    }
    rotate_exact = True
    query_changed = True
    changed_top = 0
    nonzero_rows = 0
    residual_values: list[float] = []
    probe_rows = 0
    source_positions: list[int] = []
    target_positions: list[int] = []
    model_before = _model_digest(core)
    with torch.no_grad():
        for pair in pairs:
            for side in ("first", "second"):
                episode = getattr(pair, side).to(device)
                twin = getattr(pair, "second" if side == "first" else "first").to(device)
                state, _, _, _ = v15._write_anchors(
                    _PairModeView(core, True), episode, twin, core.initial_state(), arm="true", detach_state=True,
                )
                source_digest = core.state_digest(state)
                source_non_target = _non_sidecar_state_digest(state)
                deranged = _rotate_public_sidecars(state)
                occupied = torch.nonzero(state.usage > 0, as_tuple=False).flatten()
                rotate_exact &= (
                    core.state_digest(deranged) != source_digest
                    and _non_sidecar_state_digest(deranged) == source_non_target
                    and bool(torch.equal(deranged.public_relations[occupied], state.public_relations[occupied.roll(-1)]))
                )
                for target, source in zip(PROBE_INDICES, QUERY_SWAP, strict=True):
                    target_positions.append(target); source_positions.append(source)
                    query_changed &= not torch.equal(episode.relation_features[target], episode.relation_features[source])
                    args = (
                        episode.relation_features[target:target + 1],
                        episode.temporal_features[target:target + 1],
                        episode.base_logits[target:target + 1],
                    )
                    before = {
                        "matched": core.state_digest(state),
                        "residual_zero": core.state_digest(state),
                        "pair_query_swap": core.state_digest(state),
                        "sidecar_deranged": core.state_digest(deranged),
                    }
                    matched = core.predict(*args, state=state, pair_residual_enabled=True)
                    zero = core.predict(*args, state=state, pair_residual_enabled=False)
                    shifted = core.predict(*args, state=deranged, pair_residual_enabled=True)
                    swapped = core.predict(
                        *args, state=state, pair_residual_enabled=True,
                        pair_query_features=episode.relation_features[source:source + 1],
                    )
                    outputs = {"matched": matched, "residual_zero": zero, "sidecar_deranged": shifted, "pair_query_swap": swapped}
                    states = {"matched": state, "residual_zero": state, "pair_query_swap": state, "sidecar_deranged": deranged}
                    for name, output in outputs.items():
                        logits[name].append(float(output.logits.item()))
                        unchanged[name] &= before[name] == core.state_digest(states[name])
                    label = float(episode.outcomes[target].item())
                    labels.append(label)
                    for name in ("matched", "pair_query_swap"):
                        target_labels[name].append(label)
                        exposure[name]["family"][episode.generator_family] += 1
                        exposure[name]["transition"][episode.transition_group] += 1
                        exposure[name]["probe_position"][target] += 1
                    for name in query_swap_output_exact:
                        query_swap_output_exact[name] &= torch.equal(
                            getattr(matched, name), getattr(swapped, name),
                        )
                    query_swap_raw_source_exact &= (
                        torch.equal(
                            swapped.pair_query_relations,
                            episode.relation_features[source:source + 1],
                        )
                        and torch.equal(
                            matched.pair_query_relations,
                            episode.relation_features[target:target + 1],
                        )
                    )
                    query_swap_state_exact &= all(
                        torch.equal(getattr(states["matched"], name), getattr(states["pair_query_swap"], name))
                        for name in ("keys", "values", "usage", "acquisition", "public_relations")
                    ) and states["matched"].step == states["pair_query_swap"].step
                    changed_top += int(int(matched.read_weights.argmax()) != int(zero.read_weights.argmax()))
                    active = matched.pair_address_residuals[0, occupied]
                    nonzero_rows += int(bool((active.abs() > 1.0e-8).any().item()))
                    residual_values.extend(float(value) for value in active.abs().cpu())
                    probe_rows += 1
                non_target_exact &= source_digest == core.state_digest(state) and source_non_target == _non_sidecar_state_digest(deranged)
    metrics = {name: _branch_metrics(values, labels) for name, values in logits.items()}
    matched_exact = (
        abs(metrics["matched"]["balanced_accuracy"] - float(matched_expected["matched_balanced_accuracy"])) <= 1.0e-12
        and abs(metrics["matched"]["mean_probe_nll"] - float(matched_expected["mean_nll"]["matched"])) <= 1.0e-6
    )
    query_multiset_exact = Counter(source_positions) == Counter(target_positions)
    family_exposure_exact = exposure["matched"]["family"] == exposure["pair_query_swap"]["family"]
    transition_exposure_exact = exposure["matched"]["transition"] == exposure["pair_query_swap"]["transition"]
    probe_exposure_exact = exposure["matched"]["probe_position"] == exposure["pair_query_swap"]["probe_position"]
    labels_exact = target_labels["matched"] == target_labels["pair_query_swap"] == labels
    model_after = _model_digest(core)
    return {
        **metrics,
        "branch_state_immutable": unchanged,
        "source_state_exact": non_target_exact,
        "sidecar_rotate_by_one_exact": rotate_exact,
        "sidecar_derangement_zero_fixed_points": rotate_exact,
        "query_swap_permutation": list(QUERY_SWAP),
        "query_swap_zero_fixed_points": all(left != right for left, right in zip(PROBE_INDICES, QUERY_SWAP, strict=True)),
        "query_rows_all_changed": query_changed,
        "query_multiset_exact": query_multiset_exact,
        "pair_query_swap_non_target_outputs_exact": query_swap_output_exact,
        "pair_query_swap_state_exact": query_swap_state_exact,
        "pair_query_swap_raw_source_exact": query_swap_raw_source_exact,
        "pair_query_swap_target_labels_exact": labels_exact,
        "family_exposure_exact": family_exposure_exact,
        "transition_exposure_exact": transition_exposure_exact,
        "probe_position_exposure_exact": probe_exposure_exact and query_multiset_exact,
        "exposure_counters": {
            branch: {axis: dict(sorted(counter.items(), key=lambda item: str(item[0]))) for axis, counter in axes.items()}
            for branch, axes in exposure.items()
        },
        "matched_v17_operand_exact": matched_exact,
        "probe_rows": probe_rows,
        "nonzero_pair_residual_fraction": nonzero_rows / probe_rows,
        "changed_top_read_fraction": changed_top / probe_rows,
        "mean_absolute_pair_residual": sum(residual_values) / len(residual_values),
        "maximum_absolute_pair_residual": max(residual_values),
        "no_feedback_write_or_optimizer_step": (
            all(unchanged.values()) and model_before == model_after
        ),
    }


def _evaluate(
    core: PairedPublicRelationCreditMemoryCore,
    pairs: Sequence[Any],
    affine: tuple[float, float],
    *,
    pair_enabled: bool,
    include_controls: bool,
) -> dict[str, Any]:
    view = _PairModeView(core, pair_enabled)
    with _v19_control_scope():
        metrics = v15._evaluate(view, pairs, affine)
    metrics["probe_relation_query_swap"] = v16._relation_query_swap_control(view, pairs)
    metrics["corresponding_anchor_read_mass"] = v17._corresponding_anchor_read_report(view, pairs)
    if include_controls:
        metrics["paired_public_relation_controls"] = _paired_controls(core, pairs, metrics["state_controls"])
        metrics["residual_zero_corresponding_anchor_read_mass"] = v17._corresponding_anchor_read_report(_PairModeView(core, False), pairs)
    return metrics


def _protocol(training: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "exact_training_schedule": training["completed_updates"] == TRAIN_UPDATES and set(training["exposure_counts"].values()) == {2},
        "all_training_gradients_finite": all(math.isfinite(row[arm]["gradient_norm"]) for row in training["updates"] for arm in ("full", "unary")),
        "true_deranged_paths_differentiable": all(row["true_deranged_paths_differentiable"] for row in training["updates"]),
        "zero_baseline_detached": all(row["zero_baseline_detached"] for row in training["updates"]),
        "predict_before_feedback_chronology": all(row["chronology_exact"] for row in training["updates"]),
        "eight_pair_state_horizon": metrics["maximum_state_step"] == PAIRS_PER_UPDATE * len(ACQUISITION_INDICES),
        "no_auxiliary_objective": training["auxiliary_objective"] is False,
    }


def _lesion_effect(control: Mapping[str, float], matched: Mapping[str, float]) -> bool:
    return bool(
        matched["balanced_accuracy"] - control["balanced_accuracy"] >= 0.10
        or control["mean_probe_nll"] - matched["mean_probe_nll"] >= 0.02
    )


def _component_gate(
    full: Mapping[str, Any], unary: Mapping[str, Any], architecture: Mapping[str, Any],
    preflight: Mapping[str, Any], *, identity_exact: bool,
) -> bool:
    try:
        left = full["corresponding_anchor_read_mass"]
        right = unary["corresponding_anchor_read_mass"]
        zero = full["residual_zero_corresponding_anchor_read_mass"]
        controls = full["paired_public_relation_controls"]
        matched = controls["matched"]
        mechanics = (
            preflight["exact"] is True and identity_exact is True
            and all(controls["branch_state_immutable"].values())
            and controls["source_state_exact"] is True
            and controls["sidecar_rotate_by_one_exact"] is True
            and controls["sidecar_derangement_zero_fixed_points"] is True
            and controls["query_swap_zero_fixed_points"] is True
            and controls["query_rows_all_changed"] is True
            and controls["query_multiset_exact"] is True
            and all(controls["pair_query_swap_non_target_outputs_exact"].values())
            and controls["pair_query_swap_state_exact"] is True
            and controls["pair_query_swap_raw_source_exact"] is True
            and controls["pair_query_swap_target_labels_exact"] is True
            and controls["family_exposure_exact"] is True
            and controls["transition_exposure_exact"] is True
            and controls["probe_position_exposure_exact"] is True
            and controls["matched_v17_operand_exact"] is True
            and controls["no_feedback_write_or_optimizer_step"] is True
        )
        return bool(
            mechanics
            and left["top1_accuracy"] - right["top1_accuracy"] >= 0.10
            and left["mean_matching_mass"] - right["mean_matching_mass"] >= 0.05
            and left["mean_matching_minus_competitor_margin"] - right["mean_matching_minus_competitor_margin"] >= 0.05
            and full["arms"]["true"]["balanced_accuracy"] >= unary["arms"]["true"]["balanced_accuracy"]
            and full["arms"]["true"]["mean_probe_nll"] <= unary["arms"]["true"]["mean_probe_nll"] + 0.02
            and v17._component_gate(full, architecture, identifiability_exact=True, identity_exact=identity_exact)
            and all(_lesion_effect(controls[name], matched) for name in ("residual_zero", "sidecar_deranged", "pair_query_swap"))
            and zero["top1_accuracy"] - right["top1_accuracy"] < 0.02
            and zero["mean_matching_mass"] - right["mean_matching_mass"] < 0.02
            and controls["nonzero_pair_residual_fraction"] >= 0.50
            and controls["changed_top_read_fraction"] >= 0.10
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _full_gate(
    full: Mapping[str, Any], unary: Mapping[str, Any], architecture: Mapping[str, Any],
    preflight: Mapping[str, Any], *, identity_exact: bool,
) -> bool:
    return bool(
        _component_gate(full, unary, architecture, preflight, identity_exact=identity_exact)
        and v17._full_gate(full, architecture, identifiability_exact=True, identity_exact=identity_exact)
    )


def _run_train_development(args: argparse.Namespace) -> None:
    full_path, unary_path = Path(args.full_checkpoint), Path(args.unary_checkpoint)
    result_path = Path(args.development_result)
    if any(path.exists() for path in (full_path, unary_path, result_path)):
        raise FileExistsError("V19 train-development identity is already consumed")
    sources_before = _source_hashes()
    donor_pins = {
        "phase6_paired_comparator_donor": PHASE6_PAIRED_DONOR_SHA256,
        "phase6_oml_relation_donor": PHASE6_OML_DONOR_SHA256,
        "v11_public_sidecar_donor": V11_SIDECAR_DONOR_SHA256,
    }
    if (
        sources_before["leaf"] != LEAF_SHA256
        or sources_before["v18_runner"] != V18_RUNNER_SHA256
        or any(sources_before.get(name) != digest for name, digest in donor_pins.items())
    ):
        raise RuntimeError("V19 leaf/V18/donor source identity changed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device, started = v16._prepare_angler_device(args.device)
    consumed = _v18_evidence(Path(args.v18_result), Path(args.v18_full_checkpoint), Path(args.v18_unary_checkpoint))
    v14_evidence = v15._v14_structure_evidence(Path(args.v14_result))
    frozen, v13_checkpoint = v15._load_frozen_v13(Path(args.v13_checkpoint), Path(args.v13_result), device)
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != v13_checkpoint["qwen_digest"]:
        raise RuntimeError("V19 Qwen does not match frozen V13 foundation")
    corpus = build_paired_latent_contingency_credit_v15()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V19 final opened despite absent final-phase authority")
    schedule = v15._training_schedule(corpus)
    train = v15._encode_pairs(qwen, frozen, corpus.train)
    development = v15._encode_pairs(qwen, frozen, corpus.development)
    full = PairedPublicRelationCreditMemoryCore(
        temporal_width=TEMPORAL_WIDTH, pair_residual_enabled_by_default=True,
    ).to(device=device, dtype=torch.float32)
    unary = PairedPublicRelationCreditMemoryCore(
        temporal_width=TEMPORAL_WIDTH, pair_residual_enabled_by_default=False,
    ).to(device=device, dtype=torch.float32)
    unary.load_state_dict(full.state_dict(), strict=True)
    launch_optimizers = {
        arm: torch.optim.AdamW(core.parameters(), lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS, eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY)
        for arm, core in (("full", full), ("unary", unary))
    }
    identifiability = v15._identifiability_preflight(full, train, development, schedule)
    architecture = _architecture_evidence(full, train[0])
    parity = _task_parity_preflight(full, unary, train, schedule[0], launch_optimizers)
    smoke = _mechanics_smoke(full, unary, launch_optimizers, train, schedule)
    preflight = {"identifiability": identifiability, "architecture": architecture, "task_parity": parity, "mechanics_smoke": smoke}
    preflight["exact"] = bool(identifiability["exact"] and architecture["exact"] and parity["exact"] and smoke["exact"])
    if preflight["exact"] is not True:
        raise RuntimeError("V19 identifiability/architecture/mechanics preflight failed")
    initial_digest = _model_digest(full)
    training, full_affine, unary_affine = _paired_fit(full, unary, train, schedule, launch_optimizers)
    v15._enforce_resources(started, device)
    full_metrics = _evaluate(full, development, full_affine, pair_enabled=True, include_controls=True)
    unary_metrics = _evaluate(unary, development, unary_affine, pair_enabled=False, include_controls=False)
    full_metrics["protocol_invariants"] = _protocol(training, full_metrics)
    unary_metrics["protocol_invariants"] = _protocol(training, unary_metrics)
    v15._enforce_resources(started, device)
    sources_after = _source_hashes()
    identity_checks = {
        "qwen_before": qwen_before, "qwen_after": foundation_tensor_digest(qwen.model),
        "v13_before": frozen_before, "v13_after": _model_digest(frozen),
        "sources_before": sources_before, "sources_after": sources_after,
        "reasoning_export_exact": sources_after["v18_reasoning_export"] == REASONING_EXPORT_SHA256,
        "final_partition_sealed": True,
    }
    identity_checks["exact"] = bool(
        qwen_before == identity_checks["qwen_after"]
        and frozen_before == identity_checks["v13_after"]
        and sources_before == sources_after
        and identity_checks["reasoning_export_exact"]
        and identity_checks["final_partition_sealed"]
    )
    component = _component_gate(full_metrics, unary_metrics, architecture, preflight, identity_exact=identity_checks["exact"])
    durable = _full_gate(full_metrics, unary_metrics, architecture, preflight, identity_exact=identity_checks["exact"])
    classification = (
        "FULL_DURABLE_PAIRED_PUBLIC_RELATION_CREDIT_SUPPORTED" if durable else
        "PAIRED_PUBLIC_RELATION_COMPARISON_SUPPORTED" if component else
        "DEVELOPMENT_NOT_SUPPORTED"
    )
    environment = v18._environment_report(device)
    common = {
        "identity": IDENTITY, "seed": SEED, "corpus_id": CORPUS_ID,
        "completed_updates": TRAIN_UPDATES, "temporal_width": TEMPORAL_WIDTH,
        "initial_core_digest": initial_digest, "source_hashes": sources_before,
        "qwen_digest": qwen_before, "v13_digest": frozen_before,
        "v18_result_sha256": V18_RESULT_SHA256,
        "frozen_identities": {
            "v18_full_checkpoint_sha256": V18_FULL_CHECKPOINT_SHA256,
            "v18_unary_checkpoint_sha256": V18_UNARY_CHECKPOINT_SHA256,
            "v18_runner_sha256": V18_RUNNER_SHA256,
            "v18_test_sha256": V18_TEST_SHA256,
            "v17_core_sha256": V17_CORE_SHA256,
            "phase6_paired_donor_sha256": PHASE6_PAIRED_DONOR_SHA256,
            "phase6_oml_donor_sha256": PHASE6_OML_DONOR_SHA256,
            "v11_sidecar_donor_sha256": V11_SIDECAR_DONOR_SHA256,
            "reasoning_export_sha256": REASONING_EXPORT_SHA256,
        },
        "preflight": preflight, "architecture_evidence": architecture,
        "identity_checks": identity_checks, "environment": environment,
    }
    full_checkpoint = {
        **common, "arm": "full", "core_digest": _model_digest(full),
        "pair_residual_enabled_by_default": True,
        "core_state": {name: value.detach().cpu() for name, value in full.state_dict().items()},
        "affine_calibrator": list(full_affine), "development_metrics": full_metrics,
    }
    unary_checkpoint = {
        **common, "arm": "unary", "core_digest": _model_digest(unary),
        "pair_residual_enabled_by_default": False,
        "core_state": {name: value.detach().cpu() for name, value in unary.state_dict().items()},
        "affine_calibrator": list(unary_affine), "development_metrics": unary_metrics,
    }
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    result: dict[str, Any] = {
        "identity": IDENTITY, "phase": "train-development", "classification": classification,
        "paired_public_relation_comparison_supported": component,
        "full_durable_supported": durable, "development_authorized": False,
        "successor_final_leaf_eligible": durable, "final_partition_sealed": True,
        "corpus_id": CORPUS_ID, "seed": SEED, "preflight": preflight,
        "architecture_evidence": architecture, "training": training,
        "development_metrics": {"full": full_metrics, "unary": unary_metrics},
        "identity_checks": identity_checks, "source_hashes": sources_before,
        "environment": environment,
        "frozen_evidence": {
            "v18_result_sha256": V18_RESULT_SHA256,
            "v18_full_checkpoint_sha256": V18_FULL_CHECKPOINT_SHA256,
            "v18_unary_checkpoint_sha256": V18_UNARY_CHECKPOINT_SHA256,
            "v18_classification": consumed["classification"],
            "v14_structure_replicated": v14_evidence["structure_replicated"],
            "donor_pins": donor_pins,
        },
        "frozen_compute": {
            "updates": TRAIN_UPDATES, "pairs_per_update": PAIRS_PER_UPDATE,
            "train_pairs": TRAIN_PAIRS, "development_pairs": DEVELOPMENT_PAIRS,
            "loss_weights": list(v15.LOSS_WEIGHTS), "auxiliary_objective": False,
            "pair_residual_bound": 2.0, "dtype": "float32", "autocast": False,
            "tf32": False, "qwen_device": "cuda:0", "v13_v19_device": "cuda:1",
        },
        "elapsed_seconds": elapsed, "peak_angler_gpu_bytes": peak,
    }
    if not identity_checks["exact"]:
        raise RuntimeError("V19 source/foundation drift during execution")
    v15._atomic_torch(full_path, full_checkpoint)
    v15._atomic_torch(unary_path, unary_checkpoint)
    result["final_seal"] = _final_checkpoint_seal(
        full_path,
        unary_path,
        expected_common=common,
        full_core_digest=full_checkpoint["core_digest"],
        unary_core_digest=unary_checkpoint["core_digest"],
        full_metrics=full_metrics,
        unary_metrics=unary_metrics,
        full_affine=full_affine,
        unary_affine=unary_affine,
    )
    result["full_checkpoint_sha256"] = result["final_seal"]["arms"]["full"]["checkpoint_sha256"]
    result["unary_checkpoint_sha256"] = result["final_seal"]["arms"]["unary"]["checkpoint_sha256"]
    v15._atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "result": str(result_path)}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev",), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--v13-checkpoint", default="/opt/angler/results/structure-protected-oml-natural-trace-v13.pt")
    parser.add_argument("--v13-result", default="/opt/angler/results/structure-protected-oml-natural-trace-v13-development.json")
    parser.add_argument("--v14-result", default="/opt/angler/results/structure-keyed-persistent-credit-v14-development.json")
    parser.add_argument("--v18-result", default="/opt/angler/results/barlow-semantic-invariance-v18-development.json")
    parser.add_argument("--v18-full-checkpoint", default="/opt/angler/results/barlow-semantic-invariance-v18-full.pt")
    parser.add_argument("--v18-unary-checkpoint", default="/opt/angler/results/barlow-semantic-invariance-v18-objective-off.pt")
    parser.add_argument("--full-checkpoint", default="/opt/angler/results/paired-public-relation-credit-v19-full.pt")
    parser.add_argument("--unary-checkpoint", default="/opt/angler/results/paired-public-relation-credit-v19-unary.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/paired-public-relation-credit-v19-development.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _run_train_development(args)


if __name__ == "__main__":
    main()
