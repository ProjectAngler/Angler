"""V18 negative-free public-relation semantic-invariance experiment.

The only learned-mechanism change from V17 is a Barlow Twins objective over
two frozen, random feature-mask views of canonical public relation rows.  The
paired objective-off arm retains the exact V17 task learner and evaluator.
V18 deliberately owns no final-evaluation phase.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import replace
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from angler.reasoning.shared_semantic_metric_credit_memory import (
    SharedSemanticMetricCreditMemoryCore,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.paired_latent_contingency_credit_v15 import (
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
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import shared_semantic_metric_credit_v17 as v17
from experiments.runners import strict_key_value_credit_v16 as v16


IDENTITY = "angler.barlow-semantic-invariance.v18-first-result"
SEED = 2026083118
TEMPORAL_WIDTH = v15.TEMPORAL_WIDTH
KEEP_PROBABILITY = 0.80
BARLOW_OFF_DIAGONAL = 0.0051
BARLOW_WEIGHT = 1.0
STANDARDIZATION_EPSILON = 1.0e-4
ROWS_PER_UPDATE = PAIRS_PER_UPDATE * 6
CODE_WIDTH = 32
MASK_SHAPE = (TRAIN_UPDATES, 2, ROWS_PER_UPDATE, 64)
MASK_RETAINED_COUNT = 472284
MASK_SHA256 = "8478D8C3883CA298FB3349B57DFA8FD2B8A8D01AF4F9FED94DB7C6BC46C2156D"

LEAF_SHA256 = "4F8F3E369636CC8DEA0976A1A10691704CC9084F1EBF7A765388435DD64851C8"
V17_RESULT_SHA256 = "C74AA58CA0E78702366E024F6BFB928B94D5E627241986001133E8112DF72894"
V17_CHECKPOINT_SHA256 = "5AA492D38555AFEBCDBB674EA31ECFA9474E0A8A25BF5AE579F5205FB989D182"
V17_CORE_SHA256 = "A25D42E9A4B8697FC441FC4C55543BFABD661AD5BD7756C4BF87EA80EAB14790"
V17_RUNNER_SHA256 = "17AA47B7A37FE54A01481AEF5B875F0B85CF805937ED6BFE35EF784F895A1633"
V17_TEST_SHA256 = "0A48D31F9418C8BF85B58915EB37757E040ADDDC68757CC8D71D3CF527D5C82A"
REASONING_EXPORT_SHA256 = "B06E4F87620974FD5C2F2AD75A433DBE55A9CBA799B0F442C5FE9A51785D5798"


def _sha256(path: str | Path) -> str:
    return v15._sha256(path).upper()


def _model_digest(model: torch.nn.Module) -> str:
    return v15._model_digest(model).upper()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inherited = {f"v17_{key}": value.upper() for key, value in v17._source_hashes().items()}
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "runner_test": _sha256(root / "tests/unit/experiments/test_barlow_semantic_invariance_v18.py"),
        "leaf": _sha256(root / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-BARLOW-SEMANTIC-INVARIANCE-V18-001.md"),
        "v17_runner_test": _sha256(root / "tests/unit/experiments/test_shared_semantic_metric_credit_v17.py"),
        "reasoning_export": _sha256(root / "src/angler/reasoning/__init__.py"),
        **inherited,
    }


def _environment_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or device.index != 1:
        raise RuntimeError("V18 requires exact Angler placement on cuda:1")
    properties = torch.cuda.get_device_properties(device)
    libraries = {}
    for name in ("numpy", "transformers", "safetensors"):
        try:
            libraries[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            libraries[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": str(device), "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "libraries": libraries,
    }


def _v17_evidence(result_path: Path, checkpoint_path: Path) -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if (
        _sha256(result_path) != V17_RESULT_SHA256
        or _sha256(checkpoint_path) != V17_CHECKPOINT_SHA256
        or _sha256(root / "src/angler/reasoning/shared_semantic_metric_credit_memory.py") != V17_CORE_SHA256
        or _sha256(root / "experiments/runners/shared_semantic_metric_credit_v17.py") != V17_RUNNER_SHA256
        or _sha256(root / "tests/unit/experiments/test_shared_semantic_metric_credit_v17.py") != V17_TEST_SHA256
        or _sha256(root / "src/angler/reasoning/__init__.py") != REASONING_EXPORT_SHA256
        or _sha256(root / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-BARLOW-SEMANTIC-INVARIANCE-V18-001.md") != LEAF_SHA256
    ):
        raise RuntimeError("V18 frozen V17/leaf/export identity changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current = v17._source_hashes()
    if (
        current.get("runner") != V17_RUNNER_SHA256
        or current.get("shared_metric_core") != V17_CORE_SHA256
        or result.get("identity") != v17.IDENTITY
        or result.get("phase") != "train-development"
        or result.get("classification") != "DEVELOPMENT_NOT_SUPPORTED"
        or result.get("development_authorized") is not False
        or result.get("checkpoint_sha256") != V17_CHECKPOINT_SHA256
        or result.get("source_hashes") != current
        or checkpoint.get("identity") != v17.IDENTITY
        or checkpoint.get("source_hashes") != current
    ):
        raise RuntimeError("consumed V17 evidence identity/source chain changed")
    return result


def _rng_snapshot() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    cuda = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return torch.random.get_rng_state().clone(), tuple(value.clone() for value in cuda)


def _rng_exact(left: tuple[torch.Tensor, tuple[torch.Tensor, ...]], right: tuple[torch.Tensor, tuple[torch.Tensor, ...]]) -> bool:
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True)
    )


def _mask_digest(masks: torch.Tensor) -> str:
    if masks.device.type != "cpu" or masks.dtype != torch.uint8 or not masks.is_contiguous():
        raise ValueError("V18 mask schedule must be contiguous CPU uint8")
    return hashlib.sha256(masks.numpy().tobytes(order="C")).hexdigest().upper()


def _precompute_mask_schedule() -> tuple[torch.Tensor, dict[str, Any]]:
    before = _rng_snapshot()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    draw = torch.rand(MASK_SHAPE, dtype=torch.float32, device="cpu", generator=generator)
    masks = (draw < KEEP_PROBABILITY).to(dtype=torch.uint8).contiguous()
    del draw
    after = _rng_snapshot()
    retained = int(masks.sum().item())
    digest = _mask_digest(masks)
    exact = (
        tuple(masks.shape) == MASK_SHAPE
        and retained == MASK_RETAINED_COUNT
        and digest == MASK_SHA256
        and _rng_exact(before, after)
    )
    if not exact:
        raise RuntimeError(
            f"V18 mask identity mismatch: shape={tuple(masks.shape)} retained={retained} hash={digest}"
        )
    return masks, {
        "shape": list(MASK_SHAPE), "dtype": "uint8", "device": "cpu",
        "retained_count": retained, "sha256": digest,
        "global_rng_unchanged": True, "single_float32_draw": True,
        "draw_order": "update,view,row,feature", "exact": True,
    }


def _mask_schedule_identity(masks: torch.Tensor) -> dict[str, Any]:
    valid_layout = (
        masks.device.type == "cpu" and masks.dtype == torch.uint8
        and masks.is_contiguous() and tuple(masks.shape) == MASK_SHAPE
    )
    retained = int(masks.sum().item()) if valid_layout else -1
    digest = _mask_digest(masks) if valid_layout else "INVALID"
    return {
        "shape_exact": tuple(masks.shape) == MASK_SHAPE,
        "layout_exact": valid_layout,
        "retained_count": retained,
        "retained_count_exact": retained == MASK_RETAINED_COUNT,
        "sha256": digest,
        "sha256_exact": digest == MASK_SHA256,
        "exact": bool(
            valid_layout and retained == MASK_RETAINED_COUNT and digest == MASK_SHA256
        ),
    }


def _canonical_public_event(event: Any) -> tuple[bytes, bytes, str]:
    payload = event.to_prediction_payload()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    binary = hashlib.sha256(encoded).digest()
    return encoded, binary, binary.hex().upper()


def _public_relation_batch(
    public_pairs: Sequence[Any], encoded_pairs: Sequence[v15.EncodedTwinPair],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Canonical twin-only deduplication without outcome/evaluator selection."""

    if len(public_pairs) != PAIRS_PER_UPDATE or len(encoded_pairs) != PAIRS_PER_UPDATE:
        raise ValueError("V18 auxiliary batch requires eight paired public mechanisms")
    seen: dict[str, tuple[bytes, torch.Tensor]] = {}
    rows: list[torch.Tensor] = []
    ordered_digests: list[str] = []
    ordered_binary: list[bytes] = []
    feature_collisions = 0
    for public_pair, encoded_pair in zip(public_pairs, encoded_pairs, strict=True):
        for side in ("first", "second"):
            public_episode = getattr(public_pair, side).public
            encoded_episode = getattr(encoded_pair, side)
            if len(public_episode.events) != 6 or encoded_episode.relation_features.shape != (6, 64):
                raise RuntimeError("V18 public/encoded event alignment changed")
            for index, event in enumerate(public_episode.events):
                payload, binary, digest = _canonical_public_event(event)
                row = encoded_episode.relation_features[index].detach()
                canonical_row = row.to(device="cpu", dtype=torch.float32).contiguous()
                if digest in seen:
                    prior_payload, prior_row = seen[digest]
                    if payload != prior_payload or not torch.equal(canonical_row, prior_row):
                        raise RuntimeError("V18 public-event digest collision or twin relation mismatch")
                    continue
                for _, existing_row in seen.values():
                    feature_collisions += int(torch.equal(canonical_row, existing_row))
                seen[digest] = (payload, canonical_row)
                rows.append(row)
                ordered_digests.append(digest)
                ordered_binary.append(binary)
    if len(rows) != ROWS_PER_UPDATE:
        raise RuntimeError(f"V18 auxiliary dedup yielded {len(rows)} rows, expected 48")
    return torch.stack(rows).detach(), {
        "row_count": len(rows),
        "ordered_event_digests": ordered_digests,
        "ordered_digest_chain_sha256": hashlib.sha256(b"".join(ordered_binary)).hexdigest().upper(),
        "deduplicated_by": "canonical_public_prediction_payload_sha256",
        "distinct_feature_collisions_retained": feature_collisions,
        "exact": True,
    }


def _standardize(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if codes.dtype != torch.float32 or codes.shape != (ROWS_PER_UPDATE, CODE_WIDTH):
        raise ValueError("V18 Barlow codes must be float32 [48,32]")
    if not bool(torch.isfinite(codes).all().item()):
        raise RuntimeError("V18 Barlow input is non-finite")
    mean = codes.mean(dim=0)
    centered = codes - mean
    variance = (centered * centered).mean(dim=0)
    standardized = centered / torch.sqrt(variance + STANDARDIZATION_EPSILON)
    if not all(bool(torch.isfinite(value).all().item()) for value in (mean, variance, standardized)):
        raise RuntimeError("V18 Barlow standardization is non-finite")
    return standardized, variance


def _effective_rank(codes: torch.Tensor) -> float:
    centered = codes - codes.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if float(total.item()) == 0.0:
        return 0.0
    probabilities = singular / total
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return float(torch.exp(entropy).detach().item())


def _barlow_objective(
    core: SharedSemanticMetricCreditMemoryCore,
    rows: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if rows.shape != (ROWS_PER_UPDATE, 64) or masks.shape != (2, ROWS_PER_UPDATE, 64):
        raise ValueError("V18 Barlow row/mask shape changed")
    if masks.dtype != torch.uint8:
        raise TypeError("V18 Barlow masks must be uint8")
    core_device = next(core.parameters()).device
    rows = rows.to(device=core_device, dtype=torch.float32)
    mask = masks.to(device=core_device, dtype=torch.float32)
    first = core.semantic_codes(rows * mask[0] / KEEP_PROBABILITY)
    second = core.semantic_codes(rows * mask[1] / KEEP_PROBABILITY)
    first_z, first_variance = _standardize(first)
    second_z, second_variance = _standardize(second)
    correlation = first_z.transpose(0, 1) @ second_z / ROWS_PER_UPDATE
    if not bool(torch.isfinite(correlation).all().item()):
        raise RuntimeError("V18 Barlow correlation is non-finite")
    diagonal = torch.diagonal(correlation)
    off_diagonal = correlation - torch.diag_embed(diagonal)
    loss = ((1.0 - diagonal).square().sum() + BARLOW_OFF_DIAGONAL * off_diagonal.square().sum()) / CODE_WIDTH
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("V18 Barlow loss is non-finite")
    combined = torch.cat((first, second), dim=0)
    diagnostics = {
        "loss": float(loss.detach().item()),
        "mean_cross_correlation_diagonal": float(diagonal.detach().mean().item()),
        "mean_absolute_off_diagonal": float(off_diagonal.detach().abs().sum().item() / (CODE_WIDTH * (CODE_WIDTH - 1))),
        "mean_code_standard_deviation": float(combined.detach().std(dim=0, correction=0).mean().item()),
        "effective_rank": _effective_rank(combined.detach()),
        "zero_variance_dimensions_first": int((first_variance == 0).sum().item()),
        "zero_variance_dimensions_second": int((second_variance == 0).sum().item()),
        "finite": True,
    }
    return loss, diagnostics


def _task_operands(
    core: SharedSemanticMetricCreditMemoryCore,
    pairs: Sequence[v15.EncodedTwinPair],
    indices: Sequence[int],
    affine_scale: torch.Tensor,
    affine_bias: torch.Tensor,
) -> dict[str, Any]:
    device = next(core.parameters()).device
    states = {
        (arm, side): core.initial_state()
        for arm in ("true", "deranged") for side in ("first", "second")
    }
    true_losses: list[torch.Tensor] = []
    deranged_losses: list[torch.Tensor] = []
    zero_losses: list[torch.Tensor] = []
    twin_margins: list[torch.Tensor] = []
    affine_losses: list[torch.Tensor] = []
    chronology = {(arm, side): [] for arm in ("true", "deranged") for side in ("first", "second")}
    for pair_index in indices:
        pair = pairs[pair_index]
        first, second = pair.first.to(device), pair.second.to(device)
        for arm in ("true", "deranged"):
            states[(arm, "first")], _, steps_a, _ = v15._write_anchors(
                core, first, second, states[(arm, "first")], arm=arm, detach_state=False,
            )
            states[(arm, "second")], _, steps_b, _ = v15._write_anchors(
                core, second, first, states[(arm, "second")], arm=arm, detach_state=False,
            )
            chronology[(arm, "first")].extend(steps_a)
            chronology[(arm, "second")].extend(steps_b)
        first_logits, first_loss = v15._probe_losses(core, first, states[("true", "first")])
        second_logits, second_loss = v15._probe_losses(core, second, states[("true", "second")])
        _, deranged_first = v15._probe_losses(core, first, states[("deranged", "first")])
        _, deranged_second = v15._probe_losses(core, second, states[("deranged", "second")])
        zero = core.initial_state()
        _, zero_first = v15._probe_losses(core, first, zero, read_enabled=False)
        _, zero_second = v15._probe_losses(core, second, zero, read_enabled=False)
        for local, probe in enumerate(PROBE_INDICES):
            true_losses.extend((first_loss[local], second_loss[local]))
            deranged_losses.extend((deranged_first[local], deranged_second[local]))
            zero_losses.extend((zero_first[local], zero_second[local]))
            first_y, second_y = first.outcomes[probe].reshape(()), second.outcomes[probe].reshape(())
            first_logit, second_logit = first_logits[local].reshape(()), second_logits[local].reshape(())
            twin_margins.extend((
                first_y * (first_logit - second_logit),
                second_y * (second_logit - first_logit),
            ))
            affine_losses.extend((
                F.softplus(-first.outcomes[probe] * (affine_scale * first.base_logits[probe] + affine_bias)),
                F.softplus(-second.outcomes[probe] * (affine_scale * second.base_logits[probe] + affine_bias)),
            ))
    objective, deranged, zero, separation, ranking, total = v15._causal_training_losses(
        torch.stack(true_losses), torch.stack(deranged_losses),
        torch.stack(zero_losses), torch.stack(twin_margins),
    )
    return {
        "objective": objective, "deranged": deranged, "zero": zero,
        "separation": separation, "ranking": ranking, "total": total,
        "affine_loss": torch.stack(affine_losses).mean(),
        "true_deranged_paths_differentiable": all(
            value.grad_fn is not None for value in (objective, deranged, separation, ranking)
        ),
        "zero_baseline_detached": zero.grad_fn is None,
        "chronology_exact": all(
            steps == list(range(PAIRS_PER_UPDATE * 2)) for steps in chronology.values()
        ),
    }


def _mutated_auxiliary_inputs(pairs: Sequence[v15.EncodedTwinPair]) -> tuple[v15.EncodedTwinPair, ...]:
    changed = []
    for pair in pairs:
        def episode(value: v15.EncodedContingencyEpisode) -> v15.EncodedContingencyEpisode:
            return replace(
                value, episode_ref="changed", pair_ref="changed", twin_ref="changed",
                generator_family="changed", transition_group="changed",
                outcomes=-value.outcomes,
                relation_classes=tuple(reversed(value.relation_classes)),
            )
        changed.append(v15.EncodedTwinPair("changed", episode(pair.first), episode(pair.second)))
    return tuple(changed)


def _auxiliary_preflight(
    full: SharedSemanticMetricCreditMemoryCore,
    objective_off: SharedSemanticMetricCreditMemoryCore,
    public_pairs: Sequence[Any],
    encoded_pairs: Sequence[v15.EncodedTwinPair],
    masks: torch.Tensor,
) -> dict[str, Any]:
    rng_before = _rng_snapshot()
    rows, row_report = _public_relation_batch(public_pairs, encoded_pairs)
    public_only = tuple(
        SimpleNamespace(
            first=SimpleNamespace(public=pair.first.public, supervision="changed", metadata="changed"),
            second=SimpleNamespace(public=pair.second.public, supervision="changed", metadata="changed"),
        ) for pair in public_pairs
    )
    changed_rows, changed_report = _public_relation_batch(public_only, _mutated_auxiliary_inputs(encoded_pairs))
    loss, diagnostics = _barlow_objective(full, rows, masks)
    permutation = torch.arange(ROWS_PER_UPDATE - 1, -1, -1, device=rows.device)
    permuted, _ = _barlow_objective(full, rows[permutation], masks[:, permutation.cpu()])
    swapped, _ = _barlow_objective(full, rows, masks.flip(0))
    parameters = tuple(full.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(value for _, value in parameters), allow_unused=True)
    semantic = {
        name: gradient is not None and bool(torch.isfinite(gradient).all().item()) and bool((gradient != 0).any().item())
        for (name, _), gradient in zip(parameters, gradients, strict=True)
        if name.startswith("semantic_metric_network.")
    }
    outside = tuple(
        name for (name, _), gradient in zip(parameters, gradients, strict=True)
        if not name.startswith("semantic_metric_network.")
        and gradient is not None and bool((gradient != 0).any().item())
    )
    rng_after = _rng_snapshot()
    report = {
        "canonical_rows": row_report,
        "changed_sidecar_rows_exact": torch.equal(rows, changed_rows),
        "changed_sidecar_digests_exact": row_report == changed_report,
        "row_permutation_invariant": abs(float(loss.detach()) - float(permuted.detach())) <= 1.0e-6,
        "view_swap_invariant": abs(float(loss.detach()) - float(swapped.detach())) <= 1.0e-6,
        "semantic_parameter_gradients": semantic,
        "all_semantic_parameters_reached": bool(semantic) and all(semantic.values()),
        "outside_auxiliary_gradient_parameters": list(outside),
        "auxiliary_gradient_scope_exact": not outside,
        "global_rng_unchanged": _rng_exact(rng_before, rng_after),
        "initial_barlow_diagnostics": diagnostics,
        "negative_table_absent": True,
        "outcome_dependent_filter_absent": True,
        "task_specific_assignment_head_absent": True,
        "arm_initial_parameters_exact": _model_digest(full) == _model_digest(objective_off),
    }
    report["exact"] = bool(
        row_report["exact"] and report["changed_sidecar_rows_exact"]
        and report["changed_sidecar_digests_exact"]
        and report["row_permutation_invariant"] and report["view_swap_invariant"]
        and report["all_semantic_parameters_reached"]
        and report["auxiliary_gradient_scope_exact"]
        and report["global_rng_unchanged"]
        and report["arm_initial_parameters_exact"]
    )
    return report


def _task_parity_preflight(
    full: SharedSemanticMetricCreditMemoryCore,
    objective_off: SharedSemanticMetricCreditMemoryCore,
    pairs: Sequence[v15.EncodedTwinPair],
    indices: Sequence[int],
) -> dict[str, Any]:
    device = next(full.parameters()).device
    scale_a = torch.ones((), device=device, requires_grad=True)
    bias_a = torch.zeros((), device=device, requires_grad=True)
    scale_b = torch.ones((), device=device, requires_grad=True)
    bias_b = torch.zeros((), device=device, requires_grad=True)
    left = _task_operands(full, pairs, indices, scale_a, bias_a)
    right = _task_operands(objective_off, pairs, indices, scale_b, bias_b)
    keys = ("objective", "deranged", "zero", "separation", "ranking", "total", "affine_loss")
    loss_exact = all(torch.equal(left[key].detach(), right[key].detach()) for key in keys)
    left_parameters = tuple(full.parameters())
    right_parameters = tuple(objective_off.parameters())
    left_grad = torch.autograd.grad(left["total"], left_parameters, allow_unused=True)
    right_grad = torch.autograd.grad(right["total"], right_parameters, allow_unused=True)
    gradient_exact = all(
        (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
        for a, b in zip(left_grad, right_grad, strict=True)
    )
    return {
        "initial_parameter_digest_exact": _model_digest(full) == _model_digest(objective_off),
        "task_loss_operands_exact": loss_exact,
        "task_gradients_exact": gradient_exact,
        "chronology_exact": left["chronology_exact"] and right["chronology_exact"],
        "optimizer_configuration_exact": True,
        "exact": bool(loss_exact and gradient_exact and left["chronology_exact"] and right["chronology_exact"]),
    }


def _v17_integrity_preflight(core: SharedSemanticMetricCreditMemoryCore, development: Sequence[v15.EncodedTwinPair]) -> dict[str, Any]:
    metrics = v17._evaluate(core, development, (1.0, 0.0))
    arms = metrics["arms"]
    exact = bool(
        metrics["pair_count"] == DEVELOPMENT_PAIRS
        and all(arm["finite"] for arm in arms.values())
        and all(metrics["outcome_blind_integrity"].values())
        and metrics["replay"]["exact"] is True
        and metrics["replay"]["maximum_logit_error"] <= 1.0e-6
        and metrics["state_bounds"]["finite"] is True
        and metrics["probe_relation_query_swap"]["only_read_query_swapped"] is True
        and metrics["corresponding_anchor_read_mass"]["anchor_slot_lineage_exact"] is True
        and metrics["corresponding_anchor_read_mass"]["prediction_before_sidecars_exact"] is True
        and metrics["corresponding_anchor_read_mass"]["learner_state_unchanged"] is True
    )
    return {"exact": exact, "metrics": metrics}


def _paired_fit(
    full: SharedSemanticMetricCreditMemoryCore,
    objective_off: SharedSemanticMetricCreditMemoryCore,
    public_train: Sequence[Any],
    encoded_train: Sequence[v15.EncodedTwinPair],
    schedule: Sequence[Sequence[int]],
    mask_schedule: torch.Tensor,
) -> tuple[dict[str, Any], tuple[float, float], tuple[float, float]]:
    if len(encoded_train) != TRAIN_PAIRS or len(schedule) != TRAIN_UPDATES:
        raise ValueError("V18 fit requires the frozen V15 corpus and schedule")
    device = next(full.parameters()).device
    optimizers = {
        "full": torch.optim.AdamW(full.parameters(), lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS, eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY),
        "objective_off": torch.optim.AdamW(objective_off.parameters(), lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS, eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY),
    }
    affine = {
        arm: (torch.nn.Parameter(torch.ones((), device=device)), torch.nn.Parameter(torch.zeros((), device=device)))
        for arm in ("full", "objective_off")
    }
    affine_optimizers = {
        arm: torch.optim.AdamW(affine[arm], lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS, eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY)
        for arm in affine
    }
    updates = []
    full.train(); objective_off.train()
    for update_index, indices in enumerate(schedule):
        arm_rows: dict[str, Any] = {}
        for arm, core in (("full", full), ("objective_off", objective_off)):
            optimizers[arm].zero_grad(set_to_none=True)
            affine_optimizers[arm].zero_grad(set_to_none=True)
            task = _task_operands(core, encoded_train, indices, *affine[arm])
            arm_rows[arm] = task
        public_batch = tuple(public_train[index] for index in indices)
        encoded_batch = tuple(encoded_train[index] for index in indices)
        relation_rows, row_report = _public_relation_batch(public_batch, encoded_batch)
        barlow, diagnostics = _barlow_objective(full, relation_rows, mask_schedule[update_index])
        totals = {
            "full": arm_rows["full"]["total"] + BARLOW_WEIGHT * barlow,
            "objective_off": arm_rows["objective_off"]["total"],
        }
        gradient_norms = {}
        for arm, core in (("full", full), ("objective_off", objective_off)):
            if not bool(torch.isfinite(totals[arm]).item()):
                raise RuntimeError("V18 paired training loss is non-finite")
            totals[arm].backward()
            gradients = [parameter.grad for parameter in core.parameters() if parameter.grad is not None]
            if not gradients or any(not bool(torch.isfinite(value).all().item()) for value in gradients):
                raise RuntimeError("V18 paired core gradients are absent/non-finite")
            gradient_norms[arm] = float(torch.nn.utils.clip_grad_norm_(core.parameters(), v15.GRADIENT_CLIP).detach())
            optimizers[arm].step()
            arm_rows[arm]["affine_loss"].backward()
            affine_optimizers[arm].step()
        updates.append({
            "update": update_index, "pair_indices": list(indices),
            "canonical_rows": row_report,
            "barlow": diagnostics,
            "full": {
                "true_probe_loss": float(arm_rows["full"]["objective"].detach()),
                "deranged_probe_loss": float(arm_rows["full"]["deranged"].detach()),
                "zero_probe_loss": float(arm_rows["full"]["zero"].detach()),
                "separation_loss": float(arm_rows["full"]["separation"].detach()),
                "twin_ranking_loss": float(arm_rows["full"]["ranking"].detach()),
                "task_total_loss": float(arm_rows["full"]["total"].detach()),
                "total_loss": float(totals["full"].detach()),
                "gradient_norm": gradient_norms["full"],
            },
            "objective_off": {
                "true_probe_loss": float(arm_rows["objective_off"]["objective"].detach()),
                "deranged_probe_loss": float(arm_rows["objective_off"]["deranged"].detach()),
                "zero_probe_loss": float(arm_rows["objective_off"]["zero"].detach()),
                "separation_loss": float(arm_rows["objective_off"]["separation"].detach()),
                "twin_ranking_loss": float(arm_rows["objective_off"]["ranking"].detach()),
                "task_total_loss": float(arm_rows["objective_off"]["total"].detach()),
                "total_loss": float(totals["objective_off"].detach()),
                "gradient_norm": gradient_norms["objective_off"],
            },
            "chronology_exact": arm_rows["full"]["chronology_exact"] and arm_rows["objective_off"]["chronology_exact"],
            "true_deranged_paths_differentiable": all(
                arm_rows[arm]["true_deranged_paths_differentiable"]
                for arm in ("full", "objective_off")
            ),
            "zero_baseline_detached": all(
                arm_rows[arm]["zero_baseline_detached"]
                for arm in ("full", "objective_off")
            ),
        })
    exposure = dict(sorted(Counter(index for row in schedule for index in row).items()))
    return ({
        "completed_updates": len(updates), "updates": updates,
        "exposure_counts": exposure, "loss_weights": list(v15.LOSS_WEIGHTS),
        "barlow_weight": BARLOW_WEIGHT, "barlow_off_diagonal": BARLOW_OFF_DIAGONAL,
        "keep_probability": KEEP_PROBABILITY,
    }, tuple(float(value.detach()) for value in affine["full"]), tuple(float(value.detach()) for value in affine["objective_off"]))


def _development_diagnostics(
    core: SharedSemanticMetricCreditMemoryCore,
    public_pairs: Sequence[Any],
    encoded_pairs: Sequence[v15.EncodedTwinPair],
    masks: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    for block in range(0, len(encoded_pairs), PAIRS_PER_UPDATE):
        batch, _ = _public_relation_batch(
            public_pairs[block:block + PAIRS_PER_UPDATE],
            encoded_pairs[block:block + PAIRS_PER_UPDATE],
        )
        _, diagnostics = _barlow_objective(core, batch, masks[(block // PAIRS_PER_UPDATE) % TRAIN_UPDATES])
        rows.append(diagnostics)
    keys = (
        "loss", "mean_cross_correlation_diagonal", "mean_absolute_off_diagonal",
        "mean_code_standard_deviation", "effective_rank",
    )
    return {
        **{key: sum(float(row[key]) for row in rows) / len(rows) for key in keys},
        "zero_variance_dimensions_first": max(row["zero_variance_dimensions_first"] for row in rows),
        "zero_variance_dimensions_second": max(row["zero_variance_dimensions_second"] for row in rows),
        "block_count": len(rows), "finite": all(row["finite"] for row in rows),
    }


def _protocol(training: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "exact_training_schedule": training["completed_updates"] == TRAIN_UPDATES and set(training["exposure_counts"].values()) == {2},
        "all_training_gradients_finite": all(
            math.isfinite(row[arm]["gradient_norm"])
            for row in training["updates"] for arm in ("full", "objective_off")
        ),
        "true_deranged_paths_differentiable": all(
            row["true_deranged_paths_differentiable"] for row in training["updates"]
        ),
        "zero_baseline_detached": all(
            row["zero_baseline_detached"] for row in training["updates"]
        ),
        "predict_before_feedback_chronology": all(row["chronology_exact"] for row in training["updates"]),
        "canonical_48_rows_every_update": all(row["canonical_rows"]["row_count"] == ROWS_PER_UPDATE for row in training["updates"]),
        "mask_schedule_fixed": bool(
            training["mask_schedule_after_preflight"]["exact"]
            and training["mask_schedule_after_training"]["exact"]
        ),
        "eight_pair_state_horizon": metrics["maximum_state_step"] == PAIRS_PER_UPDATE * 2,
    }


def _objective_gate(
    full_metrics: Mapping[str, Any], off_metrics: Mapping[str, Any],
    architecture: Mapping[str, Any], preflight: Mapping[str, Any], *, identity_exact: bool,
) -> bool:
    try:
        full_read = full_metrics["corresponding_anchor_read_mass"]
        off_read = off_metrics["corresponding_anchor_read_mass"]
        full_true = full_metrics["arms"]["true"]
        off_true = off_metrics["arms"]["true"]
        return bool(
            preflight["exact"] is True and identity_exact is True
            and full_read["top1_accuracy"] - off_read["top1_accuracy"] >= 0.10
            and full_read["mean_matching_mass"] - off_read["mean_matching_mass"] >= 0.05
            and full_read["mean_matching_minus_competitor_margin"]
            - off_read["mean_matching_minus_competitor_margin"] >= 0.05
            and full_true["balanced_accuracy"] >= off_true["balanced_accuracy"]
            and full_true["mean_probe_nll"] <= off_true["mean_probe_nll"] + 0.02
            and v17._component_gate(
                full_metrics, architecture,
                identifiability_exact=True, identity_exact=identity_exact,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _full_gate(
    full_metrics: Mapping[str, Any], off_metrics: Mapping[str, Any],
    architecture: Mapping[str, Any], preflight: Mapping[str, Any], *, identity_exact: bool,
) -> bool:
    return bool(
        _objective_gate(full_metrics, off_metrics, architecture, preflight, identity_exact=identity_exact)
        and v17._full_gate(
            full_metrics, architecture,
            identifiability_exact=True, identity_exact=identity_exact,
        )
    )


def _run_train_development(args: argparse.Namespace) -> None:
    full_path = Path(args.full_checkpoint)
    off_path = Path(args.objective_off_checkpoint)
    result_path = Path(args.development_result)
    if any(path.exists() for path in (full_path, off_path, result_path)):
        raise FileExistsError("V18 train-development identity is already consumed")
    sources_before = _source_hashes()
    if sources_before["leaf"] != LEAF_SHA256 or sources_before["reasoning_export"] != REASONING_EXPORT_SHA256:
        raise RuntimeError("V18 leaf/export source identity changed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks, mask_report = _precompute_mask_schedule()
    device, started = v16._prepare_angler_device(args.device)
    consumed_v17 = _v17_evidence(Path(args.v17_result), Path(args.v17_checkpoint))
    v14_evidence = v15._v14_structure_evidence(Path(args.v14_result))
    frozen, v13_checkpoint = v15._load_frozen_v13(Path(args.v13_checkpoint), Path(args.v13_result), device)
    frozen_before = _model_digest(frozen)
    qwen = v5._load_qwen(args.model_path)
    qwen_before = foundation_tensor_digest(qwen.model)
    if qwen_before != v13_checkpoint["qwen_digest"]:
        raise RuntimeError("V18 Qwen does not match frozen V13 foundation")
    corpus = build_paired_latent_contingency_credit_v15()
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V18 final opened despite absent final-phase authority")
    schedule = v15._training_schedule(corpus)
    train = v15._encode_pairs(qwen, frozen, corpus.train)
    development = v15._encode_pairs(qwen, frozen, corpus.development)
    full = SharedSemanticMetricCreditMemoryCore(temporal_width=TEMPORAL_WIDTH).to(device=device, dtype=torch.float32)
    objective_off = copy.deepcopy(full)
    identifiability = v15._identifiability_preflight(full, train, development, schedule)
    architecture = v17._architecture_evidence(full, train[0])
    task_parity = _task_parity_preflight(full, objective_off, train, schedule[0])
    auxiliary = _auxiliary_preflight(
        full, objective_off,
        tuple(corpus.train[index] for index in schedule[0]),
        tuple(train[index] for index in schedule[0]), masks[0].clone(),
    )
    mask_after_preflight = _mask_schedule_identity(masks)
    inherited_integrity = _v17_integrity_preflight(full, development)
    preflight = {
        "identifiability": identifiability, "architecture_exact": architecture["exact"],
        "task_parity": task_parity, "auxiliary": auxiliary,
        "inherited_v17_integrity": inherited_integrity, "mask_schedule": mask_report,
    }
    preflight["exact"] = bool(
        identifiability["exact"] is True and architecture["exact"] is True
        and task_parity["exact"] is True and auxiliary["exact"] is True
        and inherited_integrity["exact"] is True and mask_report["exact"] is True
        and mask_after_preflight["exact"] is True
    )
    preflight["mask_schedule_after_preflight"] = mask_after_preflight
    if preflight["exact"] is not True:
        raise RuntimeError("V18 paired/auxiliary preflight failed")
    initial_digest = _model_digest(full)
    training, full_affine, off_affine = _paired_fit(
        full, objective_off, corpus.train, train, schedule, masks,
    )
    mask_after_training = _mask_schedule_identity(masks)
    if mask_after_training["exact"] is not True:
        raise RuntimeError("V18 mask schedule changed during training")
    training["mask_schedule_after_preflight"] = mask_after_preflight
    training["mask_schedule_after_training"] = mask_after_training
    v15._enforce_resources(started, device)
    full_metrics = v17._evaluate(full, development, full_affine)
    off_metrics = v17._evaluate(objective_off, development, off_affine)
    full_metrics["protocol_invariants"] = _protocol(training, full_metrics)
    off_metrics["protocol_invariants"] = _protocol(training, off_metrics)
    diagnostics = {
        "full": _development_diagnostics(full, corpus.development, development, masks),
        "objective_off": _development_diagnostics(objective_off, corpus.development, development, masks),
    }
    v15._enforce_resources(started, device)
    sources_after = _source_hashes()
    identity_checks = {
        "qwen_before": qwen_before, "qwen_after": foundation_tensor_digest(qwen.model),
        "v13_before": frozen_before, "v13_after": _model_digest(frozen),
        "sources_before": sources_before, "sources_after": sources_after,
        "reasoning_export_exact": sources_after["reasoning_export"] == REASONING_EXPORT_SHA256,
        "final_partition_sealed": True,
    }
    identity_checks["exact"] = bool(
        qwen_before == identity_checks["qwen_after"]
        and frozen_before == identity_checks["v13_after"]
        and sources_before == sources_after
        and identity_checks["reasoning_export_exact"]
        and identity_checks["final_partition_sealed"]
    )
    supported = _objective_gate(
        full_metrics, off_metrics, architecture, preflight,
        identity_exact=bool(identity_checks["exact"]),
    )
    durable = _full_gate(
        full_metrics, off_metrics, architecture, preflight,
        identity_exact=bool(identity_checks["exact"]),
    )
    classification = (
        "FULL_DURABLE_BARLOW_SEMANTIC_CREDIT_SUPPORTED" if durable else
        "BARLOW_SEMANTIC_OBJECTIVE_SUPPORTED" if supported else
        "DEVELOPMENT_NOT_SUPPORTED"
    )
    environment = _environment_report(device)
    elapsed_seconds = time.perf_counter() - started
    peak_angler_gpu_bytes = torch.cuda.max_memory_allocated(device)
    common = {
        "identity": IDENTITY, "seed": SEED, "corpus_id": CORPUS_ID,
        "completed_updates": TRAIN_UPDATES, "temporal_width": TEMPORAL_WIDTH,
        "initial_core_digest": initial_digest, "source_hashes": sources_before,
        "qwen_digest": qwen_before, "v13_digest": frozen_before,
        "v17_result_sha256": V17_RESULT_SHA256,
        "v17_checkpoint_sha256": V17_CHECKPOINT_SHA256,
        "preflight": preflight, "architecture_evidence": architecture,
        "identity_checks": identity_checks,
        "environment": environment,
        "elapsed_seconds": elapsed_seconds,
        "peak_angler_gpu_bytes": peak_angler_gpu_bytes,
    }
    full_checkpoint = {
        **common, "arm": "full", "core_digest": _model_digest(full),
        "core_state": {name: value.detach().cpu() for name, value in full.state_dict().items()},
        "affine_calibrator": list(full_affine), "development_metrics": full_metrics,
        "barlow_diagnostics": diagnostics["full"],
    }
    off_checkpoint = {
        **common, "arm": "objective_off", "core_digest": _model_digest(objective_off),
        "core_state": {name: value.detach().cpu() for name, value in objective_off.state_dict().items()},
        "affine_calibrator": list(off_affine), "development_metrics": off_metrics,
        "barlow_diagnostics": diagnostics["objective_off"],
    }
    result: dict[str, Any] = {
        "identity": IDENTITY, "phase": "train-development",
        "classification": classification,
        "barlow_semantic_objective_supported": supported,
        "full_durable_supported": durable,
        "development_authorized": False,
        "successor_final_leaf_eligible": durable,
        "final_partition_sealed": True,
        "corpus_id": CORPUS_ID, "seed": SEED,
        "preflight": preflight, "architecture_evidence": architecture,
        "training": training,
        "development_metrics": {"full": full_metrics, "objective_off": off_metrics},
        "barlow_diagnostics": diagnostics,
        "identity_checks": identity_checks, "source_hashes": sources_before,
        "environment": environment,
        "frozen_evidence": {
            "v17_result_sha256": V17_RESULT_SHA256,
            "v17_checkpoint_sha256": V17_CHECKPOINT_SHA256,
            "v17_classification": consumed_v17["classification"],
            "v14_structure_replicated": v14_evidence["structure_replicated"],
        },
        "frozen_compute": {
            "updates": TRAIN_UPDATES, "pairs_per_update": PAIRS_PER_UPDATE,
            "train_pairs": TRAIN_PAIRS, "development_pairs": DEVELOPMENT_PAIRS,
            "loss_weights": list(v15.LOSS_WEIGHTS), "barlow_weight": BARLOW_WEIGHT,
            "keep_probability": KEEP_PROBABILITY,
            "off_diagonal_coefficient": BARLOW_OFF_DIAGONAL,
            "standardization_epsilon": STANDARDIZATION_EPSILON,
            "dtype": "float32", "autocast": False, "tf32": False,
            "qwen_device": "cuda:0", "v13_v18_device": "cuda:1",
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_angler_gpu_bytes": peak_angler_gpu_bytes,
    }
    if not identity_checks["exact"]:
        raise RuntimeError("V18 source/foundation drift during execution")
    v15._atomic_torch(full_path, full_checkpoint)
    v15._atomic_torch(off_path, off_checkpoint)
    result["full_checkpoint_sha256"] = _sha256(full_path)
    result["objective_off_checkpoint_sha256"] = _sha256(off_path)
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
    parser.add_argument("--v17-result", default="/opt/angler/results/shared-semantic-metric-credit-v17-development.json")
    parser.add_argument("--v17-checkpoint", default="/opt/angler/results/shared-semantic-metric-credit-v17.pt")
    parser.add_argument("--full-checkpoint", default="/opt/angler/results/barlow-semantic-invariance-v18-full.pt")
    parser.add_argument("--objective-off-checkpoint", default="/opt/angler/results/barlow-semantic-invariance-v18-objective-off.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/barlow-semantic-invariance-v18-development.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _run_train_development(args)


if __name__ == "__main__":
    main()
