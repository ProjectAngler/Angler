"""Frozen sufficient-horizon DNC allocation evaluation V10."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from angler.reasoning import DncAllocatedFactorizedCausalMemoryCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import content_addressed_causal_memory_v7 as v7
from experiments.runners import factorized_key_value_causal_memory_v9 as v9


IDENTITY = "angler.dnc-allocated-factorized-causal-memory.v10-first-result"
SEED = 2026083110
RANK = v9.RANK
MEMORY_SLOTS = 64
TEMPORAL_WIDTH = v9.TEMPORAL_WIDTH
MINIMUM_EFFECTIVE_WRITE_SLOTS = 32.0
MAXIMUM_AGGREGATE_WRITE_SHARE = 0.05


@contextmanager
def _v10_slot_scope():
    """Reuse V7 metrics with a scoped slot width, restoring it on every exit."""

    if v7.MEMORY_SLOTS != 16:
        raise RuntimeError("inherited V7 memory width was unexpectedly changed")
    v7.MEMORY_SLOTS = MEMORY_SLOTS
    try:
        yield
    finally:
        v7.MEMORY_SLOTS = 16


def _train_core(core, rows):
    with _v10_slot_scope():
        return v9._train_core(core, rows)


def _evaluate_all(core, rows):
    with _v10_slot_scope():
        return v9._evaluate_all(core, rows)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "runner": v7.v6.v5._sha256(Path(__file__).resolve()),
        "core": v7.v6.v5._sha256(
            root
            / "src/angler/reasoning/dnc_allocated_factorized_causal_memory.py"
        ),
        "corpus": v7.v6.v5._sha256(
            root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py"
        ),
        "v9_runner": v7.v6.v5._sha256(
            root / "experiments/runners/factorized_key_value_causal_memory_v9.py"
        ),
        "v8_runner": v7.v6.v5._sha256(
            root / "experiments/runners/sparse_memory_causal_routing_v8.py"
        ),
        "v7_runner": v7.v6.v5._sha256(
            root / "experiments/runners/content_addressed_causal_memory_v7.py"
        ),
        "v6_runner": v7.v6.v5._sha256(
            root / "experiments/runners/causal_neuromodulated_apprenticeship_v6.py"
        ),
        "v5_runner": v7.v6.v5._sha256(
            root / "experiments/runners/outcome_aware_apprenticeship_v5.py"
        ),
        "v9_core": v7.v6.v5._sha256(
            root / "src/angler/reasoning/factorized_key_value_causal_memory.py"
        ),
        "v8_core": v7.v6.v5._sha256(
            root / "src/angler/reasoning/sparse_memory_causal_routing.py"
        ),
        "v7_core": v7.v6.v5._sha256(
            root / "src/angler/reasoning/content_addressed_causal_memory.py"
        ),
        "situated_feedback": v7.v6.v5._sha256(
            root / "src/angler/reasoning/situated_feedback.py"
        ),
        "v5_corpus": v7.v6.v5._sha256(
            root / "experiments/corpora/outcome_aware_apprenticeship_v5.py"
        ),
        "qwen_runtime": v7.v6.v5._sha256(
            root / "src/angler/runtime/qwen_knowledge.py"
        ),
    }


def _load_semantic_donor(
    core: DncAllocatedFactorizedCausalMemoryCore,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != v9.IDENTITY:
        raise RuntimeError("V9 semantic donor identity mismatch")
    donor = checkpoint["core_state_dict"]
    names = (
        "query_projection.weight",
        "candidate_projection.weight",
        "temporal_projection.weight",
        "semantic_network.0.weight",
        "semantic_network.0.bias",
        "semantic_network.1.weight",
        "semantic_network.1.bias",
    )
    parameters = dict(core.named_parameters())
    with torch.no_grad():
        for name in names:
            if donor[name].shape != parameters[name].shape:
                raise RuntimeError("V9 semantic donor tensor shape mismatch")
            parameters[name].copy_(
                donor[name].to(device=parameters[name].device, dtype=parameters[name].dtype)
            )
    loaded = core.state_dict()
    if any(not torch.equal(loaded[name].cpu(), donor[name]) for name in names):
        raise RuntimeError("V9 semantic donor was not loaded byte-exactly")
    return {
        "path": str(checkpoint_path),
        "sha256": v7.v6.v5._sha256(checkpoint_path),
        "mapping": {name: name for name in names},
    }


def _allocation_metrics(
    core: DncAllocatedFactorizedCausalMemoryCore,
    rows: Sequence[v7.v6.v5.EncodedMechanism],
) -> dict[str, Any]:
    persistent = core.initial_memory_state().detached_clone()
    persistent_winners: list[int] = []
    persistent_weights: list[torch.Tensor] = []
    local_unique_counts = []
    maximum_nonzero_write_slots = 0
    attribution = []
    core.eval()
    with torch.no_grad():
        for row in rows:
            local = core.initial_memory_state().detached_clone()
            local, local_outputs = v7._apply_stream(
                core,
                row,
                local,
                detach_state=True,
                collect_outputs=True,
            )
            local_winners = [
                int(output.write_weights[0, 0].argmax().item())
                for output in local_outputs
            ]
            local_unique_counts.append(len(set(local_winners)))
            maximum_nonzero_write_slots = max(
                maximum_nonzero_write_slots,
                max(
                    int((output.write_weights[0, 0] > 0).sum().item())
                    for output in local_outputs
                ),
            )

            challenge = v7._challenge(core, row, local)
            success_slots = [
                slot
                for slot, outcome in zip(local_winners, row.episode_outcomes.tolist())
                if int(outcome) == 1
            ]
            failure_slots = [
                slot
                for slot, outcome in zip(local_winners, row.episode_outcomes.tolist())
                if int(outcome) == -1
            ]
            target_read = challenge.read_weights[0, row.target_index]
            failed_read = challenge.read_weights[0, list(row.failed_indices)]
            attribution.append(
                {
                    "mechanism_ref": row.mechanism_ref,
                    "generator_family": row.generator_family,
                    "target_success_slot_mass": float(target_read[success_slots].sum().item()),
                    "target_failure_slot_mass": float(target_read[failure_slots].sum().item()),
                    "failed_candidates_success_slot_mass": float(
                        failed_read[:, success_slots].sum(dim=-1).mean().item()
                    ),
                    "failed_candidates_failure_slot_mass": float(
                        failed_read[:, failure_slots].sum(dim=-1).mean().item()
                    ),
                }
            )

            persistent, outputs = v7._apply_stream(
                core,
                row,
                persistent,
                detach_state=True,
                collect_outputs=True,
            )
            persistent_winners.extend(
                int(output.write_weights[0, 0].argmax().item()) for output in outputs
            )
            persistent_weights.extend(output.write_weights[0, 0] for output in outputs)

    with _v10_slot_scope():
        distribution = v7._distribution_metrics(torch.stack(persistent_weights))
    occupied = int((persistent.usage > 0).sum().item())

    # Slot order is not a learned identity. A populated state and a matched
    # slot-axis permutation must produce the same retrieved value and scores.
    permutation = torch.arange(
        MEMORY_SLOTS - 1,
        -1,
        -1,
        device=persistent.keys.device,
    )
    permuted = type(persistent)(
        keys=persistent.keys[permutation].clone(),
        values=persistent.values[permutation].clone(),
        usage=persistent.usage[permutation].clone(),
        step=persistent.step,
    )
    original_read = v7._challenge(core, rows[0], persistent)
    permuted_read = v7._challenge(core, rows[0], permuted)

    # Continue a clone past capacity to prove the bounded overwrite path; this
    # diagnostic never feeds back into parameters or the 48-event evaluation.
    stress = persistent.detached_clone()
    for row in rows[:4]:
        stress, _ = v7._apply_stream(
            core,
            row,
            stress,
            detach_state=True,
        )
    stress_state = v7._state_metrics(stress)
    return {
        "event_count": len(persistent_winners),
        "persistent_unique_winning_slots": len(set(persistent_winners)),
        "minimum_local_unique_winning_slots": min(local_unique_counts),
        "no_reuse_before_capacity": len(set(persistent_winners))
        == len(persistent_winners),
        "unused_slots_after_persistent": MEMORY_SLOTS - occupied,
        "maximum_nonzero_write_slots": maximum_nonzero_write_slots,
        "aggregate_write": distribution,
        "slot_permutation_read_value_difference": float(
            (original_read.read_values - permuted_read.read_values).abs().max().item()
        ),
        "slot_permutation_residual_difference": float(
            (original_read.residuals - permuted_read.residuals).abs().max().item()
        ),
        "slot_permutation_score_difference": float(
            (original_read.scores - permuted_read.scores).abs().max().item()
        ),
        "stress_step": stress.step,
        "stress_bounds_valid": stress_state["bounds_valid"],
        "stress_state": stress_state,
        "evaluator_only_read_attribution": attribution,
    }


def _development_gate(metrics: dict[str, Any], identities_exact: bool) -> bool:
    allocation = metrics.get("allocation", {})
    aggregate = allocation.get("aggregate_write", {})
    try:
        return bool(
            v9._development_gate(metrics, identities_exact)
            and allocation.get("event_count") == 48
            and allocation.get("persistent_unique_winning_slots") == 48
            and allocation.get("minimum_local_unique_winning_slots") == 6
            and allocation.get("no_reuse_before_capacity") is True
            and allocation.get("unused_slots_after_persistent") == 16
            and allocation.get("maximum_nonzero_write_slots") == 1
            and float(aggregate["effective_slots"])
            >= MINIMUM_EFFECTIVE_WRITE_SLOTS
            and float(aggregate["maximum_share"])
            <= MAXIMUM_AGGREGATE_WRITE_SHARE
            and float(allocation["slot_permutation_read_value_difference"])
            <= 1.0e-6
            and float(allocation["slot_permutation_residual_difference"])
            <= 1.0e-6
            and float(allocation["slot_permutation_score_difference"])
            <= 1.0e-6
            and allocation.get("stress_step") == 72
            and allocation.get("stress_bounds_valid") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _new_core(
    content_width: int, device: torch.device
) -> DncAllocatedFactorizedCausalMemoryCore:
    return DncAllocatedFactorizedCausalMemoryCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        rank=RANK,
        memory_slots=MEMORY_SLOTS,
    ).to(device=device, dtype=torch.float32)


def _checkpoint_payload(core, content_width, qwen_digest, metrics):
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "content_width": content_width,
        "temporal_width": TEMPORAL_WIDTH,
        "rank": RANK,
        "memory_slots": MEMORY_SLOTS,
        "read_k": core.routing_k,
        "write_k": 1,
        "qwen_digest": qwen_digest,
        "core_digest": v7.v6.v5._module_digest(core),
        "source_hashes": _source_hashes(),
        "development_metrics": metrics,
        "core_state_dict": {
            name: tensor.detach().cpu() for name, tensor in core.state_dict().items()
        },
    }


def _validate_development_authorization(
    development: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    identity_checks = development.get("identity_checks", {})
    metrics = development.get("development_metrics")
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "DEVELOPMENT_GATE_PASSED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256")
        == v7.v6.v5._sha256(checkpoint_path)
        and isinstance(metrics, dict)
        and identity_checks.get("exact") is True
        and identity_checks.get("foundation_before")
        == identity_checks.get("foundation_after")
        and identity_checks.get("core_before_evaluation")
        == identity_checks.get("core_after_evaluation")
        and _development_gate(metrics, True)
    )
    v7.v6.v5._validate_final_authorization(valid)


def _run_train_development(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V10 train-development identity is already consumed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v7.v6.v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_causal_neuromodulated_apprenticeship_v6()
    train_cpu = v7.v6._encode_partition(qwen, corpus.train)
    development_cpu = v7.v6._encode_partition(qwen, corpus.development)
    content_width = int(train_cpu[0].episode_queries.shape[-1])
    train_rows = tuple(row.to(device) for row in train_cpu)
    development_rows = tuple(row.to(device) for row in development_cpu)
    core = _new_core(content_width, device)
    donor = _load_semantic_donor(core, Path(args.v9_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, development_rows)
    metrics["factorization"] = v9._factorization_metrics(core, development_rows)
    metrics["allocation"] = _allocation_metrics(core, development_rows)
    core_after = v7.v6.v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    exact = core_before == core_after and foundation_before == foundation_after
    authorized = _development_gate(metrics, exact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(core, content_width, foundation_before, metrics),
        checkpoint_path,
    )
    result = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": (
            "DEVELOPMENT_GATE_PASSED" if authorized else "DEVELOPMENT_NOT_SUPPORTED"
        ),
        "development_authorized": authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "training_mechanisms": 64,
            "development_mechanisms": 8,
            "final_mechanisms_opened": 0,
            "meta_pairs": 256,
            "optimizer_steps": 64,
            "rank": RANK,
            "memory_slots": MEMORY_SLOTS,
            "read_k": core.routing_k,
            "write_k": 1,
        },
        "training_diagnostics": training,
        "development_metrics": metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_before,
            "core_after_evaluation": core_after,
            "exact": exact,
        },
        "v9_semantic_donor": donor,
        "source_hashes": _source_hashes(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Development tests sufficient-horizon episodic allocation with "
            "factorized outcome-causal memory only; it is not final, arbitrary "
            "reasoning, AGI, or deployment authority."
        ),
    }
    v7.v6.v5._write_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "development_result": str(result_path),
                "checkpoint": str(checkpoint_path),
                "full_successes": metrics["successes"]["full"],
                "true_minus_shuffled": metrics["outcome_causality"][
                    "target_probability_advantage"
                ]["outcome_shuffled"],
                "unique_write_slots": metrics["allocation"][
                    "persistent_unique_winning_slots"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_final(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V10 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_development_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != IDENTITY:
        raise RuntimeError("V10 checkpoint identity mismatch")
    if checkpoint.get("source_hashes") != _source_hashes():
        raise RuntimeError("V10 source identity mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v7.v6.v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]:
        raise RuntimeError("frozen Qwen changed")
    corpus = build_causal_neuromodulated_apprenticeship_v6(include_final=True)
    rows = tuple(
        row.to(device) for row in v7.v6._encode_partition(qwen, corpus.final)
    )
    core = _new_core(checkpoint["content_width"], device)
    core.load_state_dict(checkpoint["core_state_dict"], strict=True)
    if v7.v6.v5._module_digest(core) != checkpoint["core_digest"]:
        raise RuntimeError("V10 core digest mismatch")
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, rows)
    metrics["factorization"] = v9._factorization_metrics(core, rows)
    metrics["allocation"] = _allocation_metrics(core, rows)
    core_after = v7.v6.v5._module_digest(core)
    foundation_after = foundation_tensor_digest(qwen.model)
    exact = core_before == core_after and foundation_before == foundation_after
    supported = _development_gate(metrics, exact)
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "development_result_sha256": v7.v6.v5._sha256(development_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "final_metrics": metrics,
        "identity_checks": {
            "foundation_before": foundation_before,
            "foundation_after": foundation_after,
            "core_before_evaluation": core_before,
            "core_after_evaluation": core_after,
            "exact": exact,
        },
        "source_hashes": _source_hashes(),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Supported would establish bounded sufficient-horizon episodic "
            "allocation plus factorized outcome-causal memory only, not arbitrary "
            "intelligence, AGI, or deployment readiness."
        ),
    }
    v7.v6.v5._write_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "final_result": str(result_path),
                "full_successes": metrics["successes"]["full"],
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
        "--v9-checkpoint",
        default="/opt/angler/results/factorized-key-value-causal-memory-v9.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/dnc-allocated-factorized-causal-memory-v10.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/dnc-allocated-factorized-causal-memory-v10-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/dnc-allocated-factorized-causal-memory-v10-final.json",
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
