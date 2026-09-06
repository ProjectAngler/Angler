"""Frozen factorized key-value causal apprenticeship evaluation V9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from angler.reasoning import FactorizedKeyValueCausalMemoryCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import content_addressed_causal_memory_v7 as v7
from experiments.runners import sparse_memory_causal_routing_v8 as v8


IDENTITY = "angler.factorized-key-value-causal-memory.v9-first-result"
SEED = 2026083109
RANK = v8.RANK
MEMORY_SLOTS = v8.MEMORY_SLOTS
TEMPORAL_WIDTH = v8.TEMPORAL_WIDTH

_train_core = v8._train_core
_evaluate_all = v8._evaluate_all


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "runner": v7.v6.v5._sha256(Path(__file__).resolve()),
        "core": v7.v6.v5._sha256(
            root / "src/angler/reasoning/factorized_key_value_causal_memory.py"
        ),
        "corpus": v7.v6.v5._sha256(
            root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py"
        ),
    }


def _load_semantic_donor(
    core: FactorizedKeyValueCausalMemoryCore,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != v8.IDENTITY:
        raise RuntimeError("V8 semantic donor identity mismatch")
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
                raise RuntimeError("V8 semantic donor tensor shape mismatch")
            parameters[name].copy_(
                donor[name].to(device=parameters[name].device, dtype=parameters[name].dtype)
            )
    loaded = core.state_dict()
    if any(not torch.equal(loaded[name].cpu(), donor[name]) for name in names):
        raise RuntimeError("V8 semantic donor was not loaded byte-exactly")
    return {
        "path": str(checkpoint_path),
        "sha256": v7.v6.v5._sha256(checkpoint_path),
        "mapping": {name: name for name in names},
    }


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _factorization_metrics(
    core: FactorizedKeyValueCausalMemoryCore,
    rows: Sequence[v7.v6.v5.EncodedMechanism],
) -> dict[str, Any]:
    """Measure structural address invariance without updating parameters."""

    write_key_differences = []
    write_weight_differences = []
    final_key_differences = []
    final_usage_differences = []
    read_weight_differences = []
    final_value_differences = []
    neutral_value_maxima = []
    neutral_residual_maxima = []
    maximum_read_slots = 0
    maximum_write_slots = 0
    core.eval()
    with torch.no_grad():
        for row in rows:
            initial = core.initial_memory_state().detached_clone()
            true_state, true_writes = v7._apply_stream(
                core,
                row,
                initial,
                detach_state=True,
                collect_outputs=True,
            )
            shuffled = v7.v6._counterfactual_outcomes(row, "outcome_shuffled")
            shuffled_state, shuffled_writes = v7._apply_stream(
                core,
                row,
                initial,
                outcome_values=shuffled,
                detach_state=True,
                collect_outputs=True,
            )
            for true_output, shuffled_output in zip(true_writes, shuffled_writes):
                write_key_differences.append(
                    _maximum_difference(true_output.write_keys, shuffled_output.write_keys)
                )
                write_weight_differences.append(
                    _maximum_difference(
                        true_output.write_weights, shuffled_output.write_weights
                    )
                )
                maximum_write_slots = max(
                    maximum_write_slots,
                    int((true_output.write_weights > 0).sum(dim=-1).max().item()),
                )
            final_key_differences.append(
                _maximum_difference(true_state.keys, shuffled_state.keys)
            )
            final_usage_differences.append(
                _maximum_difference(true_state.usage, shuffled_state.usage)
            )
            final_value_differences.append(
                _maximum_difference(true_state.values, shuffled_state.values)
            )
            true_challenge = v7._challenge(core, row, true_state)
            shuffled_challenge = v7._challenge(core, row, shuffled_state)
            read_weight_differences.append(
                _maximum_difference(
                    true_challenge.read_weights, shuffled_challenge.read_weights
                )
            )
            maximum_read_slots = max(
                maximum_read_slots,
                int((true_challenge.read_weights > 0).sum(dim=-1).max().item()),
            )

            neutral_state, _ = v7._apply_stream(
                core,
                row,
                initial,
                outcome_values=(0,) * 6,
                detach_state=True,
            )
            neutral_value_maxima.append(float(neutral_state.values.abs().max().item()))
            neutral_challenge = v7._challenge(core, row, neutral_state)
            neutral_residual_maxima.append(
                float(neutral_challenge.residuals.abs().max().item())
            )

    return {
        "mechanism_count": len(rows),
        "maximum_write_key_difference_under_shuffle": max(write_key_differences),
        "maximum_write_weight_difference_under_shuffle": max(write_weight_differences),
        "maximum_final_key_difference_under_shuffle": max(final_key_differences),
        "maximum_final_usage_difference_under_shuffle": max(final_usage_differences),
        "maximum_challenge_read_weight_difference_under_shuffle": max(
            read_weight_differences
        ),
        "minimum_final_value_difference_under_shuffle": min(
            final_value_differences
        ),
        "maximum_neutral_value": max(neutral_value_maxima),
        "maximum_neutral_residual": max(neutral_residual_maxima),
        "maximum_nonzero_read_slots": maximum_read_slots,
        "maximum_nonzero_write_slots": maximum_write_slots,
    }


def _development_gate(metrics: dict[str, Any], identities_exact: bool) -> bool:
    factor = metrics.get("factorization", {})
    invariant_names = (
        "maximum_write_key_difference_under_shuffle",
        "maximum_write_weight_difference_under_shuffle",
        "maximum_final_key_difference_under_shuffle",
        "maximum_final_usage_difference_under_shuffle",
        "maximum_challenge_read_weight_difference_under_shuffle",
    )
    try:
        return bool(
            v8._development_gate(metrics, identities_exact)
            and factor.get("mechanism_count") == 8
            and all(float(factor[name]) <= 1.0e-6 for name in invariant_names)
            and float(factor["minimum_final_value_difference_under_shuffle"])
            > 1.0e-6
            and float(factor["maximum_neutral_value"]) == 0.0
            and float(factor["maximum_neutral_residual"]) == 0.0
            and int(factor["maximum_nonzero_read_slots"])
            <= FactorizedKeyValueCausalMemoryCore.routing_k
            and int(factor["maximum_nonzero_write_slots"])
            <= FactorizedKeyValueCausalMemoryCore.routing_k
        )
    except (KeyError, TypeError, ValueError):
        return False


def _new_core(
    content_width: int, device: torch.device
) -> FactorizedKeyValueCausalMemoryCore:
    return FactorizedKeyValueCausalMemoryCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        rank=RANK,
        memory_slots=MEMORY_SLOTS,
    ).to(device=device, dtype=torch.float32)


def _checkpoint_payload(
    core: FactorizedKeyValueCausalMemoryCore,
    content_width: int,
    qwen_digest: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "content_width": content_width,
        "temporal_width": TEMPORAL_WIDTH,
        "rank": RANK,
        "memory_slots": MEMORY_SLOTS,
        "routing_k": FactorizedKeyValueCausalMemoryCore.routing_k,
        "qwen_digest": qwen_digest,
        "core_digest": v7.v6.v5._module_digest(core),
        "source_hashes": _source_hashes(),
        "development_metrics": metrics,
        "core_state_dict": {
            name: tensor.detach().cpu() for name, tensor in core.state_dict().items()
        },
    }


def _run_train_development(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V9 train-development identity is already consumed")
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
    donor = _load_semantic_donor(core, Path(args.v8_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, development_rows)
    metrics["factorization"] = _factorization_metrics(core, development_rows)
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
            "routing_k": FactorizedKeyValueCausalMemoryCore.routing_k,
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
        "v8_semantic_donor": donor,
        "source_hashes": _source_hashes(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Development tests factorized outcome-causal key-value memory in "
            "synthetic software mechanisms; it is not final, arbitrary reasoning, "
            "AGI, or deployment authority."
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
        raise FileExistsError("V9 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    v7.v6.v5._validate_final_authorization(
        development.get("development_authorized") is True
    )
    if development.get("checkpoint_sha256") != v7.v6.v5._sha256(checkpoint_path):
        raise RuntimeError("V9 checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != IDENTITY:
        raise RuntimeError("V9 checkpoint identity mismatch")
    if checkpoint.get("source_hashes") != _source_hashes():
        raise RuntimeError("V9 source identity mismatch")

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
        raise RuntimeError("V9 core digest mismatch")
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, rows)
    metrics["factorization"] = _factorization_metrics(core, rows)
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
            "Supported would establish bounded synthetic factorized outcome-causal "
            "memory only, not arbitrary intelligence, AGI, or deployment readiness."
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
        "--v8-checkpoint",
        default="/opt/angler/results/sparse-memory-causal-routing-v8.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/factorized-key-value-causal-memory-v9.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/factorized-key-value-causal-memory-v9-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/factorized-key-value-causal-memory-v9-final.json",
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
