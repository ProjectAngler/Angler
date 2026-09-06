"""Frozen sparse memory-dependent causal apprenticeship evaluation V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from angler.reasoning import SparseMemoryCausalRoutingCore
from angler.runtime import foundation_tensor_digest
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    CORPUS_ID,
    build_causal_neuromodulated_apprenticeship_v6,
)
from experiments.runners import content_addressed_causal_memory_v7 as v7


IDENTITY = "angler.sparse-memory-causal-routing.v8-first-result"
SEED = 2026083108
RANK = v7.RANK
MEMORY_SLOTS = v7.MEMORY_SLOTS
TEMPORAL_WIDTH = v7.TEMPORAL_WIDTH

# The scientific protocol is inherited byte-for-byte in behavior from V7.
_train_core = v7._train_core
_evaluate_all = v7._evaluate_all
_development_gate = v7._development_gate


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "runner": v7.v6.v5._sha256(Path(__file__).resolve()),
        "core": v7.v6.v5._sha256(
            root / "src/angler/reasoning/sparse_memory_causal_routing.py"
        ),
        "corpus": v7.v6.v5._sha256(
            root / "experiments/corpora/causal_neuromodulated_apprenticeship_v6.py"
        ),
    }


def _load_semantic_donor(
    core: SparseMemoryCausalRoutingCore,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != v7.IDENTITY:
        raise RuntimeError("V7 semantic donor identity mismatch")
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
                raise RuntimeError("V7 semantic donor tensor shape mismatch")
            parameters[name].copy_(
                donor[name].to(device=parameters[name].device, dtype=parameters[name].dtype)
            )
    loaded = core.state_dict()
    if any(not torch.equal(loaded[name].cpu(), donor[name]) for name in names):
        raise RuntimeError("V7 semantic donor was not loaded byte-exactly")
    return {
        "path": str(checkpoint_path),
        "sha256": v7.v6.v5._sha256(checkpoint_path),
        "mapping": {name: name for name in names},
    }


def _checkpoint_payload(
    core: SparseMemoryCausalRoutingCore,
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
        "routing_k": SparseMemoryCausalRoutingCore.routing_k,
        "qwen_digest": qwen_digest,
        "core_digest": v7.v6.v5._module_digest(core),
        "source_hashes": _source_hashes(),
        "development_metrics": metrics,
        "core_state_dict": {
            name: tensor.detach().cpu() for name, tensor in core.state_dict().items()
        },
    }


def _new_core(content_width: int, device: torch.device) -> SparseMemoryCausalRoutingCore:
    return SparseMemoryCausalRoutingCore(
        content_width=content_width,
        temporal_width=TEMPORAL_WIDTH,
        rank=RANK,
        memory_slots=MEMORY_SLOTS,
    ).to(device=device, dtype=torch.float32)


def _run_train_development(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V8 train-development identity is already consumed")

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
    donor = _load_semantic_donor(core, Path(args.v7_checkpoint))
    training = _train_core(core, train_rows)
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, development_rows)
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
            "routing_k": SparseMemoryCausalRoutingCore.routing_k,
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
        "v7_semantic_donor": donor,
        "source_hashes": _source_hashes(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": v7.v6.v5._sha256(checkpoint_path),
        "environment": v7.v6.v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "Development tests bounded sparse outcome-causal memory routing in "
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
                "effective_write_slots": metrics["memory_structure"][
                    "aggregate_effective_write_slots"
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
        raise FileExistsError("V8 final identity is already consumed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    v7.v6.v5._validate_final_authorization(
        development.get("development_authorized") is True
    )
    if development.get("checkpoint_sha256") != v7.v6.v5._sha256(checkpoint_path):
        raise RuntimeError("V8 checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != IDENTITY:
        raise RuntimeError("V8 checkpoint identity mismatch")
    if checkpoint.get("source_hashes") != _source_hashes():
        raise RuntimeError("V8 source identity mismatch")

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
        raise RuntimeError("V8 core digest mismatch")
    core_before = v7.v6.v5._module_digest(core)
    metrics = _evaluate_all(core, rows)
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
            "Supported would establish bounded sparse outcome-causal memory routing "
            "only, not arbitrary intelligence, AGI, or deployment readiness."
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
        "--v7-checkpoint",
        default="/opt/angler/results/content-addressed-causal-memory-v7.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/sparse-memory-causal-routing-v8.pt",
    )
    parser.add_argument(
        "--development-result",
        default="/opt/angler/results/sparse-memory-causal-routing-v8-development.json",
    )
    parser.add_argument(
        "--final-result",
        default="/opt/angler/results/sparse-memory-causal-routing-v8-final.json",
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
