"""Evaluation-only recovery for V8's preserved post-training checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from angler.reasoning import DonorCandidateFusion, PlasticProcedureState
from experiments.runners import compositional_procedure_v4_r1 as v4r1
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import scaled_procedural_software_v1 as v1
from experiments.runners import structured_procedure_decoder_v3 as v3
from experiments.runners import structured_relational_transfer_v7 as v7
from experiments.runners import trifecta_correspondence_composition_v8 as v8


IDENTITY = "angler.trifecta-correspondence-composition.v8-evaluation-recovery-r1"
CHECKPOINT_SHA256 = (
    "210b439ea1295c7545e5283e30e8243d1c47cec8e2f73607681ff3100c8a562c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--v19-checkpoint", required=True)
    parser.add_argument("--v20-checkpoint", required=True)
    parser.add_argument("--v8-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--angler-device", default="cuda:0")
    return parser.parse_args()


def restore_plastic_state(record, device: torch.device) -> PlasticProcedureState:
    if not isinstance(record, dict) or set(record) != {
        "keys",
        "values",
        "strengths",
        "step",
    }:
        raise RuntimeError("V8 recovery plastic-state fields changed")
    return PlasticProcedureState(
        keys=record["keys"].to(device),
        values=record["values"].to(device),
        strengths=record["strengths"].to(device),
        step=int(record["step"]),
    )


def _control_persistent(core, train, embeddings, fusion, banks, full):
    modes = {
        "donor_removed": (banks["full"], False, False),
        "oml_removed": (banks["oml_removed"], True, False),
        "paired_graph_removed": (banks["paired_graph_removed"], True, False),
        "cognee_unrelated": (banks["cognee_unrelated"], True, True),
    }
    states = {name: v4r1._anchored_initial_state(core) for name in modes}
    for stream in train:
        for name, (bank, include_donor, unrelated) in modes.items():
            states[name] = v8._advance_supports(
                core,
                stream,
                embeddings,
                states[name],
                fusion,
                bank,
                include_donor=include_donor,
                core_unrelated=unrelated,
            )
    return {"full": full, **states}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("V8 recovered result already exists")
    if v8._sha256(args.v8_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("V8 recovery checkpoint identity changed")
    for path, expected, label in (
        (args.parent_checkpoint, v8.PARENT_CHECKPOINT_SHA256, "V6 parent"),
        (args.embedding_cache, v8.EMBEDDING_CACHE_SHA256, "embedding cache"),
        (args.v19_checkpoint, v8.V19_CHECKPOINT_SHA256, "V19 donor"),
        (args.v20_checkpoint, v8.V20_CHECKPOINT_SHA256, "V20 donor"),
    ):
        if v8._sha256(path) != expected:
            raise RuntimeError(f"V8 recovery {label} changed")
    device = torch.device(args.angler_device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    evidence, prepared_train, development, final = v1.prepare_corpus()
    train = v4r1._public_training_streams(prepared_train)
    raw_groups = v8._raw_corpus()
    support_by_key, task_by_query_text, pair_by_ref = v8._episode_maps(raw_groups)
    sealed = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    cached = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
    texts = v3._needed_texts(evidence, (train, development, final))
    if cached["foundation_digest"] != sealed["foundation_digest"] or tuple(cached["texts"]) != texts:
        raise RuntimeError("V8 recovery cache/corpus binding changed")
    embeddings = {text: cached["encoded"][index] for index, text in enumerate(texts)}
    for ref, text in evidence.items():
        embeddings[ref] = embeddings[text]

    recovered = torch.load(args.v8_checkpoint, map_location="cpu", weights_only=True)
    if (
        recovered.get("identity") != v8.IDENTITY
        or recovered.get("source", {}).get("parent_checkpoint_sha256")
        != v8.PARENT_CHECKPOINT_SHA256
        or recovered.get("foundation_digest") != sealed["foundation_digest"]
    ):
        raise RuntimeError("V8 recovery payload identity changed")
    core, decoder = v7._load_parent(sealed, device)
    decoder.load_state_dict(recovered["decoder_state_dict"], strict=True)
    decoder.requires_grad_(False)
    fusion = DonorCandidateFusion(**recovered["fusion_config"]).to(device)
    fusion.load_state_dict(recovered["fusion_state_dict"], strict=True)
    fusion.requires_grad_(False)
    core.eval()
    decoder.eval()
    fusion.eval()
    full_persistent = restore_plastic_state(recovered["plastic_state"], device)

    donor_system = v20.load_oml_checkpoint(
        args.v20_checkpoint, args.v19_checkpoint, device=device
    )
    prepared_groups = (train, development, final)
    banks = {
        "full": v8.build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
        ),
        "oml_removed": v8.build_donor_bank(
            donor_system.source_v19.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
        ),
        "paired_graph_removed": v8.build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
            lesion="zero_residual",
        ),
        "cognee_unrelated": v8.build_donor_bank(
            donor_system.second_order_oml.controller,
            prepared_groups,
            support_by_key,
            task_by_query_text,
            pair_by_ref,
            unrelated=True,
        ),
    }
    donor_system = None
    persistent = _control_persistent(
        core, train, embeddings, fusion, banks, full_persistent
    )
    development_metrics, development_attribution, development_rows = v8._evaluate(
        core, decoder, development, embeddings, fusion, banks, persistent
    )
    final_metrics, final_attribution, final_rows = v8._evaluate(
        core, decoder, final, embeddings, fusion, banks, persistent
    )
    full = final_metrics["full"]
    supported = (
        development_metrics["full"] >= 0.60
        and full >= 0.60
        and full - final_metrics["donor_removed"] >= 0.20
        and full - final_metrics["oml_removed"] >= 0.10
        and full - final_metrics["moving_origin_removed"] >= 0.10
        and full - final_metrics["cognee_unrelated"] >= 0.10
        and full - final_metrics["procedure_slots_removed"] >= 0.10
        and final_attribution >= 0.75
    )
    report = {
        "identity": IDENTITY,
        "base_identity": v8.IDENTITY,
        "classification": (
            "TRIFECTA_COMPOSITION_SUPPORTED" if supported else "NOT_SUPPORTED"
        ),
        "component_supported": supported,
        "recovery": {
            "mode": "evaluation_only_from_preserved_post_training_checkpoint",
            "reason": "original runner shadowed the output Path after evaluation",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "training_repeated": False,
            "checkpoint_changed": False,
            "original_training_metrics": "unavailable_from_failed_report_write",
        },
        "development_metrics": development_metrics,
        "development_target_evidence_attribution": development_attribution,
        "development": development_rows,
        "metrics": final_metrics,
        "target_evidence_attribution": final_attribution,
        "evaluation": final_rows,
        "source": recovered["source"],
        "runtime": {
            "device": str(device),
            "name": torch.cuda.get_device_name(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "recovery_wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "deterministic_solver_used": False,
        "hidden_fields_used_for_training": False,
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "identity",
                    "classification",
                    "development_metrics",
                    "metrics",
                    "target_evidence_attribution",
                    "recovery",
                    "runtime",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

