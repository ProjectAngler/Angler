"""Evaluate the sealed V2 reader against live self-hosted Cognee recall.

This stage performs no training.  Qwen supplies detached content features and
the already-consumed V2 reader checkpoint chooses an action from Cognee's
retrieved, Moving-Origin-situated evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from angler.memory import RecallBatch, SituatedRecall
from angler.reasoning import LearnedSituatedMemoryReader, encode_situated_features
from experiments.runners.situated_memory_value_v1 import ACTIONS, encode_texts_qwen
from experiments.runners.situated_memory_value_v2 import SPEC_V2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recall(row: dict[str, object]) -> RecallBatch:
    items = []
    for raw in row["items"]:  # type: ignore[index]
        item = dict(raw)
        item["context"] = tuple(tuple(pair) for pair in item["context"])
        item["landmark_relations"] = tuple(
            tuple(pair) for pair in item["landmark_relations"]
        )
        items.append(SituatedRecall(**item))
    return RecallBatch(tuple(items), tuple(row["rejected"]))  # type: ignore[arg-type]


def _target_index(value: object) -> int:
    target = value if isinstance(value, int) else ACTIONS.index(value)
    if target < 0 or target >= len(ACTIONS):
        raise RuntimeError("live target action is outside the V2 action vocabulary")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--recall", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("live situated-memory evaluation requires CUDA")

    recall_path = Path(args.recall)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("live evaluation output already exists")
    source = json.loads(recall_path.read_text(encoding="utf-8"))
    rows = source["rows"]
    if len(rows) != 8:
        raise RuntimeError("live V2 evaluation requires exactly eight family rows")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    recalls = tuple(_recall(row) for row in rows)
    texts = tuple(
        text
        for row, batch in zip(rows, recalls, strict=True)
        for text in (row["query_text"], *(item.text for item in batch.items))
    )
    embeddings = encode_texts_qwen(
        texts,
        model_path=args.model,
        batch_size=args.embedding_batch_size,
    )
    query = torch.stack([embeddings[row["query_text"]].float() for row in rows])
    candidates = torch.stack(
        [torch.stack([embeddings[item.text].float() for item in batch.items]) for batch in recalls]
    )
    temporal, mask = encode_situated_features(
        recalls,
        now=[int(row["origin_now"]) for row in rows],
        spec=SPEC_V2,
    )

    sealed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reader = LearnedSituatedMemoryReader(
        content_width=int(sealed["content_width"]),
        temporal_width=SPEC_V2.width,
        hidden_width=256,
        action_count=len(ACTIONS),
    ).cuda()
    reader.load_state_dict(sealed["state_dict"], strict=True)
    reader.eval()
    with torch.inference_mode():
        result = reader(
            query.cuda(),
            candidates.cuda(),
            temporal.cuda(),
            mask.cuda(),
        )
    predictions = result.logits.argmax(dim=-1).cpu().tolist()
    targets = [_target_index(row["target_action"]) for row in rows]
    correct = [prediction == target for prediction, target in zip(predictions, targets, strict=True)]
    coverage = [bool(row["target_present"]) for row in rows]
    attention = result.weights.cpu()
    target_weights = []
    details = []
    for index, (row, batch) in enumerate(zip(rows, recalls, strict=True)):
        target_index = next(
            (
                candidate
                for candidate, item in enumerate(batch.items)
                if item.artifact_ref == row["target_ref"]
            ),
            None,
        )
        target_weight = None if target_index is None else float(attention[index, target_index])
        if target_weight is not None:
            target_weights.append(target_weight)
        details.append(
            {
                "family_index": row["family_index"],
                "target": ACTIONS[targets[index]],
                "prediction": ACTIONS[predictions[index]],
                "correct": correct[index],
                "retrieved": len(batch.items),
                "target_present": coverage[index],
                "target_attention": target_weight,
            }
        )

    accuracy = sum(correct) / len(correct)
    target_coverage = sum(coverage) / len(coverage)
    supported = target_coverage >= 0.90 and accuracy >= 0.65
    torch.cuda.synchronize()
    report = {
        "identity": "angler.situated-memory-cognee-live-evaluation.v2",
        "classification": "SUPPORTED_FOR_PROTOTYPE" if supported else "NOT_SUPPORTED",
        "training_performed": False,
        "sealed_reader": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "identity": "angler.situated-memory-value.v2",
        },
        "live_recall": {
            "result_sha256": _sha256(recall_path),
            "cognee_version": source["cognee_version"],
            "target_coverage": target_coverage,
            "mean_recalled": sum(len(batch.items) for batch in recalls) / len(recalls),
            "datasets_forgotten": source["datasets_forgotten"],
            "wall_seconds": source["wall_seconds"],
        },
        "metrics": {
            "action_accuracy": accuracy,
            "mean_target_attention": sum(target_weights) / len(target_weights),
            "target_coverage": target_coverage,
        },
        "thresholds": {"minimum_action_accuracy": 0.65, "minimum_target_coverage": 0.90},
        "rows": details,
        "runtime": {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
        },
        "limits": [
            "Eight synthetic semantic families; this is not unrestricted cross-domain transfer.",
            "Cognee retrieval and Moving Origin coordinates are validated, not Cognee graph reasoning.",
            "No AGI, consciousness, frontier-model, or production-readiness claim.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
