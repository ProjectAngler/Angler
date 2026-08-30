"""Self-hosted Cognee retrieval stage for the situated-memory V2 result."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cognee

from angler.memory import CogneeProjectionBackend, MemoryProjection, SituatedMemory


async def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.storage_root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Cognee live storage must be fresh for this identity")
    root.mkdir(parents=True, exist_ok=True)
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_llm_provider("ollama")
    cognee.config.set_llm_model("llama3.2:3b")
    cognee.config.set_llm_endpoint("http://127.0.0.1:11434")
    cognee.config.set_llm_api_key("ollama")
    cognee.config.set_embedding_provider("fastembed")
    cognee.config.set_embedding_model("BAAI/bge-small-en-v1.5")
    cognee.config.set_embedding_dimensions(384)

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    started = time.monotonic()
    output = []
    for row in source["rows"]:
        dataset = f"angler_situated_v2_family_{row['family_index']}"
        backend = CogneeProjectionBackend(dataset, local_models_configured=True)
        memory = SituatedMemory(backend)
        projections = tuple(
            MemoryProjection(
                artifact_ref=item["artifact_ref"],
                text=item["text"],
                source_ref=item["source_ref"],
                visibility=item["visibility"],
                context=tuple(tuple(value) for value in item["context"]),
            )
            for item in row["projections"]
        )
        await memory.remember_many(projections)
        marker = "sha256:" + hashlib.sha256(
            f"{dataset}:designation".encode()
        ).hexdigest()
        memory.origin.designate_landmark(
            "current_regime",
            target_event_ref=row["target_ref"],
            designation_event_ref=marker,
            projection_id=marker,
        )
        recalled = await memory.recall(row["query_text"], limit=12)
        target_present = any(
            item.artifact_ref == row["target_ref"] for item in recalled.items
        )
        output.append(
            {
                "family_index": row["family_index"],
                "query_text": row["query_text"],
                "target_action": row["target_action"],
                "target_ref": row["target_ref"],
                "target_present": target_present,
                "items": [asdict(item) for item in recalled.items],
                "rejected": list(recalled.rejected),
                "origin_now": memory.origin.now,
            }
        )
        await memory.forget_projection()
        print(
            json.dumps(
                {
                    "family": row["family_index"],
                    "recalled": len(recalled.items),
                    "target_present": target_present,
                    "rejected": list(recalled.rejected),
                }
            ),
            flush=True,
        )
    return {
        "identity": "angler.situated-memory-live-recall.v2",
        "cognee_version": cognee.__version__,
        "embedding_provider": "fastembed",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_provider": "ollama",
        "llm_model": "llama3.2:3b",
        "telemetry_disabled": True,
        "rows": output,
        "wall_seconds": time.monotonic() - started,
        "datasets_forgotten": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-root", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("live recall output already exists")
    result = asyncio.run(run(args))
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": True, "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
