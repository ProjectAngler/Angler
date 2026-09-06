"""Live dual-GPU conversation and first-result evaluation for situated Angler."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from angler.memory import MemoryProjection, SituatedMemory
from angler.reasoning import LearnedSituatedMemoryReader
from angler.runtime import (
    ConversationJournal,
    HybridChunkCogneeProjectionBackend,
    JournalRecord,
    LocalQwenIO,
    SituatedConversationSession,
    build_conversation_prompt,
    build_qwen_prompt,
    foundation_tensor_digest,
    freeze_knowledge_model,
    select_situated_evidence,
)
from experiments.runners.situated_memory_value_v1 import ACTIONS
from experiments.runners.situated_memory_value_v2 import SPEC_V2
from experiments.runners.situated_qwen_generation_v1 import parse_action


SEED = 20260831
READER_SHA256 = "b94e27ad0a42e3f499b7de5ff1b5217463dadd38e60934a7af7f3ebd2da4e3d4"
SOURCE_RESULT_SHA256 = "2d726321c3d8136cecd8bebba5cfa611d18d565587fa0cc0f93614c89124207f"
SOURCE_QWEN_SHA256 = "a3e9e04c7a709c913e05682055c0317a2ad8c28eaae30d370433f4ba686a1a42"
V3_RESULT_SHA256 = "0f149262462293a686515e04168a520f59a53b691c266bd82ea13b900e25d493"
RESPONSE_CONSTRAINT = (
    "The first word of your response must be the final procedure name. "
    "Return only one of: amber, cobalt, jade, violet."
)
RECALL_LIMIT = 12
_WORKSPACE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256(b"angler.situated-conversation.module.v1\0")
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _configure_cognee(storage_root: Path) -> Any:
    if os.getenv("TELEMETRY_DISABLED") != "1":
        raise RuntimeError("TELEMETRY_DISABLED=1 is required before Cognee import")
    import cognee

    storage_root.mkdir(parents=True, exist_ok=True)
    cognee.config.system_root_directory(str(storage_root / "system"))
    cognee.config.data_root_directory(str(storage_root / "data"))
    cognee.config.set_llm_provider("ollama")
    cognee.config.set_llm_model("llama3.2:3b")
    cognee.config.set_llm_endpoint("http://127.0.0.1:11434")
    cognee.config.set_llm_api_key("ollama")
    cognee.config.set_embedding_provider("fastembed")
    cognee.config.set_embedding_model("BAAI/bge-small-en-v1.5")
    cognee.config.set_embedding_dimensions(384)
    return cognee


def _load_reader(checkpoint: Path, device: torch.device) -> LearnedSituatedMemoryReader:
    if _sha256(checkpoint) != READER_SHA256:
        raise RuntimeError("sealed situated reader checkpoint identity mismatches")
    sealed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reader = LearnedSituatedMemoryReader(
        content_width=int(sealed["content_width"]),
        temporal_width=SPEC_V2.width,
        hidden_width=256,
        action_count=len(ACTIONS),
    ).to(device)
    reader.load_state_dict(sealed["state_dict"], strict=True)
    reader.requires_grad_(False)
    reader.eval()
    return reader


def _load_qwen(model_path: str, device: torch.device) -> LocalQwenIO:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
    )
    freeze_knowledge_model(model)
    return LocalQwenIO(
        model,
        tokenizer,
        embedding_batch_size=32,
        max_new_tokens=64,
        enable_thinking=False,
    )


def _projection(raw: dict[str, object]) -> MemoryProjection:
    return MemoryProjection(
        artifact_ref=raw["artifact_ref"],  # type: ignore[arg-type]
        text=raw["text"],  # type: ignore[arg-type]
        source_ref=raw["source_ref"],  # type: ignore[arg-type]
        visibility=raw["visibility"],  # type: ignore[arg-type]
        context=tuple(tuple(pair) for pair in raw["context"]),  # type: ignore[arg-type]
    )


def _validate_chat_readiness(path: Path) -> None:
    if _sha256(path) != V3_RESULT_SHA256:
        raise RuntimeError("chat readiness result identity mismatches")
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.get("metrics", {})
    frozen = result.get("frozen", {})
    supported = (
        result.get("identity") == "angler.situated-conversation.v3-hybrid-first-result"
        and result.get("classification") == "CONVERSATION_PROTOTYPE_READY"
        and metrics.get("angler_selected_accuracy", 0.0) >= 0.75
        and metrics.get("angler_selected_accuracy", 0.0)
        - metrics.get("qwen_alone_accuracy", 1.0)
        >= 0.25
        and metrics.get("angler_selected_accuracy", 0.0)
        - metrics.get("fair_rag_accuracy", 1.0)
        >= 0.25
        and metrics.get("target_coverage", 0.0) >= 0.90
        and metrics.get("selector_max_causal_drop", 0.0) >= 0.20
        and metrics.get("feedback_recalled_next_turn") is True
        and metrics.get("two_turn_history_visible") is True
        and frozen.get("qwen_before") == frozen.get("qwen_after")
        and frozen.get("reader_before") == frozen.get("reader_after")
        and frozen.get("reader_checkpoint_sha256") == READER_SHA256
    )
    if not supported:
        raise RuntimeError("chat readiness result does not satisfy frozen conditions")


async def _seed_family(
    row: dict[str, object],
    *,
    state_root: Path,
    dataset: str,
) -> tuple[SituatedMemory, ConversationJournal]:
    journal = ConversationJournal(state_root / "journals" / f"{dataset}.jsonl")
    if journal.records():
        raise RuntimeError(f"fresh evaluation journal unexpectedly exists: {dataset}")
    for raw in row["projections"]:  # type: ignore[index]
        projection = _projection(raw)
        journal.append(
            JournalRecord(
                projection,
                current_regime=projection.artifact_ref == row["target_ref"],
            )
        )
    backend = HybridChunkCogneeProjectionBackend(dataset, local_models_configured=True)
    memory = SituatedMemory(backend, origin=journal.build_origin())
    await journal.rebuild_projection(memory, forget_existing=False)
    return memory, journal


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("situated conversation V2 result already exists")
    checkpoint = Path(args.checkpoint)
    source_result = Path(args.source_result)
    source_qwen = Path(args.source_qwen_result)
    if _sha256(source_result) != SOURCE_RESULT_SHA256:
        raise RuntimeError("situated-memory V2 source result identity mismatches")
    if _sha256(source_qwen) != SOURCE_QWEN_SHA256:
        raise RuntimeError("situated-Qwen V2 source result identity mismatches")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("live evaluation requires both workstation GPUs")
    state_root = Path(args.storage_root)
    if state_root.exists() and any(state_root.iterdir()):
        raise RuntimeError("first-result storage root must be fresh")
    cognee = _configure_cognee(state_root)
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = source["rows"]
    if len(rows) != 8:
        raise RuntimeError("the frozen evaluation requires eight family rows")

    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    for device_index in range(2):
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
    memories: list[SituatedMemory] = []
    journals: list[ConversationJournal] = []
    recalls = []
    frozen_recalls = []
    for index, row in enumerate(rows):
        memory, journal = await _seed_family(
            row,
            state_root=state_root,
            dataset=f"angler_situated_conversation_v2_family_{index}",
        )
        recall = await memory.recall(str(row["query_text"]), limit=RECALL_LIMIT)
        frozen = await memory.recall(
            str(row["query_text"]), limit=RECALL_LIMIT, frozen_origin=True
        )
        memories.append(memory)
        journals.append(journal)
        recalls.append(recall)
        frozen_recalls.append(frozen)

    # Cognee's self-hosted ingestion checks/uses Ollama. Complete that bounded
    # projection stage before occupying the 5080 with Qwen; both neural models
    # remain absent from and uninvolved in retrieval construction.
    qwen = _load_qwen(args.model, torch.device("cuda:0"))
    reader = _load_reader(checkpoint, torch.device("cuda:1"))
    qwen_before = foundation_tensor_digest(qwen.model)
    reader_before = _module_digest(reader)

    all_texts = tuple(
        dict.fromkeys(
            text
            for row, recall in zip(rows, recalls, strict=True)
            for text in (str(row["query_text"]), *(item.text for item in recall.items))
        )
    )
    encoded = qwen.embed(all_texts)
    embeddings = {text: encoded[index] for index, text in enumerate(all_texts)}

    torch.manual_seed(SEED + 1)
    reset_reader = LearnedSituatedMemoryReader(
        content_width=reader.content_width,
        temporal_width=reader.temporal_width,
        hidden_width=reader.hidden_width,
        action_count=reader.action_count,
    ).to("cuda:1")
    reset_reader.requires_grad_(False)
    reset_reader.eval()

    selectors: dict[str, list[bool]] = {
        "full": [],
        "moving_origin_removed": [],
        "reader_reset": [],
        "retrieval_removed": [],
    }
    selections = []
    row_records = []
    prompts: list[tuple[int, str, str]] = []
    for index, (row, recall, frozen) in enumerate(zip(rows, recalls, frozen_recalls, strict=True)):
        query = str(row["query_text"])
        query_features = embeddings[query]

        def select(active_reader, active_recall, now):
            return select_situated_evidence(
                active_reader,
                query_features=query_features,
                candidate_features=torch.stack([embeddings[item.text] for item in active_recall.items]),
                recall=active_recall,
                now=now,
                spec=SPEC_V2,
            )

        full = select(reader, recall, memories[index].origin.now)
        moving_removed = select(reader, frozen, memories[index].origin.now)
        reset = select(reset_reader, recall, memories[index].origin.now)
        mismatch = recalls[(index + 1) % len(recalls)]
        mismatch_texts = tuple(item.text for item in mismatch.items)
        missing = [text for text in mismatch_texts if text not in embeddings]
        if missing:
            extra = qwen.embed(missing)
            embeddings.update({text: extra[position] for position, text in enumerate(missing)})
        retrieval_removed = select(
            reader, mismatch, memories[(index + 1) % len(memories)].origin.now
        )
        selections.append(full)
        target_ref = str(row["target_ref"])
        selected_by_arm = {
            "full": full,
            "moving_origin_removed": moving_removed,
            "reader_reset": reset,
            "retrieval_removed": retrieval_removed,
        }
        for name, value in selected_by_arm.items():
            selectors[name].append(value.artifact_ref == target_ref)
        for mode, prompt in (
            ("qwen_alone", build_qwen_prompt(query, response_constraint=RESPONSE_CONSTRAINT)),
            ("fair_rag", build_qwen_prompt(query, fair_rag=recall, response_constraint=RESPONSE_CONSTRAINT)),
            ("angler_selected", build_conversation_prompt(query, selected=full, response_constraint=RESPONSE_CONSTRAINT)),
        ):
            prompts.append((index, mode, prompt))
        row_records.append(
            {
                "family_index": row["family_index"],
                "target_ref": target_ref,
                "target_present": any(item.artifact_ref == target_ref for item in recall.items),
                "selected_refs": {name: value.artifact_ref for name, value in selected_by_arm.items()},
                "full_attention": full.attention,
            }
        )

    generated = [qwen.generate(prompt) for _, _, prompt in prompts]
    generation_correct: dict[str, list[bool]] = {
        "qwen_alone": [],
        "fair_rag": [],
        "angler_selected": [],
    }
    for (index, mode, prompt), response in zip(prompts, generated, strict=True):
        target_raw = rows[index]["target_action"]
        target_index = int(target_raw) if isinstance(target_raw, int) else ACTIONS.index(target_raw)
        parsed = parse_action(response)
        correct = parsed == ACTIONS[target_index]
        generation_correct[mode].append(correct)
        row_records[index].setdefault("generation", {})[mode] = {  # type: ignore[index]
            "prompt": prompt,
            "response": response,
            "parsed": parsed,
            "correct": correct,
        }

    session = SituatedConversationSession(
        memories[0],
        journals[0],
        reader,
        spec=SPEC_V2,
        embed=qwen.embed,
        generate=qwen.generate,
        max_history_messages=4,
    )
    feedback_first = await session.answer(
        str(rows[0]["query_text"]),
        recall_limit=RECALL_LIMIT,
        response_constraint=RESPONSE_CONSTRAINT,
    )
    feedback_record = await session.record_feedback(
        outcome="success",
        procedure_note="use the externally verified amber procedure for dependency repair",
        family="dependency repair",
        source_ref="sha256:" + hashlib.sha256(b"angler-v2-synthetic-human-feedback").hexdigest(),
    )
    feedback_second = await session.answer(
        "Recall the externally verified dependency repair procedure and explain why it applies.",
        recall_limit=RECALL_LIMIT,
        response_constraint="Answer concisely and identify the remembered procedure.",
    )
    feedback_recalled = any(
        item.artifact_ref == feedback_record.projection.artifact_ref
        for item in feedback_second.recalled.items
    )
    history_visible = (
        "Recent conversation (context only):" in feedback_second.prompt
        and feedback_first.query in feedback_second.prompt
        and feedback_first.response in feedback_second.prompt
    )

    qwen_after = foundation_tensor_digest(qwen.model)
    reader_after = _module_digest(reader)
    target_coverage = sum(bool(row["target_present"]) for row in row_records) / len(row_records)
    generation_accuracy = {
        name: sum(values) / len(values) for name, values in generation_correct.items()
    }
    selector_accuracy = {name: sum(values) / len(values) for name, values in selectors.items()}
    causal_drop = selector_accuracy["full"] - min(
        selector_accuracy["moving_origin_removed"],
        selector_accuracy["reader_reset"],
        selector_accuracy["retrieval_removed"],
    )
    supported = (
        generation_accuracy["angler_selected"] >= 0.75
        and generation_accuracy["angler_selected"] - generation_accuracy["qwen_alone"] >= 0.25
        and generation_accuracy["angler_selected"] - generation_accuracy["fair_rag"] >= 0.25
        and target_coverage >= 0.90
        and causal_drop >= 0.20
        and feedback_recalled
        and history_visible
        and qwen_before == qwen_after
        and reader_before == reader_after
    )
    torch.cuda.synchronize()
    report = {
        "identity": "angler.situated-conversation.v3-hybrid-first-result",
        "classification": "CONVERSATION_PROTOTYPE_READY" if supported else "NOT_READY",
        "training_performed": False,
        "metrics": {
            **{f"{name}_accuracy": value for name, value in generation_accuracy.items()},
            **{f"selector_{name}_accuracy": value for name, value in selector_accuracy.items()},
            "selector_max_causal_drop": causal_drop,
            "target_coverage": target_coverage,
            "feedback_recalled_next_turn": feedback_recalled,
            "two_turn_history_visible": history_visible,
        },
        "frozen": {
            "qwen_before": qwen_before,
            "qwen_after": qwen_after,
            "reader_before": reader_before,
            "reader_after": reader_after,
            "reader_checkpoint_sha256": _sha256(checkpoint),
        },
        "inputs": {
            "source_memory_result_sha256": _sha256(source_result),
            "source_qwen_result_sha256": _sha256(source_qwen),
            "live_input_sha256": _sha256(Path(args.input)),
        },
        "feedback": {
            "artifact_ref": feedback_record.projection.artifact_ref,
            "journal_sha256": journals[0].digest(),
            "second_turn_selected_ref": None if feedback_second.selected is None else feedback_second.selected.artifact_ref,
        },
        "rows": row_records,
        "runtime": {
            "cognee_projection_mode": "just_chunks_vector_plus_lexical",
            "recall_limit": RECALL_LIMIT,
            "qwen_device": "cuda:0",
            "qwen_gpu": torch.cuda.get_device_name(0),
            "reader_device": "cuda:1",
            "reader_gpu": torch.cuda.get_device_name(1),
            "peak_allocated_bytes": {
                "cuda:0": torch.cuda.max_memory_allocated(0),
                "cuda:1": torch.cuda.max_memory_allocated(1),
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "cognee": cognee.__version__,
            "wall_seconds": time.monotonic() - started,
        },
        "limits": [
            "The capability result remains bounded to the same eight synthetic mechanism families.",
            "Feedback persistence is memory acquisition, not a supported online neural-learning claim.",
            "No autonomous grading, tool execution, arbitrary cross-domain reasoning, AGI, or production claim.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


async def chat(args: argparse.Namespace) -> None:
    if not _WORKSPACE.fullmatch(args.workspace):
        raise ValueError("workspace must match [a-z0-9][a-z0-9_-]{0,63}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("chat requires both workstation GPUs")
    _validate_chat_readiness(Path(args.readiness_result))
    state_root = Path(args.storage_root)
    _configure_cognee(state_root)
    journal = ConversationJournal(state_root / "journals" / f"{args.workspace}.jsonl")
    backend = HybridChunkCogneeProjectionBackend(
        f"angler_situated_conversation_{args.workspace}", local_models_configured=True
    )
    memory = SituatedMemory(backend, origin=journal.build_origin())
    if args.rebuild_memory:
        await journal.rebuild_projection(memory)
    qwen = _load_qwen(args.model, torch.device("cuda:0"))
    reader = _load_reader(Path(args.checkpoint), torch.device("cuda:1"))
    session = SituatedConversationSession(
        memory,
        journal,
        reader,
        spec=SPEC_V2,
        embed=qwen.embed,
        generate=qwen.generate,
        max_history_messages=args.history_messages,
    )
    print("Angler's validated bounded conversation prototype is local and ready. Commands: /success family | note, /failure family | note, /inspect, /rebuild, /quit")
    while True:
        query = input("you> ").strip()
        if not query:
            continue
        if query == "/quit":
            return
        if query == "/inspect":
            print(json.dumps({"journal": journal.digest(), "records": len(journal.records()), "origin_now": memory.origin.now}, sort_keys=True))
            continue
        if query == "/rebuild":
            await session.rebuild_memory()
            print("angler> disposable Cognee projection rebuilt from the journal")
            continue
        if query.startswith(("/success ", "/failure ")):
            outcome = "success" if query.startswith("/success ") else "failure"
            raw = query.split(" ", 1)[1]
            if "|" not in raw:
                print("angler> feedback format: /success family | procedure note")
                continue
            family, note = (item.strip() for item in raw.split("|", 1))
            record = await session.record_feedback(
                outcome=outcome,
                procedure_note=note,
                family=family,
                source_ref="sha256:" + hashlib.sha256(f"human:{args.workspace}".encode()).hexdigest(),
            )
            print(f"angler> feedback remembered as {record.projection.artifact_ref}")
            continue
        turn = await session.answer(query, recall_limit=RECALL_LIMIT)
        if turn.selected is None:
            print("evidence> none (visible Qwen-alone fallback)")
        else:
            print(f"evidence> {turn.selected.artifact_ref} attention={turn.selected.attention:.4f}\n{turn.selected.text}")
        print(f"angler> {turn.response}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    evaluate_parser.add_argument("--input", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--source-result", required=True)
    evaluate_parser.add_argument("--source-qwen-result", required=True)
    evaluate_parser.add_argument("--storage-root", required=True)
    evaluate_parser.add_argument("--output", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    chat_parser.add_argument("--checkpoint", required=True)
    chat_parser.add_argument("--storage-root", default="/opt/angler/state/situated-conversation-v3/live")
    chat_parser.add_argument(
        "--readiness-result",
        default="/opt/angler/results/situated-conversation-v3-evaluation.json",
    )
    chat_parser.add_argument("--workspace", default="general")
    chat_parser.add_argument("--history-messages", type=int, default=8)
    chat_parser.add_argument("--rebuild-memory", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate":
        result = asyncio.run(evaluate(args))
        print(json.dumps({"classification": result["classification"], "metrics": result["metrics"], "runtime": result["runtime"]}, sort_keys=True))
    else:
        asyncio.run(chat(args))


if __name__ == "__main__":
    main()
