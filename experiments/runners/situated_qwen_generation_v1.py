"""First same-Qwen generation test for learned situated memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from angler.reasoning import LearnedSituatedMemoryReader
from angler.runtime import (
    build_qwen_prompt,
    encode_detached_segments,
    freeze_knowledge_model,
    select_situated_evidence,
)
from experiments.runners.situated_memory_cognee_evaluate_v2 import _recall, _target_index
from experiments.runners.situated_memory_value_v1 import ACTIONS
from experiments.runners.situated_memory_value_v2 import SPEC_V2


SEED = 20260830
_ACTION_PATTERN = re.compile(r"\b(?:amber|cobalt|jade|violet)\b", re.IGNORECASE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_action(response: str) -> str | None:
    found = {match.group(0).lower() for match in _ACTION_PATTERN.finditer(response)}
    return next(iter(found)) if len(found) == 1 else None


def _generate(
    model: torch.nn.Module,
    tokenizer: object,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    responses: list[str] = []
    prior_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in batch
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[:, encoded["input_ids"].shape[1] :]
            responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prior_padding
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--recall", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("situated Qwen generation requires CUDA")
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("situated Qwen result already exists")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    recall_path = Path(args.recall)
    checkpoint_path = Path(args.checkpoint)
    source = json.loads(recall_path.read_text(encoding="utf-8"))
    rows = source["rows"]
    if len(rows) != 8:
        raise RuntimeError("the frozen V1 generation identity requires eight rows")
    recalls = tuple(_recall(row) for row in rows)
    started = time.monotonic()
    torch.manual_seed(SEED + 4)
    torch.cuda.manual_seed_all(SEED + 4)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    texts = tuple(
        text
        for row, recall in zip(rows, recalls, strict=True)
        for text in (row["query_text"], *(item.text for item in recall.items))
    )
    unique = tuple(dict.fromkeys(texts))
    encoded = encode_detached_segments(
        model,
        tokenizer,
        unique,
        batch_size=args.embedding_batch_size,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: encoded[index].float() for index, text in enumerate(unique)}

    sealed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reader = LearnedSituatedMemoryReader(
        content_width=int(sealed["content_width"]),
        temporal_width=SPEC_V2.width,
        hidden_width=256,
        action_count=len(ACTIONS),
    ).cuda()
    reader.load_state_dict(sealed["state_dict"], strict=True)
    reader.eval()

    response_constraint = (
        "Return exactly one procedure name: amber, cobalt, jade, or violet."
    )
    selections = []
    prompt_records = []
    prompts = []
    for row, recall in zip(rows, recalls, strict=True):
        selection = select_situated_evidence(
            reader,
            query_features=embeddings[row["query_text"]],
            candidate_features=torch.stack([embeddings[item.text] for item in recall.items]),
            recall=recall,
            now=int(row["origin_now"]),
            spec=SPEC_V2,
        )
        selections.append(selection)
        for mode, prompt in (
            (
                "qwen_alone",
                build_qwen_prompt(
                    row["query_text"],
                    response_constraint=response_constraint,
                ),
            ),
            (
                "fair_rag",
                build_qwen_prompt(
                    row["query_text"],
                    fair_rag=recall,
                    response_constraint=response_constraint,
                ),
            ),
            (
                "angler_selected",
                build_qwen_prompt(
                    row["query_text"],
                    selected=selection,
                    response_constraint=response_constraint,
                ),
            ),
        ):
            prompts.append(prompt)
            prompt_records.append((int(row["family_index"]), mode, prompt))

    generated = _generate(
        model,
        tokenizer,
        prompts,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    by_family: dict[int, dict[str, object]] = {
        int(row["family_index"]): {
            "family_index": int(row["family_index"]),
            "target": ACTIONS[_target_index(row["target_action"])],
            "target_present": bool(row["target_present"]),
            "selected_ref": selections[index].artifact_ref,
            "selected_attention": selections[index].attention,
            "arms": {},
        }
        for index, row in enumerate(rows)
    }
    for (family_index, mode, prompt), response in zip(prompt_records, generated, strict=True):
        parsed = parse_action(response)
        target = by_family[family_index]["target"]
        by_family[family_index]["arms"][mode] = {
            "prompt": prompt,
            "response": response,
            "parsed": parsed,
            "correct": parsed == target,
        }

    family_rows = [by_family[int(row["family_index"])] for row in rows]
    accuracy = {
        mode: sum(bool(row["arms"][mode]["correct"]) for row in family_rows) / len(family_rows)
        for mode in ("qwen_alone", "fair_rag", "angler_selected")
    }
    target_coverage = sum(bool(row["target_present"]) for row in family_rows) / len(family_rows)
    supported = (
        accuracy["angler_selected"] >= 0.75
        and accuracy["angler_selected"] - accuracy["qwen_alone"] >= 0.25
        and accuracy["angler_selected"] - accuracy["fair_rag"] >= 0.25
        and target_coverage >= 0.90
    )
    torch.cuda.synchronize()
    report = {
        "identity": "angler.situated-qwen-generation.v1",
        "classification": (
            "QWEN_GENERATION_BENEFIT_SUPPORTED" if supported else "NOT_SUPPORTED"
        ),
        "training_performed": False,
        "metrics": {
            **{f"{name}_accuracy": value for name, value in accuracy.items()},
            "target_coverage": target_coverage,
        },
        "thresholds": {
            "minimum_angler_accuracy": 0.75,
            "minimum_gain_over_each_control": 0.25,
            "minimum_target_coverage": 0.90,
        },
        "inputs": {
            "live_recall_sha256": _sha256(recall_path),
            "reader_checkpoint_sha256": _sha256(checkpoint_path),
            "foundation_model": args.model,
        },
        "rows": family_rows,
        "runtime": {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "seed": SEED + 4,
        },
        "limits": [
            "Fresh instances of the same eight synthetic mechanisms only.",
            "No online learning occurred in this generation test.",
            "No arbitrary conversation, AGI, consciousness, or production claim.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
