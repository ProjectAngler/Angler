"""First causal value test for Angler + Moving Origin + Cognee candidates.

Cognee retrieval is exercised separately against the same evidence contract.
This runner trains only the small Angler reader over frozen Qwen features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn import functional as F

from angler.memory import (
    FairNaiveTemporalView,
    MovingOriginIndex,
    RecallBatch,
    SituatedRecall,
)
from angler.reasoning import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    encode_situated_features,
)
from angler.runtime.qwen_knowledge import encode_detached_segments, freeze_knowledge_model


SEED = 20260830
ACTIONS = ("amber", "cobalt", "jade", "violet")
FAMILIES = (
    "dependency repair",
    "route recovery",
    "constraint ordering",
    "interface adaptation",
    "instrument calibration",
    "resource balancing",
    "symbolic reconstruction",
    "counterfactual planning",
)
TRAIN_SURFACES = ("granite", "cedar", "silver", "harbor", "meadow", "quartz")
EVAL_SURFACES = ("basalt", "willow", "copper", "island", "prairie", "opal")
SPEC = SituatedFeatureSpec(("current_regime",))


@dataclass(frozen=True, slots=True)
class RawSample:
    query_text: str
    candidate_texts: tuple[str, ...]
    live: RecallBatch
    frozen: RecallBatch
    target_action: int
    target_candidate: int
    family_index: int
    now: int
    maintained_inspected: int
    naive_inspected: int


@dataclass(frozen=True, slots=True)
class TensorDataset:
    query: torch.Tensor
    candidates: torch.Tensor
    live: torch.Tensor
    frozen: torch.Tensor
    mask: torch.Tensor
    target_mask: torch.Tensor
    targets: torch.Tensor
    held_family: torch.Tensor
    maintained_inspected: float
    naive_inspected: float


def _ref(sample: int, event: int, salt: str) -> str:
    payload = f"{salt}:{sample}:{event}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_samples(
    count: int,
    *,
    seed: int,
    family_indices: tuple[int, ...],
    evaluation_surfaces: bool,
) -> tuple[RawSample, ...]:
    if count <= 0 or not family_indices:
        raise ValueError("sample count and family set must be non-empty")
    rng = random.Random(seed)
    surfaces = EVAL_SURFACES if evaluation_surfaces else TRAIN_SURFACES
    samples: list[RawSample] = []
    for sample_index in range(count):
        family_index = family_indices[rng.randrange(len(family_indices))]
        family = FAMILIES[family_index]
        target = rng.randrange(len(ACTIONS))
        origin = MovingOriginIndex()
        rows: list[tuple[str, str, bool, int]] = []

        old_actions = [item for item in range(len(ACTIONS)) if item != target]
        rng.shuffle(old_actions)
        success_order = old_actions + [target]
        for event_index, action_index in enumerate(success_order):
            surface = surfaces[rng.randrange(len(surfaces))]
            text = (
                f"A prior {family} trial on {surface} succeeded using "
                f"procedure {ACTIONS[action_index]}."
            )
            event_ref = _ref(sample_index, event_index, f"{seed}:success")
            origin.append(event_ref, _ref(sample_index, event_index, "projection"))
            rows.append((event_ref, text, action_index == target, action_index))

        for offset in range(2):
            event_index = len(rows)
            action_index = rng.randrange(len(ACTIONS))
            surface = surfaces[rng.randrange(len(surfaces))]
            text = (
                f"A prior {family} trial on {surface} failed using "
                f"procedure {ACTIONS[action_index]}."
            )
            event_ref = _ref(sample_index, event_index, f"{seed}:failure")
            origin.append(event_ref, _ref(sample_index, event_index, "projection"))
            rows.append((event_ref, text, False, action_index))

        target_ref = rows[3][0]
        origin.designate_landmark(
            "current_regime",
            target_event_ref=target_ref,
            designation_event_ref=_ref(sample_index, 6, f"{seed}:designation"),
            projection_id=_ref(sample_index, 6, "projection"),
        )
        naive = FairNaiveTemporalView.from_index(origin)
        order = list(range(len(rows)))
        rng.shuffle(order)
        live_items: list[SituatedRecall] = []
        frozen_items: list[SituatedRecall] = []
        target_candidate = -1
        naive_work = 0
        for candidate_index, row_index in enumerate(order):
            event_ref, text, is_target, _ = rows[row_index]
            backend_score = 0.65 + 0.30 * rng.random()
            live_position = origin.position(event_ref)
            naive_position, inspected = naive.position(event_ref)
            if live_position != naive_position:
                raise RuntimeError("maintained and fair-naive coordinates diverged")
            naive_work += inspected
            frozen_position = origin.frozen_position(event_ref)
            common: dict[str, Any] = {
                "artifact_ref": event_ref,
                "text": text,
                "source_ref": _ref(sample_index, row_index, "source"),
                "context": (("family", family),),
                "visibility": "LEARNER_VISIBLE",
                "world_valid_at_query": None,
                "backend_score": backend_score,
                "backend_ref": f"synthetic-{sample_index}-{row_index}",
            }
            live_items.append(
                SituatedRecall(
                    **common,
                    acquired_ordinal=live_position.acquired_ordinal,
                    age=live_position.age,
                    landmark_relations=live_position.landmark_relations,
                    world_valid_from=live_position.world_valid_from,
                    world_valid_until=live_position.world_valid_until,
                )
            )
            frozen_items.append(
                SituatedRecall(
                    **common,
                    acquired_ordinal=frozen_position.acquired_ordinal,
                    age=frozen_position.age,
                    landmark_relations=frozen_position.landmark_relations,
                    world_valid_from=frozen_position.world_valid_from,
                    world_valid_until=frozen_position.world_valid_until,
                )
            )
            if is_target:
                target_candidate = candidate_index
        if target_candidate < 0:
            raise RuntimeError("target evidence is absent")
        query_surface = surfaces[rng.randrange(len(surfaces))]
        query_text = (
            f"Choose the reusable procedure for a fresh {family} instance "
            f"on {query_surface}."
        )
        samples.append(
            RawSample(
                query_text=query_text,
                candidate_texts=tuple(item.text for item in live_items),
                live=RecallBatch(tuple(live_items)),
                frozen=RecallBatch(tuple(frozen_items)),
                target_action=target,
                target_candidate=target_candidate,
                family_index=family_index,
                now=origin.now,
                maintained_inspected=len(rows),
                naive_inspected=naive_work,
            )
        )
    return tuple(samples)


def encode_texts_qwen(
    texts: tuple[str, ...],
    *,
    model_path: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    unique = tuple(dict.fromkeys(texts))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    encoded = encode_detached_segments(
        model,
        tokenizer,
        unique,
        batch_size=batch_size,
        storage_dtype=torch.bfloat16,
    )
    del model
    torch.cuda.empty_cache()
    return {text: encoded[index] for index, text in enumerate(unique)}


def tensorize(
    samples: tuple[RawSample, ...],
    embeddings: dict[str, torch.Tensor],
) -> TensorDataset:
    query = torch.stack([embeddings[item.query_text].float() for item in samples])
    candidates = torch.stack(
        [torch.stack([embeddings[text].float() for text in item.candidate_texts]) for item in samples]
    )
    live, mask = encode_situated_features(
        [item.live for item in samples],
        now=[item.now for item in samples],
        spec=SPEC,
    )
    frozen, frozen_mask = encode_situated_features(
        [item.frozen for item in samples],
        now=[item.now for item in samples],
        spec=SPEC,
    )
    if not torch.equal(mask, frozen_mask):
        raise RuntimeError("live and frozen candidate topology differs")
    target_mask = mask.clone()
    for index, item in enumerate(samples):
        target_mask[index, item.target_candidate] = False
    return TensorDataset(
        query=query,
        candidates=candidates,
        live=live,
        frozen=frozen,
        mask=mask,
        target_mask=target_mask,
        targets=torch.tensor([item.target_action for item in samples]),
        held_family=torch.tensor([item.family_index >= 6 for item in samples]),
        maintained_inspected=sum(item.maintained_inspected for item in samples) / len(samples),
        naive_inspected=sum(item.naive_inspected for item in samples) / len(samples),
    )


def _slice(dataset: TensorDataset, indices: torch.Tensor, device: torch.device):
    return tuple(
        value[indices].to(device)
        for value in (
            dataset.query,
            dataset.candidates,
            dataset.live,
            dataset.mask,
            dataset.targets,
        )
    )


def train_reader(
    reader: LearnedSituatedMemoryReader,
    dataset: TensorDataset,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    device = next(reader.parameters()).device
    optimizer = torch.optim.AdamW(reader.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    reader.train()
    for _ in range(epochs):
        order = torch.randperm(dataset.targets.shape[0], generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            query, candidates, temporal, mask, targets = _slice(dataset, indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = reader(query, candidates, temporal, mask)
            loss = F.cross_entropy(output.logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        losses.append(total / seen)
    return losses


@torch.inference_mode()
def accuracy(
    reader: LearnedSituatedMemoryReader,
    dataset: TensorDataset,
    *,
    temporal: torch.Tensor,
    candidates: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    selector: torch.Tensor | None = None,
    batch_size: int = 256,
) -> float:
    device = next(reader.parameters()).device
    indices = torch.arange(dataset.targets.shape[0])
    if selector is not None:
        indices = indices[selector]
    correct = 0
    reader.eval()
    candidate_source = dataset.candidates if candidates is None else candidates
    mask_source = dataset.mask if mask is None else mask
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        output = reader(
            dataset.query[selected].to(device),
            candidate_source[selected].to(device),
            temporal[selected].to(device),
            mask_source[selected].to(device),
        )
        predicted = output.logits.argmax(dim=-1).cpu()
        correct += int((predicted == dataset.targets[selected]).sum())
    return correct / len(indices)


def evaluate(reader: LearnedSituatedMemoryReader, dataset: TensorDataset) -> dict[str, float]:
    zeros_temporal = torch.zeros_like(dataset.live)
    zeros_content = torch.zeros_like(dataset.candidates)
    reset = LearnedSituatedMemoryReader(
        content_width=reader.content_width,
        temporal_width=reader.temporal_width,
        hidden_width=reader.hidden_width,
        action_count=reader.action_count,
    ).to(next(reader.parameters()).device)
    return {
        "full": accuracy(reader, dataset, temporal=dataset.live),
        "held_family": accuracy(
            reader,
            dataset,
            temporal=dataset.live,
            selector=dataset.held_family,
        ),
        "frozen_origin": accuracy(reader, dataset, temporal=dataset.frozen),
        "coordinates_removed": accuracy(reader, dataset, temporal=zeros_temporal),
        "content_removed": accuracy(
            reader,
            dataset,
            temporal=dataset.live,
            candidates=zeros_content,
        ),
        "target_removed": accuracy(
            reader,
            dataset,
            temporal=dataset.live,
            mask=dataset.target_mask,
        ),
        "competence_reset": accuracy(reset, dataset, temporal=dataset.live),
        # Fair-naive recomputed coordinates were asserted equal during construction.
        "fair_naive": accuracy(reader, dataset, temporal=dataset.live),
    }


def classify(metrics: dict[str, float], dataset: TensorDataset) -> str:
    comparison_arms = (
        "frozen_origin",
        "coordinates_removed",
        "content_removed",
        "target_removed",
        "competence_reset",
    )
    supported = (
        metrics["full"] >= 0.80
        and metrics["held_family"] >= 0.70
        and all(metrics["full"] - metrics[name] >= 0.15 for name in comparison_arms)
        and abs(metrics["full"] - metrics["fair_naive"]) <= 1e-6
        and dataset.naive_inspected >= 2.0 * dataset.maintained_inspected
    )
    return "CAUSAL_CORE_SUPPORTED_LIVE_COGNEE_PENDING" if supported else "NOT_SUPPORTED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--eval-count", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen experiment requires CUDA")
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    train_samples = build_samples(
        args.train_count,
        seed=SEED,
        family_indices=tuple(range(6)),
        evaluation_surfaces=False,
    )
    eval_samples = build_samples(
        args.eval_count,
        seed=SEED + 1,
        family_indices=tuple(range(8)),
        evaluation_surfaces=True,
    )
    all_texts = tuple(
        text
        for sample in train_samples + eval_samples
        for text in (sample.query_text, *sample.candidate_texts)
    )
    embeddings = encode_texts_qwen(
        all_texts,
        model_path=args.model,
        batch_size=args.embedding_batch_size,
    )
    train = tensorize(train_samples, embeddings)
    evaluation = tensorize(eval_samples, embeddings)
    content_width = train.query.shape[-1]
    device = torch.device("cuda")
    reader = LearnedSituatedMemoryReader(
        content_width=content_width,
        temporal_width=SPEC.width,
        hidden_width=256,
        action_count=len(ACTIONS),
    ).to(device)
    losses = train_reader(
        reader,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=SEED,
    )
    metrics = evaluate(reader, evaluation)
    torch.cuda.synchronize()
    result = {
        "identity": "angler.situated-memory-value.v1",
        "classification": classify(metrics, evaluation),
        "metrics": metrics,
        "counts": {
            "train": len(train_samples),
            "evaluation": len(eval_samples),
            "families": len(FAMILIES),
            "held_families": 2,
            "candidates_per_query": len(eval_samples[0].candidate_texts),
            "epochs": args.epochs,
        },
        "optimization": {
            "first_loss": losses[0],
            "final_loss": losses[-1],
        },
        "operational": {
            "maintained_mean_inspected": evaluation.maintained_inspected,
            "fair_naive_mean_inspected": evaluation.naive_inspected,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "seed": SEED,
        },
        "limits": [
            "Same-mechanism fresh-instance and held-family transfer only.",
            "Live self-hosted Cognee acceptance remains a separate required clause.",
            "No AGI, consciousness, unrestricted transfer, or frontier-model claim.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = output.with_suffix(".reader.pt")
    torch.save({"state_dict": reader.state_dict(), "content_width": content_width}, checkpoint)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
