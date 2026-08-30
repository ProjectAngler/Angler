"""Confound-corrected situated-memory causal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from angler.memory import FairNaiveTemporalView, MovingOriginIndex, RecallBatch, SituatedRecall
from angler.reasoning import LearnedSituatedMemoryReader, SituatedFeatureSpec, encode_situated_features

from experiments.runners.situated_memory_value_v1 import (
    ACTIONS,
    EVAL_SURFACES,
    FAMILIES,
    SEED,
    TRAIN_SURFACES,
    RawSample,
    TensorDataset,
    accuracy,
    classify,
    encode_texts_qwen,
    train_reader,
)


SPEC_V2 = SituatedFeatureSpec(("current_regime",), include_acquired_ordinal=False)


def _ref(sample: int, event: int, salt: str) -> str:
    return "sha256:" + hashlib.sha256(f"{salt}:{sample}:{event}".encode()).hexdigest()


def build_samples_v2(
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
        older = rng.choice([item for item in range(len(ACTIONS)) if item != target])
        failures = [(action, "failure") for action in range(len(ACTIONS))]
        rng.shuffle(failures)
        prefix_count = rng.randrange(0, 3)
        remaining = failures[prefix_count:]
        middle_count = rng.randrange(0, len(remaining) + 1)
        event_plan = (
            failures[:prefix_count]
            + [(older, "success")]
            + remaining[:middle_count]
            + [(target, "success")]
            + remaining[middle_count:]
        )

        origin = MovingOriginIndex()
        rows: list[tuple[str, str, bool]] = []
        for event_index, (action_index, outcome) in enumerate(event_plan):
            surface = surfaces[rng.randrange(len(surfaces))]
            text = (
                f"For a prior {family} instance on {surface}, procedure "
                f"{ACTIONS[action_index]} was observed. Result {outcome}; "
                f"procedure {ACTIONS[action_index]}; status {outcome}."
            )
            event_ref = _ref(sample_index, event_index, f"v2:{seed}:{outcome}")
            origin.append(event_ref, _ref(sample_index, event_index, "v2:projection"))
            rows.append((event_ref, text, outcome == "success" and action_index == target))

        target_ref = next(event_ref for event_ref, _, is_target in rows if is_target)
        origin.designate_landmark(
            "current_regime",
            target_event_ref=target_ref,
            designation_event_ref=_ref(sample_index, 6, f"v2:{seed}:designation"),
            projection_id=_ref(sample_index, 6, "v2:projection"),
        )
        naive = FairNaiveTemporalView.from_index(origin)
        order = list(range(len(rows)))
        rng.shuffle(order)
        live_items: list[SituatedRecall] = []
        frozen_items: list[SituatedRecall] = []
        target_candidate = -1
        naive_work = 0
        for candidate_index, row_index in enumerate(order):
            event_ref, text, is_target = rows[row_index]
            live_position = origin.position(event_ref)
            naive_position, inspected = naive.position(event_ref)
            if live_position != naive_position:
                raise RuntimeError("maintained and fair-naive coordinates diverged")
            naive_work += inspected
            frozen_position = origin.frozen_position(event_ref)
            common: dict[str, Any] = {
                "artifact_ref": event_ref,
                "text": text,
                "source_ref": _ref(sample_index, row_index, "v2:source"),
                "context": (("family", family),),
                "visibility": "LEARNER_VISIBLE",
                "world_valid_at_query": None,
                "backend_score": 0.65 + 0.30 * rng.random(),
                "backend_ref": f"synthetic-v2-{sample_index}-{row_index}",
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
        query_surface = surfaces[rng.randrange(len(surfaces))]
        samples.append(
            RawSample(
                query_text=(
                    f"Choose the reusable procedure for a fresh {family} instance "
                    f"on {query_surface}."
                ),
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


def tensorize_v2(
    samples: tuple[RawSample, ...],
    embeddings: dict[str, torch.Tensor],
) -> TensorDataset:
    query = torch.stack([embeddings[item.query_text].float() for item in samples])
    candidates = torch.stack(
        [torch.stack([embeddings[text].float() for text in item.candidate_texts]) for item in samples]
    )
    live, mask = encode_situated_features(
        [item.live for item in samples], now=[item.now for item in samples], spec=SPEC_V2
    )
    frozen, frozen_mask = encode_situated_features(
        [item.frozen for item in samples], now=[item.now for item in samples], spec=SPEC_V2
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


def _mismatched_indices(targets: torch.Tensor) -> torch.Tensor:
    selected = []
    for index, target in enumerate(targets.tolist()):
        candidate = (index + 1) % len(targets)
        while targets[candidate].item() == target:
            candidate = (candidate + 1) % len(targets)
        selected.append(candidate)
    return torch.tensor(selected)


def evaluate_v2(
    reader: LearnedSituatedMemoryReader,
    dataset: TensorDataset,
) -> dict[str, float]:
    zeros_temporal = torch.zeros_like(dataset.live)
    zeros_content = torch.zeros_like(dataset.candidates)
    mismatched = _mismatched_indices(dataset.targets)
    reset = LearnedSituatedMemoryReader(
        content_width=reader.content_width,
        temporal_width=reader.temporal_width,
        hidden_width=reader.hidden_width,
        action_count=reader.action_count,
    ).to(next(reader.parameters()).device)
    return {
        "full": accuracy(reader, dataset, temporal=dataset.live),
        "held_family": accuracy(
            reader, dataset, temporal=dataset.live, selector=dataset.held_family
        ),
        "frozen_origin": accuracy(reader, dataset, temporal=dataset.frozen),
        "coordinates_removed": accuracy(reader, dataset, temporal=zeros_temporal),
        "content_removed": accuracy(
            reader, dataset, temporal=dataset.live, candidates=zeros_content
        ),
        "target_removed": accuracy(
            reader,
            dataset,
            temporal=dataset.live[mismatched],
            candidates=dataset.candidates[mismatched],
            mask=dataset.mask[mismatched],
        ),
        "competence_reset": accuracy(reset, dataset, temporal=dataset.live),
        "fair_naive": accuracy(reader, dataset, temporal=dataset.live),
    }


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
    torch.manual_seed(SEED + 2)
    torch.cuda.manual_seed_all(SEED + 2)
    torch.cuda.reset_peak_memory_stats()
    training_samples = build_samples_v2(
        args.train_count,
        seed=SEED + 2,
        family_indices=tuple(range(6)),
        evaluation_surfaces=False,
    )
    evaluation_samples = build_samples_v2(
        args.eval_count,
        seed=SEED + 3,
        family_indices=tuple(range(8)),
        evaluation_surfaces=True,
    )
    texts = tuple(
        text
        for sample in training_samples + evaluation_samples
        for text in (sample.query_text, *sample.candidate_texts)
    )
    embeddings = encode_texts_qwen(
        texts, model_path=args.model, batch_size=args.embedding_batch_size
    )
    training = tensorize_v2(training_samples, embeddings)
    evaluation = tensorize_v2(evaluation_samples, embeddings)
    reader = LearnedSituatedMemoryReader(
        content_width=training.query.shape[-1],
        temporal_width=SPEC_V2.width,
        hidden_width=256,
        action_count=len(ACTIONS),
    ).cuda()
    losses = train_reader(
        reader,
        training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=SEED + 2,
    )
    metrics = evaluate_v2(reader, evaluation)
    torch.cuda.synchronize()
    result = {
        "identity": "angler.situated-memory-value.v2",
        "classification": classify(metrics, evaluation),
        "predecessor": {
            "identity": "angler.situated-memory-value.v1",
            "classification": "NOT_SUPPORTED",
            "result_sha256": "4df3c604bea1116e8398312435c2b52481942894d75e55d8cdafa551e4dbd71d",
        },
        "metrics": metrics,
        "counts": {
            "train": len(training_samples),
            "evaluation": len(evaluation_samples),
            "families": len(FAMILIES),
            "held_families": 2,
            "candidates_per_query": len(evaluation_samples[0].candidate_texts),
            "epochs": args.epochs,
        },
        "optimization": {"first_loss": losses[0], "final_loss": losses[-1]},
        "operational": {
            "maintained_mean_inspected": evaluation.maintained_inspected,
            "fair_naive_mean_inspected": evaluation.naive_inspected,
        },
        "runtime": {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "seed": SEED + 2,
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
    torch.save(
        {"state_dict": reader.state_dict(), "content_width": training.query.shape[-1]},
        output.with_suffix(".reader.pt"),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
