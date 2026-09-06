#!/usr/bin/env python3
"""Materialize disjoint Jenny 2.0 temporal or mixed initial curricula."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from angler.runtime.temporal_curriculum import (
    build_mixed_initial_curriculum,
    build_temporal_curriculum,
)


def _write(path: Path, rows) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument(
        "--profile",
        choices=("temporal-v1", "mixed-initial-v1"),
        default="temporal-v1",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    builder = (
        build_temporal_curriculum
        if args.profile == "temporal-v1"
        else build_mixed_initial_curriculum
    )
    train_seed = 2026090201 if args.profile == "temporal-v1" else 2026090204
    eval_seed = 2026090202 if args.profile == "temporal-v1" else 2026090205
    train = builder(split="train", count=args.train_count, seed=train_seed)
    evaluation = builder(split="eval", count=args.eval_count, seed=eval_seed)
    train_identities = {row.identity for row in train}
    eval_identities = {row.identity for row in evaluation}
    if len(train_identities) != len(train) or len(eval_identities) != len(evaluation):
        raise RuntimeError("curriculum contains a duplicate identity")
    if train_identities & eval_identities:
        raise RuntimeError("train and evaluation identities overlap")
    train_path, eval_path = output / "train.jsonl", output / "eval.jsonl"
    manifest = {
        "schema": (
            "jenny2.temporal-curriculum-manifest.v1"
            if args.profile == "temporal-v1"
            else "jenny2.mixed-initial-curriculum-manifest.v1"
        ),
        "profile": args.profile,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_sha256": _write(train_path, train),
        "eval_sha256": _write(eval_path, evaluation),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_skill_counts": dict(sorted(Counter(row.skill for row in train).items())),
        "eval_skill_counts": dict(
            sorted(Counter(row.skill for row in evaluation).items())
        ),
        "train_boundary_counts": dict(
            sorted(Counter(getattr(row, "boundary", "public_cortex") for row in train).items())
        ),
        "eval_boundary_counts": dict(
            sorted(
                Counter(
                    getattr(row, "boundary", "public_cortex")
                    for row in evaluation
                ).items()
            )
        ),
        "train_eval_identity_overlap": 0,
        "temporal_fraction": 1.0 if args.profile == "temporal-v1" else 0.25,
        "runtime_answer_lookup": False,
        "scripted_personality_or_feeling_labels": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
