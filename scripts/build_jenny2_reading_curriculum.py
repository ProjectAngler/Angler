#!/usr/bin/env python3
"""Materialize disjoint synthetic Jenny 2.0 reading curricula."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from angler.runtime.reading_curriculum import (
    READING_CURRICULUM_SCHEMA,
    build_reading_curriculum,
)


MANIFEST_SCHEMA = "jenny2.reading-capability-curriculum-manifest.v1"
TRAIN_SEED = 2026090301
EVAL_SEED = 2026090302


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _write(path: Path, rows: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(
                row.to_mapping(),  # type: ignore[attr-defined]
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = build_reading_curriculum(
        split="train", count=args.train_count, seed=TRAIN_SEED
    )
    evaluation = build_reading_curriculum(
        split="eval", count=args.eval_count, seed=EVAL_SEED
    )
    train_identities = {row.identity for row in train}
    eval_identities = {row.identity for row in evaluation}
    if len(train_identities) != len(train) or len(eval_identities) != len(evaluation):
        raise RuntimeError("reading curriculum contains a duplicate identity")
    if train_identities & eval_identities:
        raise RuntimeError("reading train and evaluation identities overlap")
    train_message_refs = {
        _identity([message for message in row.messages[1:]]) for row in train
    }
    eval_message_refs = {
        _identity([message for message in row.messages[1:]]) for row in evaluation
    }
    if train_message_refs & eval_message_refs:
        raise RuntimeError("reading train and evaluation lesson content overlaps")

    train_path = output / "train.jsonl"
    eval_path = output / "eval.jsonl"
    train_sha = _write(train_path, train)
    eval_sha = _write(eval_path, evaluation)
    manifest_core = {
        "schema": MANIFEST_SCHEMA,
        "curriculum_schema": READING_CURRICULUM_SCHEMA,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "train_skill_counts": dict(
            sorted(Counter(row.skill for row in train).items())
        ),
        "eval_skill_counts": dict(
            sorted(Counter(row.skill for row in evaluation).items())
        ),
        "train_boundary_counts": dict(
            sorted(Counter(row.boundary for row in train).items())
        ),
        "eval_boundary_counts": dict(
            sorted(Counter(row.boundary for row in evaluation).items())
        ),
        "train_eval_identity_overlap": 0,
        "train_eval_content_overlap": 0,
        "synthetic_text_only": True,
        "live_library_opened": False,
        "runtime_answer_lookup": False,
        "raw_library_memorization": False,
        "scripted_personality_feelings_or_consciousness": False,
    }
    manifest = {
        **manifest_core,
        "dataset_identity": _identity(manifest_core),
        "train_identity": f"sha256:{train_sha}",
        "eval_identity": f"sha256:{eval_sha}",
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

