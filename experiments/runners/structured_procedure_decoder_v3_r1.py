"""Evaluation-preserving recovery for the V3 inference-tensor harness defect."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import experiments.runners.structured_procedure_decoder_v3 as original


IDENTITY = "angler.structured-procedure-decoder.v3-r1"
ORIGINAL_RUNNER_SHA256 = "28880e1e44c2bf3d4d3ae298678702755e085e2370fbf5c7a97ea03b393dbe41"
FAILED_DECODER_SHA256 = "11900a023506605fbddce437d374039f98b752ae4586479eaf1963e183a62bc0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _argument_path(name: str) -> Path:
    try:
        index = sys.argv.index(name)
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing recovery argument {name}") from error


def main() -> None:
    if _sha256(Path(original.__file__)) != ORIGINAL_RUNNER_SHA256:
        raise RuntimeError("consumed V3 runner identity changed")
    original.IDENTITY = IDENTITY
    original.main()
    output_path = _argument_path("--output")
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("identity") != IDENTITY:
        raise RuntimeError("recovery output identity mismatch")
    report["recovery"] = {
        "classification": "TENSOR_MODE_ONLY",
        "failed_decoder_sha256": FAILED_DECODER_SHA256,
        "original_runner_sha256": ORIGINAL_RUNNER_SHA256,
        "recovery_runner_sha256": _sha256(Path(__file__)),
    }
    report["runner_sha256"] = report["recovery"]["recovery_runner_sha256"]
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
