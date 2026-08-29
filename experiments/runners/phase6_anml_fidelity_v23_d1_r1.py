"""Publication-only recovery for the consumed V23-D1 ANML fidelity pilot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any

import torch

from experiments.runners import phase6_anml_fidelity_v23_d1 as source
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


PROTOCOL_ID = "phase6.public-anml-fidelity.v23-d1-r1"
SOURCE_PROTOCOL_ID = source.PROTOCOL_ID
SOURCE_RUNNER_SHA256 = "0DA7D4FFCBF7111ACB2E6F06CE5A33BF0ADFA1DB35E5C42748FB8DC2B0F99BE9"
SOURCE_CLAIM_SHA256 = "51DC9DB4C72DCBC076A78B8ED9392FAD6B348D52E350C01F3136B697C8052CAE"
SOURCE_FAILURE_SHA256 = "B1719D5580A40DA82520042EACC69DB8CE8D901A549D7DDF29EF7060EDAF7852"
SOURCE_CLAIM = Path("/opt/angler/results/phase6-anml-fidelity-v23-d1.claim.json")
SOURCE_RESULT = Path("/opt/angler/results/phase6-anml-fidelity-v23-d1.json")
SOURCE_FAILURE = Path("/opt/angler/results/phase6-anml-fidelity-v23-d1.failure.json")

# These aliases make the absence of scientific changes mechanically inspectable.
CONFIGS = source.CONFIGS
OUTER_UPDATES = source.OUTER_UPDATES
LIFETIME_UPDATES = source.LIFETIME_UPDATES
PROBE_MILESTONES = source.PROBE_MILESTONES
classify_result = source.classify_result


def json_ready(value: object) -> object:
    """Return strict JSON data while converting mapping keys without collisions."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            text = str(key)
            if text in converted:
                raise ValueError(f"V23-D1-R1 JSON key stringification collision: {text!r}")
            converted[text] = json_ready(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("V23-D1-R1 output contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V23-D1-R1 value is not JSON-safe: {type(value).__name__}")


def _accept_source_terminal_value(_value: object) -> None:
    """Defer validation until integer milestone keys have been stringified."""


def recover_source_run(device: torch.device | str = "cuda") -> dict[str, object]:
    """Run frozen V23-D1 once and repair only its terminal representation."""

    if source.v22 is not v22:
        raise RuntimeError("V23-D1-R1 source and recovery do not share the V22 module")
    original_validator = v22._validate_json_value
    try:
        v22._validate_json_value = _accept_source_terminal_value
        raw = source.run(device)
    finally:
        v22._validate_json_value = original_validator
    if v22._validate_json_value is not original_validator:
        raise RuntimeError("V23-D1-R1 did not restore the strict V22 validator")
    recovered = json_ready(raw)
    if not isinstance(recovered, dict):
        raise TypeError("V23-D1-R1 recovered source output is not a mapping")
    if recovered.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError("V23-D1-R1 source protocol identity changed")
    recovered["artifact_schema"] = "angler.anml-fidelity-v23-d1-r1.result.v1"
    recovered["protocol_id"] = PROTOCOL_ID
    recovered["recovery"] = {
        "kind": "terminal_json_key_publication_only",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_outcome": "HARNESS_ERROR_PRESERVED",
        "source_claim_sha256": SOURCE_CLAIM_SHA256,
        "source_failure_sha256": SOURCE_FAILURE_SHA256,
        "scientific_configuration_changed": False,
        "metrics_observed_before_recovery": False,
    }
    original_validator(recovered)
    return recovered


def synthetic_preflight(device: torch.device | str = "cuda") -> dict[str, object]:
    source_result = source.synthetic_preflight(device)
    regression = json_ready({"evaluation": {0: {"loss": 1.0}, 512: {"loss": 0.5}}})
    if regression != {"evaluation": {"0": {"loss": 1.0}, "512": {"loss": 0.5}}}:
        raise RuntimeError("V23-D1-R1 milestone-key regression failed")
    return {
        "passed": bool(source_result["passed"]),
        "source_preflight": json_ready(source_result),
        "milestone_key_regression": regression,
        "scientific_configuration_inherited": bool(
            CONFIGS is source.CONFIGS
            and classify_result is source.classify_result
            and OUTER_UPDATES == 48
            and LIFETIME_UPDATES == 512
        ),
    }


def run(device: torch.device | str = "cuda") -> dict[str, object]:
    return recover_source_run(device)

