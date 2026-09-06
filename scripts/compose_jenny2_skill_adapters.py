#!/usr/bin/env python3
"""Compile exact LoRA skill deltas into one uniformly served Jenny adapter."""

from __future__ import annotations

import argparse
import json

from angler.runtime.skill_adapter_composition import (
    SkillAdapterComponent,
    compose_skill_adapters,
)


_COMPONENT_FIELDS = {
    "name",
    "path",
    "adapter_model_sha256",
    "adapter_config_sha256",
    "weight",
}


def _component(value: str) -> SkillAdapterComponent:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("component must be a JSON object") from exc
    if type(payload) is not dict or set(payload) != _COMPONENT_FIELDS:
        raise argparse.ArgumentTypeError(
            "component fields must be exactly " + ", ".join(sorted(_COMPONENT_FIELDS))
        )
    if any(type(payload[field]) is not str for field in _COMPONENT_FIELDS):
        raise argparse.ArgumentTypeError("every component field must be a string")
    return SkillAdapterComponent(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and exactly rank-concatenate two or more compatible LoRA "
            "components. The output is inert and no runtime is changed."
        )
    )
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        type=_component,
        help=(
            "repeat JSON: {name,path,adapter_model_sha256," 
            "adapter_config_sha256,weight}; weight is a decimal string"
        ),
    )
    parser.add_argument("--training-result-source", required=True)
    parser.add_argument("--expected-training-result-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    composition = compose_skill_adapters(
        args.component,
        training_result_source=args.training_result_source,
        expected_training_result_sha256=args.expected_training_result_sha256,
        output_root=args.output,
    )
    print(
        "JENNY2_SKILL_ADAPTER_COMPOSITION="
        + json.dumps(
            {
                "adapter_config_sha256": composition.adapter_config_sha256,
                "adapter_model_sha256": composition.adapter_model_sha256,
                "adapter_path": str(composition.adapter_path),
                "manifest_path": str(composition.manifest_path),
                "manifest_ref": composition.manifest_ref,
                "manifest_sha256": composition.manifest_sha256,
                "output_root": str(composition.output_root),
                "runtime_changed": False,
                "training_result_path": str(composition.training_result_path),
                "training_result_sha256": composition.training_result_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
