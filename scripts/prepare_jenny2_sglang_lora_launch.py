#!/usr/bin/env python3
"""Validate one finished Jenny adapter and emit, but never execute, its launch plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from angler.runtime.jenny2_runtime import SGLangLoRABinding


def _write_new_binding(path: Path, binding: SGLangLoRABinding) -> None:
    if not path.is_absolute():
        raise ValueError("binding output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(binding.to_payload(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("binding write did not advance")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a content-addressed Jenny/SGLang LoRA binding and print the "
            "exact inert docker argv. This command never invokes Docker."
        )
    )
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--container-name", default="jenny-qwen38-lora")
    parser.add_argument(
        "--cache-path", default="/opt/angler/runtime-cache/sglang-qwen38"
    )
    parser.add_argument(
        "--selector-qualification-ref",
        help=(
            "fresh qualification evidence for this exact adapter; omit to keep "
            "learned autonomous selection unqualified"
        ),
    )
    parser.add_argument("--expected-training-result-sha256", required=True)
    parser.add_argument("--expected-adapter-model-sha256", required=True)
    parser.add_argument("--expected-profile", default="mixed-initial-v1")
    args = parser.parse_args()

    for label, value in (
        ("expected training-result SHA-256", args.expected_training_result_sha256),
        ("expected adapter-model SHA-256", args.expected_adapter_model_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} is malformed")

    binding = SGLangLoRABinding.from_training_run(
        args.training_run,
        selector_qualification_ref=args.selector_qualification_ref,
    )
    if binding.training_result_sha256 != args.expected_training_result_sha256:
        raise ValueError("finished training-result hash differs from the expected identity")
    if binding.adapter_model_sha256 != args.expected_adapter_model_sha256:
        raise ValueError("finished adapter-model hash differs from the expected identity")
    if binding.training_profile != args.expected_profile:
        raise ValueError("finished training profile differs from the expected profile")
    output = Path(args.output).resolve()
    _write_new_binding(output, binding)
    # Reloading validates the just-written bytes and every referenced local
    # artifact before any launch command is disclosed.
    bound = SGLangLoRABinding.from_file(output)
    launch_plan = {
        "schema": "jenny2.sglang-lora-launch-plan.v1",
        "binding_path": str(output),
        "binding_ref": bound.binding_ref,
        "served_model": bound.served_model,
        "runtime_environment": {
            "JENNY2_MODEL_BINDING": str(output),
            "JENNY2_MODEL_ENDPOINT": bound.endpoint,
            "JENNY2_SERVED_MODEL": bound.served_model,
        },
        "docker_argv": list(
            bound.docker_argv(
                container_name=args.container_name,
                cache_path=Path(args.cache_path).resolve(),
            )
        ),
        "launch_executed": False,
        "readiness_claim": False,
    }
    print("JENNY2_LORA_LAUNCH_PLAN=" + json.dumps(launch_plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
