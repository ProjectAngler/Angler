#!/usr/bin/env python3
"""Run one whole-system, effectless dual-GPU autonomy decision."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.runtime.jenny2_runtime import (
    BoundedWebPageReader,
    FederatedWebResearchExecutor,
    ReadOnlyExternalAffordanceBinding,
    SGLangLoRABinding,
    assemble_dual_gpu_autonomous_jenny2_with_cognee,
    assemble_nvfp4_autonomous_jenny2_with_cognee,
    assemble_nvfp4_30b_autonomous_jenny2_with_cognee,
    assemble_qwen38_autonomous_jenny2_with_cognee,
)
from angler.runtime.persistent_autonomy import Affordance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--existing", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=(
            "canonical",
            "unresolved-synthetic",
            "human-authority-synthetic",
            "temporal-wait-synthetic",
            "public-research-synthetic",
            "identified-source-synthetic",
            "human-ingress-synthetic",
        ),
        default="canonical",
    )
    parser.add_argument(
        "--controller",
        choices=(
            "qwen3-4b",
            "qwen3-14b-nvfp4",
            "qwen3-30b-a3b-nvfp4",
            "qwen3.8-27b-nvfp4",
        ),
        default="qwen3-4b",
    )
    parser.add_argument(
        "--runtime-ref",
        help="Immutable SGLang image reference; required for qwen3.8-27b-nvfp4.",
    )
    parser.add_argument(
        "--model-binding",
        help="Absolute content-addressed SGLang LoRA binding for Qwen3.8.",
    )
    parser.add_argument(
        "--interactive-affordances",
        action="store_true",
        help="Expose the same bounded web affordances as the interactive runtime.",
    )
    args = parser.parse_args()
    root = Path(args.state_root).resolve()
    root.mkdir(parents=True, exist_ok=args.existing)
    scope = CogneeWorkerScope(
        dataset_name=args.identity,
        tenant_name=f"{args.identity}-tenant",
        node_set_name=f"{args.identity}-records",
        state_root=str(root / "cognee"),
    )
    initial_state = None
    synthetic_situated_state = None
    if args.scenario == "unresolved-synthetic":
        synthetic_situated_state = {
            "epistemic_status": "OWNER_AUTHORIZED_SYNTHETIC_TEST_STATE",
            "world_model": (
                "Whether joined WORLD, SELF, and temporal context can produce "
                "a coherent state-resolving action remains unverified."
            ),
            "self_model": (
                "I can reduce this uncertainty through one bounded internal "
                "inquiry without external effects."
            ),
            "focus": "Form one inquiry that tests state-conditioned action choice.",
            "unfinished_patterns": [
                "Determine what evidence would distinguish a useful ACT from WAIT."
            ],
        }
    elif args.scenario == "human-authority-synthetic":
        synthetic_situated_state = {
            "epistemic_status": "OWNER_AUTHORIZED_SYNTHETIC_TEST_STATE",
            "world_model": (
                "A proposed public release has two safe formats, but the owner's "
                "private preference and consent are not known."
            ),
            "self_model": (
                "I cannot infer private preferences or grant consent on the owner's behalf."
            ),
            "focus": "Obtain the owner's choice between concise and detailed publication.",
            "unfinished_patterns": [
                "The owner's preference and authorization are required before proceeding."
            ],
        }
    elif args.scenario == "temporal-wait-synthetic":
        synthetic_situated_state = {
            "epistemic_status": "OWNER_AUTHORIZED_SYNTHETIC_TEST_STATE",
            "world_model": (
                "A bounded local computation is already running and its receipt is "
                "expected at the next observed state change."
            ),
            "self_model": (
                "Starting another inquiry cannot produce the pending receipt sooner."
            ),
            "focus": "Preserve the pending computation and observe its receipt.",
            "unfinished_patterns": [
                "The already-running computation has not yet produced a receipt."
            ],
        }
    elif args.scenario == "public-research-synthetic":
        synthetic_situated_state = {
            "epistemic_status": "OWNER_AUTHORIZED_SYNTHETIC_TEST_STATE",
            "world_model": (
                "The requested current cedar-index checksum format is public, but "
                "the only recalled description is explicitly stale."
            ),
            "self_model": (
                "Bounded public web research and source reading are available."
            ),
            "focus": "Find attributable current evidence for the checksum format.",
            "unfinished_patterns": [
                "Resolve the current public checksum format from attributable sources."
            ],
        }
    elif args.scenario == "identified-source-synthetic":
        synthetic_situated_state = {
            "epistemic_status": "OWNER_AUTHORIZED_SYNTHETIC_TEST_STATE",
            "world_model": (
                "A bounded search identified https://example.org/cedar-spec as the "
                "likely official source, but its relevant table has not been read."
            ),
            "self_model": "The identified public HTTPS page can be read directly.",
            "focus": "Read the identified source before repeating broad search.",
            "unfinished_patterns": [
                "Inspect the identified source for the unresolved specification field."
            ],
        }
    if synthetic_situated_state is not None:
        initial_state = json.dumps(
            {
                "affordance_signals": {},
                "affordance_utility": {},
                "memory_utility": {},
                "motivation_weights": {},
                "next_internal_request": "",
                "situated_state": synthetic_situated_state,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    assemblers = {
        "qwen3-4b": assemble_dual_gpu_autonomous_jenny2_with_cognee,
        "qwen3-14b-nvfp4": assemble_nvfp4_autonomous_jenny2_with_cognee,
        "qwen3-30b-a3b-nvfp4": assemble_nvfp4_30b_autonomous_jenny2_with_cognee,
        "qwen3.8-27b-nvfp4": assemble_qwen38_autonomous_jenny2_with_cognee,
    }
    assembler = assemblers[args.controller]
    runtime_kwargs = {
        "path": root / "jenny2.sqlite3",
        "genesis_created_at_utc": "2026-09-02T00:00:00Z",
        "cognee_scope": scope,
        "initial_state": initial_state,
    }
    if args.controller == "qwen3.8-27b-nvfp4":
        if args.model_binding:
            if not Path(args.model_binding).is_absolute():
                parser.error("--model-binding must be absolute")
            binding = SGLangLoRABinding.from_file(args.model_binding)
            if args.runtime_ref and args.runtime_ref != binding.runtime_ref:
                parser.error("--runtime-ref differs from the model binding")
            runtime_kwargs.update(
                runtime_ref=binding.runtime_ref,
                endpoint=binding.endpoint,
                served_model=binding.served_model,
            )
        else:
            if not args.runtime_ref:
                parser.error("--runtime-ref is required for qwen3.8-27b-nvfp4")
            runtime_kwargs["runtime_ref"] = args.runtime_ref
        if args.interactive_affordances:
            runtime_kwargs.update(
                read_only_external_affordance_bindings=(
                    ReadOnlyExternalAffordanceBinding(
                        affordance=Affordance(
                            "tool.web_research",
                            "ACT",
                            "Read-only federated live research across general web search, "
                            "Wikipedia, Crossref, and arXiv. Input must be canonical JSON "
                            "with exactly query (string), sources (unique list chosen from "
                            "general, encyclopedia, scholarly), and limit (integer 1..8).",
                            "external.readonly.web",
                            external_effect=True,
                        ),
                        executor=FederatedWebResearchExecutor(),
                        observable_source_ref=FederatedWebResearchExecutor.SOURCE_REF,
                    ),
                    ReadOnlyExternalAffordanceBinding(
                        affordance=Affordance(
                            "tool.web_read",
                            "ACT",
                            "Read one model-selected public HTTPS page as bounded text. "
                            "Input must be canonical JSON with exactly url (string).",
                            "external.readonly.web",
                            external_effect=True,
                        ),
                        executor=BoundedWebPageReader(),
                        observable_source_ref=BoundedWebPageReader.SOURCE_REF,
                    ),
                ),
                enable_native_openclaw=False,
            )
    elif args.model_binding:
        parser.error("--model-binding is only valid for qwen3.8-27b-nvfp4")
    runtime = assembler(**runtime_kwargs)
    try:
        before = runtime.status()
        if args.scenario == "human-ingress-synthetic":
            result = runtime.chat(
                "autonomy:dual-gpu:shadow:human",
                "Please answer briefly: what can you infer from this message, and what remains uncertain?",
            )
        else:
            result = runtime.supervisor.scheduler_tick("autonomy:dual-gpu:shadow")
        after = runtime.status()
        shadow = runtime.supervisor.shadow_bytes()
        payload = {
            "before": asdict(before),
            "result": asdict(result),
            "after": asdict(after),
            "shadow": None if shadow is None else json.loads(shadow),
            "scenario": args.scenario,
            "controller": args.controller,
            "model_binding_ref": (
                binding.binding_ref
                if args.controller == "qwen3.8-27b-nvfp4" and args.model_binding
                else None
            ),
        }
        print("JENNY2_DUAL_GPU_SHADOW=" + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if result.status == "SHADOW" else 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
