"""Isolated parallel episode execution for Angler training/evaluation workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import torch

from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


PROTOCOL_ID = "phase6.parallel-episode-executor.v1-r1"
V22_CHECKPOINT = Path("/opt/angler/results/phase6-anml-selective-plasticity-v22.pt")
V22_CHECKPOINT_SHA256 = "37DACF2F27092EED51473757C3C6EC6631D7DA407A480F4C3E58EEA6B58F54B4"
PANELS = (0, 1, 2, 3)
BENCHMARK_STEPS = 64
WORKER_TIMEOUT_SECONDS = 20.0 * 60.0
AGGREGATE_CUDA_CEILING_BYTES = 2 * 1024**3
ARM_NAMES = v22.PRIMARY_LIFETIME_ARMS


class WorkerExecutionError(RuntimeError):
    pass


class SemanticMismatchError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_panel_steps(panel: int, steps: int) -> None:
    if type(panel) is not int or panel not in PANELS:
        raise ValueError("parallel episode panel must be one of 0..3")
    if type(steps) is not int or not 1 <= steps <= v22.LIFETIME_UPDATES:
        raise ValueError("parallel episode step count is outside the lifetime")


def _new_accumulators() -> dict[str, dict[str, object]]:
    return {
        name: {
            "loss_sum": 0.0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_max": 0.0,
            "count": 0,
        }
        for name in ARM_NAMES
    }


def run_v22_panel_segment(
    panel: int,
    steps: int,
    *,
    checkpoint: str | Path = V22_CHECKPOINT,
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    """Run one panel independently while preserving its sequential update order."""

    validate_panel_steps(panel, steps)
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("parallel episode worker requires CUDA")
    checkpoint_path = Path(checkpoint)
    if sha256_file(checkpoint_path) != V22_CHECKPOINT_SHA256:
        raise RuntimeError("parallel episode V22 checkpoint identity changed")
    v22.configure_anml_numerics(selected)
    system = v22.load_anml_checkpoint(checkpoint_path, device=selected)
    if system.completed_meta_updates != v22.OUTER_UPDATES:
        raise RuntimeError("parallel episode worker requires completed V22 meta-fit")
    controller_before = v22._controller_digest(system.controller)
    gates_before = {name: v22._gate_digest(system.arm(name).gate) for name in v22.LEARNED_ARMS}
    states = {name: v22.fresh_fast_state(system.fast_initial_weight) for name in ARM_NAMES}
    accumulators = _new_accumulators()
    index = 0 if selected.index is None else selected.index
    torch.cuda.reset_peak_memory_stats(index)
    started = time.monotonic()
    for step in range(steps):
        record = v22._lifetime_record(panel, step)
        stream = v22._make_stream(record)
        bundle = v22.capture_feature_bundle(system.controller, system.fast_initial_weight, stream)
        for name in ARM_NAMES:
            module, lesion, permutation = v22._arm_gate_configuration(system, name, for_probe=False)
            next_state, diagnostic = v22._online_fast_step(
                system,
                states[name],
                bundle,
                module,
                lesion=lesion,
                permutation=permutation,
            )
            states[name] = next_state
            accumulator = accumulators[name]
            loss = float(diagnostic["loss"])
            gradient_norm = float(diagnostic["gradient_norm"])
            accumulator["loss_sum"] = float(accumulator["loss_sum"]) + loss
            accumulator["gradient_norm_sum"] = float(accumulator["gradient_norm_sum"]) + gradient_norm
            accumulator["gradient_norm_max"] = max(float(accumulator["gradient_norm_max"]), gradient_norm)
            accumulator["count"] = int(accumulator["count"]) + 1
        del bundle, stream
    maximum = int(torch.cuda.max_memory_allocated(index))
    semantic = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": v22.PROTOCOL_ID,
        "panel": panel,
        "updates": steps,
        "arms": {
            name: {
                **accumulators[name],
                "terminal_fast_step": states[name].optimizer_state[0].step,
                "terminal_fast_state_digest": v22.fast_state_digest(states[name]),
            }
            for name in ARM_NAMES
        },
        "controller_digest": controller_before,
        "learned_gate_digests": gates_before,
        "checkpoint_sha256": V22_CHECKPOINT_SHA256,
        "hook_cleanup": v22.active_gate_hook_count(system.controller) == 0,
    }
    if (
        v22._controller_digest(system.controller) != controller_before
        or any(v22._gate_digest(system.arm(name).gate) != gates_before[name] for name in v22.LEARNED_ARMS)
        or not semantic["hook_cleanup"]
        or any(record["terminal_fast_step"] != steps for record in semantic["arms"].values())
        or maximum > AGGREGATE_CUDA_CEILING_BYTES
    ):
        raise RuntimeError("parallel episode worker mechanical validity failed")
    v22._validate_json_value(semantic)
    return {
        "artifact_schema": "angler.parallel-episode-worker.v1",
        "semantic": semantic,
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "maximum_allocated_bytes": maximum,
            "device": str(selected),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }


def worker_command(panel: int, steps: int, output: str | Path) -> tuple[str, ...]:
    validate_panel_steps(panel, steps)
    return (
        sys.executable,
        "-B",
        "-m",
        "experiments.runners.parallel_episode_executor",
        "worker",
        "--panel",
        str(panel),
        "--steps",
        str(steps),
        "--output",
        str(Path(output)),
    )


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return environment


def launch_panel_workers(
    output_paths: Mapping[int, str | Path],
    *,
    steps: int = BENCHMARK_STEPS,
    parallel: bool,
    timeout_seconds: float = WORKER_TIMEOUT_SECONDS,
) -> dict[str, object]:
    if set(output_paths) != set(PANELS):
        raise ValueError("parallel episode outputs must cover panels 0..3")
    paths = {panel: Path(output_paths[panel]) for panel in PANELS}
    occupied = tuple(str(path) for path in paths.values() if path.exists())
    if occupied:
        raise RuntimeError(f"parallel episode worker output is occupied: {occupied}")
    started = time.monotonic()
    records: dict[int, dict[str, object]] = {}
    if parallel:
        processes = {
            panel: subprocess.Popen(
                worker_command(panel, steps, paths[panel]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_worker_environment(),
            )
            for panel in PANELS
        }
        try:
            for panel in PANELS:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0.0:
                    raise subprocess.TimeoutExpired(processes[panel].args, timeout_seconds)
                stdout, stderr = processes[panel].communicate(timeout=remaining)
                records[panel] = {"returncode": processes[panel].returncode, "stdout": stdout, "stderr": stderr}
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.kill()
                process.wait()
            raise
    else:
        for panel in PANELS:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(worker_command(panel, steps, paths[panel]), timeout_seconds)
            completed = subprocess.run(
                worker_command(panel, steps, paths[panel]),
                capture_output=True,
                text=True,
                env=_worker_environment(),
                timeout=remaining,
                check=False,
            )
            records[panel] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    failures = {panel: record for panel, record in records.items() if record["returncode"] != 0 or not paths[panel].is_file()}
    if failures:
        raise WorkerExecutionError(f"parallel episode worker failed: {failures}")
    return {
        "mode": "parallel" if parallel else "sequential",
        "elapsed_seconds": time.monotonic() - started,
        "worker_outputs": {str(panel): str(paths[panel]) for panel in PANELS},
    }


def load_worker_receipts(paths: Mapping[int, str | Path]) -> tuple[dict[str, object], ...]:
    if set(paths) != set(PANELS):
        raise ValueError("parallel episode receipts must cover panels 0..3")
    receipts = []
    for panel in PANELS:
        value = json.loads(Path(paths[panel]).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("artifact_schema") != "angler.parallel-episode-worker.v1":
            raise RuntimeError("parallel episode worker receipt is invalid")
        semantic = value.get("semantic")
        if not isinstance(semantic, dict) or semantic.get("panel") != panel:
            raise RuntimeError("parallel episode worker panel identity changed")
        receipts.append(value)
    return tuple(receipts)


def compare_semantics(
    sequential: Sequence[Mapping[str, object]],
    parallel: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(sequential) != len(PANELS) or len(parallel) != len(PANELS):
        raise ValueError("parallel episode comparison requires four ordered receipts")
    matches = []
    for panel, (left, right) in enumerate(zip(sequential, parallel, strict=True)):
        equal = left.get("semantic") == right.get("semantic")
        matches.append({"panel": panel, "exact_semantic_match": equal})
    if not all(record["exact_semantic_match"] for record in matches):
        raise SemanticMismatchError("parallel episode semantics changed under concurrency")
    return {"exact_match": True, "panels": tuple(matches)}


def fallback_eligible(error: BaseException) -> bool:
    return isinstance(error, (WorkerExecutionError, subprocess.TimeoutExpired, OSError)) and not isinstance(error, SemanticMismatchError)


def _worker_main(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("parallel episode worker output is occupied")
    result = run_v22_panel_segment(args.panel, args.steps)
    v22.atomic_write_json(output, result)
    print(json.dumps({"status": "complete", "panel": args.panel, "output": str(output)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--panel", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        _worker_main(args)


if __name__ == "__main__":
    main()
