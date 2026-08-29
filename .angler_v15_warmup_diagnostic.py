from __future__ import annotations

import json
import os
import time

import torch

from experiments.runners import phase6_cross_variation_plasticity as v15


def timed(label: str, function):
    started = time.perf_counter()
    value = function()
    return value, {label: time.perf_counter() - started}


def main() -> None:
    torch.set_num_threads(1)
    replicate = int(os.environ.get("ANGLER_V15_DIAGNOSTIC_REPLICATE", "0"))
    timings: dict[str, float] = {}
    pair, current = timed(
        "build_pair",
        lambda: v15.build_cross_variation_pair(replicate),
    )
    timings.update(current)
    arm = pair[0]
    batches, current = timed(
        "build_batch",
        lambda: v15.build_training_batches(replicate, updates=1),
    )
    timings.update(current)
    batch = batches[0]
    evidence, current = timed(
        "collect_evidence",
        lambda: v15.collect_cross_variation_evidence(
            arm.controller,
            batch.streams,
            arm.cell_optimizer_state,
        ),
    )
    timings.update(current)
    allocations, current = timed(
        "combined_allocations",
        lambda: v15._combined_allocations(
            evidence,
            batch,
            arm.router,
            learned_plasticity=False,
        ),
    )
    timings.update(current)
    meta, current = timed(
        "meta_gradients",
        lambda: v15.cross_variation_meta_gradients(
            arm.controller,
            arm.router,
            batch,
            arm.cell_optimizer_state,
            evidence,
        ),
    )
    timings.update(current)
    gradient_records = []
    for (name, _), gradient in zip(
        arm.router.named_parameters(),
        meta.gradients,
        strict=True,
    ):
        gradient_records.append(
            {
                "name": name,
                "reported_fp32_norm": dict(meta.parameter_gradient_norms)[name],
                "fp64_norm": float(gradient.to(torch.float64).norm().item()),
                "max_abs": float(gradient.abs().max().item()),
                "nonzero_values": int(torch.count_nonzero(gradient).item()),
                "values": gradient.numel(),
                "finite": bool(torch.isfinite(gradient).all().item()),
            }
        )
    print(
        json.dumps(
            {
                "device": "cpu",
                "replicate": replicate,
                "timings": timings,
                "first_allocation_exact_uniform": bool(
                    torch.equal(
                        allocations[0],
                        torch.full_like(allocations[0], 0.25),
                    )
                ),
                "meta_objective": meta.objective,
                "gradients": gradient_records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
