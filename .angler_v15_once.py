from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from experiments.runners.phase6_cross_variation_plasticity import (
    fit_cross_variation_pilot,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v15-plasticity.json"
)
CHECKPOINT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v15-plasticity.pt"
)
TEMP_PATH = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
EXPECTED_HASHES = {
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
        "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
    ),
    "experiments/evaluators/software_pipeline_reconstruction_suite.py": (
        "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
    ),
    "tests/unit/experiments/test_software_pipeline_reconstruction_suite.py": (
        "CFB02F8D66CFFC9E326705969E6B3309025FB2BEE4A6D066A0D4780EB86586D3"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "tests/unit/experiments/test_phase6_software_pipeline_reconstruction.py": (
        "2E6D844D24DB0A9326D84A19AEC56ED5BF6288B94C67AD5926AC05933FB6DF32"
    ),
    "experiments/runners/phase6_counterfactual_plasticity_router.py": (
        "1AA64AAC3716F5C2C8333EE46852F839D19FC80AD39B1F5ED041E1738210C068"
    ),
    "tests/unit/experiments/test_phase6_counterfactual_plasticity_router.py": (
        "5B8AA19F536EF1788D25CF5BF54D982BA1F1BA033176E09126DB7CCB8D23BFFF"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    observed = {
        relative: file_sha256(ROOT / relative) for relative in EXPECTED_HASHES
    }
    if observed != EXPECTED_HASHES:
        raise RuntimeError("V15 launch hash preflight changed")
    existing = tuple(
        str(path) for path in (REPORT_PATH, CHECKPOINT_PATH, TEMP_PATH) if path.exists()
    )
    if existing:
        raise RuntimeError(f"V15 launch targets are not fresh: {existing!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V15 launch requires exactly one visible CUDA device")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_wall = time.time()
    started_monotonic = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "FIT_INVOKED",
                "protocol": "phase6.public-anonymous-cross-variation-plasticity.paired.v15",
                "runner_sha256": observed[
                    "experiments/runners/phase6_cross_variation_plasticity.py"
                ],
                "test_sha256": observed[
                    "tests/unit/experiments/test_phase6_cross_variation_plasticity.py"
                ],
                "device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report = fit_cross_variation_pilot(
        device="cuda:0",
        checkpoint_path=CHECKPOINT_PATH,
    )
    torch.cuda.synchronize()
    report["execution"] = {
        "started_unix_seconds": started_wall,
        "completed_unix_seconds": time.time(),
        "harness_elapsed_seconds": time.perf_counter() - started_monotonic,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "launch_hashes": observed,
    }
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with TEMP_PATH.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(TEMP_PATH, REPORT_PATH)
    print(
        json.dumps(
            {
                "event": "FIT_COMPLETE",
                "status": report["status"],
                "support_checks": report["support_checks"],
                "elapsed_seconds": report["elapsed_seconds"],
                "harness_elapsed_seconds": report["execution"][
                    "harness_elapsed_seconds"
                ],
                "peak_allocated_bytes": report["execution"][
                    "peak_allocated_bytes"
                ],
                "peak_reserved_bytes": report["execution"][
                    "peak_reserved_bytes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
