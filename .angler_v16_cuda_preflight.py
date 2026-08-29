from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.runners import phase6_cross_variation_plasticity_v16 as v16


OUTPUT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-preflight-cuda.json"
)
TEMP = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
EXPECTED_PLAN_DIGEST = (
    "sha256:6d357f781843816287ca70017fe5a0fa76fe500293284bd421ade08e8ca20731"
)
EXPECTED = {
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
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity_v16.py": (
        "95A01F4C65EDF962422E6E379B05A2C1D98241F78FA6BC9F57D23A3903585637"
    ),
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
        "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    observed = {relative: file_sha256(ROOT / relative) for relative in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError("V16 CUDA preflight hashes changed")
    plan_digest = v16.cross_variation_plan_digest()
    if plan_digest != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V16 CUDA preflight plan digest changed")
    if OUTPUT.exists() or TEMP.exists():
        raise RuntimeError("V16 CUDA preflight target is not fresh")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V16 CUDA preflight requires exactly one visible device")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "V16_CUDA_PREFLIGHT_INVOKED",
                "device": torch.cuda.get_device_name(0),
                "runner_sha256": observed[
                    "experiments/runners/phase6_cross_variation_plasticity_v16.py"
                ],
                "test_sha256": observed[
                    "tests/unit/experiments/test_phase6_cross_variation_plasticity_v16.py"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report = v16.structural_preflight("cuda:0")
    torch.cuda.synchronize()
    envelope = {
        "protocol_id": v16._PROTOCOL_ID,
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "plan_digest": plan_digest,
        "launch_hashes": observed,
        "preflight": report,
        "semantic_fit_performed": False,
        "evaluation_or_classification_performed": False,
    }
    encoded = json.dumps(
        envelope,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TEMP.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(TEMP, OUTPUT)
    print(
        json.dumps(
            {
                "event": "V16_CUDA_PREFLIGHT_COMPLETE",
                "status": "PASS",
                "arms_checked": report["arms_checked"],
                "optimizer_steps": report["optimizer_steps"],
                "all_routes_exact_uniform": report["all_routes_exact_uniform"],
                "all_upstream_meta_gradients_exact_zero": report[
                    "all_upstream_meta_gradients_exact_zero"
                ],
                "elapsed_seconds": envelope["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
