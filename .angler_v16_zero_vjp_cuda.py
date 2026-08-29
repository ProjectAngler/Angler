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

from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16


OUTPUT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-zero-vjp-cuda.json"
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


def zero_vjp(function, *, device: torch.device) -> dict[str, object]:
    parameter = torch.linspace(-0.3, 0.3, 19, dtype=torch.float32, device=device)
    direction = torch.zeros_like(parameter, requires_grad=True)
    zero = torch.zeros_like(parameter)
    slot = v15.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone())
    updated, state = function(
        (parameter,), (direction,), (slot,), (3.0e-4,)
    )
    probe = torch.linspace(0.5, 1.5, 19, dtype=torch.float32, device=device)
    (gradient,) = torch.autograd.grad((updated[0] * probe).sum(), (direction,))
    finite = torch.isfinite(gradient)
    return {
        "vjp_values": gradient.numel(),
        "vjp_finite_values": int(finite.sum().item()),
        "vjp_nonfinite_values": int((~finite).sum().item()),
        "vjp_all_finite": bool(finite.all().item()),
        "forward_equals_input": bool(torch.equal(updated[0], parameter)),
        "state_step": state[0].step,
        "state_exp_avg_nonzero": int(torch.count_nonzero(state[0].exp_avg).item()),
        "state_exp_avg_sq_nonzero": int(
            torch.count_nonzero(state[0].exp_avg_sq).item()
        ),
    }


def forward_parity(*, device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(2_026_083_601)
    initial = torch.randn(17, generator=generator, dtype=torch.float32, device=device)
    gradients = (
        torch.randn(17, generator=generator, dtype=torch.float32, device=device),
        torch.zeros(17, dtype=torch.float32, device=device),
        None,
        torch.randn(17, generator=generator, dtype=torch.float32, device=device),
    )
    zero = torch.zeros_like(initial)
    old_parameters = (initial.clone(),)
    new_parameters = (initial.clone(),)
    old_state = (
        v15.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),
    )
    new_state = (
        v15.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),
    )
    records = []
    for index, gradient in enumerate(gradients):
        old_parameters, old_state = v15.functional_adamw_step(
            old_parameters, (gradient,), old_state, (3.0e-4,)
        )
        new_parameters, new_state = v16.functional_adamw_step(
            new_parameters, (gradient,), new_state, (3.0e-4,)
        )
        records.append(
            {
                "functional_step": index,
                "gradient_is_none": gradient is None,
                "gradient_is_exact_zero": bool(
                    gradient is not None
                    and torch.count_nonzero(gradient).item() == 0
                ),
                "parameter_bit_equal": bool(
                    torch.equal(old_parameters[0], new_parameters[0])
                ),
                "exp_avg_bit_equal": bool(
                    torch.equal(old_state[0].exp_avg, new_state[0].exp_avg)
                ),
                "exp_avg_sq_bit_equal": bool(
                    torch.equal(old_state[0].exp_avg_sq, new_state[0].exp_avg_sq)
                ),
                "step_equal": old_state[0].step == new_state[0].step,
            }
        )
    return {
        "functional_steps": len(records),
        "records": records,
        "all_parameter_and_state_values_bit_equal": all(
            all(
                bool(record[key])
                for key in (
                    "parameter_bit_equal",
                    "exp_avg_bit_equal",
                    "exp_avg_sq_bit_equal",
                    "step_equal",
                )
            )
            for record in records
        ),
    }


def main() -> None:
    observed = {relative: file_sha256(ROOT / relative) for relative in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError("V16 zero-VJP probe hashes changed")
    if v16.cross_variation_plan_digest() != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V16 zero-VJP probe plan digest changed")
    if OUTPUT.exists() or TEMP.exists():
        raise RuntimeError("V16 zero-VJP probe target is not fresh")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V16 zero-VJP probe requires exactly one CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    original = zero_vjp(v15.functional_adamw_step, device=device)
    repaired = zero_vjp(v16.functional_adamw_step, device=device)
    parity = forward_parity(device=device)
    torch.cuda.synchronize()
    if original["vjp_all_finite"] is not False:
        raise RuntimeError("V15 zero-direction defect was not reproduced")
    if repaired != {
        "vjp_values": 19,
        "vjp_finite_values": 19,
        "vjp_nonfinite_values": 0,
        "vjp_all_finite": True,
        "forward_equals_input": True,
        "state_step": 1,
        "state_exp_avg_nonzero": 0,
        "state_exp_avg_sq_nonzero": 0,
    }:
        raise RuntimeError("V16 exact-zero CUDA VJP contract failed")
    if parity["all_parameter_and_state_values_bit_equal"] is not True:
        raise RuntimeError("V16 CUDA forward parity changed")
    report = {
        "protocol_id": v16._PROTOCOL_ID,
        "kind": "DIRECT_EXACT_ZERO_DIRECTION_VJP_CUDA_PROBE",
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "launch_hashes": observed,
        "original_v15": original,
        "repaired_v16": repaired,
        "forward_parity": parity,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": 0,
        "semantic_fit_performed": False,
        "evaluation_or_classification_performed": False,
    }
    encoded = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
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
                "event": "V16_ZERO_VJP_CUDA_COMPLETE",
                "status": "PASS",
                "original_vjp_finite": original["vjp_all_finite"],
                "repaired_vjp_finite": repaired["vjp_all_finite"],
                "forward_bit_equal": parity[
                    "all_parameter_and_state_values_bit_equal"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
