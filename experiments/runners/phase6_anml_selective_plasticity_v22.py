"""ANML-style selective plasticity for Project Angler's frozen V20 core.

V22 independently implements the *algorithmic* separation described by ANML:
one slow neuromodulator controls which coordinates of a fixed representation
are exposed to a tiny online prediction learner.  No donor implementation is
copied.  The exact V20/V19 controller and public credit-row builder remain the
authority; this module adds only a 64-coordinate gate and orchestration for a
constant-size 64-value fast head with two Adam moments.

The learner never receives commitment, task, query, seed, panel, or evaluator
identity.  Those values exist only in the frozen orchestration plan.  Lifetime
updates use no replay and never expand the learned state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
import copy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.nn import functional as F

from experiments.evaluators import phase6_v19_paired_graph_context_recovery as v19r1
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-anml-selective-plasticity.v22"
CHECKPOINT_VERSION = "angler.phase6-anml-selective-plasticity.v1"
ACTIVE_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-ANML-SELECTIVE-PLASTICITY-V22-001.md"
)
ACTIVE_LEAF_SHA256 = (
    "9E7E4F6D3B57DF66B1FF2E696FC4370397CF2A72DFE17D5168E3B2BB12B7B8EC"
)

V20_CHECKPOINT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt"
)
V20_REPORT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.json"
)
V19_SOURCE_CHECKPOINT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
V20_CHECKPOINT_SHA256 = (
    "D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48"
)
V20_REPORT_SHA256 = (
    "5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498"
)
V19_SOURCE_CHECKPOINT_SHA256 = v20.SOURCE_CHECKPOINT_SHA256
V20_TERMINAL_SYSTEM_DIGEST = (
    "sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3"
)
V20_RUNNER_SHA256 = (
    "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
)

FROZEN_DEPENDENCY_HASHES = {
    ACTIVE_LEAF: ACTIVE_LEAF_SHA256,
    "experiments/runners/phase6_oml_relation_representation.py": V20_RUNNER_SHA256,
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py": (
        "E9656044749805E626C2DD443EBB5C34E95656CE11128AB5B4D6A3425C927517"
    ),
}

FAST_PARAMETER_NAME = v20.FAST_PARAMETER_NAME
GATE_WIDTH = 64
GATE_PARAMETER_COUNT = 8_320
FAST_PARAMETER_COUNT = 64
FAST_STATE_VALUE_COUNT = 192
FAST_STATE_BYTES_FP32 = 768

ARM_SECOND_ORDER = "second_order_anml"
ARM_FIRST_ORDER = "first_order_gate"
ARM_ALWAYS_OPEN = "always_open"
ARM_FORWARD_ONLY = "forward_only"
ARM_MEAN_GATE = "mean_gate"
ARM_PERMUTED_GATE = "permuted_gate"
ARM_RANDOM_GATE = "random_gate"
LEARNED_ARMS = (ARM_SECOND_ORDER, ARM_FIRST_ORDER)
PRIMARY_LIFETIME_ARMS = (
    ARM_SECOND_ORDER,
    ARM_FIRST_ORDER,
    ARM_ALWAYS_OPEN,
    ARM_FORWARD_ONLY,
    ARM_MEAN_GATE,
    ARM_PERMUTED_GATE,
)

OUTER_UPDATES = 240
INNER_STEPS = 8
OUTER_STREAMS = 8
INNER_LEARNING_RATE = 1.0e-3
OUTER_LEARNING_RATE = 3.0e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1.0e-8
ADAM_WEIGHT_DECAY = 0.0
OUTER_GRADIENT_CLIP = 5.0
PROGRESS_META_INTERVAL = 40

LIFETIME_PANELS = 4
LIFETIME_UPDATES = 4_096
LIFETIME_COMMITMENTS = tuple(range(32, 64))
LIFETIME_EXPOSURES_PER_COMMITMENT = 128
PROBE_MILESTONES = (0, 512, 2_048, 4_096)
PROGRESS_LIFETIME_INTERVAL = 512
LIFETIME_RESUME_STEPS = tuple(range(0, LIFETIME_UPDATES + 1, PROGRESS_LIFETIME_INTERVAL))
PROBE_COMMITMENTS = tuple(range(64))
PROBE_FAMILIES = {
    "original": tuple(range(0, 8)),
    "meta_fit": tuple(range(8, 32)),
    "unseen": tuple(range(32, 64)),
    "fully_v20_heldout": tuple(range(56, 64)),
}

META_SEED_BASES = {
    "inner": 31_000_000_001,
    "outer_current": 32_000_000_001,
    "outer_remember": 33_000_000_001,
}
LIFETIME_SEED_BASE = 40_000_000_001
PROBE_SEED_BASE = 60_000_000_001
ANML_INITIALIZATION_SEED = 34_000_000_001
RANDOM_GATE_SEED = 34_100_000_001
PERMUTATION_SEED = 34_200_000_001

ALLOCATED_MEMORY_CEILING_BYTES = 12 * 1024**3
SEMANTIC_WALL_TIME_CEILING_SECONDS = 6.0 * 60.0 * 60.0
FEATURE_EQUIVALENCE_TOLERANCE = 1.0e-6

_PLAN_DIGEST_DOMAIN = b"project-angler.anml-selective-plasticity.plan.v1\x00"
_GATE_DIGEST_DOMAIN = b"project-angler.anml-selective-plasticity.gate.v1\x00"
_FAST_DIGEST_DOMAIN = b"project-angler.anml-selective-plasticity.fast.v1\x00"
_SYSTEM_DIGEST_DOMAIN = b"project-angler.anml-selective-plasticity.system.v1\x00"
_OBJECT_DIGEST_DOMAIN = b"project-angler.anml-selective-plasticity.object.v1\x00"

AdamWSlot = v16.AdamWSlot
functional_adamw_step = v16.functional_adamw_step


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _update_tensor_digest(
    digest: "hashlib._Hash", name: str, value: torch.Tensor
) -> None:
    tensor = value.detach().cpu().contiguous()
    encoded_name = name.encode("utf-8")
    encoded_dtype = str(tensor.dtype).encode("ascii")
    digest.update(len(encoded_name).to_bytes(4, "big"))
    digest.update(encoded_name)
    digest.update(len(encoded_dtype).to_bytes(4, "big"))
    digest.update(encoded_dtype)
    digest.update(tensor.ndim.to_bytes(4, "big"))
    for size in tensor.shape:
        digest.update(int(size).to_bytes(8, "big"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def _update_object_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        _update_tensor_digest(digest, "tensor", value)
    elif value is None:
        digest.update(b"none\x00")
    elif isinstance(value, bool):
        digest.update(b"bool\x00" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\x00" + str(value).encode("ascii") + b"\x00")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("digest payload contains a non-finite float")
        digest.update(b"float\x00" + value.hex().encode("ascii") + b"\x00")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str\x00" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, Mapping):
        digest.update(b"mapping\x00")
        for key in sorted(value, key=lambda item: str(item)):
            _update_object_digest(digest, str(key))
            _update_object_digest(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\x00" + len(value).to_bytes(8, "big"))
        for item in value:
            _update_object_digest(digest, item)
    else:
        raise TypeError(f"unsupported digest payload type: {type(value).__name__}")


def _object_digest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256(domain)
    _update_object_digest(digest, value)
    return "sha256:" + digest.hexdigest()


def _mapping_digest(domain: bytes, values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(domain)
    for name, value in sorted(values.items()):
        _update_tensor_digest(digest, name, value)
    return "sha256:" + digest.hexdigest()


def _require_finite_tensor(label: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RuntimeError(f"V22 {label} is not a finite floating tensor")


def _allocated_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    index = torch.cuda.current_device() if device.index is None else device.index
    return int(torch.cuda.max_memory_allocated(index))


class ANMLNeuromodulator(nn.Module):
    """The independent 8,320-parameter 64-coordinate V22 gate."""

    def __init__(self, *, zero_final: bool = True) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(GATE_WIDTH, elementwise_affine=False),
            nn.Linear(GATE_WIDTH, GATE_WIDTH),
            nn.SiLU(),
            nn.Linear(GATE_WIDTH, GATE_WIDTH),
        )
        if zero_final:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        if sum(parameter.numel() for parameter in self.parameters()) != GATE_PARAMETER_COUNT:
            raise RuntimeError("V22 neuromodulator parameter count changed")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return centered_gate(self, hidden)


def centered_gate(module: ANMLNeuromodulator, hidden: torch.Tensor) -> torch.Tensor:
    """Return ``2*sigmoid(logits)`` without exposing orchestration identity."""

    if not isinstance(module, ANMLNeuromodulator):
        raise TypeError("V22 centered gate requires an ANMLNeuromodulator")
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.shape[-1:] != (GATE_WIDTH,)
        or not hidden.is_floating_point()
        or not bool(torch.isfinite(hidden).all().item())
    ):
        raise ValueError("V22 gate input must be finite [...,64]")
    logits = module.network(hidden)
    gate = 2.0 * torch.sigmoid(logits)
    _require_finite_tensor("gate", gate)
    return gate


GateMode = Literal["live", "open", "mean", "permuted"]
_ACTIVE_GATE_HOOKS: WeakKeyDictionary[nn.Module, int] = WeakKeyDictionary()


def active_gate_hook_count(
    controller: v19.V12ChampionPairedGraphContextController,
) -> int:
    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V22 hook count requires the exact V19 controller")
    return int(_ACTIVE_GATE_HOOKS.get(controller.relation_comparator[2], 0))


def _validate_permutation(
    permutation: torch.Tensor | Sequence[int] | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if permutation is None:
        raise ValueError("V22 permuted gate requires a permutation")
    result = torch.as_tensor(permutation, dtype=torch.long, device=device)
    if result.shape != (GATE_WIDTH,) or set(result.detach().cpu().tolist()) != set(
        range(GATE_WIDTH)
    ):
        raise ValueError("V22 gate permutation is not a permutation of 0..63")
    return result


def _gate_for_hidden(
    hidden: torch.Tensor,
    gate_module: ANMLNeuromodulator | None,
    *,
    mode: GateMode,
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    if mode == "open":
        if gate_module is not None and not isinstance(gate_module, ANMLNeuromodulator):
            raise TypeError("V22 gate module has the wrong type")
        return torch.ones_like(hidden)
    if not isinstance(gate_module, ANMLNeuromodulator):
        raise TypeError("V22 live gate modes require an ANMLNeuromodulator")
    gate = centered_gate(gate_module, hidden)
    if mode == "live":
        return gate
    if mode == "mean":
        return gate.mean(dim=-1, keepdim=True).expand_as(gate)
    if mode == "permuted":
        order = _validate_permutation(permutation, device=hidden.device)
        return gate.index_select(-1, order)
    raise ValueError(f"unknown V22 gate mode: {mode}")


def apply_gate_lesion(
    gate: torch.Tensor,
    lesion: GateMode,
    *,
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply a declared lesion to already-computed gate values."""

    _require_finite_tensor("gate lesion input", gate)
    if gate.shape[-1:] != (GATE_WIDTH,):
        raise ValueError("V22 gate lesion input must end in width 64")
    if lesion == "live":
        return gate
    if lesion == "open":
        return torch.ones_like(gate)
    if lesion == "mean":
        return gate.mean(dim=-1, keepdim=True).expand_as(gate)
    if lesion == "permuted":
        order = _validate_permutation(permutation, device=gate.device)
        return gate.index_select(-1, order)
    raise ValueError(f"unknown V22 gate lesion: {lesion}")


@contextmanager
def scoped_anml_gate(
    controller: v19.V12ChampionPairedGraphContextController,
    gate_module: ANMLNeuromodulator | None,
    mode: GateMode = "live",
    permutation: torch.Tensor | Sequence[int] | None = None,
    *,
    capture: list[torch.Tensor] | None = None,
) -> Iterator[None]:
    """Install exactly one temporary pre-hook and remove it in ``finally``."""

    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V22 hook requires the exact V19 controller")
    if mode not in {"live", "open", "mean", "permuted"}:
        raise ValueError("V22 gate mode is invalid")
    layer = controller.relation_comparator[2]
    if _ACTIVE_GATE_HOOKS.get(layer, 0) != 0:
        raise RuntimeError("nested V22 gate hooks are forbidden")

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        if len(inputs) != 1:
            raise RuntimeError("V22 comparator final layer input arity changed")
        hidden = inputs[0]
        if capture is not None:
            capture.append(hidden.detach().clone())
        gate = _gate_for_hidden(
            hidden, gate_module, mode=mode, permutation=permutation
        )
        return (hidden * gate,)

    handle = layer.register_forward_pre_hook(hook)
    _ACTIVE_GATE_HOOKS[layer] = 1
    try:
        yield
    finally:
        handle.remove()
        _ACTIVE_GATE_HOOKS.pop(layer, None)


class _V19FunctionalAdapter(nn.Module):
    def __init__(self, controller: v19.V12ChampionPairedGraphContextController) -> None:
        super().__init__()
        if type(controller) is not v19.V12ChampionPairedGraphContextController:
            raise TypeError("V22 adapter requires the exact V19 controller")
        self.controller = controller

    def forward(self, stream: object):
        return v19.public_paired_graph_credit_rows(self.controller, stream)


def functional_credit_rows(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream: object,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode = "live",
    permutation: torch.Tensor | Sequence[int] | None = None,
    capture: list[torch.Tensor] | None = None,
) -> tuple[v19.V19PairedGraphCreditRow, ...]:
    """Run the unchanged public V19 row path with one virtual fast weight."""

    v20._validate_parameter_partition(controller)
    original = controller.relation_comparator[2].weight
    if (
        fast_weight.shape != original.shape
        or fast_weight.device != original.device
        or fast_weight.dtype != original.dtype
    ):
        raise ValueError("V22 functional fast weight is not aligned")
    _require_finite_tensor("functional fast weight", fast_weight)
    adapter = _V19FunctionalAdapter(controller)
    parameter_name = f"controller.{FAST_PARAMETER_NAME}"
    try:
        with scoped_anml_gate(
            controller,
            gate_module,
            lesion,
            permutation,
            capture=capture,
        ):
            rows = torch.func.functional_call(
                adapter,
                {parameter_name: fast_weight},
                (stream,),
                tie_weights=True,
                strict=False,
            )
    finally:
        if active_gate_hook_count(controller) != 0:
            raise RuntimeError("V22 leaked its temporary gate hook")
        if controller.relation_comparator[2].weight is not original:
            raise RuntimeError("V22 functional call did not restore the fast parameter")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ANMLFeatureRow:
    heldout_index: int
    transition_index: int
    positive_index: int
    negative_index: int
    positive_hidden: torch.Tensor
    negative_hidden: torch.Tensor
    context_weights: torch.Tensor
    context_null_weight: torch.Tensor
    context_real_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ANMLFeatureBundle:
    """Transient immutable hidden features from one exact public row build."""

    rows: tuple[ANMLFeatureRow, ...]
    source_row_count: int


def capture_feature_bundle(
    controller: v19.V12ChampionPairedGraphContextController,
    reference_fast_weight: torch.Tensor,
    stream: object,
) -> ANMLFeatureBundle:
    """Capture raw final-layer inputs; never persist or expose them to routing."""

    captures: list[torch.Tensor] = []
    rows = functional_credit_rows(
        controller,
        reference_fast_weight,
        stream,
        None,
        lesion="open",
        capture=captures,
    )
    if len(rows) != v20.ROWS_PER_STREAM or len(captures) != 2 * len(rows):
        raise RuntimeError("V22 public feature capture call structure changed")
    bundled = []
    for index, row in enumerate(rows):
        positive_hidden = captures[2 * index]
        negative_hidden = captures[2 * index + 1]
        if (
            positive_hidden.ndim != 3
            or negative_hidden.ndim != 3
            or positive_hidden.shape[-1] != GATE_WIDTH
            or negative_hidden.shape[-1] != GATE_WIDTH
            or positive_hidden.shape != negative_hidden.shape
        ):
            raise RuntimeError("V22 captured hidden feature shape changed")
        bundled.append(
            ANMLFeatureRow(
                heldout_index=row.heldout_index,
                transition_index=row.transition_index,
                positive_index=row.positive_index,
                negative_index=row.negative_index,
                positive_hidden=positive_hidden,
                negative_hidden=negative_hidden,
                context_weights=row.context_weights.detach().clone(),
                context_null_weight=row.context_null_weight.detach().clone(),
                context_real_logits=row.context_real_logits.detach().clone(),
            )
        )
    return ANMLFeatureBundle(rows=tuple(bundled), source_row_count=len(rows))


def _feature_row_to_public_row(
    row: ANMLFeatureRow,
    fast_weight: torch.Tensor,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode,
    permutation: torch.Tensor | Sequence[int] | None,
) -> v19.V19PairedGraphCreditRow:
    positive_gate = _gate_for_hidden(
        row.positive_hidden,
        gate_module,
        mode=lesion,
        permutation=permutation,
    )
    negative_gate = _gate_for_hidden(
        row.negative_hidden,
        gate_module,
        mode=lesion,
        permutation=permutation,
    )
    positive_logits = torch.tanh(
        F.linear(row.positive_hidden * positive_gate, fast_weight).squeeze(-1)
    )
    negative_logits = torch.tanh(
        F.linear(row.negative_hidden * negative_gate, fast_weight).squeeze(-1)
    )
    slot_positive = positive_logits[row.positive_index] - positive_logits[row.negative_index]
    slot_negative = negative_logits[row.positive_index] - negative_logits[row.negative_index]
    metrics = v12._relation_valid_set_metrics(
        slot_positive.detach(),
        slot_negative.detach(),
        row.context_weights,
        row.context_null_weight,
    )
    valid_mask = metrics["valid_mask"]
    if not isinstance(valid_mask, torch.Tensor):
        raise RuntimeError("V22 valid-set helper lost its tensor mask")
    positive_scores = (row.context_weights * positive_logits).sum(dim=-1)
    negative_scores = (row.context_weights * negative_logits).sum(dim=-1)
    result = v19.V19PairedGraphCreditRow(
        heldout_index=row.heldout_index,
        transition_index=row.transition_index,
        positive_index=row.positive_index,
        negative_index=row.negative_index,
        positive_margin=positive_scores[row.positive_index]
        - positive_scores[row.negative_index],
        negative_margin=negative_scores[row.positive_index]
        - negative_scores[row.negative_index],
        slot_positive_margins=slot_positive,
        slot_negative_margins=slot_negative,
        context_weights=row.context_weights,
        context_null_weight=row.context_null_weight,
        context_real_logits=row.context_real_logits,
        valid_mask=valid_mask.detach(),
    )
    return result


def rows_from_feature_bundle(
    bundle: ANMLFeatureBundle,
    fast_weight: torch.Tensor,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode = "live",
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> tuple[v19.V19PairedGraphCreditRow, ...]:
    if not isinstance(bundle, ANMLFeatureBundle) or bundle.source_row_count != v20.ROWS_PER_STREAM:
        raise TypeError("V22 feature bundle is invalid")
    _require_finite_tensor("feature-path fast weight", fast_weight)
    return tuple(
        _feature_row_to_public_row(
            row,
            fast_weight,
            gate_module,
            lesion=lesion,
            permutation=permutation,
        )
        for row in bundle.rows
    )


def _stream_rows(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream_or_bundle: object,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode = "live",
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> tuple[v19.V19PairedGraphCreditRow, ...]:
    if isinstance(stream_or_bundle, ANMLFeatureBundle):
        return rows_from_feature_bundle(
            stream_or_bundle,
            fast_weight,
            gate_module,
            lesion=lesion,
            permutation=permutation,
        )
    return functional_credit_rows(
        controller,
        fast_weight,
        stream_or_bundle,
        gate_module,
        lesion=lesion,
        permutation=permutation,
    )


def _stream_loss(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream_or_bundle: object,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode = "live",
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    rows = _stream_rows(
        controller,
        fast_weight,
        stream_or_bundle,
        gate_module,
        lesion=lesion,
        permutation=permutation,
    )
    return v20._stream_loss_from_rows(rows)


def verify_feature_bundle_equivalence(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream: object,
    gate_module: ANMLNeuromodulator,
    *,
    tolerance: float = FEATURE_EQUIVALENCE_TOLERANCE,
) -> dict[str, object]:
    """Compare rows, loss, fast gradient, and one AdamW transition exactly."""

    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("V22 equivalence tolerance is invalid")
    bundle = capture_feature_bundle(controller, fast_weight, stream)

    def compute(use_bundle: bool):
        weight = fast_weight.detach().clone().requires_grad_(True)
        rows = (
            rows_from_feature_bundle(bundle, weight, gate_module)
            if use_bundle
            else functional_credit_rows(controller, weight, stream, gate_module)
        )
        loss = v20._stream_loss_from_rows(rows)
        gradient = torch.autograd.grad(loss, weight, create_graph=False)[0]
        zero = torch.zeros_like(weight)
        (updated,), state = functional_adamw_step(
            (weight,),
            (gradient.detach(),),
            (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),),
            (INNER_LEARNING_RATE,),
            beta1=ADAM_BETA1,
            beta2=ADAM_BETA2,
            epsilon=ADAM_EPSILON,
            weight_decay=ADAM_WEIGHT_DECAY,
        )
        return rows, loss.detach(), gradient.detach(), updated.detach(), state[0]

    direct = compute(False)
    cached = compute(True)
    deltas = {
        "loss": float((direct[1] - cached[1]).abs().item()),
        "fast_gradient": float((direct[2] - cached[2]).abs().amax().item()),
        "adamw_weight": float((direct[3] - cached[3]).abs().amax().item()),
        "adamw_first_moment": float(
            (direct[4].exp_avg - cached[4].exp_avg).abs().amax().item()
        ),
        "adamw_second_moment": float(
            (direct[4].exp_avg_sq - cached[4].exp_avg_sq).abs().amax().item()
        ),
    }
    row_deltas = []
    for left, right in zip(direct[0], cached[0], strict=True):
        values = (
            (left.positive_margin, right.positive_margin),
            (left.negative_margin, right.negative_margin),
            (left.slot_positive_margins, right.slot_positive_margins),
            (left.slot_negative_margins, right.slot_negative_margins),
            (left.context_weights, right.context_weights),
            (left.context_null_weight, right.context_null_weight),
            (left.context_real_logits, right.context_real_logits),
        )
        row_deltas.append(
            max(float((a - b).abs().amax().item()) for a, b in values)
        )
        if not torch.equal(left.valid_mask, right.valid_mask):
            raise RuntimeError("V22 feature path changed a row valid mask")
    deltas["rows"] = max(row_deltas)
    passed = all(value <= tolerance for value in deltas.values())
    if not passed:
        raise RuntimeError("V22 shared-feature path is not exact-equivalent")
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_deltas": deltas,
        "row_count": len(direct[0]),
        "hook_count_after_success": active_gate_hook_count(controller),
    }


def exact_equivalent_feature_parity(
    controller: v19.V12ChampionPairedGraphContextController,
    gate_module: ANMLNeuromodulator,
    stream: object,
    *,
    duplicate_same_contract: bool = False,
    fast_weight: torch.Tensor | None = None,
    tolerance: float = FEATURE_EQUIVALENCE_TOLERANCE,
) -> dict[str, object]:
    """Preclaim parity, including V19's non-dedup ``zero_residual`` path.

    ``duplicate_same_contract=True`` deliberately activates V19's
    ``zero_residual`` lesion.  That path returns before the paired-graph
    duplicate-row reuse loop and therefore exercises the inherited
    same-contract rows without the dedup optimization; no hidden tensor is
    synthetically duplicated by V22.
    """

    if type(duplicate_same_contract) is not bool:
        raise TypeError("V22 duplicate-same-contract flag must be bool")
    selected_weight = (
        controller.relation_comparator[2].weight.detach().clone()
        if fast_weight is None
        else fast_weight
    )
    baseline_hooks = active_gate_hook_count(controller)
    audit: dict[str, object] | None = None
    if duplicate_same_contract:
        audit = {
            "frozen_method_calls": 0,
            "zero_residual_calls": 0,
            "delegated_nonzero_lesion_calls": 0,
            "zero_residual_rows": 0,
            "representative_rows": 0,
            "duplicate_groups": 0,
            "duplicate_rows_projected": 0,
            "maximum_raw_duplicate_logit_difference": 0.0,
        }
        with v19r1._temporary_zero_residual_projection(controller, audit):
            with controller.paired_graph_lesion("zero_residual"):
                report = verify_feature_bundle_equivalence(
                    controller,
                    selected_weight,
                    stream,
                    gate_module,
                    tolerance=tolerance,
                )
        if (
            int(audit["zero_residual_calls"]) <= 0
            or int(audit["duplicate_groups"]) <= 0
            or int(audit["duplicate_rows_projected"]) <= 0
        ):
            raise RuntimeError("V22 duplicate parity did not exercise projected duplicate rows")
    else:
        with nullcontext():
            report = verify_feature_bundle_equivalence(
                controller,
                selected_weight,
                stream,
                gate_module,
                tolerance=tolerance,
            )
    if active_gate_hook_count(controller) != baseline_hooks:
        raise RuntimeError("V22 parity check leaked a gate hook")
    deltas = report["maximum_absolute_deltas"]
    maximum = max(float(value) for value in deltas.values())
    exact = maximum <= tolerance
    return {
        **report,
        "row_values_exact": exact,
        "row_order_exact": exact,
        "row_masks_exact": exact,
        "loss_exact": float(deltas["loss"]) <= tolerance,
        "fast_gradient_exact": float(deltas["fast_gradient"]) <= tolerance,
        "adamw_transition_exact": max(
            float(deltas["adamw_weight"]),
            float(deltas["adamw_first_moment"]),
            float(deltas["adamw_second_moment"]),
        )
        <= tolerance,
        "duplicate_same_contract_exercised": duplicate_same_contract,
        "v19_non_dedup_zero_residual_path": duplicate_same_contract,
        "zero_residual_projection_audit": copy.deepcopy(audit),
        "zero_residual_wrapper_restored": (
            "_paired_graph_context_logits" not in controller.__dict__
            and "_v19_evaluation_recovery_wrapper_active" not in controller.__dict__
        ),
        "maximum_abs_delta": maximum,
    }


public_feature_parity_report = exact_equivalent_feature_parity


def _meta_seed(role: str, update: int, position: int, kind: int) -> int:
    if role not in META_SEED_BASES or kind not in (0, 1):
        raise ValueError("V22 meta seed role or kind is invalid")
    return META_SEED_BASES[role] + 100_000 * update + 1_000 * position + 500_000_000 * kind


def _meta_record(
    role: str,
    update: int,
    position: int,
    commitment_index: int,
) -> dict[str, object]:
    return {
        "role": role,
        "update": update,
        "position": position,
        "commitment_index": commitment_index,
        "topology_seed": _meta_seed(role, update, position, 0),
        "surface_seed": _meta_seed(role, update, position, 1),
    }


@lru_cache(maxsize=4)
def lifetime_order(panel: int) -> tuple[int, ...]:
    if type(panel) is not int or not 0 <= panel < LIFETIME_PANELS:
        raise ValueError("V22 lifetime panel is invalid")
    if panel < 2:
        order = tuple(
            commitment
            for commitment in LIFETIME_COMMITMENTS
            for _ in range(LIFETIME_EXPOSURES_PER_COMMITMENT)
        )
    else:
        order = tuple(
            32 + ((13 * position + 7 * cycle) % 32)
            for cycle in range(LIFETIME_EXPOSURES_PER_COMMITMENT)
            for position in range(32)
        )
    if (
        len(order) != LIFETIME_UPDATES
        or any(order.count(index) != LIFETIME_EXPOSURES_PER_COMMITMENT for index in LIFETIME_COMMITMENTS)
    ):
        raise RuntimeError("V22 lifetime order balance changed")
    return order


def _lifetime_record(panel: int, step: int) -> dict[str, object]:
    order = lifetime_order(panel)
    if type(step) is not int or not 0 <= step < LIFETIME_UPDATES:
        raise ValueError("V22 lifetime step is invalid")
    base = LIFETIME_SEED_BASE + 2_000_000_000 * panel
    return {
        "role": "lifetime_update",
        "panel": panel,
        "step": step,
        "commitment_index": order[step],
        "topology_seed": base + 10_000 * step,
        "surface_seed": base + 1_000_000_000 + 10_000 * step,
    }


def _probe_record(panel: int, commitment_index: int) -> dict[str, object]:
    if not 0 <= panel < LIFETIME_PANELS or not 0 <= commitment_index < 64:
        raise ValueError("V22 probe panel or commitment is invalid")
    base = PROBE_SEED_BASE + 2_000_000_000 * panel
    return {
        "role": "fixed_probe",
        "panel": panel,
        "commitment_index": commitment_index,
        "topology_seed": base + 10_000 * commitment_index,
        "surface_seed": base + 1_000_000_000 + 10_000 * commitment_index,
    }


@lru_cache(maxsize=1)
def _frozen_plan_payload() -> dict[str, object]:
    commitments = tuple(v12.software_pipeline_mechanism_partition("train")[:64])
    if len(commitments) != 64 or len(set(commitments)) != 64:
        raise RuntimeError("V22 requires 64 distinct public train commitments")
    meta_updates = []
    meta_records = []
    target_counts = [0] * 24
    for update in range(OUTER_UPDATES):
        target = 8 + update % 24
        target_counts[target - 8] += 1
        inner = tuple(_meta_record("inner", update, position, target) for position in range(8))
        current = tuple(
            _meta_record("outer_current", update, position, target)
            for position in range(4)
        )
        remember = tuple(
            _meta_record(
                "outer_remember",
                update,
                position,
                8 + ((target - 8 + offset) % 24),
            )
            for position, offset in enumerate((5, 10, 15, 20))
        )
        records = inner + current + remember
        meta_records.extend(records)
        meta_updates.append(
            {
                "update": update,
                "target_commitment_index": target,
                "inner": inner,
                "outer_current": current,
                "outer_remember": remember,
                "outer": current + remember,
            }
        )
    lifetime_records = tuple(
        _lifetime_record(panel, step)
        for panel in range(LIFETIME_PANELS)
        for step in range(LIFETIME_UPDATES)
    )
    probe_records = tuple(
        _probe_record(panel, commitment)
        for panel in range(LIFETIME_PANELS)
        for commitment in PROBE_COMMITMENTS
    )
    meta_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"]))
        for record in meta_records
    }
    lifetime_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"]))
        for record in lifetime_records
    }
    probe_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"]))
        for record in probe_records
    }
    all_pairs = meta_pairs | lifetime_pairs | probe_pairs
    all_seed_values = {value for pair in all_pairs for value in pair}
    if (
        target_counts != [10] * 24
        or len(meta_records) != OUTER_UPDATES * 16
        or len(meta_pairs) != len(meta_records)
        or len(lifetime_records) != LIFETIME_PANELS * LIFETIME_UPDATES
        or len(lifetime_pairs) != len(lifetime_records)
        or len(probe_records) != LIFETIME_PANELS * len(PROBE_COMMITMENTS)
        or len(probe_pairs) != len(probe_records)
        or len(all_pairs) != len(meta_pairs) + len(lifetime_pairs) + len(probe_pairs)
        or len(all_seed_values) != 2 * len(all_pairs)
    ):
        raise RuntimeError("V22 stream schedule uniqueness or balance changed")
    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "source": {
            "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
            "v20_report_sha256": V20_REPORT_SHA256,
            "v20_terminal_system_digest": V20_TERMINAL_SYSTEM_DIGEST,
        },
        "commitments": commitments,
        "historical_retention_indices": tuple(range(0, 8)),
        "meta_fit_indices": tuple(range(8, 32)),
        "lifetime_indices": LIFETIME_COMMITMENTS,
        "fully_v20_heldout_indices": tuple(range(56, 64)),
        "meta_updates": tuple(meta_updates),
        "meta_target_counts": tuple(target_counts),
        "meta_unique_streams": len(meta_pairs),
        "lifetime": {
            "panels": LIFETIME_PANELS,
            "updates_per_panel": LIFETIME_UPDATES,
            "exposures_per_commitment": LIFETIME_EXPOSURES_PER_COMMITMENT,
            "panel_kinds": ("blocked", "blocked", "interleaved", "interleaved"),
            "unique_update_streams": len(lifetime_pairs),
            "fixed_probe_streams": len(probe_pairs),
            "probe_milestones": PROBE_MILESTONES,
        },
        "gate": {
            "width": GATE_WIDTH,
            "parameter_count": GATE_PARAMETER_COUNT,
            "formula": "2*sigmoid(logits)",
            "initial_gate": 1.0,
        },
        "fast_state": {
            "weight_values": FAST_PARAMETER_COUNT,
            "adam_moment_values": 2 * FAST_PARAMETER_COUNT,
            "total_values": FAST_STATE_VALUE_COUNT,
            "fp32_bytes": FAST_STATE_BYTES_FP32,
        },
        "optimization": {
            "outer_updates": OUTER_UPDATES,
            "inner_steps": INNER_STEPS,
            "inner_learning_rate": INNER_LEARNING_RATE,
            "outer_learning_rate": OUTER_LEARNING_RATE,
            "betas": (ADAM_BETA1, ADAM_BETA2),
            "epsilon": ADAM_EPSILON,
            "weight_decay": ADAM_WEIGHT_DECAY,
            "gradient_clip": OUTER_GRADIENT_CLIP,
        },
        "numerical": {
            "device": "cuda",
            "dtype": "torch.float32",
            "tf32": False,
            "autocast": False,
            "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
            "wall_time_ceiling_seconds": SEMANTIC_WALL_TIME_CEILING_SECONDS,
        },
    }
    digest = hashlib.sha256(_PLAN_DIGEST_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    payload["plan_digest"] = "sha256:" + digest.hexdigest()
    return payload


def anml_fit_plan() -> dict[str, object]:
    plan = copy.deepcopy(_frozen_plan_payload())
    # Stable flat aliases keep inspection/test tooling independent of the
    # nested descriptive sections while preserving one underlying schedule.
    plan["outer_updates"] = OUTER_UPDATES
    plan["meta_fit_commitments"] = tuple(range(8, 32))
    plan["unseen_commitments"] = tuple(range(32, 64))
    plan["updates"] = tuple(
        {
            **record,
            "target_commitment": record["target_commitment_index"],
        }
        for record in plan["meta_updates"]
    )
    plan["lifetime_records"] = tuple(
        tuple(_lifetime_record(panel, step) for step in range(LIFETIME_UPDATES))
        for panel in range(LIFETIME_PANELS)
    )
    plan["probe_records"] = tuple(
        tuple(_probe_record(panel, index) for index in PROBE_COMMITMENTS)
        for panel in range(LIFETIME_PANELS)
    )
    return plan


def anml_plan() -> dict[str, object]:
    return anml_fit_plan()


def anml_plan_digest() -> str:
    return str(_frozen_plan_payload()["plan_digest"])


def _make_stream(record: Mapping[str, object]):
    commitments = _frozen_plan_payload()["commitments"]
    index = int(record["commitment_index"])
    return v12.make_software_pipeline_stream(
        int(record["topology_seed"]),
        surface_seed=int(record["surface_seed"]),
        supports_per_motif=2,
        queries=1,
        maximum_steps=4,
        mechanism_commitment=commitments[index],
        mechanism_partition="train",
    )


def build_meta_fit_streams(update: int) -> dict[str, tuple[object, ...]]:
    if type(update) is not int or not 0 <= update < OUTER_UPDATES:
        raise ValueError("V22 meta update is outside the frozen plan")
    record = _frozen_plan_payload()["meta_updates"][update]
    return {
        "inner": tuple(_make_stream(item) for item in record["inner"]),
        "outer": tuple(_make_stream(item) for item in record["outer"]),
    }


def build_lifetime_streams(
    panel: int,
    *,
    start: int = 0,
    stop: int = LIFETIME_UPDATES,
) -> tuple[dict[str, object], ...]:
    """Return inspectable frozen stream specifications, not learned inputs."""

    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start <= stop <= LIFETIME_UPDATES
    ):
        raise ValueError("V22 lifetime stream slice is invalid")
    return tuple(_lifetime_record(panel, step) for step in range(start, stop))


def materialize_lifetime_streams(
    panel: int,
    *,
    start: int = 0,
    stop: int = LIFETIME_UPDATES,
) -> tuple[object, ...]:
    return tuple(_make_stream(record) for record in build_lifetime_streams(panel, start=start, stop=stop))


def build_probe_streams(panel: int) -> tuple[dict[str, object], ...]:
    return tuple(_probe_record(panel, index) for index in PROBE_COMMITMENTS)


def materialize_probe_streams(panel: int) -> tuple[object, ...]:
    return tuple(_make_stream(record) for record in build_probe_streams(panel))


@dataclass(slots=True)
class ANMLArm:
    name: str
    gate: ANMLNeuromodulator
    outer_optimizer: torch.optim.Optimizer
    meta_updates: int = 0


@dataclass(slots=True)
class ANMLSystem:
    controller: v19.V12ChampionPairedGraphContextController
    fast_initial_weight: torch.Tensor
    second_order_anml: ANMLArm
    first_order_gate: ANMLArm
    random_gate: ANMLNeuromodulator
    gate_permutation: torch.Tensor
    source_controller_digest: str
    source_v20_system_digest: str
    source_checkpoint_sha256: str
    source_auxiliary_digest: str
    completed_meta_updates: int = 0
    feature_equivalence_verified: bool = False
    harness_state: dict[str, object] = field(default_factory=dict)

    def arm(self, name: str) -> ANMLArm:
        if name == ARM_SECOND_ORDER:
            return self.second_order_anml
        if name == ARM_FIRST_ORDER:
            return self.first_order_gate
        raise KeyError(f"unknown V22 learned arm: {name}")


def _seeded_gate(
    seed: int,
    *,
    device: torch.device,
    zero_final: bool,
) -> ANMLNeuromodulator:
    cpu_state = torch.get_rng_state()
    cuda_states = (
        tuple(torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else ()
    )
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gate = ANMLNeuromodulator(zero_final=zero_final).to(device=device, dtype=torch.float32)
    finally:
        torch.set_rng_state(cpu_state)
        for index, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, index)
    return gate


def _gate_digest(gate: ANMLNeuromodulator) -> str:
    return _mapping_digest(_GATE_DIGEST_DOMAIN, gate.state_dict())


def _controller_digest(controller: v19.V12ChampionPairedGraphContextController) -> str:
    return v20.oml_controller_digest(controller)


def _configure_frozen_controller(
    controller: v19.V12ChampionPairedGraphContextController,
) -> None:
    v20._validate_parameter_partition(controller)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _assert_system_integrity(system: ANMLSystem) -> None:
    if not isinstance(system, ANMLSystem):
        raise TypeError("V22 requires an ANMLSystem")
    if type(system.controller) is not v19.V12ChampionPairedGraphContextController:
        raise RuntimeError("V22 controller type changed")
    if any(parameter.requires_grad for parameter in system.controller.parameters()):
        raise RuntimeError("V22 source controller is not frozen")
    if _controller_digest(system.controller) != system.source_controller_digest:
        raise RuntimeError("V22 changed a frozen V20 controller tensor")
    original = system.controller.state_dict()[FAST_PARAMETER_NAME]
    if not torch.equal(original.detach(), system.fast_initial_weight.detach()):
        raise RuntimeError("V22 changed the fixed V20 fast initialization")
    if system.fast_initial_weight.shape != (1, GATE_WIDTH):
        raise RuntimeError("V22 fast initialization shape changed")
    if (
        type(system.completed_meta_updates) is not int
        or not 0 <= system.completed_meta_updates <= OUTER_UPDATES
        or system.second_order_anml.meta_updates != system.completed_meta_updates
        or system.first_order_gate.meta_updates != system.completed_meta_updates
    ):
        raise RuntimeError("V22 paired meta-update counters diverged")
    for arm in (system.second_order_anml, system.first_order_gate):
        if arm.name not in LEARNED_ARMS:
            raise RuntimeError("V22 learned arm identity changed")
        if sum(parameter.numel() for parameter in arm.gate.parameters()) != GATE_PARAMETER_COUNT:
            raise RuntimeError("V22 learned gate capacity changed")
        if any(not parameter.requires_grad for parameter in arm.gate.parameters()):
            raise RuntimeError("V22 learned gate trainability changed")
        for parameter in arm.gate.parameters():
            _require_finite_tensor(f"{arm.name} gate parameter", parameter)
    if any(parameter.requires_grad for parameter in system.random_gate.parameters()):
        raise RuntimeError("V22 random control gate is trainable")
    if system.gate_permutation.shape != (GATE_WIDTH,):
        raise RuntimeError("V22 gate permutation shape changed")
    _validate_permutation(system.gate_permutation, device=system.fast_initial_weight.device)
    if not isinstance(system.harness_state, dict):
        raise RuntimeError("V22 harness state must be a dictionary")
    if active_gate_hook_count(system.controller) != 0:
        raise RuntimeError("V22 system retained a gate hook")


def anml_system_digest(system: ANMLSystem) -> str:
    _assert_system_integrity(system)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": anml_plan_digest(),
        "source_checkpoint_sha256": system.source_checkpoint_sha256,
        "source_v20_system_digest": system.source_v20_system_digest,
        "source_controller_digest": system.source_controller_digest,
        "source_auxiliary_digest": system.source_auxiliary_digest,
        "fast_initial_weight": system.fast_initial_weight,
        ARM_SECOND_ORDER: {
            "gate": system.second_order_anml.gate.state_dict(),
            "optimizer": system.second_order_anml.outer_optimizer.state_dict(),
            "updates": system.second_order_anml.meta_updates,
        },
        ARM_FIRST_ORDER: {
            "gate": system.first_order_gate.gate.state_dict(),
            "optimizer": system.first_order_gate.outer_optimizer.state_dict(),
            "updates": system.first_order_gate.meta_updates,
        },
        "random_gate": system.random_gate.state_dict(),
        "gate_permutation": system.gate_permutation,
        "completed_meta_updates": system.completed_meta_updates,
        "feature_equivalence_verified": system.feature_equivalence_verified,
    }
    return _object_digest(_SYSTEM_DIGEST_DOMAIN, payload)


def verify_anml_dependencies(
    v20_checkpoint: str | Path = V20_CHECKPOINT_PATH,
    v19_source_checkpoint: str | Path = V19_SOURCE_CHECKPOINT_PATH,
    v20_report: str | Path = V20_REPORT_PATH,
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    expected_modules = {
        "v19r1": root / "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py",
        "v12": root / "experiments/runners/phase6_software_pipeline_reconstruction.py",
        "v16": root / "experiments/runners/phase6_cross_variation_plasticity_v16.py",
        "v19": root / "experiments/runners/phase6_v12_champion_paired_graph_context.py",
        "v20": root / "experiments/runners/phase6_oml_relation_representation.py",
    }
    observed_modules = {
        "v19r1": Path(v19r1.__file__).resolve(),
        "v12": Path(v12.__file__).resolve(),
        "v16": Path(v16.__file__).resolve(),
        "v19": Path(v19.__file__).resolve(),
        "v20": Path(v20.__file__).resolve(),
    }
    if observed_modules != {name: path.resolve() for name, path in expected_modules.items()}:
        raise RuntimeError("V22 imported a shadowed dependency")
    frozen = {
        name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES
    }
    required = dict(FROZEN_DEPENDENCY_HASHES)
    if expected_hashes is not None:
        required.update({str(key): str(value).upper() for key, value in expected_hashes.items()})
    for name, expected in required.items():
        path = Path(name)
        actual = _sha256_file(path if path.is_absolute() else root / path)
        if actual != expected:
            raise RuntimeError(f"V22 frozen dependency changed: {name}")
    artifacts = {
        "v20_checkpoint_sha256": _sha256_file(v20_checkpoint),
        "v19_source_checkpoint_sha256": _sha256_file(v19_source_checkpoint),
        "v20_report_sha256": _sha256_file(v20_report),
    }
    if artifacts != {
        "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
        "v19_source_checkpoint_sha256": V19_SOURCE_CHECKPOINT_SHA256,
        "v20_report_sha256": V20_REPORT_SHA256,
    }:
        raise RuntimeError("V22 source artifact binding changed")
    return {
        "frozen_dependency_hashes": frozen,
        **artifacts,
        "active_leaf_sha256": ACTIVE_LEAF_SHA256,
    }


def _make_outer_optimizer(gate: ANMLNeuromodulator) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        tuple(gate.parameters()),
        lr=OUTER_LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )


def configure_anml_numerics(
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V22 CUDA numerical mode requires an available GPU")
        index = 0 if selected.index is None else selected.index
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.is_autocast_enabled() or (
        selected.type == "cuda" and torch.is_autocast_enabled("cuda")
    ):
        raise RuntimeError("V22 numerical mode forbids autocast")
    return {
        "device": str(selected),
        "dtype": "torch.float32",
        "tf32": False,
        "autocast": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
    }


def build_anml_system(
    v20_checkpoint: str | Path = V20_CHECKPOINT_PATH,
    v19_source_checkpoint: str | Path = V19_SOURCE_CHECKPOINT_PATH,
    *,
    device: torch.device | str = "cpu",
    verify_hashes: bool = True,
) -> ANMLSystem:
    """Load the sealed V20 second-order controller and add two fresh gates."""

    selected = torch.device(device)
    if verify_hashes:
        if _sha256_file(v20_checkpoint) != V20_CHECKPOINT_SHA256:
            raise RuntimeError("V22 V20 checkpoint SHA-256 changed")
        if _sha256_file(v19_source_checkpoint) != V19_SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("V22 V19 source checkpoint SHA-256 changed")
    source = v20.load_oml_checkpoint(
        v20_checkpoint,
        v19_source_checkpoint,
        device=selected,
    )
    if source.completed_updates != v20.OUTER_UPDATES:
        raise RuntimeError("V22 source V20 fit is incomplete")
    source_digest = v20.oml_system_digest(source)
    if source_digest != V20_TERMINAL_SYSTEM_DIGEST:
        raise RuntimeError("V22 source V20 system digest changed")
    controller = source.second_order_oml.controller
    fast_initial = source.second_order_oml.fast_initial_weight.detach().clone()
    source_auxiliary = source.second_order_oml.source_auxiliary_digest
    _configure_frozen_controller(controller)
    controller_digest = _controller_digest(controller)

    # The V20 loader restores V20's RNG.  V22 deliberately resets only now.
    torch.manual_seed(ANML_INITIALIZATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ANML_INITIALIZATION_SEED)
    second_gate = ANMLNeuromodulator(zero_final=True).to(
        device=selected, dtype=torch.float32
    )
    first_gate = copy.deepcopy(second_gate)
    random_gate = _seeded_gate(
        RANDOM_GATE_SEED, device=selected, zero_final=False
    )
    random_gate.eval()
    for parameter in random_gate.parameters():
        parameter.requires_grad_(False)
    cpu_state = torch.get_rng_state()
    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(PERMUTATION_SEED)
        permutation = torch.randperm(GATE_WIDTH, generator=generator).to(selected)
    finally:
        torch.set_rng_state(cpu_state)
    result = ANMLSystem(
        controller=controller,
        fast_initial_weight=fast_initial,
        second_order_anml=ANMLArm(
            ARM_SECOND_ORDER, second_gate, _make_outer_optimizer(second_gate)
        ),
        first_order_gate=ANMLArm(
            ARM_FIRST_ORDER, first_gate, _make_outer_optimizer(first_gate)
        ),
        random_gate=random_gate,
        gate_permutation=permutation,
        source_controller_digest=controller_digest,
        source_v20_system_digest=source_digest,
        source_checkpoint_sha256=V20_CHECKPOINT_SHA256,
        source_auxiliary_digest=source_auxiliary,
    )
    if _gate_digest(second_gate) != _gate_digest(first_gate):
        raise RuntimeError("V22 paired gates did not start byte-identical")
    _assert_system_integrity(result)
    return result


def mark_feature_equivalence_verified(
    system: ANMLSystem,
    reports: Sequence[Mapping[str, object]],
) -> None:
    _assert_system_integrity(system)
    if len(reports) < 2 or not all(report.get("passed") is True for report in reports):
        raise RuntimeError("V22 requires ordinary and duplicate feature parity reports")
    system.feature_equivalence_verified = True
    _assert_system_integrity(system)


def _fresh_fast_state_tensors(
    initial_weight: torch.Tensor,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...]]:
    _require_finite_tensor("initial fast weight", initial_weight)
    fast = initial_weight.detach().clone().requires_grad_(True)
    zero = torch.zeros_like(fast)
    return fast, (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _meta_unroll(
    system: ANMLSystem,
    arm: ANMLArm,
    streams: Sequence[object],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], dict[str, object]]:
    if len(streams) != INNER_STEPS:
        raise ValueError("V22 meta-unroll requires eight streams")
    fast, state = _fresh_fast_state_tensors(system.fast_initial_weight)
    diagnostics = []
    for index, stream in enumerate(streams):
        loss = _stream_loss(system.controller, fast, stream, arm.gate)
        gradient = torch.autograd.grad(
            loss,
            fast,
            create_graph=second_order,
            retain_graph=second_order,
            allow_unused=False,
        )[0]
        _require_finite_tensor("meta inner fast gradient", gradient)
        used = gradient if second_order else gradient.detach()
        (fast,), state = functional_adamw_step(
            (fast,),
            (used,),
            state,
            (INNER_LEARNING_RATE,),
            beta1=ADAM_BETA1,
            beta2=ADAM_BETA2,
            epsilon=ADAM_EPSILON,
            weight_decay=ADAM_WEIGHT_DECAY,
        )
        diagnostics.append(
            {
                "step": index + 1,
                "loss": float(loss.detach().item()),
                "gradient_norm": float(gradient.detach().to(torch.float64).norm().item()),
                "gradient_detached": not second_order,
                "fast_identity_path_preserved": fast.grad_fn is not None,
            }
        )
    return fast, state, {
        "steps": INNER_STEPS,
        "second_order": second_order,
        "step_diagnostics": tuple(diagnostics),
        "terminal_fast_norm": float(fast.detach().to(torch.float64).norm().item()),
        "terminal_moment_step": state[0].step,
    }


def _meta_outer_gradients(
    system: ANMLSystem,
    arm: ANMLArm,
    inner_streams: Sequence[object],
    outer_streams: Sequence[object],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
    if len(outer_streams) != OUTER_STREAMS:
        raise ValueError("V22 meta outer objective requires eight streams")
    fast, state, inner = _meta_unroll(
        system, arm, inner_streams, second_order=second_order
    )
    losses = torch.stack(
        tuple(
            _stream_loss(system.controller, fast, stream, arm.gate)
            for stream in outer_streams
        )
    )
    objective = v20._anonymous_entropic_objective(losses, OUTER_STREAMS)
    parameters = tuple(arm.gate.parameters())
    gradients = torch.autograd.grad(
        objective,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    for gradient in gradients:
        _require_finite_tensor("gate outer gradient", gradient)
    return objective.detach(), tuple(gradient.detach() for gradient in gradients), {
        "objective": float(objective.detach().item()),
        "outer_stream_losses": tuple(float(value) for value in losses.detach().tolist()),
        "inner": inner,
        "terminal_fast_step": state[0].step,
    }


def _apply_gate_step(arm: ANMLArm, gradients: Sequence[torch.Tensor]) -> dict[str, object]:
    parameters = tuple(arm.gate.parameters())
    if len(parameters) != len(gradients):
        raise RuntimeError("V22 gate gradient alignment changed")
    arm.outer_optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient.shape != parameter.shape or gradient.device != parameter.device:
            raise RuntimeError("V22 gate gradient shape or device changed")
        parameter.grad = gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(parameters, OUTER_GRADIENT_CLIP)
    _require_finite_tensor("gate gradient norm", norm)
    arm.outer_optimizer.step()
    arm.outer_optimizer.zero_grad(set_to_none=True)
    arm.meta_updates += 1
    for parameter in parameters:
        _require_finite_tensor("updated gate parameter", parameter)
    return {
        "gradient_norm_before_clip": float(norm.detach().item()),
        "gradient_clip": OUTER_GRADIENT_CLIP,
        "meta_update": arm.meta_updates,
        "gate_digest": _gate_digest(arm.gate),
    }


def fit_anml_update(
    system: ANMLSystem,
    update_index: int | None = None,
    *,
    streams: Mapping[str, Sequence[object]] | None = None,
    use_feature_cache: bool = True,
) -> dict[str, object]:
    _assert_system_integrity(system)
    update = system.completed_meta_updates if update_index is None else update_index
    if update != system.completed_meta_updates or not 0 <= update < OUTER_UPDATES:
        raise RuntimeError("V22 meta update does not continue the active identity")
    built = build_meta_fit_streams(update) if streams is None else dict(streams)
    inner_raw = tuple(built.get("inner", ()))
    outer_raw = tuple(built.get("outer", ()))
    if len(inner_raw) != INNER_STEPS or len(outer_raw) != OUTER_STREAMS:
        raise ValueError("V22 meta update lost its 8+8 streams")
    if use_feature_cache:
        if not system.feature_equivalence_verified:
            raise RuntimeError("V22 shared-feature execution lacks preclaim parity evidence")
        all_bundles = tuple(
            capture_feature_bundle(system.controller, system.fast_initial_weight, stream)
            for stream in inner_raw + outer_raw
        )
        inner: tuple[object, ...] = all_bundles[:INNER_STEPS]
        outer: tuple[object, ...] = all_bundles[INNER_STEPS:]
    else:
        inner = inner_raw
        outer = outer_raw
    controller_before = _controller_digest(system.controller)
    second_objective, second_gradients, second_diagnostic = _meta_outer_gradients(
        system,
        system.second_order_anml,
        inner,
        outer,
        second_order=True,
    )
    first_objective, first_gradients, first_diagnostic = _meta_outer_gradients(
        system,
        system.first_order_gate,
        inner,
        outer,
        second_order=False,
    )
    before_equal = (
        second_diagnostic["outer_stream_losses"]
        == first_diagnostic["outer_stream_losses"]
        and float(second_objective.item()) == float(first_objective.item())
    )
    if update == 0 and not before_equal:
        raise RuntimeError("V22 paired gates lost numeric equality before update zero")
    second_step = _apply_gate_step(system.second_order_anml, second_gradients)
    first_step = _apply_gate_step(system.first_order_gate, first_gradients)
    system.completed_meta_updates += 1
    if _controller_digest(system.controller) != controller_before:
        raise RuntimeError("V22 meta-fit changed the frozen controller")
    _assert_system_integrity(system)
    allocated = _allocated_bytes(system.fast_initial_weight.device)
    if allocated > ALLOCATED_MEMORY_CEILING_BYTES:
        raise RuntimeError("V22 exceeded the 12-GiB allocation ceiling")
    return {
        "update": update,
        "completed_meta_updates": system.completed_meta_updates,
        "unique_streams": 16,
        "public_rows": 64,
        "shared_feature_path": use_feature_cache,
        "paired_forward_equal_before_owner_step": before_equal,
        ARM_SECOND_ORDER: {**second_diagnostic, "owner_step": second_step},
        ARM_FIRST_ORDER: {**first_diagnostic, "owner_step": first_step},
        "controller_digest": controller_before,
        "allocated_bytes": allocated,
        "system_digest": anml_system_digest(system),
    }


def fit_anml(
    system: ANMLSystem,
    *,
    progress_callback: Callable[[ANMLSystem, dict[str, object]], None] | None = None,
    deadline_callback: Callable[[], None] | None = None,
    wall_time_limit_seconds: float = SEMANTIC_WALL_TIME_CEILING_SECONDS,
    use_feature_cache: bool = True,
) -> dict[str, object]:
    _assert_system_integrity(system)
    if (
        not math.isfinite(float(wall_time_limit_seconds))
        or wall_time_limit_seconds <= 0.0
        or wall_time_limit_seconds > SEMANTIC_WALL_TIME_CEILING_SECONDS
    ):
        raise ValueError("V22 wall-time limit is invalid")
    device = system.fast_initial_weight.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V22 semantic meta-fit requires CUDA")
    if system.fast_initial_weight.dtype != torch.float32:
        raise RuntimeError("V22 semantic meta-fit requires FP32")
    configure_anml_numerics(device)
    started = time.monotonic()
    start_update = system.completed_meta_updates
    diagnostics = []
    torch.cuda.reset_peak_memory_stats(0 if device.index is None else device.index)
    for update in range(start_update, OUTER_UPDATES):
        if deadline_callback is not None:
            deadline_callback()
        if time.monotonic() - started >= wall_time_limit_seconds:
            raise RuntimeError("V22 reached its cumulative wall-time ceiling")
        diagnostic = fit_anml_update(
            system, update, use_feature_cache=use_feature_cache
        )
        diagnostics.append(diagnostic)
        if (
            progress_callback is not None
            and system.completed_meta_updates % PROGRESS_META_INTERVAL == 0
        ):
            progress_callback(
                system,
                {
                    "phase": "meta_fit",
                    "completed_meta_updates": system.completed_meta_updates,
                    "elapsed_seconds": time.monotonic() - started,
                    "allocated_bytes": _allocated_bytes(device),
                    "system_digest": anml_system_digest(system),
                },
            )
    if system.completed_meta_updates != OUTER_UPDATES:
        raise RuntimeError("V22 meta-fit did not complete")
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": anml_plan_digest(),
        "start_update": start_update,
        "terminal_update": system.completed_meta_updates,
        "paired_gate_updates": OUTER_UPDATES - start_update,
        "unique_stream_uses": 16 * (OUTER_UPDATES - start_update),
        "elapsed_seconds": time.monotonic() - started,
        "maximum_allocated_bytes": _allocated_bytes(device),
        "update_diagnostics": tuple(diagnostics),
        "controller_digest": system.source_controller_digest,
        "system_digest": anml_system_digest(system),
    }


@dataclass(frozen=True, slots=True)
class ANMLFastState:
    """Constant-size deployment state: 64 weights and two Adam moments."""

    weight: torch.Tensor
    optimizer_state: tuple[AdamWSlot, ...]

    def __post_init__(self) -> None:
        _validate_fast_state(self)


def _validate_fast_state(state: ANMLFastState) -> None:
    if not isinstance(state, ANMLFastState):
        raise TypeError("V22 fast state has the wrong type")
    if len(state.optimizer_state) != 1:
        raise ValueError("V22 fast state requires one Adam slot")
    slot = state.optimizer_state[0]
    tensors = (state.weight, slot.exp_avg, slot.exp_avg_sq)
    if (
        state.weight.shape != (1, GATE_WIDTH)
        or any(value.shape != state.weight.shape for value in tensors[1:])
        or any(value.dtype != torch.float32 for value in tensors)
        or any(value.device != state.weight.device for value in tensors)
        or any(value.requires_grad for value in tensors)
        or any(not bool(torch.isfinite(value).all().item()) for value in tensors)
        or type(slot.step) is not int
        or slot.step < 0
    ):
        raise ValueError("V22 fast state is not an aligned finite FP32 state")
    if sum(value.numel() for value in tensors) != FAST_STATE_VALUE_COUNT:
        raise RuntimeError("V22 fast state value count changed")
    if sum(value.numel() * value.element_size() for value in tensors) != FAST_STATE_BYTES_FP32:
        raise RuntimeError("V22 fast state byte count changed")


def fresh_fast_state(initial_weight: torch.Tensor) -> ANMLFastState:
    if initial_weight.shape != (1, GATE_WIDTH) or initial_weight.dtype != torch.float32:
        raise ValueError("V22 initial fast weight must be FP32 [1,64]")
    _require_finite_tensor("fresh fast weight", initial_weight)
    weight = initial_weight.detach().clone()
    zero = torch.zeros_like(weight)
    return ANMLFastState(
        weight=weight,
        optimizer_state=(
            AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),
        ),
    )


def fast_state_digest(state: ANMLFastState) -> str:
    _validate_fast_state(state)
    slot = state.optimizer_state[0]
    return _object_digest(
        _FAST_DIGEST_DOMAIN,
        {
            "weight": state.weight,
            "optimizer_state": {
                "step": slot.step,
                "exp_avg": slot.exp_avg,
                "exp_avg_sq": slot.exp_avg_sq,
            },
        },
    )


def snapshot_fast_state(state: ANMLFastState) -> dict[str, object]:
    _validate_fast_state(state)
    slot = state.optimizer_state[0]
    payload: dict[str, object] = {
        "version": "angler.anml-fast-state.v1",
        "weight": state.weight.detach().cpu().clone(),
        "optimizer_state": (
            {
                "step": slot.step,
                "exp_avg": slot.exp_avg.detach().cpu().clone(),
                "exp_avg_sq": slot.exp_avg_sq.detach().cpu().clone(),
            },
        ),
    }
    payload["digest"] = fast_state_digest(state)
    return payload


def restore_fast_state(
    snapshot: Mapping[str, object],
    device: torch.device | str = "cpu",
) -> ANMLFastState:
    expected = {"version", "weight", "optimizer_state", "digest"}
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected:
        raise ValueError("V22 fast-state snapshot fields are invalid")
    if snapshot["version"] != "angler.anml-fast-state.v1":
        raise ValueError("V22 fast-state snapshot version changed")
    records = snapshot["optimizer_state"]
    if not isinstance(records, (tuple, list)) or len(records) != 1:
        raise ValueError("V22 fast-state snapshot lost its Adam slot")
    record = records[0]
    if not isinstance(record, Mapping) or set(record) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("V22 fast-state Adam fields are invalid")
    selected = torch.device(device)
    weight = snapshot["weight"]
    if not isinstance(weight, torch.Tensor):
        raise ValueError("V22 fast-state weight is not a tensor")
    try:
        result = ANMLFastState(
            weight=weight.detach().to(selected).clone(),
            optimizer_state=(
                AdamWSlot(
                    step=int(record["step"]),
                    exp_avg=record["exp_avg"].detach().to(selected).clone(),
                    exp_avg_sq=record["exp_avg_sq"].detach().to(selected).clone(),
                ),
            ),
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("V22 fast-state snapshot tensors are invalid") from error
    if fast_state_digest(result) != snapshot["digest"]:
        raise RuntimeError("V22 fast-state snapshot digest changed")
    return result


anml_fast_state_digest = fast_state_digest
snapshot_anml_fast_state = snapshot_fast_state
restore_anml_fast_state = restore_fast_state


def _online_fast_step(
    system: ANMLSystem,
    state: ANMLFastState,
    stream_or_bundle: object,
    gate_module: ANMLNeuromodulator | None,
    *,
    lesion: GateMode,
    permutation: torch.Tensor | Sequence[int] | None = None,
) -> tuple[ANMLFastState, dict[str, object]]:
    _validate_fast_state(state)
    weight = state.weight.detach().clone().requires_grad_(True)
    loss = _stream_loss(
        system.controller,
        weight,
        stream_or_bundle,
        gate_module,
        lesion=lesion,
        permutation=permutation,
    )
    gradient = torch.autograd.grad(
        loss, weight, create_graph=False, retain_graph=False, allow_unused=False
    )[0]
    _require_finite_tensor("lifetime fast gradient", gradient)
    (updated,), slots = functional_adamw_step(
        (weight,),
        (gradient.detach(),),
        state.optimizer_state,
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    next_state = ANMLFastState(
        weight=updated.detach(),
        optimizer_state=tuple(
            AdamWSlot(
                step=slot.step,
                exp_avg=slot.exp_avg.detach(),
                exp_avg_sq=slot.exp_avg_sq.detach(),
            )
            for slot in slots
        ),
    )
    return next_state, {
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient.detach().to(torch.float64).norm().item()),
        "fast_step": next_state.optimizer_state[0].step,
        "fast_state_digest": fast_state_digest(next_state),
    }


_CLASSIFICATION_GATE_NAMES = {
    "second_beats_first_auc",
    "second_beats_open_auc",
    "second_beats_first_terminal",
    "second_beats_open_terminal",
    "second_beats_forward_only",
    "second_beats_mean_gate",
    "second_beats_permuted_gate",
    "panel_direction_supported",
    "no_catastrophic_panel",
    "terminal_supported",
    "retention_supported",
    "original_nonregression",
    "fully_heldout_improved",
    "reset_attribution_supported",
    "surface_transfer_supported",
    "early_improvement",
    "acquisition_improved",
}


def classify_anml(
    gates: Mapping[str, bool],
    mechanical_validity: bool,
) -> str:
    """Apply the frozen exclusive interpretation to precomputed booleans."""

    if type(mechanical_validity) is not bool:
        raise ValueError("V22 mechanical validity must be bool")
    if not isinstance(gates, Mapping) or set(gates) != _CLASSIFICATION_GATE_NAMES:
        raise ValueError("V22 classification gate fields are incomplete")
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("V22 classification gates must be bool")
    if not mechanical_validity:
        return "INVALID_NO_CLAIM"
    ordinary_credit = gates["second_beats_first_auc"] and gates["second_beats_open_auc"]
    if ordinary_credit and not gates["second_beats_forward_only"]:
        return "STATIC_GATE_ONLY"
    if ordinary_credit and (
        not gates["second_beats_mean_gate"]
        or not gates["second_beats_permuted_gate"]
    ):
        return "SELECTIVITY_ATTRIBUTION_NOT_SUPPORTED"
    if gates["early_improvement"] and not gates["terminal_supported"]:
        return "SHORT_HORIZON_ONLY"
    retention_group = (
        gates["retention_supported"]
        and gates["original_nonregression"]
        and gates["fully_heldout_improved"]
        and gates["reset_attribution_supported"]
        and gates["surface_transfer_supported"]
    )
    if gates["acquisition_improved"] and not retention_group:
        return "ACQUISITION_RETENTION_TRADEOFF"
    required = (
        ordinary_credit
        and gates["second_beats_first_terminal"]
        and gates["second_beats_open_terminal"]
        and gates["second_beats_forward_only"]
        and gates["second_beats_mean_gate"]
        and gates["second_beats_permuted_gate"]
        and gates["panel_direction_supported"]
        and gates["no_catastrophic_panel"]
        and gates["terminal_supported"]
        and retention_group
    )
    return (
        "ANML_SELECTIVE_PLASTICITY_SUPPORTED"
        if required
        else "ANML_NOT_SUPPORTED"
    )


_classify_anml = classify_anml


def _finite_metric(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V22 metric is non-finite: {label}")
    return result


def _loss_improves(candidate: float, baseline: float, fraction: float) -> bool:
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("V22 loss comparisons require non-negative losses")
    return candidate <= (1.0 - fraction) * baseline


def compute_anml_gates(metrics: Mapping[str, object]) -> dict[str, bool]:
    """Convert primitive recorded losses/ratios into frozen gate booleans.

    Expected fields are the direct output shape of :func:`evaluate_anml`:
    ``aggregate.arms``, ``aggregate`` retention/causality primitives, and four
    panel summaries.  Thresholds are inclusive and never selected from values.
    """

    try:
        aggregate = metrics["aggregate"]
        arms = aggregate["arms"]
        second = arms[ARM_SECOND_ORDER]
        first = arms[ARM_FIRST_ORDER]
        opened = arms[ARM_ALWAYS_OPEN]
        forward = arms[ARM_FORWARD_ONLY]
        mean = arms[ARM_MEAN_GATE]
        permuted = arms[ARM_PERMUTED_GATE]
        second_auc = _finite_metric(second["loss_auc"], "second AUC")
        first_auc = _finite_metric(first["loss_auc"], "first AUC")
        open_auc = _finite_metric(opened["loss_auc"], "open AUC")
        forward_auc = _finite_metric(forward["loss_auc"], "forward AUC")
        mean_auc = _finite_metric(mean["loss_auc"], "mean AUC")
        permuted_auc = _finite_metric(permuted["loss_auc"], "permuted AUC")
        second_terminal = _finite_metric(second["terminal_loss"], "second terminal")
        first_terminal = _finite_metric(first["terminal_loss"], "first terminal")
        open_terminal = _finite_metric(opened["terminal_loss"], "open terminal")
        panels = tuple(metrics["panels"])
        if len(panels) != LIFETIME_PANELS:
            raise ValueError("V22 classification requires four panels")
        credit_direction_count = 0
        selectivity_direction_count = 0
        no_catastrophic = True
        early = False
        for panel in panels:
            panel_arms = panel["arms"]
            panel_second = _finite_metric(
                panel_arms[ARM_SECOND_ORDER]["loss_auc"], "panel second AUC"
            )
            comparisons = tuple(
                _finite_metric(panel_arms[name]["loss_auc"], f"panel {name} AUC")
                for name in (
                    ARM_FIRST_ORDER,
                    ARM_ALWAYS_OPEN,
                    ARM_FORWARD_ONLY,
                    ARM_MEAN_GATE,
                    ARM_PERMUTED_GATE,
                )
            )
            credit_direction_count += int(
                panel_second < comparisons[0] and panel_second < comparisons[1]
            )
            selectivity_direction_count += int(
                panel_second < comparisons[2]
                and panel_second < comparisons[3]
                and panel_second < comparisons[4]
            )
            no_catastrophic = no_catastrophic and all(
                panel_second <= 1.05 * value for value in comparisons[:2]
            )
            milestones = panel["probe_milestones"]
            for milestone in (512, 2_048):
                if str(milestone) in milestones:
                    record = milestones[str(milestone)]["arms"]
                    early_second = _finite_metric(
                        record[ARM_SECOND_ORDER]["mean_loss"], "early second loss"
                    )
                    early = early or (
                        early_second
                        < _finite_metric(record[ARM_FIRST_ORDER]["mean_loss"], "early first")
                        and early_second
                        < _finite_metric(record[ARM_ALWAYS_OPEN]["mean_loss"], "early open")
                    )
        retention = _finite_metric(
            aggregate["retained_acquisition_fraction"], "retention fraction"
        )
        original_pre = _finite_metric(aggregate["original_pre_loss"], "original pre")
        original_terminal = _finite_metric(
            aggregate["original_terminal_loss"], "original terminal"
        )
        heldout_pre = _finite_metric(
            aggregate["fully_heldout_pre_loss"], "heldout pre"
        )
        heldout_terminal = _finite_metric(
            aggregate["fully_heldout_terminal_loss"], "heldout terminal"
        )
        reset = _finite_metric(
            aggregate["reset_removed_fraction"], "reset attribution"
        )
        transfer = _finite_metric(
            aggregate["surface_transfer_retained_fraction"], "surface transfer"
        )
        unseen_pre = _finite_metric(aggregate["unseen_pre_loss"], "unseen pre")
        unseen_terminal = _finite_metric(
            aggregate["unseen_terminal_loss"], "unseen terminal"
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        raise ValueError("V22 classification metrics are incomplete") from error
    second_beats_first_terminal = _loss_improves(second_terminal, first_terminal, 0.05)
    second_beats_open_terminal = _loss_improves(second_terminal, open_terminal, 0.05)
    gates = {
        "second_beats_first_auc": _loss_improves(second_auc, first_auc, 0.05),
        "second_beats_open_auc": _loss_improves(second_auc, open_auc, 0.05),
        "second_beats_first_terminal": second_beats_first_terminal,
        "second_beats_open_terminal": second_beats_open_terminal,
        "second_beats_forward_only": _loss_improves(second_auc, forward_auc, 0.05),
        "second_beats_mean_gate": _loss_improves(second_auc, mean_auc, 0.03),
        "second_beats_permuted_gate": _loss_improves(second_auc, permuted_auc, 0.03),
        "panel_direction_supported": (
            credit_direction_count >= 3 and selectivity_direction_count >= 3
        ),
        "no_catastrophic_panel": no_catastrophic,
        "terminal_supported": second_beats_first_terminal and second_beats_open_terminal,
        "retention_supported": retention >= 0.80,
        "original_nonregression": original_terminal <= 1.05 * original_pre,
        "fully_heldout_improved": heldout_terminal < heldout_pre,
        "reset_attribution_supported": reset >= 0.80,
        "surface_transfer_supported": transfer >= 0.80,
        "early_improvement": early or second_auc < first_auc or second_auc < open_auc,
        "acquisition_improved": unseen_terminal < unseen_pre,
    }
    if set(gates) != _CLASSIFICATION_GATE_NAMES:
        raise AssertionError("V22 computed classification gates changed")
    return gates


def _validate_json_value(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("V22 JSON output contains a non-finite float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("V22 JSON output keys must be strings")
            _validate_json_value(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_json_value(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"V22 JSON output type is unsupported: {type(value).__name__}")


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    _validate_json_value(payload)
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise RuntimeError("V22 atomic JSON target or temporary already exists")
    try:
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_torch_save(path: str | Path, payload: object) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise RuntimeError("V22 atomic checkpoint target or temporary already exists")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


_atomic_write_json = atomic_write_json
_atomic_torch_save = atomic_torch_save


def _checkpoint_payload(
    system: ANMLSystem,
    harness_state: Mapping[str, object] | None,
) -> dict[str, object]:
    _assert_system_integrity(system)
    selected_harness = (
        copy.deepcopy(system.harness_state)
        if harness_state is None
        else copy.deepcopy(dict(harness_state))
    )
    if not isinstance(selected_harness, dict):
        raise ValueError("V22 checkpoint harness state is invalid")
    cuda_rng = (
        tuple(torch.cuda.get_rng_state(index).cpu() for index in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else ()
    )
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": anml_plan_digest(),
        "source_bindings": {
            "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
            "v19_source_checkpoint_sha256": V19_SOURCE_CHECKPOINT_SHA256,
            "v20_terminal_system_digest": V20_TERMINAL_SYSTEM_DIGEST,
            "source_controller_digest": system.source_controller_digest,
            "source_auxiliary_digest": system.source_auxiliary_digest,
        },
        "completed_meta_updates": system.completed_meta_updates,
        "feature_equivalence_verified": system.feature_equivalence_verified,
        "arms": {
            ARM_SECOND_ORDER: {
                "gate_state": {
                    name: value.detach().cpu().clone()
                    for name, value in system.second_order_anml.gate.state_dict().items()
                },
                "optimizer_state": copy.deepcopy(
                    system.second_order_anml.outer_optimizer.state_dict()
                ),
                "meta_updates": system.second_order_anml.meta_updates,
            },
            ARM_FIRST_ORDER: {
                "gate_state": {
                    name: value.detach().cpu().clone()
                    for name, value in system.first_order_gate.gate.state_dict().items()
                },
                "optimizer_state": copy.deepcopy(
                    system.first_order_gate.outer_optimizer.state_dict()
                ),
                "meta_updates": system.first_order_gate.meta_updates,
            },
        },
        "random_gate_state": {
            name: value.detach().cpu().clone()
            for name, value in system.random_gate.state_dict().items()
        },
        "gate_permutation": system.gate_permutation.detach().cpu().clone(),
        "harness_state": selected_harness,
        "harness_state_digest": _object_digest(_OBJECT_DIGEST_DOMAIN, selected_harness),
        "cpu_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_states": cuda_rng,
        "system_digest": anml_system_digest(system),
    }
    payload["checkpoint_digest"] = _object_digest(_OBJECT_DIGEST_DOMAIN, payload)
    return payload


def save_anml_checkpoint(
    path: str | Path,
    system: ANMLSystem,
    harness_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write exactly one checkpoint path; the harness owns outer replacement."""

    if harness_state is not None:
        system.harness_state = copy.deepcopy(dict(harness_state))
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"V22 checkpoint target already exists: {target}")
    torch.save(_checkpoint_payload(system, harness_state), target)
    size = target.stat().st_size
    if size > 16 * 1024**2:
        raise RuntimeError("V22 checkpoint exceeds the 16-MiB ceiling")
    return {
        "path": str(target),
        "bytes": size,
        "sha256": _sha256_file(target),
        "completed_meta_updates": system.completed_meta_updates,
        "system_digest": anml_system_digest(system),
        "harness_state": copy.deepcopy(system.harness_state),
    }


def load_anml_checkpoint(
    path: str | Path,
    v20_checkpoint: str | Path = V20_CHECKPOINT_PATH,
    v19_source_checkpoint: str | Path = V19_SOURCE_CHECKPOINT_PATH,
    *,
    device: torch.device | str = "cpu",
    verify_hashes: bool = True,
) -> ANMLSystem:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected = {
        "version",
        "protocol_id",
        "plan_digest",
        "source_bindings",
        "completed_meta_updates",
        "feature_equivalence_verified",
        "arms",
        "random_gate_state",
        "gate_permutation",
        "harness_state",
        "harness_state_digest",
        "cpu_rng_state",
        "cuda_rng_states",
        "system_digest",
        "checkpoint_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError("V22 checkpoint fields are invalid")
    stored_checkpoint_digest = payload.pop("checkpoint_digest")
    if _object_digest(_OBJECT_DIGEST_DOMAIN, payload) != stored_checkpoint_digest:
        raise RuntimeError("V22 checkpoint content digest changed")
    payload["checkpoint_digest"] = stored_checkpoint_digest
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"] != anml_plan_digest()
    ):
        raise RuntimeError("V22 checkpoint identity changed")
    bindings = payload["source_bindings"]
    if not isinstance(bindings, Mapping) or bindings.get("v20_checkpoint_sha256") != V20_CHECKPOINT_SHA256 or bindings.get("v19_source_checkpoint_sha256") != V19_SOURCE_CHECKPOINT_SHA256 or bindings.get("v20_terminal_system_digest") != V20_TERMINAL_SYSTEM_DIGEST:
        raise RuntimeError("V22 checkpoint source binding changed")
    system = build_anml_system(
        v20_checkpoint,
        v19_source_checkpoint,
        device=device,
        verify_hashes=verify_hashes,
    )
    if (
        bindings.get("source_controller_digest") != system.source_controller_digest
        or bindings.get("source_auxiliary_digest") != system.source_auxiliary_digest
    ):
        raise RuntimeError("V22 checkpoint frozen source digest changed")
    completed = payload["completed_meta_updates"]
    if type(completed) is not int or not 0 <= completed <= OUTER_UPDATES:
        raise RuntimeError("V22 checkpoint meta-update count is invalid")
    arms = payload["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(LEARNED_ARMS):
        raise RuntimeError("V22 checkpoint arm fields are invalid")
    for name in LEARNED_ARMS:
        record = arms[name]
        if not isinstance(record, Mapping) or set(record) != {
            "gate_state", "optimizer_state", "meta_updates"
        }:
            raise RuntimeError("V22 checkpoint learned arm is invalid")
        arm = system.arm(name)
        arm.gate.load_state_dict(record["gate_state"], strict=True)
        arm.outer_optimizer.load_state_dict(record["optimizer_state"])
        arm.meta_updates = int(record["meta_updates"])
        if arm.meta_updates != completed:
            raise RuntimeError("V22 checkpoint paired counters diverged")
    system.random_gate.load_state_dict(payload["random_gate_state"], strict=True)
    for parameter in system.random_gate.parameters():
        parameter.requires_grad_(False)
    permutation = payload["gate_permutation"]
    if not isinstance(permutation, torch.Tensor):
        raise RuntimeError("V22 checkpoint permutation is invalid")
    system.gate_permutation = permutation.to(device=device, dtype=torch.long).clone()
    system.completed_meta_updates = completed
    feature_verified = payload["feature_equivalence_verified"]
    if type(feature_verified) is not bool:
        raise RuntimeError("V22 checkpoint parity state is invalid")
    system.feature_equivalence_verified = feature_verified
    harness = payload["harness_state"]
    if not isinstance(harness, dict):
        raise RuntimeError("V22 checkpoint harness state is invalid")
    if _object_digest(_OBJECT_DIGEST_DOMAIN, harness) != payload["harness_state_digest"]:
        raise RuntimeError("V22 checkpoint harness-state digest changed")
    system.harness_state = copy.deepcopy(harness)
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    cuda_states = tuple(payload["cuda_rng_states"])
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or len(cuda_states) != 1:
            raise RuntimeError("V22 checkpoint CUDA RNG topology changed")
        torch.cuda.set_rng_state(cuda_states[0].cpu(), 0)
    elif len(cuda_states) not in (0, 1):
        raise RuntimeError("V22 checkpoint CUDA RNG topology changed")
    _assert_system_integrity(system)
    if anml_system_digest(system) != payload["system_digest"]:
        raise RuntimeError("V22 checkpoint system digest changed")
    return system


def anml_checkpoint_summary(system: ANMLSystem) -> dict[str, object]:
    _assert_system_integrity(system)
    return {
        "completed_meta_updates": system.completed_meta_updates,
        "meta_fit_complete": system.completed_meta_updates == OUTER_UPDATES,
        "feature_equivalence_verified": system.feature_equivalence_verified,
        "controller_digest": system.source_controller_digest,
        "second_gate_digest": _gate_digest(system.second_order_anml.gate),
        "first_gate_digest": _gate_digest(system.first_order_gate.gate),
        "random_gate_digest": _gate_digest(system.random_gate),
        "harness_state": copy.deepcopy(system.harness_state),
        "harness_state_digest": _object_digest(_OBJECT_DIGEST_DOMAIN, system.harness_state),
        "system_digest": anml_system_digest(system),
    }


def synthetic_cuda_preflight(
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    """Exercise only synthetic gate/credit mechanics; construct no public stream."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V22 synthetic preflight requires CUDA")
    configure_anml_numerics(selected)
    index = 0 if selected.index is None else selected.index
    torch.cuda.reset_peak_memory_stats(index)
    gate = _seeded_gate(
        ANML_INITIALIZATION_SEED, device=selected, zero_final=True
    )
    first_gate = copy.deepcopy(gate)
    hidden = torch.linspace(-1.0, 1.0, 8 * GATE_WIDTH, device=selected).reshape(
        8, GATE_WIDTH
    )
    initial_open = torch.equal(centered_gate(gate, hidden), torch.ones_like(hidden))
    initial = torch.linspace(-0.1, 0.1, GATE_WIDTH, device=selected).reshape(1, -1)

    def trajectory(module: ANMLNeuromodulator, second_order: bool):
        fast = initial.detach().clone().requires_grad_(True)
        zero = torch.zeros_like(fast)
        state = (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)
        for step in range(3):
            features = hidden[step : step + 2]
            target = features.new_tensor((0.1 * step, -0.05 * step))
            prediction = F.linear(features * centered_gate(module, features), fast).squeeze(-1)
            loss = (prediction - target).square().mean()
            gradient = torch.autograd.grad(loss, fast, create_graph=second_order)[0]
            (fast,), state = functional_adamw_step(
                (fast,),
                (gradient if second_order else gradient.detach(),),
                state,
                (INNER_LEARNING_RATE,),
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                epsilon=ADAM_EPSILON,
                weight_decay=ADAM_WEIGHT_DECAY,
            )
        outer_features = hidden[6:]
        outer = F.linear(
            outer_features * centered_gate(module, outer_features), fast
        ).square().mean()
        gradients = torch.autograd.grad(outer, tuple(module.parameters()))
        flat = torch.cat(tuple(value.reshape(-1) for value in gradients))
        return fast.detach(), state, outer.detach(), flat.detach()

    second = trajectory(gate, True)
    first = trajectory(first_gate, False)
    if not torch.equal(second[0], first[0]) or not torch.equal(
        second[1][0].exp_avg.detach(), first[1][0].exp_avg.detach()
    ):
        raise RuntimeError("V22 synthetic paired forwards diverged")
    if not bool(torch.isfinite(second[3]).all().item()) or not bool(
        torch.isfinite(first[3]).all().item()
    ):
        raise RuntimeError("V22 synthetic meta-credit is non-finite")
    credit_delta = float((second[3] - first[3]).abs().amax().item())
    if credit_delta <= 0.0:
        raise RuntimeError("V22 synthetic detach did not isolate update consequence")
    permutation = torch.arange(GATE_WIDTH - 1, -1, -1, device=selected)
    live = centered_gate(gate, hidden)
    lesions = {
        mode: apply_gate_lesion(live, mode, permutation=permutation)
        for mode in ("live", "open", "mean", "permuted")
    }
    for value in lesions.values():
        _require_finite_tensor("synthetic gate lesion", value)
    maximum = _allocated_bytes(selected)
    if maximum > ALLOCATED_MEMORY_CEILING_BYTES:
        raise RuntimeError("V22 synthetic preflight exceeded its allocation ceiling")
    return {
        "status": "PASS",
        "synthetic_only": True,
        "semantic_streams_generated": 0,
        "semantic_updates_performed": False,
        "device": str(selected),
        "dtype": "torch.float32",
        "initial_gate_exactly_open": initial_open,
        "gate_parameter_count": sum(parameter.numel() for parameter in gate.parameters()),
        "paired_fast_forward_exact": True,
        "second_first_meta_gradient_max_abs_delta": credit_delta,
        "lesions_finite": True,
        "maximum_allocated_bytes": maximum,
        "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
    }


def _arm_gate_configuration(
    system: ANMLSystem,
    arm_name: str,
    *,
    for_probe: bool,
) -> tuple[ANMLNeuromodulator | None, GateMode, torch.Tensor | None]:
    if arm_name == ARM_SECOND_ORDER:
        return system.second_order_anml.gate, "live", None
    if arm_name == ARM_FIRST_ORDER:
        return system.first_order_gate.gate, "live", None
    if arm_name == ARM_ALWAYS_OPEN:
        return None, "open", None
    if arm_name == ARM_FORWARD_ONLY:
        return (
            (system.second_order_anml.gate, "live", None)
            if for_probe
            else (None, "open", None)
        )
    if arm_name == ARM_MEAN_GATE:
        return system.second_order_anml.gate, "mean", None
    if arm_name == ARM_PERMUTED_GATE:
        return system.second_order_anml.gate, "permuted", system.gate_permutation
    if arm_name == ARM_RANDOM_GATE:
        return system.random_gate, "live", None
    raise KeyError(f"unknown V22 lifetime arm: {arm_name}")


def _family_losses(member_losses: Sequence[float]) -> dict[str, dict[str, object]]:
    if len(member_losses) != len(PROBE_COMMITMENTS):
        raise ValueError("V22 probe member-loss count changed")
    result = {}
    for family, indices in PROBE_FAMILIES.items():
        values = tuple(float(member_losses[index]) for index in indices)
        if not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError("V22 probe family loss is invalid")
        result[family] = {
            "mean_loss": sum(values) / len(values),
            "member_losses": values,
            "commitment_indices": indices,
        }
    return result


def _gate_statistics_for_bundle(
    bundle: ANMLFeatureBundle,
    second: ANMLNeuromodulator,
    first: ANMLNeuromodulator,
) -> dict[str, object]:
    count = 0
    value_sum = 0.0
    value_square_sum = 0.0
    entropy_sum = 0.0
    coordinate_sum = torch.zeros(GATE_WIDTH, dtype=torch.float64)
    cosine_sum = 0.0
    top_overlap_sum = 0.0
    with torch.no_grad():
        for row in bundle.rows:
            for hidden in (row.positive_hidden, row.negative_hidden):
                second_values = centered_gate(second, hidden).reshape(-1, GATE_WIDTH)
                first_values = centered_gate(first, hidden).reshape(-1, GATE_WIDTH)
                probabilities = second_values / second_values.sum(dim=-1, keepdim=True)
                entropy = -(
                    probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
                ).sum(dim=-1) / math.log(GATE_WIDTH)
                cosine = F.cosine_similarity(second_values, first_values, dim=-1)
                second_top = torch.topk(second_values, GATE_WIDTH // 4, dim=-1).indices
                first_top = torch.topk(first_values, GATE_WIDTH // 4, dim=-1).indices
                second_mask = torch.zeros_like(second_values, dtype=torch.bool).scatter_(
                    -1, second_top, True
                )
                first_mask = torch.zeros_like(first_values, dtype=torch.bool).scatter_(
                    -1, first_top, True
                )
                intersection = (second_mask & first_mask).sum(dim=-1).to(torch.float64)
                union = (second_mask | first_mask).sum(dim=-1).to(torch.float64)
                vector_count = second_values.shape[0]
                count += vector_count
                value_sum += float(second_values.to(torch.float64).sum().item())
                value_square_sum += float(second_values.to(torch.float64).square().sum().item())
                entropy_sum += float(entropy.to(torch.float64).sum().item())
                coordinate_sum += second_values.to(torch.float64).sum(dim=0).cpu()
                cosine_sum += float(cosine.to(torch.float64).sum().item())
                top_overlap_sum += float((intersection / union).sum().item())
    if count <= 0:
        raise RuntimeError("V22 gate statistics saw no hidden vectors")
    scalar_count = count * GATE_WIDTH
    mean = value_sum / scalar_count
    variance = max(0.0, value_square_sum / scalar_count - mean * mean)
    return {
        "hidden_vector_count": count,
        "gate_mean": mean,
        "gate_variance": variance,
        "normalized_entropy": entropy_sum / count,
        "coordinate_means": tuple(float(value / count) for value in coordinate_sum.tolist()),
        "second_first_coordinate_cosine": cosine_sum / count,
        "second_first_top_quartile_jaccard": top_overlap_sum / count,
    }


def _merge_gate_statistics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("V22 gate statistic merge is empty")
    total = sum(int(record["hidden_vector_count"]) for record in records)
    if total <= 0:
        raise RuntimeError("V22 gate statistic count is invalid")
    weighted = lambda name: sum(
        int(record["hidden_vector_count"]) * float(record[name]) for record in records
    ) / total
    global_mean = weighted("gate_mean")
    global_second_moment = sum(
        int(record["hidden_vector_count"])
        * (float(record["gate_variance"]) + float(record["gate_mean"]) ** 2)
        for record in records
    ) / total
    coordinates = tuple(
        sum(
            int(record["hidden_vector_count"]) * float(record["coordinate_means"][index])
            for record in records
        )
        / total
        for index in range(GATE_WIDTH)
    )
    return {
        "hidden_vector_count": total,
        "gate_mean": global_mean,
        "gate_variance": max(0.0, global_second_moment - global_mean**2),
        "normalized_entropy": weighted("normalized_entropy"),
        "coordinate_means": coordinates,
        "second_first_coordinate_cosine": weighted("second_first_coordinate_cosine"),
        "second_first_top_quartile_jaccard": weighted(
            "second_first_top_quartile_jaccard"
        ),
    }


def _score_probe_boundary(
    system: ANMLSystem,
    panel: int,
    states: Mapping[str, ANMLFastState],
    arm_names: Sequence[str],
    *,
    deadline_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    if set(states) != set(arm_names):
        raise ValueError("V22 probe states lost arm alignment")
    controller_before = _controller_digest(system.controller)
    accumulators = {
        name: {
            "member_losses": [],
            "row_losses": [],
            "supported_rows": 0,
            "informative_rows": 0,
            "qualifying_streams": 0,
        }
        for name in arm_names
    }
    static_losses = []
    static_rows = []
    static_supported = 0
    static_informative = 0
    static_qualifying = 0
    gate_statistics = []
    for record in build_probe_streams(panel):
        if deadline_callback is not None:
            deadline_callback()
        stream = _make_stream(record)
        bundle = capture_feature_bundle(
            system.controller, system.fast_initial_weight, stream
        )
        gate_statistics.append(
            _gate_statistics_for_bundle(
                bundle,
                system.second_order_anml.gate,
                system.first_order_gate.gate,
            )
        )
        with torch.no_grad():
            for name in arm_names:
                module, lesion, permutation = _arm_gate_configuration(
                    system, name, for_probe=True
                )
                rows = rows_from_feature_bundle(
                    bundle,
                    states[name].weight,
                    module,
                    lesion=lesion,
                    permutation=permutation,
                )
                row_losses = tuple(float(v20._row_loss(row).item()) for row in rows)
                loss = float(v20._stream_loss_from_rows(rows).item())
                coverage = v19._credit_rows_metrics((rows,))
                accumulator = accumulators[name]
                accumulator["member_losses"].append(loss)
                accumulator["row_losses"].append(row_losses)
                accumulator["supported_rows"] += int(coverage["supported_rows"])
                accumulator["informative_rows"] += int(coverage["informative_rows"])
                accumulator["qualifying_streams"] += int(coverage["qualifying_streams"])
            static_rows_public = rows_from_feature_bundle(
                bundle,
                system.fast_initial_weight,
                system.second_order_anml.gate,
                lesion="live",
            )
            static_row_loss = tuple(
                float(v20._row_loss(row).item()) for row in static_rows_public
            )
            static_loss = float(v20._stream_loss_from_rows(static_rows_public).item())
            static_coverage = v19._credit_rows_metrics((static_rows_public,))
            static_losses.append(static_loss)
            static_rows.append(static_row_loss)
            static_supported += int(static_coverage["supported_rows"])
            static_informative += int(static_coverage["informative_rows"])
            static_qualifying += int(static_coverage["qualifying_streams"])
        del stream, bundle
    arms = {}
    for name, accumulator in accumulators.items():
        member_losses = tuple(float(value) for value in accumulator["member_losses"])
        arms[name] = {
            "mean_loss": sum(member_losses) / len(member_losses),
            "member_losses": member_losses,
            "row_losses": tuple(accumulator["row_losses"]),
            "families": _family_losses(member_losses),
            "supported_rows": int(accumulator["supported_rows"]),
            "informative_rows": int(accumulator["informative_rows"]),
            "qualifying_streams": int(accumulator["qualifying_streams"]),
            "fast_state_digest": fast_state_digest(states[name]),
            "fast_adamw_step": states[name].optimizer_state[0].step,
        }
    static_member_losses = tuple(static_losses)
    static = {
        "mean_loss": sum(static_member_losses) / len(static_member_losses),
        "member_losses": static_member_losses,
        "row_losses": tuple(static_rows),
        "families": _family_losses(static_member_losses),
        "supported_rows": static_supported,
        "informative_rows": static_informative,
        "qualifying_streams": static_qualifying,
        "fast_state_digest": fast_state_digest(fresh_fast_state(system.fast_initial_weight)),
        "fast_adamw_step": 0,
    }
    if _controller_digest(system.controller) != controller_before:
        raise RuntimeError("V22 probe changed the frozen controller")
    if active_gate_hook_count(system.controller) != 0:
        raise RuntimeError("V22 probe leaked a gate hook")
    return {
        "arms": arms,
        "controls": {
            "second_static_no_update": static,
            "boundary_reset": copy.deepcopy(static),
            "reset_static_exact": True,
        },
        "gate_statistics": _merge_gate_statistics(gate_statistics),
        "probe_streams": len(PROBE_COMMITMENTS),
        "controller_digest": controller_before,
    }


def _normalized_trapezoid_auc(
    milestone_losses: Mapping[int, float],
) -> float:
    if tuple(sorted(milestone_losses)) != PROBE_MILESTONES:
        raise ValueError("V22 AUC milestones changed")
    total = 0.0
    for left, right in zip(PROBE_MILESTONES[:-1], PROBE_MILESTONES[1:], strict=True):
        left_loss = _finite_metric(milestone_losses[left], "AUC left loss")
        right_loss = _finite_metric(milestone_losses[right], "AUC right loss")
        total += 0.5 * (left_loss + right_loss) * (right - left)
    result = total / LIFETIME_UPDATES
    if not math.isfinite(result):
        raise RuntimeError("V22 normalized AUC is non-finite")
    return result


def _panel_summary(
    panel: int,
    probe_reports: Mapping[str, Mapping[str, object]],
    online: Mapping[str, Mapping[str, object]],
    arm_names: Sequence[str],
) -> dict[str, object]:
    if set(probe_reports) != {str(value) for value in PROBE_MILESTONES}:
        raise ValueError("V22 panel probe milestones are incomplete")
    arms = {}
    for name in arm_names:
        losses = {
            milestone: float(probe_reports[str(milestone)]["arms"][name]["mean_loss"])
            for milestone in PROBE_MILESTONES
        }
        arms[name] = {
            "loss_auc": _normalized_trapezoid_auc(losses),
            "terminal_loss": losses[LIFETIME_UPDATES],
            "milestone_mean_losses": losses,
            "online_update_loss_mean": float(online[name]["loss_sum"])
            / int(online[name]["count"]),
            "online_update_count": int(online[name]["count"]),
            "terminal_fast_state_digest": probe_reports[str(LIFETIME_UPDATES)]["arms"][name]["fast_state_digest"],
        }
    pre_second = probe_reports["0"]["arms"][ARM_SECOND_ORDER]
    terminal_second = probe_reports[str(LIFETIME_UPDATES)]["arms"][ARM_SECOND_ORDER]
    retained_numerator = 0.0
    retained_denominator = 0.0
    per_commitment = []
    for commitment in LIFETIME_COMMITMENTS:
        pre = float(pre_second["member_losses"][commitment])
        observed = tuple(
            float(probe_reports[str(milestone)]["arms"][ARM_SECOND_ORDER]["member_losses"][commitment])
            for milestone in PROBE_MILESTONES[1:]
        )
        best_gain = max((pre - value for value in observed), default=0.0)
        terminal_gain = pre - observed[-1]
        if best_gain > 0.0:
            retained_denominator += best_gain
            retained_numerator += terminal_gain
        per_commitment.append(
            {
                "commitment_index": commitment,
                "pre_loss": pre,
                "milestone_losses": observed,
                "best_acquisition": best_gain,
                "terminal_acquisition": terminal_gain,
            }
        )
    retained_fraction = (
        max(0.0, retained_numerator) / retained_denominator
        if retained_denominator > 0.0
        else 0.0
    )
    reset_loss = float(
        probe_reports[str(LIFETIME_UPDATES)]["controls"]["boundary_reset"]["mean_loss"]
    )
    live_loss = float(terminal_second["mean_loss"])
    pre_loss = float(pre_second["mean_loss"])
    live_advantage = pre_loss - live_loss
    reset_gain = pre_loss - reset_loss
    reset_removed = (
        (live_advantage - reset_gain) / live_advantage
        if live_advantage > 0.0
        else 0.0
    )
    return {
        "panel": panel,
        "order_kind": "blocked" if panel < 2 else "interleaved",
        "arms": arms,
        "probe_milestones": dict(probe_reports),
        "retention": {
            "numerator": retained_numerator,
            "denominator": retained_denominator,
            "fraction": retained_fraction,
            "per_commitment": tuple(per_commitment),
        },
        "reset_attribution": {
            "pre_loss": pre_loss,
            "live_terminal_loss": live_loss,
            "reset_loss": reset_loss,
            "live_advantage": live_advantage,
            "reset_gain": reset_gain,
            "removed_fraction": reset_removed,
        },
        "terminal_gate_statistics": probe_reports[str(LIFETIME_UPDATES)]["gate_statistics"],
    }


def _new_online_accumulators(arm_names: Sequence[str]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "loss_sum": 0.0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_max": 0.0,
            "count": 0,
            "commitment_loss_sum": [0.0] * 32,
            "commitment_count": [0] * 32,
        }
        for name in arm_names
    }


def _evaluation_progress_payload(
    *,
    next_panel: int,
    next_step: int,
    states: Mapping[str, ANMLFastState] | None,
    current_probes: Mapping[str, object],
    current_online: Mapping[str, object],
    panels: Sequence[Mapping[str, object]],
    surface_snapshots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "version": "angler.anml-evaluation-progress.v1",
        "next_panel": next_panel,
        "next_step": next_step,
        "states": (
            {name: snapshot_fast_state(state) for name, state in states.items()}
            if states is not None
            else {}
        ),
        "current_probes": copy.deepcopy(dict(current_probes)),
        "current_online": copy.deepcopy(dict(current_online)),
        "panels": copy.deepcopy(tuple(panels)),
        "surface_snapshots": copy.deepcopy(dict(surface_snapshots)),
    }


def _restore_evaluation_progress(
    value: Mapping[str, object],
    *,
    arm_names: Sequence[str],
    device: torch.device,
) -> tuple[
    int,
    int,
    dict[str, ANMLFastState] | None,
    dict[str, object],
    dict[str, object],
    list[Mapping[str, object]],
    dict[str, Mapping[str, object]],
]:
    expected = {
        "version",
        "next_panel",
        "next_step",
        "states",
        "current_probes",
        "current_online",
        "panels",
        "surface_snapshots",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("V22 evaluation resume fields are invalid")
    if value["version"] != "angler.anml-evaluation-progress.v1":
        raise RuntimeError("V22 evaluation resume version changed")
    panel = value["next_panel"]
    step = value["next_step"]
    if (
        type(panel) is not int
        or type(step) is not int
        or not 0 <= panel <= LIFETIME_PANELS
        or step not in LIFETIME_RESUME_STEPS
        or (panel == LIFETIME_PANELS and step != 0)
    ):
        raise RuntimeError("V22 evaluation resume cursor is invalid")
    state_records = value["states"]
    if not isinstance(state_records, Mapping):
        raise RuntimeError("V22 evaluation resume states are invalid")
    if panel < LIFETIME_PANELS:
        if set(state_records) != set(arm_names):
            raise RuntimeError("V22 evaluation resume arms changed")
        states = {
            name: restore_fast_state(state_records[name], device=device)
            for name in arm_names
        }
        if any(state.optimizer_state[0].step != step for state in states.values()):
            raise RuntimeError("V22 evaluation resume Adam cursor changed")
    else:
        if state_records:
            raise RuntimeError("V22 complete evaluation retained active states")
        states = None
    probes = copy.deepcopy(dict(value["current_probes"]))
    online = copy.deepcopy(dict(value["current_online"]))
    panels = list(copy.deepcopy(tuple(value["panels"])))
    surface = copy.deepcopy(dict(value["surface_snapshots"]))
    if len(panels) != panel:
        raise RuntimeError("V22 evaluation completed-panel count changed")
    if panel < LIFETIME_PANELS:
        expected_probe_steps = {0}
        expected_probe_steps.update(
            milestone for milestone in PROBE_MILESTONES[1:] if milestone <= step
        )
        if set(probes) != {str(milestone) for milestone in expected_probe_steps}:
            raise RuntimeError("V22 evaluation resume probe chronology changed")
        if set(online) != set(arm_names):
            raise RuntimeError("V22 evaluation resume online arms changed")
        for name in arm_names:
            record = online[name]
            if (
                not isinstance(record, Mapping)
                or int(record.get("count", -1)) != step
                or sum(int(value) for value in record.get("commitment_count", ())) != step
                or len(record.get("commitment_count", ())) != 32
                or len(record.get("commitment_loss_sum", ())) != 32
            ):
                raise RuntimeError("V22 evaluation resume online chronology changed")
    elif probes or online:
        raise RuntimeError("V22 complete evaluation retained active aggregates")
    return panel, step, states, probes, online, panels, surface


def _aggregate_evaluation(
    panels: Sequence[Mapping[str, object]],
    arm_names: Sequence[str],
    transfer: Mapping[str, object],
) -> dict[str, object]:
    if len(panels) != LIFETIME_PANELS:
        raise ValueError("V22 aggregate requires four panels")
    arms = {
        name: {
            "loss_auc": sum(float(panel["arms"][name]["loss_auc"]) for panel in panels)
            / LIFETIME_PANELS,
            "terminal_loss": sum(
                float(panel["arms"][name]["terminal_loss"]) for panel in panels
            )
            / LIFETIME_PANELS,
        }
        for name in arm_names
    }
    retained_numerator = sum(float(panel["retention"]["numerator"]) for panel in panels)
    retained_denominator = sum(float(panel["retention"]["denominator"]) for panel in panels)
    reset_numerator = 0.0
    reset_denominator = 0.0
    for panel in panels:
        record = panel["reset_attribution"]
        live = float(record["live_advantage"])
        removed = live - float(record["reset_gain"])
        if live > 0.0:
            reset_denominator += live
            reset_numerator += removed
    def family_mean(milestone: int, family: str) -> float:
        return sum(
            float(
                panel["probe_milestones"][str(milestone)]["arms"][ARM_SECOND_ORDER][
                    "families"
                ][family]["mean_loss"]
            )
            for panel in panels
        ) / LIFETIME_PANELS
    return {
        "arms": arms,
        "retained_acquisition_numerator": retained_numerator,
        "retained_acquisition_denominator": retained_denominator,
        "retained_acquisition_fraction": (
            max(0.0, retained_numerator) / retained_denominator
            if retained_denominator > 0.0
            else 0.0
        ),
        "original_pre_loss": family_mean(0, "original"),
        "original_terminal_loss": family_mean(LIFETIME_UPDATES, "original"),
        "unseen_pre_loss": family_mean(0, "unseen"),
        "unseen_terminal_loss": family_mean(LIFETIME_UPDATES, "unseen"),
        "fully_heldout_pre_loss": family_mean(0, "fully_v20_heldout"),
        "fully_heldout_terminal_loss": family_mean(
            LIFETIME_UPDATES, "fully_v20_heldout"
        ),
        "reset_removed_numerator": reset_numerator,
        "reset_removed_denominator": reset_denominator,
        "reset_removed_fraction": (
            max(0.0, reset_numerator) / reset_denominator
            if reset_denominator > 0.0
            else 0.0
        ),
        "surface_transfer_retained_numerator": float(transfer["terminal_numerator"]),
        "surface_transfer_retained_denominator": float(transfer["terminal_denominator"]),
        "surface_transfer_retained_fraction": float(transfer["terminal_fraction"]),
    }


def _surface_transfer_evaluation(
    system: ANMLSystem,
    panels: Sequence[Mapping[str, object]],
    surface_snapshots: Mapping[str, Mapping[str, object]],
    *,
    deadline_callback: Callable[[], None] | None,
) -> dict[str, object]:
    records = []
    terminal_numerator = 0.0
    terminal_denominator = 0.0
    for source_panel, target_panel in ((0, 1), (1, 0), (2, 3), (3, 2)):
        for milestone in (2_048, 4_096):
            key = f"{source_panel}:{milestone}"
            if key not in surface_snapshots:
                raise RuntimeError("V22 surface-transfer state snapshot is absent")
            source_state = restore_fast_state(
                surface_snapshots[key], device=system.fast_initial_weight.device
            )
            scored = _score_probe_boundary(
                system,
                target_panel,
                {ARM_SECOND_ORDER: source_state},
                (ARM_SECOND_ORDER,),
                deadline_callback=deadline_callback,
            )
            cross_loss = float(scored["arms"][ARM_SECOND_ORDER]["mean_loss"])
            target_probe = panels[target_panel]["probe_milestones"][str(milestone)]
            local_loss = float(target_probe["arms"][ARM_SECOND_ORDER]["mean_loss"])
            reset_loss = float(target_probe["controls"]["boundary_reset"]["mean_loss"])
            local_advantage = reset_loss - local_loss
            transferred_advantage = reset_loss - cross_loss
            retained = (
                transferred_advantage / local_advantage
                if local_advantage > 0.0
                else 0.0
            )
            if milestone == LIFETIME_UPDATES and local_advantage > 0.0:
                terminal_denominator += local_advantage
                terminal_numerator += transferred_advantage
            records.append(
                {
                    "source_panel": source_panel,
                    "target_panel": target_panel,
                    "milestone": milestone,
                    "source_state_digest": fast_state_digest(source_state),
                    "cross_surface_loss": cross_loss,
                    "target_local_loss": local_loss,
                    "target_reset_loss": reset_loss,
                    "local_advantage": local_advantage,
                    "transferred_advantage": transferred_advantage,
                    "retained_fraction": retained,
                }
            )
    return {
        "records": tuple(records),
        "terminal_numerator": terminal_numerator,
        "terminal_denominator": terminal_denominator,
        "terminal_fraction": (
            max(0.0, terminal_numerator) / terminal_denominator
            if terminal_denominator > 0.0
            else 0.0
        ),
    }


def evaluate_anml(
    system: ANMLSystem,
    *,
    include_random_control: bool = False,
    progress_callback: Callable[[ANMLSystem, dict[str, object]], None] | None = None,
    deadline_callback: Callable[[], None] | None = None,
    resume_state: Mapping[str, object] | None = None,
    elapsed_before_seconds: float = 0.0,
    wall_time_limit_seconds: float = SEMANTIC_WALL_TIME_CEILING_SECONDS,
) -> dict[str, object]:
    """Run four lockstep 4,096-update replay-free lifetimes.

    Each public feature bundle is captured once, consumed by every arm, and
    released before the next experience.  Only streaming aggregates and fast
    states are checkpointed; hidden features and examples are never retained.
    """

    _assert_system_integrity(system)
    if system.completed_meta_updates != OUTER_UPDATES:
        raise RuntimeError("V22 lifetime evaluation requires completed meta-fit")
    if not system.feature_equivalence_verified:
        raise RuntimeError("V22 lifetime evaluation requires feature-path parity")
    if type(include_random_control) is not bool:
        raise TypeError("V22 random-control flag must be bool")
    elapsed_before = _finite_metric(elapsed_before_seconds, "elapsed-before seconds")
    wall_limit = _finite_metric(wall_time_limit_seconds, "wall-time limit")
    if elapsed_before < 0.0 or wall_limit <= 0.0 or wall_limit > SEMANTIC_WALL_TIME_CEILING_SECONDS or elapsed_before >= wall_limit:
        raise ValueError("V22 cumulative wall-time parameters are invalid")
    device = system.fast_initial_weight.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V22 semantic lifetime evaluation requires CUDA")
    configure_anml_numerics(device)
    arm_names = PRIMARY_LIFETIME_ARMS + ((ARM_RANDOM_GATE,) if include_random_control else ())
    controller_before = _controller_digest(system.controller)
    gate_before = {
        name: _gate_digest(system.arm(name).gate) for name in LEARNED_ARMS
    }
    random_before = _gate_digest(system.random_gate)
    started = time.monotonic()

    def check_deadline() -> None:
        if deadline_callback is not None:
            deadline_callback()
        if elapsed_before + (time.monotonic() - started) >= wall_limit:
            raise RuntimeError("V22 reached its cumulative wall-time ceiling")

    if resume_state is None:
        panel_cursor = 0
        step_cursor = 0
        states: dict[str, ANMLFastState] | None = None
        current_probes: dict[str, object] = {}
        current_online: dict[str, object] = {}
        panels: list[Mapping[str, object]] = []
        surface_snapshots: dict[str, Mapping[str, object]] = {}
    else:
        (
            panel_cursor,
            step_cursor,
            states,
            current_probes,
            current_online,
            panels,
            surface_snapshots,
        ) = _restore_evaluation_progress(
            resume_state, arm_names=arm_names, device=device
        )
    torch.cuda.reset_peak_memory_stats(0 if device.index is None else device.index)
    for panel in range(panel_cursor, LIFETIME_PANELS):
        check_deadline()
        if panel != panel_cursor or states is None:
            states = {
                name: fresh_fast_state(system.fast_initial_weight) for name in arm_names
            }
            current_online = _new_online_accumulators(arm_names)
            current_probes = {
                "0": _score_probe_boundary(
                    system,
                    panel,
                    states,
                    arm_names,
                    deadline_callback=check_deadline,
                )
            }
            step_cursor = 0
        if set(states) != set(arm_names) or set(current_online) != set(arm_names):
            raise RuntimeError("V22 active lifetime arm alignment changed")
        for step in range(step_cursor, LIFETIME_UPDATES):
            check_deadline()
            record = _lifetime_record(panel, step)
            stream = _make_stream(record)
            bundle = capture_feature_bundle(
                system.controller, system.fast_initial_weight, stream
            )
            commitment_offset = int(record["commitment_index"]) - 32
            for name in arm_names:
                check_deadline()
                module, lesion, permutation = _arm_gate_configuration(
                    system, name, for_probe=False
                )
                next_state, diagnostic = _online_fast_step(
                    system,
                    states[name],
                    bundle,
                    module,
                    lesion=lesion,
                    permutation=permutation,
                )
                states[name] = next_state
                accumulator = current_online[name]
                loss = float(diagnostic["loss"])
                gradient_norm = float(diagnostic["gradient_norm"])
                accumulator["loss_sum"] = float(accumulator["loss_sum"]) + loss
                accumulator["gradient_norm_sum"] = float(accumulator["gradient_norm_sum"]) + gradient_norm
                accumulator["gradient_norm_max"] = max(
                    float(accumulator["gradient_norm_max"]), gradient_norm
                )
                accumulator["count"] = int(accumulator["count"]) + 1
                accumulator["commitment_loss_sum"][commitment_offset] += loss
                accumulator["commitment_count"][commitment_offset] += 1
            del stream, bundle
            completed = step + 1
            if completed in PROBE_MILESTONES[1:]:
                probe = _score_probe_boundary(
                    system,
                    panel,
                    states,
                    arm_names,
                    deadline_callback=check_deadline,
                )
                current_probes[str(completed)] = probe
                if completed in (2_048, 4_096):
                    surface_snapshots[f"{panel}:{completed}"] = snapshot_fast_state(
                        states[ARM_SECOND_ORDER]
                    )
            if completed % PROGRESS_LIFETIME_INTERVAL == 0 and progress_callback is not None:
                progress = _evaluation_progress_payload(
                    next_panel=panel,
                    next_step=completed,
                    states=states,
                    current_probes=current_probes,
                    current_online=current_online,
                    panels=panels,
                    surface_snapshots=surface_snapshots,
                )
                progress_callback(
                    system,
                    {
                        "phase": "lifetime",
                        "panel": panel,
                        "completed_lifetime_updates": completed,
                        "evaluation_state": progress,
                        "elapsed_seconds": elapsed_before + time.monotonic() - started,
                        "allocated_bytes": _allocated_bytes(device),
                        "system_digest": anml_system_digest(system),
                    },
                )
        panels.append(_panel_summary(panel, current_probes, current_online, arm_names))
        states = None
        current_probes = {}
        current_online = {}
        step_cursor = 0
        panel_cursor = panel + 1
    if len(panels) != LIFETIME_PANELS:
        raise RuntimeError("V22 did not complete all four lifetimes")
    transfer = _surface_transfer_evaluation(
        system,
        panels,
        surface_snapshots,
        deadline_callback=check_deadline,
    )
    aggregate = _aggregate_evaluation(panels, arm_names, transfer)
    metric_record = {
        "panels": tuple(panels),
        "aggregate": aggregate,
        "surface_transfer": transfer,
    }
    mechanical = (
        _controller_digest(system.controller) == controller_before
        and all(_gate_digest(system.arm(name).gate) == gate_before[name] for name in LEARNED_ARMS)
        and _gate_digest(system.random_gate) == random_before
        and active_gate_hook_count(system.controller) == 0
        and _allocated_bytes(device) <= ALLOCATED_MEMORY_CEILING_BYTES
    )
    gates = compute_anml_gates(metric_record)
    classification = classify_anml(gates, mechanical)
    if not mechanical:
        raise RuntimeError("V22 terminal mechanical validity failed")
    complete_progress = _evaluation_progress_payload(
        next_panel=LIFETIME_PANELS,
        next_step=0,
        states=None,
        current_probes={},
        current_online={},
        panels=panels,
        surface_snapshots=surface_snapshots,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": anml_plan_digest(),
        "classification": classification,
        "gates": gates,
        "metrics": metric_record,
        "mechanical_validity": {
            "passed": mechanical,
            "controller_digest_before": controller_before,
            "controller_digest_after": _controller_digest(system.controller),
            "gate_digests_preserved": True,
            "hook_cleanup": active_gate_hook_count(system.controller) == 0,
            "feature_equivalence_verified": system.feature_equivalence_verified,
            "maximum_allocated_bytes": _allocated_bytes(device),
            "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
        },
        "completed_lifetime_updates": LIFETIME_PANELS * LIFETIME_UPDATES,
        "include_random_control": include_random_control,
        "terminal_evaluation_state": complete_progress,
        "elapsed_seconds": time.monotonic() - started,
        "cumulative_elapsed_seconds": elapsed_before + time.monotonic() - started,
        "first_result_accepted_without_tuning": True,
        "bounded_interpretation": (
            "fresh instances of public mechanisms at the 64-feature cut; "
            "not unrestricted domain transfer, consciousness, or AGI"
        ),
        "system_digest": anml_system_digest(system),
    }
