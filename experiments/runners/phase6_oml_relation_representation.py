"""Online-aware meta-learned relation representations for Project Angler.

V20 keeps the exact V19 controller and public credit-row implementation.  It
changes only the inherited relation representation through an independently
implemented OML objective: a slow representation is trained through eight
functional AdamW updates of one small, fixed-initialization prediction head.
The paired first-order arm detaches only the inner gradients, providing the
pre-registered causal comparison for second-order meta-credit.

This module contains no task solver, mechanism router, answer lookup, or replay
policy.  Commitment indices and seeds live exclusively in orchestration code;
the learned objective receives only the public tensors produced by V19.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.evaluators import phase6_v19_paired_graph_context_recovery as v19r1
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-oml-relation-representation.v20"
CHECKPOINT_VERSION = "angler.phase6-oml-relation-representation.v1"
SOURCE_CHECKPOINT_SHA256 = (
    "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
)
SOURCE_RESULT_SHA256 = (
    "55592E9861EC16301603D0CD7BB2A104E596BAAA97BDC65D50DCC517951A0800"
)
SOURCE_RESULT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context-"
    "eval-recovery-r1.json"
)
D2_RESULT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json"
)
D2_RESULT_SHA256 = (
    "69D56232A4E70720AFD8428208A0F5ED4B4C2C75AED3D71DFF678F5BA10E6C9F"
)
D2_RESULT_SCHEMA = "angler.phase6-v11-d2-representation-overlap-report.v1"

ACTIVE_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001.md"
)
FROZEN_DEPENDENCY_HASHES = {
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
    ACTIVE_LEAF: (
        "BB2CCDDB80B25ACD5B79CB4DDC52F347ED77C8F843D972B7D4049C2B4546F257"
    ),
}

RLN_PARAMETER_PREFIXES = (
    "evidence_pair_encoder.",
    "relation_pool_attention.",
    "relation_pool_projection.",
    "relation_incidence_readout.",
    "relation_incidence_projection.",
)
RLN_EXACT_PARAMETER_PREFIX = "relation_comparator.0."
FAST_PARAMETER_NAME = "relation_comparator.2.weight"
CONTEXT_PARAMETER_PREFIXES = (
    "evidence_context_encoder.",
    "relation_context_",
    "paired_graph_",
)

ARM_SECOND_ORDER = "second_order_oml"
ARM_FIRST_ORDER = "first_order_meta"
ARM_SOURCE_ONLINE = "source_v19_online"
ARM_SECOND_NO_UPDATE = "second_order_no_update"
ARM_FIRST_NO_UPDATE = "first_order_no_update"
LEARNED_ARMS = (ARM_SECOND_ORDER, ARM_FIRST_ORDER)
EVALUATION_ARMS = (
    ARM_SECOND_ORDER,
    ARM_FIRST_ORDER,
    ARM_SOURCE_ONLINE,
    ARM_SECOND_NO_UPDATE,
    ARM_FIRST_NO_UPDATE,
)
FAMILIES = ("original", "meta_seen", "heldout")

OUTER_UPDATES = 240
INNER_STEPS = 8
OUTER_STREAMS = 8
ROWS_PER_STREAM = 4
INNER_LEARNING_RATE = 1.0e-3
OUTER_LEARNING_RATE = 3.0e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1.0e-8
ADAM_WEIGHT_DECAY = 0.0
OUTER_GRADIENT_CLIP = 5.0
ENTROPIC_TEMPERATURE = 0.05
ENTROPIC_MEAN_WEIGHT = 0.5
ENTROPIC_ROBUST_WEIGHT = 0.5
SLOT_LOSS_WEIGHT = 0.25
ALLOCATED_MEMORY_CEILING_BYTES = 12 * 1024**3
SEMANTIC_WALL_TIME_CEILING_SECONDS = 150.0 * 60.0
PROGRESS_INTERVAL = 40

TRAIN_SEED_BASES = {
    "inner": (10_001_000_001, 10_101_000_001),
    "outer_same": (10_201_000_001, 10_301_000_001),
    "outer_cross": (10_401_000_001, 10_501_000_001),
}
EVALUATION_SEED_BASE = 20_000_000_001

_PLAN_DIGEST_DOMAIN = b"project-angler.oml-relation-representation.plan.v1\x00"
_CONTROLLER_DIGEST_DOMAIN = b"project-angler.oml.controller.v1\x00"
_FROZEN_DIGEST_DOMAIN = b"project-angler.oml.frozen.v1\x00"
_CONTEXT_DIGEST_DOMAIN = b"project-angler.oml.context.v1\x00"
_AUXILIARY_DIGEST_DOMAIN = b"project-angler.oml.auxiliary-system.v1\x00"
_ARM_DIGEST_DOMAIN = b"project-angler.oml.arm.v1\x00"
_SYSTEM_DIGEST_DOMAIN = b"project-angler.oml.system.v1\x00"

AdamWSlot = v16.AdamWSlot
functional_adamw_step = v16.functional_adamw_step


@dataclass(slots=True)
class OMLArm:
    """One persistent slow-representation lineage and its outer optimizer."""

    name: str
    system: v19.V12ChampionPairedGraphContextSystem
    outer_optimizer: torch.optim.Optimizer
    fast_initial_weight: torch.Tensor
    source_frozen_digest: str
    source_context_digest: str
    source_auxiliary_digest: str
    outer_updates: int = 0

    @property
    def controller(self) -> v19.V12ChampionPairedGraphContextController:
        return self.system.controller


@dataclass(slots=True)
class OMLSystem:
    """Paired learned arms plus one untouched V19 evaluation source."""

    second_order_oml: OMLArm
    first_order_meta: OMLArm
    source_v19: v19.V12ChampionPairedGraphContextSystem
    source_checkpoint_sha256: str
    completed_updates: int = 0
    outer_mode: str | None = None

    def arm(self, name: str) -> OMLArm:
        if name == ARM_SECOND_ORDER:
            return self.second_order_oml
        if name == ARM_FIRST_ORDER:
            return self.first_order_meta
        raise KeyError(f"unknown learned OML arm: {name}")


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


def _mapping_digest(domain: bytes, values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(domain)
    for name, value in sorted(values.items()):
        _update_tensor_digest(digest, name, value)
    return "sha256:" + digest.hexdigest()


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


def _rln_parameter_name(name: str) -> bool:
    return name.startswith(RLN_PARAMETER_PREFIXES) or name.startswith(
        RLN_EXACT_PARAMETER_PREFIX
    )


def _controller_state_groups(
    controller: v19.V12ChampionPairedGraphContextController,
) -> dict[str, dict[str, torch.Tensor]]:
    state = controller.state_dict()
    rln = {name: value for name, value in state.items() if _rln_parameter_name(name)}
    fast = {name: state[name] for name in (FAST_PARAMETER_NAME,)}
    frozen = {
        name: value
        for name, value in state.items()
        if name not in rln and name not in fast
    }
    context = {
        name: value
        for name, value in state.items()
        if name.startswith(CONTEXT_PARAMETER_PREFIXES)
    }
    return {"rln": rln, "fast": fast, "frozen": frozen, "context": context}


def _validate_parameter_partition(
    controller: v19.V12ChampionPairedGraphContextController,
) -> dict[str, object]:
    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V20 requires the exact V19 controller type")
    named = dict(controller.named_parameters())
    rln_names = tuple(name for name in named if _rln_parameter_name(name))
    fast_names = tuple(name for name in named if name == FAST_PARAMETER_NAME)
    overlap = set(rln_names) & set(fast_names)
    if (
        len(rln_names) != 67
        or sum(named[name].numel() for name in rln_names) != 61_898
        or fast_names != (FAST_PARAMETER_NAME,)
        or named[FAST_PARAMETER_NAME].shape != (1, 64)
        or named[FAST_PARAMETER_NAME].numel() != 64
        or overlap
    ):
        raise RuntimeError("V20 RLN/PLN parameter partition changed")
    frozen_names = tuple(name for name in named if name not in set(rln_names + fast_names))
    if set(rln_names) | set(fast_names) | set(frozen_names) != set(named):
        raise RuntimeError("V20 parameter ownership is not exhaustive")
    paired_names = tuple(name for name in named if name.startswith("paired_graph_"))
    if len(paired_names) != 21 or sum(named[name].numel() for name in paired_names) != 34_048:
        raise RuntimeError("V20 frozen V19 paired-graph component changed")
    return {
        "rln_parameter_names": rln_names,
        "rln_tensor_count": len(rln_names),
        "rln_parameter_count": sum(named[name].numel() for name in rln_names),
        "fast_parameter_names": fast_names,
        "pln_parameter_names": fast_names,
        "fast_tensor_count": len(fast_names),
        "fast_parameter_count": named[FAST_PARAMETER_NAME].numel(),
        "frozen_parameter_names": frozen_names,
        "frozen_tensor_count": len(frozen_names),
        "frozen_parameter_count": sum(named[name].numel() for name in frozen_names),
        "paired_graph_tensor_count": len(paired_names),
        "paired_graph_parameter_count": sum(named[name].numel() for name in paired_names),
    }


def _configure_oml_controller(
    controller: v19.V12ChampionPairedGraphContextController,
    *,
    learn_rln: bool,
) -> dict[str, object]:
    report = _validate_parameter_partition(controller)
    for name, parameter in controller.named_parameters():
        parameter.requires_grad_(learn_rln and name in report["rln_parameter_names"])
        parameter.grad = None
    expected = set(report["rln_parameter_names"]) if learn_rln else set()
    actual = {name for name, value in controller.named_parameters() if value.requires_grad}
    if actual != expected:
        raise RuntimeError("V20 controller trainability differs from its owner split")
    controller.eval()
    return report


def oml_controller_digest(
    controller: v19.V12ChampionPairedGraphContextController,
) -> str:
    return _mapping_digest(_CONTROLLER_DIGEST_DOMAIN, controller.state_dict())


def oml_frozen_digest(
    controller: v19.V12ChampionPairedGraphContextController,
) -> str:
    return _mapping_digest(_FROZEN_DIGEST_DOMAIN, _controller_state_groups(controller)["frozen"])


def oml_context_digest(
    controller: v19.V12ChampionPairedGraphContextController,
) -> str:
    return _mapping_digest(_CONTEXT_DIGEST_DOMAIN, _controller_state_groups(controller)["context"])


def oml_auxiliary_system_digest(
    system: v19.V12ChampionPairedGraphContextSystem,
) -> str:
    if type(system) is not v19.V12ChampionPairedGraphContextSystem:
        raise TypeError("V20 auxiliary digest requires the exact V19 system type")
    source = system.source
    payload = {
        "mixer_state": system.mixer.state_dict(),
        "competence_state": v19.snapshot_v19_reconstruction_state(
            system.competence_state
        ),
        "source": {
            "checkpoint_sha256": source.checkpoint_sha256,
            "controller_digest": source.controller_digest,
            "mixer_digest": source.mixer_digest,
            "competence_digest": source.competence_digest,
            "system_digest": source.system_digest,
        },
        "context_updates": system.context_updates,
        "optimizer_state": system.optimizer_state,
    }
    return _object_digest(_AUXILIARY_DIGEST_DOMAIN, payload)


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RuntimeError(f"V20 {name} is not a finite floating tensor")


def _assert_arm_integrity(arm: OMLArm) -> None:
    if arm.name not in LEARNED_ARMS:
        raise RuntimeError("V20 learned arm identity changed")
    report = _validate_parameter_partition(arm.controller)
    trainable = {
        name for name, value in arm.controller.named_parameters() if value.requires_grad
    }
    if trainable != set(report["rln_parameter_names"]):
        raise RuntimeError("V20 learned arm trainability changed")
    if oml_frozen_digest(arm.controller) != arm.source_frozen_digest:
        raise RuntimeError("V20 changed a frozen controller tensor")
    if oml_context_digest(arm.controller) != arm.source_context_digest:
        raise RuntimeError("V20 changed frozen V19 context tensors")
    if oml_auxiliary_system_digest(arm.system) != arm.source_auxiliary_digest:
        raise RuntimeError("V20 changed frozen mixer, competence, or V19 lineage state")
    named = dict(arm.controller.named_parameters())
    if not torch.equal(
        named[FAST_PARAMETER_NAME].detach(), arm.fast_initial_weight.detach()
    ):
        raise RuntimeError("V20 wrote the fixed PLN initialization")
    if arm.outer_updates < 0 or arm.outer_updates > OUTER_UPDATES:
        raise RuntimeError("V20 arm update count is invalid")


def frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES}


def _read_d2_result(path: str | Path = D2_RESULT_PATH) -> dict[str, object]:
    actual = _sha256_file(path)
    if actual != D2_RESULT_SHA256:
        raise RuntimeError("V20 V11-D2 result SHA-256 changed")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("artifact_schema") != D2_RESULT_SCHEMA
        or value.get("protocol_id")
        != "phase6.public-representation-overlap.v11-d2"
    ):
        raise RuntimeError("V20 V11-D2 result identity changed")
    return value


def verify_oml_dependencies(
    source_checkpoint: str | Path,
    d2_path: str | Path = D2_RESULT_PATH,
    source_result_path: str | Path = SOURCE_RESULT_PATH,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    expected_modules = {
        "v12": root / "experiments/runners/phase6_software_pipeline_reconstruction.py",
        "v16": root / "experiments/runners/phase6_cross_variation_plasticity_v16.py",
        "v19": root / "experiments/runners/phase6_v12_champion_paired_graph_context.py",
        "v19r1": root / "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py",
    }
    observed_modules = {
        "v12": Path(v12.__file__).resolve(),
        "v16": Path(v16.__file__).resolve(),
        "v19": Path(v19.__file__).resolve(),
        "v19r1": Path(v19r1.__file__).resolve(),
    }
    if observed_modules != {key: value.resolve() for key, value in expected_modules.items()}:
        raise RuntimeError("V20 imported a shadowed dependency")
    dependencies = frozen_dependency_hashes()
    if dependencies != FROZEN_DEPENDENCY_HASHES:
        raise RuntimeError("V20 frozen dependency bytes changed")
    checkpoint_hash = _sha256_file(source_checkpoint)
    if checkpoint_hash != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V20 source checkpoint SHA-256 changed")
    source_result_hash = _sha256_file(source_result_path)
    if source_result_hash != SOURCE_RESULT_SHA256:
        raise RuntimeError("V20 accepted V19 recovery report SHA-256 changed")
    d2 = _read_d2_result(d2_path)
    return {
        "frozen_dependency_hashes": dependencies,
        "source_checkpoint_sha256": checkpoint_hash,
        "accepted_v19_recovery_report_sha256": source_result_hash,
        "d2_result_sha256": D2_RESULT_SHA256,
        "d2_classification": d2["classification"],
    }


def _training_record(
    role: str,
    update: int,
    position: int,
    commitment_index: int,
) -> dict[str, object]:
    topology_base, surface_base = TRAIN_SEED_BASES[role]
    return {
        "role": role,
        "update": update,
        "position": position,
        "commitment_index": commitment_index,
        "topology_seed": topology_base + 100_000 * update + 1_000 * position,
        "surface_seed": surface_base + 100_000 * update + 1_000 * position,
    }


def _family_commitment_indices(family: str, panel: int) -> tuple[int, ...]:
    if family not in FAMILIES or not 0 <= panel < 4:
        raise ValueError("V20 evaluation family or panel is invalid")
    if family == "original":
        return tuple(range(8))
    if family == "meta_seen":
        return tuple(8 + ((6 * position + panel) % 48) for position in range(8))
    return tuple(range(56, 64))


def _evaluation_base(family_index: int, role_index: int, kind_index: int) -> int:
    return (
        EVALUATION_SEED_BASE
        + 1_000_000_000 * family_index
        + 100_000_000 * role_index
        + 50_000_000 * kind_index
    )


def _evaluation_record(
    family: str,
    panel: int,
    role: str,
    step: int,
    position: int,
    commitment_index: int,
) -> dict[str, object]:
    family_index = FAMILIES.index(family)
    role_index = {"update": 0, "probe": 1, "terminal_credit": 2}[role]
    if role == "update":
        offset = 1_000_000 * panel + 1_000 * step
    elif role == "probe":
        offset = 1_000_000 * panel + 100_000 * step + 1_000 * position
    else:
        offset = 1_000_000 * panel + 1_000 * position
    return {
        "family": family,
        "panel": panel,
        "role": role,
        "step": step,
        "position": position,
        "commitment_index": commitment_index,
        "topology_seed": _evaluation_base(family_index, role_index, 0) + offset,
        "surface_seed": _evaluation_base(family_index, role_index, 1) + offset,
    }


@lru_cache(maxsize=1)
def _frozen_plan_payload() -> dict[str, object]:
    commitments = tuple(v12.software_pipeline_mechanism_partition("train")[:64])
    if len(commitments) != 64 or len(set(commitments)) != 64:
        raise RuntimeError("V20 requires 64 distinct public train commitments")
    updates = []
    target_counts = [0] * 48
    exposure_counts = [0] * 48
    for update in range(OUTER_UPDATES):
        targets = tuple((4 * update + slot) % 48 for slot in range(4))
        for target in targets:
            target_counts[target] += 1
            exposure_counts[target] += 2
        inner = tuple(
            _training_record(
                "inner",
                update,
                step,
                8 + targets[step % 4],
            )
            for step in range(INNER_STEPS)
        )
        same = tuple(
            _training_record("outer_same", update, slot, 8 + targets[slot])
            for slot in range(4)
        )
        cross = tuple(
            _training_record(
                "outer_cross",
                update,
                slot,
                8 + ((targets[slot] + 24) % 48),
            )
            for slot in range(4)
        )
        updates.append(
            {
                "update": update,
                "target_offsets": targets,
                "inner": inner,
                "outer_same": same,
                "outer_cross": cross,
                "outer": same + cross,
            }
        )
    panels = {}
    evaluation_records = []
    for family in FAMILIES:
        family_panels = []
        for panel in range(4):
            indices = _family_commitment_indices(family, panel)
            update_records = tuple(
                _evaluation_record(family, panel, "update", step, step, index)
                for step, index in enumerate(indices)
            )
            probe_records = tuple(
                tuple(
                    _evaluation_record(family, panel, "probe", step, position, index)
                    for position, index in enumerate(indices)
                )
                for step in range(8)
            )
            terminal_records = tuple(
                _evaluation_record(
                    family, panel, "terminal_credit", 0, position, index
                )
                for position, index in enumerate(indices)
            )
            records = update_records + tuple(
                record for group in probe_records for record in group
            ) + terminal_records
            evaluation_records.extend(records)
            family_panels.append(
                {
                    "panel": panel,
                    "commitment_indices": indices,
                    "update": update_records,
                    "probe": probe_records,
                    "terminal_credit": terminal_records,
                }
            )
        panels[family] = tuple(family_panels)
    training_records = tuple(
        record
        for update in updates
        for key in ("inner", "outer_same", "outer_cross")
        for record in update[key]
    )
    training_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"]))
        for record in training_records
    }
    evaluation_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"]))
        for record in evaluation_records
    }
    training_seed_values = {seed for pair in training_pairs for seed in pair}
    evaluation_seed_values = {seed for pair in evaluation_pairs for seed in pair}
    if (
        target_counts != [20] * 48
        or exposure_counts != [40] * 48
        or len(training_records) != OUTER_UPDATES * 16
        or len(training_pairs) != len(training_records)
        or len(training_seed_values) != 2 * len(training_records)
        or len(evaluation_records) != 960
        or len(evaluation_pairs) != 960
        or len(evaluation_seed_values) != 1_920
        or training_seed_values & evaluation_seed_values
        or training_pairs & evaluation_pairs
        or min(seed for pair in training_pairs | evaluation_pairs for seed in pair)
        <= 10_000_000_000
    ):
        raise RuntimeError("V20 seed schedule balance or isolation changed")
    for family in FAMILIES:
        if family == "meta_seen":
            panel_sets = [set(panel["commitment_indices"]) for panel in panels[family]]
            if any(left & right for index, left in enumerate(panel_sets) for right in panel_sets[index + 1 :]):
                raise RuntimeError("V20 meta-seen panels are not disjoint")
    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "accepted_v19_recovery_report_sha256": SOURCE_RESULT_SHA256,
        "d2_result_sha256": D2_RESULT_SHA256,
        "commitments": commitments,
        "retention_commitment_indices": tuple(range(8)),
        "meta_training_commitment_indices": tuple(range(8, 56)),
        "heldout_commitment_indices": tuple(range(56, 64)),
        "updates": tuple(updates),
        "evaluation_panels": panels,
        "target_counts": tuple(target_counts),
        "inner_exposure_counts": tuple(exposure_counts),
        "unique_training_streams": len(training_pairs),
        "unique_evaluation_streams": len(evaluation_pairs),
        "unique_training_seed_values": len(training_seed_values),
        "unique_evaluation_seed_values": len(evaluation_seed_values),
        "parameter_partition": {
            "rln_prefixes": RLN_PARAMETER_PREFIXES,
            "rln_exact_prefix": RLN_EXACT_PARAMETER_PREFIX,
            "rln_tensor_count": 67,
            "rln_parameter_count": 61_898,
            "pln_parameter_name": FAST_PARAMETER_NAME,
            "pln_tensor_count": 1,
            "pln_parameter_count": 64,
            "paired_graph_frozen_tensor_count": 21,
            "paired_graph_frozen_parameter_count": 34_048,
        },
        "optimization": {
            "outer_updates": OUTER_UPDATES,
            "inner_steps": INNER_STEPS,
            "inner_learning_rate": INNER_LEARNING_RATE,
            "outer_learning_rate": OUTER_LEARNING_RATE,
            "betas": (ADAM_BETA1, ADAM_BETA2),
            "epsilon": ADAM_EPSILON,
            "weight_decay": ADAM_WEIGHT_DECAY,
            "outer_gradient_clip": OUTER_GRADIENT_CLIP,
        },
        "numerical": {
            "device": "cuda",
            "dtype": "torch.float32",
            "tf32": False,
            "autocast": False,
            "exact_second_order": True,
            "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
            "wall_time_ceiling_seconds": SEMANTIC_WALL_TIME_CEILING_SECONDS,
            "progress_interval": PROGRESS_INTERVAL,
        },
    }
    digest = hashlib.sha256(_PLAN_DIGEST_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    payload["plan_digest"] = "sha256:" + digest.hexdigest()
    return payload


def oml_fit_plan() -> dict[str, object]:
    """Return an inspectable copy of the complete frozen V20 schedule."""

    return copy.deepcopy(_frozen_plan_payload())


def oml_plan_digest() -> str:
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


def _build_fit_streams(update: int) -> dict[str, tuple[object, ...]]:
    if type(update) is not int or not 0 <= update < OUTER_UPDATES:
        raise ValueError("V20 update is outside the frozen plan")
    record = _frozen_plan_payload()["updates"][update]
    return {
        "inner": tuple(_make_stream(item) for item in record["inner"]),
        "outer": tuple(_make_stream(item) for item in record["outer"]),
    }


def _build_evaluation_streams(family: str, panel: int) -> dict[str, object]:
    specification = _frozen_plan_payload()["evaluation_panels"][family][panel]
    return {
        "update": tuple(_make_stream(item) for item in specification["update"]),
        "probe": tuple(
            tuple(_make_stream(item) for item in group) for group in specification["probe"]
        ),
        "terminal_credit": tuple(
            _make_stream(item) for item in specification["terminal_credit"]
        ),
        "commitment_indices": specification["commitment_indices"],
    }


def _training_update_spec(update_index: int) -> dict[str, object]:
    if type(update_index) is not int or not 0 <= update_index < OUTER_UPDATES:
        raise ValueError("V20 update is outside the frozen plan")
    return copy.deepcopy(_frozen_plan_payload()["updates"][update_index])


def _evaluation_panel_spec(family: str, panel: int) -> dict[str, object]:
    if family not in FAMILIES or type(panel) is not int or not 0 <= panel < 4:
        raise ValueError("V20 evaluation family or panel is invalid")
    return copy.deepcopy(_frozen_plan_payload()["evaluation_panels"][family][panel])


class _V19FunctionalAdapter(nn.Module):
    """Call the exact V19 public row builder under a functional PLN weight."""

    def __init__(self, controller: v19.V12ChampionPairedGraphContextController) -> None:
        super().__init__()
        if type(controller) is not v19.V12ChampionPairedGraphContextController:
            raise TypeError("V20 functional adapter requires the exact V19 controller")
        self.controller = controller

    def forward(self, stream: object):
        return v19.public_paired_graph_credit_rows(self.controller, stream)


def _functional_credit_rows(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream: object,
):
    report = _validate_parameter_partition(controller)
    del report
    original = controller.relation_comparator[2].weight
    if (
        fast_weight.shape != original.shape
        or fast_weight.device != original.device
        or fast_weight.dtype != original.dtype
    ):
        raise ValueError("V20 functional PLN weight is not aligned")
    _require_finite_tensor("functional PLN weight", fast_weight)
    adapter = _V19FunctionalAdapter(controller)
    parameter_name = f"controller.{FAST_PARAMETER_NAME}"
    try:
        return torch.func.functional_call(
            adapter,
            {parameter_name: fast_weight},
            (stream,),
            tie_weights=True,
            strict=False,
        )
    finally:
        if controller.relation_comparator[2].weight is not original:
            raise RuntimeError("V20 functional adapter did not restore the PLN parameter")


def _row_loss(row: v19.V19PairedGraphCreditRow) -> torch.Tensor:
    if not isinstance(row, v19.V19PairedGraphCreditRow):
        raise TypeError("V20 row objective requires a V19 public credit row")
    pair = v12._paired_relation_margin_loss(row.positive_margin, row.negative_margin)
    slot = v12._relation_instance_losses(
        row.slot_positive_margins, row.slot_negative_margins
    )[1]
    loss = pair + SLOT_LOSS_WEIGHT * slot
    if loss.shape != ():
        raise RuntimeError("V20 row objective is not scalar")
    _require_finite_tensor("row objective", loss)
    return loss


def _anonymous_entropic_objective(
    losses: torch.Tensor,
    expected_count: int | None = None,
) -> torch.Tensor:
    if (
        losses.ndim != 1
        or losses.numel() == 0
        or not losses.is_floating_point()
        or (expected_count is not None and losses.numel() != expected_count)
    ):
        raise ValueError("V20 anonymous objective requires the declared loss vector")
    if not bool(torch.isfinite(losses).all().item()):
        raise ValueError("V20 anonymous losses must be finite")
    temperature = losses.new_tensor(ENTROPIC_TEMPERATURE)
    objective = ENTROPIC_MEAN_WEIGHT * losses.mean() + ENTROPIC_ROBUST_WEIGHT * temperature * (
        torch.logsumexp(losses / temperature, dim=0) - math.log(losses.numel())
    )
    _require_finite_tensor("anonymous objective", objective)
    return objective


def _anonymous_entropic_gradient_weights(losses: torch.Tensor) -> torch.Tensor:
    _require_finite_tensor("anonymous losses", losses)
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("V20 anonymous losses must be a nonempty vector")
    weights = (
        ENTROPIC_MEAN_WEIGHT / losses.numel()
        + ENTROPIC_ROBUST_WEIGHT
        * torch.softmax(losses.detach() / ENTROPIC_TEMPERATURE, dim=0)
    )
    _require_finite_tensor("anonymous gradient weights", weights)
    return weights


def _stream_loss_from_rows(
    rows: Sequence[v19.V19PairedGraphCreditRow],
) -> torch.Tensor:
    if len(rows) != ROWS_PER_STREAM:
        raise ValueError("V20 stream objective requires four V19 rows")
    losses = torch.stack(tuple(_row_loss(row) for row in rows))
    return _anonymous_entropic_objective(losses, ROWS_PER_STREAM)


def _stream_loss(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream: object,
) -> torch.Tensor:
    return _stream_loss_from_rows(_functional_credit_rows(controller, fast_weight, stream))


def _fresh_fast_state(
    initial_weight: torch.Tensor,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...]]:
    _require_finite_tensor("initial PLN weight", initial_weight)
    fast = initial_weight.detach().clone().requires_grad_(True)
    zero = torch.zeros_like(fast)
    return fast, (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _inner_step(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    state: Sequence[AdamWSlot],
    stream: object,
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], dict[str, object]]:
    if type(second_order) is not bool:
        raise TypeError("V20 second_order flag must be bool")
    if len(state) != 1:
        raise ValueError("V20 fast optimizer requires one AdamW slot")
    loss = _stream_loss(controller, fast_weight, stream)
    gradient = torch.autograd.grad(
        loss,
        (fast_weight,),
        create_graph=second_order,
        retain_graph=second_order,
        allow_unused=False,
    )[0]
    _require_finite_tensor("inner PLN gradient", gradient)
    used_gradient = gradient if second_order else gradient.detach()
    updated, next_state = functional_adamw_step(
        (fast_weight,),
        (used_gradient,),
        tuple(state),
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    next_weight = updated[0]
    _require_finite_tensor("updated PLN weight", next_weight)
    for slot in next_state:
        _require_finite_tensor("PLN first moment", slot.exp_avg)
        _require_finite_tensor("PLN second moment", slot.exp_avg_sq)
    return next_weight, tuple(next_state), {
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient.detach().to(torch.float64).norm().item()),
        "step": next_state[0].step,
        "second_order": second_order,
        "gradient_detached": not second_order,
        "fast_identity_path_preserved": next_weight.grad_fn is not None,
    }


def _unroll_inner(
    controller: v19.V12ChampionPairedGraphContextController,
    initial_weight: torch.Tensor,
    streams: Sequence[object],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], dict[str, object]]:
    if len(streams) != INNER_STEPS:
        raise ValueError("V20 inner trajectory requires exactly eight streams")
    fast, state = _fresh_fast_state(initial_weight)
    diagnostics = []
    for step, stream in enumerate(streams):
        fast, state, diagnostic = _inner_step(
            controller, fast, state, stream, second_order=second_order
        )
        if diagnostic["step"] != step + 1:
            raise RuntimeError("V20 inner AdamW step sequence changed")
        diagnostics.append(diagnostic)
    return fast, state, {
        "steps": INNER_STEPS,
        "second_order": second_order,
        "step_diagnostics": tuple(diagnostics),
        "terminal_fast_norm": float(fast.detach().to(torch.float64).norm().item()),
        "terminal_moment_step": state[0].step,
    }


def _rln_parameters(arm: OMLArm) -> tuple[tuple[str, nn.Parameter], ...]:
    report = _validate_parameter_partition(arm.controller)
    named = dict(arm.controller.named_parameters())
    return tuple((name, named[name]) for name in report["rln_parameter_names"])


def _outer_gradients_full(
    arm: OMLArm,
    inner_streams: Sequence[object],
    outer_streams: Sequence[object],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
    _assert_arm_integrity(arm)
    if len(outer_streams) != OUTER_STREAMS:
        raise ValueError("V20 outer objective requires exactly eight streams")
    fast, state, inner = _unroll_inner(
        arm.controller,
        arm.fast_initial_weight,
        inner_streams,
        second_order=second_order,
    )
    losses = torch.stack(
        tuple(_stream_loss(arm.controller, fast, stream) for stream in outer_streams)
    )
    objective = _anonymous_entropic_objective(losses, OUTER_STREAMS)
    parameters = tuple(value for _, value in _rln_parameters(arm))
    gradients = torch.autograd.grad(
        objective,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    for name, gradient in zip((name for name, _ in _rln_parameters(arm)), gradients, strict=True):
        _require_finite_tensor(f"outer gradient {name}", gradient)
    return objective.detach(), tuple(gradient.detach() for gradient in gradients), {
        "mode": "full",
        "objective": float(objective.detach().item()),
        "outer_stream_losses": tuple(float(value) for value in losses.detach().tolist()),
        "inner": inner,
        "terminal_fast_step": state[0].step,
    }


def _outer_gradients_split(
    arm: OMLArm,
    inner_streams: Sequence[object],
    outer_streams: Sequence[object],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
    """Equivalent two-by-four outer evaluation with gradient accumulation."""

    _assert_arm_integrity(arm)
    if len(outer_streams) != OUTER_STREAMS:
        raise ValueError("V20 split objective requires exactly eight streams")
    fast, state, inner = _unroll_inner(
        arm.controller,
        arm.fast_initial_weight,
        inner_streams,
        second_order=second_order,
    )
    with torch.no_grad():
        detached_losses = torch.stack(
            tuple(_stream_loss(arm.controller, fast, stream) for stream in outer_streams)
        )
    objective = _anonymous_entropic_objective(detached_losses, OUTER_STREAMS)
    coefficients = _anonymous_entropic_gradient_weights(detached_losses)
    named_parameters = _rln_parameters(arm)
    parameters = tuple(value for _, value in named_parameters)
    accumulated = [torch.zeros_like(value) for value in parameters]
    for group_index, start in enumerate((0, 4)):
        group_losses = torch.stack(
            tuple(
                _stream_loss(arm.controller, fast, stream)
                for stream in outer_streams[start : start + 4]
            )
        )
        weighted = (
            coefficients[start : start + 4].to(group_losses) * group_losses
        ).sum()
        gradients = torch.autograd.grad(
            weighted,
            parameters,
            create_graph=False,
            retain_graph=group_index == 0,
            allow_unused=False,
        )
        for index, gradient in enumerate(gradients):
            _require_finite_tensor(f"split outer gradient {named_parameters[index][0]}", gradient)
            accumulated[index] = accumulated[index] + gradient.detach()
    return objective.detach(), tuple(accumulated), {
        "mode": "split_4_plus_4",
        "objective": float(objective.detach().item()),
        "outer_stream_losses": tuple(float(value) for value in detached_losses.tolist()),
        "gradient_coefficients": tuple(float(value) for value in coefficients.tolist()),
        "inner": inner,
        "terminal_fast_step": state[0].step,
    }


def _apply_outer_step(
    arm: OMLArm,
    gradients: Sequence[torch.Tensor],
) -> dict[str, object]:
    _assert_arm_integrity(arm)
    named_parameters = _rln_parameters(arm)
    if len(gradients) != len(named_parameters):
        raise ValueError("V20 outer gradients lost RLN alignment")
    frozen_before = oml_frozen_digest(arm.controller)
    context_before = oml_context_digest(arm.controller)
    fast_before = arm.controller.state_dict()[FAST_PARAMETER_NAME].detach().clone()
    arm.outer_optimizer.zero_grad(set_to_none=True)
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        if gradient.shape != parameter.shape or gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            raise ValueError(f"V20 outer gradient is misaligned: {name}")
        _require_finite_tensor(f"outer gradient {name}", gradient)
        parameter.grad = gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(
        tuple(value for _, value in named_parameters), OUTER_GRADIENT_CLIP
    )
    _require_finite_tensor("outer gradient norm", norm)
    arm.outer_optimizer.step()
    arm.outer_optimizer.zero_grad(set_to_none=True)
    arm.outer_updates += 1
    for name, parameter in named_parameters:
        _require_finite_tensor(f"updated RLN parameter {name}", parameter)
    if (
        oml_frozen_digest(arm.controller) != frozen_before
        or oml_context_digest(arm.controller) != context_before
        or not torch.equal(
            arm.controller.state_dict()[FAST_PARAMETER_NAME].detach(), fast_before
        )
    ):
        raise RuntimeError("V20 outer step crossed its parameter ownership boundary")
    _assert_arm_integrity(arm)
    return {
        "gradient_norm_before_clip": float(norm.detach().item()),
        "gradient_clip": OUTER_GRADIENT_CLIP,
        "outer_update": arm.outer_updates,
        "frozen_digest": frozen_before,
        "context_digest": context_before,
        "fast_initial_unchanged": True,
    }


def _build_arm(
    name: str,
    source: v19.V12ChampionPairedGraphContextSystem,
) -> OMLArm:
    if name not in LEARNED_ARMS:
        raise ValueError("V20 learned arm identity is invalid")
    system = copy.deepcopy(source)
    if system.context_updates != 512:
        raise RuntimeError("V20 source is not the terminal 512-update V19 system")
    source_frozen = oml_frozen_digest(system.controller)
    source_context = oml_context_digest(system.controller)
    source_auxiliary = oml_auxiliary_system_digest(system)
    report = _configure_oml_controller(system.controller, learn_rln=True)
    named = dict(system.controller.named_parameters())
    fast_initial = named[FAST_PARAMETER_NAME].detach().clone()
    optimizer = torch.optim.AdamW(
        tuple(named[name] for name in report["rln_parameter_names"]),
        lr=OUTER_LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    arm = OMLArm(
        name=name,
        system=system,
        outer_optimizer=optimizer,
        fast_initial_weight=fast_initial,
        source_frozen_digest=source_frozen,
        source_context_digest=source_context,
        source_auxiliary_digest=source_auxiliary,
    )
    _assert_arm_integrity(arm)
    return arm


def build_oml_system(
    source_checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> OMLSystem:
    """Load three exact source copies and establish successor-local ownership."""

    if _sha256_file(source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V20 source checkpoint SHA-256 changed")
    selected = torch.device(device)
    source = v19.load_v12_champion_paired_graph_context_checkpoint(
        source_checkpoint, device=selected
    )
    second = _build_arm(ARM_SECOND_ORDER, source)
    first = _build_arm(ARM_FIRST_ORDER, source)
    _configure_oml_controller(source.controller, learn_rln=False)
    if (
        oml_controller_digest(second.controller)
        != oml_controller_digest(first.controller)
        or oml_controller_digest(second.controller)
        != oml_controller_digest(source.controller)
        or not torch.equal(second.fast_initial_weight, first.fast_initial_weight)
    ):
        raise RuntimeError("V20 paired arms did not start byte-identical")
    result = OMLSystem(
        second_order_oml=second,
        first_order_meta=first,
        source_v19=source,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
    )
    _assert_system_integrity(result)
    return result


def _assert_system_integrity(system: OMLSystem) -> None:
    if not isinstance(system, OMLSystem):
        raise TypeError("V20 requires an OMLSystem")
    if system.source_checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V20 system source binding changed")
    if (
        type(system.completed_updates) is not int
        or not 0 <= system.completed_updates <= OUTER_UPDATES
        or system.second_order_oml.outer_updates != system.completed_updates
        or system.first_order_meta.outer_updates != system.completed_updates
    ):
        raise RuntimeError("V20 paired update counters diverged")
    if system.outer_mode not in {None, "full", "split_4_plus_4"} or (
        system.completed_updates > 0 and system.outer_mode is None
    ):
        raise RuntimeError("V20 system outer-mode binding is invalid")
    _assert_arm_integrity(system.second_order_oml)
    _assert_arm_integrity(system.first_order_meta)
    if type(system.source_v19.controller) is not v19.V12ChampionPairedGraphContextController:
        raise RuntimeError("V20 source evaluation controller type changed")
    if any(value.requires_grad for value in system.source_v19.controller.parameters()):
        raise RuntimeError("V20 source evaluation controller is not frozen")
    if (
        oml_frozen_digest(system.source_v19.controller)
        != system.second_order_oml.source_frozen_digest
        or oml_context_digest(system.source_v19.controller)
        != system.second_order_oml.source_context_digest
        or oml_auxiliary_system_digest(system.source_v19)
        != system.second_order_oml.source_auxiliary_digest
    ):
        raise RuntimeError("V20 source and learned frozen state diverged")


def oml_arm_digest(arm: OMLArm) -> str:
    _assert_arm_integrity(arm)
    payload = {
        "name": arm.name,
        "controller_digest": oml_controller_digest(arm.controller),
        "outer_optimizer": arm.outer_optimizer.state_dict(),
        "fast_initial_weight": arm.fast_initial_weight,
        "source_frozen_digest": arm.source_frozen_digest,
        "source_context_digest": arm.source_context_digest,
        "source_auxiliary_digest": arm.source_auxiliary_digest,
        "outer_updates": arm.outer_updates,
    }
    return _object_digest(_ARM_DIGEST_DOMAIN, payload)


def oml_system_digest(system: OMLSystem) -> str:
    _assert_system_integrity(system)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": oml_plan_digest(),
        "source_checkpoint_sha256": system.source_checkpoint_sha256,
        "completed_updates": system.completed_updates,
        "outer_mode": system.outer_mode,
        ARM_SECOND_ORDER: oml_arm_digest(system.second_order_oml),
        ARM_FIRST_ORDER: oml_arm_digest(system.first_order_meta),
        "source_controller_digest": oml_controller_digest(system.source_v19.controller),
        "source_frozen_digest": oml_frozen_digest(system.source_v19.controller),
        "source_context_digest": oml_context_digest(system.source_v19.controller),
        "source_auxiliary_digest": oml_auxiliary_system_digest(system.source_v19),
    }
    return _object_digest(_SYSTEM_DIGEST_DOMAIN, payload)


def _allocated_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    index = torch.cuda.current_device() if device.index is None else device.index
    return int(torch.cuda.max_memory_allocated(index))


def bind_oml_outer_mode(system: OMLSystem, outer_mode: str) -> str:
    """Bind full versus split execution once for the complete run identity."""

    _assert_system_integrity(system)
    if outer_mode not in {"full", "split_4_plus_4"}:
        raise ValueError("V20 outer mode must be full or split_4_plus_4")
    if system.outer_mode is None:
        if system.completed_updates != 0:
            raise RuntimeError("V20 cannot bind outer mode after learning begins")
        system.outer_mode = outer_mode
    elif system.outer_mode != outer_mode:
        raise RuntimeError("V20 cannot switch outer mode inside one identity")
    return outer_mode


def fit_oml_update(
    system: OMLSystem,
    update_index: int | None = None,
    *,
    outer_mode: str = "full",
    streams: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, object]:
    """Apply one paired owner update using one immutable public stream set."""

    _assert_system_integrity(system)
    update = system.completed_updates if update_index is None else update_index
    if update != system.completed_updates or not 0 <= update < OUTER_UPDATES:
        raise RuntimeError("V20 update does not continue the active identity")
    bind_oml_outer_mode(system, outer_mode)
    built = _build_fit_streams(update) if streams is None else dict(streams)
    inner = tuple(built.get("inner", ()))
    outer = tuple(built.get("outer", ()))
    if len(inner) != INNER_STEPS or len(outer) != OUTER_STREAMS:
        raise ValueError("V20 fit update lost its 8+8 public streams")
    gradient_function = (
        _outer_gradients_full if outer_mode == "full" else _outer_gradients_split
    )
    second_objective, second_gradients, second_diagnostic = gradient_function(
        system.second_order_oml,
        inner,
        outer,
        second_order=True,
    )
    first_objective, first_gradients, first_diagnostic = gradient_function(
        system.first_order_meta,
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
        raise RuntimeError("V20 paired arms lost numeric forward equality before update zero")
    second_step = _apply_outer_step(system.second_order_oml, second_gradients)
    first_step = _apply_outer_step(system.first_order_meta, first_gradients)
    system.completed_updates += 1
    _assert_system_integrity(system)
    device = next(system.second_order_oml.controller.parameters()).device
    allocated = _allocated_bytes(device)
    if allocated > ALLOCATED_MEMORY_CEILING_BYTES:
        raise RuntimeError("V20 exceeded the 12-GiB allocated-memory ceiling")
    return {
        "update": update,
        "completed_updates": system.completed_updates,
        "unique_streams": 16,
        "public_rows": 64,
        "inner_steps_per_arm": INNER_STEPS,
        "paired_forward_equal_before_owner_step": before_equal,
        ARM_SECOND_ORDER: {
            **second_diagnostic,
            "owner_step": second_step,
        },
        ARM_FIRST_ORDER: {
            **first_diagnostic,
            "owner_step": first_step,
        },
        "allocated_bytes": allocated,
        "system_digest": oml_system_digest(system),
    }


def fit_oml(
    system: OMLSystem,
    *,
    outer_mode: str = "full",
    progress_callback: Callable[[OMLSystem, dict[str, object]], None] | None = None,
    wall_time_limit_seconds: float = SEMANTIC_WALL_TIME_CEILING_SECONDS,
) -> dict[str, object]:
    """Finish the frozen 240-update paired identity, resuming exactly if needed."""

    _assert_system_integrity(system)
    if (
        not math.isfinite(float(wall_time_limit_seconds))
        or wall_time_limit_seconds <= 0.0
        or wall_time_limit_seconds > SEMANTIC_WALL_TIME_CEILING_SECONDS
    ):
        raise ValueError("V20 semantic wall-time limit is invalid")
    device = next(system.second_order_oml.controller.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V20 semantic fit requires one CUDA GPU")
    if any(
        parameter.device != device or parameter.dtype != torch.float32
        for arm in (system.second_order_oml, system.first_order_meta)
        for parameter in arm.controller.parameters()
    ):
        raise RuntimeError("V20 semantic fit requires aligned CUDA FP32 controllers")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("V20 semantic fit forbids autocast")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("V20 semantic fit requires TF32 disabled")
    bind_oml_outer_mode(system, outer_mode)
    start_update = system.completed_updates
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(
        torch.cuda.current_device() if device.index is None else device.index
    )
    diagnostics = []
    for update in range(start_update, OUTER_UPDATES):
        elapsed = time.monotonic() - started
        if elapsed >= wall_time_limit_seconds:
            raise RuntimeError("V20 reached its semantic wall-time ceiling")
        diagnostic = fit_oml_update(system, update, outer_mode=outer_mode)
        diagnostics.append(diagnostic)
        if progress_callback is not None and system.completed_updates % PROGRESS_INTERVAL == 0:
            progress_callback(
                system,
                {
                    "protocol_id": PROTOCOL_ID,
                    "plan_digest": oml_plan_digest(),
                    "completed_updates": system.completed_updates,
                    "outer_mode": outer_mode,
                    "elapsed_seconds": time.monotonic() - started,
                    "allocated_bytes": _allocated_bytes(device),
                    "system_digest": oml_system_digest(system),
                },
            )
    elapsed = time.monotonic() - started
    if system.completed_updates != OUTER_UPDATES:
        raise RuntimeError("V20 fit did not complete its frozen update budget")
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": oml_plan_digest(),
        "start_update": start_update,
        "terminal_update": system.completed_updates,
        "outer_mode": outer_mode,
        "paired_arm_updates": OUTER_UPDATES - start_update,
        "unique_stream_uses": 16 * (OUTER_UPDATES - start_update),
        "inner_loss_uses_per_arm": INNER_STEPS * (OUTER_UPDATES - start_update),
        "elapsed_seconds": elapsed,
        "maximum_allocated_bytes": _allocated_bytes(device),
        "update_diagnostics": tuple(diagnostics),
        "system_digest": oml_system_digest(system),
    }


def _select_outer_mode_from_allocations(
    full_bytes: int,
    split_bytes: int,
) -> tuple[str, int]:
    """Select the frozen run mode from separately measured CUDA peaks."""

    for name, value in (("full", full_bytes), ("split", split_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V20 {name}-mode allocation must be a non-negative integer")
    if full_bytes <= ALLOCATED_MEMORY_CEILING_BYTES:
        return "full", full_bytes
    if split_bytes <= ALLOCATED_MEMORY_CEILING_BYTES:
        return "split_4_plus_4", split_bytes
    raise RuntimeError("V20 full and split modes both exceed the 12-GiB allocation ceiling")


def synthetic_cuda_preflight(device: torch.device | str = "cuda") -> dict[str, object]:
    """Measure independent synthetic full/split second-order trajectories."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V20 synthetic preflight requires CUDA")
    device_index = 0 if selected.index is None else selected.index
    torch.cuda.set_device(device_index)

    def compute(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in {"full", "split_4_plus_4"}:
            raise ValueError("V20 synthetic preflight mode is invalid")
        theta = torch.tensor(
            (0.15, -0.20, 0.30, -0.10),
            device=selected,
            dtype=torch.float32,
            requires_grad=True,
        )
        fast = torch.tensor(
            ((0.10, -0.05, 0.02, 0.03),),
            device=selected,
            dtype=torch.float32,
            requires_grad=True,
        )
        state = (
            AdamWSlot(
                step=0,
                exp_avg=torch.zeros_like(fast),
                exp_avg_sq=torch.zeros_like(fast),
            ),
        )
        for step in range(2):
            feature = theta * (1.0 + 0.1 * step)
            target = theta.new_tensor(0.2 - 0.05 * step)
            loss = (F.linear(feature, fast).reshape(()) - target).square()
            gradient = torch.autograd.grad(loss, fast, create_graph=True)[0]
            (fast,), state = functional_adamw_step(
                (fast,),
                (gradient,),
                state,
                (INNER_LEARNING_RATE,),
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                epsilon=ADAM_EPSILON,
                weight_decay=ADAM_WEIGHT_DECAY,
            )
        def outer_loss(index: int) -> torch.Tensor:
            return (
                F.linear(theta * (0.8 + 0.03 * index), fast).reshape(())
                - theta.new_tensor(-0.1 + 0.025 * index)
            ).square()

        if mode == "full":
            losses = torch.stack(tuple(outer_loss(index) for index in range(8)))
            objective = _anonymous_entropic_objective(losses, 8)
            gradient = torch.autograd.grad(objective, theta)[0]
        else:
            with torch.no_grad():
                detached_losses = torch.stack(
                    tuple(outer_loss(index) for index in range(8))
                )
                objective = _anonymous_entropic_objective(detached_losses, 8)
                coefficients = _anonymous_entropic_gradient_weights(detached_losses)
            gradient = torch.zeros_like(theta)
            for group, start in enumerate((0, 4)):
                group_losses = torch.stack(
                    tuple(outer_loss(index) for index in range(start, start + 4))
                )
                weighted = (
                    coefficients[start : start + 4].to(group_losses) * group_losses
                ).sum()
                group_gradient = torch.autograd.grad(
                    weighted,
                    theta,
                    retain_graph=group == 0,
                )[0]
                gradient = gradient + group_gradient.detach()
                del group_losses, weighted, group_gradient
        _require_finite_tensor(f"synthetic {mode} objective", objective)
        _require_finite_tensor(f"synthetic {mode} gradient", gradient)
        return objective.detach().cpu(), gradient.detach().cpu()

    def measure(mode: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        torch.cuda.synchronize(device_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        objective, gradient = compute(mode)
        torch.cuda.synchronize(device_index)
        maximum = int(torch.cuda.max_memory_allocated(device_index))
        return objective, gradient, maximum

    full_objective, full_gradient, full_allocated = measure("full")
    split_objective, split_gradient, split_allocated = measure("split_4_plus_4")
    objective_delta = float((full_objective - split_objective).abs().item())
    gradient_delta = float((full_gradient - split_gradient).abs().amax().item())
    if objective_delta > 1.0e-6 or gradient_delta > 1.0e-6:
        raise RuntimeError("V20 synthetic full/split preflight is not equivalent")
    selected_mode, selected_allocated = _select_outer_mode_from_allocations(
        full_allocated,
        split_allocated,
    )
    return {
        "synthetic_only": True,
        "device": str(selected),
        "dtype": "torch.float32",
        "full_split_objective_abs_delta": objective_delta,
        "full_split_max_gradient_abs_delta": gradient_delta,
        "equivalence_tolerance": 1.0e-6,
        "full_mode_maximum_allocated_bytes": full_allocated,
        "split_mode_maximum_allocated_bytes": split_allocated,
        "selected_mode_maximum_allocated_bytes": selected_allocated,
        "overall_maximum_allocated_bytes": max(full_allocated, split_allocated),
        "maximum_allocated_bytes": max(full_allocated, split_allocated),
        "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
        "selected_outer_mode": selected_mode,
    }


def _arm_checkpoint_payload(arm: OMLArm) -> dict[str, object]:
    _assert_arm_integrity(arm)
    return {
        "name": arm.name,
        "controller_state": {
            name: value.detach().cpu().clone()
            for name, value in arm.controller.state_dict().items()
        },
        "outer_optimizer_state": copy.deepcopy(arm.outer_optimizer.state_dict()),
        "fast_initial_weight": arm.fast_initial_weight.detach().cpu().clone(),
        "source_frozen_digest": arm.source_frozen_digest,
        "source_context_digest": arm.source_context_digest,
        "source_auxiliary_digest": arm.source_auxiliary_digest,
        "outer_updates": arm.outer_updates,
        "arm_digest": oml_arm_digest(arm),
    }


def _checkpoint_payload(system: OMLSystem) -> dict[str, object]:
    _assert_system_integrity(system)
    cuda_rng = (
        tuple(torch.cuda.get_rng_state(index).cpu() for index in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else ()
    )
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": oml_plan_digest(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "completed_updates": system.completed_updates,
        "outer_mode": system.outer_mode,
        ARM_SECOND_ORDER: _arm_checkpoint_payload(system.second_order_oml),
        ARM_FIRST_ORDER: _arm_checkpoint_payload(system.first_order_meta),
        "cpu_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_states": cuda_rng,
        "system_digest": oml_system_digest(system),
    }


def save_oml_checkpoint(path: str | Path, system: OMLSystem) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(_checkpoint_payload(system), temporary)
    temporary.replace(target)


def load_oml_checkpoint(
    path: str | Path,
    source_checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> OMLSystem:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected = {
        "version",
        "protocol_id",
        "plan_digest",
        "source_checkpoint_sha256",
        "completed_updates",
        "outer_mode",
        ARM_SECOND_ORDER,
        ARM_FIRST_ORDER,
        "cpu_rng_state",
        "cuda_rng_states",
        "system_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError("V20 checkpoint fields are invalid")
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"] != oml_plan_digest()
        or payload["source_checkpoint_sha256"] != SOURCE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("V20 checkpoint identity changed")
    system = build_oml_system(source_checkpoint, device=device)
    completed = payload["completed_updates"]
    if type(completed) is not int or not 0 <= completed <= OUTER_UPDATES:
        raise RuntimeError("V20 checkpoint update count is invalid")
    outer_mode = payload["outer_mode"]
    if outer_mode not in {None, "full", "split_4_plus_4"} or (
        completed > 0 and outer_mode is None
    ):
        raise RuntimeError("V20 checkpoint outer-mode binding is invalid")
    for name in LEARNED_ARMS:
        arm = system.arm(name)
        record = payload[name]
        required = {
            "name",
            "controller_state",
            "outer_optimizer_state",
            "fast_initial_weight",
            "source_frozen_digest",
            "source_context_digest",
            "source_auxiliary_digest",
            "outer_updates",
            "arm_digest",
        }
        if not isinstance(record, dict) or set(record) != required or record["name"] != name:
            raise RuntimeError("V20 arm checkpoint fields are invalid")
        arm.controller.load_state_dict(record["controller_state"], strict=True)
        _configure_oml_controller(arm.controller, learn_rln=True)
        arm.outer_optimizer.load_state_dict(record["outer_optimizer_state"])
        arm.fast_initial_weight = record["fast_initial_weight"].to(device).detach().clone()
        arm.source_frozen_digest = str(record["source_frozen_digest"])
        arm.source_context_digest = str(record["source_context_digest"])
        arm.source_auxiliary_digest = str(record["source_auxiliary_digest"])
        arm.outer_updates = int(record["outer_updates"])
        if arm.outer_updates != completed or oml_arm_digest(arm) != record["arm_digest"]:
            raise RuntimeError("V20 arm checkpoint digest changed")
    system.completed_updates = completed
    system.outer_mode = outer_mode
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if torch.cuda.is_available():
        states = tuple(payload["cuda_rng_states"])
        if len(states) != torch.cuda.device_count():
            raise RuntimeError("V20 checkpoint CUDA RNG device count changed")
        for index, state in enumerate(states):
            torch.cuda.set_rng_state(state.cpu(), index)
    _assert_system_integrity(system)
    if oml_system_digest(system) != payload["system_digest"]:
        raise RuntimeError("V20 checkpoint system digest changed")
    return system


def _credit_signature_digest(signature: object) -> str:
    digest = hashlib.sha256(b"project-angler.oml.relation-signature.v1\x00")
    digest.update(json.dumps(signature, separators=(",", ":")).encode("ascii"))
    return "sha256:" + digest.hexdigest()


def _zero_projection_audit() -> dict[str, object]:
    return {
        "frozen_method_calls": 0,
        "zero_residual_calls": 0,
        "delegated_nonzero_lesion_calls": 0,
        "zero_residual_rows": 0,
        "representative_rows": 0,
        "duplicate_groups": 0,
        "duplicate_rows_projected": 0,
        "maximum_raw_duplicate_logit_difference": 0.0,
    }


def _terminal_credit_metrics(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    streams: Sequence[object],
    *,
    zero_residual: bool = False,
) -> dict[str, object]:
    if len(streams) != 8:
        raise ValueError("V20 terminal credit requires eight streams")
    audit = _zero_projection_audit()

    def collect():
        with torch.no_grad():
            return tuple(
                _functional_credit_rows(controller, fast_weight, stream)
                for stream in streams
            )

    if zero_residual:
        with v19r1._temporary_zero_residual_projection(controller, audit):
            with controller.paired_graph_lesion("zero_residual"):
                row_groups = collect()
        if (
            "_paired_graph_context_logits" in controller.__dict__
            or "_v19_evaluation_recovery_wrapper_active" in controller.__dict__
        ):
            raise RuntimeError("V20 zero-residual projection wrapper was not restored")
    else:
        row_groups = collect()
    metrics = v19._credit_rows_metrics(row_groups)
    signature = metrics.pop("relation_signature")
    result = {
        key: tuple(value) if isinstance(value, tuple) else value
        for key, value in metrics.items()
    }
    result["relation_signature_digest"] = _credit_signature_digest(signature)
    result["zero_residual"] = zero_residual
    if zero_residual:
        result["projection_audit"] = audit
    return result


def _terminal_context_causal_metrics(
    full_panels: Sequence[Mapping[str, object]],
    zero_panels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(full_panels) != 4 or len(zero_panels) != 4:
        raise ValueError("V20 context causal metrics require four paired panels")
    full = v19._aggregate_metric_arms(full_panels)
    zero = v19._aggregate_metric_arms(zero_panels)
    panel_deltas = []
    recurrent_count = 0
    nonregressed = True
    relation_exact = True
    for panel, (learned, lesion) in enumerate(zip(full_panels, zero_panels, strict=True)):
        top_delta = int(learned["unique_valid_top_one"]) - int(
            lesion["unique_valid_top_one"]
        )
        mass_delta = float(learned["real_normalized_valid_mass"]) - float(
            lesion["real_normalized_valid_mass"]
        )
        recurrent = top_delta >= 0 and mass_delta >= 0.0 and (
            top_delta >= 1 or mass_delta >= 0.01
        )
        recurrent_count += int(recurrent)
        panel_nonregressed = top_delta >= -1 and mass_delta >= -0.01
        nonregressed = nonregressed and panel_nonregressed
        panel_relation_exact = (
            learned["relation_signature_digest"]
            == lesion["relation_signature_digest"]
        )
        relation_exact = relation_exact and panel_relation_exact
        panel_deltas.append(
            {
                "panel": panel,
                "unique_valid_top_one_delta": top_delta,
                "real_normalized_valid_mass_delta": mass_delta,
                "positive_recurrence": recurrent,
                "nonregressed": panel_nonregressed,
                "relation_signature_exact": panel_relation_exact,
            }
        )
    top_gain = int(full["unique_valid_top_one"]) - int(zero["unique_valid_top_one"])
    mass_gain = float(full["real_normalized_valid_mass"]) - float(
        zero["real_normalized_valid_mass"]
    )
    margin_gain = float(full["mean_informative_log_weight_margin"]) - float(
        zero["mean_informative_log_weight_margin"]
    )
    supported = (
        top_gain >= 12
        and mass_gain >= 0.05
        and margin_gain >= 0.05
        and recurrent_count >= 3
        and nonregressed
        and relation_exact
    )
    return {
        "full": full,
        "zero_residual": zero,
        "aggregate_top_one_gain": top_gain,
        "aggregate_real_normalized_valid_mass_gain": mass_gain,
        "aggregate_informative_margin_gain": margin_gain,
        "positive_recurrence_panels": recurrent_count,
        "all_panels_nonregressed": nonregressed,
        "relation_signatures_exact": relation_exact,
        "panel_deltas": tuple(panel_deltas),
        "frozen_v19_context_effect_supported": supported,
    }


def _probe_loss(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    stream: object,
) -> float:
    with torch.no_grad():
        return float(_stream_loss(controller, fast_weight, stream).item())


def _evaluate_arm_panel(
    controller: v19.V12ChampionPairedGraphContextController,
    initial_fast_weight: torch.Tensor,
    streams: Mapping[str, object],
    *,
    arm_name: str,
    updates_enabled: bool,
) -> dict[str, object]:
    if arm_name not in EVALUATION_ARMS or type(updates_enabled) is not bool:
        raise ValueError("V20 evaluation arm configuration is invalid")
    update_streams = tuple(streams["update"])
    probe_streams = tuple(tuple(group) for group in streams["probe"])
    terminal_streams = tuple(streams["terminal_credit"])
    if (
        len(update_streams) != 8
        or len(probe_streams) != 8
        or any(len(group) != 8 for group in probe_streams)
        or len(terminal_streams) != 8
    ):
        raise ValueError("V20 evaluation panel stream shape changed")
    fast, state = _fresh_fast_state(initial_fast_weight)
    online_pre = []
    probe_pre = []
    probe_post = []
    step_diagnostics = []
    for step in range(8):
        if updates_enabled:
            online_loss = _stream_loss(controller, fast, update_streams[step])
            online_pre.append(float(online_loss.detach().item()))
        else:
            online_loss = None
            online_pre.append(_probe_loss(controller, fast, update_streams[step]))
        pre = tuple(
            _probe_loss(controller, fast, stream) for stream in probe_streams[step]
        )
        probe_pre.append(pre)
        if updates_enabled:
            assert online_loss is not None
            gradient = torch.autograd.grad(
                online_loss,
                (fast,),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
            _require_finite_tensor("evaluation PLN gradient", gradient)
            (fast,), state = functional_adamw_step(
                (fast,),
                (gradient.detach(),),
                state,
                (INNER_LEARNING_RATE,),
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                epsilon=ADAM_EPSILON,
                weight_decay=ADAM_WEIGHT_DECAY,
            )
            _require_finite_tensor("evaluation PLN weight", fast)
            step_diagnostics.append(
                {
                    "step": step,
                    "gradient_norm": float(
                        gradient.detach().to(torch.float64).norm().item()
                    ),
                    "adamw_step": state[0].step,
                }
            )
        else:
            step_diagnostics.append(
                {"step": step, "gradient_norm": 0.0, "adamw_step": 0}
            )
        post = tuple(
            _probe_loss(controller, fast, stream) for stream in probe_streams[step]
        )
        probe_post.append(post)
    terminal = tuple(
        _probe_loss(controller, fast, probe_streams[step][step]) for step in range(8)
    )
    full_credit = _terminal_credit_metrics(
        controller, fast, terminal_streams, zero_residual=False
    )
    zero_credit = _terminal_credit_metrics(
        controller, fast, terminal_streams, zero_residual=True
    )
    auc = (
        0.5 * online_pre[0]
        + sum(online_pre[1:7])
        + 0.5 * online_pre[7]
    ) / 7.0
    return {
        "arm": arm_name,
        "updates_enabled": updates_enabled,
        "online_pre_loss": tuple(online_pre),
        "online_loss_auc": auc,
        "probe_pre_loss": tuple(probe_pre),
        "probe_post_loss": tuple(probe_post),
        "probe_terminal_loss": terminal,
        "terminal_credit": {"full": full_credit, "zero_residual": zero_credit},
        "step_diagnostics": tuple(step_diagnostics),
        "terminal_fast_step": state[0].step,
        "terminal_fast_norm": float(fast.detach().to(torch.float64).norm().item()),
    }


def _aggregate_arm_panels(
    panels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(panels) != 4:
        raise ValueError("V20 family aggregation requires four panels")
    immediate = []
    terminal = []
    forward = []
    for panel in panels:
        pre = panel["probe_pre_loss"]
        post = panel["probe_post_loss"]
        final = panel["probe_terminal_loss"]
        for step in range(8):
            immediate.append(float(pre[step][step]) - float(post[step][step]))
            terminal.append(float(pre[step][step]) - float(final[step]))
            for position in range(step + 1, 8):
                forward.append(
                    float(pre[step][position]) - float(post[step][position])
                )
    immediate_gain = sum(immediate) / len(immediate)
    terminal_gain = sum(terminal) / len(terminal)
    retained = terminal_gain / immediate_gain if immediate_gain > 0.0 else None
    full_panels = tuple(panel["terminal_credit"]["full"] for panel in panels)
    zero_panels = tuple(
        panel["terminal_credit"]["zero_residual"] for panel in panels
    )
    return {
        "online_loss_auc": sum(float(panel["online_loss_auc"]) for panel in panels)
        / 4.0,
        "supported_rows": sum(int(panel["supported_rows"]) for panel in full_panels),
        "qualifying_streams": sum(
            int(panel["qualifying_streams"]) for panel in full_panels
        ),
        "immediate_gain": immediate_gain,
        "terminal_gain": terminal_gain,
        "retained_fraction": retained,
        "retention_valid": immediate_gain > 0.0,
        "forward_gain": sum(forward) / len(forward),
        "panel_supported_rows": tuple(
            int(panel["supported_rows"]) for panel in full_panels
        ),
        "panel_qualifying_streams": tuple(
            int(panel["qualifying_streams"]) for panel in full_panels
        ),
        "context_causal": _terminal_context_causal_metrics(
            full_panels, zero_panels
        ),
    }


def _evaluate_family(system: OMLSystem, family: str) -> dict[str, object]:
    if family not in FAMILIES:
        raise ValueError("V20 evaluation family is invalid")
    panel_reports = []
    by_arm: dict[str, list[Mapping[str, object]]] = {
        name: [] for name in EVALUATION_ARMS
    }
    arm_configurations = (
        (
            ARM_SECOND_ORDER,
            system.second_order_oml.controller,
            system.second_order_oml.fast_initial_weight,
            True,
        ),
        (
            ARM_FIRST_ORDER,
            system.first_order_meta.controller,
            system.first_order_meta.fast_initial_weight,
            True,
        ),
        (
            ARM_SOURCE_ONLINE,
            system.source_v19.controller,
            system.second_order_oml.fast_initial_weight,
            True,
        ),
        (
            ARM_SECOND_NO_UPDATE,
            system.second_order_oml.controller,
            system.second_order_oml.fast_initial_weight,
            False,
        ),
        (
            ARM_FIRST_NO_UPDATE,
            system.first_order_meta.controller,
            system.first_order_meta.fast_initial_weight,
            False,
        ),
    )
    for panel in range(4):
        streams = _build_evaluation_streams(family, panel)
        arms = {}
        for name, controller, initial, enabled in arm_configurations:
            report = _evaluate_arm_panel(
                controller,
                initial,
                streams,
                arm_name=name,
                updates_enabled=enabled,
            )
            arms[name] = report
            by_arm[name].append(report)
        panel_reports.append(
            {
                "panel": panel,
                "commitment_indices": tuple(streams["commitment_indices"]),
                "arms": arms,
            }
        )
    return {
        "family": family,
        "panels": tuple(panel_reports),
        "arms": {
            name: _aggregate_arm_panels(tuple(reports))
            for name, reports in by_arm.items()
        },
    }


def _d2_same_module_overlap(report: Mapping[str, object] | None) -> bool:
    try:
        if report is None or report["classification"] != "REPRESENTATION_OVERLAP_INTERFERENCE_SUPPORTED":
            return False
        cells = report["evaluation"]["stages"]["relation_comparator_hidden"]["cells"]
        for cell_id in ("t0_s0", "t0_s1"):
            cell = cells[cell_id]
            if cell["group_overlap_summary"]["easy_hard_exceeds_both_within_groups"] is not True:
                return False
            shared = cell["gradient_alignment"]["shared_comparator"]
            burden = float(shared["observed_mean_off_diagonal_burden"])
            p_value = float(shared["p_value_one_sided"])
            if (
                not math.isfinite(burden)
                or not math.isfinite(p_value)
                or burden <= 0.0
                or not 0.0 <= p_value <= 0.05
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _selective_plasticity_metrics(
    heldout_family: Mapping[str, object],
) -> dict[str, object]:
    deltas = []
    target_gains = []
    positive_harms = []
    harmful_steps = 0
    total_steps = 0
    panels = tuple(heldout_family["panels"])
    if len(panels) != 4:
        raise ValueError("V20 selective-plasticity metrics require four panels")
    for panel in panels:
        arm = panel["arms"][ARM_SECOND_ORDER]
        pre = tuple(tuple(row) for row in arm["probe_pre_loss"])
        post = tuple(tuple(row) for row in arm["probe_post_loss"])
        if (
            len(pre) != 8
            or len(post) != 8
            or any(len(row) != 8 for row in pre)
            or any(len(row) != 8 for row in post)
        ):
            raise ValueError("V20 selective-plasticity probe matrix shape changed")
        for step in range(8):
            row = tuple(float(post[step][index]) - float(pre[step][index]) for index in range(8))
            if any(not math.isfinite(value) for value in row):
                raise ValueError("V20 selective-plasticity observables are non-finite")
            deltas.append(row)
            target_gain = -row[step]
            target_gains.append(target_gain)
            off_target = tuple(max(value, 0.0) for index, value in enumerate(row) if index != step)
            positive_harms.extend(off_target)
            mean_harm = sum(off_target) / len(off_target)
            harmful_steps += int(target_gain > 0.0 and mean_harm >= 0.25 * target_gain)
            total_steps += 1
    target_gain_mean = sum(target_gains) / len(target_gains)
    harm_mean = sum(positive_harms) / len(positive_harms)
    ratio = harm_mean / target_gain_mean if target_gain_mean > 0.0 else None
    fraction = harmful_steps / total_steps
    harmful = (
        target_gain_mean > 0.0
        and ratio is not None
        and ratio >= 0.25
        and fraction >= 0.50
    )
    return {
        "post_update_loss_delta": tuple(deltas),
        "target_gain_mean": target_gain_mean,
        "positive_off_target_harm_mean": harm_mean,
        "off_target_harm_to_target_gain_ratio": ratio,
        "harmful_panel_step_fraction": fraction,
        "selective_plasticity_harm": harmful,
    }


def _finite_metric(value: object, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"V20 classification metric is non-finite: {label}")
    return converted


def _classify_oml(
    families: Mapping[str, object],
    d2_report: Mapping[str, object] | None,
) -> dict[str, object]:
    """Apply the frozen V20 gates to pure recorded metrics."""

    try:
        heldout = families["heldout"]
        held_arms = heldout["arms"]
        second = held_arms[ARM_SECOND_ORDER]
        first = held_arms[ARM_FIRST_ORDER]
        source = held_arms[ARM_SOURCE_ONLINE]
        second_no = held_arms[ARM_SECOND_NO_UPDATE]
        first_no = held_arms[ARM_FIRST_NO_UPDATE]
        second_auc = _finite_metric(second["online_loss_auc"], "heldout second-order AUC")
        first_auc = _finite_metric(first["online_loss_auc"], "heldout first-order AUC")
        source_auc = _finite_metric(source["online_loss_auc"], "heldout source AUC")
        second_no_auc = _finite_metric(
            second_no["online_loss_auc"], "heldout second no-update AUC"
        )
        first_no_auc = _finite_metric(
            first_no["online_loss_auc"], "heldout first no-update AUC"
        )
        second_retained = (
            None
            if second["retained_fraction"] is None
            else _finite_metric(second["retained_fraction"], "heldout retained fraction")
        )
        second_forward = _finite_metric(second["forward_gain"], "heldout forward gain")
        panel_improvements = 0
        panel_nonregression = True
        panel_comparison = []
        heldout_panels = tuple(heldout["panels"])
        if len(heldout_panels) != 4:
            raise ValueError("V20 heldout classification requires four panels")
        for panel in heldout_panels:
            second_credit = panel["arms"][ARM_SECOND_ORDER]["terminal_credit"]["full"]
            first_credit = panel["arms"][ARM_FIRST_ORDER]["terminal_credit"]["full"]
            left = (
                int(second_credit["supported_rows"]),
                int(second_credit["qualifying_streams"]),
            )
            right = (
                int(first_credit["supported_rows"]),
                int(first_credit["qualifying_streams"]),
            )
            improved = left > right
            nonregressed = left[0] >= right[0] - 1
            panel_improvements += int(improved)
            panel_nonregression = panel_nonregression and nonregressed
            panel_comparison.append(
                {
                    "panel": panel["panel"],
                    "second_order": left,
                    "first_order": right,
                    "lexicographically_improved": improved,
                    "row_nonregressed": nonregressed,
                }
            )
        row_delta = int(second["supported_rows"]) - int(first["supported_rows"])
        stream_delta = int(second["qualifying_streams"]) - int(first["qualifying_streams"])
        second_order_credit = (
            second_auc < first_auc
            and second_auc <= 0.95 * first_auc
            and (row_delta >= 4 or stream_delta >= 2)
            and panel_improvements >= 3
            and panel_nonregression
            and second_auc < source_auc
            and second_auc < second_no_auc
        )
        cross = (
            second_order_credit
            and int(second["supported_rows"]) >= 96
            and int(second["qualifying_streams"]) >= 24
            and second_retained is not None
            and second_retained >= 0.80
            and second_forward >= 0.0
        )
        original = families["original"]
        original_second = original["arms"][ARM_SECOND_ORDER]
        original_source = original["arms"][ARM_SOURCE_ONLINE]
        original_row_nonregression = all(
            int(left) >= int(right) - 1
            for left, right in zip(
                original_second["panel_supported_rows"],
                original_source["panel_supported_rows"],
                strict=True,
            )
        )
        context_supported = bool(
            original_second["context_causal"]["frozen_v19_context_effect_supported"]
        )
        harmonized = (
            cross
            and int(original_second["supported_rows"]) >= 96
            and int(original_second["qualifying_streams"]) >= 24
            and original_row_nonregression
            and context_supported
        )
        fast_adaptation = (
            second_auc < second_no_auc
            and second_auc < source_auc
            and first_auc < first_no_auc
            and first_auc < source_auc
        )
        selective = _selective_plasticity_metrics(heldout)
        d2_overlap = _d2_same_module_overlap(d2_report)
        anml_trigger = (
            second_order_credit
            and not harmonized
            and d2_overlap
            and bool(selective["selective_plasticity_harm"])
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError, ZeroDivisionError) as error:
        raise ValueError("V20 classification metrics are incomplete or invalid") from error
    if harmonized:
        classification = "OML_V19_HARMONIZED_ADVANCEMENT"
    elif second_order_credit and not harmonized and anml_trigger:
        classification = "OML_COMPONENT_SUPPORTED_NOT_INTEGRATED"
    elif cross and not harmonized and not anml_trigger:
        classification = "OML_CROSS_MECHANISM_ADVANCEMENT"
    elif second_order_credit and not cross and not anml_trigger:
        classification = "SECOND_ORDER_OML_CREDIT_SUPPORTED"
    elif not second_order_credit and fast_adaptation:
        classification = "FAST_ADAPTATION_SUPPORTED_OML_ATTRIBUTION_NOT_ESTABLISHED"
    else:
        classification = "OML_NOT_SUPPORTED"
    return {
        "classification": classification,
        "gates": {
            "SECOND_ORDER_OML_CREDIT_SUPPORTED": second_order_credit,
            "OML_CROSS_MECHANISM_ADVANCEMENT": cross,
            "OML_V19_HARMONIZED_ADVANCEMENT": harmonized,
            "fast_adaptation_supported": fast_adaptation,
            "d2_same_module_overlap": d2_overlap,
            "selective_plasticity_harm": selective["selective_plasticity_harm"],
            "anml_trigger": anml_trigger,
        },
        "comparisons": {
            "heldout_second_order_auc": second_auc,
            "heldout_first_order_auc": first_auc,
            "heldout_source_online_auc": source_auc,
            "heldout_supported_row_delta_vs_first_order": row_delta,
            "heldout_qualifying_stream_delta_vs_first_order": stream_delta,
            "heldout_lexicographically_improved_panels": panel_improvements,
            "heldout_all_panels_row_nonregressed": panel_nonregression,
            "heldout_panel_comparison": tuple(panel_comparison),
            "original_row_nonregression_vs_source": original_row_nonregression,
            "original_context_effect_supported": context_supported,
            "selective_plasticity": selective,
        },
        "d2_binding": {
            "path": str(D2_RESULT_PATH),
            "sha256": D2_RESULT_SHA256,
            "artifact_schema": D2_RESULT_SCHEMA,
            "classification": (
                d2_report.get("classification") if isinstance(d2_report, Mapping) else None
            ),
            "same_module_overlap": d2_overlap,
        },
    }


def evaluate_oml(
    system: OMLSystem,
    *,
    d2_path: str | Path = D2_RESULT_PATH,
) -> dict[str, object]:
    """Run the frozen original/meta-seen/heldout online evaluation."""

    _assert_system_integrity(system)
    if system.completed_updates != OUTER_UPDATES:
        raise RuntimeError("V20 terminal evaluation requires all 240 outer updates")
    before = oml_system_digest(system)
    started = time.monotonic()
    d2 = _read_d2_result(d2_path)
    families = {family: _evaluate_family(system, family) for family in FAMILIES}
    classification = _classify_oml(families, d2)
    after = oml_system_digest(system)
    if after != before:
        raise RuntimeError("V20 evaluation changed the learned system")
    meta_gap = {
        name: float(families["heldout"]["arms"][name]["online_loss_auc"])
        - float(families["meta_seen"]["arms"][name]["online_loss_auc"])
        for name in EVALUATION_ARMS
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": oml_plan_digest(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "accepted_v19_recovery_report_sha256": SOURCE_RESULT_SHA256,
        "classification": classification["classification"],
        "families": families,
        "comparisons": {
            **classification["comparisons"],
            "meta_seen_to_heldout_auc_gap": meta_gap,
        },
        "gates": classification["gates"],
        "d2_binding": classification["d2_binding"],
        "first_result_accepted_without_tuning": True,
        "elapsed_seconds": time.monotonic() - started,
        "frozen_context_digest": system.second_order_oml.source_context_digest,
        "system_digest": before,
    }


__all__ = [
    "ACTIVE_LEAF",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPSILON",
    "ADAM_WEIGHT_DECAY",
    "ALLOCATED_MEMORY_CEILING_BYTES",
    "ARM_FIRST_NO_UPDATE",
    "ARM_FIRST_ORDER",
    "ARM_SECOND_NO_UPDATE",
    "ARM_SECOND_ORDER",
    "ARM_SOURCE_ONLINE",
    "AdamWSlot",
    "CHECKPOINT_VERSION",
    "D2_RESULT_PATH",
    "D2_RESULT_SCHEMA",
    "D2_RESULT_SHA256",
    "FAST_PARAMETER_NAME",
    "FAMILIES",
    "FROZEN_DEPENDENCY_HASHES",
    "INNER_LEARNING_RATE",
    "INNER_STEPS",
    "OMLArm",
    "OMLSystem",
    "OUTER_LEARNING_RATE",
    "OUTER_STREAMS",
    "OUTER_UPDATES",
    "PROTOCOL_ID",
    "RLN_PARAMETER_PREFIXES",
    "SEMANTIC_WALL_TIME_CEILING_SECONDS",
    "SOURCE_CHECKPOINT_SHA256",
    "SOURCE_RESULT_PATH",
    "SOURCE_RESULT_SHA256",
    "_V19FunctionalAdapter",
    "_anonymous_entropic_objective",
    "_apply_outer_step",
    "_classify_oml",
    "_d2_same_module_overlap",
    "_evaluation_panel_spec",
    "_evaluate_arm_panel",
    "_evaluate_family",
    "_fresh_fast_state",
    "_inner_step",
    "_outer_gradients_full",
    "_outer_gradients_split",
    "_row_loss",
    "_select_outer_mode_from_allocations",
    "_stream_loss",
    "_terminal_context_causal_metrics",
    "_terminal_credit_metrics",
    "_training_update_spec",
    "_unroll_inner",
    "_validate_parameter_partition",
    "bind_oml_outer_mode",
    "build_oml_system",
    "evaluate_oml",
    "fit_oml",
    "fit_oml_update",
    "frozen_dependency_hashes",
    "functional_adamw_step",
    "load_oml_checkpoint",
    "oml_arm_digest",
    "oml_context_digest",
    "oml_controller_digest",
    "oml_fit_plan",
    "oml_frozen_digest",
    "oml_plan_digest",
    "oml_system_digest",
    "save_oml_checkpoint",
    "synthetic_cuda_preflight",
    "verify_oml_dependencies",
]
