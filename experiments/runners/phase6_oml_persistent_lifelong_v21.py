"""Replay-free persistent fast-state evaluation for the frozen V20 OML result.

V21-A freezes every slow V20/V19 tensor and carries only one 64-value relation
head plus its two Adam moments through the declared 256-experience chronology.
The learned dependency closure receives four public support tasks; orchestration
metadata and evaluator-private pair objects never enter the loss function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-oml-persistent-lifelong.v21a"
CHECKPOINT_VERSION = "angler.phase6-oml-persistent-lifelong.v1"
ACTIVE_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-OML-PERSISTENT-LIFELONG-V21-001.md"
)
V20_CONTINUATION_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001.md"
)

SOURCE_V20_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt"
)
SOURCE_V20_REPORT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.json"
)
SOURCE_V19_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
SOURCE_V20_CHECKPOINT_SHA256 = (
    "D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48"
)
SOURCE_V20_REPORT_SHA256 = (
    "5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498"
)
SOURCE_V20_SYSTEM_DIGEST = (
    "sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3"
)
SOURCE_V19_CHECKPOINT_SHA256 = (
    "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
)
SOURCE_V20_CLASSIFICATION = "OML_V19_HARMONIZED_ADVANCEMENT"

FROZEN_DEPENDENCY_HASHES = {
    "experiments/evaluators/glyph_machine_trace_suite.py": (
        "259118357A042A9867DA90514EFD82292C36709A573EAD13AE956089DBD3BC7E"
    ),
    "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py": (
        "E9656044749805E626C2DD443EBB5C34E95656CE11128AB5B4D6A3425C927517"
    ),
    "experiments/evaluators/software_pipeline_reconstruction_suite.py": (
        "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
    ),
    "experiments/runners/phase5_glyph_machine_trace.py": (
        "BCA01B3BA152200E0E1C79EAD993DEBD2D65CECC5D881DAA8152B863A6FA1066"
    ),
    "experiments/runners/phase5_skill_memory_stream.py": (
        "CAB8CB309D80083ED22E6BA86451B64A4A30BFD0CBE696B0B17E63966CAE1924"
    ),
    "experiments/runners/phase6_counterfactual_plasticity_router.py": (
        "1AA64AAC3716F5C2C8333EE46852F839D19FC80AD39B1F5ED041E1738210C068"
    ),
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "experiments/runners/phase6_oml_relation_representation.py": (
        "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
    ),
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "src/angler/procedures/__init__.py": (
        "A6FC55919DBCE2D73A822F9F0A900C78CACEA31031068CEF251029A526FF637E"
    ),
    "src/angler/procedures/execution.py": (
        "DEC18CAB5A9F93F2463D2CBAED115848282C7D1E58DC87FFAFC81B6E2EB03AC8"
    ),
    "src/angler/procedures/learning.py": (
        "346791D1D9D81BF0827B40C20AE6C5A9931F6FDA30B682D198DE0ED203C63616"
    ),
    "src/angler/procedures/operators.py": (
        "1F6726BD66AF6666B44CD9B94C43FCE12F4CF4F03179FC3723C472C600A746EA"
    ),
    "src/angler/procedures/records.py": (
        "2FBEB4DD34249BDC9B5340F122C8C9AD2EF8DD0F8DF3232CFDFC9A3F8A4BFD0D"
    ),
    "src/angler/procedures/skill_memory.py": (
        "DDA7B7992970E6839A09E7C9C14A6B2E32E1F3B8ADB0DD4748EDC19A0672D36D"
    ),
    "src/angler/procedures/trunk.py": (
        "9C7730345FE036B4991E6C6187F620B26229F4421B9F29EF3B8CEC9E40B99D5B"
    ),
    "src/angler/reasoning/__init__.py": (
        "FF237005A2F80D98F02D7A6509F37F11FB4FF703B8A0A747E02FF5603499C79C"
    ),
    "src/angler/reasoning/adaptive_core.py": (
        "5C78F1F8E9CDF4ADE1DEEE40BBEADF48FD481A68834A3D484DB0221612858584"
    ),
    "src/angler/reasoning/recurrent_core.py": (
        "C6504E286A01E2E95F6F10F43BADAE2B2A8FB8DEBFD4E09C7EFB855D722FA0B6"
    ),
    "src/angler/reasoning/self_referential_memory.py": (
        "D87642BAC7F5073386AB88E4CCE6B29ED3C900EB5044D7ADDC95FB84420831B4"
    ),
    V20_CONTINUATION_LEAF: (
        "1C0130928F77D80F2AAA9047E44DD02B4DEDEE951CEB2E41837EAA2A086B66F5"
    ),
    ACTIVE_LEAF: (
        "B85D47AABE390761E932FE49EABD123109663802CC7BA3346E6607F6D29F092F"
    ),
}

ARM_CANDIDATE = "second_order_oml_persistent"
ARM_FIRST_ORDER = "first_order_meta_persistent"
ARM_SOURCE = "source_v19_persistent"
ARM_BOUNDARY = "second_order_boundary_reset"
UPDATED_ARMS = (ARM_CANDIDATE, ARM_FIRST_ORDER, ARM_SOURCE, ARM_BOUNDARY)

CONTROL_SECOND_NO_UPDATE = "second_order_no_update"
CONTROL_FIRST_NO_UPDATE = "first_order_no_update"
CONTROL_SOURCE_NO_UPDATE = "source_v19_no_update"
NO_UPDATE_CONTROLS = (
    CONTROL_SECOND_NO_UPDATE,
    CONTROL_FIRST_NO_UPDATE,
    CONTROL_SOURCE_NO_UPDATE,
)
MEASUREMENT_CONDITIONS = UPDATED_ARMS + NO_UPDATE_CONTROLS

PROBE_BOUNDARIES = ("pre", "end_A", "end_B")
PROBE_GROUPS = ("original", "v20_heldout", "stage_a", "dev_acquired", "dev_unseen")
PROBE_GROUP_SIZES = {
    "original": 8,
    "v20_heldout": 8,
    "stage_a": 48,
    "dev_acquired": 8,
    "dev_unseen": 8,
}

STAGE_A_EXPERIENCES = 192
STAGE_B_EXPERIENCES = 64
TOTAL_EXPERIENCES = STAGE_A_EXPERIENCES + STAGE_B_EXPERIENCES
DIAGNOSTIC_STREAMS = 56
PROBE_STREAMS = 80
PROGRESS_INTERVAL = 32
PROGRESS_CURSORS = (32, 64, 96, 128, 160, 192, 224, 256)

INNER_LEARNING_RATE = v20.INNER_LEARNING_RATE
ADAM_BETA1 = v20.ADAM_BETA1
ADAM_BETA2 = v20.ADAM_BETA2
ADAM_EPSILON = v20.ADAM_EPSILON
ADAM_WEIGHT_DECAY = v20.ADAM_WEIGHT_DECAY
ALLOCATED_MEMORY_CEILING_BYTES = 2 * 1024**3
SEMANTIC_WALL_TIME_CEILING_SECONDS = 45.0 * 60.0
CHECKPOINT_SIZE_CEILING_BYTES = 16 * 1024**2
TERMINAL_JSON_SIZE_CEILING_BYTES = 4 * 1024**2
PERSISTENT_FLOAT_VALUES = 192
PERSISTENT_FLOAT_BYTES = 768

_PLAN_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.plan.v1\x00"
_FAST_STATE_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.fast-state.v1\x00"
_LEARNED_STATE_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.learned.v1\x00"
_SYSTEM_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.system.v1\x00"
_HARNESS_STATE_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.harness.v1\x00"
_CHECKPOINT_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.checkpoint.v1\x00"
_ROW_DIGEST_DOMAIN = b"project-angler.oml-persistent-lifelong.rows.v1\x00"
_LITERAL_IDENTITY_DIGEST_DOMAIN = (
    b"project-angler.oml-persistent-lifelong.literal-identity.v1\x00"
)
_PRIOR_SEED_UPPER_BOUND = 30_000_000_000

AdamWSlot = v16.AdamWSlot
functional_adamw_step = v16.functional_adamw_step


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _object_digest(domain: bytes, value: object) -> str:
    return v20._object_digest(domain, value)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_finite_tensor(label: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RuntimeError(f"V21-A {label} is not a finite floating tensor")


def configure_persistent_lifelong_numerics() -> dict[str, object]:
    """Select the frozen FP32 deterministic mode; no semantic work is performed."""

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "dtype": "torch.float32",
        "autocast": False,
        "tf32": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


@dataclass(frozen=True, slots=True)
class PersistentFastState:
    """The complete fixed-capacity online-writable state for one arm."""

    weight: torch.Tensor
    optimizer_state: tuple[AdamWSlot, ...]
    lifetime_updates: int = 0
    reset_count: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.weight, torch.Tensor)
            or self.weight.shape != (1, 64)
            or self.weight.dtype != torch.float32
            or self.weight.requires_grad
            or self.weight.grad_fn is not None
            or not bool(torch.isfinite(self.weight).all().item())
            or len(self.optimizer_state) != 1
            or isinstance(self.lifetime_updates, bool)
            or not isinstance(self.lifetime_updates, int)
            or self.lifetime_updates < 0
            or isinstance(self.reset_count, bool)
            or not isinstance(self.reset_count, int)
            or self.reset_count not in (0, 1)
        ):
            raise ValueError("V21-A persistent fast state is invalid")
        slot = self.optimizer_state[0]
        if (
            not isinstance(slot, AdamWSlot)
            or isinstance(slot.step, bool)
            or not isinstance(slot.step, int)
            or slot.step < 0
            or slot.exp_avg.shape != self.weight.shape
            or slot.exp_avg_sq.shape != self.weight.shape
            or slot.exp_avg.device != self.weight.device
            or slot.exp_avg_sq.device != self.weight.device
            or slot.exp_avg.dtype != self.weight.dtype
            or slot.exp_avg_sq.dtype != self.weight.dtype
            or slot.exp_avg.requires_grad
            or slot.exp_avg_sq.requires_grad
            or slot.exp_avg.grad_fn is not None
            or slot.exp_avg_sq.grad_fn is not None
            or not bool(torch.isfinite(slot.exp_avg).all().item())
            or not bool(torch.isfinite(slot.exp_avg_sq).all().item())
        ):
            raise ValueError("V21-A persistent AdamW state is invalid")

    @property
    def persistent_float_values(self) -> int:
        return self.weight.numel() + self.optimizer_state[0].exp_avg.numel() + self.optimizer_state[0].exp_avg_sq.numel()


@dataclass(slots=True)
class PersistentArm:
    """One frozen slow representation paired with one persistent fast state."""

    name: str
    controller: v19.V12ChampionPairedGraphContextController
    initial_weight: torch.Tensor
    state: PersistentFastState
    source_controller_digest: str


@dataclass(slots=True)
class PersistentLifelongSystem:
    """Four matched online arms and their non-learned evidence cursor."""

    second_order_oml_persistent: PersistentArm
    first_order_meta_persistent: PersistentArm
    source_v19_persistent: PersistentArm
    second_order_boundary_reset: PersistentArm
    source_bindings: dict[str, str]
    next_experience: int = 0
    boundary_reset_applied: bool = False
    probes: dict[str, object] = field(default_factory=dict)
    gradient_geometry: dict[str, object] | None = None
    stage_b_online_pre_loss: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in MEASUREMENT_CONDITIONS}
    )
    end_a_exactness: dict[str, object] | None = None
    public_train_parity: dict[str, object] | None = None
    identity_ledger: dict[str, dict[str, object]] = field(
        default_factory=lambda: {"diagnostic": {}, "update": {}, "probe": {}}
    )
    harness_state: dict[str, object] = field(default_factory=dict)

    def arm(self, name: str) -> PersistentArm:
        if name == ARM_CANDIDATE:
            return self.second_order_oml_persistent
        if name == ARM_FIRST_ORDER:
            return self.first_order_meta_persistent
        if name == ARM_SOURCE:
            return self.source_v19_persistent
        if name == ARM_BOUNDARY:
            return self.second_order_boundary_reset
        raise KeyError(f"unknown V21-A arm: {name}")


def _schedule_record(
    *,
    role: str,
    index: int,
    group: str | None,
    member: int,
    partition: str,
    commitment_index: int,
    topology_seed: int,
    surface_seed: int,
    stage: str | None = None,
    variation: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "index": index,
        "group": group,
        "member": member,
        "partition": partition,
        "commitment_index": commitment_index,
        "topology_seed": topology_seed,
        "surface_seed": surface_seed,
    }
    if stage is not None:
        record["stage"] = stage
    if variation is not None:
        record["variation"] = variation
    return record


@lru_cache(maxsize=1)
def _frozen_plan_payload() -> dict[str, object]:
    train = tuple(v12.software_pipeline_mechanism_partition("train"))
    development = tuple(v12.software_pipeline_mechanism_partition("development"))
    if len(train) != 64 or len(development) != 16 or len(set(train + development)) != 80:
        raise RuntimeError("V21-A public mechanism partitions changed")

    experiences = []
    for index in range(STAGE_A_EXPERIENCES):
        pass_index, position = divmod(index, 48)
        experiences.append(
            _schedule_record(
                role="update",
                index=index,
                group=None,
                member=position,
                stage="A",
                variation=pass_index,
                partition="train",
                commitment_index=8 + ((position + 13 * pass_index) % 48),
                topology_seed=31_000_000_001 + 100_000 * index,
                surface_seed=31_500_000_001 + 100_000 * index,
            )
        )
    for local_index in range(STAGE_B_EXPERIENCES):
        pass_index, position = divmod(local_index, 8)
        experiences.append(
            _schedule_record(
                role="update",
                index=STAGE_A_EXPERIENCES + local_index,
                group=None,
                member=position,
                stage="B",
                variation=pass_index,
                partition="development",
                commitment_index=2 * ((position + 3 * pass_index) % 8),
                topology_seed=32_000_000_001 + 100_000 * local_index,
                surface_seed=32_500_000_001 + 100_000 * local_index,
            )
        )

    diagnostic = []
    for index in range(DIAGNOSTIC_STREAMS):
        diagnostic.append(
            _schedule_record(
                role="diagnostic",
                index=index,
                group="stage_a" if index < 48 else "dev_acquired",
                member=index if index < 48 else index - 48,
                partition="train" if index < 48 else "development",
                commitment_index=8 + index if index < 48 else 2 * (index - 48),
                topology_seed=30_000_000_001 + 100_000 * index,
                surface_seed=30_500_000_001 + 100_000 * index,
            )
        )

    group_indices = {
        "original": ("train", tuple(range(8))),
        "v20_heldout": ("train", tuple(range(56, 64))),
        "stage_a": ("train", tuple(range(8, 56))),
        "dev_acquired": ("development", tuple(range(0, 16, 2))),
        "dev_unseen": ("development", tuple(range(1, 16, 2))),
    }
    probes = []
    record_index = 0
    for group_index, group in enumerate(PROBE_GROUPS):
        partition, indices = group_indices[group]
        for member, commitment_index in enumerate(indices):
            probes.append(
                _schedule_record(
                    role="probe",
                    index=record_index,
                    group=group,
                    member=member,
                    partition=partition,
                    commitment_index=commitment_index,
                    topology_seed=33_000_000_001 + 10_000_000 * group_index + 10_000 * member,
                    surface_seed=34_000_000_001 + 10_000_000 * group_index + 10_000 * member,
                )
            )
            record_index += 1

    all_records = tuple(diagnostic) + tuple(experiences) + tuple(probes)
    seed_pairs = {
        (int(record["topology_seed"]), int(record["surface_seed"])) for record in all_records
    }
    seed_values = {value for pair in seed_pairs for value in pair}
    stage_a_counts = [0] * 48
    stage_b_counts = [0] * 8
    for record in experiences[:STAGE_A_EXPERIENCES]:
        stage_a_counts[int(record["commitment_index"]) - 8] += 1
    for record in experiences[STAGE_A_EXPERIENCES:]:
        stage_b_counts[int(record["commitment_index"]) // 2] += 1
    if (
        len(experiences) != TOTAL_EXPERIENCES
        or len(diagnostic) != DIAGNOSTIC_STREAMS
        or len(probes) != PROBE_STREAMS
        or len(seed_pairs) != 392
        or len(seed_values) != 784
        or min(seed_values) <= _PRIOR_SEED_UPPER_BOUND
        or stage_a_counts != [4] * 48
        or stage_b_counts != [8] * 8
        or any(PROBE_GROUP_SIZES[group] != sum(record["group"] == group for record in probes) for group in PROBE_GROUPS)
    ):
        raise RuntimeError("V21-A frozen schedule arithmetic changed")

    prior_seed_values = set()
    for plan in (
        v20.oml_fit_plan(),
        v19.v12_champion_paired_graph_context_plan(),
        v16.cross_variation_fit_plan(),
        v12.public_relation_fit_plan(),
        v12.public_relation_conflict_fit_plan(),
        v12.capacity_matched_relation_cluster_fit_plan(),
    ):
        def collect(value: object, key: str | None = None) -> None:
            if isinstance(value, Mapping):
                for nested_key, nested in value.items():
                    collect(nested, str(nested_key))
            elif isinstance(value, (tuple, list)):
                for nested in value:
                    collect(nested, key)
            elif type(value) is int and key is not None and "seed" in key:
                prior_seed_values.add(value)
        collect(plan)
    if prior_seed_values & seed_values or (prior_seed_values and max(prior_seed_values) >= min(seed_values)):
        raise RuntimeError("V21-A seed namespace overlaps a frozen predecessor")

    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "source_v20_checkpoint_sha256": SOURCE_V20_CHECKPOINT_SHA256,
        "source_v20_report_sha256": SOURCE_V20_REPORT_SHA256,
        "source_v20_system_digest": SOURCE_V20_SYSTEM_DIGEST,
        "source_v19_checkpoint_sha256": SOURCE_V19_CHECKPOINT_SHA256,
        "train_commitments": train,
        "development_commitments": development,
        "experiences": tuple(experiences),
        "diagnostic": tuple(diagnostic),
        "probe_cohort": tuple(probes),
        "probe_boundaries": PROBE_BOUNDARIES,
        "probe_group_sizes": dict(PROBE_GROUP_SIZES),
        "total_experiences": TOTAL_EXPERIENCES,
        "stage_a_experiences": STAGE_A_EXPERIENCES,
        "stage_b_experiences": STAGE_B_EXPERIENCES,
        "diagnostic_streams": DIAGNOSTIC_STREAMS,
        "probe_streams": PROBE_STREAMS,
        "distinct_generated_streams": len(seed_pairs),
        "distinct_seed_values": len(seed_values),
        "progress_interval": PROGRESS_INTERVAL,
        "progress_cursors": PROGRESS_CURSORS,
        "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
        "semantic_wall_time_ceiling_seconds": SEMANTIC_WALL_TIME_CEILING_SECONDS,
        "checkpoint_size_ceiling_bytes": CHECKPOINT_SIZE_CEILING_BYTES,
        "terminal_json_size_ceiling_bytes": TERMINAL_JSON_SIZE_CEILING_BYTES,
        "persistent_float_values": PERSISTENT_FLOAT_VALUES,
        "persistent_float_bytes": PERSISTENT_FLOAT_BYTES,
        "updated_arms": UPDATED_ARMS,
        "stateless_controls": NO_UPDATE_CONTROLS,
        "matched_no_update": {
            ARM_CANDIDATE: CONTROL_SECOND_NO_UPDATE,
            ARM_FIRST_ORDER: CONTROL_FIRST_NO_UPDATE,
            ARM_SOURCE: CONTROL_SOURCE_NO_UPDATE,
            ARM_BOUNDARY: CONTROL_SECOND_NO_UPDATE,
        },
        "persistent_state": {
            "weight_values": 64,
            "adam_first_moment_values": 64,
            "adam_second_moment_values": 64,
            "integer_step": 1,
            "replay_values": 0,
            "boundary_reset_cursor": STAGE_A_EXPERIENCES,
            "boundary_reset_count": 1,
        },
        "public_objective": {
            "rows_per_stream": v20.ROWS_PER_STREAM,
            "slot_loss_weight": v20.SLOT_LOSS_WEIGHT,
            "entropic_temperature": v20.ENTROPIC_TEMPERATURE,
            "mean_weight": v20.ENTROPIC_MEAN_WEIGHT,
            "robust_weight": v20.ENTROPIC_ROBUST_WEIGHT,
        },
        "optimization": {
            "learning_rate": INNER_LEARNING_RATE,
            "betas": (ADAM_BETA1, ADAM_BETA2),
            "epsilon": ADAM_EPSILON,
            "weight_decay": ADAM_WEIGHT_DECAY,
        },
        "gates": {
            "substantive_fast_acquisition": {
                "auc_ratio_max": 0.95,
                "terminal_loss_ratio_max": 0.95,
                "supported_row_gain_min": 4,
                "qualifying_stream_gain_min": 2,
                "paired_better_min": 6,
                "members": 8,
            },
            "stage_a_acquired": {
                "loss_ratio_max": 0.95,
                "supported_row_gain_min": 4,
                "qualifying_stream_gain_min": 2,
                "paired_better_min": 36,
                "members": 48,
            },
            "oml_fast_attribution": {
                "normalized_gain_margin_min": 0.02,
                "candidate_auc_no_higher": True,
            },
            "stage_a_retained": {
                "retained_fraction_min": 0.80,
                "coverage_ratio_min": 0.95,
                "paired_retained_ratio": 1.05,
                "paired_retained_min": 43,
                "members": 48,
            },
            "unseen_development_transfer": {
                "loss_ratio_max": 0.95,
                "supported_row_gain_min": 2,
                "qualifying_stream_gain_min": 1,
                "paired_better_min": 6,
                "members": 8,
            },
            "persistent_boundary_nonregression": {
                "stage_a_loss_ratio_max": 0.98,
                "paired_better_min": 36,
                "members": 48,
            },
            "inherited_nonregression": {
                "loss_ratio_max": 1.05,
                "coverage_ratio_min": 0.95,
                "paired_ratio": 1.05,
                "paired_ratio_min": 7,
                "members": 8,
            },
            "static_representation_floor": {
                "supported_rows_min": 24,
                "qualifying_streams_min": 6,
            },
        },
        "classification_priority": (
            "INVALID_NO_CLAIM",
            "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
            "STAGE_A_NOT_ACQUIRED",
            "FAST_ACQUISITION_WITH_FORGETTING",
            "INHERITED_CAPABILITY_REGRESSION",
            "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED",
            "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER",
            "STATIC_REPRESENTATION_DOMINATES",
            "PERSISTENT_OML_NOT_SUPPORTED",
        ),
    }
    digest = hashlib.sha256(_PLAN_DIGEST_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))
    payload["plan_digest"] = "sha256:" + digest.hexdigest()
    return payload


def persistent_lifelong_plan() -> dict[str, object]:
    """Return a deep copy of metadata-only V21-A chronology."""

    return copy.deepcopy(_frozen_plan_payload())


def persistent_lifelong_plan_digest() -> str:
    return str(_frozen_plan_payload()["plan_digest"])


def _experience_spec(index: int) -> dict[str, object]:
    if type(index) is not int or not 0 <= index < TOTAL_EXPERIENCES:
        raise ValueError("V21-A experience index is outside the frozen plan")
    return copy.deepcopy(_frozen_plan_payload()["experiences"][index])


def _probe_specs() -> tuple[dict[str, object], ...]:
    return copy.deepcopy(_frozen_plan_payload()["probe_cohort"])


def _diagnostic_specs() -> tuple[dict[str, object], ...]:
    return copy.deepcopy(_frozen_plan_payload()["diagnostic"])


def public_paired_graph_credit_rows_from_supports(
    controller: v19.V12ChampionPairedGraphContextController,
    supports: tuple[
        v12.PublicSoftwarePipelineTask,
        v12.PublicSoftwarePipelineTask,
        v12.PublicSoftwarePipelineTask,
        v12.PublicSoftwarePipelineTask,
    ],
    *,
    reverse_evidence_order: bool = False,
    reverse_public_presentation: bool = False,
) -> tuple[v19.V19PairedGraphCreditRow, ...]:
    """Run the exact V19 public row computation from public support tasks only.

    The frozen V19 entry point restricts its input container to the train
    partition.  This successor-local boundary removes only that orchestration
    restriction: it receives four learner-visible public tasks and otherwise
    preserves V19's tensor operation order and live controller methods.
    """

    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V21-A public rows require the exact V19 controller")
    if type(supports) is not tuple:
        raise TypeError("V21-A public rows require an exact tuple boundary")
    public_tasks = supports
    if len(public_tasks) != 4:
        raise ValueError("V21-A public rows require exactly four public support tasks")
    if any(type(task) is not v12.PublicSoftwarePipelineTask for task in public_tasks):
        raise TypeError("V21-A learned input requires exact public support-task values")
    if type(reverse_evidence_order) is not bool or type(reverse_public_presentation) is not bool:
        raise TypeError("V21-A covariance flags must be bool")

    tasks = []
    for task in public_tasks:
        if reverse_public_presentation:
            task = replace(
                task,
                components=tuple(reversed(task.components)),
                grounded_candidates=tuple(reversed(task.grounded_candidates)),
                states=tuple(reversed(task.states)),
            )
        tasks.append(task)
    encoded_tasks = tuple(v19._v19_relation_credit_task(controller, task) for task in tasks)
    rows = []
    for heldout_index, encoded_query in enumerate(encoded_tasks):
        query_transitions, query_observed, query_alternatives, query_encoding = encoded_query
        del query_transitions
        discriminating = tuple(
            index for index, alternative in enumerate(query_alternatives) if alternative is not None
        )
        if len(discriminating) != 1:
            raise RuntimeError("each V21-A public support must expose one contrast")
        transition_index = discriminating[0]
        positive_index = query_observed[transition_index]
        negative_index = query_alternatives[transition_index]
        if negative_index is None:
            raise AssertionError("validated V21-A public contrast disappeared")
        if not torch.equal(
            query_encoding.relation_context_embeddings[positive_index],
            query_encoding.relation_context_embeddings[negative_index],
        ) or not torch.equal(
            query_encoding.context_graph_adjacencies[positive_index],
            query_encoding.context_graph_adjacencies[negative_index],
        ) or not torch.equal(
            query_encoding.context_graph_masks[positive_index],
            query_encoding.context_graph_masks[negative_index],
        ):
            raise RuntimeError("same-contract V21-A alternatives changed predecessor context")
        pair_indices = torch.tensor(
            (positive_index, negative_index),
            device=query_encoding.relation_component_embeddings.device,
            dtype=torch.long,
        )
        query_contexts = query_encoding.relation_context_embeddings
        query_relations = query_encoding.relation_component_embeddings
        query_graphs = query_encoding.context_graph_adjacencies
        query_masks = query_encoding.context_graph_masks
        evidence_indices = [index for index in range(len(encoded_tasks)) if index != heldout_index]
        if reverse_evidence_order:
            evidence_indices.reverse()
        stored_contexts = []
        stored_graphs = []
        stored_masks = []
        positive_values = []
        negative_values = []
        for evidence_index in evidence_indices:
            _, evidence_observed, evidence_alternatives, evidence_encoding = encoded_tasks[
                evidence_index
            ]
            for observed_index, alternative_index in zip(
                evidence_observed, evidence_alternatives, strict=True
            ):
                stored_contexts.append(
                    evidence_encoding.relation_context_embeddings[observed_index]
                )
                stored_graphs.append(
                    evidence_encoding.context_graph_adjacencies[observed_index]
                )
                stored_masks.append(evidence_encoding.context_graph_masks[observed_index])
                positive_values.append(
                    evidence_encoding.relation_component_embeddings[observed_index]
                )
                if alternative_index is None:
                    negative_values.append(
                        evidence_encoding.relation_component_embeddings[observed_index]
                    )
                else:
                    if not torch.equal(
                        evidence_encoding.context_graph_adjacencies[observed_index],
                        evidence_encoding.context_graph_adjacencies[alternative_index],
                    ) or not torch.equal(
                        evidence_encoding.context_graph_masks[observed_index],
                        evidence_encoding.context_graph_masks[alternative_index],
                    ):
                        raise RuntimeError("V21-A evidence alternative changed predecessor graph")
                    negative_values.append(
                        evidence_encoding.relation_component_embeddings[alternative_index]
                    )
        context_matrix = torch.stack(stored_contexts)
        graph_matrix = torch.stack(stored_graphs)
        mask_matrix = torch.stack(stored_masks)
        positive_matrix = torch.stack(positive_values)
        negative_matrix = torch.stack(negative_values)
        present = (
            (context_matrix.norm(dim=-1) > 1.0e-8)
            & (positive_matrix.norm(dim=-1) > 1.0e-8)
            & (negative_matrix.norm(dim=-1) > 1.0e-8)
            & mask_matrix.any(dim=-1)
        )
        if not bool(present.any().item()):
            raise RuntimeError("V21-A public credit has no transferable slots")
        context_matrix = context_matrix[present]
        graph_matrix = graph_matrix[present]
        mask_matrix = mask_matrix[present]
        positive_matrix = positive_matrix[present]
        negative_matrix = negative_matrix[present]
        (
            positive_scores,
            positive_weights,
            positive_nulls,
            positive_relation_logits,
            positive_context_logits,
        ) = controller._paired_graph_evidence_read(
            query_contexts,
            query_relations,
            query_graphs,
            query_masks,
            context_matrix,
            positive_matrix,
            graph_matrix,
            mask_matrix,
        )
        (
            negative_scores,
            negative_weights,
            negative_nulls,
            negative_relation_logits,
            negative_context_logits,
        ) = controller._paired_graph_evidence_read(
            query_contexts,
            query_relations,
            query_graphs,
            query_masks,
            context_matrix,
            negative_matrix,
            graph_matrix,
            mask_matrix,
        )
        for left, right, label in (
            (positive_weights, negative_weights, "weights"),
            (positive_nulls, negative_nulls, "nulls"),
            (positive_context_logits, negative_context_logits, "real logits"),
        ):
            if not torch.equal(left, right):
                raise RuntimeError(f"V21-A relation alternatives changed context {label}")
        positive_logits = positive_relation_logits.index_select(0, pair_indices)
        negative_logits = negative_relation_logits.index_select(0, pair_indices)
        slot_positive = positive_logits[0] - positive_logits[1]
        slot_negative = negative_logits[0] - negative_logits[1]
        pair_weights = positive_weights.index_select(0, pair_indices)
        pair_nulls = positive_nulls.index_select(0, pair_indices)
        pair_real_logits = positive_context_logits.index_select(0, pair_indices)
        if not torch.equal(pair_weights[0], pair_weights[1]) or not torch.equal(
            pair_nulls[0], pair_nulls[1]
        ) or not torch.equal(pair_real_logits[0], pair_real_logits[1]):
            raise RuntimeError("same-contract V21-A query alternatives changed context")
        metrics = v12._relation_valid_set_metrics(
            slot_positive.detach(),
            slot_negative.detach(),
            pair_weights[0],
            pair_nulls[0],
        )
        valid_mask = metrics["valid_mask"]
        assert isinstance(valid_mask, torch.Tensor)
        selected_positive_scores = positive_scores.index_select(0, pair_indices)
        selected_negative_scores = negative_scores.index_select(0, pair_indices)
        rows.append(
            v19.V19PairedGraphCreditRow(
                heldout_index=heldout_index,
                transition_index=transition_index,
                positive_index=positive_index,
                negative_index=negative_index,
                positive_margin=selected_positive_scores[0] - selected_positive_scores[1],
                negative_margin=selected_negative_scores[0] - selected_negative_scores[1],
                slot_positive_margins=slot_positive,
                slot_negative_margins=slot_negative,
                context_weights=pair_weights[0],
                context_null_weight=pair_nulls[0],
                context_real_logits=pair_real_logits[0],
                valid_mask=valid_mask.detach(),
            )
        )
    if len(rows) != v20.ROWS_PER_STREAM:
        raise RuntimeError("V21-A stream lost a public credit row")
    return tuple(rows)


class _V21PublicFunctionalAdapter(nn.Module):
    """Apply one external fast weight to the public-task-only row boundary."""

    def __init__(self, controller: v19.V12ChampionPairedGraphContextController) -> None:
        super().__init__()
        if type(controller) is not v19.V12ChampionPairedGraphContextController:
            raise TypeError("V21-A functional adapter requires the exact V19 controller")
        self.controller = controller

    def forward(self, supports: tuple[v12.PublicSoftwarePipelineTask, ...]):
        return public_paired_graph_credit_rows_from_supports(self.controller, supports)


def _functional_public_rows(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> tuple[v19.V19PairedGraphCreditRow, ...]:
    v20._validate_parameter_partition(controller)
    original = controller.relation_comparator[2].weight
    if (
        fast_weight.shape != original.shape
        or fast_weight.device != original.device
        or fast_weight.dtype != original.dtype
    ):
        raise ValueError("V21-A functional fast weight is not aligned")
    _require_finite_tensor("functional fast weight", fast_weight)
    adapter = _V21PublicFunctionalAdapter(controller)
    try:
        return torch.func.functional_call(
            adapter,
            {f"controller.{v20.FAST_PARAMETER_NAME}": fast_weight},
            (supports,),
            tie_weights=True,
            strict=False,
        )
    finally:
        if controller.relation_comparator[2].weight is not original:
            raise RuntimeError("V21-A functional adapter did not restore the controller weight")


def _public_stream_loss_from_rows(
    rows: Sequence[v19.V19PairedGraphCreditRow],
) -> torch.Tensor:
    return v20._stream_loss_from_rows(rows)


def _public_stream_loss(
    controller: v19.V12ChampionPairedGraphContextController,
    fast_weight: torch.Tensor,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> torch.Tensor:
    return _public_stream_loss_from_rows(
        _functional_public_rows(controller, fast_weight, supports)
    )


def _rows_digest(rows: Sequence[v19.V19PairedGraphCreditRow]) -> str:
    payload = tuple(
        {
            "heldout_index": row.heldout_index,
            "transition_index": row.transition_index,
            "positive_index": row.positive_index,
            "negative_index": row.negative_index,
            "positive_margin": row.positive_margin,
            "negative_margin": row.negative_margin,
            "slot_positive_margins": row.slot_positive_margins,
            "slot_negative_margins": row.slot_negative_margins,
            "context_weights": row.context_weights,
            "context_null_weight": row.context_null_weight,
            "context_real_logits": row.context_real_logits,
            "valid_mask": row.valid_mask,
        }
        for row in rows
    )
    return _object_digest(_ROW_DIGEST_DOMAIN, payload)


def _rows_exact(
    left: Sequence[v19.V19PairedGraphCreditRow],
    right: Sequence[v19.V19PairedGraphCreditRow],
) -> bool:
    if len(left) != len(right):
        return False
    scalar_fields = ("heldout_index", "transition_index", "positive_index", "negative_index")
    tensor_fields = (
        "positive_margin",
        "negative_margin",
        "slot_positive_margins",
        "slot_negative_margins",
        "context_weights",
        "context_null_weight",
        "context_real_logits",
        "valid_mask",
    )
    return all(
        all(getattr(a, name) == getattr(b, name) for name in scalar_fields)
        and all(torch.equal(getattr(a, name), getattr(b, name)) for name in tensor_fields)
        for a, b in zip(left, right, strict=True)
    )


def public_train_parity_report(
    controller: v19.V12ChampionPairedGraphContextController,
    stream: v12.SoftwarePipelineStream,
) -> dict[str, object]:
    """Prove successor rows/loss/fast and all-RLN gradients equal frozen V19."""

    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V21-A train parity requires the exact V19 controller")
    if type(stream) is not v12.SoftwarePipelineStream:
        raise TypeError("V21-A train parity requires a software-pipeline stream")
    if stream.mechanism_partition != "train" or stream.control_arm != "correct":
        raise ValueError("V21-A train parity accepts only a correct train stream")
    supports = tuple(pair.learner for pair in stream.supports)
    legacy_controller = copy.deepcopy(controller)
    generic_controller = copy.deepcopy(controller)
    v20._configure_oml_controller(legacy_controller, learn_rln=True)
    v20._configure_oml_controller(generic_controller, learn_rln=True)
    if v20.oml_controller_digest(legacy_controller) != v20.oml_controller_digest(
        generic_controller
    ):
        raise RuntimeError("V21-A parity controllers did not start byte-identical")
    legacy_fast = (
        legacy_controller.relation_comparator[2].weight.detach().clone().requires_grad_(True)
    )
    generic_fast = (
        generic_controller.relation_comparator[2].weight.detach().clone().requires_grad_(True)
    )
    legacy_rows = v20._functional_credit_rows(legacy_controller, legacy_fast, stream)
    generic_rows = _functional_public_rows(generic_controller, generic_fast, supports)
    if not _rows_exact(legacy_rows, generic_rows):
        raise RuntimeError("V21-A generalized public rows differ from frozen V19")
    legacy_loss = v20._stream_loss_from_rows(legacy_rows)
    generic_loss = _public_stream_loss_from_rows(generic_rows)
    if not torch.equal(legacy_loss, generic_loss):
        raise RuntimeError("V21-A generalized public loss differs from frozen V20")
    legacy_report = v20._validate_parameter_partition(legacy_controller)
    generic_report = v20._validate_parameter_partition(generic_controller)
    names = tuple(legacy_report["rln_parameter_names"])
    if names != tuple(generic_report["rln_parameter_names"]):
        raise RuntimeError("V21-A parity RLN ownership changed")
    legacy_named = dict(legacy_controller.named_parameters())
    generic_named = dict(generic_controller.named_parameters())
    legacy_gradients = torch.autograd.grad(
        legacy_loss,
        (legacy_fast,) + tuple(legacy_named[name] for name in names),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    generic_gradients = torch.autograd.grad(
        generic_loss,
        (generic_fast,) + tuple(generic_named[name] for name in names),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    if any(
        not torch.equal(left, right)
        for left, right in zip(legacy_gradients, generic_gradients, strict=True)
    ):
        raise RuntimeError("V21-A generalized public gradients differ from frozen V19/V20")
    return {
        "row_values_exact": True,
        "row_order_exact": True,
        "row_masks_exact": True,
        "loss_exact": True,
        "fast_gradient_exact": True,
        "rln_gradients_exact": True,
        "rln_tensor_count": len(names),
        "rows": len(legacy_rows),
        "row_digest": _rows_digest(legacy_rows),
    }


def _record_commitment(record: Mapping[str, object]) -> str:
    plan = _frozen_plan_payload()
    partition = str(record["partition"])
    index = int(record["commitment_index"])
    if partition == "train":
        commitments = plan["train_commitments"]
    elif partition == "development":
        commitments = plan["development_commitments"]
    else:
        raise RuntimeError("V21-A schedule selected a forbidden partition")
    try:
        return str(commitments[index])
    except (IndexError, TypeError) as error:
        raise RuntimeError("V21-A schedule selected an invalid commitment") from error


def _make_protocol_stream(record: Mapping[str, object]) -> v12.SoftwarePipelineStream:
    return v12.make_software_pipeline_stream(
        int(record["topology_seed"]),
        surface_seed=int(record["surface_seed"]),
        supports_per_motif=2,
        queries=1,
        maximum_steps=4,
        mechanism_commitment=_record_commitment(record),
        mechanism_partition=str(record["partition"]),
    )


def _public_supports_from_stream(
    stream: v12.SoftwarePipelineStream,
    record: Mapping[str, object],
) -> tuple[v12.PublicSoftwarePipelineTask, ...]:
    if type(stream) is not v12.SoftwarePipelineStream:
        raise TypeError("V21-A stream factory returned an invalid stream")
    if (
        stream.control_arm != "correct"
        or stream.mechanism_partition != record["partition"]
        or stream.mechanism_commitment != _record_commitment(record)
        or len(stream.supports) != 4
        or len(stream.queries) != 1
    ):
        raise RuntimeError("V21-A materialized stream differs from its frozen record")
    supports = tuple(pair.learner for pair in stream.supports)
    if len(supports) != 4 or any(
        type(task) is not v12.PublicSoftwarePipelineTask for task in supports
    ):
        raise RuntimeError("V21-A public projection lost a support task")
    public_digests = _public_support_tuple_digests(supports)
    if len(set(public_digests)) != 4:
        raise RuntimeError("V21-A materialized duplicate public support tasks")
    return supports


def _public_support_tuple_digests(
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> tuple[str, ...]:
    if type(supports) is not tuple or len(supports) != 4 or any(
        type(task) is not v12.PublicSoftwarePipelineTask for task in supports
    ):
        raise TypeError("V21-A public support digest requires an exact four-task tuple")
    return tuple(
        "sha256:"
        + hashlib.sha256(
            json.dumps(task.to_canonical(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        for task in supports
    )


def _literal_identity_payload(value: object) -> object:
    """Encode exact field and sequence order for control-only identity checks."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": tuple(
                (item.name, _literal_identity_payload(getattr(value, item.name)))
                for item in fields(value)
            ),
        }
    if isinstance(value, tuple):
        return {"tuple": tuple(_literal_identity_payload(item) for item in value)}
    if isinstance(value, list):
        return {"list": tuple(_literal_identity_payload(item) for item in value)}
    if isinstance(value, frozenset):
        encoded = tuple(_literal_identity_payload(item) for item in value)
        return {
            "frozenset": tuple(
                sorted(
                    encoded,
                    key=lambda item: _object_digest(
                        _LITERAL_IDENTITY_DIGEST_DOMAIN,
                        item,
                    ),
                )
            )
        }
    if isinstance(value, Mapping):
        return {
            "mapping": tuple(
                (
                    _literal_identity_payload(key),
                    _literal_identity_payload(value[key]),
                )
                for key in sorted(value, key=lambda item: str(item))
            )
        }
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("V21-A literal identity contains a non-finite float")
        return {"float_hex": value.hex()}
    raise TypeError(
        f"V21-A literal identity cannot encode {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _literal_identity_digest(value: object) -> str:
    return _object_digest(
        _LITERAL_IDENTITY_DIGEST_DOMAIN,
        _literal_identity_payload(value),
    )


def _stream_identity_record(
    stream: v12.SoftwarePipelineStream,
    record: Mapping[str, object],
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> dict[str, object]:
    if type(stream) is not v12.SoftwarePipelineStream:
        raise TypeError("V21-A identity ledger requires an exact stream")
    if tuple(pair.learner for pair in stream.supports) != supports:
        raise RuntimeError("V21-A identity ledger public projection changed")
    public_queries = tuple(pair.learner for pair in stream.queries)
    support_packages = tuple(_literal_identity_digest(pair) for pair in stream.supports)
    query_packages = tuple(_literal_identity_digest(pair) for pair in stream.queries)
    public_supports = tuple(_literal_identity_digest(task) for task in supports)
    public_query_tasks = tuple(_literal_identity_digest(task) for task in public_queries)
    if (
        len(set(support_packages + query_packages))
        != len(support_packages) + len(query_packages)
        or len(set(public_supports + public_query_tasks))
        != len(public_supports) + len(public_query_tasks)
    ):
        raise RuntimeError("V21-A materialized duplicate package/task identity")
    return {
        "schedule_digest": _object_digest(
            _LITERAL_IDENTITY_DIGEST_DOMAIN,
            dict(record),
        ),
        "stream_literal_digest": _literal_identity_digest(stream),
        "support_package_literal_digests": support_packages,
        "query_package_literal_digests": query_packages,
        "public_support_literal_digests": public_supports,
        "public_query_literal_digests": public_query_tasks,
        "public_support_canonical_digests": _public_support_tuple_digests(
            supports
        ),
    }


def _identity_record_digest_values(identity: Mapping[str, object]) -> tuple[str, ...]:
    values = [str(identity["stream_literal_digest"])]
    for field_name in (
        "support_package_literal_digests",
        "query_package_literal_digests",
        "public_support_literal_digests",
        "public_query_literal_digests",
    ):
        values.extend(str(item) for item in identity[field_name])
    return tuple(values)


def _register_stream_identity(
    system: PersistentLifelongSystem,
    record: Mapping[str, object],
    stream: v12.SoftwarePipelineStream,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
    *,
    probe_boundary: str | None = None,
) -> dict[str, object]:
    """Record control evidence without exposing it to any learned component."""

    role = record.get("role")
    index = record.get("index")
    if role not in ("diagnostic", "update", "probe") or type(index) is not int:
        raise RuntimeError("V21-A materialized stream has no frozen role/index")
    if set(system.identity_ledger) != {"diagnostic", "update", "probe"}:
        raise RuntimeError("V21-A identity ledger roles changed")
    key = str(index)
    identity = _stream_identity_record(stream, record, supports)
    role_records = system.identity_ledger[str(role)]
    existing = role_records.get(key)
    if role == "probe":
        if probe_boundary not in PROBE_BOUNDARIES:
            raise RuntimeError("V21-A probe identity has no frozen boundary")
        if existing is None:
            if probe_boundary != "pre":
                raise RuntimeError("V21-A probe identity was not established at pre")
            next_record = {"identity": identity, "boundaries": ("pre",)}
        else:
            if not isinstance(existing, Mapping) or existing.get("identity") != identity:
                raise RuntimeError("V21-A probe literal identity changed across boundaries")
            boundaries = tuple(existing.get("boundaries", ()))
            expected = PROBE_BOUNDARIES[len(boundaries)] if len(boundaries) < 3 else None
            if probe_boundary != expected:
                raise RuntimeError("V21-A probe identity boundary chronology changed")
            next_record = {
                "identity": identity,
                "boundaries": boundaries + (probe_boundary,),
            }
    else:
        if probe_boundary is not None or existing is not None:
            raise RuntimeError("V21-A non-probe identity was materialized twice")
        next_record = {"identity": identity}

    current_values = set(_identity_record_digest_values(identity))
    for other_role, records in system.identity_ledger.items():
        for other_key, other in records.items():
            if other_role == role and other_key == key:
                continue
            if not isinstance(other, Mapping) or not isinstance(other.get("identity"), Mapping):
                raise RuntimeError("V21-A identity ledger record changed")
            other_values = set(_identity_record_digest_values(other["identity"]))
            if current_values & other_values:
                raise RuntimeError(
                    "V21-A constructed stream/package/task identity collided across roles"
                )
    role_records[key] = next_record
    return copy.deepcopy(next_record)


StreamFactory = Callable[[Mapping[str, object]], v12.SoftwarePipelineStream]


def _materialize_public_supports(
    record: Mapping[str, object],
    stream_factory: StreamFactory | None,
    *,
    system: PersistentLifelongSystem | None = None,
    probe_boundary: str | None = None,
) -> tuple[v12.PublicSoftwarePipelineTask, ...]:
    stream = (stream_factory or _make_protocol_stream)(record)
    supports = _public_supports_from_stream(stream, record)
    if system is not None:
        _register_stream_identity(
            system,
            record,
            stream,
            supports,
            probe_boundary=probe_boundary,
        )
    return supports


def frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES}


def verify_persistent_lifelong_dependencies(
    v20_checkpoint: str | Path = SOURCE_V20_CHECKPOINT,
    v20_report: str | Path = SOURCE_V20_REPORT,
    v19_checkpoint: str | Path = SOURCE_V19_CHECKPOINT,
) -> dict[str, object]:
    """Verify frozen repository and consumed-result identities without loading a model."""

    observed = frozen_dependency_hashes()
    if observed != FROZEN_DEPENDENCY_HASHES:
        raise RuntimeError("V21-A frozen repository dependency bytes changed")
    artifact_hashes = {
        "v20_checkpoint": _sha256_file(v20_checkpoint),
        "v20_report": _sha256_file(v20_report),
        "v19_checkpoint": _sha256_file(v19_checkpoint),
    }
    expected_artifacts = {
        "v20_checkpoint": SOURCE_V20_CHECKPOINT_SHA256,
        "v20_report": SOURCE_V20_REPORT_SHA256,
        "v19_checkpoint": SOURCE_V19_CHECKPOINT_SHA256,
    }
    if artifact_hashes != expected_artifacts:
        raise RuntimeError("V21-A frozen source artifact bytes changed")
    with Path(v20_report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    source_integrity = report.get("source_integrity")
    if (
        not isinstance(report, dict)
        or report.get("artifact_schema") != "angler.phase6-v20-oml-report.v1"
        or report.get("protocol_id") != v20.PROTOCOL_ID
        or report.get("classification") != SOURCE_V20_CLASSIFICATION
        or report.get("passed") is not True
        or not isinstance(source_integrity, dict)
        or source_integrity.get("terminal_system_digest") != SOURCE_V20_SYSTEM_DIGEST
    ):
        raise RuntimeError("V21-A consumed V20 report state changed")
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "repository_hashes": observed,
        "artifact_hashes": artifact_hashes,
        "source_v20_classification": report["classification"],
        "source_v20_system_digest": source_integrity["terminal_system_digest"],
    }


def _detached_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _fresh_persistent_fast_state(initial_weight: torch.Tensor) -> PersistentFastState:
    _require_finite_tensor("initial fast weight", initial_weight)
    weight = _detached_clone(initial_weight)
    zero = torch.zeros_like(weight)
    return PersistentFastState(
        weight=weight,
        optimizer_state=(
            AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),
        ),
        lifetime_updates=0,
        reset_count=0,
    )


def snapshot_fast_state(state: PersistentFastState) -> dict[str, object]:
    if not isinstance(state, PersistentFastState):
        raise TypeError("V21-A snapshot requires a persistent fast state")
    slot = state.optimizer_state[0]
    return {
        "weight": state.weight.detach().cpu().clone(),
        "optimizer_state": {
            "step": slot.step,
            "exp_avg": slot.exp_avg.detach().cpu().clone(),
            "exp_avg_sq": slot.exp_avg_sq.detach().cpu().clone(),
        },
        "lifetime_updates": state.lifetime_updates,
        "reset_count": state.reset_count,
        "state_digest": persistent_fast_state_digest(state),
    }


def restore_fast_state(
    snapshot: Mapping[str, object],
    *,
    device: torch.device | str,
) -> PersistentFastState:
    required = {
        "weight",
        "optimizer_state",
        "lifetime_updates",
        "reset_count",
        "state_digest",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != required:
        raise RuntimeError("V21-A fast-state snapshot fields are invalid")
    optimizer = snapshot["optimizer_state"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"step", "exp_avg", "exp_avg_sq"}:
        raise RuntimeError("V21-A fast optimizer snapshot fields are invalid")
    selected = torch.device(device)
    state = PersistentFastState(
        weight=torch.as_tensor(snapshot["weight"]).to(selected).detach().clone(),
        optimizer_state=(
            AdamWSlot(
                step=int(optimizer["step"]),
                exp_avg=torch.as_tensor(optimizer["exp_avg"]).to(selected).detach().clone(),
                exp_avg_sq=torch.as_tensor(optimizer["exp_avg_sq"])
                .to(selected)
                .detach()
                .clone(),
            ),
        ),
        lifetime_updates=int(snapshot["lifetime_updates"]),
        reset_count=int(snapshot["reset_count"]),
    )
    if persistent_fast_state_digest(state) != snapshot["state_digest"]:
        raise RuntimeError("V21-A restored fast-state digest changed")
    return state


def persistent_fast_state_digest(state: PersistentFastState) -> str:
    if not isinstance(state, PersistentFastState):
        raise TypeError("V21-A digest requires a persistent fast state")
    slot = state.optimizer_state[0]
    return _object_digest(
        _FAST_STATE_DIGEST_DOMAIN,
        {
            "weight": state.weight,
            "step": slot.step,
            "exp_avg": slot.exp_avg,
            "exp_avg_sq": slot.exp_avg_sq,
            "lifetime_updates": state.lifetime_updates,
            "reset_count": state.reset_count,
        },
    )


def reset_fast_state(
    initial_weight: torch.Tensor,
    *,
    lifetime_updates: int = 0,
    reset_count: int = 0,
) -> PersistentFastState:
    if (
        isinstance(lifetime_updates, bool)
        or not isinstance(lifetime_updates, int)
        or lifetime_updates < 0
        or isinstance(reset_count, bool)
        or not isinstance(reset_count, int)
        or reset_count not in (0, 1)
    ):
        raise ValueError("V21-A reset metadata is invalid")
    weight = _detached_clone(initial_weight)
    zero = torch.zeros_like(weight)
    return PersistentFastState(
        weight=weight,
        optimizer_state=(AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),),
        lifetime_updates=lifetime_updates,
        reset_count=reset_count,
    )


def _freeze_controller(controller: v19.V12ChampionPairedGraphContextController) -> str:
    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V21-A requires the exact V19 controller")
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    controller.eval()
    if any(parameter.requires_grad or parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V21-A failed to freeze a slow controller")
    return v20.oml_controller_digest(controller)


def _arm_learned_digest(arm: PersistentArm) -> str:
    return _object_digest(
        _LEARNED_STATE_DIGEST_DOMAIN,
        {
            "name": arm.name,
            "source_controller_digest": arm.source_controller_digest,
            "initial_weight": arm.initial_weight,
            "state_digest": persistent_fast_state_digest(arm.state),
        },
    )


def persistent_learned_state_digest(system: PersistentLifelongSystem) -> str:
    return _object_digest(
        _LEARNED_STATE_DIGEST_DOMAIN,
        {
            name: _arm_learned_digest(system.arm(name)) for name in UPDATED_ARMS
        },
    )


def _validate_harness_state(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None or len(value) == 0:
        return {}
    required = {
        "claim_sha256",
        "claim_created_utc",
        "identity_deadline_utc",
        "last_identity_age_seconds",
        "publication_cursor",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V21-A harness-state fields are invalid")
    result = dict(value)
    if (
        not isinstance(result["claim_sha256"], str)
        or len(result["claim_sha256"]) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in result["claim_sha256"])
        or not isinstance(result["claim_created_utc"], str)
        or not result["claim_created_utc"]
        or not isinstance(result["identity_deadline_utc"], str)
        or not result["identity_deadline_utc"]
        or isinstance(result["publication_cursor"], bool)
        or not isinstance(result["publication_cursor"], int)
        or not 0 <= result["publication_cursor"] <= TOTAL_EXPERIENCES
    ):
        raise ValueError("V21-A harness-state value is invalid")
    age = _finite_float(result["last_identity_age_seconds"], "identity age")
    if age < 0.0 or age > SEMANTIC_WALL_TIME_CEILING_SECONDS:
        raise ValueError("V21-A identity age is outside its cumulative ceiling")
    result["last_identity_age_seconds"] = age
    return result


def build_persistent_lifelong_system(
    v20_checkpoint: str | Path = SOURCE_V20_CHECKPOINT,
    v19_checkpoint: str | Path = SOURCE_V19_CHECKPOINT,
    *,
    device: torch.device | str = "cpu",
) -> PersistentLifelongSystem:
    """Load the consumed V20 result once, extract controllers, and freeze them."""

    configure_persistent_lifelong_numerics()
    if _sha256_file(v20_checkpoint) != SOURCE_V20_CHECKPOINT_SHA256:
        raise RuntimeError("V21-A V20 checkpoint SHA-256 changed")
    if _sha256_file(v19_checkpoint) != SOURCE_V19_CHECKPOINT_SHA256:
        raise RuntimeError("V21-A V19 source checkpoint SHA-256 changed")
    selected = torch.device(device)
    source = v20.load_oml_checkpoint(
        v20_checkpoint,
        v19_checkpoint,
        device=selected,
    )
    if (
        source.completed_updates != v20.OUTER_UPDATES
        or source.outer_mode != "full"
        or v20.oml_system_digest(source) != SOURCE_V20_SYSTEM_DIGEST
    ):
        raise RuntimeError("V21-A consumed V20 terminal system changed")
    candidate_controller = source.second_order_oml.controller
    first_controller = source.first_order_meta.controller
    source_controller = source.source_v19.controller
    candidate_w0 = _detached_clone(source.second_order_oml.fast_initial_weight)
    first_w0 = _detached_clone(source.first_order_meta.fast_initial_weight)
    source_w0 = _detached_clone(source_controller.relation_comparator[2].weight)
    if not torch.equal(candidate_w0, first_w0) or not torch.equal(candidate_w0, source_w0):
        raise RuntimeError("V21-A matched arms lost their common W0")
    candidate_digest = _freeze_controller(candidate_controller)
    first_digest = _freeze_controller(first_controller)
    source_digest = _freeze_controller(source_controller)
    bindings = {
        "v20_checkpoint_sha256": SOURCE_V20_CHECKPOINT_SHA256,
        "v20_report_sha256": SOURCE_V20_REPORT_SHA256,
        "v20_terminal_system_digest": SOURCE_V20_SYSTEM_DIGEST,
        "v19_checkpoint_sha256": SOURCE_V19_CHECKPOINT_SHA256,
        "candidate_controller_digest": candidate_digest,
        "first_order_controller_digest": first_digest,
        "source_v19_controller_digest": source_digest,
    }
    result = PersistentLifelongSystem(
        second_order_oml_persistent=PersistentArm(
            ARM_CANDIDATE,
            candidate_controller,
            candidate_w0,
            _fresh_persistent_fast_state(candidate_w0),
            candidate_digest,
        ),
        first_order_meta_persistent=PersistentArm(
            ARM_FIRST_ORDER,
            first_controller,
            first_w0,
            _fresh_persistent_fast_state(first_w0),
            first_digest,
        ),
        source_v19_persistent=PersistentArm(
            ARM_SOURCE,
            source_controller,
            source_w0,
            _fresh_persistent_fast_state(source_w0),
            source_digest,
        ),
        second_order_boundary_reset=PersistentArm(
            ARM_BOUNDARY,
            candidate_controller,
            candidate_w0.detach().clone(),
            _fresh_persistent_fast_state(candidate_w0),
            candidate_digest,
        ),
        source_bindings=bindings,
    )
    del source
    _assert_system_integrity(result)
    return result


def _assert_identity_ledger(system: PersistentLifelongSystem) -> None:
    ledger = system.identity_ledger
    roles = {"diagnostic", "update", "probe"}
    if not isinstance(ledger, dict) or set(ledger) != roles or any(
        not isinstance(ledger[role], dict) for role in roles
    ):
        raise RuntimeError("V21-A identity ledger structure changed")
    expected_keys = {
        "diagnostic": (
            {str(index) for index in range(DIAGNOSTIC_STREAMS)}
            if system.gradient_geometry is not None
            else set()
        ),
        "update": {str(index) for index in range(system.next_experience)},
        "probe": (
            {str(index) for index in range(PROBE_STREAMS)} if system.probes else set()
        ),
    }
    completed_probe_boundaries = tuple(
        boundary for boundary in PROBE_BOUNDARIES if boundary in system.probes
    )
    all_values: set[str] = set()
    identity_fields = {
        "schedule_digest",
        "stream_literal_digest",
        "support_package_literal_digests",
        "query_package_literal_digests",
        "public_support_literal_digests",
        "public_query_literal_digests",
        "public_support_canonical_digests",
    }
    for role in ("diagnostic", "update", "probe"):
        records = ledger[role]
        if set(records) != expected_keys[role]:
            raise RuntimeError(f"V21-A {role} identity ledger chronology changed")
        for key, record in records.items():
            expected_record_fields = (
                {"identity", "boundaries"} if role == "probe" else {"identity"}
            )
            if not isinstance(record, Mapping) or set(record) != expected_record_fields:
                raise RuntimeError("V21-A identity ledger record fields changed")
            identity = record.get("identity")
            if not isinstance(identity, Mapping) or set(identity) != identity_fields:
                raise RuntimeError("V21-A materialized identity fields changed")
            if role == "probe" and tuple(record.get("boundaries", ())) != completed_probe_boundaries:
                raise RuntimeError("V21-A probe identity reuse chronology changed")
            expected_lengths = {
                "support_package_literal_digests": 4,
                "query_package_literal_digests": 1,
                "public_support_literal_digests": 4,
                "public_query_literal_digests": 1,
                "public_support_canonical_digests": 4,
            }
            for field_name, length in expected_lengths.items():
                values = identity.get(field_name)
                if not isinstance(values, (tuple, list)) or len(values) != length:
                    raise RuntimeError("V21-A materialized identity cardinality changed")
            digest_values = (
                str(identity.get("schedule_digest")),
                str(identity.get("stream_literal_digest")),
                *tuple(
                    str(item)
                    for field_name in expected_lengths
                    for item in identity[field_name]
                ),
            )
            if any(
                len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
                for value in digest_values
            ):
                raise RuntimeError("V21-A materialized identity digest changed")
            unique_values = set(_identity_record_digest_values(identity))
            if all_values & unique_values:
                raise RuntimeError("V21-A materialized identity disjointness changed")
            all_values.update(unique_values)


def _assert_system_integrity(
    system: PersistentLifelongSystem,
    *,
    allow_boundary_transition: bool = False,
) -> None:
    if not isinstance(system, PersistentLifelongSystem):
        raise TypeError("V21-A requires a PersistentLifelongSystem")
    if (
        type(system.next_experience) is not int
        or not 0 <= system.next_experience <= TOTAL_EXPERIENCES
        or type(system.boundary_reset_applied) is not bool
        or system.source_bindings.get("v20_checkpoint_sha256")
        != SOURCE_V20_CHECKPOINT_SHA256
        or system.source_bindings.get("v20_report_sha256") != SOURCE_V20_REPORT_SHA256
        or system.source_bindings.get("v20_terminal_system_digest")
        != SOURCE_V20_SYSTEM_DIGEST
        or system.source_bindings.get("v19_checkpoint_sha256")
        != SOURCE_V19_CHECKPOINT_SHA256
    ):
        raise RuntimeError("V21-A system identity or cursor changed")
    if system.second_order_boundary_reset.controller is not system.second_order_oml_persistent.controller:
        raise RuntimeError("V21-A candidate and boundary no longer share one frozen controller")
    for name in UPDATED_ARMS:
        arm = system.arm(name)
        if (
            arm.name != name
            or type(arm.controller) is not v19.V12ChampionPairedGraphContextController
            or arm.initial_weight.shape != (1, 64)
            or arm.initial_weight.dtype != torch.float32
            or arm.initial_weight.requires_grad
            or arm.initial_weight.grad_fn is not None
            or not torch.equal(arm.initial_weight, system.second_order_oml_persistent.initial_weight)
            or v20.oml_controller_digest(arm.controller) != arm.source_controller_digest
            or system.source_bindings[
                "candidate_controller_digest"
                if name in (ARM_CANDIDATE, ARM_BOUNDARY)
                else "first_order_controller_digest"
                if name == ARM_FIRST_ORDER
                else "source_v19_controller_digest"
            ]
            != arm.source_controller_digest
            or any(
                parameter.requires_grad or parameter.grad is not None
                for parameter in arm.controller.parameters()
            )
            or arm.controller.training
            or arm.state.weight.device != arm.initial_weight.device
            or arm.state.persistent_float_values != PERSISTENT_FLOAT_VALUES
        ):
            raise RuntimeError(f"V21-A arm integrity changed: {name}")
    cursor = system.next_experience
    for name in (ARM_CANDIDATE, ARM_FIRST_ORDER, ARM_SOURCE):
        state = system.arm(name).state
        if state.lifetime_updates != cursor or state.reset_count != 0 or state.optimizer_state[0].step != cursor:
            raise RuntimeError(f"V21-A persistent counter changed: {name}")
    boundary = system.second_order_boundary_reset.state
    if system.boundary_reset_applied:
        if (
            cursor < STAGE_A_EXPERIENCES
            or boundary.lifetime_updates != cursor
            or boundary.reset_count != 1
            or boundary.optimizer_state[0].step != cursor - STAGE_A_EXPERIENCES
        ):
            raise RuntimeError("V21-A boundary-reset counters changed")
    elif (
        boundary.lifetime_updates != cursor
        or boundary.reset_count != 0
        or boundary.optimizer_state[0].step != cursor
        or cursor > STAGE_A_EXPERIENCES
        or (cursor == STAGE_A_EXPERIENCES and not allow_boundary_transition)
    ):
        raise RuntimeError("V21-A pre-boundary counters changed")
    expected_stage_b = max(0, cursor - STAGE_A_EXPERIENCES)
    if set(system.stage_b_online_pre_loss) != set(MEASUREMENT_CONDITIONS) or any(
        len(system.stage_b_online_pre_loss[name]) != expected_stage_b
        for name in MEASUREMENT_CONDITIONS
    ):
        raise RuntimeError("V21-A Stage-B online-loss chronology changed")
    if set(system.probes) - set(PROBE_BOUNDARIES):
        raise RuntimeError("V21-A probe boundary record is invalid")
    if "end_A" in system.probes and cursor < STAGE_A_EXPERIENCES:
        raise RuntimeError("V21-A end-A probe exists before Stage A completed")
    if "end_B" in system.probes and cursor != TOTAL_EXPERIENCES:
        raise RuntimeError("V21-A end-B probe exists before the terminal cursor")
    _assert_identity_ledger(system)
    _validate_harness_state(system.harness_state)


def persistent_lifelong_system_digest(system: PersistentLifelongSystem) -> str:
    _assert_system_integrity(
        system,
        allow_boundary_transition=(
            system.next_experience == STAGE_A_EXPERIENCES
            and not system.boundary_reset_applied
        ),
    )
    return _object_digest(
        _SYSTEM_DIGEST_DOMAIN,
        {
            "protocol_id": PROTOCOL_ID,
            "plan_digest": persistent_lifelong_plan_digest(),
            "source_bindings": system.source_bindings,
            "next_experience": system.next_experience,
            "boundary_reset_applied": system.boundary_reset_applied,
            "learned_state_digest": persistent_learned_state_digest(system),
            "probes": system.probes,
            "gradient_geometry": system.gradient_geometry,
            "stage_b_online_pre_loss": system.stage_b_online_pre_loss,
            "end_a_exactness": system.end_a_exactness,
            "public_train_parity": system.public_train_parity,
            "identity_ledger": system.identity_ledger,
        },
    )


def persistent_lifelong_checkpoint_summary(
    system: PersistentLifelongSystem,
) -> dict[str, object]:
    _assert_system_integrity(
        system,
        allow_boundary_transition=(
            system.next_experience == STAGE_A_EXPERIENCES
            and not system.boundary_reset_applied
        ),
    )
    cursor = system.next_experience
    stage = "stage_a" if cursor < STAGE_A_EXPERIENCES else "stage_b" if cursor < TOTAL_EXPERIENCES else "complete"
    return {
        "cursor": cursor,
        "stage": stage,
        "end_a_complete": "end_A" in system.probes,
        "end_b_complete": "end_B" in system.probes,
        "boundary_reset_applied": system.boundary_reset_applied,
        "probe_keys": tuple(key for key in PROBE_BOUNDARIES if key in system.probes),
        "system_digest": persistent_lifelong_system_digest(system),
        "learned_state_digest": persistent_learned_state_digest(system),
        "harness_state_digest": _object_digest(
            _HARNESS_STATE_DIGEST_DOMAIN, system.harness_state
        ),
        "harness_state": copy.deepcopy(system.harness_state),
        "identity_ledger_digest": _object_digest(
            _LITERAL_IDENTITY_DIGEST_DOMAIN,
            system.identity_ledger,
        ),
        "identity_ledger_counts": {
            role: len(system.identity_ledger[role])
            for role in ("diagnostic", "update", "probe")
        },
        "arm_counters": {
            name: {
                "lifetime_updates": system.arm(name).state.lifetime_updates,
                "adamw_step": system.arm(name).state.optimizer_state[0].step,
                "reset_count": system.arm(name).state.reset_count,
            }
            for name in UPDATED_ARMS
        },
    }


def _allocated_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _assert_resource_ceiling(system: PersistentLifelongSystem) -> int:
    device = system.second_order_oml_persistent.state.weight.device
    allocated = _allocated_bytes(device)
    if allocated > ALLOCATED_MEMORY_CEILING_BYTES:
        raise RuntimeError("V21-A allocated-memory ceiling exceeded")
    return allocated


def _persistent_step(
    controller: v19.V12ChampionPairedGraphContextController,
    state: PersistentFastState,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> tuple[PersistentFastState, dict[str, object]]:
    """Perform one replay-free online update of the sole 64-value fast weight."""

    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V21-A persistent step requires the exact V19 controller")
    if not isinstance(state, PersistentFastState):
        raise TypeError("V21-A persistent step requires a fast state")
    before_controller = v20.oml_controller_digest(controller)
    fast = state.weight.detach().clone().requires_grad_(True)
    loss = _public_stream_loss(controller, fast, supports)
    gradient = torch.autograd.grad(
        loss,
        (fast,),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    _require_finite_tensor("persistent fast gradient", gradient)
    (updated,), updated_optimizer = functional_adamw_step(
        (fast,),
        (gradient.detach(),),
        state.optimizer_state,
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    next_state = PersistentFastState(
        weight=updated.detach().clone(),
        optimizer_state=tuple(
            AdamWSlot(
                step=slot.step,
                exp_avg=slot.exp_avg.detach().clone(),
                exp_avg_sq=slot.exp_avg_sq.detach().clone(),
            )
            for slot in updated_optimizer
        ),
        lifetime_updates=state.lifetime_updates + 1,
        reset_count=state.reset_count,
    )
    if (
        v20.oml_controller_digest(controller) != before_controller
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in controller.parameters()
        )
    ):
        raise RuntimeError("V21-A persistent step mutated a frozen slow tensor")
    return next_state, {
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient.detach().to(torch.float64).norm().item()),
        "adamw_step": next_state.optimizer_state[0].step,
        "lifetime_updates": next_state.lifetime_updates,
        "state_digest": persistent_fast_state_digest(next_state),
    }


def _condition_arm(
    system: PersistentLifelongSystem,
    condition: str,
) -> tuple[v19.V12ChampionPairedGraphContextController, torch.Tensor]:
    if condition in UPDATED_ARMS:
        arm = system.arm(condition)
        return arm.controller, arm.state.weight
    if condition == CONTROL_SECOND_NO_UPDATE:
        arm = system.second_order_oml_persistent
    elif condition == CONTROL_FIRST_NO_UPDATE:
        arm = system.first_order_meta_persistent
    elif condition == CONTROL_SOURCE_NO_UPDATE:
        arm = system.source_v19_persistent
    else:
        raise KeyError(f"unknown V21-A measurement condition: {condition}")
    return arm.controller, arm.initial_weight


def _probe_member(
    controller: v19.V12ChampionPairedGraphContextController,
    weight: torch.Tensor,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...],
) -> tuple[dict[str, object], tuple[v19.V19PairedGraphCreditRow, ...]]:
    with torch.no_grad():
        rows = _functional_public_rows(controller, weight, supports)
        loss = _public_stream_loss_from_rows(rows)
    metrics = v19._credit_rows_metrics((rows,))
    signature = metrics.pop("relation_signature")
    signed_margins = tuple(
        float((row.positive_margin - row.negative_margin).detach().item()) for row in rows
    )
    result = {
        "loss": float(loss.item()),
        "supported_rows": int(metrics["supported_rows"]),
        "informative_rows": int(metrics["informative_rows"]),
        "qualifying": bool(int(metrics["qualifying_streams"]) == 1),
        "signed_margins": signed_margins,
        "row_digest": _rows_digest(rows),
        "relation_signature_digest": _object_digest(_ROW_DIGEST_DOMAIN, signature),
    }
    if not math.isfinite(result["loss"]) or any(
        not math.isfinite(value) for value in signed_margins
    ):
        raise RuntimeError("V21-A probe member contains a non-finite metric")
    return result, rows


def _aggregate_probe_group(
    member_records: Sequence[Mapping[str, object]],
    row_groups: Sequence[Sequence[v19.V19PairedGraphCreditRow]],
) -> dict[str, object]:
    if not member_records or len(member_records) != len(row_groups):
        raise ValueError("V21-A probe group is incomplete")
    metrics = v19._credit_rows_metrics(row_groups)
    signature = metrics.pop("relation_signature")
    losses = tuple(_finite_float(record["loss"], "probe member loss") for record in member_records)
    return {
        "mean_loss": sum(losses) / len(losses),
        "member_losses": losses,
        "supported_rows": int(metrics["supported_rows"]),
        "informative_rows": int(metrics["informative_rows"]),
        "qualifying_streams": int(metrics["qualifying_streams"]),
        "signed_margins": tuple(
            tuple(_finite_float(value, "probe signed margin") for value in record["signed_margins"])
            for record in member_records
        ),
        "row_digests": tuple(str(record["row_digest"]) for record in member_records),
        "relation_signature_digest": _object_digest(_ROW_DIGEST_DOMAIN, signature),
    }


def _probe_state_guard(system: PersistentLifelongSystem) -> dict[str, object]:
    return {
        "learned": persistent_learned_state_digest(system),
        "controllers": {
            key: v20.oml_controller_digest(system.arm(key).controller)
            for key in (ARM_CANDIDATE, ARM_FIRST_ORDER, ARM_SOURCE)
        },
    }


def evaluate_probe_boundary(
    system: PersistentLifelongSystem,
    boundary: str,
    *,
    stream_factory: StreamFactory | None = None,
    deadline_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Evaluate the one immutable 80-stream cohort without changing learned state."""

    expected_cursor = {"pre": 0, "end_A": STAGE_A_EXPERIENCES, "end_B": TOTAL_EXPERIENCES}
    if boundary not in expected_cursor or system.next_experience != expected_cursor[boundary]:
        raise ValueError("V21-A probe boundary does not match the chronology")
    if boundary in system.probes:
        raise RuntimeError("V21-A probe boundary was already consumed")
    if boundary == "end_A" and system.boundary_reset_applied:
        raise RuntimeError("V21-A end-A probe must precede the sole boundary reset")
    if boundary == "end_B" and not system.boundary_reset_applied:
        raise RuntimeError("V21-A end-B probe requires the boundary reset")
    _assert_system_integrity(
        system,
        allow_boundary_transition=(boundary == "end_A"),
    )
    before = _probe_state_guard(system)
    conditions: dict[str, dict[str, list[object]]] = {
        name: {
            group: [] for group in PROBE_GROUPS
        }
        for name in MEASUREMENT_CONDITIONS
    }
    row_groups: dict[str, dict[str, list[tuple[v19.V19PairedGraphCreditRow, ...]]]] = {
        name: {group: [] for group in PROBE_GROUPS} for name in MEASUREMENT_CONDITIONS
    }
    causal_records: dict[str, dict[str, list[object]]] = {}
    causal_rows: dict[str, dict[str, list[tuple[v19.V19PairedGraphCreditRow, ...]]]] = {}
    cohort_public_digests: list[tuple[str, ...]] = []
    clean_controller = None
    clean_controller_digest_exact = False
    swapped_state = None
    terminal_reset_state = None
    if boundary == "end_B":
        clean_controller = copy.deepcopy(system.second_order_oml_persistent.controller)
        clean_controller_digest_exact = (
            _freeze_controller(clean_controller)
            == system.second_order_oml_persistent.source_controller_digest
        )
        if not clean_controller_digest_exact:
            raise RuntimeError("V21-A clean swap controller differs from candidate RLN")
        swapped_state = restore_fast_state(
            snapshot_fast_state(system.second_order_oml_persistent.state),
            device=system.second_order_oml_persistent.state.weight.device,
        )
        terminal_reset_state = reset_fast_state(
            system.second_order_oml_persistent.initial_weight
        )
        for name in ("candidate_state_swap", "candidate_w0_reset"):
            causal_records[name] = {group: [] for group in PROBE_GROUPS}
            causal_rows[name] = {group: [] for group in PROBE_GROUPS}

    for record in _probe_specs():
        if deadline_callback is not None:
            deadline_callback()
        group = str(record["group"])
        supports = _materialize_public_supports(
            record,
            stream_factory,
            system=system,
            probe_boundary=boundary,
        )
        cohort_public_digests.append(
            tuple(_literal_identity_digest(task) for task in supports)
        )
        for condition in MEASUREMENT_CONDITIONS:
            controller, weight = _condition_arm(system, condition)
            member, rows = _probe_member(controller, weight, supports)
            conditions[condition][group].append(member)
            row_groups[condition][group].append(rows)
        if clean_controller is not None:
            assert swapped_state is not None and terminal_reset_state is not None
            for name, weight in (
                ("candidate_state_swap", swapped_state.weight),
                ("candidate_w0_reset", terminal_reset_state.weight),
            ):
                member, rows = _probe_member(clean_controller, weight, supports)
                causal_records[name][group].append(member)
                causal_rows[name][group].append(rows)
        if deadline_callback is not None:
            deadline_callback()

    result_conditions = {
        condition: {
            group: _aggregate_probe_group(
                conditions[condition][group], row_groups[condition][group]
            )
            for group in PROBE_GROUPS
        }
        for condition in MEASUREMENT_CONDITIONS
    }
    result: dict[str, object] = {
        "boundary": boundary,
        "cursor": system.next_experience,
        "cohort_plan_digest": persistent_lifelong_plan_digest(),
        "cohort_digest_mode": "order_sensitive_literal_dataclass_fields_v1",
        "cohort_public_digests": tuple(cohort_public_digests),
        "cohort_bytes_exact": True,
        "conditions": result_conditions,
        "learned_state_preserved": False,
        "slow_state_preserved": False,
    }
    if "pre" in system.probes:
        result["cohort_bytes_exact"] = (
            tuple(system.probes["pre"].get("cohort_public_digests", ()))
            == result["cohort_public_digests"]
        )
        if not result["cohort_bytes_exact"]:
            raise RuntimeError("V21-A immutable probe cohort bytes changed across boundaries")
    if clean_controller is not None:
        assert swapped_state is not None and terminal_reset_state is not None
        causal_aggregates = {
            name: {
                group: _aggregate_probe_group(
                    causal_records[name][group], causal_rows[name][group]
                )
                for group in PROBE_GROUPS
            }
            for name in causal_records
        }
        swap_exact = causal_aggregates["candidate_state_swap"] == result_conditions[ARM_CANDIDATE]
        reset_exact = causal_aggregates["candidate_w0_reset"] == result_conditions[
            CONTROL_SECOND_NO_UPDATE
        ]
        swap_state_digest_exact = persistent_fast_state_digest(swapped_state) == persistent_fast_state_digest(
            system.second_order_oml_persistent.state
        )
        reset_slot = terminal_reset_state.optimizer_state[0]
        reset_state_valid = (
            terminal_reset_state.lifetime_updates == 0
            and terminal_reset_state.reset_count == 0
            and reset_slot.step == 0
            and torch.equal(
                terminal_reset_state.weight,
                system.second_order_oml_persistent.initial_weight,
            )
            and bool(torch.count_nonzero(reset_slot.exp_avg).item()) is False
            and bool(torch.count_nonzero(reset_slot.exp_avg_sq).item()) is False
        )
        result["terminal_causal"] = {
            "state_swap_exact": swap_exact,
            "state_swap_digest_exact": swap_state_digest_exact,
            "clean_controller_digest_exact": clean_controller_digest_exact,
            "w0_reset_exact": reset_exact,
            "reset_state_valid": reset_state_valid,
            "candidate_state_digest": persistent_fast_state_digest(
                system.second_order_oml_persistent.state
            ),
            "swapped_state_digest": persistent_fast_state_digest(swapped_state),
            "reset_state_digest": persistent_fast_state_digest(terminal_reset_state),
            "reset_state_counters": {
                "lifetime_updates": terminal_reset_state.lifetime_updates,
                "adamw_step": reset_slot.step,
                "reset_count": terminal_reset_state.reset_count,
            },
            "state_swap": causal_aggregates["candidate_state_swap"],
            "w0_reset": causal_aggregates["candidate_w0_reset"],
        }
    after = _probe_state_guard(system)
    result["learned_state_preserved"] = before["learned"] == after["learned"]
    result["slow_state_preserved"] = before["controllers"] == after["controllers"]
    if not result["learned_state_preserved"] or not result["slow_state_preserved"]:
        raise RuntimeError("V21-A probe mutated learned or frozen state")
    system.probes[boundary] = result
    _assert_resource_ceiling(system)
    return copy.deepcopy(result)


def apply_persistent_experience(
    system: PersistentLifelongSystem,
    experience_index: int,
    *,
    supports: tuple[v12.PublicSoftwarePipelineTask, ...] | None = None,
    stream_factory: StreamFactory | None = None,
) -> dict[str, object]:
    """Apply exactly one frozen experience to all four persistent arms."""

    if type(experience_index) is not int or experience_index != system.next_experience:
        raise ValueError("V21-A experience does not match the persistent cursor")
    if not 0 <= experience_index < TOTAL_EXPERIENCES:
        raise ValueError("V21-A persistent fit is already complete")
    if experience_index >= STAGE_A_EXPERIENCES and not system.boundary_reset_applied:
        raise RuntimeError("V21-A Stage B cannot begin before the ordered boundary reset")
    _assert_system_integrity(system)
    record = _experience_spec(experience_index)
    if supports is None:
        stream = (stream_factory or _make_protocol_stream)(record)
        supports = _public_supports_from_stream(stream, record)
        _register_stream_identity(system, record, stream, supports)
        if experience_index == 0:
            system.public_train_parity = public_train_parity_report(
                system.second_order_oml_persistent.controller,
                stream,
            )
    elif type(supports) is not tuple or len(supports) != 4 or any(
        type(task) is not v12.PublicSoftwarePipelineTask for task in supports
    ):
        raise TypeError("V21-A injected supports must be an exact four-task public tuple")

    online_pre: dict[str, float] = {}
    if experience_index >= STAGE_A_EXPERIENCES:
        for condition in MEASUREMENT_CONDITIONS:
            controller, weight = _condition_arm(system, condition)
            with torch.no_grad():
                value = float(_public_stream_loss(controller, weight, supports).item())
            if not math.isfinite(value):
                raise RuntimeError("V21-A Stage-B online loss is non-finite")
            online_pre[condition] = value

    diagnostics: dict[str, object] = {}
    for name in UPDATED_ARMS:
        arm = system.arm(name)
        arm.state, diagnostics[name] = _persistent_step(arm.controller, arm.state, supports)
    system.next_experience += 1
    if experience_index >= STAGE_A_EXPERIENCES:
        for condition in MEASUREMENT_CONDITIONS:
            system.stage_b_online_pre_loss[condition].append(online_pre[condition])

    if system.next_experience == STAGE_A_EXPERIENCES:
        left = system.second_order_oml_persistent.state
        right = system.second_order_boundary_reset.state
        exact = (
            persistent_fast_state_digest(left) == persistent_fast_state_digest(right)
            and torch.equal(left.weight, right.weight)
            and left.optimizer_state[0].step == right.optimizer_state[0].step
            and torch.equal(left.optimizer_state[0].exp_avg, right.optimizer_state[0].exp_avg)
            and torch.equal(left.optimizer_state[0].exp_avg_sq, right.optimizer_state[0].exp_avg_sq)
        )
        if not exact:
            raise RuntimeError("V21-A candidate/boundary arms diverged before the reset")
        system.end_a_exactness = {
            "state_exact": True,
            "candidate_state_digest": persistent_fast_state_digest(left),
            "boundary_state_digest": persistent_fast_state_digest(right),
        }
        _assert_system_integrity(system, allow_boundary_transition=True)
    else:
        _assert_system_integrity(system)
    allocated = _assert_resource_ceiling(system)
    return {
        "experience_index": experience_index,
        "cursor": system.next_experience,
        "stage": record["stage"],
        "commitment_index": record["commitment_index"],
        "arm_diagnostics": diagnostics,
        "online_pre_loss": online_pre,
        "system_digest": persistent_lifelong_system_digest(system),
        "allocated_bytes": allocated,
    }


def _cosine_summary(vectors: torch.Tensor, indices: Sequence[int]) -> dict[str, object]:
    values = []
    selected = tuple(indices)
    for position, left_index in enumerate(selected):
        left = vectors[left_index]
        left_norm = left.norm()
        for right_index in selected[position + 1 :]:
            right = vectors[right_index]
            denominator = left_norm * right.norm()
            cosine = (
                float(torch.dot(left, right).div(denominator).item())
                if float(denominator.item()) > 0.0
                else 0.0
            )
            values.append(cosine)
    return _distribution_summary(values)


def _between_cosine_summary(
    vectors: torch.Tensor,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
) -> dict[str, object]:
    values = []
    for left_index in left_indices:
        left = vectors[left_index]
        left_norm = left.norm()
        for right_index in right_indices:
            right = vectors[right_index]
            denominator = left_norm * right.norm()
            values.append(
                float(torch.dot(left, right).div(denominator).item())
                if float(denominator.item()) > 0.0
                else 0.0
            )
    return _distribution_summary(values)


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    finite = tuple(_finite_float(value, "gradient cosine") for value in values)
    if not finite:
        return {
            "count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
            "negative_fraction": 0.0,
            "values": (),
        }
    return {
        "count": len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "mean": sum(finite) / len(finite),
        "negative_fraction": sum(value < 0.0 for value in finite) / len(finite),
        "values": finite,
    }


def run_initial_gradient_geometry(
    system: PersistentLifelongSystem,
    *,
    stream_factory: StreamFactory | None = None,
    deadline_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Measure the frozen descriptive 56-by-64 W0 gradient geometry once."""

    if system.next_experience != 0 or system.gradient_geometry is not None:
        raise RuntimeError("V21-A gradient geometry is a one-time pre-update diagnostic")
    _assert_system_integrity(system)
    before = _probe_state_guard(system)
    arm = system.second_order_oml_persistent
    gradients = []
    for record in _diagnostic_specs():
        if deadline_callback is not None:
            deadline_callback()
        supports = _materialize_public_supports(
            record,
            stream_factory,
            system=system,
        )
        fast = arm.initial_weight.detach().clone().requires_grad_(True)
        loss = _public_stream_loss(arm.controller, fast, supports)
        gradient = torch.autograd.grad(
            loss,
            (fast,),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        _require_finite_tensor("diagnostic gradient", gradient)
        gradients.append(gradient.detach().reshape(-1).to(torch.float32).cpu())
        if deadline_callback is not None:
            deadline_callback()
    matrix = torch.stack(gradients)
    if matrix.shape != (DIAGNOSTIC_STREAMS, 64):
        raise RuntimeError("V21-A diagnostic matrix shape changed")
    singular_values = torch.linalg.svdvals(matrix.to(torch.float64))
    largest = float(singular_values.max().item()) if singular_values.numel() else 0.0
    tolerance = max(matrix.shape) * torch.finfo(torch.float32).eps * largest
    rank = int((singular_values > tolerance).sum().item())
    norms = matrix.to(torch.float64).norm(dim=1)
    normalized = matrix.to(torch.float64) / norms.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(1)
    cosine = normalized @ normalized.T
    upper = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
    all_cosines = tuple(float(value) for value in upper.tolist())
    stage_a_indices = tuple(range(48))
    stage_b_indices = tuple(range(48, 56))
    result = {
        "matrix_shape": (DIAGNOSTIC_STREAMS, 64),
        "matrix": tuple(tuple(float(value) for value in row.tolist()) for row in matrix),
        "matrix_digest": _object_digest(_ROW_DIGEST_DOMAIN, matrix),
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_values": tuple(float(value) for value in singular_values.tolist()),
        "gradient_norms": tuple(float(value) for value in norms.tolist()),
        "zero_gradient_count": int((norms == 0.0).sum().item()),
        "pairwise_cosines": _distribution_summary(all_cosines),
        "stage_a_within": _cosine_summary(matrix.to(torch.float64), stage_a_indices),
        "stage_b_within": _cosine_summary(matrix.to(torch.float64), stage_b_indices),
        "stage_a_stage_b_between": _between_cosine_summary(
            matrix.to(torch.float64), stage_a_indices, stage_b_indices
        ),
        "descriptive_only": True,
        "learned_state_preserved": False,
        "slow_state_preserved": False,
    }
    after = _probe_state_guard(system)
    result["learned_state_preserved"] = before["learned"] == after["learned"]
    result["slow_state_preserved"] = before["controllers"] == after["controllers"]
    if not result["learned_state_preserved"] or not result["slow_state_preserved"]:
        raise RuntimeError("V21-A gradient diagnostic mutated system state")
    system.gradient_geometry = result
    _assert_resource_ceiling(system)
    return copy.deepcopy(result)


def _apply_boundary_reset(system: PersistentLifelongSystem) -> dict[str, object]:
    if (
        system.next_experience != STAGE_A_EXPERIENCES
        or system.boundary_reset_applied
        or "end_A" not in system.probes
        or not isinstance(system.end_a_exactness, Mapping)
        or system.end_a_exactness.get("state_exact") is not True
    ):
        raise RuntimeError("V21-A boundary reset is out of order")
    candidate_metrics = system.probes["end_A"]["conditions"][ARM_CANDIDATE]
    boundary_metrics = system.probes["end_A"]["conditions"][ARM_BOUNDARY]
    metrics_exact = candidate_metrics == boundary_metrics
    if not metrics_exact:
        raise RuntimeError("V21-A candidate/boundary end-A metrics differ")
    system.end_a_exactness = {
        **dict(system.end_a_exactness),
        "metrics_exact": True,
        "candidate_probe_digest": _object_digest(_ROW_DIGEST_DOMAIN, candidate_metrics),
        "boundary_probe_digest": _object_digest(_ROW_DIGEST_DOMAIN, boundary_metrics),
    }
    arm = system.second_order_boundary_reset
    arm.state = reset_fast_state(
        arm.initial_weight,
        lifetime_updates=STAGE_A_EXPERIENCES,
        reset_count=1,
    )
    system.boundary_reset_applied = True
    _assert_system_integrity(system)
    return {
        "cursor": system.next_experience,
        "boundary_reset_applied": True,
        "boundary_state_digest": persistent_fast_state_digest(arm.state),
        "candidate_boundary_exact_through_end_a": True,
    }


ProgressCallback = Callable[[PersistentLifelongSystem, Mapping[str, object]], None]
DeadlineCallback = Callable[[], None]


def _invoke_deadline(callback: DeadlineCallback | None) -> None:
    if callback is not None:
        callback()


def _progress_event(system: PersistentLifelongSystem) -> dict[str, object]:
    summary = persistent_lifelong_checkpoint_summary(system)
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "cursor": summary["cursor"],
        "stage": summary["stage"],
        "end_a_complete": summary["end_a_complete"],
        "end_b_complete": summary["end_b_complete"],
        "boundary_reset_applied": summary["boundary_reset_applied"],
        "system_digest": summary["system_digest"],
        "allocated_bytes": _assert_resource_ceiling(system),
    }


def fit_persistent_lifelong(
    system: PersistentLifelongSystem,
    *,
    progress_callback: ProgressCallback | None = None,
    deadline_callback: DeadlineCallback | None = None,
    stream_factory: StreamFactory | None = None,
) -> dict[str, object]:
    """Resume and complete the single frozen 256-experience chronology."""

    _assert_system_integrity(system)
    start_cursor = system.next_experience
    emitted: list[int] = []
    if system.gradient_geometry is None:
        if system.next_experience != 0 or system.probes:
            raise RuntimeError("V21-A resume is missing its initial diagnostic")
        _invoke_deadline(deadline_callback)
        run_initial_gradient_geometry(
            system,
            stream_factory=stream_factory,
            deadline_callback=deadline_callback,
        )
    if "pre" not in system.probes:
        if system.next_experience != 0:
            raise RuntimeError("V21-A resume is missing its pre probe")
        _invoke_deadline(deadline_callback)
        evaluate_probe_boundary(
            system,
            "pre",
            stream_factory=stream_factory,
            deadline_callback=deadline_callback,
        )

    while system.next_experience < TOTAL_EXPERIENCES:
        if system.next_experience == STAGE_A_EXPERIENCES:
            if not system.boundary_reset_applied:
                _invoke_deadline(deadline_callback)
                if "end_A" not in system.probes:
                    evaluate_probe_boundary(
                        system,
                        "end_A",
                        stream_factory=stream_factory,
                        deadline_callback=deadline_callback,
                    )
                _apply_boundary_reset(system)
                if progress_callback is not None:
                    progress_callback(system, _progress_event(system))
                emitted.append(STAGE_A_EXPERIENCES)
            elif "end_A" not in system.probes:
                raise RuntimeError("V21-A reset checkpoint is missing its end-A probe")
        _invoke_deadline(deadline_callback)
        apply_persistent_experience(
            system,
            system.next_experience,
            stream_factory=stream_factory,
        )
        _invoke_deadline(deadline_callback)
        cursor = system.next_experience
        if cursor in PROGRESS_CURSORS and cursor != STAGE_A_EXPERIENCES:
            if progress_callback is not None:
                progress_callback(system, _progress_event(system))
            emitted.append(cursor)

    if "end_A" not in system.probes or not system.boundary_reset_applied:
        raise RuntimeError("V21-A terminal state lacks its boundary transition")
    if "end_B" not in system.probes:
        _invoke_deadline(deadline_callback)
        evaluate_probe_boundary(
            system,
            "end_B",
            stream_factory=stream_factory,
            deadline_callback=deadline_callback,
        )
    _invoke_deadline(deadline_callback)
    _assert_system_integrity(system)
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "start_cursor": start_cursor,
        "terminal_cursor": system.next_experience,
        "system_digest": persistent_lifelong_system_digest(system),
        "progress_cursors": tuple(emitted),
        "allocated_bytes": _assert_resource_ceiling(system),
    }


def _checkpoint_payload(system: PersistentLifelongSystem) -> dict[str, object]:
    _assert_system_integrity(system)
    cuda_rng = (
        tuple(torch.cuda.get_rng_state(index).cpu() for index in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else ()
    )
    selected_device = system.second_order_oml_persistent.state.weight.device
    if selected_device.type == "cuda" and (
        torch.cuda.device_count() != 1 or len(cuda_rng) != 1
    ):
        raise RuntimeError("V21-A semantic checkpoint requires exactly one CUDA RNG state")
    harness_state = _validate_harness_state(system.harness_state)
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "source_bindings": copy.deepcopy(system.source_bindings),
        "next_experience": system.next_experience,
        "boundary_reset_applied": system.boundary_reset_applied,
        "arms": {name: snapshot_fast_state(system.arm(name).state) for name in UPDATED_ARMS},
        "probes": copy.deepcopy(system.probes),
        "gradient_geometry": copy.deepcopy(system.gradient_geometry),
        "stage_b_online_pre_loss": copy.deepcopy(system.stage_b_online_pre_loss),
        "end_a_exactness": copy.deepcopy(system.end_a_exactness),
        "public_train_parity": copy.deepcopy(system.public_train_parity),
        "identity_ledger": copy.deepcopy(system.identity_ledger),
        "harness_state": harness_state,
        "harness_state_digest": _object_digest(_HARNESS_STATE_DIGEST_DOMAIN, harness_state),
        "cpu_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_states": cuda_rng,
        "system_digest": persistent_lifelong_system_digest(system),
    }
    payload["checkpoint_digest"] = _object_digest(_CHECKPOINT_DIGEST_DOMAIN, payload)
    return payload


def save_persistent_lifelong_checkpoint(
    path: str | Path,
    system: PersistentLifelongSystem,
    *,
    harness_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write exactly the supplied path; the harness owns outer atomic replacement."""

    if harness_state is not None:
        system.harness_state = _validate_harness_state(harness_state)
    else:
        system.harness_state = _validate_harness_state(system.harness_state)
    if system.harness_state and system.harness_state["publication_cursor"] != system.next_experience:
        raise RuntimeError("V21-A checkpoint publication cursor differs from system cursor")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"V21-A checkpoint target already exists: {target}")
    torch.save(_checkpoint_payload(system), target)
    size = target.stat().st_size
    if size > CHECKPOINT_SIZE_CEILING_BYTES:
        raise RuntimeError("V21-A checkpoint exceeds its 16 MiB ceiling")
    return {
        "path": str(target),
        "bytes": size,
        "sha256": _sha256_file(target),
        "cursor": system.next_experience,
        "system_digest": persistent_lifelong_system_digest(system),
    }


def load_persistent_lifelong_checkpoint(
    path: str | Path,
    v20_checkpoint: str | Path = SOURCE_V20_CHECKPOINT,
    v19_checkpoint: str | Path = SOURCE_V19_CHECKPOINT,
    *,
    device: torch.device | str = "cpu",
) -> PersistentLifelongSystem:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected = {
        "version",
        "protocol_id",
        "plan_digest",
        "source_bindings",
        "next_experience",
        "boundary_reset_applied",
        "arms",
        "probes",
        "gradient_geometry",
        "stage_b_online_pre_loss",
        "end_a_exactness",
        "public_train_parity",
        "identity_ledger",
        "harness_state",
        "harness_state_digest",
        "cpu_rng_state",
        "cuda_rng_states",
        "system_digest",
        "checkpoint_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError("V21-A checkpoint fields are invalid")
    observed_checkpoint_digest = payload.pop("checkpoint_digest")
    if _object_digest(_CHECKPOINT_DIGEST_DOMAIN, payload) != observed_checkpoint_digest:
        raise RuntimeError("V21-A checkpoint content digest changed")
    payload["checkpoint_digest"] = observed_checkpoint_digest
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"] != persistent_lifelong_plan_digest()
        or payload["source_bindings"].get("v20_checkpoint_sha256")
        != SOURCE_V20_CHECKPOINT_SHA256
        or payload["source_bindings"].get("v19_checkpoint_sha256")
        != SOURCE_V19_CHECKPOINT_SHA256
    ):
        raise RuntimeError("V21-A checkpoint identity changed")
    system = build_persistent_lifelong_system(
        v20_checkpoint,
        v19_checkpoint,
        device=device,
    )
    if system.source_bindings != payload["source_bindings"]:
        raise RuntimeError("V21-A checkpoint frozen source binding changed")
    arms = payload["arms"]
    if not isinstance(arms, Mapping) or set(arms) != set(UPDATED_ARMS):
        raise RuntimeError("V21-A checkpoint arm records are invalid")
    for name in UPDATED_ARMS:
        system.arm(name).state = restore_fast_state(arms[name], device=device)
    cursor = payload["next_experience"]
    reset = payload["boundary_reset_applied"]
    if type(cursor) is not int or not 0 <= cursor <= TOTAL_EXPERIENCES or type(reset) is not bool:
        raise RuntimeError("V21-A checkpoint cursor fields are invalid")
    if cursor == STAGE_A_EXPERIENCES and not reset:
        raise RuntimeError("V21-A cursor-192 checkpoint must follow the sole reset")
    system.next_experience = cursor
    system.boundary_reset_applied = reset
    system.probes = copy.deepcopy(payload["probes"])
    system.gradient_geometry = copy.deepcopy(payload["gradient_geometry"])
    system.stage_b_online_pre_loss = copy.deepcopy(payload["stage_b_online_pre_loss"])
    system.end_a_exactness = copy.deepcopy(payload["end_a_exactness"])
    system.public_train_parity = copy.deepcopy(payload["public_train_parity"])
    system.identity_ledger = copy.deepcopy(payload["identity_ledger"])
    system.harness_state = _validate_harness_state(payload["harness_state"])
    if system.harness_state and system.harness_state["publication_cursor"] != system.next_experience:
        raise RuntimeError("V21-A restored publication cursor differs from system cursor")
    if (
        _object_digest(_HARNESS_STATE_DIGEST_DOMAIN, system.harness_state)
        != payload["harness_state_digest"]
    ):
        raise RuntimeError("V21-A checkpoint harness-state digest changed")
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    cuda_rng = tuple(payload["cuda_rng_states"])
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or len(cuda_rng) != 1
        ):
            raise RuntimeError("V21-A checkpoint CUDA RNG topology changed")
        torch.cuda.set_rng_state(cuda_rng[0].cpu(), 0)
    elif len(cuda_rng) not in (0, 1):
        raise RuntimeError("V21-A checkpoint CUDA RNG topology changed")
    _assert_system_integrity(system)
    if persistent_lifelong_system_digest(system) != payload["system_digest"]:
        raise RuntimeError("V21-A checkpoint system digest changed")
    return system


def validate_persistent_lifelong_checkpoint(
    path: str | Path,
    v20_checkpoint: str | Path = SOURCE_V20_CHECKPOINT,
    v19_checkpoint: str | Path = SOURCE_V19_CHECKPOINT,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    system = load_persistent_lifelong_checkpoint(
        path,
        v20_checkpoint,
        v19_checkpoint,
        device=device,
    )
    result = persistent_lifelong_checkpoint_summary(system)
    result["checkpoint_sha256"] = _sha256_file(path)
    result["checkpoint_bytes"] = Path(path).stat().st_size
    return result


def _online_auc(losses: Sequence[float]) -> float:
    if isinstance(losses, (str, bytes, bytearray)) or len(losses) != STAGE_B_EXPERIENCES:
        raise ValueError("V21-A online AUC requires exactly 64 losses")
    values = tuple(_finite_float(value, "Stage-B online loss") for value in losses)
    if any(value < 0.0 for value in values):
        raise ValueError("V21-A online losses must be nonnegative")
    result = (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1]) / 63.0
    if not math.isfinite(result):
        raise ValueError("V21-A online AUC is non-finite")
    return result


def _paired_counts(
    left: Sequence[float],
    right: Sequence[float],
    *,
    ratio: float = 1.0,
    strict: bool = False,
) -> int:
    if type(strict) is not bool or len(left) != len(right) or len(left) == 0:
        raise ValueError("V21-A paired comparison shape changed")
    selected_ratio = _finite_float(ratio, "paired ratio")
    if selected_ratio <= 0.0:
        raise ValueError("V21-A paired ratio must be positive")
    left_values = tuple(_finite_float(value, "paired left loss") for value in left)
    right_values = tuple(_finite_float(value, "paired right loss") for value in right)
    if strict:
        return sum(a < selected_ratio * b for a, b in zip(left_values, right_values, strict=True))
    return sum(a <= selected_ratio * b for a, b in zip(left_values, right_values, strict=True))


def _probe_metric(
    probes: Mapping[str, object],
    boundary: str,
    condition: str,
    group: str,
) -> Mapping[str, object]:
    try:
        record = probes[boundary]["conditions"][condition][group]
    except (KeyError, TypeError) as error:
        raise ValueError("V21-A probe metric is missing") from error
    required = {"mean_loss", "member_losses", "supported_rows", "qualifying_streams"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ValueError("V21-A probe metric fields are incomplete")
    expected = PROBE_GROUP_SIZES[group]
    losses = record["member_losses"]
    if not isinstance(losses, (tuple, list)) or len(losses) != expected:
        raise ValueError("V21-A probe member count changed")
    _finite_float(record["mean_loss"], "probe mean loss")
    tuple(_finite_float(value, "probe member loss") for value in losses)
    rows = record["supported_rows"]
    streams = record["qualifying_streams"]
    if (
        type(rows) is not int
        or not 0 <= rows <= 4 * expected
        or type(streams) is not int
        or not 0 <= streams <= expected
    ):
        raise ValueError("V21-A probe coverage is outside its literal bounds")
    return record


def _terminal_comparisons(system: PersistentLifelongSystem) -> dict[str, object]:
    _assert_system_integrity(system)
    if system.next_experience != TOTAL_EXPERIENCES or set(system.probes) != set(PROBE_BOUNDARIES):
        raise RuntimeError("V21-A terminal comparisons require a complete chronology")
    auc = {
        condition: _online_auc(system.stage_b_online_pre_loss[condition])
        for condition in MEASUREMENT_CONDITIONS
    }
    matched = {
        ARM_CANDIDATE: CONTROL_SECOND_NO_UPDATE,
        ARM_FIRST_ORDER: CONTROL_FIRST_NO_UPDATE,
        ARM_SOURCE: CONTROL_SOURCE_NO_UPDATE,
        ARM_BOUNDARY: CONTROL_SECOND_NO_UPDATE,
    }
    normalized_gain = {}
    for arm, control in matched.items():
        denominator = auc[control]
        if denominator <= 0.0:
            raise ValueError("V21-A normalized fast gain has a zero denominator")
        normalized_gain[arm] = 1.0 - auc[arm] / denominator

    paired_better: dict[str, int] = {}
    paired_ratio: dict[str, int] = {}
    for boundary in PROBE_BOUNDARIES:
        for group in PROBE_GROUPS:
            for left, right in (
                (ARM_CANDIDATE, CONTROL_SECOND_NO_UPDATE),
                (ARM_CANDIDATE, ARM_BOUNDARY),
            ):
                left_losses = _probe_metric(system.probes, boundary, left, group)["member_losses"]
                right_losses = _probe_metric(system.probes, boundary, right, group)["member_losses"]
                key = f"{left}|{right}|{group}|{boundary}"
                paired_better[key] = _paired_counts(left_losses, right_losses, strict=True)
                paired_ratio[key + "|1.05"] = _paired_counts(
                    left_losses, right_losses, ratio=1.05
                )
    end_a_losses = _probe_metric(system.probes, "end_A", ARM_CANDIDATE, "stage_a")[
        "member_losses"
    ]
    end_b_losses = _probe_metric(system.probes, "end_B", ARM_CANDIDATE, "stage_a")[
        "member_losses"
    ]
    retained_count = _paired_counts(end_b_losses, end_a_losses, ratio=1.05)
    no_update_end_a = _finite_float(
        _probe_metric(system.probes, "end_A", CONTROL_SECOND_NO_UPDATE, "stage_a")[
            "mean_loss"
        ],
        "no-update end-A Stage-A loss",
    )
    candidate_end_a = _finite_float(
        _probe_metric(system.probes, "end_A", ARM_CANDIDATE, "stage_a")["mean_loss"],
        "candidate end-A Stage-A loss",
    )
    no_update_end_b = _finite_float(
        _probe_metric(system.probes, "end_B", CONTROL_SECOND_NO_UPDATE, "stage_a")[
            "mean_loss"
        ],
        "no-update end-B Stage-A loss",
    )
    candidate_end_b = _finite_float(
        _probe_metric(system.probes, "end_B", ARM_CANDIDATE, "stage_a")["mean_loss"],
        "candidate end-B Stage-A loss",
    )
    improvement_end_a = no_update_end_a - candidate_end_a
    improvement_end_b = no_update_end_b - candidate_end_b
    retained_fraction = (
        improvement_end_b / improvement_end_a if improvement_end_a > 0.0 else None
    )
    return {
        "auc": auc,
        "normalized_fast_gain": normalized_gain,
        "matched_no_update": matched,
        "probes": copy.deepcopy(system.probes),
        "paired_better": paired_better,
        "paired_ratio": paired_ratio,
        "stage_a_retention": {
            "improvement_end_a": improvement_end_a,
            "improvement_end_b": improvement_end_b,
            "denominator_valid": improvement_end_a > 0.0,
            "retained_fraction": retained_fraction,
            "paired_retained_1.05": retained_count,
        },
    }


def _compute_gates(comparisons: Mapping[str, object]) -> dict[str, bool]:
    """Compute the seven frozen causal gates from terminal numeric evidence."""

    if not isinstance(comparisons, Mapping):
        raise TypeError("V21-A gates require a comparison mapping")
    try:
        auc = comparisons["auc"]
        gain = comparisons["normalized_fast_gain"]
        probes = comparisons["probes"]
        paired_better = comparisons["paired_better"]
        paired_ratio = comparisons["paired_ratio"]
        retention = comparisons["stage_a_retention"]
    except KeyError as error:
        raise ValueError("V21-A gate evidence is incomplete") from error
    if not all(isinstance(value, Mapping) for value in (auc, gain, probes, paired_better, paired_ratio, retention)):
        raise ValueError("V21-A gate evidence has an invalid structure")

    def finite(mapping: Mapping[str, object], key: str) -> float:
        if key not in mapping:
            raise ValueError(f"V21-A gate operand is missing: {key}")
        return _finite_float(mapping[key], key)

    def cell(boundary: str, condition: str, group: str) -> Mapping[str, object]:
        return _probe_metric(probes, boundary, condition, group)

    def coverage_delta(left: Mapping[str, object], right: Mapping[str, object], rows: int, streams: int) -> bool:
        return (
            int(left["supported_rows"]) >= int(right["supported_rows"]) + rows
            or int(left["qualifying_streams"]) >= int(right["qualifying_streams"]) + streams
        )

    candidate_auc = finite(auc, ARM_CANDIDATE)
    second_no_auc = finite(auc, CONTROL_SECOND_NO_UPDATE)
    acquired_candidate = cell("end_B", ARM_CANDIDATE, "dev_acquired")
    acquired_no_update = cell("end_B", CONTROL_SECOND_NO_UPDATE, "dev_acquired")
    key_acquired = f"{ARM_CANDIDATE}|{CONTROL_SECOND_NO_UPDATE}|dev_acquired|end_B"
    A = (
        candidate_auc <= 0.95 * second_no_auc
        and finite(acquired_candidate, "mean_loss") <= 0.95 * finite(acquired_no_update, "mean_loss")
        and coverage_delta(acquired_candidate, acquired_no_update, 4, 2)
        and int(paired_better[key_acquired]) >= 6
    )

    stage_a_candidate_end_a = cell("end_A", ARM_CANDIDATE, "stage_a")
    stage_a_no_update_end_a = cell("end_A", CONTROL_SECOND_NO_UPDATE, "stage_a")
    key_stage_a_end_a = f"{ARM_CANDIDATE}|{CONTROL_SECOND_NO_UPDATE}|stage_a|end_A"
    S = (
        finite(stage_a_candidate_end_a, "mean_loss")
        <= 0.95 * finite(stage_a_no_update_end_a, "mean_loss")
        and coverage_delta(stage_a_candidate_end_a, stage_a_no_update_end_a, 4, 2)
        and int(paired_better[key_stage_a_end_a]) >= 36
    )

    candidate_gain = finite(gain, ARM_CANDIDATE)
    T = all(
        candidate_gain >= finite(gain, control) + 0.02
        and candidate_auc <= finite(auc, control)
        for control in (ARM_FIRST_ORDER, ARM_SOURCE)
    )

    stage_a_candidate_end_b = cell("end_B", ARM_CANDIDATE, "stage_a")
    retained_fraction = retention.get("retained_fraction")
    R = (
        S
        and finite(retention, "improvement_end_a") > 0.0
        and retained_fraction is not None
        and _finite_float(retained_fraction, "retained fraction") >= 0.80
        and int(stage_a_candidate_end_b["supported_rows"])
        >= 0.95 * int(stage_a_candidate_end_a["supported_rows"])
        and int(stage_a_candidate_end_b["qualifying_streams"])
        >= 0.95 * int(stage_a_candidate_end_a["qualifying_streams"])
        and int(retention["paired_retained_1.05"]) >= 43
    )

    unseen_candidate = cell("end_B", ARM_CANDIDATE, "dev_unseen")
    unseen_no_update = cell("end_B", CONTROL_SECOND_NO_UPDATE, "dev_unseen")
    key_unseen = f"{ARM_CANDIDATE}|{CONTROL_SECOND_NO_UPDATE}|dev_unseen|end_B"
    U = (
        finite(unseen_candidate, "mean_loss") <= 0.95 * finite(unseen_no_update, "mean_loss")
        and coverage_delta(unseen_candidate, unseen_no_update, 2, 1)
        and int(paired_better[key_unseen]) >= 6
    )

    acquired_boundary = cell("end_B", ARM_BOUNDARY, "dev_acquired")
    stage_a_boundary = cell("end_B", ARM_BOUNDARY, "stage_a")
    key_boundary_stage_a = f"{ARM_CANDIDATE}|{ARM_BOUNDARY}|stage_a|end_B"
    B = (
        candidate_auc <= finite(auc, ARM_BOUNDARY)
        and finite(acquired_candidate, "mean_loss") <= finite(acquired_boundary, "mean_loss")
        and int(acquired_candidate["supported_rows"]) >= int(acquired_boundary["supported_rows"])
        and int(acquired_candidate["qualifying_streams"])
        >= int(acquired_boundary["qualifying_streams"])
        and finite(stage_a_candidate_end_b, "mean_loss")
        <= 0.98 * finite(stage_a_boundary, "mean_loss")
        and int(stage_a_candidate_end_b["supported_rows"])
        >= int(stage_a_boundary["supported_rows"])
        and int(stage_a_candidate_end_b["qualifying_streams"])
        >= int(stage_a_boundary["qualifying_streams"])
        and int(paired_better[key_boundary_stage_a]) >= 36
    )

    inherited_checks = []
    for group in ("original", "v20_heldout"):
        candidate = cell("end_B", ARM_CANDIDATE, group)
        no_update = cell("end_B", CONTROL_SECOND_NO_UPDATE, group)
        key = f"{ARM_CANDIDATE}|{CONTROL_SECOND_NO_UPDATE}|{group}|end_B|1.05"
        inherited_checks.append(
            finite(candidate, "mean_loss") <= 1.05 * finite(no_update, "mean_loss")
            and int(candidate["supported_rows"]) >= 0.95 * int(no_update["supported_rows"])
            and int(candidate["qualifying_streams"])
            >= 0.95 * int(no_update["qualifying_streams"])
            and int(paired_ratio[key]) >= 7
        )
    N = all(inherited_checks)
    static_competent = (
        int(acquired_no_update["supported_rows"]) >= 24
        and int(acquired_no_update["qualifying_streams"]) >= 6
    )
    return {
        "substantive_fast_acquisition": A,
        "stage_a_acquired": S,
        "oml_fast_attribution": T,
        "stage_a_retained": R,
        "unseen_development_transfer": U,
        "persistent_boundary_nonregression": B,
        "inherited_nonregression": N,
        "static_no_update_competent": static_competent,
        "A": A,
        "S": S,
        "T": T,
        "R": R,
        "U": U,
        "B": B,
        "N": N,
    }


def _classify_v21a(
    gates: Mapping[str, object],
    mechanical_validity: Mapping[str, object] | bool,
) -> str:
    """Apply the frozen exclusive first-result priority tree."""

    valid = (
        mechanical_validity
        if type(mechanical_validity) is bool
        else mechanical_validity.get("valid")
        if isinstance(mechanical_validity, Mapping)
        else None
    )
    if type(valid) is not bool:
        raise ValueError("V21-A mechanical validity must be an exact boolean")
    required = ("A", "S", "T", "R", "U", "B", "N", "static_no_update_competent")
    if not isinstance(gates, Mapping) or any(type(gates.get(key)) is not bool for key in required):
        raise ValueError("V21-A classifier gates are incomplete")
    if not valid:
        return "INVALID_NO_CLAIM"
    A, S, T, R, U, B, N = (bool(gates[key]) for key in ("A", "S", "T", "R", "U", "B", "N"))
    if A and S and T and R and U and B and N:
        return "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED"
    if not S:
        return "STAGE_A_NOT_ACQUIRED"
    if S and not R:
        return "FAST_ACQUISITION_WITH_FORGETTING"
    if S and R and not N:
        return "INHERITED_CAPABILITY_REGRESSION"
    if S and R and N and A and not T:
        return "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED"
    if S and R and N and A and T and (not U or not B):
        return "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER"
    if S and R and N and not A and gates["static_no_update_competent"]:
        return "STATIC_REPRESENTATION_DOMINATES"
    return "PERSISTENT_OML_NOT_SUPPORTED"


def _nested_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return True
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return True
    if isinstance(value, Mapping):
        return all(_nested_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_nested_finite(item) for item in value)
    return False


def _compute_mechanical_validity(
    system: PersistentLifelongSystem,
    comparisons: Mapping[str, object],
) -> dict[str, object]:
    _assert_system_integrity(system)
    terminal_causal = system.probes.get("end_B", {}).get("terminal_causal", {})
    parity = system.public_train_parity or {}
    probe_preservation = all(
        system.probes.get(boundary, {}).get("learned_state_preserved") is True
        and system.probes.get(boundary, {}).get("slow_state_preserved") is True
        for boundary in PROBE_BOUNDARIES
    )
    no_update_probe_exact = all(
        system.probes["pre"]["conditions"][condition]
        == system.probes["end_A"]["conditions"][condition]
        == system.probes["end_B"]["conditions"][condition]
        for condition in NO_UPDATE_CONTROLS
    )
    probe_cohort_exact = all(
        system.probes[boundary].get("cohort_bytes_exact") is True
        and system.probes[boundary].get("cohort_digest_mode")
        == "order_sensitive_literal_dataclass_fields_v1"
        and tuple(system.probes[boundary].get("cohort_public_digests", ()))
        == tuple(system.probes["pre"].get("cohort_public_digests", ()))
        and len(system.probes[boundary].get("cohort_public_digests", ()))
        == PROBE_STREAMS
        for boundary in PROBE_BOUNDARIES
    )
    arm_counters = {
        name: (
            system.arm(name).state.lifetime_updates,
            system.arm(name).state.optimizer_state[0].step,
            system.arm(name).state.reset_count,
        )
        for name in UPDATED_ARMS
    }
    counter_exact = arm_counters == {
        ARM_CANDIDATE: (256, 256, 0),
        ARM_FIRST_ORDER: (256, 256, 0),
        ARM_SOURCE: (256, 256, 0),
        ARM_BOUNDARY: (256, 64, 1),
    }
    numerical_mode = (
        torch.are_deterministic_algorithms_enabled()
        and not bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
        and not bool(getattr(torch.backends.cudnn, "allow_tf32", False))
        and all(system.arm(name).state.weight.dtype == torch.float32 for name in UPDATED_ARMS)
    )
    retention = comparisons.get("stage_a_retention", {})
    checks = {
        "chronology_complete": system.next_experience == TOTAL_EXPERIENCES,
        "probe_boundaries_complete": tuple(system.probes) == PROBE_BOUNDARIES,
        "public_train_parity_exact": all(
            parity.get(key) is True
            for key in (
                "row_values_exact",
                "row_order_exact",
                "row_masks_exact",
                "loss_exact",
                "fast_gradient_exact",
                "rln_gradients_exact",
            )
        ),
        "candidate_boundary_exact_through_end_a": bool(
            isinstance(system.end_a_exactness, Mapping)
            and system.end_a_exactness.get("state_exact") is True
            and system.end_a_exactness.get("metrics_exact") is True
        ),
        "terminal_state_swap_exact": terminal_causal.get("state_swap_exact") is True,
        "terminal_state_swap_digest_exact": terminal_causal.get(
            "state_swap_digest_exact"
        )
        is True,
        "terminal_clean_controller_digest_exact": terminal_causal.get(
            "clean_controller_digest_exact"
        )
        is True,
        "terminal_w0_reset_exact": terminal_causal.get("w0_reset_exact") is True,
        "terminal_reset_state_valid": terminal_causal.get("reset_state_valid") is True,
        "no_update_probe_metrics_exact_across_boundaries": no_update_probe_exact,
        "immutable_probe_cohort_bytes_exact": probe_cohort_exact,
        "constructed_identity_ledger_complete": {
            role: len(system.identity_ledger[role])
            for role in ("diagnostic", "update", "probe")
        }
        == {
            "diagnostic": DIAGNOSTIC_STREAMS,
            "update": TOTAL_EXPERIENCES,
            "probe": PROBE_STREAMS,
        }
        and all(
            tuple(record.get("boundaries", ())) == PROBE_BOUNDARIES
            for record in system.identity_ledger["probe"].values()
        ),
        "probe_state_preserved": probe_preservation,
        "diagnostic_state_preserved": bool(
            isinstance(system.gradient_geometry, Mapping)
            and system.gradient_geometry.get("learned_state_preserved") is True
            and system.gradient_geometry.get("slow_state_preserved") is True
        ),
        "slow_parameters_frozen": all(
            not parameter.requires_grad and parameter.grad is None
            for name in UPDATED_ARMS
            for parameter in system.arm(name).controller.parameters()
        ),
        "arm_counters_exact": counter_exact,
        "no_update_counters_zero": True,
        "online_auc_complete": all(
            len(system.stage_b_online_pre_loss[name]) == STAGE_B_EXPERIENCES
            for name in MEASUREMENT_CONDITIONS
        ),
        "finite_complete": _nested_finite(comparisons)
        and _nested_finite(system.probes)
        and _nested_finite(system.gradient_geometry)
        and _nested_finite(system.stage_b_online_pre_loss),
        "retention_denominator_valid": isinstance(retention, Mapping)
        and retention.get("denominator_valid") is True
        and retention.get("retained_fraction") is not None,
        "numerical_mode_exact": numerical_mode,
        "allocated_memory_within_ceiling": _assert_resource_ceiling(system)
        <= ALLOCATED_MEMORY_CEILING_BYTES,
        "single_boundary_reset": system.boundary_reset_applied
        and system.second_order_boundary_reset.state.reset_count == 1,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "arm_counters": arm_counters,
        "persistent_float_values_per_arm": {
            name: system.arm(name).state.persistent_float_values for name in UPDATED_ARMS
        },
        "no_update_counters": {
            name: {"lifetime_updates": 0, "adamw_step": 0, "reset_count": 0}
            for name in NO_UPDATE_CONTROLS
        },
    }


def evaluate_persistent_lifelong(
    system: PersistentLifelongSystem,
    *,
    stream_factory: StreamFactory | None = None,
    deadline_callback: DeadlineCallback | None = None,
) -> dict[str, object]:
    """Evaluate and classify the first completed V21-A identity without tuning."""

    if system.next_experience != TOTAL_EXPERIENCES:
        raise RuntimeError("V21-A terminal evaluation requires 256 completed experiences")
    if "end_B" not in system.probes:
        evaluate_probe_boundary(
            system,
            "end_B",
            stream_factory=stream_factory,
            deadline_callback=deadline_callback,
        )
    _invoke_deadline(deadline_callback)
    comparisons = _terminal_comparisons(system)
    gates = _compute_gates(comparisons)
    mechanical = _compute_mechanical_validity(system, comparisons)
    classification = _classify_v21a(gates, mechanical)
    report: dict[str, object] = {
        "artifact_schema": "angler.phase6-v21a-persistent-lifelong-report.v1",
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "classification": classification,
        "passed": classification == "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
        "system_digest": persistent_lifelong_system_digest(system),
        "gates": gates,
        "comparisons": comparisons,
        "mechanical_validity": mechanical,
        "families": {
            group: {
                condition: copy.deepcopy(
                    system.probes["end_B"]["conditions"][condition][group]
                )
                for condition in MEASUREMENT_CONDITIONS
            }
            for group in PROBE_GROUPS
        },
        "gradient_geometry": copy.deepcopy(system.gradient_geometry),
        "public_train_parity": copy.deepcopy(system.public_train_parity),
        "source_bindings": copy.deepcopy(system.source_bindings),
        "first_result_accepted_without_tuning": True,
        "nonclaims": {
            "no_anml_authority": True,
            "no_replay": True,
            "no_final_or_sealed_access": True,
            "no_promoted_state_mutation": True,
            "no_deployment": True,
        },
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    report["terminal_json_bytes_without_size_field"] = len(encoded)
    if len(encoded) > TERMINAL_JSON_SIZE_CEILING_BYTES:
        report["mechanical_validity"]["checks"]["terminal_json_within_ceiling"] = False
        report["mechanical_validity"]["valid"] = False
        report["classification"] = "INVALID_NO_CLAIM"
        report["passed"] = False
    else:
        report["mechanical_validity"]["checks"]["terminal_json_within_ceiling"] = True
    return report


def synthetic_cuda_preflight(
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    """Exercise only synthetic 64-value state math; construct no protocol object."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V21-A synthetic preflight requires CUDA")
    configure_persistent_lifelong_numerics()
    torch.cuda.synchronize(selected)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(selected)
    initial = torch.linspace(-0.25, 0.25, 64, device=selected, dtype=torch.float32).reshape(1, 64)
    initial = initial.detach()
    gradients = tuple(
        torch.sin(initial * float(index + 1) + 0.01 * index).detach()
        for index in range(8)
    )

    def advance(state: PersistentFastState, gradient: torch.Tensor) -> PersistentFastState:
        (weight,), optimizer = functional_adamw_step(
            (state.weight,),
            (gradient,),
            state.optimizer_state,
            (INNER_LEARNING_RATE,),
            beta1=ADAM_BETA1,
            beta2=ADAM_BETA2,
            epsilon=ADAM_EPSILON,
            weight_decay=ADAM_WEIGHT_DECAY,
        )
        return PersistentFastState(
            weight=weight.detach().clone(),
            optimizer_state=tuple(
                AdamWSlot(
                    slot.step,
                    slot.exp_avg.detach().clone(),
                    slot.exp_avg_sq.detach().clone(),
                )
                for slot in optimizer
            ),
            lifetime_updates=state.lifetime_updates + 1,
            reset_count=state.reset_count,
        )

    uninterrupted = _fresh_persistent_fast_state(initial)
    interrupted = _fresh_persistent_fast_state(initial)
    for index, gradient in enumerate(gradients):
        uninterrupted = advance(uninterrupted, gradient)
        if index < 4:
            interrupted = advance(interrupted, gradient)
    serialized_snapshot = io.BytesIO()
    torch.save(snapshot_fast_state(interrupted), serialized_snapshot)
    serialized_snapshot.seek(0)
    round_tripped_snapshot = torch.load(
        serialized_snapshot,
        map_location=selected,
        weights_only=True,
    )
    restored = restore_fast_state(round_tripped_snapshot, device=selected)
    for gradient in gradients[4:]:
        restored = advance(restored, gradient)
    continuation_exact = persistent_fast_state_digest(uninterrupted) == persistent_fast_state_digest(
        restored
    )

    functional = _fresh_persistent_fast_state(initial)
    parameter = nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW(
        (parameter,),
        lr=INNER_LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    for gradient in gradients:
        functional = advance(functional, gradient)
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = gradient.detach().clone()
        optimizer.step()
    functional_delta = float((functional.weight - parameter.detach()).abs().max().item())
    functional_parity = functional_delta <= 1.0e-6

    capacity_state = _fresh_persistent_fast_state(initial)
    observed_capacities = []
    for index in range(TOTAL_EXPERIENCES):
        deterministic_gradient = torch.cos(initial + 0.001 * index).detach()
        capacity_state = advance(capacity_state, deterministic_gradient)
        observed_capacities.append(capacity_state.persistent_float_values)
    selected_state_constant_capacity = set(observed_capacities) == {PERSISTENT_FLOAT_VALUES}
    torch.cuda.synchronize(selected)
    maximum = int(torch.cuda.max_memory_allocated(selected))
    if (
        not continuation_exact
        or not functional_parity
        or not selected_state_constant_capacity
        or maximum > ALLOCATED_MEMORY_CEILING_BYTES
    ):
        raise RuntimeError("V21-A synthetic CUDA preflight failed")
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "plan_digest": persistent_lifelong_plan_digest(),
        "synthetic_only": True,
        "device": str(selected),
        "detach_continuation_exact": True,
        "functional_adamw_parity": True,
        "functional_adamw_max_abs_delta": functional_delta,
        "checkpoint_resume_exact": continuation_exact,
        "selected_state_constant_capacity": selected_state_constant_capacity,
        "persistent_float_values": PERSISTENT_FLOAT_VALUES,
        "persistent_float_bytes": PERSISTENT_FLOAT_BYTES,
        "semantic_streams_generated": 0,
        "semantic_updates_performed": False,
        "maximum_allocated_bytes": maximum,
        "allocated_memory_ceiling_bytes": ALLOCATED_MEMORY_CEILING_BYTES,
    }


__all__ = (
    "PROTOCOL_ID",
    "PersistentFastState",
    "PersistentArm",
    "PersistentLifelongSystem",
    "persistent_lifelong_plan",
    "persistent_lifelong_plan_digest",
    "verify_persistent_lifelong_dependencies",
    "public_paired_graph_credit_rows_from_supports",
    "public_train_parity_report",
    "snapshot_fast_state",
    "restore_fast_state",
    "persistent_fast_state_digest",
    "reset_fast_state",
    "persistent_learned_state_digest",
    "persistent_lifelong_system_digest",
    "persistent_lifelong_checkpoint_summary",
    "build_persistent_lifelong_system",
    "apply_persistent_experience",
    "run_initial_gradient_geometry",
    "evaluate_probe_boundary",
    "fit_persistent_lifelong",
    "save_persistent_lifelong_checkpoint",
    "load_persistent_lifelong_checkpoint",
    "validate_persistent_lifelong_checkpoint",
    "evaluate_persistent_lifelong",
    "synthetic_cuda_preflight",
    "configure_persistent_lifelong_numerics",
)
