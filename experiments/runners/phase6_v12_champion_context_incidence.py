"""V12-champion incidence-aware context successor.

V18 changes only the information available to V12's context selector.  A
context-specific copy of the existing anonymous relation-axis set readout sees
the raw pair tensor before the lossy flat pool.  Its bias-free output
projection starts at exact zero, so migration is function preserving and an
analytic projection-zero lesion remains available on the learned system.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_context_residual as v17


PROTOCOL_ID = "phase6.public-v12-champion-context-incidence.v18"
CHECKPOINT_VERSION = "angler.phase6-v12-champion-context-incidence.v1"
V12_CHECKPOINT_SHA256 = v17.V12_CHECKPOINT_SHA256
V12_CONTROLLER_DIGEST = v17.V12_CONTROLLER_DIGEST
V12_MIXER_DIGEST = v17.V12_MIXER_DIGEST
V12_COMPETENCE_DIGEST = v17.V12_COMPETENCE_DIGEST
V12_SYSTEM_DIGEST = v17.V12_SYSTEM_DIGEST

ACTIVE_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md"
)
FROZEN_DEPENDENCY_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_v12_champion_context_residual.py": (
        "3B5B05CA4122F08133213AA811D5C5EDCA6B9869EF56B132273A90CE42724333"
    ),
    ACTIVE_LEAF: (
        "1A00F9FE8FE4004BF1E1F9FFD2D2441B2D57808CE6F7B3C87DA9CBB27DB964CF"
    ),
}

_PLAN_DIGEST_DOMAIN = b"project-angler.v12-champion-context-incidence.plan.v1\x00"
_MUTABLE_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-context-incidence.mutable.v1\x00"
)
_OPTIMIZER_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-context-incidence.optimizer.v1\x00"
)
_SYSTEM_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-context-incidence.system.v1\x00"
)

_CONTEXT_WIDTH = 32
_READOUT_SEED = 2_026_083_801
_CONTEXT_UPDATES = 256
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_PANEL_COUNT = 4
_READOUT_LEARNING_RATE = 3.0e-4
_PROJECTION_LEARNING_RATE = 1.0e-3
_GRADIENT_CLIP = 5.0
_PRESENCE_WEIGHT = 0.25
_ABSTAIN_WEIGHT = 0.25

_TRAIN_TOPOLOGY_BASE = 9_001_000_001
_TRAIN_SURFACE_BASE = 9_101_000_001
_PANEL_TOPOLOGY_BASE = 9_201_000_001
_PANEL_SURFACE_BASE = 9_301_000_001

_CONTEXT_TOP_ONE_THRESHOLD = 0.80
_CONTEXT_MASS_THRESHOLD = 0.60
_RELATION_SUPPORTED_ROWS_THRESHOLD = 96
_RELATION_QUALIFYING_STREAMS_THRESHOLD = 24
_CAUSAL_TOP_ONE_GAIN = 12
_CAUSAL_REAL_NORMALIZED_MASS_GAIN = 0.05
_REQUIRED_IMPROVED_PANELS = 3
_MAX_FOURTH_PANEL_TOP_ONE_REGRESSION = 1
_MAX_FOURTH_PANEL_MASS_REGRESSION = 0.01

_READOUT_PARAMETER_NAMES = (
    "context_incidence_readout.row_attention.weight",
    "context_incidence_readout.column_attention.weight",
    "context_incidence_readout.node_projection.0.weight",
    "context_incidence_readout.node_projection.0.bias",
    "context_incidence_readout.node_projection.1.weight",
    "context_incidence_readout.node_projection.1.bias",
    "context_incidence_readout.node_projection.3.weight",
    "context_incidence_readout.node_projection.3.bias",
    "context_incidence_readout.node_pool_attention.weight",
)
_PROJECTION_PARAMETER_NAMES = ("context_incidence_projection.weight",)
MUTABLE_PARAMETER_NAMES = _READOUT_PARAMETER_NAMES + _PROJECTION_PARAMETER_NAMES
_MUTABLE_PREFIXES = (
    "context_incidence_readout.",
    "context_incidence_projection.",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES}


def _verify_frozen_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    if Path(v12.__file__).resolve() != (
        root / "experiments/runners/phase6_software_pipeline_reconstruction.py"
    ).resolve():
        raise RuntimeError("V18 imported a shadowed V12 dependency")
    if Path(v17.__file__).resolve() != (
        root / "experiments/runners/phase6_v12_champion_context_residual.py"
    ).resolve():
        raise RuntimeError("V18 imported a shadowed V17 utility dependency")
    if frozen_dependency_hashes() != FROZEN_DEPENDENCY_HASHES:
        raise RuntimeError("V18 frozen source or active-leaf dependency changed")


def _plan_payload() -> dict[str, object]:
    commitments = tuple(v12.software_pipeline_mechanism_partition("train")[:8])
    if len(commitments) != _STREAMS_PER_UPDATE or len(set(commitments)) != 8:
        raise RuntimeError("V18 requires eight distinct public commitments")
    training = tuple(
        tuple(
            (
                _TRAIN_TOPOLOGY_BASE + 100_000 * update + 1_000 * commitment,
                _TRAIN_SURFACE_BASE + 100_000 * update + 1_000 * commitment,
            )
            for commitment in range(_STREAMS_PER_UPDATE)
        )
        for update in range(_CONTEXT_UPDATES)
    )
    panels = tuple(
        tuple(
            (
                _PANEL_TOPOLOGY_BASE + 100_000 * panel + 1_000 * commitment,
                _PANEL_SURFACE_BASE + 100_000 * panel + 1_000 * commitment,
            )
            for commitment in range(_STREAMS_PER_UPDATE)
        )
        for panel in range(_PANEL_COUNT)
    )
    training_pairs = {pair for batch in training for pair in batch}
    panel_pairs = {pair for panel in panels for pair in panel}
    if (
        len(training_pairs) != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE
        or len(panel_pairs) != _PANEL_COUNT * _STREAMS_PER_UPDATE
        or training_pairs & panel_pairs
        or any(seed < _TRAIN_TOPOLOGY_BASE for pair in training_pairs | panel_pairs for seed in pair)
    ):
        raise RuntimeError("V18 stream identities are not fresh and disjoint")
    return {
        "protocol_id": PROTOCOL_ID,
        "source_v12_checkpoint_sha256": V12_CHECKPOINT_SHA256,
        "source_v12_digests": {
            "controller": V12_CONTROLLER_DIGEST,
            "mixer": V12_MIXER_DIGEST,
            "competence": V12_COMPETENCE_DIGEST,
            "system": V12_SYSTEM_DIGEST,
        },
        "frozen_dependency_hashes": dict(FROZEN_DEPENDENCY_HASHES),
        "commitments": commitments,
        "readout_seed": _READOUT_SEED,
        "context_width": _CONTEXT_WIDTH,
        "training_seed_batches": training,
        "panel_seed_pairs": panels,
        "context_updates": _CONTEXT_UPDATES,
        "streams_per_update": _STREAMS_PER_UPDATE,
        "rows_per_stream": _ROWS_PER_STREAM,
        "readout_learning_rate": _READOUT_LEARNING_RATE,
        "projection_learning_rate": _PROJECTION_LEARNING_RATE,
        "weight_decay": 0.0,
        "gradient_clip": _GRADIENT_CLIP,
        "objective": {
            "supported_rank": "-log(valid_real_mass/all_real_mass)",
            "supported_presence": "-log(all_real_mass)",
            "unsupported_abstain": "-log(null_mass)",
            "rank_weight": 1.0,
            "presence_weight": _PRESENCE_WEIGHT,
            "abstain_weight": _ABSTAIN_WEIGHT,
            "relation_margins_detached": True,
        },
        "component_support": {
            "aggregate_unique_valid_real_top_one_gain": _CAUSAL_TOP_ONE_GAIN,
            "aggregate_real_normalized_mass_gain": (
                _CAUSAL_REAL_NORMALIZED_MASS_GAIN
            ),
            "aggregate_unique_valid_rank_margin_positive": True,
            "causal_informative_rank_margin_gain_positive": True,
            "improved_panels": _REQUIRED_IMPROVED_PANELS,
            "maximum_fourth_panel_top_one_regression": (
                _MAX_FOURTH_PANEL_TOP_ONE_REGRESSION
            ),
            "maximum_fourth_panel_mass_regression": (
                _MAX_FOURTH_PANEL_MASS_REGRESSION
            ),
        },
        "full_advancement": {
            "relation_supported_rows": _RELATION_SUPPORTED_ROWS_THRESHOLD,
            "relation_qualifying_streams": _RELATION_QUALIFYING_STREAMS_THRESHOLD,
            "supported_context_top_one_fraction": _CONTEXT_TOP_ONE_THRESHOLD,
            "supported_valid_set_mass": _CONTEXT_MASS_THRESHOLD,
        },
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
        "joint_training": False,
        "replay": False,
        "srwm": False,
        "router": False,
    }


def v12_champion_context_incidence_plan() -> dict[str, object]:
    """Return the complete frozen V18 plan and canonical digest."""

    payload = _plan_payload()
    return {
        **payload,
        "plan_digest": v17._json_digest(_PLAN_DIGEST_DOMAIN, payload),
    }


class V12ChampionContextIncidenceController(v12.SoftwarePipelineController):
    """V12 controller with one zero-output raw-context incidence residual."""

    def __init__(self, profile: v12.SoftwarePipelineRunProfile) -> None:
        if profile != v12.SOFTWARE_PIPELINE_PROFILES["smoke"]:
            raise ValueError("V18 accepts only the V12 smoke profile")
        cpu_rng_state = torch.get_rng_state()
        try:
            super().__init__(profile)
            torch.default_generator.manual_seed(_READOUT_SEED)
            self.context_incidence_readout = v12.RelationAxisSetReadout(profile)
            self.context_incidence_projection = nn.Linear(
                4 * profile.width,
                profile.width,
                bias=False,
            )
            nn.init.zeros_(self.context_incidence_projection.weight)
        finally:
            torch.set_rng_state(cpu_rng_state)
        self._context_incidence_enabled = True
        _enforce_mutable_scope(self)

    def _pool_context_tensor(self, pair_states: torch.Tensor) -> torch.Tensor:
        if (
            pair_states.ndim != 3
            or pair_states.shape[0] != pair_states.shape[1]
            or pair_states.shape[-1] != self.profile.width
            or pair_states.shape[0] <= 0
            or not pair_states.is_floating_point()
            or not bool(torch.isfinite(pair_states).all().item())
        ):
            raise ValueError("context pair states must be finite [nodes, nodes, width]")
        cells = pair_states.reshape(-1, self.profile.width)
        attention = torch.softmax(
            self.relation_context_pool_attention(cells).transpose(0, 1),
            dim=-1,
        )
        attended = attention @ cells
        pooled = torch.cat(
            (
                attended.reshape(-1),
                cells.mean(dim=0),
                cells.amax(dim=0),
            ),
            dim=-1,
        )
        precode = self.relation_context_pool_projection(pooled)
        if self._context_incidence_enabled:
            incidence = self.context_incidence_readout(pair_states)
            precode = precode + self.context_incidence_projection(incidence)
        return F.normalize(precode, dim=-1, eps=1.0e-8)

    @contextmanager
    def projection_zero_lesion(self) -> Iterator[None]:
        if not self._context_incidence_enabled:
            raise RuntimeError("nested V18 incidence lesions are forbidden")
        self._context_incidence_enabled = False
        try:
            yield
        finally:
            self._context_incidence_enabled = True


def _enforce_mutable_scope(controller: V12ChampionContextIncidenceController) -> None:
    actual = tuple(
        name
        for name, _ in controller.named_parameters()
        if name.startswith(_MUTABLE_PREFIXES)
    )
    if actual != MUTABLE_PARAMETER_NAMES or len(actual) != 10:
        raise RuntimeError("V18 mutable tensor identity changed")
    named = dict(controller.named_parameters())
    if sum(named[name].numel() for name in actual) != 16_992:
        raise RuntimeError("V18 mutable parameter count changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name in MUTABLE_PARAMETER_NAMES)


def context_incidence_parameter_report(
    controller: V12ChampionContextIncidenceController,
    mixer: v12.AnonymousConflictMixer,
) -> dict[str, object]:
    if not isinstance(controller, V12ChampionContextIncidenceController):
        raise TypeError("controller must be the V18 controller")
    _enforce_mutable_scope(controller)
    controller_parameters = sum(value.numel() for value in controller.parameters())
    mixer_parameters = sum(value.numel() for value in mixer.parameters())
    return {
        "protocol_id": PROTOCOL_ID,
        "inherited_v12_controller_parameters": 265_606,
        "new_trainable_tensors": len(MUTABLE_PARAMETER_NAMES),
        "new_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in controller.named_parameters()
            if name in MUTABLE_PARAMETER_NAMES
        ),
        "controller_parameters": controller_parameters,
        "mixer_parameters": mixer_parameters,
        "complete_learned_system_parameters": controller_parameters + mixer_parameters,
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
    }


V12ChampionSourceBinding = v17.V12ChampionSourceBinding


@dataclass(slots=True)
class V12ChampionContextIncidenceSystem:
    controller: V12ChampionContextIncidenceController
    mixer: v12.AnonymousConflictMixer
    competence_state: v12.SoftwareReconstructionState
    source: V12ChampionSourceBinding
    context_updates: int = 0
    optimizer_state: dict[str, object] | None = None


def _expected_source_binding() -> V12ChampionSourceBinding:
    return V12ChampionSourceBinding(
        checkpoint_sha256=V12_CHECKPOINT_SHA256,
        controller_digest=V12_CONTROLLER_DIGEST,
        mixer_digest=V12_MIXER_DIGEST,
        competence_digest=V12_COMPETENCE_DIGEST,
        system_digest=V12_SYSTEM_DIGEST,
    )


def _inherited_state(
    controller: V12ChampionContextIncidenceController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if not name.startswith(_MUTABLE_PREFIXES)
    }


def _mutable_state(
    controller: V12ChampionContextIncidenceController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if name.startswith(_MUTABLE_PREFIXES)
    }


def _base_controller_from_successor(
    controller: V12ChampionContextIncidenceController,
) -> v12.SoftwarePipelineController:
    cpu_rng_state = torch.get_rng_state()
    try:
        base = v12.SoftwarePipelineController(controller.profile)
    finally:
        torch.set_rng_state(cpu_rng_state)
    base.load_state_dict(
        {
            name: value.detach().cpu().clone()
            for name, value in _inherited_state(controller).items()
        },
        strict=True,
    )
    base.eval()
    return base


def inherited_v12_controller_digest(
    controller: V12ChampionContextIncidenceController,
) -> str:
    return v12.software_pipeline_model_digest(_base_controller_from_successor(controller))


def context_incidence_mutable_digest(
    controller: V12ChampionContextIncidenceController,
) -> str:
    return v17._mapping_digest(_MUTABLE_DIGEST_DOMAIN, _mutable_state(controller))


def _source_system_digest(system: V12ChampionContextIncidenceSystem) -> str:
    return v12.public_relation_conflict_system_digest(
        _base_controller_from_successor(system.controller),
        system.mixer,
        system.competence_state,
    )


def context_incidence_system_digest(
    system: V12ChampionContextIncidenceSystem,
) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_context_incidence_plan()["plan_digest"],
        "source": asdict(system.source),
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": context_incidence_mutable_digest(system.controller),
        "mixer_digest": v12.anonymous_conflict_mixer_digest(system.mixer),
        "competence_digest": v12.software_reconstruction_state_digest(
            system.competence_state
        ),
        "context_updates": system.context_updates,
        "optimizer_digest": context_incidence_optimizer_digest(
            system.optimizer_state
        ),
    }
    return v17._json_digest(_SYSTEM_DIGEST_DOMAIN, payload)


def _assert_source_lineage(system: V12ChampionContextIncidenceSystem) -> None:
    if system.source != _expected_source_binding():
        raise RuntimeError("V18 terminal V12 source binding changed")
    if inherited_v12_controller_digest(system.controller) != V12_CONTROLLER_DIGEST:
        raise RuntimeError("V18 changed an inherited V12 controller byte")
    if v12.anonymous_conflict_mixer_digest(system.mixer) != V12_MIXER_DIGEST:
        raise RuntimeError("V18 changed the inherited V12 conflict mixer")
    if any(parameter.requires_grad for parameter in system.mixer.parameters()):
        raise RuntimeError("V18 inherited conflict mixer is not frozen")
    if (
        v12.software_reconstruction_state_digest(system.competence_state)
        != V12_COMPETENCE_DIGEST
    ):
        raise RuntimeError("V18 changed the inherited V12 competence state")
    if _source_system_digest(system) != V12_SYSTEM_DIGEST:
        raise RuntimeError("V18 source system lineage changed")
    if type(system.context_updates) is not int or not 0 <= system.context_updates <= _CONTEXT_UPDATES:
        raise RuntimeError("V18 context update count is invalid")
    if system.context_updates == 0:
        if system.optimizer_state is not None:
            raise RuntimeError("V18 fresh lineage unexpectedly has optimizer moments")
    else:
        if not isinstance(system.optimizer_state, Mapping):
            raise RuntimeError("V18 learned lineage lost optimizer moments")
        _validate_optimizer_state(
            system.optimizer_state,
            system.controller,
            expected_steps=system.context_updates,
        )
    _enforce_mutable_scope(system.controller)


def _migrate_loaded_v12_system(
    controller: v12.SoftwarePipelineController,
    mixer: v12.AnonymousConflictMixer,
    state: v12.SoftwareReconstructionState,
    *,
    source_checkpoint_sha256: str,
    device: torch.device | str = "cpu",
) -> V12ChampionContextIncidenceSystem:
    _verify_frozen_dependencies()
    if source_checkpoint_sha256.upper() != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V18 accepts only the frozen terminal V12 checkpoint")
    if type(controller) is not v12.SoftwarePipelineController:
        raise TypeError("V18 migration requires the exact V12 controller type")
    source = V12ChampionSourceBinding(
        checkpoint_sha256=V12_CHECKPOINT_SHA256,
        controller_digest=v12.software_pipeline_model_digest(controller),
        mixer_digest=v12.anonymous_conflict_mixer_digest(mixer),
        competence_digest=v12.software_reconstruction_state_digest(state),
        system_digest=v12.public_relation_conflict_system_digest(
            controller,
            mixer,
            state,
        ),
    )
    if source != _expected_source_binding():
        raise RuntimeError("V18 source payload is not the terminal V12 lineage")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = v17._cuda_rng_snapshot(device)
    try:
        successor = V12ChampionContextIncidenceController(controller.profile).to(device)
        load_result = successor.load_state_dict(controller.state_dict(), strict=False)
        if tuple(sorted(load_result.missing_keys)) != tuple(
            sorted(MUTABLE_PARAMETER_NAMES)
        ) or load_result.unexpected_keys:
            raise RuntimeError("V18 migration state boundary changed")
        cloned_mixer = v12.AnonymousConflictMixer(
            feature_count=mixer.feature_count,
            hidden_width=mixer.hidden_width,
            anchor_weight=mixer.anchor_weight,
        ).to(device)
        cloned_mixer.load_state_dict(mixer.state_dict(), strict=True)
        for parameter in cloned_mixer.parameters():
            parameter.requires_grad_(False)
    finally:
        torch.set_rng_state(cpu_rng_state)
        v17._restore_cuda_rng_snapshot(cuda_rng_states)
    cloned_state = v12.restore_software_reconstruction_state(
        {
            name: value.detach().to(device).clone()
            for name, value in v12.snapshot_software_reconstruction_state(state).items()
        }
    )
    _enforce_mutable_scope(successor)
    inherited = _inherited_state(successor)
    if inherited.keys() != controller.state_dict().keys() or any(
        not torch.equal(value.detach().cpu(), controller.state_dict()[name].detach().cpu())
        for name, value in inherited.items()
    ):
        raise RuntimeError("V18 migration changed inherited V12 state")
    system = V12ChampionContextIncidenceSystem(
        controller=successor,
        mixer=cloned_mixer,
        competence_state=cloned_state,
        source=source,
    )
    successor.eval()
    cloned_mixer.eval()
    _assert_source_lineage(system)
    report = context_incidence_parameter_report(successor, cloned_mixer)
    if (
        report["controller_parameters"] != 282_598
        or report["complete_learned_system_parameters"] != 286_002
    ):
        raise RuntimeError("V18 migrated parameter count changed")
    return system


def load_v12_champion_context_incidence_source(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionContextIncidenceSystem:
    actual = _sha256_file(path)
    if actual != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V18 source checkpoint SHA-256 is not frozen V12")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = v17._cuda_rng_snapshot(device)
    try:
        controller, mixer, state = v12.load_public_relation_conflict_checkpoint(
            path,
            device=device,
        )
        return _migrate_loaded_v12_system(
            controller,
            mixer,
            state,
            source_checkpoint_sha256=actual,
            device=device,
        )
    finally:
        torch.set_rng_state(cpu_rng_state)
        v17._restore_cuda_rng_snapshot(cuda_rng_states)


def _context_optimizer(
    controller: V12ChampionContextIncidenceController,
) -> torch.optim.AdamW:
    named = dict(controller.named_parameters())
    return torch.optim.AdamW(
        (
            {
                "params": [named[name] for name in _READOUT_PARAMETER_NAMES],
                "lr": _READOUT_LEARNING_RATE,
            },
            {
                "params": [named[name] for name in _PROJECTION_PARAMETER_NAMES],
                "lr": _PROJECTION_LEARNING_RATE,
            },
        ),
        weight_decay=0.0,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )


def _optimizer_groups() -> tuple[dict[str, object], ...]:
    return (
        {"names": _READOUT_PARAMETER_NAMES, "lr": _READOUT_LEARNING_RATE},
        {"names": _PROJECTION_PARAMETER_NAMES, "lr": _PROJECTION_LEARNING_RATE},
    )


def _canonical_optimizer_state(
    optimizer: torch.optim.AdamW,
    controller: V12ChampionContextIncidenceController,
) -> dict[str, object]:
    named = dict(controller.named_parameters())
    for actual, expected in zip(optimizer.param_groups, _optimizer_groups(), strict=True):
        actual_names = tuple(
            next(name for name, parameter in named.items() if parameter is value)
            for value in actual["params"]
        )
        if actual_names != expected["names"] or float(actual["lr"]) != expected["lr"]:
            raise RuntimeError("V18 optimizer parameter groups changed")
    slots: dict[str, dict[str, torch.Tensor]] = {}
    for name in MUTABLE_PARAMETER_NAMES:
        state = optimizer.state.get(named[name])
        if not isinstance(state, dict) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(f"V18 optimizer state is incomplete: {name}")
        slots[name] = {
            slot_name: value.detach().cpu().clone()
            for slot_name, value in state.items()
        }
    result: dict[str, object] = {
        "version": "adamw-name-keyed.v1",
        "groups": _optimizer_groups(),
        "hyperparameters": {
            "betas": (0.9, 0.999),
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
            "capturable": False,
            "differentiable": False,
        },
        "state": slots,
    }
    _validate_optimizer_state(result, controller)
    return result


def _validate_optimizer_state(
    value: Mapping[str, object],
    controller: V12ChampionContextIncidenceController,
    *,
    expected_steps: int | None = None,
) -> None:
    expected_hyperparameters = {
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    }
    if (
        set(value) != {"version", "groups", "hyperparameters", "state"}
        or value["version"] != "adamw-name-keyed.v1"
        or value["groups"] != _optimizer_groups()
        or value["hyperparameters"] != expected_hyperparameters
    ):
        raise RuntimeError("V18 optimizer configuration changed")
    slots = value["state"]
    if not isinstance(slots, Mapping) or set(slots) != set(MUTABLE_PARAMETER_NAMES):
        raise RuntimeError("V18 optimizer slot ownership changed")
    named = dict(controller.named_parameters())
    observed_steps = []
    for name in MUTABLE_PARAMETER_NAMES:
        slot = slots[name]
        if not isinstance(slot, Mapping) or set(slot) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(f"V18 optimizer slot fields changed: {name}")
        step = slot["step"]
        exp_avg = slot["exp_avg"]
        exp_avg_sq = slot["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all().item())
            or float(step.item()) <= 0.0
            or not float(step.item()).is_integer()
            or not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.shape != named[name].shape
            or exp_avg_sq.shape != named[name].shape
            or exp_avg.dtype != named[name].dtype
            or exp_avg_sq.dtype != named[name].dtype
            or not bool(torch.isfinite(exp_avg).all().item())
            or not bool(torch.isfinite(exp_avg_sq).all().item())
        ):
            raise RuntimeError(f"V18 optimizer slot tensor changed: {name}")
        observed_steps.append(int(step.item()))
    if len(set(observed_steps)) != 1 or (
        expected_steps is not None and observed_steps[0] != expected_steps
    ):
        raise RuntimeError("V18 optimizer step counters changed")


def context_incidence_optimizer_digest(
    value: Mapping[str, object] | None,
) -> str:
    digest = hashlib.sha256(_OPTIMIZER_DIGEST_DOMAIN)
    if value is None:
        digest.update(b"none")
        return "sha256:" + digest.hexdigest()
    metadata = {
        "version": value.get("version"),
        "groups": value.get("groups"),
        "hyperparameters": value.get("hyperparameters"),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    slots = value.get("state")
    if not isinstance(slots, Mapping):
        raise RuntimeError("V18 optimizer digest requires named slots")
    for parameter_name in sorted(slots):
        slot = slots[parameter_name]
        if not isinstance(slot, Mapping):
            raise RuntimeError("V18 optimizer digest slot is invalid")
        for slot_name in sorted(slot):
            tensor = slot[slot_name]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("V18 optimizer digest value is not a tensor")
            v17._update_tensor_digest(
                digest,
                f"{parameter_name}.{slot_name}",
                tensor,
            )
    return "sha256:" + digest.hexdigest()


def restore_context_incidence_optimizer(
    system: V12ChampionContextIncidenceSystem,
) -> torch.optim.AdamW:
    if system.optimizer_state is None:
        raise RuntimeError("V18 system has no learned optimizer state")
    _validate_optimizer_state(system.optimizer_state, system.controller)
    optimizer = _context_optimizer(system.controller)
    named = dict(system.controller.named_parameters())
    slots = system.optimizer_state["state"]
    assert isinstance(slots, Mapping)
    for name in MUTABLE_PARAMETER_NAMES:
        slot = slots[name]
        assert isinstance(slot, Mapping)
        optimizer.state[named[name]] = {
            slot_name: value.detach().to(named[name].device).clone()
            for slot_name, value in slot.items()
            if isinstance(value, torch.Tensor)
        }
    return optimizer


def _context_incidence_row_terms(
    row: v12.PublicRelationCreditRow,
) -> tuple[dict[str, torch.Tensor | None], dict[str, object]]:
    """Build rank/presence/abstention terms from detached relation support."""

    if not isinstance(row, v12.PublicRelationCreditRow):
        raise TypeError("V18 context objective requires a public credit row")
    metrics = v12._relation_valid_set_metrics(
        row.slot_positive_margins.detach(),
        row.slot_negative_margins.detach(),
        row.context_weights,
        row.context_null_weight,
    )
    valid_mask = metrics["valid_mask"]
    assert isinstance(valid_mask, torch.Tensor)
    tiny = torch.finfo(row.context_weights.dtype).tiny
    if metrics["relation_supported"] is True:
        real_mass = row.context_weights.sum().clamp_min(tiny)
        valid_mass = row.context_weights.masked_select(valid_mask).sum().clamp_min(tiny)
        real_normalized = (valid_mass / real_mass).clamp(max=1.0)
        informative = int(metrics["valid_slot_count"]) < row.context_weights.numel()
        terms = {
            "rank": (
                -torch.log(real_normalized.clamp_min(tiny))
                if informative
                else None
            ),
            "presence": -torch.log(real_mass),
            "abstain": None,
        }
        diagnostic = {
            "supported": True,
            "informative": informative,
            "unique_valid": int(metrics["valid_slot_count"]) == 1,
            "valid_slot_count": int(metrics["valid_slot_count"]),
            "rank_loss": (
                float(terms["rank"].detach().item())
                if terms["rank"] is not None
                else None
            ),
            "presence_loss": float(terms["presence"].detach().item()),
            "real_normalized_valid_mass": float(real_normalized.detach().item()),
            "full_valid_mass": float(valid_mass.detach().item()),
            "null_mass": float(row.context_null_weight.detach().item()),
        }
    else:
        abstain = -torch.log(row.context_null_weight.clamp_min(tiny))
        terms = {"rank": None, "presence": None, "abstain": abstain}
        diagnostic = {
            "supported": False,
            "informative": False,
            "unique_valid": False,
            "valid_slot_count": 0,
            "abstain_loss": float(abstain.detach().item()),
            "null_mass": float(row.context_null_weight.detach().item()),
        }
    return terms, diagnostic


def _context_incidence_objective(
    row_groups: Sequence[Sequence[v12.PublicRelationCreditRow]],
) -> tuple[torch.Tensor, dict[str, object]]:
    rows = tuple(row for group in row_groups for row in group)
    if not rows:
        raise ValueError("V18 objective requires public rows")
    rank_terms = []
    presence_terms = []
    abstain_terms = []
    diagnostics = []
    for row in rows:
        terms, diagnostic = _context_incidence_row_terms(row)
        diagnostics.append(diagnostic)
        if terms["rank"] is not None:
            rank_terms.append(terms["rank"])
        if terms["presence"] is not None:
            assert terms["presence"] is not None
            presence_terms.append(terms["presence"])
        if terms["abstain"] is not None:
            abstain_terms.append(terms["abstain"])
    zero = rows[0].context_weights.sum() * 0.0
    rank_mean = torch.stack(tuple(rank_terms)).mean() if rank_terms else zero
    presence_mean = (
        torch.stack(tuple(presence_terms)).mean() if presence_terms else zero
    )
    abstain_mean = torch.stack(tuple(abstain_terms)).mean() if abstain_terms else zero
    objective = rank_mean + _PRESENCE_WEIGHT * presence_mean + _ABSTAIN_WEIGHT * abstain_mean
    if not bool(torch.isfinite(objective).item()):
        raise RuntimeError("V18 context objective is non-finite")
    return objective, {
        "rows": len(rows),
        "supported_rows": len(presence_terms),
        "informative_rows": len(rank_terms),
        "unique_valid_rows": sum(
            diagnostic["unique_valid"] is True for diagnostic in diagnostics
        ),
        "unsupported_rows": len(abstain_terms),
        "rank_mean": float(rank_mean.detach().item()),
        "presence_mean": float(presence_mean.detach().item()),
        "abstain_mean": float(abstain_mean.detach().item()),
        "row_diagnostics": tuple(diagnostics),
    }


def _fit_context_incidence_batches(
    system: V12ChampionContextIncidenceSystem,
    stream_batches: Sequence[Sequence[v12.SoftwarePipelineStream]],
) -> dict[str, object]:
    if not stream_batches or any(not batch for batch in stream_batches):
        raise ValueError("V18 context fit requires nonempty public stream batches")
    if system.context_updates + len(stream_batches) > _CONTEXT_UPDATES:
        raise RuntimeError("V18 context fit exceeds the frozen update budget")
    if not system.controller._context_incidence_enabled:
        raise RuntimeError("V18 cannot train under the projection-zero lesion")
    _assert_source_lineage(system)
    controller = system.controller
    named = dict(controller.named_parameters())
    inherited_before = {
        name: value.detach().clone()
        for name, value in controller.state_dict().items()
        if name not in MUTABLE_PARAMETER_NAMES
    }
    mutable_before = {
        name: named[name].detach().clone() for name in MUTABLE_PARAMETER_NAMES
    }
    mixer_before = {
        name: value.detach().clone() for name, value in system.mixer.state_dict().items()
    }
    competence_before = {
        name: value.detach().clone()
        for name, value in v12.snapshot_software_reconstruction_state(
            system.competence_state
        ).items()
    }
    optimizer = (
        _context_optimizer(controller)
        if system.optimizer_state is None
        else restore_context_incidence_optimizer(system)
    )
    parameters = tuple(named[name] for name in MUTABLE_PARAMETER_NAMES)
    losses = []
    objective_diagnostics = []
    gradient_norms = []
    nonzero_gradient_names = []
    first_projection_gradient_nonzero: bool | None = None
    first_trunk_gradients_exact_zero: bool | None = None
    later_trunk_gradient_reached = False
    start_update = system.context_updates
    was_training = controller.training
    controller.train()
    try:
        for local_index, batch in enumerate(stream_batches):
            global_update = start_update + local_index
            if len(batch) != _STREAMS_PER_UPDATE:
                raise RuntimeError("V18 context update lost a public stream")
            row_groups = tuple(
                v12.public_relation_credit_rows(controller, stream) for stream in batch
            )
            objective, diagnostics = _context_incidence_objective(row_groups)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for name in MUTABLE_PARAMETER_NAMES:
                gradient = named[name].grad
                if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(f"V18 context gradient is absent/non-finite: {name}")
            nonzero = tuple(
                name
                for name in MUTABLE_PARAMETER_NAMES
                if bool(torch.count_nonzero(named[name].grad).item())
            )
            nonzero_gradient_names.append(nonzero)
            if global_update == 0:
                projection_gradient = named[_PROJECTION_PARAMETER_NAMES[0]].grad
                first_projection_gradient_nonzero = bool(
                    torch.count_nonzero(projection_gradient).item()
                )
                first_trunk_gradients_exact_zero = all(
                    not bool(torch.count_nonzero(named[name].grad).item())
                    for name in _READOUT_PARAMETER_NAMES
                )
                if not first_projection_gradient_nonzero or not first_trunk_gradients_exact_zero:
                    raise RuntimeError("V18 zero-projection gradient staging changed")
            elif any(name in nonzero for name in _READOUT_PARAMETER_NAMES):
                later_trunk_gradient_reached = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, _GRADIENT_CLIP)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("V18 clipped gradient norm is non-finite")
            optimizer.step()
            system.context_updates += 1
            system.optimizer_state = _canonical_optimizer_state(optimizer, controller)
            if any(
                not bool(torch.isfinite(named[name]).all().item())
                for name in MUTABLE_PARAMETER_NAMES
            ):
                raise RuntimeError("V18 context update produced non-finite parameters")
            current_state = controller.state_dict()
            for name, before in inherited_before.items():
                if not torch.equal(before, current_state[name].detach()):
                    raise RuntimeError(f"V18 changed inherited tensor: {name}")
            losses.append(float(objective.detach().item()))
            objective_diagnostics.append(diagnostics)
            gradient_norms.append(float(gradient_norm.detach().item()))
    finally:
        controller.train(was_training)
    if any(
        not torch.equal(before, system.mixer.state_dict()[name].detach())
        for name, before in mixer_before.items()
    ):
        raise RuntimeError("V18 context fit changed the frozen mixer")
    current_competence = v12.snapshot_software_reconstruction_state(system.competence_state)
    if any(
        not torch.equal(before, current_competence[name].detach())
        for name, before in competence_before.items()
    ):
        raise RuntimeError("V18 context fit changed frozen competence")
    _assert_source_lineage(system)
    changed_names = tuple(
        name
        for name in MUTABLE_PARAMETER_NAMES
        if not torch.equal(mutable_before[name], named[name].detach())
    )
    reached_names = tuple(
        name
        for name in MUTABLE_PARAMETER_NAMES
        if any(name in update_names for update_names in nonzero_gradient_names)
    )
    return {
        "stage": "context_incidence",
        "start_update": start_update,
        "optimizer_steps": len(stream_batches),
        "terminal_update": system.context_updates,
        "streams": sum(len(batch) for batch in stream_batches),
        "rows": sum(len(batch) for batch in stream_batches) * _ROWS_PER_STREAM,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": tuple(losses),
        "objective_diagnostics": tuple(objective_diagnostics),
        "gradient_norms": tuple(gradient_norms),
        "nonzero_gradient_parameter_names": tuple(nonzero_gradient_names),
        "gradient_reached_parameter_names": reached_names,
        "first_projection_gradient_nonzero": first_projection_gradient_nonzero,
        "first_trunk_gradients_exact_zero": first_trunk_gradients_exact_zero,
        "later_trunk_gradient_reached": later_trunk_gradient_reached,
        "changed_parameter_names": changed_names,
        "unchanged_allowed_parameter_names": tuple(
            name for name in MUTABLE_PARAMETER_NAMES if name not in changed_names
        ),
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
        "inherited_controller_exact": True,
        "mixer_exact": True,
        "competence_state_exact": True,
        "weight_decay": 0.0,
        "gradient_clip": _GRADIENT_CLIP,
        "joint_training": False,
    }


def _canonical_rows(panel: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = panel.get("row_reports")
    if not isinstance(rows, (tuple, list)) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise RuntimeError("V18 panel lost its row reports")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                int(row["stream_index"]),
                int(row["heldout_index"]),
                int(row["transition_index"]),
            ),
        )
    )


def _public_content_alignment_keys(
    streams: Sequence[v12.SoftwarePipelineStream],
    panel: Mapping[str, object],
) -> tuple[str, ...]:
    """Align evaluation rows only by the learner-visible public projection."""

    keys = []
    for row in _canonical_rows(panel):
        stream_index = int(row["stream_index"])
        heldout_index = int(row["heldout_index"])
        transition_index = int(row["transition_index"])
        if not 0 <= stream_index < len(streams):
            raise RuntimeError("V18 alignment stream is outside the public panel")
        stream = streams[stream_index]
        if not 0 <= heldout_index < len(stream.supports):
            raise RuntimeError("V18 alignment support is outside the public stream")
        task = stream.supports[heldout_index].learner
        transitions = v12._public_transitions(task)
        if not 0 <= transition_index < len(transitions):
            raise RuntimeError("V18 alignment transition is outside public evidence")
        payload = {
            "task": task.to_canonical(),
            "transition_digest": transitions[transition_index].digest,
        }
        digest = hashlib.sha256(b"project-angler.v18.public-row-alignment.v1\x00")
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        keys.append("sha256:" + digest.hexdigest())
    if len(set(keys)) != len(keys):
        raise RuntimeError("V18 public-content alignment keys are not unique")
    return tuple(keys)


def _panel_context_metrics(panel: Mapping[str, object]) -> dict[str, float | int]:
    rows = _canonical_rows(panel)
    supported = tuple(row for row in rows if row.get("relation_supported") is True)
    if len(supported) != panel.get("relation_supported_rows"):
        raise RuntimeError("V18 panel supported-row accounting changed")
    informative = tuple(
        row
        for row in supported
        if 1
        <= int(row.get("valid_slot_count", -1))
        < len(row.get("context_weights", ()))
    )
    unique = tuple(
        row for row in informative if int(row.get("valid_slot_count", -1)) == 1
    )
    informative_valid_mass_sum = 0.0
    informative_real_mass_sum = 0.0
    informative_rank_margin_sum = 0.0
    unique_real_top_one = []
    for row in informative:
        weights = tuple(float(value) for value in row["context_weights"])
        valid_slots = tuple(int(value) for value in row["valid_slots"])
        valid_set = set(valid_slots)
        if (
            not valid_slots
            or len(valid_set) != len(valid_slots)
            or any(not 0 <= index < len(weights) for index in valid_slots)
            or len(valid_slots) >= len(weights)
        ):
            raise RuntimeError("V18 informative row is malformed")
        valid_weights = tuple(weights[index] for index in valid_slots)
        invalid = tuple(
            weight for index, weight in enumerate(weights) if index not in valid_set
        )
        real_mass = sum(weights)
        valid_mass = sum(valid_weights)
        valid_max = max(valid_weights)
        invalid_max = max(invalid)
        if (
            not math.isfinite(real_mass)
            or real_mass <= 0.0
            or valid_max <= 0.0
            or invalid_max <= 0.0
        ):
            raise RuntimeError("V18 informative real-slot mass is invalid")
        informative_valid_mass_sum += valid_mass
        informative_real_mass_sum += real_mass
        informative_rank_margin_sum += math.log(valid_max) - math.log(invalid_max)
        if len(valid_slots) == 1:
            unique_real_top_one.append(valid_max > invalid_max)
    return {
        "rows": len(rows),
        "supported_rows": len(supported),
        "qualifying_streams": int(panel["streams_with_three_supported_rows"]),
        "supported_full_top_one_successes": sum(
            row.get("context_valid_set_top_one") is True for row in supported
        ),
        "supported_full_valid_set_mass": (
            sum(float(row["context_valid_set_mass"]) for row in supported)
            / len(supported)
            if supported
            else 0.0
        ),
        "unique_valid_rows": len(unique),
        "unique_valid_real_top_one_successes": sum(unique_real_top_one),
        "informative_rows": len(informative),
        "informative_valid_mass_sum": informative_valid_mass_sum,
        "informative_real_mass_sum": informative_real_mass_sum,
        "informative_real_normalized_mass": (
            informative_valid_mass_sum / informative_real_mass_sum
            if informative_real_mass_sum > 0.0
            else 0.0
        ),
        "informative_rank_margin_sum": informative_rank_margin_sum,
        "informative_rank_margin": (
            informative_rank_margin_sum / len(informative) if informative else 0.0
        ),
    }


def _relation_boundary_signature(panel: Mapping[str, object]) -> tuple[object, ...]:
    row_fields = (
        "target_slot",
        "valid_slots",
        "valid_slot_count",
        "relation_supported",
        "slot_positive_margins",
        "slot_negative_margins",
        "slot_losses",
        "responsibilities",
    )
    return (
        panel.get("streams"),
        panel.get("rows"),
        panel.get("relation_supported_rows"),
        panel.get("streams_with_three_supported_rows"),
        panel.get("supported_rows_per_stream"),
        panel.get("valid_slot_count_histogram"),
        tuple(tuple(row.get(field) for field in row_fields) for row in _canonical_rows(panel)),
    )


def _evaluate_panel(
    controller: V12ChampionContextIncidenceController,
    streams: Sequence[v12.SoftwarePipelineStream],
) -> dict[str, object]:
    learned = v12.evaluate_public_relation_credit_panel(controller, streams)
    with controller.projection_zero_lesion():
        lesion = v12.evaluate_public_relation_credit_panel(controller, streams)
    learned_alignment = _public_content_alignment_keys(streams, learned)
    lesion_alignment = _public_content_alignment_keys(streams, lesion)
    invariants = v12._evaluate_public_relation_credit_invariants(controller, streams)
    learned_metrics = _panel_context_metrics(learned)
    lesion_metrics = _panel_context_metrics(lesion)
    top_one_gain = int(learned_metrics["unique_valid_real_top_one_successes"]) - int(
        lesion_metrics["unique_valid_real_top_one_successes"]
    )
    mass_gain = float(learned_metrics["informative_real_normalized_mass"]) - float(
        lesion_metrics["informative_real_normalized_mass"]
    )
    rank_margin_gain = float(learned_metrics["informative_rank_margin"]) - float(
        lesion_metrics["informative_rank_margin"]
    )
    return {
        "learned": learned,
        "projection_zero_lesion": lesion,
        "learned_context_metrics": learned_metrics,
        "projection_zero_context_metrics": lesion_metrics,
        "causal_unique_valid_real_top_one_gain": top_one_gain,
        "causal_real_normalized_mass_gain": mass_gain,
        "causal_informative_rank_margin_gain": rank_margin_gain,
        "panel_improved": (
            top_one_gain >= 0
            and mass_gain >= 0.0
            and (top_one_gain >= 1 or mass_gain >= 0.01)
        ),
        "no_material_regression": (
            top_one_gain >= -_MAX_FOURTH_PANEL_TOP_ONE_REGRESSION
            and mass_gain >= -_MAX_FOURTH_PANEL_MASS_REGRESSION
        ),
        "canonical_public_alignment_keys": learned_alignment,
        "integrity_checks": {
            "canonical_public_alignment_exact": (
                learned_alignment == lesion_alignment
            ),
            "relation_boundary_exact_under_projection_zero": (
                _relation_boundary_signature(learned)
                == _relation_boundary_signature(lesion)
            ),
            "permutation_covariant": invariants.get("permutation_covariant") is True,
            "empty_memory_zero_exact": invariants.get("empty_memory_zero_exact") is True,
        },
    }


def _classify_context_incidence_panels(
    panel_reports: Sequence[Mapping[str, object]],
    *,
    integrity_passed: bool,
) -> dict[str, object]:
    if len(panel_reports) != _PANEL_COUNT:
        raise ValueError("V18 classification requires four fresh panels")
    learned_metrics = tuple(
        panel["learned_context_metrics"] for panel in panel_reports
    )
    lesion_metrics = tuple(
        panel["projection_zero_context_metrics"] for panel in panel_reports
    )
    if any(not isinstance(value, Mapping) for value in (*learned_metrics, *lesion_metrics)):
        raise TypeError("V18 panel metrics are invalid")
    learned_unique_success = sum(
        int(value["unique_valid_real_top_one_successes"])
        for value in learned_metrics
    )
    lesion_unique_success = sum(
        int(value["unique_valid_real_top_one_successes"])
        for value in lesion_metrics
    )
    learned_unique_rows = sum(int(value["unique_valid_rows"]) for value in learned_metrics)
    lesion_unique_rows = sum(int(value["unique_valid_rows"]) for value in lesion_metrics)
    if learned_unique_rows != lesion_unique_rows or learned_unique_rows <= 0:
        raise RuntimeError("V18 causal panels lost unique-valid row alignment")
    learned_informative_rows = sum(
        int(value["informative_rows"]) for value in learned_metrics
    )
    lesion_informative_rows = sum(
        int(value["informative_rows"]) for value in lesion_metrics
    )
    if (
        learned_informative_rows != lesion_informative_rows
        or learned_informative_rows <= 0
    ):
        raise RuntimeError("V18 causal panels lost informative-row alignment")
    learned_real_mass_sum = sum(
        float(value["informative_real_mass_sum"]) for value in learned_metrics
    )
    lesion_real_mass_sum = sum(
        float(value["informative_real_mass_sum"]) for value in lesion_metrics
    )
    if learned_real_mass_sum <= 0.0 or lesion_real_mass_sum <= 0.0:
        raise RuntimeError("V18 causal panels lost informative real mass")
    learned_mass = sum(
        float(value["informative_valid_mass_sum"]) for value in learned_metrics
    ) / learned_real_mass_sum
    lesion_mass = sum(
        float(value["informative_valid_mass_sum"]) for value in lesion_metrics
    ) / lesion_real_mass_sum
    learned_rank_margin = sum(
        float(value["informative_rank_margin_sum"]) for value in learned_metrics
    ) / learned_informative_rows
    lesion_rank_margin = sum(
        float(value["informative_rank_margin_sum"]) for value in lesion_metrics
    ) / lesion_informative_rows
    supported_rows = sum(int(value["supported_rows"]) for value in learned_metrics)
    supported_top_one = sum(
        int(value["supported_full_top_one_successes"]) for value in learned_metrics
    )
    supported_mass = (
        sum(
            float(value["supported_full_valid_set_mass"])
            * int(value["supported_rows"])
            for value in learned_metrics
        )
        / supported_rows
        if supported_rows
        else 0.0
    )
    qualifying_streams = sum(
        int(value["qualifying_streams"]) for value in learned_metrics
    )
    improved_panels = sum(panel.get("panel_improved") is True for panel in panel_reports)
    nonregressed_panels = sum(
        panel.get("no_material_regression") is True for panel in panel_reports
    )
    causal_top_one_gain = learned_unique_success - lesion_unique_success
    causal_mass_gain = learned_mass - lesion_mass
    causal_rank_margin_gain = learned_rank_margin - lesion_rank_margin
    component_checks = {
        "causal_unique_valid_real_top_one_gain": (
            causal_top_one_gain >= _CAUSAL_TOP_ONE_GAIN
        ),
        "causal_real_normalized_mass_gain": (
            causal_mass_gain >= _CAUSAL_REAL_NORMALIZED_MASS_GAIN
        ),
        "positive_aggregate_informative_rank_margin": learned_rank_margin > 0.0,
        "positive_causal_informative_rank_margin_gain": (
            causal_rank_margin_gain > 0.0
        ),
        "improvement_on_three_panels": improved_panels >= _REQUIRED_IMPROVED_PANELS,
        "no_material_regression_on_any_panel": nonregressed_panels == _PANEL_COUNT,
        "projection_zero_erases_material_gain": (
            causal_top_one_gain >= _CAUSAL_TOP_ONE_GAIN
            and causal_mass_gain >= _CAUSAL_REAL_NORMALIZED_MASS_GAIN
        ),
    }
    full_checks = {
        "relation_supported_rows": (
            supported_rows >= _RELATION_SUPPORTED_ROWS_THRESHOLD
        ),
        "relation_qualifying_streams": (
            qualifying_streams >= _RELATION_QUALIFYING_STREAMS_THRESHOLD
        ),
        "supported_context_top_one_fraction": (
            supported_rows > 0
            and supported_top_one / supported_rows >= _CONTEXT_TOP_ONE_THRESHOLD
        ),
        "supported_valid_set_mass": supported_mass >= _CONTEXT_MASS_THRESHOLD,
    }
    component_supported = all(component_checks.values())
    full_advancement = component_supported and all(full_checks.values())
    if not integrity_passed:
        classification = "INVALID_NO_CLAIM"
    elif full_advancement:
        classification = "CONTEXT_INCIDENCE_SUPPORTED"
    elif component_supported:
        classification = "CONTEXT_INCIDENCE_COMPONENT_SUPPORTED"
    else:
        classification = "CONTEXT_INCIDENCE_NOT_SUPPORTED"
    return {
        "classification": classification,
        "passed": classification == "CONTEXT_INCIDENCE_SUPPORTED",
        "component_supported": component_supported and integrity_passed,
        "component_support_checks": component_checks,
        "full_advancement_checks": full_checks,
        "improved_panels": improved_panels,
        "nonregressed_panels": nonregressed_panels,
        "aggregate_unique_valid_rows": learned_unique_rows,
        "aggregate_informative_rows": learned_informative_rows,
        "aggregate_unique_valid_real_top_one_successes": learned_unique_success,
        "projection_zero_unique_valid_real_top_one_successes": lesion_unique_success,
        "causal_unique_valid_real_top_one_gain": causal_top_one_gain,
        "aggregate_informative_real_normalized_mass": learned_mass,
        "projection_zero_informative_real_normalized_mass": lesion_mass,
        "causal_real_normalized_mass_gain": causal_mass_gain,
        "aggregate_informative_rank_margin": learned_rank_margin,
        "projection_zero_informative_rank_margin": lesion_rank_margin,
        "causal_informative_rank_margin_gain": causal_rank_margin_gain,
        "aggregate_relation_supported_rows": supported_rows,
        "aggregate_relation_qualifying_streams": qualifying_streams,
        "aggregate_supported_context_top_one_fraction": (
            supported_top_one / supported_rows if supported_rows else 0.0
        ),
        "aggregate_supported_valid_set_mass": supported_mass,
    }


def evaluate_v12_champion_context_incidence(
    system: V12ChampionContextIncidenceSystem,
) -> dict[str, object]:
    _assert_source_lineage(system)
    plan = v12_champion_context_incidence_plan()
    commitments = plan["commitments"]
    panel_seed_pairs = plan["panel_seed_pairs"]
    assert isinstance(commitments, tuple)
    assert isinstance(panel_seed_pairs, tuple)
    panels = tuple(
        _evaluate_panel(
            system.controller,
            v12._relation_credit_panel_streams(commitments, seed_pairs),
        )
        for seed_pairs in panel_seed_pairs
    )
    source_exact = (
        inherited_v12_controller_digest(system.controller) == V12_CONTROLLER_DIGEST
        and v12.anonymous_conflict_mixer_digest(system.mixer) == V12_MIXER_DIGEST
        and v12.software_reconstruction_state_digest(system.competence_state)
        == V12_COMPETENCE_DIGEST
    )
    integrity = {
        "source_lineage_exact": source_exact,
        "context_updates_exact": system.context_updates == _CONTEXT_UPDATES,
        "four_disjoint_fresh_panels": len(panels) == _PANEL_COUNT,
        "panel_integrity": all(
            all(panel["integrity_checks"].values()) for panel in panels
        ),
    }
    classification = _classify_context_incidence_panels(
        panels,
        integrity_passed=all(integrity.values()),
    )
    return {
        "protocol_id": PROTOCOL_ID,
        **classification,
        "plan_digest": plan["plan_digest"],
        "panels": panels,
        "integrity_checks": integrity,
        "joint_training_performed": False,
        "srwm_used": False,
        "replay_used": False,
        "stored_examples_used": False,
        "deterministic_solver_used": False,
        "identity_inputs_used": False,
        "development_or_final_access": False,
        "scalar_judge_calls": 0,
        "control_streams_used": 0,
        "wrong_evidence_training_streams": 0,
    }


def fit_v12_champion_context_incidence(
    system: V12ChampionContextIncidenceSystem,
) -> dict[str, object]:
    """Execute the one frozen 256-update V18 fit without writing artifacts."""

    if system.context_updates != 0 or system.optimizer_state is not None:
        raise RuntimeError("V18 is a one-shot context fit")
    plan = v12_champion_context_incidence_plan()
    commitments = plan["commitments"]
    seed_batches = plan["training_seed_batches"]
    assert isinstance(commitments, tuple)
    assert isinstance(seed_batches, tuple)
    batches = v12._relation_credit_stream_batches(commitments, seed_batches)
    fit = _fit_context_incidence_batches(system, batches)
    if (
        fit["optimizer_steps"] != _CONTEXT_UPDATES
        or fit["streams"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE
        or fit["rows"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM
        or fit["inherited_controller_exact"] is not True
        or fit["mixer_exact"] is not True
        or fit["competence_state_exact"] is not True
        or fit["first_projection_gradient_nonzero"] is not True
        or fit["first_trunk_gradients_exact_zero"] is not True
        or fit["later_trunk_gradient_reached"] is not True
        or set(fit["gradient_reached_parameter_names"]) != set(MUTABLE_PARAMETER_NAMES)
        or set(fit["changed_parameter_names"]) != set(MUTABLE_PARAMETER_NAMES)
    ):
        raise RuntimeError("V18 semantic fit lost its frozen accounting")
    evaluation = evaluate_v12_champion_context_incidence(system)
    return {
        "protocol_id": PROTOCOL_ID,
        "classification": evaluation["classification"],
        "passed": evaluation["passed"],
        "plan": plan,
        "fit": fit,
        "evaluation": evaluation,
        "parameter_report": context_incidence_parameter_report(
            system.controller,
            system.mixer,
        ),
        "source": asdict(system.source),
        "terminal_controller_digest": v12.software_pipeline_model_digest(
            system.controller
        ),
        "terminal_mutable_digest": context_incidence_mutable_digest(
            system.controller
        ),
        "terminal_system_digest": context_incidence_system_digest(system),
        "context_updates": system.context_updates,
        "joint_training_performed": False,
    }


def _checkpoint_config() -> dict[str, object]:
    return {
        "context_width": _CONTEXT_WIDTH,
        "readout_seed": _READOUT_SEED,
        "readout_type": "RelationAxisSetReadout",
        "projection_bias": False,
        "mutable_parameter_names": MUTABLE_PARAMETER_NAMES,
    }


def _checkpoint_payload(
    system: V12ChampionContextIncidenceSystem,
) -> dict[str, object]:
    _assert_source_lineage(system)
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_context_incidence_plan()["plan_digest"],
        "source": asdict(system.source),
        "profile": asdict(system.controller.profile),
        "config": _checkpoint_config(),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in system.controller.state_dict().items()
        },
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": context_incidence_mutable_digest(system.controller),
        "inherited_controller_digest": inherited_v12_controller_digest(
            system.controller
        ),
        "mixer_config": {
            "feature_count": system.mixer.feature_count,
            "hidden_width": system.mixer.hidden_width,
            "anchor_weight": system.mixer.anchor_weight,
        },
        "mixer_state": {
            name: value.detach().cpu().clone()
            for name, value in system.mixer.state_dict().items()
        },
        "mixer_digest": v12.anonymous_conflict_mixer_digest(system.mixer),
        "competence_state": {
            name: value.detach().cpu().clone()
            for name, value in v12.snapshot_software_reconstruction_state(
                system.competence_state
            ).items()
        },
        "competence_digest": v12.software_reconstruction_state_digest(
            system.competence_state
        ),
        "context_updates": system.context_updates,
        "optimizer_state": copy.deepcopy(system.optimizer_state),
        "optimizer_digest": context_incidence_optimizer_digest(
            system.optimizer_state
        ),
        "parameter_report": context_incidence_parameter_report(
            system.controller,
            system.mixer,
        ),
        "system_digest": context_incidence_system_digest(system),
    }


def save_v12_champion_context_incidence_checkpoint(
    path: str | Path,
    system: V12ChampionContextIncidenceSystem,
) -> None:
    torch.save(_checkpoint_payload(system), Path(path))


def load_v12_champion_context_incidence_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionContextIncidenceSystem:
    _verify_frozen_dependencies()
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected_keys = {
        "version",
        "protocol_id",
        "plan_digest",
        "source",
        "profile",
        "config",
        "model_state",
        "controller_digest",
        "mutable_digest",
        "inherited_controller_digest",
        "mixer_config",
        "mixer_state",
        "mixer_digest",
        "competence_state",
        "competence_digest",
        "context_updates",
        "optimizer_state",
        "optimizer_digest",
        "parameter_report",
        "system_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("V18 checkpoint fields are invalid")
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"]
        != v12_champion_context_incidence_plan()["plan_digest"]
        or payload["config"] != _checkpoint_config()
        or payload["source"] != asdict(_expected_source_binding())
    ):
        raise RuntimeError("V18 checkpoint identity is invalid")
    profile = v12.SoftwarePipelineRunProfile(**payload["profile"])
    if v12.SOFTWARE_PIPELINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("V18 checkpoint profile is not registered")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = v17._cuda_rng_snapshot(device)
    try:
        controller = V12ChampionContextIncidenceController(profile).to(device)
        controller.load_state_dict(payload["model_state"], strict=True)
        mixer_config = payload["mixer_config"]
        if not isinstance(mixer_config, dict) or set(mixer_config) != {
            "feature_count",
            "hidden_width",
            "anchor_weight",
        }:
            raise RuntimeError("V18 mixer config is invalid")
        mixer = v12.AnonymousConflictMixer(**mixer_config).to(device)
        mixer.load_state_dict(payload["mixer_state"], strict=True)
        for parameter in mixer.parameters():
            parameter.requires_grad_(False)
    finally:
        torch.set_rng_state(cpu_rng_state)
        v17._restore_cuda_rng_snapshot(cuda_rng_states)
    try:
        state = v12.restore_software_reconstruction_state(payload["competence_state"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("V18 checkpoint competence state is invalid") from error
    updates = payload["context_updates"]
    optimizer_state = payload["optimizer_state"]
    if type(updates) is not int or not 0 <= updates <= _CONTEXT_UPDATES:
        raise RuntimeError("V18 checkpoint context update count is invalid")
    if updates == 0:
        if optimizer_state is not None:
            raise RuntimeError("V18 fresh checkpoint unexpectedly has optimizer state")
    elif not isinstance(optimizer_state, Mapping):
        raise RuntimeError("V18 learned checkpoint lost optimizer state")
    else:
        _validate_optimizer_state(
            optimizer_state,
            controller,
            expected_steps=updates,
        )
    system = V12ChampionContextIncidenceSystem(
        controller=controller,
        mixer=mixer,
        competence_state=state,
        source=_expected_source_binding(),
        context_updates=updates,
        optimizer_state=copy.deepcopy(optimizer_state),
    )
    if (
        v12.software_pipeline_model_digest(controller) != payload["controller_digest"]
        or context_incidence_mutable_digest(controller) != payload["mutable_digest"]
        or inherited_v12_controller_digest(controller)
        != payload["inherited_controller_digest"]
        or v12.anonymous_conflict_mixer_digest(mixer) != payload["mixer_digest"]
        or v12.software_reconstruction_state_digest(state)
        != payload["competence_digest"]
        or context_incidence_optimizer_digest(system.optimizer_state)
        != payload["optimizer_digest"]
        or context_incidence_parameter_report(controller, mixer)
        != payload["parameter_report"]
        or context_incidence_system_digest(system) != payload["system_digest"]
    ):
        raise RuntimeError("V18 checkpoint digest or report mismatch")
    _assert_source_lineage(system)
    controller.eval()
    mixer.eval()
    return system
