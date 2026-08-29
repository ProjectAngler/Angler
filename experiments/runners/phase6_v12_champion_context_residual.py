"""V12-champion context-only residual composition successor.

V17 retains the complete terminal V12 controller, conflict mixer, and
competence state.  Four zero-output anonymous residual experts expand only the
context read.  Their outputs are composed by the already-audited symmetric
V15 implementation of V13's all-active soft composer.
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

from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_software_pipeline_reconstruction as v12


PROTOCOL_ID = "phase6.public-v12-champion-context-residual.v17"
CHECKPOINT_VERSION = "angler.phase6-v12-champion-context-residual.v1"
V12_CHECKPOINT_SHA256 = (
    "B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C"
)
V12_CONTROLLER_DIGEST = (
    "sha256:30fc9a965e3bd683afff162171600ab3095c387dfd194cf9613c45dbdc2111b9"
)
V12_MIXER_DIGEST = (
    "sha256:e7d82d68ebc23ec9b0c78b27017062c50f57274d74e6ec94ccee3e0e47eb19fc"
)
V12_COMPETENCE_DIGEST = (
    "sha256:41686aa48bc15412e2be15721cedc7bad7237914f0fb8ff4bca6233022eb9ab6"
)
V12_SYSTEM_DIGEST = (
    "sha256:1e4c0e3e0afe608ac220b2892d34ca873d27b685cbd1ad4b05f107f66e610bd6"
)
FROZEN_DEPENDENCY_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
}

_PLAN_DIGEST_DOMAIN = b"project-angler.v12-champion-context-residual.plan.v1\x00"
_MUTABLE_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-context-residual.mutable.v1\x00"
)
_OPTIMIZER_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-context-residual.optimizer.v1\x00"
)
_SYSTEM_DIGEST_DOMAIN = b"project-angler.v12-champion-context-residual.system.v1\x00"

_EXPERT_COUNT = 4
_EXPERT_RANK = 8
_CONTEXT_WIDTH = 32
_COMPOSER_HIDDEN_WIDTH = 41
_COMPOSER_ANCHOR_WEIGHT = 0.50
_EXPERT_SEEDS = (2_026_083_701, 2_026_083_702, 2_026_083_703, 2_026_083_704)
_COMPOSER_SEED = 2_026_083_705

_CONTEXT_UPDATES = 25
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_EXPERT_LEARNING_RATE = 3.0e-4
_COMPOSER_LEARNING_RATE = 1.0e-3
_GRADIENT_CLIP = 5.0
_TRAIN_TOPOLOGY_BASE = 8_001_000_001
_TRAIN_SURFACE_BASE = 8_041_000_001
_PANEL_TOPOLOGY_BASE = 8_081_000_001
_PANEL_SURFACE_BASE = 8_121_000_001
_RERENDER_SURFACE_BASE = 8_161_000_001

_CONTEXT_TOP_ONE_THRESHOLD = 0.80
_CONTEXT_MASS_THRESHOLD = 0.60
_RESIDUAL_TOP_ONE_GAIN = 3
_RESIDUAL_MASS_GAIN = 0.05
_COMPOSER_TOP_ONE_GAIN = 2
_COMPOSER_MASS_GAIN = 0.02
_LESION_TOP_ONE_LOSS = 1
_LESION_MASS_LOSS = 0.01
_REQUIRED_CAUSAL_LESIONS = 2

_EXPERT_PARAMETER_NAMES = tuple(
    f"context_residual_experts.{index}.{projection}.weight"
    for index in range(_EXPERT_COUNT)
    for projection in ("down", "up")
)
_COMPOSER_PARAMETER_NAMES = (
    "context_composer.local_encoder.0.weight",
    "context_composer.local_encoder.0.bias",
    "context_composer.local_encoder.1.weight",
    "context_composer.local_encoder.1.bias",
    "context_composer.local_encoder.3.weight",
    "context_composer.local_encoder.3.bias",
    "context_composer.residual_scorer.0.weight",
    "context_composer.residual_scorer.0.bias",
    "context_composer.residual_scorer.2.weight",
)
MUTABLE_PARAMETER_NAMES = _EXPERT_PARAMETER_NAMES + _COMPOSER_PARAMETER_NAMES
_MUTABLE_PREFIXES = ("context_residual_experts.", "context_composer.")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _cuda_rng_snapshot(device: torch.device | str) -> tuple[torch.Tensor, ...] | None:
    selected = torch.device(device)
    if selected.type != "cuda":
        return None
    return tuple(torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count()))


def _restore_cuda_rng_snapshot(states: tuple[torch.Tensor, ...] | None) -> None:
    if states is not None:
        for index, state in enumerate(states):
            torch.cuda.set_rng_state(state, index)


def frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES}


def _verify_frozen_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    expected_modules = {
        Path(v12.__file__).resolve(): (
            root / "experiments/runners/phase6_software_pipeline_reconstruction.py"
        ).resolve(),
        Path(v15.__file__).resolve(): (
            root / "experiments/runners/phase6_cross_variation_plasticity.py"
        ).resolve(),
    }
    if any(observed != expected for observed, expected in expected_modules.items()):
        raise RuntimeError("V17 imported a shadowed V12/V15 dependency")
    if frozen_dependency_hashes() != FROZEN_DEPENDENCY_HASHES:
        raise RuntimeError("V17 frozen V12/V15 source dependency changed")


def _update_tensor_digest(
    digest: "hashlib._Hash",
    name: str,
    value: torch.Tensor,
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


def _json_digest(domain: bytes, payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(domain)
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def _plan_payload() -> dict[str, object]:
    commitments = tuple(v12.software_pipeline_mechanism_partition("train")[:8])
    if len(commitments) != _STREAMS_PER_UPDATE or len(set(commitments)) != 8:
        raise RuntimeError("V17 requires eight distinct public commitments")
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
    panel = tuple(
        (
            _PANEL_TOPOLOGY_BASE + 1_000 * commitment,
            _PANEL_SURFACE_BASE + 1_000 * commitment,
        )
        for commitment in range(_STREAMS_PER_UPDATE)
    )
    rerender = tuple(
        (
            _PANEL_TOPOLOGY_BASE + 1_000 * commitment,
            _RERENDER_SURFACE_BASE + 1_000 * commitment,
        )
        for commitment in range(_STREAMS_PER_UPDATE)
    )
    training_pairs = {pair for batch in training for pair in batch}
    all_pairs = tuple(pair for batch in training for pair in batch) + panel + rerender
    if (
        len(training_pairs) != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE
        or training_pairs & set(panel)
        or training_pairs & set(rerender)
        or set(panel) & set(rerender)
        or any(seed < 8_000_000_001 for pair in all_pairs for seed in pair)
    ):
        raise RuntimeError("V17 stream identities are not fresh and disjoint")
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
        "expert_seeds": _EXPERT_SEEDS,
        "composer_seed": _COMPOSER_SEED,
        "expert_count": _EXPERT_COUNT,
        "expert_rank": _EXPERT_RANK,
        "context_width": _CONTEXT_WIDTH,
        "composer_hidden_width": _COMPOSER_HIDDEN_WIDTH,
        "composer_anchor_weight": _COMPOSER_ANCHOR_WEIGHT,
        "training_seed_batches": training,
        "panel_seed_pairs": panel,
        "rerender_seed_pairs": rerender,
        "context_updates": _CONTEXT_UPDATES,
        "streams_per_update": _STREAMS_PER_UPDATE,
        "rows_per_stream": _ROWS_PER_STREAM,
        "expert_learning_rate": _EXPERT_LEARNING_RATE,
        "composer_learning_rate": _COMPOSER_LEARNING_RATE,
        "weight_decay": 0.0,
        "gradient_clip": _GRADIENT_CLIP,
        "context_gate": {
            "supported_top_one": _CONTEXT_TOP_ONE_THRESHOLD,
            "supported_valid_set_mass": _CONTEXT_MASS_THRESHOLD,
        },
        "causal_support": {
            "residual_top_one_gain": _RESIDUAL_TOP_ONE_GAIN,
            "residual_mass_gain": _RESIDUAL_MASS_GAIN,
            "composer_top_one_gain": _COMPOSER_TOP_ONE_GAIN,
            "composer_mass_gain": _COMPOSER_MASS_GAIN,
            "lesion_top_one_loss": _LESION_TOP_ONE_LOSS,
            "lesion_mass_loss": _LESION_MASS_LOSS,
            "required_distinct_lesions": _REQUIRED_CAUSAL_LESIONS,
        },
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
        "joint_training": False,
        "replay": False,
        "write_router": False,
    }


def v12_champion_context_residual_plan() -> dict[str, object]:
    """Return the complete frozen V17 plan and its canonical digest."""

    payload = _plan_payload()
    return {
        **payload,
        "plan_digest": _json_digest(_PLAN_DIGEST_DOMAIN, payload),
    }


class ZeroOutputContextResidualExpert(nn.Module):
    """Small anonymous context transform with exact-zero initial output."""

    def __init__(self, width: int = _CONTEXT_WIDTH, rank: int = _EXPERT_RANK) -> None:
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
        ):
            raise ValueError("context residual dimensions must be positive integers")
        self.width = width
        self.rank = rank
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim < 2
            or values.shape[-1] != self.width
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all().item())
        ):
            raise ValueError("context residual input must be a finite code tensor")
        result = values + self.up(F.silu(self.down(values)))
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("context residual expert produced non-finite codes")
        return result


class V12ChampionContextResidualController(v12.SoftwarePipelineController):
    """V12 controller with one additive anonymous context-read expansion."""

    def __init__(self, profile: v12.SoftwarePipelineRunProfile) -> None:
        if profile != v12.SOFTWARE_PIPELINE_PROFILES["smoke"]:
            raise ValueError("V17 accepts only the V12 smoke profile")
        cpu_rng_state = torch.get_rng_state()
        try:
            super().__init__(profile)
            experts = []
            for seed in _EXPERT_SEEDS:
                torch.default_generator.manual_seed(seed)
                experts.append(ZeroOutputContextResidualExpert())
            self.context_residual_experts = nn.ModuleList(experts)
            torch.default_generator.manual_seed(_COMPOSER_SEED)
            self.context_composer = v15.SymmetricV15RelationComposer(
                cell_count=_EXPERT_COUNT,
                cell_width=_CONTEXT_WIDTH,
                hidden_width=_COMPOSER_HIDDEN_WIDTH,
                anchor_weight=_COMPOSER_ANCHOR_WEIGHT,
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
        self._context_diagnostic: tuple[str, int | None] | None = None
        self._context_capture: list[dict[str, object]] | None = None
        _enforce_mutable_scope(self)

    def _context_composed_read(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diagnostic = self._context_diagnostic
        if diagnostic is not None and diagnostic[0] == "residual_off":
            fused = v12.SoftwarePipelineController._context_pair_logits(
                self,
                query_codes,
                stored_codes,
            )
            empty = fused.new_empty((*fused.shape, 0))
            self._capture_context_result(fused, empty, empty)
            return fused, empty, empty
        query_cells = torch.stack(
            tuple(expert(query_codes) for expert in self.context_residual_experts),
            dim=1,
        )
        stored_cells = torch.stack(
            tuple(expert(stored_codes) for expert in self.context_residual_experts),
            dim=1,
        )
        cell_logits = torch.stack(
            tuple(
                v12.SoftwarePipelineController._context_pair_logits(
                    self,
                    query_cells[:, index, :],
                    stored_cells[:, index, :],
                )
                for index in range(_EXPERT_COUNT)
            ),
            dim=-1,
        )
        fused, weights, _, _ = self.context_composer(
            query_cells,
            stored_cells,
            cell_logits,
        )
        if diagnostic is not None:
            kind, index = diagnostic
            if kind == "uniform":
                weights = torch.full_like(weights, 1.0 / _EXPERT_COUNT)
            elif kind == "drop" and index is not None and 0 <= index < _EXPERT_COUNT:
                weights = weights.clone()
                weights[..., index] = 0.0
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).tiny
                )
            elif kind != "learned":
                raise RuntimeError("V17 context diagnostic is invalid")
            fused = (
                weights.to(torch.float64) * cell_logits.to(torch.float64)
            ).sum(dim=-1).to(cell_logits.dtype)
        self._capture_context_result(fused, weights, cell_logits)
        return fused, weights, cell_logits

    def _capture_context_result(
        self,
        fused: torch.Tensor,
        weights: torch.Tensor,
        cell_logits: torch.Tensor,
    ) -> None:
        if self._context_capture is not None:
            pairwise = (
                tuple(
                    float(
                        (cell_logits[..., left] - cell_logits[..., right])
                        .detach()
                        .abs()
                        .mean()
                        .item()
                    )
                    for left in range(_EXPERT_COUNT)
                    for right in range(left + 1, _EXPERT_COUNT)
                )
                if cell_logits.shape[-1] == _EXPERT_COUNT
                else ()
            )
            self._context_capture.append(
                {
                    "mean_weights": (
                        tuple(
                            float(value)
                            for value in weights.detach().mean(dim=(0, 1)).tolist()
                        )
                        if weights.shape[-1] == _EXPERT_COUNT
                        else ()
                    ),
                    "mean_pairwise_logit_differences": pairwise,
                    "context_logits": fused.detach().clone(),
                    "context_weights": torch.softmax(
                        torch.cat(
                            (
                                fused / v12._RELATION_CONTEXT_AUX_TEMPERATURE,
                                fused.new_zeros(fused.shape[0], 1),
                            ),
                            dim=-1,
                        ),
                        dim=-1,
                    )[:, : fused.shape[1]].detach().clone(),
                    "context_null_weights": torch.softmax(
                        torch.cat(
                            (
                                fused / v12._RELATION_CONTEXT_AUX_TEMPERATURE,
                                fused.new_zeros(fused.shape[0], 1),
                            ),
                            dim=-1,
                        ),
                        dim=-1,
                    )[:, fused.shape[1]].detach().clone(),
                    "final_evidence_scores": None,
                }
            )

    def _context_pair_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self._context_composed_read(query_codes, stored_codes)[0]

    @contextmanager
    def context_diagnostic(
        self,
        kind: str,
        index: int | None = None,
    ) -> Iterator[None]:
        if self._context_diagnostic is not None:
            raise RuntimeError("nested V17 diagnostics are forbidden")
        if kind not in {"residual_off", "uniform", "learned", "drop"}:
            raise ValueError("unknown V17 context diagnostic")
        if (kind == "drop") != (type(index) is int and 0 <= index < _EXPERT_COUNT):
            raise ValueError("drop diagnostics require one valid expert index")
        self._context_diagnostic = (kind, index)
        try:
            yield
        finally:
            self._context_diagnostic = None

    @contextmanager
    def capture_context_reads(self) -> Iterator[list[dict[str, object]]]:
        if self._context_capture is not None:
            raise RuntimeError("nested V17 context capture is forbidden")
        records: list[dict[str, object]] = []
        self._context_capture = records

        def relation_hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not records or records[-1]["final_evidence_scores"] is not None:
                raise RuntimeError("V17 evidence capture lost read-call alignment")
            relation_logits = torch.tanh(output.squeeze(-1)).detach()
            context_weights = records[-1]["context_weights"]
            assert isinstance(context_weights, torch.Tensor)
            if context_weights.shape != relation_logits.shape:
                raise RuntimeError("V17 evidence capture tensor shapes changed")
            records[-1]["final_evidence_scores"] = (
                context_weights * relation_logits
            ).sum(dim=-1).detach().clone()

        hook = self.relation_comparator.register_forward_hook(relation_hook)
        try:
            yield records
        finally:
            hook.remove()
            self._context_capture = None


def _enforce_mutable_scope(controller: V12ChampionContextResidualController) -> None:
    actual = tuple(name for name, _ in controller.named_parameters() if name.startswith(_MUTABLE_PREFIXES))
    if actual != MUTABLE_PARAMETER_NAMES:
        raise RuntimeError("V17 mutable parameter identity changed")
    if len(actual) != 17:
        raise RuntimeError("V17 must expose exactly 17 mutable tensors")
    named = dict(controller.named_parameters())
    if sum(named[name].numel() for name in actual) != 10_007:
        raise RuntimeError("V17 mutable parameter count changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name in MUTABLE_PARAMETER_NAMES)


def context_residual_parameter_report(
    controller: V12ChampionContextResidualController,
    mixer: v12.AnonymousConflictMixer,
) -> dict[str, object]:
    if not isinstance(controller, V12ChampionContextResidualController):
        raise TypeError("controller must be the V17 controller")
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


@dataclass(frozen=True, slots=True)
class V12ChampionSourceBinding:
    checkpoint_sha256: str
    controller_digest: str
    mixer_digest: str
    competence_digest: str
    system_digest: str


def _expected_source_binding() -> V12ChampionSourceBinding:
    return V12ChampionSourceBinding(
        checkpoint_sha256=V12_CHECKPOINT_SHA256,
        controller_digest=V12_CONTROLLER_DIGEST,
        mixer_digest=V12_MIXER_DIGEST,
        competence_digest=V12_COMPETENCE_DIGEST,
        system_digest=V12_SYSTEM_DIGEST,
    )


@dataclass(slots=True)
class V12ChampionContextResidualSystem:
    controller: V12ChampionContextResidualController
    mixer: v12.AnonymousConflictMixer
    competence_state: v12.SoftwareReconstructionState
    source: V12ChampionSourceBinding
    context_updates: int = 0
    optimizer_state: dict[str, object] | None = None


def _inherited_state(
    controller: V12ChampionContextResidualController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if not name.startswith(_MUTABLE_PREFIXES)
    }


def _mutable_state(
    controller: V12ChampionContextResidualController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if name.startswith(_MUTABLE_PREFIXES)
    }


def _base_controller_from_successor(
    controller: V12ChampionContextResidualController,
) -> v12.SoftwarePipelineController:
    cpu_rng_state = torch.get_rng_state()
    try:
        base = v12.SoftwarePipelineController(controller.profile)
    finally:
        torch.set_rng_state(cpu_rng_state)
    state = {
        name: value.detach().cpu().clone()
        for name, value in _inherited_state(controller).items()
    }
    base.load_state_dict(state, strict=True)
    base.eval()
    return base


def inherited_v12_controller_digest(
    controller: V12ChampionContextResidualController,
) -> str:
    return v12.software_pipeline_model_digest(_base_controller_from_successor(controller))


def context_residual_mutable_digest(
    controller: V12ChampionContextResidualController,
) -> str:
    return _mapping_digest(_MUTABLE_DIGEST_DOMAIN, _mutable_state(controller))


def _source_system_digest(system: V12ChampionContextResidualSystem) -> str:
    return v12.public_relation_conflict_system_digest(
        _base_controller_from_successor(system.controller),
        system.mixer,
        system.competence_state,
    )


def context_residual_system_digest(system: V12ChampionContextResidualSystem) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_context_residual_plan()["plan_digest"],
        "source": asdict(system.source),
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": context_residual_mutable_digest(system.controller),
        "mixer_digest": v12.anonymous_conflict_mixer_digest(system.mixer),
        "competence_digest": v12.software_reconstruction_state_digest(
            system.competence_state
        ),
        "context_updates": system.context_updates,
        "optimizer_digest": context_residual_optimizer_digest(
            system.optimizer_state
        ),
    }
    return _json_digest(_SYSTEM_DIGEST_DOMAIN, payload)


def _assert_source_lineage(system: V12ChampionContextResidualSystem) -> None:
    if system.source != _expected_source_binding():
        raise RuntimeError("V17 terminal V12 source binding changed")
    if inherited_v12_controller_digest(system.controller) != system.source.controller_digest:
        raise RuntimeError("V17 changed an inherited V12 controller byte")
    if v12.anonymous_conflict_mixer_digest(system.mixer) != system.source.mixer_digest:
        raise RuntimeError("V17 changed the inherited V12 conflict mixer")
    if any(parameter.requires_grad for parameter in system.mixer.parameters()):
        raise RuntimeError("V17 inherited conflict mixer is not frozen")
    if (
        v12.software_reconstruction_state_digest(system.competence_state)
        != system.source.competence_digest
    ):
        raise RuntimeError("V17 changed the inherited V12 competence state")
    if _source_system_digest(system) != system.source.system_digest:
        raise RuntimeError("V17 source system lineage changed")
    if type(system.context_updates) is not int:
        raise RuntimeError("V17 context update count is invalid")
    if system.context_updates == 0:
        if system.optimizer_state is not None:
            raise RuntimeError("V17 fresh lineage unexpectedly has optimizer moments")
    elif 0 < system.context_updates <= _CONTEXT_UPDATES:
        if not isinstance(system.optimizer_state, Mapping):
            raise RuntimeError("V17 learned lineage lost optimizer moments")
        _validate_optimizer_state(
            system.optimizer_state,
            system.controller,
            expected_steps=system.context_updates,
        )
    else:
        raise RuntimeError("V17 context update count is invalid")
    _enforce_mutable_scope(system.controller)


def _migrate_loaded_v12_system(
    controller: v12.SoftwarePipelineController,
    mixer: v12.AnonymousConflictMixer,
    state: v12.SoftwareReconstructionState,
    *,
    source_checkpoint_sha256: str,
    device: torch.device | str = "cpu",
) -> V12ChampionContextResidualSystem:
    """Lift an already strictly verified V12 system into the additive V17 type."""

    _verify_frozen_dependencies()
    if source_checkpoint_sha256.upper() != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V17 accepts only the frozen terminal V12 checkpoint")
    if type(controller) is not v12.SoftwarePipelineController:
        raise TypeError("V17 migration requires the exact V12 controller type")
    if not isinstance(mixer, v12.AnonymousConflictMixer):
        raise TypeError("V17 migration requires the V12 conflict mixer")
    if not isinstance(state, v12.SoftwareReconstructionState):
        raise TypeError("V17 migration requires the V12 competence state")
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
        raise RuntimeError("V17 source payload is not the terminal V12 lineage")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = _cuda_rng_snapshot(device)
    try:
        successor = V12ChampionContextResidualController(controller.profile).to(device)
        load_result = successor.load_state_dict(controller.state_dict(), strict=False)
        if tuple(sorted(load_result.missing_keys)) != tuple(sorted(MUTABLE_PARAMETER_NAMES)):
            raise RuntimeError("V17 migration missing-key set changed")
        if load_result.unexpected_keys:
            raise RuntimeError("V17 migration received unexpected V12 keys")
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
        _restore_cuda_rng_snapshot(cuda_rng_states)
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
        raise RuntimeError("V17 migration changed inherited V12 state")
    system = V12ChampionContextResidualSystem(
        controller=successor,
        mixer=cloned_mixer,
        competence_state=cloned_state,
        source=source,
    )
    successor.eval()
    cloned_mixer.eval()
    _assert_source_lineage(system)
    report = context_residual_parameter_report(successor, cloned_mixer)
    if report["controller_parameters"] != 275_613 or report["complete_learned_system_parameters"] != 279_017:
        raise RuntimeError("V17 migrated parameter count changed")
    return system


def load_v12_champion_context_residual_source(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionContextResidualSystem:
    """Strictly hash, load, and migrate the one accepted terminal V12 checkpoint."""

    actual = _sha256_file(path)
    if actual != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V17 source checkpoint SHA-256 is not the frozen V12 identity")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = _cuda_rng_snapshot(device)
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
        _restore_cuda_rng_snapshot(cuda_rng_states)


def _context_optimizer(
    controller: V12ChampionContextResidualController,
) -> torch.optim.AdamW:
    named = dict(controller.named_parameters())
    return torch.optim.AdamW(
        (
            {
                "params": [named[name] for name in _EXPERT_PARAMETER_NAMES],
                "lr": _EXPERT_LEARNING_RATE,
            },
            {
                "params": [named[name] for name in _COMPOSER_PARAMETER_NAMES],
                "lr": _COMPOSER_LEARNING_RATE,
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
        {"names": _EXPERT_PARAMETER_NAMES, "lr": _EXPERT_LEARNING_RATE},
        {"names": _COMPOSER_PARAMETER_NAMES, "lr": _COMPOSER_LEARNING_RATE},
    )


def _canonical_optimizer_state(
    optimizer: torch.optim.AdamW,
    controller: V12ChampionContextResidualController,
) -> dict[str, object]:
    named = dict(controller.named_parameters())
    expected_groups = _optimizer_groups()
    for actual, expected in zip(optimizer.param_groups, expected_groups, strict=True):
        actual_names = tuple(
            next(name for name, parameter in named.items() if parameter is value)
            for value in actual["params"]
        )
        if actual_names != expected["names"] or float(actual["lr"]) != expected["lr"]:
            raise RuntimeError("V17 optimizer parameter groups changed")
    slots: dict[str, dict[str, torch.Tensor]] = {}
    for name in MUTABLE_PARAMETER_NAMES:
        state = optimizer.state.get(named[name])
        if not isinstance(state, dict) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(f"V17 optimizer state is incomplete: {name}")
        slots[name] = {
            slot_name: value.detach().cpu().clone()
            for slot_name, value in state.items()
        }
    result: dict[str, object] = {
        "version": "adamw-name-keyed.v1",
        "groups": expected_groups,
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
    controller: V12ChampionContextResidualController,
    *,
    expected_steps: int | None = None,
) -> None:
    if set(value) != {"version", "groups", "hyperparameters", "state"}:
        raise RuntimeError("V17 optimizer checkpoint fields changed")
    if (
        value["version"] != "adamw-name-keyed.v1"
        or value["groups"] != _optimizer_groups()
        or value["hyperparameters"]
        != {
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
    ):
        raise RuntimeError("V17 optimizer configuration changed")
    slots = value["state"]
    if not isinstance(slots, Mapping) or set(slots) != set(MUTABLE_PARAMETER_NAMES):
        raise RuntimeError("V17 optimizer slot ownership changed")
    named = dict(controller.named_parameters())
    observed_steps = []
    for name in MUTABLE_PARAMETER_NAMES:
        slot = slots[name]
        if not isinstance(slot, Mapping) or set(slot) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(f"V17 optimizer slot fields changed: {name}")
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
            raise RuntimeError(f"V17 optimizer slot tensor changed: {name}")
        observed_steps.append(int(step.item()))
    if len(set(observed_steps)) != 1 or (
        expected_steps is not None and observed_steps[0] != expected_steps
    ):
        raise RuntimeError("V17 optimizer step counters changed")


def context_residual_optimizer_digest(
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
        raise RuntimeError("V17 optimizer digest requires named slots")
    for parameter_name in sorted(slots):
        slot = slots[parameter_name]
        if not isinstance(slot, Mapping):
            raise RuntimeError("V17 optimizer digest slot is invalid")
        for slot_name in sorted(slot):
            tensor = slot[slot_name]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("V17 optimizer digest value is not a tensor")
            _update_tensor_digest(
                digest,
                f"{parameter_name}.{slot_name}",
                tensor,
            )
    return "sha256:" + digest.hexdigest()


def restore_context_residual_optimizer(
    system: V12ChampionContextResidualSystem,
) -> torch.optim.AdamW:
    """Restore the terminal name-keyed AdamW moments for a future successor."""

    if system.optimizer_state is None:
        raise RuntimeError("V17 system has no learned optimizer state")
    _validate_optimizer_state(system.optimizer_state, system.controller)
    optimizer = _context_optimizer(system.controller)
    named = dict(system.controller.named_parameters())
    slots = system.optimizer_state["state"]
    assert isinstance(slots, Mapping)
    for name in MUTABLE_PARAMETER_NAMES:
        parameter = named[name]
        slot = slots[name]
        assert isinstance(slot, Mapping)
        optimizer.state[parameter] = {
            slot_name: value.detach().to(parameter.device).clone()
            for slot_name, value in slot.items()
            if isinstance(value, torch.Tensor)
        }
    return optimizer


def _fit_context_residual_batches(
    system: V12ChampionContextResidualSystem,
    stream_batches: Sequence[Sequence[v12.SoftwarePipelineStream]],
) -> dict[str, object]:
    if not stream_batches or any(not batch for batch in stream_batches):
        raise ValueError("V17 context fit requires nonempty public stream batches")
    if system.context_updates != 0 or system.optimizer_state is not None:
        raise RuntimeError("V17 context batches require a fresh migrated lineage")
    if system.controller._context_diagnostic is not None:
        raise RuntimeError("V17 cannot train under a diagnostic lesion")
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
    optimizer = _context_optimizer(controller)
    parameters = tuple(named[name] for name in MUTABLE_PARAMETER_NAMES)
    losses: list[float] = []
    gradient_norms: list[float] = []
    nonzero_gradient_names: list[tuple[str, ...]] = []
    supported_rows: list[int] = []
    valid_masses: list[float] = []
    valid_top_one: list[float] = []
    first_up_nonzero: dict[str, bool] = {}
    composer_gradient_reached = False
    first_post_divergence_composer_credit = False
    was_training = controller.training
    controller.train()
    try:
        for update_index, batch in enumerate(stream_batches):
            if len(batch) != _STREAMS_PER_UPDATE:
                raise RuntimeError("V17 context update lost a public stream")
            row_groups = tuple(
                v12.public_relation_credit_rows(controller, stream) for stream in batch
            )
            per_stream_terms = []
            counts = []
            diagnostics: list[dict[str, object]] = []
            for group in row_groups:
                terms = []
                for row in group:
                    term, diagnostic = v12._context_valid_set_training_term(row)
                    if term is not None:
                        terms.append(term)
                        diagnostics.append(diagnostic)
                counts.append(len(terms))
                per_stream_terms.append(
                    torch.stack(tuple(terms)).mean()
                    if terms
                    else group[0].context_weights.sum() * 0.0
                )
            if not diagnostics:
                raise RuntimeError("V17 context update has no supported public row")
            per_stream_losses = torch.stack(tuple(per_stream_terms))
            row_counts = per_stream_losses.new_tensor(counts)
            objective, _, _, _, _, _ = v12._relation_credit_stream_objective(
                per_stream_losses,
                stage="context",
                stream_row_counts=row_counts,
            )
            if not bool(torch.isfinite(objective).item()):
                raise RuntimeError("V17 context objective is non-finite")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for name in MUTABLE_PARAMETER_NAMES:
                gradient = named[name].grad
                if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(f"V17 context gradient is non-finite: {name}")
            nonzero_gradient_names.append(
                tuple(
                    name
                    for name in MUTABLE_PARAMETER_NAMES
                    if named[name].grad is not None
                    and bool(torch.count_nonzero(named[name].grad).item())
                )
            )
            if update_index == 0:
                for name in _EXPERT_PARAMETER_NAMES:
                    if name.endswith(".up.weight"):
                        gradient = named[name].grad
                        first_up_nonzero[name] = gradient is not None and bool(
                            torch.count_nonzero(gradient).item()
                        )
                if not all(first_up_nonzero.values()):
                    raise RuntimeError("V17 first credit did not reach every zero Up")
            composer_gradient_reached = composer_gradient_reached or any(
                named[name].grad is not None
                and bool(torch.count_nonzero(named[name].grad).item())
                for name in _COMPOSER_PARAMETER_NAMES
            )
            if update_index == 1:
                first_post_divergence_composer_credit = any(
                    named[name].grad is not None
                    and bool(torch.isfinite(named[name].grad).all().item())
                    and bool(torch.count_nonzero(named[name].grad).item())
                    for name in _COMPOSER_PARAMETER_NAMES
                )
                if not first_post_divergence_composer_credit:
                    raise RuntimeError(
                        "V17 first post-divergence batch did not reach the composer"
                    )
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, _GRADIENT_CLIP)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("V17 clipped gradient norm is non-finite")
            optimizer.step()
            # Mark the invocation consumed immediately after every mutation so
            # a later integrity failure cannot make the same object rerunnable.
            system.context_updates += 1
            system.optimizer_state = _canonical_optimizer_state(
                optimizer,
                controller,
            )
            if any(not bool(torch.isfinite(named[name]).all().item()) for name in MUTABLE_PARAMETER_NAMES):
                raise RuntimeError("V17 context update produced non-finite parameters")
            if update_index == 0:
                first_up_states = tuple(
                    named[f"context_residual_experts.{index}.up.weight"].detach()
                    for index in range(_EXPERT_COUNT)
                )
                if not any(
                    not torch.equal(first_up_states[left], first_up_states[right])
                    for left in range(_EXPERT_COUNT)
                    for right in range(left + 1, _EXPERT_COUNT)
                ):
                    raise RuntimeError("V17 experts did not diverge after update zero")
            current_state = controller.state_dict()
            for name, before in inherited_before.items():
                if not torch.equal(before, current_state[name].detach()):
                    raise RuntimeError(f"V17 changed inherited parameter or buffer: {name}")
            losses.append(float(objective.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
            supported_rows.append(len(diagnostics))
            valid_masses.append(
                sum(float(value["context_valid_set_mass"]) for value in diagnostics)
                / len(diagnostics)
            )
            valid_top_one.append(
                sum(value["context_valid_set_top_one"] is True for value in diagnostics)
                / len(diagnostics)
            )
    finally:
        controller.train(was_training)
    if any(
        not torch.equal(before, system.mixer.state_dict()[name].detach())
        for name, before in mixer_before.items()
    ):
        raise RuntimeError("V17 context fit changed the frozen V12 mixer")
    current_competence = v12.snapshot_software_reconstruction_state(system.competence_state)
    if any(
        not torch.equal(before, current_competence[name].detach())
        for name, before in competence_before.items()
    ):
        raise RuntimeError("V17 context fit changed the frozen competence state")
    _assert_source_lineage(system)
    changed_names = tuple(
        name
        for name in MUTABLE_PARAMETER_NAMES
        if not torch.equal(mutable_before[name], named[name].detach())
    )
    expert_up_states = tuple(
        named[f"context_residual_experts.{index}.up.weight"].detach()
        for index in range(_EXPERT_COUNT)
    )
    expert_diverged = any(
        not torch.equal(expert_up_states[left], expert_up_states[right])
        for left in range(_EXPERT_COUNT)
        for right in range(left + 1, _EXPERT_COUNT)
    )
    gradient_reached_names = tuple(
        name
        for name in MUTABLE_PARAMETER_NAMES
        if any(name in update_names for update_names in nonzero_gradient_names)
    )
    return {
        "stage": "context_residual",
        "optimizer_steps": len(stream_batches),
        "streams": sum(len(batch) for batch in stream_batches),
        "rows": sum(len(batch) for batch in stream_batches) * _ROWS_PER_STREAM,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": tuple(losses),
        "gradient_norms": tuple(gradient_norms),
        "nonzero_gradient_parameter_names": tuple(nonzero_gradient_names),
        "gradient_reached_parameter_names": gradient_reached_names,
        "supported_rows": tuple(supported_rows),
        "context_valid_set_mass": tuple(valid_masses),
        "context_valid_set_top_one_fraction": tuple(valid_top_one),
        "first_zero_up_gradients_nonzero": first_up_nonzero,
        "composer_gradient_reached": composer_gradient_reached,
        "first_post_divergence_composer_credit": (
            first_post_divergence_composer_credit
        ),
        "experts_diverged": expert_diverged,
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


def _panel_context_metrics(panel: Mapping[str, object]) -> dict[str, float | int]:
    reports = panel.get("row_reports")
    if not isinstance(reports, (tuple, list)):
        raise RuntimeError("V17 panel lost its row reports")
    supported = tuple(
        row for row in reports if isinstance(row, Mapping) and row.get("relation_supported") is True
    )
    if len(supported) != panel.get("relation_supported_rows"):
        raise RuntimeError("V17 panel supported-row accounting changed")
    return {
        "supported_rows": len(supported),
        "top_one_successes": sum(
            row.get("context_valid_set_top_one") is True for row in supported
        ),
        "valid_set_mass": float(panel["context_valid_set_mass_mean_supported"]),
    }


def _relation_boundary_signature(panel: Mapping[str, object]) -> tuple[object, ...]:
    reports = panel.get("row_reports")
    if not isinstance(reports, (tuple, list)):
        raise RuntimeError("V17 relation-boundary panel lost row reports")
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
        tuple(
            tuple(row.get(field) for field in row_fields)
            for row in reports
            if isinstance(row, Mapping)
        ),
    )


def _evaluate_mode(
    controller: V12ChampionContextResidualController,
    streams: Sequence[v12.SoftwarePipelineStream],
    kind: str,
    index: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    with controller.context_diagnostic(kind, index), controller.capture_context_reads() as records:
        panel = v12.evaluate_public_relation_credit_panel(controller, streams)
    if any(record["final_evidence_scores"] is None for record in records):
        raise RuntimeError("V17 evidence capture ended before final scoring")
    weights = tuple(
        record["mean_weights"]
        for record in records
        if len(record["mean_weights"]) == _EXPERT_COUNT
    )
    pairwise = tuple(
        float(value)
        for record in records
        for value in record["mean_pairwise_logit_differences"]
    )
    diagnostics = {
        "context_reads": len(records),
        "mean_weights": (
            tuple(
                sum(float(row[cell]) for row in weights) / len(weights)
                for cell in range(_EXPERT_COUNT)
            )
            if weights
            else ()
        ),
        "mean_pairwise_logit_difference": (
            sum(pairwise) / len(pairwise) if pairwise else 0.0
        ),
        "evidence_reads": tuple(
            {
                "context_weights": tuple(
                    tuple(float(value) for value in row)
                    for row in record["context_weights"].detach().cpu().tolist()
                ),
                "context_null_weights": tuple(
                    float(value)
                    for value in record["context_null_weights"].detach().cpu().tolist()
                ),
                "final_evidence_scores": tuple(
                    float(value)
                    for value in record["final_evidence_scores"].detach().cpu().tolist()
                ),
            }
            for record in records
        ),
    }
    return panel, diagnostics


def _classify_context_result(
    learned: Mapping[str, float | int],
    residual_off: Mapping[str, float | int],
    uniform: Mapping[str, float | int],
    lesions: Sequence[Mapping[str, float | int]],
    *,
    context_gate_passed: bool,
    integrity_passed: bool,
) -> dict[str, object]:
    if len(lesions) != _EXPERT_COUNT:
        raise ValueError("V17 classification requires four drop-one lesions")
    residual_top_one_gain = int(learned["top_one_successes"]) - int(
        residual_off["top_one_successes"]
    )
    residual_mass_gain = float(learned["valid_set_mass"]) - float(
        residual_off["valid_set_mass"]
    )
    uniform_top_one_gain = int(learned["top_one_successes"]) - int(
        uniform["top_one_successes"]
    )
    uniform_mass_gain = float(learned["valid_set_mass"]) - float(
        uniform["valid_set_mass"]
    )
    lesion_effects = tuple(
        {
            "expert": index,
            "top_one_loss": int(learned["top_one_successes"])
            - int(metrics["top_one_successes"]),
            "mass_loss": float(learned["valid_set_mass"])
            - float(metrics["valid_set_mass"]),
        }
        for index, metrics in enumerate(lesions)
    )
    causal_lesions = sum(
        effect["top_one_loss"] >= _LESION_TOP_ONE_LOSS
        or effect["mass_loss"] >= _LESION_MASS_LOSS
        for effect in lesion_effects
    )
    support = {
        "context_gate": context_gate_passed,
        "residual_top_one_gain": residual_top_one_gain >= _RESIDUAL_TOP_ONE_GAIN,
        "residual_mass_gain": residual_mass_gain >= _RESIDUAL_MASS_GAIN,
        "composer_nonregress_top_one": uniform_top_one_gain >= 0,
        "composer_nonregress_mass": uniform_mass_gain >= 0.0,
        "composer_material_gain": uniform_top_one_gain >= _COMPOSER_TOP_ONE_GAIN
        or uniform_mass_gain >= _COMPOSER_MASS_GAIN,
        "two_causal_drop_one_lesions": causal_lesions >= _REQUIRED_CAUSAL_LESIONS,
    }
    if not integrity_passed:
        classification = "INVALID_NO_CLAIM"
    elif all(support.values()):
        classification = "CONTEXT_RESIDUAL_SUPPORTED"
    else:
        classification = "CONTEXT_RESIDUAL_NOT_SUPPORTED"
    return {
        "classification": classification,
        "passed": classification == "CONTEXT_RESIDUAL_SUPPORTED",
        "support_checks": support,
        "residual_top_one_gain": residual_top_one_gain,
        "residual_mass_gain": residual_mass_gain,
        "uniform_top_one_gain": uniform_top_one_gain,
        "uniform_mass_gain": uniform_mass_gain,
        "lesion_effects": lesion_effects,
        "causal_lesion_count": causal_lesions,
    }


def _evaluate_surface_suite(
    controller: V12ChampionContextResidualController,
    streams: Sequence[v12.SoftwarePipelineStream],
) -> dict[str, object]:
    learned, learned_read = _evaluate_mode(controller, streams, "learned")
    residual_off, _ = _evaluate_mode(controller, streams, "residual_off")
    uniform, uniform_read = _evaluate_mode(controller, streams, "uniform")
    lesions = tuple(
        _evaluate_mode(controller, streams, "drop", index)[0]
        for index in range(_EXPERT_COUNT)
    )
    invariants = v12._evaluate_public_relation_credit_invariants(controller, streams)
    learned_metrics = _panel_context_metrics(learned)
    off_metrics = _panel_context_metrics(residual_off)
    uniform_metrics = _panel_context_metrics(uniform)
    lesion_metrics = tuple(_panel_context_metrics(panel) for panel in lesions)
    learned_relation_signature = _relation_boundary_signature(learned)
    relation_identity = all(
        _relation_boundary_signature(panel) == learned_relation_signature
        for panel in (residual_off, uniform, *lesions)
    )
    context_gate = v12._relation_credit_context_gate(learned, invariants)
    integrity = {
        "relation_boundary_exact_across_modes": relation_identity,
        "permutation_covariant": invariants.get("permutation_covariant") is True,
        "empty_memory_zero_exact": invariants.get("empty_memory_zero_exact") is True,
    }
    classification = _classify_context_result(
        learned_metrics,
        off_metrics,
        uniform_metrics,
        lesion_metrics,
        context_gate_passed=context_gate.get("passed") is True,
        integrity_passed=all(integrity.values()),
    )
    return {
        "learned": learned,
        "residual_off": residual_off,
        "forced_uniform": uniform,
        "drop_one_lesions": lesions,
        "learned_context_metrics": learned_metrics,
        "residual_off_context_metrics": off_metrics,
        "uniform_context_metrics": uniform_metrics,
        "lesion_context_metrics": lesion_metrics,
        "context_gate": context_gate,
        "integrity_checks": integrity,
        "classification": classification,
        "read_diagnostics": {"learned": learned_read, "uniform": uniform_read},
    }


def _cross_surface_checks(
    base_panel: Mapping[str, object],
    rerender_panel: Mapping[str, object],
    base_read: Mapping[str, object],
    rerender_read: Mapping[str, object],
) -> dict[str, object]:
    base_rows = base_panel.get("row_reports")
    rerender_rows = rerender_panel.get("row_reports")
    if (
        not isinstance(base_rows, (tuple, list))
        or not isinstance(rerender_rows, (tuple, list))
        or len(base_rows) != len(rerender_rows)
    ):
        raise RuntimeError("V17 cross-surface row alignment changed")
    relation_masks_exact = (
        base_panel.get("relation_supported_rows")
        == rerender_panel.get("relation_supported_rows")
        and base_panel.get("streams_with_three_supported_rows")
        == rerender_panel.get("streams_with_three_supported_rows")
        and base_panel.get("supported_rows_per_stream")
        == rerender_panel.get("supported_rows_per_stream")
        and all(
            (
                left.get("stream_index"),
                left.get("heldout_index"),
                left.get("transition_index"),
            )
            == (
                right.get("stream_index"),
                right.get("heldout_index"),
                right.get("transition_index"),
            )
            and left.get("valid_slots") == right.get("valid_slots")
            and left.get("relation_supported") == right.get("relation_supported")
            for left, right in zip(base_rows, rerender_rows, strict=True)
        )
    )
    top_one_success_exact = all(
        left.get("context_valid_set_top_one")
        == right.get("context_valid_set_top_one")
        for left, right in zip(base_rows, rerender_rows, strict=True)
    )
    top_one_choice_exact = all(
        max(
            range(len(left["context_weights"]) + 1),
            key=lambda index: (
                left["context_null_mass"]
                if index == len(left["context_weights"])
                else left["context_weights"][index]
            ),
        )
        == max(
            range(len(right["context_weights"]) + 1),
            key=lambda index: (
                right["context_null_mass"]
                if index == len(right["context_weights"])
                else right["context_weights"][index]
            ),
        )
        for left, right in zip(base_rows, rerender_rows, strict=True)
    )
    weight_delta = 0.0
    null_delta = 0.0
    valid_mass_delta = 0.0
    final_score_delta = 0.0
    for left, right in zip(base_rows, rerender_rows, strict=True):
        left_weights = left.get("context_weights")
        right_weights = right.get("context_weights")
        if not isinstance(left_weights, (tuple, list)) or not isinstance(
            right_weights, (tuple, list)
        ) or len(left_weights) != len(right_weights):
            raise RuntimeError("V17 cross-surface context weights lost alignment")
        weight_delta = max(
            weight_delta,
            *(abs(float(a) - float(b)) for a, b in zip(left_weights, right_weights, strict=True)),
        )
        null_delta = max(
            null_delta,
            abs(float(left["context_null_mass"]) - float(right["context_null_mass"])),
        )
        valid_mass_delta = max(
            valid_mass_delta,
            abs(
                float(left["context_valid_set_mass"])
                - float(right["context_valid_set_mass"])
            ),
        )
    base_reads = base_read.get("evidence_reads")
    rerender_reads = rerender_read.get("evidence_reads")
    if (
        not isinstance(base_reads, (tuple, list))
        or not isinstance(rerender_reads, (tuple, list))
        or len(base_reads) != len(rerender_reads)
    ):
        raise RuntimeError("V17 cross-surface evidence-read alignment changed")
    full_weight_delta = 0.0
    full_null_delta = 0.0
    for left, right in zip(base_reads, rerender_reads, strict=True):
        for left_row, right_row in zip(
            left["context_weights"],
            right["context_weights"],
            strict=True,
        ):
            full_weight_delta = max(
                full_weight_delta,
                *(abs(a - b) for a, b in zip(left_row, right_row, strict=True)),
            )
        full_null_delta = max(
            full_null_delta,
            *(abs(a - b) for a, b in zip(
                left["context_null_weights"],
                right["context_null_weights"],
                strict=True,
            )),
        )
        final_score_delta = max(
            final_score_delta,
            *(abs(a - b) for a, b in zip(
                left["final_evidence_scores"],
                right["final_evidence_scores"],
                strict=True,
            )),
        )
    weight_delta = max(weight_delta, full_weight_delta)
    null_delta = max(null_delta, full_null_delta)
    return {
        "relation_masks_and_counts_exact": relation_masks_exact,
        "context_top_one_success_exact": top_one_success_exact,
        "context_top_one_choice_exact": top_one_choice_exact,
        "context_weight_max_delta": weight_delta,
        "context_null_weight_max_delta": null_delta,
        "valid_set_mass_max_delta": valid_mass_delta,
        "final_evidence_score_max_delta": final_score_delta,
        "continuous_values_within_1e-6": max(
            weight_delta,
            null_delta,
            valid_mass_delta,
            final_score_delta,
        )
        <= 1.0e-6,
    }


def evaluate_v12_champion_context_residual(
    system: V12ChampionContextResidualSystem,
) -> dict[str, object]:
    """Evaluate both frozen surfaces and every causal read diagnostic."""

    _assert_source_lineage(system)
    plan = v12_champion_context_residual_plan()
    commitments = plan["commitments"]
    panel_pairs = plan["panel_seed_pairs"]
    rerender_pairs = plan["rerender_seed_pairs"]
    assert isinstance(commitments, tuple)
    assert isinstance(panel_pairs, tuple)
    assert isinstance(rerender_pairs, tuple)
    base_streams = v12._relation_credit_panel_streams(commitments, panel_pairs)
    rerender_streams = v12._relation_credit_panel_streams(commitments, rerender_pairs)
    base = _evaluate_surface_suite(system.controller, base_streams)
    rerender = _evaluate_surface_suite(system.controller, rerender_streams)
    cross_surface = _cross_surface_checks(
        base["learned"],
        rerender["learned"],
        base["read_diagnostics"]["learned"],
        rerender["read_diagnostics"]["learned"],
    )
    source_exact = (
        inherited_v12_controller_digest(system.controller)
        == system.source.controller_digest
        and v12.anonymous_conflict_mixer_digest(system.mixer)
        == system.source.mixer_digest
        and v12.software_reconstruction_state_digest(system.competence_state)
        == system.source.competence_digest
    )
    integrity = {
        "source_lineage_exact": source_exact,
        "context_updates_exact": system.context_updates == _CONTEXT_UPDATES,
        "base_surface_integrity": all(base["integrity_checks"].values()),
        "rerender_surface_integrity": all(rerender["integrity_checks"].values()),
        "cross_surface_relation_exact": cross_surface[
            "relation_masks_and_counts_exact"
        ]
        is True,
        "cross_surface_top_one_exact": (
            cross_surface["context_top_one_success_exact"] is True
            and cross_surface["context_top_one_choice_exact"] is True
        ),
        "cross_surface_continuous_within_1e-6": cross_surface[
            "continuous_values_within_1e-6"
        ]
        is True,
    }
    base_causal_indices = {
        int(effect["expert"])
        for effect in base["classification"]["lesion_effects"]
        if effect["top_one_loss"] >= _LESION_TOP_ONE_LOSS
        or effect["mass_loss"] >= _LESION_MASS_LOSS
    }
    rerender_causal_indices = {
        int(effect["expert"])
        for effect in rerender["classification"]["lesion_effects"]
        if effect["top_one_loss"] >= _LESION_TOP_ONE_LOSS
        or effect["mass_loss"] >= _LESION_MASS_LOSS
    }
    shared_causal_indices = tuple(sorted(base_causal_indices & rerender_causal_indices))
    shared_causal_lesion_index_support = (
        len(shared_causal_indices) >= _REQUIRED_CAUSAL_LESIONS
    )
    if not all(integrity.values()):
        classification = "INVALID_NO_CLAIM"
    elif (
        base["classification"]["classification"] == "CONTEXT_RESIDUAL_SUPPORTED"
        and rerender["classification"]["classification"]
        == "CONTEXT_RESIDUAL_SUPPORTED"
    ):
        classification = "CONTEXT_RESIDUAL_SUPPORTED"
    else:
        classification = "CONTEXT_RESIDUAL_NOT_SUPPORTED"
    return {
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "passed": classification == "CONTEXT_RESIDUAL_SUPPORTED",
        "plan_digest": plan["plan_digest"],
        "base_surface": base,
        "surface_rerender": rerender,
        "cross_surface_checks": cross_surface,
        "cross_surface_causal_lesion_indices": shared_causal_indices,
        "shared_causal_lesion_index_support": shared_causal_lesion_index_support,
        "integrity_checks": integrity,
        "joint_training_performed": False,
        "write_router_used": False,
        "replay_used": False,
        "stored_examples_used": False,
        "deterministic_solver_used": False,
        "identity_inputs_used": False,
        "development_or_final_access": False,
        "scalar_judge_calls": 0,
        "control_streams_used": 0,
        "wrong_evidence_training_streams": 0,
    }


def fit_v12_champion_context_residual(
    system: V12ChampionContextResidualSystem,
) -> dict[str, object]:
    """Execute the one frozen C25 successor fit without writing artifacts."""

    if system.context_updates != 0:
        raise RuntimeError("V17 is a one-shot context fit")
    plan = v12_champion_context_residual_plan()
    commitments = plan["commitments"]
    seed_batches = plan["training_seed_batches"]
    assert isinstance(commitments, tuple)
    assert isinstance(seed_batches, tuple)
    batches = v12._relation_credit_stream_batches(commitments, seed_batches)
    fit = _fit_context_residual_batches(system, batches)
    changed = set(fit["changed_parameter_names"])
    allowed = set(MUTABLE_PARAMETER_NAMES)
    if (
        fit["optimizer_steps"] != _CONTEXT_UPDATES
        or fit["streams"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE
        or fit["rows"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM
        or fit["inherited_controller_exact"] is not True
        or fit["mixer_exact"] is not True
        or fit["competence_state_exact"] is not True
        or fit["experts_diverged"] is not True
        or fit["composer_gradient_reached"] is not True
        or fit["first_post_divergence_composer_credit"] is not True
        or not changed
        or not changed <= allowed
        or not set(_EXPERT_PARAMETER_NAMES) <= changed
        or "context_composer.residual_scorer.2.weight" not in changed
        or set(fit["gradient_reached_parameter_names"]) != allowed
    ):
        raise RuntimeError("V17 semantic fit lost its frozen accounting")
    evaluation = evaluate_v12_champion_context_residual(system)
    return {
        "protocol_id": PROTOCOL_ID,
        "classification": evaluation["classification"],
        "passed": evaluation["passed"],
        "plan": plan,
        "fit": fit,
        "evaluation": evaluation,
        "parameter_report": context_residual_parameter_report(
            system.controller,
            system.mixer,
        ),
        "source": asdict(system.source),
        "terminal_controller_digest": v12.software_pipeline_model_digest(
            system.controller
        ),
        "terminal_mutable_digest": context_residual_mutable_digest(system.controller),
        "terminal_system_digest": context_residual_system_digest(system),
        "context_updates": system.context_updates,
        "joint_training_performed": False,
    }


def _checkpoint_payload(system: V12ChampionContextResidualSystem) -> dict[str, object]:
    _assert_source_lineage(system)
    payload = {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_context_residual_plan()["plan_digest"],
        "source": asdict(system.source),
        "profile": asdict(system.controller.profile),
        "config": {
            "expert_count": _EXPERT_COUNT,
            "expert_rank": _EXPERT_RANK,
            "context_width": _CONTEXT_WIDTH,
            "expert_seeds": _EXPERT_SEEDS,
            "composer_seed": _COMPOSER_SEED,
            "composer_hidden_width": _COMPOSER_HIDDEN_WIDTH,
            "composer_anchor_weight": _COMPOSER_ANCHOR_WEIGHT,
        },
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in system.controller.state_dict().items()
        },
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": context_residual_mutable_digest(system.controller),
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
        "optimizer_digest": context_residual_optimizer_digest(
            system.optimizer_state
        ),
        "parameter_report": context_residual_parameter_report(
            system.controller,
            system.mixer,
        ),
        "system_digest": context_residual_system_digest(system),
    }
    return payload


def save_v12_champion_context_residual_checkpoint(
    path: str | Path,
    system: V12ChampionContextResidualSystem,
) -> None:
    """Persist one complete V17 lineage without mutating its source checkpoint."""

    torch.save(_checkpoint_payload(system), Path(path))


def load_v12_champion_context_residual_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionContextResidualSystem:
    """Strictly restore and validate every V17 lineage component."""

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
        raise RuntimeError("V17 checkpoint fields are invalid")
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"]
        != v12_champion_context_residual_plan()["plan_digest"]
    ):
        raise RuntimeError("V17 checkpoint identity is invalid")
    expected_config = _checkpoint_payload_config()
    if payload["config"] != expected_config:
        raise RuntimeError("V17 checkpoint architecture changed")
    source_value = payload["source"]
    if not isinstance(source_value, dict) or set(source_value) != {
        "checkpoint_sha256",
        "controller_digest",
        "mixer_digest",
        "competence_digest",
        "system_digest",
    }:
        raise RuntimeError("V17 source binding is invalid")
    source = V12ChampionSourceBinding(**source_value)
    if source != _expected_source_binding():
        raise RuntimeError("V17 checkpoint changed its terminal V12 source")
    profile = v12.SoftwarePipelineRunProfile(**payload["profile"])
    if v12.SOFTWARE_PIPELINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("V17 checkpoint profile is not registered")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = _cuda_rng_snapshot(device)
    try:
        controller = V12ChampionContextResidualController(profile).to(device)
        controller.load_state_dict(payload["model_state"], strict=True)
        mixer_config = payload["mixer_config"]
        if not isinstance(mixer_config, dict) or set(mixer_config) != {
            "feature_count",
            "hidden_width",
            "anchor_weight",
        }:
            raise RuntimeError("V17 mixer config is invalid")
        mixer = v12.AnonymousConflictMixer(**mixer_config).to(device)
        mixer.load_state_dict(payload["mixer_state"], strict=True)
        for parameter in mixer.parameters():
            parameter.requires_grad_(False)
    finally:
        torch.set_rng_state(cpu_rng_state)
        _restore_cuda_rng_snapshot(cuda_rng_states)
    try:
        state = v12.restore_software_reconstruction_state(payload["competence_state"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("V17 checkpoint competence state is invalid") from error
    updates = payload["context_updates"]
    if type(updates) is not int or not 0 <= updates <= _CONTEXT_UPDATES:
        raise RuntimeError("V17 checkpoint context update count is invalid")
    optimizer_state = payload["optimizer_state"]
    if updates == 0:
        if optimizer_state is not None:
            raise RuntimeError("V17 fresh checkpoint unexpectedly has optimizer state")
    else:
        if not isinstance(optimizer_state, Mapping):
            raise RuntimeError("V17 learned checkpoint lost optimizer state")
        _validate_optimizer_state(
            optimizer_state,
            controller,
            expected_steps=updates,
        )
    _enforce_mutable_scope(controller)
    system = V12ChampionContextResidualSystem(
        controller=controller,
        mixer=mixer,
        competence_state=state,
        source=source,
        context_updates=updates,
        optimizer_state=copy.deepcopy(optimizer_state),
    )
    if (
        v12.software_pipeline_model_digest(controller) != payload["controller_digest"]
        or context_residual_mutable_digest(controller) != payload["mutable_digest"]
        or inherited_v12_controller_digest(controller)
        != payload["inherited_controller_digest"]
        or v12.anonymous_conflict_mixer_digest(mixer) != payload["mixer_digest"]
        or v12.software_reconstruction_state_digest(state)
        != payload["competence_digest"]
        or context_residual_optimizer_digest(system.optimizer_state)
        != payload["optimizer_digest"]
        or context_residual_parameter_report(controller, mixer)
        != payload["parameter_report"]
        or context_residual_system_digest(system) != payload["system_digest"]
    ):
        raise RuntimeError("V17 checkpoint digest or report mismatch")
    _assert_source_lineage(system)
    controller.eval()
    mixer.eval()
    return system


def _checkpoint_payload_config() -> dict[str, object]:
    return {
        "expert_count": _EXPERT_COUNT,
        "expert_rank": _EXPERT_RANK,
        "context_width": _CONTEXT_WIDTH,
        "expert_seeds": _EXPERT_SEEDS,
        "composer_seed": _COMPOSER_SEED,
        "composer_hidden_width": _COMPOSER_HIDDEN_WIDTH,
        "composer_anchor_weight": _COMPOSER_ANCHOR_WEIGHT,
    }
