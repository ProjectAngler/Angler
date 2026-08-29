"""V12-champion paired graph-context successor.

V19 preserves the terminal V12 reasoner and replaces no learned path.  It adds
one function-preserving residual that compares a query predecessor graph with
each stored predecessor graph *before* either graph is pooled.  The mechanism
is used by both public credit construction and ordinary production scoring.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runners import phase6_software_pipeline_reconstruction as v12


PROTOCOL_ID = "phase6.public-v12-champion-paired-graph-context.v19"
CHECKPOINT_VERSION = "angler.phase6-v12-champion-paired-graph-context.v1"
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

ACTIVE_LEAF = (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md"
)
FROZEN_DEPENDENCY_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    ACTIVE_LEAF: (
        "B819DA5F6D10151E7613ADECBBA076DF7642559D35BEA2EA74551FD791C6668D"
    ),
}

_PLAN_DIGEST_DOMAIN = b"project-angler.v12-champion-paired-graph-context.plan.v1\x00"
_MUTABLE_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-paired-graph-context.mutable.v1\x00"
)
_OPTIMIZER_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-paired-graph-context.optimizer.v1\x00"
)
_SYSTEM_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-paired-graph-context.system.v1\x00"
)
_STATE_DIGEST_DOMAIN = (
    b"project-angler.v12-champion-paired-graph-context.state.v1\x00"
)

_WIDTH = 32
_HIDDEN_WIDTH = 64
_MAX_GRAPH_NODES = 32
_NODE_ENCODER_SEED = 2_026_083_901
_GRAPH_UPDATE_SEED = 2_026_083_902
_PAIR_SCORER_SEED = 2_026_083_903
_CONTEXT_UPDATES = 512
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_PANEL_COUNT = 4
_TRUNK_LEARNING_RATE = 3.0e-4
_SCORER_LEARNING_RATE = 1.0e-3
_GRADIENT_CLIP = 5.0
_CONTEXT_TEMPERATURE = 0.25
_RESIDUAL_BOUND = 0.5
_PAIR_MARGIN = 0.10
_PAIR_TEMPERATURE = 0.05

_TRAIN_TOPOLOGY_BASE = 9_401_000_001
_TRAIN_SURFACE_BASE = 9_501_000_001
_PANEL_TOPOLOGY_BASE = 9_601_000_001
_PANEL_SURFACE_BASE = 9_701_000_001

_CONTEXT_TOP_ONE_THRESHOLD = 0.80
_CONTEXT_MASS_THRESHOLD = 0.60
_RELATION_SUPPORTED_ROWS_THRESHOLD = 96
_RELATION_QUALIFYING_STREAMS_THRESHOLD = 24
_CAUSAL_TOP_ONE_GAIN = 12
_CAUSAL_REAL_NORMALIZED_MASS_GAIN = 0.05
_CAUSAL_MARGIN_GAIN = 0.05
_REQUIRED_IMPROVED_PANELS = 3
_MAX_PANEL_TOP_ONE_REGRESSION = 1
_MAX_PANEL_MASS_REGRESSION = 0.01
_ATTRIBUTION_TOP_ONE_REMOVAL = 4
_ATTRIBUTION_MASS_REMOVAL = 0.02
_ATTRIBUTION_MARGIN_FRACTION = 0.5

_NODE_ENCODER_PARAMETER_NAMES = (
    "paired_graph_node_encoder.row_attention.weight",
    "paired_graph_node_encoder.column_attention.weight",
    "paired_graph_node_encoder.node_projection.0.weight",
    "paired_graph_node_encoder.node_projection.0.bias",
    "paired_graph_node_encoder.node_projection.1.weight",
    "paired_graph_node_encoder.node_projection.1.bias",
    "paired_graph_node_encoder.node_projection.3.weight",
    "paired_graph_node_encoder.node_projection.3.bias",
)
_GRAPH_UPDATE_PARAMETER_NAMES = (
    "paired_graph_update.0.weight",
    "paired_graph_update.0.bias",
    "paired_graph_update.1.weight",
    "paired_graph_update.1.bias",
    "paired_graph_update.3.weight",
    "paired_graph_update.3.bias",
)
_PAIR_SCORER_PARAMETER_NAMES = (
    "paired_graph_scorer.0.weight",
    "paired_graph_scorer.0.bias",
    "paired_graph_scorer.1.weight",
    "paired_graph_scorer.1.bias",
    "paired_graph_scorer.3.weight",
    "paired_graph_scorer.3.bias",
    "paired_graph_scorer.5.weight",
)
MUTABLE_PARAMETER_NAMES = (
    _NODE_ENCODER_PARAMETER_NAMES
    + _GRAPH_UPDATE_PARAMETER_NAMES
    + _PAIR_SCORER_PARAMETER_NAMES
)
_MUTABLE_PREFIXES = (
    "paired_graph_node_encoder.",
    "paired_graph_update.",
    "paired_graph_scorer.",
)


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
    return tuple(
        torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())
    )


def _restore_cuda_rng_snapshot(states: tuple[torch.Tensor, ...] | None) -> None:
    if states is not None:
        for index, state in enumerate(states):
            torch.cuda.set_rng_state(state, index)


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


def _json_digest(domain: bytes, payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(domain)
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256_file(root / name) for name in FROZEN_DEPENDENCY_HASHES}


def _verify_frozen_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = (
        root / "experiments/runners/phase6_software_pipeline_reconstruction.py"
    ).resolve()
    if Path(v12.__file__).resolve() != expected:
        raise RuntimeError("V19 imported a shadowed V12 dependency")
    if frozen_dependency_hashes() != FROZEN_DEPENDENCY_HASHES:
        raise RuntimeError("V19 frozen V12 source or active leaf changed")


def _plan_payload() -> dict[str, object]:
    commitments = tuple(v12.software_pipeline_mechanism_partition("train")[:8])
    if len(commitments) != _STREAMS_PER_UPDATE or len(set(commitments)) != 8:
        raise RuntimeError("V19 requires eight distinct public commitments")
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
    ):
        raise RuntimeError("V19 stream identities are not fresh and disjoint")
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
        "donor": {
            "repository": "google-deepmind/deepmind-research",
            "repository_commit": "f5de0ede8430809180254ee957abf36ed62579ef",
            "gmn_subtree_commit": "451d2964904a4e71d8d28ac45cdc5f33c1db1b19",
            "license": "Apache-2.0",
            "adaptation": "cross_graph_attention_and_mismatch_messages",
            "runtime_imported": False,
        },
        "commitments": commitments,
        "module_seeds": {
            "node_encoder": _NODE_ENCODER_SEED,
            "graph_update": _GRAPH_UPDATE_SEED,
            "pair_scorer": _PAIR_SCORER_SEED,
        },
        "architecture": {
            "width": _WIDTH,
            "hidden_width": _HIDDEN_WIDTH,
            "maximum_graph_nodes": _MAX_GRAPH_NODES,
            "trainable_tensors": 21,
            "trainable_parameters": 34_048,
            "cross_attention_rounds": 1,
            "residual_bound": _RESIDUAL_BOUND,
            "context_temperature": _CONTEXT_TEMPERATURE,
            "raw_graph_state": True,
            "frozen_pair_encoder_before_padding": True,
        },
        "training_seed_batches": training,
        "panel_seed_pairs": panels,
        "context_updates": _CONTEXT_UPDATES,
        "streams_per_update": _STREAMS_PER_UPDATE,
        "rows_per_stream": _ROWS_PER_STREAM,
        "trunk_learning_rate": _TRUNK_LEARNING_RATE,
        "scorer_learning_rate": _SCORER_LEARNING_RATE,
        "weight_decay": 0.0,
        "gradient_clip": _GRADIENT_CLIP,
        "objective": {
            "informative_rows_only": True,
            "list_loss_weight": 0.5,
            "pair_loss_weight": 0.5,
            "pair_margin": _PAIR_MARGIN,
            "pair_temperature": _PAIR_TEMPERATURE,
            "relation_support_detached": True,
            "presence_or_abstention_loss": False,
        },
        "component_support": {
            "aggregate_unique_valid_real_top_one_gain": _CAUSAL_TOP_ONE_GAIN,
            "aggregate_real_normalized_mass_gain": (
                _CAUSAL_REAL_NORMALIZED_MASS_GAIN
            ),
            "learned_margin_positive": True,
            "causal_margin_gain": _CAUSAL_MARGIN_GAIN,
            "improved_panels": _REQUIRED_IMPROVED_PANELS,
            "maximum_panel_top_one_regression": _MAX_PANEL_TOP_ONE_REGRESSION,
            "maximum_panel_mass_regression": _MAX_PANEL_MASS_REGRESSION,
        },
        "full_advancement": {
            "relation_supported_rows": _RELATION_SUPPORTED_ROWS_THRESHOLD,
            "relation_qualifying_streams": _RELATION_QUALIFYING_STREAMS_THRESHOLD,
            "supported_context_top_one_fraction": _CONTEXT_TOP_ONE_THRESHOLD,
            "supported_valid_set_mass": _CONTEXT_MASS_THRESHOLD,
        },
        "attribution": {
            "lesions": ("uniform_cross_graph_attention", "mismatch_zero"),
            "each_margin_gain_removal_fraction": _ATTRIBUTION_MARGIN_FRACTION,
            "each_top_one_removal": _ATTRIBUTION_TOP_ONE_REMOVAL,
            "each_mass_removal": _ATTRIBUTION_MASS_REMOVAL,
        },
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
        "replay": False,
        "early_stop": False,
        "seed_retry": False,
        "rescue_identity": False,
        "adaptive_threshold": False,
        "srwm": False,
        "router": False,
    }


def v12_champion_paired_graph_context_plan() -> dict[str, object]:
    """Return the complete frozen V19 plan and canonical digest."""

    payload = _plan_payload()
    return {**payload, "plan_digest": _json_digest(_PLAN_DIGEST_DOMAIN, payload)}


@dataclass(frozen=True, slots=True)
class V19SoftwareTaskEncoding(v12.SoftwareTaskEncoding):
    context_graph_adjacencies: torch.Tensor
    context_graph_masks: torch.Tensor

    def __post_init__(self) -> None:
        reference = self.relation_context_embeddings
        action_count = reference.shape[0]
        if (
            self.context_graph_adjacencies.shape
            != (action_count, _MAX_GRAPH_NODES, _MAX_GRAPH_NODES)
            or self.context_graph_masks.shape != (action_count, _MAX_GRAPH_NODES)
            or self.context_graph_adjacencies.device != reference.device
            or self.context_graph_adjacencies.dtype != torch.bool
            or self.context_graph_masks.device != reference.device
            or self.context_graph_masks.dtype != torch.bool
        ):
            raise ValueError("V19 task graph sidecars are not aligned")
        support = self.context_graph_masks.unsqueeze(1) & self.context_graph_masks.unsqueeze(2)
        if bool(self.context_graph_adjacencies.masked_select(~support).ne(0.0).any().item()):
            raise ValueError("V19 task graph padding must be exact zero")


@dataclass(frozen=True, slots=True)
class V19SoftwareReconstructionState(v12.SoftwareReconstructionState):
    context_trace_graphs: torch.Tensor
    context_trace_graph_masks: torch.Tensor

    def __post_init__(self) -> None:
        v12.SoftwareReconstructionState.__post_init__(self)
        expected_graphs = (
            self.role.batch_size,
            self.role.slot_count,
            _MAX_GRAPH_NODES,
            _MAX_GRAPH_NODES,
        )
        expected_masks = (
            self.role.batch_size,
            self.role.slot_count,
            _MAX_GRAPH_NODES,
        )
        if (
            self.context_trace_graphs.shape != expected_graphs
            or self.context_trace_graph_masks.shape != expected_masks
            or self.context_trace_graphs.device != self.role.keys.device
            or self.context_trace_graphs.dtype != torch.bool
            or self.context_trace_graph_masks.device != self.role.keys.device
            or self.context_trace_graph_masks.dtype != torch.bool
        ):
            raise ValueError("V19 graph state must match the role lane")
        trace_slots = self.role.slot_count // 2
        allowed = self.role.occupied.clone()
        allowed[:, trace_slots:] = False
        present = self.context_trace_graph_masks.any(dim=-1)
        context_present = self.context_trace_keys.norm(dim=-1) > 1.0e-8
        relation_present = self.relation_trace_values.norm(dim=-1) > 1.0e-8
        expected_present = allowed & context_present & relation_present
        if not torch.equal(present, expected_present):
            raise ValueError("V19 raw graphs lost trace-slot alignment")
        support = self.context_trace_graph_masks.unsqueeze(-1) & self.context_trace_graph_masks.unsqueeze(-2)
        if bool(self.context_trace_graphs.masked_select(~support).ne(0.0).any().item()):
            raise ValueError("V19 graph-state padding must be exact zero")


def snapshot_v19_reconstruction_state(
    state: V19SoftwareReconstructionState,
) -> dict[str, torch.Tensor]:
    if type(state) is not V19SoftwareReconstructionState:
        raise TypeError("V19 snapshot requires the versioned graph state")
    result = v12.snapshot_software_reconstruction_state(state)
    result["context_trace_graphs"] = state.context_trace_graphs.detach().clone()
    result["context_trace_graph_masks"] = (
        state.context_trace_graph_masks.detach().clone()
    )
    return result


def restore_v19_reconstruction_state(
    snapshot: Mapping[str, torch.Tensor],
) -> V19SoftwareReconstructionState:
    extras = {"context_trace_graphs", "context_trace_graph_masks"}
    if not extras <= set(snapshot):
        raise ValueError("V19 reconstruction snapshot lacks graph sidecars")
    base = v12.restore_software_reconstruction_state(
        {name: value for name, value in snapshot.items() if name not in extras}
    )
    return V19SoftwareReconstructionState(
        pointer=base.pointer,
        role=base.role,
        context_trace_keys=base.context_trace_keys,
        relation_trace_values=base.relation_trace_values,
        context_trace_graphs=snapshot["context_trace_graphs"].detach().clone(),
        context_trace_graph_masks=(
            snapshot["context_trace_graph_masks"].detach().clone()
        ),
    )


def v19_reconstruction_state_digest(state: V19SoftwareReconstructionState) -> str:
    return _mapping_digest(_STATE_DIGEST_DOMAIN, snapshot_v19_reconstruction_state(state))


class ContextAxisNodeEncoder(nn.Module):
    """Permutation-equivariant endpoint-incidence encoder without pooling."""

    _AXIS_HEADS = 2

    def __init__(self, profile: v12.SoftwarePipelineRunProfile) -> None:
        super().__init__()
        width = profile.width
        self.width = width
        self.row_attention = nn.Linear(width, self._AXIS_HEADS, bias=False)
        self.column_attention = nn.Linear(width, self._AXIS_HEADS, bias=False)
        self.node_projection = nn.Sequential(
            nn.LayerNorm(5 * width),
            nn.Linear(5 * width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )

    def forward(self, pair_states: torch.Tensor) -> torch.Tensor:
        if (
            pair_states.ndim != 3
            or pair_states.shape[0] != pair_states.shape[1]
            or pair_states.shape[0] <= 0
            or pair_states.shape[-1] != self.width
            or not pair_states.is_floating_point()
            or not bool(torch.isfinite(pair_states).all().item())
        ):
            raise ValueError("V19 pair states must be finite [nodes,nodes,width]")
        node_count = pair_states.shape[0]
        row_weights = torch.softmax(self.row_attention(pair_states), dim=1)
        rows = torch.einsum("ijh,ijw->ihw", row_weights, pair_states)
        columns_state = pair_states.transpose(0, 1)
        column_weights = torch.softmax(
            self.column_attention(columns_state), dim=1
        )
        columns = torch.einsum(
            "ijh,ijw->ihw", column_weights, columns_state
        )
        index = torch.arange(node_count, device=pair_states.device)
        diagonal = pair_states[index, index]
        return self.node_projection(
            torch.cat(
                (rows.reshape(node_count, -1), columns.reshape(node_count, -1), diagonal),
                dim=-1,
            )
        )


def _seeded_module(seed: int, constructor):
    cpu_rng = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(seed)
        return constructor()
    finally:
        torch.set_rng_state(cpu_rng)


class V12ChampionPairedGraphContextController(v12.SoftwarePipelineController):
    """Terminal V12 plus one production-wired paired graph residual."""

    def __init__(self, profile: v12.SoftwarePipelineRunProfile) -> None:
        if profile != v12.SOFTWARE_PIPELINE_PROFILES["smoke"]:
            raise ValueError("V19 accepts only the frozen V12 smoke profile")
        cpu_rng = torch.get_rng_state()
        try:
            super().__init__(profile)
            self.paired_graph_node_encoder = _seeded_module(
                _NODE_ENCODER_SEED, lambda: ContextAxisNodeEncoder(profile)
            )
            self.paired_graph_update = _seeded_module(
                _GRAPH_UPDATE_SEED,
                lambda: nn.Sequential(
                    nn.LayerNorm(2 * profile.width),
                    nn.Linear(2 * profile.width, profile.hidden_width),
                    nn.SiLU(),
                    nn.Linear(profile.hidden_width, profile.width),
                ),
            )
            self.paired_graph_scorer = _seeded_module(
                _PAIR_SCORER_SEED,
                lambda: nn.Sequential(
                    nn.LayerNorm(6 * profile.width),
                    nn.Linear(6 * profile.width, profile.hidden_width),
                    nn.SiLU(),
                    nn.Linear(profile.hidden_width, profile.width),
                    nn.SiLU(),
                    nn.Linear(profile.width, 1, bias=False),
                ),
            )
            nn.init.zeros_(self.paired_graph_scorer[-1].weight)
        finally:
            torch.set_rng_state(cpu_rng)
        self._paired_graph_lesion: str | None = None
        _enforce_mutable_scope(self)

    def initial_state(self, batch_size: int = 1) -> V19SoftwareReconstructionState:
        base = super().initial_state(batch_size)
        return V19SoftwareReconstructionState(
            pointer=base.pointer,
            role=base.role,
            context_trace_keys=base.context_trace_keys,
            relation_trace_values=base.relation_trace_values,
            context_trace_graphs=base.role.keys.new_zeros(
                (batch_size, base.role.slot_count, _MAX_GRAPH_NODES, _MAX_GRAPH_NODES)
            ).to(torch.bool),
            context_trace_graph_masks=torch.zeros(
                (batch_size, base.role.slot_count, _MAX_GRAPH_NODES),
                device=base.role.keys.device,
                dtype=torch.bool,
            ),
        )

    def encode_task(self, task: v12.PublicSoftwarePipelineTask) -> V19SoftwareTaskEncoding:
        base = super().encode_task(task)
        components = v12._components_in_candidate_order(task)
        graphs = []
        masks = []
        for candidate in components:
            predecessors = tuple(
                predecessor
                for predecessor in components
                if predecessor.output_type == candidate.input_type
            )
            if len(predecessors) > 1:
                raise ValueError("V19 context action has multiple public predecessors")
            graph = torch.zeros(
                (_MAX_GRAPH_NODES, _MAX_GRAPH_NODES),
                device=base.relation_context_embeddings.device,
                dtype=torch.bool,
            )
            mask = torch.zeros(
                (_MAX_GRAPH_NODES,),
                device=base.relation_context_embeddings.device,
                dtype=torch.bool,
            )
            if predecessors:
                _, adjacency = v12._incidence_graph(predecessors[0])
                node_count = len(adjacency)
                if not 1 <= node_count <= _MAX_GRAPH_NODES:
                    raise ValueError("V19 predecessor graph exceeds the declared node capacity")
                active = torch.tensor(
                    adjacency,
                    device=base.relation_context_embeddings.device,
                    dtype=torch.bool,
                )
                graph[:node_count, :node_count] = active
                mask[:node_count] = True
            graphs.append(graph)
            masks.append(mask)
        values = {field.name: getattr(base, field.name) for field in fields(base)}
        return V19SoftwareTaskEncoding(
            **values,
            context_graph_adjacencies=torch.stack(graphs),
            context_graph_masks=torch.stack(masks),
        )

    def _graph_node_tokens(
        self, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        capacity = adjacency.shape[0] if adjacency.ndim == 2 else 0
        if (
            adjacency.shape != (capacity, capacity)
            or not 1 <= capacity <= _MAX_GRAPH_NODES
            or node_mask.shape != (capacity,)
            or adjacency.device != node_mask.device
            or adjacency.dtype != torch.bool
            or node_mask.dtype != torch.bool
            or not bool(node_mask.any().item())
        ):
            raise ValueError("V19 raw graph and node mask are invalid")
        active_index = node_mask.nonzero(as_tuple=False).flatten()
        active = adjacency.index_select(0, active_index).index_select(1, active_index)
        active = active.to(dtype=self.procedure_start.dtype)
        # Critical ordering: crop before the frozen pair encoder.  Padding can
        # never become a learned graph feature.
        pair_states = self.evidence_context_encoder(active, torch.zeros_like(active))
        active_tokens = self.paired_graph_node_encoder(pair_states)
        tokens = active_tokens.new_zeros((_MAX_GRAPH_NODES, self.profile.width))
        compact_mask = torch.zeros(
            (_MAX_GRAPH_NODES,), device=node_mask.device, dtype=torch.bool
        )
        compact_mask[: active_tokens.shape[0]] = True
        tokens[: active_tokens.shape[0]] = active_tokens
        return tokens, compact_mask

    def _paired_graph_raw_residual(
        self,
        query_adjacency: torch.Tensor,
        query_mask: torch.Tensor,
        stored_adjacency: torch.Tensor,
        stored_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, query_mask = self._graph_node_tokens(query_adjacency, query_mask)
        stored, stored_mask = self._graph_node_tokens(stored_adjacency, stored_mask)
        scale = math.sqrt(self.profile.width)
        similarities = query @ stored.transpose(0, 1) / scale
        query_valid = query_mask.unsqueeze(1)
        stored_valid = stored_mask.unsqueeze(0)
        if self._paired_graph_lesion == "uniform_cross_graph_attention":
            query_to_stored = stored_mask.to(query.dtype).unsqueeze(0)
            query_to_stored = query_to_stored / query_to_stored.sum(dim=1, keepdim=True)
            query_to_stored = query_to_stored.expand(_MAX_GRAPH_NODES, -1)
            query_to_stored = query_to_stored * query_valid.to(query.dtype)
            stored_to_query = query_mask.to(query.dtype).unsqueeze(1)
            stored_to_query = stored_to_query / stored_to_query.sum(dim=0, keepdim=True)
            stored_to_query = stored_to_query.expand(-1, _MAX_GRAPH_NODES)
            stored_to_query = stored_to_query * stored_valid.to(query.dtype)
        else:
            query_to_stored = torch.softmax(
                similarities.masked_fill(~stored_valid, -torch.inf), dim=1
            )
            query_to_stored = torch.where(
                query_valid, query_to_stored, torch.zeros_like(query_to_stored)
            )
            stored_to_query = torch.softmax(
                similarities.masked_fill(~query_valid, -torch.inf), dim=0
            )
            stored_to_query = torch.where(
                stored_valid, stored_to_query, torch.zeros_like(stored_to_query)
            )
        mismatch_query = query - query_to_stored @ stored
        mismatch_stored = stored - stored_to_query.transpose(0, 1) @ query
        if self._paired_graph_lesion == "mismatch_zero":
            mismatch_query = torch.zeros_like(mismatch_query)
            mismatch_stored = torch.zeros_like(mismatch_stored)
        updated_query = query + self.paired_graph_update(
            torch.cat((query, mismatch_query), dim=-1)
        )
        updated_stored = stored + self.paired_graph_update(
            torch.cat((stored, mismatch_stored), dim=-1)
        )
        updated_query = updated_query * query_mask.unsqueeze(-1)
        updated_stored = updated_stored * stored_mask.unsqueeze(-1)

        def pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            selected = tokens.masked_select(mask.unsqueeze(-1)).reshape(-1, self.profile.width)
            return torch.cat((selected.mean(dim=0), selected.amax(dim=0)), dim=-1)

        pooled_query = pool(updated_query, query_mask)
        pooled_stored = pool(updated_stored, stored_mask)
        features = torch.cat(
            (
                0.5 * (pooled_query + pooled_stored),
                (pooled_query - pooled_stored).abs(),
                pooled_query * pooled_stored,
            ),
            dim=-1,
        )
        return self.paired_graph_scorer(features).reshape(())

    def _paired_graph_context_logits(
        self,
        query_context_codes: torch.Tensor,
        query_graphs: torch.Tensor,
        query_masks: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_graphs: torch.Tensor,
        stored_masks: torch.Tensor,
    ) -> torch.Tensor:
        inherited = self._context_pair_logits(query_context_codes, stored_contexts)
        if inherited.shape[1] <= 1 or self._paired_graph_lesion == "zero_residual":
            return inherited
        adjusted_rows = []
        for query_index in range(query_graphs.shape[0]):
            inherited_row = inherited[query_index]
            duplicate_index = next(
                (
                    previous
                    for previous in range(query_index)
                    if torch.equal(
                        query_context_codes[query_index],
                        query_context_codes[previous],
                    )
                    and torch.equal(query_graphs[query_index], query_graphs[previous])
                    and torch.equal(query_masks[query_index], query_masks[previous])
                ),
                None,
            )
            if duplicate_index is not None:
                adjusted_rows.append(adjusted_rows[duplicate_index])
                continue
            if not bool(query_masks[query_index].any().item()):
                adjusted_rows.append(inherited_row)
                continue
            raw = torch.stack(
                tuple(
                    self._paired_graph_raw_residual(
                        query_graphs[query_index],
                        query_masks[query_index],
                        stored_graphs[stored_index],
                        stored_masks[stored_index],
                    )
                    for stored_index in range(stored_graphs.shape[0])
                )
            )
            residual = _RESIDUAL_BOUND * torch.tanh(raw)
            work = inherited_row.to(torch.float64)
            residual_work = residual.to(torch.float64)
            correction = _CONTEXT_TEMPERATURE * (
                torch.logsumexp((work + residual_work) / _CONTEXT_TEMPERATURE, dim=-1)
                - torch.logsumexp(work / _CONTEXT_TEMPERATURE, dim=-1)
            )
            adjusted_rows.append(
                (work + residual_work - correction).to(inherited.dtype)
            )
        return torch.stack(adjusted_rows)

    def _paired_graph_evidence_read(
        self,
        query_context_codes: torch.Tensor,
        query_relation_codes: torch.Tensor,
        query_graphs: torch.Tensor,
        query_masks: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_relations: torch.Tensor,
        stored_graphs: torch.Tensor,
        stored_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_logits = self._paired_graph_context_logits(
            query_context_codes,
            query_graphs,
            query_masks,
            stored_contexts,
            stored_graphs,
            stored_masks,
        )
        all_weights = torch.softmax(
            torch.cat(
                (
                    context_logits / _CONTEXT_TEMPERATURE,
                    context_logits.new_zeros(context_logits.shape[0], 1),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        context_weights = all_weights[:, : stored_contexts.shape[0]]
        null_weights = all_weights[:, stored_contexts.shape[0]]
        relation_logits = self._relation_pair_logits(
            query_relation_codes, stored_relations
        )
        scores = (context_weights * relation_logits).sum(dim=-1)
        return scores, context_weights, null_weights, relation_logits, context_logits

    @contextmanager
    def paired_graph_lesion(self, kind: str) -> Iterator[None]:
        if kind not in {
            "zero_residual",
            "uniform_cross_graph_attention",
            "mismatch_zero",
        }:
            raise ValueError("unknown V19 paired-graph lesion")
        if self._paired_graph_lesion is not None:
            raise RuntimeError("nested V19 lesions are forbidden")
        self._paired_graph_lesion = kind
        try:
            yield
        finally:
            self._paired_graph_lesion = None

    def _paired_graph_evidence_scores(
        self,
        encoding: V19SoftwareTaskEncoding,
        state: V19SoftwareReconstructionState,
    ) -> torch.Tensor:
        _validate_v19_state(self, state)
        if type(encoding) is not V19SoftwareTaskEncoding:
            raise TypeError("V19 production scoring requires V19 task encoding")
        trace_slots = self.role_memory.trace_slot_count
        occupied = state.role.occupied[0, :trace_slots]
        occupied = (
            occupied
            & (state.context_trace_keys[0, :trace_slots].norm(dim=-1) > 1.0e-8)
            & (state.relation_trace_values[0, :trace_slots].norm(dim=-1) > 1.0e-8)
            & state.context_trace_graph_masks[0, :trace_slots].any(dim=-1)
        )
        query_present = (
            (encoding.relation_context_embeddings.norm(dim=-1) > 1.0e-8)
            & (encoding.relation_component_embeddings.norm(dim=-1) > 1.0e-8)
            & encoding.context_graph_masks.any(dim=-1)
        )
        if not bool(occupied.any().item()) or not bool(query_present.any().item()):
            return encoding.relation_component_embeddings.new_zeros(
                encoding.relation_component_embeddings.shape[0]
            )
        scores, _, _, _, _ = self._paired_graph_evidence_read(
            encoding.relation_context_embeddings,
            encoding.relation_component_embeddings,
            encoding.context_graph_adjacencies,
            encoding.context_graph_masks,
            state.context_trace_keys[0, :trace_slots][occupied],
            state.relation_trace_values[0, :trace_slots][occupied],
            state.context_trace_graphs[0, :trace_slots][occupied],
            state.context_trace_graph_masks[0, :trace_slots][occupied],
        )
        return torch.where(query_present, scores, torch.zeros_like(scores))

    def score_actions(
        self,
        task: v12.PublicSoftwarePipelineTask,
        state: V19SoftwareReconstructionState,
        *,
        current_state_belief: torch.Tensor | None = None,
        steps_remaining: int | None = None,
        encoding: V19SoftwareTaskEncoding | None = None,
        include_pointer_memory: bool = True,
        include_role_memory: bool = True,
        include_backward_reasoning: bool = True,
        detach_evidence_action_input: bool = False,
        use_legacy_evidence: bool = False,
    ) -> v12.SoftwareStepScores:
        _validate_v19_state(self, state)
        encoded = self.encode_task(task) if encoding is None else encoding
        if type(encoded) is not V19SoftwareTaskEncoding:
            raise TypeError("V19 score_actions rejects base V12 encoding")
        base_scores = super().score_actions(
            task,
            state,
            current_state_belief=current_state_belief,
            steps_remaining=steps_remaining,
            encoding=encoded,
            include_pointer_memory=include_pointer_memory,
            include_role_memory=include_role_memory,
            include_backward_reasoning=include_backward_reasoning,
            detach_evidence_action_input=detach_evidence_action_input,
            use_legacy_evidence=use_legacy_evidence,
        )
        if (
            use_legacy_evidence
            or not include_role_memory
            or self._paired_graph_lesion == "zero_residual"
            or not bool(torch.count_nonzero(self.paired_graph_scorer[-1].weight).item())
        ):
            return base_scores
        paired_scores = self._paired_graph_evidence_scores(encoded, state)
        base_input = (
            base_scores.evidence_match_scores.detach()
            if detach_evidence_action_input
            else base_scores.evidence_match_scores
        )
        paired_input = paired_scores.detach() if detach_evidence_action_input else paired_scores
        contribution_delta = self._evidence_action_contribution(
            paired_input
        ) - self._evidence_action_contribution(base_input)
        action_logits = base_scores.action_logits + contribution_delta
        logits = torch.cat((action_logits, base_scores.stop_logit.unsqueeze(0)), dim=0)
        return v12.SoftwareStepScores(
            logits=logits,
            action_logits=action_logits,
            stop_logit=base_scores.stop_logit,
            successor_state_logits=base_scores.successor_state_logits,
            pointer_contexts=base_scores.pointer_contexts,
            role_contexts=base_scores.role_contexts,
            outcome_contexts=base_scores.outcome_contexts,
            evidence_match_scores=paired_scores,
            reasoning_node_codes=base_scores.reasoning_node_codes,
            current_state_belief=base_scores.current_state_belief,
        )


def _enforce_mutable_scope(controller: V12ChampionPairedGraphContextController) -> None:
    actual = tuple(
        name
        for name, _ in controller.named_parameters()
        if name.startswith(_MUTABLE_PREFIXES)
    )
    if actual != MUTABLE_PARAMETER_NAMES or len(actual) != 21:
        raise RuntimeError("V19 mutable tensor identity changed")
    named = dict(controller.named_parameters())
    if sum(named[name].numel() for name in actual) != 34_048:
        raise RuntimeError("V19 mutable parameter count changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name in MUTABLE_PARAMETER_NAMES)


def _validate_v19_state(
    controller: V12ChampionPairedGraphContextController,
    state: V19SoftwareReconstructionState,
) -> None:
    if type(state) is not V19SoftwareReconstructionState:
        raise TypeError("V19 requires its versioned raw-graph state")
    v12._validate_controller_state(controller, state)
    state.__post_init__()


def paired_graph_parameter_report(
    controller: V12ChampionPairedGraphContextController,
    mixer: v12.AnonymousConflictMixer,
) -> dict[str, object]:
    if not isinstance(controller, V12ChampionPairedGraphContextController):
        raise TypeError("controller must be the V19 controller")
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


@dataclass(slots=True)
class V12ChampionPairedGraphContextSystem:
    controller: V12ChampionPairedGraphContextController
    mixer: v12.AnonymousConflictMixer
    competence_state: V19SoftwareReconstructionState
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
    controller: V12ChampionPairedGraphContextController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if not name.startswith(_MUTABLE_PREFIXES)
    }


def _mutable_state(
    controller: V12ChampionPairedGraphContextController,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in controller.state_dict().items()
        if name.startswith(_MUTABLE_PREFIXES)
    }


def _base_controller_from_successor(
    controller: V12ChampionPairedGraphContextController,
) -> v12.SoftwarePipelineController:
    cpu_rng = torch.get_rng_state()
    try:
        base = v12.SoftwarePipelineController(controller.profile)
    finally:
        torch.set_rng_state(cpu_rng)
    base.load_state_dict(
        {
            name: value.detach().cpu().clone()
            for name, value in _inherited_state(controller).items()
        },
        strict=True,
    )
    base.eval()
    return base


def _base_state_from_v19(
    state: V19SoftwareReconstructionState,
) -> v12.SoftwareReconstructionState:
    snapshot = v12.snapshot_software_reconstruction_state(state)
    return v12.restore_software_reconstruction_state(
        {name: value.detach().cpu().clone() for name, value in snapshot.items()}
    )


def inherited_v12_controller_digest(
    controller: V12ChampionPairedGraphContextController,
) -> str:
    return v12.software_pipeline_model_digest(_base_controller_from_successor(controller))


def paired_graph_mutable_digest(
    controller: V12ChampionPairedGraphContextController,
) -> str:
    return _mapping_digest(_MUTABLE_DIGEST_DOMAIN, _mutable_state(controller))


def _source_system_digest(system: V12ChampionPairedGraphContextSystem) -> str:
    return v12.public_relation_conflict_system_digest(
        _base_controller_from_successor(system.controller),
        system.mixer,
        _base_state_from_v19(system.competence_state),
    )


def paired_graph_system_digest(system: V12ChampionPairedGraphContextSystem) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_paired_graph_context_plan()["plan_digest"],
        "source": asdict(system.source),
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": paired_graph_mutable_digest(system.controller),
        "mixer_digest": v12.anonymous_conflict_mixer_digest(system.mixer),
        "competence_digest": v19_reconstruction_state_digest(system.competence_state),
        "context_updates": system.context_updates,
        "optimizer_digest": paired_graph_optimizer_digest(system.optimizer_state),
    }
    return _json_digest(_SYSTEM_DIGEST_DOMAIN, payload)


def _assert_source_lineage(system: V12ChampionPairedGraphContextSystem) -> None:
    if system.source != _expected_source_binding():
        raise RuntimeError("V19 terminal V12 source binding changed")
    if inherited_v12_controller_digest(system.controller) != V12_CONTROLLER_DIGEST:
        raise RuntimeError("V19 changed an inherited V12 controller byte")
    if v12.anonymous_conflict_mixer_digest(system.mixer) != V12_MIXER_DIGEST:
        raise RuntimeError("V19 changed the inherited conflict mixer")
    if any(parameter.requires_grad for parameter in system.mixer.parameters()):
        raise RuntimeError("V19 inherited conflict mixer is not frozen")
    base_competence = _base_state_from_v19(system.competence_state)
    if v12.software_reconstruction_state_digest(base_competence) != V12_COMPETENCE_DIGEST:
        raise RuntimeError("V19 changed inherited V12 competence")
    if _source_system_digest(system) != V12_SYSTEM_DIGEST:
        raise RuntimeError("V19 source system lineage changed")
    if type(system.context_updates) is not int or not 0 <= system.context_updates <= _CONTEXT_UPDATES:
        raise RuntimeError("V19 context update count is invalid")
    if system.context_updates == 0:
        if system.optimizer_state is not None:
            raise RuntimeError("fresh V19 lineage unexpectedly has optimizer moments")
    elif not isinstance(system.optimizer_state, Mapping):
        raise RuntimeError("learned V19 lineage lost optimizer moments")
    else:
        _validate_optimizer_state(
            system.optimizer_state,
            system.controller,
            expected_steps=system.context_updates,
        )
    _validate_v19_state(system.controller, system.competence_state)
    _enforce_mutable_scope(system.controller)


def _migrate_loaded_v12_system(
    controller: v12.SoftwarePipelineController,
    mixer: v12.AnonymousConflictMixer,
    state: v12.SoftwareReconstructionState,
    *,
    source_checkpoint_sha256: str,
    device: torch.device | str = "cpu",
) -> V12ChampionPairedGraphContextSystem:
    _verify_frozen_dependencies()
    if source_checkpoint_sha256.upper() != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V19 accepts only the frozen terminal V12 checkpoint")
    if type(controller) is not v12.SoftwarePipelineController:
        raise TypeError("V19 migration requires the exact V12 controller type")
    trace_slots = state.role.slot_count // 2
    if bool(state.role.occupied[:, :trace_slots].any().item()):
        raise RuntimeError("V19 refuses legacy trace state without raw graphs")
    source = V12ChampionSourceBinding(
        checkpoint_sha256=V12_CHECKPOINT_SHA256,
        controller_digest=v12.software_pipeline_model_digest(controller),
        mixer_digest=v12.anonymous_conflict_mixer_digest(mixer),
        competence_digest=v12.software_reconstruction_state_digest(state),
        system_digest=v12.public_relation_conflict_system_digest(controller, mixer, state),
    )
    if source != _expected_source_binding():
        raise RuntimeError("V19 source payload is not terminal V12 lineage")
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_snapshot(device)
    try:
        successor = V12ChampionPairedGraphContextController(controller.profile).to(device)
        loaded = successor.load_state_dict(controller.state_dict(), strict=False)
        if tuple(sorted(loaded.missing_keys)) != tuple(sorted(MUTABLE_PARAMETER_NAMES)) or loaded.unexpected_keys:
            raise RuntimeError("V19 migration state boundary changed")
        cloned_mixer = v12.AnonymousConflictMixer(
            feature_count=mixer.feature_count,
            hidden_width=mixer.hidden_width,
            anchor_weight=mixer.anchor_weight,
        ).to(device)
        cloned_mixer.load_state_dict(mixer.state_dict(), strict=True)
        for parameter in cloned_mixer.parameters():
            parameter.requires_grad_(False)
    finally:
        torch.set_rng_state(cpu_rng)
        _restore_cuda_rng_snapshot(cuda_rng)
    base_snapshot = {
        name: value.detach().to(device).clone()
        for name, value in v12.snapshot_software_reconstruction_state(state).items()
    }
    base_state = v12.restore_software_reconstruction_state(base_snapshot)
    competence = V19SoftwareReconstructionState(
        pointer=base_state.pointer,
        role=base_state.role,
        context_trace_keys=base_state.context_trace_keys,
        relation_trace_values=base_state.relation_trace_values,
        context_trace_graphs=base_state.role.keys.new_zeros(
            (
                base_state.role.batch_size,
                base_state.role.slot_count,
                _MAX_GRAPH_NODES,
                _MAX_GRAPH_NODES,
            )
        ).to(torch.bool),
        context_trace_graph_masks=torch.zeros(
            (
                base_state.role.batch_size,
                base_state.role.slot_count,
                _MAX_GRAPH_NODES,
            ),
            device=base_state.role.keys.device,
            dtype=torch.bool,
        ),
    )
    inherited = _inherited_state(successor)
    if inherited.keys() != controller.state_dict().keys() or any(
        not torch.equal(value.detach().cpu(), controller.state_dict()[name].detach().cpu())
        for name, value in inherited.items()
    ):
        raise RuntimeError("V19 migration changed inherited V12 state")
    system = V12ChampionPairedGraphContextSystem(
        controller=successor,
        mixer=cloned_mixer,
        competence_state=competence,
        source=source,
    )
    successor.eval()
    cloned_mixer.eval()
    _assert_source_lineage(system)
    report = paired_graph_parameter_report(successor, cloned_mixer)
    if (
        report["controller_parameters"] != 299_654
        or report["complete_learned_system_parameters"] != 303_058
    ):
        raise RuntimeError("V19 migrated parameter count changed")
    return system


def load_v12_champion_paired_graph_context_source(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionPairedGraphContextSystem:
    actual = _sha256_file(path)
    if actual != V12_CHECKPOINT_SHA256:
        raise RuntimeError("V19 source checkpoint SHA-256 is not frozen V12")
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_snapshot(device)
    try:
        controller, mixer, state = v12.load_public_relation_conflict_checkpoint(
            path, device=device
        )
        return _migrate_loaded_v12_system(
            controller,
            mixer,
            state,
            source_checkpoint_sha256=actual,
            device=device,
        )
    finally:
        torch.set_rng_state(cpu_rng)
        _restore_cuda_rng_snapshot(cuda_rng)


def acquire_v19_public_pipeline_traces(
    controller: V12ChampionPairedGraphContextController,
    task: v12.PublicSoftwarePipelineTask,
    state: V19SoftwareReconstructionState,
) -> v12.SoftwareTraceAcquisition:
    """Atomically retain V12 trace events plus aligned raw predecessor graphs."""

    _validate_v19_state(controller, state)
    transitions = v12._public_transitions(task)
    if not transitions:
        return v12.SoftwareTraceAcquisition(state, 0, 0, 0)
    encoded = controller.encode_task(task)
    pointer_keys = []
    pointer_values = []
    pointer_pair_ids = []
    pointer_successor_ids = []
    role_keys = []
    role_values = []
    action_indices = []
    for transition in transitions:
        before_index = v12._state_index(task.states, transition.before)
        after_index = v12._state_index(task.states, transition.after)
        action_index = v12._action_index(task.grounded_candidates, transition.action)
        action_indices.append(action_index)
        pointer_keys.append(
            F.normalize(
                encoded.pointer_state_embeddings[before_index]
                + encoded.pointer_component_embeddings[action_index],
                dim=-1,
                eps=1.0e-8,
            )
        )
        pointer_values.append(encoded.pointer_state_embeddings[after_index])
        pointer_pair_ids.append(encoded.pointer_pair_ids[before_index, action_index])
        pointer_successor_ids.append(encoded.pointer_successor_ids[after_index])
        role_keys.append(encoded.role_pair_keys[before_index, action_index])
        role_values.append(
            controller.trace_role_value(
                encoded.local_pair_embeddings[before_index, action_index],
                encoded.operator_embeddings[action_index],
                encoded.relative_effect_embeddings[before_index, action_index, after_index],
            )
        )
    pointer_write = controller.pointer_memory.write_events(
        torch.stack(pointer_keys),
        torch.stack(pointer_values),
        state.pointer,
        lane="trace",
        public_source_action_ids=torch.stack(pointer_pair_ids),
        public_successor_ids=torch.stack(pointer_successor_ids),
    )
    zero_ids = torch.zeros(
        (len(role_keys), v12._POINTER_WORDS),
        device=state.role.keys.device,
        dtype=torch.long,
    )
    role_write = controller.role_memory.write_events(
        torch.stack(role_keys),
        torch.stack(role_values),
        state.role,
        lane="trace",
        public_source_action_ids=zero_ids,
        public_successor_ids=zero_ids,
    )
    if not pointer_write.accepted or not role_write.accepted:
        return v12.SoftwareTraceAcquisition(state, len(transitions), 0, 0)
    context_keys = state.context_trace_keys.clone()
    relation_values = state.relation_trace_values.clone()
    raw_graphs = state.context_trace_graphs.clone()
    raw_masks = state.context_trace_graph_masks.clone()
    if len(role_write.write_slots) != len(action_indices):
        raise RuntimeError("V19 graph events lost role-slot alignment")
    for slot, action_index in zip(role_write.write_slots, action_indices, strict=True):
        if not 0 <= slot < controller.role_memory.trace_slot_count:
            raise RuntimeError("V19 graph event escaped the role trace partition")
        context_keys[0, slot] = encoded.relation_context_embeddings[action_index]
        relation_values[0, slot] = encoded.relation_component_embeddings[action_index]
        raw_graphs[0, slot] = encoded.context_graph_adjacencies[action_index]
        raw_masks[0, slot] = encoded.context_graph_masks[action_index]
    try:
        candidate = V19SoftwareReconstructionState(
            pointer=pointer_write.state,
            role=role_write.state,
            context_trace_keys=context_keys,
            relation_trace_values=relation_values,
            context_trace_graphs=raw_graphs,
            context_trace_graph_masks=raw_masks,
        )
        finite = bool(torch.isfinite(controller.score_actions(task, candidate).logits).all().item())
    except (RuntimeError, TypeError, ValueError):
        finite = False
    if not finite:
        return v12.SoftwareTraceAcquisition(state, len(transitions), 0, 0)
    return v12.SoftwareTraceAcquisition(
        candidate,
        len(transitions),
        len(pointer_write.write_slots),
        len(role_write.write_slots),
    )


def _paired_graph_optimizer(
    controller: V12ChampionPairedGraphContextController,
) -> torch.optim.AdamW:
    named = dict(controller.named_parameters())
    return torch.optim.AdamW(
        (
            {
                "params": [
                    named[name]
                    for name in (
                        _NODE_ENCODER_PARAMETER_NAMES + _GRAPH_UPDATE_PARAMETER_NAMES
                    )
                ],
                "lr": _TRUNK_LEARNING_RATE,
            },
            {
                "params": [named[name] for name in _PAIR_SCORER_PARAMETER_NAMES],
                "lr": _SCORER_LEARNING_RATE,
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
        {
            "names": _NODE_ENCODER_PARAMETER_NAMES + _GRAPH_UPDATE_PARAMETER_NAMES,
            "lr": _TRUNK_LEARNING_RATE,
        },
        {"names": _PAIR_SCORER_PARAMETER_NAMES, "lr": _SCORER_LEARNING_RATE},
    )


def _canonical_optimizer_state(
    optimizer: torch.optim.AdamW,
    controller: V12ChampionPairedGraphContextController,
) -> dict[str, object]:
    named = dict(controller.named_parameters())
    frozen_group_values = {
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
    for actual, expected in zip(optimizer.param_groups, _optimizer_groups(), strict=True):
        actual_names = tuple(
            next(name for name, parameter in named.items() if parameter is value)
            for value in actual["params"]
        )
        if (
            actual_names != expected["names"]
            or float(actual["lr"]) != expected["lr"]
            or any(actual.get(name) != expected_value for name, expected_value in frozen_group_values.items())
        ):
            raise RuntimeError("V19 optimizer parameter groups changed")
    slots: dict[str, dict[str, torch.Tensor]] = {}
    for name in MUTABLE_PARAMETER_NAMES:
        state = optimizer.state.get(named[name])
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"V19 optimizer state is incomplete: {name}")
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
    controller: V12ChampionPairedGraphContextController,
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
        raise RuntimeError("V19 optimizer configuration changed")
    slots = value["state"]
    if not isinstance(slots, Mapping) or set(slots) != set(MUTABLE_PARAMETER_NAMES):
        raise RuntimeError("V19 optimizer slot ownership changed")
    named = dict(controller.named_parameters())
    steps = []
    for name in MUTABLE_PARAMETER_NAMES:
        slot = slots[name]
        if not isinstance(slot, Mapping) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"V19 optimizer slot fields changed: {name}")
        step, exp_avg, exp_avg_sq = slot["step"], slot["exp_avg"], slot["exp_avg_sq"]
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
            raise RuntimeError(f"V19 optimizer slot tensor changed: {name}")
        steps.append(int(step.item()))
    if len(set(steps)) != 1 or (
        expected_steps is not None and steps[0] != expected_steps
    ):
        raise RuntimeError("V19 optimizer step counters changed")


def paired_graph_optimizer_digest(value: Mapping[str, object] | None) -> str:
    digest = hashlib.sha256(_OPTIMIZER_DIGEST_DOMAIN)
    if value is None:
        digest.update(b"none")
        return "sha256:" + digest.hexdigest()
    metadata = {
        "version": value.get("version"),
        "groups": value.get("groups"),
        "hyperparameters": value.get("hyperparameters"),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii"))
    slots = value.get("state")
    if not isinstance(slots, Mapping):
        raise RuntimeError("V19 optimizer digest requires named slots")
    for parameter_name in sorted(slots):
        slot = slots[parameter_name]
        if not isinstance(slot, Mapping):
            raise RuntimeError("V19 optimizer digest slot is invalid")
        for slot_name in sorted(slot):
            tensor = slot[slot_name]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("V19 optimizer digest value is not a tensor")
            _update_tensor_digest(digest, f"{parameter_name}.{slot_name}", tensor)
    return "sha256:" + digest.hexdigest()


def restore_paired_graph_optimizer(
    system: V12ChampionPairedGraphContextSystem,
) -> torch.optim.AdamW:
    if system.optimizer_state is None:
        raise RuntimeError("V19 optimizer continuation has no named state")
    _validate_optimizer_state(
        system.optimizer_state,
        system.controller,
        expected_steps=system.context_updates,
    )
    optimizer = _paired_graph_optimizer(system.controller)
    named = dict(system.controller.named_parameters())
    slots = system.optimizer_state["state"]
    assert isinstance(slots, Mapping)
    for name in MUTABLE_PARAMETER_NAMES:
        slot = slots[name]
        assert isinstance(slot, Mapping)
        optimizer.state[named[name]] = {
            slot_name: tensor.detach().to(named[name].device).clone()
            for slot_name, tensor in slot.items()
            if isinstance(tensor, torch.Tensor)
        }
    return optimizer


@dataclass(frozen=True, slots=True)
class V19PairedGraphCreditRow:
    """One public row whose slot credit uses the production paired matcher."""

    heldout_index: int
    transition_index: int
    positive_index: int
    negative_index: int
    positive_margin: torch.Tensor
    negative_margin: torch.Tensor
    slot_positive_margins: torch.Tensor
    slot_negative_margins: torch.Tensor
    context_weights: torch.Tensor
    context_null_weight: torch.Tensor
    context_real_logits: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        vectors = (
            self.slot_positive_margins,
            self.slot_negative_margins,
            self.context_weights,
            self.context_real_logits,
        )
        if (
            any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
                for value in vectors
            )
            or len({value.shape for value in vectors}) != 1
            or self.valid_mask.shape != self.context_weights.shape
            or self.valid_mask.dtype != torch.bool
            or self.valid_mask.device != self.context_weights.device
            or self.context_null_weight.shape != ()
            or not bool(torch.isfinite(self.context_null_weight).item())
            or bool((self.context_weights < 0.0).any().item())
            or abs(
                float((self.context_weights.sum() + self.context_null_weight).item())
                - 1.0
            )
            > 1.0e-6
        ):
            raise ValueError("V19 credit row tensors are not aligned")


def _v19_relation_credit_task(
    controller: V12ChampionPairedGraphContextController,
    task: v12.PublicSoftwarePipelineTask,
) -> tuple[
    tuple[v12.Transition, ...],
    tuple[int, ...],
    tuple[int | None, ...],
    V19SoftwareTaskEncoding,
]:
    encoded = controller.encode_task(task)
    transitions = v12._public_transitions(task)
    if not transitions:
        raise ValueError("V19 relation credit requires a public observation")
    components = v12._components_in_candidate_order(task)
    observed = tuple(
        v12._action_index(task.grounded_candidates, transition.action)
        for transition in transitions
    )
    alternatives = tuple(
        v12._same_contract_alternative_index(components, index) for index in observed
    )
    return transitions, observed, alternatives, encoded


def public_paired_graph_credit_rows(
    controller: V12ChampionPairedGraphContextController,
    stream: v12.SoftwarePipelineStream,
    *,
    reverse_evidence_order: bool = False,
    reverse_public_presentation: bool = False,
) -> tuple[V19PairedGraphCreditRow, ...]:
    """Build public relation-derived context rows through the live V19 matcher."""

    if type(controller) is not V12ChampionPairedGraphContextController:
        raise TypeError("V19 credit rows require the V19 controller")
    if not isinstance(stream, v12.SoftwarePipelineStream):
        raise TypeError("V19 credit requires a software-pipeline stream")
    if type(reverse_evidence_order) is not bool or type(reverse_public_presentation) is not bool:
        raise TypeError("V19 covariance flags must be bool")
    if stream.mechanism_partition != "train" or stream.control_arm != "correct":
        raise ValueError("V19 credit accepts only original train streams")
    if len(stream.supports) != 4:
        raise ValueError("V19 credit requires four public support packages")
    tasks = []
    for pair in stream.supports:
        task = pair.learner
        if reverse_public_presentation:
            task = replace(
                task,
                components=tuple(reversed(task.components)),
                grounded_candidates=tuple(reversed(task.grounded_candidates)),
                states=tuple(reversed(task.states)),
            )
        tasks.append(task)
    encoded_tasks = tuple(_v19_relation_credit_task(controller, task) for task in tasks)
    rows = []
    for heldout_index, encoded_query in enumerate(encoded_tasks):
        query_transitions, query_observed, query_alternatives, query_encoding = encoded_query
        discriminating = tuple(
            index for index, alternative in enumerate(query_alternatives) if alternative is not None
        )
        if len(discriminating) != 1:
            raise RuntimeError("each V19 public support must expose one contrast")
        transition_index = discriminating[0]
        positive_index = query_observed[transition_index]
        negative_index = query_alternatives[transition_index]
        if negative_index is None:
            raise AssertionError("validated V19 public contrast disappeared")
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
            raise RuntimeError("same-contract V19 alternatives changed predecessor context")
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
            _, evidence_observed, evidence_alternatives, evidence_encoding = encoded_tasks[evidence_index]
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
                        raise RuntimeError("V19 evidence alternative changed predecessor graph")
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
            raise RuntimeError("V19 public credit has no transferable slots")
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
                raise RuntimeError(f"V19 relation alternatives changed context {label}")
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
            raise RuntimeError("same-contract V19 query alternatives changed context")
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
            V19PairedGraphCreditRow(
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
    if len(rows) != _ROWS_PER_STREAM:
        raise RuntimeError("V19 stream lost a public credit row")
    return tuple(rows)


def _paired_graph_row_objective(
    row: V19PairedGraphCreditRow,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    if not isinstance(row, V19PairedGraphCreditRow):
        raise TypeError("V19 objective requires a paired graph credit row")
    valid_count = int(row.valid_mask.sum().item())
    slot_count = row.valid_mask.numel()
    informative = 0 < valid_count < slot_count
    if not informative:
        return None, {
            "informative": False,
            "valid_slot_count": valid_count,
            "slot_count": slot_count,
        }
    probabilities = torch.softmax(
        row.context_real_logits / _CONTEXT_TEMPERATURE, dim=0
    )
    valid_mass = probabilities.masked_select(row.valid_mask).sum()
    list_loss = -torch.log(valid_mass.clamp_min(torch.finfo(valid_mass.dtype).tiny))
    valid_logits = row.context_real_logits.masked_select(row.valid_mask)
    invalid_logits = row.context_real_logits.masked_select(~row.valid_mask)
    differences = valid_logits.unsqueeze(1) - invalid_logits.unsqueeze(0)
    pair_loss = _PAIR_TEMPERATURE * F.softplus(
        (_PAIR_MARGIN - differences) / _PAIR_TEMPERATURE
    ).mean()
    loss = 0.5 * list_loss + 0.5 * pair_loss
    return loss, {
        "informative": True,
        "valid_slot_count": valid_count,
        "slot_count": slot_count,
        "list_loss": float(list_loss.detach().item()),
        "pair_loss": float(pair_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }


def _paired_graph_objective(
    row_groups: Sequence[Sequence[V19PairedGraphCreditRow]],
) -> tuple[torch.Tensor, dict[str, object]]:
    rows = tuple(row for group in row_groups for row in group)
    if not rows:
        raise ValueError("V19 objective requires public rows")
    losses = []
    diagnostics = []
    for row in rows:
        loss, diagnostic = _paired_graph_row_objective(row)
        diagnostics.append(diagnostic)
        if loss is not None:
            losses.append(loss)
    if not losses:
        raise RuntimeError("V19 update has no informative public row")
    objective = torch.stack(losses).mean()
    if not bool(torch.isfinite(objective).item()):
        raise RuntimeError("V19 paired graph objective is non-finite")
    return objective, {
        "rows": len(rows),
        "informative_rows": len(losses),
        "excluded_rows": len(rows) - len(losses),
        "row_diagnostics": tuple(diagnostics),
    }


def _fit_paired_graph_batches(
    system: V12ChampionPairedGraphContextSystem,
    stream_batches: Sequence[Sequence[v12.SoftwarePipelineStream]],
) -> dict[str, object]:
    """Fit a bounded prefix of the frozen V19 plan through production rows."""

    if type(system.controller) is not V12ChampionPairedGraphContextController:
        raise TypeError("V19 fit rejects a base V12 controller")
    if type(system.competence_state) is not V19SoftwareReconstructionState:
        raise TypeError("V19 fit rejects base V12 state")
    if not stream_batches or any(not batch for batch in stream_batches):
        raise ValueError("V19 fit requires nonempty public stream batches")
    if system.context_updates + len(stream_batches) > _CONTEXT_UPDATES:
        raise RuntimeError("V19 fit exceeds the frozen update budget")
    if system.controller._paired_graph_lesion is not None:
        raise RuntimeError("V19 cannot train under a diagnostic lesion")
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
        for name, value in snapshot_v19_reconstruction_state(system.competence_state).items()
    }
    optimizer = (
        _paired_graph_optimizer(controller)
        if system.optimizer_state is None
        else restore_paired_graph_optimizer(system)
    )
    parameters = tuple(named[name] for name in MUTABLE_PARAMETER_NAMES)
    losses = []
    gradient_norms = []
    objective_diagnostics = []
    nonzero_gradient_history = []
    first_head_gradient_nonzero: bool | None = None
    first_upstream_gradients_exact_zero: bool | None = None
    later_upstream_gradient_reached = False
    reached = set()
    start_update = system.context_updates
    was_training = controller.training
    controller.train()
    try:
        for local_index, batch in enumerate(stream_batches):
            global_update = start_update + local_index
            if len(batch) != _STREAMS_PER_UPDATE:
                raise RuntimeError("V19 update lost a public stream")
            row_groups = tuple(
                public_paired_graph_credit_rows(controller, stream) for stream in batch
            )
            objective, diagnostics = _paired_graph_objective(row_groups)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for name in MUTABLE_PARAMETER_NAMES:
                gradient = named[name].grad
                if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                    raise RuntimeError(f"V19 gradient is absent/non-finite: {name}")
            nonzero = tuple(
                name
                for name in MUTABLE_PARAMETER_NAMES
                if bool(torch.count_nonzero(named[name].grad).item())
            )
            reached.update(nonzero)
            nonzero_gradient_history.append(nonzero)
            if global_update == 0:
                head_name = _PAIR_SCORER_PARAMETER_NAMES[-1]
                first_head_gradient_nonzero = head_name in nonzero
                first_upstream_gradients_exact_zero = all(
                    name not in nonzero for name in MUTABLE_PARAMETER_NAMES[:-1]
                )
                if not first_head_gradient_nonzero or not first_upstream_gradients_exact_zero:
                    raise RuntimeError("V19 zero-head gradient staging changed")
            elif any(name in nonzero for name in MUTABLE_PARAMETER_NAMES[:-1]):
                later_upstream_gradient_reached = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, _GRADIENT_CLIP)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("V19 clipped gradient norm is non-finite")
            optimizer.step()
            system.context_updates += 1
            system.optimizer_state = _canonical_optimizer_state(optimizer, controller)
            if any(
                not bool(torch.isfinite(named[name]).all().item())
                for name in MUTABLE_PARAMETER_NAMES
            ):
                raise RuntimeError("V19 update produced a non-finite parameter")
            current_state = controller.state_dict()
            for name, before in inherited_before.items():
                if not torch.equal(before, current_state[name].detach()):
                    raise RuntimeError(f"V19 changed inherited tensor: {name}")
            losses.append(float(objective.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
            objective_diagnostics.append(diagnostics)
    finally:
        controller.train(was_training)
    if any(
        not torch.equal(before, system.mixer.state_dict()[name].detach())
        for name, before in mixer_before.items()
    ):
        raise RuntimeError("V19 fit changed the frozen mixer")
    competence_after = snapshot_v19_reconstruction_state(system.competence_state)
    if any(
        not torch.equal(before, competence_after[name].detach())
        for name, before in competence_before.items()
    ):
        raise RuntimeError("V19 fit changed frozen competence state")
    _assert_source_lineage(system)
    changed = tuple(
        name
        for name in MUTABLE_PARAMETER_NAMES
        if not torch.equal(mutable_before[name], named[name].detach())
    )
    return {
        "stage": "paired_graph_context",
        "start_update": start_update,
        "optimizer_steps": len(stream_batches),
        "terminal_update": system.context_updates,
        "streams": sum(len(batch) for batch in stream_batches),
        "rows": sum(len(batch) for batch in stream_batches) * _ROWS_PER_STREAM,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": tuple(losses),
        "gradient_norms": tuple(gradient_norms),
        "objective_diagnostics": tuple(objective_diagnostics),
        "nonzero_gradient_parameter_names": tuple(nonzero_gradient_history),
        "gradient_reached_parameter_names": tuple(
            name for name in MUTABLE_PARAMETER_NAMES if name in reached
        ),
        "first_head_gradient_nonzero": first_head_gradient_nonzero,
        "first_upstream_gradients_exact_zero": first_upstream_gradients_exact_zero,
        "later_upstream_gradient_reached": later_upstream_gradient_reached,
        "changed_parameter_names": changed,
        "unchanged_allowed_parameter_names": tuple(
            name for name in MUTABLE_PARAMETER_NAMES if name not in changed
        ),
        "trainable_parameter_names": MUTABLE_PARAMETER_NAMES,
        "inherited_controller_exact": True,
        "mixer_exact": True,
        "competence_state_exact": True,
        "weight_decay": 0.0,
        "gradient_clip": _GRADIENT_CLIP,
        "common_production_matcher": "_paired_graph_evidence_read",
    }


def fit_v12_champion_paired_graph_context(
    system: V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    """Execute the single fixed 512-update V19 training identity."""

    if system.context_updates != 0 or system.optimizer_state is not None:
        raise RuntimeError("the frozen V19 fit requires a fresh migrated system")
    plan = v12_champion_paired_graph_context_plan()
    batches = v12._relation_credit_stream_batches(
        plan["commitments"], plan["training_seed_batches"]
    )
    result = _fit_paired_graph_batches(system, batches)
    if (
        result["optimizer_steps"] != _CONTEXT_UPDATES
        or result["streams"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE
        or result["rows"] != _CONTEXT_UPDATES * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM
        or not result["later_upstream_gradient_reached"]
        or set(result["gradient_reached_parameter_names"]) != set(MUTABLE_PARAMETER_NAMES)
    ):
        raise RuntimeError("V19 fixed fit did not reach its declared mechanism")
    return {
        **result,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan["plan_digest"],
        "terminal_system_digest": paired_graph_system_digest(system),
    }


def _credit_rows_metrics(
    row_groups: Sequence[Sequence[V19PairedGraphCreditRow]],
) -> dict[str, object]:
    rows = tuple(row for group in row_groups for row in group)
    supported_rows = 0
    informative_rows = 0
    unique_rows = 0
    unique_top_one = 0
    full_top_one = 0
    full_valid_mass = 0.0
    valid_real_mass = 0.0
    all_real_mass = 0.0
    margins = []
    relation_signature = []
    stream_supported_counts = []
    for group in row_groups:
        stream_supported = 0
        for row in group:
            valid = row.valid_mask
            count = int(valid.sum().item())
            slots = valid.numel()
            relation_signature.append(
                (
                    tuple(bool(value) for value in valid.tolist()),
                    tuple(float(value) for value in row.slot_positive_margins.tolist()),
                    tuple(float(value) for value in row.slot_negative_margins.tolist()),
                )
            )
            if count == 0:
                continue
            supported_rows += 1
            stream_supported += 1
            valid_weights = row.context_weights.masked_select(valid)
            invalid_weights = row.context_weights.masked_select(~valid)
            valid_max = valid_weights.max()
            invalid_max = (
                invalid_weights.max()
                if invalid_weights.numel()
                else row.context_null_weight.new_tensor(-torch.inf)
            )
            if bool((valid_max > torch.maximum(invalid_max, row.context_null_weight)).item()):
                full_top_one += 1
            full_valid_mass += float(valid_weights.sum().item())
            if count == slots:
                continue
            informative_rows += 1
            valid_real_mass += float(valid_weights.sum().item())
            all_real_mass += float(row.context_weights.sum().item())
            margin = torch.log(valid_max) - torch.log(invalid_max)
            margins.append(float(margin.item()))
            if count == 1:
                unique_rows += 1
                if bool((valid_max > invalid_max).item()):
                    unique_top_one += 1
        stream_supported_counts.append(stream_supported)
    # Retain V12's relation-gate meaning: a stream qualifies when at least
    # three of its four public rows have a valid relation witness.
    qualifying_streams = sum(value >= 3 for value in stream_supported_counts)
    return {
        "rows": len(rows),
        "supported_rows": supported_rows,
        "informative_rows": informative_rows,
        "unique_valid_rows": unique_rows,
        "unique_valid_top_one": unique_top_one,
        "real_normalized_valid_mass": (
            valid_real_mass / all_real_mass if all_real_mass > 0.0 else 0.0
        ),
        "valid_real_mass_numerator": valid_real_mass,
        "all_real_mass_denominator": all_real_mass,
        "mean_informative_log_weight_margin": (
            sum(margins) / len(margins) if margins else -math.inf
        ),
        "margin_sum": sum(margins),
        "margin_count": len(margins),
        "full_supported_top_one_fraction": (
            full_top_one / supported_rows if supported_rows else 0.0
        ),
        "full_supported_valid_set_mass": (
            full_valid_mass / supported_rows if supported_rows else 0.0
        ),
        "full_top_one_count": full_top_one,
        "full_valid_mass_sum": full_valid_mass,
        "qualifying_streams": qualifying_streams,
        "stream_supported_counts": tuple(stream_supported_counts),
        "relation_signature": tuple(relation_signature),
    }


def _evaluate_streams(
    controller: V12ChampionPairedGraphContextController,
    streams: Sequence[v12.SoftwarePipelineStream],
) -> dict[str, object]:
    with torch.no_grad():
        rows = tuple(public_paired_graph_credit_rows(controller, stream) for stream in streams)
    return _credit_rows_metrics(rows)


def _aggregate_metric_arms(arms: Sequence[Mapping[str, object]]) -> dict[str, object]:
    supported = sum(int(arm["supported_rows"]) for arm in arms)
    informative = sum(int(arm["informative_rows"]) for arm in arms)
    unique = sum(int(arm["unique_valid_rows"]) for arm in arms)
    real_numerator = sum(float(arm["valid_real_mass_numerator"]) for arm in arms)
    real_denominator = sum(float(arm["all_real_mass_denominator"]) for arm in arms)
    margin_sum = sum(float(arm["margin_sum"]) for arm in arms)
    margin_count = sum(int(arm["margin_count"]) for arm in arms)
    full_mass = sum(float(arm["full_valid_mass_sum"]) for arm in arms)
    return {
        "supported_rows": supported,
        "informative_rows": informative,
        "unique_valid_rows": unique,
        "unique_valid_top_one": sum(int(arm["unique_valid_top_one"]) for arm in arms),
        "real_normalized_valid_mass": (
            real_numerator / real_denominator if real_denominator > 0.0 else 0.0
        ),
        "mean_informative_log_weight_margin": (
            margin_sum / margin_count if margin_count else -math.inf
        ),
        "full_supported_top_one_fraction": (
            sum(int(arm["full_top_one_count"]) for arm in arms) / supported
            if supported
            else 0.0
        ),
        "full_supported_valid_set_mass": full_mass / supported if supported else 0.0,
        "qualifying_streams": sum(int(arm["qualifying_streams"]) for arm in arms),
    }


def evaluate_v12_champion_paired_graph_context(
    system: V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    """Evaluate four frozen panels and same-checkpoint causal lesions."""

    _assert_source_lineage(system)
    if system.context_updates != _CONTEXT_UPDATES:
        raise RuntimeError("V19 evaluation requires the complete fixed fit")
    controller = system.controller
    plan = v12_champion_paired_graph_context_plan()
    panel_streams = tuple(
        v12._relation_credit_panel_streams(plan["commitments"], seed_pairs)
        for seed_pairs in plan["panel_seed_pairs"]
    )
    learned_panels = tuple(_evaluate_streams(controller, streams) for streams in panel_streams)
    lesion_panels: dict[str, tuple[dict[str, object], ...]] = {}
    for lesion in (
        "zero_residual",
        "uniform_cross_graph_attention",
        "mismatch_zero",
    ):
        with controller.paired_graph_lesion(lesion):
            lesion_panels[lesion] = tuple(
                _evaluate_streams(controller, streams) for streams in panel_streams
            )
    zero_panels = lesion_panels["zero_residual"]
    relation_exact = all(
        learned["relation_signature"] == zero["relation_signature"]
        for learned, zero in zip(learned_panels, zero_panels, strict=True)
    )
    panel_deltas = []
    for learned, zero in zip(learned_panels, zero_panels, strict=True):
        top_delta = int(learned["unique_valid_top_one"]) - int(zero["unique_valid_top_one"])
        mass_delta = float(learned["real_normalized_valid_mass"]) - float(
            zero["real_normalized_valid_mass"]
        )
        recurrent = top_delta >= 0 and mass_delta >= 0.0 and (
            top_delta >= 1 or mass_delta >= 0.01
        )
        nonregressed = (
            top_delta >= -_MAX_PANEL_TOP_ONE_REGRESSION
            and mass_delta >= -_MAX_PANEL_MASS_REGRESSION
        )
        panel_deltas.append(
            {
                "unique_valid_top_one": top_delta,
                "real_normalized_valid_mass": mass_delta,
                "positive_recurrence": recurrent,
                "nonregressed": nonregressed,
            }
        )
    aggregates = {
        "learned": _aggregate_metric_arms(learned_panels),
        "zero_residual": _aggregate_metric_arms(zero_panels),
        "uniform_cross_graph_attention": _aggregate_metric_arms(
            lesion_panels["uniform_cross_graph_attention"]
        ),
        "mismatch_zero": _aggregate_metric_arms(lesion_panels["mismatch_zero"]),
    }
    learned = aggregates["learned"]
    zero = aggregates["zero_residual"]
    causal = {
        "unique_valid_top_one": int(learned["unique_valid_top_one"])
        - int(zero["unique_valid_top_one"]),
        "real_normalized_valid_mass": float(learned["real_normalized_valid_mass"])
        - float(zero["real_normalized_valid_mass"]),
        "mean_informative_log_weight_margin": float(
            learned["mean_informative_log_weight_margin"]
        )
        - float(zero["mean_informative_log_weight_margin"]),
    }
    attribution: dict[str, dict[str, object]] = {}
    for lesion in ("uniform_cross_graph_attention", "mismatch_zero"):
        arm = aggregates[lesion]
        margin_removed = float(learned["mean_informative_log_weight_margin"]) - float(
            arm["mean_informative_log_weight_margin"]
        )
        top_removed = int(learned["unique_valid_top_one"]) - int(
            arm["unique_valid_top_one"]
        )
        mass_removed = float(learned["real_normalized_valid_mass"]) - float(
            arm["real_normalized_valid_mass"]
        )
        passed = (
            margin_removed
            >= _ATTRIBUTION_MARGIN_FRACTION
            * float(causal["mean_informative_log_weight_margin"])
            and (
                top_removed >= _ATTRIBUTION_TOP_ONE_REMOVAL
                or mass_removed >= _ATTRIBUTION_MASS_REMOVAL
            )
        )
        attribution[lesion] = {
            "margin_removed": margin_removed,
            "unique_valid_top_one_removed": top_removed,
            "real_normalized_valid_mass_removed": mass_removed,
            "passed": passed,
        }
    component_supported = (
        sum(value["positive_recurrence"] is True for value in panel_deltas)
        >= _REQUIRED_IMPROVED_PANELS
        and all(value["nonregressed"] is True for value in panel_deltas)
        and causal["unique_valid_top_one"] >= _CAUSAL_TOP_ONE_GAIN
        and causal["real_normalized_valid_mass"]
        >= _CAUSAL_REAL_NORMALIZED_MASS_GAIN
        and float(learned["mean_informative_log_weight_margin"]) > 0.0
        and causal["mean_informative_log_weight_margin"] >= _CAUSAL_MARGIN_GAIN
        and relation_exact
    )
    attribution_supported = all(value["passed"] is True for value in attribution.values())
    full_advance = (
        component_supported
        and attribution_supported
        and int(learned["supported_rows"]) >= _RELATION_SUPPORTED_ROWS_THRESHOLD
        and int(learned["qualifying_streams"]) >= _RELATION_QUALIFYING_STREAMS_THRESHOLD
        and float(learned["full_supported_top_one_fraction"])
        >= _CONTEXT_TOP_ONE_THRESHOLD
        and float(learned["full_supported_valid_set_mass"])
        >= _CONTEXT_MASS_THRESHOLD
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan["plan_digest"],
        "panels": tuple(
            {
                "learned": learned_panel,
                "zero_residual": zero_panel,
                "uniform_cross_graph_attention": uniform_panel,
                "mismatch_zero": mismatch_panel,
                "causal_delta": delta,
            }
            for learned_panel, zero_panel, uniform_panel, mismatch_panel, delta in zip(
                learned_panels,
                zero_panels,
                lesion_panels["uniform_cross_graph_attention"],
                lesion_panels["mismatch_zero"],
                panel_deltas,
                strict=True,
            )
        ),
        "aggregate": aggregates,
        "causal_delta": causal,
        "attribution": attribution,
        "relation_exact_under_primary_lesion": relation_exact,
        "component_supported": component_supported,
        "attribution_supported": attribution_supported,
        "full_v12_replacement": full_advance,
        "classification": (
            "FULL_V12_REPLACEMENT"
            if full_advance
            else "PAIRED_GRAPH_COMPONENT_SUPPORTED"
            if component_supported
            else "PAIRED_GRAPH_CONTEXT_NOT_SUPPORTED"
        ),
        "terminal_system_digest": paired_graph_system_digest(system),
    }


def _checkpoint_config() -> dict[str, object]:
    return {
        "width": _WIDTH,
        "hidden_width": _HIDDEN_WIDTH,
        "maximum_graph_nodes": _MAX_GRAPH_NODES,
        "node_encoder_seed": _NODE_ENCODER_SEED,
        "graph_update_seed": _GRAPH_UPDATE_SEED,
        "pair_scorer_seed": _PAIR_SCORER_SEED,
        "cross_attention_rounds": 1,
        "context_temperature": _CONTEXT_TEMPERATURE,
        "residual_bound": _RESIDUAL_BOUND,
        "mutable_parameter_names": MUTABLE_PARAMETER_NAMES,
    }


def _checkpoint_payload(
    system: V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    _assert_source_lineage(system)
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "plan_digest": v12_champion_paired_graph_context_plan()["plan_digest"],
        "source": asdict(system.source),
        "profile": asdict(system.controller.profile),
        "config": _checkpoint_config(),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in system.controller.state_dict().items()
        },
        "controller_digest": v12.software_pipeline_model_digest(system.controller),
        "mutable_digest": paired_graph_mutable_digest(system.controller),
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
            for name, value in snapshot_v19_reconstruction_state(
                system.competence_state
            ).items()
        },
        "competence_digest": v19_reconstruction_state_digest(
            system.competence_state
        ),
        "context_updates": system.context_updates,
        "optimizer_state": copy.deepcopy(system.optimizer_state),
        "optimizer_digest": paired_graph_optimizer_digest(system.optimizer_state),
        "parameter_report": paired_graph_parameter_report(
            system.controller, system.mixer
        ),
        "system_digest": paired_graph_system_digest(system),
    }


def save_v12_champion_paired_graph_context_checkpoint(
    path: str | Path,
    system: V12ChampionPairedGraphContextSystem,
) -> None:
    torch.save(_checkpoint_payload(system), Path(path))


def load_v12_champion_paired_graph_context_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V12ChampionPairedGraphContextSystem:
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
        raise RuntimeError("V19 checkpoint fields are invalid")
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["protocol_id"] != PROTOCOL_ID
        or payload["plan_digest"]
        != v12_champion_paired_graph_context_plan()["plan_digest"]
        or payload["config"] != _checkpoint_config()
        or payload["source"] != asdict(_expected_source_binding())
    ):
        raise RuntimeError("V19 checkpoint identity is invalid")
    profile = v12.SoftwarePipelineRunProfile(**payload["profile"])
    if v12.SOFTWARE_PIPELINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("V19 checkpoint profile is not registered")
    cpu_rng = torch.get_rng_state()
    cuda_rng = _cuda_rng_snapshot(device)
    try:
        controller = V12ChampionPairedGraphContextController(profile).to(device)
        controller.load_state_dict(payload["model_state"], strict=True)
        mixer_config = payload["mixer_config"]
        if not isinstance(mixer_config, dict) or set(mixer_config) != {
            "feature_count",
            "hidden_width",
            "anchor_weight",
        }:
            raise RuntimeError("V19 mixer config is invalid")
        mixer = v12.AnonymousConflictMixer(**mixer_config).to(device)
        mixer.load_state_dict(payload["mixer_state"], strict=True)
        for parameter in mixer.parameters():
            parameter.requires_grad_(False)
    finally:
        torch.set_rng_state(cpu_rng)
        _restore_cuda_rng_snapshot(cuda_rng)
    try:
        state = restore_v19_reconstruction_state(payload["competence_state"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("V19 checkpoint competence state is invalid") from error
    updates = payload["context_updates"]
    optimizer_state = payload["optimizer_state"]
    if type(updates) is not int or not 0 <= updates <= _CONTEXT_UPDATES:
        raise RuntimeError("V19 checkpoint update count is invalid")
    if updates == 0:
        if optimizer_state is not None:
            raise RuntimeError("fresh V19 checkpoint unexpectedly has optimizer state")
    elif not isinstance(optimizer_state, Mapping):
        raise RuntimeError("learned V19 checkpoint lost optimizer state")
    else:
        _validate_optimizer_state(optimizer_state, controller, expected_steps=updates)
    system = V12ChampionPairedGraphContextSystem(
        controller=controller,
        mixer=mixer,
        competence_state=state,
        source=_expected_source_binding(),
        context_updates=updates,
        optimizer_state=copy.deepcopy(optimizer_state),
    )
    if (
        v12.software_pipeline_model_digest(controller) != payload["controller_digest"]
        or paired_graph_mutable_digest(controller) != payload["mutable_digest"]
        or inherited_v12_controller_digest(controller)
        != payload["inherited_controller_digest"]
        or v12.anonymous_conflict_mixer_digest(mixer) != payload["mixer_digest"]
        or v19_reconstruction_state_digest(state) != payload["competence_digest"]
        or paired_graph_optimizer_digest(system.optimizer_state)
        != payload["optimizer_digest"]
        or paired_graph_parameter_report(controller, mixer)
        != payload["parameter_report"]
        or paired_graph_system_digest(system) != payload["system_digest"]
    ):
        raise RuntimeError("V19 checkpoint digest or report mismatch")
    _assert_source_lineage(system)
    controller.eval()
    mixer.eval()
    return system


__all__ = [
    "CHECKPOINT_VERSION",
    "FROZEN_DEPENDENCY_HASHES",
    "MUTABLE_PARAMETER_NAMES",
    "PROTOCOL_ID",
    "V12_CHECKPOINT_SHA256",
    "V12ChampionPairedGraphContextController",
    "V12ChampionPairedGraphContextSystem",
    "V19PairedGraphCreditRow",
    "V19SoftwareReconstructionState",
    "V19SoftwareTaskEncoding",
    "acquire_v19_public_pipeline_traces",
    "evaluate_v12_champion_paired_graph_context",
    "fit_v12_champion_paired_graph_context",
    "frozen_dependency_hashes",
    "load_v12_champion_paired_graph_context_checkpoint",
    "load_v12_champion_paired_graph_context_source",
    "paired_graph_mutable_digest",
    "paired_graph_parameter_report",
    "paired_graph_system_digest",
    "public_paired_graph_credit_rows",
    "restore_v19_reconstruction_state",
    "save_v12_champion_paired_graph_context_checkpoint",
    "snapshot_v19_reconstruction_state",
    "v12_champion_paired_graph_context_plan",
    "v19_reconstruction_state_digest",
]
