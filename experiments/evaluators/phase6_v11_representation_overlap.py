from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
import itertools
import json
import math
from types import MethodType, ModuleType

import torch


PROTOCOL_ID = "phase6.public-representation-overlap.v11-d2"
SUPPORTED = "REPRESENTATION_OVERLAP_INTERFERENCE_SUPPORTED"
NOT_SUPPORTED = "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED"

CELL_IDS = ("t0_s0", "t0_s1", "t1_s0", "t1_s1")
STAGE_IDS = ("fused_relation_code", "relation_comparator_hidden")
GRADIENT_GROUPS = (
    "shared_encoder",
    "global_pool",
    "incidence_branch",
    "shared_comparator",
)
EASY_STREAMS = (0, 2, 3)
HARD_STREAMS = (4, 5, 7)
INTERMEDIATE_STREAMS = (1, 6)
PERMUTATION_COUNT = math.factorial(8)
MAXIMUM_JSON_BYTES = 262_144

EXPECTED_FROZEN_IDENTITY = {
    "preserved_tree": "6d54d3fe66d7e27b30e550b65c94d3f82c22bb1f",
    "runner_blob": "cf7fe45fb31531435a2b51e4485ec3137d40ed4e",
    "runner_sha256": (
        "305A083B4A108E5CA3784BD8834DDA74CA813D6E06C25CE08F75C61AE39D0B01"
    ),
    "checkpoint_sha256": (
        "2CF650BA5C9B62F1205CBA7F096CF9A078B752E699B98584FBC436FE1F5F0694"
    ),
    "source_report_sha256": (
        "EFDAC9461F34BE20226F54B718FB7A6F29375F74D9EFAD293D847867E071AE43"
    ),
    "d1_report_sha256": (
        "3191E1D9962A11BCCE9E8664E315D13CADB2F893C02994B7AA8B25233DF142BA"
    ),
    "terminal_model_digest": (
        "sha256:3833c9a01e986d5d7206802969b909747e34a5136266b54c72546f28436d9581"
    ),
}

_ACTIVE_MARKER = "_v11_d2_representation_capture_active"
_FACTORIZED_METHOD = "_factorized_relation_embeddings"
_EVIDENCE_METHOD = "_relation_evidence_read"
_FORBIDDEN_SERIALIZED_KEYS = {
    "activation",
    "activations",
    "activation_rows",
    "raw_activation",
    "raw_activations",
    "component_name",
    "component_names",
    "graph_id",
    "graph_ids",
    "commitment",
    "commitments",
    "answer",
    "answers",
    "public_task",
    "public_tasks",
}


def validate_frozen_identity(identity: Mapping[str, object]) -> None:
    observed = {key: identity.get(key) for key in EXPECTED_FROZEN_IDENTITY}
    if observed != EXPECTED_FROZEN_IDENTITY:
        raise RuntimeError("V11-D2 frozen source or evidence identity changed")


def covariance_matrix(activations: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(activations, torch.Tensor)
        or activations.ndim != 2
        or activations.shape[0] < 2
        or activations.shape[1] < 2
        or not activations.is_floating_point()
        or not bool(torch.isfinite(activations).all().item())
    ):
        raise ValueError("activations must be a finite floating matrix")
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(values.shape[0] - 1, 1)
    if not bool(torch.isfinite(covariance).all().item()) or float(
        torch.linalg.vector_norm(covariance).item()
    ) == 0.0:
        raise ValueError("activation covariance has zero variance")
    return covariance


def frobenius_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.ndim != 2
        or left.shape != right.shape
        or left.shape[0] != left.shape[1]
        or not left.is_floating_point()
        or not right.is_floating_point()
    ):
        raise ValueError("overlap requires aligned square floating matrices")
    lhs = left.detach().to(device="cpu", dtype=torch.float64)
    rhs = right.detach().to(device="cpu", dtype=torch.float64)
    lhs_norm = torch.linalg.vector_norm(lhs)
    rhs_norm = torch.linalg.vector_norm(rhs)
    denominator = lhs_norm * rhs_norm
    if (
        not bool(torch.isfinite(lhs).all().item())
        or not bool(torch.isfinite(rhs).all().item())
        or float(denominator.item()) == 0.0
    ):
        raise ValueError("overlap is undefined for zero or non-finite covariance")
    value = float(((lhs * rhs).sum() / denominator).item())
    if not math.isfinite(value):
        raise ValueError("overlap is non-finite")
    return max(-1.0, min(1.0, value))


def activation_description(activations: torch.Tensor) -> dict[str, object]:
    covariance = covariance_matrix(activations)
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    width = values.shape[1]
    l1 = values.abs().sum(dim=1)
    l2 = torch.linalg.vector_norm(values, dim=1)
    root_width = math.sqrt(width)
    row_hoyer = torch.where(
        l2 > 0.0,
        (root_width - l1 / l2) / (root_width - 1.0),
        torch.zeros_like(l2),
    ).clamp(0.0, 1.0)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total.item()) == 0.0:
        raise ValueError("effective rank is undefined for zero covariance")
    probabilities = eigenvalues / total
    nonzero = probabilities > 0.0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    effective_rank = float(torch.exp(entropy).item())
    return {
        "rows": values.shape[0],
        "width": width,
        "mean_row_hoyer_sparsity": float(row_hoyer.mean().item()),
        "near_zero_fraction": float((values.abs() <= 1.0e-6).double().mean().item()),
        "covariance_effective_rank": effective_rank,
    }


def overlap_matrix(covariances: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(covariances) != 8:
        raise ValueError("representation overlap requires exactly eight streams")
    matrix = torch.empty((8, 8), dtype=torch.float64)
    for left in range(8):
        for right in range(left, 8):
            value = frobenius_overlap(covariances[left], covariances[right])
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _validate_symmetric_matrix(value: torch.Tensor, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (8, 8)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{label} must be a finite 8x8 floating matrix")
    matrix = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.allclose(matrix, matrix.transpose(0, 1), atol=1.0e-12, rtol=0.0):
        raise ValueError(f"{label} must be symmetric")
    return matrix


def _off_diagonal_pair_values(matrix: torch.Tensor) -> torch.Tensor:
    indices = torch.triu_indices(8, 8, offset=1)
    return matrix[indices[0], indices[1]]


def burden_statistic(
    overlap: torch.Tensor,
    gradient_cosine: torch.Tensor,
) -> float:
    overlap = _validate_symmetric_matrix(overlap, "overlap")
    gradient = _validate_symmetric_matrix(gradient_cosine, "gradient cosine")
    burden = overlap * (-gradient).clamp_min(0.0)
    return float(_off_diagonal_pair_values(burden).mean().item())


@lru_cache(maxsize=1)
def _permutation_indices() -> tuple[tuple[int, ...], ...]:
    permutations = tuple(itertools.permutations(range(8)))
    if len(permutations) != PERMUTATION_COUNT or any(
        len(permutation) != 8 for permutation in permutations
    ):
        raise AssertionError("exact 8! permutation grid changed")
    return permutations


def exact_relative_permutation_test(
    overlap: torch.Tensor,
    gradient_cosine: torch.Tensor,
    *,
    jointly_permute_both: bool = False,
) -> dict[str, object]:
    if type(jointly_permute_both) is not bool:
        raise TypeError("joint permutation flag must be bool")
    if jointly_permute_both:
        raise ValueError(
            "jointly permuting overlap and gradient matrices is an invalid no-op"
        )
    overlap = _validate_symmetric_matrix(overlap, "overlap")
    gradient = _validate_symmetric_matrix(gradient_cosine, "gradient cosine")
    permutations = _permutation_indices()
    pairs = tuple(itertools.combinations(range(8), 2))
    overlap_values = overlap.tolist()
    fixed_weights = (-gradient).clamp_min(0.0).tolist()

    def statistic(permutation: Sequence[int]) -> float:
        # Deliberately use the same ordered scalar reduction for the identity
        # and every null permutation.  Mixed tensor reductions can move exact
        # ties by one ULP and alter the permutation count.
        return sum(
            overlap_values[permutation[left]][permutation[right]]
            * fixed_weights[left][right]
            for left, right in pairs
        ) / len(pairs)

    observed = statistic(permutations[0])
    reference_observed = burden_statistic(overlap, gradient)
    if not math.isclose(observed, reference_observed, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("permutation identity statistic disagrees with burden reference")
    exceedances = sum(statistic(permutation) >= observed for permutation in permutations)
    return {
        "observed_mean_off_diagonal_burden": observed,
        "permutations": PERMUTATION_COUNT,
        "exceedances": exceedances,
        "p_value_one_sided": exceedances / PERMUTATION_COUNT,
        "permuted_matrix": "representation_overlap_only",
        "fixed_matrix": "negative_gradient_cosine",
    }


def _pair_mean(
    matrix: torch.Tensor,
    left: Sequence[int],
    right: Sequence[int],
    *,
    within: bool,
) -> float:
    if within:
        pairs = tuple(itertools.combinations(left, 2))
    else:
        pairs = tuple((i, j) for i in left for j in right)
    return sum(float(matrix[i, j].item()) for i, j in pairs) / len(pairs)


def group_overlap_summary(overlap: torch.Tensor) -> dict[str, object]:
    matrix = _validate_symmetric_matrix(overlap, "overlap")
    easy_hard = _pair_mean(matrix, EASY_STREAMS, HARD_STREAMS, within=False)
    easy_easy = _pair_mean(matrix, EASY_STREAMS, EASY_STREAMS, within=True)
    hard_hard = _pair_mean(matrix, HARD_STREAMS, HARD_STREAMS, within=True)
    return {
        "easy_hard_mean": easy_hard,
        "easy_easy_mean": easy_easy,
        "hard_hard_mean": hard_hard,
        "easy_hard_pair_count": 9,
        "easy_easy_pair_count": 3,
        "hard_hard_pair_count": 3,
        "easy_hard_exceeds_both_within_groups": (
            easy_hard > easy_easy and easy_hard > hard_hard
        ),
    }


def classify_interference(cell_results: Mapping[str, object]) -> str:
    required_surface_rerenders = []
    for cell_id in ("t0_s0", "t0_s1"):
        cell = cell_results.get(cell_id)
        if not isinstance(cell, Mapping):
            raise ValueError("classification requires both t0 cells")
        summary = cell.get("group_overlap_summary")
        tests = cell.get("gradient_alignment")
        if not isinstance(summary, Mapping) or not isinstance(tests, Mapping):
            raise ValueError("classification inputs are incomplete")
        significant = sum(
            isinstance(tests.get(group), Mapping)
            and float(tests[group]["observed_mean_off_diagonal_burden"]) > 0.0
            and float(tests[group]["p_value_one_sided"]) <= 0.05
            for group in GRADIENT_GROUPS
        )
        required_surface_rerenders.append(
            summary.get("easy_hard_exceeds_both_within_groups") is True
            and significant >= 3
        )
    return SUPPORTED if all(required_surface_rerenders) else NOT_SUPPORTED


def _same_contract_pair(source_module: ModuleType, components: Sequence[object]) -> tuple[int, int]:
    key_function = getattr(source_module, "_public_static_contract_fields", None)
    if not callable(key_function):
        raise RuntimeError("V11 source omitted public static contract fields")
    groups: dict[object, list[int]] = {}
    for index, component in enumerate(components):
        groups.setdefault(key_function(component), []).append(index)
    pairs = tuple(tuple(indices) for indices in groups.values() if len(indices) == 2)
    if len(pairs) != 1 or any(len(indices) > 2 for indices in groups.values()):
        raise RuntimeError("public support did not expose exactly one same-contract pair")
    return pairs[0]


@contextmanager
def _temporary_capture(
    source_module: ModuleType,
    controller: object,
) -> Iterator[dict[str, list[torch.Tensor]]]:
    controller_type = getattr(source_module, "SoftwarePipelineController", None)
    if not isinstance(controller_type, type) or type(controller) is not controller_type:
        raise TypeError("capture requires the exact supplied V11 controller type")
    if bool(getattr(controller, "__dict__", {}).get(_ACTIVE_MARKER, False)):
        raise RuntimeError("nested V11-D2 capture is forbidden")
    for name in (_FACTORIZED_METHOD, _EVIDENCE_METHOD):
        if name in controller.__dict__:
            raise RuntimeError("V11-D2 refuses a pre-existing instance override")

    original_factorized = getattr(controller, _FACTORIZED_METHOD)
    original_evidence = getattr(controller, _EVIDENCE_METHOD)
    relation_pairs: dict[int, tuple[int, int]] = {}
    active_pair: list[tuple[int, int] | None] = [None]
    captured: dict[str, list[torch.Tensor]] = {
        "fused_relation_code": [],
        "relation_comparator_hidden": [],
    }

    def factorized_wrapper(instance: object, components: Sequence[object], reference: torch.Tensor):
        if instance is not controller:
            raise RuntimeError("V11-D2 factorized wrapper escaped its controller")
        context_codes, relation_codes = original_factorized(components, reference)
        pair = _same_contract_pair(source_module, components)
        indices = torch.tensor(pair, device=relation_codes.device, dtype=torch.long)
        selected = relation_codes.index_select(0, indices)
        if selected.shape != (2, 32):
            raise RuntimeError("V11-D2 fused relation capture shape changed")
        captured["fused_relation_code"].append(selected.detach().cpu().clone())
        relation_pairs[id(relation_codes)] = pair
        return context_codes, relation_codes

    def evidence_wrapper(
        instance: object,
        query_context_codes: torch.Tensor,
        query_relation_codes: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_relations: torch.Tensor,
    ):
        if instance is not controller:
            raise RuntimeError("V11-D2 evidence wrapper escaped its controller")
        pair = relation_pairs.get(id(query_relation_codes))
        if pair is None or active_pair[0] is not None:
            raise RuntimeError("V11-D2 could not bind the public same-contract pair")
        active_pair[0] = pair
        try:
            return original_evidence(
                query_context_codes,
                query_relation_codes,
                stored_contexts,
                stored_relations,
            )
        finally:
            active_pair[0] = None

    def hidden_hook(_module: object, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        pair = active_pair[0]
        if pair is None:
            raise RuntimeError("V11-D2 comparator hook ran outside public evidence read")
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] != 64:
            raise RuntimeError("V11-D2 comparator hidden capture shape changed")
        indices = torch.tensor(pair, device=output.device, dtype=torch.long)
        selected = output.index_select(0, indices).reshape(-1, 64)
        captured["relation_comparator_hidden"].append(
            selected.detach().cpu().clone()
        )

    object.__setattr__(controller, _ACTIVE_MARKER, True)
    object.__setattr__(
        controller,
        _FACTORIZED_METHOD,
        MethodType(factorized_wrapper, controller),
    )
    object.__setattr__(
        controller,
        _EVIDENCE_METHOD,
        MethodType(evidence_wrapper, controller),
    )
    hook = controller.relation_comparator[1].register_forward_hook(hidden_hook)
    try:
        yield captured
    finally:
        hook.remove()
        for name in (_FACTORIZED_METHOD, _EVIDENCE_METHOD, _ACTIVE_MARKER):
            if name in controller.__dict__:
                object.__delattr__(controller, name)


def capture_stream_representations(
    source_module: ModuleType,
    controller: object,
    stream: object,
) -> tuple[dict[str, torch.Tensor], int]:
    public_rows = getattr(source_module, "public_relation_credit_rows", None)
    if not callable(public_rows):
        raise RuntimeError("V11 source omitted public_relation_credit_rows")
    with _temporary_capture(source_module, controller) as captured:
        with torch.no_grad():
            rows = public_rows(controller, stream)
    matrices = {}
    for stage, chunks in captured.items():
        if not chunks:
            raise RuntimeError(f"V11-D2 captured no {stage} activations")
        matrices[stage] = torch.cat(chunks, dim=0).to(torch.float64)
    if matrices["fused_relation_code"].shape != (8, 32):
        raise RuntimeError("V11-D2 fused relation row count changed")
    if matrices["relation_comparator_hidden"].shape != (48, 64):
        raise RuntimeError("V11-D2 comparator hidden row count changed")
    if len(rows) != 4:
        raise RuntimeError("V11-D2 public support row count changed")
    return matrices, len(rows)


def _gradient_matrices(d1_report: Mapping[str, object], cell_id: str) -> dict[str, torch.Tensor]:
    try:
        groups = d1_report["cells"][cell_id]["gradient_diagnostic"]["groups"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("V11-D1 report omitted gradient geometry") from error
    matrices = {}
    for group in GRADIENT_GROUPS:
        try:
            value = torch.tensor(groups[group]["cosine_matrix"], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"V11-D1 report omitted {group} cosine matrix") from error
        matrices[group] = _validate_symmetric_matrix(value, group)
    return matrices


def _validate_public_inputs(
    source_module: ModuleType,
    source_report: Mapping[str, object],
    d1_report: Mapping[str, object],
    streams_by_cell: Mapping[str, Sequence[object]],
) -> None:
    try:
        after_relation = source_report["panels"]["after_relation"]
        source_pattern = after_relation["supported_rows_per_stream"]
        identity = d1_report["identity"]
        d1_cells = d1_report["cells"]
        declared_commitments = tuple(identity["anonymous_commitments"])
    except (KeyError, TypeError) as error:
        raise RuntimeError("V11 source reports omitted frozen public identity") from error
    if list(source_pattern) != [4, 2, 4, 4, 2, 2, 3, 0]:
        raise RuntimeError("V11 after_relation support pattern changed")
    if tuple(streams_by_cell) != CELL_IDS or len(declared_commitments) != 8:
        raise RuntimeError("V11-D2 public cell identity changed")
    expected_commitments = tuple(source_module.software_pipeline_mechanism_partition("train")[:8])
    if declared_commitments != expected_commitments:
        raise RuntimeError("V11-D1 commitments changed")
    for cell_id in CELL_IDS:
        streams = streams_by_cell[cell_id]
        seed_pairs = identity["seed_grid"][cell_id]["seed_pairs"]
        if len(streams) != 8 or len(seed_pairs) != 8:
            raise RuntimeError("V11-D2 cell lost a public stream")
        if tuple(stream.mechanism_commitment for stream in streams) != declared_commitments:
            raise RuntimeError("V11-D2 reconstructed commitments changed")
        panel = d1_cells[cell_id]["fused_panel"]
        if panel["rows"] != 32 or panel["streams"] != 8:
            raise RuntimeError("V11-D1 public row counts changed")
        if cell_id in ("t0_s0", "t0_s1") and list(
            panel["supported_rows_per_stream"]
        ) != [4, 2, 4, 4, 2, 2, 3, 2]:
            raise RuntimeError("V11-D1 t0 support recurrence changed")
        _gradient_matrices(d1_report, cell_id)


def evaluate_v11_representation_overlap(
    source_module: ModuleType,
    controller: object,
    source_report: Mapping[str, object],
    d1_report: Mapping[str, object],
    streams_by_cell: Mapping[str, Sequence[object]],
    frozen_identity: Mapping[str, object],
) -> dict[str, object]:
    validate_frozen_identity(frozen_identity)
    if not isinstance(source_module, ModuleType):
        raise TypeError("source_module must be the loaded exact V11 module")
    controller_type = getattr(source_module, "SoftwarePipelineController", None)
    if not isinstance(controller_type, type) or type(controller) is not controller_type:
        raise TypeError("controller must have the exact V11 source type")
    if torch.get_num_threads() != 1:
        raise RuntimeError("V11-D2 requires exactly one torch thread")
    if any(parameter.device.type != "cpu" for parameter in controller.parameters()):
        raise RuntimeError("V11-D2 requires the CPU controller")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V11-D2 refuses pre-existing parameter gradients")
    model_digest = source_module.software_pipeline_model_digest(controller)
    if model_digest != EXPECTED_FROZEN_IDENTITY["terminal_model_digest"]:
        raise RuntimeError("V11-D2 terminal controller digest changed")
    _validate_public_inputs(source_module, source_report, d1_report, streams_by_cell)

    captured: dict[str, dict[str, torch.Tensor]] = {stage: {} for stage in STAGE_IDS}
    descriptions: dict[str, dict[str, dict[str, object]]] = {
        stage: {} for stage in STAGE_IDS
    }
    total_rows = 0
    total_streams = 0
    for cell_id in CELL_IDS:
        stage_streams: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGE_IDS}
        for stream in streams_by_cell[cell_id]:
            matrices, row_count = capture_stream_representations(
                source_module,
                controller,
                stream,
            )
            total_rows += row_count
            total_streams += 1
            for stage in STAGE_IDS:
                stage_streams[stage].append(matrices[stage])
        for stage in STAGE_IDS:
            covariances = tuple(covariance_matrix(value) for value in stage_streams[stage])
            matrix = overlap_matrix(covariances)
            captured[stage][cell_id] = matrix
            reverse_delta = float(
                (
                    matrix
                    - overlap_matrix(
                        tuple(
                            covariance_matrix(value.flip(0))
                            for value in stage_streams[stage]
                        )
                    )
                )
                .abs()
                .amax()
                .item()
            )
            if reverse_delta > 1.0e-12:
                raise RuntimeError("V11-D2 sample-order invariance control failed")
            descriptions[stage][cell_id] = {
                "streams": [activation_description(value) for value in stage_streams[stage]],
                "reverse_sample_order_max_abs_delta": reverse_delta,
            }

    if total_streams != 32 or total_rows != 128:
        raise RuntimeError("V11-D2 exposure bounds changed")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V11-D2 accumulated a parameter gradient")
    after_digest = source_module.software_pipeline_model_digest(controller)
    if after_digest != model_digest:
        raise RuntimeError("V11-D2 changed the terminal controller")

    stages: dict[str, object] = {}
    for stage in STAGE_IDS:
        cells: dict[str, object] = {}
        for cell_id in CELL_IDS:
            matrix = captured[stage][cell_id]
            gradients = _gradient_matrices(d1_report, cell_id)
            cells[cell_id] = {
                "overlap_matrix": matrix.tolist(),
                "descriptive": descriptions[stage][cell_id],
                "group_overlap_summary": group_overlap_summary(matrix),
                "gradient_alignment": {
                    group: exact_relative_permutation_test(matrix, gradients[group])
                    for group in GRADIENT_GROUPS
                },
            }
        rerender_delta = float(
            (captured[stage]["t0_s0"] - captured[stage]["t0_s1"])
            .abs()
            .amax()
            .item()
        )
        if rerender_delta > 1.0e-6:
            raise RuntimeError("V11-D2 t0 surface rerender control failed")
        stages[stage] = {
            "cells": cells,
            "t0_surface_rerender_max_abs_delta": rerender_delta,
            "t0_surface_rerender_within_1e_6": rerender_delta <= 1.0e-6,
        }

    comparator_cells = stages["relation_comparator_hidden"]["cells"]
    classification = classify_interference(comparator_cells)
    return {
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "frozen_identity": dict(EXPECTED_FROZEN_IDENTITY),
        "integrity": {
            "terminal_model_digest_before": model_digest,
            "terminal_model_digest_after": after_digest,
            "model_exactly_unchanged": after_digest == model_digest,
            "parameters_with_accumulated_grad": [],
        },
        "public_exposure": {
            "cells": 4,
            "streams": total_streams,
            "support_rows": total_rows,
            "commitment_count": 8,
            "development_final_control_or_query_access": False,
        },
        "groups": {
            "easy_stream_indices": list(EASY_STREAMS),
            "hard_stream_indices": list(HARD_STREAMS),
            "intermediate_stream_indices": list(INTERMEDIATE_STREAMS),
        },
        "stages": stages,
        "classification_rule": {
            "primary_stage": "relation_comparator_hidden",
            "required_t0_surface_rerender_cells": ["t0_s0", "t0_s1"],
            "surface_rerenders_are_determinism_control_not_replication": True,
            "raw_easy_hard_overlap_exceeds_both_within_groups": True,
            "minimum_significant_gradient_groups_per_cell": 3,
            "maximum_p_value_one_sided": 0.05,
            "topology_transfer_cells_descriptive_only": True,
        },
        "bounds": {
            "permutations_per_test": PERMUTATION_COUNT,
            "optimizer_creations": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "checkpoint_writes": 0,
            "maximum_json_bytes": MAXIMUM_JSON_BYTES,
        },
    }


def _reject_forbidden_serialized_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_SERIALIZED_KEYS:
                raise ValueError(f"forbidden raw diagnostic field: {key}")
            _reject_forbidden_serialized_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_serialized_keys(item)


def serialize_bounded_report(
    report: Mapping[str, object],
    *,
    maximum_bytes: int = MAXIMUM_JSON_BYTES,
) -> bytes:
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer")
    _reject_forbidden_serialized_keys(report)
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(payload) > maximum_bytes:
        raise ValueError("V11-D2 terminal JSON exceeds its byte ceiling")
    return payload


__all__ = [
    "PROTOCOL_ID",
    "SUPPORTED",
    "NOT_SUPPORTED",
    "PERMUTATION_COUNT",
    "MAXIMUM_JSON_BYTES",
    "EXPECTED_FROZEN_IDENTITY",
    "covariance_matrix",
    "frobenius_overlap",
    "activation_description",
    "overlap_matrix",
    "burden_statistic",
    "exact_relative_permutation_test",
    "group_overlap_summary",
    "classify_interference",
    "capture_stream_representations",
    "evaluate_v11_representation_overlap",
    "serialize_bounded_report",
    "validate_frozen_identity",
]
