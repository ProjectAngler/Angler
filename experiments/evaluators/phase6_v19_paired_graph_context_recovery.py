from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import math
from types import MethodType

import torch

from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-v12-champion-paired-graph-context-eval-recovery.v19r1"

_RAW_DUPLICATE_LOGIT_DIFFERENCE_CEILING = 1.0e-6
_WRAPPED_METHOD_NAME = "_paired_graph_context_logits"
_ACTIVE_MARKER = "_v19_evaluation_recovery_wrapper_active"


def _terminal_state_identity(
    system: v19.V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    return {
        "context_updates": system.context_updates,
        "system_digest": v19.paired_graph_system_digest(system),
        "mutable_digest": v19.paired_graph_mutable_digest(system.controller),
        "optimizer_digest": v19.paired_graph_optimizer_digest(
            system.optimizer_state
        ),
    }


def _same_query_row(
    left: int,
    right: int,
    query_context_codes: torch.Tensor,
    query_graphs: torch.Tensor,
    query_masks: torch.Tensor,
) -> bool:
    return (
        torch.equal(query_context_codes[left], query_context_codes[right])
        and torch.equal(query_graphs[left], query_graphs[right])
        and torch.equal(query_masks[left], query_masks[right])
    )


@contextmanager
def _temporary_zero_residual_projection(
    controller: v19.V12ChampionPairedGraphContextController,
    audit: dict[str, object],
) -> Iterator[None]:
    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V19 recovery requires the exact frozen V19 controller type")
    if bool(controller.__dict__.get(_ACTIVE_MARKER, False)):
        raise RuntimeError("nested V19 evaluation-recovery wrappers are forbidden")
    if _WRAPPED_METHOD_NAME in controller.__dict__:
        raise RuntimeError("V19 recovery refuses a pre-existing instance method override")

    original = controller._paired_graph_context_logits
    if (
        getattr(original, "__self__", None) is not controller
        or getattr(original, "__func__", None)
        is not v19.V12ChampionPairedGraphContextController._paired_graph_context_logits
    ):
        raise RuntimeError("V19 recovery did not resolve the frozen context matcher")

    def wrapped(
        instance: v19.V12ChampionPairedGraphContextController,
        query_context_codes: torch.Tensor,
        query_graphs: torch.Tensor,
        query_masks: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_graphs: torch.Tensor,
        stored_masks: torch.Tensor,
    ) -> torch.Tensor:
        if instance is not controller:
            raise RuntimeError("V19 recovery wrapper escaped its controller instance")
        raw = original(
            query_context_codes,
            query_graphs,
            query_masks,
            stored_contexts,
            stored_graphs,
            stored_masks,
        )
        audit["frozen_method_calls"] = int(audit["frozen_method_calls"]) + 1
        if instance._paired_graph_lesion != "zero_residual":
            audit["delegated_nonzero_lesion_calls"] = (
                int(audit["delegated_nonzero_lesion_calls"]) + 1
            )
            return raw
        audit["zero_residual_calls"] = int(audit["zero_residual_calls"]) + 1
        if (
            not isinstance(raw, torch.Tensor)
            or raw.ndim != 2
            or raw.shape[0] != query_context_codes.shape[0]
            or query_graphs.shape[0] != raw.shape[0]
            or query_masks.shape[0] != raw.shape[0]
        ):
            raise RuntimeError("V19 recovery received misaligned frozen logits")
        if not bool(torch.isfinite(raw).all().item()):
            raise RuntimeError("V19 recovery received non-finite frozen logits")

        row_count = raw.shape[0]
        audit["zero_residual_rows"] = int(audit["zero_residual_rows"]) + row_count
        representatives: list[int] = []
        group_sizes: dict[int, int] = {}
        duplicate_rows: list[tuple[int, int]] = []
        for row in range(row_count):
            representative = next(
                (
                    candidate
                    for candidate in representatives
                    if _same_query_row(
                        row,
                        candidate,
                        query_context_codes,
                        query_graphs,
                        query_masks,
                    )
                ),
                None,
            )
            if representative is None:
                representatives.append(row)
                group_sizes[row] = 1
            else:
                group_sizes[representative] += 1
                duplicate_rows.append((row, representative))

        audit["representative_rows"] = (
            int(audit["representative_rows"]) + len(representatives)
        )
        audit["duplicate_groups"] = int(audit["duplicate_groups"]) + sum(
            size > 1 for size in group_sizes.values()
        )
        if not duplicate_rows:
            return raw

        projected = raw.clone()
        for row, representative in duplicate_rows:
            difference = raw[row].to(torch.float64) - raw[representative].to(
                torch.float64
            )
            maximum = (
                float(difference.abs().amax().item())
                if difference.numel()
                else 0.0
            )
            if (
                not math.isfinite(maximum)
                or maximum > _RAW_DUPLICATE_LOGIT_DIFFERENCE_CEILING
            ):
                raise RuntimeError(
                    "exact duplicate V19 query rows exceeded the frozen raw-logit "
                    "difference ceiling"
                )
            audit["maximum_raw_duplicate_logit_difference"] = max(
                float(audit["maximum_raw_duplicate_logit_difference"]),
                maximum,
            )
            projected[row].copy_(raw[representative])
        audit["duplicate_rows_projected"] = (
            int(audit["duplicate_rows_projected"]) + len(duplicate_rows)
        )
        return projected

    object.__setattr__(controller, _ACTIVE_MARKER, True)
    object.__setattr__(
        controller,
        _WRAPPED_METHOD_NAME,
        MethodType(wrapped, controller),
    )
    try:
        yield
    finally:
        if _WRAPPED_METHOD_NAME in controller.__dict__:
            object.__delattr__(controller, _WRAPPED_METHOD_NAME)
        if _ACTIVE_MARKER in controller.__dict__:
            object.__delattr__(controller, _ACTIVE_MARKER)


def evaluate_v19_paired_graph_context_recovery(
    system: v19.V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    """Run the frozen V19 causal evaluator with one duplicate-row projection."""

    if type(system) is not v19.V12ChampionPairedGraphContextSystem:
        raise TypeError("V19 recovery requires the exact frozen V19 system type")
    before = _terminal_state_identity(system)
    audit: dict[str, object] = {
        "projection": "exact-query-duplicate-first-row.v1",
        "grouping_fields": (
            "query_context_code",
            "raw_graph_adjacency",
            "node_mask",
        ),
        "raw_duplicate_logit_difference_ceiling": (
            _RAW_DUPLICATE_LOGIT_DIFFERENCE_CEILING
        ),
        "frozen_method_calls": 0,
        "zero_residual_calls": 0,
        "delegated_nonzero_lesion_calls": 0,
        "zero_residual_rows": 0,
        "representative_rows": 0,
        "duplicate_groups": 0,
        "duplicate_rows_projected": 0,
        "maximum_raw_duplicate_logit_difference": 0.0,
    }
    try:
        with _temporary_zero_residual_projection(system.controller, audit):
            evaluation = v19.evaluate_v12_champion_paired_graph_context(system)
    finally:
        after = _terminal_state_identity(system)
        if after != before:
            raise RuntimeError("V19 recovery evaluation changed learned system state")
    audit["wrapper_restored"] = (
        _WRAPPED_METHOD_NAME not in system.controller.__dict__
        and _ACTIVE_MARKER not in system.controller.__dict__
    )
    audit["terminal_state_identity"] = before
    return {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": v19.PROTOCOL_ID,
        "evaluation": evaluation,
        "projection_audit": audit,
    }


__all__ = ["PROTOCOL_ID", "evaluate_v19_paired_graph_context_recovery"]
