"""Fresh learned graph decoder over public candidate roles."""

from __future__ import annotations

import torch
from torch import nn

from .candidate_procedure_decoder import CandidateProcedureDecode
from .edge_aware_procedure_decoder import EdgeAwareActionTraceDecoder
from .structured_relational_encoder import PublicProcedureRelations


PUBLIC_ROLE_WIDTH = 12


def build_public_candidate_roles(
    graph: PublicProcedureRelations,
    *,
    maximum_actions: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tensorize public unary incidence without inferring a procedure."""

    if not isinstance(graph, PublicProcedureRelations):
        raise TypeError("graph must be PublicProcedureRelations")
    if len(graph.components) > maximum_actions:
        raise ValueError("public candidates exceed role tensor capacity")
    rows = torch.zeros(
        (1, maximum_actions, PUBLIC_ROLE_WIDTH), device=device, dtype=dtype
    )
    origin = frozenset(graph.origin)
    goal = frozenset(graph.goal)
    forbidden = frozenset(graph.forbidden)
    for index, component in enumerate(graph.components):
        reads = frozenset(component.reads)
        writes = frozenset(component.writes)
        incoming = sum(
            other.output_type == component.input_type
            for other in graph.components
            if other is not component
        )
        outgoing = sum(
            component.output_type == other.input_type
            for other in graph.components
            if other is not component
        )
        nodes = {value for edge in component.topology_edges for value in edge}
        rows[0, index] = rows.new_tensor(
            (
                bool(reads & origin),
                bool(writes & goal),
                bool(reads & forbidden),
                bool(writes & forbidden),
                incoming > 0,
                outgoing > 0,
                incoming / max(len(graph.components) - 1, 1),
                outgoing / max(len(graph.components) - 1, 1),
                len(reads) / max(len(origin | goal | forbidden | reads | writes), 1),
                len(writes) / max(len(origin | goal | forbidden | reads | writes), 1),
                len(nodes) / 16.0,
                len(component.topology_edges) / 32.0,
            )
        )
    return rows


class RoleAwareGraphProcedureDecoder(EdgeAwareActionTraceDecoder):
    """Inject public unary roles before learned edge-aware decoding.

    The role tensor is descriptive, not prescriptive: it records membership and
    incidence facts but never constructs a path or assigns an action target.
    All inherited decoder parameters are freshly initialized for this model.
    """

    def __init__(self, *, role_width: int, **config: int) -> None:
        if type(role_width) is not int or role_width <= 0:
            raise ValueError("role_width must be positive")
        super().__init__(**config)
        self.role_width = role_width
        self.role_projection = nn.Sequential(
            nn.LayerNorm(role_width),
            nn.Linear(role_width, self.content_width),
            nn.GELU(),
            nn.Linear(self.content_width, self.content_width, bias=False),
        )
        self.role_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        procedure_slots: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        role_features: torch.Tensor,
        include_roles: bool = True,
        **kwargs,
    ) -> CandidateProcedureDecode:
        if (
            role_features.shape
            != (*action_features.shape[:2], self.role_width)
            or role_features.device != action_features.device
            or role_features.dtype != action_features.dtype
            or not bool(torch.isfinite(role_features).all().item())
        ):
            raise ValueError("public role features are invalid")
        if type(include_roles) is not bool:
            raise TypeError("include_roles must be bool")
        if include_roles:
            action_features = action_features + torch.sigmoid(
                self.role_gate
            ) * self.role_projection(role_features)
        return super().forward(
            procedure_slots,
            action_features,
            action_mask,
            **kwargs,
        )


__all__ = [
    "PUBLIC_ROLE_WIDTH",
    "RoleAwareGraphProcedureDecoder",
    "build_public_candidate_roles",
]
