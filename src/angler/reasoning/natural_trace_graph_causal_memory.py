"""Natural-language trace graphs with outcome-factorized causal memory.

Only the public ``Step N:`` framing is interpreted.  Step meanings remain
detached foundation embeddings and every structural operation is generic path
incidence.  The memory keeps outcome-blind public-step sidecars in the same
least-used slots as V10.  A bounded, uniform whole-graph comparison augments
content-address logits before the inherited top-two read; pooled candidate
semantics are unchanged, and the graph never performs cross-node attention or
infers which procedure is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F

from .content_addressed_causal_memory import (
    ContentAddressedCausalMemoryOutput,
    ContentAddressedCausalMemoryState,
)
from .dnc_allocated_factorized_causal_memory import (
    DncAllocatedFactorizedCausalMemoryCore,
)
from .situated_feedback import _normalize_masked_scores


_STEP_MARKER = re.compile(r"(?:^|(?<=\s))Step ([1-9][0-9]*):[ \t]*")


def parse_step_trace(text: str) -> tuple[str, ...]:
    """Split a public trace without interpreting any sentence or solution.

    The grammar is deliberately small: a trace is a contiguous, one-based
    series of ``Step N:`` fields.  Returned values are the raw, trimmed public
    sentence spans.  No role vocabulary or outcome information is consulted.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("step trace must be a non-empty string")
    matches = tuple(_STEP_MARKER.finditer(text))
    if not matches or matches[0].start() != 0:
        raise ValueError("step trace must begin with Step 1:")
    steps: list[str] = []
    for expected, match in enumerate(matches, start=1):
        if int(match.group(1)) != expected:
            raise ValueError("step trace ordinals must be contiguous and one-based")
        end = matches[expected].start() if expected < len(matches) else len(text)
        sentence = text[match.end() : end].strip()
        if not sentence:
            raise ValueError("step trace sentences must be non-empty")
        steps.append(sentence)
    return tuple(steps)


class NaturalLanguageTraceGraphEncoder(nn.Module):
    """Encode public step embeddings with exactly two shared path rounds."""

    reasoning_rounds = 2

    def __init__(self, *, step_width: int, relational_width: int = 32) -> None:
        super().__init__()
        for name, value in (("step_width", step_width), ("relational_width", relational_width)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.step_width = step_width
        self.relational_width = relational_width
        self.step_projection = nn.Linear(step_width, relational_width, bias=False)
        # Normalized ordinal, first endpoint, and last endpoint are public,
        # task-neutral context.  No absolute mechanism or family identity is
        # represented.
        self.ordinal_endpoint_projection = nn.Linear(3, relational_width, bias=False)
        self.previous_message = nn.Linear(relational_width, relational_width, bias=False)
        self.self_message = nn.Linear(relational_width, relational_width, bias=False)
        self.next_message = nn.Linear(relational_width, relational_width, bias=False)
        self.message_update = nn.Sequential(
            nn.LayerNorm(relational_width * 2),
            nn.Linear(relational_width * 2, relational_width),
            nn.SiLU(),
        )
        self.output_norm = nn.LayerNorm(relational_width)
        self.pooling = nn.Sequential(
            nn.LayerNorm(relational_width * 4),
            nn.Linear(relational_width * 4, relational_width),
            nn.SiLU(),
        )

    def _validate(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        allow_empty: bool,
    ) -> None:
        reference = self.step_projection.weight
        if step_features.ndim != 4 or step_features.shape[-1] != self.step_width:
            raise ValueError("step_features must be [batch, candidates, steps, step_width]")
        if step_mask.dtype is not torch.bool or step_mask.shape != step_features.shape[:-1]:
            raise ValueError("step_mask must be boolean [batch, candidates, steps]")
        if step_features.device != reference.device or step_features.dtype != reference.dtype:
            raise ValueError("step_features must match the encoder device and dtype")
        if step_mask.device != reference.device:
            raise ValueError("step_mask must match the encoder device")
        if step_features.shape[2] == 0:
            raise ValueError("step tensors must expose at least one step column")
        lengths = step_mask.sum(dim=-1)
        if not allow_empty and bool((lengths == 0).any().item()):
            raise ValueError("every visible trace must contain at least one step")
        # A path is represented by a left-aligned prefix.  Rejecting holes
        # avoids silently inventing adjacency across missing public nodes.
        seen_padding = (~step_mask).cumsum(dim=-1) > 0
        if bool((step_mask & seen_padding).any().item()):
            raise ValueError("step_mask must be a left-aligned contiguous prefix")
        visible = step_features[step_mask]
        if visible.numel() and not bool(torch.isfinite(visible).all().item()):
            raise ValueError("visible step features must be finite")

    def _encode(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool,
        allow_empty: bool,
    ) -> torch.Tensor:
        self._validate(step_features, step_mask, allow_empty=allow_empty)
        batch, candidates, maximum_steps, _ = step_features.shape
        mask = step_mask.unsqueeze(-1)
        detached = step_features.detach().masked_fill(~mask, 0.0)
        lengths = step_mask.sum(dim=-1)
        denominator = (lengths - 1).clamp_min(1).to(dtype=detached.dtype)
        ordinal = torch.arange(
            maximum_steps,
            device=detached.device,
            dtype=detached.dtype,
        ).view(1, 1, maximum_steps)
        ordinal = ordinal / denominator.unsqueeze(-1)
        first = torch.zeros_like(step_mask)
        first[..., 0] = lengths > 0
        last = torch.zeros_like(step_mask)
        last_index = (lengths - 1).clamp_min(0).unsqueeze(-1)
        last.scatter_(-1, last_index, lengths.gt(0).unsqueeze(-1))
        context = torch.stack(
            (ordinal, first.to(detached.dtype), last.to(detached.dtype)),
            dim=-1,
        ).masked_fill(~mask, 0.0)

        states = self.step_projection(detached) + self.ordinal_endpoint_projection(context)
        states = states.masked_fill(~mask, 0.0)
        for _ in range(self.reasoning_rounds):
            previous = torch.zeros_like(states)
            following = torch.zeros_like(states)
            if include_direction and maximum_steps > 1:
                previous[..., 1:, :] = states[..., :-1, :]
                following[..., :-1, :] = states[..., 1:, :]
            messages = self.self_message(states)
            if include_direction:
                messages = (
                    messages
                    + self.previous_message(previous)
                    + self.next_message(following)
                )
            update = self.message_update(torch.cat((states, messages), dim=-1))
            states = self.output_norm(states + update).masked_fill(~mask, 0.0)

        safe_lengths = lengths.clamp_min(1)
        first_state = states[..., 0, :]
        gather_index = (safe_lengths - 1).view(batch, candidates, 1, 1).expand(
            -1, -1, 1, self.relational_width
        )
        last_state = states.gather(2, gather_index).squeeze(2)
        mean_state = states.sum(dim=2) / safe_lengths.to(states.dtype).unsqueeze(-1)
        maximum_state = states.masked_fill(~mask, -torch.inf).amax(dim=2)
        empty = lengths == 0
        maximum_state = maximum_state.masked_fill(empty.unsqueeze(-1), 0.0)
        first_state = first_state.masked_fill(empty.unsqueeze(-1), 0.0)
        last_state = last_state.masked_fill(empty.unsqueeze(-1), 0.0)
        pooled = self.pooling(
            torch.cat((first_state, last_state, mean_state, maximum_state), dim=-1)
        )
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)

    def forward(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
    ) -> torch.Tensor:
        return self._encode(
            step_features,
            step_mask,
            include_direction=include_direction,
            allow_empty=False,
        )

    def encode_sidecars(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
    ) -> torch.Tensor:
        """Encode fixed slot sidecars, permitting unused all-padding slots."""

        return self._encode(
            step_features,
            step_mask,
            include_direction=include_direction,
            allow_empty=True,
        )


@dataclass(frozen=True, slots=True)
class NaturalTraceGraphMemoryState(ContentAddressedCausalMemoryState):
    graph_step_features: torch.Tensor
    graph_step_mask: torch.Tensor

    @property
    def bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.keys,
                self.values,
                self.usage,
                self.graph_step_features,
                self.graph_step_mask,
            )
        )

    def detached_clone(self) -> "NaturalTraceGraphMemoryState":
        return NaturalTraceGraphMemoryState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            usage=self.usage.detach().clone(),
            step=self.step,
            graph_step_features=self.graph_step_features.detach().clone(),
            graph_step_mask=self.graph_step_mask.detach().clone(),
        )


@dataclass(frozen=True, slots=True)
class NaturalTraceGraphCausalMemoryOutput(ContentAddressedCausalMemoryOutput):
    candidate_step_features: torch.Tensor
    candidate_step_mask: torch.Tensor
    relational_features: torch.Tensor
    graph_address_logits: torch.Tensor


class NaturalTraceGraphCausalMemoryCore(DncAllocatedFactorizedCausalMemoryCore):
    """V10 memory plus learned public-trace graph addresses and sidecars."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        step_width: int | None = None,
        relational_width: int = 32,
        maximum_trace_steps: int = 8,
        maximum_graph_logit: float = 2.0,
        **kwargs,
    ) -> None:
        if type(maximum_trace_steps) is not int or maximum_trace_steps <= 0:
            raise ValueError("maximum_trace_steps must be a positive integer")
        if not isinstance(maximum_graph_logit, (int, float)) or maximum_graph_logit <= 0:
            raise ValueError("maximum_graph_logit must be positive")
        super().__init__(
            content_width=content_width,
            temporal_width=temporal_width,
            **kwargs,
        )
        resolved_step_width = content_width if step_width is None else step_width
        self.maximum_trace_steps = maximum_trace_steps
        self.step_width = resolved_step_width
        self.relational_width = relational_width
        self.maximum_graph_logit = float(maximum_graph_logit)
        self.trace_graph_encoder = NaturalLanguageTraceGraphEncoder(
            step_width=resolved_step_width,
            relational_width=relational_width,
        )
        # Symmetric, graph-wide comparison only.  There is no Q@S attention,
        # node correspondence, role vocabulary, or extra propagation round.
        self.graph_comparator = nn.Sequential(
            nn.LayerNorm(relational_width * 2),
            nn.Linear(relational_width * 2, relational_width),
            nn.SiLU(),
            nn.Linear(relational_width, 1, bias=False),
        )

    def initial_memory_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NaturalTraceGraphMemoryState:
        base = super().initial_memory_state(device=device, dtype=dtype)
        return NaturalTraceGraphMemoryState(
            keys=base.keys,
            values=base.values,
            usage=base.usage,
            step=base.step,
            graph_step_features=torch.zeros(
                self.memory_slots,
                self.maximum_trace_steps,
                self.step_width,
                device=base.keys.device,
                dtype=base.keys.dtype,
            ),
            graph_step_mask=torch.zeros(
                self.memory_slots,
                self.maximum_trace_steps,
                device=base.keys.device,
                dtype=torch.bool,
            ),
        )

    initial_plastic_state = initial_memory_state

    def _validate_state(self, state: ContentAddressedCausalMemoryState) -> None:
        super()._validate_state(state)
        if not isinstance(state, NaturalTraceGraphMemoryState):
            raise TypeError("memory_state must include natural-trace graph sidecars")
        expected_features = (
            self.memory_slots,
            self.maximum_trace_steps,
            self.step_width,
        )
        if state.graph_step_features.shape != expected_features:
            raise ValueError("memory graph sidecar features have the wrong shape")
        if state.graph_step_mask.shape != expected_features[:-1] or state.graph_step_mask.dtype is not torch.bool:
            raise ValueError("memory graph sidecar mask has the wrong shape or dtype")
        if state.graph_step_features.device != state.keys.device or state.graph_step_features.dtype != state.keys.dtype:
            raise ValueError("memory graph sidecars must match key device and dtype")
        if state.graph_step_mask.device != state.keys.device:
            raise ValueError("memory graph sidecar mask must match key device")
        visible = state.graph_step_features[state.graph_step_mask]
        if visible.numel() and not bool(torch.isfinite(visible).all().item()):
            raise ValueError("memory graph sidecars must be finite")

    def encode_candidate_traces(
        self,
        candidate_step_features: torch.Tensor,
        candidate_step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
    ) -> torch.Tensor:
        return self.trace_graph_encoder(
            candidate_step_features,
            candidate_step_mask,
            include_direction=include_direction,
        )

    def _graph_logits(
        self,
        relational_features: torch.Tensor,
        state: NaturalTraceGraphMemoryState,
        *,
        include_direction: bool,
        include_graph_address: bool,
    ) -> torch.Tensor:
        shape = (*relational_features.shape[:-1], self.memory_slots)
        if not include_graph_address:
            return relational_features.new_zeros(shape)
        stored = self.trace_graph_encoder.encode_sidecars(
            state.graph_step_features.unsqueeze(0),
            state.graph_step_mask.unsqueeze(0),
            include_direction=include_direction,
        ).squeeze(0)
        current = relational_features.unsqueeze(-2)
        stored = stored.view(1, 1, self.memory_slots, self.relational_width)
        symmetric = torch.cat(
            ((current - stored).abs(), current * stored),
            dim=-1,
        )
        logits = self.maximum_graph_logit * torch.tanh(
            self.graph_comparator(symmetric).squeeze(-1)
        )
        occupied = state.graph_step_mask.any(dim=-1).view(1, 1, self.memory_slots)
        return logits.masked_fill(~occupied, 0.0)

    def _graph_augmented_read_weights(
        self,
        read_keys: torch.Tensor,
        state: NaturalTraceGraphMemoryState,
        strengths: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> torch.Tensor:
        query_norm = F.normalize(read_keys, dim=-1, eps=1.0e-6)
        key_norm = F.normalize(state.keys, dim=-1, eps=1.0e-6)
        logits = torch.einsum("...r,sr->...s", query_norm, key_norm)
        logits = logits * strengths.unsqueeze(-1)
        logits = logits + torch.log(state.usage.clamp_min(1.0e-4)) + graph_logits
        dense = torch.softmax(logits, dim=-1)
        return self._top_k_preserve_mass(dense, self.routing_k)

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        *,
        candidate_step_features: torch.Tensor,
        candidate_step_mask: torch.Tensor,
        memory_state: NaturalTraceGraphMemoryState | None = None,
        include_direction: bool = True,
        include_graph_address: bool = True,
    ) -> NaturalTraceGraphCausalMemoryOutput:
        self._validate_inputs(
            query_features,
            candidate_features,
            temporal_features,
            candidate_outcomes,
            candidate_mask,
            base_scores,
        )
        if candidate_step_features.shape[:2] != candidate_features.shape[:2] or candidate_step_mask.shape[:2] != candidate_mask.shape:
            raise ValueError("candidate step tensors must align with candidate features")
        if candidate_step_features.shape[2] > self.maximum_trace_steps:
            raise ValueError("candidate trace exceeds maximum_trace_steps")
        state = self.initial_memory_state() if memory_state is None else memory_state
        self._validate_state(state)

        relational = self.encode_candidate_traces(
            candidate_step_features,
            candidate_step_mask,
            include_direction=include_direction,
        )
        query = self.query_projection(query_features.detach())
        # Keep V9/V10 pooled semantics exact.  V11 intervenes only in the
        # read-address logits through the graph sidecar comparison below.
        candidates = self.candidate_projection(candidate_features.detach())
        temporal_hidden = self.temporal_projection(temporal_features.detach())
        expanded_query = query.unsqueeze(1).expand_as(candidates)
        semantic = self.semantic_network(
            torch.cat(
                (candidates, expanded_query, candidates * expanded_query, temporal_hidden),
                dim=-1,
            )
        )

        address_input = torch.cat((semantic, temporal_hidden), dim=-1)
        read_keys = self.read_key_network(address_input)
        read_strengths = F.softplus(self.read_strength(address_input).squeeze(-1)) + 1.0
        graph_logits = self._graph_logits(
            relational,
            state,
            include_direction=include_direction,
            include_graph_address=include_graph_address,
        )
        read_weights = self._graph_augmented_read_weights(
            read_keys,
            state,
            read_strengths,
            graph_logits,
        ).masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        read_values = torch.einsum("bns,sr->bnr", read_weights, state.values)

        score_features = torch.cat((semantic, read_values, semantic * read_values), dim=-1)
        feature_gates = torch.sigmoid(self.neuromodulator(score_features)).masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )
        gated_semantic = semantic * feature_gates
        raw_residual = self.score_network(
            torch.cat(
                (gated_semantic, read_values, gated_semantic * read_values),
                dim=-1,
            )
        ).squeeze(-1)
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)

        controls = self.address_controls(address_input)
        write_gates = torch.sigmoid(controls[..., : self.rank])
        visible = candidate_mask.to(dtype=write_gates.dtype)
        write_gates = write_gates * visible.unsqueeze(-1)
        write_strengths = torch.sigmoid(controls[..., self.rank]) * visible
        content_strengths = (
            F.softplus(controls[..., self.rank + 2]) + 1.0
        ) * visible
        write_keys = read_keys * write_gates

        outcome = self._outcome_features(candidate_outcomes)
        semantic_payload = self.semantic_payload_projection(semantic)
        outcome_payload = self.outcome_payload_projection(outcome)
        bound_payload = semantic_payload * outcome_payload
        write_values = torch.tanh(self.writer(bound_payload)) * write_gates
        erase_gates = write_gates

        allocation = self._allocation_weights(state.usage).view(1, 1, -1)
        write_weights = write_strengths.unsqueeze(-1) * allocation
        return NaturalTraceGraphCausalMemoryOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
            pair_features=semantic,
            temporal_hidden=temporal_hidden,
            read_keys=read_keys,
            read_weights=read_weights,
            read_values=read_values,
            feature_gates=feature_gates,
            write_gates=write_gates,
            write_keys=write_keys,
            write_values=write_values,
            erase_gates=erase_gates,
            write_strengths=write_strengths,
            allocation_mix=torch.zeros_like(write_strengths),
            content_strengths=content_strengths,
            write_weights=write_weights,
            candidate_step_features=candidate_step_features.detach(),
            candidate_step_mask=candidate_step_mask.detach(),
            relational_features=relational,
            graph_address_logits=graph_logits,
        )

    def apply_mixed_outcome_update(
        self,
        state: NaturalTraceGraphMemoryState,
        output: NaturalTraceGraphCausalMemoryOutput,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> NaturalTraceGraphMemoryState:
        self._validate_state(state)
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        if candidate_mask.shape[0] != 1:
            raise ValueError("one persistent memory accepts one chronological event at a time")
        if int(candidate_mask.sum().item()) != 1:
            raise ValueError("a trace-sidecar update requires exactly one visible candidate")
        self._validate_output(output, candidate_mask)

        keys, values, usage = state.keys, state.values, state.usage
        graph_features = state.graph_step_features.clone()
        graph_mask = state.graph_step_mask.clone()
        visible_candidate = int(torch.nonzero(candidate_mask[0], as_tuple=False).item())
        for candidate in range(candidate_mask.shape[1]):
            slot_weights = output.write_weights[0, candidate]
            slot = slot_weights.unsqueeze(-1)
            write_key = output.write_keys[0, candidate].unsqueeze(0)
            write_value = output.write_values[0, candidate].unsqueeze(0)
            erase = output.erase_gates[0, candidate].unsqueeze(0)
            keys = (1.0 - slot) * keys + slot * write_key
            erased_values = values * (1.0 - slot * erase)
            values = (1.0 - slot) * erased_values + slot * write_value
            usage = usage + slot_weights * (1.0 - usage)

        winner = int(output.write_weights[0, visible_candidate].argmax().item())
        trace_length = output.candidate_step_features.shape[2]
        graph_features[winner].zero_()
        graph_mask[winner].zero_()
        graph_features[winner, :trace_length] = output.candidate_step_features[
            0, visible_candidate
        ]
        graph_mask[winner, :trace_length] = output.candidate_step_mask[
            0, visible_candidate
        ]
        updated = NaturalTraceGraphMemoryState(
            keys=keys,
            values=values,
            usage=usage,
            step=state.step + 1,
            graph_step_features=graph_features,
            graph_step_mask=graph_mask,
        )
        self._validate_state(updated)
        return updated.detached_clone() if detach_state else updated


__all__ = [
    "NaturalLanguageTraceGraphEncoder",
    "NaturalTraceGraphCausalMemoryCore",
    "NaturalTraceGraphCausalMemoryOutput",
    "NaturalTraceGraphMemoryState",
    "parse_step_trace",
]
