"""A small, explicit self-referential fast-weight memory.

This module adapts the stateful weight-offset representation and the four
block delta update from Irie et al., *A Modern Self-Referential Weight Matrix
That Learns to Modify Itself* (ICML 2022).  The equations and initialization
were checked against the MIT-licensed ``IDSIA/automated-cl`` donor at commit
``3d7b53adb4b6b43acd82b9a381a2c631d0e59a5d``; none of its custom CUDA code is
used here.  It intentionally contains no task semantics: callers provide
generic vectors and carry the returned numeric state between encounters.

The slow parameters are the initial ``y``, ``q``, ``k``, and ``beta`` weight
blocks.  A :class:`SelfReferentialState` stores only offsets from those bases.
Consequently, applying an experience never mutates an ``nn.Parameter``.  The
same effective weights produce an output, modifier key, analyser query, and
four learning rates, then modify themselves through a rank-one delta rule.

``read`` is strictly read-only.  ``write`` applies one update for every input
token and returns a new state; it never changes the supplied state in place.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping

import torch
from torch import nn


_STATE_KEYS = ("delta_y", "delta_q", "delta_k", "delta_beta")
_STATE_DIGEST_DOMAIN = b"project-angler.self-referential-state.v1\x00"


@dataclass(frozen=True, slots=True)
class SelfReferentialState:
    """The sole persistent competence state: offsets from four slow bases."""

    delta_y: torch.Tensor
    delta_q: torch.Tensor
    delta_k: torch.Tensor
    delta_beta: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.delta_y.shape[0])

    def numel(self) -> int:
        """Return the fixed number of stored scalar values."""

        return sum(
            tensor.numel()
            for tensor in (
                self.delta_y,
                self.delta_q,
                self.delta_k,
                self.delta_beta,
            )
        )


def snapshot_self_referential_state(
    state: SelfReferentialState,
) -> dict[str, torch.Tensor]:
    """Copy a state exactly into detached tensors with no storage aliases."""

    _validate_state_structure(state)
    return {
        name: getattr(state, name).detach().clone()
        for name in _STATE_KEYS
    }


def restore_self_referential_state(
    snapshot: Mapping[str, torch.Tensor],
) -> SelfReferentialState:
    """Construct an independent state from an exact four-tensor snapshot."""

    if set(snapshot) != set(_STATE_KEYS):
        missing = sorted(set(_STATE_KEYS) - set(snapshot))
        extra = sorted(set(snapshot) - set(_STATE_KEYS))
        raise ValueError(
            f"state snapshot keys differ; missing={missing}, extra={extra}"
        )
    if any(not isinstance(snapshot[name], torch.Tensor) for name in _STATE_KEYS):
        raise TypeError("every state snapshot value must be a tensor")
    state = SelfReferentialState(
        **{
            name: snapshot[name].detach().clone()
            for name in _STATE_KEYS
        }
    )
    _validate_state_structure(state)
    return state


def detach_self_referential_state(
    state: SelfReferentialState,
) -> SelfReferentialState:
    """Truncate autograd history without resetting the numeric state."""

    _validate_state_structure(state)
    return SelfReferentialState(
        **{
            name: getattr(state, name).detach().clone()
            for name in _STATE_KEYS
        }
    )


def self_referential_state_digest(state: SelfReferentialState) -> str:
    """Return a deterministic identity for the four numeric state offsets."""

    _validate_state_structure(state)
    digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
    for name in _STATE_KEYS:
        tensor = getattr(state, name).detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", tensor.ndim))
        digest.update(struct.pack(f">{tensor.ndim}Q", *tensor.shape))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


class SelfReferentialMemory(nn.Module):
    """Multi-head self-referential matrix with explicit read/write operations.

    Tensors presented to :meth:`read` or :meth:`write` have shape
    ``[batch, length, width]``.  Internally, ``width`` is divided uniformly
    among ``heads``.  The state has four offset tensors with donor-compatible
    orientations:

    - ``delta_y``, ``delta_q``, ``delta_k``: ``[B, H, D, D]``;
    - ``delta_beta``: ``[B, H, D, 4]``.

    The penultimate dimension is the input/key dimension.  Thus a row vector
    ``x`` reads a block as ``x @ W``.  With ``H=8`` and ``D=64``, one stream
    stores 100,352 scalars regardless of how many events it has encountered.
    """

    def __init__(
        self,
        width: int,
        *,
        heads: int = 1,
        input_softmax: bool = False,
        beta_init: float = -1.0,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0:
            raise ValueError("width and heads must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not math.isfinite(beta_init):
            raise ValueError("beta_init must be finite")

        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.input_softmax = input_softmax

        matrix_shape = (1, heads, self.head_width, self.head_width)
        beta_shape = (1, heads, self.head_width, 4)
        self.base_y = nn.Parameter(torch.empty(matrix_shape))
        self.base_q = nn.Parameter(torch.empty(matrix_shape))
        self.base_k = nn.Parameter(torch.empty(matrix_shape))
        self.base_beta = nn.Parameter(torch.empty(beta_shape))
        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init: float = -1.0) -> None:
        """Initialize the slow bases as in the public SRWM donor layer."""

        if not math.isfinite(beta_init):
            raise ValueError("beta_init must be finite")
        standard_deviation = 1.0 / math.sqrt(self.head_width)
        query_deviation = 0.01 / math.sqrt(self.head_width)
        nn.init.normal_(self.base_y, mean=0.0, std=standard_deviation)
        nn.init.normal_(self.base_q, mean=0.0, std=query_deviation)
        nn.init.normal_(self.base_k, mean=0.0, std=standard_deviation)
        nn.init.normal_(
            self.base_beta,
            mean=beta_init,
            std=standard_deviation,
        )

    def state_numel(self, batch_size: int = 1) -> int:
        """Return state capacity, which is independent of stream length."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        per_head = 3 * self.head_width * self.head_width
        per_head += self.head_width * 4
        return batch_size * self.heads * per_head

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SelfReferentialState:
        """Create the zero-offset state for independent stream(s)."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        target_device = self.base_y.device if device is None else torch.device(device)
        target_dtype = self.base_y.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise ValueError("state dtype must be floating point")
        matrix_shape = (
            batch_size,
            self.heads,
            self.head_width,
            self.head_width,
        )
        beta_shape = (batch_size, self.heads, self.head_width, 4)
        return SelfReferentialState(
            delta_y=torch.zeros(
                matrix_shape,
                device=target_device,
                dtype=target_dtype,
            ),
            delta_q=torch.zeros(
                matrix_shape,
                device=target_device,
                dtype=target_dtype,
            ),
            delta_k=torch.zeros(
                matrix_shape,
                device=target_device,
                dtype=target_dtype,
            ),
            delta_beta=torch.zeros(
                beta_shape,
                device=target_device,
                dtype=target_dtype,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: SelfReferentialState,
    ) -> torch.Tensor:
        """Read without writing; equivalent to :meth:`read`."""

        return self.read(inputs, state)

    def read(
        self,
        inputs: torch.Tensor,
        state: SelfReferentialState,
    ) -> torch.Tensor:
        """Return current ``y`` values without changing ``state``."""

        head_inputs = self._validate_and_split_inputs(inputs, state)
        weights_y, _, _, _ = self._effective_weights(state)
        normalized_inputs = self._normalize_inputs(head_inputs)
        outputs = torch.einsum(
            "bhld,bhdo->bhlo",
            normalized_inputs,
            weights_y,
        )
        return self._merge_heads(outputs)

    def write(
        self,
        inputs: torch.Tensor,
        state: SelfReferentialState,
    ) -> tuple[torch.Tensor, SelfReferentialState]:
        """Apply one self-modification per token and return a new state.

        Each returned output is read from the effective weights immediately
        before that token's update, matching equations 5--8 of the donor
        formulation.  The state passed by the caller is never modified.
        """

        head_inputs = self._validate_and_split_inputs(inputs, state)
        weights_y, weights_q, weights_k, weights_beta = (
            self._effective_weights(state)
        )
        outputs: list[torch.Tensor] = []

        for position in range(head_inputs.shape[2]):
            event = self._normalize_inputs(head_inputs[:, :, position, :])
            output_y = torch.einsum("bhd,bhdo->bho", event, weights_y)
            query = torch.einsum("bhd,bhdo->bho", event, weights_q)
            key = torch.einsum("bhd,bhdo->bho", event, weights_k)
            learning_rates = torch.einsum(
                "bhd,bhdc->bhc",
                event,
                weights_beta,
            )

            query = torch.softmax(query, dim=-1)
            key = torch.softmax(key, dim=-1)

            y_from_query = torch.einsum(
                "bhd,bhdo->bho",
                query,
                weights_y,
            )
            y_from_key = torch.einsum(
                "bhd,bhdo->bho",
                key,
                weights_y,
            )
            q_from_query = torch.einsum(
                "bhd,bhdo->bho",
                query,
                weights_q,
            )
            q_from_key = torch.einsum(
                "bhd,bhdo->bho",
                key,
                weights_q,
            )
            k_from_query = torch.einsum(
                "bhd,bhdo->bho",
                query,
                weights_k,
            )
            k_from_key = torch.einsum(
                "bhd,bhdo->bho",
                key,
                weights_k,
            )
            beta_from_query = torch.einsum(
                "bhd,bhdc->bhc",
                query,
                weights_beta,
            )
            beta_from_key = torch.einsum(
                "bhd,bhdc->bhc",
                key,
                weights_beta,
            )

            rates = torch.sigmoid(learning_rates)
            weights_y = weights_y + rates[..., 0, None, None] * torch.einsum(
                "bhi,bho->bhio",
                key,
                y_from_query - y_from_key,
            )
            weights_q = weights_q + rates[..., 1, None, None] * torch.einsum(
                "bhi,bho->bhio",
                key,
                q_from_query - q_from_key,
            )
            weights_k = weights_k + rates[..., 2, None, None] * torch.einsum(
                "bhi,bho->bhio",
                key,
                k_from_query - k_from_key,
            )
            weights_beta = (
                weights_beta
                + rates[..., 3, None, None]
                * torch.einsum(
                    "bhi,bhc->bhic",
                    key,
                    beta_from_query - beta_from_key,
                )
            )
            outputs.append(output_y)

        new_state = SelfReferentialState(
            delta_y=weights_y - self.base_y,
            delta_q=weights_q - self.base_q,
            delta_k=weights_k - self.base_k,
            delta_beta=weights_beta - self.base_beta,
        )
        merged_outputs = self._merge_heads(torch.stack(outputs, dim=2))
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (
                merged_outputs,
                new_state.delta_y,
                new_state.delta_q,
                new_state.delta_k,
                new_state.delta_beta,
            )
        ):
            raise RuntimeError("self-referential write produced a non-finite value")
        return merged_outputs, new_state

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.input_softmax:
            return torch.softmax(inputs, dim=-1)
        return inputs

    def _effective_weights(
        self,
        state: SelfReferentialState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.base_y + state.delta_y,
            self.base_q + state.delta_q,
            self.base_k + state.delta_k,
            self.base_beta + state.delta_beta,
        )

    def _validate_and_split_inputs(
        self,
        inputs: torch.Tensor,
        state: SelfReferentialState,
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, length, width]")
        if inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
            raise ValueError("input batch and length must be positive")
        if inputs.shape[2] != self.width:
            raise ValueError(
                f"input width {inputs.shape[2]} does not match {self.width}"
            )
        if not inputs.is_floating_point():
            raise ValueError("inputs must use a floating-point dtype")
        if inputs.device != self.base_y.device or inputs.dtype != self.base_y.dtype:
            raise ValueError("inputs must match the memory parameter device and dtype")
        if not bool(torch.isfinite(inputs).all().item()):
            raise ValueError("inputs must be finite")

        self._validate_state(state, inputs.shape[0], inputs.device, inputs.dtype)
        return inputs.reshape(
            inputs.shape[0],
            inputs.shape[1],
            self.heads,
            self.head_width,
        ).permute(0, 2, 1, 3)

    def _validate_state(
        self,
        state: SelfReferentialState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        _validate_state_structure(state)
        matrix_shape = (
            batch_size,
            self.heads,
            self.head_width,
            self.head_width,
        )
        beta_shape = (batch_size, self.heads, self.head_width, 4)
        if any(
            tensor.shape != matrix_shape
            for tensor in (state.delta_y, state.delta_q, state.delta_k)
        ) or state.delta_beta.shape != beta_shape:
            raise ValueError("state topology does not match this memory")
        if any(
            tensor.device != device or tensor.dtype != dtype
            for tensor in (
                state.delta_y,
                state.delta_q,
                state.delta_k,
                state.delta_beta,
            )
        ):
            raise ValueError("state must match the input device and dtype")

    @staticmethod
    def _merge_heads(values: torch.Tensor) -> torch.Tensor:
        return values.permute(0, 2, 1, 3).reshape(
            values.shape[0],
            values.shape[2],
            values.shape[1] * values.shape[3],
        )


def _validate_state_structure(state: SelfReferentialState) -> None:
    if not isinstance(state, SelfReferentialState):
        raise TypeError("state must be a SelfReferentialState")
    matrices = (state.delta_y, state.delta_q, state.delta_k)
    tensors = (*matrices, state.delta_beta)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("every state field must be a tensor")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("every state tensor must be rank four")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("state tensors must use floating-point dtypes")
    if any(tensor.shape != matrices[0].shape for tensor in matrices[1:]):
        raise ValueError("y, q, and k state shapes must match")
    batch, heads, input_width, output_width = matrices[0].shape
    if batch <= 0 or heads <= 0 or input_width <= 0:
        raise ValueError("state dimensions must be positive")
    if input_width != output_width:
        raise ValueError("y, q, and k state matrices must be square")
    if state.delta_beta.shape != (batch, heads, input_width, 4):
        raise ValueError("beta state must have shape [batch, heads, width, 4]")
    reference = matrices[0]
    if any(
        tensor.device != reference.device or tensor.dtype != reference.dtype
        for tensor in tensors[1:]
    ):
        raise ValueError("all state tensors must share device and dtype")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("state tensors must be finite")


__all__ = [
    "SelfReferentialMemory",
    "SelfReferentialState",
    "detach_self_referential_state",
    "restore_self_referential_state",
    "self_referential_state_digest",
    "snapshot_self_referential_state",
]
