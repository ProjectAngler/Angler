"""Bounded feedback-time updates for one active PEFT LoRA state.

This module is deliberately narrower than an agent or reasoning policy.  The
caller supplies tokenized text that the model already generated and masks the
reflection/revision region with ``-100`` outside that region.  This module
only computes teacher-forced causal loss, applies bounded gradients to the
sole active LoRA adapter, and exposes an exact adapter rollback transaction.

It does not generate critiques, prescribe reasoning, inspect hidden answers,
select tools, or decide whether a candidate should be retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.runtime.qwen_peft import (
    adapter_tensor_digest,
    foundation_tensor_digest,
    validate_foundation_frozen,
)

_IGNORE_INDEX = -100


class BoundedUpdateError(RuntimeError):
    """Raised when an update violates its declared scope or budget."""


@dataclass(frozen=True, slots=True)
class TeacherForcedReflectionBatch:
    """Tokenized model output with loss enabled only for revision tokens.

    ``reflection_labels`` has the same shape as ``input_ids``.  Positions
    outside the model-generated reflection/revision text must be ``-100``;
    supervised positions contain their token IDs.  No answer or reasoning
    text is created or selected by this structure.
    """

    input_ids: torch.Tensor
    reflection_labels: torch.Tensor
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class LoraUpdateBudget:
    """Hard bounds for one candidate LoRA update."""

    max_steps: int
    max_input_tokens: int
    max_supervised_tokens: int
    learning_rate: float
    max_gradient_norm: float
    max_adapter_delta_norm: float

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_supervised_tokens <= 0:
            raise ValueError("max_supervised_tokens must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.max_gradient_norm)
            or self.max_gradient_norm <= 0.0
        ):
            raise ValueError("max_gradient_norm must be finite and positive")
        if (
            not math.isfinite(self.max_adapter_delta_norm)
            or self.max_adapter_delta_norm <= 0.0
        ):
            raise ValueError(
                "max_adapter_delta_norm must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class LoraUpdateStep:
    """Measured result of one bounded optimizer step."""

    step: int
    loss: float
    gradient_norm_before_clip: float
    adapter_delta_norm_after_step: float
    input_tokens: int
    supervised_tokens: int


@dataclass(frozen=True, slots=True)
class CandidateUpdateReceipt:
    """Identity and resource facts for an unresolved adapter candidate."""

    adapter_name: str
    parent_adapter_digest: str
    candidate_adapter_digest: str
    foundation_digest: str
    foundation_numel: int
    adapter_numel: int
    steps: tuple[LoraUpdateStep, ...]
    total_input_tokens: int
    total_supervised_tokens: int
    max_gradient_norm: float
    max_adapter_delta_norm: float
    candidate_adapter_delta_norm: float


@dataclass(frozen=True, slots=True)
class _FoundationParameterSentinel:
    """Cheap per-step guard complementing full transaction-boundary hashes."""

    name: str
    object_id: int
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    device: str
    version: int


@dataclass(frozen=True, slots=True)
class FinalizedUpdateReceipt:
    """Final state after an external decision retains or rejects a candidate."""

    candidate: CandidateUpdateReceipt
    disposition: Literal["candidate_retained", "rejected_rolled_back"]
    final_adapter_digest: str


class LoraUpdateCandidate:
    """An unresolved candidate with exact, adapter-only rollback material.

    The caller evaluates the candidate outside this module, then calls
    :meth:`retain` or :meth:`reject`.  Used as a context manager, an unresolved
    candidate is automatically rejected on exit, including exceptional exit.
    """

    __slots__ = (
        "_model",
        "_parent_tensors",
        "_foundation_digest",
        "_foundation_sentinel",
        "_receipt",
        "_resolved",
    )

    def __init__(
        self,
        model: nn.Module,
        parent_tensors: dict[str, torch.Tensor],
        foundation_digest: str,
        foundation_sentinel: tuple[_FoundationParameterSentinel, ...],
        receipt: CandidateUpdateReceipt,
    ) -> None:
        self._model = model
        self._parent_tensors: dict[str, torch.Tensor] | None = parent_tensors
        self._foundation_digest = foundation_digest
        self._foundation_sentinel = foundation_sentinel
        self._receipt = receipt
        self._resolved = False

    @property
    def receipt(self) -> CandidateUpdateReceipt:
        return self._receipt

    @property
    def resolved(self) -> bool:
        return self._resolved

    def retain(self) -> FinalizedUpdateReceipt:
        """Retain the externally accepted candidate without promoting it."""

        self._require_open()
        _verify_foundation_sentinel(
            self._model,
            self._foundation_sentinel,
            adapter_name=self._receipt.adapter_name,
        )
        if foundation_tensor_digest(self._model) != self._foundation_digest:
            self._rollback_after_integrity_failure()
            raise BoundedUpdateError(
                "foundation content changed before the external decision; "
                "the adapter was rolled back and the foundation must be reloaded"
            )
        current_digest = adapter_tensor_digest(
            self._model,
            adapter_name=self._receipt.adapter_name,
        )
        if current_digest != self._receipt.candidate_adapter_digest:
            self._rollback_after_integrity_failure()
            raise BoundedUpdateError(
                "candidate adapter changed before the external decision; "
                "the parent adapter was restored"
            )

        self._resolved = True
        self._parent_tensors = None
        return FinalizedUpdateReceipt(
            candidate=self._receipt,
            disposition="candidate_retained",
            final_adapter_digest=current_digest,
        )

    def reject(self) -> FinalizedUpdateReceipt:
        """Restore the exact parent adapter after external rejection."""

        self._require_open()
        parent_tensors = self._parent_tensors
        if parent_tensors is None:  # pragma: no cover - guarded by _require_open
            raise BoundedUpdateError("candidate has no rollback material")

        _restore_adapter_tensors(self._model, parent_tensors)
        _verify_foundation_sentinel(
            self._model,
            self._foundation_sentinel,
            adapter_name=self._receipt.adapter_name,
        )
        if foundation_tensor_digest(self._model) != self._foundation_digest:
            self._resolved = True
            self._parent_tensors = None
            raise BoundedUpdateError(
                "adapter rollback succeeded but foundation content changed; "
                "the foundation must be reloaded"
            )
        final_digest = adapter_tensor_digest(
            self._model,
            adapter_name=self._receipt.adapter_name,
        )
        if final_digest != self._receipt.parent_adapter_digest:
            raise BoundedUpdateError("exact adapter rollback verification failed")

        self._resolved = True
        self._parent_tensors = None
        return FinalizedUpdateReceipt(
            candidate=self._receipt,
            disposition="rejected_rolled_back",
            final_adapter_digest=final_digest,
        )

    def __enter__(self) -> LoraUpdateCandidate:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc_value, traceback
        if not self._resolved:
            self.reject()
        return False

    def _require_open(self) -> None:
        if self._resolved:
            raise BoundedUpdateError("candidate decision is already finalized")

    def _rollback_after_integrity_failure(self) -> None:
        parent_tensors = self._parent_tensors
        if parent_tensors is not None:
            _restore_adapter_tensors(self._model, parent_tensors)
        self._resolved = True
        self._parent_tensors = None


def teacher_forced_causal_loss(
    logits: torch.Tensor,
    reflection_labels: torch.Tensor,
) -> torch.Tensor:
    """Return next-token loss over only caller-marked revision positions."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if reflection_labels.ndim != 2:
        raise ValueError("reflection_labels must have shape [batch, sequence]")
    if logits.shape[:2] != reflection_labels.shape:
        raise ValueError("logits and reflection_labels shapes do not align")
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires a sequence of at least two tokens")
    if logits.shape[2] <= 0:
        raise ValueError("logits vocabulary must be non-empty")
    if reflection_labels.dtype != torch.long:
        raise ValueError("reflection_labels must use torch.long token IDs")
    if reflection_labels.device != logits.device:
        raise ValueError("logits and reflection_labels must share a device")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = reflection_labels[:, 1:].contiguous()
    supervised = shifted_labels.ne(_IGNORE_INDEX)
    if not bool(supervised.any().item()):
        raise ValueError("reflection_labels contain no causal target tokens")

    visible_labels = shifted_labels[supervised]
    if bool((visible_labels < 0).any().item()) or bool(
        (visible_labels >= logits.shape[2]).any().item()
    ):
        raise ValueError("reflection_labels contain an invalid token ID")

    return F.cross_entropy(
        shifted_logits.float().view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=_IGNORE_INDEX,
        reduction="mean",
    )


def propose_bounded_lora_update(
    model: nn.Module,
    batches: Sequence[TeacherForcedReflectionBatch],
    budget: LoraUpdateBudget,
    *,
    adapter_name: str | None = None,
) -> LoraUpdateCandidate:
    """Apply bounded AdamW steps to the sole active LoRA adapter.

    All batch and budget checks occur before the first mutation.  If forward,
    loss, backward, clipping, or optimizer execution fails, the parent adapter
    is restored and verified before the exception is surfaced.
    """

    inventory = validate_foundation_frozen(
        model,
        adapter_name=adapter_name,
        require_trainable_lora=True,
    )
    named_parameters = dict(model.named_parameters())
    foundation_parameters = tuple(
        named_parameters[record.name] for record in inventory.foundation
    )
    adapter_parameters = tuple(
        named_parameters[record.name] for record in inventory.lora
    )
    foundation_sentinel = _capture_foundation_sentinel(
        inventory.foundation,
        named_parameters,
    )

    if any(parameter.grad is not None for parameter in foundation_parameters):
        raise BoundedUpdateError(
            "foundation parameters contain pre-existing gradients"
        )
    batches = tuple(batches)
    if not batches:
        raise BoundedUpdateError("at least one reflection batch is required")
    if len(batches) > budget.max_steps:
        raise BoundedUpdateError(
            f"{len(batches)} steps exceed max_steps={budget.max_steps}"
        )

    batch_counts = tuple(_validate_batch(batch) for batch in batches)
    total_input_tokens = sum(counts[0] for counts in batch_counts)
    total_supervised_tokens = sum(counts[1] for counts in batch_counts)
    if total_input_tokens > budget.max_input_tokens:
        raise BoundedUpdateError(
            f"{total_input_tokens} input tokens exceed "
            f"max_input_tokens={budget.max_input_tokens}"
        )
    if total_supervised_tokens > budget.max_supervised_tokens:
        raise BoundedUpdateError(
            f"{total_supervised_tokens} supervised tokens exceed "
            f"max_supervised_tokens={budget.max_supervised_tokens}"
        )

    parent_digest = adapter_tensor_digest(
        model,
        adapter_name=inventory.adapter_name,
    )
    parent_foundation_digest = foundation_tensor_digest(model)
    parent_tensors = {
        record.name: named_parameters[record.name].detach().to("cpu").clone()
        for record in inventory.lora
    }
    prior_training_modes = tuple(
        (module, module.training) for module in model.modules()
    )
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=budget.learning_rate,
        weight_decay=0.0,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    adapter_parameter_ids = {id(parameter) for parameter in adapter_parameters}
    foundation_parameter_ids = {
        id(parameter) for parameter in foundation_parameters
    }
    if (
        optimizer_parameter_ids != adapter_parameter_ids
        or optimizer_parameter_ids & foundation_parameter_ids
    ):
        raise BoundedUpdateError(
            "optimizer parameter scope is not exactly the active adapter"
        )
    steps: list[LoraUpdateStep] = []
    candidate_delta_norm = 0.0

    try:
        model.train(True)
        for step_index, (batch, counts) in enumerate(
            zip(batches, batch_counts, strict=True),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            forward_kwargs: dict[str, torch.Tensor] = {
                "input_ids": batch.input_ids,
            }
            if batch.attention_mask is not None:
                forward_kwargs["attention_mask"] = batch.attention_mask

            outputs = model(**forward_kwargs)
            logits = _extract_logits(outputs)
            loss = teacher_forced_causal_loss(logits, batch.reflection_labels)
            if not bool(torch.isfinite(loss).item()):
                raise BoundedUpdateError("teacher-forced loss is not finite")

            loss.backward()
            if any(
                parameter.grad is not None
                for parameter in foundation_parameters
            ):
                raise BoundedUpdateError(
                    "a foundation parameter received a gradient"
                )
            if not any(
                parameter.grad is not None for parameter in adapter_parameters
            ):
                raise BoundedUpdateError("the active adapter received no gradient")

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter_parameters,
                max_norm=budget.max_gradient_norm,
                error_if_nonfinite=True,
            )
            gradient_norm_value = float(gradient_norm.detach().cpu().item())
            optimizer.step()

            if not all(
                bool(torch.isfinite(parameter.detach()).all().item())
                for parameter in adapter_parameters
            ):
                raise BoundedUpdateError("an adapter parameter is not finite")
            _verify_foundation_sentinel(
                model,
                foundation_sentinel,
                adapter_name=inventory.adapter_name,
            )
            candidate_delta_norm = _adapter_delta_norm(model, parent_tensors)
            if candidate_delta_norm > budget.max_adapter_delta_norm:
                raise BoundedUpdateError(
                    f"adapter delta norm {candidate_delta_norm:.8g} exceeds "
                    f"max_adapter_delta_norm={budget.max_adapter_delta_norm:.8g}"
                )
            steps.append(
                LoraUpdateStep(
                    step=step_index,
                    loss=float(loss.detach().cpu().item()),
                    gradient_norm_before_clip=gradient_norm_value,
                    adapter_delta_norm_after_step=candidate_delta_norm,
                    input_tokens=counts[0],
                    supervised_tokens=counts[1],
                )
            )

        candidate_digest = adapter_tensor_digest(
            model,
            adapter_name=inventory.adapter_name,
        )
        if candidate_digest == parent_digest:
            raise BoundedUpdateError("bounded update did not change adapter state")
        if foundation_tensor_digest(model) != parent_foundation_digest:
            raise BoundedUpdateError("foundation content changed during update")

        _clear_parameter_gradients(adapter_parameters)
        _restore_training_modes(prior_training_modes)
        receipt = CandidateUpdateReceipt(
            adapter_name=inventory.adapter_name,
            parent_adapter_digest=parent_digest,
            candidate_adapter_digest=candidate_digest,
            foundation_digest=parent_foundation_digest,
            foundation_numel=inventory.foundation_numel,
            adapter_numel=inventory.lora_numel,
            steps=tuple(steps),
            total_input_tokens=total_input_tokens,
            total_supervised_tokens=total_supervised_tokens,
            max_gradient_norm=budget.max_gradient_norm,
            max_adapter_delta_norm=budget.max_adapter_delta_norm,
            candidate_adapter_delta_norm=candidate_delta_norm,
        )
        return LoraUpdateCandidate(
            model,
            parent_tensors,
            parent_foundation_digest,
            foundation_sentinel,
            receipt,
        )
    except BaseException as update_error:
        _clear_parameter_gradients(adapter_parameters)
        _restore_training_modes(prior_training_modes)
        try:
            _restore_adapter_tensors(model, parent_tensors)
            restored_digest = adapter_tensor_digest(
                model,
                adapter_name=inventory.adapter_name,
            )
            if restored_digest != parent_digest:
                raise BoundedUpdateError(
                    "failure rollback did not restore the parent adapter"
                )
            if foundation_tensor_digest(model) != parent_foundation_digest:
                raise BoundedUpdateError(
                    "adapter rollback succeeded but foundation content changed; "
                    "the foundation must be reloaded"
                )
        except BaseException as rollback_error:
            if isinstance(rollback_error, BoundedUpdateError) and (
                "foundation content changed" in str(rollback_error)
            ):
                raise rollback_error from update_error
            raise BoundedUpdateError(
                "bounded update failed and exact rollback also failed"
            ) from rollback_error
        raise


def _validate_batch(batch: TeacherForcedReflectionBatch) -> tuple[int, int]:
    input_ids = batch.input_ids
    labels = batch.reflection_labels
    attention_mask = batch.attention_mask

    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, sequence>=2]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long token IDs")
    if labels.shape != input_ids.shape or labels.dtype != torch.long:
        raise ValueError(
            "reflection_labels must be torch.long with the input_ids shape"
        )
    if labels.device != input_ids.device:
        raise ValueError("input_ids and reflection_labels must share a device")
    if bool(labels[:, 0].ne(_IGNORE_INDEX).any().item()):
        raise ValueError("the first token cannot be a causal supervision target")

    supervised_mask = labels.ne(_IGNORE_INDEX)
    if bool(
        labels.masked_select(supervised_mask)
        .ne(input_ids.masked_select(supervised_mask))
        .any()
        .item()
    ):
        raise ValueError(
            "supervised labels must be tokens from the supplied model output"
        )

    input_tokens = input_ids.numel()
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the input_ids shape")
        if attention_mask.device != input_ids.device:
            raise ValueError("input_ids and attention_mask must share a device")
        if not bool(
            torch.logical_or(attention_mask == 0, attention_mask == 1)
            .all()
            .item()
        ):
            raise ValueError("attention_mask values must be zero or one")
        if bool(
            labels.masked_select(attention_mask == 0)
            .ne(_IGNORE_INDEX)
            .any()
            .item()
        ):
            raise ValueError("padded positions must not be supervised")
    supervised_tokens = int(labels[:, 1:].ne(_IGNORE_INDEX).sum().item())
    if supervised_tokens <= 0:
        raise ValueError("reflection_labels contain no causal target tokens")
    return input_tokens, supervised_tokens


def _extract_logits(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise BoundedUpdateError("causal model output does not contain logits")
    return logits


def _capture_foundation_sentinel(
    records: Sequence[Any],
    named_parameters: dict[str, nn.Parameter],
) -> tuple[_FoundationParameterSentinel, ...]:
    sentinels: list[_FoundationParameterSentinel] = []
    for record in records:
        parameter = named_parameters[record.name]
        if parameter.layout is not torch.strided or parameter.device.type == "meta":
            raise BoundedUpdateError(
                f"foundation parameter {record.name!r} cannot be integrity checked"
            )
        sentinels.append(
            _FoundationParameterSentinel(
                name=record.name,
                object_id=id(parameter),
                data_ptr=parameter.data_ptr(),
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                storage_offset=parameter.storage_offset(),
                dtype=str(parameter.dtype),
                device=str(parameter.device),
                version=parameter._version,
            )
        )
    return tuple(sentinels)


def _verify_foundation_sentinel(
    model: nn.Module,
    expected: tuple[_FoundationParameterSentinel, ...],
    *,
    adapter_name: str,
) -> None:
    inventory = validate_foundation_frozen(
        model,
        adapter_name=adapter_name,
        require_trainable_lora=True,
    )
    named_parameters = dict(model.named_parameters())
    actual = _capture_foundation_sentinel(
        inventory.foundation,
        named_parameters,
    )
    if actual != expected:
        raise BoundedUpdateError("foundation parameter identity or content changed")
    if any(
        named_parameters[record.name].grad is not None
        for record in inventory.foundation
    ):
        raise BoundedUpdateError("a foundation parameter received a gradient")


def _adapter_delta_norm(
    model: nn.Module,
    parent_tensors: dict[str, torch.Tensor],
) -> float:
    named_parameters = dict(model.named_parameters())
    squared_norm = 0.0
    for name, parent_value in parent_tensors.items():
        parameter = named_parameters[name]
        difference = parameter.detach().float() - parent_value.to(
            device=parameter.device,
            dtype=torch.float32,
        )
        squared_norm += float(torch.sum(difference * difference).cpu().item())
    return math.sqrt(squared_norm)


def _clear_parameter_gradients(parameters: Sequence[nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _restore_training_modes(
    modes: Sequence[tuple[nn.Module, bool]],
) -> None:
    for module, training in modes:
        module.training = training


def _restore_adapter_tensors(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    current = dict(model.named_parameters())
    if set(current).isdisjoint(snapshot):
        raise BoundedUpdateError("adapter rollback parameters are missing")

    with torch.no_grad():
        for name, parent_value in snapshot.items():
            parameter = current.get(name)
            if parameter is None:
                raise BoundedUpdateError(
                    f"adapter rollback parameter {name!r} is missing"
                )
            if parameter.shape != parent_value.shape:
                raise BoundedUpdateError(
                    f"adapter rollback parameter {name!r} changed shape"
                )
            parameter.copy_(
                parent_value.to(device=parameter.device, dtype=parameter.dtype)
            )
