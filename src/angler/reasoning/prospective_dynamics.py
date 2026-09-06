"""Bounded internal prospective dynamics for one composite competence state.

This module is a preparatory learned mechanism. Its world, self, and focus
tensors are hypotheses inside plastic state, not canonical identities, truth,
permission, sentience, or persisted Prospective-Origin records. The isolated
multiworld API accepts only synthetic/public feature rows. The runtime-facing
composite core deliberately consumes only the relation, temporal, base-logit,
and state inputs already bound by DurableAbilityLearner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .structure_keyed_credit_memory import (
    RELATION_WIDTH,
    StructureKeyedCreditEvent,
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditOutput,
    StructureKeyedCreditSnapshot,
    StructureKeyedCreditState,
)


_EVENT_SCHEMA = "angler.composite-prospective-credit-event.v1"


def _positive_int(value: object, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" at most {maximum}"
        raise ValueError(f"{label} must be a positive integer{suffix}")
    return value


def _finite_positive(value: object, label: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        or (maximum is not None and float(value) > maximum)
    ):
        suffix = "" if maximum is None else f" no greater than {maximum}"
        raise ValueError(f"{label} must be finite and positive{suffix}")
    return float(value)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _update_tensor_digest(digest: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(_tensor_bytes(tensor))


def _hex_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_identity(value: object, label: str) -> str:
    if type(value) is not str or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a SHA-256 identity")
    _hex_digest(value.removeprefix("sha256:"), label)
    return value


@dataclass(frozen=True, slots=True)
class ProspectiveDynamicsConfig:
    temporal_width: int
    relation_width: int = RELATION_WIDTH
    latent_width: int = 16
    world_slots: int = 8
    maximum_world_reads: int = 4
    maximum_branches: int = 64
    maximum_recurrent_steps: int = 4
    maximum_update_backtracks: int = 12
    maximum_residual: float = 2.0
    update_rate: float = 0.5

    def __post_init__(self) -> None:
        _positive_int(self.temporal_width, "temporal_width", maximum=4_096)
        if self.relation_width != RELATION_WIDTH:
            raise ValueError("prospective dynamics requires public 64-D relation features")
        _positive_int(self.latent_width, "latent_width", maximum=1_024)
        _positive_int(self.world_slots, "world_slots", maximum=256)
        _positive_int(
            self.maximum_world_reads,
            "maximum_world_reads",
            maximum=self.world_slots,
        )
        _positive_int(self.maximum_branches, "maximum_branches", maximum=4_096)
        _positive_int(
            self.maximum_recurrent_steps,
            "maximum_recurrent_steps",
            maximum=64,
        )
        _positive_int(
            self.maximum_update_backtracks,
            "maximum_update_backtracks",
            maximum=64,
        )
        _finite_positive(self.maximum_residual, "maximum_residual", maximum=64.0)
        rate = _finite_positive(self.update_rate, "update_rate", maximum=1.0)
        object.__setattr__(self, "update_rate", rate)

    def to_record(self) -> dict[str, int | float]:
        return {
            "latent_width": self.latent_width,
            "maximum_branches": self.maximum_branches,
            "maximum_recurrent_steps": self.maximum_recurrent_steps,
            "maximum_residual": self.maximum_residual,
            "maximum_update_backtracks": self.maximum_update_backtracks,
            "maximum_world_reads": self.maximum_world_reads,
            "relation_width": self.relation_width,
            "temporal_width": self.temporal_width,
            "update_rate": self.update_rate,
            "world_slots": self.world_slots,
        }

    @property
    def digest(self) -> str:
        material = json.dumps(
            self.to_record(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(
            b"angler.prospective-dynamics-config.v1\0" + material
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProspectiveResourceBudget:
    world_reads: int
    branches: int
    recurrent_steps: int


@dataclass(frozen=True, slots=True)
class ProspectiveDynamicsState:
    credit_state: StructureKeyedCreditState
    world_hypotheses: torch.Tensor
    self_hypotheses: torch.Tensor
    focus_hypothesis: torch.Tensor
    outcome_hypothesis: torch.Tensor
    world_usage: torch.Tensor
    step: int

    @property
    def bytes(self) -> int:
        values = (
            self.credit_state.keys,
            self.credit_state.values,
            self.credit_state.usage,
            self.credit_state.acquisition,
            self.world_hypotheses,
            self.self_hypotheses,
            self.focus_hypothesis,
            self.outcome_hypothesis,
            self.world_usage,
        )
        return sum(value.numel() * value.element_size() for value in values)

    def detached_clone(self) -> "ProspectiveDynamicsState":
        return ProspectiveDynamicsState(
            credit_state=self.credit_state.detached_clone(),
            world_hypotheses=self.world_hypotheses.detach().clone(),
            self_hypotheses=self.self_hypotheses.detach().clone(),
            focus_hypothesis=self.focus_hypothesis.detach().clone(),
            outcome_hypothesis=self.outcome_hypothesis.detach().clone(),
            world_usage=self.world_usage.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class ProspectiveDynamicsSnapshot:
    credit_keys: torch.Tensor
    credit_values: torch.Tensor
    credit_usage: torch.Tensor
    credit_acquisition: torch.Tensor
    world_hypotheses: torch.Tensor
    self_hypotheses: torch.Tensor
    focus_hypothesis: torch.Tensor
    outcome_hypothesis: torch.Tensor
    world_usage: torch.Tensor
    step: int
    config_digest: str
    checkpoint_identity: str


@dataclass(frozen=True, slots=True)
class ProspectiveComponentStateIntegrity:
    """Domain-separated integrity material for one composite parent state."""

    step: int
    checkpoint_ref: str
    config_ref: str
    world_state_digest: str
    self_state_digest: str
    focus_state_digest: str
    outcome_state_digest: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("component state step must be a non-negative integer")
        for label, value in (
            ("checkpoint_ref", self.checkpoint_ref),
            ("config_ref", self.config_ref),
            ("world_state_digest", self.world_state_digest),
            ("self_state_digest", self.self_state_digest),
            ("focus_state_digest", self.focus_state_digest),
            ("outcome_state_digest", self.outcome_state_digest),
        ):
            _sha256_identity(value, label)


@dataclass(frozen=True, slots=True)
class ProspectiveFocusOutput:
    future_latents: torch.Tensor
    outcome_logits: torch.Tensor
    uncertainties: torch.Tensor
    selection_residuals: torch.Tensor
    focus_weights: torch.Tensor
    focused_mask: torch.Tensor
    recurrent_steps: int
    state_step: int


@dataclass(frozen=True, slots=True)
class CompositeProspectiveCreditEvent:
    credit_event: StructureKeyedCreditEvent
    prospective_state_digest: str
    prospective_residual: float
    focused_slots: tuple[int, ...]
    config_digest: str
    checkpoint_identity: str
    read_enabled: bool
    outcome_loss_before: float
    outcome_loss_after: float

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return self.credit_event.evidence_refs

    def _validate_record_fields(self) -> None:
        self.credit_event._validate_record_fields()
        _hex_digest(self.prospective_state_digest, "prospective_state_digest")
        _hex_digest(self.config_digest, "config_digest")
        _sha256_identity(self.checkpoint_identity, "checkpoint_identity")
        if not math.isfinite(self.prospective_residual):
            raise ValueError("prospective_residual must be finite")
        if type(self.read_enabled) is not bool:
            raise ValueError("read_enabled must be bool")
        if (
            not math.isfinite(self.outcome_loss_before)
            or not math.isfinite(self.outcome_loss_after)
            or self.outcome_loss_before < 0.0
            or self.outcome_loss_after < 0.0
            or self.outcome_loss_after > self.outcome_loss_before
        ):
            raise ValueError("outcome losses must be finite, non-negative, and non-increasing")
        if (
            type(self.focused_slots) is not tuple
            or not self.focused_slots
            or any(type(item) is not int or item < 0 for item in self.focused_slots)
            or len(set(self.focused_slots)) != len(self.focused_slots)
        ):
            raise ValueError("focused_slots must be distinct non-negative integers")

    def to_record(self) -> dict[str, Any]:
        self._validate_record_fields()
        return {
            "checkpoint_identity": self.checkpoint_identity,
            "config_digest": self.config_digest,
            "credit_event": self.credit_event.to_record(),
            "focused_slots": list(self.focused_slots),
            "outcome_loss_after": self.outcome_loss_after,
            "outcome_loss_before": self.outcome_loss_before,
            "prospective_residual": self.prospective_residual,
            "prospective_state_digest": self.prospective_state_digest,
            "read_enabled": self.read_enabled,
            "schema": _EVENT_SCHEMA,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CompositeProspectiveCreditEvent":
        expected = {
            "checkpoint_identity",
            "config_digest",
            "credit_event",
            "focused_slots",
            "outcome_loss_after",
            "outcome_loss_before",
            "prospective_residual",
            "prospective_state_digest",
            "read_enabled",
            "schema",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("composite prospective event fields are malformed")
        if record.get("schema") != _EVENT_SCHEMA:
            raise ValueError("composite prospective event schema is unsupported")
        try:
            event = cls(
                credit_event=StructureKeyedCreditEvent.from_record(
                    record["credit_event"]
                ),
                prospective_state_digest=_hex_digest(
                    record["prospective_state_digest"],
                    "prospective_state_digest",
                ),
                prospective_residual=float(record["prospective_residual"]),
                focused_slots=tuple(record["focused_slots"]),
                config_digest=_hex_digest(record["config_digest"], "config_digest"),
                checkpoint_identity=_sha256_identity(
                    record["checkpoint_identity"],
                    "checkpoint_identity",
                ),
                read_enabled=record["read_enabled"],
                outcome_loss_before=float(record["outcome_loss_before"]),
                outcome_loss_after=float(record["outcome_loss_after"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("composite prospective event is malformed") from exc
        event._validate_record_fields()
        return event


class CompositeProspectiveCreditCore(nn.Module):
    """One replayable procedure-credit and prospective-hypothesis core."""

    def __init__(
        self,
        credit_core: StructureKeyedCreditMemoryCore,
        config: ProspectiveDynamicsConfig,
    ) -> None:
        super().__init__()
        if not isinstance(credit_core, StructureKeyedCreditMemoryCore):
            raise TypeError("credit_core must be StructureKeyedCreditMemoryCore")
        if not isinstance(config, ProspectiveDynamicsConfig):
            raise TypeError("config must be ProspectiveDynamicsConfig")
        if (
            credit_core.relation_width != config.relation_width
            or credit_core.temporal_width != config.temporal_width
        ):
            raise ValueError("credit and prospective feature widths must match")
        self.credit_core = credit_core
        self.config = config
        self.relation_width = config.relation_width
        self.temporal_width = config.temporal_width
        self.rank = credit_core.rank
        self.memory_slots = credit_core.memory_slots
        self.maximum_residual = (
            credit_core.maximum_residual + config.maximum_residual
        )

        latent = config.latent_width
        self.action_network = nn.Sequential(
            nn.LayerNorm(config.relation_width),
            nn.Linear(config.relation_width, latent),
            nn.Tanh(),
        )
        self.world_network = nn.Sequential(
            nn.LayerNorm(config.temporal_width),
            nn.Linear(config.temporal_width, latent),
            nn.Tanh(),
        )
        self.self_network = nn.Sequential(
            nn.LayerNorm(config.temporal_width),
            nn.Linear(config.temporal_width, latent),
            nn.Tanh(),
        )
        self.focus_query = nn.Linear(latent * 2, latent, bias=False)
        self.focus_key = nn.Linear(latent, latent, bias=False)
        self.future_cell = nn.GRUCell(latent * 3, latent)
        head_width = latent * 4
        self.outcome_network = nn.Linear(head_width, 1, bias=False)
        self.uncertainty_network = nn.Linear(head_width, 1)
        self.residual_gate = nn.Sequential(
            nn.Linear(latent * 2, latent),
            nn.Sigmoid(),
        )
        self.residual_network = nn.Sequential(
            nn.Linear(latent * 2, latent, bias=False),
            nn.SiLU(),
            nn.Linear(latent, 1, bias=False),
        )

    @property
    def checkpoint_identity(self) -> str:
        digest = hashlib.sha256(b"angler.composite-prospective-checkpoint.v1\0")
        digest.update(self.config.digest.encode("ascii"))
        for name, value in sorted(self.state_dict().items()):
            _update_tensor_digest(digest, name, value)
        return "sha256:" + digest.hexdigest()

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ProspectiveDynamicsState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        credit = self.credit_core.initial_state(
            device=resolved_device,
            dtype=resolved_dtype,
        )
        slots = self.config.world_slots
        latent = self.config.latent_width
        return ProspectiveDynamicsState(
            credit_state=credit,
            world_hypotheses=torch.zeros(
                slots, latent, device=resolved_device, dtype=resolved_dtype
            ),
            self_hypotheses=torch.zeros(
                slots, latent, device=resolved_device, dtype=resolved_dtype
            ),
            focus_hypothesis=torch.zeros(
                latent, device=resolved_device, dtype=resolved_dtype
            ),
            outcome_hypothesis=torch.zeros(
                latent, device=resolved_device, dtype=resolved_dtype
            ),
            world_usage=torch.zeros(
                slots, device=resolved_device, dtype=resolved_dtype
            ),
            step=0,
        )

    initial_ability_state = initial_state

    def _validate_budget(
        self,
        budget: ProspectiveResourceBudget,
        *,
        batch: int,
        worlds: int,
        eligible: torch.Tensor,
    ) -> None:
        if not isinstance(budget, ProspectiveResourceBudget):
            raise TypeError("budget must be ProspectiveResourceBudget")
        reads = _positive_int(
            budget.world_reads,
            "world_reads",
            maximum=self.config.maximum_world_reads,
        )
        branches = _positive_int(
            budget.branches,
            "branches",
            maximum=self.config.maximum_branches,
        )
        _positive_int(
            budget.recurrent_steps,
            "recurrent_steps",
            maximum=self.config.maximum_recurrent_steps,
        )
        if branches < batch:
            raise ValueError("branch budget must cover every supplied branch")
        if reads > worlds or bool((eligible.sum(dim=-1) < reads).any().item()):
            raise ValueError("world-read budget exceeds an eligible world set")

    def prospect(
        self,
        relation_features: torch.Tensor,
        world_features: torch.Tensor,
        eligible_mask: torch.Tensor,
        *,
        state: ProspectiveDynamicsState,
        budget: ProspectiveResourceBudget,
    ) -> ProspectiveFocusOutput:
        """Run the isolated public-feature prospective/focus component."""

        self.validate_state(state)
        reference = next(self.parameters())
        if (
            relation_features.ndim != 2
            or relation_features.shape[-1] != self.relation_width
            or relation_features.shape[0] == 0
        ):
            raise ValueError("relation_features must be nonempty [branch,64]")
        batch = relation_features.shape[0]
        if (
            world_features.ndim != 3
            or world_features.shape[0] != batch
            or world_features.shape[2] != self.temporal_width
            or not 1 <= world_features.shape[1] <= self.config.world_slots
        ):
            raise ValueError(
                "world_features must be [branch,world,temporal_width] within capacity"
            )
        worlds = world_features.shape[1]
        if eligible_mask.shape != (batch, worlds) or eligible_mask.dtype != torch.bool:
            raise ValueError("eligible_mask must be a boolean [branch,world] tensor")
        for name, value in (
            ("relation_features", relation_features),
            ("world_features", world_features),
            ("eligible_mask", eligible_mask),
        ):
            if value.device != reference.device:
                raise ValueError(f"{name} must match core device")
        if (
            relation_features.dtype != reference.dtype
            or world_features.dtype != reference.dtype
            or not bool(torch.isfinite(relation_features).all().item())
            or not bool(torch.isfinite(world_features).all().item())
        ):
            raise ValueError("prospective feature rows must match dtype and be finite")
        self._validate_budget(
            budget,
            batch=batch,
            worlds=worlds,
            eligible=eligible_mask,
        )

        relations = relation_features.detach()
        worlds_observed = world_features.detach()
        action = self.action_network(relations)
        encoded_worlds = self.world_network(
            worlds_observed.reshape(batch * worlds, self.temporal_width)
        ).reshape(batch, worlds, self.config.latent_width)
        encoded_self = self.self_network(
            worlds_observed.reshape(batch * worlds, self.temporal_width)
        ).reshape(batch, worlds, self.config.latent_width)
        stored_worlds = state.world_hypotheses[:worlds].unsqueeze(0)
        stored_self = state.self_hypotheses[:worlds].unsqueeze(0)
        keys = self.focus_key(encoded_worlds + stored_worlds + stored_self)
        query = self.focus_query(
            torch.cat(
                (
                    action,
                    state.focus_hypothesis.unsqueeze(0).expand(batch, -1),
                ),
                dim=-1,
            )
        )
        focus_logits = torch.einsum("bwl,bl->bw", keys, query)
        focus_logits = focus_logits / math.sqrt(float(self.config.latent_width))
        masked_logits = focus_logits.masked_fill(~eligible_mask, float("-inf"))
        top = torch.topk(
            masked_logits,
            k=budget.world_reads,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        focused_mask = torch.zeros_like(eligible_mask)
        focused_mask.scatter_(1, top, True)
        focused_mask &= eligible_mask
        selected_logits = masked_logits.masked_fill(~focused_mask, float("-inf"))
        focus_weights = torch.softmax(selected_logits, dim=-1)
        focus_weights = torch.where(
            focused_mask,
            focus_weights,
            torch.zeros_like(focus_weights),
        )
        context = torch.sum(
            focus_weights.unsqueeze(-1)
            * (encoded_worlds + stored_worlds + stored_self),
            dim=1,
        )
        self_context = torch.sum(
            focus_weights.unsqueeze(-1) * (encoded_self + stored_self),
            dim=1,
        )
        hidden = state.focus_hypothesis.unsqueeze(0).expand(batch, -1)
        recurrent_input = torch.cat((action, context, self_context), dim=-1)
        for _ in range(budget.recurrent_steps):
            hidden = self.future_cell(recurrent_input, hidden)
        future = hidden
        outcome_context = state.outcome_hypothesis.unsqueeze(0).expand(batch, -1)
        head_input = torch.cat((future, action, context, outcome_context), dim=-1)
        outcome_logits = self.outcome_network(head_input).squeeze(-1)
        uncertainties = F.softplus(self.uncertainty_network(head_input).squeeze(-1))
        uncertainties = uncertainties + torch.finfo(uncertainties.dtype).eps

        learned_memory = (
            torch.sum(
                focus_weights.unsqueeze(-1) * (stored_worlds + stored_self),
                dim=1,
            )
            + state.focus_hypothesis.unsqueeze(0)
            + state.outcome_hypothesis.unsqueeze(0)
        )
        gate = self.residual_gate(torch.cat((action, context), dim=-1))
        gated = learned_memory * gate
        raw_residual = self.residual_network(
            torch.cat((gated, gated * action), dim=-1)
        ).squeeze(-1)
        selection_residuals = self.config.maximum_residual * torch.tanh(
            raw_residual
        )
        output = ProspectiveFocusOutput(
            future_latents=future,
            outcome_logits=outcome_logits,
            uncertainties=uncertainties,
            selection_residuals=selection_residuals,
            focus_weights=focus_weights,
            focused_mask=focused_mask,
            recurrent_steps=budget.recurrent_steps,
            state_step=state.step,
        )
        self._validate_prospective_output(
            output,
            batch=batch,
            worlds=worlds,
            eligible_mask=eligible_mask,
            state=state,
            budget=budget,
        )
        return output

    def _one_world_prospect(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        state: ProspectiveDynamicsState,
    ) -> ProspectiveFocusOutput:
        batch = relation_features.shape[0]
        worlds = temporal_features.unsqueeze(1)
        eligible = torch.ones(
            batch,
            1,
            device=relation_features.device,
            dtype=torch.bool,
        )
        return self.prospect(
            relation_features,
            worlds,
            eligible,
            state=state,
            budget=ProspectiveResourceBudget(
                world_reads=1,
                branches=batch,
                recurrent_steps=1,
            ),
        )

    def predict_with_prospective(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: ProspectiveDynamicsState | None = None,
        read_enabled: bool = True,
        prospective_read_enabled: bool = True,
    ) -> tuple[StructureKeyedCreditOutput, ProspectiveFocusOutput]:
        """Return combined credit and the complete same-call prospective view."""

        if type(read_enabled) is not bool:
            raise TypeError("read_enabled must be bool")
        if type(prospective_read_enabled) is not bool:
            raise TypeError("prospective_read_enabled must be bool")
        current = self.initial_state() if state is None else state
        self.validate_state(current)
        credit = self.credit_core.predict(
            relation_features,
            temporal_features,
            base_logits,
            state=current.credit_state,
            read_enabled=read_enabled,
        )
        prospective = self._one_world_prospect(
            credit.observed_relations,
            credit.observed_temporal,
            current,
        )
        prospective_residual = prospective.selection_residuals
        if not read_enabled or not prospective_read_enabled:
            prospective_residual = torch.zeros_like(prospective_residual)
        residuals = credit.residuals + prospective_residual
        output = StructureKeyedCreditOutput(
            logits=credit.base_logits + residuals,
            base_logits=credit.base_logits,
            residuals=residuals,
            read_queries=credit.read_queries,
            read_weights=credit.read_weights,
            read_values=credit.read_values,
            temporal_hidden=credit.temporal_hidden,
            write_keys=credit.write_keys,
            write_strengths=credit.write_strengths,
            observed_relations=credit.observed_relations,
            observed_temporal=credit.observed_temporal,
            state_step=current.step,
            read_enabled=read_enabled,
        )
        self._validate_output(output)
        return output, prospective

    def predict(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: ProspectiveDynamicsState | None = None,
        read_enabled: bool = True,
    ) -> StructureKeyedCreditOutput:
        output, _prospective = self.predict_with_prospective(
            relation_features,
            temporal_features,
            base_logits,
            state=state,
            read_enabled=read_enabled,
            prospective_read_enabled=True,
        )
        return output

    forward = predict

    def outcome_loss(
        self,
        output: StructureKeyedCreditOutput,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_output(output)
        return self.credit_core.outcome_loss(output, outcomes)

    def _validate_outcomes(self, outcomes: torch.Tensor, *, batch: int) -> None:
        reference = next(self.parameters())
        if (
            not isinstance(outcomes, torch.Tensor)
            or outcomes.shape != (batch,)
            or outcomes.device != reference.device
            or outcomes.dtype != reference.dtype
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be one aligned +/-1 tensor per branch")

    def prospective_training_loss(
        self,
        output: ProspectiveFocusOutput,
        *,
        future_targets: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """Declared differentiable objective for a future trained checkpoint.

        Runtime feedback learns bounded fast state separately. This objective
        exposes future, outcome, uncertainty, focus, and selection heads to
        caller-supplied targets without inventing task labels or observations.
        """

        if not isinstance(output, ProspectiveFocusOutput):
            raise TypeError("output must be ProspectiveFocusOutput")
        reference = next(self.parameters())
        if (
            output.future_latents.ndim != 2
            or output.future_latents.shape[0] == 0
            or output.future_latents.shape[1] != self.config.latent_width
            or output.outcome_logits.shape != (output.future_latents.shape[0],)
            or output.uncertainties.shape != (output.future_latents.shape[0],)
            or output.selection_residuals.shape != (output.future_latents.shape[0],)
        ):
            raise ValueError("prospective training output has malformed shapes")
        batch = output.outcome_logits.shape[0]
        self._validate_outcomes(outcomes, batch=batch)
        training_tensors = (
            output.future_latents,
            output.outcome_logits,
            output.uncertainties,
            output.selection_residuals,
        )
        if (
            not isinstance(future_targets, torch.Tensor)
            or future_targets.shape != output.future_latents.shape
            or future_targets.device != reference.device
            or future_targets.dtype != reference.dtype
            or not bool(torch.isfinite(future_targets).all().item())
        ):
            raise ValueError("future_targets must align as finite latent targets")
        if (
            any(
                value.device != reference.device or value.dtype != reference.dtype
                for value in training_tensors
            )
            or not all(
                bool(torch.isfinite(value).all().item())
                for value in training_tensors
            )
            or not bool((output.uncertainties > 0.0).all().item())
        ):
            raise ValueError("prospective training output is malformed")

        future_error = (output.future_latents - future_targets).square().mean(dim=-1)
        uncertainty_target = torch.sqrt(
            future_error.detach() + torch.finfo(reference.dtype).eps
        )
        outcome_error = F.softplus(-outcomes * output.outcome_logits)
        selection_error = F.softplus(-outcomes * output.selection_residuals)
        uncertainty_error = F.smooth_l1_loss(
            output.uncertainties,
            uncertainty_target,
            reduction="none",
        )
        loss = (
            future_error + outcome_error + selection_error + uncertainty_error
        ).mean()
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise ValueError("prospective training loss must be one finite scalar")
        return loss

    def _loss_directed_hypothesis_update(
        self,
        state: ProspectiveDynamicsState,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
    ]:
        """Take one bounded online gradient step on objective outcome error."""

        self._validate_outcomes(outcomes, batch=1)
        originals = (
            state.world_hypotheses.detach().clone(),
            state.self_hypotheses.detach().clone(),
            state.focus_hypothesis.detach().clone(),
            state.outcome_hypothesis.detach().clone(),
        )
        with torch.enable_grad():
            variables = tuple(value.requires_grad_(True) for value in originals)
            probe_state = ProspectiveDynamicsState(
                credit_state=state.credit_state,
                world_hypotheses=variables[0],
                self_hypotheses=variables[1],
                focus_hypothesis=variables[2],
                outcome_hypothesis=variables[3],
                world_usage=state.world_usage.detach().clone(),
                step=state.step,
            )
            probe = self._one_world_prospect(
                relation_features,
                temporal_features,
                probe_state,
            )
            parent_loss = F.softplus(
                -outcomes.detach() * probe.outcome_logits
            ).mean()
            gradients = torch.autograd.grad(parent_loss, variables)
        if any(not bool(torch.isfinite(value).all().item()) for value in gradients):
            raise RuntimeError("prospective outcome gradient is non-finite")

        before = float(parent_loss.detach().cpu().item())
        best_values = originals
        best_loss = before
        usage = state.world_usage.detach().clone()
        usage[0] = 1.0
        for exponent in range(self.config.maximum_update_backtracks):
            scale = self.config.update_rate / float(2**exponent)
            with torch.no_grad():
                candidate_values = tuple(
                    torch.clamp(value - scale * gradient, min=-1.0, max=1.0)
                    for value, gradient in zip(originals, gradients, strict=True)
                )
                candidate_state = ProspectiveDynamicsState(
                    credit_state=state.credit_state,
                    world_hypotheses=candidate_values[0],
                    self_hypotheses=candidate_values[1],
                    focus_hypothesis=candidate_values[2],
                    outcome_hypothesis=candidate_values[3],
                    world_usage=usage,
                    step=state.step,
                )
                candidate = self._one_world_prospect(
                    relation_features,
                    temporal_features,
                    candidate_state,
                )
                candidate_loss = float(
                    F.softplus(-outcomes * candidate.outcome_logits)
                    .mean()
                    .cpu()
                    .item()
                )
            if candidate_loss <= best_loss:
                best_values = candidate_values
                best_loss = candidate_loss
            if candidate_loss <= before:
                break
        return (*best_values, usage, before, best_loss)

    def apply_feedback(
        self,
        state: ProspectiveDynamicsState,
        output: StructureKeyedCreditOutput,
        outcomes: torch.Tensor,
        *,
        evidence_refs: str | Sequence[str],
        detach_state: bool = True,
    ) -> tuple[ProspectiveDynamicsState, CompositeProspectiveCreditEvent]:
        if type(detach_state) is not bool:
            raise TypeError("detach_state must be bool")
        self.validate_state(state)
        self._validate_output(output)
        if output.logits.shape != (1,) or output.state_step != state.step:
            raise ValueError("one current prospective output is required per feedback")
        credit_output = self.credit_core.predict(
            output.observed_relations,
            output.observed_temporal,
            output.base_logits,
            state=state.credit_state,
            read_enabled=output.read_enabled,
        )
        prospective = self._one_world_prospect(
            output.observed_relations,
            output.observed_temporal,
            state,
        )
        full_prospective = prospective.selection_residuals
        if not output.read_enabled:
            full_prospective = torch.zeros_like(full_prospective)
        full_residual = credit_output.residuals + full_prospective
        lesioned_prospective = torch.zeros_like(full_prospective)
        lesioned_residual = credit_output.residuals
        if torch.equal(output.residuals, full_residual):
            applied_prospective = full_prospective
            expected_residual = full_residual
        elif torch.equal(output.residuals, lesioned_residual):
            applied_prospective = lesioned_prospective
            expected_residual = lesioned_residual
        else:
            raise ValueError("prospective output does not match the composite parent")
        if not torch.equal(output.logits, output.base_logits + expected_residual):
            raise ValueError("prospective output does not match the composite parent")
        for name in (
            "read_queries",
            "read_weights",
            "read_values",
            "temporal_hidden",
            "write_keys",
            "write_strengths",
            "observed_relations",
            "observed_temporal",
        ):
            if not torch.equal(getattr(output, name), getattr(credit_output, name)):
                raise ValueError("prospective output altered procedural evidence fields")

        self._validate_outcomes(outcomes, batch=1)
        (
            world,
            self_state,
            focus,
            outcome_state,
            usage,
            outcome_loss_before,
            outcome_loss_after,
        ) = self._loss_directed_hypothesis_update(
            state,
            output.observed_relations,
            output.observed_temporal,
            outcomes,
        )
        credit_state, credit_event = self.credit_core.apply_feedback(
            state.credit_state,
            credit_output,
            outcomes,
            evidence_refs=evidence_refs,
            detach_state=detach_state,
        )
        updated = ProspectiveDynamicsState(
            credit_state=credit_state,
            world_hypotheses=world,
            self_hypotheses=self_state,
            focus_hypothesis=focus,
            outcome_hypothesis=outcome_state,
            world_usage=usage,
            step=credit_state.step,
        )
        self.validate_state(updated)
        result = updated.detached_clone() if detach_state else updated
        event = CompositeProspectiveCreditEvent(
            credit_event=credit_event,
            prospective_state_digest=self._prospective_state_digest(result),
            prospective_residual=float(applied_prospective[0].detach().cpu().item()),
            focused_slots=(0,),
            config_digest=self.config.digest,
            checkpoint_identity=self.checkpoint_identity,
            read_enabled=output.read_enabled,
            outcome_loss_before=outcome_loss_before,
            outcome_loss_after=outcome_loss_after,
        )
        event._validate_record_fields()
        return result, event

    def capture_state(
        self,
        state: ProspectiveDynamicsState,
    ) -> ProspectiveDynamicsSnapshot:
        self.validate_state(state)
        credit = self.credit_core.capture_state(state.credit_state)
        return ProspectiveDynamicsSnapshot(
            credit_keys=credit.keys,
            credit_values=credit.values,
            credit_usage=credit.usage,
            credit_acquisition=credit.acquisition,
            world_hypotheses=state.world_hypotheses.detach().cpu().clone(),
            self_hypotheses=state.self_hypotheses.detach().cpu().clone(),
            focus_hypothesis=state.focus_hypothesis.detach().cpu().clone(),
            outcome_hypothesis=state.outcome_hypothesis.detach().cpu().clone(),
            world_usage=state.world_usage.detach().cpu().clone(),
            step=state.step,
            config_digest=self.config.digest,
            checkpoint_identity=self.checkpoint_identity,
        )

    def restore_state(
        self,
        snapshot: ProspectiveDynamicsSnapshot,
    ) -> ProspectiveDynamicsState:
        if not isinstance(snapshot, ProspectiveDynamicsSnapshot):
            raise TypeError("snapshot must be ProspectiveDynamicsSnapshot")
        if snapshot.config_digest != self.config.digest:
            raise ValueError("prospective snapshot has another configuration")
        if snapshot.checkpoint_identity != self.checkpoint_identity:
            raise ValueError("prospective snapshot has another parameter checkpoint")
        credit = self.credit_core.restore_state(
            StructureKeyedCreditSnapshot(
                keys=snapshot.credit_keys,
                values=snapshot.credit_values,
                usage=snapshot.credit_usage,
                acquisition=snapshot.credit_acquisition,
                step=snapshot.step,
            )
        )
        reference = next(self.parameters())
        state = ProspectiveDynamicsState(
            credit_state=credit,
            world_hypotheses=snapshot.world_hypotheses.detach().to(reference).clone(),
            self_hypotheses=snapshot.self_hypotheses.detach().to(reference).clone(),
            focus_hypothesis=snapshot.focus_hypothesis.detach().to(reference).clone(),
            outcome_hypothesis=snapshot.outcome_hypothesis.detach().to(reference).clone(),
            world_usage=snapshot.world_usage.detach().to(reference).clone(),
            step=snapshot.step,
        )
        self.validate_state(state)
        return state

    def state_digest(self, state: ProspectiveDynamicsState) -> str:
        self.validate_state(state)
        digest = hashlib.sha256(b"angler.composite-prospective-state.v1\0")
        digest.update(self.checkpoint_identity.encode("ascii") + b"\0")
        digest.update(self.config.digest.encode("ascii") + b"\0")
        digest.update(self.credit_core.state_digest(state.credit_state).encode("ascii"))
        digest.update(b"\0")
        for name, value in (
            ("world_hypotheses", state.world_hypotheses),
            ("self_hypotheses", state.self_hypotheses),
            ("focus_hypothesis", state.focus_hypothesis),
            ("outcome_hypothesis", state.outcome_hypothesis),
            ("world_usage", state.world_usage),
        ):
            _update_tensor_digest(digest, name, value)
        digest.update(str(state.step).encode("ascii"))
        return digest.hexdigest()

    def component_state_integrity(
        self,
        state: ProspectiveDynamicsState,
    ) -> ProspectiveComponentStateIntegrity:
        """Expose integrity-only component refs without interpreting state."""

        self.validate_state(state)

        def component_ref(
            domain: str,
            values: tuple[tuple[str, torch.Tensor], ...],
        ) -> str:
            digest = hashlib.sha256(
                b"angler.prospective-component-state.v1\0"
                + domain.encode("ascii")
                + b"\0"
            )
            for name, value in values:
                _update_tensor_digest(digest, name, value)
            return "sha256:" + digest.hexdigest()

        return ProspectiveComponentStateIntegrity(
            step=state.step,
            checkpoint_ref=self.checkpoint_identity,
            config_ref="sha256:" + self.config.digest,
            world_state_digest=component_ref(
                "world",
                (
                    ("world_hypotheses", state.world_hypotheses),
                    ("world_usage", state.world_usage),
                ),
            ),
            self_state_digest=component_ref(
                "self", (("self_hypotheses", state.self_hypotheses),)
            ),
            focus_state_digest=component_ref(
                "focus", (("focus_hypothesis", state.focus_hypothesis),)
            ),
            outcome_state_digest=component_ref(
                "outcome", (("outcome_hypothesis", state.outcome_hypothesis),)
            ),
        )

    def _prospective_state_digest(self, state: ProspectiveDynamicsState) -> str:
        self.validate_state(state)
        digest = hashlib.sha256(b"angler.prospective-hypothesis-state.v1\0")
        digest.update(self.checkpoint_identity.encode("ascii") + b"\0")
        digest.update(self.config.digest.encode("ascii") + b"\0")
        for name, value in (
            ("world_hypotheses", state.world_hypotheses),
            ("self_hypotheses", state.self_hypotheses),
            ("focus_hypothesis", state.focus_hypothesis),
            ("outcome_hypothesis", state.outcome_hypothesis),
            ("world_usage", state.world_usage),
        ):
            _update_tensor_digest(digest, name, value)
        digest.update(str(state.step).encode("ascii"))
        return digest.hexdigest()

    def zero_state_like(
        self,
        state: ProspectiveDynamicsState,
    ) -> ProspectiveDynamicsState:
        self.validate_state(state)
        return self.initial_state(
            device=state.world_hypotheses.device,
            dtype=state.world_hypotheses.dtype,
        )

    def zero_prospective_state_like(
        self,
        state: ProspectiveDynamicsState,
    ) -> ProspectiveDynamicsState:
        """Lesion only prospective hypotheses while preserving procedure credit."""

        self.validate_state(state)
        return ProspectiveDynamicsState(
            credit_state=state.credit_state.detached_clone(),
            world_hypotheses=torch.zeros_like(state.world_hypotheses),
            self_hypotheses=torch.zeros_like(state.self_hypotheses),
            focus_hypothesis=torch.zeros_like(state.focus_hypothesis),
            outcome_hypothesis=torch.zeros_like(state.outcome_hypothesis),
            world_usage=torch.zeros_like(state.world_usage),
            step=state.step,
        )

    def replay(
        self,
        events: Sequence[CompositeProspectiveCreditEvent],
        *,
        state: ProspectiveDynamicsState | None = None,
        detach_state: bool = True,
    ) -> tuple[ProspectiveDynamicsState, tuple[StructureKeyedCreditOutput, ...]]:
        if type(detach_state) is not bool:
            raise TypeError("detach_state must be bool")
        current = self.initial_state() if state is None else state
        self.validate_state(current)
        reference = next(self.parameters())
        outputs = []
        for event in tuple(events):
            if not isinstance(event, CompositeProspectiveCreditEvent):
                raise TypeError("replay accepts CompositeProspectiveCreditEvent objects")
            event._validate_record_fields()
            if event.config_digest != self.config.digest:
                raise ValueError("event belongs to another prospective configuration")
            if event.checkpoint_identity != self.checkpoint_identity:
                raise ValueError("event belongs to another prospective checkpoint")
            credit_event = event.credit_event
            output, _prospective = self.predict_with_prospective(
                credit_event.relation_features.to(reference),
                credit_event.temporal_features.to(reference),
                credit_event.base_logits.to(reference),
                state=current,
                read_enabled=event.read_enabled,
                prospective_read_enabled=event.prospective_residual != 0.0,
            )
            current, reconstructed = self.apply_feedback(
                current,
                output,
                credit_event.outcomes.to(reference),
                evidence_refs=credit_event.evidence_refs,
                detach_state=detach_state,
            )
            if (
                reconstructed.prospective_state_digest
                != event.prospective_state_digest
                or reconstructed.focused_slots != event.focused_slots
                or reconstructed.checkpoint_identity != event.checkpoint_identity
                or reconstructed.read_enabled != event.read_enabled
                or abs(
                    reconstructed.prospective_residual
                    - event.prospective_residual
                )
                > 1.0e-7
                or abs(
                    reconstructed.outcome_loss_before
                    - event.outcome_loss_before
                )
                > 1.0e-7
                or abs(
                    reconstructed.outcome_loss_after
                    - event.outcome_loss_after
                )
                > 1.0e-7
                or reconstructed.credit_event.to_record()
                != event.credit_event.to_record()
            ):
                raise RuntimeError("composite prospective event replay diverged")
            outputs.append(output)
        return current, tuple(outputs)

    def event_from_record(
        self,
        record: Mapping[str, Any],
    ) -> CompositeProspectiveCreditEvent:
        event = CompositeProspectiveCreditEvent.from_record(record)
        if event.config_digest != self.config.digest:
            raise ValueError("event belongs to another prospective configuration")
        if event.checkpoint_identity != self.checkpoint_identity:
            raise ValueError("event belongs to another prospective checkpoint")
        if any(slot >= self.config.world_slots for slot in event.focused_slots):
            raise ValueError("event focused slot exceeds prospective capacity")
        return event

    def validate_state(self, state: ProspectiveDynamicsState) -> None:
        if not isinstance(state, ProspectiveDynamicsState):
            raise TypeError("state must be ProspectiveDynamicsState")
        self.credit_core.validate_state(state.credit_state)
        reference = next(self.parameters())
        latent = self.config.latent_width
        slots = self.config.world_slots
        expected = (
            ("world_hypotheses", state.world_hypotheses, (slots, latent)),
            ("self_hypotheses", state.self_hypotheses, (slots, latent)),
            ("focus_hypothesis", state.focus_hypothesis, (latent,)),
            ("outcome_hypothesis", state.outcome_hypothesis, (latent,)),
            ("world_usage", state.world_usage, (slots,)),
        )
        for name, value, shape in expected:
            if value.shape != shape:
                raise ValueError(f"{name} has the wrong shape")
            if (
                value.device != reference.device
                or value.dtype != reference.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"{name} must match core device/dtype and be finite")
            if name != "world_usage" and not bool(
                (value.abs() <= 1.0 + 1.0e-6).all().item()
            ):
                raise ValueError(f"{name} exceeded [-1,1]")
        if not bool(
            ((state.world_usage >= 0.0) & (state.world_usage <= 1.0)).all().item()
        ):
            raise ValueError("world_usage exceeded [0,1]")
        if (
            type(state.step) is not int
            or state.step != state.credit_state.step
            or not 0 <= state.step <= self.memory_slots
        ):
            raise ValueError("composite state step is invalid or unsynchronized")

    def _validate_output(self, output: StructureKeyedCreditOutput) -> None:
        self.credit_core._validate_output(output)
        if not torch.equal(output.logits, output.base_logits + output.residuals):
            raise ValueError("composite logit decomposition is invalid")
        if not bool(
            (output.residuals.abs() <= self.maximum_residual + 1.0e-6).all().item()
        ):
            raise ValueError("composite residual exceeds its configured bound")

    def _validate_prospective_output(
        self,
        output: ProspectiveFocusOutput,
        *,
        batch: int,
        worlds: int,
        eligible_mask: torch.Tensor,
        state: ProspectiveDynamicsState,
        budget: ProspectiveResourceBudget,
    ) -> None:
        if not isinstance(output, ProspectiveFocusOutput):
            raise TypeError("output must be ProspectiveFocusOutput")
        if (
            eligible_mask.shape != (batch, worlds)
            or eligible_mask.dtype != torch.bool
        ):
            raise ValueError("eligible_mask must align as boolean output context")
        self._validate_budget(
            budget,
            batch=batch,
            worlds=worlds,
            eligible=eligible_mask,
        )
        latent = self.config.latent_width
        if (
            output.future_latents.shape != (batch, latent)
            or output.outcome_logits.shape != (batch,)
            or output.uncertainties.shape != (batch,)
            or output.selection_residuals.shape != (batch,)
            or output.focus_weights.shape != (batch, worlds)
            or output.focused_mask.shape != (batch, worlds)
            or output.focused_mask.dtype != torch.bool
            or type(output.recurrent_steps) is not int
            or output.recurrent_steps != budget.recurrent_steps
            or type(output.state_step) is not int
            or output.state_step != state.step
        ):
            raise ValueError("prospective output fields have wrong shape or type")
        reference = next(self.parameters())
        tensors = (
            output.future_latents,
            output.outcome_logits,
            output.uncertainties,
            output.selection_residuals,
            output.focus_weights,
        )
        if (
            any(
                value.device != reference.device or value.dtype != reference.dtype
                for value in tensors
            )
            or output.focused_mask.device != reference.device
            or eligible_mask.device != reference.device
        ):
            raise ValueError("prospective output must match core device and dtype")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("prospective output must be finite")
        if not bool((output.uncertainties > 0.0).all().item()):
            raise ValueError("prospective uncertainty must be positive")
        if not bool(
            ((output.focus_weights >= 0.0) & (output.focus_weights <= 1.0))
            .all()
            .item()
        ):
            raise ValueError("prospective focus weights must be probabilities")
        if bool((output.focused_mask & ~eligible_mask).any().item()):
            raise ValueError("prospective focus escaped the eligible world set")
        if not bool(
            (output.focused_mask.sum(dim=-1) == budget.world_reads).all().item()
        ):
            raise ValueError("prospective focus count does not match its read budget")
        if bool((output.focus_weights.masked_select(~output.focused_mask) != 0).any().item()):
            raise ValueError("unfocused worlds received allocation")
        if not torch.allclose(
            output.focus_weights.sum(dim=-1),
            torch.ones(
                batch,
                device=output.focus_weights.device,
                dtype=output.focus_weights.dtype,
            ),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("prospective focus weights do not sum to one")
        if not bool(
            (
                output.selection_residuals.abs()
                <= self.config.maximum_residual + 1.0e-6
            ).all().item()
        ):
            raise ValueError("prospective residual exceeded its configured bound")

    def parameter_report(self) -> dict[str, Any]:
        return {
            "checkpoint_identity": self.checkpoint_identity,
            "config": self.config.to_record(),
            "parameter_count": self.parameter_count,
            "component_only_multiworld": True,
            "runtime_multiworld_context": False,
            "declared_training_objective": True,
            "runtime_loss_directed_fast_state": True,
            "trained_checkpoint_supplied": False,
            "authorization_inputs": False,
            "truth_inputs": False,
            "task_identity_inputs": False,
        }


__all__ = [
    "CompositeProspectiveCreditCore",
    "CompositeProspectiveCreditEvent",
    "ProspectiveDynamicsConfig",
    "ProspectiveDynamicsSnapshot",
    "ProspectiveDynamicsState",
    "ProspectiveComponentStateIntegrity",
    "ProspectiveFocusOutput",
    "ProspectiveResourceBudget",
]
