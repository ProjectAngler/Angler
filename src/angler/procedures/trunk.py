"""Model-agnostic neural primitives for learned procedural operators.

The module consumes the immutable record contracts in :mod:`.records`, but it
does not execute actions or inspect an environment.  States, goals, and action
schemas are encoded from their public structure.  In particular, operators do
not receive a learned lookup-table ID: changing a schema changes the features
from which its embedding is computed.

The tensor-level methods are intentionally public.  They make the learned
boundary explicit and let evaluators train or ablate initiation, dynamics,
termination, primitive decoding, and direct proposal independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from torch import nn


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/+-]+|[^\s]")
_HASH_DOMAIN = b"project-angler.procedure-features.v1\x00"


@lru_cache(maxsize=65_536)
def _cached_hash_row(
    text: str,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build one immutable CPU hash row once, before device transfer.

    The hash encoder has no trainable state.  Caching its normalized input
    rows avoids repeating SHA-256 work and, critically, avoids one tiny GPU
    indexing kernel per lexical token inside every optimizer step.  Callers
    receive a stacked copy, so cached storage is never exposed for mutation.
    """

    result = torch.zeros(width, dtype=dtype, device="cpu")
    lexical_tokens = _TOKEN_PATTERN.findall(text.casefold())
    tokens = [f"whole:{text}", f"length:{len(text)}", *lexical_tokens]
    for position, token in enumerate(tokens):
        material = (
            _HASH_DOMAIN
            + position.to_bytes(4, "big")
            + token.encode("utf-8")
        )
        digest = hashlib.sha256(material).digest()
        index = int.from_bytes(digest[:8], "big") % width
        result[index] += 1.0 if digest[8] & 1 else -1.0
    for token in lexical_tokens:
        digest = hashlib.sha256(
            _HASH_DOMAIN + b"bag\x00" + token.encode("utf-8")
        ).digest()
        index = int.from_bytes(digest[:8], "big") % width
        result[index] += 1.0 if digest[8] & 1 else -1.0
    return result / result.norm().clamp_min(1.0)


@runtime_checkable
class RecordEncoder(Protocol):
    """Structural interface for a model-independent record feature source.

    A future text model, graph model, or hand-built sensor can implement this
    protocol.  The neural operator core depends only on fixed-width record
    features and never on a particular foundation-model class.
    """

    output_width: int

    def encode_records(
        self,
        records: Sequence[Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return one feature row per record, shaped ``[records, width]``."""


def _hash_texts(
    texts: Sequence[str],
    width: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if width <= 0:
        raise ValueError("hash width must be positive")
    if not dtype.is_floating_point:
        raise ValueError("hashed features require a floating-point dtype")
    rows: list[torch.Tensor] = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("hashed inputs must be strings")
        rows.append(_cached_hash_row(text, width, dtype))
    if not rows:
        return torch.zeros((0, width), device=device, dtype=dtype)
    return torch.stack(rows).to(device=device)


def _record_text(record: Any) -> str:
    try:
        predicate = record.predicate
        arguments = record.arguments
    except AttributeError as error:
        raise TypeError("records must expose predicate and arguments") from error
    if not isinstance(predicate, str):
        raise TypeError("record predicate must be a string")
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        raise TypeError("record arguments must be a finite sequence")
    payload = {
        "predicate": predicate,
        "arguments": [str(argument) for argument in arguments],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def canonical_schema_text(schema: Any) -> str:
    """Return structural schema material without using a free embedding ID.

    ``ActionSchema`` is handled directly.  Richer induced operators can expose
    ``to_canonical``; their complete canonical form is embedded rather than
    their digest alone.
    """

    if hasattr(schema, "to_canonical"):
        # LearnedOperator canonical forms include lineage and exemplar
        # provenance.  Those fields are essential evidence, but they are not
        # operator semantics and would turn an embedding into an episode ID.
        payload = _without_provenance(schema.to_canonical())
    elif hasattr(schema, "name") and hasattr(schema, "parameters"):
        parameters = []
        for parameter in schema.parameters:
            try:
                parameters.append(
                    {"name": parameter.name, "type_name": parameter.type_name}
                )
            except AttributeError as error:
                raise TypeError(
                    "schema parameters must expose name and type_name"
                ) from error
        description = getattr(schema, "description", None)
        payload = {
            "description": description,
            "name": schema.name,
            "parameters": parameters,
        }
    else:
        raise TypeError(
            "operator schemas must expose name/parameters or to_canonical()"
        )
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("operator canonical form must be JSON-serializable") from error


def _without_provenance(value: Any) -> Any:
    """Remove evidence identity while retaining executable schema structure."""

    if isinstance(value, dict):
        excluded = {
            "digest",
            "exemplars",
            "parent_digest",
            "revision",
            "trace_digest",
            "before_state_digest",
            "after_state_digest",
            "action_digests",
        }
        return {
            key: _without_provenance(item)
            for key, item in value.items()
            if key not in excluded and not key.endswith("_digest")
        }
    if isinstance(value, (tuple, list)):
        return [_without_provenance(item) for item in value]
    return value


class FrozenHashTextEncoder(nn.Module):
    """Deterministic, parameter-free hashed text/record features.

    This is an intentionally small bootstrap encoder for unit tests and
    controlled experiments.  It can be replaced by any ``RecordEncoder``
    without changing the operator trunk.
    """

    def __init__(self, output_width: int = 128) -> None:
        super().__init__()
        if output_width <= 0:
            raise ValueError("output_width must be positive")
        self.output_width = output_width
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        target_device = self._device_anchor.device if device is None else device
        return _hash_texts(
            texts,
            self.output_width,
            device=target_device,
            dtype=dtype,
        )

    def encode_records(
        self,
        records: Sequence[Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.encode_texts(
            [_record_text(record) for record in records],
            device=device,
            dtype=dtype,
        )

    def forward(self, records: Sequence[Any]) -> torch.Tensor:
        return self.encode_records(records)


# A discoverable name for callers that want to emphasize the record contract.
FrozenHashRecordEncoder = FrozenHashTextEncoder


class PermutationInvariantSetEncoder(nn.Module):
    """Trainable DeepSets-style encoder over unordered typed records."""

    def __init__(
        self,
        record_encoder: RecordEncoder,
        *,
        hidden_width: int = 128,
        output_width: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(record_encoder.output_width, int) or record_encoder.output_width <= 0:
            raise ValueError("record_encoder.output_width must be positive")
        if hidden_width <= 0 or output_width <= 0:
            raise ValueError("set-encoder widths must be positive")
        self.record_encoder = record_encoder
        self.input_width = record_encoder.output_width
        self.output_width = output_width
        self.item_network = nn.Sequential(
            nn.LayerNorm(self.input_width),
            nn.Linear(self.input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        # Sum, mean, max, namespace features, and one log-count scalar.
        pooled_width = hidden_width * 3 + self.input_width + 1
        self.pool_network = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_width),
        )
        self.empty_item = nn.Parameter(torch.zeros(hidden_width))

    def encode_record_sets(
        self,
        record_sets: Sequence[Sequence[Any]],
        *,
        namespaces: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if not record_sets:
            raise ValueError("at least one record set is required")
        if namespaces is None:
            namespaces = ("",) * len(record_sets)
        if len(namespaces) != len(record_sets):
            raise ValueError("namespaces and record_sets must have equal length")
        reference = next(self.item_network.parameters())
        rows: list[torch.Tensor] = []
        for records, namespace in zip(record_sets, namespaces, strict=True):
            if not isinstance(namespace, str):
                raise TypeError("state namespace must be a string")
            raw = self.record_encoder.encode_records(
                tuple(records),
                device=reference.device,
                dtype=reference.dtype,
            )
            expected = (len(records), self.input_width)
            if raw.shape != expected:
                raise ValueError(
                    f"record encoder returned {tuple(raw.shape)}; expected {expected}"
                )
            if not bool(torch.isfinite(raw).all().item()):
                raise ValueError("record encoder returned non-finite features")
            if len(records):
                items = self.item_network(raw)
                summed = items.sum(dim=0) / math.sqrt(len(records))
                mean = items.mean(dim=0)
                maximum = items.max(dim=0).values
            else:
                summed = self.empty_item
                mean = self.empty_item
                maximum = self.empty_item
            namespace_features = _hash_texts(
                (f"namespace:{namespace}",),
                self.input_width,
                device=reference.device,
                dtype=reference.dtype,
            )[0]
            count = torch.tensor(
                [math.log1p(len(records))],
                device=reference.device,
                dtype=reference.dtype,
            )
            rows.append(
                self.pool_network(
                    torch.cat((summed, mean, maximum, namespace_features, count))
                )
            )
        return torch.stack(rows)

    def forward(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            raise ValueError("at least one state is required")
        try:
            records = [state.records for state in states]
            namespaces = [state.namespace for state in states]
        except AttributeError as error:
            raise TypeError("states must expose namespace and records") from error
        return self.encode_record_sets(records, namespaces=namespaces)


class GoalSetEncoder(nn.Module):
    """Encode required/forbidden record sets and exactness compositionally."""

    def __init__(
        self,
        set_encoder: PermutationInvariantSetEncoder,
        *,
        output_width: int,
    ) -> None:
        super().__init__()
        if output_width <= 0:
            raise ValueError("output_width must be positive")
        self.set_encoder = set_encoder
        self.output_width = output_width
        width = set_encoder.output_width
        self.projection = nn.Sequential(
            nn.LayerNorm(width * 2 + 1),
            nn.Linear(width * 2 + 1, max(width, output_width)),
            nn.SiLU(),
            nn.Linear(max(width, output_width), output_width),
        )

    def forward(self, goals: Sequence[Any]) -> torch.Tensor:
        if not goals:
            raise ValueError("at least one goal is required")
        try:
            required = [goal.required for goal in goals]
            forbidden = [goal.forbidden for goal in goals]
            required_namespaces = [f"{goal.namespace}|required" for goal in goals]
            forbidden_namespaces = [f"{goal.namespace}|forbidden" for goal in goals]
            exact_values = [bool(goal.exact) for goal in goals]
        except AttributeError as error:
            raise TypeError(
                "goals must expose namespace, required, forbidden, and exact"
            ) from error
        required_features = self.set_encoder.encode_record_sets(
            required,
            namespaces=required_namespaces,
        )
        forbidden_features = self.set_encoder.encode_record_sets(
            forbidden,
            namespaces=forbidden_namespaces,
        )
        exact = torch.tensor(
            exact_values,
            device=required_features.device,
            dtype=required_features.dtype,
        ).unsqueeze(-1)
        return self.projection(
            torch.cat((required_features, forbidden_features, exact), dim=-1)
        )


class SchemaEncoder(nn.Module):
    """Derive operator embeddings from complete public schema structure."""

    def __init__(
        self,
        *,
        hash_width: int = 192,
        hidden_width: int = 128,
        output_width: int = 128,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or output_width <= 0:
            raise ValueError("schema-encoder widths must be positive")
        self.feature_encoder = FrozenHashTextEncoder(hash_width)
        self.output_width = output_width
        self._canonical_text_cache: dict[int, tuple[Any, str]] = {}
        self.projection = nn.Sequential(
            nn.LayerNorm(hash_width),
            nn.Linear(hash_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_width),
        )

    def _canonical_text(self, schema: Any) -> str:
        dataclass_parameters = getattr(type(schema), "__dataclass_params__", None)
        if dataclass_parameters is None or not dataclass_parameters.frozen:
            return canonical_schema_text(schema)
        key = id(schema)
        cached = self._canonical_text_cache.get(key)
        if cached is not None:
            if cached[0] is not schema:
                raise RuntimeError("schema text cache identity collision")
            return cached[1]
        text = canonical_schema_text(schema)
        if len(self._canonical_text_cache) >= 4_096:
            self._canonical_text_cache.pop(next(iter(self._canonical_text_cache)))
        self._canonical_text_cache[key] = (schema, text)
        return text

    def forward(self, schemas: Sequence[Any]) -> torch.Tensor:
        if not schemas:
            raise ValueError("at least one operator schema is required")
        reference = next(self.projection.parameters())
        raw = self.feature_encoder.encode_texts(
            [self._canonical_text(schema) for schema in schemas],
            device=reference.device,
            dtype=reference.dtype,
        )
        return self.projection(raw)

    def encode_values(self, values: Sequence[str]) -> torch.Tensor:
        """Embed candidate argument values through the same structural path."""

        if not values:
            raise ValueError("at least one value is required")
        reference = next(self.projection.parameters())
        raw = self.feature_encoder.encode_texts(
            [json.dumps({"value": value}, separators=(",", ":")) for value in values],
            device=reference.device,
            dtype=reference.dtype,
        )
        return self.projection(raw)


class GoalConditionedOperatorProposer(nn.Module):
    """Pointer-style scorer over schema-derived operator candidates."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.query = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.keys = nn.Linear(width, width, bias=False)
        self.bias = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        operators: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states, goals, operators = _validate_context_tensors(states, goals, operators)
        context = torch.cat((states, goals, goals - states), dim=-1)
        query = self.query(context).unsqueeze(1)
        logits = (self.keys(operators) * query).sum(dim=-1) / math.sqrt(self.width)
        pair_features = torch.cat(
            (
                context.unsqueeze(1).expand(-1, operators.shape[1], -1),
                operators,
            ),
            dim=-1,
        )
        logits = logits + self.bias(pair_features).squeeze(-1)
        return _apply_candidate_mask(logits, mask, "operator mask")


@dataclass(frozen=True, slots=True)
class PrimitiveScores:
    """Pointer logits for primitive action schemas and their arguments."""

    action_logits: torch.Tensor
    argument_logits: torch.Tensor


class PrimitiveDecoder(nn.Module):
    """Shared context-conditioned pointer decoder for actions and arguments."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.context = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_query = nn.Linear(width, width, bias=False)
        self.action_key = nn.Linear(width, width, bias=False)
        self.argument_query = nn.Linear(width, width, bias=False)
        self.action_to_argument = nn.Linear(width, width, bias=False)
        self.argument_key = nn.Linear(width, width, bias=False)

    def forward(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        selected_operators: torch.Tensor,
        action_candidates: torch.Tensor,
        argument_candidates: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        argument_mask: torch.Tensor | None = None,
    ) -> PrimitiveScores:
        if states.ndim != 2 or goals.shape != states.shape:
            raise ValueError("states and goals must share shape [batch, width]")
        if selected_operators.shape != states.shape:
            raise ValueError("selected_operators must share the state shape")
        if states.shape[-1] != self.width:
            raise ValueError("context width does not match the decoder")
        actions = _expand_candidates(action_candidates, states.shape[0], self.width)
        arguments = _expand_argument_candidates(
            argument_candidates,
            states.shape[0],
            actions.shape[1],
            self.width,
        )
        _require_same_device_dtype(states, goals, selected_operators, actions, arguments)
        context = self.context(torch.cat((states, goals, selected_operators), dim=-1))
        action_logits = (
            self.action_key(actions) * self.action_query(context).unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(self.width)
        action_logits = _apply_candidate_mask(action_logits, action_mask, "action mask")

        argument_query = self.argument_query(context).unsqueeze(1)
        argument_query = argument_query + self.action_to_argument(actions)
        argument_logits = (
            self.argument_key(arguments) * argument_query.unsqueeze(2)
        ).sum(dim=-1) / math.sqrt(self.width)
        argument_logits = _apply_candidate_mask(
            argument_logits,
            argument_mask,
            "argument mask",
        )
        return PrimitiveScores(action_logits, argument_logits)


@dataclass(frozen=True, slots=True)
class OperatorPredictions:
    """All independently ablatable predictions for one candidate set."""

    state_embeddings: torch.Tensor
    goal_embeddings: torch.Tensor
    operator_embeddings: torch.Tensor
    initiation_logits: torch.Tensor
    effect_embeddings: torch.Tensor
    predecessor_embeddings: torch.Tensor
    termination_logits: torch.Tensor
    proposer_logits: torch.Tensor


class NeuralOperatorCore(nn.Module):
    """Shared neural trunk for operator learning, proposal, and execution.

    Effect prediction is deliberately goal-independent: a requested goal may
    prioritize an operator, but it cannot bend that operator's learned state
    transition toward the answer.
    """

    def __init__(
        self,
        record_encoder: RecordEncoder | None = None,
        *,
        width: int = 128,
        hidden_width: int = 192,
        schema_hash_width: int = 192,
    ) -> None:
        super().__init__()
        if width <= 0 or hidden_width <= 0:
            raise ValueError("core widths must be positive")
        if record_encoder is None:
            record_encoder = FrozenHashTextEncoder(max(64, width))
        self.width = width
        self.state_encoder = PermutationInvariantSetEncoder(
            record_encoder,
            hidden_width=hidden_width,
            output_width=width,
        )
        self.goal_encoder = GoalSetEncoder(self.state_encoder, output_width=width)
        self.schema_encoder = SchemaEncoder(
            hash_width=schema_hash_width,
            hidden_width=hidden_width,
            output_width=width,
        )
        directed_width = width * 2 + 2
        self.initiation_head = _scalar_head(directed_width, hidden_width)
        self.effect_head = _vector_head(directed_width, hidden_width, width)
        self.termination_head = _scalar_head(width * 3, hidden_width)
        self.proposer = GoalConditionedOperatorProposer(width)
        self.primitive_decoder = PrimitiveDecoder(width)

    def encode_states(self, states: Sequence[Any]) -> torch.Tensor:
        return self.state_encoder(states)

    def encode_goals(self, goals: Sequence[Any]) -> torch.Tensor:
        return self.goal_encoder(goals)

    def encode_goal_states(self, goals: Sequence[Any]) -> torch.Tensor:
        """Encode exact goals in state space for a backward-search root.

        A partial predicate set is not a concrete state and therefore cannot
        seed backward dynamics without a separately learned completion model.
        """

        if not goals:
            raise ValueError("at least one goal is required")
        try:
            exact = [goal.exact for goal in goals]
            records = [goal.required for goal in goals]
            namespaces = [goal.namespace for goal in goals]
        except AttributeError as error:
            raise TypeError(
                "goals must expose namespace, required, and exact"
            ) from error
        if not all(value is True for value in exact):
            raise ValueError(
                "bidirectional goal roots require exact goals or a learned "
                "goal-completion model"
            )
        return self.state_encoder.encode_record_sets(
            records,
            namespaces=namespaces,
        )

    def encode_operators(self, schemas: Sequence[Any]) -> torch.Tensor:
        return self.schema_encoder(schemas)

    def initiation_logits(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        states, operators = _validate_state_operator_tensors(
            states,
            operators,
            self.width,
        )
        direction = _direction_features(
            reverse,
            batch=states.shape[0],
            candidates=operators.shape[1],
            reference=states,
        )
        expanded_states = states.unsqueeze(1).expand(-1, operators.shape[1], -1)
        features = torch.cat((expanded_states, operators, direction), dim=-1)
        return self.initiation_head(features).squeeze(-1)

    def predict_effects(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        states, operators = _validate_state_operator_tensors(
            states,
            operators,
            self.width,
        )
        direction = _direction_features(
            reverse,
            batch=states.shape[0],
            candidates=operators.shape[1],
            reference=states,
        )
        expanded_states = states.unsqueeze(1).expand(-1, operators.shape[1], -1)
        features = torch.cat((expanded_states, operators, direction), dim=-1)
        return expanded_states + self.effect_head(features)

    def termination_logits(
        self,
        candidate_states: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        squeezed = False
        if candidate_states.ndim == 2:
            candidate_states = candidate_states.unsqueeze(1)
            squeezed = True
        if candidate_states.ndim != 3 or candidate_states.shape[-1] != self.width:
            raise ValueError("candidate_states must have shape [batch, candidates, width]")
        if goals.ndim != 2 or goals.shape != (
            candidate_states.shape[0],
            self.width,
        ):
            raise ValueError("goals must have shape [batch, width]")
        _require_same_device_dtype(candidate_states, goals)
        expanded_goals = goals.unsqueeze(1).expand_as(candidate_states)
        features = torch.cat(
            (candidate_states, expanded_goals, expanded_goals - candidate_states),
            dim=-1,
        )
        logits = self.termination_head(features).squeeze(-1)
        return logits.squeeze(1) if squeezed else logits

    def proposer_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        operators: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.proposer(states, goals, operators, mask=mask)

    def forward(
        self,
        states: Sequence[Any],
        goals: Sequence[Any],
        operators: Sequence[Any],
    ) -> OperatorPredictions:
        if len(states) != len(goals):
            raise ValueError("states and goals must have equal batch size")
        state_embeddings = self.encode_states(states)
        goal_embeddings = self.encode_goals(goals)
        operator_embeddings = self.encode_operators(operators)
        initiation = self.initiation_logits(state_embeddings, operator_embeddings)
        effects = self.predict_effects(state_embeddings, operator_embeddings)
        predecessors = self.predict_effects(
            state_embeddings,
            operator_embeddings,
            reverse=True,
        )
        termination = self.termination_logits(effects, goal_embeddings)
        proposal = self.proposer_logits(
            state_embeddings,
            goal_embeddings,
            operator_embeddings,
        )
        return OperatorPredictions(
            state_embeddings=state_embeddings,
            goal_embeddings=goal_embeddings,
            operator_embeddings=operator_embeddings,
            initiation_logits=initiation,
            effect_embeddings=effects,
            predecessor_embeddings=predecessors,
            termination_logits=termination,
            proposer_logits=proposal,
        )


def _scalar_head(input_width: int, hidden_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_width),
        nn.Linear(input_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, 1),
    )


def _vector_head(
    input_width: int,
    hidden_width: int,
    output_width: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_width),
        nn.Linear(input_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, output_width),
    )


def _validate_state_operator_tensors(
    states: torch.Tensor,
    operators: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or states.shape[-1] != width:
        raise ValueError("states must have shape [batch, width]")
    operators = _expand_candidates(operators, states.shape[0], width)
    _require_same_device_dtype(states, operators)
    if not bool(torch.isfinite(states).all().item()) or not bool(
        torch.isfinite(operators).all().item()
    ):
        raise ValueError("state and operator embeddings must be finite")
    return states, operators


def _validate_context_tensors(
    states: torch.Tensor,
    goals: torch.Tensor,
    operators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or goals.shape != states.shape:
        raise ValueError("states and goals must share shape [batch, width]")
    operators = _expand_candidates(
        operators,
        states.shape[0],
        states.shape[-1],
    )
    _require_same_device_dtype(states, goals, operators)
    return states, goals, operators


def _expand_candidates(
    candidates: torch.Tensor,
    batch_size: int,
    width: int,
) -> torch.Tensor:
    if candidates.ndim == 2:
        if candidates.shape[1] != width:
            raise ValueError("candidate width does not match context width")
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)
    elif candidates.ndim == 3:
        if candidates.shape[0] != batch_size or candidates.shape[2] != width:
            raise ValueError("batched candidates must have shape [batch, count, width]")
    else:
        raise ValueError("candidates must have shape [count, width] or [batch, count, width]")
    if candidates.shape[1] <= 0:
        raise ValueError("at least one candidate is required")
    return candidates


def _expand_argument_candidates(
    candidates: torch.Tensor,
    batch_size: int,
    action_count: int,
    width: int,
) -> torch.Tensor:
    if candidates.ndim == 3:
        if candidates.shape[0] != action_count or candidates.shape[2] != width:
            raise ValueError(
                "arguments must have shape [actions, values, width]"
            )
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif candidates.ndim == 4:
        if candidates.shape[0] != batch_size or candidates.shape[1] != action_count or candidates.shape[3] != width:
            raise ValueError(
                "batched arguments must have shape [batch, actions, values, width]"
            )
    else:
        raise ValueError(
            "argument candidates must have rank three or four"
        )
    if candidates.shape[2] <= 0:
        raise ValueError("each action needs at least one argument candidate")
    return candidates


def _apply_candidate_mask(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")
    if mask.shape == logits.shape[1:]:
        mask = mask.unsqueeze(0).expand_as(logits)
    elif mask.shape != logits.shape:
        raise ValueError(f"{name} shape does not match logits")
    if mask.device != logits.device:
        raise ValueError(f"{name} must share the logits device")
    return logits.masked_fill(~mask, -torch.inf)


def _direction_features(
    reverse: bool,
    *,
    batch: int,
    candidates: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = (0.0, 1.0) if reverse else (1.0, 0.0)
    return torch.tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    ).view(1, 1, 2).expand(batch, candidates, -1)


def _require_same_device_dtype(*tensors: torch.Tensor) -> None:
    reference = tensors[0]
    for tensor in tensors[1:]:
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError("all neural inputs must share device and dtype")


__all__ = [
    "FrozenHashRecordEncoder",
    "FrozenHashTextEncoder",
    "GoalConditionedOperatorProposer",
    "GoalSetEncoder",
    "NeuralOperatorCore",
    "OperatorPredictions",
    "PermutationInvariantSetEncoder",
    "PrimitiveDecoder",
    "PrimitiveScores",
    "RecordEncoder",
    "SchemaEncoder",
    "canonical_schema_text",
]
