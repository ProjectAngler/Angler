"""Bridge between Jenny's learned choices, local inference, and tool operations.

The bridge projects one exact, allowlisted tool manifest into Jenny's dynamic
affordance registry and records selected calls in :mod:`tool_operations`.  It
never invokes a tool, chooses an affordance, assigns reward, mutates learned
state directly, or projects cognitive memory.  Its optional model adapter can
call only an uncredentialed loopback OpenAI-compatible inference endpoint.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Literal, Mapping, Protocol

from .persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    DynamicAffordanceRegistry,
    ObservableConsequence,
)
from .tool_operations import (
    _connect_private_sqlite,
    _ensure_private_sqlite_sidecars,
    ToolCallReservation,
    ToolDefinition,
    ToolEvent,
    ToolManifest,
    ToolOperationStore,
)


DEFAULT_ALLOWED_PERMISSION_SCOPES = frozenset(("external.web.read",))
CONTROLLER_TOOL_DESCRIPTION_MAX_BYTES = 1_024
CONTROLLER_MANIFEST_DESCRIPTION_MAX_BYTES = 32_768
COGNITIVE_TOOL_RESULT_MAX_BYTES = 16_384
NATIVE_TURN_TRACE_MAX_BYTES = 128 * 1_024
NATIVE_TURN_MAX_TOOL_CALLS = 8
NATIVE_TOOL_TURN_CONTRACT = "ANG-CTR-JENNY-NATIVE-TOOL-TURN-001@0.1.0"
NATIVE_OPENCLAW_CATALOG_MAX_BYTES = 256 * 1_024
NATIVE_MODEL_CHOICE_MAX_BYTES = 128 * 1_024
NATIVE_MODEL_INPUT_MAXIMUM_CHARACTERS = 262_144

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*$")
_NATIVE_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,255}$")
_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_STATUSES = frozenset(("COMPLETED", "ERROR", "DENIED"))
_TURN_STATUSES = frozenset(
    ("OPEN", "TOOL_PENDING", "CONTINUATION_READY", "FINAL_READY", "COMMITTED")
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object(value: str, label: str) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains a non-finite constant: {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    if type(decoded) is not dict:
        raise ValueError(f"{label} must encode an object")
    if _canonical(decoded) != value:
        raise ValueError(f"{label} must be canonical JSON")
    return decoded


def _decoded_object(value: str, label: str) -> dict[str, object]:
    """Decode an object safely without requiring its source to be canonical."""

    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains a non-finite constant: {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    if type(decoded) is not dict:
        raise ValueError(f"{label} must encode an object")
    return decoded


def _array(value: str, label: str) -> list[object]:
    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains a non-finite constant: {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    if type(decoded) is not list:
        raise ValueError(f"{label} must encode an array")
    if _canonical(decoded) != value:
        raise ValueError(f"{label} must be canonical JSON")
    return decoded


def _reference(value: str, label: str) -> str:
    if type(value) is not str or not _REFERENCE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _visible_text(
    value: str,
    label: str,
    maximum_bytes: int,
    *,
    allow_empty: bool = False,
) -> str:
    if (
        type(value) is not str
        or len(value.encode("utf-8")) > maximum_bytes
        or (not allow_empty and not value.strip())
    ):
        raise ValueError(f"{label} must be bounded text")
    return value


class DeferredEffectSupervisor(Protocol):
    """Only the exact supervisor seam needed by this bridge."""

    def pending_deferred_effect(
        self,
    ) -> tuple[Affordance, AffordanceRequest] | None: ...


@dataclass(frozen=True, slots=True)
class DecodedToolOperation:
    """JSON-ready metadata for one exact, already-reserved operation."""

    reservation: ToolCallReservation
    definition: ToolDefinition
    affordance_id: str
    arguments: Mapping[str, object]
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    events: tuple[ToolEvent, ...]


@dataclass(frozen=True, slots=True)
class NativeOpenClawToolCard:
    """One provider-native card with an honest, separate effect classification."""

    name: str
    description: str
    input_schema_json: str
    effect_classification: Literal[
        "HOST_SUPERVISED_UNKNOWN",
        "HOST_DECLARED_NON_EFFECT",
        "HOST_DECLARED_EXTERNAL_EFFECT",
    ]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _NATIVE_TOOL_NAME.fullmatch(self.name):
            raise ValueError("native OpenClaw tool name is malformed")
        _visible_text(self.description, "native OpenClaw description", 16_384)
        _object(self.input_schema_json, "native OpenClaw input schema")
        if self.effect_classification not in (
            "HOST_SUPERVISED_UNKNOWN",
            "HOST_DECLARED_NON_EFFECT",
            "HOST_DECLARED_EXTERNAL_EFFECT",
        ):
            raise ValueError("native OpenClaw effect classification is unsupported")

    def provider_payload(self) -> dict[str, object]:
        return {
            "description": self.description,
            "input_schema": _object(
                self.input_schema_json, "native OpenClaw input schema"
            ),
            "name": self.name,
        }

    def policy_payload(self) -> dict[str, object]:
        return {
            **self.provider_payload(),
            "effect_classification": self.effect_classification,
        }


class NativeOpenClawCatalog:
    """Owner-present native tools, distinct from read-only registry affordances.

    These cards are returned only for a human-initiated, bridge-authenticated,
    owner-present turn.  They are deliberately never registered in the global
    affordance registry, so the scheduler/life loop cannot discover or invoke
    them.  OpenClaw's guarded tool cards and session policy remain the executor
    permission boundary; this catalog makes no execution grant.
    """

    def __init__(
        self,
        tools: tuple[Mapping[str, object], ...],
        *,
        catalog_hash: str | None = None,
    ) -> None:
        if type(tools) is not tuple or not tools or len(tools) > 256:
            raise ValueError("native OpenClaw tools must contain 1 through 256 cards")
        cards: list[NativeOpenClawToolCard] = []
        names: set[str] = set()
        for raw in tools:
            if not isinstance(raw, Mapping):
                raise TypeError("native OpenClaw tool cards must be mappings")
            keys = set(raw)
            if keys not in (
                {"name", "description", "input_schema"},
                {"name", "description", "input_schema", "external_effect"},
            ):
                raise ValueError("native OpenClaw tool card fields are unsupported")
            name, description, input_schema = (
                raw["name"],
                raw["description"],
                raw["input_schema"],
            )
            if type(name) is not str or not _NATIVE_TOOL_NAME.fullmatch(name):
                raise ValueError("native OpenClaw tool name is malformed")
            if name in names:
                raise ValueError("native OpenClaw tool names must be unique")
            names.add(name)
            if type(description) is not str:
                raise TypeError("native OpenClaw description must be text")
            if type(input_schema) is not dict:
                raise TypeError("native OpenClaw input_schema must be an object")
            schema_json = _canonical(input_schema)
            # Reparse to reject any non-finite values hidden in nested input.
            _object(schema_json, "native OpenClaw input schema")
            supplied_effect = raw.get("external_effect")
            if "external_effect" not in raw:
                classification = "HOST_SUPERVISED_UNKNOWN"
            elif type(supplied_effect) is not bool:
                raise TypeError("native OpenClaw external_effect must be boolean")
            elif supplied_effect:
                classification = "HOST_DECLARED_EXTERNAL_EFFECT"
            else:
                classification = "HOST_DECLARED_NON_EFFECT"
            cards.append(
                NativeOpenClawToolCard(
                    name=name,
                    description=description,
                    input_schema_json=schema_json,
                    effect_classification=classification,  # type: ignore[arg-type]
                )
            )
        provider_json = _canonical([card.provider_payload() for card in cards])
        if len(provider_json.encode("utf-8")) > NATIVE_OPENCLAW_CATALOG_MAX_BYTES:
            raise ValueError("native OpenClaw catalog exceeds 256 KiB")
        observed_hash = _digest(provider_json)
        if catalog_hash is not None:
            _reference(catalog_hash, "native OpenClaw catalog_hash")
            if catalog_hash != observed_hash:
                raise ValueError("native OpenClaw catalog hash differs")
        self._cards = tuple(cards)
        self._provider_json = provider_json
        self._catalog_hash = observed_hash
        self._policy_ref = _digest(
            _canonical([card.policy_payload() for card in cards])
        )

    @property
    def cards(self) -> tuple[NativeOpenClawToolCard, ...]:
        return self._cards

    @property
    def catalog_hash(self) -> str:
        return self._catalog_hash

    @property
    def policy_ref(self) -> str:
        """Bind host declarations separately from the provider catalog hash."""

        return self._policy_ref

    @property
    def guarded_affordance(self) -> Affordance:
        """Return one proposal boundary for the host-supervised native catalog."""

        tool_names = [card.name for card in self.cards]
        description: dict[str, object] = {
            "catalog_hash": self.catalog_hash,
            "effect_classification": "HOST_GUARDED_PER_TOOL_CARD",
            "policy_ref": self.policy_ref,
            "purpose": "Request an owner-present guarded tool turn when its capabilities are needed.",
            "tool_count": len(self.cards),
            "tool_names": tool_names,
        }
        encoded = _canonical(description)
        if len(encoded) > CONTROLLER_TOOL_DESCRIPTION_MAX_BYTES:
            namespaces = sorted(
                {
                    re.split(r"[.:/_-]", name, maxsplit=1)[0]
                    for name in tool_names
                }
            )
            description.pop("tool_names")
            description["tool_name_set_ref"] = _digest(_canonical(tool_names))
            description["tool_namespaces"] = namespaces
            encoded = _canonical(description)
        return Affordance(
            affordance_id="openclaw.supervised.turn",
            disposition="ACT",
            description=encoded,
            permission_scope="external.openclaw.guarded",
            external_effect=True,
            authorization_mode="HOST_GUARDED",
        )

    def register_guarded_affordance(
        self, registry: DynamicAffordanceRegistry
    ) -> None:
        """Register only the aggregate deferred proposal, never 48 executors.

        The learned-cycle caller must exclude ``HOST_GUARDED`` affordances from
        scheduler observations.  Native cards are released separately through
        :meth:`tools_for_supervised_turn` only after the human/session checks.
        """

        if not isinstance(registry, DynamicAffordanceRegistry):
            raise TypeError("registry must be a DynamicAffordanceRegistry")
        affordance = self.guarded_affordance
        if affordance.affordance_id in {
            item.affordance_id for item in registry.definitions()
        }:
            raise ValueError("native OpenClaw affordance collides with the registry")
        registry.register(
            affordance,
            observable_source_ref=self.catalog_hash,
            observable_source_kind="TOOL",
            execution_mode="DEFERRED",
        )

    def tools_for_supervised_turn(
        self,
        *,
        observation_source: str,
        bridge_authenticated: bool,
        owner_present: bool,
    ) -> tuple[dict[str, object], ...]:
        if (
            observation_source != "HUMAN"
            or bridge_authenticated is not True
            or owner_present is not True
        ):
            raise PermissionError(
                "native OpenClaw tools require an owner-present authenticated human turn"
            )
        decoded = json.loads(self._provider_json)
        if type(decoded) is not list or any(type(item) is not dict for item in decoded):
            raise RuntimeError("stored native OpenClaw catalog is malformed")
        return tuple(decoded)

    def final_receipt(
        self,
        turn: "NativeToolTurn",
        request: AffordanceRequest,
    ) -> AffordanceReceipt:
        """Build the sole completion receipt after a final response is durable."""

        if not isinstance(turn, NativeToolTurn) or turn.status != "FINAL_READY":
            raise ValueError("native OpenClaw turn is not FINAL_READY")
        if not isinstance(request, AffordanceRequest):
            raise TypeError("request must be an AffordanceRequest")
        if (
            turn.catalog_hash != self.catalog_hash
            or turn.original_request_ref != request.idempotency_key
            or request.affordance_id != self.guarded_affordance.affordance_id
        ):
            raise ValueError("final native turn differs from its Jenny reservation")
        if turn.final_response is None or turn.aggregate_observation_json is None:
            raise RuntimeError("FINAL_READY turn lost its final material")
        operation_refs = sorted(
            {
                item["operation_ref"]
                for item in turn.trace
                if item.get("kind") == "TOOL_RESULT"
            }
        )
        evidence_refs = sorted(
            {
                item["event_ref"]
                for item in turn.trace
                if item.get("kind") == "TOOL_RESULT"
            }
        )
        if any(type(item) is not str for item in (*operation_refs, *evidence_refs)):
            raise RuntimeError("native turn result references are malformed")
        observed = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind="TOOL",
            source_ref=self.catalog_hash,
            observation_json=turn.aggregate_observation_json,
            artifact_refs=tuple(operation_refs),  # type: ignore[arg-type]
            evidence_refs=tuple(evidence_refs),  # type: ignore[arg-type]
        )
        return AffordanceReceipt(
            status="COMPLETED",
            output=turn.final_response,
            consequence=(),
            observable_consequence=observed,
        )


class NativeToolModelTransport(Protocol):
    """Injectable JSON transport for one loopback model request."""

    def __call__(
        self,
        endpoint: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class NativeModelToolCall:
    """One exact OpenAI function call selected by the native model."""

    call_id: str
    tool_name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _visible_text(self.call_id, "native model call_id", 256)
        if any(ord(character) < 33 or ord(character) > 126 for character in self.call_id):
            raise ValueError("native model call_id must use visible ASCII")
        if type(self.tool_name) is not str or not _NATIVE_TOOL_NAME.fullmatch(
            self.tool_name
        ):
            raise ValueError("native model tool_name must be a stable identifier")
        _object(self.arguments_json, "native model tool arguments")
        if len(self.arguments_json.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
            raise ValueError("native model tool arguments exceed the cognitive ceiling")


@dataclass(frozen=True, slots=True)
class NativeToolModelResult:
    """Exclusive final-text or tool-call outcome from one model pass."""

    route: Literal["FINAL", "TOOL_REQUESTED"]
    final_content: str | None
    tool_calls: tuple[NativeModelToolCall, ...]
    choice_json: str
    finish_reason: str | None

    def __post_init__(self) -> None:
        if self.route not in ("FINAL", "TOOL_REQUESTED"):
            raise ValueError("native model route is unsupported")
        if type(self.tool_calls) is not tuple or any(
            not isinstance(item, NativeModelToolCall) for item in self.tool_calls
        ):
            raise TypeError("native model tool_calls must be a tuple")
        if len(self.tool_calls) > NATIVE_TURN_MAX_TOOL_CALLS:
            raise ValueError("native model returned more than eight tool calls")
        if len({item.call_id for item in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("native model call identifiers must be unique")
        _object(self.choice_json, "native model choice")
        if len(self.choice_json.encode("utf-8")) > NATIVE_MODEL_CHOICE_MAX_BYTES:
            raise ValueError("native model choice exceeds 128 KiB")
        if self.finish_reason is not None:
            _visible_text(
                self.finish_reason,
                "native model finish_reason",
                256,
                allow_empty=True,
            )
        if self.route == "FINAL":
            if self.tool_calls or self.final_content is None:
                raise ValueError("FINAL must contain only final content")
            _visible_text(
                self.final_content,
                "native model final content",
                COGNITIVE_TOOL_RESULT_MAX_BYTES,
            )
        elif self.final_content is not None or not self.tool_calls:
            raise ValueError("TOOL_REQUESTED must contain only tool calls")


class NativeToolModel(Protocol):
    """Protocol used by the native pending-turn coordinator."""

    def generate(
        self,
        *,
        system_context: str,
        user_message: str,
        tools: tuple[Mapping[str, object], ...],
        catalog_hash: str,
        prior_trace_json: str = "[]",
        max_new_tokens: int | None = None,
    ) -> NativeToolModelResult: ...


def _loopback_openai_json_transport(
    endpoint: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> Mapping[str, object]:
    body = _canonical(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(512 * 1_024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("local native tool model request failed") from exc
    if len(raw) > 512 * 1_024:
        raise RuntimeError("local native tool model response exceeds 512 KiB")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("local native tool model returned malformed JSON") from exc
    if type(value) is not dict:
        raise RuntimeError("local native tool model response must be an object")
    return value


class OpenAICompatibleNativeToolModel:
    """Strict native function-calling adapter for one loopback model server."""

    def __init__(
        self,
        *,
        endpoint: str,
        served_model: str,
        transport: NativeToolModelTransport | None = None,
        maximum_input_characters: int = NATIVE_MODEL_INPUT_MAXIMUM_CHARACTERS,
        maximum_output_tokens: int = 4_096,
        timeout_seconds: float = 180.0,
        maximum_choice_bytes: int = NATIVE_MODEL_CHOICE_MAX_BYTES,
    ) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"
        ):
            raise ValueError("endpoint must be an uncredentialed loopback /v1 HTTP URL")
        if type(served_model) is not str or not served_model.strip():
            raise ValueError("served_model must be non-empty text")
        if (
            type(maximum_input_characters) is not int
            or not 1_024 <= maximum_input_characters <= 2_000_000
        ):
            raise ValueError("maximum_input_characters is outside its boundary")
        if (
            type(maximum_output_tokens) is not int
            or not 1 <= maximum_output_tokens <= 32_768
        ):
            raise ValueError("maximum_output_tokens must be 1 through 32768")
        if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be in [1, 600]")
        if (
            type(maximum_choice_bytes) is not int
            or not 1_024 <= maximum_choice_bytes <= NATIVE_MODEL_CHOICE_MAX_BYTES
        ):
            raise ValueError("maximum_choice_bytes must be 1 KiB through 128 KiB")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.served_model = served_model
        self.maximum_input_characters = maximum_input_characters
        self.maximum_output_tokens = maximum_output_tokens
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_choice_bytes = maximum_choice_bytes
        self._transport = transport or _loopback_openai_json_transport

    def generate(
        self,
        *,
        system_context: str,
        user_message: str,
        tools: tuple[Mapping[str, object], ...],
        catalog_hash: str,
        prior_trace_json: str = "[]",
        max_new_tokens: int | None = None,
    ) -> NativeToolModelResult:
        if type(system_context) is not str or type(user_message) is not str:
            raise TypeError("system_context and user_message must be text")
        _visible_text(
            system_context,
            "native model system context",
            self.maximum_input_characters,
        )
        _visible_text(
            user_message,
            "native model user message",
            self.maximum_input_characters,
        )
        if type(prior_trace_json) is not str:
            raise TypeError("prior_trace_json must be canonical JSON text")
        if len(prior_trace_json.encode("utf-8")) > NATIVE_TURN_TRACE_MAX_BYTES:
            raise ValueError("native prior trace exceeds 128 KiB")
        trace = _array(prior_trace_json, "native prior trace")
        catalog = NativeOpenClawCatalog(tools, catalog_hash=catalog_hash)
        provider_tools = tuple(card.provider_payload() for card in catalog.cards)
        input_characters = (
            len(system_context)
            + len(user_message)
            + len(prior_trace_json)
            + len(_canonical(provider_tools))
        )
        if input_characters > self.maximum_input_characters:
            raise ValueError("native model input exceeds its character ceiling")
        if max_new_tokens is None:
            selected_max_tokens = self.maximum_output_tokens
        elif (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= self.maximum_output_tokens
        ):
            raise ValueError("max_new_tokens exceeds the model output boundary")
        else:
            selected_max_tokens = max_new_tokens
        known_names = frozenset(card.name for card in catalog.cards)
        messages = self._messages_from_trace(
            system_context=system_context,
            user_message=user_message,
            trace=trace,
            known_names=known_names,
        )
        function_tools = [
            {
                "function": {
                    "description": card.description,
                    "name": card.name,
                    "parameters": _object(
                        card.input_schema_json, "native OpenClaw input schema"
                    ),
                },
                "type": "function",
            }
            for card in catalog.cards
        ]
        payload: dict[str, object] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
            "max_tokens": selected_max_tokens,
            "messages": messages,
            "model": self.served_model,
            "seed": 20260902,
            "temperature": 0.0,
            "tool_choice": "auto",
            "tools": function_tools,
            "top_p": 1.0,
        }
        response = self._transport(
            self.endpoint,
            payload,
            self.timeout_seconds,
        )
        return self._parse_response(response, known_names=known_names)

    @staticmethod
    def _messages_from_trace(
        *,
        system_context: str,
        user_message: str,
        trace: list[object],
        known_names: frozenset[str],
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {"content": system_context, "role": "system"},
            {"content": user_message, "role": "user"},
        ]
        pending: dict[str, str] = {}
        seen: set[str] = set()
        call_count = 0
        for entry in trace:
            if type(entry) is not dict or type(entry.get("kind")) is not str:
                raise ValueError("native prior trace entry is malformed")
            kind = entry["kind"]
            if kind == "MODEL_TOOL_REQUESTS":
                if pending:
                    raise ValueError("native prior trace starts a request before results")
                raw_calls = entry.get("calls")
                if type(raw_calls) is not list or not raw_calls:
                    raise ValueError("native prior trace tool request is empty")
                assistant_calls: list[dict[str, object]] = []
                for raw_call in raw_calls:
                    if type(raw_call) is not dict:
                        raise ValueError("native prior trace call is malformed")
                    call_id = raw_call.get("call_id")
                    tool_name = raw_call.get("tool_name")
                    arguments_json = raw_call.get("arguments_json")
                    if type(call_id) is not str or type(tool_name) is not str:
                        raise ValueError("native prior trace call identity is malformed")
                    _visible_text(call_id, "native prior call_id", 256)
                    if (
                        any(ord(character) < 33 or ord(character) > 126 for character in call_id)
                        or call_id in seen
                    ):
                        raise ValueError("native prior trace call identifiers differ")
                    if tool_name not in known_names:
                        raise ValueError("native prior trace names a tool outside the catalog")
                    if type(arguments_json) is not str:
                        raise ValueError("native prior trace arguments are malformed")
                    _object(arguments_json, "native prior trace arguments")
                    seen.add(call_id)
                    pending[call_id] = tool_name
                    assistant_calls.append(
                        {
                            "function": {
                                "arguments": arguments_json,
                                "name": tool_name,
                            },
                            "id": call_id,
                            "type": "function",
                        }
                    )
                    call_count += 1
                    if call_count > NATIVE_TURN_MAX_TOOL_CALLS:
                        raise ValueError("native prior trace exceeds eight tool calls")
                messages.append(
                    {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": assistant_calls,
                    }
                )
            elif kind == "TOOL_RESULT":
                call_id = entry.get("call_id")
                tool_name = entry.get("tool_name")
                result = entry.get("result")
                status = entry.get("status")
                if (
                    type(call_id) is not str
                    or call_id not in pending
                    or type(tool_name) is not str
                    or pending[call_id] != tool_name
                    or type(result) is not dict
                    or status not in _TERMINAL_STATUSES
                ):
                    raise ValueError("native prior trace result differs from its call")
                result_json = _canonical({"result": result, "status": status})
                _object(result_json, "native prior trace result")
                if len(result_json.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
                    raise ValueError("native prior trace result exceeds the cognitive ceiling")
                messages.append(
                    {
                        "content": result_json,
                        "name": tool_name,
                        "role": "tool",
                        "tool_call_id": call_id,
                    }
                )
                del pending[call_id]
            elif kind in ("FINAL_RESPONSE", "COMMITTED"):
                raise ValueError("a finalized native trace cannot be continued")
            else:
                raise ValueError("native prior trace kind is unsupported")
        if pending:
            raise ValueError("native prior trace has outstanding tool results")
        return messages

    def _parse_response(
        self,
        response: Mapping[str, object],
        *,
        known_names: frozenset[str],
    ) -> NativeToolModelResult:
        if not isinstance(response, Mapping):
            raise RuntimeError("native tool model response must be a mapping")
        choices = response.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise RuntimeError("native tool model must return exactly one choice")
        choice = choices[0]
        try:
            choice_json = _canonical(choice)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("native tool model choice is not canonicalizable") from exc
        if len(choice_json.encode("utf-8")) > self.maximum_choice_bytes:
            raise RuntimeError("native tool model choice exceeds its bounded ceiling")
        message = choice.get("message")
        if type(message) is not dict:
            raise RuntimeError("native tool model choice has no message")
        content = message.get("content")
        if content is not None and type(content) is not str:
            raise RuntimeError("native tool model content must be text or null")
        final_content = content if isinstance(content, str) and content.strip() else None
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if type(raw_calls) is not list:
            raise RuntimeError("native tool model tool_calls must be an array")
        if final_content is not None and raw_calls:
            raise RuntimeError("native tool model returned both content and tool calls")
        if len(raw_calls) > NATIVE_TURN_MAX_TOOL_CALLS:
            raise RuntimeError("native tool model returned more than eight tool calls")
        parsed_calls: list[NativeModelToolCall] = []
        for raw_call in raw_calls:
            if type(raw_call) is not dict or raw_call.get("type") != "function":
                raise RuntimeError("native tool model call is not an OpenAI function call")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if type(call_id) is not str or type(function) is not dict:
                raise RuntimeError("native tool model call identity is malformed")
            name = function.get("name")
            arguments = function.get("arguments")
            if type(name) is not str or name not in known_names:
                raise RuntimeError("native tool model selected an unknown tool")
            if type(arguments) is not str:
                raise RuntimeError("native tool model arguments must be JSON text")
            if len(arguments.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
                raise RuntimeError("native tool model arguments exceed the cognitive ceiling")
            try:
                canonical_arguments = _canonical(
                    _decoded_object(arguments, "native tool model arguments")
                )
                parsed_calls.append(
                    NativeModelToolCall(
                        call_id=call_id,
                        tool_name=name,
                        arguments_json=canonical_arguments,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("native tool model function call is malformed") from exc
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and type(finish_reason) is not str:
            raise RuntimeError("native tool model finish_reason is malformed")
        if final_content is not None:
            return NativeToolModelResult(
                route="FINAL",
                final_content=final_content,
                tool_calls=(),
                choice_json=choice_json,
                finish_reason=finish_reason,
            )
        if not parsed_calls:
            raise RuntimeError("native tool model returned neither content nor tool calls")
        if len({item.call_id for item in parsed_calls}) != len(parsed_calls):
            raise RuntimeError("native tool model reused a call identifier")
        return NativeToolModelResult(
            route="TOOL_REQUESTED",
            final_content=None,
            tool_calls=tuple(parsed_calls),
            choice_json=choice_json,
            finish_reason=finish_reason,
        )


@dataclass(frozen=True, slots=True)
class NativeToolCall:
    """One model-selected native provider call inside a pending turn."""

    call_id: str
    tool_name: str
    operation_ref: str
    permission_reservation_ref: str
    arguments_json: str

    def __post_init__(self) -> None:
        _visible_text(self.call_id, "native call_id", 256)
        if any(ord(character) < 33 or ord(character) > 126 for character in self.call_id):
            raise ValueError("native call_id must use visible ASCII")
        if type(self.tool_name) is not str or not _NATIVE_TOOL_NAME.fullmatch(self.tool_name):
            raise ValueError("native tool_name must be a stable identifier")
        _reference(self.operation_ref, "native operation_ref")
        _reference(
            self.permission_reservation_ref, "native permission_reservation_ref"
        )
        _object(self.arguments_json, "native tool arguments")
        if len(self.arguments_json.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
            raise ValueError("native tool arguments exceed the cognitive bridge ceiling")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "arguments_json": self.arguments_json,
            "call_id": self.call_id,
            "operation_ref": self.operation_ref,
            "permission_reservation_ref": self.permission_reservation_ref,
            "tool_name": self.tool_name,
        }


@dataclass(frozen=True, slots=True)
class NativeToolTurn:
    """One durable provider turn that commits cognitively only when final."""

    turn_id: str
    revision: int
    status: Literal[
        "OPEN", "TOOL_PENDING", "CONTINUATION_READY", "FINAL_READY", "COMMITTED"
    ]
    catalog_hash: str
    original_request_ref: str
    original_observation_ref: str
    trace_json: str
    outstanding_calls: tuple[NativeToolCall, ...] = ()
    permission_reservation_refs: tuple[str, ...] = ()
    final_response: str | None = None
    aggregate_observation_json: str | None = None
    committed_episode_ref: str | None = None

    def __post_init__(self) -> None:
        if type(self.turn_id) is not str or not re.fullmatch(r"turn\.[0-9a-f]{64}", self.turn_id):
            raise ValueError("turn_id must be derived from one SHA-256 request reference")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("turn revision must be non-negative")
        if self.status not in _TURN_STATUSES:
            raise ValueError("native turn status is unsupported")
        _reference(self.catalog_hash, "native catalog_hash")
        _reference(self.original_request_ref, "native original_request_ref")
        _reference(self.original_observation_ref, "native original_observation_ref")
        if len(self.trace_json.encode("utf-8")) > NATIVE_TURN_TRACE_MAX_BYTES:
            raise ValueError("native turn trace exceeds 128 KiB")
        trace = _array(self.trace_json, "native turn trace")
        if any(type(item) is not dict for item in trace):
            raise ValueError("native turn trace entries must be objects")
        if type(self.outstanding_calls) is not tuple or any(
            not isinstance(item, NativeToolCall) for item in self.outstanding_calls
        ):
            raise TypeError("outstanding_calls must contain NativeToolCall values")
        call_ids = tuple(item.call_id for item in self.outstanding_calls)
        operation_refs = tuple(item.operation_ref for item in self.outstanding_calls)
        if len(call_ids) != len(set(call_ids)) or len(operation_refs) != len(
            set(operation_refs)
        ):
            raise ValueError("outstanding native calls must be unique")
        if len(self.outstanding_calls) > NATIVE_TURN_MAX_TOOL_CALLS:
            raise ValueError("native turn exceeds eight outstanding calls")
        if (
            type(self.permission_reservation_refs) is not tuple
            or any(type(item) is not str for item in self.permission_reservation_refs)
            or len(self.permission_reservation_refs)
            != len(set(self.permission_reservation_refs))
        ):
            raise ValueError("permission reservation references must be a unique tuple")
        for reference in self.permission_reservation_refs:
            _reference(reference, "native permission reservation reference")
        requested_calls = sum(
            len(item.get("calls", []))
            for item in trace
            if isinstance(item, dict) and item.get("kind") == "MODEL_TOOL_REQUESTS"
        )
        if requested_calls > NATIVE_TURN_MAX_TOOL_CALLS:
            raise ValueError("native turn trace exceeds eight tool calls")
        if self.status == "OPEN" and (trace or self.outstanding_calls):
            raise ValueError("an open native turn cannot already contain tool activity")
        if (self.status == "TOOL_PENDING") != bool(self.outstanding_calls):
            raise ValueError("TOOL_PENDING must exactly match outstanding calls")
        if self.status in ("FINAL_READY", "COMMITTED"):
            if self.outstanding_calls:
                raise ValueError("a final native turn cannot have outstanding calls")
            if self.final_response is None or self.aggregate_observation_json is None:
                raise ValueError("a final native turn requires response and observation")
            _visible_text(
                self.final_response,
                "native final response",
                COGNITIVE_TOOL_RESULT_MAX_BYTES,
            )
            _object(
                self.aggregate_observation_json,
                "native aggregate observation",
            )
            if (
                len(self.aggregate_observation_json.encode("utf-8"))
                > COGNITIVE_TOOL_RESULT_MAX_BYTES
            ):
                raise ValueError("aggregate observation exceeds the cognitive ceiling")
        elif self.final_response is not None or self.aggregate_observation_json is not None:
            raise ValueError("non-final native turn cannot claim a final response")
        if self.status == "COMMITTED":
            if self.committed_episode_ref is None:
                raise ValueError("committed native turn requires an episode reference")
            _reference(self.committed_episode_ref, "committed episode_ref")
        elif self.committed_episode_ref is not None:
            raise ValueError("only a committed native turn may name an episode")

    @property
    def trace(self) -> tuple[dict[str, object], ...]:
        values = _array(self.trace_json, "native turn trace")
        return tuple(values)  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "aggregate_observation_json": self.aggregate_observation_json,
            "catalog_hash": self.catalog_hash,
            "committed_episode_ref": self.committed_episode_ref,
            "contract": NATIVE_TOOL_TURN_CONTRACT,
            "final_response": self.final_response,
            "original_observation_ref": self.original_observation_ref,
            "original_request_ref": self.original_request_ref,
            "outstanding_calls": tuple(
                item.canonical_payload() for item in self.outstanding_calls
            ),
            "permission_reservation_refs": self.permission_reservation_refs,
            "revision": self.revision,
            "status": self.status,
            "trace_json": self.trace_json,
            "turn_id": self.turn_id,
        }

    def provider_metadata(self) -> dict[str, object]:
        """Expose replay-stable turn fields without leaking trace or results."""

        return {
            "catalog_hash": self.catalog_hash,
            "outstanding_calls": [
                {
                    "call_id": item.call_id,
                    "name": item.tool_name,
                    "permission_reservation_ref": item.permission_reservation_ref,
                }
                for item in self.outstanding_calls
            ],
            "permission_reservation_refs": list(
                self.permission_reservation_refs
            ),
            "status": self.status,
            "turn_id": self.turn_id,
            "turn_revision": self.revision,
        }

    @property
    def envelope_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


class NativeToolTurnStore:
    """CAS-revisioned persistence for exactly one active native tool turn.

    This store holds transport continuation state only.  ``FINAL_READY`` is
    durable before the caller performs the sole supervisor completion, and
    ``COMMITTED`` records that completion afterward.  This class never calls a
    supervisor, model, tool, network, memory system, or clock.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        catalog_hash: str,
        tool_names: frozenset[str],
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path")
        _reference(catalog_hash, "native catalog_hash")
        if (
            type(tool_names) is not frozenset
            or not tool_names
            or any(
                type(name) is not str or not _NATIVE_TOOL_NAME.fullmatch(name)
                for name in tool_names
            )
        ):
            raise ValueError("tool_names must be a non-empty identifier frozenset")
        self.database_path = database_path
        self.catalog_hash = catalog_hash
        self.tool_names = tool_names
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = _connect_private_sqlite(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jenny_native_tool_turns (
                    turn_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0,1)),
                    envelope_ref TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_jenny_native_tool_turn
                    ON jenny_native_tool_turns(active) WHERE active = 1;
                """
            )
            connection.commit()
            _ensure_private_sqlite_sidecars(self.database_path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def turn_id_for_request(request_ref: str) -> str:
        _reference(request_ref, "native original_request_ref")
        return "turn." + request_ref[7:]

    def _from_row(self, row: sqlite3.Row) -> NativeToolTurn:
        payload = json.loads(row["payload_json"])
        turn = _native_turn_from_payload(payload)
        if (
            turn.turn_id != row["turn_id"]
            or turn.revision != row["revision"]
            or turn.status != row["status"]
            or (turn.status != "COMMITTED") != bool(row["active"])
            or turn.envelope_ref != row["envelope_ref"]
            or _canonical(turn.canonical_payload()) != row["payload_json"]
        ):
            raise RuntimeError("stored native tool turn failed integrity validation")
        return turn

    def _get(
        self, connection: sqlite3.Connection, turn_id: str
    ) -> NativeToolTurn | None:
        row = connection.execute(
            "SELECT * FROM jenny_native_tool_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        return None if row is None else self._from_row(row)

    def get(self, turn_id: str) -> NativeToolTurn | None:
        if type(turn_id) is not str or not re.fullmatch(r"turn\.[0-9a-f]{64}", turn_id):
            raise ValueError("turn_id is malformed")
        with self._lock, closing(self._connect()) as connection:
            return self._get(connection, turn_id)

    def active_turn(self) -> NativeToolTurn | None:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM jenny_native_tool_turns WHERE active=1"
            ).fetchall()
            if len(rows) > 1:
                raise RuntimeError("more than one native tool turn is active")
            return None if not rows else self._from_row(rows[0])

    def begin(
        self, *, original_request_ref: str, original_observation_ref: str
    ) -> NativeToolTurn:
        turn_id = self.turn_id_for_request(original_request_ref)
        _reference(original_observation_ref, "native original_observation_ref")
        candidate = NativeToolTurn(
            turn_id=turn_id,
            revision=0,
            status="OPEN",
            catalog_hash=self.catalog_hash,
            original_request_ref=original_request_ref,
            original_observation_ref=original_observation_ref,
            trace_json="[]",
        )
        with self._lock, self._transaction() as connection:
            existing = self._get(connection, turn_id)
            if existing is not None:
                if (
                    existing.catalog_hash != candidate.catalog_hash
                    or existing.original_request_ref != candidate.original_request_ref
                    or existing.original_observation_ref
                    != candidate.original_observation_ref
                ):
                    raise RuntimeError("turn identity conflicts with stored provenance")
                return existing
            active = connection.execute(
                "SELECT turn_id FROM jenny_native_tool_turns WHERE active=1"
            ).fetchone()
            if active is not None:
                raise RuntimeError("another native tool turn is already active")
            payload = _canonical(candidate.canonical_payload())
            connection.execute(
                "INSERT INTO jenny_native_tool_turns VALUES(?,?,?,?,?,?)",
                (
                    candidate.turn_id,
                    candidate.revision,
                    candidate.status,
                    1,
                    candidate.envelope_ref,
                    payload,
                ),
            )
        return candidate

    def _replace_cas(
        self,
        current: NativeToolTurn,
        updated: NativeToolTurn,
        connection: sqlite3.Connection,
    ) -> NativeToolTurn:
        if updated.turn_id != current.turn_id or updated.revision != current.revision + 1:
            raise RuntimeError("native turn transition identity or revision is invalid")
        payload = _canonical(updated.canonical_payload())
        changed = connection.execute(
            "UPDATE jenny_native_tool_turns SET revision=?,status=?,active=?,"
            "envelope_ref=?,payload_json=? WHERE turn_id=? AND revision=?",
            (
                updated.revision,
                updated.status,
                int(updated.status != "COMMITTED"),
                updated.envelope_ref,
                payload,
                current.turn_id,
                current.revision,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("native turn revision compare-and-swap failed")
        return updated

    def _current_for_cas(
        self,
        connection: sqlite3.Connection,
        turn_id: str,
        expected_revision: int,
    ) -> NativeToolTurn:
        current = self._get(connection, turn_id)
        if current is None:
            raise KeyError("native tool turn is absent")
        if current.revision != expected_revision:
            raise RuntimeError("native tool turn revision conflict")
        if current.catalog_hash != self.catalog_hash:
            raise RuntimeError("native tool turn catalog differs from this runtime")
        return current

    def record_tool_requests(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        calls: tuple[NativeToolCall, ...],
        continuation_choice_json: str,
    ) -> NativeToolTurn:
        if type(calls) is not tuple or not calls or any(
            not isinstance(item, NativeToolCall) for item in calls
        ):
            raise ValueError("native model selection must contain tool calls")
        if len({item.call_id for item in calls}) != len(calls):
            raise ValueError("native model call identifiers must be unique")
        if any(item.tool_name not in self.tool_names for item in calls):
            raise ValueError("native model selected a tool outside the exact catalog")
        choice = _object(continuation_choice_json, "native continuation choice")
        with self._lock, self._transaction() as connection:
            current = self._current_for_cas(
                connection, turn_id, expected_revision
            )
            if current.status not in ("OPEN", "CONTINUATION_READY"):
                raise RuntimeError("native turn is not ready for a model choice")
            known_ids = {
                call["call_id"]
                for entry in current.trace
                if entry.get("kind") == "MODEL_TOOL_REQUESTS"
                for call in entry.get("calls", [])  # type: ignore[union-attr]
            }
            if known_ids & {item.call_id for item in calls}:
                raise ValueError("native model reused a call identifier")
            previous_count = len(known_ids)
            if previous_count + len(calls) > NATIVE_TURN_MAX_TOOL_CALLS:
                raise ValueError("native turn exceeds eight tool calls")
            trace = list(current.trace)
            trace.append(
                {
                    "calls": [item.canonical_payload() for item in calls],
                    "choice": choice,
                    "kind": "MODEL_TOOL_REQUESTS",
                }
            )
            permission_refs = tuple(
                dict.fromkeys(
                    (*current.permission_reservation_refs,)
                    + tuple(item.permission_reservation_ref for item in calls)
                )
            )
            updated = replace(
                current,
                revision=current.revision + 1,
                status="TOOL_PENDING",
                trace_json=_canonical(trace),
                outstanding_calls=calls,
                permission_reservation_refs=permission_refs,
            )
            return self._replace_cas(current, updated, connection)

    def record_tool_result(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        call_id: str,
        terminal_event: ToolEvent,
    ) -> NativeToolTurn:
        return self.record_tool_results(
            turn_id,
            expected_revision=expected_revision,
            results=((call_id, terminal_event),),
        )

    def record_tool_results(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        results: tuple[tuple[str, ToolEvent], ...],
    ) -> NativeToolTurn:
        """Atomically record one bounded host-result batch.

        A provider may return several calls from one model choice.  Validating
        and appending the entire batch in one SQLite transaction prevents a
        malformed later item or lost HTTP response from leaving a half-applied
        continuation.
        """

        if (
            type(results) is not tuple
            or not 1 <= len(results) <= NATIVE_TURN_MAX_TOOL_CALLS
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not isinstance(item[1], ToolEvent)
                for item in results
            )
        ):
            raise ValueError("native result batch is malformed")
        if len({item[0] for item in results}) != len(results):
            raise ValueError("native result batch repeats a call_id")
        decoded: dict[str, tuple[ToolEvent, dict[str, object]]] = {}
        for call_id, terminal_event in results:
            _visible_text(call_id, "native call_id", 256)
            if (
                terminal_event.sequence != 2
                or terminal_event.status not in _TERMINAL_STATUSES
            ):
                raise ValueError("native result requires one terminal ToolEvent")
            if (
                len(terminal_event.result_json.encode("utf-8"))
                > COGNITIVE_TOOL_RESULT_MAX_BYTES
            ):
                raise ValueError("native tool result exceeds the cognitive bridge ceiling")
            decoded[call_id] = (
                terminal_event,
                _object(terminal_event.result_json, "native tool result"),
            )
        with self._lock, self._transaction() as connection:
            current = self._current_for_cas(
                connection, turn_id, expected_revision
            )
            if current.status != "TOOL_PENDING":
                raise RuntimeError("native turn has no outstanding tool result")
            outstanding = {item.call_id: item for item in current.outstanding_calls}
            if not set(decoded) <= set(outstanding):
                raise ValueError("native result call_id is not outstanding")
            for call_id, (terminal_event, _) in decoded.items():
                if outstanding[call_id].operation_ref != terminal_event.operation_ref:
                    raise ValueError("native result operation differs from its call")
            remaining = tuple(
                item for item in current.outstanding_calls if item.call_id not in decoded
            )
            trace = list(current.trace)
            for selected in current.outstanding_calls:
                if selected.call_id not in decoded:
                    continue
                terminal_event, result = decoded[selected.call_id]
                trace.append(
                    {
                        "call_id": selected.call_id,
                        "event_ref": terminal_event.event_ref,
                        "kind": "TOOL_RESULT",
                        "operation_ref": terminal_event.operation_ref,
                        "recorded_at_utc": terminal_event.recorded_at_utc,
                        "result": result,
                        "status": terminal_event.status,
                        "tool_name": selected.tool_name,
                    }
                )
            updated = replace(
                current,
                revision=current.revision + 1,
                status="TOOL_PENDING" if remaining else "CONTINUATION_READY",
                trace_json=_canonical(trace),
                outstanding_calls=remaining,
            )
            return self._replace_cas(current, updated, connection)

    def stage_final(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        response: str,
        continuation_choice_json: str,
        aggregate_observation_json: str | None = None,
    ) -> NativeToolTurn:
        _visible_text(
            response, "native final response", COGNITIVE_TOOL_RESULT_MAX_BYTES
        )
        choice = _object(continuation_choice_json, "native final choice")
        with self._lock, self._transaction() as connection:
            current = self._current_for_cas(
                connection, turn_id, expected_revision
            )
            if current.status not in ("OPEN", "CONTINUATION_READY"):
                raise RuntimeError("native turn is not ready for a final response")
            result_evidence = [
                {
                    "call_id": item["call_id"],
                    "event_ref": item["event_ref"],
                    "operation_ref": item["operation_ref"],
                    "result_ref": _digest(_canonical(item["result"])),
                    "status": item["status"],
                    "tool_name": item["tool_name"],
                }
                for item in current.trace
                if item.get("kind") == "TOOL_RESULT"
            ]
            aggregate = {
                "catalog_hash": current.catalog_hash,
                "tool_results": result_evidence,
                "turn_id": current.turn_id,
            }
            derived_aggregate_json = _canonical(aggregate)
            if (
                aggregate_observation_json is not None
                and aggregate_observation_json != derived_aggregate_json
            ):
                raise ValueError(
                    "aggregate observation differs from the exact durable trace"
                )
            if (
                len(derived_aggregate_json.encode("utf-8"))
                > COGNITIVE_TOOL_RESULT_MAX_BYTES
            ):
                raise ValueError("aggregate observation exceeds the cognitive ceiling")
            trace = list(current.trace)
            trace.append(
                {
                    "aggregate_observation_ref": _digest(_canonical(aggregate)),
                    "choice": choice,
                    "kind": "FINAL_RESPONSE",
                    "response_ref": _digest(response),
                }
            )
            updated = replace(
                current,
                revision=current.revision + 1,
                status="FINAL_READY",
                trace_json=_canonical(trace),
                final_response=response,
                aggregate_observation_json=derived_aggregate_json,
            )
            return self._replace_cas(current, updated, connection)

    def mark_committed(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        episode_ref: str,
    ) -> NativeToolTurn:
        _reference(episode_ref, "native committed episode_ref")
        with self._lock, self._transaction() as connection:
            current = self._current_for_cas(
                connection, turn_id, expected_revision
            )
            if current.status != "FINAL_READY":
                raise RuntimeError("native turn is not staged for final commit")
            trace = list(current.trace)
            trace.append({"episode_ref": episode_ref, "kind": "COMMITTED"})
            updated = replace(
                current,
                revision=current.revision + 1,
                status="COMMITTED",
                trace_json=_canonical(trace),
                committed_episode_ref=episode_ref,
            )
            return self._replace_cas(current, updated, connection)


def _native_turn_from_payload(payload: object) -> NativeToolTurn:
    fields = {
        "aggregate_observation_json",
        "catalog_hash",
        "committed_episode_ref",
        "contract",
        "final_response",
        "original_observation_ref",
        "original_request_ref",
        "outstanding_calls",
        "permission_reservation_refs",
        "revision",
        "status",
        "trace_json",
        "turn_id",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise RuntimeError("stored native tool turn fields are incompatible")
    if payload.get("contract") != NATIVE_TOOL_TURN_CONTRACT:
        raise RuntimeError("stored native tool turn contract is unsupported")
    raw_calls = payload.get("outstanding_calls")
    raw_permissions = payload.get("permission_reservation_refs")
    if type(raw_calls) is not list or type(raw_permissions) is not list:
        raise RuntimeError("stored native tool turn collections are malformed")
    calls: list[NativeToolCall] = []
    call_fields = {
        "arguments_json",
        "call_id",
        "operation_ref",
        "permission_reservation_ref",
        "tool_name",
    }
    for item in raw_calls:
        if type(item) is not dict or set(item) != call_fields:
            raise RuntimeError("stored outstanding native call is malformed")
        calls.append(NativeToolCall(**item))  # type: ignore[arg-type]
    return NativeToolTurn(
        turn_id=payload["turn_id"],  # type: ignore[arg-type]
        revision=payload["revision"],  # type: ignore[arg-type]
        status=payload["status"],  # type: ignore[arg-type]
        catalog_hash=payload["catalog_hash"],  # type: ignore[arg-type]
        original_request_ref=payload["original_request_ref"],  # type: ignore[arg-type]
        original_observation_ref=payload["original_observation_ref"],  # type: ignore[arg-type]
        trace_json=payload["trace_json"],  # type: ignore[arg-type]
        outstanding_calls=tuple(calls),
        permission_reservation_refs=tuple(raw_permissions),  # type: ignore[arg-type]
        final_response=payload["final_response"],  # type: ignore[arg-type]
        aggregate_observation_json=payload["aggregate_observation_json"],  # type: ignore[arg-type]
        committed_episode_ref=payload["committed_episode_ref"],  # type: ignore[arg-type]
    )


class ToolManifestDispatcher:
    """Project and reserve read-only tools without executing them."""

    def __init__(
        self,
        manifest: ToolManifest,
        store: ToolOperationStore,
        *,
        allowed_permission_scopes: frozenset[str] = DEFAULT_ALLOWED_PERMISSION_SCOPES,
    ) -> None:
        if type(manifest) is not ToolManifest:
            raise TypeError("manifest must be an exact ToolManifest")
        if not isinstance(store, ToolOperationStore):
            raise TypeError("store must be a ToolOperationStore")
        if (
            type(allowed_permission_scopes) is not frozenset
            or not allowed_permission_scopes
            or any(
                type(scope) is not str or not _IDENTIFIER.fullmatch(scope)
                for scope in allowed_permission_scopes
            )
        ):
            raise ValueError(
                "allowed_permission_scopes must be a non-empty identifier frozenset"
            )
        expected_manifest_ref = _digest(_canonical(manifest.canonical_payload()))
        if manifest.manifest_ref != expected_manifest_ref:
            raise ValueError("manifest reference is not content-addressed")

        projected: list[tuple[ToolDefinition, Affordance]] = []
        cumulative_bytes = 0
        for definition in manifest.tools:
            expected_definition_ref = _digest(
                _canonical(definition.canonical_payload())
            )
            if definition.definition_ref != expected_definition_ref:
                raise ValueError("tool definition reference is not content-addressed")
            if definition.external_effect:
                raise PermissionError(
                    "external-effect tools cannot enter the read-only bridge"
                )
            if definition.permission_scope not in allowed_permission_scopes:
                raise PermissionError("tool permission scope is not allowlisted")
            controller_description = _canonical(
                {
                    "definition_ref": definition.definition_ref,
                    "description": definition.description,
                    "input_schema": _object(
                        definition.input_schema_json, "tool input schema"
                    ),
                    "output_schema": _object(
                        definition.output_schema_json, "tool output schema"
                    ),
                    "tool_version": definition.tool_version,
                }
            )
            description_bytes = len(controller_description.encode("utf-8"))
            if description_bytes > CONTROLLER_TOOL_DESCRIPTION_MAX_BYTES:
                raise ValueError(
                    "full tool contract exceeds the individual controller-visible ceiling"
                )
            cumulative_bytes += description_bytes
            if cumulative_bytes > CONTROLLER_MANIFEST_DESCRIPTION_MAX_BYTES:
                raise ValueError(
                    "full tool manifest exceeds the cumulative controller-visible ceiling"
                )
            projected.append(
                (
                    definition,
                    Affordance(
                        affordance_id=f"tool.{definition.tool_id}",
                        disposition="ACT",
                        description=controller_description,
                        permission_scope=definition.permission_scope,
                        external_effect=False,
                    ),
                )
            )

        registered_ref = store.register_manifest(manifest)
        if registered_ref != manifest.manifest_ref:
            raise RuntimeError("operation store returned a different manifest reference")
        self.manifest = manifest
        self.store = store
        self.allowed_permission_scopes = allowed_permission_scopes
        self._by_affordance = {
            affordance.affordance_id: (definition, affordance)
            for definition, affordance in projected
        }
        self._by_definition = {
            definition.definition_ref: (definition, affordance)
            for definition, affordance in projected
        }
        self._native_catalog_json = _canonical(
            [
                {
                    "description": affordance.description,
                    "input_schema": _object(
                        definition.input_schema_json, "tool input schema"
                    ),
                    "name": affordance.affordance_id,
                }
                for definition, affordance in projected
            ]
        )

    @property
    def permission_scopes(self) -> frozenset[str]:
        return frozenset(
            definition.permission_scope
            for definition, _ in self._by_affordance.values()
        )

    @property
    def affordances(self) -> tuple[Affordance, ...]:
        return tuple(
            value[1] for _, value in sorted(self._by_affordance.items())
        )

    @property
    def native_tools(self) -> tuple[dict[str, object], ...]:
        """Return the exact provider-native tools array as fresh mappings."""

        decoded = json.loads(self._native_catalog_json)
        if type(decoded) is not list or any(type(item) is not dict for item in decoded):
            raise RuntimeError("stored native tool catalog is malformed")
        return tuple(decoded)

    @property
    def catalog_hash(self) -> str:
        """Hash recursively key-sorted canonical JSON for ``native_tools``."""

        return _digest(self._native_catalog_json)

    def require_catalog_hash(self, catalog_hash: str) -> None:
        if catalog_hash != self.catalog_hash:
            raise ValueError("native tool catalog hash differs")

    def register_affordances(self, registry: DynamicAffordanceRegistry) -> None:
        if not isinstance(registry, DynamicAffordanceRegistry):
            raise TypeError("registry must be a DynamicAffordanceRegistry")
        existing = {item.affordance_id for item in registry.definitions()}
        incoming = set(self._by_affordance)
        collisions = existing & incoming
        if collisions:
            raise ValueError(
                "tool affordance collides with the live registry: "
                + ",".join(sorted(collisions))
            )
        for affordance_id in sorted(self._by_affordance):
            definition, affordance = self._by_affordance[affordance_id]
            registry.register(
                affordance,
                observable_source_ref=definition.definition_ref,
                observable_source_kind="TOOL",
                execution_mode="DEFERRED",
            )

    def reserve(
        self,
        affordance: Affordance,
        request: AffordanceRequest,
    ) -> ToolCallReservation:
        if not isinstance(affordance, Affordance):
            raise TypeError("affordance must be an Affordance")
        if not isinstance(request, AffordanceRequest):
            raise TypeError("request must be an AffordanceRequest")
        try:
            definition, expected_affordance = self._by_affordance[
                request.affordance_id
            ]
        except KeyError as exc:
            raise ValueError("request selected a tool outside the exact manifest") from exc
        if affordance != expected_affordance:
            raise ValueError("pending affordance differs from the exact tool projection")
        if request.affordance_id != affordance.affordance_id:
            raise ValueError("request and pending affordance identifiers differ")
        _object(request.action_payload, "tool action payload")
        reservation = ToolCallReservation(
            call_id=f"call.{request.idempotency_key[7:]}",
            manifest_ref=self.manifest.manifest_ref,
            tool_ref=definition.definition_ref,
            requester_ref=request.idempotency_key,
            arguments_json=request.action_payload,
        )
        stored = self.store.reserve(reservation)
        if stored != reservation:
            raise RuntimeError("operation store returned a different reservation")
        return stored

    def reserve_pending(
        self, supervisor: DeferredEffectSupervisor
    ) -> ToolCallReservation:
        pending = supervisor.pending_deferred_effect()
        if pending is None:
            raise RuntimeError("Jenny has no deferred effect to reserve")
        return self.reserve(*pending)

    def resolve(self, operation_ref: str) -> DecodedToolOperation:
        reservation = self.store.reservation_for_operation(operation_ref)
        if reservation is None:
            raise KeyError("tool operation is absent")
        if reservation.operation_ref != operation_ref:
            raise RuntimeError("resolved operation identity differs")
        if reservation.manifest_ref != self.manifest.manifest_ref:
            raise ValueError("operation belongs to a different tool manifest")
        try:
            definition, affordance = self._by_definition[reservation.tool_ref]
        except KeyError as exc:
            raise ValueError("operation selected an unknown tool definition") from exc
        return DecodedToolOperation(
            reservation=reservation,
            definition=definition,
            affordance_id=affordance.affordance_id,
            arguments=_object(reservation.arguments_json, "tool arguments"),
            input_schema=_object(definition.input_schema_json, "tool input schema"),
            output_schema=_object(definition.output_schema_json, "tool output schema"),
            events=self.store.events_for_operation(operation_ref),
        )

    def terminal_receipt(
        self,
        affordance: Affordance,
        request: AffordanceRequest,
        event: ToolEvent,
    ) -> AffordanceReceipt:
        """Create a staged raw receipt without committing Jenny's pending turn.

        The caller must retain this observation across native tool continuation
        and pass a final receipt to the supervisor only after the provider has
        produced the final Jenny response.  A tool result by itself is not a
        cognitive episode and must not advance Moving Origin.
        """

        if not isinstance(event, ToolEvent):
            raise TypeError("event must be a ToolEvent")
        if event.sequence != 2 or event.status not in _TERMINAL_STATUSES:
            raise ValueError("only a terminal event can become a cognitive receipt")
        operation = self.resolve(event.operation_ref)
        if (
            len(operation.events) != 2
            or operation.events[0].sequence != 1
            or operation.events[0].status != "RUNNING"
            or operation.events[1] != event
        ):
            raise ValueError("terminal event requires an exact prior RUNNING event")
        definition = operation.definition
        expected_affordance = self._by_definition[definition.definition_ref][1]
        if affordance != expected_affordance:
            raise ValueError("pending affordance differs from the operation tool")
        reservation = operation.reservation
        if (
            request.affordance_id != affordance.affordance_id
            or request.idempotency_key != reservation.requester_ref
            or request.action_payload != reservation.arguments_json
        ):
            raise ValueError("pending Jenny request differs from the tool reservation")
        if len(event.result_json.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
            raise ValueError("tool result exceeds the cognitive bridge ceiling")
        result = _object(event.result_json, "tool result")
        observation_json = _canonical(
            {
                "operation_ref": event.operation_ref,
                "recorded_at_utc": event.recorded_at_utc,
                "result": result,
                "status": event.status,
            }
        )
        if len(observation_json.encode("utf-8")) > COGNITIVE_TOOL_RESULT_MAX_BYTES:
            raise ValueError("tool observation exceeds the cognitive bridge ceiling")
        observed = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind="TOOL",
            source_ref=definition.definition_ref,
            observation_json=observation_json,
            artifact_refs=(event.operation_ref,),
            evidence_refs=(event.event_ref,),
        )
        return AffordanceReceipt(
            status=event.status,
            output=event.result_json,
            consequence=(),
            observable_consequence=observed,
        )
