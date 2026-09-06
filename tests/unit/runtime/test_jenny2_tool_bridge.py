from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from typing import Mapping
import unittest

from angler.runtime.jenny2_tool_bridge import (
    COGNITIVE_TOOL_RESULT_MAX_BYTES,
    NativeOpenClawCatalog,
    NativeModelToolCall,
    NativeToolCall,
    NativeToolModelResult,
    NativeToolTurnStore,
    OpenAICompatibleNativeToolModel,
    ToolManifestDispatcher,
)
from angler.runtime.persistent_autonomy import (
    Affordance,
    AffordanceRequest,
    DynamicAffordanceRegistry,
)
from angler.runtime.tool_operations import (
    ToolDefinition,
    ToolEvent,
    ToolManifest,
    ToolOperationStore,
)


def _ref(character: str) -> str:
    return "sha256:" + character * 64


def _tool(
    tool_id: str,
    *,
    external_effect: bool = False,
    permission_scope: str = "external.web.read",
    description: str | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        tool_version="1.0.0",
        description=description or f"Use {tool_id} for bounded read-only research.",
        permission_scope=permission_scope,
        input_schema_json=(
            '{"additionalProperties":false,"properties":{"query":'
            '{"type":"string"}},"required":["query"],"type":"object"}'
        ),
        output_schema_json=(
            '{"additionalProperties":false,"properties":{"answer":'
            '{"type":"string"}},"required":["answer"],"type":"object"}'
        ),
        external_effect=external_effect,
    )


def _manifest(*tools: ToolDefinition, version: str = "1.0.0") -> ToolManifest:
    return ToolManifest(
        manifest_id="openclaw.readonly",
        manifest_version=version,
        tools=tuple(sorted(tools, key=lambda item: (item.tool_id, item.tool_version))),
    )


def _request(
    affordance_id: str,
    *,
    request_ref: str = _ref("a"),
    action_payload: str = '{"query":"moving origin"}',
) -> AffordanceRequest:
    return AffordanceRequest(
        idempotency_key=request_ref,
        trigger_ref="agent:test-turn",
        affordance_id=affordance_id,
        observation_ref=_ref("b"),
        state_head_ref=_ref("c"),
        action_payload=action_payload,
        choice_context_json="{}",
    )


def _events(store: ToolOperationStore, operation_ref: str) -> tuple[ToolEvent, ToolEvent]:
    running = ToolEvent(
        operation_ref=operation_ref,
        sequence=1,
        status="RUNNING",
        result_json="{}",
        recorded_at_utc="2026-09-02T14:00:00.000000Z",
    )
    terminal = ToolEvent(
        operation_ref=operation_ref,
        sequence=2,
        status="COMPLETED",
        result_json='{"answer":"bounded evidence"}',
        recorded_at_utc="2026-09-02T14:00:01.000000Z",
    )
    store.append_event(running)
    store.append_event(terminal)
    return running, terminal


class _DeferredSupervisor:
    def __init__(self, pending: tuple[Affordance, AffordanceRequest]) -> None:
        self.pending = pending
    def pending_deferred_effect(
        self,
    ) -> tuple[Affordance, AffordanceRequest] | None:
        return self.pending


class ToolManifestProjectionTests(unittest.TestCase):
    def test_projects_full_dynamic_contracts_as_deferred_affordances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ToolOperationStore(Path(directory) / "tools.sqlite3")
            search, fetch = _tool("web.search"), _tool("web.fetch")
            dispatcher = ToolManifestDispatcher(_manifest(search, fetch), store)
            registry = DynamicAffordanceRegistry()
            dispatcher.register_affordances(registry)

            native_tools = dispatcher.native_tools
            self.assertEqual(
                [item["name"] for item in native_tools],
                ["tool.web.fetch", "tool.web.search"],
            )
            canonical_catalog = json.dumps(
                list(native_tools),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self.assertEqual(
                dispatcher.catalog_hash,
                "sha256:"
                + hashlib.sha256(canonical_catalog.encode("utf-8")).hexdigest(),
            )
            dispatcher.require_catalog_hash(dispatcher.catalog_hash)
            with self.assertRaisesRegex(ValueError, "catalog hash differs"):
                dispatcher.require_catalog_hash(_ref("7"))

            self.assertEqual(
                [item.affordance_id for item in registry.definitions()],
                ["tool.web.fetch", "tool.web.search"],
            )
            for definition in (search, fetch):
                affordance = registry.definition(f"tool.{definition.tool_id}")
                visible = json.loads(affordance.description)
                self.assertEqual(visible["definition_ref"], definition.definition_ref)
                self.assertEqual(
                    visible["input_schema"], json.loads(definition.input_schema_json)
                )
                self.assertEqual(
                    visible["output_schema"], json.loads(definition.output_schema_json)
                )
                self.assertEqual(
                    registry.execution_mode(affordance.affordance_id), "DEFERRED"
                )
                self.assertEqual(
                    registry.observable_source_ref(affordance.affordance_id),
                    definition.definition_ref,
                )
                self.assertEqual(
                    registry.observable_source_kind(affordance.affordance_id), "TOOL"
                )
                self.assertIsNone(
                    registry.consequence_mapper(affordance.affordance_id)
                )
                with self.assertRaisesRegex(ValueError, "no executor"):
                    registry.executor(affordance.affordance_id)

    def test_rejects_mutating_nonallowlisted_collision_and_oversize_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(PermissionError, "external-effect"):
                ToolManifestDispatcher(
                    _manifest(_tool("web.write", external_effect=True)),
                    ToolOperationStore(root / "mutating.sqlite3"),
                )
            with self.assertRaisesRegex(PermissionError, "not allowlisted"):
                ToolManifestDispatcher(
                    _manifest(
                        _tool("filesystem.read", permission_scope="filesystem.read")
                    ),
                    ToolOperationStore(root / "scope.sqlite3"),
                )
            with self.assertRaisesRegex(ValueError, "individual controller-visible"):
                ToolManifestDispatcher(
                    _manifest(_tool("web.large", description="x" * 950)),
                    ToolOperationStore(root / "individual.sqlite3"),
                )

            many = tuple(
                _tool(f"web.item{index:03d}", description="x" * 480)
                for index in range(64)
            )
            with self.assertRaisesRegex(ValueError, "cumulative controller-visible"):
                ToolManifestDispatcher(
                    _manifest(*many), ToolOperationStore(root / "cumulative.sqlite3")
                )

            dispatcher = ToolManifestDispatcher(
                _manifest(_tool("web.fetch"), _tool("web.search")),
                ToolOperationStore(root / "collision.sqlite3"),
            )
            registry = DynamicAffordanceRegistry()
            colliding = dispatcher.affordances[1]
            registry.register(
                colliding,
                observable_source_ref=json.loads(colliding.description)[
                    "definition_ref"
                ],
                observable_source_kind="TOOL",
                execution_mode="DEFERRED",
            )
            before = registry.definitions()
            with self.assertRaisesRegex(ValueError, "collides"):
                dispatcher.register_affordances(registry)
            self.assertEqual(registry.definitions(), before)


class NativeOpenClawCatalogTests(unittest.TestCase):
    def test_accepts_full_catalog_with_truthful_separate_effect_classification(self) -> None:
        tools: list[dict[str, object]] = []
        for index in range(48):
            card: dict[str, object] = {
                "name": f"openclaw.tool_{index:02d}",
                "description": f"Tool {index}: " + "x" * 1_100,
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "type": "object",
                },
            }
            if index == 0:
                card["external_effect"] = True
            elif index == 1:
                card["external_effect"] = False
            tools.append(card)
        catalog = NativeOpenClawCatalog(tuple(tools))
        projected = catalog.tools_for_supervised_turn(
            observation_source="HUMAN",
            bridge_authenticated=True,
            owner_present=True,
        )
        self.assertEqual(len(projected), 48)
        self.assertEqual(set(projected[0]), {"name", "description", "input_schema"})
        canonical = json.dumps(
            list(projected),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(
            catalog.catalog_hash,
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            catalog.cards[0].effect_classification,
            "HOST_DECLARED_EXTERNAL_EFFECT",
        )
        self.assertEqual(
            catalog.cards[1].effect_classification,
            "HOST_DECLARED_NON_EFFECT",
        )
        self.assertEqual(
            catalog.cards[2].effect_classification,
            "HOST_SUPERVISED_UNKNOWN",
        )
        self.assertNotEqual(catalog.policy_ref, catalog.catalog_hash)
        registry = DynamicAffordanceRegistry()
        catalog.register_guarded_affordance(registry)
        guarded = registry.definition("openclaw.supervised.turn")
        self.assertTrue(guarded.external_effect)
        self.assertEqual(guarded.authorization_mode, "HOST_GUARDED")
        self.assertEqual(guarded.permission_scope, "external.openclaw.guarded")
        self.assertEqual(
            registry.execution_mode(guarded.affordance_id), "DEFERRED"
        )
        self.assertEqual(
            registry.observable_source_ref(guarded.affordance_id),
            catalog.catalog_hash,
        )
        self.assertEqual(
            registry.observable_source_kind(guarded.affordance_id), "TOOL"
        )
        with self.assertRaisesRegex(PermissionError, "owner-present"):
            catalog.tools_for_supervised_turn(
                observation_source="SCHEDULER",
                bridge_authenticated=True,
                owner_present=True,
            )

    def test_final_ready_turn_yields_one_catalog_bound_zero_score_receipt(self) -> None:
        card = {
            "name": "openclaw.search",
            "description": "Search under host supervision.",
            "input_schema": {"type": "object"},
        }
        catalog = NativeOpenClawCatalog((card,))
        with tempfile.TemporaryDirectory() as directory:
            turns = NativeToolTurnStore(
                Path(directory) / "turns.sqlite3",
                catalog_hash=catalog.catalog_hash,
                tool_names=frozenset(("openclaw.search",)),
            )
            request = _request(
                catalog.guarded_affordance.affordance_id,
                request_ref=_ref("1"),
                action_payload='{"message":"please research this"}',
            )
            opened = turns.begin(
                original_request_ref=request.idempotency_key,
                original_observation_ref=request.observation_ref,
            )
            with self.assertRaisesRegex(ValueError, "not FINAL_READY"):
                catalog.final_receipt(opened, request)
            final = turns.stage_final(
                opened.turn_id,
                expected_revision=0,
                response="Here is Jenny's final response.",
                continuation_choice_json=(
                    '{"outstanding_call_ids":[],"route":"FINAL",'
                    '"turn_revision":0}'
                ),
            )
            receipt = catalog.final_receipt(final, request)
            self.assertEqual(receipt.status, "COMPLETED")
            self.assertEqual(receipt.output, "Here is Jenny's final response.")
            self.assertEqual(receipt.consequence, ())
            self.assertEqual(
                receipt.observable_consequence.source_ref,  # type: ignore[union-attr]
                catalog.catalog_hash,
            )
            self.assertEqual(
                receipt.observable_consequence.artifact_refs,  # type: ignore[union-attr]
                (),
            )
        with self.assertRaisesRegex(PermissionError, "owner-present"):
            catalog.tools_for_supervised_turn(
                observation_source="HUMAN",
                bridge_authenticated=False,
                owner_present=True,
            )

    def test_catalog_hash_duplicate_names_and_256k_ceiling_fail_closed(self) -> None:
        card = {
            "name": "openclaw.search",
            "description": "Search under host supervision.",
            "input_schema": {"type": "object"},
        }
        valid = NativeOpenClawCatalog((card,))
        with self.assertRaisesRegex(ValueError, "catalog hash differs"):
            NativeOpenClawCatalog((card,), catalog_hash=_ref("0"))
        self.assertTrue(valid.catalog_hash.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            NativeOpenClawCatalog((card, dict(card)))
        oversized = tuple(
            {
                "name": f"openclaw.large_{index:02d}",
                "description": "z" * 6_000,
                "input_schema": {"type": "object"},
            }
            for index in range(48)
        )
        with self.assertRaisesRegex(ValueError, "exceeds 256 KiB"):
            NativeOpenClawCatalog(oversized)


class OpenAICompatibleNativeToolModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = (
            {
                "name": "openclaw.search",
                "description": "Search for current evidence under host supervision.",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "type": "object",
                },
            },
            {
                "name": "openclaw.fetch",
                "description": "Fetch one selected source under host supervision.",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "type": "object",
                },
            },
        )
        self.catalog = NativeOpenClawCatalog(self.tools)

    def _model(self, response: dict[str, object], calls: list[object]):
        def transport(
            endpoint: str, payload: Mapping[str, object], timeout_seconds: float
        ) -> Mapping[str, object]:
            calls.append((endpoint, payload, timeout_seconds))
            return response

        return OpenAICompatibleNativeToolModel(
            endpoint="http://127.0.0.1:30000/v1",
            served_model="jenny-qwen3.8-27b",
            transport=transport,
            maximum_output_tokens=2_048,
        )

    def test_final_response_uses_exact_function_catalog_and_greedy_thinking(self) -> None:
        choice = {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "I found enough evidence to answer directly.",
                "reasoning_content": "bounded internal reasoning",
                "role": "assistant",
            },
        }
        calls: list[object] = []
        model = self._model({"choices": [choice]}, calls)
        result = model.generate(
            system_context="You are Jenny in an owner-present supervised turn.",
            user_message="What should we inspect?",
            tools=self.tools,
            catalog_hash=self.catalog.catalog_hash,
            max_new_tokens=512,
        )

        self.assertIsInstance(result, NativeToolModelResult)
        self.assertEqual(result.route, "FINAL")
        self.assertEqual(
            result.final_content, "I found enough evidence to answer directly."
        )
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(json.loads(result.choice_json), choice)
        self.assertEqual(result.finish_reason, "stop")
        endpoint, payload, timeout = calls[0]  # type: ignore[misc]
        self.assertEqual(endpoint, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(timeout, 180.0)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["seed"], 20260902)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "low"},
        )
        self.assertEqual(
            payload["tools"],
            [
                {
                    "function": {
                        "description": item["description"],
                        "name": item["name"],
                        "parameters": item["input_schema"],
                    },
                    "type": "function",
                }
                for item in self.tools
            ],
        )
        self.assertEqual(
            payload["messages"],
            [
                {
                    "content": "You are Jenny in an owner-present supervised turn.",
                    "role": "system",
                },
                {"content": "What should we inspect?", "role": "user"},
            ],
        )

    def test_tool_calls_are_known_unique_and_arguments_become_canonical(self) -> None:
        calls: list[object] = []
        model = self._model(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "content": None,
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '{"query": "Moving Origin"}',
                                        "name": "openclaw.search",
                                    },
                                    "id": "call_search_1",
                                    "type": "function",
                                },
                                {
                                    "function": {
                                        "arguments": '{"url":"https://example.test"}',
                                        "name": "openclaw.fetch",
                                    },
                                    "id": "call_fetch_1",
                                    "type": "function",
                                },
                            ],
                        },
                    }
                ]
            },
            calls,
        )
        result = model.generate(
            system_context="System context.",
            user_message="Research this.",
            tools=self.tools,
            catalog_hash=self.catalog.catalog_hash,
        )
        self.assertEqual(result.route, "TOOL_REQUESTED")
        self.assertIsNone(result.final_content)
        self.assertEqual(
            result.tool_calls,
            (
                NativeModelToolCall(
                    "call_search_1",
                    "openclaw.search",
                    '{"query":"Moving Origin"}',
                ),
                NativeModelToolCall(
                    "call_fetch_1",
                    "openclaw.fetch",
                    '{"url":"https://example.test"}',
                ),
            ),
        )

    def test_reconstructs_assistant_calls_and_tool_results_for_continuation(self) -> None:
        trace = [
            {
                "calls": [
                    {
                        "arguments_json": '{"query":"temporal memory"}',
                        "call_id": "call_search_1",
                        "operation_ref": _ref("1"),
                        "permission_reservation_ref": _ref("2"),
                        "tool_name": "openclaw.search",
                    }
                ],
                "choice": {"route": "TOOL_REQUESTED"},
                "kind": "MODEL_TOOL_REQUESTS",
            },
            {
                "call_id": "call_search_1",
                "event_ref": _ref("3"),
                "kind": "TOOL_RESULT",
                "operation_ref": _ref("1"),
                "recorded_at_utc": "2026-09-02T15:00:00.000000Z",
                "result": {"answer": "Situated recall uses valid time."},
                "status": "COMPLETED",
                "tool_name": "openclaw.search",
            },
        ]
        calls: list[object] = []
        model = self._model(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "The retrieved evidence distinguishes valid time.",
                            "role": "assistant",
                        },
                    }
                ]
            },
            calls,
        )
        model.generate(
            system_context="System context.",
            user_message="Explain temporal recall.",
            tools=self.tools,
            catalog_hash=self.catalog.catalog_hash,
            prior_trace_json=json.dumps(
                trace,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        messages = calls[0][1]["messages"]  # type: ignore[index]
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(
            messages[2]["tool_calls"],
            [
                {
                    "function": {
                        "arguments": '{"query":"temporal memory"}',
                        "name": "openclaw.search",
                    },
                    "id": "call_search_1",
                    "type": "function",
                }
            ],
        )
        self.assertEqual(messages[3]["tool_call_id"], "call_search_1")
        self.assertEqual(
            messages[3]["content"],
            '{"result":{"answer":"Situated recall uses valid time."},"status":"COMPLETED"}',
        )

    def test_continuation_exposes_denied_status_to_the_model(self) -> None:
        trace = [
            {
                "calls": [
                    {
                        "arguments_json": "{}",
                        "call_id": "call_denied_1",
                        "operation_ref": _ref("denied-op"),
                        "permission_reservation_ref": _ref("denied-permission"),
                        "tool_name": "openclaw.search",
                    }
                ],
                "choice": {"route": "TOOL_REQUESTED"},
                "kind": "MODEL_TOOL_REQUESTS",
            },
            {
                "call_id": "call_denied_1",
                "event_ref": _ref("denied-event"),
                "kind": "TOOL_RESULT",
                "operation_ref": _ref("denied-op"),
                "recorded_at_utc": "2026-09-02T15:00:00.000000Z",
                "result": {"reason": "owner approval was not granted"},
                "status": "DENIED",
                "tool_name": "openclaw.search",
            },
        ]
        calls: list[object] = []
        model = self._model(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "I did not run that action.", "role": "assistant"},
                    }
                ]
            },
            calls,
        )
        model.generate(
            system_context="System context.",
            user_message="Try the guarded action.",
            tools=self.tools,
            catalog_hash=self.catalog.catalog_hash,
            prior_trace_json=json.dumps(trace, separators=(",", ":"), sort_keys=True),
        )
        self.assertEqual(
            calls[0][1]["messages"][3]["content"],  # type: ignore[index]
            '{"result":{"reason":"owner approval was not granted"},"status":"DENIED"}',
        )

    def test_malformed_model_outputs_and_trace_fail_closed(self) -> None:
        malformed = {
            "both": {
                "content": "answer",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{}",
                            "name": "openclaw.search",
                        },
                        "id": "call_1",
                        "type": "function",
                    }
                ],
            },
            "neither": {"content": None},
            "unknown": {
                "content": None,
                "tool_calls": [
                    {
                        "function": {"arguments": "{}", "name": "shell.exec"},
                        "id": "call_1",
                        "type": "function",
                    }
                ],
            },
            "nonobject": {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "[]",
                            "name": "openclaw.search",
                        },
                        "id": "call_1",
                        "type": "function",
                    }
                ],
            },
            "duplicates": {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{}",
                            "name": "openclaw.search",
                        },
                        "id": "call_same",
                        "type": "function",
                    },
                    {
                        "function": {
                            "arguments": "{}",
                            "name": "openclaw.fetch",
                        },
                        "id": "call_same",
                        "type": "function",
                    },
                ],
            },
        }
        for label, message in malformed.items():
            with self.subTest(label=label):
                model = self._model(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": message,
                            }
                        ]
                    },
                    [],
                )
                with self.assertRaises(RuntimeError):
                    model.generate(
                        system_context="System context.",
                        user_message="User request.",
                        tools=self.tools,
                        catalog_hash=self.catalog.catalog_hash,
                    )

        outstanding_trace = json.dumps(
            [
                {
                    "calls": [
                        {
                            "arguments_json": "{}",
                            "call_id": "call_pending",
                            "tool_name": "openclaw.search",
                        }
                    ],
                    "choice": {},
                    "kind": "MODEL_TOOL_REQUESTS",
                }
            ],
            separators=(",", ":"),
            sort_keys=True,
        )
        valid_model = self._model(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "unused", "role": "assistant"},
                    }
                ]
            },
            [],
        )
        with self.assertRaisesRegex(ValueError, "outstanding tool results"):
            valid_model.generate(
                system_context="System context.",
                user_message="User request.",
                tools=self.tools,
                catalog_hash=self.catalog.catalog_hash,
                prior_trace_json=outstanding_trace,
            )
        with self.assertRaisesRegex(ValueError, "loopback"):
            OpenAICompatibleNativeToolModel(
                endpoint="https://api.example.test/v1",
                served_model="forbidden-remote-model",
            )


class ToolReservationAndReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ToolOperationStore(
            Path(self.temporary.name) / "tool-operations.sqlite3"
        )
        self.search = _tool("web.search")
        self.fetch = _tool("web.fetch")
        self.dispatcher = ToolManifestDispatcher(
            _manifest(self.search, self.fetch), self.store
        )
        self.affordance = next(
            item
            for item in self.dispatcher.affordances
            if item.affordance_id == "tool.web.search"
        )
        self.request = _request(self.affordance.affordance_id)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_requires_canonical_arguments_and_reserves_idempotently(self) -> None:
        first = self.dispatcher.reserve(self.affordance, self.request)
        second = self.dispatcher.reserve(self.affordance, self.request)
        self.assertEqual(first, second)
        self.assertEqual(first.call_id, "call." + self.request.idempotency_key[7:])
        self.assertEqual(first.manifest_ref, self.dispatcher.manifest.manifest_ref)
        self.assertEqual(first.tool_ref, self.search.definition_ref)
        self.assertEqual(first.requester_ref, self.request.idempotency_key)
        resolved = self.dispatcher.resolve(first.operation_ref)
        self.assertEqual(resolved.arguments, {"query": "moving origin"})
        self.assertEqual(resolved.definition, self.search)

        noncanonical = _request(
            self.affordance.affordance_id,
            request_ref=_ref("d"),
            action_payload='{ "query": "moving origin" }',
        )
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            self.dispatcher.reserve(self.affordance, noncanonical)

    def test_terminal_receipt_requires_exact_operation_tool_request_and_running(self) -> None:
        reservation = self.dispatcher.reserve(self.affordance, self.request)
        running = ToolEvent(
            operation_ref=reservation.operation_ref,
            sequence=1,
            status="RUNNING",
            result_json="{}",
            recorded_at_utc="2026-09-02T14:00:00.000000Z",
        )
        self.store.append_event(running)
        with self.assertRaisesRegex(ValueError, "only a terminal"):
            self.dispatcher.terminal_receipt(
                self.affordance, self.request, running
            )
        terminal = ToolEvent(
            operation_ref=reservation.operation_ref,
            sequence=2,
            status="COMPLETED",
            result_json='{"answer":"bounded evidence"}',
            recorded_at_utc="2026-09-02T14:00:01.000000Z",
        )
        with self.assertRaisesRegex(ValueError, "exact prior RUNNING"):
            self.dispatcher.terminal_receipt(
                self.affordance, self.request, terminal
            )
        self.store.append_event(terminal)

        wrong_request = replace(self.request, idempotency_key=_ref("e"))
        with self.assertRaisesRegex(ValueError, "Jenny request differs"):
            self.dispatcher.terminal_receipt(
                self.affordance, wrong_request, terminal
            )
        wrong_affordance = next(
            item
            for item in self.dispatcher.affordances
            if item.affordance_id == "tool.web.fetch"
        )
        with self.assertRaisesRegex(ValueError, "pending affordance differs"):
            self.dispatcher.terminal_receipt(
                wrong_affordance, self.request, terminal
            )
        unknown_event = replace(terminal, operation_ref=_ref("f"))
        with self.assertRaisesRegex(KeyError, "operation is absent"):
            self.dispatcher.terminal_receipt(
                self.affordance, self.request, unknown_event
            )

    def test_terminal_observation_is_source_bound_and_has_no_scalar_score(self) -> None:
        reservation = self.dispatcher.reserve(self.affordance, self.request)
        _, terminal = _events(self.store, reservation.operation_ref)
        receipt = self.dispatcher.terminal_receipt(
            self.affordance, self.request, terminal
        )
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.consequence, ())
        self.assertIsNotNone(receipt.observable_consequence)
        observed = receipt.observable_consequence
        assert observed is not None
        self.assertEqual(observed.request_ref, self.request.idempotency_key)
        self.assertEqual(observed.source_kind, "TOOL")
        self.assertEqual(observed.source_ref, self.search.definition_ref)
        self.assertEqual(observed.artifact_refs, (reservation.operation_ref,))
        self.assertEqual(observed.evidence_refs, (terminal.event_ref,))
        self.assertEqual(
            json.loads(observed.observation_json),
            {
                "operation_ref": reservation.operation_ref,
                "recorded_at_utc": terminal.recorded_at_utc,
                "result": {"answer": "bounded evidence"},
                "status": "COMPLETED",
            },
        )

    def test_supervisor_seam_reserves_without_executing_or_completing(self) -> None:
        supervisor = _DeferredSupervisor((self.affordance, self.request))
        reservation = self.dispatcher.reserve_pending(supervisor)
        self.assertEqual(reservation.requester_ref, self.request.idempotency_key)
        self.assertEqual(self.store.events_for_operation(reservation.operation_ref), ())

    def test_result_over_bridge_ceiling_is_rejected(self) -> None:
        reservation = self.dispatcher.reserve(self.affordance, self.request)
        running = ToolEvent(
            reservation.operation_ref,
            1,
            "RUNNING",
            "{}",
            "2026-09-02T14:00:00.000000Z",
        )
        terminal = ToolEvent(
            reservation.operation_ref,
            2,
            "COMPLETED",
            '{"text":"' + "x" * COGNITIVE_TOOL_RESULT_MAX_BYTES + '"}',
            "2026-09-02T14:00:01.000000Z",
        )
        self.store.append_event(running)
        self.store.append_event(terminal)
        with self.assertRaisesRegex(ValueError, "cognitive bridge ceiling"):
            self.dispatcher.terminal_receipt(
                self.affordance, self.request, terminal
            )


class NativeToolTurnStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "shared-tools.sqlite3"
        operation_store = ToolOperationStore(self.database)
        self.dispatcher = ToolManifestDispatcher(
            _manifest(_tool("web.fetch"), _tool("web.search")),
            operation_store,
        )
        self.store = NativeToolTurnStore(
            self.database,
            catalog_hash=self.dispatcher.catalog_hash,
            tool_names=frozenset(
                item["name"] for item in self.dispatcher.native_tools  # type: ignore[misc]
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_native_turn_database_and_live_sidecars_are_private(self) -> None:
        self.assertEqual(stat.S_IMODE(self.database.stat().st_mode), 0o600)
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            sidecars = tuple(
                Path(f"{self.database}{suffix}") for suffix in ("-wal", "-shm")
            )
            self.assertTrue(all(path.is_file() for path in sidecars))
            self.assertTrue(
                all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in sidecars)
            )
        finally:
            connection.rollback()
            connection.close()

        target = Path(self.temporary.name) / "not-the-turn-store.txt"
        target.write_text("preserve", encoding="utf-8")
        linked = Path(self.temporary.name) / "linked-turns.sqlite3"
        linked.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "cannot be opened safely"):
            NativeToolTurnStore(
                linked,
                catalog_hash=self.dispatcher.catalog_hash,
                tool_names=frozenset(
                    item["name"] for item in self.dispatcher.native_tools  # type: ignore[misc]
                ),
            )
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve")

    def test_one_turn_survives_restart_and_commits_only_after_final_ready(self) -> None:
        turn = self.store.begin(
            original_request_ref=_ref("1"), original_observation_ref=_ref("2")
        )
        self.assertEqual(turn.turn_id, "turn." + "1" * 64)
        self.assertEqual(turn.status, "OPEN")
        self.assertEqual(
            self.store.begin(
                original_request_ref=_ref("1"),
                original_observation_ref=_ref("2"),
            ),
            turn,
        )
        with self.assertRaisesRegex(RuntimeError, "another native tool turn"):
            self.store.begin(
                original_request_ref=_ref("3"),
                original_observation_ref=_ref("4"),
            )

        call = NativeToolCall(
            call_id="provider-call-1",
            tool_name="tool.web.search",
            operation_ref=_ref("5"),
            permission_reservation_ref=_ref("6"),
            arguments_json='{"query":"temporal memory"}',
        )
        pending = self.store.record_tool_requests(
            turn.turn_id,
            expected_revision=0,
            calls=(call,),
            continuation_choice_json=(
                '{"outstanding_call_ids":["provider-call-1"],'
                '"route":"TOOL_REQUESTED","turn_revision":1}'
            ),
        )
        self.assertEqual(pending.status, "TOOL_PENDING")
        self.assertEqual(pending.outstanding_calls, (call,))
        self.assertEqual(pending.permission_reservation_refs, (_ref("6"),))
        self.assertEqual(
            pending.provider_metadata(),
            {
                "catalog_hash": self.dispatcher.catalog_hash,
                "outstanding_calls": [
                    {
                        "call_id": "provider-call-1",
                        "name": "tool.web.search",
                        "permission_reservation_ref": _ref("6"),
                    }
                ],
                "permission_reservation_refs": [_ref("6")],
                "status": "TOOL_PENDING",
                "turn_id": turn.turn_id,
                "turn_revision": 1,
            },
        )
        self.assertIsNone(pending.final_response)
        self.assertIsNone(pending.committed_episode_ref)

        reopened = NativeToolTurnStore(
            self.database,
            catalog_hash=self.dispatcher.catalog_hash,
            tool_names=frozenset(
                item["name"] for item in self.dispatcher.native_tools  # type: ignore[misc]
            ),
        )
        self.assertEqual(reopened.active_turn(), pending)
        terminal = ToolEvent(
            operation_ref=call.operation_ref,
            sequence=2,
            status="COMPLETED",
            result_json='{"answer":"bounded evidence"}',
            recorded_at_utc="2026-09-02T15:00:00.000000Z",
        )
        with self.assertRaisesRegex(RuntimeError, "revision conflict"):
            reopened.record_tool_result(
                turn.turn_id,
                expected_revision=0,
                call_id=call.call_id,
                terminal_event=terminal,
            )
        continued = reopened.record_tool_result(
            turn.turn_id,
            expected_revision=1,
            call_id=call.call_id,
            terminal_event=terminal,
        )
        self.assertEqual(continued.status, "CONTINUATION_READY")
        self.assertEqual(continued.outstanding_calls, ())
        self.assertEqual(continued.trace[-1]["result"], {"answer": "bounded evidence"})
        self.assertIsNotNone(reopened.active_turn())

        final = reopened.stage_final(
            turn.turn_id,
            expected_revision=2,
            response="The evidence is now sufficient.",
            continuation_choice_json=(
                '{"outstanding_call_ids":[],"route":"FINAL",'
                '"turn_revision":2}'
            ),
        )
        self.assertEqual(final.status, "FINAL_READY")
        self.assertIsNone(final.committed_episode_ref)
        self.assertEqual(reopened.active_turn(), final)

        committed = reopened.mark_committed(
            turn.turn_id,
            expected_revision=3,
            episode_ref=_ref("8"),
        )
        self.assertEqual(committed.status, "COMMITTED")
        self.assertEqual(committed.committed_episode_ref, _ref("8"))
        self.assertIsNone(reopened.active_turn())
        self.assertEqual(reopened.get(turn.turn_id), committed)
        next_turn = reopened.begin(
            original_request_ref=_ref("3"), original_observation_ref=_ref("4")
        )
        self.assertEqual(next_turn.status, "OPEN")

    def test_rejects_unknown_tools_duplicate_calls_and_more_than_eight(self) -> None:
        turn = self.store.begin(
            original_request_ref=_ref("a"), original_observation_ref=_ref("b")
        )
        unknown = NativeToolCall(
            "unknown-call",
            "tool.web.unknown",
            _ref("c"),
            _ref("d"),
            "{}",
        )
        with self.assertRaisesRegex(ValueError, "outside the exact catalog"):
            self.store.record_tool_requests(
                turn.turn_id,
                expected_revision=0,
                calls=(unknown,),
                continuation_choice_json='{"route":"TOOL_REQUESTED"}',
            )
        duplicate = NativeToolCall(
            "duplicate-call",
            "tool.web.search",
            _ref("e"),
            _ref("f"),
            "{}",
        )
        with self.assertRaisesRegex(ValueError, "identifiers must be unique"):
            self.store.record_tool_requests(
                turn.turn_id,
                expected_revision=0,
                calls=(duplicate, duplicate),
                continuation_choice_json='{"route":"TOOL_REQUESTED"}',
            )
        too_many = tuple(
            NativeToolCall(
                f"provider-call-{index}",
                "tool.web.search",
                "sha256:" + f"{index + 10:064x}",
                "sha256:" + f"{index + 30:064x}",
                "{}",
            )
            for index in range(9)
        )
        with self.assertRaisesRegex(ValueError, "exceeds eight"):
            self.store.record_tool_requests(
                turn.turn_id,
                expected_revision=0,
                calls=too_many,
                continuation_choice_json='{"route":"TOOL_REQUESTED"}',
            )


if __name__ == "__main__":
    unittest.main()
