from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import http.client
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from time import monotonic
import unittest
from unittest.mock import patch

from angler.runtime.jenny2_runtime import (
    Jenny2Status,
    NativeAgentTurnResult,
    NativeHostToolResult,
)
from angler.runtime.jenny2_activity import Jenny2ActivitySnapshot
from angler.runtime.persistent_autonomy import SupervisorResult
from angler.runtime.authored_artifact import (
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AuthoredArtifactAction,
    AuthoredArtifactExecutor,
)
from angler.runtime.persistent_autonomy import AffordanceRequest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "jenny2_chat.py"
SPEC = importlib.util.spec_from_file_location("jenny2_chat_api_under_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery invariant
    raise RuntimeError("Jenny API script could not be loaded")
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)

UI_TOKEN = "ui-" + "u" * 61
BRIDGE_TOKEN = "bridge-" + "b" * 57
EPISODE_REF = "sha256:" + "e" * 64
OPERATION_REF = "sha256:" + "o" * 64
PERMISSION_REF = "sha256:" + "p" * 64
CATALOG_HASH = "sha256:" + "c" * 64


def _episode_payloads():
    general_event_ref = "sha256:" + "f" * 64
    observed = API.ObservableConsequence(
        request_ref="sha256:" + "1" * 64,
        source_kind="WORLD",
        source_ref="sha256:" + "2" * 64,
        observation_json=json.dumps(
            {"private_observation": "SECRET-OBSERVATION"},
            separators=(",", ":"),
            sort_keys=True,
        ),
        artifact_refs=("sha256:" + "3" * 64,),
        evidence_refs=("sha256:" + "4" * 64,),
    )
    general = {
        "event_ref": general_event_ref,
        "parent_state_ref": "sha256:" + "5" * 64,
        "child_state_ref": "sha256:" + "6" * 64,
        "observation": {
            "content": "SECRET-USER-CONTENT",
            "source": "HUMAN",
            "trigger_ref": "sha256:" + "0" * 64,
        },
        "choice": {
            "selected_affordance_id": "cortex.respond",
            "rankings": [
                {"affordance_id": "cortex.respond", "score": 0.75},
                {"affordance_id": "internal.self-observation-diary", "score": 0.25},
            ],
            "prediction": "A bounded natural response should resolve the request.",
            "uncertainty": 0.2,
            "action_payload": "SECRET-ACTION-PAYLOAD",
            "context_json": json.dumps(
                {
                    "memories": ["SECRET-MEMORY"],
                    "intent_proposal": {
                        "epistemic_status": "MODEL_PROPOSAL_BEFORE_ACTION",
                        "state_assessment": "A response is requested.",
                        "resolution_target": "Answer clearly.",
                        "desired_state_change": "The request is resolved.",
                        "selection_basis": "Conversation is the fitting path.",
                        "intent_candidates": ["SECRET-CANDIDATE-REASONING"],
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
        "receipt": {
            "status": "COMPLETED_UNEVALUATED",
            "output": "SECRET-RECEIPT-OUTPUT",
            "consequence": [],
            "observable_consequence": observed.canonical_payload(),
        },
        "reflection": json.dumps(
            {
                "analysis": "The public response remains unevaluated.",
                "revision": "Wait for attributable evidence.",
                "retained_principle": "Do not invent outcome credit.",
                "raw_model_output": "SECRET-SCRATCH-REASONING",
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "temporal": {
            "recorded_time_utc": "2026-09-03T01:00:00.000000Z",
            "timezone": "America/New_York",
        },
    }

    proposal = {
        "alternative_explanations": ["SECRET-DIARY-ALTERNATIVE"],
        "candidate_label": "careful momentum",
        "estimated_strength": 0.6,
        "functional_description": "A tendency to proceed while checking uncertainty.",
        "observable_signals": ["SECRET-DIARY-SIGNAL"],
        "uncertainty": 0.3,
    }
    diary_observation = API.ObservableConsequence(
        request_ref="sha256:" + "7" * 64,
        source_kind=API.SELF_OBSERVATION_DIARY_SOURCE_KIND,
        source_ref=API.SELF_OBSERVATION_DIARY_SOURCE_REF,
        observation_json=json.dumps(
            {
                "contract": API.SELF_OBSERVATION_DIARY_CONTRACT,
                "proposal": proposal,
                "purpose": API.SELF_OBSERVATION_DIARY_PURPOSE,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    diary_event_ref = "sha256:" + "8" * 64
    diary = {
        "event_ref": diary_event_ref,
        "parent_state_ref": "sha256:" + "6" * 64,
        "child_state_ref": "sha256:" + "9" * 64,
        "observation": {"content": "SECRET-DIARY-TRIGGER"},
        "choice": {
            "selected_affordance_id": API.SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
            "rankings": [
                {
                    "affordance_id": API.SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
                    "score": 0.8,
                }
            ],
            "prediction": "A provisional hypothesis will be recorded.",
            "uncertainty": 0.3,
            "action_payload": "SECRET-DIARY-ACTION",
            "context_json": "{}",
        },
        "receipt": {
            "status": "COMPLETED",
            "output": "SECRET-DIARY-RECEIPT",
            "consequence": [],
            "observable_consequence": diary_observation.canonical_payload(),
        },
        "reflection": json.dumps(
            {
                "epistemic_status": "UNVERIFIED",
                "analysis": "The hypothesis remains provisional.",
                "raw_model_output": "SECRET-DIARY-SCRATCH",
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "temporal": {
            "recorded_time_utc": "2026-09-03T01:01:00.000000Z",
            "timezone": "America/New_York",
        },
    }
    return (
        _FakeEpisode("sha256:" + "a" * 64, general_event_ref, 0, general),
        _FakeEpisode("sha256:" + "b" * 64, diary_event_ref, 1, diary),
    )


class _FakeEpisode:
    def __init__(self, episode_ref: str, event_ref: str, ordinal: int, payload) -> None:
        self.episode_ref = episode_ref
        self.event_ref = event_ref
        self.ordinal = ordinal
        self.payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _authored_episode() -> _FakeEpisode:
    evidence_ref = "sha256:" + "1" * 64
    source_episode_ref = "sha256:" + "2" * 64
    action = AuthoredArtifactAction(
        version=1,
        kind="short reflective fiction",
        title="A Window Remembering Rain",
        body="The room kept the shape of the weather after the clouds had moved on.",
        purpose="Preserve one revisable model-authored work for later reflection.",
        evidence_refs=(evidence_ref,),
        source_episode_refs=(source_episode_ref,),
    )
    request = AffordanceRequest(
        idempotency_key="sha256:" + "3" * 64,
        trigger_ref="test:authored-artifact-ui",
        affordance_id=AUTHORED_ARTIFACT_AFFORDANCE_ID,
        observation_ref="sha256:" + "4" * 64,
        state_head_ref="sha256:" + "5" * 64,
        action_payload=action.canonical_json(),
    )
    receipt = AuthoredArtifactExecutor()(request)
    assert receipt.observable_consequence is not None
    event_ref = "sha256:" + "6" * 64
    payload = {
        "event_ref": event_ref,
        "parent_state_ref": "sha256:" + "7" * 64,
        "child_state_ref": "sha256:" + "8" * 64,
        "observation": {"content": "SECRET-AUTHORED-TRIGGER"},
        "choice": {
            "selected_affordance_id": AUTHORED_ARTIFACT_AFFORDANCE_ID,
            "rankings": [
                {"affordance_id": AUTHORED_ARTIFACT_AFFORDANCE_ID, "score": 1.0}
            ],
            "prediction": "One private work will be durably observed.",
            "uncertainty": 0.2,
            "action_payload": "SECRET-AUTHORED-ACTION",
            "context_json": "{}",
        },
        "receipt": {
            "status": receipt.status,
            "output": "SECRET-AUTHORED-RECEIPT",
            "consequence": [],
            "observable_consequence": receipt.observable_consequence.canonical_payload(),
        },
        "reflection": json.dumps(
            {
                "epistemic_status": "MODEL_AUTHORED_WORK_UNVERIFIED",
                "analysis": "The work was preserved without turning its prose into fact.",
                "raw_model_output": "SECRET-AUTHORED-SCRATCH",
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "temporal": {
            "recorded_time_utc": "2026-09-04T00:20:00.000000Z",
            "timezone": "America/New_York",
        },
    }
    return _FakeEpisode("sha256:" + "9" * 64, event_ref, 2, payload)


class _FakeStateHead:
    moving_origin_ordinal = 1


class _FakeSupervisor:
    def __init__(self) -> None:
        self.episodes = _episode_payloads()

    def state_head(self):
        return _FakeStateHead()

    def episode_items(self, *, after_ordinal: int, limit: int):
        return tuple(
            episode
            for episode in self.episodes
            if episode.ordinal > after_ordinal
        )[:limit]


class _FakeRuntime:
    def __init__(self) -> None:
        self.life_running = True
        self.supervisor = _FakeSupervisor()
        self.chat_calls: list[tuple[str, str]] = []
        self.native_chat_calls: list[tuple[str, str, tuple[dict, ...], str]] = []
        self.native_continue_calls: list[
            tuple[str, int, tuple[NativeHostToolResult, ...]]
        ] = []
        self.composition_calls = 0
        self.foreground_defer_calls = 0
        self.foreground_deferred = threading.Event()

    def defer_life_for_foreground(self) -> None:
        self.foreground_defer_calls += 1
        self.foreground_deferred.set()

    def composition_manifest(self):
        self.composition_calls += 1

        class Manifest:
            @staticmethod
            def to_dict():
                return {
                    "schema": "jenny2.live-composition-manifest.v1",
                    "manifest_ref": "sha256:" + "m" * 64,
                    "integrity_status": "FRESH",
                }

        return Manifest()

    def status(self) -> Jenny2Status:
        return Jenny2Status(
            agent_ref="jenny.agent.v2",
            genesis_ref="sha256:" + "g" * 64,
            state_ref="sha256:" + "s" * 64,
            moving_origin_time_utc="2026-09-03T01:00:00.000000Z",
            moving_origin_local_time="2026-09-02T21:00:00.000000-04:00",
            moving_origin_timezone="America/New_York",
            clock_uncertainty_ms=1.0,
            clock_jump_detected=False,
            moving_origin_ordinal=26,
            last_event_ref="sha256:" + "l" * 64,
            scheduler_enabled=True,
            selector_qualification_ref="sha256:" + "q" * 64,
            pending_operation=False,
            pending_projections=0,
            external_effects_enabled=False,
            host_guarded_tools_enabled=True,
            projection_error=None,
            life_error=None,
        )

    def activity(self) -> Jenny2ActivitySnapshot:
        return Jenny2ActivitySnapshot(
            version="jenny.runtime-activity.v1",
            sequence=4,
            phase="DELIBERATE",
            phase_started_at_utc="2026-09-03T01:00:00.000Z",
            phase_elapsed_seconds=1.25,
            operation_source="HUMAN",
            current_status="RUNNING",
            error_code=None,
            last_completed=None,
        )

    def chat(self, trigger_ref: str, content: str) -> SupervisorResult:
        self.chat_calls.append((trigger_ref, content))
        return SupervisorResult(
            "COMMITTED",
            trigger_ref,
            "cortex.respond",
            EPISODE_REF,
            None,
            26,
            "receipt, learned update, episode, and state head committed",
        )

    def turn_output(self, episode_ref: str) -> tuple[str, str]:
        if episode_ref != EPISODE_REF:
            raise KeyError(episode_ref)
        return "Hello from Jenny.", "COMPLETED_UNEVALUATED"

    def native_chat(
        self,
        trigger_ref: str,
        content: str,
        *,
        tools: tuple[dict, ...],
        catalog_hash: str,
    ) -> NativeAgentTurnResult:
        self.native_chat_calls.append((trigger_ref, content, tools, catalog_hash))
        return NativeAgentTurnResult(
            status="TOOL_REQUESTED",
            catalog_hash=catalog_hash,
            turn_id="turn." + "t" * 64,
            turn_revision=1,
            tool_calls=(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"Jenny"}',
                    },
                    "operation_ref": OPERATION_REF,
                    "permission_reservation_ref": PERMISSION_REF,
                },
            ),
            permission_reservation_refs=(PERMISSION_REF,),
            message=None,
            episode_ref=None,
            moving_origin_ordinal=26,
        )

    def native_continue(
        self,
        *,
        turn_id: str,
        turn_revision: int,
        tool_results: tuple[NativeHostToolResult, ...],
    ) -> NativeAgentTurnResult:
        self.native_continue_calls.append((turn_id, turn_revision, tool_results))
        return NativeAgentTurnResult(
            status="COMMITTED",
            catalog_hash=CATALOG_HASH,
            turn_id=turn_id,
            turn_revision=turn_revision + 2,
            tool_calls=(),
            permission_reservation_refs=(PERMISSION_REF,),
            message="I found the grounded result.",
            episode_ref=EPISODE_REF,
            moving_origin_ordinal=27,
        )


@contextmanager
def _running_server(runtime: _FakeRuntime, *, operation_lock=None):
    operation_lock = operation_lock or threading.Lock()
    server = API._build_api_server(
        runtime,
        identity="jenny2-api-test",
        bind="127.0.0.1",
        port=0,
        ui_token=UI_TOKEN,
        bridge_token=BRIDGE_TOKEN,
        operation_lock=operation_lock,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(address, path: str, raw: str, token: str, request_id: str = "request-1"):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
    connection.request(
        "POST",
        path,
        body=raw.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
        },
    )
    response = connection.getresponse()
    payload = json.loads(response.read())
    result = (response.status, dict(response.getheaders()), payload)
    connection.close()
    return result


def _get(address, path: str, token: str | None = None):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=2)
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    result = (response.status, dict(response.getheaders()), payload)
    connection.close()
    return result


def _get_raw(address, path: str, token: str | None = None):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=2)
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    payload = response.read().decode("utf-8")
    result = (response.status, dict(response.getheaders()), payload)
    connection.close()
    return result


def _tool_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": "web_search",
            "description": "Search the web under host supervision.",
            "input_schema": {
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "type": "object",
            },
        }
    ]


class Jenny2ChatApiTests(unittest.TestCase):
    def test_composition_view_is_ui_authenticated_read_only_and_no_store(self):
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            unauthorized, _, unauthorized_payload = _get(
                address, "/v1/composition"
            )
            status, headers, payload = _get(
                address, "/v1/composition", UI_TOKEN
            )
            query_status, _, query_payload = _get(
                address, "/v1/composition?refresh=true", UI_TOKEN
            )

        self.assertEqual(unauthorized, 401)
        self.assertEqual(unauthorized_payload, {"error": "unauthorized"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["schema"], "jenny2.live-composition-manifest.v1")
        self.assertEqual(payload["integrity_status"], "FRESH")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(runtime.composition_calls, 1)
        self.assertEqual(query_status, 400)
        self.assertEqual(
            query_payload, {"error": "query_parameters_are_unsupported"}
        )

    def test_base_binding_rejects_unbound_adapter_name(self) -> None:
        resolved = API._resolve_model_binding(
            binding_path=None,
            endpoint=None,
            served_model=None,
        )
        self.assertEqual(resolved[0], API.QWEN38_MODEL_ENDPOINT)
        self.assertEqual(resolved[1], API.QWEN38_BASE_SERVED_MODEL)
        self.assertEqual(resolved[3], API.QUALIFICATION_REF)
        self.assertIsNone(resolved[4])
        with self.assertRaisesRegex(ValueError, "requires a content-addressed binding"):
            API._resolve_model_binding(
                binding_path=None,
                endpoint=API.QWEN38_MODEL_ENDPOINT,
                served_model="jenny-qwen3.8-27b--lora-sha256-" + "a" * 64,
            )

    def test_model_health_requires_the_exact_served_identity(self) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read(_maximum):
                return json.dumps(
                    {
                        "data": [
                            {"id": API.QWEN38_BASE_SERVED_MODEL},
                            {"id": "bound-adapter"},
                        ]
                    }
                ).encode("utf-8")

        with patch.object(API, "urlopen", return_value=Response()):
            self.assertTrue(
                API._model_runtime_ready(
                    API.QWEN38_MODEL_ENDPOINT, "bound-adapter"
                )
            )
            self.assertFalse(
                API._model_runtime_ready(
                    API.QWEN38_MODEL_ENDPOINT, "missing-adapter"
                )
            )

    def test_activity_is_ui_authenticated_and_does_not_take_operation_lock(self) -> None:
        runtime = _FakeRuntime()
        operation_lock = threading.Lock()
        with _running_server(runtime, operation_lock=operation_lock) as address:
            unauthorized, _, payload = _get(address, "/v1/activity")
            self.assertEqual(unauthorized, 401)
            self.assertEqual(payload, {"error": "unauthorized"})

            operation_lock.acquire()
            try:
                started = monotonic()
                status, headers, payload = _get(
                    address, "/v1/activity", UI_TOKEN
                )
                elapsed = monotonic() - started
            finally:
                operation_lock.release()
        self.assertEqual(status, 200)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(payload["phase"], "DELIBERATE")
        self.assertEqual(payload["current_status"], "RUNNING")
        self.assertEqual(payload["operation_source"], "HUMAN")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_owner_page_is_authenticated_self_contained_and_credential_free(self) -> None:
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            unauthorized, _, payload = _get(address, "/")
            bridge_status, _, bridge_payload = _get(address, "/ui", BRIDGE_TOKEN)
            status, headers, document = _get_raw(address, "/", UI_TOKEN)

        self.assertEqual(unauthorized, 401)
        self.assertEqual(payload, {"error": "unauthorized"})
        self.assertEqual(bridge_status, 401)
        self.assertEqual(bridge_payload, {"error": "unauthorized"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertNotIn("'unsafe-inline'", headers["Content-Security-Policy"])
        for label in ("Chat", "Journal", "Cognitive Trace"):
            self.assertIn(f">{label}</button>", document)
        for relative_path in (
            '"/v1/chat"',
            '"/v1/ui-state"',
            '"/v1/diary"',
            '"/v1/cognitive-trace"',
        ):
            self.assertIn(relative_path, document)
        self.assertIn("payload.message", document)
        self.assertIn("syncConversation(state.conversation)", document)
        self.assertNotIn("payload.detail", document)
        self.assertIn("const MAX_QUEUE = 8", document)
        self.assertIn("replaced before send", document)
        self.assertNotIn("innerHTML", document)
        self.assertNotIn(UI_TOKEN, document)
        self.assertNotIn(BRIDGE_TOKEN, document)
        self.assertNotIn("192.168.137.5", document)
        self.assertNotIn("127.0.0.1:30000", document)

    def test_ui_state_is_authenticated_lock_isolated_and_bounded(self) -> None:
        runtime = _FakeRuntime()
        operation_lock = threading.Lock()
        with _running_server(runtime, operation_lock=operation_lock) as address:
            operation_lock.acquire()
            try:
                started = monotonic()
                status, _, payload = _get(address, "/v1/ui-state", UI_TOKEN)
                elapsed = monotonic() - started
            finally:
                operation_lock.release()
            query_status, _, query_payload = _get(
                address, "/v1/ui-state?extra=true", UI_TOKEN
            )

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(payload["schema"], "jenny2.ui-state.v1")
        self.assertIs(payload["life_loop_running"], True)
        self.assertEqual(payload["activity"]["phase"], "DELIBERATE")
        self.assertEqual(len(payload["conversation"]), 1)
        self.assertEqual(
            payload["conversation"][0]["message"], "SECRET-USER-CONTENT"
        )
        self.assertEqual(
            payload["conversation"][0]["response"], "SECRET-RECEIPT-OUTPUT"
        )
        self.assertRegex(payload["last_update_utc"], r"Z$")
        self.assertEqual(query_status, 400)
        self.assertEqual(
            query_payload,
            {"error": "query_parameters_are_unsupported"},
        )

    def test_owner_conversation_joins_tool_continuation_once(self) -> None:
        runtime = _FakeRuntime()
        trigger_ref = "sha256:" + "d" * 64
        human = _FakeEpisode(
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            2,
            {
                "observation": {
                    "source": "HUMAN",
                    "trigger_ref": trigger_ref,
                    "content": "Please inspect the library.",
                },
                "choice": {"selected_affordance_id": "internal.library-read"},
                "receipt": {"status": "COMPLETED", "output": "PRIVATE TOOL OUTPUT"},
            },
        )
        continuation = _FakeEpisode(
            "sha256:" + "3" * 64,
            "sha256:" + "4" * 64,
            3,
            {
                "observation": {
                    "source": "CONTINUATION",
                    "content": json.dumps(
                        {
                            "contract": "jenny.library.response-continuation.v1",
                            "human_trigger_ref": trigger_ref,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
                "choice": {"selected_affordance_id": "cortex.respond"},
                "receipt": {
                    "status": "COMPLETED_UNEVALUATED",
                    "output": "The library contains verified works.",
                },
            },
        )
        runtime.supervisor.episodes = (
            *runtime.supervisor.episodes,
            human,
            continuation,
        )

        conversation = API._recent_owner_conversation(runtime)

        self.assertEqual(conversation[-1]["message"], "Please inspect the library.")
        self.assertEqual(
            conversation[-1]["response"],
            "The library contains verified works.",
        )
        self.assertNotIn("PRIVATE TOOL OUTPUT", json.dumps(conversation))

    def test_diary_view_is_ui_only_source_bound_and_redacted(self) -> None:
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            unauthorized, _, _ = _get(address, "/v1/diary")
            bridge_status, _, _ = _get(address, "/v1/diary", BRIDGE_TOKEN)
            status, headers, payload = _get(address, "/v1/diary", UI_TOKEN)

        self.assertEqual(unauthorized, 401)
        self.assertEqual(bridge_status, 401)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(payload["schema"], "jenny2.diary-view.v1")
        self.assertIs(payload["read_only"], True)
        self.assertIs(payload["direct_write_supported"], False)
        self.assertEqual(
            payload["epistemic_status"],
            API.SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS,
        )
        self.assertEqual(len(payload["entries"]), 1)
        entry = payload["entries"][0]
        self.assertEqual(entry["candidate_label"], "careful momentum")
        self.assertEqual(entry["observable_signal_count"], 1)
        self.assertEqual(entry["alternative_explanation_count"], 1)
        self.assertRegex(entry["entry_ref"], r"^sha256:[0-9a-f]{64}$")
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "SECRET-DIARY-SIGNAL",
            "SECRET-DIARY-ALTERNATIVE",
            "SECRET-DIARY-TRIGGER",
            "SECRET-DIARY-ACTION",
            "SECRET-DIARY-RECEIPT",
            "SECRET-DIARY-SCRATCH",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_diary_view_shows_validated_authored_work_without_scratch(self) -> None:
        runtime = _FakeRuntime()
        runtime.supervisor.episodes = (
            *runtime.supervisor.episodes,
            _authored_episode(),
        )
        with _running_server(runtime) as address:
            status, _, payload = _get(address, "/v1/diary", UI_TOKEN)

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["authored_artifact_contract"],
            API.AUTHORED_ARTIFACT_ACTION_CONTRACT,
        )
        self.assertEqual(len(payload["authored_artifacts"]), 1)
        artifact = payload["authored_artifacts"][0]
        self.assertEqual(artifact["kind"], "short reflective fiction")
        self.assertEqual(artifact["title"], "A Window Remembering Rain")
        self.assertIn("shape of the weather", artifact["body"])
        self.assertEqual(artifact["visibility"], "PRIVATE")
        self.assertRegex(artifact["artifact_ref"], r"^sha256:[0-9a-f]{64}$")
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "SECRET-AUTHORED-TRIGGER",
            "SECRET-AUTHORED-ACTION",
            "SECRET-AUTHORED-RECEIPT",
            "SECRET-AUTHORED-SCRATCH",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_cognitive_trace_is_explicit_whitelist_not_raw_reasoning(self) -> None:
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            unauthorized, _, _ = _get(address, "/v1/cognitive-trace")
            status, _, payload = _get(
                address, "/v1/cognitive-trace", UI_TOKEN
            )

        self.assertEqual(unauthorized, 401)
        self.assertEqual(status, 200)
        self.assertEqual(payload["schema"], "jenny2.cognitive-trace-view.v1")
        self.assertIs(payload["read_only"], True)
        self.assertIn("not hidden chain-of-thought", payload["notice"])
        self.assertEqual(len(payload["entries"]), 2)
        general = payload["entries"][1]
        self.assertEqual(
            general["candidates"]["selected_affordance_id"],
            "cortex.respond",
        )
        self.assertEqual(general["candidates"]["count"], 2)
        self.assertEqual(general["ranking"][0]["position"], 1)
        self.assertEqual(
            general["evidence"]["evidence_refs"],
            ["sha256:" + "4" * 64],
        )
        self.assertEqual(
            general["reflection"]["analysis"],
            "The public response remains unevaluated.",
        )
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "SECRET-OBSERVATION",
            "SECRET-USER-CONTENT",
            "SECRET-ACTION-PAYLOAD",
            "SECRET-MEMORY",
            "SECRET-CANDIDATE-REASONING",
            "SECRET-RECEIPT-OUTPUT",
            "SECRET-SCRATCH-REASONING",
            "SECRET-DIARY-SIGNAL",
            "SECRET-DIARY-ALTERNATIVE",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_diary_source_tamper_fails_closed_without_detail(self) -> None:
        runtime = _FakeRuntime()
        diary_episode = runtime.supervisor.episodes[1]
        payload = json.loads(diary_episode.payload_json)
        payload["receipt"]["observable_consequence"]["source_ref"] = (
            "sha256:" + "0" * 64
        )
        diary_episode.payload_json = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        with _running_server(runtime) as address:
            status, _, response = _get(address, "/v1/diary", UI_TOKEN)

        self.assertEqual(status, 409)
        self.assertEqual(response, {"error": "view_unavailable"})
        self.assertNotIn("source", json.dumps(response))

    def test_tokens_are_distinct_same_owner_regular_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ui_path = root / "ui-token.txt"
            bridge_path = root / "bridge-token.txt"
            ui = API._load_or_create_token(ui_path, label="UI")
            bridge = API._load_or_create_token(bridge_path, label="bridge")

            self.assertNotEqual(ui, bridge)
            self.assertTrue(API._token_paths_are_distinct(ui_path, bridge_path))
            self.assertEqual(stat.S_IMODE(ui_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(bridge_path.stat().st_mode), 0o600)
            self.assertEqual(ui_path.stat().st_uid, os.getuid())
            self.assertEqual(API._load_or_create_token(ui_path, label="UI"), ui)

            ui_path.chmod(0o640)
            with self.assertRaisesRegex(ValueError, "mode must be 0600"):
                API._load_or_create_token(ui_path, label="UI")
            with self.assertRaisesRegex(ValueError, "distinct bounded secrets"):
                API._build_api_server(
                    _FakeRuntime(),
                    identity="test",
                    bind="127.0.0.1",
                    port=0,
                    ui_token=UI_TOKEN,
                    bridge_token=UI_TOKEN,
                    operation_lock=threading.Lock(),
                )

    def test_ui_chat_remains_legacy_speech_schema_with_cors(self) -> None:
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            status, headers, payload = _post(
                address, "/v1/chat", '{"message":"Hello"}', UI_TOKEN
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(payload["message"], "Hello from Jenny.")
        self.assertEqual(payload["receipt_status"], "COMPLETED_UNEVALUATED")
        self.assertEqual(payload["request_id"], "request-1")
        self.assertEqual(len(runtime.chat_calls), 1)
        self.assertTrue(runtime.chat_calls[0][1].startswith("Speaker: "))
        self.assertTrue(runtime.chat_calls[0][1].endswith("\n\nHello"))
        self.assertEqual(runtime.native_chat_calls, [])
        self.assertEqual(runtime.foreground_defer_calls, 2)

    def test_chat_defers_autonomy_before_waiting_for_operation_lock(self) -> None:
        runtime = _FakeRuntime()
        operation_lock = threading.Lock()
        operation_lock.acquire()
        response: dict[str, object] = {}

        def send() -> None:
            response["value"] = _post(
                address, "/v1/chat", '{"message":"Foreground"}', UI_TOKEN
            )

        try:
            with _running_server(runtime, operation_lock=operation_lock) as address:
                thread = threading.Thread(target=send, daemon=True)
                thread.start()
                self.assertTrue(runtime.foreground_deferred.wait(1.0))
                self.assertTrue(thread.is_alive())
                self.assertEqual(runtime.chat_calls, [])
                operation_lock.release()
                thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())
        finally:
            if operation_lock.locked():
                operation_lock.release()

        status, _, payload = response["value"]
        self.assertEqual(status, 200)
        self.assertEqual(payload["message"], "Hello from Jenny.")
        self.assertEqual(runtime.foreground_defer_calls, 2)

    def test_bridge_chat_dispatches_by_credential_and_has_no_cors(self) -> None:
        runtime = _FakeRuntime()
        rich_body = json.dumps(
            {
                "message": "Research this.",
                "tools": _tool_catalog(),
                "catalog_hash": CATALOG_HASH,
            },
            separators=(",", ":"),
        )
        with _running_server(runtime) as address:
            status, headers, payload = _post(
                address, "/v1/chat", rich_body, BRIDGE_TOKEN
            )
            ui_status, ui_headers, _ = _post(
                address, "/v1/chat", rich_body, UI_TOKEN, "ui-rich"
            )
            bridge_legacy_status, bridge_legacy_headers, _ = _post(
                address,
                "/v1/chat",
                '{"message":"legacy"}',
                BRIDGE_TOKEN,
                "bridge-legacy",
            )

        self.assertEqual(status, 202)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(payload["status"], "TOOL_REQUESTED")
        self.assertEqual(payload["turn_revision"], 1)
        self.assertEqual(payload["tool_calls"][0]["id"], "call-1")
        self.assertEqual(payload["permission_reservation_refs"], [PERMISSION_REF])
        self.assertEqual(len(runtime.native_chat_calls), 1)
        trigger, message, tools, catalog_hash = runtime.native_chat_calls[0]
        self.assertTrue(trigger.startswith("sha256:"))
        self.assertEqual(message, "Research this.")
        self.assertEqual(tools, tuple(_tool_catalog()))
        self.assertEqual(catalog_hash, CATALOG_HASH)
        self.assertEqual(ui_status, 400)
        self.assertEqual(ui_headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(bridge_legacy_status, 400)
        self.assertNotIn("Access-Control-Allow-Origin", bridge_legacy_headers)
        self.assertEqual(runtime.chat_calls, [])

    def test_bridge_continue_converts_exact_results_and_is_bridge_only(self) -> None:
        runtime = _FakeRuntime()
        body = json.dumps(
            {
                "turn_id": "turn." + "t" * 64,
                "turn_revision": 1,
                "tool_results": [
                    {
                        "call_id": "call-1",
                        "tool_name": "web_search",
                        "operation_ref": OPERATION_REF,
                        "permission_reservation_ref": PERMISSION_REF,
                        "status": "COMPLETED",
                        "result": {"answer": "grounded"},
                    }
                ],
            },
            separators=(",", ":"),
        )
        with _running_server(runtime) as address:
            status, headers, payload = _post(
                address, "/v1/chat/continue", body, BRIDGE_TOKEN
            )
            ui_status, ui_headers, _ = _post(
                address, "/v1/chat/continue", body, UI_TOKEN, "ui-continue"
            )

        self.assertEqual(status, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(payload["status"], "COMMITTED")
        self.assertEqual(payload["message"], "I found the grounded result.")
        self.assertEqual(len(runtime.native_continue_calls), 1)
        turn_id, revision, results = runtime.native_continue_calls[0]
        self.assertEqual(turn_id, "turn." + "t" * 64)
        self.assertEqual(revision, 1)
        self.assertEqual(results[0].result, {"answer": "grounded"})
        self.assertEqual(results[0].status, "COMPLETED")
        self.assertEqual(ui_status, 401)
        self.assertNotIn("Access-Control-Allow-Origin", ui_headers)

    def test_strict_json_fields_and_request_identity_fail_closed(self) -> None:
        runtime = _FakeRuntime()
        with _running_server(runtime) as address:
            duplicate_status, duplicate_headers, _ = _post(
                address,
                "/v1/chat",
                '{"message":"x","message":"y","tools":[],"catalog_hash":"x"}',
                BRIDGE_TOKEN,
                "duplicate",
            )
            nonfinite_status, _, _ = _post(
                address,
                "/v1/chat",
                '{"message":"x","tools":[{"name":"t","description":"d",'
                '"input_schema":{"limit":NaN}}],"catalog_hash":"x"}',
                BRIDGE_TOKEN,
                "nonfinite",
            )
            unknown_status, _, _ = _post(
                address,
                "/v1/chat",
                json.dumps(
                    {
                        "message": "x",
                        "tools": _tool_catalog(),
                        "catalog_hash": CATALOG_HASH,
                        "request_id": "not-allowed",
                    },
                    separators=(",", ":"),
                ),
                BRIDGE_TOKEN,
                "unknown",
            )
            conflict_status, conflict_headers, _ = _post(
                address,
                "/v1/chat",
                '{"message":"x","request_id":"body-id"}',
                UI_TOKEN,
                "header-id",
            )
            bool_revision_status, _, _ = _post(
                address,
                "/v1/chat/continue",
                json.dumps(
                    {
                        "turn_id": "turn.x",
                        "turn_revision": True,
                        "tool_results": [
                            {
                                "call_id": "x",
                                "tool_name": "x",
                                "operation_ref": OPERATION_REF,
                                "permission_reservation_ref": PERMISSION_REF,
                                "status": "ERROR",
                                "result": {},
                            }
                        ],
                    },
                    separators=(",", ":"),
                ),
                BRIDGE_TOKEN,
                "bool-revision",
            )

        self.assertEqual(duplicate_status, 400)
        self.assertNotIn("Access-Control-Allow-Origin", duplicate_headers)
        self.assertEqual(nonfinite_status, 400)
        self.assertEqual(unknown_status, 400)
        self.assertEqual(conflict_status, 400)
        self.assertEqual(conflict_headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(bool_revision_status, 400)
        self.assertEqual(runtime.chat_calls, [])
        self.assertEqual(runtime.native_chat_calls, [])
        self.assertEqual(runtime.native_continue_calls, [])

    def test_bridge_result_is_the_runtime_dataclass_without_ui_metadata(self) -> None:
        expected = _FakeRuntime().native_chat(
            "sha256:" + "r" * 64,
            "Research this.",
            tools=tuple(_tool_catalog()),
            catalog_hash=CATALOG_HASH,
        )
        payload = asdict(expected)
        self.assertNotIn("elapsed_seconds", payload)
        self.assertNotIn("receipt_status", payload)
        self.assertIsNone(payload["message"])


if __name__ == "__main__":
    unittest.main()
